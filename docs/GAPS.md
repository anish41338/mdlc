# GAPS — ground-truth audit (Phase 0, 2026-07-04)

Everything below was measured by running commands in this repo at this commit,
not copied from docs. Reproduce with the commands shown. This file is the
work-order for Phase 1+; items get struck through only when a test proves them
closed.

## 1. Test-suite ground truth

`python -m pytest -q -m "not gpu"` — **72 passed, 0 failed, 0 xfail** (42 at
baseline commit + 30 added by this audit: 26 structural-verifier/idempotence
tests, 4 Clip-fusion regressions). No GPU-marked tests exist yet (Phase 2).

## 2. Headline numbers: reproduced ✔

`python -m mdlc.tools.compile examples/resnet18.onnx --report` confirms every
README claim: 141 → 32 ops, 17 fused groups (9 FusedConvAct + 8
FusedElementwise), 12 distinct kernels, 52 launches, activations 13.119 MB →
4.594 MB (**65.0%** saved), parity vs ONNX Runtime PASS (max_abs 5.2e-6,
max_rel 8.9e-5 on O(0.1) logits — non-vacuous).

## 3. Bugs found and fixed during the audit

1. **Silent ReLU6 drop (numerical corruption, the exact failure mode this
   project defends against).** `FuseActivation` fused `Clip` nodes whose
   min/max arrived as `Constant`-node outputs (the standard opset≥11 torch
   export), failed to resolve them, and recorded `clip_min=None/clip_max=None`
   — every ReLU6 in MobileNetV2 became identity. Fixed twice over:
   `ConstantFolding` now materializes `Constant` nodes as initializers, and
   `FuseActivation` refuses to fuse when a bound is present but unresolvable.
   Regression tests: `test_clip_with_unresolvable_bounds_is_not_fused`,
   `test_clip_bounds_resolved_through_pipeline`.
2. **The oracle had no power to catch bug #1.** Untrained MobileNetV2 shrinks
   activations geometrically (~0.7×/layer, 52 layers) to logits ≈ 1e-8 —
   under atol=1e-4 *any* output passes. Two defenses added: the harness now
   fails a comparison as **VACUOUS** when golden magnitude < 100×atol, and the
   exporters LSUV-calibrate weights (unit output std per Conv/Linear on a
   seeded dummy) so every layer keeps O(1) signal. RepViT had the same
   disease (uncalibrated logits ~1e-8; now 3.9).
3. **Pass-hygiene: orphan nodes.** Caught by the new structural verifier:
   fusing Clip left its Constant producers orphaned mid-pipeline. Gone via
   fix #1; the verifier now enforces orphan-freedom after every pass.

## 4. Corrected fact: the batch gap is N>1, not N=1

The build brief said "batch-1 conv codegen broken". Measured truth (STATUS.md
had it right): **batch-1 works end-to-end; N>1 crashes** — the im2col→GEMM
lowering emits P = OH·OW columns for a single image, so at N=2 the output
reshape fails (`cannot reshape array of size 512 into shape (2,8,8,8)`).
Phase 1 must batch the im2col (P = N·OH·OW) or loop over N, and permanently
add N∈{1,2,8} to every conv parity test.

## 5. Host-fallback inventory (from `compile --report`, now printed per model)

### ResNet-18 — 3 host launches
| Fallback | Count | Burn-down |
|---|---|---|
| MaxPool | 1 | pooling kernel (Phase 1) |
| GlobalAveragePool | 1 | reduction kernel (Phase 1, §5.3) |
| Flatten | 1 | zero-copy view in runtime, not a kernel (Phase 1) |

### MobileNetV2 — 29 host launches
| Fallback | Count | Burn-down |
|---|---|---|
| FusedConvAct(group=C) depthwise 3×3, s1/s2 | 17 | direct depthwise kernel (Phase 1, §5.2) |
| Add (bare residual — linear bottleneck leaves no act to chain) | 10 | singleton elementwise lowering (Phase 1) |
| GlobalAveragePool | 1 | reduction kernel |
| Flatten | 1 | view |

Note: MobileNetV2's residual Adds expose a lowering gap distinct from fusion:
`emit.py` sends *any* bare elementwise op (Relu/Add/Mul/…) to host. Fix is to
route single elementwise ops through the existing elementwise kernel emitter.

### RepViT-m0_9 — cannot compile yet; blocked one rung earlier
Frontend import: **works**. Reference executor (oracle rung 1) is missing:

| Op | Count | Needed for |
|---|---|---|
| Erf | 27 | GELU decomposition (x/√2 → Erf → +1 → ×x → ×0.5) |
| ReduceMean | 11 | SE squeeze over H,W + final global pool |

After those land, codegen will additionally hit (from the op inventory):
depthwise 3×3 convs ×26 (group≠1 → host), ReduceMean + Erf (**no lowering at
all — codegen raises NotImplementedError**, they must join the elementwise
table / reduction template), bare Add/Mul/Div singletons, and SE's broadcast
`Mul((N,C,1,1), (N,C,H,W))` which the elementwise emitter rejects (element
counts differ → host). §5.3's per-input stride vectors fix that.

## 6. RepViT-m0_9 measured op inventory (deploy form, opset 17, batch 1)

`python -m mdlc.tools.build_repvit` → `examples/repvit_m0_9.onnx`, 463 nodes:
Conv×103 (75 pointwise 1×1; 2 standard 3×3 s2 stem; 23 depthwise 3×3 s1; 3
depthwise 3×3 s2), Add×53, Mul×64, Div×27, Erf×27, ReduceMean×11, Relu×10,
Sigmoid×10, Gemm×1, Constant×81, Identity×76 (the last two fold away).

Differences vs the brief's guess, so we implement reality: **no** MaxPool, no
HardSigmoid/HardSwish, no GELU-tanh (erf form only), no Flatten/Reshape, no
5×5/7×7 depthwise. All depthwise is 3×3. SE gate is plain Sigmoid.

## 7. Export decisions (frozen)

* Exporters: `tools/build_resnet18.py`, `tools/build_mobilenetv2.py`,
  `tools/build_repvit.py` (timm `repvit_m0_9` default, `reparameterize_model`
  deploy form). Public weights path only; random seeded weights +
  LSUV calibration (we verify agreement, not accuracy).
* opset 17, batch dim **fixed** (default 1) — kernels specialize on static
  shapes; a different batch is a different compile.

## 8. Tolerance contract: frozen in `mdlc/testing/tolerances.py`

Four tiers by comparison class — BITWISE (0/0), FP32_SAME_ORDER (1e-4/1e-5),
FP32_REDUCTION (1e-4/1e-4), FP32_NETWORK (1e-3/1e-4) — all call sites import
from there; every change from the previous ad-hoc values was a *tightening*
(im2col is now compared bit-exact; GEMM kernels went 1e-3→1e-4). Changing a
tier requires a written numerical justification in that file + human sign-off.

## 9. Other measured gaps (unchanged from STATUS, now with owners)

1. GEMM `alpha`/`beta` ≠ 1 ignored in the kernel epilogue (fine for torch
   exports which emit 1.0; assert at lowering time until fixed).
2. Memory planner has **no alignment guarantee** — offsets are raw byte
   sums. Phase 2 GPU bring-up needs 256-byte-aligned pool offsets (vectorized
   loads + coalescing); add `align=256` to the planner and an assert.
3. Autotuner on CPU ranks by analytical cost model only (by design); measured
   tuning is GPU-gated (Phase 2).
4. No CUDA-graph/stream work; launches are serial (fine — batch-1 latency
   story is fusion + fewer launches).
5. INT8/DP4A path not started (Phase 3).

## 10. Phase 1 acceptance restated against this audit

`compile --report` must print `host fallbacks: none` for ResNet-18,
MobileNetV2, and RepViT-m0_9; conv parity tests run N∈{1,2,8}; RepViT passes
reference + sim executors within FP32_NETWORK; fusion report shows depthwise
Conv(+BN)+Act fused groups.
