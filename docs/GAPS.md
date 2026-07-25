# GAPS — measured gap inventory

Everything below was measured by running commands in this repo, not copied
from docs. Reproduce with the commands shown. Items get struck through only
when a test proves them closed.

## Phase 1 result (2026-07-04): zero host fallbacks on all three models

`python -m mdlc.tools.compile <model> --report` now prints
**`host fallbacks: none`** for ResNet-18, MobileNetV2, and RepViT-m0_9, with
ORT parity PASS on non-vacuous logits:

| Model | Launches (by kind) | Parity vs ORT |
|---|---|---|
| ResNet-18 | elementwise×8 gemm×21 im2col×20 pool×1 reduce×1 view×1 | max_abs 5.2e-6 |
| MobileNetV2 | depthwise×17 elementwise×10 gemm×36 im2col×35 reduce×1 view×1 | max_abs 2.1e-6 |
| RepViT-m0_9 | depthwise×26 elementwise×63 gemm×78 im2col×77 reduce×11 | max_abs 1.7e-6 |

Closed by Phase 1 (each with sim-executed parity tests on the exact emitted
source): batch-N conv (§4 below — N∈{1,2,8} permanent in tests), direct
depthwise kernel with fused BN+act epilogue (125-case matrix), general grouped
conv per-group slices, broadcast elementwise codegen (per-input stride
vectors), singleton elementwise lowering, Erf/GELU chain fusion, ReduceMean +
GlobalAveragePool reduction kernel (deterministic smem tree), MaxPool kernel,
Flatten/Reshape/Squeeze/Unsqueeze/Identity as metadata-only views, Erf +
ReduceMean in the reference executor.

## Phase 2 pre-flight (2026-07-25): three device-path bugs found *without* a GPU

The first real `pytest -m gpu` attempt on a Kaggle P100 (sm_60) exposed that the
device path had never executed a single kernel. Three defects, each invisible to
the CPU simulator by construction:

1. **No `gpu` marker existed.** `pytest -m gpu` collected 0 tests and exited 5
   — so "written but not yet device-validated" (§9.5) was optimistic: there was
   nothing to run. `tests/test_gpu.py` now carries 24 device tests mirroring the
   sim suite (depthwise, grouped, batch-N offsets, reduce/pool/view, pooled
   allocator, both target models); the marker is registered in `pyproject.toml`.
2. **Emitted source was not self-contained.** Four templates opened
   `#include <math_constants.h>`. NVRTC compiles a string with no include search
   path, so this is a `catastrophic error` on device — while the sim compiled
   happily, because its shim *textually strips that line* and defines the
   constant itself. Templates now carry a `PREAMBLE` defining `CUDART_INF_F` as
   `__int_as_float(0x7f800000)` (verbatim how the toolkit header defines it).
   `tests/test_nvrtc_compat.py` asserts self-containment on CPU.
3. **`transA`/`transB` were silently ignored on device.** `_lower_gemm` recorded
   them in meta, but the emitted kernel assumed row-major `A[MxK] @ B[KxN]`, and
   **only `sim_executor` compensated — by transposing the operands host-side.**
   The sim therefore matched the reference while the GPU path computed a
   different result, with no test able to see it. Since every torch
   `nn.Linear` exports as `Gemm(transB=1)`, this was wrong output for the
   classifier head of all three target models. Fixed at the right level: the
   transpose is baked into the kernel's load indices
   (`B[gc * K + gr]`), costs nothing at runtime, works for activations as well
   as weights, and the sim's host-side transpose is *removed* so both
   executors now feed the kernel identical buffers. `trans_a`/`trans_b` are
   part of the kernel-cache key. Pinned by
   `test_gemm_kernel_handles_transposed_operands` (transA/transB/both) and
   `test_transposed_gemm_emits_distinct_index_math`.

Method note: (2) and (3) are both cases of *the simulator being more capable
than the device* — a host harness silently repairing something the GPU cannot.
That asymmetry, not kernel math, is where the device-path bugs were. The
`test_nvrtc_compat.py` guards (no includes, launch dims are plain `int`,
≤1024 threads/block, y/z grid ≤65535) exist to keep that class CPU-detectable.

Also hardened while auditing (unvalidated-on-device code, no behaviour change
on CPU): the ctypes driver layer now declares `argtypes`/`restype` rather than
letting every undeclared argument default to C `int` (`size_t` copy/alloc sizes
were being passed as 32-bit), and resolves the `_v2` entry points that the CUDA
headers `#define` over `cuMemAlloc`/`cuMemcpy*`/`cuCtxCreate` — dlsym bypasses
those macros and would otherwise bind the legacy 32-bit-size ABI.

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

## 4. Corrected fact: the batch gap was N>1, not N=1 — CLOSED in Phase 1

The build brief said "batch-1 conv codegen broken". Measured truth (STATUS.md
had it right): batch-1 worked end-to-end; N>1 crashed — the im2col→GEMM
lowering emits P = OH·OW columns for a single image, so at N=2 the output
reshape failed. **Fixed**: the plan now carries one im2col+GEMM launch pair
per image with explicit element offsets (`batch_index`/`in_offset`/
`out_offset` in launch meta); N∈{1,2,8} is a permanent axis of the conv
parity tests (`test_conv_models_batch_n`, depthwise cross-product).

## 5. Host-fallback inventory — ALL CLOSED (Phase 1)

The Phase-0 inventory (ResNet-18: MaxPool/GAP/Flatten ×3; MobileNetV2:
17 depthwise convs + 10 bare residual Adds + GAP + Flatten = 29; RepViT:
blocked at the reference executor on Erf/ReduceMean, then depthwise +
broadcast-Mul + singleton elementwise in codegen) is fully burned down — see
the Phase 1 result table at the top. Ops with *no* model driving them remain
honestly host-listed in `emit.HOST_FALLBACK` (Transpose, Concat, Pad,
Softmax, AveragePool, exotic ReduceMean axes, MaxPool with ceil_mode) and
would print in any report that hits them.

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

## 9. Remaining measured gaps (Phase 2+ owners)

1. GEMM `alpha`/`beta` ≠ 1 still not applied in the kernel epilogue (fine for
   torch exports, which emit 1.0) — but **no longer silent**: as of the Phase 2
   pre-flight, `_lower_gemm` raises `NotImplementedError` instead of emitting a
   kernel that drops them (`test_gemm_rejects_unapplied_alpha_beta`). Folding
   them into the weights/bias is the real fix.
2. ~~Memory planner has no alignment guarantee~~ **closed** — `ALIGN = 256` in
   `runtime/memory_planner.py`, `GpuExecutor` asserts every pooled offset is
   256-byte aligned before binding, and the pooled path is covered on device by
   `test_generated_module_matches_reference_pooled_on_device`.
3. ~~Autotuner ranks by cost model only; depthwise knobs unsearched~~
   **closed** — measured tuning ran on T4 (sm_75) and P100 (sm_60): 55 GEMM
   shapes + 17 depthwise sigs, 72 entries per arch. Depthwise *is* searched
   (`tune_depthwise`, tile_h/w × smem-vs-direct). CPU-side compiles still use
   the cost model by design, and now prefer a committed measured entry.
4. Batched conv costs N× launches (per-image im2col+GEMM pairs) — correct but
   launch-bound. **Now priced**: batch-8 is 1.85–3.79× slower than torch eager
   across the three models, versus winning two of three at batch-1 (STATUS.md).
   Batching the im2col columns (P = N·OH·OW with an output permute) or implicit
   GEMM is the Phase 4 fix, and this is the number it has to beat.
5. ~~GPU executor's new-kind handling not device-validated~~ **closed** —
   `pytest -m gpu` 24/24 on T4 and P100 (2026-07-25). None of the pre-flagged
   risk spots failed; the real bugs were found by static audit beforehand
   (see "Phase 2 pre-flight" above), not by the device run.
6. The reduction kernel uses a shared-memory tree, not warp shuffles: the
   cpu-sim shim has no warp-lockstep to emulate `__shfl_down_sync`. Portable
   and deterministic; a shuffle variant is a labeled Phase-4 tuner option.
7. No CUDA-graph/stream work; launches are serial (fine — batch-1 latency
   story is fusion + fewer launches).
8. INT8/DP4A path not started (Phase 3). Needs sm_61+ for `__dp4a`, so it can
   be validated on a T4 but not on a P100.
9. **No ORT-CUDA baseline yet.** Both 2026-07-25 runs failed to load the CUDA
   EP: the PyPI `onnxruntime-gpu` wheel is built against CUDA 13
   (`libcudart.so.13`) and the Kaggle image ships CUDA 12. Every `ort-cuda` row
   is an error, so torch eager is currently the only baseline. `kaggle_run.sh`
   now tries the CUDA-12 ORT build against torch's bundled `nvidia-*` runtime
   before falling back to CPU, and `bench_ort_cuda` asserts the EP is actually
   bound (ORT otherwise falls back to CPU silently — a CPU timing labelled
   "ort-cuda" would be a fabricated baseline).

## 11. Phase 2 measured results (T4, sm_75, run 20260725_162618)

`pytest -m gpu` 24/24. Tuning: 72 entries/arch, best GEMM win 70.9%
(`512x49x4608`), depthwise 1.7–26.8%, `576x196x96` won 0.0% (default already
optimal — the tuner is not fitting noise).

Latency (median ms) vs PyTorch eager, same card:

| Model | b1 mdlc | b1 torch | b8 mdlc | b8 torch |
|---|---|---|---|---|
| MobileNetV2 | **2.218** | 5.738 | 13.897 | 7.529 |
| RepViT-m0_9 | **5.049** | 7.226 | 30.744 | 8.134 |
| ResNet-18 | 4.440 | 2.748 | 35.749 | 9.438 |

Reading it honestly: we beat eager 2.59× (MobileNetV2) and 1.43× (RepViT) at
batch-1 — the depthwise-heavy mobile graphs where fusion and launch-count
reduction pay. We lose ResNet-18 at 1.62×, whose dense 3×3 convs are cuDNN's
best case, and we lose every batch-8 row by 1.85–3.79× to the per-image launch
gap (§9.4). No tensor cores, no double-buffering, no CUDA graphs — §9.7 and the
Phase-4 list are the reasons, and they are now quantified rather than asserted.

## 10. Phase 1 acceptance — MET

`compile --report` prints `host fallbacks: none` for all three models (table
at top); conv parity tests run N∈{1,2,8} permanently; RepViT and MobileNetV2
pass the reference executor vs ORT at 224² and the sim executor (exact
emitted kernels) at reduced resolution within FP32_NETWORK; the fusion path
emits depthwise kernels with fused BN(+bias)+activation epilogues
(`test_depthwise_fused_emit_path`, MobileNetV2 ReLU6-on-depthwise asserted in
`test_mobilenetv2_sim_end_to_end`).
