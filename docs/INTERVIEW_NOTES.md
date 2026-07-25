# Interview defense notes

Appended per phase. Rule: nothing goes on the resume that can't survive five
minutes of follow-up questions; each entry records the decision, the rejected
alternative, and the number/formula that anchors it.

## Phase 1 — total conv/op coverage (2026-07-04)

### Why depthwise conv gets a direct kernel, with the arithmetic

im2col rewrites conv as GEMM by materializing a (C·KH·KW, OH·OW) patch
matrix — KH·KW=9× the input's memory for a 3×3. That trade buys you a GEMM
whose K-dimension (C·KH·KW) amortizes the traffic across C_out outputs. For
*depthwise*, each output channel reads exactly ONE input channel: K collapses
to KH·KW=9 and there is no cross-channel reuse to amortize — you'd pay the 9×
blow-up for a skinny GEMM that's pure memory-bound anyway. The direct kernel
reads each input element once into shared memory (tile + halo,
(T−1)·s+(K−1)·d+1 per dim), so arithmetic intensity is the theoretical max
for the op. Rejected alternative: routing depthwise through grouped-GEMM
(what naive frameworks do) — measured as the #1 fallback source in Phase 0.

### Kernel design decisions to defend

- `blockIdx.z = n·C_out + c_out` makes batch a grid dimension — batch-N
  needed zero new code in the depthwise path. One thread per output pixel;
  the block's channel is uniform so the K×K filter is staged once (9 floats).
- Shape specialization: every conv parameter is a baked compile-time
  constant. This is the NVRTC JIT bet: `#pragma unroll` on literal bounds
  fully unrolls K×K, bounds checks on interior tiles fold away. Cost: one
  kernel per distinct (shape, schedule) — mitigated by content dedup, and
  it's exactly what the tuner wants to specialize anyway.
- Flat 1-D blocks with tx/ty derived in-kernel: the identical source runs
  under the cpu-sim shim (which only models threadIdx.x) and NVRTC.
- `direct_load` schedule knob (skip smem, read global with per-tap bounds
  checks): at stride 2 the halo nearly doubles per dim, so staging cost can
  exceed reuse benefit — that's a *measured* question, hence a tuner knob,
  not a belief.

### Broadcast fusion = per-input stride vectors

One mechanism handles SE gates, bias patterns, and runtime scalar Clip
bounds: right-align the input shape against the output, stride:=0 on size-1
dims, and emit `sum_d ((i / ostride_d) % odim_d) * istride_d` per input. Key
legality insight: values on a broadcast branch (the (N,C,1,1) sigmoid inside
an SE fusion group) are *recomputed per output element* — pure functions make
this safe, and the redundant flops are noise next to the eliminated
intermediate tensors. What is NOT safe: a group output whose shape differs
from the fusion output (would write out-shaped data into a smaller buffer) —
the emitter checks and refuses.

### Determinism in reductions

The spatial-mean kernel: each thread grid-strides a fixed slice (fixed
accumulation order), then a fixed smem tree combine — no atomics, bit-identical
across runs (Prime Directive 8). Warp shuffles would cut the smem round-trip
but the cpu-sim shim cannot emulate warp-lockstep semantics; a portable,
sim-verifiable kernel beats a marginally faster unverifiable one at this
stage. Numbers: reduction is over H·W ≤ 50176 elements per (n,c); at 128
threads that's ≤392 sequential adds per thread + 7 tree levels.

### Views are not launches

Flatten/Reshape/Squeeze/Unsqueeze/Identity became `view` plan entries — the
output aliases the input buffer (device pointer copy, zero kernels, zero
bytes). Counting them as "launches" would inflate the fusion win we report;
the benchmark launch count excludes them. Bug class this kills: "compilers"
that dispatch a kernel to copy a tensor into a different shape.

### The batch-N fix and its honest cost

Per-image im2col+GEMM pairs with element offsets (`in_offset`/`out_offset`)
— minimal, reuses the tuned single-image GEMM shape, and the offsets are
exactly what the GPU pointer arithmetic needs. Honest cost: launches scale
with N (batch-8 ResNet-18 ≈ 8×41 conv launches), so batch-8 stays
launch-heavy until implicit GEMM (Phase 4) folds N into the GEMM's N
dimension. Batch-1 — the inference case the resume claims — is unaffected.

### RepViT compile facts (for the "walk me through your compiler" question)

463 ONNX nodes → 178 IR nodes → 255 launches, zero host. 27 GELU chains
(Div-Erf-Add-Mul-Mul) each fuse to one kernel; 10 SE gates lower as
broadcast Mul; 11 ReduceMeans hit the reduction kernel; 26 depthwise convs
hit the direct kernel; activation memory pools 42.8→2.9 MB (93.3% — SE
bottlenecks make RepViT unusually pool-friendly: many short-lived
same-shaped tensors).

## Phase 0 — ground-truth audit (2026-07-04)

### The vacuous-oracle lesson (best war story in the repo)

**What happened.** The audit found MobileNetV2 "passing" parity vs ONNX Runtime
(max_abs 9e-8) while every ReLU6 in the compiled graph had silently become
identity. Root cause chain: (1) torch exports Clip bounds as `Constant` nodes
(opset ≥ 11 moved min/max from attributes to inputs); (2) our constant folder
skipped zero-input nodes, so the bounds never became initializers; (3) the
fusion pass couldn't resolve them and recorded `None` — identity clip; (4) the
oracle never noticed because an *untrained* MobileNetV2's activations decay
~0.7× per layer through 52 layers, leaving logits at ~1e-8, far below
atol=1e-4 — `allclose(got, want)` passes for any `got` when `want ≈ 0`.

**The transferable principle.** A tolerance check has a signal-to-tolerance
ratio: comparisons only have power when `|golden| >> atol`. Test power is a
property you must *engineer*, not assume. Defenses now in the repo: the
harness fails any comparison whose golden magnitude < 100×atol ("VACUOUS"),
and exporters LSUV-calibrate random weights (rescale each Conv/Linear so
output std ≈ 1 on a seeded input — one forward pass, exact: y/s is realized
as W/s, b/s).

**Interview form.** "My compiler's correctness harness once passed a model
whose ReLU6s I had silently deleted. Here's why allclose lied to me, and the
two structural fixes." Follow-ups to be ready for: why 0.7×/layer decay
(kaiming fan_out preserves variance under ReLU assumptions untrained BN +
depthwise + ReLU6 don't satisfy); why LSUV instead of bigger init std (std
compounds multiplicatively; LSUV is closed-loop per layer).

### Fusion legality: what the Clip fix teaches

Fusing `A→B` into one node is legal only if (a) B is A's sole consumer of the
intermediate (checked via use-count), (b) the intermediate isn't a graph
output, and (c) **every parameter of B is available at compile time** — (c) is
the one this bug added. An epilogue is baked into generated source; a runtime
tensor can't be baked, so the pass must refuse, not approximate. Conv+BN fold
is different in kind: it's exact arithmetic on weights (w' = w·γ/√(σ²+ε)),
not an epilogue — same accumulation order, so it's held to the tighter
FP32_SAME_ORDER tier.

### Tolerance tiers are keyed by comparison class, not by op

What makes two *correct* computations differ is the association order of fp32
sums: none (BITWISE, e.g. im2col gather — compared bit-exact), same order
(1e-4/1e-5), one reordered reduction (tiled K-loop vs NumPy pairwise:
1e-4/1e-4), or 20+ stacked reductions vs a different backend (network tier
1e-3/1e-4, observed ResNet-18 max_rel 8.9e-5 → ~10× headroom, not more).
Anchor number: fp32 eps = 2⁻²³ ≈ 1.19e-7; a K-term sum's reassociation error
grows ~K·eps·running-magnitude, so K≈4600 (ResNet's widest im2col GEMM) gives
~5e-4 worst-case relative — the reduction tier is set at observed reality
(≤1e-4) with the theory as ceiling.

### Structural verification is a separate axis from numerical verification

A pass can preserve numerics and still corrupt the graph (orphan nodes, edge
shadowing, non-idempotence) — failures that surface far from the cause. The
verifier costs O(V+E) after each pass and proved its value on day one by
catching the orphaned Constants. Idempotence (`p(p(g)) == p(g)`) is tested for
all five passes; it's the property that makes the fixed-point pass manager
terminate for free.

### Facts to have loaded

- ResNet-18: 141→32 nodes (Identity×72 + BN×20 folded, 9 Conv+Act, 8 Add+Relu
  groups), 12 distinct kernels / 52 launches vs ~141 eager dispatches,
  activation pool 13.1→4.6 MB = 65% via liveness interval packing.
- RepViT-m0_9 deploy = 463 ONNX nodes; the compiler-relevant core: 75×
  pointwise conv, 26× depthwise 3×3, GELU only as Erf decomposition, SE =
  ReduceMean→1×1 convs→Sigmoid→broadcast Mul. No MaxPool anywhere.
- The N>1 conv bug: im2col emits (C·KH·KW, OH·OW) for one image; batching
  needs P = N·OH·OW columns (or an outer loop). Batch-1 was never broken —
  audit corrected the brief.
