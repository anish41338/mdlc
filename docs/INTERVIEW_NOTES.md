# Interview defense notes

Appended per phase. Rule: nothing goes on the resume that can't survive five
minutes of follow-up questions; each entry records the decision, the rejected
alternative, and the number/formula that anchors it.

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
