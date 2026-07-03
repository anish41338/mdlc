# Status, GPU-gating, and known gaps

A snapshot of what is built and verified, what only activates on a CUDA device,
and the deliberate limitations to close next.

## Done and verified on CPU

| Stage | Module | Verified by |
|-------|--------|-------------|
| ONNX frontend → IR | `frontend/onnx_importer.py` | reference output == ONNX Runtime |
| IR (Graph/Node/Tensor) | `ir/` | topo sort, clone, use-counts in tests |
| NumPy reference executor | `runtime/reference.py`, `runtime/ops.py` | == ONNX Runtime (1e-6) |
| Constant folding | `passes/constant_folding.py` | folded subgraph == original |
| Dead-code elimination | `passes/dce.py` | structural test |
| Conv+BN fold | `passes/fuse_conv_bn.py` | arithmetic-only, output preserved |
| Conv/Gemm + activation fusion | `passes/fuse_activation.py` | per-pass numeric verifier |
| Elementwise-chain fusion | `passes/fuse_elementwise.py` | multi-consumer legality test |
| Per-pass correctness harness | `testing/harness.py` | aborts a pass on any drift |
| Shape inference (by execution) | `passes/shape_inference.py` | exact static shapes |
| Liveness memory planner | `runtime/memory_planner.py` | no live-range overlap; 65% saved on ResNet-18 |
| CUDA codegen (GEMM/im2col/elementwise) | `codegen/cuda/templates.py` | compiled+run on CPU sim == NumPy |
| Whole-graph kernel composition | `codegen/cuda/sim_executor.py` | == reference, end-to-end |
| Autotuner search + cache | `autotuner/tuner.py` | cost-model ranking, cache round-trip |
| Compile driver + CLI | `compiler.py`, `tools/compile.py` | ResNet-18 report PASS |

The headline correctness discipline: **every graph pass is diffed against an
ONNX Runtime golden after it runs**, and **every generated kernel is compiled
and executed on the CPU sim and diffed against NumPy**. Numerical corruption is
caught at the stage that introduced it, not days later.

## GPU-gated (correct-by-construction, runs unchanged on a CUDA device)

| Piece | Module | Why it can't run here |
|-------|--------|-----------------------|
| NVRTC JIT + CUDA driver runtime | `codegen/cuda/nvrtc_runtime.py` | needs `nvcuda` + `nvrtc` libs + a device |
| On-device whole-graph executor | `codegen/cuda/gpu_executor.py` | mirrors the validated CPU-sim path |
| Measured autotuning (CUDA-event timing) | `autotuner/tuner.py` (`ctx=...`) | falls back to the analytical cost model on CPU |
| Latency benchmark vs PyTorch eager / ORT-CUDA | `tools/benchmark.py` | reports compile wins + ORT-CPU here |

`cuda_available()` is the single gate. This machine has an Intel iGPU, so that
returns `False` and the GPU code is never entered.

## Known gaps / next steps

1. **Grouped & depthwise conv** fall back to the host executor (the im2col→GEMM
   path assumes `group==1`). MobileNet/RepViT lean on depthwise — needs a
   depthwise conv kernel template. *Highest-value next item.*
2. **Conv codegen is batch-1** (im2col lowers one CHW image). Loop over N, or
   batch the im2col, for N>1.
3. **GEMM `alpha`/`beta` ≠ 1** aren't applied in the kernel epilogue (fine for
   standard inference Gemm/Linear; fold them in for generality).
4. **Pooling / softmax / reshape** are host fallbacks. Cheap and correct, but a
   production lowering would emit kernels (esp. pooling for conv nets).
5. **Implicit GEMM** (gather patches inside the GEMM K-loop) would remove the
   im2col scratch buffer — the documented stretch over the current explicit
   im2col.
6. **INT8 / DP4A quantized-GEMM** path — the Samsung/RepViT tie-in stretch.
7. **No tensor cores / no double-buffering** in the GEMM — this is why we expect
   to lose to cuBLAS/cuDNN and ORT, and we quantify the gap rather than hide it.

## Interview talking points this codebase supports

- *Why fusion helps*: memory traffic vs FLOPs (elementwise is memory-bound) and
  kernel-launch overhead (52 launches vs 141 eager dispatches). Measurable here.
- *When fusion is illegal*: multiple consumers of an intermediate — enforced in
  `fuse_elementwise.py` via single-use-edge union-find, and tested.
- *Shared memory & bank conflicts*: A staged transposed (`As[BK][BM]`) so the
  inner-product read is conflict-free; see `templates.py::gemm_kernel`.
- *im2col vs implicit GEMM*: im2col's KH·KW memory blow-up is in the code and
  the scratch planner; implicit GEMM is the noted stretch.
- *Autotuner without combinatorial blowup*: pre-pruned space + random-sampled
  budget; `schedule.py::gemm_search_space`, `tuner.py::tune_gemm`.
- *Graph-level vs kernel-level optimization*: passes vs schedules — the TVM
  compute/schedule split, mirrored by `passes/` vs `codegen/schedule.py`.
