# Status, GPU-gating, and known gaps

## Phase 2 (2026-07-25) — it runs on a GPU, and here is what it costs

Kaggle **Tesla T4 (sm_75)**, driver 580.159.04, CUDA 12.8, torch 2.10.0+cu128.
Artifact: `artifacts/gpu_run_20260725_173804/`, generated at git `b1d581f`.
Every number below is reproduced by `docs/BENCHMARKS.md`, which
`mdlc.tools.gen_benchmarks` writes from `bench.json` — none are hand-typed.

The device gate is met: **`pytest -m gpu` 24/24** on the T4 (and again on a
P100, sm_60) — depthwise, grouped conv, batch-N offsets, reduce/pool/views, the
pooled allocator, and MobileNetV2 + RepViT end-to-end vs the ORT golden.

Measured autotuning: 55 GEMM shapes + 17 depthwise signatures, **72 cache
entries** (all `sm_75`), ≤64 configs each, median-of-50 CUDA-event samples,
noisy configs discarded by an IQR gate. Best GEMM win **70.9%** over the naive
default (`512x49x4608`); depthwise wins 1.7–26.8%. The shape of the result is
what an honest tuner looks like: big wins on tall-skinny classifier GEMMs, and
~0% where the default was already right (`576x196x96`: 0.0%).

Latency, median ms over 200 iterations after 50 warmups, H2D/D2H excluded,
CUDA-event timed, vs PyTorch eager (`cudnn.benchmark=True`) on the same card:

| Model | b1 mdlc | b1 torch | b1 result | b8 mdlc | b8 torch | b8 result |
|---|---|---|---|---|---|---|
| MobileNetV2 | **2.303** | 5.746 | **2.50× faster** | 13.882 | 7.743 | 1.79× slower |
| RepViT-m0_9 | **5.186** | 7.259 | **1.40× faster** | 31.046 | 8.245 | 3.77× slower |
| ResNet-18 | 4.400 | 2.728 | 1.61× slower | 35.822 | 9.559 | 3.75× slower |

**Every one of those latencies is parity-checked on the device**: each mdlc row
carries `parity_max_abs_vs_ort` against an ORT golden for the same seeded
inputs — 1.5e-06 to 5.7e-06 across all six — with `host fallbacks: none`. A
fast wrong answer is not a result, so the timing and the check ship together.

Fewer launches, less memory (both from the same artifact), batch-1:

| Model | mdlc launches | torch launches | mdlc device MB | torch device MB |
|---|---|---|---|---|
| ResNet-18 | 51 | 91 | 110.9 | 1193.2 |
| MobileNetV2 | 99 | 153 | 35.8 | 971.0 |
| RepViT-m0_9 | 255 | 290 | 45.2 | 714.6 |

(The memory columns use each system's own accounting — ours is exact allocation
bookkeeping, torch's is `max_memory_allocated` over a caching allocator — so
read the order of magnitude, not the digits. The planner-managed activation
pool is the precise number we control: 4.82 MB for ResNet-18 b1.)

This is the result the design predicted, in both directions. **We win batch-1
on the depthwise-heavy mobile models** — exactly where fusion and launch-count
reduction pay and where cuDNN's conv kernels have least to work with. **We lose
ResNet-18**, whose dense 3×3 convs are what cuDNN is most tuned for, and **we
lose every batch-8 row by 1.8–3.8×** because batched conv still issues *N
separate im2col+GEMM launch pairs per conv* (GAPS §9.4) — visible directly in
the launch counts (ResNet-18 b1 51 → b8 331). That is a known, documented,
Phase-4 gap — column-batching or implicit GEMM — now with a price tag on it
rather than a guess.

**Baseline caveat, stated plainly:** eager is the *soft* comparison — it pays
Python dispatch per op, precisely the overhead a compiler exists to remove, so
beating it is necessary but not sufficient. The ORT CUDA EP column is empty
(its wheel targets CUDA 13, the image ships CUDA 12) and a `torch.compile`
(Inductor) column is now wired into the harness but not yet run. Until both
land, treat "2.50× faster than eager" as the honest but weakest-baseline claim
that it is. Optional Run 2 in `artifacts/GPU_TODO.md` collects them.

## Phase 2 pre-flight (2026-07-25) — device path made executable

First `pytest -m gpu` attempt on a Kaggle P100 proved the device path had never
run a kernel. Fixed, with CPU-detectable guards so the class cannot recur
(full write-up in [GAPS.md](GAPS.md) "Phase 2 pre-flight"):

- `tests/test_gpu.py` — 24 device tests (the `gpu` marker previously matched
  **nothing**, so `pytest -m gpu` exited 5 "no tests collected").
- Emitted CUDA is now **self-contained**: NVRTC has no include search path, so
  `#include <math_constants.h>` was fatal on device while the sim silently
  stripped it. `tests/test_nvrtc_compat.py` asserts the invariant on CPU.
- **`transA`/`transB` are now resolved inside the GEMM kernel.** They had been
  applied *only* host-side by the simulator, so the sim matched the reference
  while the GPU path computed something different — wrong logits for every
  `nn.Linear` head (all three target models). The transpose is baked into the
  load indices and the sim's host-side transpose is gone, so both executors
  now hand the kernel byte-identical buffers.
- ctypes driver layer: explicit `argtypes` (32-bit `int` was standing in for
  `size_t`) and `_v2` entry-point resolution.
- Exporters pin the legacy TorchScript ONNX exporter (`dynamo=False`); newer
  torch defaults to a dynamo path that targets opset 18 and then fails
  converting down to our frozen 17.

A snapshot of what is built and verified, what only activates on a CUDA device,
and the deliberate limitations to close next. The exhaustive measured gap
inventory (per-model fallbacks, RepViT op set, audit findings) lives in
[GAPS.md](GAPS.md).

## Phase 1 (2026-07-04) — total op coverage: zero host fallbacks on all 3 models

- **ResNet-18, MobileNetV2, RepViT-m0_9 all compile with `host fallbacks:
  none`** and pass ORT parity (per-model launch tables in GAPS.md).
- Direct depthwise conv kernel: `blockIdx.z = n·C+c`, shared-memory input
  tile with halo, staged per-channel filter, fully-unrolled K×K, fused
  bias+activation epilogue, channel multiplier, `direct_load` tuner variant.
  125 sim-parity tests (96-config cross-product + 25 randomized + fused
  emit-path at N∈{1,2,8} and mult=2).
- Batch-N conv fixed: per-image im2col+GEMM launch pairs with explicit
  element offsets; N∈{1,2,8} permanent in conv parity tests.
- Elementwise codegen generalized: NumPy-style broadcasting via baked
  stride-0 index expressions (SE's (N,C,1,1)⊙(N,C,H,W) gate), non-scalar
  constants as buffers, Clip bounds from inputs, Erf; lone elementwise ops
  lower through the same emitter — never to host.
- New kernels: spatial-mean reduction (one block per (n,c), deterministic
  smem tree — GlobalAveragePool + ReduceMean[2,3]), MaxPool; Flatten/Reshape/
  Squeeze/Unsqueeze/Identity are metadata-only views (buffer aliases).
- General grouped conv (1<g<C_in): per-(image,group) im2col+GEMM slices on
  the device path.
- cpu_sim: content-addressed exe cache (~/.cache/mdlc-sim) — whole-model
  simulation and 100+-kernel parity sweeps became tractable.
- GPU executor updated for all new launch kinds + offsets (device validation
  is the Phase 2 gate).

## Phase 0 (2026-07-04) — ground-truth audit: complete

- Headline ResNet-18 numbers (141→32, 12 kernels, 65% memory) reproduced from
  commands; MobileNetV2 and RepViT-m0_9 exporters added (`tools/build_*.py`).
- **Fixed a silent-corruption bug**: Clip/ReLU6 fused with unresolvable bounds
  became identity on MobileNetV2 — caught only because the audit probed the
  oracle's power. Constant nodes now fold to initializers; unresolvable
  activation params block fusion; the harness fails **vacuous** comparisons
  (golden ≈ atol); exporters LSUV-calibrate so every layer keeps O(1) signal.
- Structural graph verifier (`mdlc/ir/verify.py`) runs after every pass in any
  verified pipeline: dangling edges, duplicate producers, cycles, orphans,
  initializer/shape drift. All five passes proven idempotent in tests.
- Tolerance contract frozen in `mdlc/testing/tolerances.py` (4 tiers, all call
  sites migrated, only tightenings).
- `compile --report` now prints the named host-fallback list per model.
- Corrected fact: conv codegen works at N=1 and breaks at N>1 (not the
  reverse); Phase 1 adds N∈{1,2,8} parity tests permanently.

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

## Known gaps / next steps (measured inventory in GAPS.md §9)

1. ~~Grouped & depthwise conv host fallback~~ **closed** (direct depthwise
   kernel; per-group slices for general grouped) — Phase 1.
2. ~~Conv codegen batch-1 only~~ **closed** (per-image launch pairs; N∈{1,2,8}
   tested) — Phase 1. Batched-N still costs N× launches; implicit GEMM or
   column-batching is the Phase 4 fix.
3. **GEMM `alpha`/`beta` ≠ 1** aren't applied in the kernel epilogue (fine for
   standard inference Gemm/Linear; fold them in for generality).
4. ~~Pooling / reshape host fallbacks~~ **closed** (reduce/pool kernels,
   views). Softmax/Transpose/Concat/Pad remain host-listed — no target model
   hits them; any report that does will say so.
5. **Memory-planner alignment**: pooled offsets need a 256-byte guarantee
   before GPU bring-up (Phase 2).
6. **Implicit GEMM** (gather patches inside the GEMM K-loop) would remove the
   im2col scratch buffer — the documented stretch over the current explicit
   im2col.
7. **INT8 / DP4A quantized-GEMM** path — the AIMET/RepViT tie-in (Phase 3).
8. **No tensor cores / no double-buffering** in the GEMM — this is why we expect
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
