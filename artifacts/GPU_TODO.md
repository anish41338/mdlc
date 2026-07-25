# GPU_TODO — ordered checklist for the next Kaggle session

*Maintained by the compiler side; the human executes top to bottom on a Kaggle
T4 notebook (see `tools/kaggle_notebook.md` for copy-paste cells) and commits
the artifacts back. When this file is empty, Phase 2 GPU work is done.*

## Run 1 — DONE (2026-07-25, Kaggle **Tesla T4, sm_75**)

`artifacts/gpu_run_20260725_162618/`. **Phase 2's device gate is met**: the
kernels execute on real hardware and agree with the oracle.

- [x] 1. `bash tools/kaggle_run.sh` completed end to end.
- [x] 2. `pytest -q -m gpu` — **24/24 passed** on the T4, and again on a P100
      (sm_60) earlier the same day: depthwise, grouped conv, batch-N conv
      offsets, reduce/pool/views, the pooled-memory executor, and MobileNetV2 +
      RepViT end-to-end against the ORT golden. None of the pre-flagged risk
      spots (NVRTC arch flag, >1024-thread launches, pooled-offset binding,
      per-image pointer math, grouped weight-row slices) actually failed.
- [x] 3. Autotuning filled the cache on both architectures: 55 GEMM shapes +
      17 depthwise sigs = **72 entries** per arch, keyed by `sm_arch`.
- [x] 4. Benchmark matrix ran; `docs/BENCHMARKS.md` regenerated. Numbers in
      docs/STATUS.md and docs/GAPS.md §11.

### Still to commit from that run

The artifacts live in the Kaggle zip, not yet in git:
`artifacts/gpu_run_20260725_162618/` (bench.json, tune.json, logs,
BENCHMARKS.md), `artifacts/tune_cache.json`, `docs/BENCHMARKS.md`. Commit them
so every number in the docs is backed by a file in the repo.

## Run 2 — OPTIONAL, buys exactly one thing: the ORT-CUDA baseline column

Nothing is blocked on this. Both Run-1 attempts hit the same wall: the PyPI
`onnxruntime-gpu` wheel is built against **CUDA 13** (`libcudart.so.13`) and
the Kaggle image ships CUDA 12, so the CUDA EP never loaded — every `ort-cuda`
row is an error and the torch-eager comparison is the only baseline we have.

`tools/kaggle_run.sh` now tries three things in order, verifying each by
*running* a CUDA session (ORT silently falls back to CPU otherwise, and a CPU
timing recorded as "ort-cuda" would be a lie):

1. the default wheel (works if the image ever ships cudart 13),
2. the CUDA-12 build from ORT's official feed, run against the CUDA 12 runtime
   that torch's bundled `nvidia-*` wheels already provide (`LD_LIBRARY_PATH` is
   pointed at them),
3. the CPU build — the parity oracle is mandatory, the baseline column is not.

`bench_ort_cuda` now asserts the CUDA EP is actually bound before timing.

Re-run only if you want that column. Everything else from Run 1 stands.

## Phase 3 (INT8/DP4A) — when the CPU-side code is ready

Will add: AIMET quantization of RepViT (needs torch+aimet on Kaggle),
`pytest -m gpu` coverage for the DP4A kernels, and an int8-vs-fp32 latency row.
**Requires sm_61+ for `__dp4a`** — a T4 (sm_75) is fine, a P100 (sm_60) is not.
