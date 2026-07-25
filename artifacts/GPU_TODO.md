# GPU_TODO — ordered checklist for the next Kaggle session

*Maintained by the compiler side; the human executes top to bottom on a Kaggle
T4 notebook (see `tools/kaggle_notebook.md` for copy-paste cells) and commits
the artifacts back. When this file is empty, Phase 2 GPU work is done.*

## Run 1 — bring-up + tune + first honest benchmark (blocks Phase 2 sign-off)

- [ ] 1. Run `bash tools/kaggle_run.sh` on a T4 notebook. It will:
      - verify NVRTC/driver load and print `Tesla T4 sm_75`,
      - export the six ONNX files (3 models × batch {1,8}),
      - run `pytest -q -m gpu` (first-ever device validation of the Phase-1
        kernels: depthwise, reduce, pool, views, batched conv offsets,
        pooled-memory executor),
      - fill `artifacts/tune_cache.json` (≤64 configs/shape, median-of-50,
        noisy configs discarded),
      - run the benchmark suite (3 models × batch {1,8} × {mdlc, torch eager,
        ORT CUDA EP}) into `artifacts/gpu_run_<ts>/bench.json`,
      - regenerate `docs/BENCHMARKS.md` from that artifact.
- [ ] 2. Bring back (zip or push): `artifacts/gpu_run_<ts>/` (all JSON/MD/TXT),
      `artifacts/tune_cache.json`, `docs/BENCHMARKS.md`.
- [ ] 3. If `pytest -m gpu` failed anywhere, include `pytest_gpu.txt` —
      expected first-run risk spots (pre-empted but unproven on device):
      NVRTC arch flag, launch-config bounds >1024 threads, pooled-offset
      binding, per-image `in_offset/out_offset` pointer math, grouped-conv
      weight-row slices.

## Nothing else is GPU-blocked right now

INT8/DP4A (Phase 3) will add: AIMET quantization of RepViT (needs torch+aimet
on Kaggle), `pytest -m gpu` for DP4A kernels, and an int8-vs-fp32 latency row.
Those items land here once the CPU-side code exists.
