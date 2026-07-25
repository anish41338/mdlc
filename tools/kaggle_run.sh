#!/usr/bin/env bash
# mdlc GPU validation + tuning + benchmark run (Kaggle T4, sm_75).
# Usage (in a Kaggle notebook cell, GPU accelerator ON):
#   !git clone https://github.com/<owner>/mdlc && cd mdlc && bash tools/kaggle_run.sh
# Everything lands in artifacts/gpu_run_<timestamp>/ — commit that directory
# (JSON/MD only) plus artifacts/tune_cache.json back to the repo.
set -euxo pipefail

TS=$(date +%Y%m%d_%H%M%S)
OUT="artifacts/gpu_run_${TS}"
mkdir -p "$OUT"

# ---- 0. environment ---------------------------------------------------------
nvidia-smi | tee "$OUT/nvidia_smi.txt"
python -m pip install -q -e ".[ref,dev]"
python -m pip install -q timm onnxscript   # torch/torchvision preinstalled on Kaggle
python -m pip install -q onnxruntime-gpu || true

# onnxruntime-gpu wheels are built against a specific CUDA runtime: the current
# one wants libcudart.so.13, which this image does not ship, so `import
# onnxruntime` dies with an ImportError. That oracle is what every parity check
# and per-pass verifier uses, so fall back to the CPU build rather than run
# unverified. Costs the ORT-CUDA baseline column (recorded as an error row),
# never the correctness of our own numbers.
if ! python -c "import onnxruntime" 2>/dev/null; then
    echo "WARNING: onnxruntime-gpu unusable here; falling back to CPU onnxruntime"
    python -m pip uninstall -q -y onnxruntime-gpu || true
    python -m pip install -q --force-reinstall onnxruntime
    python -c "import onnxruntime as o; print('onnxruntime', o.__version__, o.get_available_providers())"
fi
python - <<'EOF'
from mdlc.codegen.cuda.nvrtc_runtime import cuda_available, CudaContext
assert cuda_available(), "CUDA driver/NVRTC not found - is the GPU accelerator on?"
ctx = CudaContext()
print("device:", ctx.device_name(), ctx.arch)
EOF

# ---- 1. export the target models (fixed batch 1 and 8) ----------------------
python -m mdlc.tools.build_resnet18
python -m mdlc.tools.build_mobilenetv2
python -m mdlc.tools.build_repvit
python -m mdlc.tools.build_resnet18    --out examples/resnet18_b8.onnx     --batch 8
python -m mdlc.tools.build_mobilenetv2 --out examples/mobilenetv2_b8.onnx  --batch 8
python -m mdlc.tools.build_repvit      --out examples/repvit_m0_9_b8.onnx  --batch 8

# ---- 2. GPU test suite -------------------------------------------------------
python -m pytest -q -m gpu 2>&1 | tee "$OUT/pytest_gpu.txt"

# ---- 3. autotune every shape the models hit (fills the committed cache) -----
python -m mdlc.tools.gpu_tune --budget 64 \
  --cache artifacts/tune_cache.json \
  --report "$OUT/tune.json" 2>&1 | tee "$OUT/tune_log.txt"

# ---- 4. benchmark matrix ------------------------------------------------------
python -m mdlc.tools.benchmark --suite --out "$OUT" \
  --tune-cache artifacts/tune_cache.json 2>&1 | tee "$OUT/bench_log.txt"

# ---- 5. regenerate the doc from the artifact ---------------------------------
python -m mdlc.tools.gen_benchmarks "$OUT/bench.json"
cp docs/BENCHMARKS.md "$OUT/BENCHMARKS.md"

echo "DONE. Download/commit: $OUT (JSON/MD/TXT) + artifacts/tune_cache.json + docs/BENCHMARKS.md"
