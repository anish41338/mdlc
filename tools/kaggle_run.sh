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

# ---- ONNX Runtime: the CPU oracle is mandatory, the CUDA EP best-effort ----
# The PyPI onnxruntime-gpu wheel is built against CUDA 13 (needs
# libcudart.so.13); Kaggle images ship CUDA 12, so `import onnxruntime` dies.
# But torch's bundled nvidia-* wheels carry the full CUDA 12 runtime (cudart,
# cublas, cudnn, cufft, ...) — not on the loader path by default. Expose them,
# then try wheels in order, verifying each by *running* a CUDA session (ORT
# lists/falls back silently, so only a bound session proves anything).
NVIDIA_LIBS=$(python - <<'PY'
import glob, os, sysconfig
site = sysconfig.get_paths()["purelib"]
print(":".join(sorted(glob.glob(os.path.join(site, "nvidia", "*", "lib")))))
PY
)
export LD_LIBRARY_PATH="${NVIDIA_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

ort_cuda_ok() {
    python - <<'PY'
import sys
try:
    import numpy as np
    from onnx import TensorProto, helper
    import onnxruntime as ort
    g = helper.make_graph(
        [helper.make_node("Identity", ["x"], ["y"])], "probe",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 9
    sess = ort.InferenceSession(m.SerializeToString(),
                                providers=["CUDAExecutionProvider"])
    assert sess.get_providers()[0] == "CUDAExecutionProvider", sess.get_providers()
    sess.run(None, {"x": np.zeros(1, np.float32)})
    print("ORT CUDA EP OK:", ort.__version__)
except Exception as e:
    print("ORT CUDA EP unavailable:", type(e).__name__, e)
    sys.exit(1)
PY
}

# Attempt 1: default PyPI wheel (CUDA 13 — works only if the image has cudart 13).
python -m pip install -q onnxruntime-gpu || true
if ! ort_cuda_ok; then
    # Attempt 2: the CUDA-12 build from the official ORT feed, running against
    # torch's bundled nvidia libs exposed above. --no-deps so pip cannot churn
    # numpy/protobuf; ORT's own small deps are ensured explicitly.
    echo "trying the CUDA-12 onnxruntime-gpu build"
    python -m pip uninstall -q -y onnxruntime-gpu onnxruntime || true
    python -m pip install -q coloredlogs flatbuffers || true
    python -m pip install -q --no-deps onnxruntime-gpu \
      --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/ || true
fi
if ! ort_cuda_ok; then
    # Fallback: CPU build. The oracle behind every parity check must exist —
    # running unverified is not an option. Costs only the ORT-CUDA baseline
    # column, which the benchmark records as an error row.
    echo "WARNING: no CUDA-capable onnxruntime; installing the CPU build (oracle only)"
    python -m pip uninstall -q -y onnxruntime-gpu onnxruntime || true
    python -m pip install -q --no-deps --force-reinstall onnxruntime
fi
python -c "import onnxruntime as o; print('onnxruntime', o.__version__, o.get_available_providers())"
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
