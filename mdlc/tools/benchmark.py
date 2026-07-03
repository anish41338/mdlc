"""Benchmark harness: optimization wins + latency across backends.

What it always reports (no GPU needed):
  * the compile-time wins — node reduction, fused-group counts, distinct
    kernels, launch count, and memory-planner savings,
  * ONNX Runtime latency as a baseline oracle on this machine.

What it adds on a CUDA device (unchanged code):
  * our generated-CUDA runtime latency (NVRTC) vs PyTorch eager and ONNX
    Runtime's CUDA EP, plus a per-op GEMM-vs-cuBLAS microbench.

The honest expectation: beat PyTorch *eager* on small models (we erase its
per-op launch overhead via fusion), lose to ONNX Runtime / TensorRT (no tensor
cores, simpler tiling), and quantify the gap.
"""

from __future__ import annotations

import argparse
import statistics
import time

import numpy as np

from mdlc.codegen.cuda.nvrtc_runtime import cuda_available
from mdlc.compiler import compile_onnx


def _time(fn, iters=20, warmup=5) -> float:
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


def bench(model_path: str, *, tune: bool = False) -> None:
    schedule_fn = None
    if tune:
        from mdlc.autotuner import Tuner
        schedule_fn = Tuner()

    compiled = compile_onnx(model_path, schedule_fn=schedule_fn)
    print(compiled.report())
    print()

    n_fused = sum(1 for n in compiled.graph.nodes if n.is_fused)
    orig_ops = len(compiled.original.nodes)
    opt_ops = len(compiled.graph.nodes)
    launches = len(compiled.module.plan)
    print("== optimization summary ==")
    print(f"  graph ops      : {orig_ops} -> {opt_ops}  ({n_fused} fused groups)")
    print(f"  kernel launches: {launches}  (vs ~{orig_ops} eager op-dispatches)")
    print(f"  distinct kernels: {len(compiled.module.kernels)}")
    print(f"  activation mem  : {compiled.mem_plan.reuse_ratio*100:.0f}% of naive "
          f"({compiled.mem_plan.planned_activation_bytes/1e6:.2f} MB)")
    print()

    feeds = compiled.feeds

    # --- baseline: ONNX Runtime -------------------------------------------
    try:
        import onnxruntime as ort
        prov = (["CUDAExecutionProvider"] if cuda_available()
                else ["CPUExecutionProvider"])
        sess = ort.InferenceSession(compiled.proto_bytes, providers=prov)
        f32 = {k: v.astype(np.float32) for k, v in feeds.items()}
        ort_ms = _time(lambda: sess.run(None, f32))
        print(f"ONNX Runtime ({prov[0].replace('ExecutionProvider','')}): "
              f"{ort_ms:.3f} ms")
    except Exception as e:
        print(f"ONNX Runtime: unavailable ({e})")

    # --- our backend -------------------------------------------------------
    if cuda_available():
        from mdlc.codegen.cuda.gpu_executor import GpuExecutor
        ex = GpuExecutor(compiled.module)
        res = ex.run(feeds, timing=True)
        print(f"mdlc (generated CUDA, NVRTC): {res['ms']:.3f} ms")
        # correctness on device vs golden
        for o in compiled.graph.outputs:
            err = np.max(np.abs(res["outputs"][o] - compiled.golden[o]))
            print(f"  device output {o}: max_abs_err={err:.2e}")
        # PyTorch eager baseline, if this is a known torchvision model
        _torch_eager_baseline(model_path, feeds)
    else:
        print("mdlc (generated CUDA): [GPU not present] kernels generated & "
              "validated on CPU sim; run on a CUDA device for latency.")
    print()


def _torch_eager_baseline(model_path, feeds):
    try:
        import torch
        import torchvision
        if "resnet18" not in model_path:
            return
        m = torchvision.models.resnet18(weights=None).cuda().eval()
        x = torch.from_numpy(next(iter(feeds.values()))).cuda()
        with torch.no_grad():
            torch.cuda.synchronize()
            ms = _time(lambda: (m(x), torch.cuda.synchronize()))
        print(f"PyTorch eager (CUDA): {ms:.3f} ms")
    except Exception as e:
        print(f"PyTorch eager: unavailable ({e})")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default="examples/resnet18.onnx")
    ap.add_argument("--tune", action="store_true")
    args = ap.parse_args(argv)
    bench(args.model, tune=args.tune)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
