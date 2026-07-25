"""Benchmark harness: optimization wins + honest latency across systems.

Two modes:

* ``python -m mdlc.tools.benchmark <model.onnx>`` — quick single-model report:
  compile-time wins (op collapse, launches, memory pool) + a latency spot
  check against whatever oracle this machine has.

* ``python -m mdlc.tools.benchmark --suite --out artifacts/gpu_run_X`` — the
  full matrix for docs/BENCHMARKS.md: models × batch {1,8} × systems
  {mdlc (tuned), PyTorch eager, ONNX Runtime CUDA EP}, 200 timed iterations
  after 50 warmups, median/p10/p90, kernel-launch counts, device memory, full
  environment capture. Everything lands in a machine-readable JSON — numbers
  in docs are *generated* from these artifacts, never typed by hand.

Methodology (stated in every artifact):
  - Inputs are fixed seeded tensors staged on device before timing; H2D/D2H
    are excluded for all systems (mdlc stages feeds, torch gets a resident
    tensor, ORT uses io_binding).
  - mdlc + torch are CUDA-event timed per iteration; ORT is wall-clock around
    ``run_with_iobinding`` (it synchronizes internally) — noted per row.
  - Baselines get their best settings: ``torch.backends.cudnn.benchmark=True``,
    ORT graph optimizations ALL.
  - fast-math is OFF for mdlc (parity-grade kernels; a fast-math variant may
    appear only as a separately labeled row).

The honest expectation: beat PyTorch *eager* at batch-1 on small models (we
erase per-op dispatch/launch overhead via fusion), lose to ORT CUDA EP on
conv-heavy graphs (cuDNN algo selection), and quantify the gap.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import time
from datetime import datetime, timezone

import numpy as np

from mdlc.codegen.cuda.nvrtc_runtime import cuda_available
from mdlc.compiler import compile_onnx

SEED = 20260705
ITERS, WARMUP = 200, 50


def _pcts(samples: list[float]) -> dict:
    s = sorted(samples)
    n = len(s)
    return {
        "median_ms": s[n // 2],
        "p10_ms": s[n // 10],
        "p90_ms": s[(9 * n) // 10],
        "n": n,
    }


def _time_wall(fn, iters=ITERS, warmup=WARMUP) -> list[float]:
    for _ in range(warmup):
        fn()
    out = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t0) * 1e3)
    return out


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True).stdout.strip()
    except Exception:
        return "unknown"


def capture_env(ctx=None) -> dict:
    env = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import onnxruntime as ort
        env["onnxruntime"] = ort.__version__
    except Exception:
        pass
    try:
        import torch
        env["torch"] = torch.__version__
        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
            env["cuda_runtime"] = torch.version.cuda
    except Exception:
        pass
    if ctx is not None:
        env.setdefault("gpu", ctx.device_name())
        env["sm_arch"] = ctx.arch
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10)
        if smi.returncode == 0:
            env["driver"] = smi.stdout.strip().splitlines()[0]
    except Exception:
        pass
    return env


# ---------------------------------------------------------------------------
# systems
# ---------------------------------------------------------------------------

def bench_mdlc(model_path: str, batch: int, *, tune_cache: str, ctx) -> dict:
    """Compile with the committed tune cache and time the launch plan."""
    from mdlc.autotuner import AutotuneCache, Tuner
    from mdlc.codegen.cuda.gpu_executor import GpuExecutor

    tuner = Tuner(cache=AutotuneCache(tune_cache), ctx=None)   # cached lookups
    compiled = compile_onnx(_model_for_batch(model_path, batch),
                            schedule_fn=tuner,
                            dw_schedule_fn=tuner.dw_schedule_for)
    fallbacks = compiled.module.fallbacks()
    ex = GpuExecutor(compiled.module, ctx=ctx, mem_plan=compiled.mem_plan)
    feeds = {k: _seeded_input(v.shape) for k, v in compiled.feeds.items()}
    res = ex.bench(feeds, iters=ITERS, warmup=WARMUP)
    out = _pcts(res["samples_ms"])
    out.update({
        "system": "mdlc",
        "timing": "cuda_events",
        "launches": res["launches"],
        "device_bytes": res["device_bytes"],
        "pool_bytes": compiled.mem_plan.pool_bytes,
        "fallbacks": fallbacks,
    })
    # On-device parity vs the ORT golden for these exact feeds. The timing above
    # is already measured and valid; an unavailable oracle (no onnxruntime, or a
    # wheel built against a CUDA runtime this image lacks) must therefore be
    # recorded as a missing *check*, not allowed to discard the measurement.
    # The result stays clearly labelled either way — an unverified latency is
    # never reported as a verified one.
    try:
        golden = _ort_golden(compiled.proto_bytes, feeds)
        out["parity_max_abs_vs_ort"] = max(
            float(np.max(np.abs(res["outputs"][o] - golden[i])))
            for i, o in enumerate(compiled.graph.outputs))
    except Exception as e:
        out["parity_error"] = f"{type(e).__name__}: {e}"
    return out


def _seeded_input(shape) -> np.ndarray:
    return np.random.default_rng(SEED).standard_normal(shape).astype(np.float32)


def _ort_golden(proto_bytes: bytes, feeds: dict) -> list[np.ndarray]:
    import onnxruntime as ort
    sess = ort.InferenceSession(proto_bytes, providers=["CPUExecutionProvider"])
    return sess.run(None, {k: v for k, v in feeds.items()})


def bench_ort_cuda(model_path: str, batch: int) -> dict:
    """ONNX Runtime CUDA EP, graph optimizations ALL, io_binding (no H2D/D2H
    in the timed loop)."""
    import onnx
    import onnxruntime as ort

    model = onnx.load(_model_for_batch(model_path, batch))
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(model.SerializeToString(), so,
                                providers=["CUDAExecutionProvider"])
    inp = sess.get_inputs()[0]
    shape = [batch if isinstance(d, str) or d is None else d for d in inp.shape]
    if shape[0] != batch:
        shape[0] = batch
    x = _seeded_input(tuple(shape))
    x_dev = ort.OrtValue.ortvalue_from_numpy(x, "cuda", 0)
    io = sess.io_binding()
    io.bind_ortvalue_input(inp.name, x_dev)
    for o in sess.get_outputs():
        io.bind_output(o.name, "cuda", 0)
    samples = _time_wall(lambda: sess.run_with_iobinding(io))
    out = _pcts(samples)
    out.update({"system": "ort-cuda", "timing": "wall_clock_iobinding",
                "launches": None,
                "launches_note": "not instrumented (no public counter)"})
    return out


def _model_for_batch(model_path: str, batch: int) -> str:
    """Models are exported with a *fixed* batch (kernels specialize on shape).
    batch=1 uses the file as-is; batch=N expects a sibling ``<stem>_b<N>.onnx``
    (the Kaggle runbook exports it)."""
    if batch == 1:
        return model_path
    stem, ext = os.path.splitext(model_path)
    cand = f"{stem}_b{batch}{ext}"
    if not os.path.exists(cand):
        raise FileNotFoundError(
            f"{cand} not found — export it (build_* --batch {batch})")
    return cand


def _torch_model_for(model_path: str):
    import torch
    name = os.path.basename(model_path).lower()
    if "resnet18" in name:
        import torchvision
        return torchvision.models.resnet18(weights=None)
    if "mobilenet" in name:
        import torchvision
        return torchvision.models.mobilenet_v2(weights=None)
    if "repvit" in name:
        import timm
        from timm.utils.model import reparameterize_model
        variant = "repvit_m0_9" if "m0_9" in name else name.split(".")[0]
        m = timm.create_model(variant, pretrained=False)
        m.eval()
        return reparameterize_model(m)   # deploy form, same as our export
    raise ValueError(f"no torch baseline builder for {model_path}")


def bench_torch_eager(model_path: str, batch: int, res: int = 224) -> dict:
    """PyTorch eager on CUDA with its best settings (cudnn.benchmark=True).
    Latency depends on architecture, not weights, so an untrained instance of
    the identical (deploy-form) architecture is a fair timing baseline."""
    import torch
    torch.backends.cudnn.benchmark = True
    m = _torch_model_for(model_path).cuda().eval()
    x = torch.from_numpy(_seeded_input((batch, 3, res, res))).cuda()
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    samples = []
    with torch.no_grad():
        for _ in range(WARMUP):
            m(x)
        torch.cuda.synchronize()
        for _ in range(ITERS):
            start.record()
            m(x)
            stop.record()
            torch.cuda.synchronize()
            samples.append(start.elapsed_time(stop))
    out = _pcts(samples)
    out.update({"system": "torch-eager", "timing": "cuda_events",
                "device_bytes": int(torch.cuda.max_memory_allocated()),
                "launches": _torch_kernel_count(m, x)})
    return out


def _torch_kernel_count(m, x):
    """Count CUDA kernel launches for one forward via torch.profiler; None if
    profiling is unavailable (older builds)."""
    import torch
    try:
        from torch.profiler import ProfilerActivity, profile
        with torch.no_grad(), profile(
                activities=[ProfilerActivity.CUDA]) as prof:
            m(x)
            torch.cuda.synchronize()
        return sum(1 for e in prof.events()
                   if getattr(e, "device_type", None) is not None
                   and "cuda" in str(e.device_type).lower()
                   and e.self_device_time_total > 0)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# suite driver
# ---------------------------------------------------------------------------

def run_suite(models: list[str], batches: list[int], out_dir: str,
              tune_cache: str = "artifacts/tune_cache.json") -> str:
    if not cuda_available():
        raise SystemExit("--suite needs a CUDA device (this is the Kaggle step; "
                         "see artifacts/GPU_TODO.md)")
    from mdlc.codegen.cuda.nvrtc_runtime import CudaContext
    ctx = CudaContext()

    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for model in models:
        for batch in batches:
            for system, fn in (
                ("mdlc", lambda: bench_mdlc(model, batch, tune_cache=tune_cache,
                                            ctx=ctx)),
                ("torch-eager", lambda: bench_torch_eager(model, batch)),
                ("ort-cuda", lambda: bench_ort_cuda(model, batch)),
            ):
                try:
                    row = fn()
                except Exception as e:   # a missing baseline is a note, not a crash
                    row = {"system": system, "error": f"{type(e).__name__}: {e}"}
                row.update({"model": os.path.basename(model), "batch": batch})
                rows.append(row)
                label = row.get("median_ms")
                print(f"  {row['model']} b{batch} {system}: "
                      + (f"{label:.3f} ms" if label else row.get("error", "?")))

    artifact = {
        "kind": "mdlc_benchmark_suite",
        "env": capture_env(ctx),
        "config": {"iters": ITERS, "warmup": WARMUP, "seed": SEED,
                   "h2d_d2h_excluded": True, "fast_math": False,
                   "cudnn_benchmark": True, "ort_graph_opt": "ALL"},
        "rows": rows,
    }
    path = os.path.join(out_dir, "bench.json")
    with open(path, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"wrote {path}")
    return path


# ---------------------------------------------------------------------------
# quick single-model report (no GPU required)
# ---------------------------------------------------------------------------

def bench(model_path: str, *, tune: bool = False) -> None:
    schedule_fn = dw_fn = None
    if tune:
        from mdlc.autotuner import Tuner
        tuner = Tuner()
        schedule_fn, dw_fn = tuner, tuner.dw_schedule_for

    compiled = compile_onnx(model_path, schedule_fn=schedule_fn,
                            dw_schedule_fn=dw_fn)
    print(compiled.report())
    print()

    n_fused = sum(1 for n in compiled.graph.nodes if n.is_fused)
    orig_ops = len(compiled.original.nodes)
    opt_ops = len(compiled.graph.nodes)
    # views are metadata-only (buffer aliases), not kernel launches
    launches = sum(1 for l in compiled.module.plan if l.kind != "view")
    print("== optimization summary ==")
    print(f"  graph ops      : {orig_ops} -> {opt_ops}  ({n_fused} fused groups)")
    print(f"  kernel launches: {launches}  (vs ~{orig_ops} eager op-dispatches)")
    print(f"  distinct kernels: {len(compiled.module.kernels)}")
    print(f"  activation mem  : {compiled.mem_plan.reuse_ratio*100:.0f}% of naive "
          f"({compiled.mem_plan.planned_activation_bytes/1e6:.2f} MB pool, "
          f"256-byte aligned)")
    print()

    feeds = compiled.feeds

    # --- baseline: ONNX Runtime -------------------------------------------
    try:
        import onnxruntime as ort
        prov = (["CUDAExecutionProvider"] if cuda_available()
                else ["CPUExecutionProvider"])
        sess = ort.InferenceSession(compiled.proto_bytes, providers=prov)
        f32 = {k: v.astype(np.float32) for k, v in feeds.items()}
        samples = _time_wall(lambda: sess.run(None, f32), iters=20, warmup=5)
        print(f"ONNX Runtime ({prov[0].replace('ExecutionProvider','')}): "
              f"{statistics.median(samples):.3f} ms")
    except Exception as e:
        print(f"ONNX Runtime: unavailable ({e})")

    # --- our backend -------------------------------------------------------
    if cuda_available():
        from mdlc.codegen.cuda.gpu_executor import GpuExecutor
        ex = GpuExecutor(compiled.module, mem_plan=compiled.mem_plan)
        res = ex.run(feeds, timing=True)
        print(f"mdlc (generated CUDA, NVRTC): {res['ms']:.3f} ms")
        for o in compiled.graph.outputs:
            err = np.max(np.abs(res["outputs"][o] - compiled.golden[o]))
            print(f"  device output {o}: max_abs_err={err:.2e}")
    else:
        print("mdlc (generated CUDA): [GPU not present] kernels generated & "
              "validated on CPU sim; run on a CUDA device for latency.")
    print()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default="examples/resnet18.onnx")
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--suite", action="store_true",
                    help="full model x batch x system matrix (needs CUDA)")
    ap.add_argument("--models", nargs="+",
                    default=["examples/resnet18.onnx",
                             "examples/mobilenetv2.onnx",
                             "examples/repvit_m0_9.onnx"])
    ap.add_argument("--batches", nargs="+", type=int, default=[1, 8])
    ap.add_argument("--out", default=None,
                    help="artifact dir (default artifacts/gpu_run_<ts>)")
    ap.add_argument("--tune-cache", default="artifacts/tune_cache.json")
    args = ap.parse_args(argv)
    if args.suite:
        out = args.out or f"artifacts/gpu_run_{datetime.now():%Y%m%d_%H%M%S}"
        run_suite(args.models, args.batches, out, tune_cache=args.tune_cache)
    else:
        bench(args.model, tune=args.tune)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
