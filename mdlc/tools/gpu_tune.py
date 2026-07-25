"""Tune every kernel shape the target models actually hit, on the local GPU.

    python -m mdlc.tools.gpu_tune [models...] --budget 64 \
        --cache artifacts/tune_cache.json [--report artifacts/gpu_run_X/tune.json]

Compiles each model (default schedules), harvests the distinct GEMM shapes and
depthwise signatures from the launch plans, then measured-tunes each on the
device: ≤budget configs per shape, random-then-local-refine, median-of-50
CUDA-event samples after 10 warmups, configs with IQR/median > 15% discarded
(shared-GPU clocks; the count is recorded). Results persist in the committed
cache keyed (kernel, shape, dtype, sm_arch) so a later compile — anywhere —
reuses them without re-tuning, and every entry stores the naive-default metric
so reports can say "autotuning won X% on kernel Y".
"""

from __future__ import annotations

import argparse
import json
import os

from mdlc.autotuner import AutotuneCache, tune_depthwise, tune_gemm
from mdlc.autotuner.tuner import depthwise_sig
from mdlc.codegen.cuda.nvrtc_runtime import CudaContext, cuda_available
from mdlc.compiler import compile_onnx

DEFAULT_MODELS = ["examples/resnet18.onnx", "examples/mobilenetv2.onnx",
                  "examples/repvit_m0_9.onnx"]


def harvest(models: list[str]) -> tuple[set, list]:
    """Distinct (M,N,K) GEMM shapes and depthwise sig dicts across models."""
    gemms: set[tuple] = set()
    dws: list[dict] = []
    seen_dw = set()
    for path in models:
        compiled = compile_onnx(path)
        for l in compiled.module.plan:
            if l.kind == "gemm":
                m = l.meta
                gemms.add((m["M"], m["N"], m["K"]))
            elif l.kind == "depthwise":
                sig = l.meta.get("dw_sig")
                if sig is None:
                    continue
                k = depthwise_sig(sig)
                if k not in seen_dw:
                    seen_dw.add(k)
                    dws.append(sig)
    return gemms, dws


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*", default=None)
    ap.add_argument("--budget", type=int, default=64)
    ap.add_argument("--cache", default="artifacts/tune_cache.json")
    ap.add_argument("--report", default=None,
                    help="also dump per-shape tuning results as JSON")
    args = ap.parse_args(argv)
    models = args.models or [m for m in DEFAULT_MODELS if os.path.exists(m)]

    if not cuda_available():
        raise SystemExit("gpu_tune needs a CUDA device (this is the Kaggle "
                         "step; see artifacts/GPU_TODO.md)")
    ctx = CudaContext()
    cache = AutotuneCache(args.cache)
    print(f"device: {ctx.device_name()} ({ctx.arch}); "
          f"cache: {args.cache} ({len(cache.entries)} entries)")

    gemms, dws = harvest(models)
    print(f"harvested {len(gemms)} GEMM shapes, {len(dws)} depthwise sigs "
          f"from {len(models)} models")

    results = []
    for (M, N, K) in sorted(gemms):
        if cache.get(M, N, K, arch=ctx.arch) is not None:
            continue
        res = tune_gemm(M, N, K, budget=args.budget, ctx=ctx)
        cache.put(M, N, K, res.best, res.best_metric, res.measured,
                  arch=ctx.arch, default_metric=res.default_metric)
        cache.save()   # crash-safe: every result lands immediately
        print(f"  gemm {M}x{N}x{K}: {res.best.key()} {res.best_metric:.4f} ms "
              f"(naive {res.default_metric:.4f}, won {res.win_pct:.1f}%, "
              f"{res.n_noisy_discarded} noisy discarded)")
        results.append(res)
    for sig in dws:
        if cache.get_depthwise(sig, arch=ctx.arch) is not None:
            continue
        res = tune_depthwise(sig, budget=min(args.budget, 32), ctx=ctx)
        cache.put_depthwise(sig, res.best, res.best_metric, res.measured,
                            arch=ctx.arch, default_metric=res.default_metric)
        cache.save()
        print(f"  dw {res.key}: {res.best.key()} {res.best_metric:.4f} ms "
              f"(naive {res.default_metric:.4f}, won {res.win_pct:.1f}%)")
        results.append(res)

    print(f"cache now has {len(cache.entries)} entries")
    if args.report:
        os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
        with open(args.report, "w") as f:
            json.dump([{
                "key": r.key, "best": r.best.key(),
                "best_metric_ms": r.best_metric,
                "default_metric_ms": r.default_metric,
                "win_pct": r.win_pct, "n_evaluated": r.n_evaluated,
                "n_noisy_discarded": r.n_noisy_discarded,
            } for r in results], f, indent=2)
        print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
