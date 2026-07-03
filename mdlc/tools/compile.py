"""CLI: compile an ONNX model and report what the compiler did.

    python -m mdlc.tools.compile examples/resnet18.onnx --report
    python -m mdlc.tools.compile examples/resnet18.onnx --emit out.cu
    python -m mdlc.tools.compile examples/resnet18.onnx --tune
"""

from __future__ import annotations

import argparse

from mdlc.compiler import compile_onnx


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="mdlc: ONNX -> fused CUDA compiler")
    ap.add_argument("model", help="path to .onnx model")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--report", action="store_true", help="print full compile report")
    ap.add_argument("--emit", metavar="FILE", help="write generated CUDA to FILE")
    ap.add_argument("--tune", action="store_true", help="autotune GEMM schedules")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip per-pass numerical verification")
    args = ap.parse_args(argv)

    schedule_fn = None
    if args.tune:
        from mdlc.autotuner import Tuner
        schedule_fn = Tuner()

    compiled = compile_onnx(args.model, batch=args.batch, schedule_fn=schedule_fn,
                            verify_passes=not args.no_verify)

    if args.report or not args.emit:
        print(compiled.report())

    if args.tune and schedule_fn is not None:
        schedule_fn.cache.save()
        print(f"\nautotune cache: {len(schedule_fn.cache.entries)} shapes -> "
              f"{schedule_fn.cache.path}")

    if args.emit:
        with open(args.emit, "w") as f:
            f.write(compiled.module.sources())
        print(f"\nwrote generated CUDA ({len(compiled.module.kernels)} kernels) "
              f"to {args.emit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
