"""Correctness tooling: diff graphs/kernels against golden references."""

from mdlc.testing.harness import (
    CheckResult,
    compare_arrays,
    check_graph_against_reference,
    make_pass_verifier,
    onnxruntime_available,
)

__all__ = [
    "CheckResult",
    "compare_arrays",
    "check_graph_against_reference",
    "make_pass_verifier",
    "onnxruntime_available",
]
