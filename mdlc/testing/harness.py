"""The correctness harness — the project's safety net.

Bugs in a codegen compiler are silent numerical corruption: a conv that's off
by 0.3% costs days of bisecting tiling indices. The defense is to diff *every*
stage against a golden reference, per-tensor, with explicit tolerances, from
day one.

This module provides:
  * ``compare_arrays`` — tolerance check with informative diagnostics.
  * ``check_graph_against_reference`` — run a (possibly optimized) graph and
    compare its outputs to a golden source (ONNX Runtime if available, else the
    NumPy reference run of the *original* graph).
  * ``make_pass_verifier`` — a callback the PassManager runs after every pass,
    so an optimization that breaks numerics fails immediately and loudly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from mdlc.ir import Graph
from mdlc.runtime.reference import run_reference
from mdlc.testing.tolerances import FP32_NETWORK


def onnxruntime_available() -> bool:
    try:
        import onnxruntime  # noqa: F401
        return True
    except Exception:
        return False


@dataclass
class CheckResult:
    name: str
    ok: bool
    max_abs: float
    max_rel: float
    detail: str = ""

    def __str__(self) -> str:
        flag = "OK  " if self.ok else "FAIL"
        return f"[{flag}] {self.name:24s} max_abs={self.max_abs:.3e} max_rel={self.max_rel:.3e} {self.detail}"


def compare_arrays(
    name: str,
    got: np.ndarray,
    want: np.ndarray,
    *,
    rtol: float = FP32_NETWORK.rtol,
    atol: float = FP32_NETWORK.atol,
    vacuous_factor: float = 100.0,
) -> CheckResult:
    got = np.asarray(got, dtype=np.float64)
    want = np.asarray(want, dtype=np.float64)
    if got.shape != want.shape:
        return CheckResult(name, False, float("inf"), float("inf"),
                           f"shape mismatch got{got.shape} want{want.shape}")
    diff = np.abs(got - want)
    max_abs = float(diff.max()) if diff.size else 0.0
    denom = np.abs(want) + atol
    max_rel = float((diff / denom).max()) if diff.size else 0.0
    ok = bool(np.allclose(got, want, rtol=rtol, atol=atol))
    detail = ""
    if not ok:
        idx = np.unravel_index(int(diff.argmax()), diff.shape)
        detail = f"worst@{idx}: got={got[idx]:.6g} want={want[idx]:.6g}"
    # A comparison where the golden values are at atol scale passes no matter
    # what we computed — a vacuous PASS is more dangerous than a FAIL (this is
    # how a dropped ReLU6 on an untrained MobileNetV2 went unnoticed). Flag it.
    # ``vacuous_factor`` scales the power demand: fp32 tiers use 100x; the
    # quantized tier (atol = 2 output quanta by construction) uses a smaller
    # factor since golden magnitudes are inherently a modest number of quanta.
    want_mag = float(np.abs(want).max()) if want.size else 0.0
    if ok and atol > 0 and want_mag < vacuous_factor * atol:
        return CheckResult(
            name, False, max_abs, max_rel,
            f"VACUOUS: golden magnitude {want_mag:.3e} < "
            f"{vacuous_factor:g}*atol={vacuous_factor * atol:.0e}; "
            "comparison has no power — rescale the model/inputs")
    return CheckResult(name, ok, max_abs, max_rel, detail)


def _golden_outputs(
    model_proto_bytes: Optional[bytes],
    original_graph: Graph,
    feeds: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Compute reference outputs. Prefer ONNX Runtime on the original model;
    fall back to our NumPy reference run of the original (pre-optimization)
    graph, which is itself validated against ORT in the test suite."""
    if model_proto_bytes is not None and onnxruntime_available():
        import onnxruntime as ort

        sess = ort.InferenceSession(model_proto_bytes, providers=["CPUExecutionProvider"])
        names = [o.name for o in sess.get_outputs()]
        outs = sess.run(None, {k: np.asarray(v, dtype=np.float32) for k, v in feeds.items()})
        return dict(zip(names, outs))
    return run_reference(original_graph, feeds)


def check_graph_against_reference(
    graph: Graph,
    feeds: dict[str, np.ndarray],
    *,
    golden: Optional[dict[str, np.ndarray]] = None,
    model_proto_bytes: Optional[bytes] = None,
    original_graph: Optional[Graph] = None,
    rtol: float = FP32_NETWORK.rtol,
    atol: float = FP32_NETWORK.atol,
    vacuous_factor: float = 100.0,
) -> list[CheckResult]:
    """Run ``graph`` through the reference executor and diff its outputs
    against a golden source. Returns one ``CheckResult`` per graph output."""
    if golden is None:
        golden = _golden_outputs(model_proto_bytes, original_graph or graph, feeds)
    got = run_reference(graph, feeds)
    results = []
    for oname in graph.outputs:
        if oname not in golden:
            results.append(CheckResult(oname, False, float("inf"), float("inf"),
                                       "missing in golden"))
            continue
        results.append(compare_arrays(oname, got[oname], golden[oname],
                                      rtol=rtol, atol=atol,
                                      vacuous_factor=vacuous_factor))
    return results


def make_pass_verifier(
    feeds: dict[str, np.ndarray],
    golden: dict[str, np.ndarray],
    *,
    rtol: float = FP32_NETWORK.rtol,
    atol: float = FP32_NETWORK.atol,
    vacuous_factor: float = 100.0,
) -> Callable[[Graph, str], None]:
    """Build a verifier for ``PassManager.verify``. After every pass it checks
    the graph's *structural* invariants (``mdlc.ir.verify_graph``), then re-runs
    the graph and raises ``AssertionError`` if any output drifts past tolerance."""
    from mdlc.ir.verify import verify_graph

    def verify(graph: Graph, pass_name: str) -> None:
        verify_graph(graph, context=pass_name)
        results = check_graph_against_reference(
            graph, feeds, golden=golden, rtol=rtol, atol=atol,
            vacuous_factor=vacuous_factor,
        )
        bad = [r for r in results if not r.ok]
        if bad:
            msg = "\n".join(str(r) for r in bad)
            raise AssertionError(f"pass {pass_name!r} broke numerics:\n{msg}")

    return verify
