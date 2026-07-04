"""The frozen tolerance contract — single source of truth for every parity check.

Every harness/test comparison imports a tier from here. Nobody hand-picks
rtol/atol at a call site: if a comparison fails its tier, the response is to
find the bug (or escalate with a divergence analysis), never to loosen the
number in place. Changing a tier requires a documented numerical justification
in this docstring and human sign-off.

Tiers are keyed by *comparison class* — what makes two correct computations of
the same value differ at all:

  BITWISE           Pure data movement (im2col gather, reshape, transpose) and
                    integer/quantized codes. No arithmetic happens, so any
                    difference is a bug. Also exact small-integer fp math.

  FP32_SAME_ORDER   Algebraically identical fp32 math evaluated in the same
                    association order: elementwise chains, folded Conv+BN
                    (per-channel scale multiplies each weight once; the conv's
                    accumulation order is unchanged). Error budget is a few ulp
                    per element from rounding of the transformed constants.

  FP32_REDUCTION    One reduction-type kernel (GEMM, conv, pooling, mean) vs a
                    reference that accumulates in a different order (tiled
                    K-loop vs NumPy's pairwise summation). fp32 reassociation
                    over K up to a few thousand terms; bounded by ~K·eps of the
                    running magnitude, observed well under 1e-4 relative.

  FP32_NETWORK      A whole network end-to-end vs ONNX Runtime. Per-layer
                    reduction drift compounds through 20+ conv layers, so this
                    is the loosest fp32 tier. ResNet-18 observed max_abs
                    5.2e-6 / max_rel 8.9e-5 against ORT — the tier leaves
                    ~10x headroom over observed, not more.

History:
  2026-07-04  Frozen at Phase 0. These *tighten* the pre-contract ad-hoc
              values (which were 1e-3/1e-3 in the loosest spots); nothing was
              loosened to pass.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Tol:
    rtol: float
    atol: float


BITWISE = Tol(rtol=0.0, atol=0.0)
FP32_SAME_ORDER = Tol(rtol=1e-4, atol=1e-5)
FP32_REDUCTION = Tol(rtol=1e-4, atol=1e-4)
FP32_NETWORK = Tol(rtol=1e-3, atol=1e-4)
