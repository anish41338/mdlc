"""Autotuner: search the schedule space for the best GEMM config per shape.

This is the "baby TVM" piece: compute is fixed, the *schedule* (tile/block/
register-blocking factors) is searched. Two evaluators:

  * **measured** — on a GPU, JIT each candidate and time it with CUDA events.
  * **modeled**  — without a GPU, score candidates with an analytical cost model
    (memory traffic + tile-quantization waste + occupancy). This keeps the
    search logic exercised and the pruning honest everywhere.

The space is deliberately small (a few proven tile shapes, pre-pruned by
validity and shared-memory capacity), and we cap the per-shape budget with
random sampling — that's how the combinatorial blowup is avoided.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from typing import Optional

from mdlc.codegen.schedule import (
    DEFAULT_GEMM,
    GemmSchedule,
    gemm_search_space,
)


def analytical_cost(sched: GemmSchedule, M: int, N: int, K: int) -> float:
    """Lower is better. A proxy for runtime with no device:

    * global traffic — A is re-read once per column-block, B once per row-block,
      so larger BN/BM cut traffic (the reuse fusion buys).
    * quantization waste — tiles that overhang a small problem burn threads.
    * occupancy — bias toward ~128-512 threads/block.
    """
    gx = (N + sched.BN - 1) // sched.BN
    gy = (M + sched.BM - 1) // sched.BM
    a_loads = gx * M * K          # elements of A streamed across the grid
    b_loads = gy * K * N          # elements of B streamed across the grid
    traffic = a_loads + b_loads
    waste = (gx * sched.BN - N) / max(N, 1) + (gy * sched.BM - M) / max(M, 1)
    threads = sched.threads_per_block()
    occ = abs(256 - threads) / 256.0
    return traffic * (1.0 + waste) * (1.0 + 0.15 * occ)


@dataclass
class TuneResult:
    mnk: tuple
    best: GemmSchedule
    best_metric: float          # ms (measured) or cost (modeled)
    measured: bool
    n_evaluated: int
    ranking: list = field(default_factory=list)   # (schedule.key, metric)


class AutotuneCache:
    """Persist best schedule per (M,N,K) so we tune a shape once."""

    def __init__(self, path: str = "autotune_cache/gemm.json") -> None:
        self.path = path
        self.entries: dict[str, dict] = {}
        if os.path.exists(path):
            with open(path) as f:
                self.entries = json.load(f)

    @staticmethod
    def key(M, N, K) -> str:
        return f"{M}x{N}x{K}"

    def get(self, M, N, K) -> Optional[GemmSchedule]:
        e = self.entries.get(self.key(M, N, K))
        if not e:
            return None
        return GemmSchedule(**e["sched"])

    def put(self, M, N, K, sched: GemmSchedule, metric: float, measured: bool) -> None:
        self.entries[self.key(M, N, K)] = {
            "sched": dict(BM=sched.BM, BN=sched.BN, BK=sched.BK,
                          TM=sched.TM, TN=sched.TN),
            "metric": metric, "measured": measured,
        }

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.entries, f, indent=2)


def _measure_gemm(sched, M, N, K, ctx, rng) -> float:
    """Time one candidate on the GPU. Imported lazily so the no-GPU path never
    touches the driver."""
    import numpy as np

    from mdlc.codegen.cuda import templates as T
    from mdlc.codegen.cuda.nvrtc_runtime import compile_to_ptx

    name, src = T.gemm_kernel("tune_gemm", sched, with_bias=False, activation=None)
    ptx = compile_to_ptx(src, arch=ctx.arch)
    mod = ctx.load_ptx(ptx)
    import ctypes
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    dA, dB = ctx.to_device(A), ctx.to_device(B)
    dC = ctx.malloc(M * N * 4)
    gx = (N + sched.BN - 1) // sched.BN
    gy = (M + sched.BM - 1) // sched.BM
    block = sched.threads_per_block()

    def launch():
        mod.launch(name, (gx, gy, 1), (block, 1, 1),
                   [dA, dB, dC, ctypes.c_int(M), ctypes.c_int(N), ctypes.c_int(K)])

    ms = ctx.time_ms(launch)
    ctx.free(dA); ctx.free(dB); ctx.free(dC)
    return ms


def tune_gemm(M, N, K, *, budget: int = 24, ctx=None, seed: int = 0) -> TuneResult:
    """Search schedules for one GEMM shape. Uses GPU timing if ``ctx`` is given,
    else the analytical cost model. ``budget`` caps random samples."""
    rng = random.Random(seed)
    candidates = [s for s in gemm_search_space()]
    if len(candidates) > budget:
        candidates = rng.sample(candidates, budget)
    if DEFAULT_GEMM not in candidates:
        candidates.append(DEFAULT_GEMM)

    measured = ctx is not None
    import numpy as np
    nprng = np.random.default_rng(seed)
    scored = []
    for s in candidates:
        if measured:
            try:
                metric = _measure_gemm(s, M, N, K, ctx, nprng)
            except Exception:
                continue
        else:
            metric = analytical_cost(s, M, N, K)
        scored.append((s, metric))

    scored.sort(key=lambda t: t[1])
    best, best_metric = scored[0]
    return TuneResult(
        mnk=(M, N, K), best=best, best_metric=best_metric, measured=measured,
        n_evaluated=len(scored),
        ranking=[(s.key(), m) for s, m in scored],
    )


class Tuner:
    """A cache-backed schedule oracle the codegen can query as ``schedule_fn``."""

    def __init__(self, cache: Optional[AutotuneCache] = None, ctx=None,
                 budget: int = 24) -> None:
        self.cache = cache or AutotuneCache()
        self.ctx = ctx
        self.budget = budget

    def schedule_for(self, node, mnk) -> GemmSchedule:
        M, N, K = mnk
        cached = self.cache.get(M, N, K)
        if cached is not None and cached.is_valid():
            return cached
        res = tune_gemm(M, N, K, budget=self.budget, ctx=self.ctx)
        self.cache.put(M, N, K, res.best, res.best_metric, res.measured)
        return res.best

    def __call__(self, node, mnk) -> GemmSchedule:
        return self.schedule_for(node, mnk)
