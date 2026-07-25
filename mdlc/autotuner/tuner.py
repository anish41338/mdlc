"""Autotuner: search the schedule space for the best kernel config per shape.

This is the "baby TVM" piece: compute is fixed, the *schedule* (tile/block/
register-blocking factors) is searched. Two evaluators:

  * **measured** — on a GPU, JIT each candidate and time it with CUDA events:
    per-iteration samples, median-of-50 after 10 warmups, and configs whose
    IQR/median exceeds 15% are discarded (shared-GPU clocks — Kaggle T4 —
    can't be locked, so noisy configs are untrustworthy, not unlucky).
  * **modeled** — without a GPU, score candidates with an analytical cost model
    (memory traffic + tile-quantization waste + occupancy). This keeps the
    search logic exercised and the pruning honest everywhere.

Search strategy: random sample within a budget, then local refinement around
the best candidates (vary one knob one step). The budget caps total evals at
≤64 per (kernel, shape, dtype) — that's how the combinatorial blowup is
avoided.

Results persist in ``artifacts/tune_cache.json`` keyed by
``(kernel_kind, shape_sig, dtype, sm_arch)``, committed to the repo so tuned
schedules reproduce without re-tuning. The default (untuned) schedule is
always evaluated too, so every entry can report "tuning won X% over naive".
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass, field
from typing import Optional

from mdlc.codegen.schedule import (
    DEFAULT_DEPTHWISE,
    DEFAULT_GEMM,
    DepthwiseSchedule,
    GemmSchedule,
    depthwise_search_space,
    gemm_search_space,
)

# Measurement noise gate: reject a config whose IQR/median exceeds this.
NOISE_GATE = 0.15


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


def depthwise_analytical_cost(sched: DepthwiseSchedule, sig: dict) -> float:
    """Cost proxy for the direct depthwise kernel (per image; N scales all
    candidates equally so it cancels).

    * smem variant: each block loads tile+halo once plus the filter; the halo
      grows with stride, so big tiles amortize it.
    * direct variant: every thread re-reads its K×K window from global.
    * waste: tile overhang on the output plane burns threads.
    """
    OH, OW = sig["OH"], sig["OW"]
    KH, KW, SH, DH = sig["KH"], sig["KW"], sig["SH"], sig.get("DH", 1)
    C_out = sig["C"] * sig.get("mult", 1)
    th, tw = sched.tile_h, sched.tile_w
    gx = (OW + tw - 1) // tw
    gy = (OH + th - 1) // th
    blocks = gx * gy * C_out
    if sched.direct_load:
        loads = th * tw * KH * KW
    else:
        smh = (th - 1) * SH + (KH - 1) * DH + 1
        smw = (tw - 1) * SH + (KW - 1) * DH + 1
        loads = smh * smw + KH * KW
        loads *= 1.05              # barrier + staging overhead nudge
    waste = (gx * tw - OW) / max(OW, 1) + (gy * th - OH) / max(OH, 1)
    threads = sched.threads_per_block()
    occ = abs(128 - threads) / 128.0
    return blocks * loads * (1.0 + waste) * (1.0 + 0.1 * occ)


@dataclass
class TuneResult:
    key: str                    # cache key this result belongs to
    best: object                # GemmSchedule | DepthwiseSchedule
    best_metric: Optional[float]     # ms (measured) or cost; None = nothing measurable
    default_metric: Optional[float]  # untuned default; None = baseline too noisy
    measured: bool
    n_evaluated: int
    n_noisy_discarded: int = 0
    ranking: list = field(default_factory=list)   # (schedule.key, metric)

    @property
    def win_pct(self) -> Optional[float]:
        """How much tuning beat the naive default, in % of default.

        None when there is no usable baseline (the default's own timing was
        rejected by the noise gate on a shared GPU) — a missing measurement is
        reported as missing, never as a made-up percentage.
        """
        if (self.default_metric is None or self.default_metric <= 0
                or self.best_metric is None):
            return None
        return 100.0 * (self.default_metric - self.best_metric) / self.default_metric


# ----- persistent cache -----------------------------------------------------

def gemm_sig(M: int, N: int, K: int) -> str:
    return f"{M}x{N}x{K}"


def depthwise_sig(sig: dict) -> str:
    return (f"C{sig['C']}m{sig.get('mult', 1)}"
            f"_H{sig['H']}W{sig['W']}_O{sig['OH']}x{sig['OW']}"
            f"_K{sig['KH']}x{sig['KW']}_S{sig['SH']}x{sig['SW']}"
            f"_P{sig['PH']}x{sig['PW']}_D{sig.get('DH', 1)}x{sig.get('DW', 1)}")


class AutotuneCache:
    """Persist best schedules keyed ``(kernel_kind, shape_sig, dtype, sm_arch)``.

    ``sm_arch`` is the device the number was measured on (e.g. ``sm_75``) or
    ``"model"`` for analytical results. Lookup prefers an exact-arch measured
    entry, then any measured entry (a T4-tuned cache still beats the cost
    model on another card), then a modeled one.
    """

    def __init__(self, path: str = "artifacts/tune_cache.json") -> None:
        self.path = path
        self.entries: dict[str, dict] = {}
        if os.path.exists(path):
            with open(path) as f:
                self.entries = json.load(f)
        self.entries.pop("_meta", None)

    @staticmethod
    def key(kind: str, sig: str, arch: str, dtype: str = "f32") -> str:
        return f"{kind}|{dtype}|{sig}|{arch}"

    def _lookup(self, kind: str, sig: str, arch: Optional[str],
                dtype: str = "f32") -> Optional[dict]:
        if arch:
            e = self.entries.get(self.key(kind, sig, arch, dtype))
            if e:
                return e
        prefix = f"{kind}|{dtype}|{sig}|"
        measured = [e for k, e in self.entries.items()
                    if k.startswith(prefix) and e.get("measured")]
        if measured:
            return measured[0]
        modeled = [e for k, e in self.entries.items() if k.startswith(prefix)]
        return modeled[0] if modeled else None

    def get(self, M: int, N: int, K: int,
            arch: Optional[str] = None) -> Optional[GemmSchedule]:
        e = self._lookup("gemm", gemm_sig(M, N, K), arch)
        if not e:
            return None
        return GemmSchedule(**e["sched"])

    def get_depthwise(self, sig: dict,
                      arch: Optional[str] = None) -> Optional[DepthwiseSchedule]:
        e = self._lookup("depthwise", depthwise_sig(sig), arch)
        if not e:
            return None
        return DepthwiseSchedule(**e["sched"])

    @staticmethod
    def _finite_or_none(v: Optional[float]) -> Optional[float]:
        """JSON boundary guard: json.dump writes bare ``Infinity``/``NaN`` for
        non-finite floats, which is not JSON. A missing baseline is null."""
        return v if v is not None and math.isfinite(v) else None

    def put(self, M: int, N: int, K: int, sched: GemmSchedule, metric: float,
            measured: bool, arch: Optional[str] = None,
            default_metric: Optional[float] = None) -> None:
        self.entries[self.key("gemm", gemm_sig(M, N, K),
                              arch or ("model" if not measured else "gpu"))] = {
            "sched": dict(BM=sched.BM, BN=sched.BN, BK=sched.BK,
                          TM=sched.TM, TN=sched.TN),
            "metric": self._finite_or_none(metric), "measured": measured,
            "default_metric": self._finite_or_none(default_metric),
        }

    def put_depthwise(self, sig: dict, sched: DepthwiseSchedule, metric: float,
                      measured: bool, arch: Optional[str] = None,
                      default_metric: Optional[float] = None) -> None:
        self.entries[self.key("depthwise", depthwise_sig(sig),
                              arch or ("model" if not measured else "gpu"))] = {
            "sched": dict(tile_h=sched.tile_h, tile_w=sched.tile_w,
                          direct_load=sched.direct_load),
            "metric": self._finite_or_none(metric), "measured": measured,
            "default_metric": self._finite_or_none(default_metric),
        }

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        out = dict(self.entries)
        with open(self.path, "w") as f:
            # allow_nan=False: refuse loudly rather than write invalid JSON —
            # the committed cache must stay readable by any JSON parser.
            json.dump(out, f, indent=2, sort_keys=True, allow_nan=False)


# ----- measurement ----------------------------------------------------------

def robust_median_ms(ctx, launch_fn, *, iters: int = 50,
                     warmup: int = 10) -> tuple[float, float]:
    """(median_ms, iqr_over_median) from per-iteration CUDA-event samples."""
    samples = sorted(ctx.time_ms_samples(launch_fn, iters=iters, warmup=warmup))
    n = len(samples)
    med = samples[n // 2]
    iqr = samples[(3 * n) // 4] - samples[n // 4]
    return med, (iqr / med if med > 0 else 0.0)


def _measure_gemm(sched, M, N, K, ctx, rng) -> tuple[float, float]:
    """Median-time one candidate on the GPU. Imported lazily so the no-GPU
    path never touches the driver. Returns (median_ms, rel_iqr)."""
    import ctypes

    import numpy as np

    from mdlc.codegen.cuda import templates as T
    from mdlc.codegen.cuda.nvrtc_runtime import compile_to_ptx

    name, src = T.gemm_kernel("tune_gemm", sched, with_bias=False, activation=None)
    ptx = compile_to_ptx(src, arch=ctx.arch)
    mod = ctx.load_ptx(ptx)
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

    med, rel_iqr = robust_median_ms(ctx, launch)
    ctx.free(dA); ctx.free(dB); ctx.free(dC)
    return med, rel_iqr


def _measure_depthwise(sched: DepthwiseSchedule, sig: dict, ctx,
                       rng) -> tuple[float, float]:
    import numpy as np

    from mdlc.codegen.cuda import templates as T
    from mdlc.codegen.cuda.nvrtc_runtime import compile_to_ptx

    C, mult = sig["C"], sig.get("mult", 1)
    H, W, OH, OW = sig["H"], sig["W"], sig["OH"], sig["OW"]
    name, src = T.depthwise_conv_kernel(
        "tune_dw", C=C, mult=mult, H=H, W=W, OH=OH, OW=OW,
        KH=sig["KH"], KW=sig["KW"], SH=sig["SH"], SW=sig["SW"],
        PH=sig["PH"], PW=sig["PW"], DH=sig.get("DH", 1), DW=sig.get("DW", 1),
        tile_h=sched.tile_h, tile_w=sched.tile_w,
        with_bias=False, activation=None, direct_load=sched.direct_load)
    ptx = compile_to_ptx(src, arch=ctx.arch)
    mod = ctx.load_ptx(ptx)
    x = rng.standard_normal((1, C, H, W)).astype(np.float32)
    w = rng.standard_normal((C * mult, 1, sig["KH"], sig["KW"])).astype(np.float32)
    dX, dW = ctx.to_device(x), ctx.to_device(w)
    dY = ctx.malloc(C * mult * OH * OW * 4)
    grid = ((OW + sched.tile_w - 1) // sched.tile_w,
            (OH + sched.tile_h - 1) // sched.tile_h,
            C * mult)

    def launch():
        mod.launch(name, grid, (sched.threads_per_block(), 1, 1), [dX, dW, dY])

    med, rel_iqr = robust_median_ms(ctx, launch)
    ctx.free(dX); ctx.free(dW); ctx.free(dY)
    return med, rel_iqr


# ----- search ---------------------------------------------------------------

_GEMM_KNOBS = {"BM": (32, 64, 128), "BN": (32, 64, 128), "BK": (8, 16, 32),
               "TM": (2, 4, 8), "TN": (2, 4, 8)}
_DW_KNOBS = {"tile_h": (4, 8, 16, 32), "tile_w": (4, 8, 16, 32),
             "direct_load": (False, True)}


def _neighbors(sched, knobs, make):
    """One-knob-one-step neighbors of a schedule (local refinement moves)."""
    d = {k: getattr(sched, k) for k in knobs}
    out = []
    for k, values in knobs.items():
        vs = list(values)
        i = vs.index(d[k]) if d[k] in vs else -1
        for j in (i - 1, i + 1):
            if 0 <= j < len(vs) and vs[j] != d[k]:
                nd = dict(d)
                nd[k] = vs[j]
                s = make(**nd)
                out.append(s)
    return out


def _search(candidates, default, evaluate, *, budget: int, seed: int,
            knobs, make, is_valid):
    """Random sample then local refinement, ≤ budget evaluations total.

    ``evaluate`` returns (metric, rel_iqr) — rel_iqr > NOISE_GATE discards the
    config (measured runs on shared GPUs); modeled evaluators return iqr 0.
    """
    rng = random.Random(seed)
    pool = [c for c in candidates if c != default]
    rng.shuffle(pool)

    scored: list[tuple[object, float]] = []
    discarded = 0
    seen = set()

    def try_eval(s) -> None:
        nonlocal discarded
        if s.key() in seen:
            return
        seen.add(s.key())
        try:
            metric, rel_iqr = evaluate(s)
        except Exception:
            return
        if rel_iqr > NOISE_GATE:
            discarded += 1
            return
        scored.append((s, metric))

    # The naive default is always evaluated (it anchors the win-% report). Its
    # own sample can be the noisy one on a shared GPU: that used to leave
    # default_metric = inf, which flowed into the report as "won nan%" and into
    # the JSON cache as a bare `Infinity`. Retry once; a still-noisy baseline
    # is recorded as None ("no baseline"), never as a number.
    default_metric = None
    default_noisy = False
    seen.add(default.key())
    for _ in range(2):
        try:
            metric, rel_iqr = evaluate(default)
        except Exception:
            break
        if rel_iqr <= NOISE_GATE:
            default_metric = metric
            scored.append((default, metric))
            break
        default_noisy = True
    if default_noisy and default_metric is None:
        discarded += 1   # the default counts once, however many attempts

    n_random = max(1, (budget * 3) // 4)
    for s in pool[:n_random]:
        if len(seen) >= budget:
            break
        try_eval(s)

    # local refinement around the current top-3
    for s, _ in sorted(scored, key=lambda t: t[1])[:3]:
        for nb in _neighbors(s, knobs, make):
            if len(seen) >= budget:
                break
            if is_valid(nb):
                try_eval(nb)

    scored.sort(key=lambda t: t[1])
    return scored, default_metric, discarded


def tune_gemm(M, N, K, *, budget: int = 64, ctx=None, seed: int = 0) -> TuneResult:
    """Search schedules for one GEMM shape. Uses GPU timing if ``ctx`` is given,
    else the analytical cost model. ``budget`` caps total evaluations."""
    measured = ctx is not None
    if measured:
        import numpy as np
        nprng = np.random.default_rng(seed)

        def evaluate(s):
            return _measure_gemm(s, M, N, K, ctx, nprng)
    else:
        def evaluate(s):
            return analytical_cost(s, M, N, K), 0.0

    scored, default_metric, discarded = _search(
        list(gemm_search_space()), DEFAULT_GEMM, evaluate,
        budget=budget, seed=seed, knobs=_GEMM_KNOBS, make=GemmSchedule,
        is_valid=lambda s: s.is_valid() and s.smem_bytes() <= 48 * 1024)

    best, best_metric = scored[0] if scored else (DEFAULT_GEMM, None)
    return TuneResult(
        key=AutotuneCache.key("gemm", gemm_sig(M, N, K),
                              ctx.arch if measured else "model"),
        best=best, best_metric=best_metric, default_metric=default_metric,
        measured=measured, n_evaluated=len(scored), n_noisy_discarded=discarded,
        ranking=[(s.key(), m) for s, m in scored],
    )


def tune_depthwise(sig: dict, *, budget: int = 32, ctx=None,
                   seed: int = 0) -> TuneResult:
    """Search depthwise schedules for one conv signature (dict with C, mult,
    H, W, OH, OW, KH, KW, SH, SW, PH, PW[, DH, DW])."""
    measured = ctx is not None
    if measured:
        import numpy as np
        nprng = np.random.default_rng(seed)

        def evaluate(s):
            return _measure_depthwise(s, sig, ctx, nprng)
    else:
        def evaluate(s):
            return depthwise_analytical_cost(s, sig), 0.0

    scored, default_metric, discarded = _search(
        list(depthwise_search_space()), DEFAULT_DEPTHWISE, evaluate,
        budget=budget, seed=seed, knobs=_DW_KNOBS, make=DepthwiseSchedule,
        is_valid=lambda s: s.is_valid(max_threads=256))

    # Nothing measurable (every config, default included, rejected as noisy):
    # keep the default schedule. A tuning run on a busy GPU must degrade to
    # "untuned", never crash — codegen still needs a valid schedule back.
    best, best_metric = scored[0] if scored else (DEFAULT_DEPTHWISE, None)
    return TuneResult(
        key=AutotuneCache.key("depthwise", depthwise_sig(sig),
                              ctx.arch if measured else "model"),
        best=best, best_metric=best_metric, default_metric=default_metric,
        measured=measured, n_evaluated=len(scored), n_noisy_discarded=discarded,
        ranking=[(s.key(), m) for s, m in scored],
    )


# ----- compile-time oracle ----------------------------------------------------

class Tuner:
    """A cache-backed schedule oracle codegen queries as ``schedule_fn`` /
    ``dw_schedule_fn``. Records every decision for the tuning report."""

    def __init__(self, cache: Optional[AutotuneCache] = None, ctx=None,
                 budget: int = 64) -> None:
        self.cache = cache or AutotuneCache()
        self.ctx = ctx
        self.budget = budget
        self.log: list[TuneResult] = []

    @property
    def arch(self) -> Optional[str]:
        return self.ctx.arch if self.ctx is not None else None

    def schedule_for(self, node, mnk) -> GemmSchedule:
        M, N, K = mnk
        cached = self.cache.get(M, N, K, arch=self.arch)
        if cached is not None and cached.is_valid():
            return cached
        res = tune_gemm(M, N, K, budget=self.budget, ctx=self.ctx)
        self.log.append(res)
        self.cache.put(M, N, K, res.best, res.best_metric, res.measured,
                       arch=self.arch, default_metric=res.default_metric)
        return res.best

    def dw_schedule_for(self, node, sig: dict) -> DepthwiseSchedule:
        cached = self.cache.get_depthwise(sig, arch=self.arch)
        if cached is not None and cached.is_valid(max_threads=256):
            return cached
        res = tune_depthwise(sig, budget=min(self.budget, 32), ctx=self.ctx)
        self.log.append(res)
        self.cache.put_depthwise(sig, res.best, res.best_metric, res.measured,
                                 arch=self.arch, default_metric=res.default_metric)
        return res.best

    def report_lines(self) -> list[str]:
        lines = []
        for r in self.log:
            unit = "ms" if r.measured else "cost"
            naive = (f"{r.default_metric:.4g}{unit}"
                     if r.default_metric is not None else "n/a (noisy)")
            best = (f"{r.best_metric:.4g}{unit}"
                    if r.best_metric is not None else "n/a (all noisy)")
            won = f"{r.win_pct:.1f}%" if r.win_pct is not None else "n/a"
            lines.append(
                f"  {r.key}: tuned {r.best.key()} {best} "
                f"vs naive {naive} -> won {won}"
                + (f" ({r.n_noisy_discarded} noisy configs discarded)"
                   if r.n_noisy_discarded else ""))
        return lines

    def __call__(self, node, mnk) -> GemmSchedule:
        return self.schedule_for(node, mnk)
