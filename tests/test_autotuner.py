"""Autotuner: cost-model ranking is sane and the cache round-trips."""

import os

from mdlc.autotuner import AutotuneCache, Tuner, analytical_cost, tune_gemm
from mdlc.codegen.schedule import GemmSchedule


def test_cost_model_prefers_no_quant_waste_on_small_gemm():
    M = N = K = 64
    perfect = GemmSchedule(64, 64, 16, 4, 4)       # tiles the 64x64 exactly
    oversized = GemmSchedule(128, 128, 16, 8, 8)   # huge overhang -> wasteful
    assert analytical_cost(perfect, M, N, K) < analytical_cost(oversized, M, N, K)


def test_tune_gemm_returns_valid_best():
    res = tune_gemm(512, 512, 512, budget=16)
    assert res.best.is_valid()
    assert res.n_evaluated >= 1
    # ranking sorted ascending by metric
    metrics = [m for _, m in res.ranking]
    assert metrics == sorted(metrics)


def test_cache_roundtrip(tmp_path):
    path = os.path.join(tmp_path, "cache.json")
    c = AutotuneCache(path)
    s = GemmSchedule(64, 64, 16, 4, 4)
    c.put(128, 256, 64, s, 1.23, measured=False)
    c.save()
    c2 = AutotuneCache(path)
    got = c2.get(128, 256, 64)
    assert got == s


def test_tuner_caches_per_shape(tmp_path):
    cache = AutotuneCache(os.path.join(tmp_path, "c.json"))
    tuner = Tuner(cache=cache, budget=8)
    s1 = tuner.schedule_for(None, (256, 256, 256))
    s2 = tuner.schedule_for(None, (256, 256, 256))   # served from cache
    assert s1 == s2
    assert cache.get(256, 256, 256) is not None
