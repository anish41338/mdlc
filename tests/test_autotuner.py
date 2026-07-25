"""Autotuner: cost-model ranking is sane, the budgeted search behaves, the
cache round-trips (arch-aware), and tuned schedules flow through codegen."""

import os

import numpy as np

from conftest import requires_gpp
from mdlc.autotuner import (
    AutotuneCache,
    Tuner,
    analytical_cost,
    depthwise_analytical_cost,
    tune_depthwise,
    tune_gemm,
)
from mdlc.codegen.schedule import DepthwiseSchedule, GemmSchedule


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
    # the naive default is always evaluated so win% is anchored
    assert res.default_metric >= res.best_metric
    assert 0.0 <= res.win_pct < 100.0


def test_tune_gemm_budget_caps_evaluations():
    res = tune_gemm(256, 256, 256, budget=8)
    assert res.n_evaluated <= 8


def test_tune_depthwise_modeled():
    sig = dict(C=32, mult=1, H=56, W=56, OH=56, OW=56,
               KH=3, KW=3, SH=1, SW=1, PH=1, PW=1)
    res = tune_depthwise(sig, budget=16)
    assert isinstance(res.best, DepthwiseSchedule)
    assert res.best.is_valid(max_threads=256)
    assert res.default_metric >= res.best_metric


def test_depthwise_cost_model_penalizes_tile_overhang():
    sig = dict(C=8, mult=1, H=7, W=7, OH=7, OW=7,
               KH=3, KW=3, SH=1, SW=1, PH=1, PW=1)
    snug = DepthwiseSchedule(8, 8, False)      # 7x7 fits one 8x8 tile
    huge = DepthwiseSchedule(16, 16, False)    # 4x the threads, same 49 pixels
    assert depthwise_analytical_cost(snug, sig) < \
        depthwise_analytical_cost(huge, sig)


def test_cache_roundtrip(tmp_path):
    path = os.path.join(tmp_path, "cache.json")
    c = AutotuneCache(path)
    s = GemmSchedule(64, 64, 16, 4, 4)
    c.put(128, 256, 64, s, 1.23, measured=False)
    c.save()
    c2 = AutotuneCache(path)
    got = c2.get(128, 256, 64)
    assert got == s


def test_cache_prefers_measured_over_modeled(tmp_path):
    path = os.path.join(tmp_path, "cache.json")
    c = AutotuneCache(path)
    modeled = GemmSchedule(64, 64, 16, 4, 4)
    measured = GemmSchedule(128, 64, 16, 8, 4)
    c.put(64, 64, 64, modeled, 100.0, measured=False)
    c.put(64, 64, 64, measured, 0.05, measured=True, arch="sm_75")
    # a CPU-side compile (no arch) should still pick up the T4-measured entry
    assert c.get(64, 64, 64) == measured
    # exact-arch lookup too
    assert c.get(64, 64, 64, arch="sm_75") == measured


def test_cache_depthwise_roundtrip(tmp_path):
    path = os.path.join(tmp_path, "cache.json")
    sig = dict(C=32, mult=1, H=28, W=28, OH=14, OW=14,
               KH=3, KW=3, SH=2, SW=2, PH=1, PW=1)
    c = AutotuneCache(path)
    s = DepthwiseSchedule(16, 8, True)
    c.put_depthwise(sig, s, 0.7, measured=True, arch="sm_75")
    c.save()
    assert AutotuneCache(path).get_depthwise(sig) == s


def test_tuner_caches_per_shape(tmp_path):
    cache = AutotuneCache(os.path.join(tmp_path, "c.json"))
    tuner = Tuner(cache=cache, budget=8)
    s1 = tuner.schedule_for(None, (256, 256, 256))
    s2 = tuner.schedule_for(None, (256, 256, 256))   # served from cache
    assert s1 == s2
    assert cache.get(256, 256, 256) is not None
    # first tuning is logged with a naive-vs-tuned line
    assert len(tuner.log) == 1
    assert tuner.report_lines()


@requires_gpp
def test_tuned_schedule_flows_through_codegen_and_sim(tmp_path):
    """A non-default cached schedule must reach the emitted kernel AND the
    sim launch dims — parity proves the whole tuned path, not just emission."""
    from onnx import TensorProto, helper, numpy_helper

    from mdlc.codegen.cuda.emit import emit_cuda_module
    from mdlc.codegen.cuda.sim_executor import simulate_module
    from mdlc.frontend import import_onnx_model
    from mdlc.passes.shape_inference import infer_shapes_by_execution
    from mdlc.runtime import run_reference
    from mdlc.testing.tolerances import FP32_REDUCTION

    rng = np.random.default_rng(3)
    M, K, N = 32, 48, 40
    w = rng.standard_normal((K, N)).astype(np.float32)
    model = helper.make_model(helper.make_graph(
        [helper.make_node("Gemm", ["x", "w"], ["y"])], "t",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [M, K])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [M, N])],
        [numpy_helper.from_array(w, "w")]),
        opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9

    # pin a deliberately non-default schedule in the cache
    pinned = GemmSchedule(BM=32, BN=32, BK=8, TM=2, TN=2)
    assert pinned != GemmSchedule() and pinned.is_valid()
    cache = AutotuneCache(os.path.join(tmp_path, "c.json"))
    cache.put(M, N, K, pinned, 1.0, measured=True, arch="sm_75")
    tuner = Tuner(cache=cache)

    g = import_onnx_model(model)
    feeds = {"x": rng.standard_normal((M, K)).astype(np.float32)}
    golden = run_reference(g.clone(), feeds)
    infer_shapes_by_execution(g, feeds)
    module = emit_cuda_module(g, schedule_fn=tuner)

    (gemm_launch,) = [l for l in module.plan if l.kind == "gemm"]
    assert gemm_launch.meta["sched"] == pinned
    assert gemm_launch.block[0] == pinned.threads_per_block()

    got = simulate_module(module, feeds)
    np.testing.assert_allclose(got["y"], golden["y"],
                               rtol=FP32_REDUCTION.rtol, atol=FP32_REDUCTION.atol)
