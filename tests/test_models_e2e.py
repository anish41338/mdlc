"""End-to-end on the real target models: compile -> zero host fallbacks ->
generated kernels compose on the CPU sim -> outputs match ONNX Runtime.

CI note: these need torch+timm to export (skipped where absent — CI covers the
same kernels via hand-built models); resolution is reduced to keep the
one-OS-thread-per-CUDA-thread sim fast. One full-resolution (224x224) sim run
per model is executed locally per phase and its result recorded in
docs/GAPS.md — same kernels, same code paths, bigger grids.
"""

import numpy as np
import pytest

from conftest import requires_gpp, requires_ort, requires_torch_models
from mdlc.compiler import compile_onnx
from mdlc.testing.tolerances import FP32_NETWORK

RES = 64            # divisible by 32 (both nets downsample 5x)


@pytest.fixture(scope="session")
def small_mobilenetv2(tmp_path_factory):
    from mdlc.tools.build_mobilenetv2 import export
    path = tmp_path_factory.mktemp("models") / "mobilenetv2_small.onnx"
    export(str(path), res=RES)
    return str(path)


@pytest.fixture(scope="session")
def small_repvit(tmp_path_factory):
    from mdlc.tools.build_repvit import export
    path = tmp_path_factory.mktemp("models") / "repvit_small.onnx"
    export(str(path), res=RES)
    return str(path)


def _feeds(res=RES, seed=0):
    return {"input": np.random.default_rng(seed).standard_normal(
        (1, 3, res, res)).astype(np.float32)}


def _sim_parity(model_path):
    """Compile (per-pass verified vs ORT golden), demand zero fallbacks, run
    the emitted kernels via the sim executor, diff against the ORT golden."""
    from mdlc.codegen.cuda.sim_executor import simulate_module

    feeds = _feeds()
    compiled = compile_onnx(model_path, feeds=feeds)
    assert compiled.module.fallbacks() == [], compiled.module.fallbacks()

    got = simulate_module(compiled.module, feeds)
    for oname in compiled.graph.outputs:
        want = compiled.golden[oname]
        assert float(np.abs(want).max()) > 100 * FP32_NETWORK.atol, \
            "golden magnitude too small — vacuous comparison"
        np.testing.assert_allclose(got[oname], want,
                                   rtol=FP32_NETWORK.rtol, atol=FP32_NETWORK.atol)
    return compiled


@requires_torch_models
@requires_ort
@requires_gpp
def test_mobilenetv2_sim_end_to_end(small_mobilenetv2):
    compiled = _sim_parity(small_mobilenetv2)
    kinds = {l.kind for l in compiled.module.plan}
    assert "depthwise" in kinds, "depthwise convs must use the direct kernel"
    # depthwise Conv+BN+ReLU6 fused groups exist
    fused_dw = [l for l in compiled.module.plan if l.kind == "depthwise"
                and l.meta["node"].attrs.get("activation") == "Clip"]
    assert fused_dw, "Conv+BN+ReLU6 should fuse onto depthwise convs"


@requires_torch_models
@requires_ort
@requires_gpp
def test_repvit_sim_end_to_end(small_repvit):
    compiled = _sim_parity(small_repvit)
    kinds = {l.kind for l in compiled.module.plan}
    assert {"depthwise", "elementwise", "reduce", "gemm"} <= kinds
    # the GELU chains fused: at least one elementwise kernel computes erff
    srcs = "\n".join(compiled.module.kernels.values())
    assert "erff(" in srcs, "fused GELU (Erf) kernels expected"


@requires_torch_models
@requires_ort
def test_models_reference_parity_full_res():
    """Full 224x224 through the reference path (fast, no sim): compile report
    truth — zero fallbacks and ORT parity for all three models."""
    import os
    for path in ("examples/resnet18.onnx", "examples/mobilenetv2.onnx",
                 "examples/repvit_m0_9.onnx"):
        if not os.path.exists(path):
            pytest.skip(f"{path} not built (run the exporters in mdlc.tools)")
        compiled = compile_onnx(path)
        assert compiled.module.fallbacks() == [], (path, compiled.module.fallbacks())
        checks = compiled.verify()
        assert all(c.ok for c in checks), (path, [str(c) for c in checks])
