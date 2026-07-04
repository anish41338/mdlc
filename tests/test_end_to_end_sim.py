"""Whole-graph: the generated kernels compose to the reference result."""

import numpy as np
import pytest

from conftest import build_graph, requires_gpp
from mdlc.codegen.cuda.emit import emit_cuda_module
from mdlc.passes import default_pipeline
from mdlc.passes.shape_inference import infer_shapes_by_execution
from mdlc.runtime import run_reference
from mdlc.testing.tolerances import FP32_NETWORK


@requires_gpp
def test_generated_module_matches_reference(model_case):
    name, kw = model_case
    g, feeds, _ = build_graph(name, **kw)
    g = default_pipeline().run(g)
    infer_shapes_by_execution(g, feeds)
    module = emit_cuda_module(g)

    from mdlc.codegen.cuda.sim_executor import simulate_module
    got = simulate_module(module, feeds)
    ref = run_reference(g, feeds)
    for o in g.outputs:
        np.testing.assert_allclose(got[o], ref[o],
                                   rtol=FP32_NETWORK.rtol, atol=FP32_NETWORK.atol)


@requires_gpp
@pytest.mark.parametrize("n", [1, 2, 8])
@pytest.mark.parametrize("model_name,kw", [
    ("conv_bn_relu", dict(cin=3, cout=8, h=10, w=10)),
    ("residual_block", dict(c=4, h=8, w=8)),
])
def test_conv_models_batch_n(model_name, kw, n):
    """Batch is a permanent axis of every conv parity test: the N>1 im2col
    lowering bug (audit finding) must be structurally impossible to reintroduce."""
    from mdlc.codegen.cuda.sim_executor import simulate_module
    from mdlc.frontend import import_onnx_model
    from mdlc.tools import build_models as bm

    model = bm.build(model_name, n=n, **kw)
    g = import_onnx_model(model)
    feeds = {g.runtime_inputs[0]:
             np.random.default_rng(n).standard_normal(
                 g.info(g.runtime_inputs[0]).shape).astype(np.float32)}
    g = default_pipeline().run(g)
    infer_shapes_by_execution(g, feeds)
    module = emit_cuda_module(g)
    got = simulate_module(module, feeds)
    ref = run_reference(g, feeds)
    for o in g.outputs:
        np.testing.assert_allclose(got[o], ref[o],
                                   rtol=FP32_NETWORK.rtol, atol=FP32_NETWORK.atol)


def test_module_emits_sources_without_gpu(model_case):
    """Source generation needs no compiler or GPU."""
    name, kw = model_case
    g, feeds, _ = build_graph(name, **kw)
    g = default_pipeline().run(g)
    infer_shapes_by_execution(g, feeds)
    module = emit_cuda_module(g)
    src = module.sources()
    assert "__global__" in src
    assert len(module.plan) >= 1
