"""Whole-graph: the generated kernels compose to the reference result."""

import numpy as np

from conftest import build_graph, requires_gpp
from mdlc.codegen.cuda.emit import emit_cuda_module
from mdlc.passes import default_pipeline
from mdlc.passes.shape_inference import infer_shapes_by_execution
from mdlc.runtime import run_reference


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
        np.testing.assert_allclose(got[o], ref[o], rtol=1e-3, atol=1e-3)


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
