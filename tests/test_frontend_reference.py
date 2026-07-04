"""Frontend import + reference executor agree with ONNX Runtime."""

import numpy as np
import pytest

from conftest import build_graph, requires_ort
from mdlc.runtime import run_reference
from mdlc.testing.tolerances import FP32_NETWORK


def test_import_preserves_io():
    g, feeds, _ = build_graph("conv_bn_relu")
    assert g.runtime_inputs == ["x"]
    assert g.outputs == ["y"]
    # topo sort is a valid linearization
    order = g.topo_sort()
    assert len(order) == len(g.nodes)


def test_reference_runs_all_models(model_case):
    name, kw = model_case
    g, feeds, _ = build_graph(name, **kw)
    out = run_reference(g, feeds)
    assert set(out) == set(g.outputs)
    for v in out.values():
        assert np.all(np.isfinite(v))


@requires_ort
def test_reference_matches_onnxruntime(model_case):
    import onnxruntime as ort

    name, kw = model_case
    g, feeds, model = build_graph(name, **kw)
    ref = run_reference(g, feeds)
    sess = ort.InferenceSession(model.SerializeToString(),
                                providers=["CPUExecutionProvider"])
    outs = sess.run(None, {k: v.astype(np.float32) for k, v in feeds.items()})
    for (oname, oval) in zip([o.name for o in sess.get_outputs()], outs):
        np.testing.assert_allclose(ref[oname], oval,
                                   rtol=FP32_NETWORK.rtol, atol=FP32_NETWORK.atol)
