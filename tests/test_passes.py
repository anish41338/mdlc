"""Graph passes: numerical preservation, fusion structure, legality."""

import numpy as np
import pytest

from conftest import build_graph
from mdlc.frontend import import_onnx_model
from mdlc.passes import (
    ConstantFolding,
    DeadCodeElimination,
    FuseActivation,
    FuseConvBN,
    FuseElementwise,
    default_pipeline,
)
from mdlc.passes.base import PassManager
from mdlc.runtime import run_reference
from mdlc.testing.harness import make_pass_verifier
from mdlc.tools import build_models as bm


def _golden(g, feeds):
    return run_reference(g.clone(), feeds)


def test_pipeline_preserves_numerics(model_case):
    name, kw = model_case
    g, feeds, _ = build_graph(name, **kw)
    golden = _golden(g, feeds)
    pm = default_pipeline(verify=make_pass_verifier(feeds, golden))
    g2 = pm.run(g)             # verifier raises if any pass drifts
    out = run_reference(g2, feeds)
    for o in g2.outputs:
        np.testing.assert_allclose(out[o], golden[o], rtol=1e-3, atol=1e-4)


def test_conv_bn_relu_fuses_to_single_node():
    g, feeds, _ = build_graph("conv_bn_relu")
    g = default_pipeline().run(g)
    assert len(g.nodes) == 1
    assert g.nodes[0].op_type == "FusedConvAct"
    assert g.nodes[0].attrs["activation"] == "Relu"


def test_conv_bn_fold_is_arithmetic_only():
    g, feeds, _ = build_graph("conv_bn_relu")
    before = run_reference(g.clone(), feeds)
    FuseConvBN().run(g)
    # BN gone, conv now carries a bias initializer
    assert not any(n.op_type == "BatchNormalization" for n in g.nodes)
    after = run_reference(g, feeds)
    for o in g.outputs:
        np.testing.assert_allclose(after[o], before[o], rtol=1e-4, atol=1e-5)


def test_elementwise_fusion_groups_add_relu():
    g, feeds, _ = build_graph("residual_block", c=4, h=8, w=8)
    g = default_pipeline().run(g)
    fe = [n for n in g.nodes if n.op_type == "FusedElementwise"]
    assert len(fe) == 1
    ops = [n.op_type for n in fe[0].attrs["subgraph"]]
    assert ops == ["Add", "Relu"]


def test_fusion_respects_multiple_consumers():
    """An intermediate with two consumers must not be swallowed into one."""
    # x feeds both a conv and the residual Add; it must remain a shared input.
    g, feeds, _ = build_graph("residual_block", c=4, h=8, w=8)
    uc = g.use_count()
    assert uc["x"] >= 2
    g2 = default_pipeline().run(g)
    # 'x' is still referenced by more than one node after fusion
    cons = g2.consumers()
    assert len(cons["x"]) >= 2


def test_constant_folding_removes_const_subgraph():
    # A graph that adds two constants then multiplies by input.
    from onnx import TensorProto, helper, numpy_helper

    a = np.full((4,), 2.0, np.float32)
    b = np.full((4,), 3.0, np.float32)
    nodes = [
        helper.make_node("Add", ["a", "b"], ["c"]),
        helper.make_node("Mul", ["x", "c"], ["y"]),
    ]
    graph = helper.make_graph(
        nodes, "constfold",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [4])],
        [numpy_helper.from_array(a, "a"), numpy_helper.from_array(b, "b")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    g = import_onnx_model(model)
    changed = ConstantFolding().run(g)
    assert changed
    assert not any(n.op_type == "Add" for n in g.nodes)   # folded away
    assert g.is_constant("c")
    out = run_reference(g, {"x": np.ones(4, np.float32)})
    np.testing.assert_allclose(out["y"], np.full(4, 5.0), rtol=0, atol=1e-6)


def test_dce_drops_dead_nodes():
    from onnx import TensorProto, helper

    nodes = [
        helper.make_node("Relu", ["x"], ["live"]),
        helper.make_node("Sigmoid", ["x"], ["dead"]),   # never used
    ]
    graph = helper.make_graph(
        nodes, "dce",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [4])],
        [helper.make_tensor_value_info("live", TensorProto.FLOAT, [4])],
        [],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    g = import_onnx_model(model)
    DeadCodeElimination().run(g)
    assert [n.op_type for n in g.nodes] == ["Relu"]
