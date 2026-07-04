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
from mdlc.testing.tolerances import BITWISE, FP32_NETWORK, FP32_SAME_ORDER
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
        np.testing.assert_allclose(out[o], golden[o],
                                   rtol=FP32_NETWORK.rtol, atol=FP32_NETWORK.atol)


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
        np.testing.assert_allclose(after[o], before[o],
                                   rtol=FP32_SAME_ORDER.rtol, atol=FP32_SAME_ORDER.atol)


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
    np.testing.assert_allclose(out["y"], np.full(4, 5.0),
                               rtol=BITWISE.rtol, atol=BITWISE.atol)


def _clip_from_constants_model():
    """Conv -> Clip(0,6) where the bounds come from Constant *nodes* (the
    opset>=11 torch-export pattern that once silently fused to identity)."""
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(7)
    w = rng.standard_normal((4, 3, 3, 3)).astype(np.float32)
    nodes = [
        helper.make_node("Constant", [], ["lo"],
                         value=numpy_helper.from_array(np.float32(0.0), "lo_v")),
        helper.make_node("Constant", [], ["hi"],
                         value=numpy_helper.from_array(np.float32(6.0), "hi_v")),
        helper.make_node("Conv", ["x", "w"], ["c"],
                         kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        helper.make_node("Clip", ["c", "lo", "hi"], ["y"]),
    ]
    graph = helper.make_graph(
        nodes, "clip_const",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 8, 8])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4, 8, 8])],
        [numpy_helper.from_array(w, "w")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    return import_onnx_model(model)


def test_clip_with_unresolvable_bounds_is_not_fused():
    """FuseActivation alone (no constant folding) cannot resolve Constant-node
    bounds — it must leave the Clip in place rather than fuse a wrong identity."""
    g = _clip_from_constants_model()
    FuseActivation().run(g)
    assert any(n.op_type == "Clip" for n in g.nodes), \
        "Clip with runtime-tensor bounds must not be fused away"


def test_clip_bounds_resolved_through_pipeline():
    """Through the default pipeline, Constants fold to initializers and the
    Clip fuses with correct bounds; the clamp must actually happen."""
    g = _clip_from_constants_model()
    feeds = {"x": np.random.default_rng(8).standard_normal((1, 3, 8, 8)).astype(np.float32) * 3}
    golden = _golden(g, feeds)
    g = default_pipeline(verify=make_pass_verifier(feeds, golden)).run(g)
    fused = [n for n in g.nodes if n.op_type == "FusedConvAct"]
    assert len(fused) == 1
    assert fused[0].attrs["clip_min"] == 0.0
    assert fused[0].attrs["clip_max"] == 6.0
    out = run_reference(g, feeds)["y"]
    assert out.min() >= 0.0 and out.max() <= 6.0
    np.testing.assert_allclose(out, golden["y"],
                               rtol=FP32_SAME_ORDER.rtol, atol=FP32_SAME_ORDER.atol)


def test_constant_folding_materializes_constant_nodes():
    g = _clip_from_constants_model()
    ConstantFolding().run(g)
    assert not any(n.op_type == "Constant" for n in g.nodes)
    assert g.is_constant("lo") and g.is_constant("hi")


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
