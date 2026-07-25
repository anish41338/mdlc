"""Memory planner: correctness of liveness reuse, view aliasing, alignment,
and reported savings."""

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from conftest import build_graph
from mdlc.frontend import import_onnx_model
from mdlc.ir import VIEW_OPS
from mdlc.passes import default_pipeline
from mdlc.passes.shape_inference import infer_shapes_by_execution
from mdlc.runtime.memory_planner import plan_memory


def _roots(graph):
    """Resolve every tensor to its storage root through view chains."""
    alias = {}
    for n in graph.topo_sort():
        if n.op_type in VIEW_OPS:
            alias[n.outputs[0]] = n.inputs[0]

    def root(name):
        while name in alias:
            name = alias[name]
        return name

    return root


def _no_live_overlap(graph, plan):
    """No two *storage roots* sharing a buffer may be simultaneously live.
    A view and its root share a buffer by design; their union interval is the
    storage's live range."""
    order = graph.topo_sort()
    root = _roots(graph)
    INF = len(order) + 1
    # union live interval per storage root
    birth, death = {}, {}
    for k, n in enumerate(order):
        for o in n.outputs:
            if o in plan.assignment:
                r = root(o)
                birth.setdefault(r, k)
    for k, n in enumerate(order):
        for e in n.real_inputs():
            if e in plan.assignment or root(e) in plan.assignment:
                r = root(e)
                death[r] = max(death.get(r, k), k)
    for o in graph.outputs:
        r = root(o)
        if r in plan.assignment:
            death[r] = INF
    by_buf = {}
    for t, b in plan.assignment.items():
        r = root(t)
        if r in plan.assignment:
            by_buf.setdefault(b, set()).add(r)
    for b, roots_ in by_buf.items():
        ivals = sorted((birth.get(t, 0), death.get(t, INF)) for t in roots_)
        for (s1, e1), (s2, e2) in zip(ivals, ivals[1:]):
            assert e1 <= s2, f"buffer {b}: live ranges overlap {(s1,e1)} {(s2,e2)}"


def test_plan_reuses_and_is_safe():
    g, feeds, _ = build_graph("residual_block", c=8, h=16, w=16)
    g = default_pipeline().run(g)
    infer_shapes_by_execution(g, feeds)
    plan = plan_memory(g)
    assert plan.planned_activation_bytes <= plan.naive_activation_bytes
    assert len(plan.buffers) >= 1
    _no_live_overlap(g, plan)


def test_plan_saves_memory_on_deep_graph():
    import onnx
    from mdlc.frontend import import_onnx

    onnx.load("examples/resnet18.onnx")   # ensure file exists
    g = import_onnx("examples/resnet18.onnx")
    from mdlc.passes.shape_inference import make_dummy_feeds
    feeds = make_dummy_feeds(g, batch=1)
    g = default_pipeline().run(g)
    infer_shapes_by_execution(g, feeds)
    plan = plan_memory(g)
    # pooling should beat the naive sum substantially on a deep net
    assert plan.planned_activation_bytes < 0.6 * plan.naive_activation_bytes
    _no_live_overlap(g, plan)


def test_offsets_are_256_aligned_and_pool_is_dense():
    g, feeds, _ = build_graph("residual_block", c=8, h=16, w=16)
    g = default_pipeline().run(g)
    infer_shapes_by_execution(g, feeds)
    plan = plan_memory(g)
    assert plan.align == 256
    for b in plan.buffers:
        assert b.offset % 256 == 0
        assert b.size % 256 == 0
    assert plan.pool_bytes == sum(b.size for b in plan.buffers)
    # offsets tile the pool back to back
    assert sorted(b.offset for b in plan.buffers) == \
        [sum(x.size for x in plan.buffers[:i]) for i in range(len(plan.buffers))]
    for t in plan.assignment:
        assert plan.offset_of(t) % 256 == 0


def _view_hazard_model():
    """Relu -> Flatten -> Gemm: the executors alias Flatten's output onto its
    input buffer, so the planner must keep that storage live through the Gemm
    even though the Relu output's last *direct* use is the Flatten itself."""
    w = numpy_helper.from_array(
        np.random.default_rng(0).standard_normal((64, 8)).astype(np.float32), "w")
    nodes = [
        helper.make_node("Relu", ["x"], ["a"]),
        helper.make_node("Flatten", ["a"], ["f"], axis=1),
        helper.make_node("Gemm", ["f", "w"], ["g"]),
        helper.make_node("Relu", ["g"], ["y"]),
    ]
    graph = helper.make_graph(
        nodes, "viewhazard",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4, 4, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 8])],
        [w])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    return model


def test_view_output_shares_buffer_and_extends_lifetime():
    g = import_onnx_model(_view_hazard_model())
    feeds = {"x": np.random.default_rng(1).standard_normal(
        (1, 4, 4, 4)).astype(np.float32)}
    infer_shapes_by_execution(g, feeds)
    plan = plan_memory(g)
    # the view output is the same storage as its input
    assert plan.assignment["f"] == plan.assignment["a"]
    # nothing else moves into that buffer while the view is still consumed:
    # the Gemm output must NOT share the (a, f) buffer
    assert plan.assignment["g"] != plan.assignment["a"]
    _no_live_overlap(g, plan)
    # views own no bytes: naive total counts a, g, y (+nothing for f)
    per = {t: int(np.prod(s)) * 4
           for t, s in (("a", (1, 4, 4, 4)), ("g", (1, 8)), ("y", (1, 8)))}
    assert plan.naive_activation_bytes == sum(per.values())
