"""Memory planner: correctness of liveness reuse and reported savings."""

import numpy as np

from conftest import build_graph
from mdlc.passes import default_pipeline
from mdlc.passes.shape_inference import infer_shapes_by_execution
from mdlc.runtime.memory_planner import plan_memory


def _no_live_overlap(graph, plan):
    """No two tensors sharing a buffer may be simultaneously live."""
    order = graph.topo_sort()
    pos = {id(n): k for k, n in enumerate(order)}
    # birth/death per pooled tensor
    birth, death = {}, {}
    INF = len(order) + 1
    for k, n in enumerate(order):
        for o in n.outputs:
            if o in plan.assignment:
                birth[o] = k
    for k, n in enumerate(order):
        for e in n.real_inputs():
            if e in plan.assignment:
                death[e] = k
    for o in graph.outputs:
        if o in plan.assignment:
            death[o] = INF
    # group by buffer, check intervals are disjoint
    by_buf = {}
    for t, b in plan.assignment.items():
        by_buf.setdefault(b, []).append(t)
    for b, tensors in by_buf.items():
        ivals = sorted((birth.get(t, 0), death.get(t, INF)) for t in tensors)
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
