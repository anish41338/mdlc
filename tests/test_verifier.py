"""Structural verifier: catches bad graphs, passes keep invariants, and every
pass is idempotent (pass(pass(g)) == pass(g))."""

import numpy as np
import pytest

from conftest import build_graph
from mdlc.ir import Graph, GraphVerifyError, Node, verify_graph
from mdlc.passes import (
    ConstantFolding,
    DeadCodeElimination,
    FuseActivation,
    FuseConvBN,
    FuseElementwise,
    default_pipeline,
)

ALL_PASSES = [ConstantFolding, FuseConvBN, DeadCodeElimination,
              FuseActivation, FuseElementwise]


def _structure(g: Graph):
    """A comparable snapshot of graph structure."""
    return [(n.op_type, tuple(n.inputs), tuple(n.outputs)) for n in g.topo_sort()]


# -- the verifier itself catches broken graphs ------------------------------

def test_verifier_accepts_valid_graph(model_case):
    name, kw = model_case
    g, _, _ = build_graph(name, **kw)
    verify_graph(g)


def test_verifier_rejects_undefined_input():
    g = Graph("bad")
    g.inputs = ["x"]
    g.outputs = ["y"]
    g.add_node(Node("Relu", ["nonexistent"], ["y"]))
    with pytest.raises(GraphVerifyError, match="undefined"):
        verify_graph(g)


def test_verifier_rejects_duplicate_producer():
    g = Graph("bad")
    g.inputs = ["x"]
    g.outputs = ["y"]
    g.add_node(Node("Relu", ["x"], ["y"]))
    g.add_node(Node("Sigmoid", ["x"], ["y"]))
    with pytest.raises(GraphVerifyError, match="two producers"):
        verify_graph(g)


def test_verifier_rejects_cycle():
    g = Graph("bad")
    g.inputs = ["x"]
    g.outputs = ["y"]
    g.add_node(Node("Add", ["x", "b"], ["a"]))
    g.add_node(Node("Add", ["a", "x"], ["b"]))
    g.add_node(Node("Relu", ["a"], ["y"]))
    with pytest.raises(GraphVerifyError):
        verify_graph(g)


def test_verifier_rejects_orphan_node():
    g = Graph("bad")
    g.inputs = ["x"]
    g.outputs = ["y"]
    g.add_node(Node("Relu", ["x"], ["y"]))
    g.add_node(Node("Sigmoid", ["x"], ["dead"]))
    with pytest.raises(GraphVerifyError, match="orphan"):
        verify_graph(g)


def test_verifier_rejects_missing_output():
    g = Graph("bad")
    g.inputs = ["x"]
    g.outputs = ["y", "never_made"]
    g.add_node(Node("Relu", ["x"], ["y"]))
    with pytest.raises(GraphVerifyError, match="not produced"):
        verify_graph(g)


# -- every pass preserves the invariants and is idempotent -------------------

@pytest.mark.parametrize("pass_cls", ALL_PASSES)
def test_pass_keeps_invariants_and_is_idempotent(pass_cls, model_case):
    name, kw = model_case
    g, _, _ = build_graph(name, **kw)
    # Exercise each pass on the graph the pipeline would hand it: run the
    # preceding default-pipeline passes first, verifying at every step.
    for cls in ALL_PASSES:
        p = cls()
        p.run(g)
        verify_graph(g, context=p.name)
        if cls is pass_cls:
            break
    snap = _structure(g)
    changed_again = pass_cls().run(g)
    verify_graph(g, context=f"{pass_cls.__name__} (2nd run)")
    assert _structure(g) == snap, f"{pass_cls.__name__} is not idempotent"
    assert not changed_again, (
        f"{pass_cls.__name__} reported changes on an already-converged graph")


def test_full_pipeline_output_verifies(model_case):
    name, kw = model_case
    g, _, _ = build_graph(name, **kw)
    g = default_pipeline().run(g)
    verify_graph(g, context="full pipeline")
