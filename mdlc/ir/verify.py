"""Structural graph verifier — cheap invariant checks run after every pass.

A pass that corrupts graph *structure* (dangling edge, duplicate producer,
cycle, orphan node) produces failures far from the cause, so we assert the
invariants at the moment a pass finishes. This is the structural complement to
the numerical pass verifier in ``mdlc/testing/harness.py``; both are wired into
``make_pass_verifier`` so any verified pipeline gets both for free.

Checks:
  * every real input of every node is defined (graph input, initializer, or
    some node's output);
  * every value has exactly one producer;
  * every graph output is defined;
  * the graph is a DAG (via topo_sort, which raises on cycles);
  * no orphan nodes (every node's outputs feed another node or a graph output);
  * initializer arrays agree with their recorded TensorInfo shape.
"""

from __future__ import annotations

from mdlc.ir.graph import Graph


class GraphVerifyError(AssertionError):
    """A structural invariant of the IR was violated."""


def verify_graph(g: Graph, *, context: str = "") -> None:
    """Raise ``GraphVerifyError`` on the first violated invariant."""
    where = f" (after {context})" if context else ""

    defined = set(g.inputs) | set(g.initializers)
    producer_of: dict[str, str] = {}
    for node in g.nodes:
        for o in node.outputs:
            if not o:
                continue
            if o in producer_of:
                raise GraphVerifyError(
                    f"value {o!r} has two producers ({producer_of[o]} and "
                    f"{node.op_type}){where}")
            if o in defined:
                raise GraphVerifyError(
                    f"value {o!r} produced by {node.op_type} shadows a graph "
                    f"input/initializer{where}")
            producer_of[o] = node.op_type
    defined |= set(producer_of)

    for node in g.nodes:
        for i in node.real_inputs():
            if i not in defined:
                raise GraphVerifyError(
                    f"node {node.op_type}({node.name!r}) reads undefined "
                    f"value {i!r}{where}")

    for o in g.outputs:
        if o not in defined:
            raise GraphVerifyError(f"graph output {o!r} is not produced{where}")

    # DAG check: topo_sort raises ValueError on a cycle / dangling dependency.
    try:
        g.topo_sort()
    except ValueError as e:
        raise GraphVerifyError(f"{e}{where}") from e

    # Orphans: a node none of whose outputs is consumed or exported. Passes
    # must clean up after themselves (or be followed by DCE inside the same
    # pass) so dead work never reaches codegen.
    consumed = {i for node in g.nodes for i in node.real_inputs()}
    live_sinks = consumed | set(g.outputs)
    for node in g.nodes:
        outs = [o for o in node.outputs if o]
        if outs and not any(o in live_sinks for o in outs):
            raise GraphVerifyError(
                f"orphan node {node.op_type}({node.name!r}): outputs {outs} "
                f"are never consumed{where}")

    for name, arr in g.initializers.items():
        info = g.info(name)
        if info is not None and info.is_static and tuple(info.shape) != tuple(arr.shape):
            raise GraphVerifyError(
                f"initializer {name!r} shape {arr.shape} != recorded "
                f"TensorInfo {info.shape}{where}")
