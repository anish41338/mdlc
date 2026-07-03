"""Dead-code elimination: drop nodes and initializers nothing observes."""

from __future__ import annotations

from mdlc.ir import Graph
from mdlc.passes.base import Pass


class DeadCodeElimination(Pass):
    name = "dce"

    def run(self, graph: Graph) -> bool:
        changed = False

        # Backward reachability from graph outputs.
        producer = graph.producers()
        needed: set[str] = set(graph.outputs)
        stack = list(graph.outputs)
        live_nodes: set[int] = set()
        while stack:
            val = stack.pop()
            node = producer.get(val)
            if node is None or id(node) in live_nodes:
                continue
            live_nodes.add(id(node))
            for i in node.real_inputs():
                if i not in needed:
                    needed.add(i)
                    stack.append(i)

        before = len(graph.nodes)
        graph.nodes = [n for n in graph.nodes if id(n) in live_nodes]
        if len(graph.nodes) != before:
            changed = True

        # Drop initializers and value_info nothing references anymore.
        used: set[str] = set(graph.outputs) | set(graph.inputs)
        for n in graph.nodes:
            used.update(n.real_inputs())
            used.update(o for o in n.outputs if o)
        for k in list(graph.initializers):
            if k not in used:
                del graph.initializers[k]
                changed = True
        for k in list(graph.value_info):
            if k not in used:
                del graph.value_info[k]

        return changed
