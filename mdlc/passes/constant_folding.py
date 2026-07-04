"""Constant folding: precompute subgraphs whose inputs are all constant.

Any node all of whose real inputs are initializers can be evaluated at compile
time with the reference executor; its outputs become new initializers and the
node is removed. We iterate to a fixed point so chains collapse.
"""

from __future__ import annotations

import numpy as np

from mdlc.ir import Graph, Node
from mdlc.passes.base import Pass
from mdlc.runtime.reference import ReferenceExecutor

# Ops we refuse to fold even if constant: stateful, or so large that folding
# would bloat the model (none here, but a hook for future policy).
_NEVER_FOLD = {"FusedElementwise"}


class ConstantFolding(Pass):
    name = "constant_folding"

    def __init__(self, max_numel: int = 8_000_000) -> None:
        # Guardrail so we don't materialize an enormous constant tensor.
        self.max_numel = max_numel

    def run(self, graph: Graph) -> bool:
        changed = False
        executor = ReferenceExecutor(graph)

        progress = True
        while progress:
            progress = False
            for node in list(graph.nodes):
                if node.op_type in _NEVER_FOLD:
                    continue
                ins = node.real_inputs()
                # Zero-input nodes: only Constant is safe to fold (its value
                # lives in an attribute); anything else zero-input is either
                # stateful or nondeterministic (e.g. RandomNormal).
                if not ins and node.op_type != "Constant":
                    continue
                if not all(graph.is_constant(i) for i in ins):
                    continue
                if not self._safe_size(graph, node):
                    continue
                try:
                    outs = executor._exec_node(node, dict(graph.initializers))
                except (NotImplementedError, Exception):
                    continue
                for oname, oval in zip(node.outputs, outs):
                    if oname:
                        graph.set_initializer(oname, np.asarray(oval))
                graph.remove_nodes([node])
                changed = True
                progress = True

        return changed

    def _safe_size(self, graph: Graph, node: Node) -> bool:
        total = 0
        for i in node.real_inputs():
            arr = graph.initializers.get(i)
            if arr is not None:
                total += arr.size
        return total <= self.max_numel
