"""Fuse a pointwise activation onto the Conv/Gemm that feeds it.

Conv -> Relu becomes a single ``FusedConvAct`` node; Gemm -> Relu becomes
``FusedGemmAct``. The activation is applied in-register at the tail of the
matmul/conv epilogue, so the intermediate pre-activation tensor never touches
global memory. This is the GEMM+bias+activation and Conv+activation fusion.
"""

from __future__ import annotations

import numpy as np

from mdlc.ir import Graph, Node
from mdlc.passes.base import Pass

# Activations we can express in a kernel epilogue.
_ACTS = {"Relu", "Clip", "Sigmoid", "Tanh", "HardSwish", "HardSigmoid", "LeakyRelu"}
_PRODUCERS = {"Conv": "FusedConvAct", "Gemm": "FusedGemmAct"}


class FuseActivation(Pass):
    name = "fuse_activation"

    def run(self, graph: Graph) -> bool:
        changed = False
        producer = graph.producers()
        use_count = graph.use_count()

        for act in list(graph.nodes):
            if act.op_type not in _ACTS:
                continue
            src = producer.get(act.inputs[0])
            if src is None or src.op_type not in _PRODUCERS:
                continue
            feed = src.outputs[0]
            if use_count.get(feed, 0) != 1 or feed in graph.outputs:
                continue

            fused = Node(
                op_type=_PRODUCERS[src.op_type],
                inputs=list(src.inputs),
                outputs=list(act.outputs),
                attrs=dict(src.attrs),
                name=(src.name or src.op_type) + "+" + act.op_type,
                fused_from=[src.op_type, act.op_type],
            )
            fused.attrs["activation"] = act.op_type
            if not self._capture_act_params(graph, act, fused):
                # A parameter (e.g. Clip bound) is a runtime tensor we cannot
                # bake into the epilogue. Fusing anyway would silently change
                # the math, so leave the activation as its own node.
                continue

            # Splice: new fused node where the source was, drop both originals.
            idx = graph.nodes.index(src)
            graph.nodes[idx] = fused
            graph.remove_nodes([act])
            changed = True

        return changed

    def _capture_act_params(self, graph: Graph, act: Node, fused: Node) -> bool:
        """Bake the activation's parameters into the fused node's attrs.

        Returns False when a parameter exists but cannot be resolved to a
        compile-time constant — in that case fusion must not happen (a Clip
        whose bounds we drop would silently become identity).
        """
        if act.op_type == "Clip":
            ok_lo, lo = self._scalar(graph, act, 1, act.attr("min"))
            ok_hi, hi = self._scalar(graph, act, 2, act.attr("max"))
            if not (ok_lo and ok_hi):
                return False
            fused.attrs["clip_min"] = lo
            fused.attrs["clip_max"] = hi
        elif act.op_type == "LeakyRelu":
            fused.attrs["alpha"] = act.attr("alpha", 0.01)
        elif act.op_type == "HardSigmoid":
            fused.attrs["alpha"] = act.attr("alpha", 0.2)
            fused.attrs["beta"] = act.attr("beta", 0.5)
        return True

    def _scalar(self, graph: Graph, node: Node, idx: int, attr_fallback):
        """Resolve an optional scalar input to (resolved, value).

        An absent/empty input slot is a legitimate None (e.g. one-sided Clip);
        a *present* input that is not a constant initializer is unresolvable.
        """
        if len(node.inputs) > idx and node.inputs[idx]:
            arr = graph.initializers.get(node.inputs[idx])
            if arr is None:
                return False, None
            return True, float(np.asarray(arr).item())
        return True, (None if attr_fallback is None else float(attr_fallback))
