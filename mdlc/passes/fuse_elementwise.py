"""Fuse maximal chains of pointwise ops into one ``FusedElementwise`` node.

Elementwise ops (Add/Mul/Relu/Sigmoid/...) are memory-bound: each one reads and
writes a full tensor for a trivial amount of arithmetic. Chaining them into a
single kernel means the whole chain is computed per element in registers, with
one global read of each input and one global write of each output — eliminating
every intermediate tensor.

Legality (the classic rule): we may absorb a producer's output into a consumer
only if that output has exactly one consumer and is not a graph output. We
encode this as union-find over *single-use, non-output* edges, so every edge
internal to a group is produced and consumed exactly once inside it. An
intermediate with multiple consumers becomes a materialized group output
instead — never silently duplicated.
"""

from __future__ import annotations

from collections import defaultdict

from mdlc.ir import Graph, Node
from mdlc.passes.base import Pass

ELEMENTWISE = {
    "Add", "Sub", "Mul", "Div",
    "Relu", "Clip", "Sigmoid", "Tanh", "HardSwish", "HardSigmoid", "LeakyRelu",
    "Erf",   # GELU decomposes to Div-Erf-Add-Mul-Mul at opset 17
}


class _DSU:
    def __init__(self, n: int) -> None:
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


class FuseElementwise(Pass):
    name = "fuse_elementwise"

    def run(self, graph: Graph) -> bool:
        nodes = graph.nodes
        idx_of = {id(n): k for k, n in enumerate(nodes)}
        ew = [k for k, n in enumerate(nodes) if n.op_type in ELEMENTWISE]
        if not ew:
            return False

        producer = graph.producers()
        use_count = graph.use_count()
        consumers = graph.consumers()
        out_set = set(graph.outputs)

        dsu = _DSU(len(nodes))
        for k in ew:
            node = nodes[k]
            for e in node.real_inputs():
                p = producer.get(e)
                if p is None or p.op_type not in ELEMENTWISE:
                    continue
                # union only across an edge used exactly once and not exported
                if use_count.get(e, 0) == 1 and e not in out_set:
                    dsu.union(idx_of[id(p)], k)

        groups: dict[int, list[int]] = defaultdict(list)
        for k in ew:
            groups[dsu.find(k)].append(k)

        changed = False
        # Preserve overall order: emit fused node at the position of the group's
        # last member, drop the rest.
        replacement: dict[int, Node] = {}
        drop: set[int] = set()

        for root, members in groups.items():
            if len(members) < 2:
                continue
            members_sorted = sorted(members)
            member_set = set(members_sorted)
            inner = [nodes[k] for k in members_sorted]

            internal_outputs = {o for n in inner for o in n.outputs if o}

            # Group inputs: edges read by the group but produced outside it.
            group_inputs: list[str] = []
            seen_in: set[str] = set()
            for n in inner:
                for e in n.real_inputs():
                    if e not in internal_outputs and e not in seen_in:
                        seen_in.add(e)
                        group_inputs.append(e)

            # Group outputs: produced inside, but needed outside (or exported).
            group_outputs: list[str] = []
            for n in inner:
                for o in n.outputs:
                    if not o:
                        continue
                    external = any(id(c) not in {id(nodes[m]) for m in member_set}
                                   for c in consumers.get(o, []))
                    if o in out_set or external:
                        group_outputs.append(o)

            fused = Node(
                op_type="FusedElementwise",
                inputs=group_inputs,
                outputs=group_outputs,
                attrs={"subgraph": inner},
                name="fused_ew_" + "_".join(str(k) for k in members_sorted),
                fused_from=[nodes[k].op_type for k in members_sorted],
            )
            replacement[members_sorted[-1]] = fused
            for k in members_sorted[:-1]:
                drop.add(k)
            drop.add(members_sorted[-1])  # original at this slot is replaced
            changed = True

        if not changed:
            return False

        new_nodes: list[Node] = []
        for k, n in enumerate(nodes):
            if k in replacement:
                new_nodes.append(replacement[k])
            elif k in drop:
                continue
            else:
                new_nodes.append(n)
        graph.nodes = new_nodes
        return changed
