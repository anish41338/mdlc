"""The graph IR container and its core structural operations."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Iterable, Iterator, Optional

import numpy as np

from mdlc.ir.node import Node
from mdlc.ir.tensor import DType, TensorInfo


class Graph:
    """A dataflow graph of ``Node`` operators over named tensor edges.

    Invariants we try to keep:
      * Every value referenced by a node input/output has a ``TensorInfo`` in
        ``self.value_info`` (possibly with an unknown shape).
      * ``initializers`` holds constant tensors (weights, folded constants) as
        NumPy arrays, keyed by value name.
      * Graph inputs that also appear in ``initializers`` are treated as
        constants, not runtime inputs (ONNX allows this overlap).
    """

    def __init__(self, name: str = "graph") -> None:
        self.name = name
        self.nodes: list[Node] = []
        self.inputs: list[str] = []
        self.outputs: list[str] = []
        self.initializers: dict[str, np.ndarray] = {}
        self.value_info: dict[str, TensorInfo] = {}

    # ----- construction helpers -------------------------------------------

    def add_node(self, node: Node) -> Node:
        self.nodes.append(node)
        return node

    def set_info(self, info: TensorInfo) -> None:
        self.value_info[info.name] = info

    def info(self, name: str) -> Optional[TensorInfo]:
        return self.value_info.get(name)

    def set_initializer(self, name: str, array: np.ndarray) -> None:
        array = np.asarray(array)
        self.initializers[name] = array
        # Keep type/shape metadata in sync with the constant data.
        try:
            dtype = DType.from_numpy(array.dtype)
        except KeyError:
            dtype = DType.FLOAT32
        self.set_info(TensorInfo(name, dtype, tuple(int(d) for d in array.shape)))

    def is_constant(self, name: str) -> bool:
        return name in self.initializers

    # ----- graph queries --------------------------------------------------

    @property
    def runtime_inputs(self) -> list[str]:
        """Graph inputs that are *not* also constants — the real feed dict."""
        return [i for i in self.inputs if i not in self.initializers]

    def producers(self) -> dict[str, Node]:
        """Map each value name to the node that produces it."""
        out: dict[str, Node] = {}
        for node in self.nodes:
            for o in node.outputs:
                if o:
                    out[o] = node
        return out

    def consumers(self) -> dict[str, list[Node]]:
        """Map each value name to the list of nodes that consume it."""
        out: dict[str, list[Node]] = defaultdict(list)
        for node in self.nodes:
            for i in node.real_inputs():
                out[i].append(node)
        return out

    def use_count(self) -> dict[str, int]:
        """How many *consumer slots* reference each value (graph outputs +1).

        A value with use_count > 1 is shared, which makes fusing its producer
        into a single consumer illegal (the intermediate is still needed).
        """
        counts: dict[str, int] = defaultdict(int)
        for node in self.nodes:
            for i in node.real_inputs():
                counts[i] += 1
        for o in self.outputs:
            counts[o] += 1
        return counts

    # ----- ordering -------------------------------------------------------

    def topo_sort(self) -> list[Node]:
        """Return nodes in a valid execution order (Kahn's algorithm).

        Available values start as graph inputs + initializers. A node becomes
        ready once all its real inputs are available. Raises on a cycle.
        """
        available: set[str] = set(self.inputs) | set(self.initializers.keys())
        producer = self.producers()

        # in-degree = number of distinct not-yet-available producers feeding it
        indeg: dict[int, int] = {}
        deps: dict[int, set[int]] = defaultdict(set)
        dependents: dict[int, list[int]] = defaultdict(list)
        idx_of = {id(n): k for k, n in enumerate(self.nodes)}

        for k, node in enumerate(self.nodes):
            need: set[int] = set()
            for i in node.real_inputs():
                if i in available:
                    continue
                pnode = producer.get(i)
                if pnode is not None:
                    need.add(idx_of[id(pnode)])
            deps[k] = need
            indeg[k] = len(need)
            for d in need:
                dependents[d].append(k)

        ready = deque(k for k in range(len(self.nodes)) if indeg[k] == 0)
        order: list[Node] = []
        while ready:
            k = ready.popleft()
            order.append(self.nodes[k])
            for m in dependents[k]:
                indeg[m] -= 1
                if indeg[m] == 0:
                    ready.append(m)

        if len(order) != len(self.nodes):
            raise ValueError("graph has a cycle or dangling dependency")
        return order

    def reorder_topologically(self) -> None:
        self.nodes = self.topo_sort()

    def __iter__(self) -> Iterator[Node]:
        return iter(self.nodes)

    def __len__(self) -> int:
        return len(self.nodes)

    # ----- mutation -------------------------------------------------------

    def remove_nodes(self, to_remove: Iterable[Node]) -> None:
        drop = {id(n) for n in to_remove}
        self.nodes = [n for n in self.nodes if id(n) not in drop]

    def clone(self) -> "Graph":
        """Deep-ish copy: new Node objects, shared (immutable) weight arrays."""
        g = Graph(self.name)
        g.inputs = list(self.inputs)
        g.outputs = list(self.outputs)
        g.initializers = dict(self.initializers)
        g.value_info = {k: TensorInfo(v.name, v.dtype, v.shape) for k, v in self.value_info.items()}
        for n in self.nodes:
            g.nodes.append(
                Node(
                    op_type=n.op_type,
                    inputs=list(n.inputs),
                    outputs=list(n.outputs),
                    attrs=dict(n.attrs),
                    name=n.name,
                    fused_from=None if n.fused_from is None else list(n.fused_from),
                )
            )
        return g

    # ----- debug ----------------------------------------------------------

    def summary(self) -> str:
        lines = [f"Graph {self.name!r}: {len(self.nodes)} nodes"]
        lines.append(f"  inputs : {', '.join(self.runtime_inputs) or '(none)'}")
        lines.append(f"  outputs: {', '.join(self.outputs) or '(none)'}")
        op_hist: dict[str, int] = defaultdict(int)
        for n in self.nodes:
            op_hist[n.op_type] += 1
        hist = ", ".join(f"{k}x{v}" for k, v in sorted(op_hist.items()))
        lines.append(f"  ops    : {hist}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"<Graph {self.name!r} nodes={len(self.nodes)}>"
