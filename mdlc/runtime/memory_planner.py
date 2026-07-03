"""Liveness-based memory planner.

Intermediate activations dominate inference memory, yet most are short-lived: a
tensor is born when its producer runs and dies after its last consumer. By
computing those liveness intervals and reusing buffers via a greedy best-fit
allocator, many tensors share the same physical memory — the same idea as a
register allocator's linear scan, applied to device buffers.

This runs entirely on shape metadata (no device needed), so it is fully
testable on CPU and reports the headline number: planned bytes vs the naive
"every tensor its own buffer" total.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from mdlc.ir import DType, Graph


@dataclass
class Buffer:
    id: int
    size: int                       # bytes (high-water mark across reuses)
    tenants: list[str] = field(default_factory=list)


@dataclass
class MemoryPlan:
    assignment: dict[str, int]      # tensor name -> buffer id
    buffers: list[Buffer]
    weights_bytes: int
    naive_activation_bytes: int     # sum of all intermediate tensor sizes
    planned_activation_bytes: int   # sum of pooled buffer sizes

    @property
    def reuse_ratio(self) -> float:
        if self.naive_activation_bytes == 0:
            return 1.0
        return self.planned_activation_bytes / self.naive_activation_bytes

    def report(self) -> str:
        mb = 1024 * 1024
        lines = ["Memory plan:"]
        lines.append(f"  weights            : {self.weights_bytes / mb:8.3f} MB")
        lines.append(f"  activations (naive): {self.naive_activation_bytes / mb:8.3f} MB"
                     f"  across {len({t for b in self.buffers for t in b.tenants})} tensors")
        lines.append(f"  activations (pool) : {self.planned_activation_bytes / mb:8.3f} MB"
                     f"  across {len(self.buffers)} buffers")
        saved = self.naive_activation_bytes - self.planned_activation_bytes
        pct = 100.0 * saved / self.naive_activation_bytes if self.naive_activation_bytes else 0.0
        lines.append(f"  saved              : {saved / mb:8.3f} MB ({pct:.1f}%)")
        return "\n".join(lines)


def _nbytes(graph: Graph, name: str) -> int:
    info = graph.info(name)
    if info is None or not info.is_static:
        raise ValueError(f"tensor {name!r} has no static shape; run shape inference first")
    nb = info.nbytes()
    assert nb is not None
    return nb


def plan_memory(graph: Graph) -> MemoryPlan:
    """Compute a buffer assignment for all intermediate activations."""
    order = graph.topo_sort()
    pos = {id(n): k for k, n in enumerate(order)}

    weights = set(graph.initializers.keys())
    runtime_inputs = set(graph.runtime_inputs)
    outputs = set(graph.outputs)

    # last-use index for each value (graph outputs never die).
    last_use: dict[str, int] = {}
    for k, node in enumerate(order):
        for e in node.real_inputs():
            last_use[e] = k
    INF = len(order) + 1
    for o in outputs:
        last_use[o] = INF

    # Tensors we actually pool: produced-by-a-node activations (not weights,
    # not graph inputs — those have fixed external storage).
    assignment: dict[str, int] = {}
    buffers: list[Buffer] = []
    free: list[int] = []            # buffer ids available for reuse
    naive_total = 0
    pooled_tensors: set[str] = set()

    def acquire(name: str) -> int:
        need = _nbytes(graph, name)
        # best-fit among free buffers
        best = -1
        best_size = None
        for bid in free:
            bsz = buffers[bid].size
            if bsz >= need and (best_size is None or bsz < best_size):
                best, best_size = bid, bsz
        if best == -1:
            # reuse the largest free buffer (grow it) or make a new one
            if free:
                best = max(free, key=lambda b: buffers[b].size)
                buffers[best].size = max(buffers[best].size, need)
            else:
                best = len(buffers)
                buffers.append(Buffer(id=best, size=need))
        free.remove(best) if best in free else None
        buffers[best].tenants.append(name)
        return best

    for k, node in enumerate(order):
        # produce outputs first (so an output can't alias a still-live input of
        # the same node — inputs are freed only after this step).
        for o in node.outputs:
            if not o or o in weights or o in runtime_inputs:
                continue
            if o in assignment:
                continue
            naive_total += _nbytes(graph, o)
            pooled_tensors.add(o)
            assignment[o] = acquire(o)

        # free inputs whose last use is now, returning their buffers to pool
        # (but never free graph outputs or weights).
        for e in node.real_inputs():
            if e in weights or e in runtime_inputs or e in outputs:
                continue
            if last_use.get(e, -1) <= k and e in assignment:
                bid = assignment[e]
                if bid not in free:
                    free.append(bid)

    weights_bytes = 0
    for w, arr in graph.initializers.items():
        weights_bytes += int(arr.size) * arr.dtype.itemsize

    planned = sum(b.size for b in buffers)
    return MemoryPlan(
        assignment=assignment,
        buffers=buffers,
        weights_bytes=weights_bytes,
        naive_activation_bytes=naive_total,
        planned_activation_bytes=planned,
    )
