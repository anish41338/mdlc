"""Liveness-based memory planner.

Intermediate activations dominate inference memory, yet most are short-lived: a
tensor is born when its producer runs and dies after its last consumer. By
computing those liveness intervals and reusing buffers via a greedy best-fit
allocator, many tensors share the same physical memory — the same idea as a
register allocator's linear scan, applied to device buffers.

Two properties matter for the GPU path:

* **View aliasing.** Flatten/Reshape/&c. are zero-copy in the executors (the
  output aliases the input's bytes), so the planner must treat a view output
  as the *same storage* as its input — otherwise the input's buffer could be
  recycled while the view is still live. Uses of an alias count as uses of
  its root, and view outputs get no buffer of their own (they also don't
  count toward the naive total: honest accounting, a view was never a copy).

* **Alignment.** Every buffer size is rounded up to ``align`` (256 bytes, the
  guarantee cuMemAlloc itself gives) and the plan carries an explicit byte
  offset per buffer into one contiguous pool, so a single device allocation
  serves every activation with vectorized-load/coalescing-safe pointers.

This runs entirely on shape metadata (no device needed), so it is fully
testable on CPU and reports the headline number: planned bytes vs the naive
"every tensor its own buffer" total.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mdlc.ir import VIEW_OPS, Graph

ALIGN = 256  # bytes; matches the CUDA driver's own allocation granularity


@dataclass
class Buffer:
    id: int
    size: int                       # bytes (aligned high-water mark across reuses)
    offset: int = 0                 # byte offset into the pooled allocation
    tenants: list[str] = field(default_factory=list)


@dataclass
class MemoryPlan:
    assignment: dict[str, int]      # tensor name -> buffer id (aliases resolved)
    buffers: list[Buffer]
    weights_bytes: int
    naive_activation_bytes: int     # sum of all intermediate tensor sizes
    planned_activation_bytes: int   # sum of pooled buffer sizes
    align: int = ALIGN

    @property
    def pool_bytes(self) -> int:
        """Size of the single device allocation that serves all activations."""
        return self.planned_activation_bytes

    def offset_of(self, tensor: str) -> int:
        return self.buffers[self.assignment[tensor]].offset

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
                     f"  across {len(self.buffers)} buffers "
                     f"({self.align}-byte aligned offsets)")
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


def plan_memory(graph: Graph, *, align: int = ALIGN) -> MemoryPlan:
    """Compute a pooled, aligned buffer assignment for all intermediate
    activations."""
    order = graph.topo_sort()

    weights = set(graph.initializers.keys())
    runtime_inputs = set(graph.runtime_inputs)
    outputs = set(graph.outputs)

    # View outputs alias their input's storage; resolve every name to the
    # tensor that actually owns bytes.
    alias: dict[str, str] = {}
    for node in order:
        if node.op_type in VIEW_OPS:
            alias[node.outputs[0]] = node.inputs[0]

    def root(name: str) -> str:
        while name in alias:
            name = alias[name]
        return name

    # last-use index per storage root (graph outputs never die).
    last_use: dict[str, int] = {}
    for k, node in enumerate(order):
        for e in node.real_inputs():
            last_use[root(e)] = k
    INF = len(order) + 1
    for o in outputs:
        last_use[root(o)] = INF

    # Tensors we actually pool: produced-by-a-node activations (not weights,
    # not graph inputs — those have fixed external storage; not view outputs —
    # those are their root's bytes).
    assignment: dict[str, int] = {}
    buffers: list[Buffer] = []
    free: list[int] = []            # buffer ids available for reuse
    naive_total = 0

    def aligned(nb: int) -> int:
        return (nb + align - 1) // align * align

    def acquire(name: str) -> int:
        need = aligned(_nbytes(graph, name))
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
            r = root(o)
            if r != o:
                # view: share the root's buffer if the root is pooled; roots
                # that are weights/graph-inputs have external storage.
                if r in assignment:
                    assignment[o] = assignment[r]
                    buffers[assignment[r]].tenants.append(o)
                continue
            naive_total += _nbytes(graph, o)
            assignment[o] = acquire(o)

        # free inputs whose storage root's last use is now, returning their
        # buffers to the pool (graph outputs never free: their roots carry
        # an infinite last use).
        for e in node.real_inputs():
            r = root(e)
            if r in weights or r in runtime_inputs:
                continue
            if last_use.get(r, -1) <= k and r in assignment:
                bid = assignment[r]
                if bid not in free:
                    free.append(bid)

    # Lay the buffers out in one pool: aligned sizes stacked back to back
    # yield aligned offsets by construction; assert the invariant anyway.
    off = 0
    for b in buffers:
        b.size = aligned(b.size)
        b.offset = off
        assert b.offset % align == 0, (b.id, b.offset)
        off += b.size

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
        align=align,
    )
