"""Shape inference by execution.

Rather than reimplement every op's shape rule, we run the NumPy reference
executor once on a concrete (or dummy) input and record the actual shape and
dtype of every value. Exact, robust, and it naturally handles fused ops. The
only cost is one CPU forward pass at compile time.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from mdlc.ir import DType, Graph, TensorInfo
from mdlc.runtime.reference import run_reference


def make_dummy_feeds(graph: Graph, batch: int = 1, seed: int = 0) -> dict[str, np.ndarray]:
    """Fabricate inputs for shape inference, binding symbolic dims to ``batch``."""
    rng = np.random.default_rng(seed)
    feeds = {}
    for name in graph.runtime_inputs:
        info = graph.info(name)
        if info is None or not info.shape:
            raise ValueError(f"input {name!r} has no shape; cannot fabricate")
        shape = [batch if not isinstance(d, int) or d < 0 else int(d) for d in info.shape]
        np_dt = DType.to_numpy(info.dtype)
        if np.issubdtype(np_dt, np.integer):
            feeds[name] = rng.integers(0, 2, size=shape).astype(np_dt)
        else:
            feeds[name] = rng.standard_normal(shape).astype(np_dt)
    return feeds


def infer_shapes_by_execution(
    graph: Graph, feeds: Optional[dict[str, np.ndarray]] = None, *, batch: int = 1
) -> Graph:
    """Populate ``graph.value_info`` with concrete static shapes/dtypes."""
    if feeds is None:
        feeds = make_dummy_feeds(graph, batch=batch)
    env = run_reference(graph, feeds, return_all=True)
    for name, arr in env.items():
        arr = np.asarray(arr)
        try:
            dt = DType.from_numpy(arr.dtype)
        except KeyError:
            dt = DType.FLOAT32
        graph.set_info(TensorInfo(name, dt, tuple(int(d) for d in arr.shape)))
    return graph
