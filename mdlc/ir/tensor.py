"""Tensor type/shape metadata for the IR."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np

# A dimension is either a concrete int, or a symbolic name (e.g. "batch"), or
# None when entirely unknown. Symbolic dims survive until we bind a concrete
# input shape; codegen requires fully-static shapes.
Dim = Union[int, str, None]
Shape = tuple[Dim, ...]


class DType:
    """Thin wrapper mapping our dtype names to NumPy dtypes.

    We keep the set small on purpose. INT8 is included for the quantized
    GEMM/DP4A codegen path (the Samsung tie-in), even though the reference
    executor treats it as a normal integer matmul.
    """

    FLOAT32 = "float32"
    FLOAT16 = "float16"
    INT64 = "int64"
    INT32 = "int32"
    INT8 = "int8"
    BOOL = "bool"

    _TO_NUMPY = {
        FLOAT32: np.float32,
        FLOAT16: np.float16,
        INT64: np.int64,
        INT32: np.int32,
        INT8: np.int8,
        BOOL: np.bool_,
    }

    @classmethod
    def to_numpy(cls, name: str) -> np.dtype:
        if name not in cls._TO_NUMPY:
            raise KeyError(f"unsupported dtype: {name!r}")
        return np.dtype(cls._TO_NUMPY[name])

    @classmethod
    def from_numpy(cls, dt: np.dtype) -> str:
        dt = np.dtype(dt)
        for name, npt in cls._TO_NUMPY.items():
            if np.dtype(npt) == dt:
                return name
        raise KeyError(f"unsupported numpy dtype: {dt!r}")


@dataclass
class TensorInfo:
    """Static metadata about a named value (edge) in the graph.

    This is *not* the data — weights live in ``Graph.initializers``. This is
    the type and shape the value carries, used by shape inference, the memory
    planner (to size buffers), and codegen (to emit loop bounds).
    """

    name: str
    dtype: str = DType.FLOAT32
    shape: Shape = field(default_factory=tuple)

    @property
    def is_static(self) -> bool:
        """True when every dimension is a concrete non-negative int."""
        return all(isinstance(d, int) and d >= 0 for d in self.shape)

    @property
    def rank(self) -> int:
        return len(self.shape)

    def numel(self) -> Optional[int]:
        """Element count, or None if any dim is symbolic/unknown."""
        if not self.is_static:
            return None
        n = 1
        for d in self.shape:
            n *= int(d)
        return n

    def nbytes(self) -> Optional[int]:
        n = self.numel()
        if n is None:
            return None
        return n * DType.to_numpy(self.dtype).itemsize

    def with_shape(self, shape: Shape) -> "TensorInfo":
        return TensorInfo(self.name, self.dtype, tuple(shape))

    def __repr__(self) -> str:
        shp = "x".join(str(d) for d in self.shape) if self.shape else "scalar"
        return f"{self.name}:{self.dtype}[{shp}]"
