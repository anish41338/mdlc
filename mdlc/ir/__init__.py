"""The compiler's own intermediate representation.

Deliberately small and explicit: a ``Graph`` owns ``Node`` objects and
``TensorInfo`` metadata, with constant weights held as NumPy arrays in
``initializers``. Everything downstream (passes, codegen, runtime) operates on
this IR, never on ONNX protobuf directly.
"""

from mdlc.ir.tensor import DType, TensorInfo
from mdlc.ir.node import Node, VIEW_OPS
from mdlc.ir.graph import Graph
from mdlc.ir.verify import GraphVerifyError, verify_graph

__all__ = ["DType", "TensorInfo", "Node", "Graph", "GraphVerifyError",
           "verify_graph", "VIEW_OPS"]
