"""Parse an ONNX model into our graph IR.

The ``onnx`` library is used purely as a protobuf reader and (optionally) for
shape inference. Once imported, nothing downstream touches ONNX again.
"""

from __future__ import annotations

from typing import Any, Union

import numpy as np
import onnx
from onnx import numpy_helper

from mdlc.ir import DType, Graph, Node, TensorInfo

# ONNX TensorProto dtype enum -> our dtype name. Only the dtypes we support.
_ONNX_DTYPE = {
    onnx.TensorProto.FLOAT: DType.FLOAT32,
    onnx.TensorProto.FLOAT16: DType.FLOAT16,
    onnx.TensorProto.INT64: DType.INT64,
    onnx.TensorProto.INT32: DType.INT32,
    onnx.TensorProto.INT8: DType.INT8,
    onnx.TensorProto.BOOL: DType.BOOL,
}


def _decode_attr(attr: onnx.AttributeProto) -> Any:
    """Decode one ONNX attribute into a plain Python value."""
    t = attr.type
    A = onnx.AttributeProto
    if t == A.INT:
        return int(attr.i)
    if t == A.FLOAT:
        return float(attr.f)
    if t == A.STRING:
        return attr.s.decode("utf-8")
    if t == A.INTS:
        return [int(v) for v in attr.ints]
    if t == A.FLOATS:
        return [float(v) for v in attr.floats]
    if t == A.STRINGS:
        return [s.decode("utf-8") for s in attr.strings]
    if t == A.TENSOR:
        return numpy_helper.to_array(attr.t)
    raise NotImplementedError(f"unsupported attribute type {t} on {attr.name!r}")


def _shape_of(vi: onnx.ValueInfoProto) -> tuple:
    """Extract a shape tuple from a ValueInfoProto, keeping symbolic dims."""
    tt = vi.type.tensor_type
    dims: list[Union[int, str, None]] = []
    for d in tt.shape.dim:
        if d.HasField("dim_value"):
            dims.append(int(d.dim_value))
        elif d.HasField("dim_param") and d.dim_param:
            dims.append(d.dim_param)
        else:
            dims.append(None)
    return tuple(dims)


def _dtype_of(vi: onnx.ValueInfoProto) -> str:
    elem = vi.type.tensor_type.elem_type
    return _ONNX_DTYPE.get(elem, DType.FLOAT32)


def import_onnx_model(
    model: onnx.ModelProto,
    *,
    run_shape_inference: bool = True,
) -> Graph:
    """Convert a loaded ``onnx.ModelProto`` into a ``Graph``."""
    if run_shape_inference:
        try:
            model = onnx.shape_inference.infer_shapes(model)
        except Exception:
            # Shape inference is best-effort; our own pass can fill gaps later.
            pass

    og = model.graph
    g = Graph(name=og.name or "graph")

    # Initializers (weights / constants).
    for init in og.initializer:
        g.set_initializer(init.name, numpy_helper.to_array(init))

    # Value info for inputs, outputs, and (post-inference) intermediates.
    for vi in list(og.input) + list(og.output) + list(og.value_info):
        if vi.name not in g.value_info or g.info(vi.name).rank == 0:
            g.set_info(TensorInfo(vi.name, _dtype_of(vi), _shape_of(vi)))

    g.inputs = [vi.name for vi in og.input]
    g.outputs = [vi.name for vi in og.output]

    # Nodes.
    for i, np_node in enumerate(og.node):
        attrs = {a.name: _decode_attr(a) for a in np_node.attribute}
        node = Node(
            op_type=np_node.op_type,
            inputs=list(np_node.input),
            outputs=list(np_node.output),
            attrs=attrs,
            name=np_node.name or f"{np_node.op_type}_{i}",
        )
        g.add_node(node)
        # Make sure every edge has at least a placeholder TensorInfo.
        for e in list(node.inputs) + list(node.outputs):
            if e and e not in g.value_info:
                g.set_info(TensorInfo(e))

    g.reorder_topologically()
    return g


def import_onnx(path: str, *, run_shape_inference: bool = True) -> Graph:
    """Load an ONNX file from disk and import it into our IR."""
    model = onnx.load(path)
    return import_onnx_model(model, run_shape_inference=run_shape_inference)
