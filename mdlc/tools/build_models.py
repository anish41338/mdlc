"""Build small ONNX models for tests and demos using ``onnx.helper``.

These are hand-built so the test suite needs no network and no torch. A
ResNet-18 exporter (via torchvision, random weights) is provided separately in
``build_resnet18.py`` for the end-to-end benchmark.
"""

from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def _init(name, array):
    return numpy_helper.from_array(np.asarray(array, dtype=np.float32), name)


def conv_bn_relu_model(
    *,
    n=1, cin=3, cout=8, h=16, w=16, k=3, seed=0,
) -> onnx.ModelProto:
    """A single Conv -> BatchNorm -> Relu block. The canonical fusion target."""
    rng = np.random.default_rng(seed)
    pad = k // 2

    weight = rng.standard_normal((cout, cin, k, k)).astype(np.float32) * 0.1
    bias = rng.standard_normal(cout).astype(np.float32) * 0.1
    gamma = (rng.standard_normal(cout).astype(np.float32) * 0.2 + 1.0)
    beta = rng.standard_normal(cout).astype(np.float32) * 0.1
    mean = rng.standard_normal(cout).astype(np.float32) * 0.1
    var = np.abs(rng.standard_normal(cout).astype(np.float32)) + 0.5

    nodes = [
        helper.make_node("Conv", ["x", "w", "b"], ["conv_out"],
                         kernel_shape=[k, k], pads=[pad, pad, pad, pad], strides=[1, 1]),
        helper.make_node("BatchNormalization",
                         ["conv_out", "gamma", "beta", "mean", "var"], ["bn_out"],
                         epsilon=1e-5),
        helper.make_node("Relu", ["bn_out"], ["y"]),
    ]
    inits = [
        _init("w", weight), _init("b", bias), _init("gamma", gamma),
        _init("beta", beta), _init("mean", mean), _init("var", var),
    ]
    g = helper.make_graph(
        nodes, "conv_bn_relu",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [n, cin, h, w])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [n, cout, h, w])],
        inits,
    )
    return _finish(g)


def residual_block_model(*, n=1, c=8, h=16, w=16, seed=1) -> onnx.ModelProto:
    """Two conv-bn-relu paths joined by a residual Add -> Relu.

    Exercises a *shared* intermediate (the block input feeds both the conv path
    and the skip Add), so the fusion legality check (multiple consumers) gets
    tested, plus elementwise Add+Relu chain fusion.
    """
    rng = np.random.default_rng(seed)

    def conv_weights(tag):
        return [
            _init(f"w_{tag}", rng.standard_normal((c, c, 3, 3)).astype(np.float32) * 0.1),
            _init(f"gamma_{tag}", rng.standard_normal(c).astype(np.float32) * 0.2 + 1.0),
            _init(f"beta_{tag}", rng.standard_normal(c).astype(np.float32) * 0.1),
            _init(f"mean_{tag}", rng.standard_normal(c).astype(np.float32) * 0.1),
            _init(f"var_{tag}", np.abs(rng.standard_normal(c).astype(np.float32)) + 0.5),
        ]

    nodes = [
        helper.make_node("Conv", ["x", "w_a", ""], ["a_conv"],
                         kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        helper.make_node("BatchNormalization",
                         ["a_conv", "gamma_a", "beta_a", "mean_a", "var_a"], ["a_bn"]),
        helper.make_node("Relu", ["a_bn"], ["a_relu"]),
        helper.make_node("Conv", ["a_relu", "w_b", ""], ["b_conv"],
                         kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        helper.make_node("BatchNormalization",
                         ["b_conv", "gamma_b", "beta_b", "mean_b", "var_b"], ["b_bn"]),
        helper.make_node("Add", ["b_bn", "x"], ["sum"]),
        helper.make_node("Relu", ["sum"], ["y"]),
    ]
    inits = conv_weights("a") + conv_weights("b")
    g = helper.make_graph(
        nodes, "residual_block",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [n, c, h, w])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [n, c, h, w])],
        inits,
    )
    return _finish(g)


def mlp_model(*, n=4, din=16, dh=32, dout=8, seed=2) -> onnx.ModelProto:
    """Gemm -> Relu -> Gemm. Exercises GEMM+bias+activation fusion."""
    rng = np.random.default_rng(seed)
    nodes = [
        helper.make_node("Gemm", ["x", "w1", "b1"], ["h1"], transB=1),
        helper.make_node("Relu", ["h1"], ["h1r"]),
        helper.make_node("Gemm", ["h1r", "w2", "b2"], ["y"], transB=1),
    ]
    inits = [
        _init("w1", rng.standard_normal((dh, din)).astype(np.float32) * 0.1),
        _init("b1", rng.standard_normal(dh).astype(np.float32) * 0.1),
        _init("w2", rng.standard_normal((dout, dh)).astype(np.float32) * 0.1),
        _init("b2", rng.standard_normal(dout).astype(np.float32) * 0.1),
    ]
    g = helper.make_graph(
        nodes, "mlp",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [n, din])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [n, dout])],
        inits,
    )
    return _finish(g)


def _finish(graph: onnx.GraphProto) -> onnx.ModelProto:
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9  # compatible with onnxruntime 1.16+
    onnx.checker.check_model(model)
    return model


_REGISTRY = {
    "conv_bn_relu": conv_bn_relu_model,
    "residual_block": residual_block_model,
    "mlp": mlp_model,
}


def build(name: str, **kw) -> onnx.ModelProto:
    return _REGISTRY[name](**kw)
