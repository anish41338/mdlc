"""Sim parity for the Phase-1 lowerings beyond depthwise: grouped conv,
broadcast elementwise (SE gate, GELU chain), spatial-mean reduction, maxpool,
and views. Every test compiles and runs the exact emitted CUDA source."""

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from conftest import requires_gpp
from mdlc.codegen.cuda.emit import emit_cuda_module
from mdlc.codegen.cuda.sim_executor import simulate_module
from mdlc.frontend import import_onnx_model
from mdlc.passes import default_pipeline
from mdlc.passes.shape_inference import infer_shapes_by_execution
from mdlc.runtime import run_reference
from mdlc.testing.tolerances import FP32_NETWORK, FP32_SAME_ORDER


def _model(nodes, inputs, outputs, inits):
    graph = helper.make_graph(nodes, "t", inputs, outputs, inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    return model


def _compile_and_sim(model, feeds, *, expect_kinds=None, forbid_host=True):
    g = import_onnx_model(model)
    golden = run_reference(g.clone(), feeds)
    g = default_pipeline().run(g)
    infer_shapes_by_execution(g, feeds)
    module = emit_cuda_module(g)
    if forbid_host:
        assert module.fallbacks() == [], f"host fallbacks: {module.fallbacks()}"
    if expect_kinds is not None:
        assert sorted({l.kind for l in module.plan}) == sorted(expect_kinds), \
            [l.kind for l in module.plan]
    got = simulate_module(module, feeds)
    for o in g.outputs:
        np.testing.assert_allclose(got[o], golden[o],
                                   rtol=FP32_NETWORK.rtol, atol=FP32_NETWORK.atol)
    return module


# -- general grouped conv (1 < g < C_in) -------------------------------------

@requires_gpp
@pytest.mark.parametrize("n,g_", [(1, 2), (2, 4)])
def test_grouped_conv_per_group_slices(n, g_):
    rng = np.random.default_rng(5)
    C, OC, H, W, K = 8, 12, 10, 10, 3
    w = (rng.standard_normal((OC, C // g_, K, K)) * 0.3).astype(np.float32)
    b = rng.standard_normal(OC).astype(np.float32)
    model = _model(
        [helper.make_node("Conv", ["x", "w", "b"], ["y"], kernel_shape=[K, K],
                          pads=[1, 1, 1, 1], group=g_)],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [n, C, H, W])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [n, OC, H, W])],
        [numpy_helper.from_array(w, "w"), numpy_helper.from_array(b, "b")],
    )
    feeds = {"x": rng.standard_normal((n, C, H, W)).astype(np.float32)}
    module = _compile_and_sim(model, feeds, expect_kinds=["im2col", "gemm"])
    # per-(image, group) launch pairs
    assert sum(1 for l in module.plan if l.kind == "gemm") == n * g_


# -- SE-style broadcast Mul: (N,C,1,1) gate x (N,C,H,W) trunk ----------------

@requires_gpp
@pytest.mark.parametrize("n", [1, 2])
def test_se_broadcast_mul_chain(n):
    """GlobalAveragePool -> Sigmoid -> broadcast Mul, the squeeze-excite tail.
    The Sigmoid+Mul chain must fuse and lower to one broadcast elementwise
    kernel — no host launch."""
    rng = np.random.default_rng(6)
    C, H, W = 6, 8, 8
    model = _model(
        [
            helper.make_node("GlobalAveragePool", ["x"], ["p"]),
            helper.make_node("Sigmoid", ["p"], ["s"]),
            helper.make_node("Mul", ["s", "x"], ["y"]),
        ],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [n, C, H, W])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [n, C, H, W])],
        [],
    )
    feeds = {"x": rng.standard_normal((n, C, H, W)).astype(np.float32)}
    module = _compile_and_sim(model, feeds)
    kinds = [l.kind for l in module.plan]
    assert "elementwise" in kinds and "reduce" in kinds and "host" not in kinds


# -- GELU decomposition chain (Div-Erf-Add-Mul-Mul) fuses to one kernel ------

@requires_gpp
def test_gelu_erf_chain_fuses_and_matches():
    rng = np.random.default_rng(7)
    shape = [1, 4, 6, 6]
    half = np.float32(0.5)
    one = np.float32(1.0)
    rsqrt2 = np.float32(1.4142135)
    model = _model(
        [
            helper.make_node("Div", ["x", "c_sqrt2"], ["d"]),
            helper.make_node("Erf", ["d"], ["e"]),
            helper.make_node("Add", ["e", "c_one"], ["a"]),
            helper.make_node("Mul", ["x", "a"], ["m"]),
            helper.make_node("Mul", ["m", "c_half"], ["y"]),
        ],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, shape)],
        [numpy_helper.from_array(rsqrt2, "c_sqrt2"),
         numpy_helper.from_array(one, "c_one"),
         numpy_helper.from_array(half, "c_half")],
    )
    x = rng.standard_normal(shape).astype(np.float32) * 2
    module = _compile_and_sim(model, {"x": x},
                              expect_kinds=["elementwise"])
    assert len(module.plan) == 1, "GELU must be a single fused kernel"
    # and it really is GELU
    import math
    got = simulate_module(module, {"x": x})["y"]
    want = 0.5 * x * (1.0 + np.vectorize(math.erf)(x / np.sqrt(2.0)))
    np.testing.assert_allclose(got, want, rtol=FP32_SAME_ORDER.rtol, atol=1e-4)


# -- reduction + maxpool kernels ---------------------------------------------

@requires_gpp
@pytest.mark.parametrize("n,c,h,w", [(1, 4, 7, 9), (2, 8, 16, 16)])
def test_global_average_pool_kernel(n, c, h, w):
    rng = np.random.default_rng(8)
    model = _model(
        [helper.make_node("GlobalAveragePool", ["x"], ["y"])],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [n, c, h, w])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [n, c, 1, 1])],
        [],
    )
    feeds = {"x": rng.standard_normal((n, c, h, w)).astype(np.float32)}
    _compile_and_sim(model, feeds, expect_kinds=["reduce"])


@requires_gpp
def test_reduce_mean_hw_keepdims_false():
    """RepViT's classifier-head pooling: ReduceMean axes=[2,3], keepdims=0."""
    rng = np.random.default_rng(9)
    n, c, h, w = 2, 5, 6, 6
    model = _model(
        [helper.make_node("ReduceMean", ["x"], ["y"], axes=[2, 3], keepdims=0)],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [n, c, h, w])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [n, c])],
        [],
    )
    feeds = {"x": rng.standard_normal((n, c, h, w)).astype(np.float32)}
    _compile_and_sim(model, feeds, expect_kinds=["reduce"])


@requires_gpp
@pytest.mark.parametrize("stride,pad", [(2, 1), (1, 0)])
def test_maxpool_kernel(stride, pad):
    rng = np.random.default_rng(10)
    n, c, h, w, k = 2, 3, 11, 11, 3
    oh = (h + 2 * pad - k) // stride + 1
    model = _model(
        [helper.make_node("MaxPool", ["x"], ["y"], kernel_shape=[k, k],
                          strides=[stride, stride], pads=[pad, pad, pad, pad])],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [n, c, h, w])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [n, c, oh, oh])],
        [],
    )
    feeds = {"x": rng.standard_normal((n, c, h, w)).astype(np.float32)}
    _compile_and_sim(model, feeds, expect_kinds=["pool"])


# -- broadcast fuzz: random elementwise chains with random broadcast shapes --

@requires_gpp
@pytest.mark.parametrize("case", range(20))
def test_broadcast_elementwise_fuzz(case):
    """Seeded random binary chains where each operand randomly broadcasts:
    full shape, (1,C,1,1), (N,C,1,1), scalar. Diffed against NumPy through
    the emitted kernel."""
    rng = np.random.default_rng(2000 + case)
    n, c, h, w = int(rng.choice([1, 2])), int(rng.integers(2, 6)), \
        int(rng.integers(2, 7)), int(rng.integers(2, 7))
    out_shape = [n, c, h, w]
    shapes = [out_shape, [1, c, 1, 1], [n, c, 1, 1], [1], [1, 1, h, w]]
    ops_pool = ["Add", "Mul", "Sub"]

    nodes = []
    inits = []
    inputs_vi = [helper.make_tensor_value_info("x", TensorProto.FLOAT, out_shape)]
    feeds = {"x": rng.standard_normal(out_shape).astype(np.float32)}
    cur = "x"
    depth = int(rng.integers(2, 5))
    for d in range(depth):
        op = ops_pool[int(rng.integers(len(ops_pool)))]
        s = shapes[int(rng.integers(len(shapes)))]
        oname = f"t{d}"
        bname = f"b{d}"
        arr = rng.standard_normal(s).astype(np.float32)
        inits.append(numpy_helper.from_array(arr, bname))
        nodes.append(helper.make_node(op, [cur, bname], [oname]))
        cur = oname
    nodes.append(helper.make_node("Relu", [cur], ["y"]))
    model = _model(nodes, inputs_vi,
                   [helper.make_tensor_value_info("y", TensorProto.FLOAT, out_shape)],
                   inits)
    _compile_and_sim(model, feeds, forbid_host=True)
