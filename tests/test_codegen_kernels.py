"""Generated CUDA kernels are numerically correct (compiled+run via CPU sim)."""

import numpy as np
import pytest

from conftest import requires_gpp
from mdlc.codegen.schedule import DEFAULT_GEMM, GemmSchedule, gemm_search_space
from mdlc.codegen.cuda.templates import (
    activation_expr,
    elementwise_kernel,
    gemm_kernel,
    im2col_kernel,
)
from mdlc.runtime import ops
from mdlc.testing.tolerances import BITWISE, FP32_REDUCTION, FP32_SAME_ORDER


def test_schedule_validity_and_pruning():
    space = list(gemm_search_space())
    assert space, "search space should be non-empty"
    for s in space:
        assert s.is_valid()
        assert s.smem_bytes() <= 48 * 1024
        assert 32 <= s.threads_per_block() <= 1024


def test_activation_expr_smoke():
    assert "fmaxf" in activation_expr("Relu", "v", {})
    assert activation_expr(None, "v", {}) == "v"


@requires_gpp
@pytest.mark.parametrize("sched", [DEFAULT_GEMM, GemmSchedule(32, 32, 8, 2, 2),
                                   GemmSchedule(128, 64, 16, 8, 4)])
@pytest.mark.parametrize("act,bias", [("Relu", True), (None, False), ("Sigmoid", True)])
def test_gemm_kernel_matches_numpy(sched, act, bias):
    from mdlc.codegen.cuda.cpu_sim import simulate_gemm

    rng = np.random.default_rng(0)
    M, N, K = 65, 48, 33   # non-tile-multiples exercise the bounds checks
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    b = rng.standard_normal(N).astype(np.float32) if bias else None
    name, src = gemm_kernel("g", sched, with_bias=bias, activation=act)
    got = simulate_gemm(src, name, sched, A, B, b)
    ref = A @ B
    if bias:
        ref = ref + b
    if act == "Relu":
        ref = np.maximum(ref, 0)
    elif act == "Sigmoid":
        ref = 1.0 / (1.0 + np.exp(-ref))
    np.testing.assert_allclose(got, ref,
                               rtol=FP32_REDUCTION.rtol, atol=FP32_REDUCTION.atol)


@requires_gpp
@pytest.mark.parametrize("trans_a,trans_b", [(False, True), (True, False), (True, True)])
def test_gemm_kernel_handles_transposed_operands(trans_a, trans_b):
    """ONNX transA/transB must be resolved *inside* the kernel.

    Regression guard: the transpose used to be applied host-side by the sim
    harness only, so the CPU path matched the reference while the device path
    — which has no such step — silently computed a different result. Every
    torch ``nn.Linear`` exports as ``Gemm(transB=1)``, so this hit the
    classifier head of all three target models. The kernel now bakes the
    layout into its load indices; this pins that behaviour.
    """
    from mdlc.codegen.cuda.cpu_sim import simulate_gemm

    rng = np.random.default_rng(3)
    M, N, K = 65, 48, 33            # non-tile-multiples exercise bounds checks
    A_log = rng.standard_normal((M, K)).astype(np.float32)
    B_log = rng.standard_normal((K, N)).astype(np.float32)
    # Store each operand transposed exactly as ONNX would.
    A = A_log.T.copy() if trans_a else A_log
    B = B_log.T.copy() if trans_b else B_log

    name, src = gemm_kernel("gt", DEFAULT_GEMM, with_bias=False, activation=None,
                            trans_a=trans_a, trans_b=trans_b)
    got = simulate_gemm(src, name, DEFAULT_GEMM, A, B, None, mnk=(M, N, K))
    np.testing.assert_allclose(got, A_log @ B_log,
                               rtol=FP32_REDUCTION.rtol, atol=FP32_REDUCTION.atol)


@pytest.mark.parametrize("attrs", [{"alpha": 2.0}, {"beta": 0.5}])
def test_gemm_rejects_unapplied_alpha_beta(attrs):
    """alpha/beta are not in the epilogue, so lowering must refuse rather than
    emit a kernel that quietly drops them (GAPS §9.1)."""
    import numpy as np_
    from onnx import TensorProto, helper, numpy_helper

    from mdlc.codegen.cuda.emit import emit_cuda_module
    from mdlc.frontend import import_onnx_model
    from mdlc.passes.shape_inference import infer_shapes_by_execution

    M, N, K = 4, 3, 5
    w = np_.random.default_rng(0).standard_normal((K, N)).astype(np_.float32)
    b = np_.random.default_rng(1).standard_normal(N).astype(np_.float32)
    node = helper.make_node("Gemm", ["x", "w", "b"], ["y"], **attrs)
    model = helper.make_model(
        helper.make_graph(
            [node], "t",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [M, K])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [M, N])],
            [numpy_helper.from_array(w, "w"), numpy_helper.from_array(b, "b")]),
        opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    g = import_onnx_model(model)
    feeds = {"x": np_.zeros((M, K), np_.float32)}
    infer_shapes_by_execution(g, feeds)
    with pytest.raises(NotImplementedError, match="alpha|beta"):
        emit_cuda_module(g)


def test_transposed_gemm_emits_distinct_index_math():
    """The specialization is real: transposed variants differ in source and do
    not collide in the emitter's kernel cache."""
    src_plain = gemm_kernel("g0", DEFAULT_GEMM)[1]
    src_tb = gemm_kernel("g0", DEFAULT_GEMM, trans_b=True)[1]
    src_ta = gemm_kernel("g0", DEFAULT_GEMM, trans_a=True)[1]
    assert src_plain != src_tb != src_ta
    assert "B[gc * K + gr]" in src_tb and "B[gr * N + gc]" in src_plain
    assert "A[gc * M + gr]" in src_ta and "A[gr * K + gc]" in src_plain


@requires_gpp
def test_im2col_kernel_matches_numpy():
    from mdlc.codegen.cuda.cpu_sim import simulate_im2col

    rng = np.random.default_rng(1)
    C, H, W = 3, 9, 9
    KH = KW = 3
    SH = SW = 1
    PH = PW = 1
    DH = DW = 1
    OH = OW = 9
    x = rng.standard_normal((C, H, W)).astype(np.float32)
    name, src = im2col_kernel("im2col")
    cols = simulate_im2col(src, name, x, KH=KH, KW=KW, OH=OH, OW=OW,
                           SH=SH, SW=SW, PH=PH, PW=PW, DH=DH, DW=DW)
    ref, _, _ = ops.im2col(x[None], KH, KW, SH, SW, DH, DW, ((PH, PH), (PW, PW)))
    np.testing.assert_allclose(cols, ref[0],
                               rtol=BITWISE.rtol, atol=BITWISE.atol)


@requires_gpp
def test_conv_via_im2col_and_gemm():
    from mdlc.codegen.cuda.cpu_sim import simulate_gemm, simulate_im2col

    rng = np.random.default_rng(2)
    C, H, W, OC = 3, 9, 9, 8
    KH = KW = 3
    x = rng.standard_normal((C, H, W)).astype(np.float32)
    w = rng.standard_normal((OC, C, KH, KW)).astype(np.float32)
    bias = rng.standard_normal(OC).astype(np.float32)
    OH = OW = 9
    nm, src = im2col_kernel("im2col")
    cols = simulate_im2col(src, nm, x, KH=KH, KW=KW, OH=OH, OW=OW,
                           SH=1, SW=1, PH=1, PW=1, DH=1, DW=1)
    gnm, gsrc = gemm_kernel("cg", DEFAULT_GEMM, with_bias=True,
                            bias_mode="row", activation="Relu")
    out = simulate_gemm(gsrc, gnm, DEFAULT_GEMM, w.reshape(OC, -1), cols, bias)
    out = out.reshape(OC, OH, OW)
    ref = np.maximum(ops.conv(x[None], w, bias, strides=[1, 1], pads=[1, 1, 1, 1],
                              dilations=[1, 1], group=1)[0], 0)
    np.testing.assert_allclose(out, ref,
                               rtol=FP32_REDUCTION.rtol, atol=FP32_REDUCTION.atol)


@requires_gpp
def test_elementwise_kernel_matches_numpy():
    from mdlc.codegen.cuda.cpu_sim import simulate_elementwise
    from mdlc.ir import Node

    # Build a small fused chain: y = relu(a + b) * c
    sub = [
        Node("Add", ["a", "b"], ["s"]),
        Node("Relu", ["s"], ["r"]),
        Node("Mul", ["r", "c"], ["y"]),
    ]
    fe = Node("FusedElementwise", ["a", "b", "c"], ["y"], attrs={"subgraph": sub})
    name, src, ext = elementwise_kernel("ew", fe)
    rng = np.random.default_rng(3)
    arrs = [rng.standard_normal(256).astype(np.float32) for _ in ext]
    outs = simulate_elementwise(src, name, ext, arrs, 1)
    env = dict(zip(ext, arrs))
    ref = np.maximum(env["a"] + env["b"], 0) * env["c"]
    np.testing.assert_allclose(outs[0], ref,
                               rtol=FP32_SAME_ORDER.rtol, atol=FP32_SAME_ORDER.atol)
