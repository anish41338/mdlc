"""Direct depthwise conv: the emitted CUDA source is compiled and executed on
the CPU sim and diffed against the NumPy reference over the full config
cross-product, randomized configs, and the fused emit path."""

import numpy as np
import pytest

from conftest import requires_gpp
from mdlc.codegen.cuda.cpu_sim import simulate_kernel
from mdlc.codegen.cuda.templates import depthwise_conv_kernel
from mdlc.codegen.schedule import DepthwiseSchedule, depthwise_search_space
from mdlc.runtime import ops
from mdlc.testing.tolerances import FP32_SAME_ORDER


def _run_case(*, N, C, H, W, K, S, pad_same, mult=1, act=None, bias=True,
              sched=DepthwiseSchedule(8, 8), rng=None):
    """Emit the shape-specialized kernel, run it on the sim, diff vs reference."""
    rng = rng or np.random.default_rng(0)
    P = (K // 2) if pad_same else 0
    OC = C * mult
    OH = (H + 2 * P - K) // S + 1
    OW = (W + 2 * P - K) // S + 1
    assert OH > 0 and OW > 0, "config produces empty output"

    x = rng.standard_normal((N, C, H, W)).astype(np.float32)
    w = rng.standard_normal((OC, 1, K, K)).astype(np.float32) * 0.5
    b = rng.standard_normal(OC).astype(np.float32) if bias else None

    name = "dw_case"
    _, src = depthwise_conv_kernel(
        name, C=C, mult=mult, H=H, W=W, OH=OH, OW=OW, KH=K, KW=K,
        SH=S, SW=S, PH=P, PW=P, tile_h=sched.tile_h, tile_w=sched.tile_w,
        with_bias=bias, activation=act, direct_load=sched.direct_load)
    grid = ((OW + sched.tile_w - 1) // sched.tile_w,
            (OH + sched.tile_h - 1) // sched.tile_h, N * OC)
    inputs = [x, w] + ([b] if bias else [])
    out = simulate_kernel(src, name, inputs=inputs, output_sizes=[N * OC * OH * OW],
                          grid=grid, block=sched.threads_per_block())[0]
    out = out.reshape(N, OC, OH, OW)

    ref = ops.conv(x, w, b, strides=[S, S], pads=[P, P, P, P],
                   dilations=[1, 1], group=C)
    if act == "Relu":
        ref = np.maximum(ref, 0)
    np.testing.assert_allclose(out, ref, rtol=FP32_SAME_ORDER.rtol,
                               atol=FP32_SAME_ORDER.atol)


# -- the brief's cross-product: stride x pad x K x C x H,W parity x N --------
# Spatial sizes kept small so the sim's one-OS-thread-per-CUDA-thread model
# stays fast; the kernel is shape-specialized so small shapes exercise the
# same generated code paths (tile interior, halo, edge tiles).

@requires_gpp
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("pad_same", [True, False])
@pytest.mark.parametrize("k", [3, 5])
@pytest.mark.parametrize("c", [8, 32, 96])
@pytest.mark.parametrize("hw", [(12, 12), (9, 11)])   # even and odd H,W
@pytest.mark.parametrize("n", [1, 8])
def test_depthwise_cross_product(stride, pad_same, k, c, hw, n):
    h, w = hw
    if not pad_same and k == 5 and stride == 2 and min(h, w) < 7:
        pytest.skip("degenerate output")
    _run_case(N=n, C=c, H=h, W=w, K=k, S=stride, pad_same=pad_same,
              act="Relu", rng=np.random.default_rng(hash((stride, k, c, n)) % 2**32))


# -- 25 randomized configs (seeded) over the whole knob space ----------------

@requires_gpp
@pytest.mark.parametrize("case", range(25))
def test_depthwise_randomized(case):
    rng = np.random.default_rng(1000 + case)
    scheds = [s for s in depthwise_search_space(max_threads=128)]
    sched = scheds[rng.integers(len(scheds))]
    k = int(rng.choice([3, 5]))
    _run_case(
        N=int(rng.choice([1, 2, 4])),
        C=int(rng.choice([3, 5, 8, 16])),
        H=int(rng.integers(k + 2, 20)),
        W=int(rng.integers(k + 2, 20)),
        K=k,
        S=int(rng.choice([1, 2])),
        pad_same=bool(rng.integers(2)),
        mult=int(rng.choice([1, 2])),
        act=[None, "Relu"][int(rng.integers(2))],
        bias=bool(rng.integers(2)),
        sched=sched,
        rng=rng,
    )


# -- emit path: Conv(group=C)+BN+Relu fuses and lowers to one depthwise launch

@requires_gpp
@pytest.mark.parametrize("n,mult", [(1, 1), (2, 1), (8, 1), (1, 2)])
def test_depthwise_fused_emit_path(n, mult):
    from onnx import TensorProto, helper, numpy_helper

    from mdlc.codegen.cuda.emit import emit_cuda_module
    from mdlc.codegen.cuda.sim_executor import simulate_module
    from mdlc.frontend import import_onnx_model
    from mdlc.passes import default_pipeline
    from mdlc.passes.shape_inference import infer_shapes_by_execution
    from mdlc.runtime import run_reference
    from mdlc.testing.tolerances import FP32_NETWORK

    rng = np.random.default_rng(42)
    C, H, W, K = 6, 12, 12, 3
    OC = C * mult
    w = rng.standard_normal((OC, 1, K, K)).astype(np.float32) * 0.3
    gamma = (rng.standard_normal(OC) * 0.2 + 1.0).astype(np.float32)
    beta = (rng.standard_normal(OC) * 0.1).astype(np.float32)
    mean = (rng.standard_normal(OC) * 0.1).astype(np.float32)
    var = (np.abs(rng.standard_normal(OC)) + 0.5).astype(np.float32)

    nodes = [
        helper.make_node("Conv", ["x", "w", ""], ["c"], kernel_shape=[K, K],
                         pads=[1, 1, 1, 1], group=C),
        helper.make_node("BatchNormalization",
                         ["c", "gamma", "beta", "mean", "var"], ["bn"]),
        helper.make_node("Relu", ["bn"], ["y"]),
    ]
    graph = helper.make_graph(
        nodes, "dw_fused",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [n, C, H, W])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [n, OC, H, W])],
        [numpy_helper.from_array(w, "w"), numpy_helper.from_array(gamma, "gamma"),
         numpy_helper.from_array(beta, "beta"), numpy_helper.from_array(mean, "mean"),
         numpy_helper.from_array(var, "var")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    g = import_onnx_model(model)
    feeds = {"x": rng.standard_normal((n, C, H, W)).astype(np.float32)}
    golden = run_reference(g.clone(), feeds)

    g = default_pipeline().run(g)
    # BN folded + Relu fused: a single FusedConvAct with a bias, still grouped
    assert [nd.op_type for nd in g.nodes] == ["FusedConvAct"]
    infer_shapes_by_execution(g, feeds)
    module = emit_cuda_module(g)
    kinds = [l.kind for l in module.plan]
    assert kinds == ["depthwise"], f"expected one depthwise launch, got {kinds}"
    assert module.fallbacks() == []

    got = simulate_module(module, feeds)
    np.testing.assert_allclose(got["y"], golden["y"],
                               rtol=FP32_NETWORK.rtol, atol=FP32_NETWORK.atol)
