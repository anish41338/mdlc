"""Device validation (Phase 2 gate): same graphs/kernels the CPU sim already
proved correct, replayed through the real CUDA driver via NVRTC. Everything
here is `pytest.mark.gpu` and skipped unless `cuda_available()` — this file
only executes on a CUDA host (e.g. the Kaggle T4/P100 runbook), never in the
CPU-only CI/dev loop. Structurally mirrors test_end_to_end_sim.py and
test_new_lowerings.py so any drift between the sim and device launch plans
shows up as a same-shaped failure.
"""

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from conftest import build_graph, requires_gpp
from mdlc.codegen.cuda.emit import emit_cuda_module
from mdlc.codegen.cuda.nvrtc_runtime import cuda_available
from mdlc.frontend import import_onnx_model
from mdlc.passes import default_pipeline
from mdlc.passes.shape_inference import infer_shapes_by_execution
from mdlc.runtime import run_reference
from mdlc.runtime.memory_planner import plan_memory
from mdlc.testing.tolerances import FP32_NETWORK

pytestmark = pytest.mark.gpu

requires_cuda = pytest.mark.skipif(
    not cuda_available(), reason="needs a CUDA device (NVRTC + driver)")


def _model(nodes, inputs, outputs, inits):
    graph = helper.make_graph(nodes, "t", inputs, outputs, inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    return model


def _compile_and_run(model, feeds, *, pooled=False):
    from mdlc.codegen.cuda.gpu_executor import run_on_gpu

    g = import_onnx_model(model)
    golden = run_reference(g.clone(), feeds)
    g = default_pipeline().run(g)
    infer_shapes_by_execution(g, feeds)
    module = emit_cuda_module(g)
    assert module.fallbacks() == [], f"host fallbacks: {module.fallbacks()}"
    mem_plan = plan_memory(g) if pooled else None
    got = run_on_gpu(module, feeds, mem_plan=mem_plan)["outputs"]
    for o in g.outputs:
        np.testing.assert_allclose(got[o], golden[o],
                                   rtol=FP32_NETWORK.rtol, atol=FP32_NETWORK.atol)
    return module


# -- basic hand-built models, on-device vs reference -------------------------

@requires_cuda
@requires_gpp
def test_generated_module_matches_reference_on_device(model_case):
    name, kw = model_case
    _, feeds, _ = build_graph(name, **kw)
    from mdlc.tools import build_models as bm
    model = bm.build(name, **kw)
    _compile_and_run(model, feeds)


@requires_cuda
@requires_gpp
def test_generated_module_matches_reference_pooled_on_device(model_case):
    """Same as above, but through the pooled-memory executor path (the
    256-byte-aligned single-allocation planner) — the GPU_TODO risk item."""
    name, kw = model_case
    from mdlc.tools import build_models as bm
    model = bm.build(name, **kw)
    _, feeds, _ = build_graph(name, **kw)
    _compile_and_run(model, feeds, pooled=True)


# -- batch-N conv: per-image im2col+GEMM offset math on real device pointers -

@requires_cuda
@requires_gpp
@pytest.mark.parametrize("n", [1, 2, 8])
@pytest.mark.parametrize("model_name,kw", [
    ("conv_bn_relu", dict(cin=3, cout=8, h=10, w=10)),
    ("residual_block", dict(c=4, h=8, w=8)),
])
def test_conv_models_batch_n_on_device(model_name, kw, n):
    from mdlc.tools import build_models as bm

    model = bm.build(model_name, n=n, **kw)
    g = import_onnx_model(model)
    feeds = {g.runtime_inputs[0]:
             np.random.default_rng(n).standard_normal(
                 g.info(g.runtime_inputs[0]).shape).astype(np.float32)}
    _compile_and_run(model, feeds)


# -- grouped conv: per-(image,group) weight-row device-pointer slices -------

@requires_cuda
@requires_gpp
@pytest.mark.parametrize("n,g_", [(1, 2), (2, 4)])
def test_grouped_conv_per_group_slices_on_device(n, g_):
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
    _compile_and_run(model, feeds)


# -- direct depthwise kernel on device ---------------------------------------

@requires_cuda
@requires_gpp
@pytest.mark.parametrize("n,mult", [(1, 1), (2, 1), (1, 2)])
def test_direct_depthwise_on_device(n, mult):
    rng = np.random.default_rng(11)
    C, H, W, K = 6, 12, 12, 3
    OC = C * mult
    w = (rng.standard_normal((OC, 1, K, K)) * 0.3).astype(np.float32)
    b = rng.standard_normal(OC).astype(np.float32)
    model = _model(
        [helper.make_node("Conv", ["x", "w", "b"], ["y"], kernel_shape=[K, K],
                          pads=[1, 1, 1, 1], group=C)],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [n, C, H, W])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [n, OC, H, W])],
        [numpy_helper.from_array(w, "w"), numpy_helper.from_array(b, "b")],
    )
    feeds = {"x": rng.standard_normal((n, C, H, W)).astype(np.float32)}
    module = _compile_and_run(model, feeds)
    assert "depthwise" in {l.kind for l in module.plan}


# -- reduce / pool / view kernels on device ----------------------------------

@requires_cuda
@requires_gpp
@pytest.mark.parametrize("n,c,h,w", [(1, 4, 7, 9), (2, 8, 16, 16)])
def test_global_average_pool_on_device(n, c, h, w):
    rng = np.random.default_rng(8)
    model = _model(
        [helper.make_node("GlobalAveragePool", ["x"], ["y"])],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [n, c, h, w])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [n, c, 1, 1])],
        [],
    )
    feeds = {"x": rng.standard_normal((n, c, h, w)).astype(np.float32)}
    _compile_and_run(model, feeds)


@requires_cuda
@requires_gpp
@pytest.mark.parametrize("stride,pad", [(2, 1), (1, 0)])
def test_maxpool_on_device(stride, pad):
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
    _compile_and_run(model, feeds)


@requires_cuda
@requires_gpp
def test_se_broadcast_mul_chain_on_device():
    """GlobalAveragePool -> Sigmoid -> broadcast Mul: exercises reduce +
    elementwise + the view-aliasing path together, on real device pointers."""
    rng = np.random.default_rng(6)
    n, C, H, W = 2, 6, 8, 8
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
    _compile_and_run(model, feeds)


# -- real target models, reduced resolution, full device path ---------------

RES = 64


@requires_cuda
@requires_gpp
@pytest.mark.parametrize("build_mod,name", [
    ("mdlc.tools.build_mobilenetv2", "mobilenetv2"),
    ("mdlc.tools.build_repvit", "repvit"),
])
def test_target_model_on_device(build_mod, name, tmp_path):
    torch = pytest.importorskip("torch")  # noqa: F841
    pytest.importorskip("timm")
    import importlib

    from mdlc.compiler import compile_onnx
    from mdlc.codegen.cuda.gpu_executor import run_on_gpu

    mod = importlib.import_module(build_mod)
    path = str(tmp_path / f"{name}.onnx")
    mod.export(path, res=RES)

    feeds = {"input": np.random.default_rng(0).standard_normal(
        (1, 3, RES, RES)).astype(np.float32)}
    compiled = compile_onnx(path, feeds=feeds)
    assert compiled.module.fallbacks() == [], compiled.module.fallbacks()

    got = run_on_gpu(compiled.module, feeds)["outputs"]
    for oname in compiled.graph.outputs:
        want = compiled.golden[oname]
        assert float(np.abs(want).max()) > 100 * FP32_NETWORK.atol, \
            "golden magnitude too small — vacuous comparison"
        np.testing.assert_allclose(got[oname], want,
                                   rtol=FP32_NETWORK.rtol, atol=FP32_NETWORK.atol)
