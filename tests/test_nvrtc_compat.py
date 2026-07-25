"""Emitted CUDA source must be NVRTC-compilable — checked without a GPU.

NVRTC compiles a source *string* in-process with no filesystem include path, so
anything the generated kernels `#include` is a hard `catastrophic error: could
not open source file` on device — even though the CPU simulator happily
compiles the same source, because g++ *does* have an include path and the sim
shim textually strips the offending line.

That asymmetry cost a full GPU round-trip once (`#include <math_constants.h>`
in four templates). These tests close it: they assert the self-containment
property on the CPU, over the real emitted module source for every lowering
kind, so the next such regression fails in the normal dev loop instead of on
Kaggle.
"""

import re

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from conftest import build_graph
from mdlc.codegen.cuda.emit import emit_cuda_module
from mdlc.frontend import import_onnx_model
from mdlc.passes import default_pipeline
from mdlc.passes.shape_inference import infer_shapes_by_execution

# Identifiers NVRTC supplies built-in (no header needed). Anything else that
# looks like a CUDA-toolkit constant is suspect.
_NVRTC_BUILTIN_OK = {"__int_as_float", "__float_as_int", "__dp4a",
                     "__float2int_rn", "__expf", "__syncthreads",
                     "__restrict__", "__shared__", "__global__", "__device__"}


def _all_emitted_sources():
    """Emit modules covering every lowering kind; yield (label, source)."""
    mods = []

    # hand-built graphs from the shared fixtures (conv/gemm/elementwise paths)
    for name, kw in [("conv_bn_relu", dict(cin=3, cout=8, h=10, w=10)),
                     ("residual_block", dict(c=4, h=8, w=8)),
                     ("mlp", {})]:
        g, feeds, _ = build_graph(name, **kw)
        g = default_pipeline().run(g)
        infer_shapes_by_execution(g, feeds)
        mods.append((name, emit_cuda_module(g)))

    # depthwise, grouped conv, reduce, maxpool, broadcast elementwise (Clip
    # with an open bound is the CUDART_INF_F user, so include one)
    rng = np.random.default_rng(0)

    def _one(label, nodes, in_shape, out_shape, inits):
        model = helper.make_model(
            helper.make_graph(
                nodes, "t",
                [helper.make_tensor_value_info("x", TensorProto.FLOAT, in_shape)],
                [helper.make_tensor_value_info("y", TensorProto.FLOAT, out_shape)],
                inits),
            opset_imports=[helper.make_opsetid("", 17)])
        model.ir_version = 9
        g = import_onnx_model(model)
        feeds = {"x": rng.standard_normal(in_shape).astype(np.float32)}
        g = default_pipeline().run(g)
        infer_shapes_by_execution(g, feeds)
        mods.append((label, emit_cuda_module(g)))

    dw_w = (rng.standard_normal((6, 1, 3, 3)) * 0.3).astype(np.float32)
    _one("depthwise+clip",
         [helper.make_node("Conv", ["x", "w"], ["c"], kernel_shape=[3, 3],
                           pads=[1, 1, 1, 1], group=6),
          helper.make_node("Clip", ["c"], ["y"])],
         [1, 6, 12, 12], [1, 6, 12, 12], [numpy_helper.from_array(dw_w, "w")])

    gw = (rng.standard_normal((12, 4, 3, 3)) * 0.3).astype(np.float32)
    _one("grouped_conv",
         [helper.make_node("Conv", ["x", "w"], ["y"], kernel_shape=[3, 3],
                           pads=[1, 1, 1, 1], group=2)],
         [1, 8, 10, 10], [1, 12, 10, 10], [numpy_helper.from_array(gw, "w")])

    _one("maxpool",
         [helper.make_node("MaxPool", ["x"], ["y"], kernel_shape=[3, 3],
                           strides=[2, 2], pads=[1, 1, 1, 1])],
         [1, 3, 11, 11], [1, 3, 6, 6], [])

    _one("reduce+broadcast_mul",
         [helper.make_node("GlobalAveragePool", ["x"], ["p"]),
          helper.make_node("Sigmoid", ["p"], ["s"]),
          helper.make_node("Mul", ["s", "x"], ["y"])],
         [2, 6, 8, 8], [2, 6, 8, 8], [])

    return [(label, m) for label, m in mods]


@pytest.fixture(scope="module")
def emitted():
    return _all_emitted_sources()


def test_emitted_sources_have_no_includes(emitted):
    """The self-containment invariant: NVRTC has no include search path."""
    for label, module in emitted:
        src = module.sources()
        offenders = re.findall(r"^\s*#\s*include.*$", src, re.MULTILINE)
        assert not offenders, (
            f"{label}: emitted CUDA source must be self-contained for NVRTC "
            f"(no include path exists there); found {offenders}")


def test_cudart_inf_is_defined_where_used(emitted):
    """Any source referencing CUDART_INF_F must also define it itself."""
    for label, module in emitted:
        for kname, src in module.kernels.items():
            if "CUDART_INF_F" not in src:
                continue
            assert "#define CUDART_INF_F" in src, (
                f"{label}/{kname} uses CUDART_INF_F without defining it — "
                "NVRTC cannot pull it from math_constants.h")


def test_no_unknown_double_underscore_intrinsics(emitted):
    """Catch toolkit-only helpers sneaking in: every `__foo` the source calls
    must be one NVRTC provides built-in."""
    for label, module in emitted:
        src = module.sources()
        used = set(re.findall(r"\b(__\w+)\b", src))
        unknown = used - _NVRTC_BUILTIN_OK
        assert not unknown, (
            f"{label}: unrecognized __-prefixed identifiers {sorted(unknown)}; "
            "confirm NVRTC provides them without a header, then allow-list "
            "them in _NVRTC_BUILTIN_OK")


def test_launch_configs_are_plain_ints_within_device_limits(emitted):
    """cuLaunchKernel takes unsigned ints through ctypes: numpy integers raise
    at the FFI boundary, and >1024 threads/block is a launch failure. Both are
    device-only symptoms, so assert the shape of the plan here."""
    for label, module in emitted:
        for L in module.plan:
            if L.kind == "view":
                continue
            for dim in tuple(L.grid) + tuple(L.block):
                assert type(dim) is int, (
                    f"{label}/{L.kernel_name}: launch dim {dim!r} is "
                    f"{type(dim).__name__}, not a plain int — ctypes cannot "
                    "convert numpy integers for cuLaunchKernel")
                assert dim >= 1, f"{label}/{L.kernel_name}: non-positive dim {dim}"
            threads = 1
            for b in L.block:
                threads *= b
            assert threads <= 1024, (
                f"{label}/{L.kernel_name}: {threads} threads/block exceeds the "
                "CUDA maximum of 1024")
            assert L.grid[0] <= 2**31 - 1
            for gi in (1, 2):
                if len(L.grid) > gi:
                    assert L.grid[gi] <= 65535, (
                        f"{label}/{L.kernel_name}: grid dim {gi} = {L.grid[gi]} "
                        "exceeds the 65535 limit for y/z")
