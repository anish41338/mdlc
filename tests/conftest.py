"""Shared test fixtures and capability detection."""

from __future__ import annotations

import shutil

import numpy as np
import pytest

from mdlc.frontend import import_onnx_model
from mdlc.passes.shape_inference import make_dummy_feeds
from mdlc.tools import build_models as bm


def has_gpp() -> bool:
    return shutil.which("g++") is not None


def has_ort() -> bool:
    try:
        import onnxruntime  # noqa: F401
        return True
    except Exception:
        return False


requires_gpp = pytest.mark.skipif(not has_gpp(), reason="needs g++ to compile CUDA-sim")
requires_ort = pytest.mark.skipif(not has_ort(), reason="needs onnxruntime")


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


def build_graph(name, **kw):
    """Build a model, import to IR, return (graph, feeds, onnx_model)."""
    model = bm.build(name, **kw)
    g = import_onnx_model(model)
    feeds = make_dummy_feeds(g, batch=1)
    return g, feeds, model


@pytest.fixture(params=["conv_bn_relu", "residual_block", "mlp"])
def model_case(request):
    name = request.param
    small = {"conv_bn_relu": dict(cin=3, cout=8, h=10, w=10),
             "residual_block": dict(c=4, h=8, w=8),
             "mlp": dict()}[name]
    return (name, small)
