"""mdlc — a mini deep-learning compiler: ONNX -> graph IR -> fused CUDA."""

__version__ = "0.1.0"


def compile_onnx(*args, **kwargs):
    """Convenience re-export of :func:`mdlc.compiler.compile_onnx` (lazy)."""
    from mdlc.compiler import compile_onnx as _impl
    return _impl(*args, **kwargs)
