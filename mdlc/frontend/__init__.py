"""ONNX -> IR frontend. We use the ``onnx`` package for parsing only."""

from mdlc.frontend.onnx_importer import import_onnx, import_onnx_model

__all__ = ["import_onnx", "import_onnx_model"]
