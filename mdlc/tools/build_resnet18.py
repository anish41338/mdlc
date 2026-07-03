"""Export torchvision ResNet-18 to ONNX with random weights (no download).

Used for the end-to-end benchmark. Random (untrained) weights are fine: we
benchmark *speed* and verify *numerical agreement* with ONNX Runtime, neither
of which needs trained weights.
"""

from __future__ import annotations

import argparse
import os


def export(path: str = "examples/resnet18.onnx", batch: int = 1, opset: int = 17) -> str:
    import torch
    import torchvision

    model = torchvision.models.resnet18(weights=None)
    model.eval()
    dummy = torch.randn(batch, 3, 224, 224)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.onnx.export(
        model, dummy, path,
        input_names=["input"], output_names=["logits"],
        opset_version=opset,
        do_constant_folding=False,   # leave folding to *our* pass
        dynamic_axes=None,
    )
    print(f"wrote {path}")
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="examples/resnet18.onnx")
    ap.add_argument("--batch", type=int, default=1)
    args = ap.parse_args()
    export(args.out, args.batch)
