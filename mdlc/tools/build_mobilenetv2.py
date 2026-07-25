"""Export torchvision MobileNetV2 to ONNX with random weights (no download).

Same conventions as build_resnet18.py: seeded random weights (we verify
numerical agreement, not accuracy), fixed batch dim, opset 17, constant
folding left to our own pass.
"""

from __future__ import annotations

import argparse
import os


def export(path: str = "examples/mobilenetv2.onnx", batch: int = 1,
           opset: int = 17, seed: int = 0, res: int = 224) -> str:
    import torch
    import torchvision

    from mdlc.tools.export_utils import lsuv_calibrate

    torch.manual_seed(seed)
    model = torchvision.models.mobilenet_v2(weights=None)
    model.eval()
    dummy = torch.randn(batch, 3, res, res)
    # Untrained MobileNetV2 shrinks activations to ~1e-8 at the logits, below
    # the harness atol — parity checks would pass vacuously. Rescale to O(1).
    lsuv_calibrate(model, dummy)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.onnx.export(
        model, dummy, path,
        input_names=["input"], output_names=["logits"],
        opset_version=opset,
        do_constant_folding=False,
        dynamic_axes=None,
    )
    print(f"wrote {path} (batch={batch}, res={res}, opset={opset}, seed={seed})")
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="examples/mobilenetv2.onnx")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--res", type=int, default=224,
                    help="input H=W (reduced sizes for sim-path tests)")
    args = ap.parse_args()
    export(args.out, args.batch, args.opset, args.seed, args.res)
