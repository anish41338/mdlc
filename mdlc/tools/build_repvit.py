"""Export a public timm RepViT (deploy/re-parameterized) to ONNX.

RepViT is the headline target model. We use the *public* timm implementation
with seeded random weights (no download, no external checkpoints) and export
the deploy form: ``timm.utils.model.reparameterize_model`` folds the RepVGG-style
multi-branch blocks and BN into plain convs, which is the form an inference
compiler should see.

Export decisions (fixed, documented here, relied on by tests):
  * batch dimension is FIXED (default 1) — the compiler specializes kernels on
    static shapes; a different batch is a different compile.
  * opset 17, matching build_resnet18.py. At opset 17 GELU has no dedicated op
    and exports as its decomposition (Erf or tanh-approx primitives) — we
    compile whatever the export actually emits.
"""

from __future__ import annotations

import argparse
import os

VARIANTS = ("repvit_m0_9", "repvit_m1_0", "repvit_m1_1", "repvit_m1_5", "repvit_m2_3")


def export(
    path: str = "examples/repvit_m0_9.onnx",
    *,
    variant: str = "repvit_m0_9",
    batch: int = 1,
    opset: int = 17,
    seed: int = 0,
    res: int = 224,
) -> str:
    import timm
    import torch
    from timm.utils.model import reparameterize_model

    from mdlc.tools.export_utils import lsuv_calibrate

    torch.manual_seed(seed)
    model = timm.create_model(variant, pretrained=False)
    model.eval()
    model = reparameterize_model(model)

    dummy = torch.randn(batch, 3, res, res)
    # Keep O(1) activations at every depth so parity checks have power (see
    # export_utils.lsuv_calibrate).
    lsuv_calibrate(model, dummy)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.onnx.export(
        model, dummy, path,
        input_names=["input"], output_names=["logits"],
        opset_version=opset,
        do_constant_folding=False,   # leave folding to *our* pass
        dynamic_axes=None,           # fixed batch: kernels specialize on shape
        dynamo=False,  # legacy exporter: avoids the dynamo->opset18->downgrade path breaking on some ops
    )
    print(f"wrote {path} ({variant}, batch={batch}, res={res}, opset={opset}, "
          f"seed={seed})")
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--variant", default="repvit_m0_9", choices=VARIANTS)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--res", type=int, default=224,
                    help="input H=W (reduced sizes for sim-path tests)")
    args = ap.parse_args()
    out = args.out or f"examples/{args.variant}.onnx"
    export(out, variant=args.variant, batch=args.batch, opset=args.opset,
           seed=args.seed, res=args.res)
