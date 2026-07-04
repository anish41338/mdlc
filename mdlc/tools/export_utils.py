"""Shared helpers for the torch->ONNX model exporters (tools-only, needs torch).

``lsuv_calibrate`` rescales weights so every Conv/Linear output has ~unit std
on a seeded dummy input (LSUV-style, one forward pass). Untrained deep nets —
especially ReLU6/depthwise stacks like MobileNetV2 — otherwise shrink
activations geometrically until the logits sit at ~1e-8, below the parity
harness's atol, which makes every output comparison vacuously pass. Calibrated
models keep O(1) signal at every layer so a numerical bug anywhere actually
moves the outputs. (We verify agreement, not accuracy, so rescaled random
weights are exactly as good as any other random weights.)
"""

from __future__ import annotations


def lsuv_calibrate(model, dummy, *, target_std: float = 1.0, eps: float = 1e-8):
    """Scale each Conv2d/Linear so its output std is ~``target_std``.

    In-place, single forward pass: each hook rescales the module's weight and
    bias by the observed output std and returns the rescaled output, so every
    downstream module is calibrated against already-calibrated inputs.
    """
    import torch
    from torch import nn

    handles = []

    def make_hook():
        def hook(module, inputs, output):
            std = float(output.detach().float().std())
            if std < eps:
                return output
            s = std / target_std
            with torch.no_grad():
                module.weight.div_(s)
                if getattr(module, "bias", None) is not None:
                    module.bias.div_(s)
            return output / s
        return hook

    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            handles.append(m.register_forward_hook(make_hook()))
    try:
        with torch.no_grad():
            model(dummy)
    finally:
        for h in handles:
            h.remove()
    return model
