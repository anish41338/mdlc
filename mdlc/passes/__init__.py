"""Graph-level optimization passes.

The classic ML-compiler set: constant folding, dead-code elimination, and
fusion (the heavy hitter). Each pass takes a ``Graph`` and mutates it in place,
reporting whether it changed anything so the ``PassManager`` can iterate to a
fixed point.
"""

from mdlc.passes.base import Pass, PassManager, default_pipeline
from mdlc.passes.constant_folding import ConstantFolding
from mdlc.passes.dce import DeadCodeElimination
from mdlc.passes.fuse_conv_bn import FuseConvBN
from mdlc.passes.fuse_activation import FuseActivation
from mdlc.passes.fuse_elementwise import FuseElementwise

__all__ = [
    "Pass",
    "PassManager",
    "default_pipeline",
    "ConstantFolding",
    "DeadCodeElimination",
    "FuseConvBN",
    "FuseActivation",
    "FuseElementwise",
]
