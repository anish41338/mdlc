"""A single operation in the graph IR."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Node:
    """One operator: an op_type, named input/output edges, and attributes.

    Edges are referenced by string name, matching the ``TensorInfo`` keys in
    the owning ``Graph``. An empty string in ``inputs`` denotes an optional/
    omitted input (ONNX convention, e.g. Conv with no bias).

    ``attrs`` holds op parameters already decoded from ONNX into plain Python
    (ints, floats, lists, strings). Fused ops add their own attrs, e.g. a
    ``FusedConvBNAct`` node carries ``activation="Relu"``.
    """

    op_type: str
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    attrs: dict[str, Any] = field(default_factory=dict)
    name: str = ""
    # Set on fused nodes: the list of original op_types this node replaced,
    # in execution order. Used for reporting and the autotuner cache key.
    fused_from: Optional[list[str]] = None

    def attr(self, key: str, default: Any = None) -> Any:
        return self.attrs.get(key, default)

    @property
    def is_fused(self) -> bool:
        return self.fused_from is not None

    def real_inputs(self) -> list[str]:
        """Inputs with optional/omitted ("") slots removed."""
        return [i for i in self.inputs if i]

    def replace_input(self, old: str, new: str) -> None:
        self.inputs = [new if i == old else i for i in self.inputs]

    def __repr__(self) -> str:
        ins = ", ".join(self.inputs)
        outs = ", ".join(self.outputs)
        tag = f" <-{'+'.join(self.fused_from)}" if self.fused_from else ""
        return f"{self.op_type}({ins}) -> ({outs}){tag}"
