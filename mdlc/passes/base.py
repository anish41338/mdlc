"""Pass infrastructure: the ``Pass`` protocol and a ``PassManager``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from mdlc.ir import Graph


class Pass:
    """A graph transformation. Subclasses implement ``run`` and return True if
    they modified the graph (so the manager knows whether to iterate)."""

    name: str = "pass"

    def run(self, graph: Graph) -> bool:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass
class PassRecord:
    name: str
    changed: bool
    nodes_before: int
    nodes_after: int


@dataclass
class PassManager:
    """Runs a sequence of passes, optionally to a fixed point, optionally
    verifying correctness after each pass via a user-supplied checker.

    The verifier is the project's core discipline: after every pass we can diff
    the transformed graph's outputs against a golden reference and abort the
    moment a pass introduces numerical drift, instead of discovering it days
    later in generated CUDA.
    """

    passes: list[Pass] = field(default_factory=list)
    max_iters: int = 8
    verify: Optional[Callable[[Graph, str], None]] = None
    verbose: bool = False
    history: list[PassRecord] = field(default_factory=list)

    def add(self, p: Pass) -> "PassManager":
        self.passes.append(p)
        return self

    def run(self, graph: Graph) -> Graph:
        for _ in range(self.max_iters):
            any_changed = False
            for p in self.passes:
                before = len(graph.nodes)
                changed = p.run(graph)
                after = len(graph.nodes)
                self.history.append(PassRecord(p.name, changed, before, after))
                if self.verbose and changed:
                    print(f"  [pass] {p.name}: {before} -> {after} nodes")
                if changed and self.verify is not None:
                    self.verify(graph, p.name)
                any_changed = any_changed or changed
            if not any_changed:
                break
        graph.reorder_topologically()
        return graph

    def report(self) -> str:
        lines = ["Pass history:"]
        for r in self.history:
            if r.changed:
                lines.append(f"  {r.name:24s} {r.nodes_before:3d} -> {r.nodes_after:3d}")
        return "\n".join(lines)


def default_pipeline(
    *, verify: Optional[Callable[[Graph, str], None]] = None, verbose: bool = False
) -> PassManager:
    """The standard optimization order.

    Order matters: fold constants and BN first (turns Conv+BN into a plain
    Conv), then fuse activations onto Conv/Gemm, then sweep up the remaining
    elementwise chains, with DCE cleaning up between rounds.
    """
    from mdlc.passes.constant_folding import ConstantFolding
    from mdlc.passes.dce import DeadCodeElimination
    from mdlc.passes.fuse_conv_bn import FuseConvBN
    from mdlc.passes.fuse_activation import FuseActivation
    from mdlc.passes.fuse_elementwise import FuseElementwise

    return PassManager(
        passes=[
            ConstantFolding(),
            FuseConvBN(),
            DeadCodeElimination(),
            FuseActivation(),
            FuseElementwise(),
            DeadCodeElimination(),
        ],
        verify=verify,
        verbose=verbose,
    )
