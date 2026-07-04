"""End-to-end compiler driver: ONNX path/model -> optimized, lowered module.

Ties the stages together in one call so the CLI, tests, and benchmarks share a
single pipeline:

    import -> (golden via ORT) -> passes (verified) -> shape inference
           -> memory plan -> CUDA codegen
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from mdlc.codegen.cuda.emit import CudaModule, emit_cuda_module
from mdlc.frontend import import_onnx, import_onnx_model
from mdlc.ir import Graph
from mdlc.passes import default_pipeline
from mdlc.passes.base import PassManager
from mdlc.passes.shape_inference import infer_shapes_by_execution, make_dummy_feeds
from mdlc.runtime.memory_planner import MemoryPlan, plan_memory
from mdlc.testing.harness import (
    _golden_outputs,
    check_graph_against_reference,
    make_pass_verifier,
)
from mdlc.testing.tolerances import FP32_NETWORK


@dataclass
class Compiled:
    original: Graph
    graph: Graph                 # optimized
    module: CudaModule
    mem_plan: MemoryPlan
    passes: PassManager
    feeds: dict
    golden: dict
    proto_bytes: Optional[bytes]

    def verify(self, rtol: float = FP32_NETWORK.rtol, atol: float = FP32_NETWORK.atol):
        return check_graph_against_reference(
            self.graph, self.feeds, golden=self.golden, rtol=rtol, atol=atol)

    def report(self) -> str:
        lines = []
        lines.append("== mdlc compile report ==")
        lines.append(self.original.summary())
        lines.append("-> optimized:")
        lines.append(self.graph.summary())
        lines.append(self.passes.report())
        lines.append(self.module.report())
        lines.append(self.mem_plan.report())
        checks = self.verify()
        ok = all(c.ok for c in checks)
        lines.append(f"correctness vs golden: {'PASS' if ok else 'FAIL'}")
        for c in checks:
            lines.append(f"  {c}")
        return "\n".join(lines)


def compile_onnx(
    source: Union[str, "object"],
    *,
    batch: int = 1,
    feeds: Optional[dict] = None,
    schedule_fn=None,
    verify_passes: bool = True,
) -> Compiled:
    if isinstance(source, str):
        graph = import_onnx(source)
        import onnx
        proto = onnx.load(source).SerializeToString()
    else:
        graph = import_onnx_model(source)
        proto = source.SerializeToString()

    original = graph.clone()
    if feeds is None:
        feeds = make_dummy_feeds(graph, batch=batch)
    golden = _golden_outputs(proto, original, feeds)

    verifier = make_pass_verifier(feeds, golden) if verify_passes else None
    pm = default_pipeline(verify=verifier)
    graph = pm.run(graph)

    infer_shapes_by_execution(graph, feeds)
    mem_plan = plan_memory(graph)
    module = emit_cuda_module(graph, schedule_fn=schedule_fn)

    return Compiled(original=original, graph=graph, module=module,
                    mem_plan=mem_plan, passes=pm, feeds=feeds, golden=golden,
                    proto_bytes=proto)
