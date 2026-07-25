"""End-to-end compiler driver: ONNX path/model -> optimized, lowered module.

Ties the stages together in one call so the CLI, tests, and benchmarks share a
single pipeline:

    import -> (golden via ORT) -> passes (verified) -> shape inference
           -> memory plan -> CUDA codegen

QDQ (quantized) models are auto-detected and routed through the quant
pipeline. Their per-pass verification uses a *code-space* tolerance: the fp32
outputs of the rewritten graph may differ from ORT's QDQ evaluation by at most
2 output quanta (2·y_scale) — integer-exact requant vs ORT's
float-conv-then-quantize legitimately differ by ±1 code on rounding
boundaries; anything beyond ±2 codes is a real bug.
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
from mdlc.passes.quant import graph_is_qdq, quant_pipeline
from mdlc.passes.shape_inference import infer_shapes_by_execution, make_dummy_feeds
from mdlc.runtime.memory_planner import MemoryPlan, plan_memory
from mdlc.testing.harness import (
    _golden_outputs,
    check_graph_against_reference,
    make_pass_verifier,
)
from mdlc.testing.tolerances import FP32_NETWORK


def _quant_output_atol(graph: Graph) -> float:
    """±2 quanta of the largest output-side dequantize scale (see module
    docstring); falls back to the fp32 tier when no DQ feeds an output."""
    producers = graph.producers()
    scales = []
    for o in graph.outputs:
        p = producers.get(o)
        if p is not None and p.op_type == "DequantizeLinear":
            s = graph.initializers.get(p.inputs[1])
            if s is not None:
                scales.append(float(np.asarray(s).reshape(-1)[0]))
    return (2.0 * max(scales) + FP32_NETWORK.atol) if scales else FP32_NETWORK.atol


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
    quantized: bool = False
    verify_atol: float = FP32_NETWORK.atol

    def verify(self, rtol: Optional[float] = None, atol: Optional[float] = None):
        return check_graph_against_reference(
            self.graph, self.feeds, golden=self.golden,
            rtol=(0.0 if self.quantized else FP32_NETWORK.rtol)
            if rtol is None else rtol,
            atol=self.verify_atol if atol is None else atol,
            vacuous_factor=5.0 if self.quantized else 100.0)

    def int8_coverage(self) -> tuple[int, int]:
        """(quantized compute nodes, total compute nodes) in the optimized
        graph — compute = conv/gemm/matmul/add-shaped work."""
        q = sum(1 for n in self.graph.nodes
                if n.op_type in ("QConvFused", "QGemmFused", "QAddFused"))
        heavy = q + sum(1 for n in self.graph.nodes
                        if n.op_type in ("Conv", "FusedConvAct", "FusedConvBNAct",
                                         "Gemm", "FusedGemmAct", "MatMul", "Add"))
        return q, heavy

    def report(self) -> str:
        lines = []
        lines.append("== mdlc compile report ==")
        lines.append(self.original.summary())
        lines.append("-> optimized:")
        lines.append(self.graph.summary())
        lines.append(self.passes.report())
        lines.append(self.module.report())
        lines.append(self.mem_plan.report())
        if self.quantized:
            q, total = self.int8_coverage()
            pct = 100.0 * q / total if total else 0.0
            lines.append(f"int8 coverage: {q}/{total} compute nodes ({pct:.0f}%) "
                         f"— remainder runs fp32 between explicit Q/DQ kernels")
        checks = self.verify()
        ok = all(c.ok for c in checks)
        tol_note = (f" (code-space tier: atol={self.verify_atol:.3g} = 2 output "
                    "quanta)" if self.quantized else "")
        lines.append(f"correctness vs golden: {'PASS' if ok else 'FAIL'}{tol_note}")
        for c in checks:
            lines.append(f"  {c}")
        return "\n".join(lines)


def compile_onnx(
    source: Union[str, "object"],
    *,
    batch: int = 1,
    feeds: Optional[dict] = None,
    schedule_fn=None,
    dw_schedule_fn=None,
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

    quantized = graph_is_qdq(graph)
    if quantized:
        atol = _quant_output_atol(graph)
        verifier = (make_pass_verifier(feeds, golden, rtol=0.0, atol=atol,
                                       vacuous_factor=5.0)
                    if verify_passes else None)
        pm = quant_pipeline(verify=verifier)
    else:
        atol = FP32_NETWORK.atol
        verifier = make_pass_verifier(feeds, golden) if verify_passes else None
        pm = default_pipeline(verify=verifier)
    graph = pm.run(graph)

    infer_shapes_by_execution(graph, feeds)
    mem_plan = plan_memory(graph)
    module = emit_cuda_module(graph, schedule_fn=schedule_fn,
                              dw_schedule_fn=dw_schedule_fn)

    return Compiled(original=original, graph=graph, module=module,
                    mem_plan=mem_plan, passes=pm, feeds=feeds, golden=golden,
                    proto_bytes=proto, quantized=quantized, verify_atol=atol)
