"""Execute a ``CudaModule`` end-to-end on the CPU kernel simulator.

This runs the *generated CUDA kernels* (compiled by g++ via the CUDA-semantics
shim) in the module's launch order, threading real buffers between them, with
NumPy host fallbacks for ops we don't codegen yet. If the final outputs match
the reference executor, the generated kernels are proven to compose — the same
guarantee the GPU path needs, obtained without a GPU.

It assumes batch size 1 for conv (the im2col path lowers a single CHW image),
which matches our inference benchmarks.
"""

from __future__ import annotations

import numpy as np

from mdlc.codegen.cuda import templates as T
from mdlc.codegen.cuda.cpu_sim import (
    simulate_elementwise,
    simulate_gemm,
    simulate_im2col,
)
from mdlc.codegen.cuda.emit import CudaModule
from mdlc.codegen.schedule import GemmSchedule
from mdlc.ir import Graph
from mdlc.runtime.reference import ReferenceExecutor


def simulate_module(module: CudaModule, feeds: dict[str, np.ndarray],
                    schedule_fn=None) -> dict[str, np.ndarray]:
    g: Graph = module.graph
    schedule_fn = schedule_fn or (lambda node, mnk: GemmSchedule(64, 64, 16, 4, 4))
    ref = ReferenceExecutor(g)

    env: dict[str, np.ndarray] = {k: np.asarray(v) for k, v in g.initializers.items()}
    env.update({k: np.asarray(v) for k, v in feeds.items()})

    for launch in module.plan:
        if launch.kind == "im2col":
            m = launch.meta
            x = env[launch.inputs[0]]
            img = x[0] if x.ndim == 4 else x        # assume batch 1
            _, src = T.im2col_kernel("im2col")
            cols = simulate_im2col(
                src, "im2col", img.astype(np.float32),
                KH=m["KH"], KW=m["KW"], OH=m["OH"], OW=m["OW"],
                SH=m["SH"], SW=m["SW"], PH=m["PH"], PW=m["PW"], DH=m["DH"], DW=m["DW"])
            env[launch.outputs[0]] = cols

        elif launch.kind == "gemm":
            src = module.kernels[launch.kernel_name]
            m = launch.meta
            sched = schedule_fn(None, (m["M"], m["N"], m["K"]))
            if "out_4d" in m:                       # conv-as-GEMM
                w = env[launch.inputs[0]].reshape(m["weight_2d"]).astype(np.float32)
                B = env[launch.inputs[1]].astype(np.float32)
                bias = env[launch.inputs[2]].astype(np.float32) if m["with_bias"] else None
                out = simulate_gemm(src, launch.kernel_name, sched, w, B, bias)
                env[launch.outputs[0]] = out.reshape(m["out_4d"])
            else:                                   # plain Gemm/MatMul
                A = env[launch.inputs[0]].astype(np.float32)
                B = env[launch.inputs[1]].astype(np.float32)
                if m.get("transA"):
                    A = A.T.copy()
                if m.get("transB"):
                    B = B.T.copy()
                bias = env[launch.inputs[2]].astype(np.float32) if m.get("with_bias") else None
                out = simulate_gemm(src, launch.kernel_name, sched, A, B, bias)
                env[launch.outputs[0]] = out

        elif launch.kind == "elementwise":
            src = module.kernels[launch.kernel_name]
            arrs = [env[e].astype(np.float32) for e in launch.inputs]
            outs = simulate_elementwise(src, launch.kernel_name, launch.inputs,
                                        arrs, len(launch.outputs))
            ref_shape = env[launch.inputs[0]].shape
            for oname, ov in zip(launch.outputs, outs):
                env[oname] = ov.reshape(ref_shape)

        elif launch.kind == "host":
            node = launch.meta["node"]
            outs = ref._exec_node(node, env)
            for oname, ov in zip(node.outputs, outs):
                if oname:
                    env[oname] = ov
        else:
            raise RuntimeError(f"unknown launch kind {launch.kind!r}")

    return {o: env[o] for o in g.outputs}
