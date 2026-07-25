"""Execute a ``CudaModule`` end-to-end on the CPU kernel simulator.

This runs the *generated CUDA kernels* (compiled by g++ via the CUDA-semantics
shim) in the module's launch order, threading real buffers between them, with
NumPy host fallbacks for ops we don't codegen yet. If the final outputs match
the reference executor, the generated kernels are proven to compose — the same
guarantee the GPU path needs, obtained without a GPU.

Conv batching: the plan carries one im2col+GEMM launch pair per image
(``batch_index`` in launch meta), so any N runs through the same single-image
kernels.
"""

from __future__ import annotations

import numpy as np

from mdlc.codegen.cuda import templates as T
from mdlc.codegen.cuda.cpu_sim import (
    simulate_elementwise,
    simulate_gemm,
    simulate_im2col,
    simulate_kernel,
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
            img = x[m.get("batch_index", 0)] if x.ndim == 4 else x
            if "c_range" in m:                      # grouped conv channel slice
                c0, c1 = m["c_range"]
                img = img[c0:c1]
            _, src = T.im2col_kernel("im2col")
            cols = simulate_im2col(
                src, "im2col", img.astype(np.float32),
                KH=m["KH"], KW=m["KW"], OH=m["OH"], OW=m["OW"],
                SH=m["SH"], SW=m["SW"], PH=m["PH"], PW=m["PW"], DH=m["DH"], DW=m["DW"])
            env[launch.outputs[0]] = cols

        elif launch.kind == "depthwise":
            src = module.kernels[launch.kernel_name]
            m = launch.meta
            arrs = [env[e].astype(np.float32) for e in launch.inputs]
            out_n = int(np.prod(m["out_4d"]))
            outs = simulate_kernel(src, launch.kernel_name, inputs=arrs,
                                   output_sizes=[out_n], grid=launch.grid,
                                   block=launch.block[0])
            env[launch.outputs[0]] = outs[0].reshape(m["out_4d"])

        elif launch.kind == "gemm":
            src = module.kernels[launch.kernel_name]
            m = launch.meta
            # the schedule the kernel was emitted with (tuned or default);
            # falling back to schedule_fn keeps hand-built test modules working
            sched = m.get("sched") or schedule_fn(None, (m["M"], m["N"], m["K"]))
            if "out_4d" in m:                       # conv-as-GEMM, one image
                w = env[launch.inputs[0]].reshape(m["weight_2d"]).astype(np.float32)
                bias = env[launch.inputs[2]].astype(np.float32) if m["with_bias"] else None
                if "w_rows" in m:                   # grouped conv weight slice
                    r0, r1 = m["w_rows"]
                    w = w[r0:r1]
                    if bias is not None:
                        bias = bias[r0:r1]
                B = env[launch.inputs[1]].astype(np.float32)
                out = simulate_gemm(src, launch.kernel_name, sched, w, B, bias)
                oname = launch.outputs[0]
                n4, oc, oh, ow = m["out_4d"]
                if oname not in env or env[oname].shape != tuple(m["out_4d"]):
                    env[oname] = np.zeros(m["out_4d"], np.float32)
                oc0, oc1 = m.get("oc_range", (0, oc))
                env[oname][m.get("batch_index", 0), oc0:oc1] = \
                    out.reshape(oc1 - oc0, oh, ow)
            else:                                   # plain Gemm/MatMul
                # No host-side transpose: transA/transB are baked into the
                # emitted kernel's load indices, so the operands are passed
                # exactly as the device path passes them. Transposing here
                # would double-transpose, and used to be the only reason the
                # sim agreed with the reference while the GPU path did not.
                A = env[launch.inputs[0]].astype(np.float32)
                B = env[launch.inputs[1]].astype(np.float32)
                bias = env[launch.inputs[2]].astype(np.float32) if m.get("with_bias") else None
                out = simulate_gemm(src, launch.kernel_name, sched, A, B, bias,
                                    mnk=(m["M"], m["N"], m["K"]))
                env[launch.outputs[0]] = out

        elif launch.kind == "elementwise":
            src = module.kernels[launch.kernel_name]
            m = launch.meta
            arrs = [env[e].astype(np.float32) for e in launch.inputs]
            outs = simulate_elementwise(src, launch.kernel_name, launch.inputs,
                                        arrs, len(launch.outputs), out_n=m["N"])
            out_shape = m.get("out_shape") or env[launch.inputs[0]].shape
            for oname, ov in zip(launch.outputs, outs):
                env[oname] = ov.reshape(out_shape)

        elif launch.kind in ("reduce", "pool"):
            src = module.kernels[launch.kernel_name]
            m = launch.meta
            x = env[launch.inputs[0]].astype(np.float32)
            outs = simulate_kernel(src, launch.kernel_name, inputs=[x],
                                   output_sizes=[m["out_n"]],
                                   grid=launch.grid, block=launch.block[0])
            env[launch.outputs[0]] = outs[0].reshape(m["out_shape"])

        elif launch.kind == "view":
            env[launch.outputs[0]] = \
                env[launch.inputs[0]].reshape(launch.meta["out_shape"])

        elif launch.kind == "host":
            node = launch.meta["node"]
            outs = ref._exec_node(node, env)
            for oname, ov in zip(node.outputs, outs):
                if oname:
                    env[oname] = ov
        else:
            raise RuntimeError(f"unknown launch kind {launch.kind!r}")

    return {o: env[o] for o in g.outputs}
