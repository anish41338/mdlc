"""Execute a ``CudaModule`` on a real GPU via NVRTC (GPU-gated).

Structurally identical to ``sim_executor`` — same launch plan, same arg order —
but buffers live in device memory and kernels run through the CUDA driver. The
CPU simulator validates this path's correctness bit-for-bit; here we get speed
and event-timed benchmarks. Requires ``cuda_available()``.

Device buffers are allocated per tensor for clarity; ``mdlc.runtime.memory_planner``
provides the liveness plan a production version would use to pool them.
"""

from __future__ import annotations

import ctypes
import time

import numpy as np

from mdlc.codegen.cuda.emit import CudaModule
from mdlc.codegen.cuda.nvrtc_runtime import CudaContext, compile_to_ptx, cuda_available
from mdlc.ir import Graph
from mdlc.runtime.reference import ReferenceExecutor


class GpuExecutor:
    def __init__(self, module: CudaModule, ctx: CudaContext | None = None) -> None:
        if not cuda_available():
            raise RuntimeError("no CUDA device; use sim_executor for CPU validation")
        self.module = module
        self.ctx = ctx or CudaContext()
        # NVRTC-compile every distinct kernel once into a single module.
        self.loaded = self.ctx.load_ptx(compile_to_ptx(module.sources(), arch=self.ctx.arch))
        self.ref = ReferenceExecutor(module.graph)

    def _i(self, v):  # int arg
        return ctypes.c_int(int(v))

    def run(self, feeds: dict[str, np.ndarray], *, timing: bool = False) -> dict:
        g: Graph = self.module.graph
        dev: dict[str, ctypes.c_void_p] = {}
        shapes: dict[str, tuple] = {}

        def shape_of(name):
            info = g.info(name)
            return tuple(int(d) for d in info.shape)

        # Stage weights + feeds onto the device.
        for name, arr in g.initializers.items():
            dev[name] = self.ctx.to_device(np.ascontiguousarray(arr, dtype=np.float32))
            shapes[name] = arr.shape
        for name, arr in feeds.items():
            dev[name] = self.ctx.to_device(np.ascontiguousarray(arr, dtype=np.float32))
            shapes[name] = arr.shape
        for name, shp in self.module.scratch_shapes.items():
            dev[name] = self.ctx.malloc(int(np.prod(shp)) * 4)
            shapes[name] = shp

        def ensure(name, shp):
            if name not in dev:
                dev[name] = self.ctx.malloc(int(np.prod(shp)) * 4)
                shapes[name] = shp

        def issue():
            for L in self.module.plan:
                if L.kind == "im2col":
                    m = L.meta
                    ensure(L.outputs[0], self.module.scratch_shapes[L.outputs[0]])
                    args = [dev[L.inputs[0]], dev[L.outputs[0]],
                            *[self._i(m[k]) for k in
                              ("C", "H", "W", "KH", "KW", "OH", "OW",
                               "SH", "SW", "PH", "PW", "DH", "DW")]]
                    self.loaded.launch("im2col", L.grid, L.block, args)
                elif L.kind == "gemm":
                    m = L.meta
                    ensure(L.outputs[0], shape_of(L.outputs[0]))
                    ptrs = [dev[i] for i in L.inputs]
                    args = [*ptrs[:2]]
                    if m.get("with_bias"):
                        args.append(ptrs[2])
                    args += [dev[L.outputs[0]], self._i(m["M"]), self._i(m["N"]), self._i(m["K"])]
                    self.loaded.launch(L.kernel_name, L.grid, L.block, args)
                elif L.kind == "elementwise":
                    m = L.meta
                    for o in L.outputs:
                        ensure(o, shape_of(o))
                    args = [dev[i] for i in L.inputs] + [dev[o] for o in L.outputs]
                    args.append(self._i(m["N"]))
                    self.loaded.launch(L.kernel_name, L.grid, L.block, args)
                elif L.kind == "host":
                    self._run_host(L, dev, shapes, g)
            self.ctx.synchronize()

        issue()
        out = {o: self.ctx.from_device(dev[o], shape_of(o)) for o in g.outputs}
        result = {"outputs": out}
        if timing:
            result["ms"] = self.ctx.time_ms(issue)
        return result

    def _run_host(self, L, dev, shapes, g):
        """Bounce host-fallback ops through CPU: D2H, run reference op, H2D."""
        node = L.meta["node"]
        env = {}
        for e in node.real_inputs():
            env[e] = self.ctx.from_device(dev[e], shapes[e])
        outs = self.ref._exec_node(node, env)
        for oname, ov in zip(node.outputs, outs):
            if not oname:
                continue
            ov = np.ascontiguousarray(ov, dtype=np.float32)
            shapes[oname] = ov.shape
            if oname in dev:
                self.ctx.free(dev[oname])
            dev[oname] = self.ctx.to_device(ov)


def run_on_gpu(module: CudaModule, feeds, *, timing: bool = False):
    return GpuExecutor(module).run(feeds, timing=timing)
