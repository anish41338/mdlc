"""Execute a ``CudaModule`` on a real GPU via NVRTC (GPU-gated).

Structurally identical to ``sim_executor`` — same launch plan, same arg order —
but buffers live in device memory and kernels run through the CUDA driver. The
CPU simulator validates this path's correctness bit-for-bit; here we get speed
and event-timed benchmarks. Requires ``cuda_available()``.

Pass a ``MemoryPlan`` to pool activations: one device allocation sized
``plan.pool_bytes`` serves every planned tensor at its 256-byte-aligned
offset (the planner's liveness analysis guarantees no live-range overlap,
including through zero-copy views). Without a plan, buffers are allocated
per tensor.
"""

from __future__ import annotations

import ctypes
import time

import numpy as np

from mdlc.codegen.cuda.emit import CudaModule
from mdlc.codegen.cuda.nvrtc_runtime import CudaContext, compile_to_ptx, cuda_available
from mdlc.ir import Graph
from mdlc.runtime.memory_planner import MemoryPlan
from mdlc.runtime.reference import ReferenceExecutor


class GpuExecutor:
    def __init__(self, module: CudaModule, ctx: CudaContext | None = None,
                 mem_plan: MemoryPlan | None = None) -> None:
        if not cuda_available():
            raise RuntimeError("no CUDA device; use sim_executor for CPU validation")
        self.module = module
        self.ctx = ctx or CudaContext()
        # NVRTC-compile every distinct kernel once into a single module.
        self.loaded = self.ctx.load_ptx(compile_to_ptx(module.sources(), arch=self.ctx.arch))
        self.ref = ReferenceExecutor(module.graph)
        self.mem_plan = mem_plan
        self._pool: ctypes.c_void_p | None = None
        self._pooled: set[str] = set()

    def _i(self, v):  # int arg
        return ctypes.c_int(int(v))

    @staticmethod
    def _off(ptr: ctypes.c_void_p, elems: int) -> ctypes.c_void_p:
        """Device pointer offset in float32 elements (for per-image batched
        conv and per-group weight/bias/output slices)."""
        return ctypes.c_void_p(ptr.value + int(elems) * 4)

    def _stage(self, feeds: dict[str, np.ndarray]):
        """Upload weights/feeds, bind pooled activations, and build the
        ``issue`` closure that replays the launch plan. H2D happens here,
        once — timing loops replay ``issue`` only (D2H likewise excluded)."""
        g: Graph = self.module.graph
        dev: dict[str, ctypes.c_void_p] = {}
        shapes: dict[str, tuple] = {}
        alloc_bytes = [0]

        def shape_of(name):
            info = g.info(name)
            return tuple(int(d) for d in info.shape)

        def _malloc(nbytes):
            alloc_bytes[0] += int(nbytes)
            return self.ctx.malloc(int(nbytes))

        # Stage weights + feeds onto the device.
        for name, arr in g.initializers.items():
            dev[name] = self.ctx.to_device(np.ascontiguousarray(arr, dtype=np.float32))
            shapes[name] = arr.shape
            alloc_bytes[0] += arr.size * 4
        for name, arr in feeds.items():
            dev[name] = self.ctx.to_device(np.ascontiguousarray(arr, dtype=np.float32))
            shapes[name] = arr.shape
            alloc_bytes[0] += int(np.prod(arr.shape)) * 4
        for name, shp in self.module.scratch_shapes.items():
            dev[name] = _malloc(int(np.prod(shp)) * 4)
            shapes[name] = shp

        # Pooled activations: one allocation, planner-assigned aligned offsets.
        if self.mem_plan is not None and self.mem_plan.pool_bytes:
            if self._pool is None:
                self._pool = _malloc(self.mem_plan.pool_bytes)
            base = self._pool.value
            for tname, bid in self.mem_plan.assignment.items():
                off = self.mem_plan.buffers[bid].offset
                assert off % self.mem_plan.align == 0, (tname, off)
                dev[tname] = ctypes.c_void_p(base + off)
                self._pooled.add(tname)

        def ensure(name, shp):
            if name not in dev:
                dev[name] = _malloc(int(np.prod(shp)) * 4)
            shapes[name] = shp

        def issue():
            for L in self.module.plan:
                if L.kind == "im2col":
                    m = L.meta
                    ensure(L.outputs[0], self.module.scratch_shapes[L.outputs[0]])
                    xin = dev[L.inputs[0]]
                    if m.get("in_offset"):
                        xin = self._off(xin, m["in_offset"])
                    args = [xin, dev[L.outputs[0]],
                            *[self._i(m[k]) for k in
                              ("C", "H", "W", "KH", "KW", "OH", "OW",
                               "SH", "SW", "PH", "PW", "DH", "DW")]]
                    self.loaded.launch("im2col", L.grid, L.block, args)
                elif L.kind == "gemm":
                    m = L.meta
                    ensure(L.outputs[0], shape_of(L.outputs[0]))
                    ptrs = [dev[i] for i in L.inputs]
                    a_ptr = ptrs[0]
                    if "w_rows" in m:     # grouped conv: weight-row slice
                        a_ptr = self._off(a_ptr, m["w_rows"][0] * m["K"])
                    args = [a_ptr, ptrs[1]]
                    if m.get("with_bias"):
                        b_ptr = ptrs[2]
                        if "oc_range" in m:
                            b_ptr = self._off(b_ptr, m["oc_range"][0])
                        args.append(b_ptr)
                    c_ptr = dev[L.outputs[0]]
                    if m.get("out_offset"):
                        c_ptr = self._off(c_ptr, m["out_offset"])
                    args += [c_ptr, self._i(m["M"]), self._i(m["N"]), self._i(m["K"])]
                    self.loaded.launch(L.kernel_name, L.grid, L.block, args)
                elif L.kind == "depthwise":
                    ensure(L.outputs[0], L.meta["out_4d"])
                    args = [dev[i] for i in L.inputs] + [dev[L.outputs[0]]]
                    self.loaded.launch(L.kernel_name, L.grid, L.block, args)
                elif L.kind in ("reduce", "pool"):
                    ensure(L.outputs[0], L.meta["out_shape"])
                    args = [dev[L.inputs[0]], dev[L.outputs[0]]]
                    self.loaded.launch(L.kernel_name, L.grid, L.block, args)
                elif L.kind == "view":
                    # metadata-only: the output aliases the input buffer
                    dev[L.outputs[0]] = dev[L.inputs[0]]
                    shapes[L.outputs[0]] = L.meta["out_shape"]
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

        return dev, shapes, shape_of, issue, alloc_bytes

    def run(self, feeds: dict[str, np.ndarray], *, timing: bool = False) -> dict:
        g: Graph = self.module.graph
        dev, shapes, shape_of, issue, _ = self._stage(feeds)
        issue()
        out = {o: self.ctx.from_device(dev[o], shape_of(o)) for o in g.outputs}
        result = {"outputs": out}
        if timing:
            result["ms"] = self.ctx.time_ms(issue)
        return result

    def bench(self, feeds: dict[str, np.ndarray], *, iters: int = 200,
              warmup: int = 50) -> dict:
        """Latency samples for the whole compiled forward.

        Method: feeds/weights staged once (H2D excluded), then ``iters``
        CUDA-event-timed replays of the launch plan after ``warmup`` untimed
        ones; outputs stay on device (D2H excluded). Returns per-iteration
        samples plus exact launch/memory accounting."""
        g: Graph = self.module.graph
        dev, shapes, shape_of, issue, alloc_bytes = self._stage(feeds)
        issue()   # allocate lazies + first-touch before timing
        samples = self.ctx.time_ms_samples(issue, iters=iters, warmup=warmup)
        launches = sum(1 for l in self.module.plan if l.kind != "view")
        return {
            "samples_ms": samples,
            "launches": launches,
            "device_bytes": alloc_bytes[0],
            "outputs": {o: self.ctx.from_device(dev[o], shape_of(o))
                        for o in g.outputs},
        }

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
            if oname in dev and oname not in self._pooled:
                self.ctx.free(dev[oname])
            dev[oname] = self.ctx.to_device(ov)


def run_on_gpu(module: CudaModule, feeds, *, timing: bool = False,
               mem_plan: MemoryPlan | None = None):
    return GpuExecutor(module, mem_plan=mem_plan).run(feeds, timing=timing)
