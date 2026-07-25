"""NVRTC + CUDA Driver API runtime via ctypes (GPU-gated).

This is the real device backend: it JIT-compiles generated kernel source to PTX
with NVRTC, loads it through the CUDA driver API, manages device buffers, and
launches kernels with event-based timing. It needs no Python CUDA package — only
the NVIDIA shared libraries present on any machine with a driver + toolkit.

On a machine without CUDA, ``cuda_available()`` returns False and the rest of
the module is never exercised; the CPU simulator (``cpu_sim``) covers
correctness in the meantime. None of this code changes when a GPU appears.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import sys
from typing import Optional

import numpy as np

# ----- dynamic library loading -------------------------------------------

def _load(names: list[str]) -> Optional[ctypes.CDLL]:
    for n in names:
        try:
            return ctypes.CDLL(n)
        except OSError:
            continue
    # last resort: let the loader search
    for n in names:
        found = ctypes.util.find_library(n)
        if found:
            try:
                return ctypes.CDLL(found)
            except OSError:
                pass
    return None


def _cuda_libs():
    if sys.platform == "win32":
        driver = _load(["nvcuda.dll"])
        # nvrtc dll is version-suffixed; try a range of known names.
        nvrtc = _load([f"nvrtc64_{v}_0.dll" for v in (130, 128, 125, 124, 123,
                                                      122, 121, 120, 112, 111, 110)]
                      + ["nvrtc64_120_0.dll", "nvrtc.dll"])
    else:
        driver = _load(["libcuda.so", "libcuda.so.1"])
        nvrtc = _load(["libnvrtc.so", "libnvrtc.so.12", "libnvrtc.so.11"])
    return driver, nvrtc


_DRIVER, _NVRTC = _cuda_libs()


# ----- driver prototypes --------------------------------------------------
#
# ctypes defaults every undeclared argument to C `int`. That silently truncates
# the 64-bit values in this API — `CUdeviceptr` is an `unsigned long long` and
# the copy/alloc sizes are `size_t` — so the prototypes are declared explicitly
# rather than relying on the ABI zero-extending a 32-bit register write.
#
# The driver also ships versioned entry points: the *unsuffixed* `cuMemAlloc`,
# `cuMemcpy*` and `cuCtxCreate` symbols in libcuda are the legacy v1 ABI (32-bit
# sizes), and the CUDA headers `#define` them to `_v2`. dlsym/ctypes bypasses
# those macros, so resolve `_v2` first and fall back only if absent.
CUdeviceptr = ctypes.c_ulonglong


def _sym(name: str):
    """Resolve a driver symbol, preferring the modern `_v2` ABI."""
    if _DRIVER is None:
        return None
    for candidate in (f"{name}_v2", name):
        fn = getattr(_DRIVER, candidate, None)
        if fn is not None:
            return fn
    return None


def _declare_driver_prototypes() -> None:
    if _DRIVER is None:
        return
    protos = {
        "cuInit": [ctypes.c_uint],
        "cuDeviceGetCount": [ctypes.POINTER(ctypes.c_int)],
        "cuDeviceGet": [ctypes.POINTER(ctypes.c_int), ctypes.c_int],
        "cuDeviceGetAttribute": [ctypes.POINTER(ctypes.c_int), ctypes.c_int,
                                 ctypes.c_int],
        "cuDeviceGetName": [ctypes.c_char_p, ctypes.c_int, ctypes.c_int],
        "cuCtxSynchronize": [],
        "cuModuleLoadData": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p],
        "cuModuleGetFunction": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p,
                                ctypes.c_char_p],
        "cuLaunchKernel": [ctypes.c_void_p,
                           ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
                           ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
                           ctypes.c_uint, ctypes.c_void_p,
                           ctypes.POINTER(ctypes.c_void_p),
                           ctypes.POINTER(ctypes.c_void_p)],
        "cuEventCreate": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint],
        "cuEventRecord": [ctypes.c_void_p, ctypes.c_void_p],
        "cuEventSynchronize": [ctypes.c_void_p],
        "cuEventElapsedTime": [ctypes.POINTER(ctypes.c_float), ctypes.c_void_p,
                               ctypes.c_void_p],
    }
    for name, argtypes in protos.items():
        fn = getattr(_DRIVER, name, None)
        if fn is not None:
            fn.argtypes = argtypes
            fn.restype = ctypes.c_int

    # Versioned entry points, declared on the resolved `_v2` object.
    # `CUdeviceptr` is an integer handle, but it is the same 64-bit width as a
    # pointer, so the module keeps `c_void_p` as its single pointer currency
    # (see `GpuExecutor._off`, which does arithmetic on `.value`). What matters
    # here is that the *sizes* are `size_t`, not `int`.
    versioned = {
        "cuCtxCreate": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint,
                        ctypes.c_int],
        "cuMemAlloc": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t],
        "cuMemFree": [ctypes.c_void_p],
        "cuMemcpyHtoD": [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t],
        "cuMemcpyDtoH": [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t],
    }
    for name, argtypes in versioned.items():
        fn = _sym(name)
        if fn is not None:
            fn.argtypes = argtypes
            fn.restype = ctypes.c_int


_declare_driver_prototypes()


def cuda_available() -> bool:
    """True iff the CUDA driver + NVRTC are present and a device initializes."""
    if _DRIVER is None or _NVRTC is None:
        return False
    try:
        if _DRIVER.cuInit(0) != 0:
            return False
        count = ctypes.c_int(0)
        if _DRIVER.cuDeviceGetCount(ctypes.byref(count)) != 0:
            return False
        return count.value > 0
    except Exception:
        return False


# ----- error checking -----------------------------------------------------

class CudaError(RuntimeError):
    pass


def _ck(code: int, what: str) -> None:
    if code != 0:
        raise CudaError(f"{what} failed with CUDA code {code}")


def _ckn(code: int, what: str) -> None:
    if code != 0:
        msg = b""
        try:
            getstr = _NVRTC.nvrtcGetErrorString
            getstr.restype = ctypes.c_char_p
            msg = getstr(code)
        except Exception:
            pass
        raise CudaError(f"{what} failed: NVRTC code {code} {msg!r}")


# ----- NVRTC compile ------------------------------------------------------

def compile_to_ptx(source: str, name: str = "kern.cu",
                   arch: Optional[str] = None, *,
                   fast_math: bool = False) -> bytes:
    """Compile CUDA C source to PTX with NVRTC. ``arch`` like 'sm_75'.

    ``fast_math`` is OFF by default: correctness/parity runs must use IEEE
    math so the GPU matches the sim and ORT. It exists only as a *labeled*
    benchmark variant — never enable it for a run whose output is compared
    against a golden.
    """
    if _NVRTC is None:
        raise CudaError("NVRTC library not found")
    prog = ctypes.c_void_p()
    _ckn(_NVRTC.nvrtcCreateProgram(ctypes.byref(prog), source.encode(),
                                   name.encode(), 0, None, None),
         "nvrtcCreateProgram")
    opts = []
    if arch:
        opts.append(f"--gpu-architecture=compute_{arch.split('_')[-1]}".encode())
    if fast_math:
        opts.append(b"--use_fast_math")
    arr = (ctypes.c_char_p * len(opts))(*opts)
    rc = _NVRTC.nvrtcCompileProgram(prog, len(opts), arr)
    # always fetch the build log for diagnostics
    log_size = ctypes.c_size_t()
    _NVRTC.nvrtcGetProgramLogSize(prog, ctypes.byref(log_size))
    log = ctypes.create_string_buffer(log_size.value)
    _NVRTC.nvrtcGetProgramLog(prog, log)
    if rc != 0:
        raise CudaError(f"NVRTC compile failed:\n{log.value.decode(errors='replace')}")
    ptx_size = ctypes.c_size_t()
    _NVRTC.nvrtcGetPTXSize(prog, ctypes.byref(ptx_size))
    ptx = ctypes.create_string_buffer(ptx_size.value)
    _NVRTC.nvrtcGetPTX(prog, ptx)
    _NVRTC.nvrtcDestroyProgram(ctypes.byref(prog))
    return ptx.raw


# ----- device context / launching ----------------------------------------

class CudaModule:
    """A loaded PTX module: resolve and launch its kernels."""

    def __init__(self, ctx: "CudaContext", ptx: bytes) -> None:
        self.ctx = ctx
        self._mod = ctypes.c_void_p()
        _ck(_DRIVER.cuModuleLoadData(ctypes.byref(self._mod), ptx), "cuModuleLoadData")
        self._funcs: dict[str, ctypes.c_void_p] = {}

    def func(self, name: str) -> ctypes.c_void_p:
        if name not in self._funcs:
            f = ctypes.c_void_p()
            _ck(_DRIVER.cuModuleGetFunction(ctypes.byref(f), self._mod, name.encode()),
                f"cuModuleGetFunction({name})")
            self._funcs[name] = f
        return self._funcs[name]

    def launch(self, name: str, grid, block, args: list, *, shared_bytes: int = 0):
        """Launch ``name``. ``args`` is a list of (ctype_value) — device
        pointers as c_void_p, scalars as c_int/c_float."""
        f = self.func(name)
        kargs = (ctypes.c_void_p * len(args))()
        keep = []
        for i, a in enumerate(args):
            keep.append(a)
            kargs[i] = ctypes.cast(ctypes.byref(a), ctypes.c_void_p)
        # int() rather than pass-through: a numpy integer from a shape
        # computation has no ctypes conversion and would raise at the FFI
        # boundary on device only.
        gx, gy, gz = (int(v) for v in (list(grid) + [1, 1])[:3])
        bx, by, bz = (int(v) for v in (list(block) + [1, 1])[:3])
        if bx * by * bz > 1024:
            raise CudaError(
                f"{name}: {bx * by * bz} threads/block exceeds the CUDA limit "
                f"of 1024 (block={bx},{by},{bz})")
        _ck(_DRIVER.cuLaunchKernel(f, gx, gy, gz, bx, by, bz,
                                   int(shared_bytes), None, kargs, None),
            f"cuLaunchKernel({name})")


class CudaContext:
    """Owns a device + context and provides allocation/copy/timing."""

    def __init__(self, device: int = 0) -> None:
        if not cuda_available():
            raise CudaError("no CUDA device available")
        _ck(_DRIVER.cuInit(0), "cuInit")
        dev = ctypes.c_int()
        _ck(_DRIVER.cuDeviceGet(ctypes.byref(dev), device), "cuDeviceGet")
        self.device = dev
        self._ctx = ctypes.c_void_p()
        _ck(_sym("cuCtxCreate")(ctypes.byref(self._ctx), 0, dev), "cuCtxCreate")
        self.arch = self._compute_arch()

    def _compute_arch(self) -> str:
        major, minor = ctypes.c_int(), ctypes.c_int()
        # CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR=75, MINOR=76
        _DRIVER.cuDeviceGetAttribute(ctypes.byref(major), 75, self.device)
        _DRIVER.cuDeviceGetAttribute(ctypes.byref(minor), 76, self.device)
        return f"sm_{major.value}{minor.value}"

    def device_name(self) -> str:
        buf = ctypes.create_string_buffer(256)
        _DRIVER.cuDeviceGetName(buf, 256, self.device)
        return buf.value.decode(errors="replace")

    def malloc(self, nbytes: int) -> ctypes.c_void_p:
        ptr = ctypes.c_void_p()
        _ck(_sym("cuMemAlloc")(ctypes.byref(ptr), int(nbytes)), "cuMemAlloc")
        return ptr

    def free(self, ptr: ctypes.c_void_p) -> None:
        _ck(_sym("cuMemFree")(ptr), "cuMemFree")

    def to_device(self, arr: np.ndarray) -> ctypes.c_void_p:
        arr = np.ascontiguousarray(arr)
        ptr = self.malloc(arr.nbytes)
        _ck(_sym("cuMemcpyHtoD")(ptr, arr.ctypes.data_as(ctypes.c_void_p),
                                 int(arr.nbytes)),
            "cuMemcpyHtoD")
        return ptr

    def from_device(self, ptr: ctypes.c_void_p, shape, dtype=np.float32) -> np.ndarray:
        out = np.empty(shape, dtype=dtype)
        _ck(_sym("cuMemcpyDtoH")(out.ctypes.data_as(ctypes.c_void_p), ptr,
                                 int(out.nbytes)),
            "cuMemcpyDtoH")
        return out

    def synchronize(self) -> None:
        _ck(_DRIVER.cuCtxSynchronize(), "cuCtxSynchronize")

    def load_ptx(self, ptx: bytes) -> CudaModule:
        return CudaModule(self, ptx)

    # event-based timing of a callable that issues launches
    def time_ms(self, fn, iters: int = 50, warmup: int = 10) -> float:
        """Mean ms/iteration over one bracketing event pair (cheap, but blind
        to per-iteration variance — prefer ``time_ms_samples`` + median)."""
        start, stop = ctypes.c_void_p(), ctypes.c_void_p()
        _ck(_DRIVER.cuEventCreate(ctypes.byref(start), 0), "cuEventCreate")
        _ck(_DRIVER.cuEventCreate(ctypes.byref(stop), 0), "cuEventCreate")
        for _ in range(warmup):
            fn()
        self.synchronize()
        _ck(_DRIVER.cuEventRecord(start, None), "cuEventRecord")
        for _ in range(iters):
            fn()
        _ck(_DRIVER.cuEventRecord(stop, None), "cuEventRecord")
        _ck(_DRIVER.cuEventSynchronize(stop), "cuEventSynchronize")
        ms = ctypes.c_float()
        _ck(_DRIVER.cuEventElapsedTime(ctypes.byref(ms), start, stop), "cuEventElapsedTime")
        return ms.value / iters

    def time_ms_samples(self, fn, iters: int = 50, warmup: int = 10) -> list[float]:
        """Per-iteration ms samples, one CUDA-event pair per iteration, so the
        caller can take a median and reject noisy configs (IQR gate) — the
        honest way to time on shared GPUs whose clocks can't be locked."""
        start, stop = ctypes.c_void_p(), ctypes.c_void_p()
        _ck(_DRIVER.cuEventCreate(ctypes.byref(start), 0), "cuEventCreate")
        _ck(_DRIVER.cuEventCreate(ctypes.byref(stop), 0), "cuEventCreate")
        for _ in range(warmup):
            fn()
        self.synchronize()
        out = []
        ms = ctypes.c_float()
        for _ in range(iters):
            _ck(_DRIVER.cuEventRecord(start, None), "cuEventRecord")
            fn()
            _ck(_DRIVER.cuEventRecord(stop, None), "cuEventRecord")
            _ck(_DRIVER.cuEventSynchronize(stop), "cuEventSynchronize")
            _ck(_DRIVER.cuEventElapsedTime(ctypes.byref(ms), start, stop),
                "cuEventElapsedTime")
            out.append(float(ms.value))
        return out
