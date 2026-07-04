"""Run generated CUDA-C kernels on the CPU, faithfully, for correctness tests.

No GPU is required to *validate* codegen. We compile the exact generated kernel
source with a small shim that emulates CUDA's execution model on the host:

  * one OS thread per CUDA thread within a block (Win32 ``_beginthreadex``),
  * ``__shared__`` -> a block-static array shared by those threads,
  * ``__syncthreads()`` -> a reusable sense-reversing barrier over the block,
  * blocks executed serially (so the block-static shared memory is reused).

This is the same trick CPU CUDA backends use. A tiling/index bug in the
generated GEMM is caught here, on CPU, instead of as silent corruption on a GPU
we don't have — and the *identical* source runs unchanged via NVRTC on a real
device.

The threading uses the Win32 API directly so it works on MinGW builds that ship
the win32 (non-POSIX) GCC threading model, where ``std::thread`` is absent.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile

import numpy as np

_SHIM = r"""
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0600   // Vista+: CONDITION_VARIABLE, condition vars
#endif
#include <windows.h>
#include <process.h>
#include <vector>
#include <string>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstddef>

struct dim3 { unsigned x, y, z; };
static dim3 blockIdx, blockDim, gridDim;
__thread dim3 threadIdx;

#define __global__
#define __shared__ static
#define __restrict__
#define __device__
#define __forceinline__ inline
#define __expf expf
#ifndef CUDART_INF_F
#define CUDART_INF_F (__builtin_huge_valf())
#endif

// Reusable sense-reversing barrier over a block's threads.
struct Barrier {
    CRITICAL_SECTION cs; CONDITION_VARIABLE cv;
    int count, total, sense;
    void init(int t) {
        InitializeCriticalSection(&cs);
        InitializeConditionVariable(&cv);
        count = 0; total = t; sense = 0;
    }
    void destroy() { DeleteCriticalSection(&cs); }
    void wait() {
        EnterCriticalSection(&cs);
        int s = sense;
        if (++count == total) { count = 0; sense ^= 1; WakeAllConditionVariable(&cv); }
        else { while (s == sense) SleepConditionVariableCS(&cv, &cs, INFINITE); }
        LeaveCriticalSection(&cs);
    }
};
static Barrier g_bar;
#define __syncthreads() g_bar.wait()

typedef void (*BlockBody)();
static BlockBody g_body = nullptr;
static unsigned __stdcall _threadproc(void* p) {
    unsigned t = (unsigned)(size_t)p;
    threadIdx.x = t; threadIdx.y = 0; threadIdx.z = 0;
    g_body();
    return 0;
}
static void launch_block(unsigned nthreads) {
    g_bar.init((int)nthreads);
    std::vector<HANDLE> hs(nthreads);
    for (unsigned t = 0; t < nthreads; ++t)
        hs[t] = (HANDLE)_beginthreadex(0, 0, _threadproc, (void*)(size_t)t, 0, 0);
    for (unsigned t = 0; t < nthreads; ++t) {
        WaitForSingleObject(hs[t], INFINITE);
        CloseHandle(hs[t]);
    }
    g_bar.destroy();
}

static std::vector<float> load_floats(const char* path, long n) {
    std::vector<float> v(n);
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(2); }
    fread(v.data(), sizeof(float), (size_t)n, f);
    fclose(f);
    return v;
}
static void save_floats(const char* path, const float* p, long n) {
    FILE* f = fopen(path, "wb");
    fwrite(p, sizeof(float), (size_t)n, f);
    fclose(f);
}
"""


def _cache_dir() -> str:
    d = os.environ.get("MDLC_SIM_CACHE") or os.path.join(
        os.path.expanduser("~"), ".cache", "mdlc-sim")
    os.makedirs(d, exist_ok=True)
    return d


def _compile_cached(cpp_source: str) -> str:
    """Compile a sim harness once per distinct source; reuse the exe after.

    Whole-model simulation issues hundreds of launches whose sources repeat
    (same kernel, different data), and parity sweeps compile many small
    variants — content-addressed caching turns ~1s of g++ per launch into a
    one-time cost. Keyed by source hash; exe written atomically so concurrent
    pytest workers can share the cache.
    """
    key = hashlib.sha256(cpp_source.encode()).hexdigest()[:24]
    exe_path = os.path.join(_cache_dir(), f"sim_{key}.exe")
    if os.path.exists(exe_path):
        return exe_path
    with tempfile.TemporaryDirectory() as bd:
        src_path = os.path.join(bd, "kernel_test.cpp")
        tmp_exe = os.path.join(bd, "kernel_test.exe")
        with open(src_path, "w") as f:
            f.write(cpp_source)
        cc = subprocess.run(
            ["g++", "-O2", "-std=c++14", src_path, "-o", tmp_exe],
            capture_output=True, text=True,
        )
        if cc.returncode != 0:
            raise RuntimeError(f"g++ failed:\n{cc.stderr}")
        try:
            os.replace(tmp_exe, exe_path)
        except OSError:
            # Another worker won the race; theirs is identical.
            if not os.path.exists(exe_path):
                raise
    return exe_path


def _compile_and_run(cpp_source: str, workdir: str) -> None:
    exe_path = _compile_cached(cpp_source)
    run = subprocess.run([exe_path, workdir], capture_output=True, text=True)
    if run.returncode != 0:
        raise RuntimeError(f"kernel run failed (rc={run.returncode}):\n{run.stdout}\n{run.stderr}")


def _prep_kernel(source: str) -> str:
    return source.replace("#include <math_constants.h>", "")


def _dump(workdir: str, name: str, arr: np.ndarray) -> str:
    path = os.path.join(workdir, name + ".bin")
    arr.astype(np.float32).ravel().tofile(path)
    return path


def simulate_gemm(source, kernel_name, sched, A, B, bias):
    """Compile+run a generated GEMM kernel on CPU; return C = A@B (+bias)->act."""
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    gx = (N + sched.BN - 1) // sched.BN
    gy = (M + sched.BM - 1) // sched.BM
    nthreads = sched.threads_per_block()
    with_bias = bias is not None

    with tempfile.TemporaryDirectory() as wd:
        _dump(wd, "A", A)
        _dump(wd, "B", B)
        if with_bias:
            _dump(wd, "bias", bias)
        bias_load = ('auto bias = load_floats((dir+"/bias.bin").c_str(), N);'
                     if with_bias else "")
        call = (f"{kernel_name}(gA, gB, " + ("gBias, " if with_bias else "")
                + "gC, gM, gN, gK)")
        main = f"""
{_SHIM}
{_prep_kernel(source)}

static const float *gA, *gB, *gBias; static float* gC;
static int gM, gN, gK;
static void body() {{ {call}; }}

int main(int argc, char** argv) {{
    std::string dir = argv[1];
    int M = {M}, N = {N}, K = {K};
    auto A = load_floats((dir+"/A.bin").c_str(), (long)M*K);
    auto B = load_floats((dir+"/B.bin").c_str(), (long)K*N);
    {bias_load}
    std::vector<float> C((long)M*N, 0.0f);
    gA = A.data(); gB = B.data(); {"gBias = bias.data();" if with_bias else ""}
    gC = C.data(); gM = M; gN = N; gK = K;
    blockDim.x = {nthreads}; blockDim.y = 1; blockDim.z = 1;
    gridDim.x = {gx}; gridDim.y = {gy}; gridDim.z = 1;
    g_body = body;
    for (unsigned by = 0; by < {gy}; ++by)
      for (unsigned bx = 0; bx < {gx}; ++bx) {{
        blockIdx.x = bx; blockIdx.y = by; blockIdx.z = 0;
        launch_block({nthreads});
      }}
    save_floats((dir+"/C.bin").c_str(), C.data(), (long)M*N);
    return 0;
}}
"""
        _compile_and_run(main, wd)
        out = np.fromfile(os.path.join(wd, "C.bin"), dtype=np.float32)
    return out.reshape(M, N)


def simulate_im2col(source, kernel_name, x, *, KH, KW, OH, OW,
                    SH, SW, PH, PW, DH, DW):
    """Compile+run the generated im2col kernel for a single CHW image.

    Returns the column matrix of shape (C*KH*KW, OH*OW). The kernel has no
    shared memory or barriers, so each thread is independent; we still launch
    real threads to exercise the same 2D-grid indexing the GPU would use.
    """
    C, H, W = x.shape
    n_rows = C * KH * KW
    n_cols = OH * OW
    bx, byd = 16, 16
    gx = (n_cols + bx - 1) // bx
    gy = (n_rows + byd - 1) // byd

    with tempfile.TemporaryDirectory() as wd:
        _dump(wd, "x", x)
        main = f"""
{_SHIM}
{_prep_kernel(source)}

static const float* g_x; static float* g_cols;
static int gC,gH,gW,gKH,gKW,gOH,gOW,gSH,gSW,gPH,gPW,gDH,gDW;
static unsigned g_bx, g_byd;
static void body() {{
    // collapse the (bx,byd) 2D block into a flat thread id from threadIdx.x
    unsigned tx = threadIdx.x % g_bx;
    unsigned ty = threadIdx.x / g_bx;
    threadIdx.x = tx; threadIdx.y = ty;
    {kernel_name}(g_x, g_cols, gC,gH,gW,gKH,gKW,gOH,gOW,gSH,gSW,gPH,gPW,gDH,gDW);
}}
int main(int argc, char** argv) {{
    std::string dir = argv[1];
    int C={C},H={H},W={W},KH={KH},KW={KW},OH={OH},OW={OW};
    auto x = load_floats((dir+"/x.bin").c_str(), (long)C*H*W);
    std::vector<float> cols((long){n_rows}*{n_cols}, 0.0f);
    g_x=x.data(); g_cols=cols.data();
    gC=C;gH=H;gW=W;gKH=KH;gKW=KW;gOH=OH;gOW=OW;
    gSH={SH};gSW={SW};gPH={PH};gPW={PW};gDH={DH};gDW={DW};
    g_bx={bx}; g_byd={byd};
    blockDim.x={bx}; blockDim.y={byd}; blockDim.z=1;
    g_body = body;
    for (unsigned by=0; by<{gy}; ++by)
      for (unsigned bxi=0; bxi<{gx}; ++bxi) {{
        blockIdx.x=bxi; blockIdx.y=by; blockIdx.z=0;
        // im2col threads are independent; one real OS thread per CUDA thread
        // would be wasteful, so run the block's threads serially.
        for (unsigned t=0; t<{bx*byd}; ++t) {{
            threadIdx.x=t; threadIdx.y=0; threadIdx.z=0;
            unsigned tx=t%{bx}, ty=t/{bx};
            threadIdx.x=tx; threadIdx.y=ty;
            {kernel_name}(g_x,g_cols,gC,gH,gW,gKH,gKW,gOH,gOW,gSH,gSW,gPH,gPW,gDH,gDW);
        }}
      }}
    save_floats((dir+"/cols.bin").c_str(), cols.data(), (long){n_rows}*{n_cols});
    return 0;
}}
"""
        _compile_and_run(main, wd)
        cols = np.fromfile(os.path.join(wd, "cols.bin"), dtype=np.float32)
    return cols.reshape(n_rows, n_cols)


def simulate_kernel(source, kernel_name, *, inputs, output_sizes, scalar_args=(),
                    grid=(1, 1, 1), block=64):
    """Generic launcher: compile+run any generated kernel on the CPU sim.

    Calling convention (matches all new templates): the kernel takes each
    input pointer, then each output pointer, then each int scalar, in order.
    ``grid`` is (gx, gy, gz) — blocks run serially in z, y, x order; ``block``
    is the flat thread count (kernels derive 2D/3D thread coords from
    ``threadIdx.x`` themselves so the identical source runs under NVRTC with
    1-D blocks). Returns the output arrays (flat float32).
    """
    gx, gy, gz = grid
    n_in = len(inputs)
    n_out = len(output_sizes)

    with tempfile.TemporaryDirectory() as wd:
        for i, arr in enumerate(inputs):
            _dump(wd, f"in{i}", np.ascontiguousarray(arr, dtype=np.float32))
        loads = "\n    ".join(
            f'auto in{i} = load_floats((dir+"/in{i}.bin").c_str(), {int(np.asarray(inputs[i]).size)}L);'
            for i in range(n_in))
        out_decls = "\n    ".join(
            f"std::vector<float> out{j}({int(output_sizes[j])}L, 0.0f);"
            for j in range(n_out))
        glob = "\n".join([f"static const float* g_in{i};" for i in range(n_in)] +
                         [f"static float* g_out{j};" for j in range(n_out)])
        set_g = "\n    ".join([f"g_in{i} = in{i}.data();" for i in range(n_in)] +
                              [f"g_out{j} = out{j}.data();" for j in range(n_out)])
        args = ", ".join([f"g_in{i}" for i in range(n_in)] +
                         [f"g_out{j}" for j in range(n_out)] +
                         [str(int(s)) for s in scalar_args])
        saves = "\n    ".join(
            f'save_floats((dir+"/out{j}.bin").c_str(), out{j}.data(), {int(output_sizes[j])}L);'
            for j in range(n_out))
        main = f"""
{_SHIM}
{_prep_kernel(source)}

{glob}
static void body() {{ {kernel_name}({args}); }}

int main(int argc, char** argv) {{
    std::string dir = argv[1];
    {loads}
    {out_decls}
    {set_g}
    blockDim.x = {int(block)}; blockDim.y = 1; blockDim.z = 1;
    gridDim.x = {gx}; gridDim.y = {gy}; gridDim.z = {gz};
    g_body = body;
    for (unsigned bz = 0; bz < {gz}; ++bz)
      for (unsigned by = 0; by < {gy}; ++by)
        for (unsigned bx = 0; bx < {gx}; ++bx) {{
            blockIdx.x = bx; blockIdx.y = by; blockIdx.z = bz;
            launch_block({int(block)});
        }}
    {saves}
    return 0;
}}
"""
        _compile_and_run(main, wd)
        outs = [np.fromfile(os.path.join(wd, f"out{j}.bin"), dtype=np.float32)
                for j in range(n_out)]
    return outs


def simulate_elementwise(source, kernel_name, ext_inputs, input_arrays, n_outputs,
                         out_n=None):
    """Compile+run a generated elementwise kernel on CPU; return its outputs.

    ``out_n`` is the output element count (defaults to the first input's size;
    pass it explicitly when inputs are broadcast and smaller than the output).
    """
    N = int(out_n) if out_n is not None else int(input_arrays[0].size)
    block = 128
    grid = (N + block - 1) // block

    with tempfile.TemporaryDirectory() as wd:
        for i, arr in enumerate(input_arrays):
            _dump(wd, f"in{i}", arr)
        loads = "\n    ".join(
            f'auto in{i} = load_floats((dir+"/in{i}.bin").c_str(), '
            f'{int(np.asarray(input_arrays[i]).size)}L);'
            for i in range(len(input_arrays)))
        out_decls = "\n    ".join(
            f"std::vector<float> out{j}(N, 0.0f);" for j in range(n_outputs))
        set_g_in = "\n    ".join(f"g_in{i} = in{i}.data();" for i in range(len(input_arrays)))
        set_g_out = "\n    ".join(f"g_out{j} = out{j}.data();" for j in range(n_outputs))
        glob_in = "\n".join(f"static const float* g_in{i};" for i in range(len(input_arrays)))
        glob_out = "\n".join(f"static float* g_out{j};" for j in range(n_outputs))
        in_args = ", ".join(f"g_in{i}" for i in range(len(input_arrays)))
        out_args = ", ".join(f"g_out{j}" for j in range(n_outputs))
        saves = "\n    ".join(
            f'save_floats((dir+"/out{j}.bin").c_str(), out{j}.data(), N);'
            for j in range(n_outputs))

        main = f"""
{_SHIM}
{_prep_kernel(source)}

{glob_in}
{glob_out}
static int g_N;
static void body() {{ {kernel_name}({in_args}, {out_args}, g_N); }}

int main(int argc, char** argv) {{
    std::string dir = argv[1];
    int N = {N};
    {loads}
    {out_decls}
    {set_g_in}
    {set_g_out}
    g_N = N;
    blockDim.x = {block}; blockDim.y = 1; blockDim.z = 1;
    gridDim.x = {grid}; gridDim.y = 1; gridDim.z = 1;
    g_body = body;
    for (unsigned bx = 0; bx < {grid}; ++bx) {{
        blockIdx.x = bx; blockIdx.y = 0; blockIdx.z = 0;
        launch_block({block});
    }}
    {saves}
    return 0;
}}
"""
        _compile_and_run(main, wd)
        outs = [np.fromfile(os.path.join(wd, f"out{j}.bin"), dtype=np.float32)
                for j in range(n_outputs)]
    return outs
