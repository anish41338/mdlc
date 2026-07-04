"""CUDA C kernel generators.

Each function returns a ``(kernel_name, source)`` pair of plain CUDA C, ready
for NVRTC. Three templates cover the workloads our passes produce:

  * ``gemm_kernel``        — tiled, register-blocked SGEMM with a fused
                             bias+activation epilogue.
  * ``im2col_kernel``      — lowers a conv's input patches to a column matrix so
                             the conv becomes an (implicit) GEMM.
  * ``elementwise_kernel`` — string-generates the body of a fused pointwise
                             chain, one element per thread, no intermediates.
"""

from __future__ import annotations

from typing import Optional

from mdlc.codegen.schedule import GemmSchedule
from mdlc.ir import Node


# ----- activation epilogue ------------------------------------------------

def activation_expr(activation: Optional[str], var: str, attrs: dict) -> str:
    """Return a CUDA expression applying ``activation`` to scalar ``var``."""
    if not activation or activation == "Identity":
        return var
    if activation == "Relu":
        return f"fmaxf({var}, 0.0f)"
    if activation == "Relu6":
        return f"fminf(fmaxf({var}, 0.0f), 6.0f)"
    if activation == "Clip":
        lo = attrs.get("clip_min")
        hi = attrs.get("clip_max")
        lo = "-CUDART_INF_F" if lo is None else f"{float(lo)}f"
        hi = "CUDART_INF_F" if hi is None else f"{float(hi)}f"
        return f"fminf(fmaxf({var}, {lo}), {hi})"
    if activation == "Sigmoid":
        return f"(1.0f / (1.0f + __expf(-({var}))))"
    if activation == "Tanh":
        return f"tanhf({var})"
    if activation == "LeakyRelu":
        a = float(attrs.get("alpha", 0.01))
        return f"(({var}) > 0.0f ? ({var}) : {a}f * ({var}))"
    if activation == "HardSigmoid":
        a = float(attrs.get("alpha", 0.2))
        b = float(attrs.get("beta", 0.5))
        return f"fminf(fmaxf({a}f * ({var}) + {b}f, 0.0f), 1.0f)"
    if activation == "HardSwish":
        return f"(({var}) * fminf(fmaxf(({var}) * 0.16666667f + 0.5f, 0.0f), 1.0f))"
    if activation == "Erf":
        return f"erff({var})"
    raise NotImplementedError(f"no CUDA epilogue for activation {activation!r}")


# ----- tiled GEMM ---------------------------------------------------------

def gemm_kernel(
    name: str,
    sched: GemmSchedule,
    *,
    with_bias: bool = False,
    bias_mode: str = "col",
    activation: Optional[str] = None,
    attrs: Optional[dict] = None,
) -> tuple[str, str]:
    """Generate a row-major SGEMM: C[MxN] = A[MxK] @ B[KxN] (+ bias) -> act.

    Schedule: a block computes a BM x BN output tile; each thread owns a
    TM x TN register micro-tile. A's shared slab is stored **transposed**
    ([BK][BM]) so the inner-product read across threads is conflict-free —
    consecutive threads hit consecutive banks instead of striding by BK.

    ``bias_mode`` selects the broadcast axis: ``"col"`` adds ``bias[col]`` (the
    Linear/Gemm case, bias per output feature) and ``"row"`` adds ``bias[row]``
    (the conv-as-GEMM case, bias per output channel).
    """
    attrs = attrs or {}
    BM, BN, BK, TM, TN = sched.BM, sched.BN, sched.BK, sched.TM, sched.TN
    bias_param = "const float* __restrict__ bias, " if with_bias else ""
    epi_var = "v"
    if with_bias:
        bias_idx = "row" if bias_mode == "row" else "col"
        bias_line = f"        v += bias[{bias_idx}];"
    else:
        bias_line = ""
    act_store = activation_expr(activation, epi_var, attrs)

    src = f"""
#include <math_constants.h>
extern "C" __global__ void {name}(
        const float* __restrict__ A,
        const float* __restrict__ B,
        {bias_param}float* __restrict__ C,
        int M, int N, int K) {{
    const int BM = {BM}, BN = {BN}, BK = {BK};
    const int TM = {TM}, TN = {TN};
    __shared__ float As[BK][BM];   // transposed: [k][m] avoids bank conflicts
    __shared__ float Bs[BK][BN];

    const int nThreadCol = BN / TN;
    const int tid = threadIdx.x;
    const int threadRow = tid / nThreadCol;
    const int threadCol = tid % nThreadCol;
    const int numThreads = (BM / TM) * (BN / TN);

    const int blockRow = blockIdx.y * BM;
    const int blockCol = blockIdx.x * BN;

    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;

    for (int k0 = 0; k0 < K; k0 += BK) {{
        // cooperatively stage A (BM x BK) transposed into As[BK][BM]
        for (int idx = tid; idx < BM * BK; idx += numThreads) {{
            int r = idx / BK, c = idx % BK;
            int gr = blockRow + r, gc = k0 + c;
            As[c][r] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
        }}
        // stage B (BK x BN) into Bs[BK][BN]
        for (int idx = tid; idx < BK * BN; idx += numThreads) {{
            int r = idx / BN, c = idx % BN;
            int gr = k0 + r, gc = blockCol + c;
            Bs[r][c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
        }}
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {{
            float aFrag[TM], bFrag[TN];
            #pragma unroll
            for (int i = 0; i < TM; ++i) aFrag[i] = As[kk][threadRow * TM + i];
            #pragma unroll
            for (int j = 0; j < TN; ++j) bFrag[j] = Bs[kk][threadCol * TN + j];
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j)
                    acc[i][j] += aFrag[i] * bFrag[j];
        }}
        __syncthreads();
    }}

    #pragma unroll
    for (int i = 0; i < TM; ++i) {{
        #pragma unroll
        for (int j = 0; j < TN; ++j) {{
            int row = blockRow + threadRow * TM + i;
            int col = blockCol + threadCol * TN + j;
            if (row < M && col < N) {{
                float v = acc[i][j];
{bias_line}
                C[row * N + col] = {act_store};
            }}
        }}
    }}
}}
""".strip()
    return name, src


# ----- im2col conv lowering -----------------------------------------------

def im2col_kernel(name: str) -> tuple[str, str]:
    """Generate the patch-gathering kernel for an implicit-GEMM conv.

    Produces a column matrix of shape (C*KH*KW, OH*OW) from a single NCHW image
    so the convolution reduces to weight[OC, C*KH*KW] @ cols. One thread per
    (column-row, output-pixel) entry. im2col trades memory (the lowered matrix
    is KH*KW larger than the input) for the ability to reuse the tuned GEMM;
    implicit GEMM would gather these patches on the fly inside the GEMM loop.
    """
    src = f"""
extern "C" __global__ void {name}(
        const float* __restrict__ x,   // [C, H, W] (single image)
        float* __restrict__ cols,      // [C*KH*KW, OH*OW]
        int C, int H, int W,
        int KH, int KW, int OH, int OW,
        int SH, int SW, int PH, int PW, int DH, int DW) {{
    int col = blockIdx.x * blockDim.x + threadIdx.x;   // output pixel index
    int row = blockIdx.y * blockDim.y + threadIdx.y;   // C*KH*KW index
    int n_cols = OH * OW;
    int n_rows = C * KH * KW;
    if (col >= n_cols || row >= n_rows) return;

    int kw = row % KW;
    int kh = (row / KW) % KH;
    int c  = row / (KH * KW);
    int ow = col % OW;
    int oh = col / OW;

    int ih = oh * SH - PH + kh * DH;
    int iw = ow * SW - PW + kw * DW;
    float v = 0.0f;
    if (ih >= 0 && ih < H && iw >= 0 && iw < W)
        v = x[(c * H + ih) * W + iw];
    cols[row * n_cols + col] = v;
}}
""".strip()
    return name, src


# ----- direct depthwise conv ------------------------------------------------

def depthwise_conv_kernel(
    name: str,
    *,
    C: int, mult: int, H: int, W: int, OH: int, OW: int,
    KH: int, KW: int, SH: int, SW: int, PH: int, PW: int,
    DH: int = 1, DW: int = 1,
    tile_h: int = 8, tile_w: int = 8,
    with_bias: bool = False,
    activation: Optional[str] = None,
    attrs: Optional[dict] = None,
    direct_load: bool = False,
) -> tuple[str, str]:
    """Direct depthwise convolution — no im2col (which would multiply memory
    traffic by KH*KW for zero GEMM benefit since each output channel reads
    exactly one input channel).

    Mapping: ``blockIdx.z = n * C_out + c_out`` (batching is native);
    blockIdx.x/y tile the output plane in ``tile_w x tile_h`` patches; one
    thread per output pixel. The input tile plus halo
    ((TILE-1)*stride + (K-1)*dilation + 1 per dim) and the KxKW filter are
    staged in shared memory (all threads in the block share the channel);
    ``direct_load=True`` is the tuner variant that reads global memory
    directly. Every conv parameter is baked as a compile-time constant — the
    kernel is shape-specialized, K loops fully unroll, and the block is flat
    1-D (thread x/y derived from threadIdx.x) so the identical source runs on
    the cpu-sim and under NVRTC.

    Channel multiplier: input channel = c_out / mult; weight is
    [C*mult, 1, KH, KW]. Epilogue: optional per-channel bias (Conv+BN fold
    lands there) then activation.
    """
    attrs = attrs or {}
    c_out = C * mult
    smh = (tile_h - 1) * SH + (KH - 1) * DH + 1
    smw = (tile_w - 1) * SW + (KW - 1) * DW + 1
    nthreads = tile_h * tile_w
    bias_param = "const float* __restrict__ bias, " if with_bias else ""
    bias_line = "        v += bias[co];" if with_bias else ""
    act = activation_expr(activation, "v", attrs)

    if direct_load:
        load_expr = """
        float acc = 0.0f;
        #pragma unroll
        for (int kh = 0; kh < KH; ++kh) {
            #pragma unroll
            for (int kw = 0; kw < KW; ++kw) {
                int ih = oh * SH - PH + kh * DH;
                int iw = ow * SW - PW + kw * DW;
                float xv = (ih >= 0 && ih < H && iw >= 0 && iw < W)
                               ? xin[ih * W + iw] : 0.0f;
                acc += xv * w[co * KH * KW + kh * KW + kw];
            }
        }"""
        smem_decl = ""
        stage = ""
    else:
        load_expr = """
        float acc = 0.0f;
        #pragma unroll
        for (int kh = 0; kh < KH; ++kh) {
            #pragma unroll
            for (int kw = 0; kw < KW; ++kw) {
                acc += tile[(ty * SH + kh * DH) * SMW + (tx * SW + kw * DW)]
                     * filt[kh * KW + kw];
            }
        }"""
        smem_decl = """
    __shared__ float tile[SMH * SMW];
    __shared__ float filt[KH * KW];"""
        stage = """
    for (int i = tid; i < SMH * SMW; i += NT) {
        int r = i / SMW, c = i % SMW;
        int ih = ih0 + r, iw = iw0 + c;
        tile[i] = (ih >= 0 && ih < H && iw >= 0 && iw < W) ? xin[ih * W + iw]
                                                           : 0.0f;
    }
    for (int i = tid; i < KH * KW; i += NT)
        filt[i] = w[co * KH * KW + i];
    __syncthreads();"""

    src = f"""
#include <math_constants.h>
extern "C" __global__ void {name}(
        const float* __restrict__ x,     // [N, {C}, {H}, {W}]
        const float* __restrict__ w,     // [{c_out}, 1, {KH}, {KW}]
        {bias_param}float* __restrict__ y) {{   // [N, {c_out}, {OH}, {OW}]
    const int C_OUT = {c_out}, MULT = {mult};
    const int H = {H}, W = {W}, OH = {OH}, OW = {OW};
    const int KH = {KH}, KW = {KW}, SH = {SH}, SW = {SW};
    const int PH = {PH}, PW = {PW}, DH = {DH}, DW = {DW};
    const int TILE_H = {tile_h}, TILE_W = {tile_w}, NT = {nthreads};
    const int SMH = {smh}, SMW = {smw};
    (void)DH; (void)DW; (void)PH; (void)PW; (void)SMH; (void)SMW; (void)NT;

    const int nc = blockIdx.z;             // n * C_out + c_out
    const int n  = nc / C_OUT;
    const int co = nc % C_OUT;
    const int ci = co / MULT;

    const int tid = threadIdx.x;           // flat {tile_h}x{tile_w} block
    const int tx = tid % TILE_W;
    const int ty = tid / TILE_W;

    const int oh0 = blockIdx.y * TILE_H;
    const int ow0 = blockIdx.x * TILE_W;
    const int ih0 = oh0 * SH - PH;
    const int iw0 = ow0 * SW - PW;
    (void)ih0; (void)iw0;

    const float* xin = x + (n * {C} + ci) * H * W;
{smem_decl}
{stage}

    const int oh = oh0 + ty;
    const int ow = ow0 + tx;
    if (oh < OH && ow < OW) {{
{load_expr}
        float v = acc;
{bias_line}
        y[(n * C_OUT + co) * OH * OW + oh * OW + ow] = {act};
    }}
}}
""".strip()
    return name, src


# ----- spatial mean reduction (GlobalAveragePool / ReduceMean over H,W) -----

def reduce_mean_hw_kernel(name: str, *, HW: int, block: int = 128) -> tuple[str, str]:
    """Mean over one (n, c) plane per block: grid-stride partial sums per
    thread, then a fixed shared-memory tree combine.

    Determinism: each thread accumulates a fixed stride-slice in a fixed
    order and the tree combines in a fixed order — no atomics, so the result
    is bit-identical run to run (Prime Directive: reductions use a
    deterministic in-block tree). Warp shuffles would shave the smem round
    trip on a real GPU but the cpu-sim shim has no warp-lockstep to emulate
    them; the smem tree is the portable, verifiable choice.
    """
    assert block & (block - 1) == 0, "block must be a power of two"
    src = f"""
extern "C" __global__ void {name}(
        const float* __restrict__ x,   // [N*C, {HW}]
        float* __restrict__ y) {{      // [N*C]
    const int HW = {HW}, BLOCK = {block};
    const int nc = blockIdx.x;
    const float* xin = x + nc * HW;
    __shared__ float partial[BLOCK];
    float acc = 0.0f;
    for (int i = threadIdx.x; i < HW; i += BLOCK)
        acc += xin[i];
    partial[threadIdx.x] = acc;
    __syncthreads();
    for (int s = BLOCK / 2; s > 0; s >>= 1) {{
        if (threadIdx.x < s) partial[threadIdx.x] += partial[threadIdx.x + s];
        __syncthreads();
    }}
    if (threadIdx.x == 0) y[nc] = partial[0] * (1.0f / HW);
}}
""".strip()
    return name, src


# ----- max pooling ----------------------------------------------------------

def maxpool_kernel(name: str, *, H: int, W: int, OH: int, OW: int,
                   KH: int, KW: int, SH: int, SW: int, PH: int, PW: int,
                   DH: int = 1, DW: int = 1, total: int,
                   block: int = 128) -> tuple[str, str]:
    """One thread per output element, flat over N*C*OH*OW. Out-of-bounds
    window taps are skipped (ONNX MaxPool padding semantics: -inf identity)."""
    src = f"""
#include <math_constants.h>
extern "C" __global__ void {name}(
        const float* __restrict__ x,
        float* __restrict__ y) {{
    const int H = {H}, W = {W}, OH = {OH}, OW = {OW};
    const int KH = {KH}, KW = {KW}, SH = {SH}, SW = {SW};
    const int PH = {PH}, PW = {PW}, DH = {DH}, DW = {DW};
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {total}) return;
    int ow = i % OW;
    int oh = (i / OW) % OH;
    int nc = i / (OH * OW);
    const float* xin = x + nc * H * W;
    float m = -CUDART_INF_F;
    #pragma unroll
    for (int kh = 0; kh < KH; ++kh) {{
        #pragma unroll
        for (int kw = 0; kw < KW; ++kw) {{
            int ih = oh * SH - PH + kh * DH;
            int iw = ow * SW - PW + kw * DW;
            if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                m = fmaxf(m, xin[ih * W + iw]);
        }}
    }}
    y[i] = m;
}}
""".strip()
    return name, src


# ----- fused elementwise --------------------------------------------------

_BINARY = {"Add": "+", "Sub": "-", "Mul": "*", "Div": "/"}


def broadcast_index_expr(in_shape, out_shape, idx: str = "i") -> str:
    """C expression for an input's flat index given the output's flat index.

    NumPy/ONNX multidirectional broadcasting as baked strides: the input shape
    is right-aligned against the output shape and every size-1 input dim gets
    stride 0. Same-shape inputs collapse to ``i``; fully-broadcast (scalar)
    inputs collapse to ``0``. This one mechanism covers channel-broadcast
    (N,C,1,1)x(N,C,H,W) SE gates, bias-add patterns, and runtime scalars.
    """
    r = len(out_shape)
    s = (1,) * (r - len(in_shape)) + tuple(int(d) for d in in_shape)
    for d in range(r):
        if s[d] not in (1, int(out_shape[d])):
            raise ValueError(f"shape {in_shape} not broadcastable to {out_shape}")
    # strides over the input's own (right-aligned) layout, zeroed on broadcast
    istr = [0] * r
    acc = 1
    for d in range(r - 1, -1, -1):
        istr[d] = 0 if s[d] == 1 else acc
        acc *= s[d]
    # output strides
    ostr = [0] * r
    acc = 1
    for d in range(r - 1, -1, -1):
        ostr[d] = acc
        acc *= int(out_shape[d])
    if all(istr[d] in (0, ostr[d]) for d in range(r)) and \
            all(istr[d] == ostr[d] for d in range(r) if s[d] != 1) and \
            tuple(s) == tuple(int(d) for d in out_shape):
        return idx
    terms = [f"(({idx} / {ostr[d]}) % {int(out_shape[d])}) * {istr[d]}"
             for d in range(r) if istr[d] != 0]
    return "(" + " + ".join(terms) + ")" if terms else "0"


def elementwise_kernel(
    name: str,
    node: Node,
    *,
    scalar_consts: Optional[dict[str, float]] = None,
    input_shapes: Optional[dict[str, tuple]] = None,
    out_shape: Optional[tuple] = None,
) -> tuple[str, str, list[str]]:
    """String-generate a fused pointwise kernel from a ``FusedElementwise``.

    Each non-constant group input becomes a kernel pointer argument read
    through its own baked broadcast-index expression (stride 0 on broadcast
    dims — see ``broadcast_index_expr``); each subgraph op becomes one line of
    register arithmetic; the chain's outputs are written once at the output
    index. No intermediate ever leaves a register. Values on a broadcast
    branch (e.g. an SE gate's (N,C,1,1) sigmoid) are recomputed per output
    element — pure functions, so legality is unaffected and the redundant
    flops are trivial next to the memory traffic saved.

    Returns ``(name, source, ordered_input_names)`` so the launcher knows arg
    order. Without ``input_shapes``/``out_shape`` every input is assumed
    output-shaped (the pre-broadcast behavior).
    """
    scalar_consts = scalar_consts or {}
    input_shapes = input_shapes or {}
    sub: list[Node] = node.attrs["subgraph"]

    internal = {o for n in sub for o in n.outputs if o}
    ext_inputs: list[str] = []
    seen = set()
    for n in sub:
        for e in n.real_inputs():
            if e not in internal and e not in scalar_consts and e not in seen:
                seen.add(e)
                ext_inputs.append(e)

    def var(edge: str) -> str:
        if edge in scalar_consts:
            return f"{float(scalar_consts[edge])}f"
        return "t_" + edge.replace(".", "_").replace("/", "_").replace(":", "_")

    def load_index(edge: str) -> str:
        if out_shape is None or edge not in input_shapes:
            return "i"
        return broadcast_index_expr(input_shapes[edge], out_shape)

    params = ", ".join(f"const float* __restrict__ {var(e)}_p" for e in ext_inputs)
    out_params = ", ".join(f"float* __restrict__ {var(o)}_p" for o in node.outputs)

    body = []
    for e in ext_inputs:
        body.append(f"    float {var(e)} = {var(e)}_p[{load_index(e)}];")
    for n in sub:
        out = n.outputs[0]
        if n.op_type in _BINARY:
            a, b = var(n.inputs[0]), var(n.inputs[1])
            body.append(f"    float {var(out)} = {a} {_BINARY[n.op_type]} {b};")
        elif n.op_type == "Clip":
            # opset>=11 carries bounds as inputs; they arrive here as resolved
            # scalar consts or loaded registers (a broadcast-index of 0).
            x = var(n.inputs[0])
            lo = (var(n.inputs[1]) if len(n.inputs) > 1 and n.inputs[1]
                  else _attr_f(n, "min", "-CUDART_INF_F"))
            hi = (var(n.inputs[2]) if len(n.inputs) > 2 and n.inputs[2]
                  else _attr_f(n, "max", "CUDART_INF_F"))
            body.append(f"    float {var(out)} = fminf(fmaxf({x}, {lo}), {hi});")
        else:
            x = var(n.inputs[0])
            expr = activation_expr(n.op_type, x, n.attrs)
            body.append(f"    float {var(out)} = {expr};")
    for o in node.outputs:
        body.append(f"    {var(o)}_p[i] = {var(o)};")

    src = f"""
#include <math_constants.h>
extern "C" __global__ void {name}({params}, {out_params}, int N) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
{chr(10).join(body)}
}}
""".strip()
    return name, src, ext_inputs


def _attr_f(n: Node, key: str, default: str) -> str:
    v = n.attrs.get(key)
    return default if v is None else f"{float(v)}f"
