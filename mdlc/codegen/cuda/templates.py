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


# ----- fused elementwise --------------------------------------------------

_BINARY = {"Add": "+", "Sub": "-", "Mul": "*", "Div": "/"}


def elementwise_kernel(
    name: str,
    node: Node,
    *,
    scalar_consts: Optional[dict[str, float]] = None,
) -> tuple[str, str, list[str]]:
    """String-generate a fused pointwise kernel from a ``FusedElementwise``.

    Each non-constant group input becomes a kernel pointer argument; each
    subgraph op becomes one line of register arithmetic; the chain's outputs
    are written once. No intermediate ever leaves a register. Returns
    ``(name, source, ordered_input_names)`` so the launcher knows arg order.

    Limitation (documented, not silent): inputs must be elementwise-compatible
    with the output (same element count); broadcastable constants must be
    scalar. The emitter checks this before choosing to fuse.
    """
    scalar_consts = scalar_consts or {}
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

    params = ", ".join(f"const float* __restrict__ {var(e)}_p" for e in ext_inputs)
    out_params = ", ".join(f"float* __restrict__ {var(o)}_p" for o in node.outputs)

    body = []
    for e in ext_inputs:
        body.append(f"    float {var(e)} = {var(e)}_p[i];")
    for n in sub:
        out = n.outputs[0]
        if n.op_type in _BINARY:
            a, b = var(n.inputs[0]), var(n.inputs[1])
            body.append(f"    float {var(out)} = {a} {_BINARY[n.op_type]} {b};")
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
