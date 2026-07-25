"""Pure-NumPy implementations of the tensor ops we support.

These are the *semantics* of the IR: the reference executor and the
correctness harness rely on them, and codegen is validated against them. Keep
them simple and obviously correct rather than fast.
"""

from __future__ import annotations

import math

import numpy as np


# ----- shape/window helpers ----------------------------------------------

def _resolve_pads(auto_pad: str, pads, in_hw, k_hw, s_hw, d_hw):
    """Return ((pt, pb), (pl, pr)) for 2D, honoring ONNX auto_pad."""
    ih, iw = in_hw
    kh, kw = k_hw
    sh, sw = s_hw
    dh, dw = d_hw
    if auto_pad in ("SAME_UPPER", "SAME_LOWER"):
        out_h = -(-ih // sh)  # ceil
        out_w = -(-iw // sw)
        pad_h = max(0, (out_h - 1) * sh + (kh - 1) * dh + 1 - ih)
        pad_w = max(0, (out_w - 1) * sw + (kw - 1) * dw + 1 - iw)
        if auto_pad == "SAME_UPPER":
            return (pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2)
        return (pad_h - pad_h // 2, pad_h // 2), (pad_w - pad_w // 2, pad_w // 2)
    if pads is None:
        return (0, 0), (0, 0)
    # ONNX layout: [x1_begin, x2_begin, x1_end, x2_end]
    return (pads[0], pads[2]), (pads[1], pads[3])


def im2col(x, kh, kw, sh, sw, dh, dw, pads):
    """N,C,H,W -> columns (N, C*kh*kw, out_h*out_w). Mirrors the CUDA path."""
    n, c, h, w = x.shape
    (pt, pb), (pl, pr) = pads
    xp = np.pad(x, ((0, 0), (0, 0), (pt, pb), (pl, pr)))
    out_h = (h + pt + pb - (dh * (kh - 1) + 1)) // sh + 1
    out_w = (w + pl + pr - (dw * (kw - 1) + 1)) // sw + 1
    cols = np.empty((n, c * kh * kw, out_h * out_w), dtype=x.dtype)
    idx = 0
    for ci in range(c):
        for ky in range(kh):
            for kx in range(kw):
                patch = xp[:, ci, ky * dh: ky * dh + sh * out_h: sh,
                                 kx * dw: kx * dw + sw * out_w: sw]
                cols[:, idx, :] = patch.reshape(n, out_h * out_w)
                idx += 1
    return cols, out_h, out_w


# ----- ops ----------------------------------------------------------------

def conv(x, w, b, *, strides, pads, dilations, group, auto_pad="NOTSET"):
    n, c, ih, iw = x.shape
    oc, icg, kh, kw = w.shape
    sh, sw = strides
    dh, dw = dilations
    p = _resolve_pads(auto_pad, pads, (ih, iw), (kh, kw), (sh, sw), (dh, dw))

    if group == 1:
        cols, out_h, out_w = im2col(x, kh, kw, sh, sw, dh, dw, p)
        wm = w.reshape(oc, -1)                      # (oc, c*kh*kw)
        out = np.einsum("ok,nkp->nop", wm, cols)    # (n, oc, out_h*out_w)
        out = out.reshape(n, oc, out_h, out_w)
    else:
        # grouped / depthwise: run each group independently
        outs = []
        ocg = oc // group
        for g in range(group):
            xs = x[:, g * icg:(g + 1) * icg]
            ws = w[g * ocg:(g + 1) * ocg]
            cols, out_h, out_w = im2col(xs, kh, kw, sh, sw, dh, dw, p)
            wm = ws.reshape(ocg, -1)
            outs.append(np.einsum("ok,nkp->nop", wm, cols).reshape(n, ocg, out_h, out_w))
        out = np.concatenate(outs, axis=1)
    if b is not None:
        out = out + b.reshape(1, -1, 1, 1)
    return out


def gemm(a, b, c, *, alpha=1.0, beta=1.0, transA=0, transB=0):
    if transA:
        a = a.T
    if transB:
        b = b.T
    y = alpha * (a @ b)
    if c is not None:
        y = y + beta * c
    return y


def matmul(a, b):
    return a @ b


def batchnorm(x, scale, bias, mean, var, *, epsilon=1e-5):
    shape = [1] * x.ndim
    shape[1] = x.shape[1]
    s = scale.reshape(shape)
    b = bias.reshape(shape)
    m = mean.reshape(shape)
    v = var.reshape(shape)
    return (x - m) / np.sqrt(v + epsilon) * s + b


def maxpool(x, *, kernel, strides, pads, dilations=(1, 1), auto_pad="NOTSET", ceil_mode=0):
    n, c, ih, iw = x.shape
    kh, kw = kernel
    sh, sw = strides
    dh, dw = dilations
    (pt, pb), (pl, pr) = _resolve_pads(auto_pad, pads, (ih, iw), (kh, kw), (sh, sw), (dh, dw))
    xp = np.pad(x, ((0, 0), (0, 0), (pt, pb), (pl, pr)), constant_values=-np.inf)
    div = (lambda a, bb: -(-a // bb)) if ceil_mode else (lambda a, bb: a // bb)
    out_h = div(ih + pt + pb - (dh * (kh - 1) + 1), sh) + 1
    out_w = div(iw + pl + pr - (dw * (kw - 1) + 1), sw) + 1
    out = np.full((n, c, out_h, out_w), -np.inf, dtype=x.dtype)
    for ky in range(kh):
        for kx in range(kw):
            patch = xp[:, :, ky * dh: ky * dh + sh * out_h: sh,
                             kx * dw: kx * dw + sw * out_w: sw]
            out = np.maximum(out, patch[:, :, :out_h, :out_w])
    return out


def averagepool(x, *, kernel, strides, pads, auto_pad="NOTSET",
                count_include_pad=0, ceil_mode=0):
    n, c, ih, iw = x.shape
    kh, kw = kernel
    sh, sw = strides
    (pt, pb), (pl, pr) = _resolve_pads(auto_pad, pads, (ih, iw), (kh, kw), (sh, sw), (1, 1))
    xp = np.pad(x, ((0, 0), (0, 0), (pt, pb), (pl, pr)))
    div = (lambda a, bb: -(-a // bb)) if ceil_mode else (lambda a, bb: a // bb)
    out_h = div(ih + pt + pb - kh, sh) + 1
    out_w = div(iw + pl + pr - kw, sw) + 1
    acc = np.zeros((n, c, out_h, out_w), dtype=np.float64)
    cnt = np.zeros((n, c, out_h, out_w), dtype=np.float64)
    ones = np.pad(np.ones((n, c, ih, iw)), ((0, 0), (0, 0), (pt, pb), (pl, pr)))
    for ky in range(kh):
        for kx in range(kw):
            acc += xp[:, :, ky: ky + sh * out_h: sh, kx: kx + sw * out_w: sw][:, :, :out_h, :out_w]
            cnt += ones[:, :, ky: ky + sh * out_h: sh, kx: kx + sw * out_w: sw][:, :, :out_h, :out_w]
    denom = (kh * kw) if count_include_pad else np.maximum(cnt, 1)
    return (acc / denom).astype(x.dtype)


def global_average_pool(x):
    return x.mean(axis=tuple(range(2, x.ndim)), keepdims=True).astype(x.dtype)


def reduce_mean(x, axes=None, keepdims=1):
    if axes is None:
        ax = None
    else:
        ax = tuple(int(a) % x.ndim for a in np.atleast_1d(np.asarray(axes)))
    return x.mean(axis=ax, keepdims=bool(keepdims)).astype(x.dtype)


# ----- elementwise / activations -----------------------------------------

def relu(x):
    return np.maximum(x, 0)


def clip(x, lo=None, hi=None):
    if lo is None:
        lo = -np.inf
    if hi is None:
        hi = np.inf
    return np.clip(x, lo, hi)


def leaky_relu(x, alpha=0.01):
    return np.where(x >= 0, x, alpha * x)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


def hard_sigmoid(x, alpha=0.2, beta=0.5):
    return np.clip(alpha * x + beta, 0.0, 1.0)


def hard_swish(x):
    # ONNX HardSwish: x * max(0, min(1, x/6 + 0.5))
    return x * np.clip(x / 6.0 + 0.5, 0.0, 1.0)


# Exact (double-precision libm) erf, vectorized. NumPy has no erf and scipy is
# not a core dependency; math.erf per element is slow but this is the *oracle*,
# where exactness beats speed. GELU exports decompose to Erf at opset 17.
_erf_ufunc = np.frompyfunc(math.erf, 1, 1)


def erf(x):
    return _erf_ufunc(np.asarray(x, dtype=np.float64)).astype(np.asarray(x).dtype)


def softmax(x, axis=-1):
    z = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


# Binary elementwise (with NumPy broadcasting == ONNX multidirectional).
def add(a, b):
    return a + b


def sub(a, b):
    return a - b


def mul(a, b):
    return a * b


def div(a, b):
    return a / b


# ----- quantization (ONNX QDQ semantics, exact) ----------------------------
#
# Rounding is round-half-to-even everywhere (np.rint == ONNX QuantizeLinear ==
# CUDA __float2int_rn/rintf) — get this wrong and int8 parity vs ORT fails on
# exact .5 ties.

def _qrange(zp_dtype) -> tuple[int, int]:
    return (0, 255) if np.dtype(zp_dtype) == np.uint8 else (-128, 127)


def _per_axis(arr, x_ndim, axis):
    """Reshape a 1-D per-axis scale/zp for broadcasting along ``axis``."""
    a = np.asarray(arr)
    if a.ndim == 1 and a.size > 1:
        shape = [1] * x_ndim
        shape[axis] = a.size
        return a.reshape(shape)
    return a


def quantize_linear(x, scale, zp, axis=1):
    """fp32 -> int8/uint8 codes: clamp(rint(x/scale) + zp)."""
    zp = np.asarray(zp)
    s = _per_axis(np.asarray(scale, np.float32), x.ndim, axis)
    z = _per_axis(zp, x.ndim, axis)
    qmin, qmax = _qrange(zp.dtype)
    q = np.rint(np.asarray(x, np.float32) / s).astype(np.int64) + z.astype(np.int64)
    return np.clip(q, qmin, qmax).astype(zp.dtype)


def dequantize_linear(q, scale, zp, axis=1):
    """int codes -> fp32: (q - zp) * scale."""
    s = _per_axis(np.asarray(scale, np.float32), np.asarray(q).ndim, axis)
    z = _per_axis(np.asarray(zp), np.asarray(q).ndim, axis)
    return (np.asarray(q).astype(np.int32) - z.astype(np.int32)).astype(np.float32) * s


def _requant(acc_i32, multiplier, y_zp, zp_dtype, act_qmin=None, act_qmax=None):
    """int32 accumulator -> output codes with a per-channel fp32 multiplier.

    q = clamp(rint(acc * m) + zp_y). Activations folded into the quantized
    domain arrive as clamp bounds (ReLU = floor at zp_y): the kernel does the
    same, so this reference is bit-exact against the emitted epilogue."""
    qmin, qmax = _qrange(zp_dtype)
    if act_qmin is not None:
        qmin = max(qmin, int(act_qmin))
    if act_qmax is not None:
        qmax = min(qmax, int(act_qmax))
    v = acc_i32.astype(np.float32) * multiplier.astype(np.float32)
    q = np.rint(v).astype(np.int64) + int(y_zp)
    return np.clip(q, qmin, qmax).astype(zp_dtype)


def qconv(xq, wq, bias_i32, *, x_scale, x_zp, w_scales, y_scale, y_zp,
          strides, pads, dilations, group, act_qmin=None, act_qmax=None,
          out_dtype=np.int8):
    """Quantized conv: int32-exact accumulation over centered int inputs, then
    per-channel requantization. Weight zero-points are 0 (per-channel
    symmetric — the AIMET W8A8 convention this path targets)."""
    xi = xq.astype(np.int64) - int(x_zp)
    wi = wq.astype(np.int64)
    acc = conv(xi, wi, None, strides=strides, pads=pads,
               dilations=dilations, group=group)
    if bias_i32 is not None:
        acc = acc + np.asarray(bias_i32, np.int64).reshape(1, -1, 1, 1)
    m = (np.float32(x_scale) * np.asarray(w_scales, np.float32)
         / np.float32(y_scale)).reshape(1, -1, 1, 1)
    return _requant(acc.astype(np.int32), m, y_zp, out_dtype, act_qmin, act_qmax)


def qgemm(xq, wq, bias_i32, *, x_scale, x_zp, w_scales, y_scale, y_zp,
          act_qmin=None, act_qmax=None, out_dtype=np.int8):
    """Quantized Gemm/Linear: y[M,N] over int8 x[M,K] and per-column-quantized
    w[K,N] (weights already transposed to K,N form by the pass)."""
    xi = xq.astype(np.int64) - int(x_zp)
    acc = xi @ wq.astype(np.int64)
    if bias_i32 is not None:
        acc = acc + np.asarray(bias_i32, np.int64).reshape(1, -1)
    m = (np.float32(x_scale) * np.asarray(w_scales, np.float32)
         / np.float32(y_scale)).reshape(1, -1)
    return _requant(acc.astype(np.int32), m, y_zp, out_dtype, act_qmin, act_qmax)


def qadd(aq, bq, *, a_scale, a_zp, b_scale, b_zp, y_scale, y_zp,
         act_qmin=None, act_qmax=None, out_dtype=np.int8):
    """Quantized residual Add: dequantize both sides to fp32 in-register,
    add, requantize — accurate and simple (one fused kernel on device)."""
    af = (aq.astype(np.int32) - int(a_zp)).astype(np.float32) * np.float32(a_scale)
    bf = (bq.astype(np.int32) - int(b_zp)).astype(np.float32) * np.float32(b_scale)
    v = (af + bf) / np.float32(y_scale)
    qmin, qmax = _qrange(out_dtype)
    if act_qmin is not None:
        qmin = max(qmin, int(act_qmin))
    if act_qmax is not None:
        qmax = min(qmax, int(act_qmax))
    q = np.rint(v).astype(np.int64) + int(y_zp)
    return np.clip(q, qmin, qmax).astype(out_dtype)


# ----- tensor reshaping ---------------------------------------------------

def flatten(x, axis=1):
    if axis < 0:
        axis += x.ndim
    outer = int(np.prod(x.shape[:axis])) if axis > 0 else 1
    return x.reshape(outer, -1)


def reshape(x, shape):
    shape = list(int(s) for s in np.asarray(shape).ravel())
    # ONNX: 0 means "copy from input", -1 means "infer".
    out = []
    for i, s in enumerate(shape):
        out.append(x.shape[i] if s == 0 else s)
    return x.reshape(out)


def transpose(x, perm=None):
    return np.transpose(x, perm)


def concat(arrays, axis=0):
    return np.concatenate(arrays, axis=axis)


def squeeze(x, axes=None):
    if axes is None:
        return np.squeeze(x)
    return np.squeeze(x, axis=tuple(int(a) for a in np.asarray(axes).ravel()))


def unsqueeze(x, axes):
    for a in sorted(int(v) for v in np.asarray(axes).ravel()):
        x = np.expand_dims(x, a)
    return x


def pad(x, pads, value=0.0, mode="constant"):
    pads = [int(p) for p in np.asarray(pads).ravel()]
    half = len(pads) // 2
    width = [(pads[i], pads[i + half]) for i in range(half)]
    if mode == "constant":
        return np.pad(x, width, mode="constant", constant_values=value)
    return np.pad(x, width, mode={"reflect": "reflect", "edge": "edge"}.get(mode, "constant"))
