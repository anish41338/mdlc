"""QDQ recognition: rewrite DequantizeLinear→Op→QuantizeLinear sandwiches into
fused quantized ops that codegen lowers to DP4A kernels.

Input form: ONNX QDQ as exported by AIMET (or ``tools/encodings_to_qdq.py``):
per-channel symmetric int8 weights (axis 0, zero-point 0), per-tensor
activations (int8 or uint8). Patterns handled:

  DQ(x), DQ(w) → Conv           → [Relu|Clip] → Q(y)   ⇒  QConvFused
  DQ(x), DQ(w) → Gemm(transB)   → [Relu|Clip] → Q(y)   ⇒  QGemmFused
  DQ(a), DQ(b) → Add            → [Relu|Clip] → Q(y)   ⇒  QAddFused

The fused node consumes/produces *integer code* edges directly; the activation
folds into the requant clamp (in the quantized domain ReLU is a floor at
zp_y; Clip(lo,hi) becomes [rint(lo/s)+zp, rint(hi/s)+zp]). The fp32 bias is
quantized here: ``b_i32[c] = rint(b / (s_x·s_w[c]))`` — round-half-to-even,
matching ONNX QuantizeLinear semantics.

Anything not matched stays fp32 between explicit Q/DQ boundary kernels — a
mixed-precision graph is acceptable and honest; the compile report shows the
int8 coverage fraction. This pass must run *before* constant folding, which
would otherwise fold DQ(w_int8) into a plain fp32 weight and erase the
pattern.

Weight forms supported: int8 initializer + DQ (AIMET default), and fp32
initializer + Q + DQ (the pass folds the Q itself, round-half-even).
"""

from __future__ import annotations

import numpy as np

from mdlc.ir import DType, Graph, Node, TensorInfo
from mdlc.passes.base import Pass
from mdlc.runtime import ops

_ACTS = {"Relu", "Clip"}


def graph_is_qdq(graph: Graph) -> bool:
    return any(n.op_type in ("QuantizeLinear", "DequantizeLinear")
               for n in graph.nodes)


def _scalar(graph: Graph, name: str, default=None):
    if not name:
        return default
    arr = graph.initializers.get(name)
    if arr is None:
        return None
    return np.asarray(arr).reshape(-1)


class QdqLowering(Pass):
    name = "qdq_lowering"

    def run(self, graph: Graph) -> bool:
        changed = False
        # iterate until no more sandwiches match (chains rewrite front-to-back)
        while self._rewrite_one(graph):
            changed = True
        if changed:
            graph.reorder_topologically()
        return changed

    # ------------------------------------------------------------------

    def _rewrite_one(self, graph: Graph) -> bool:
        producers = graph.producers()
        consumers = graph.consumers()
        use = graph.use_count()

        for node in graph.nodes:
            if node.op_type == "Conv" and self._try_conv_gemm(
                    graph, node, producers, consumers, use, kind="conv"):
                return True
            if node.op_type == "Gemm" and self._try_conv_gemm(
                    graph, node, producers, consumers, use, kind="gemm"):
                return True
            if node.op_type == "Add" and self._try_add(
                    graph, node, producers, consumers, use):
                return True
        return False

    # -- helpers ---------------------------------------------------------

    def _dq_of(self, graph: Graph, producers, edge: str):
        """If ``edge`` is produced by DequantizeLinear, return (dq_node,
        q_edge, scale_arr, zp_arr, axis)."""
        dq = producers.get(edge)
        if dq is None or dq.op_type != "DequantizeLinear":
            return None
        scale = _scalar(graph, dq.inputs[1])
        zp_name = dq.inputs[2] if len(dq.inputs) > 2 else ""
        zp = _scalar(graph, zp_name, default=np.zeros(1, np.int8))
        if scale is None or zp is None:
            return None                      # runtime scales: not supported
        return dq, dq.inputs[0], scale, zp, dq.attrs.get("axis", 1)

    def _weight_codes(self, graph: Graph, producers, edge: str):
        """Resolve a weight edge to (int8_codes, per_channel_scales). Handles
        int8-init + DQ and fp32-init + Q + DQ."""
        hit = self._dq_of(graph, producers, edge)
        if hit is None:
            return None
        dq, q_edge, scale, zp, axis = hit
        if not np.all(np.asarray(zp) == 0):
            return None                      # weights must be symmetric
        if graph.is_constant(q_edge):
            codes = graph.initializers[q_edge]
            if codes.dtype != np.int8:
                return None
            return dq, codes, scale.astype(np.float32)
        qn = producers.get(q_edge)
        if qn is not None and qn.op_type == "QuantizeLinear" and \
                graph.is_constant(qn.inputs[0]):
            w_fp = graph.initializers[qn.inputs[0]]
            codes = ops.quantize_linear(
                w_fp, scale, np.zeros(scale.shape, np.int8),
                qn.attrs.get("axis", 0))
            return dq, codes, scale.astype(np.float32)
        return None

    def _tail(self, graph: Graph, consumers, out_edge: str):
        """Follow out_edge through an optional activation to QuantizeLinear.
        Returns (act_node|None, q_node) or None. Every hop must be
        single-consumer (fusion legality)."""
        cs = consumers.get(out_edge, [])
        if len(cs) != 1 or out_edge in graph.outputs:
            return None
        act = None
        node = cs[0]
        if node.op_type in _ACTS:
            act = node
            cs2 = consumers.get(act.outputs[0], [])
            if len(cs2) != 1 or act.outputs[0] in graph.outputs:
                return None
            node = cs2[0]
        if node.op_type != "QuantizeLinear":
            return None
        return act, node

    def _act_bounds(self, graph: Graph, act, y_scale: float, y_zp: int):
        """Fold Relu/Clip into quantized clamp bounds."""
        if act is None:
            return None, None
        if act.op_type == "Relu":
            return int(y_zp), None
        # Clip: bounds from attrs or constant inputs
        lo = act.attrs.get("min")
        hi = act.attrs.get("max")
        if lo is None and len(act.inputs) > 1 and act.inputs[1]:
            v = _scalar(graph, act.inputs[1])
            lo = None if v is None else float(v[0])
        if hi is None and len(act.inputs) > 2 and act.inputs[2]:
            v = _scalar(graph, act.inputs[2])
            hi = None if v is None else float(v[0])
        qlo = None if lo is None else int(np.rint(lo / y_scale)) + int(y_zp)
        qhi = None if hi is None else int(np.rint(hi / y_scale)) + int(y_zp)
        return qlo, qhi

    # -- Conv / Gemm -------------------------------------------------------

    def _try_conv_gemm(self, graph, node, producers, consumers, use, *, kind):
        x_hit = self._dq_of(graph, producers, node.inputs[0])
        if x_hit is None:
            return False
        x_dq, xq_edge, x_scale, x_zp, _ = x_hit
        if x_scale.size != 1:
            return False                     # activations are per-tensor
        w_hit = self._weight_codes(graph, producers, node.inputs[1])
        if w_hit is None:
            return False
        w_dq, w_codes, w_scales = w_hit
        tail = self._tail(graph, consumers, node.outputs[0])
        if tail is None:
            return False
        act, q_node = tail
        y_scale = _scalar(graph, q_node.inputs[1])
        y_zp_arr = _scalar(graph, q_node.inputs[2] if len(q_node.inputs) > 2
                           else "", default=np.zeros(1, np.int8))
        if y_scale is None or y_zp_arr is None or y_scale.size != 1:
            return False
        y_scale_f = float(y_scale[0])
        y_zp = int(np.asarray(y_zp_arr).reshape(-1)[0])
        out_np_dtype = np.asarray(y_zp_arr).dtype

        a = node.attrs
        if kind == "gemm":
            if a.get("transA", 0) or a.get("alpha", 1.0) != 1.0 or \
                    a.get("beta", 1.0) != 1.0:
                return False
            # weights (N,K) with transB=1 (torch Linear) or (K,N) without
            w2 = w_codes if not a.get("transB", 0) else w_codes.T
            oc = w2.shape[1]
        else:
            if w_codes.ndim != 4:
                return False
            oc = w_codes.shape[0]
            w2 = w_codes
        if w_scales.size == 1:
            w_scales = np.full(oc, float(w_scales[0]), np.float32)
        if w_scales.size != oc:
            return False

        # bias: fp32 const -> int32 codes at scale s_x*s_w[c]
        bias_i32_name = ""
        if len(node.inputs) > 2 and node.inputs[2]:
            b = graph.initializers.get(node.inputs[2])
            if b is None:
                return False                 # non-const bias: leave fp32
            b_i32 = np.rint(np.asarray(b, np.float64) /
                            (float(x_scale[0]) * w_scales.astype(np.float64))
                            ).astype(np.int32)
            bias_i32_name = node.outputs[0] + "__bias_i32"
            graph.set_initializer(bias_i32_name, b_i32)

        qlo, qhi = self._act_bounds(graph, act, y_scale_f, y_zp)

        w_name = node.outputs[0] + "__w_int8"
        graph.set_initializer(w_name, np.ascontiguousarray(w2, dtype=np.int8))

        fused_from = [n.op_type for n in
                      (x_dq, w_dq, node) + ((act,) if act else ()) + (q_node,)]
        qnode = Node(
            "QConvFused" if kind == "conv" else "QGemmFused",
            [xq_edge, w_name] + ([bias_i32_name] if bias_i32_name else []),
            [q_node.outputs[0]],
            attrs={
                **({k: a[k] for k in ("strides", "pads", "dilations", "group",
                                      "auto_pad") if k in a}
                   if kind == "conv" else {}),
                "x_scale": float(x_scale[0]), "x_zp": int(np.asarray(x_zp).reshape(-1)[0]),
                "w_scales": w_scales, "y_scale": y_scale_f, "y_zp": y_zp,
                "act_qmin": qlo, "act_qmax": qhi,
                "out_dtype": str(np.dtype(out_np_dtype)),
            },
            name=(node.name or node.op_type) + "_q",
            )
        qnode.fused_from = fused_from
        graph.add_node(qnode)
        graph.set_info(TensorInfo(q_node.outputs[0],
                                  DType.from_numpy(out_np_dtype), ()))
        dead = [node, q_node] + ([act] if act else [])
        graph.remove_nodes(dead)
        self._sweep_dead(graph)
        return True

    # -- Add ---------------------------------------------------------------

    def _try_add(self, graph, node, producers, consumers, use):
        a_hit = self._dq_of(graph, producers, node.inputs[0])
        b_hit = self._dq_of(graph, producers, node.inputs[1])
        if a_hit is None or b_hit is None:
            return False
        a_dq, aq_edge, a_scale, a_zp, _ = a_hit
        b_dq, bq_edge, b_scale, b_zp, _ = b_hit
        if a_scale.size != 1 or b_scale.size != 1:
            return False
        tail = self._tail(graph, consumers, node.outputs[0])
        if tail is None:
            return False
        act, q_node = tail
        y_scale = _scalar(graph, q_node.inputs[1])
        y_zp_arr = _scalar(graph, q_node.inputs[2] if len(q_node.inputs) > 2
                           else "", default=np.zeros(1, np.int8))
        if y_scale is None or y_zp_arr is None or y_scale.size != 1:
            return False
        y_scale_f = float(y_scale[0])
        y_zp = int(np.asarray(y_zp_arr).reshape(-1)[0])
        qlo, qhi = self._act_bounds(graph, act, y_scale_f, y_zp)

        qnode = Node(
            "QAddFused", [aq_edge, bq_edge], [q_node.outputs[0]],
            attrs={
                "a_scale": float(a_scale[0]), "a_zp": int(np.asarray(a_zp).reshape(-1)[0]),
                "b_scale": float(b_scale[0]), "b_zp": int(np.asarray(b_zp).reshape(-1)[0]),
                "y_scale": y_scale_f, "y_zp": y_zp,
                "act_qmin": qlo, "act_qmax": qhi,
                "out_dtype": str(np.asarray(y_zp_arr).dtype),
            },
            name=(node.name or "Add") + "_q")
        qnode.fused_from = [n.op_type for n in
                            (a_dq, b_dq, node) + ((act,) if act else ())
                            + (q_node,)]
        graph.add_node(qnode)
        graph.set_info(TensorInfo(q_node.outputs[0],
                                  DType.from_numpy(np.asarray(y_zp_arr).dtype), ()))
        graph.remove_nodes([node, q_node] + ([act] if act else []))
        self._sweep_dead(graph)
        return True

    # -- cleanup -----------------------------------------------------------

    def _sweep_dead(self, graph: Graph) -> None:
        """Remove nodes whose outputs feed nothing (orphaned DQ/Q of the
        rewritten sandwich) — keeps the graph verifier-clean mid-pipeline."""
        while True:
            use = graph.use_count()
            dead = [n for n in graph.nodes
                    if all((not o) or use.get(o, 0) == 0 for o in n.outputs)]
            if not dead:
                return
            graph.remove_nodes(dead)


def quant_pipeline(*, verify=None, verbose: bool = False):
    """Pipeline for QDQ models: recognize quantized sandwiches FIRST (constant
    folding would fold DQ(w) into fp32 and erase them), then standard cleanup
    on the remaining fp32 sections."""
    from mdlc.passes.base import PassManager
    from mdlc.passes.constant_folding import ConstantFolding
    from mdlc.passes.dce import DeadCodeElimination
    from mdlc.passes.fuse_elementwise import FuseElementwise

    return PassManager(
        passes=[
            QdqLowering(),
            DeadCodeElimination(),
            ConstantFolding(),
            FuseElementwise(),
            DeadCodeElimination(),
        ],
        verify=verify,
        verbose=verbose,
    )
