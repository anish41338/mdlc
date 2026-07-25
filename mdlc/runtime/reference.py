"""The NumPy reference executor.

Walks the IR in topological order and produces exact (within fp tolerance)
outputs. This is the oracle: graph passes must preserve its results, and
codegen kernels are diffed against it. It also understands the fused ops we
introduce so that an optimized graph still runs end-to-end on CPU.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from mdlc.ir import Graph, Node
from mdlc.runtime import ops


def _act(name: Optional[str], x: np.ndarray, attrs: dict) -> np.ndarray:
    if not name or name == "Identity":
        return x
    if name == "Relu":
        return ops.relu(x)
    if name == "Clip":
        return ops.clip(x, attrs.get("clip_min"), attrs.get("clip_max"))
    if name == "Relu6":
        return ops.clip(x, 0.0, 6.0)
    if name == "Sigmoid":
        return ops.sigmoid(x)
    if name == "Tanh":
        return ops.tanh(x)
    if name == "LeakyRelu":
        return ops.leaky_relu(x, attrs.get("alpha", 0.01))
    if name == "HardSigmoid":
        return ops.hard_sigmoid(x, attrs.get("alpha", 0.2), attrs.get("beta", 0.5))
    if name == "HardSwish":
        return ops.hard_swish(x)
    raise NotImplementedError(f"activation {name!r} not implemented")


class ReferenceExecutor:
    """Execute a ``Graph`` over NumPy arrays."""

    def __init__(self, graph: Graph) -> None:
        self.graph = graph

    def run(
        self,
        feeds: dict[str, np.ndarray],
        *,
        return_all: bool = False,
    ) -> dict[str, np.ndarray]:
        """Run the graph. Returns graph outputs, or every intermediate value
        when ``return_all`` is set (used by the per-layer correctness harness).
        """
        env: dict[str, np.ndarray] = {}
        for name, arr in self.graph.initializers.items():
            env[name] = np.asarray(arr)
        for name, arr in feeds.items():
            env[name] = np.asarray(arr)

        for node in self.graph.topo_sort():
            outputs = self._exec_node(node, env)
            for oname, oval in zip(node.outputs, outputs):
                if oname:
                    env[oname] = oval

        if return_all:
            return env
        return {o: env[o] for o in self.graph.outputs}

    # ------------------------------------------------------------------

    def _g(self, env, name):
        """Get a tensor edge value, or None for an omitted optional input."""
        if name == "":
            return None
        return env[name]

    def _exec_node(self, node: Node, env: dict) -> list[np.ndarray]:
        op = node.op_type
        a = node.attrs
        ins = [self._g(env, i) for i in node.inputs]

        # ---- fused ops -------------------------------------------------
        if op in ("FusedConvAct", "FusedConvBNAct"):
            x, w, b = ins[0], ins[1], (ins[2] if len(ins) > 2 else None)
            y = ops.conv(
                x, w, b,
                strides=a.get("strides", [1, 1]),
                pads=a.get("pads"),
                dilations=a.get("dilations", [1, 1]),
                group=a.get("group", 1),
                auto_pad=a.get("auto_pad", "NOTSET"),
            )
            return [_act(a.get("activation"), y, a)]

        if op == "FusedGemmAct":
            y = ops.gemm(ins[0], ins[1], ins[2] if len(ins) > 2 else None,
                         alpha=a.get("alpha", 1.0), beta=a.get("beta", 1.0),
                         transA=a.get("transA", 0), transB=a.get("transB", 0))
            return [_act(a.get("activation"), y, a)]

        if op == "FusedElementwise":
            return self._exec_fused_elementwise(node, env)

        # ---- quantized ops (Phase 3: QDQ -> DP4A path) -------------------
        if op == "QuantizeLinear":
            zp = ins[2] if len(ins) > 2 and ins[2] is not None else np.int8(0)
            return [ops.quantize_linear(ins[0], ins[1], np.asarray(zp),
                                        a.get("axis", 1))]
        if op == "DequantizeLinear":
            zp = ins[2] if len(ins) > 2 and ins[2] is not None else np.int8(0)
            return [ops.dequantize_linear(ins[0], ins[1], np.asarray(zp),
                                          a.get("axis", 1))]
        if op == "QConvFused":
            x, w, b = ins[0], ins[1], (ins[2] if len(ins) > 2 else None)
            return [ops.qconv(
                x, w, b,
                x_scale=a["x_scale"], x_zp=a["x_zp"], w_scales=a["w_scales"],
                y_scale=a["y_scale"], y_zp=a["y_zp"],
                strides=a.get("strides", [1, 1]), pads=a.get("pads"),
                dilations=a.get("dilations", [1, 1]), group=a.get("group", 1),
                act_qmin=a.get("act_qmin"), act_qmax=a.get("act_qmax"),
                out_dtype=np.dtype(a.get("out_dtype", "int8")))]
        if op == "QGemmFused":
            x, w, b = ins[0], ins[1], (ins[2] if len(ins) > 2 else None)
            return [ops.qgemm(
                x, w, b,
                x_scale=a["x_scale"], x_zp=a["x_zp"], w_scales=a["w_scales"],
                y_scale=a["y_scale"], y_zp=a["y_zp"],
                act_qmin=a.get("act_qmin"), act_qmax=a.get("act_qmax"),
                out_dtype=np.dtype(a.get("out_dtype", "int8")))]
        if op == "QAddFused":
            return [ops.qadd(
                ins[0], ins[1],
                a_scale=a["a_scale"], a_zp=a["a_zp"],
                b_scale=a["b_scale"], b_zp=a["b_zp"],
                y_scale=a["y_scale"], y_zp=a["y_zp"],
                act_qmin=a.get("act_qmin"), act_qmax=a.get("act_qmax"),
                out_dtype=np.dtype(a.get("out_dtype", "int8")))]

        # ---- core ops --------------------------------------------------
        if op == "Conv":
            return [ops.conv(ins[0], ins[1], ins[2] if len(ins) > 2 else None,
                             strides=a.get("strides", [1, 1]), pads=a.get("pads"),
                             dilations=a.get("dilations", [1, 1]),
                             group=a.get("group", 1), auto_pad=a.get("auto_pad", "NOTSET"))]
        if op == "Gemm":
            return [ops.gemm(ins[0], ins[1], ins[2] if len(ins) > 2 else None,
                             alpha=a.get("alpha", 1.0), beta=a.get("beta", 1.0),
                             transA=a.get("transA", 0), transB=a.get("transB", 0))]
        if op == "MatMul":
            return [ops.matmul(ins[0], ins[1])]
        if op == "BatchNormalization":
            return [ops.batchnorm(ins[0], ins[1], ins[2], ins[3], ins[4],
                                  epsilon=a.get("epsilon", 1e-5))]
        if op == "MaxPool":
            return [ops.maxpool(ins[0], kernel=a["kernel_shape"],
                                strides=a.get("strides", [1, 1]), pads=a.get("pads"),
                                dilations=a.get("dilations", [1, 1]),
                                auto_pad=a.get("auto_pad", "NOTSET"),
                                ceil_mode=a.get("ceil_mode", 0))]
        if op == "AveragePool":
            return [ops.averagepool(ins[0], kernel=a["kernel_shape"],
                                    strides=a.get("strides", [1, 1]), pads=a.get("pads"),
                                    auto_pad=a.get("auto_pad", "NOTSET"),
                                    count_include_pad=a.get("count_include_pad", 0),
                                    ceil_mode=a.get("ceil_mode", 0))]
        if op == "GlobalAveragePool":
            return [ops.global_average_pool(ins[0])]
        if op == "ReduceMean":
            # opset<18: axes attribute; opset>=18: axes as optional input[1]
            axes = ins[1] if len(ins) > 1 and ins[1] is not None else a.get("axes")
            return [ops.reduce_mean(ins[0], axes, a.get("keepdims", 1))]
        if op == "Erf":
            return [ops.erf(ins[0])]
        if op in ("Relu", "Sigmoid", "Tanh", "HardSwish"):
            return [_act(op, ins[0], a)]
        if op == "Clip":
            lo = ins[1] if len(ins) > 1 and ins[1] is not None else a.get("min")
            hi = ins[2] if len(ins) > 2 and ins[2] is not None else a.get("max")
            lo = None if lo is None else float(np.asarray(lo).item())
            hi = None if hi is None else float(np.asarray(hi).item())
            return [ops.clip(ins[0], lo, hi)]
        if op == "LeakyRelu":
            return [ops.leaky_relu(ins[0], a.get("alpha", 0.01))]
        if op == "HardSigmoid":
            return [ops.hard_sigmoid(ins[0], a.get("alpha", 0.2), a.get("beta", 0.5))]
        if op == "Softmax":
            return [ops.softmax(ins[0], a.get("axis", -1))]
        if op == "Add":
            return [ops.add(ins[0], ins[1])]
        if op == "Sub":
            return [ops.sub(ins[0], ins[1])]
        if op == "Mul":
            return [ops.mul(ins[0], ins[1])]
        if op == "Div":
            return [ops.div(ins[0], ins[1])]
        if op == "Flatten":
            return [ops.flatten(ins[0], a.get("axis", 1))]
        if op == "Reshape":
            return [ops.reshape(ins[0], ins[1])]
        if op == "Transpose":
            return [ops.transpose(ins[0], a.get("perm"))]
        if op == "Concat":
            return [ops.concat([x for x in ins if x is not None], a.get("axis", 0))]
        if op == "Squeeze":
            axes = ins[1] if len(ins) > 1 and ins[1] is not None else a.get("axes")
            return [ops.squeeze(ins[0], axes)]
        if op == "Unsqueeze":
            axes = ins[1] if len(ins) > 1 and ins[1] is not None else a.get("axes")
            return [ops.unsqueeze(ins[0], axes)]
        if op == "Pad":
            pads = ins[1] if len(ins) > 1 and ins[1] is not None else a.get("pads")
            val = ins[2] if len(ins) > 2 and ins[2] is not None else a.get("value", 0.0)
            return [ops.pad(ins[0], pads, float(np.asarray(val).item()) if val is not None else 0.0,
                            a.get("mode", "constant"))]
        if op == "Constant":
            return [a["value"]]
        if op == "Identity":
            return [ins[0]]
        if op == "Shape":
            return [np.array(ins[0].shape, dtype=np.int64)]

        raise NotImplementedError(f"reference executor: op {op!r} not implemented")

    def _exec_fused_elementwise(self, node: Node, env: dict) -> list[np.ndarray]:
        """Run an elementwise-fusion group by executing its inner subgraph."""
        local = dict(env)
        sub: list[Node] = node.attrs["subgraph"]
        for sn in sub:
            outs = self._exec_node(sn, local)
            for oname, oval in zip(sn.outputs, outs):
                if oname:
                    local[oname] = oval
        return [local[o] for o in node.outputs]


def run_reference(graph: Graph, feeds: dict[str, np.ndarray], **kw) -> dict[str, np.ndarray]:
    return ReferenceExecutor(graph).run(feeds, **kw)
