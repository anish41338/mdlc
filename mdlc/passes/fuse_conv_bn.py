"""Fold BatchNormalization into a preceding Conv's weights and bias.

At inference time BN is an affine transform per output channel:

    y = (x - mean) / sqrt(var + eps) * gamma + beta
      = x * s + t,   s = gamma/sqrt(var+eps),   t = beta - mean*s

Folding it into the conv that produced ``x`` makes BN disappear entirely:

    w' = w * s[:, None, None, None]
    b' = (b_conv) * s + t          (b_conv = 0 if the conv had no bias)

This is pure compile-time arithmetic and removes a whole kernel launch plus a
full read/write of the feature map. Legal only when the conv output feeds the
BN exclusively and isn't a graph output.
"""

from __future__ import annotations

import numpy as np

from mdlc.ir import Graph, Node
from mdlc.passes.base import Pass


class FuseConvBN(Pass):
    name = "fuse_conv_bn"

    def run(self, graph: Graph) -> bool:
        changed = False
        producer = graph.producers()
        use_count = graph.use_count()

        for bn in list(graph.nodes):
            if bn.op_type != "BatchNormalization":
                continue
            conv = producer.get(bn.inputs[0])
            if conv is None or conv.op_type != "Conv":
                continue
            conv_out = conv.outputs[0]
            # The conv's output must feed *only* this BN (single consumer) and
            # must not itself be a graph output.
            if use_count.get(conv_out, 0) != 1 or conv_out in graph.outputs:
                continue
            # Weights and BN params must be constant to fold at compile time.
            w_name = conv.inputs[1]
            if not graph.is_constant(w_name):
                continue
            gamma, beta, mean, var = bn.inputs[1:5]
            if not all(graph.is_constant(t) for t in (gamma, beta, mean, var)):
                continue

            eps = bn.attr("epsilon", 1e-5)
            w = graph.initializers[w_name].astype(np.float32)
            g = graph.initializers[gamma].astype(np.float32)
            b_bn = graph.initializers[beta].astype(np.float32)
            m = graph.initializers[mean].astype(np.float32)
            v = graph.initializers[var].astype(np.float32)

            s = g / np.sqrt(v + eps)                      # (oc,)
            oc = w.shape[0]
            w_new = w * s.reshape(oc, *([1] * (w.ndim - 1)))

            if len(conv.inputs) > 2 and conv.inputs[2]:
                b_conv = graph.initializers[conv.inputs[2]].astype(np.float32)
            else:
                b_conv = np.zeros(oc, dtype=np.float32)
            b_new = b_conv * s + (b_bn - m * s)

            # Write folded weights as fresh initializers (don't clobber shared).
            new_w_name = w_name + "_bnfold"
            new_b_name = (conv.name or "conv") + "_bias_bnfold"
            graph.set_initializer(new_w_name, w_new)
            graph.set_initializer(new_b_name, b_new)

            conv.inputs = [conv.inputs[0], new_w_name, new_b_name]
            conv.outputs = [bn.outputs[0]]   # take over BN's output edge
            graph.remove_nodes([bn])
            changed = True

        return changed
