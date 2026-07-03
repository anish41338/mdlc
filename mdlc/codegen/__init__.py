"""Codegen: lower the optimized IR to CUDA C and JIT it with NVRTC.

The design follows TVM's key separation: a *compute* definition (what the
kernel computes) is decoupled from a *schedule* (how it's tiled, where data is
staged, how loops are unrolled). The schedule is exactly what the autotuner
searches over.
"""

from mdlc.codegen.schedule import GemmSchedule, gemm_search_space

__all__ = ["GemmSchedule", "gemm_search_space"]
