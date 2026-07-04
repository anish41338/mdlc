"""Schedules: the tunable knobs codegen exposes to the autotuner.

A schedule is the "how" that the autotuner explores while the compute stays
fixed — split/tile sizes, thread-block shape, and register-blocking/unroll
factors. Keeping it a small dataclass with an explicit, *pruned* search space
is how we avoid combinatorial blowup (the classic autotuning failure mode).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class GemmSchedule:
    """Tiling schedule for the shared-memory GEMM.

    A thread block computes a ``BM x BN`` output tile, cooperatively staging
    ``BM x BK`` and ``BK x BN`` slabs into shared memory. Each thread computes a
    ``TM x TN`` micro-tile in registers (register blocking), so the block has
    ``(BM/TM) x (BN/TN)`` threads.
    """

    BM: int = 64        # block tile rows of C
    BN: int = 64        # block tile cols of C
    BK: int = 16        # depth step staged into shared memory
    TM: int = 4         # rows of C each thread accumulates in registers
    TN: int = 4         # cols of C each thread accumulates in registers

    def is_valid(self) -> bool:
        if min(self.BM, self.BN, self.BK, self.TM, self.TN) <= 0:
            return False
        if self.BM % self.TM or self.BN % self.TN:
            return False
        threads = self.threads_per_block()
        if threads > 1024 or threads < 32:
            return False
        # Shared-memory load must distribute evenly across the thread block.
        if (self.BM * self.BK) % threads or (self.BK * self.BN) % threads:
            return False
        return True

    def threads_per_block(self) -> int:
        return (self.BM // self.TM) * (self.BN // self.TN)

    def smem_bytes(self, dtype_bytes: int = 4) -> int:
        return (self.BM * self.BK + self.BK * self.BN) * dtype_bytes

    def key(self) -> str:
        return f"BM{self.BM}_BN{self.BN}_BK{self.BK}_TM{self.TM}_TN{self.TN}"


def gemm_search_space(*, max_smem: int = 48 * 1024) -> Iterator[GemmSchedule]:
    """Yield valid GEMM schedules. The space is deliberately pruned to a few
    proven shapes rather than a full cross-product — bounded by validity,
    thread count, and shared-memory capacity."""
    for BM in (32, 64, 128):
        for BN in (32, 64, 128):
            for BK in (8, 16, 32):
                for TM in (2, 4, 8):
                    for TN in (2, 4, 8):
                        s = GemmSchedule(BM, BN, BK, TM, TN)
                        if s.is_valid() and s.smem_bytes() <= max_smem:
                            yield s


# A robust default that works across shapes without tuning.
DEFAULT_GEMM = GemmSchedule(BM=64, BN=64, BK=16, TM=4, TN=4)


@dataclass(frozen=True)
class DepthwiseSchedule:
    """Tiling schedule for the direct depthwise conv.

    A block computes a ``tile_h x tile_w`` patch of one (n, c_out) output
    plane with one thread per output pixel; ``direct_load`` bypasses the
    shared-memory input staging (profitable when the halo re-read is cheap
    relative to the barrier, e.g. tiny tiles or stride 2).
    """

    tile_h: int = 8
    tile_w: int = 8
    direct_load: bool = False

    def threads_per_block(self) -> int:
        return self.tile_h * self.tile_w

    def is_valid(self, *, max_threads: int = 1024) -> bool:
        if self.tile_h <= 0 or self.tile_w <= 0:
            return False
        return 32 <= self.threads_per_block() <= max_threads

    def smem_bytes(self, K: int, stride: int, dilation: int = 1,
                   dtype_bytes: int = 4) -> int:
        smh = (self.tile_h - 1) * stride + (K - 1) * dilation + 1
        smw = (self.tile_w - 1) * stride + (K - 1) * dilation + 1
        return (smh * smw + K * K) * dtype_bytes

    def key(self) -> str:
        return f"TH{self.tile_h}_TW{self.tile_w}_{'dl' if self.direct_load else 'sm'}"


def depthwise_search_space(*, max_threads: int = 256) -> Iterator[DepthwiseSchedule]:
    """Tuner knobs per the brief: tile_h/tile_w in {4,8,16,32}, smem vs direct
    load. Thread count capped (sim spawns an OS thread per CUDA thread, and
    >256 threads/block rarely wins for a one-pixel-per-thread kernel)."""
    for th in (4, 8, 16, 32):
        for tw in (4, 8, 16, 32):
            for dl in (False, True):
                s = DepthwiseSchedule(th, tw, dl)
                if s.is_valid(max_threads=max_threads):
                    yield s


DEFAULT_DEPTHWISE = DepthwiseSchedule(tile_h=8, tile_w=8, direct_load=False)
