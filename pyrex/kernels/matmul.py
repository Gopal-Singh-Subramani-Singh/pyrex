"""General matrix multiply kernel."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple


@dataclass
class MatMulKernel:
    """
    General matrix multiply: C = A @ B
    A: (M, K), B: (K, N) → C: (M, N)

    FLOPs: 2 * M * K * N
    Memory bytes (fp32): 4 * (M*K + K*N + M*N)
    """
    M: int = 512
    K: int = 512
    N: int = 512

    @property
    def flops(self) -> float:
        return 2.0 * self.M * self.K * self.N

    def bytes_transferred(self, precision: str = "fp32") -> float:
        bpe = 2.0 if precision == "fp16" else 4.0
        return bpe * (self.M * self.K + self.K * self.N + self.M * self.N)

    def arithmetic_intensity(self, precision: str = "fp32") -> float:
        bt = self.bytes_transferred(precision)
        return self.flops / bt if bt > 0 else 0.0

    @classmethod
    def from_params(cls, params: dict) -> "MatMulKernel":
        size = params.get("size", [512, 512, 512])
        return cls(M=size[0], K=size[1], N=size[2])
