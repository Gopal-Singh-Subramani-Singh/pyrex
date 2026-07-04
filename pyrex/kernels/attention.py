"""Scaled dot-product attention kernel."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class AttentionKernel:
    """
    Multi-head scaled dot-product attention.
    Q, K, V: (batch, heads, seq_len, head_dim)

    FLOPs: 4 * B * S^2 * H  (two matmuls + softmax approximation)
    Memory: bandwidth-intensive for large sequence lengths.
    """
    batch_size: int = 8
    seq_len: int = 512
    hidden_dim: int = 768
    num_heads: int = 12

    @property
    def head_dim(self) -> int:
        return self.hidden_dim // self.num_heads

    @property
    def flops(self) -> float:
        # QK^T matmul + AV matmul, each B*H * S*S*head_dim
        B, S, H, h = self.batch_size, self.seq_len, self.num_heads, self.head_dim
        return 2.0 * (2.0 * B * H * S * S * h)

    def bytes_transferred(self, precision: str = "fp32") -> float:
        bpe = 2.0 if precision == "fp16" else 4.0
        B, H, S, h = self.batch_size, self.num_heads, self.seq_len, self.head_dim
        # Q, K, V input + attention weights + output
        return bpe * (3.0 * B * H * S * h + B * H * S * S + B * H * S * h)

    def arithmetic_intensity(self, precision: str = "fp32") -> float:
        bt = self.bytes_transferred(precision)
        return self.flops / bt if bt > 0 else 0.0

    @classmethod
    def from_params(cls, params: dict) -> "AttentionKernel":
        return cls(
            batch_size=params.get("batch_size", 8),
            seq_len=params.get("seq_len", 512),
            hidden_dim=params.get("hidden_dim", 768),
            num_heads=params.get("num_heads", 12),
        )
