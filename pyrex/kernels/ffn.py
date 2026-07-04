"""Feed-forward network kernel: 2 linear layers + GELU activation."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class FFNKernel:
    """
    Standard Transformer FFN: Linear(H→F) → GELU → Linear(F→H)

    FLOPs: 2 * B * S * (H*F + F*H)
    """
    batch_size: int = 8
    seq_len: int = 512
    hidden_dim: int = 768
    ffn_dim: int = 3072

    @property
    def flops(self) -> float:
        B, S, H, F = self.batch_size, self.seq_len, self.hidden_dim, self.ffn_dim
        return 2.0 * B * S * (H * F + F * H)

    def bytes_transferred(self, precision: str = "fp32") -> float:
        bpe = 2.0 if precision == "fp16" else 4.0
        B, S, H, F = self.batch_size, self.seq_len, self.hidden_dim, self.ffn_dim
        # Inputs + weights for both linear layers + output
        return bpe * (B * S * H + H * F + F + B * S * F + F * H + H + B * S * H)

    def arithmetic_intensity(self, precision: str = "fp32") -> float:
        bt = self.bytes_transferred(precision)
        return self.flops / bt if bt > 0 else 0.0

    @classmethod
    def from_params(cls, params: dict) -> "FFNKernel":
        return cls(
            batch_size=params.get("batch_size", 8),
            seq_len=params.get("seq_len", 512),
            hidden_dim=params.get("hidden_dim", 768),
            ffn_dim=params.get("ffn_dim", 3072),
        )
