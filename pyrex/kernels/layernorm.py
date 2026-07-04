"""Layer normalisation kernel."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class LayerNormKernel:
    """
    Layer normalisation: y = (x - mean) / sqrt(var + eps) * gamma + beta

    Memory-bound: reads entire activation tensor twice (mean pass + norm pass).
    FLOPs: ~5 * B * S * H
    """
    batch_size: int = 8
    seq_len: int = 512
    hidden_dim: int = 768

    @property
    def flops(self) -> float:
        B, S, H = self.batch_size, self.seq_len, self.hidden_dim
        # mean + variance + norm + scale + bias = ~5 ops per element
        return 5.0 * B * S * H

    def bytes_transferred(self, precision: str = "fp32") -> float:
        bpe = 2.0 if precision == "fp16" else 4.0
        B, S, H = self.batch_size, self.seq_len, self.hidden_dim
        # Read input + write output + gamma + beta
        return bpe * (B * S * H * 2 + 2 * H)

    def arithmetic_intensity(self, precision: str = "fp32") -> float:
        bt = self.bytes_transferred(precision)
        return self.flops / bt if bt > 0 else 0.0

    @classmethod
    def from_params(cls, params: dict) -> "LayerNormKernel":
        return cls(
            batch_size=params.get("batch_size", 8),
            seq_len=params.get("seq_len", 512),
            hidden_dim=params.get("hidden_dim", 768),
        )
