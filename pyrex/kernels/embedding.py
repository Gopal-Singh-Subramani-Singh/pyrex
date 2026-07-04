"""Token embedding + positional encoding kernel."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class EmbeddingKernel:
    """
    Token embedding lookup: E[ids] where E is (vocab_size, embedding_dim)
    Purely memory-bound: no arithmetic, only table lookup + copy.

    FLOPs: 0 (pure gather, no multiply-accumulate)
    Bytes: 2 * seq_len * embedding_dim  (read embedding rows + write output)
    """
    vocab_size: int = 32000
    embedding_dim: int = 768
    seq_len: int = 512

    @property
    def flops(self) -> float:
        # Each token lookup involves fetching one row — approximately
        # 2 ops per element (read + copy)
        return 2.0 * self.seq_len * self.embedding_dim

    def bytes_transferred(self, precision: str = "fp32") -> float:
        bpe = 2.0 if precision == "fp16" else 4.0
        # Read seq_len embedding vectors from the table + write output
        return bpe * self.seq_len * self.embedding_dim * 2

    def arithmetic_intensity(self, precision: str = "fp32") -> float:
        bt = self.bytes_transferred(precision)
        return self.flops / bt if bt > 0 else 0.0

    @classmethod
    def from_params(cls, params: dict) -> "EmbeddingKernel":
        return cls(
            vocab_size=params.get("vocab_size", 32000),
            embedding_dim=params.get("embedding_dim", 768),
            seq_len=params.get("seq_len", 512),
        )
