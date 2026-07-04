from __future__ import annotations
from typing import Any, Dict, Optional
import structlog
import torch

from pyrex.backends.base import BackendBase

logger = structlog.get_logger(__name__)


class PyTorchMPSBackend(BackendBase):
    """PyTorch with Apple MPS (Metal Performance Shaders) backend."""

    @property
    def backend_id(self) -> str:
        return "pytorch_mps"

    @property
    def available(self) -> bool:
        return torch.backends.mps.is_available()

    def prepare(self, kernel_id: str, params: Dict[str, Any], precision: str) -> Any:
        if not self.available:
            raise RuntimeError("MPS not available on this system")

        device = torch.device("mps")
        dtype = torch.float16 if precision == "fp16" else torch.float32

        context = self._build_context(kernel_id, params, device, dtype)
        context["device"] = device
        context["dtype"] = dtype
        context["kernel_id"] = kernel_id
        return context

    def _build_context(
        self, kernel_id: str, params: Dict, device: torch.device, dtype: torch.dtype
    ) -> Dict:
        if kernel_id == "matmul":
            M, K, N = params.get("size", [512, 512, 512])
            A = torch.randn(M, K, device=device, dtype=dtype)
            B = torch.randn(K, N, device=device, dtype=dtype)
            return {"A": A, "B": B, "type": "matmul"}

        elif kernel_id == "attention":
            B = params.get("batch_size", 8)
            S = params.get("seq_len", 512)
            H = params.get("hidden_dim", 768)
            n_heads = params.get("num_heads", 12)
            head_dim = H // n_heads
            Q = torch.randn(B, n_heads, S, head_dim, device=device, dtype=dtype)
            K_ = torch.randn(B, n_heads, S, head_dim, device=device, dtype=dtype)
            V = torch.randn(B, n_heads, S, head_dim, device=device, dtype=dtype)
            return {"Q": Q, "K": K_, "V": V, "head_dim": head_dim, "type": "attention"}

        elif kernel_id == "ffn":
            B = params.get("batch_size", 8)
            S = params.get("seq_len", 512)
            H = params.get("hidden_dim", 768)
            F = params.get("ffn_dim", 3072)
            x = torch.randn(B, S, H, device=device, dtype=dtype)
            w1 = torch.randn(H, F, device=device, dtype=dtype)
            w2 = torch.randn(F, H, device=device, dtype=dtype)
            b1 = torch.zeros(F, device=device, dtype=dtype)
            b2 = torch.zeros(H, device=device, dtype=dtype)
            return {"x": x, "w1": w1, "w2": w2, "b1": b1, "b2": b2, "type": "ffn"}

        elif kernel_id == "layernorm":
            B = params.get("batch_size", 8)
            S = params.get("seq_len", 512)
            H = params.get("hidden_dim", 768)
            x = torch.randn(B, S, H, device=device, dtype=dtype)
            ln = torch.nn.LayerNorm(H).to(device=device, dtype=dtype)
            return {"x": x, "ln": ln, "type": "layernorm"}

        elif kernel_id == "conv2d":
            B = params.get("batch_size", 8)
            C_in = params.get("in_channels", 64)
            C_out = params.get("out_channels", 128)
            HW = params.get("image_size", 56)
            KS = params.get("kernel_size", 3)
            x = torch.randn(B, C_in, HW, HW, device=device, dtype=dtype)
            conv = torch.nn.Conv2d(C_in, C_out, KS, padding=1).to(
                device=device, dtype=dtype
            )
            return {"x": x, "conv": conv, "type": "conv2d"}

        elif kernel_id == "embedding":
            vocab = params.get("vocab_size", 32000)
            dim = params.get("embedding_dim", 768)
            S = params.get("seq_len", 512)
            ids = torch.randint(0, vocab, (1, S), device=device)
            emb = torch.nn.Embedding(vocab, dim).to(device=device, dtype=dtype)
            return {"ids": ids, "emb": emb, "type": "embedding"}

        else:
            raise ValueError(f"Unknown kernel_id: {kernel_id}")

    def run_kernel(self, context: Any) -> None:
        t = context["type"]
        if t == "matmul":
            torch.mm(context["A"], context["B"])
        elif t == "attention":
            Q, K_, V = context["Q"], context["K"], context["V"]
            scale = context["head_dim"] ** -0.5
            scores = torch.matmul(Q, K_.transpose(-2, -1)) * scale
            weights = torch.softmax(scores, dim=-1)
            torch.matmul(weights, V)
        elif t == "ffn":
            x = torch.nn.functional.linear(context["x"], context["w1"].t(), context["b1"])
            x = torch.nn.functional.gelu(x)
            torch.nn.functional.linear(x, context["w2"].t(), context["b2"])
        elif t == "layernorm":
            context["ln"](context["x"])
        elif t == "conv2d":
            context["conv"](context["x"])
        elif t == "embedding":
            context["emb"](context["ids"])

    def sync(self) -> None:
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
