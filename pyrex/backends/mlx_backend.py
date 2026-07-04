from __future__ import annotations
from typing import Any, Dict, Optional
import structlog

from pyrex.backends.base import BackendBase

logger = structlog.get_logger(__name__)

_MLX_AVAILABLE = False
try:
    import mlx.core as mx
    import mlx.nn as mlx_nn
    _MLX_AVAILABLE = True
except ImportError:
    mx = None
    mlx_nn = None


class MLXBackend(BackendBase):
    """
    Apple MLX backend.
    MLX is Apple's own ML framework, released 2023.
    Uses unified memory — no explicit host-device transfers.
    Lazy evaluation model: mx.eval() forces synchronisation.
    """

    @property
    def backend_id(self) -> str:
        return "mlx"

    @property
    def available(self) -> bool:
        return _MLX_AVAILABLE

    def prepare(self, kernel_id: str, params: Dict[str, Any], precision: str) -> Any:
        if not _MLX_AVAILABLE:
            raise RuntimeError("MLX not available. Install with: pip install mlx")

        dtype = mx.float16 if precision == "fp16" else mx.float32
        context = self._build_context(kernel_id, params, dtype)
        context["kernel_id"] = kernel_id
        # Force materialisation of all arrays before timing
        mx.eval(*[v for v in context.values() if isinstance(v, mx.array)])
        return context

    def _build_context(self, kernel_id: str, params: Dict, dtype) -> Dict:
        if kernel_id == "matmul":
            M, K, N = params.get("size", [512, 512, 512])
            A = mx.random.normal([M, K]).astype(dtype)
            B = mx.random.normal([K, N]).astype(dtype)
            mx.eval(A, B)
            return {"A": A, "B": B, "type": "matmul"}

        elif kernel_id == "attention":
            B_ = params.get("batch_size", 8)
            S = params.get("seq_len", 512)
            H = params.get("hidden_dim", 768)
            n_heads = params.get("num_heads", 12)
            head_dim = H // n_heads
            Q = mx.random.normal([B_, n_heads, S, head_dim]).astype(dtype)
            K_ = mx.random.normal([B_, n_heads, S, head_dim]).astype(dtype)
            V = mx.random.normal([B_, n_heads, S, head_dim]).astype(dtype)
            mx.eval(Q, K_, V)
            return {"Q": Q, "K": K_, "V": V, "head_dim": head_dim, "type": "attention"}

        elif kernel_id == "ffn":
            B_ = params.get("batch_size", 8)
            S = params.get("seq_len", 512)
            H = params.get("hidden_dim", 768)
            F_ = params.get("ffn_dim", 3072)
            x = mx.random.normal([B_, S, H]).astype(dtype)
            w1 = mx.random.normal([H, F_]).astype(dtype)
            w2 = mx.random.normal([F_, H]).astype(dtype)
            mx.eval(x, w1, w2)
            return {"x": x, "w1": w1, "w2": w2, "type": "ffn"}

        elif kernel_id == "layernorm":
            B_ = params.get("batch_size", 8)
            S = params.get("seq_len", 512)
            H = params.get("hidden_dim", 768)
            x = mx.random.normal([B_, S, H]).astype(dtype)
            mx.eval(x)
            return {"x": x, "H": H, "type": "layernorm"}

        elif kernel_id == "embedding":
            vocab = params.get("vocab_size", 32000)
            dim = params.get("embedding_dim", 768)
            S = params.get("seq_len", 512)
            W = mx.random.normal([vocab, dim]).astype(dtype)
            ids = mx.array([[i % vocab for i in range(S)]])
            mx.eval(W, ids)
            return {"W": W, "ids": ids, "type": "embedding"}

        elif kernel_id == "conv2d":
            # MLX conv2d: NHWC layout
            B_ = params.get("batch_size", 8)
            C_in = params.get("in_channels", 64)
            C_out = params.get("out_channels", 128)
            HW = params.get("image_size", 56)
            KS = params.get("kernel_size", 3)
            x = mx.random.normal([B_, HW, HW, C_in]).astype(dtype)
            w = mx.random.normal([C_out, KS, KS, C_in]).astype(dtype)
            mx.eval(x, w)
            return {"x": x, "w": w, "KS": KS, "type": "conv2d"}

        raise ValueError(f"Unknown kernel: {kernel_id}")

    def run_kernel(self, context: Any) -> None:
        t = context["type"]
        if t == "matmul":
            result = mx.matmul(context["A"], context["B"])
        elif t == "attention":
            Q, K_, V = context["Q"], context["K"], context["V"]
            scale = context["head_dim"] ** -0.5
            scores = (Q @ K_.transpose(0, 1, 3, 2)) * scale
            weights = mx.softmax(scores, axis=-1)
            result = weights @ V
        elif t == "ffn":
            x = context["x"] @ context["w1"]
            x = mx.maximum(0.0, x) * 0.5 * (1.0 + mx.tanh(
                0.7978845608 * (x + 0.044715 * x ** 3)
            ))  # approximate GELU
            result = x @ context["w2"]
        elif t == "layernorm":
            x = context["x"]
            mean = mx.mean(x, axis=-1, keepdims=True)
            var = mx.mean((x - mean) ** 2, axis=-1, keepdims=True)
            result = (x - mean) / mx.sqrt(var + 1e-5)
        elif t == "embedding":
            result = context["W"][context["ids"]]
        elif t == "conv2d":
            result = mx.conv2d(context["x"], context["w"], padding=1)
        else:
            return
        mx.eval(result)

    def sync(self) -> None:
        if _MLX_AVAILABLE:
            if hasattr(mx, "synchronize"):
                mx.synchronize()
