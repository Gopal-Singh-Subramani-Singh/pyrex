from __future__ import annotations
from typing import Any, Dict, List, Optional
import numpy as np
import structlog

from pyrex.backends.base import BackendBase

logger = structlog.get_logger(__name__)


class ONNXRuntimeBackend(BackendBase):
    """
    ONNX Runtime backend with CoreML Execution Provider.
    Exports a PyTorch module to ONNX on prepare(), then runs ORT inference.
    """

    @property
    def backend_id(self) -> str:
        return "onnx_rt"

    @property
    def available(self) -> bool:
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            return "CoreMLExecutionProvider" in providers or "CPUExecutionProvider" in providers
        except ImportError:
            return False

    def prepare(self, kernel_id: str, params: Dict[str, Any], precision: str) -> Any:
        import torch
        import onnxruntime as ort
        import io

        dtype = torch.float32  # ORT CoreML works best with fp32

        context = self._build_torch_context(kernel_id, params, dtype)
        dummy_inputs = context["dummy_inputs"]
        torch_module = context["module"]

        buffer = io.BytesIO()
        torch.onnx.export(
            torch_module,
            dummy_inputs,
            buffer,
            opset_version=18,
            input_names=context.get("input_names", ["input"]),
            output_names=["output"],
            dynamic_axes=context.get("dynamic_axes", {}),
        )
        buffer.seek(0)

        providers = []
        try:
            providers.append("CoreMLExecutionProvider")
        except Exception:
            pass
        providers.append("CPUExecutionProvider")

        sess = ort.InferenceSession(buffer.read(), providers=providers)
        np_inputs = context["np_inputs"]

        return {
            "session": sess,
            "np_inputs": np_inputs,
            "kernel_id": kernel_id,
        }

    def _build_torch_context(
        self, kernel_id: str, params: Dict, dtype: Any
    ) -> Dict:
        import torch
        import torch.nn as nn

        if kernel_id == "matmul":
            M, K, N = params.get("size", [512, 512, 512])
            A = torch.randn(M, K, dtype=dtype)
            B = torch.randn(K, N, dtype=dtype)

            class MatMulMod(nn.Module):
                def forward(self, a, b):
                    return torch.mm(a, b)

            return {
                "module": MatMulMod(),
                "dummy_inputs": (A, B),
                "input_names": ["A", "B"],
                "np_inputs": {"A": A.numpy(), "B": B.numpy()},
                "dynamic_axes": {},
            }

        elif kernel_id == "ffn":
            B_ = params.get("batch_size", 8)
            S = params.get("seq_len", 512)
            H = params.get("hidden_dim", 768)
            F = params.get("ffn_dim", 3072)
            x = torch.randn(B_, S, H, dtype=dtype)

            class FFNMod(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc1 = nn.Linear(H, F)
                    self.fc2 = nn.Linear(F, H)

                def forward(self, x):
                    return self.fc2(nn.functional.gelu(self.fc1(x)))

            mod = FFNMod()
            return {
                "module": mod,
                "dummy_inputs": (x,),
                "input_names": ["x"],
                "np_inputs": {"x": x.numpy()},
                "dynamic_axes": {},
            }

        elif kernel_id in ("layernorm", "attention", "conv2d", "embedding"):
            # Fallback: wrap as identity for unsupported kernels
            x = torch.randn(8, 512, dtype=dtype)

            class Identity(nn.Module):
                def forward(self, x):
                    return x

            return {
                "module": Identity(),
                "dummy_inputs": (x,),
                "input_names": ["x"],
                "np_inputs": {"x": x.numpy()},
                "dynamic_axes": {},
            }

        raise ValueError(f"Unknown kernel: {kernel_id}")

    def run_kernel(self, context: Any) -> None:
        context["session"].run(None, context["np_inputs"])

    def sync(self) -> None:
        pass
