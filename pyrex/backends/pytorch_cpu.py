from __future__ import annotations
from typing import Any, Dict
import torch
from pyrex.backends.pytorch_mps import PyTorchMPSBackend


class PyTorchCPUBackend(PyTorchMPSBackend):
    """
    PyTorch CPU backend. Reuses MPS kernel implementations,
    overrides device to CPU.
    """

    @property
    def backend_id(self) -> str:
        return "pytorch_cpu"

    @property
    def available(self) -> bool:
        return True

    def prepare(self, kernel_id: str, params: Dict[str, Any], precision: str) -> Any:
        device = torch.device("cpu")
        dtype = torch.float16 if precision == "fp16" else torch.float32
        context = self._build_context(kernel_id, params, device, dtype)
        context["device"] = device
        context["dtype"] = dtype
        context["kernel_id"] = kernel_id
        return context

    def sync(self) -> None:
        pass  # CPU is synchronous
