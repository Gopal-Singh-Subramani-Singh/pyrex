"""2D convolution kernel — ResNet-style dimensions."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class Conv2DKernel:
    """
    2D convolution with padding=1.
    Input: (B, C_in, H, W)
    Weight: (C_out, C_in, KH, KW)
    Output: (B, C_out, H, W)  [with padding=1, stride=1]

    FLOPs: 2 * B * C_out * C_in * KH * KW * H_out * W_out
    """
    batch_size: int = 8
    in_channels: int = 64
    out_channels: int = 128
    image_size: int = 56
    kernel_size: int = 3

    @property
    def output_size(self) -> int:
        # padding=1, stride=1 → same spatial size
        return self.image_size

    @property
    def flops(self) -> float:
        B = self.batch_size
        C_in, C_out = self.in_channels, self.out_channels
        KS, H_out = self.kernel_size, self.output_size
        return 2.0 * B * C_out * C_in * KS * KS * H_out * H_out

    def bytes_transferred(self, precision: str = "fp32") -> float:
        bpe = 2.0 if precision == "fp16" else 4.0
        B = self.batch_size
        C_in, C_out = self.in_channels, self.out_channels
        HW = self.image_size
        KS = self.kernel_size
        # Input + weight + output
        input_bytes = B * C_in * HW * HW
        weight_bytes = C_out * C_in * KS * KS
        output_bytes = B * C_out * HW * HW
        return bpe * (input_bytes + weight_bytes + output_bytes)

    def arithmetic_intensity(self, precision: str = "fp32") -> float:
        bt = self.bytes_transferred(precision)
        return self.flops / bt if bt > 0 else 0.0

    @classmethod
    def from_params(cls, params: dict) -> "Conv2DKernel":
        return cls(
            batch_size=params.get("batch_size", 8),
            in_channels=params.get("in_channels", 64),
            out_channels=params.get("out_channels", 128),
            image_size=params.get("image_size", 56),
            kernel_size=params.get("kernel_size", 3),
        )
