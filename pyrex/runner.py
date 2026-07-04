from __future__ import annotations
import platform
import subprocess
import time
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
import structlog
import torch
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from pyrex.models import (
    BenchmarkRun, KernelResult, LatencyStats, TelemetrySnapshot
)
from pyrex.backends.base import BackendBase
from pyrex.backends.pytorch_mps import PyTorchMPSBackend
from pyrex.backends.pytorch_cpu import PyTorchCPUBackend
from pyrex.backends.onnx_rt import ONNXRuntimeBackend
from pyrex.backends.mlx_backend import MLXBackend
from pyrex import telemetry

logger = structlog.get_logger(__name__)
console = Console()

ALL_BACKENDS: Dict[str, BackendBase] = {
    "pytorch_mps": PyTorchMPSBackend(),
    "pytorch_cpu": PyTorchCPUBackend(),
    "onnx_rt": ONNXRuntimeBackend(),
    "mlx": MLXBackend(),
}


def _get_platform_info() -> dict:
    try:
        chip = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            text=True, timeout=2
        ).strip()
    except Exception:
        chip = platform.processor()
    return {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "chip": chip,
    }


def _compute_stats(timings: list[float]) -> LatencyStats:
    arr = np.array(timings)
    return LatencyStats(
        mean_ms=float(np.mean(arr)),
        std_ms=float(np.std(arr)),
        p50_ms=float(np.percentile(arr, 50)),
        p95_ms=float(np.percentile(arr, 95)),
        p99_ms=float(np.percentile(arr, 99)),
        min_ms=float(np.min(arr)),
        max_ms=float(np.max(arr)),
        raw_ms=timings,
    )


def _estimate_flops(kernel_id: str, params: dict) -> Optional[float]:
    """Rough FLOP estimates for all kernels."""
    if kernel_id == "matmul":
        M, K, N = params.get("size", [512, 512, 512])
        return 2.0 * M * K * N
    elif kernel_id == "attention":
        B = params.get("batch_size", 8)
        S = params.get("seq_len", 512)
        H = params.get("hidden_dim", 768)
        n_heads = params.get("num_heads", 12)
        head_dim = H // n_heads
        # QK^T + softmax + AV — 2 matmuls each B*n_heads*S*S*head_dim
        return 4.0 * B * n_heads * S * S * head_dim
    elif kernel_id == "ffn":
        B = params.get("batch_size", 8)
        S = params.get("seq_len", 512)
        H = params.get("hidden_dim", 768)
        F = params.get("ffn_dim", 3072)
        return 2.0 * B * S * (H * F + F * H)
    elif kernel_id == "layernorm":
        B = params.get("batch_size", 8)
        S = params.get("seq_len", 512)
        H = params.get("hidden_dim", 768)
        return 5.0 * B * S * H  # mean + var + norm + scale + bias
    elif kernel_id == "conv2d":
        B = params.get("batch_size", 8)
        C_in = params.get("in_channels", 64)
        C_out = params.get("out_channels", 128)
        HW = params.get("image_size", 56)
        KS = params.get("kernel_size", 3)
        return 2.0 * B * C_out * C_in * KS * KS * HW * HW
    elif kernel_id == "embedding":
        S = params.get("seq_len", 512)
        D = params.get("embedding_dim", 768)
        return 2.0 * S * D  # gather + copy
    return None


def _estimate_bytes(kernel_id: str, params: dict, precision: str) -> Optional[float]:
    """Rough memory traffic estimates in bytes for all kernels."""
    bpe = 2.0 if precision == "fp16" else 4.0
    if kernel_id == "matmul":
        M, K, N = params.get("size", [512, 512, 512])
        return bpe * (M * K + K * N + M * N)
    elif kernel_id == "attention":
        B = params.get("batch_size", 8)
        S = params.get("seq_len", 512)
        H = params.get("hidden_dim", 768)
        n_heads = params.get("num_heads", 12)
        head_dim = H // n_heads
        # Q, K, V inputs + scores matrix + output
        return bpe * (3.0 * B * n_heads * S * head_dim + B * n_heads * S * S + B * n_heads * S * head_dim)
    elif kernel_id == "ffn":
        B = params.get("batch_size", 8)
        S = params.get("seq_len", 512)
        H = params.get("hidden_dim", 768)
        F = params.get("ffn_dim", 3072)
        # input + W1 + b1 + hidden + W2 + b2 + output
        return bpe * (B * S * H + H * F + F + B * S * F + F * H + H + B * S * H)
    elif kernel_id == "layernorm":
        B = params.get("batch_size", 8)
        S = params.get("seq_len", 512)
        H = params.get("hidden_dim", 768)
        # read input twice + write output + gamma + beta
        return bpe * (B * S * H * 2 + 2 * H)
    elif kernel_id == "conv2d":
        B = params.get("batch_size", 8)
        C_in = params.get("in_channels", 64)
        C_out = params.get("out_channels", 128)
        HW = params.get("image_size", 56)
        KS = params.get("kernel_size", 3)
        return bpe * (B * C_in * HW * HW + C_out * C_in * KS * KS + B * C_out * HW * HW)
    elif kernel_id == "embedding":
        S = params.get("seq_len", 512)
        D = params.get("embedding_dim", 768)
        return bpe * S * D * 2  # read embedding rows + write output
    return None


class BenchmarkRunner:
    def __init__(
        self,
        warmup_runs: int = 3,
        repeat_runs: int = 10,
        outlier_sigma: float = 3.0,
        enabled_backends: Optional[List[str]] = None,
        enabled_kernels: Optional[List[str]] = None,
        precisions: Optional[List[str]] = None,
    ):
        self.warmup_runs = warmup_runs
        self.repeat_runs = repeat_runs
        self.outlier_sigma = outlier_sigma
        self.enabled_backends = enabled_backends or list(ALL_BACKENDS.keys())
        self.enabled_kernels = enabled_kernels or [
            "matmul", "attention", "ffn", "layernorm", "conv2d", "embedding"
        ]
        self.precisions = precisions or ["fp32", "fp16"]

    def _kernel_param_sweep(self, kernel_id: str) -> List[dict]:
        """Return list of param dicts for each benchmark variant."""
        if kernel_id == "matmul":
            return [
                {"size": [512, 512, 512]},
                {"size": [1024, 1024, 1024]},
                {"size": [2048, 2048, 2048]},
            ]
        elif kernel_id == "attention":
            return [
                {"batch_size": b, "seq_len": s, "hidden_dim": 768, "num_heads": 12}
                for b in [1, 8, 32] for s in [128, 512]
            ]
        elif kernel_id == "ffn":
            return [
                {"batch_size": b, "seq_len": 512, "hidden_dim": 768, "ffn_dim": 3072}
                for b in [1, 8, 32]
            ]
        elif kernel_id == "layernorm":
            return [
                {"batch_size": b, "seq_len": 512, "hidden_dim": 768}
                for b in [1, 8, 32]
            ]
        elif kernel_id == "conv2d":
            return [
                {"batch_size": b, "in_channels": 64, "out_channels": 128,
                 "image_size": 56, "kernel_size": 3}
                for b in [1, 8]
            ]
        elif kernel_id == "embedding":
            return [
                {"vocab_size": 32000, "embedding_dim": 768, "seq_len": s}
                for s in [128, 512]
            ]
        return [{}]

    def run(self, label: Optional[str] = None) -> BenchmarkRun:
        platform_info = _get_platform_info()
        run = BenchmarkRun(
            label=label,
            **platform_info,
        )

        total = (
            len(self.enabled_backends)
            * len(self.enabled_kernels)
            * len(self.precisions)
        )
        console.print(
            f"\n[bold]Pyrex Benchmark[/bold] — {total} configurations\n"
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            console=console,
        ) as progress:
            task = progress.add_task("Benchmarking...", total=total)

            for backend_id in self.enabled_backends:
                backend = ALL_BACKENDS.get(backend_id)
                if backend is None or not backend.available:
                    console.print(
                        f"  [yellow]⚠ {backend_id} not available, skipping[/yellow]"
                    )
                    for _ in self.enabled_kernels:
                        for _ in self.precisions:
                            progress.advance(task)
                    continue

                for kernel_id in self.enabled_kernels:
                    for precision in self.precisions:
                        result = self._bench_one(
                            backend, kernel_id, precision, run.run_id
                        )
                        run.results.append(result)
                        progress.advance(task)
                        logger.debug(
                            "bench.done",
                            backend=backend_id,
                            kernel=kernel_id,
                            precision=precision,
                            p50=round(result.latency.p50_ms, 2)
                            if result.error is None else "err",
                        )

        run.finished_at = datetime.utcnow()
        run.total_seconds = (
            run.finished_at - run.started_at
        ).total_seconds()
        return run

    def _bench_one(
        self,
        backend: BackendBase,
        kernel_id: str,
        precision: str,
        run_id: str,
    ) -> KernelResult:
        param_sets = self._kernel_param_sweep(kernel_id)
        # For single-result per kernel_id/backend/precision,
        # use first param set. Full sweep stores multiple results.
        params = param_sets[0] if param_sets else {}

        try:
            ctx = backend.prepare(kernel_id, params, precision)
            tele_before = telemetry.snapshot()
            timings = backend.time_kernel(
                ctx,
                warmup_runs=self.warmup_runs,
                repeat_runs=self.repeat_runs,
                outlier_sigma=self.outlier_sigma,
            )
            tele_after = telemetry.snapshot()

            stats = _compute_stats(timings)
            flops = _estimate_flops(kernel_id, params)
            bytes_ = _estimate_bytes(kernel_id, params, precision)
            ai = (flops / bytes_) if (flops and bytes_) else None
            tflops = (flops / (stats.mean_ms / 1000) / 1e12) if flops else None

            return KernelResult(
                run_id=run_id,
                kernel_id=kernel_id,
                backend_id=backend.backend_id,
                precision=precision,
                params=params,
                latency=stats,
                throughput_ops_per_sec=tflops,
                flops=flops,
                bytes_transferred=bytes_,
                arithmetic_intensity=ai,
                telemetry=TelemetrySnapshot(
                    mps_memory_mb=tele_after.get("mps_memory_mb"),
                    cpu_memory_mb=tele_after.get("cpu_memory_mb"),
                    power_watts=tele_after.get("power_watts"),
                    cpu_percent=tele_after.get("cpu_percent"),
                ),
            )
        except Exception as exc:
            logger.warning(
                "bench.error",
                backend=backend.backend_id,
                kernel=kernel_id,
                precision=precision,
                error=str(exc),
            )
            empty_stats = LatencyStats(
                mean_ms=0, std_ms=0, p50_ms=0,
                p95_ms=0, p99_ms=0, min_ms=0, max_ms=0,
            )
            return KernelResult(
                run_id=run_id,
                kernel_id=kernel_id,
                backend_id=backend.backend_id,
                precision=precision,
                params=params,
                latency=empty_stats,
                error=str(exc),
            )

    def quick_run(self, label: Optional[str] = None) -> BenchmarkRun:
        """Fast subset for CI: only matmul + attention, fp32, MPS + CPU."""
        orig_kernels = self.enabled_kernels
        orig_backends = self.enabled_backends
        orig_prec = self.precisions
        self.enabled_kernels = ["matmul", "attention"]
        self.enabled_backends = ["pytorch_mps", "pytorch_cpu"]
        self.precisions = ["fp32"]
        run = self.run(label=label or "ci-quick")
        self.enabled_kernels = orig_kernels
        self.enabled_backends = orig_backends
        self.precisions = orig_prec
        return run
