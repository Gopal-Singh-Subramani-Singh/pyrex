from __future__ import annotations
import numpy as np
import pytest
import torch
from unittest.mock import MagicMock, patch

from pyrex.models import (
    BenchmarkRun, KernelResult, LatencyStats, TelemetrySnapshot
)


def make_latency_stats(mean=10.0, std=0.5, p50=9.8, p95=11.0, p99=12.0):
    return LatencyStats(
        mean_ms=mean, std_ms=std, p50_ms=p50,
        p95_ms=p95, p99_ms=p99,
        min_ms=mean - std * 2,
        max_ms=mean + std * 2,
        raw_ms=[mean] * 10,
    )


def make_kernel_result(
    kernel_id="matmul",
    backend_id="pytorch_mps",
    precision="fp32",
    mean_ms=10.0,
    error=None,
    flops=1e9,
    bytes_transferred=1e7,
):
    ai = flops / bytes_transferred if (flops and bytes_transferred) else None
    return KernelResult(
        run_id="test-run",
        kernel_id=kernel_id,
        backend_id=backend_id,
        precision=precision,
        params={"size": [512, 512, 512]},
        latency=make_latency_stats(mean=mean_ms),
        error=error,
        flops=flops,
        bytes_transferred=bytes_transferred,
        arithmetic_intensity=ai,
        throughput_ops_per_sec=(flops / (mean_ms / 1000) / 1e12) if flops else None,
    )


def make_run(run_id="run-001", results=None):
    return BenchmarkRun(
        run_id=run_id,
        chip="Apple M4",
        torch_version=torch.__version__,
        platform="macOS",
        results=results or [make_kernel_result()],
        total_seconds=30.0,
    )


@pytest.fixture
def sample_run():
    return make_run()


@pytest.fixture
def baseline_run():
    return make_run(
        run_id="baseline",
        results=[
            make_kernel_result("matmul", "pytorch_mps", "fp32", mean_ms=10.0),
            make_kernel_result("attention", "pytorch_mps", "fp32", mean_ms=20.0),
            make_kernel_result("matmul", "mlx", "fp32", mean_ms=8.0),
        ],
    )


@pytest.fixture
def current_run_ok():
    return make_run(
        run_id="current-ok",
        results=[
            make_kernel_result("matmul", "pytorch_mps", "fp32", mean_ms=10.2),
            make_kernel_result("attention", "pytorch_mps", "fp32", mean_ms=20.5),
            make_kernel_result("matmul", "mlx", "fp32", mean_ms=7.9),
        ],
    )


@pytest.fixture
def current_run_regression():
    return make_run(
        run_id="current-regression",
        results=[
            make_kernel_result("matmul", "pytorch_mps", "fp32", mean_ms=13.5),  # +35% regression
            make_kernel_result("attention", "pytorch_mps", "fp32", mean_ms=20.5),
            make_kernel_result("matmul", "mlx", "fp32", mean_ms=7.9),
        ],
    )
