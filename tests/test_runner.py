from __future__ import annotations
import pytest
from unittest.mock import MagicMock, patch
from pyrex.runner import BenchmarkRunner, _compute_stats


def test_compute_stats_basic():
    timings = [10.0, 11.0, 9.0, 10.5, 10.2, 9.8, 10.1, 11.5, 9.5, 10.3]
    stats = _compute_stats(timings)
    assert stats.mean_ms > 0
    assert stats.std_ms >= 0
    assert stats.p50_ms <= stats.p95_ms <= stats.p99_ms
    assert stats.min_ms <= stats.mean_ms <= stats.max_ms


def test_compute_stats_single_value():
    stats = _compute_stats([5.0])
    assert stats.mean_ms == 5.0
    assert stats.p50_ms == 5.0


def test_runner_init():
    runner = BenchmarkRunner(
        warmup_runs=2,
        repeat_runs=5,
        enabled_backends=["pytorch_cpu"],
        enabled_kernels=["matmul"],
        precisions=["fp32"],
    )
    assert runner.warmup_runs == 2
    assert runner.repeat_runs == 5
    assert "pytorch_cpu" in runner.enabled_backends


def test_runner_bench_one_cpu():
    runner = BenchmarkRunner(
        warmup_runs=1,
        repeat_runs=3,
        enabled_backends=["pytorch_cpu"],
        enabled_kernels=["matmul"],
        precisions=["fp32"],
    )
    from pyrex.backends.pytorch_cpu import PyTorchCPUBackend
    backend = PyTorchCPUBackend()
    result = runner._bench_one(backend, "matmul", "fp32", "test-run")
    assert result.error is None
    assert result.latency.mean_ms > 0
    assert result.backend_id == "pytorch_cpu"


def test_runner_handles_backend_error():
    runner = BenchmarkRunner(warmup_runs=1, repeat_runs=3)
    from pyrex.backends.base import BackendBase

    class FailingBackend(BackendBase):
        @property
        def backend_id(self): return "failing"
        def prepare(self, k, p, prec): raise RuntimeError("intentional failure")
        def run_kernel(self, ctx): pass

    result = runner._bench_one(FailingBackend(), "matmul", "fp32", "r1")
    assert result.error is not None
    assert "intentional failure" in result.error


def test_quick_run_limits_scope():
    runner = BenchmarkRunner(
        warmup_runs=1,
        repeat_runs=2,
        enabled_backends=["pytorch_cpu"],
        enabled_kernels=["matmul", "attention", "ffn", "layernorm"],
        precisions=["fp32", "fp16"],
    )
    run = runner.quick_run(label="test-quick")
    kernel_ids = {r.kernel_id for r in run.results}
    assert "ffn" not in kernel_ids
    assert "matmul" in kernel_ids


def test_estimate_flops_matmul():
    from pyrex.runner import _estimate_flops
    flops = _estimate_flops("matmul", {"size": [512, 512, 512]})
    assert flops == 2.0 * 512 * 512 * 512


def test_estimate_flops_attention():
    from pyrex.runner import _estimate_flops
    params = {"batch_size": 1, "seq_len": 128, "hidden_dim": 768}
    flops = _estimate_flops("attention", params)
    assert flops == 4.0 * 1 * 128 * 128 * 768


def test_kernel_param_sweep():
    runner = BenchmarkRunner()
    matmul_params = runner._kernel_param_sweep("matmul")
    assert len(matmul_params) == 3
    attention_params = runner._kernel_param_sweep("attention")
    assert len(attention_params) == 6  # 3 batch sizes * 2 seq lengths
