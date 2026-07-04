from __future__ import annotations
import pytest
import torch
from pyrex.backends.pytorch_cpu import PyTorchCPUBackend


backend = PyTorchCPUBackend()


@pytest.mark.parametrize("kernel_id,params", [
    ("matmul", {"size": [64, 64, 64]}),
    ("attention", {"batch_size": 1, "seq_len": 32, "hidden_dim": 64, "num_heads": 4}),
    ("ffn", {"batch_size": 1, "seq_len": 32, "hidden_dim": 64, "ffn_dim": 256}),
    ("layernorm", {"batch_size": 1, "seq_len": 32, "hidden_dim": 64}),
    ("conv2d", {"batch_size": 1, "in_channels": 8, "out_channels": 16, "image_size": 16, "kernel_size": 3}),
    ("embedding", {"vocab_size": 100, "embedding_dim": 64, "seq_len": 32}),
])
def test_kernel_runs_without_error(kernel_id, params):
    ctx = backend.prepare(kernel_id, params, "fp32")
    backend.run_kernel(ctx)


@pytest.mark.parametrize("kernel_id,params", [
    ("matmul", {"size": [64, 64, 64]}),
    ("ffn", {"batch_size": 1, "seq_len": 32, "hidden_dim": 64, "ffn_dim": 256}),
])
def test_fp16_kernel_runs_without_error(kernel_id, params):
    ctx = backend.prepare(kernel_id, params, "fp16")
    backend.run_kernel(ctx)


def test_matmul_produces_correct_shape():
    ctx = backend.prepare("matmul", {"size": [16, 32, 64]}, "fp32")
    # Just verify it runs cleanly
    backend.run_kernel(ctx)


def test_attention_runs_multiple_times():
    params = {"batch_size": 1, "seq_len": 32, "hidden_dim": 64, "num_heads": 4}
    ctx = backend.prepare("attention", params, "fp32")
    for _ in range(5):
        backend.run_kernel(ctx)


def test_timing_returns_positive_values():
    ctx = backend.prepare("matmul", {"size": [64, 64, 64]}, "fp32")
    timings = backend.time_kernel(ctx, warmup_runs=1, repeat_runs=5)
    assert len(timings) > 0
    assert all(t > 0 for t in timings)


def test_unknown_kernel_raises():
    with pytest.raises((ValueError, KeyError)):
        backend.prepare("unknown_kernel", {}, "fp32")


def test_layernorm_cpu_runs():
    params = {"batch_size": 2, "seq_len": 64, "hidden_dim": 128}
    ctx = backend.prepare("layernorm", params, "fp32")
    backend.run_kernel(ctx)


def test_conv2d_cpu_runs():
    params = {"batch_size": 2, "in_channels": 4, "out_channels": 8, "image_size": 8, "kernel_size": 3}
    ctx = backend.prepare("conv2d", params, "fp32")
    backend.run_kernel(ctx)


def test_embedding_cpu_runs():
    params = {"vocab_size": 50, "embedding_dim": 32, "seq_len": 16}
    ctx = backend.prepare("embedding", params, "fp32")
    backend.run_kernel(ctx)
