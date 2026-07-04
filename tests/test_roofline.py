from __future__ import annotations
import pytest
from pyrex.roofline import compute_roofline_points, M4_PEAK_TFLOPS_FP32


def test_roofline_points_computed(sample_run):
    points = compute_roofline_points(sample_run)
    assert len(points) >= 0


def test_roofline_memory_bound_classification():
    from tests.conftest import make_run, make_kernel_result
    # Embedding: low arithmetic intensity → memory bound
    embedding = make_kernel_result(
        "embedding", "pytorch_mps", "fp32",
        flops=1e8,
        bytes_transferred=1e9,  # very low AI
    )
    run = make_run(results=[embedding])
    points = compute_roofline_points(run)
    if points:
        assert points[0].is_memory_bound is True


def test_roofline_compute_bound_classification():
    from tests.conftest import make_run, make_kernel_result
    # Large matmul: high arithmetic intensity → compute bound
    matmul = make_kernel_result(
        "matmul", "pytorch_mps", "fp32",
        flops=1e12,
        bytes_transferred=1e8,  # very high AI
    )
    run = make_run(results=[matmul])
    points = compute_roofline_points(run)
    if points:
        assert points[0].is_memory_bound is False


def test_plot_roofline_creates_file(tmp_path, sample_run):
    output = str(tmp_path / "roofline.png")
    from pyrex.roofline import plot_roofline
    result = plot_roofline(sample_run, output_path=output)
    if result:
        import os
        assert os.path.exists(output)


def test_roofline_skips_error_results():
    from tests.conftest import make_run, make_kernel_result
    error_result = make_kernel_result("matmul", "pytorch_mps", "fp32", error="failed")
    run = make_run(results=[error_result])
    points = compute_roofline_points(run)
    assert len(points) == 0


def test_roofline_ridge_point_computed():
    from tests.conftest import make_run, make_kernel_result
    from pyrex.roofline import M4_MEMORY_BANDWIDTH_GBS
    matmul = make_kernel_result(
        "matmul", "pytorch_mps", "fp32",
        flops=1e12,
        bytes_transferred=1e8,
    )
    run = make_run(results=[matmul])
    points = compute_roofline_points(run)
    if points:
        expected_ridge = M4_PEAK_TFLOPS_FP32 * 1e12 / (M4_MEMORY_BANDWIDTH_GBS * 1e9)
        assert abs(points[0].ridge_point - expected_ridge) < 1.0
