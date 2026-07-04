from __future__ import annotations
import pytest
from pyrex.regression import RegressionDetector


def test_no_regression_stable_run(baseline_run, current_run_ok):
    detector = RegressionDetector(threshold_pct=5.0, min_z_score=2.0)
    report = detector.compare(baseline_run, current_run_ok)
    assert report.regression_count == 0


def test_detects_regression(baseline_run, current_run_regression):
    detector = RegressionDetector(threshold_pct=5.0, min_z_score=0.0)
    report = detector.compare(baseline_run, current_run_regression)
    assert report.regression_count >= 1
    assert any(r.kernel_id == "matmul" for r in report.regressions)


def test_detects_improvement():
    from tests.conftest import make_run, make_kernel_result
    base = make_run(results=[make_kernel_result("matmul", "mlx", "fp32", mean_ms=20.0)])
    cur = make_run(results=[make_kernel_result("matmul", "mlx", "fp32", mean_ms=10.0)])
    detector = RegressionDetector(threshold_pct=5.0, min_z_score=0.0)
    report = detector.compare(base, cur)
    assert report.improvement_count >= 1


def test_ci_check_pass(baseline_run, current_run_ok):
    detector = RegressionDetector()
    report = detector.compare(baseline_run, current_run_ok)
    passed, msg = detector.check_ci(report)
    assert passed is True
    assert "✅" in msg


def test_ci_check_fail(baseline_run, current_run_regression):
    detector = RegressionDetector(threshold_pct=5.0, min_z_score=0.0)
    report = detector.compare(baseline_run, current_run_regression)
    passed, msg = detector.check_ci(report)
    assert passed is False
    assert "❌" in msg


def test_delta_pct_calculation():
    from tests.conftest import make_run, make_kernel_result
    base = make_run(results=[make_kernel_result("matmul", "pytorch_mps", "fp32", mean_ms=10.0)])
    cur = make_run(results=[make_kernel_result("matmul", "pytorch_mps", "fp32", mean_ms=12.0)])
    detector = RegressionDetector(threshold_pct=5.0, min_z_score=0.0)
    report = detector.compare(base, cur)
    result = report.regressions[0] if report.regressions else report.stable[0]
    assert abs(result.delta_pct - 20.0) < 0.1


def test_missing_baseline_kernel_skipped():
    from tests.conftest import make_run, make_kernel_result
    base = make_run(results=[make_kernel_result("matmul", "pytorch_mps", "fp32")])
    cur = make_run(results=[
        make_kernel_result("matmul", "pytorch_mps", "fp32"),
        make_kernel_result("ffn", "pytorch_mps", "fp32"),  # not in baseline
    ])
    detector = RegressionDetector()
    report = detector.compare(base, cur)
    total = report.regression_count + report.improvement_count + len(report.stable)
    assert total == 1  # only matmul compared


def test_severity_critical_for_large_regression():
    from tests.conftest import make_run, make_kernel_result
    base = make_run(results=[make_kernel_result("matmul", "pytorch_mps", "fp32", mean_ms=10.0)])
    cur = make_run(results=[make_kernel_result("matmul", "pytorch_mps", "fp32", mean_ms=15.0)])
    detector = RegressionDetector(threshold_pct=5.0, min_z_score=0.0)
    report = detector.compare(base, cur)
    assert len(report.regressions) >= 1
    assert report.regressions[0].severity in ("warning", "critical")
