from __future__ import annotations
import numpy as np
from typing import List, Optional
import structlog

from pyrex.models import BenchmarkRun, RegressionResult, CompareReport

logger = structlog.get_logger(__name__)


class RegressionDetector:
    def __init__(
        self,
        threshold_pct: float = 5.0,
        min_z_score: float = 2.0,
    ):
        self.threshold_pct = threshold_pct
        self.min_z_score = min_z_score

    def compare(
        self, baseline: BenchmarkRun, current: BenchmarkRun
    ) -> CompareReport:
        baseline_map = {
            (r.kernel_id, r.backend_id, r.precision): r
            for r in baseline.results
            if r.error is None
        }

        regressions = []
        improvements = []
        stable = []

        for cur in current.results:
            if cur.error is not None:
                continue
            key = (cur.kernel_id, cur.backend_id, cur.precision)
            base = baseline_map.get(key)
            if base is None:
                continue

            result = self._compare_one(base, cur)
            if result.is_regression:
                regressions.append(result)
            elif result.is_improvement:
                improvements.append(result)
            else:
                stable.append(result)

        return CompareReport(
            run_a_id=baseline.run_id,
            run_b_id=current.run_id,
            regressions=regressions,
            improvements=improvements,
            stable=stable,
            total_kernels=len(regressions) + len(improvements) + len(stable),
            regression_count=len(regressions),
            improvement_count=len(improvements),
        )

    def _compare_one(
        self, baseline_r, current_r
    ) -> RegressionResult:
        base_mean = baseline_r.latency.mean_ms
        base_std = baseline_r.latency.std_ms
        cur_mean = current_r.latency.mean_ms

        if base_mean == 0:
            delta_pct = 0.0
            z_score = 0.0
        else:
            delta_pct = ((cur_mean - base_mean) / base_mean) * 100.0
            z_score = (
                abs(cur_mean - base_mean) / base_std
                if base_std > 0
                else 0.0
            )

        is_regression = (
            delta_pct > self.threshold_pct
            and z_score >= self.min_z_score
        )
        is_improvement = (
            delta_pct < -self.threshold_pct
            and z_score >= self.min_z_score
        )

        if is_regression:
            severity = "critical" if delta_pct > 20 else "warning"
        else:
            severity = "ok"

        return RegressionResult(
            kernel_id=current_r.kernel_id,
            backend_id=current_r.backend_id,
            precision=current_r.precision,
            params=current_r.params,
            baseline_mean_ms=base_mean,
            current_mean_ms=cur_mean,
            delta_pct=round(delta_pct, 2),
            z_score=round(z_score, 2),
            is_regression=is_regression,
            is_improvement=is_improvement,
            severity=severity,
        )

    def check_ci(self, report: CompareReport) -> tuple[bool, str]:
        """Returns (passed, message) for CI gate."""
        if report.regression_count == 0:
            return True, f"✅ No regressions ({report.total_kernels} kernels checked)"

        lines = [
            f"❌ {report.regression_count} regression(s) detected:\n"
        ]
        for r in report.regressions:
            lines.append(
                f"  {r.kernel_id}/{r.backend_id}/{r.precision}: "
                f"{r.baseline_mean_ms:.2f}ms → {r.current_mean_ms:.2f}ms "
                f"(+{r.delta_pct:.1f}%, z={r.z_score:.1f})"
            )
        return False, "\n".join(lines)
