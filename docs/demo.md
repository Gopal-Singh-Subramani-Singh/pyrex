# Pyrex — Demo Guide

## What this demo proves

- Cross-backend benchmarking runs correctly (MPS, CPU, ONNX, MLX)
- Regression detection correctly flags performance regressions
- Roofline chart is generated
- HTML report is generated and viewable
- DuckDB result store persists run history
- GitHub Actions CI gates PRs on regressions

---

## Prerequisites

```bash
pip install -r requirements.txt
pip install -e .

# Optional: MLX for Apple Silicon
pip install mlx
```

---

## Demo Commands

### 1. Run a quick benchmark

```bash
pyrex run --quick
```

Expected output:
```
Running benchmark: 2 backends × 2 kernels × 4 batch sizes × 1 precision
[MPS ] matmul   b=1   fp32 → p50=4.2ms  p99=5.1ms  mem=12MB
[MPS ] attention b=1  fp32 → p50=8.9ms  p99=10.2ms mem=18MB
[CPU ] matmul   b=1   fp32 → p50=12.3ms p99=14.1ms mem=8MB
[CPU ] attention b=1  fp32 → p50=31.2ms p99=35.0ms mem=9MB
Run saved: run_20240101_120000 (4 configs in 8.3s)
```

### 2. Save as baseline

```bash
pyrex baseline --name baseline
```

### 3. Run full benchmark

```bash
pyrex run
```

### 4. Compare to baseline (regression detection)

```bash
pyrex compare baseline <run_id>
```

Expected output when no regression:
```
Comparing run_20240101_120000 vs baseline
No regressions detected (all deltas within 5% or z-score < 2.0)
```

Expected output with regression:
```
REGRESSION DETECTED:
  matmul MPS b=32 fp32: +18.3% (z=3.2) ← regression
```

### 5. Generate HTML report

```bash
pyrex report <run_id>
open results/<run_id>/report.html
```

Report includes: per-kernel latency charts, roofline model, backend comparison table, regression summary.

Screenshot pending.

### 6. List all runs

```bash
pyrex list
```

### 7. Run tests

```bash
pytest tests/ -v
```

---

## Expected Output Summary

| Check | Expected |
|-------|----------|
| `pyrex run --quick` | Benchmark completes, results saved to DuckDB |
| `pyrex baseline` | Baseline JSON saved to `baselines/` |
| `pyrex compare` | Regression report printed (pass or fail) |
| `pyrex report` | HTML report with charts generated |
| `pyrex list` | All past runs listed with metadata |

---

## Known Limitations

- Full benchmark (all backends) requires Apple Silicon for MPS and MLX. CPU benchmarks run on any machine.
- `powermetrics` power telemetry requires passwordless sudo — skipped gracefully if unavailable.
- MLX backend is optional — graceful fallback if not installed.
- ONNX Runtime attention kernel is not available for all configurations — shows N/A.
- Benchmark results vary by thermal state and background load. Pyrex includes warmup iterations but single-run precision is limited.
