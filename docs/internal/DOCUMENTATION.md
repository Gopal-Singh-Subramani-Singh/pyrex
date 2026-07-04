# Pyrex — In-Depth Documentation

## What Is Pyrex?

Pyrex is a production-grade cross-backend ML inference benchmark suite for Apple Silicon. It answers a practical question for anyone running ML on a Mac: **which combination of backend, kernel, batch size, and precision gives the best performance — and is it getting better or worse over time?**

It benchmarks four backends (PyTorch MPS, ONNX Runtime with CoreML, Apple MLX, and PyTorch CPU) across six kernel types, generates roofline charts, detects performance regressions via z-score analysis, produces standalone HTML reports, and integrates into GitHub Actions CI.

---

## What Pyrex Benchmarks

| Dimension | Values |
|---|---|
| Backends | `pytorch_mps` · `onnx_rt` · `mlx` · `pytorch_cpu` |
| Kernels | `attention` · `ffn` · `matmul` · `layernorm` · `conv2d` · `embedding` |
| Batch sizes | 1, 8, 32, 128 |
| Precisions | fp32, fp16 |

For each combination, Pyrex:
1. Warms up the backend (3 runs discarded)
2. Times 10 repetitions
3. Filters outliers (3σ)
4. Computes p50, p95, p99, mean, std
5. Estimates FLOPs, bytes transferred, and arithmetic intensity
6. Captures MPS memory, CPU memory, CPU%, and optionally power draw

---

## Architecture

```
CLI (Typer)
│
├── pyrex run
│     └── BenchmarkRunner
│           ├── for each backend × kernel × precision:
│           │     └── BackendBase.time_kernel()
│           │           ├── warmup N times
│           │           ├── time M repetitions
│           │           └── filter outliers (3σ)
│           ├── collect TelemetrySnapshot (MPS mem, CPU, power)
│           ├── estimate FLOPs + memory bytes
│           └── compute arithmetic intensity
│
├── pyrex baseline
│     └── ResultStore.save_baseline()
│
├── pyrex compare
│     └── RegressionDetector.compare()
│           ├── match baseline ↔ current by (kernel, backend, precision)
│           ├── compute delta% and z-score
│           └── classify: regression / improvement / stable
│
├── pyrex report
│     ├── RooflineAnalyser.plot_roofline()  → PNG
│     └── ReportGenerator.generate_html_report()  → HTML
│
└── ResultStore (DuckDB + JSON)
      ├── kernel_results table  ← all measurements
      └── runs table            ← run metadata
```

**Key components:**

| Component | File | Role |
|---|---|---|
| CLI | `pyrex/cli.py` | Typer commands: run, compare, report, baseline, list |
| Runner | `pyrex/runner.py` | Core timing loop, orchestrates backends and kernels |
| Telemetry | `pyrex/telemetry.py` | MPS memory, CPU memory, CPU%, powermetrics |
| Regression | `pyrex/regression.py` | Z-score regression detection |
| Roofline | `pyrex/roofline.py` | Arithmetic intensity chart |
| Report | `pyrex/report.py` | HTML report generation (Jinja2 + matplotlib) |
| Store | `pyrex/store.py` | DuckDB + JSON persistence |
| Models | `pyrex/models.py` | Pydantic result schemas |

---

## Project Structure

```
pyrex/
├── pyrex/
│   ├── cli.py               ← Typer CLI
│   ├── runner.py            ← Core benchmark timing loop
│   ├── telemetry.py         ← powermetrics, MPS memory, psutil
│   ├── regression.py        ← z-score regression detector
│   ├── roofline.py          ← Roofline chart generator
│   ├── report.py            ← HTML report (Jinja2)
│   ├── store.py             ← DuckDB + Parquet persistence
│   ├── models.py            ← Pydantic schemas
│   ├── backends/
│   │   ├── base.py          ← BackendBase ABC + timing loop
│   │   ├── pytorch_mps.py   ← PyTorch MPS (Metal GPU)
│   │   ├── pytorch_cpu.py   ← PyTorch CPU
│   │   ├── onnx_rt.py       ← ONNX Runtime + CoreML
│   │   └── mlx_backend.py   ← Apple MLX (graceful fallback)
│   └── kernels/
│       ├── attention.py     ← Scaled dot-product attention
│       ├── ffn.py           ← FFN: 2× linear + GELU
│       ├── matmul.py        ← General matrix multiply
│       ├── layernorm.py     ← Layer normalisation
│       ├── conv2d.py        ← 2D convolution (ResNet dims)
│       └── embedding.py     ← Token embedding lookup
├── config/
│   └── config.yaml          ← Benchmark configuration
├── baselines/               ← Baseline JSON snapshots
├── results/                 ← Run results (.json, .png, .html, .duckdb)
├── tests/                   ← 53 pytest tests
├── .github/workflows/
│   └── benchmark.yml        ← GitHub Actions CI
├── requirements.txt
└── pyproject.toml
```

---

## How to Run

### Prerequisites

- macOS (Apple Silicon strongly recommended — M1/M2/M3/M4)
- Python 3.11+

### Step 1 — Install dependencies

```bash
cd "/Users/gopalsinghsubramanisingh/Documents/AI  Hive/Pyrex/pyrex"
pip install -r requirements.txt
```

### Step 2 — Install the CLI

```bash
pip install -e .
```

Verify:

```bash
pyrex --help
```

### Step 3 — (Optional) Install Apple MLX

MLX is Apple Silicon-only and excluded from `requirements.txt` to avoid breaking non-Apple environments. Install separately if you want the 4th backend:

```bash
pip install mlx
```

If not installed, Pyrex skips the MLX backend cleanly — no errors.

---

## CLI Reference

### `pyrex run` — Run benchmarks

```bash
# Full benchmark suite (all backends, all kernels, fp32 + fp16)
pyrex run

# Label a run for tracking
pyrex run --label "after-optimization-v2"

# Quick CI subset (matmul + attention only, MPS + CPU, fp32)
pyrex run --quick

# Specific backends
pyrex run --backends pytorch_mps,pytorch_cpu

# Specific kernels
pyrex run --kernels matmul,attention,ffn

# Mix and match
pyrex run --quick --backends pytorch_cpu --kernels matmul

# Skip roofline chart generation (faster)
pyrex run --skip-roofline

# Don't save results to store
pyrex run --no-save
```

Available backend IDs: `pytorch_mps` · `pytorch_cpu` · `onnx_rt` · `mlx`

Available kernel IDs: `matmul` · `attention` · `ffn` · `layernorm` · `conv2d` · `embedding`

---

### `pyrex baseline` — Save a performance baseline

```bash
# Save the most recent run as baseline
pyrex baseline

# Save a specific run ID
pyrex baseline <run_id>

# Use a custom name
pyrex baseline --name "before-refactor"
pyrex baseline <run_id> --name "v1.2-stable"
```

Baselines are saved to `baselines/<name>.json`.

---

### `pyrex compare` — Detect regressions

```bash
# Compare against default baseline
pyrex compare baseline <run_id>

# Compare two specific runs
pyrex compare <run_id_a> <run_id_b>

# Use a named baseline
pyrex compare before-refactor <run_id>

# Custom regression threshold (default 5%)
pyrex compare baseline <run_id> --threshold 10.0

# Exit with code 1 if regressions found (for CI)
pyrex compare baseline <run_id> --fail
```

Output is a table with `delta%`, `z-score`, and `regression/improvement/stable` classification per kernel × backend × precision.

---

### `pyrex report` — Generate HTML report

```bash
# Generate report for a run
pyrex report <run_id>

# Custom output path
pyrex report <run_id> --output reports/my-report.html
```

Opens `results/<run_id>_report.html` — a standalone dark-themed HTML file with a summary grid, embedded roofline chart, and full latency table.

```bash
# Open in browser
open results/<run_id>_report.html
```

---

### `pyrex list` — Show recent runs

```bash
pyrex list

# Limit results
pyrex list --limit 5
```

---

## Recommended First-Run Workflow

```bash
# 1. Install
pip install -r requirements.txt
pip install -e .

# 2. Verify with quick run
pyrex run --quick --label "initial"

# 3. Save as baseline
pyrex baseline --name baseline

# 4. Make code changes, then run again
pyrex run --quick --label "after-change"

# 5. Compare
pyrex list   # get the new run ID
pyrex compare baseline <new_run_id>

# 6. Generate report
pyrex report <new_run_id>
open results/<new_run_id>_report.html

# 7. Full sweep across all backends
pyrex run --label "full-$(date +%Y%m%d)"
```

---

## Run Tests

```bash
pytest tests/ -v
```

Expected: **53 passed** in ~3 seconds.

| Test file | Tests | What's covered |
|---|---|---|
| `test_kernels.py` | 15 | All 6 kernels on CPU, fp32 + fp16 |
| `test_regression.py` | 8 | Regression/improvement detection, z-score, CI gate |
| `test_store.py` | 7 | DuckDB save/load, baseline, history |
| `test_runner.py` | 9 | Stats computation, bench loop, error handling |
| `test_roofline.py` | 6 | Memory-bound/compute-bound classification, PNG output |
| `test_telemetry.py` | 8 | CPU memory, CPU%, MPS memory, snapshot |

---

## How Regression Detection Works

A result is flagged as a regression only when **both** conditions are true simultaneously:

1. `delta% > 5.0%` — latency increased by more than the threshold
2. `z-score >= 2.0` — the change is statistically significant

Using both conditions prevents noise from triggering false positives. A 6% slowdown that's within 1σ of normal variance won't be flagged.

```
z-score = (current_mean - baseline_mean) / baseline_std
```

Severity levels:

| Level | Condition |
|---|---|
| `ok` | No regression |
| `warning` | 5–20% regression |
| `critical` | >20% regression |

---

## How the Roofline Model Works

The roofline chart plots each kernel's achieved performance (TFLOPS) against its **arithmetic intensity** (FLOPs per byte of memory transferred), overlaid with two theoretical ceilings:

- **Memory bandwidth ceiling**: `TFLOPS = bandwidth_GB/s × intensity_FLOPs/byte`
- **Compute ceiling**: M4 peak = 3.6 TFLOPS (fp32) / 7.2 TFLOPS (fp16)

The **ridge point** is where the two ceilings intersect:
```
ridge_point = peak_TFLOPS / memory_bandwidth_GBs
            = 3.6 / 120 ≈ 30 FLOPs/byte (M4)
```

- Kernels **left of the ridge** → memory-bound: more bandwidth helps, not more compute
- Kernels **right of the ridge** → compute-bound: faster chips help, not more bandwidth

Typical M4 classifications:

| Kernel | Bound | Reason |
|---|---|---|
| matmul (large) | Compute | High matrix reuse |
| attention (long seq) | Memory | Score matrix grows with S² |
| ffn | Compute | Two large linear projections |
| layernorm | Memory | Reads/writes full activation twice |
| embedding | Memory | Pure lookup table, no arithmetic |
| conv2d (large) | Compute | High filter reuse |

---

## Backend Details

### `pytorch_mps` — PyTorch Metal GPU

Uses Apple's Metal Performance Shaders via `torch.device("mps")`. Calls `torch.mps.synchronize()` after each kernel to ensure accurate wall-clock timing. Supports fp32 and fp16.

### `pytorch_cpu` — PyTorch CPU

Inherits all kernel implementations from the MPS backend, overrides device to `cpu`. CPU is synchronous — no explicit sync needed. Always available on any macOS machine.

### `onnx_rt` — ONNX Runtime + CoreML

Exports each kernel as a PyTorch module to ONNX (opset 17), then runs inference through ONNX Runtime with `CoreMLExecutionProvider` as the first choice, falling back to `CPUExecutionProvider`.

Full ONNX graphs are available for `matmul` and `ffn`. Other kernels (`attention`, `layernorm`, `conv2d`, `embedding`) use identity modules due to ONNX export complexity with dynamic shapes — they measure ORT session overhead rather than full kernel performance.

### `mlx` — Apple MLX

Apple's own ML framework using unified memory (no explicit host-device transfers). MLX uses lazy evaluation: `mx.eval()` forces materialisation and acts as a sync point for accurate timing. If MLX is not installed, `available = False` and Pyrex skips it without errors.

---

## Telemetry

| Metric | Source | Notes |
|---|---|---|
| Latency p50/p95/p99 | `time.perf_counter()` | Per kernel, per run |
| Latency mean / std | numpy | With 3σ outlier filtering |
| MPS memory (MB) | `torch.mps.current_allocated_memory()` | GPU allocation |
| CPU memory (MB) | `psutil` process RSS | Current process |
| CPU utilisation % | `psutil.cpu_percent()` | System-wide |
| Power draw (W) | macOS `powermetrics` | Requires passwordless sudo |

Power reading is optional. If `powermetrics` fails or sudo is not configured, Pyrex continues normally with `power_watts = None`.

To enable power readings, add to `/etc/sudoers` via `visudo`:

```
yourusername ALL=(ALL) NOPASSWD: /usr/bin/powermetrics
```

---

## CI Setup — GitHub Actions

Copy `.github/workflows/benchmark.yml` to your repository. The workflow:

1. Installs all dependencies
2. Runs the test suite: `pytest tests/ -v`
3. Runs `pyrex run --quick` and captures the run ID
4. Compares against `baselines/baseline.json` if it exists
5. Posts a regression table as a PR comment
6. Exits with code 1 if any regression is detected (blocks merge)
7. On merge to main: saves the run as the new baseline automatically
8. Uploads results as build artifacts (retained 30 days)

### Enable CI

```bash
# 1. Run first benchmark and save baseline
pyrex run --quick
pyrex baseline --name baseline

# 2. Commit the baseline
git add baselines/baseline.json
git commit -m "chore: add pyrex performance baseline"
git push
```

Every subsequent PR will now auto-benchmark and post a regression report.

---

## Sample Results (Apple M4, 24GB)

| Kernel | MPS p50 | MLX p50 | ONNX p50 | CPU p50 |
|---|---|---|---|---|
| matmul | 4.21 ms | 3.18 ms | 5.40 ms | 12.3 ms |
| attention | 8.90 ms | 9.10 ms | N/A | 31.2 ms |
| ffn | 2.90 ms | 2.70 ms | 3.10 ms | 9.80 ms |

Run `pyrex run` for actual numbers on your hardware.

---

## Configuration Reference

`config/config.yaml`:

```yaml
benchmark:
  warmup_runs: 3          # runs before timing (discarded)
  repeat_runs: 10         # timed runs per config
  min_repeat_ms: 100      # minimum valid measurement time
  outlier_sigma: 3.0      # σ threshold for outlier removal

backends:
  - id: "pytorch_mps"
    enabled: true
  - id: "pytorch_cpu"
    enabled: true
  - id: "onnx_rt"
    enabled: true
  - id: "mlx"
    enabled: true

regression:
  threshold_pct: 5.0      # % change to flag
  min_z_score: 2.0        # minimum statistical significance
  min_samples: 3          # minimum prior samples needed

roofline:
  peak_tflops_fp32: 3.6   # M4 FP32 theoretical peak
  peak_tflops_fp16: 7.2   # M4 FP16 theoretical peak
  memory_bandwidth_gbs: 120.0  # M4 memory bandwidth

store:
  results_dir: "results"
  baselines_dir: "baselines"
  db_path: "results/pyrex.duckdb"
```

---

## Known Constraints

**typer pinned to 0.12.3**: `typer==0.12.5` has a bug with `click>=8.3` that raises a `TypeError` when rendering boolean flag help text. Pyrex uses `typer==0.12.3` + `click==8.1.8` — identical API, fully functional.

**MLX excluded from requirements.txt**: Apple Silicon-only package. Install with `pip install mlx` when needed.

**ONNX kernels are partial**: Full ONNX export for dynamic multi-head attention requires significant customisation. Matmul and FFN run through CoreML; other kernels use identity modules.

**powermetrics requires sudo**: Power readings are optional. Standard benchmarks work without it.

---

## Makefile Reference

```bash
make install      # pip install -r requirements.txt && pip install -e .
make run          # pyrex run (full suite)
make quick        # pyrex run --quick --label ci-YYYYMMDD
make baseline     # pyrex baseline --name baseline
make list         # pyrex list
make test         # pytest tests/ -v
make test-cov     # tests + coverage report
make clean        # remove __pycache__ and *.pyc
```

For compare and report:
```bash
make compare RUN=<run_id>
make report RUN=<run_id>
```

---

## Prometheus Metrics

Pyrex doesn't serve a live Prometheus endpoint — it's a CLI tool that runs to completion. For CI integration, the comparison output and exit codes are the primary signal.

For long-running performance tracking, use the DuckDB store:

```python
import duckdb
con = duckdb.connect("results/pyrex.duckdb")
# All historical results
df = con.execute("SELECT * FROM kernel_results ORDER BY benchmarked_at DESC").df()
# Regression trend for matmul on MPS
df = con.execute("""
    SELECT run_id, benchmarked_at, latency_p50_ms 
    FROM kernel_results 
    WHERE kernel_id='matmul' AND backend_id='pytorch_mps' AND precision='fp32'
    ORDER BY benchmarked_at
""").df()
```

The optional Prometheus + Grafana stack (`docker compose up prometheus grafana -d`) can scrape any custom metrics you expose.

---

## Production Hardening (CI Integration)

### Setting up GitHub Actions

The workflow at `.github/workflows/benchmark.yml` runs on every PR:

1. Installs dependencies and the CLI
2. Runs `pyrex run --quick`
3. Compares against the committed `baselines/baseline.json`
4. Comments the regression table on the PR
5. Exits with code 1 if any regression exceeds 5% AND z-score ≥ 2.0

To enable:

```bash
# One-time: establish baseline on your machine
pyrex run --quick
pyrex baseline --name baseline
git add baselines/baseline.json
git commit -m "chore: add pyrex performance baseline"
git push
```

### Tightening regression thresholds for sensitive kernels

Edit `config/config.yaml`:

```yaml
regression:
  threshold_pct: 3.0      # flag regressions above 3% (default 5%)
  min_z_score: 1.5        # lower z-score threshold (more sensitive)
  min_samples: 5          # require more samples before flagging
```

### Running as a pre-merge gate

```bash
# In your CI script:
pyrex run --quick --no-save
pyrex compare baseline $(pyrex list --limit 1 --format id) --fail
# exits 1 if regression detected, blocking the merge
```

### Storing baselines per branch

For long-lived feature branches, maintain per-branch baselines:

```bash
pyrex baseline --name "feature-$(git rev-parse --abbrev-ref HEAD)"
pyrex compare "feature-$(git rev-parse --abbrev-ref HEAD)" <run_id>
```

---

## Troubleshooting

### `pyrex: command not found`

Install the package in development mode:

```bash
pip install -e .
# Verify:
which pyrex
pyrex --help
```

### `MPS backend unavailable`

On Intel Macs or non-Apple hardware, `pytorch_mps` will be skipped automatically. Run with CPU and ONNX only:

```bash
pyrex run --backends pytorch_cpu,onnx_rt
```

### `MLX backend skipped`

Install MLX separately:

```bash
pip install mlx
```

If on a non-Apple Silicon machine, MLX cannot be installed — it's Apple-only. Pyrex handles this gracefully.

### `z-score NaN` in regression output

The baseline doesn't have enough samples to compute a standard deviation (`min_samples` in config, default 3). Run the baseline a few times to accumulate samples before relying on z-score regression detection.

### HTML report missing roofline chart

The roofline chart requires `matplotlib`. If it's not installed:

```bash
pip install matplotlib
```

Or skip it during the run:

```bash
pyrex run --skip-roofline
```

### ONNX export fails for a kernel

Some kernels (attention, layernorm, conv2d, embedding) use identity modules in the ONNX backend due to dynamic shape complexity. This is by design — those ONNX results measure session overhead rather than true kernel performance. Use `pytorch_mps` or `mlx` for accurate attention/conv2d measurements.

### Power readings always `None`

Power telemetry requires passwordless `sudo` for `powermetrics`. Add to `/etc/sudoers` via `visudo`:

```
yourusername ALL=(ALL) NOPASSWD: /usr/bin/powermetrics
```

All other telemetry (latency, memory, CPU%) continues normally without power readings.

### DuckDB results database grows large

Old results accumulate in `results/pyrex.duckdb`. To prune:

```python
import duckdb
con = duckdb.connect("results/pyrex.duckdb")
# Keep only last 90 days
con.execute("DELETE FROM kernel_results WHERE benchmarked_at < NOW() - INTERVAL 90 DAY")
con.execute("DELETE FROM runs WHERE started_at < NOW() - INTERVAL 90 DAY")
con.close()
```
