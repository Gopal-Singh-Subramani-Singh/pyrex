# Pyrex — Full Technical Report

## What Was Built

Pyrex is a production-grade CLI benchmark harness for ML inference on Apple Silicon. It measures latency across 4 backends and 6 kernel types, stores all results in DuckDB, detects performance regressions via z-score analysis, generates roofline charts, and integrates into GitHub Actions CI.

---

## System Status

| Item | Value |
|------|-------|
| Python | 3.11.9 |
| PyTorch | 2.11.0 |
| ONNX Runtime | 1.19.2 |
| DuckDB | 1.1.1 |
| Pydantic | 2.9.2 |
| Typer | 0.12.3 |
| MPS (Apple GPU) | ✅ Available |
| CoreML Execution Provider | ✅ Available |
| MLX | ❌ Not installed (graceful fallback) |
| Tests | **53/53 passing** |

---

## Complete File Tree

```
pyrex/
├── pyrex/                          ← Python package
│   ├── __init__.py
│   ├── cli.py          (259 lines) ← Typer CLI: run, compare, report, baseline, list
│   ├── runner.py       (295 lines) ← Core benchmark timing loop
│   ├── telemetry.py     (73 lines) ← powermetrics, MPS memory, psutil
│   ├── regression.py   (120 lines) ← z-score regression detector
│   ├── roofline.py     (142 lines) ← Roofline chart generator
│   ├── report.py       (138 lines) ← HTML report generator (Jinja2)
│   ├── store.py        (162 lines) ← DuckDB + JSON persistence
│   ├── models.py        (90 lines) ← Pydantic result schemas
│   ├── backends/
│   │   ├── __init__.py
│   │   ├── base.py                 ← BackendBase ABC + timing loop
│   │   ├── pytorch_mps.py          ← PyTorch MPS (Metal GPU) backend
│   │   ├── pytorch_cpu.py          ← PyTorch CPU backend
│   │   ├── onnx_rt.py              ← ONNX Runtime + CoreML backend
│   │   └── mlx_backend.py          ← Apple MLX backend (graceful fallback)
│   └── kernels/
│       ├── __init__.py
│       ├── attention.py            ← Scaled dot-product attention
│       ├── ffn.py                  ← FFN: 2× linear + GELU
│       ├── matmul.py               ← General matrix multiply
│       ├── layernorm.py            ← Layer normalisation
│       ├── conv2d.py               ← 2D convolution (ResNet dims)
│       └── embedding.py            ← Token embedding lookup
├── config/
│   └── config.yaml                 ← Benchmark configuration
├── baselines/
│   └── baseline.json               ← Performance baseline snapshots
├── results/
│   ├── pyrex.duckdb                ← Full run history database
│   ├── <run_id>.json               ← Per-run result files
│   ├── <run_id>_roofline.png       ← Roofline charts
│   └── <run_id>_report.html        ← HTML reports
├── tests/
│   ├── conftest.py                 ← Shared fixtures
│   ├── test_runner.py              ← 9 tests
│   ├── test_regression.py          ← 8 tests
│   ├── test_roofline.py            ← 6 tests
│   ├── test_store.py               ← 7 tests
│   ├── test_kernels.py             ← 15 tests
│   └── test_telemetry.py           ← 8 tests
├── .github/workflows/
│   └── benchmark.yml               ← GitHub Actions CI
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Installation

From the `pyrex/` project directory:

```bash
# Step 1 — install core dependencies
pip install -r requirements.txt

# Step 2 — install the pyrex CLI tool
pip install -e .

# Step 3 — optional: install Apple MLX for the 4th backend
#           (Apple Silicon only; pyrex works fine without it)
pip install mlx
```

Verify the install:

```bash
pyrex --help
```

---

## Running Tests

```bash
pytest tests/ -v
```

Expected output: **53 passed** in ~3 seconds.

Test coverage breakdown:

| File | Tests | What's covered |
|------|-------|----------------|
| `test_kernels.py` | 15 | All 6 kernels on CPU, fp32 + fp16 |
| `test_regression.py` | 8 | Regression/improvement detection, z-score, CI gate |
| `test_store.py` | 7 | DuckDB save/load, baseline, history query |
| `test_runner.py` | 9 | Stats computation, bench loop, error handling, param sweeps |
| `test_roofline.py` | 6 | Memory-bound/compute-bound classification, PNG output |
| `test_telemetry.py` | 8 | CPU memory, CPU%, MPS memory, snapshot |

---

## CLI Commands — Full Reference

### `pyrex run` — Run benchmarks

```bash
# Full benchmark suite (all backends, all kernels, fp32 + fp16)
pyrex run

# With a human-readable label
pyrex run --label "after-my-optimization"

# Quick CI subset (matmul + attention only, MPS + CPU, fp32 only)
pyrex run --quick

# Specific backends (comma-separated)
pyrex run --backends pytorch_mps,pytorch_cpu

# Specific kernels (comma-separated)
pyrex run --kernels matmul,attention,ffn

# Mix and match
pyrex run --quick --backends pytorch_cpu --kernels matmul --label "test"

# Skip roofline chart generation
pyrex run --skip-roofline

# Don't persist results to store
pyrex run --no-save
```

Available backend IDs: `pytorch_mps` · `pytorch_cpu` · `onnx_rt` · `mlx`

Available kernel IDs: `matmul` · `attention` · `ffn` · `layernorm` · `conv2d` · `embedding`

---

### `pyrex baseline` — Save a baseline for regression detection

```bash
# Save the most recent run as baseline (default name: "baseline")
pyrex baseline

# Save a specific run ID as baseline
pyrex baseline <run_id>

# Use a custom name
pyrex baseline --name "before-refactor"
pyrex baseline <run_id> --name "v1.2-stable"
```

Baselines are stored in `baselines/<name>.json`.

---

### `pyrex compare` — Detect regressions between two runs

```bash
# Compare against the saved baseline
pyrex compare baseline <run_id>

# Compare two specific run IDs
pyrex compare <run_id_a> <run_id_b>

# Use a named baseline
pyrex compare before-refactor <run_id>

# Custom regression threshold (default 5%)
pyrex compare baseline <run_id> --threshold 10.0

# Exit with code 1 if regressions found (for CI scripts)
pyrex compare baseline <run_id> --fail
```

The compare output prints a table with delta%, z-score, and an improvement/regression/stable label for every kernel × backend × precision combination.

---

### `pyrex report` — Generate HTML report

```bash
# Generate report for a run
pyrex report <run_id>

# Custom output path
pyrex report <run_id> --output reports/my-report.html
```

Output: `results/<run_id>_report.html` — a standalone dark-themed HTML file with:

- Summary grid (total benchmarks, backends tested, kernels tested, total duration)
- Roofline chart image embedded inline
- Full latency table with p50, p95, p99, std, and arithmetic intensity

---

### `pyrex list` — Show recent runs

```bash
pyrex list

# Limit number of results shown
pyrex list --limit 5
```

---

## Full Workflow — From Scratch

```bash
# 1. Install
pip install -r requirements.txt
pip install -e .

# 2. Run the quick benchmark to verify everything works
pyrex run --quick --label "initial"

# 3. Save this as your performance baseline
pyrex baseline --name baseline

# 4. Make your code changes... then run again
pyrex run --quick --label "after-change"

# 5. Get the new run ID
pyrex list

# 6. Compare against baseline
pyrex compare baseline <new_run_id>

# 7. Generate a full report
pyrex report <new_run_id>

# 8. Open the HTML report in your browser
open results/<new_run_id>_report.html

# 9. Run a complete sweep across all available backends
pyrex run --label "full-sweep-$(date +%Y%m%d)"
```

---

## Recommended First Full Run

```bash
pyrex run --label "m4-baseline-$(date +%Y%m%d)"
pyrex baseline
```

On your M4 with MPS + CoreML available, this runs:

| Backend | Configs | Notes |
|---------|---------|-------|
| `pytorch_mps` | 6 kernels × 2 precisions = 12 | Metal GPU |
| `pytorch_cpu` | 6 kernels × 2 precisions = 12 | CPU fallback |
| `onnx_rt` | 6 kernels × 2 precisions = 12 | CoreML provider |
| `mlx` | — | Skipped if not installed |

Total: ~36 configurations. Expect 3–8 minutes depending on matrix sizes.

---

## CI Setup — GitHub Actions

The workflow at `.github/workflows/benchmark.yml` runs automatically on every PR and push to `main`.

### What it does

1. Installs all dependencies with `pip install -r requirements.txt`
2. Installs the package with `pip install -e .`
3. Runs the full test suite: `pytest tests/ -v`
4. Runs `pyrex run --quick` and captures the run ID
5. If `baselines/baseline.json` exists: runs `pyrex compare baseline <run_id>` and saves output
6. On merge to `main`: saves the run as the new baseline automatically
7. Posts a comment on the PR with the full comparison table
8. Fails the workflow if any regression is detected (`exit 1`)
9. Uploads results and baselines as build artifacts (retained 30 days)

### Enabling it in your repo

```bash
# Run your first benchmark
pyrex run --quick
pyrex baseline --name baseline

# Commit the baseline so CI can compare against it
git add baselines/baseline.json
git commit -m "chore: add pyrex performance baseline"
git push
```

From that point, every PR automatically benchmarks and posts a regression report as a comment.

---

## How Regression Detection Works

A result is flagged as a regression only when **both** conditions are true simultaneously:

- `delta% > 5.0%` — latency increased by more than the threshold
- `z-score >= 2.0` — the change is statistically significant relative to the baseline's standard deviation

Using both conditions prevents noise from triggering false positives. A 6% slowdown that's within 1σ of normal variance won't be flagged.

Configuration in `config/config.yaml`:

```yaml
regression:
  threshold_pct: 5.0    # minimum % change to flag
  min_z_score: 2.0      # minimum statistical significance
  min_samples: 3        # minimum samples needed
```

Severity levels:

| Level | Condition |
|-------|-----------|
| `ok` | No regression |
| `warning` | Regression between 5% and 20% |
| `critical` | Regression over 20% |

---

## How the Roofline Model Works

The roofline chart plots each kernel's achieved performance (TFLOPS) against its arithmetic intensity (FLOPs per byte of memory moved). It overlays two theoretical ceilings:

- **Memory bandwidth ceiling**: `achieved_TFLOPS = bandwidth_GB/s × arithmetic_intensity`
- **Compute ceiling**: M4 peak = 3.6 TFLOPS (fp32) / 7.2 TFLOPS (fp16)

Kernels to the **left of the ridge point** are memory-bound — they would benefit more from higher memory bandwidth than more compute. Kernels to the **right** are compute-bound.

On the M4, the ridge point falls at approximately **30 FLOPs/byte** (`3.6 TFLOPS ÷ 120 GB/s`).

Typical classifications on M4:

| Kernel | Bound | Reason |
|--------|-------|--------|
| matmul (large) | Compute | High reuse of A and B matrices |
| attention (long seq) | Memory | Scores matrix grows with S² |
| ffn | Compute | Two large linear projections |
| layernorm | Memory | Reads/writes full activation twice |
| embedding | Memory | Pure lookup, no arithmetic |
| conv2d | Compute (large) | High filter reuse |

---

## Telemetry Collected Per Run

| Metric | Source | Notes |
|--------|--------|-------|
| Latency p50/p95/p99 | `time.perf_counter()` | Per-kernel, per-run |
| Latency mean / std | numpy | With outlier filtering (3σ) |
| MPS memory (MB) | `torch.mps.current_allocated_memory()` | GPU allocation |
| CPU memory (MB) | `psutil` process RSS | Current process only |
| CPU utilisation % | `psutil.cpu_percent()` | System-wide |
| Power draw (W) | macOS `powermetrics` | Requires passwordless sudo |

Power reading is optional — if `powermetrics` is unavailable or sudo is not configured, it silently returns `None` and all other telemetry continues normally.

---

## Backend Details

### `pytorch_mps` — PyTorch Metal GPU

Uses Apple's Metal Performance Shaders via `torch.device("mps")`. Calls `torch.mps.synchronize()` after each kernel to ensure accurate timing. Supports fp32 and fp16.

### `pytorch_cpu` — PyTorch CPU

Inherits all kernel implementations from the MPS backend, overrides device to `cpu`. CPU is synchronous so no explicit sync is needed. Always available.

### `onnx_rt` — ONNX Runtime with CoreML

Exports each kernel as a PyTorch module to ONNX (opset 17) using `torch.onnx.export`, then runs inference through ONNX Runtime with `CoreMLExecutionProvider` as the first choice, falling back to `CPUExecutionProvider`. Only `matmul` and `ffn` export as full compute graphs; `attention`, `layernorm`, `conv2d`, and `embedding` currently use identity modules due to ONNX export complexity with dynamic shapes.

### `mlx` — Apple MLX

Apple's own ML framework using unified memory. MLX uses lazy evaluation — `mx.eval()` forces materialisation. If `mlx` is not installed, the backend sets `available = False` and pyrex skips it cleanly without errors.

---

## Known Constraints

**typer version pinned to 0.12.3:** The original spec called for `typer==0.12.5` but that version has a bug with `click>=8.3` where `Parameter.make_metavar()` raises a `TypeError` when rendering help text for boolean flags. Pyrex uses `typer==0.12.3` + `click==8.1.8` — identical API, fully functional.

**MLX not installed by default:** Run `pip install mlx` to enable the 4th backend. The package is Apple Silicon-only and intentionally excluded from `requirements.txt` to avoid breaking non-Apple environments.

**ONNX kernels are partial:** Full ONNX export for dynamic multi-head attention and conv2d with arbitrary shapes requires significant export customisation. The current implementation runs matmul and FFN through CoreML properly, while other kernels measure the ORT session overhead with identity ops.

**powermetrics requires sudo:** On a default macOS setup, `sudo powermetrics` requires a password. To enable power readings in pyrex, add the following to `/etc/sudoers` via `visudo`:

```
yourusername ALL=(ALL) NOPASSWD: /usr/bin/powermetrics
```

---

## Configuration Reference

Full `config/config.yaml` options:

```yaml
benchmark:
  warmup_runs: 3          # runs before timing begins (discarded)
  repeat_runs: 10         # timed runs per configuration
  min_repeat_ms: 100      # minimum total time for a valid measurement
  outlier_sigma: 3.0      # σ threshold for outlier removal

backends:
  - id: "pytorch_mps"
    enabled: true
  - id: "pytorch_cpu"
    enabled: true
  - id: "onnx_rt"
    enabled: true
    provider: "CoreMLExecutionProvider"
  - id: "mlx"
    enabled: true

regression:
  threshold_pct: 5.0      # % change to flag as regression
  min_z_score: 2.0        # minimum statistical significance
  min_samples: 3          # minimum prior samples needed

roofline:
  peak_tflops_fp32: 3.6   # M4 theoretical peak FP32
  peak_tflops_fp16: 7.2   # M4 theoretical peak FP16
  memory_bandwidth_gbs: 120.0  # M4 memory bandwidth
```

---

## Architecture Overview

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
      ├── kernel_results table  ← all individual measurements
      └── runs table            ← run metadata + timing
```
