# Pyrex — Cross-Backend ML Inference Benchmark Suite

Systematic, CI-friendly benchmark harness for ML inference on Apple Silicon. Compares PyTorch MPS, ONNX Runtime, Apple MLX, and CPU across 6 kernel types with regression detection and HTML report generation.

---

## What it does

Pyrex runs configurable ML inference benchmarks across 4 backends and 6 kernel types, measures latency, throughput, and memory usage, and detects performance regressions against a saved baseline using z-score analysis. Results are stored in DuckDB, and reports are generated as standalone HTML files. A GitHub Actions workflow gates PRs on regressions.

---

## Why it matters

Choosing the wrong inference backend on Apple Silicon leaves performance on the table. MPS, MLX, ONNX Runtime, and CPU have different characteristics across kernel types — what is fastest for matmul may not be fastest for attention. Without systematic benchmarking with regression detection, optimisation efforts are anecdotal. Pyrex provides repeatable, CI-integrated benchmarking that treats performance as a first-class engineering concern.

---

## Architecture

```mermaid
flowchart TD
    CLI([CLI\nrun · compare · report\nbaseline · list]) --> BR

    subgraph BR [BenchmarkRunner]
        direction LR
        subgraph BACKENDS [Backends]
            MPS[PyTorch MPS]
            CPU[PyTorch CPU]
            ONNX[ONNX Runtime\nCoreML EP]
            MLX[Apple MLX\noptional]
        end
        subgraph KERNELS [Kernels]
            ATT[attention]
            FFN[ffn]
            MM[matmul]
            LN[layernorm]
            CV[conv2d]
            EMB[embedding]
        end
        subgraph TEL [Telemetry]
            MEM[MPS memory\npsutil RSS]
            CPU_P[CPU %]
            PWR[powermetrics\noptional]
        end
    end

    BR --> RD[RegressionDetector\nz-score vs baseline\nthreshold: Δ>5% AND z≥2.0]
    BR --> RA[RooflineAnalyser\nTFLOPS vs arithmetic intensity]
    BR --> RG[ReportGenerator\nHTML · Jinja2 + matplotlib]
    BR --> RS[(ResultStore\nDuckDB + Parquet\nfull run history)]

    RD -->|gates PRs| CI[GitHub Actions CI\npost regression table\nblock on regression]

    style CLI fill:#4A90D9,color:#fff
    style RD fill:#C0392B,color:#fff
    style CI fill:#2E8B57,color:#fff
```

---

## Features

- **4 backends**: PyTorch MPS, ONNX Runtime (CoreML EP), Apple MLX, CPU
- **6 kernels**: attention, FFN, matmul, layernorm, conv2d, embedding
- **Batch size sweep**: 1, 8, 32, 128
- **Precision sweep**: fp32, fp16
- **Telemetry**: MPS memory, CPU RSS, CPU %, optional power via `powermetrics`
- **Regression detection**: z-score-gated comparison against saved baselines
- **Roofline model chart**: memory-bound vs compute-bound analysis per kernel
- **HTML report**: standalone report with charts, tables, and regression summary
- **DuckDB result store**: full run history, queryable via SQL
- **Baseline management**: save, compare, label runs
- **GitHub Actions CI**: benchmark on every PR, post regression table as comment, block on >5% regression

---

## Requirements

- macOS (Apple Silicon — M1/M2/M3/M4 recommended)
- Python 3.11+
- PyTorch 2.4.0

---

## Tech Stack

Python · PyTorch (MPS + CPU) · ONNX Runtime · Apple MLX · DuckDB · Parquet · Jinja2 · matplotlib · Typer · GitHub Actions

---

## Install

```bash
# Navigate to pyrex directory
cd pyrex

# Core dependencies
pip install -r requirements.txt

# Apple MLX (optional — graceful fallback if missing)
pip install mlx

# Install pyrex as a CLI tool
pip install -e .
```

---

## Quickstart

```bash
# Navigate to pyrex directory
cd pyrex

# Run fast CI subset (matmul + attention, MPS + CPU, fp32 only)
pyrex run --quick

# Run full benchmark suite
pyrex run

# Run with specific backends and kernels
pyrex run --backends pytorch_mps,pytorch_cpu --kernels matmul,attention

# Save current run as baseline
pyrex baseline

# Compare two runs for regressions
pyrex compare baseline <run_id>

# Generate HTML report
pyrex report <run_id>

# List all saved runs
pyrex list

# Label a run
pyrex run --label "post-optimization-v2"
```

---

## Tests

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=pyrex
```

Tests cover: runner, regression detector, roofline analyser, result store, all 6 kernels, telemetry module.

---

## CI Setup

1. Copy `.github/workflows/benchmark.yml` to your repo
2. Run the first benchmark and save as baseline:
   ```bash
   pyrex run --quick
   pyrex baseline --name baseline
   ```
3. Commit `baselines/baseline.json`
4. PRs will now auto-benchmark and post a regression table as a comment

---

## Benchmark Dimensions

| Dimension | Values |
|-----------|--------|
| Kernels | attention · FFN · matmul · layernorm · conv2d · embedding |
| Backends | pytorch_mps · onnx_rt · mlx · pytorch_cpu |
| Batch | 1, 8, 32, 128 |
| Precision | fp32, fp16 |

---

## Sample Results (Apple M4, 24GB)

| Kernel | MPS p50 | MLX p50 | ONNX p50 | CPU p50 |
|--------|---------|---------|---------|---------|
| matmul | 4.21 ms | 3.18 ms | 5.40 ms | 12.3 ms |
| attention | 8.90 ms | 9.10 ms | N/A | 31.2 ms |
| ffn | 2.90 ms | 2.70 ms | 3.10 ms | 9.80 ms |

*These results are from a real run on M4 hardware. Run `pyrex run` on your hardware for your actual numbers.*

---

## Observability

Pyrex does not expose a running server or Prometheus metrics — it is a CLI tool. Results are stored in DuckDB and viewable via:

```bash
# List all runs with metadata
pyrex list

# Generate HTML report with charts
pyrex report <run_id>
# Opens: results/<run_id>/report.html

# Regression comparison summary
pyrex compare baseline <run_id>
```

---

## Demo

```bash
# Navigate to pyrex directory
cd pyrex

# Quick benchmark
pyrex run --quick

# Expected output:
# Running 2 backends × 2 kernels × 2 batch sizes × 1 precision = 8 configurations
# [MPS] matmul b=1 fp32 ... p50=4.2ms p99=5.1ms mem=12MB
# [CPU] matmul b=1 fp32 ... p50=12.3ms p99=14.1ms mem=8MB
# Run saved: run_20240101_123456

# Save as baseline
pyrex baseline --name baseline

# Make a change, re-run, compare
pyrex run --quick
pyrex compare baseline <new_run_id>

# Generate report
pyrex report <run_id>
open results/<run_id>/report.html
```

---

## Known Limitations

- **Apple Silicon only for full benchmarks**: PyTorch MPS and MLX backends require Apple Silicon. CPU backend works on any machine. ONNX Runtime with CoreML EP is macOS-specific.
- **MLX is optional**: MLX may not be installable in all environments. Pyrex falls back gracefully.
- **`powermetrics` requires sudo**: Power telemetry via `powermetrics` requires passwordless sudo access. Not available in CI environments by default. Power metrics are skipped gracefully if unavailable.
- **No live monitoring**: Pyrex is a CLI tool, not a running service. No Prometheus metrics or dashboard.
- **Warmup variation**: Results vary between runs due to thermal throttling, background processes, and MPS warmup. Pyrex mitigates this with warmup iterations, but single-run results should not be over-interpreted.
- **ONNX attention unavailable**: ONNX Runtime does not support all attention kernel configurations. Results show N/A where backends cannot execute a kernel.

---

## Future Work

- Add LLM inference benchmarks (Ollama or llama.cpp backends)
- Add throughput (tokens/sec) as a first-class metric for generative models
- Add memory bandwidth analysis
- Support non-macOS backends (CUDA via torch.cuda)
- Add time-series charting for benchmark trends across commits

---

## Resume Bullet

> Built a cross-backend ML inference benchmark suite comparing PyTorch MPS, ONNX Runtime, MLX, and CPU execution with latency, throughput, memory, and regression analysis on Apple Silicon.
