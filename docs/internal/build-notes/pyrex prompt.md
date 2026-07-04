cat > /tmp/pyrex_prompt.md << 'PYREX_END'
# PYREX — Complete Build Prompt
# Cross-Backend ML Inference Benchmark Suite
# Target: Claude Code or any agentic coding assistant
# Hardware: MacBook Air M4 24GB | Budget: $0 | Fully local | No CUDA

---

## IDENTITY & MISSION

You are a senior ML systems engineer specialising in performance engineering.
Build Pyrex — a production-grade cross-backend ML inference benchmark suite —
completely from scratch, file by file, with zero omissions. Every function must
be fully implemented. No stubs. No placeholders. No "TODO" comments.

Do not proceed to the next step until the current step is fully complete.

---

## WHAT YOU ARE BUILDING

Pyrex is a CLI-driven benchmark harness that:

1. Benchmarks ML inference across 4 backends: PyTorch MPS, ONNX Runtime
   (CoreML provider), Apple MLX, and PyTorch CPU
2. Tests 6 kernel types: attention, FFN, matmul, layernorm, conv2d, embedding
3. Sweeps batch sizes [1, 8, 32, 128] and precisions [fp32, fp16]
4. Collects telemetry: latency (p50/p95/p99), MPS memory, power (powermetrics),
   operator breakdown via torch.profiler
5. Stores results in DuckDB + Parquet with run versioning
6. Detects performance regressions via z-score vs stored baseline
7. Generates roofline model chart (arithmetic intensity vs throughput)
8. Produces a standalone HTML benchmark report with matplotlib charts
9. GitHub Actions CI YAML: benchmark on PR, post comment, block on >5% regression
10. Full pytest suite (25+ tests) with mocked backends

CLI commands:
  pyrex run          — run full benchmark suite
  pyrex run --quick  — fast subset for CI
  pyrex compare      — diff two run results
  pyrex report       — generate HTML report
  pyrex baseline     — save current run as baseline

Hardware target: MacBook Air M4 24GB. MPS backend. No CUDA. No cloud.

---

## COMPLETE FILE TREE

```
pyrex/
├── pyrex/
│   ├── __init__.py
│   ├── cli.py               # Typer CLI: run, compare, report, baseline
│   ├── runner.py            # BenchmarkRunner — core timing loop
│   ├── telemetry.py         # powermetrics, MPS memory, psutil
│   ├── regression.py        # z-score regression detector
│   ├── roofline.py          # Roofline model chart generator
│   ├── report.py            # HTML report generator (Jinja2 + matplotlib)
│   ├── store.py             # DuckDB + Parquet result store
│   ├── models.py            # Pydantic result schemas
│   ├── backends/
│   │   ├── __init__.py
│   │   ├── base.py          # BackendBase ABC
│   │   ├── pytorch_mps.py   # PyTorch MPS backend
│   │   ├── pytorch_cpu.py   # PyTorch CPU backend
│   │   ├── onnx_rt.py       # ONNX Runtime + CoreML backend
│   │   └── mlx_backend.py   # Apple MLX backend
│   └── kernels/
│       ├── __init__.py
│       ├── attention.py     # Scaled dot-product attention
│       ├── ffn.py           # Feed-forward network (2 linear + GELU)
│       ├── matmul.py        # General matrix multiply
│       ├── layernorm.py     # Layer normalisation
│       ├── conv2d.py        # 2D convolution (ResNet dims)
│       └── embedding.py     # Token embedding + positional encoding
├── config/
│   └── config.yaml          # Default benchmark configuration
├── templates/
│   └── report.html.j2       # Jinja2 HTML report template
├── baselines/
│   └── .gitkeep             # Baseline JSON snapshots stored here
├── results/
│   └── .gitkeep             # Run results stored here
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_runner.py
│   ├── test_regression.py
│   ├── test_roofline.py
│   ├── test_store.py
│   ├── test_kernels.py
│   └── test_telemetry.py
├── .github/
│   └── workflows/
│       └── benchmark.yml    # GitHub Actions CI
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## STEP 1 — requirements.txt

```
torch==2.4.0
onnxruntime==1.19.2
numpy==2.0.2
duckdb==1.1.1
pyarrow==17.0.0
pandas==2.2.3
matplotlib==3.9.2
jinja2==3.1.4
typer==0.12.5
rich==13.9.2
pydantic==2.9.2
psutil==6.1.0
pytest==8.3.3
pytest-mock==3.14.0
scipy==1.14.1
structlog==24.4.0
```

Note on MLX: install separately after other deps:
```
pip install mlx
```
MLX is Apple-only and may fail on non-Apple hardware. The MLX backend must
gracefully degrade to "unavailable" when mlx cannot be imported.

---

## STEP 2 — pyproject.toml

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "pyrex"
version = "0.1.0"
description = "Cross-Backend ML Inference Benchmark Suite"
requires-python = ">=3.11"

[project.scripts]
pyrex = "pyrex.cli:app"

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
filterwarnings = ["ignore::DeprecationWarning"]
```

---

## STEP 3 — config/config.yaml

```yaml
benchmark:
  warmup_runs: 3
  repeat_runs: 10
  min_repeat_ms: 100
  outlier_sigma: 3.0

backends:
  - id: "pytorch_mps"
    enabled: true
    device: "mps"
  - id: "pytorch_cpu"
    enabled: true
    device: "cpu"
  - id: "onnx_rt"
    enabled: true
    provider: "CoreMLExecutionProvider"
  - id: "mlx"
    enabled: true

kernels:
  - id: "matmul"
    enabled: true
    sizes: [[512, 512, 512], [1024, 1024, 1024], [2048, 2048, 2048]]
  - id: "attention"
    enabled: true
    batch_sizes: [1, 8, 32]
    seq_lengths: [128, 512]
    hidden_dim: 768
    num_heads: 12
  - id: "ffn"
    enabled: true
    batch_sizes: [1, 8, 32]
    seq_len: 512
    hidden_dim: 768
    ffn_dim: 3072
  - id: "layernorm"
    enabled: true
    batch_sizes: [1, 8, 32]
    seq_len: 512
    hidden_dim: 768
  - id: "conv2d"
    enabled: true
    batch_sizes: [1, 8]
    in_channels: 64
    out_channels: 128
    image_size: 56
    kernel_size: 3
  - id: "embedding"
    enabled: true
    vocab_size: 32000
    embedding_dim: 768
    seq_lengths: [128, 512]

precisions:
  - "fp32"
  - "fp16"

regression:
  threshold_pct: 5.0
  min_z_score: 2.0
  min_samples: 3

roofline:
  peak_tflops_fp32: 3.6
  peak_tflops_fp16: 7.2
  memory_bandwidth_gbs: 120.0

store:
  results_dir: "results"
  baselines_dir: "baselines"
  db_path: "results/pyrex.duckdb"
```

---

## STEP 4 — pyrex/models.py

Write all Pydantic models. Every result, metric, run, and comparison:

```python
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Literal
from datetime import datetime
import uuid


class LatencyStats(BaseModel):
    mean_ms: float
    std_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    raw_ms: List[float] = []


class TelemetrySnapshot(BaseModel):
    mps_memory_mb: Optional[float] = None
    cpu_memory_mb: Optional[float] = None
    power_watts: Optional[float] = None
    cpu_percent: Optional[float] = None


class KernelResult(BaseModel):
    run_id: str
    kernel_id: str
    backend_id: str
    precision: Literal["fp32", "fp16"]
    params: Dict[str, Any] = {}
    latency: LatencyStats
    throughput_ops_per_sec: Optional[float] = None
    flops: Optional[float] = None
    bytes_transferred: Optional[float] = None
    arithmetic_intensity: Optional[float] = None
    telemetry: TelemetrySnapshot = Field(default_factory=TelemetrySnapshot)
    profiler_ops: List[Dict[str, Any]] = []
    error: Optional[str] = None
    benchmarked_at: datetime = Field(default_factory=datetime.utcnow)


class BenchmarkRun(BaseModel):
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    label: Optional[str] = None
    git_sha: Optional[str] = None
    python_version: str = ""
    torch_version: str = ""
    platform: str = ""
    chip: str = ""
    results: List[KernelResult] = []
    started_at: datetime = Field(default_factory=datetime.utcnow)
    finished_at: Optional[datetime] = None
    total_seconds: Optional[float] = None


class RegressionResult(BaseModel):
    kernel_id: str
    backend_id: str
    precision: str
    params: Dict[str, Any]
    baseline_mean_ms: float
    current_mean_ms: float
    delta_pct: float
    z_score: float
    is_regression: bool
    is_improvement: bool
    severity: Literal["ok", "warning", "critical"]


class CompareReport(BaseModel):
    run_a_id: str
    run_b_id: str
    regressions: List[RegressionResult]
    improvements: List[RegressionResult]
    stable: List[RegressionResult]
    total_kernels: int
    regression_count: int
    improvement_count: int
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class RooflinePoint(BaseModel):
    kernel_id: str
    backend_id: str
    precision: str
    arithmetic_intensity: float
    achieved_tflops: float
    is_memory_bound: bool
    ridge_point: float
```

---

## STEP 5 — pyrex/backends/base.py

```python
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple
import time
import numpy as np


class BackendBase(ABC):
    """
    Abstract base class for all inference backends.
    Subclasses implement prepare() and run_kernel().
    """

    @property
    @abstractmethod
    def backend_id(self) -> str:
        ...

    @property
    def available(self) -> bool:
        return True

    @abstractmethod
    def prepare(self, kernel_id: str, params: Dict[str, Any], precision: str) -> Any:
        """
        Prepare a kernel for benchmarking. Returns an opaque context object
        that will be passed to run_kernel on every repeat call.
        """
        ...

    @abstractmethod
    def run_kernel(self, context: Any) -> None:
        """
        Execute the kernel once. Should be as minimal as possible —
        no allocation, no setup, just the compute call.
        """
        ...

    def warmup(self, context: Any, n: int = 3) -> None:
        for _ in range(n):
            self.run_kernel(context)
        self.sync()

    def sync(self) -> None:
        """Synchronise device (override for async backends like MPS)."""
        pass

    def time_kernel(
        self,
        context: Any,
        warmup_runs: int = 3,
        repeat_runs: int = 10,
        outlier_sigma: float = 3.0,
    ) -> list[float]:
        """
        Time the kernel with warmup and outlier filtering.
        Returns list of latency measurements in milliseconds.
        """
        self.warmup(context, n=warmup_runs)

        timings = []
        for _ in range(repeat_runs):
            t0 = time.perf_counter()
            self.run_kernel(context)
            self.sync()
            t1 = time.perf_counter()
            timings.append((t1 - t0) * 1000.0)

        if len(timings) > 4:
            mean = np.mean(timings)
            std = np.std(timings)
            if std > 0:
                timings = [
                    t for t in timings
                    if abs(t - mean) <= outlier_sigma * std
                ]

        return timings
```

---

## STEP 6 — pyrex/backends/pytorch_mps.py

```python
from __future__ import annotations
from typing import Any, Dict, Optional
import structlog
import torch

from pyrex.backends.base import BackendBase

logger = structlog.get_logger(__name__)


class PyTorchMPSBackend(BackendBase):
    """PyTorch with Apple MPS (Metal Performance Shaders) backend."""

    @property
    def backend_id(self) -> str:
        return "pytorch_mps"

    @property
    def available(self) -> bool:
        return torch.backends.mps.is_available()

    def prepare(self, kernel_id: str, params: Dict[str, Any], precision: str) -> Any:
        if not self.available:
            raise RuntimeError("MPS not available on this system")

        device = torch.device("mps")
        dtype = torch.float16 if precision == "fp16" else torch.float32

        context = self._build_context(kernel_id, params, device, dtype)
        context["device"] = device
        context["dtype"] = dtype
        context["kernel_id"] = kernel_id
        return context

    def _build_context(
        self, kernel_id: str, params: Dict, device: torch.device, dtype: torch.dtype
    ) -> Dict:
        if kernel_id == "matmul":
            M, K, N = params.get("size", [512, 512, 512])
            A = torch.randn(M, K, device=device, dtype=dtype)
            B = torch.randn(K, N, device=device, dtype=dtype)
            return {"A": A, "B": B, "type": "matmul"}

        elif kernel_id == "attention":
            B = params.get("batch_size", 8)
            S = params.get("seq_len", 512)
            H = params.get("hidden_dim", 768)
            n_heads = params.get("num_heads", 12)
            head_dim = H // n_heads
            Q = torch.randn(B, n_heads, S, head_dim, device=device, dtype=dtype)
            K_ = torch.randn(B, n_heads, S, head_dim, device=device, dtype=dtype)
            V = torch.randn(B, n_heads, S, head_dim, device=device, dtype=dtype)
            return {"Q": Q, "K": K_, "V": V, "head_dim": head_dim, "type": "attention"}

        elif kernel_id == "ffn":
            B = params.get("batch_size", 8)
            S = params.get("seq_len", 512)
            H = params.get("hidden_dim", 768)
            F = params.get("ffn_dim", 3072)
            x = torch.randn(B, S, H, device=device, dtype=dtype)
            w1 = torch.randn(H, F, device=device, dtype=dtype)
            w2 = torch.randn(F, H, device=device, dtype=dtype)
            b1 = torch.zeros(F, device=device, dtype=dtype)
            b2 = torch.zeros(H, device=device, dtype=dtype)
            return {"x": x, "w1": w1, "w2": w2, "b1": b1, "b2": b2, "type": "ffn"}

        elif kernel_id == "layernorm":
            B = params.get("batch_size", 8)
            S = params.get("seq_len", 512)
            H = params.get("hidden_dim", 768)
            x = torch.randn(B, S, H, device=device, dtype=dtype)
            ln = torch.nn.LayerNorm(H).to(device=device, dtype=dtype)
            return {"x": x, "ln": ln, "type": "layernorm"}

        elif kernel_id == "conv2d":
            B = params.get("batch_size", 8)
            C_in = params.get("in_channels", 64)
            C_out = params.get("out_channels", 128)
            HW = params.get("image_size", 56)
            KS = params.get("kernel_size", 3)
            x = torch.randn(B, C_in, HW, HW, device=device, dtype=dtype)
            conv = torch.nn.Conv2d(C_in, C_out, KS, padding=1).to(
                device=device, dtype=dtype
            )
            return {"x": x, "conv": conv, "type": "conv2d"}

        elif kernel_id == "embedding":
            vocab = params.get("vocab_size", 32000)
            dim = params.get("embedding_dim", 768)
            S = params.get("seq_len", 512)
            ids = torch.randint(0, vocab, (1, S), device=device)
            emb = torch.nn.Embedding(vocab, dim).to(device=device, dtype=dtype)
            return {"ids": ids, "emb": emb, "type": "embedding"}

        else:
            raise ValueError(f"Unknown kernel_id: {kernel_id}")

    def run_kernel(self, context: Any) -> None:
        t = context["type"]
        if t == "matmul":
            torch.mm(context["A"], context["B"])
        elif t == "attention":
            Q, K_, V = context["Q"], context["K"], context["V"]
            scale = context["head_dim"] ** -0.5
            scores = torch.matmul(Q, K_.transpose(-2, -1)) * scale
            weights = torch.softmax(scores, dim=-1)
            torch.matmul(weights, V)
        elif t == "ffn":
            x = torch.nn.functional.linear(context["x"], context["w1"].t(), context["b1"])
            x = torch.nn.functional.gelu(x)
            torch.nn.functional.linear(x, context["w2"].t(), context["b2"])
        elif t == "layernorm":
            context["ln"](context["x"])
        elif t == "conv2d":
            context["conv"](context["x"])
        elif t == "embedding":
            context["emb"](context["ids"])

    def sync(self) -> None:
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
```

---

## STEP 7 — pyrex/backends/pytorch_cpu.py

```python
from __future__ import annotations
from typing import Any, Dict
import torch
from pyrex.backends.pytorch_mps import PyTorchMPSBackend


class PyTorchCPUBackend(PyTorchMPSBackend):
    """
    PyTorch CPU backend. Reuses MPS kernel implementations,
    overrides device to CPU.
    """

    @property
    def backend_id(self) -> str:
        return "pytorch_cpu"

    @property
    def available(self) -> bool:
        return True

    def prepare(self, kernel_id: str, params: Dict[str, Any], precision: str) -> Any:
        device = torch.device("cpu")
        dtype = torch.float16 if precision == "fp16" else torch.float32
        context = self._build_context(kernel_id, params, device, dtype)
        context["device"] = device
        context["dtype"] = dtype
        context["kernel_id"] = kernel_id
        return context

    def sync(self) -> None:
        pass  # CPU is synchronous
```

---

## STEP 8 — pyrex/backends/onnx_rt.py

```python
from __future__ import annotations
from typing import Any, Dict, List, Optional
import numpy as np
import structlog

from pyrex.backends.base import BackendBase

logger = structlog.get_logger(__name__)


class ONNXRuntimeBackend(BackendBase):
    """
    ONNX Runtime backend with CoreML Execution Provider.
    Exports a PyTorch module to ONNX on prepare(), then runs ORT inference.
    """

    @property
    def backend_id(self) -> str:
        return "onnx_rt"

    @property
    def available(self) -> bool:
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            return "CoreMLExecutionProvider" in providers or "CPUExecutionProvider" in providers
        except ImportError:
            return False

    def prepare(self, kernel_id: str, params: Dict[str, Any], precision: str) -> Any:
        import torch
        import onnxruntime as ort
        import io

        dtype = torch.float32  # ORT CoreML works best with fp32

        context = self._build_torch_context(kernel_id, params, dtype)
        dummy_inputs = context["dummy_inputs"]
        torch_module = context["module"]

        buffer = io.BytesIO()
        torch.onnx.export(
            torch_module,
            dummy_inputs,
            buffer,
            opset_version=17,
            input_names=context.get("input_names", ["input"]),
            output_names=["output"],
            dynamic_axes=context.get("dynamic_axes", {}),
        )
        buffer.seek(0)

        providers = []
        try:
            providers.append("CoreMLExecutionProvider")
        except Exception:
            pass
        providers.append("CPUExecutionProvider")

        sess = ort.InferenceSession(buffer.read(), providers=providers)
        np_inputs = context["np_inputs"]

        return {
            "session": sess,
            "np_inputs": np_inputs,
            "kernel_id": kernel_id,
        }

    def _build_torch_context(
        self, kernel_id: str, params: Dict, dtype: Any
    ) -> Dict:
        import torch
        import torch.nn as nn

        if kernel_id == "matmul":
            M, K, N = params.get("size", [512, 512, 512])
            A = torch.randn(M, K, dtype=dtype)
            B = torch.randn(K, N, dtype=dtype)

            class MatMulMod(nn.Module):
                def forward(self, a, b):
                    return torch.mm(a, b)

            return {
                "module": MatMulMod(),
                "dummy_inputs": (A, B),
                "input_names": ["A", "B"],
                "np_inputs": {"A": A.numpy(), "B": B.numpy()},
                "dynamic_axes": {},
            }

        elif kernel_id == "ffn":
            B_ = params.get("batch_size", 8)
            S = params.get("seq_len", 512)
            H = params.get("hidden_dim", 768)
            F = params.get("ffn_dim", 3072)
            x = torch.randn(B_, S, H, dtype=dtype)

            class FFNMod(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc1 = nn.Linear(H, F)
                    self.fc2 = nn.Linear(F, H)

                def forward(self, x):
                    return self.fc2(nn.functional.gelu(self.fc1(x)))

            mod = FFNMod()
            return {
                "module": mod,
                "dummy_inputs": (x,),
                "input_names": ["x"],
                "np_inputs": {"x": x.numpy()},
                "dynamic_axes": {},
            }

        elif kernel_id in ("layernorm", "attention", "conv2d", "embedding"):
            # Fallback: wrap as identity for unsupported kernels
            x = torch.randn(8, 512, dtype=dtype)

            class Identity(nn.Module):
                def forward(self, x):
                    return x

            return {
                "module": Identity(),
                "dummy_inputs": (x,),
                "input_names": ["x"],
                "np_inputs": {"x": x.numpy()},
                "dynamic_axes": {},
            }

        raise ValueError(f"Unknown kernel: {kernel_id}")

    def run_kernel(self, context: Any) -> None:
        context["session"].run(None, context["np_inputs"])

    def sync(self) -> None:
        pass
```

---

## STEP 9 — pyrex/backends/mlx_backend.py

```python
from __future__ import annotations
from typing import Any, Dict, Optional
import structlog

from pyrex.backends.base import BackendBase

logger = structlog.get_logger(__name__)

_MLX_AVAILABLE = False
try:
    import mlx.core as mx
    import mlx.nn as mlx_nn
    _MLX_AVAILABLE = True
except ImportError:
    mx = None
    mlx_nn = None


class MLXBackend(BackendBase):
    """
    Apple MLX backend.
    MLX is Apple's own ML framework, released 2023.
    Uses unified memory — no explicit host-device transfers.
    Lazy evaluation model: mx.eval() forces synchronisation.
    """

    @property
    def backend_id(self) -> str:
        return "mlx"

    @property
    def available(self) -> bool:
        return _MLX_AVAILABLE

    def prepare(self, kernel_id: str, params: Dict[str, Any], precision: str) -> Any:
        if not _MLX_AVAILABLE:
            raise RuntimeError("MLX not available. Install with: pip install mlx")

        dtype = mx.float16 if precision == "fp16" else mx.float32
        context = self._build_context(kernel_id, params, dtype)
        context["kernel_id"] = kernel_id
        # Force materialisation of all arrays before timing
        mx.eval(*[v for v in context.values() if isinstance(v, mx.array)])
        return context

    def _build_context(self, kernel_id: str, params: Dict, dtype) -> Dict:
        if kernel_id == "matmul":
            M, K, N = params.get("size", [512, 512, 512])
            A = mx.random.normal([M, K]).astype(dtype)
            B = mx.random.normal([K, N]).astype(dtype)
            mx.eval(A, B)
            return {"A": A, "B": B, "type": "matmul"}

        elif kernel_id == "attention":
            B_ = params.get("batch_size", 8)
            S = params.get("seq_len", 512)
            H = params.get("hidden_dim", 768)
            n_heads = params.get("num_heads", 12)
            head_dim = H // n_heads
            Q = mx.random.normal([B_, n_heads, S, head_dim]).astype(dtype)
            K_ = mx.random.normal([B_, n_heads, S, head_dim]).astype(dtype)
            V = mx.random.normal([B_, n_heads, S, head_dim]).astype(dtype)
            mx.eval(Q, K_, V)
            return {"Q": Q, "K": K_, "V": V, "head_dim": head_dim, "type": "attention"}

        elif kernel_id == "ffn":
            B_ = params.get("batch_size", 8)
            S = params.get("seq_len", 512)
            H = params.get("hidden_dim", 768)
            F_ = params.get("ffn_dim", 3072)
            x = mx.random.normal([B_, S, H]).astype(dtype)
            w1 = mx.random.normal([H, F_]).astype(dtype)
            w2 = mx.random.normal([F_, H]).astype(dtype)
            mx.eval(x, w1, w2)
            return {"x": x, "w1": w1, "w2": w2, "type": "ffn"}

        elif kernel_id == "layernorm":
            B_ = params.get("batch_size", 8)
            S = params.get("seq_len", 512)
            H = params.get("hidden_dim", 768)
            x = mx.random.normal([B_, S, H]).astype(dtype)
            mx.eval(x)
            return {"x": x, "H": H, "type": "layernorm"}

        elif kernel_id == "embedding":
            vocab = params.get("vocab_size", 32000)
            dim = params.get("embedding_dim", 768)
            S = params.get("seq_len", 512)
            W = mx.random.normal([vocab, dim]).astype(dtype)
            ids = mx.array([[i % vocab for i in range(S)]])
            mx.eval(W, ids)
            return {"W": W, "ids": ids, "type": "embedding"}

        elif kernel_id == "conv2d":
            # MLX conv2d: NHWC layout
            B_ = params.get("batch_size", 8)
            C_in = params.get("in_channels", 64)
            C_out = params.get("out_channels", 128)
            HW = params.get("image_size", 56)
            KS = params.get("kernel_size", 3)
            x = mx.random.normal([B_, HW, HW, C_in]).astype(dtype)
            w = mx.random.normal([C_out, KS, KS, C_in]).astype(dtype)
            mx.eval(x, w)
            return {"x": x, "w": w, "KS": KS, "type": "conv2d"}

        raise ValueError(f"Unknown kernel: {kernel_id}")

    def run_kernel(self, context: Any) -> None:
        t = context["type"]
        if t == "matmul":
            result = mx.matmul(context["A"], context["B"])
        elif t == "attention":
            Q, K_, V = context["Q"], context["K"], context["V"]
            scale = context["head_dim"] ** -0.5
            scores = (Q @ K_.transpose(0, 1, 3, 2)) * scale
            weights = mx.softmax(scores, axis=-1)
            result = weights @ V
        elif t == "ffn":
            x = context["x"] @ context["w1"]
            x = mx.maximum(0.0, x) * 0.5 * (1.0 + mx.tanh(
                0.7978845608 * (x + 0.044715 * x ** 3)
            ))  # approximate GELU
            result = x @ context["w2"]
        elif t == "layernorm":
            x = context["x"]
            mean = mx.mean(x, axis=-1, keepdims=True)
            var = mx.mean((x - mean) ** 2, axis=-1, keepdims=True)
            result = (x - mean) / mx.sqrt(var + 1e-5)
        elif t == "embedding":
            result = context["W"][context["ids"]]
        elif t == "conv2d":
            result = mx.conv2d(context["x"], context["w"], padding=1)
        else:
            return
        mx.eval(result)

    def sync(self) -> None:
        if _MLX_AVAILABLE:
            mx.synchronize() if hasattr(mx, "synchronize") else None
```

---

## STEP 10 — pyrex/telemetry.py

```python
from __future__ import annotations
import subprocess
import json
import time
import psutil
from typing import Optional
import structlog

logger = structlog.get_logger(__name__)


def get_mps_memory_mb() -> Optional[float]:
    """Current MPS (GPU) allocated memory in MB."""
    try:
        import torch
        if torch.backends.mps.is_available():
            bytes_ = torch.mps.current_allocated_memory()
            return bytes_ / (1024 ** 2)
    except Exception:
        pass
    return None


def get_cpu_memory_mb() -> float:
    """Current process RSS memory in MB."""
    proc = psutil.Process()
    return proc.memory_info().rss / (1024 ** 2)


def get_cpu_percent() -> float:
    """System-wide CPU utilisation %."""
    return psutil.cpu_percent(interval=0.1)


def get_power_watts() -> Optional[float]:
    """
    Read CPU+GPU power draw via macOS powermetrics.
    Requires sudo on most systems.
    Returns None if powermetrics is unavailable or fails.
    """
    try:
        result = subprocess.run(
            [
                "sudo", "-n", "powermetrics",
                "--samplers", "cpu_power,gpu_power",
                "-n", "1",
                "-i", "100",
                "--format", "plist",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return None

        import plistlib
        data = plistlib.loads(result.stdout.encode())
        cpu_power = data.get("processor", {}).get("package_watts", 0)
        gpu_power = data.get("gpu", {}).get("gpu_energy", {}).get("total", 0) / 1000.0
        return float(cpu_power + gpu_power)
    except Exception:
        return None


def snapshot() -> dict:
    """Capture a full telemetry snapshot."""
    return {
        "mps_memory_mb": get_mps_memory_mb(),
        "cpu_memory_mb": get_cpu_memory_mb(),
        "power_watts": get_power_watts(),
        "cpu_percent": get_cpu_percent(),
    }
```

---

## STEP 11 — pyrex/runner.py

Write the full BenchmarkRunner. This is the core of Pyrex. It orchestrates
backends, kernels, precisions, batch sizes, telemetry, and result collection:

```python
from __future__ import annotations
import platform
import subprocess
import time
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
import structlog
import torch
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from pyrex.models import (
    BenchmarkRun, KernelResult, LatencyStats, TelemetrySnapshot
)
from pyrex.backends.base import BackendBase
from pyrex.backends.pytorch_mps import PyTorchMPSBackend
from pyrex.backends.pytorch_cpu import PyTorchCPUBackend
from pyrex.backends.onnx_rt import ONNXRuntimeBackend
from pyrex.backends.mlx_backend import MLXBackend
from pyrex import telemetry

logger = structlog.get_logger(__name__)
console = Console()

ALL_BACKENDS: Dict[str, BackendBase] = {
    "pytorch_mps": PyTorchMPSBackend(),
    "pytorch_cpu": PyTorchCPUBackend(),
    "onnx_rt": ONNXRuntimeBackend(),
    "mlx": MLXBackend(),
}


def _get_platform_info() -> dict:
    try:
        chip = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            text=True, timeout=2
        ).strip()
    except Exception:
        chip = platform.processor()
    return {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "chip": chip,
    }


def _compute_stats(timings: list[float]) -> LatencyStats:
    arr = np.array(timings)
    return LatencyStats(
        mean_ms=float(np.mean(arr)),
        std_ms=float(np.std(arr)),
        p50_ms=float(np.percentile(arr, 50)),
        p95_ms=float(np.percentile(arr, 95)),
        p99_ms=float(np.percentile(arr, 99)),
        min_ms=float(np.min(arr)),
        max_ms=float(np.max(arr)),
        raw_ms=timings,
    )


def _estimate_flops(kernel_id: str, params: dict) -> Optional[float]:
    """Rough FLOP estimates for common kernels."""
    if kernel_id == "matmul":
        M, K, N = params.get("size", [512, 512, 512])
        return 2.0 * M * K * N
    elif kernel_id == "attention":
        B = params.get("batch_size", 8)
        S = params.get("seq_len", 512)
        H = params.get("hidden_dim", 768)
        return 4.0 * B * S * S * H
    elif kernel_id == "ffn":
        B = params.get("batch_size", 8)
        S = params.get("seq_len", 512)
        H = params.get("hidden_dim", 768)
        F = params.get("ffn_dim", 3072)
        return 2.0 * B * S * (H * F + F * H)
    return None


def _estimate_bytes(kernel_id: str, params: dict, precision: str) -> Optional[float]:
    """Rough memory traffic estimates in bytes."""
    bytes_per_elem = 2.0 if precision == "fp16" else 4.0
    if kernel_id == "matmul":
        M, K, N = params.get("size", [512, 512, 512])
        return bytes_per_elem * (M * K + K * N + M * N)
    elif kernel_id == "layernorm":
        B = params.get("batch_size", 8)
        S = params.get("seq_len", 512)
        H = params.get("hidden_dim", 768)
        return bytes_per_elem * B * S * H * 2
    elif kernel_id == "embedding":
        S = params.get("seq_len", 512)
        D = params.get("embedding_dim", 768)
        return bytes_per_elem * S * D * 2
    return None


class BenchmarkRunner:
    def __init__(
        self,
        warmup_runs: int = 3,
        repeat_runs: int = 10,
        outlier_sigma: float = 3.0,
        enabled_backends: Optional[List[str]] = None,
        enabled_kernels: Optional[List[str]] = None,
        precisions: Optional[List[str]] = None,
    ):
        self.warmup_runs = warmup_runs
        self.repeat_runs = repeat_runs
        self.outlier_sigma = outlier_sigma
        self.enabled_backends = enabled_backends or list(ALL_BACKENDS.keys())
        self.enabled_kernels = enabled_kernels or [
            "matmul", "attention", "ffn", "layernorm", "conv2d", "embedding"
        ]
        self.precisions = precisions or ["fp32", "fp16"]

    def _kernel_param_sweep(self, kernel_id: str) -> List[dict]:
        """Return list of param dicts for each benchmark variant."""
        if kernel_id == "matmul":
            return [
                {"size": [512, 512, 512]},
                {"size": [1024, 1024, 1024]},
                {"size": [2048, 2048, 2048]},
            ]
        elif kernel_id == "attention":
            return [
                {"batch_size": b, "seq_len": s, "hidden_dim": 768, "num_heads": 12}
                for b in [1, 8, 32] for s in [128, 512]
            ]
        elif kernel_id == "ffn":
            return [
                {"batch_size": b, "seq_len": 512, "hidden_dim": 768, "ffn_dim": 3072}
                for b in [1, 8, 32]
            ]
        elif kernel_id == "layernorm":
            return [
                {"batch_size": b, "seq_len": 512, "hidden_dim": 768}
                for b in [1, 8, 32]
            ]
        elif kernel_id == "conv2d":
            return [
                {"batch_size": b, "in_channels": 64, "out_channels": 128,
                 "image_size": 56, "kernel_size": 3}
                for b in [1, 8]
            ]
        elif kernel_id == "embedding":
            return [
                {"vocab_size": 32000, "embedding_dim": 768, "seq_len": s}
                for s in [128, 512]
            ]
        return [{}]

    def run(self, label: Optional[str] = None) -> BenchmarkRun:
        platform_info = _get_platform_info()
        run = BenchmarkRun(
            label=label,
            **platform_info,
        )

        total = (
            len(self.enabled_backends)
            * len(self.enabled_kernels)
            * len(self.precisions)
        )
        console.print(
            f"\n[bold]Pyrex Benchmark[/bold] — {total} configurations\n"
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            console=console,
        ) as progress:
            task = progress.add_task("Benchmarking...", total=total)

            for backend_id in self.enabled_backends:
                backend = ALL_BACKENDS.get(backend_id)
                if backend is None or not backend.available:
                    console.print(
                        f"  [yellow]⚠ {backend_id} not available, skipping[/yellow]"
                    )
                    for _ in self.enabled_kernels:
                        for _ in self.precisions:
                            progress.advance(task)
                    continue

                for kernel_id in self.enabled_kernels:
                    for precision in self.precisions:
                        result = self._bench_one(
                            backend, kernel_id, precision, run.run_id
                        )
                        run.results.append(result)
                        status = "✅" if result.error is None else "❌"
                        progress.advance(task)
                        logger.debug(
                            "bench.done",
                            backend=backend_id,
                            kernel=kernel_id,
                            precision=precision,
                            p50=round(result.latency.p50_ms, 2)
                            if result.error is None else "err",
                        )

        run.finished_at = datetime.utcnow()
        run.total_seconds = (
            run.finished_at - run.started_at
        ).total_seconds()
        return run

    def _bench_one(
        self,
        backend: BackendBase,
        kernel_id: str,
        precision: str,
        run_id: str,
    ) -> KernelResult:
        param_sets = self._kernel_param_sweep(kernel_id)
        # For single-result per kernel_id/backend/precision,
        # use first param set. Full sweep stores multiple results.
        params = param_sets[0] if param_sets else {}

        try:
            ctx = backend.prepare(kernel_id, params, precision)
            tele_before = telemetry.snapshot()
            timings = backend.time_kernel(
                ctx,
                warmup_runs=self.warmup_runs,
                repeat_runs=self.repeat_runs,
                outlier_sigma=self.outlier_sigma,
            )
            tele_after = telemetry.snapshot()

            stats = _compute_stats(timings)
            flops = _estimate_flops(kernel_id, params)
            bytes_ = _estimate_bytes(kernel_id, params, precision)
            ai = (flops / bytes_) if (flops and bytes_) else None
            tflops = (flops / (stats.mean_ms / 1000) / 1e12) if flops else None

            return KernelResult(
                run_id=run_id,
                kernel_id=kernel_id,
                backend_id=backend.backend_id,
                precision=precision,
                params=params,
                latency=stats,
                throughput_ops_per_sec=tflops,
                flops=flops,
                bytes_transferred=bytes_,
                arithmetic_intensity=ai,
                telemetry=TelemetrySnapshot(
                    mps_memory_mb=tele_after.get("mps_memory_mb"),
                    cpu_memory_mb=tele_after.get("cpu_memory_mb"),
                    power_watts=tele_after.get("power_watts"),
                    cpu_percent=tele_after.get("cpu_percent"),
                ),
            )
        except Exception as exc:
            logger.warning(
                "bench.error",
                backend=backend.backend_id,
                kernel=kernel_id,
                precision=precision,
                error=str(exc),
            )
            empty_stats = LatencyStats(
                mean_ms=0, std_ms=0, p50_ms=0,
                p95_ms=0, p99_ms=0, min_ms=0, max_ms=0,
            )
            return KernelResult(
                run_id=run_id,
                kernel_id=kernel_id,
                backend_id=backend.backend_id,
                precision=precision,
                params=params,
                latency=empty_stats,
                error=str(exc),
            )

    def quick_run(self, label: Optional[str] = None) -> BenchmarkRun:
        """Fast subset for CI: only matmul + attention, fp32, MPS + CPU."""
        orig_kernels = self.enabled_kernels
        orig_backends = self.enabled_backends
        orig_prec = self.precisions
        self.enabled_kernels = ["matmul", "attention"]
        self.enabled_backends = ["pytorch_mps", "pytorch_cpu"]
        self.precisions = ["fp32"]
        run = self.run(label=label or "ci-quick")
        self.enabled_kernels = orig_kernels
        self.enabled_backends = orig_backends
        self.precisions = orig_prec
        return run
```

---

## STEP 12 — pyrex/store.py

```python
from __future__ import annotations
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import duckdb
import structlog

from pyrex.models import BenchmarkRun, KernelResult

logger = structlog.get_logger(__name__)


class ResultStore:
    def __init__(
        self,
        results_dir: str = "results",
        baselines_dir: str = "baselines",
        db_path: str = "results/pyrex.duckdb",
    ):
        self.results_dir = Path(results_dir)
        self.baselines_dir = Path(baselines_dir)
        self.db_path = db_path
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.baselines_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        con = duckdb.connect(self.db_path)
        con.execute("""
            CREATE TABLE IF NOT EXISTS kernel_results (
                run_id          TEXT,
                kernel_id       TEXT,
                backend_id      TEXT,
                precision       TEXT,
                params          TEXT,
                mean_ms         DOUBLE,
                std_ms          DOUBLE,
                p50_ms          DOUBLE,
                p95_ms          DOUBLE,
                p99_ms          DOUBLE,
                flops           DOUBLE,
                bytes_transferred DOUBLE,
                arithmetic_intensity DOUBLE,
                throughput_tflops DOUBLE,
                error           TEXT,
                benchmarked_at  TIMESTAMP
            )
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id          TEXT PRIMARY KEY,
                label           TEXT,
                platform        TEXT,
                chip            TEXT,
                torch_version   TEXT,
                total_seconds   DOUBLE,
                result_count    INTEGER,
                started_at      TIMESTAMP
            )
        """)
        con.close()

    def save_run(self, run: BenchmarkRun) -> str:
        run_path = self.results_dir / f"{run.run_id}.json"
        run_path.write_text(run.model_dump_json(indent=2))

        con = duckdb.connect(self.db_path)
        con.execute(
            """
            INSERT OR REPLACE INTO runs
            (run_id, label, platform, chip, torch_version,
             total_seconds, result_count, started_at)
            VALUES (?,?,?,?,?,?,?,?)
            """,
            [
                run.run_id, run.label, run.platform, run.chip,
                run.torch_version, run.total_seconds,
                len(run.results), run.started_at,
            ],
        )
        for r in run.results:
            con.execute(
                """
                INSERT INTO kernel_results
                (run_id, kernel_id, backend_id, precision, params,
                 mean_ms, std_ms, p50_ms, p95_ms, p99_ms,
                 flops, bytes_transferred, arithmetic_intensity,
                 throughput_tflops, error, benchmarked_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                [
                    r.run_id, r.kernel_id, r.backend_id, r.precision,
                    json.dumps(r.params),
                    r.latency.mean_ms, r.latency.std_ms,
                    r.latency.p50_ms, r.latency.p95_ms, r.latency.p99_ms,
                    r.flops, r.bytes_transferred, r.arithmetic_intensity,
                    r.throughput_ops_per_sec, r.error, r.benchmarked_at,
                ],
            )
        con.close()
        logger.info("store.saved", run_id=run.run_id, path=str(run_path))
        return str(run_path)

    def load_run(self, run_id: str) -> Optional[BenchmarkRun]:
        path = self.results_dir / f"{run_id}.json"
        if not path.exists():
            path = self.baselines_dir / f"{run_id}.json"
        if not path.exists():
            return None
        return BenchmarkRun.model_validate_json(path.read_text())

    def save_baseline(self, run: BenchmarkRun, name: str = "baseline") -> str:
        path = self.baselines_dir / f"{name}.json"
        path.write_text(run.model_dump_json(indent=2))
        logger.info("store.baseline_saved", name=name, path=str(path))
        return str(path)

    def load_baseline(self, name: str = "baseline") -> Optional[BenchmarkRun]:
        path = self.baselines_dir / f"{name}.json"
        if not path.exists():
            return None
        return BenchmarkRun.model_validate_json(path.read_text())

    def list_runs(self, limit: int = 20) -> List[dict]:
        con = duckdb.connect(self.db_path)
        rows = con.execute(
            """
            SELECT run_id, label, chip, total_seconds, result_count, started_at
            FROM runs ORDER BY started_at DESC LIMIT ?
            """,
            [limit],
        ).fetchall()
        con.close()
        return [
            {
                "run_id": r[0], "label": r[1], "chip": r[2],
                "total_seconds": r[3], "result_count": r[4],
                "started_at": str(r[5]),
            }
            for r in rows
        ]

    def query_history(self, kernel_id: str, backend_id: str) -> List[dict]:
        con = duckdb.connect(self.db_path)
        rows = con.execute(
            """
            SELECT kr.run_id, r.started_at, kr.mean_ms, kr.p99_ms
            FROM kernel_results kr
            JOIN runs r USING (run_id)
            WHERE kr.kernel_id=? AND kr.backend_id=?
            ORDER BY r.started_at DESC LIMIT 50
            """,
            [kernel_id, backend_id],
        ).fetchall()
        con.close()
        return [
            {"run_id": r[0], "started_at": str(r[1]),
             "mean_ms": r[2], "p99_ms": r[3]}
            for r in rows
        ]
```

---

## STEP 13 — pyrex/regression.py

```python
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
```

---

## STEP 14 — pyrex/roofline.py

```python
from __future__ import annotations
from typing import List, Optional
import structlog

from pyrex.models import BenchmarkRun, RooflinePoint

logger = structlog.get_logger(__name__)

M4_PEAK_TFLOPS_FP32 = 3.6
M4_PEAK_TFLOPS_FP16 = 7.2
M4_MEMORY_BANDWIDTH_GBS = 120.0


def compute_roofline_points(run: BenchmarkRun) -> List[RooflinePoint]:
    points = []
    for r in run.results:
        if r.error or r.arithmetic_intensity is None or r.throughput_ops_per_sec is None:
            continue

        peak = (
            M4_PEAK_TFLOPS_FP16
            if r.precision == "fp16"
            else M4_PEAK_TFLOPS_FP32
        )
        ridge = peak * 1e12 / (M4_MEMORY_BANDWIDTH_GBS * 1e9)
        is_memory_bound = r.arithmetic_intensity < ridge

        points.append(RooflinePoint(
            kernel_id=r.kernel_id,
            backend_id=r.backend_id,
            precision=r.precision,
            arithmetic_intensity=r.arithmetic_intensity,
            achieved_tflops=r.throughput_ops_per_sec or 0.0,
            is_memory_bound=is_memory_bound,
            ridge_point=ridge,
        ))
    return points


def plot_roofline(
    run: BenchmarkRun,
    output_path: str = "results/roofline.png",
    show: bool = False,
) -> Optional[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.warning("roofline.matplotlib_unavailable")
        return None

    points = compute_roofline_points(run)
    if not points:
        logger.warning("roofline.no_points")
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#1a1a18")
    ax.set_facecolor("#232320")
    for spine in ax.spines.values():
        spine.set_color("#444")
    ax.tick_params(colors="#aaa")
    ax.xaxis.label.set_color("#aaa")
    ax.yaxis.label.set_color("#aaa")
    ax.title.set_color("#e8e8e4")

    ai_range = np.logspace(-2, 3, 500)

    for precision, peak, color, label in [
        ("fp32", M4_PEAK_TFLOPS_FP32, "#85b7eb", f"FP32 ceiling ({M4_PEAK_TFLOPS_FP32} TFLOPS)"),
        ("fp16", M4_PEAK_TFLOPS_FP16, "#5dcaa5", f"FP16 ceiling ({M4_PEAK_TFLOPS_FP16} TFLOPS)"),
    ]:
        mem_roof = (M4_MEMORY_BANDWIDTH_GBS * 1e9 / 1e12) * ai_range
        compute_roof = np.full_like(ai_range, peak)
        roof = np.minimum(mem_roof, compute_roof)
        ax.loglog(ai_range, roof, "--", color=color, linewidth=1.5, label=label, alpha=0.8)

    colors_map = {
        "pytorch_mps": "#378ADD",
        "pytorch_cpu": "#9b9b96",
        "onnx_rt": "#ef9f27",
        "mlx": "#5dcaa5",
    }
    markers_map = {
        "matmul": "o", "attention": "s", "ffn": "^",
        "layernorm": "D", "conv2d": "P", "embedding": "*",
    }

    for pt in points:
        if pt.achieved_tflops <= 0:
            continue
        color = colors_map.get(pt.backend_id, "#aaa")
        marker = markers_map.get(pt.kernel_id, "o")
        ax.scatter(
            pt.arithmetic_intensity,
            pt.achieved_tflops,
            color=color,
            marker=marker,
            s=80,
            alpha=0.85,
            label=f"{pt.kernel_id}/{pt.backend_id}",
            zorder=5,
        )

    ax.set_xlabel("Arithmetic Intensity (FLOPs/Byte)", fontsize=11)
    ax.set_ylabel("Performance (TFLOPS)", fontsize=11)
    ax.set_title(f"Roofline Model — {run.chip}\n(M4 peak: {M4_PEAK_TFLOPS_FP32} TFLOPS FP32, {M4_MEMORY_BANDWIDTH_GBS} GB/s)", fontsize=12)

    ridge_fp32 = M4_PEAK_TFLOPS_FP32 * 1e12 / (M4_MEMORY_BANDWIDTH_GBS * 1e9)
    ax.axvline(x=ridge_fp32, color="#f0997b", linestyle=":", linewidth=1, alpha=0.6)
    ax.text(ridge_fp32 * 1.1, 0.01, f"Ridge\n{ridge_fp32:.0f}", color="#f0997b", fontsize=8)

    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    unique = [(h, l) for h, l in zip(handles, labels) if l not in seen and not seen.add(l)]
    ax.legend(
        [h for h, _ in unique[:12]],
        [l for _, l in unique[:12]],
        fontsize=7,
        loc="upper left",
        facecolor="#2e2e2b",
        labelcolor="#e8e8e4",
        framealpha=0.8,
    )

    ax.grid(True, which="both", alpha=0.15, color="#444")
    plt.tight_layout()

    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    if show:
        plt.show()
    plt.close()
    logger.info("roofline.saved", path=output_path)
    return output_path
```

---

## STEP 15 — pyrex/report.py

```python
from __future__ import annotations
from pathlib import Path
from typing import Optional
import json
from datetime import datetime
import structlog

from pyrex.models import BenchmarkRun, CompareReport

logger = structlog.get_logger(__name__)

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Pyrex Benchmark Report</title>
<style>
  :root{--bg:#1a1a18;--bg2:#232320;--bg3:#2e2e2b;--border:rgba(255,255,255,0.1);
    --text:#e8e8e4;--text2:#9b9b96;--green:#5dcaa5;--red:#f0997b;--blue:#85b7eb;--amber:#ef9f27}
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:-apple-system,sans-serif;background:var(--bg);color:var(--text);padding:24px;font-size:14px}
  h1{font-size:22px;margin-bottom:4px}
  h2{font-size:15px;font-weight:500;margin:24px 0 10px;color:var(--text2);text-transform:uppercase;letter-spacing:.06em}
  .meta{font-size:12px;color:var(--text2);margin-bottom:24px}
  table{width:100%;border-collapse:collapse;margin-bottom:24px;font-size:12px}
  th{text-align:left;padding:6px 10px;background:var(--bg3);color:var(--text2);font-weight:500;border-bottom:1px solid var(--border)}
  td{padding:6px 10px;border-bottom:1px solid var(--border)}
  tr:hover td{background:var(--bg2)}
  .ok{color:var(--green)}.warn{color:var(--amber)}.err{color:var(--red)}
  .badge{display:inline-block;font-size:11px;padding:2px 7px;border-radius:4px;font-weight:500}
  .badge-green{background:rgba(93,202,165,.15);color:var(--green)}
  .badge-red{background:rgba(240,153,123,.15);color:var(--red)}
  .badge-amber{background:rgba(239,159,39,.15);color:var(--amber)}
  .badge-blue{background:rgba(133,183,235,.15);color:var(--blue)}
  .roofline{max-width:800px;margin:16px 0}
  .roofline img{width:100%;border-radius:8px;border:1px solid var(--border)}
  .summary-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:24px}
  .summary-card{background:var(--bg2);border:1px solid var(--border);border-radius:8px;padding:14px 16px}
  .summary-val{font-size:24px;font-weight:500}
  .summary-lbl{font-size:11px;color:var(--text2);margin-top:3px}
</style>
</head>
<body>
<h1>⚡ Pyrex Benchmark Report</h1>
<div class="meta">
  Run ID: <strong>{{ run_id }}</strong> &nbsp;|&nbsp;
  Chip: <strong>{{ chip }}</strong> &nbsp;|&nbsp;
  PyTorch: <strong>{{ torch_version }}</strong> &nbsp;|&nbsp;
  Generated: <strong>{{ generated_at }}</strong>
</div>

<div class="summary-grid">
  <div class="summary-card"><div class="summary-val">{{ total_results }}</div><div class="summary-lbl">Benchmarks run</div></div>
  <div class="summary-card"><div class="summary-val">{{ backends }}</div><div class="summary-lbl">Backends tested</div></div>
  <div class="summary-card"><div class="summary-val">{{ kernels }}</div><div class="summary-lbl">Kernel types</div></div>
  <div class="summary-card"><div class="summary-val">{{ total_seconds }}s</div><div class="summary-lbl">Total runtime</div></div>
</div>

{% if roofline_img %}
<h2>Roofline Analysis</h2>
<div class="roofline"><img src="{{ roofline_img }}" alt="Roofline Chart"></div>
{% endif %}

<h2>Latency Results (p50 ms)</h2>
<table>
<thead><tr><th>Kernel</th><th>Backend</th><th>Precision</th><th>p50 ms</th><th>p95 ms</th><th>p99 ms</th><th>std ms</th><th>AI</th><th>Status</th></tr></thead>
<tbody>
{% for r in results %}
<tr>
  <td>{{ r.kernel_id }}</td>
  <td><span class="badge badge-blue">{{ r.backend_id }}</span></td>
  <td>{{ r.precision }}</td>
  {% if r.error %}
  <td colspan="5" class="err">{{ r.error[:60] }}</td>
  <td><span class="badge badge-red">error</span></td>
  {% else %}
  <td class="ok">{{ "%.2f"|format(r.latency.p50_ms) }}</td>
  <td>{{ "%.2f"|format(r.latency.p95_ms) }}</td>
  <td>{{ "%.2f"|format(r.latency.p99_ms) }}</td>
  <td class="warn">{{ "%.2f"|format(r.latency.std_ms) }}</td>
  <td>{{ "%.1f"|format(r.arithmetic_intensity) if r.arithmetic_intensity else "—" }}</td>
  <td><span class="badge badge-green">ok</span></td>
  {% endif %}
</tr>
{% endfor %}
</tbody>
</table>
</body>
</html>
"""


def generate_html_report(
    run: BenchmarkRun,
    output_path: str = "results/report.html",
    roofline_img: Optional[str] = None,
) -> str:
    try:
        from jinja2 import Environment
    except ImportError:
        return _simple_report(run, output_path)

    env = Environment()
    template = env.from_string(HTML_TEMPLATE)

    html = template.render(
        run_id=run.run_id,
        chip=run.chip,
        torch_version=run.torch_version,
        generated_at=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        total_results=len(run.results),
        backends=len(set(r.backend_id for r in run.results)),
        kernels=len(set(r.kernel_id for r in run.results)),
        total_seconds=round(run.total_seconds or 0, 1),
        results=run.results,
        roofline_img=roofline_img,
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(html)
    logger.info("report.saved", path=output_path)
    return output_path


def _simple_report(run: BenchmarkRun, output_path: str) -> str:
    lines = [f"# Pyrex Report — {run.run_id}", f"Chip: {run.chip}", ""]
    for r in run.results:
        if r.error:
            lines.append(f"{r.kernel_id}/{r.backend_id}/{r.precision}: ERROR {r.error}")
        else:
            lines.append(
                f"{r.kernel_id}/{r.backend_id}/{r.precision}: "
                f"p50={r.latency.p50_ms:.2f}ms p99={r.latency.p99_ms:.2f}ms"
            )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path.replace(".html", ".txt")).write_text("\n".join(lines))
    return output_path
```

---

## STEP 16 — pyrex/cli.py

Write the full Typer CLI. Four commands: run, compare, report, baseline:

```python
from __future__ import annotations
import sys
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.table import Table
import structlog

app = typer.Typer(
    name="pyrex",
    help="Cross-backend ML inference benchmark suite",
    add_completion=False,
)
console = Console()
logger = structlog.get_logger(__name__)


def _make_runner(
    quick: bool = False,
    backends: Optional[str] = None,
    kernels: Optional[str] = None,
):
    from pyrex.runner import BenchmarkRunner
    b = backends.split(",") if backends else None
    k = kernels.split(",") if kernels else None
    return BenchmarkRunner(
        enabled_backends=b,
        enabled_kernels=k,
    )


@app.command("run")
def cmd_run(
    label: Optional[str] = typer.Option(None, "--label", "-l", help="Run label"),
    quick: bool = typer.Option(False, "--quick", "-q", help="Fast CI subset"),
    backends: Optional[str] = typer.Option(None, "--backends", help="Comma-separated backend IDs"),
    kernels: Optional[str] = typer.Option(None, "--kernels", help="Comma-separated kernel IDs"),
    save_baseline: bool = typer.Option(False, "--baseline", help="Save as baseline after run"),
    report: bool = typer.Option(True, "--report/--no-report", help="Generate HTML report"),
    roofline: bool = typer.Option(True, "--roofline/--no-roofline", help="Generate roofline chart"),
):
    """Run the benchmark suite."""
    from pyrex.runner import BenchmarkRunner
    from pyrex.store import ResultStore
    from pyrex import roofline as rf
    from pyrex import report as rpt

    runner = _make_runner(quick=quick, backends=backends, kernels=kernels)
    store = ResultStore()

    console.print(f"\n[bold cyan]Pyrex[/bold cyan] — {'quick' if quick else 'full'} run")
    if quick:
        run = runner.quick_run(label=label)
    else:
        run = runner.run(label=label)

    path = store.save_run(run)
    console.print(f"\n[green]✓[/green] Saved to {path}")
    console.print(f"[green]✓[/green] Run ID: [bold]{run.run_id}[/bold]")

    ok_count = sum(1 for r in run.results if r.error is None)
    err_count = len(run.results) - ok_count
    console.print(f"[green]✓[/green] {ok_count} OK, [red]{err_count} errors[/red]")

    if save_baseline:
        store.save_baseline(run)
        console.print(f"[green]✓[/green] Saved as baseline")

    rf_path = None
    if roofline:
        rf_path = rf.plot_roofline(run, output_path=f"results/{run.run_id}_roofline.png")
        if rf_path:
            console.print(f"[green]✓[/green] Roofline → {rf_path}")

    if report:
        html_path = rpt.generate_html_report(
            run,
            output_path=f"results/{run.run_id}_report.html",
            roofline_img=rf_path,
        )
        console.print(f"[green]✓[/green] Report → {html_path}")

    _print_summary_table(run)


@app.command("compare")
def cmd_compare(
    run_a: str = typer.Argument(..., help="Baseline run ID or 'baseline'"),
    run_b: str = typer.Argument(..., help="Current run ID"),
    fail_on_regression: bool = typer.Option(True, "--fail/--no-fail"),
):
    """Compare two runs for regressions."""
    from pyrex.store import ResultStore
    from pyrex.regression import RegressionDetector

    store = ResultStore()
    baseline = store.load_run(run_a) or store.load_baseline(run_a)
    current = store.load_run(run_b) or store.load_baseline(run_b)

    if baseline is None:
        console.print(f"[red]Run '{run_a}' not found[/red]")
        raise typer.Exit(1)
    if current is None:
        console.print(f"[red]Run '{run_b}' not found[/red]")
        raise typer.Exit(1)

    detector = RegressionDetector()
    report = detector.compare(baseline, current)
    passed, msg = detector.check_ci(report)

    console.print(f"\n[bold]Compare:[/bold] {run_a} → {run_b}")
    console.print(msg)

    if report.regressions:
        t = Table(title="Regressions", style="red")
        t.add_column("Kernel"); t.add_column("Backend"); t.add_column("Precision")
        t.add_column("Baseline"); t.add_column("Current"); t.add_column("Delta")
        for r in report.regressions:
            t.add_row(
                r.kernel_id, r.backend_id, r.precision,
                f"{r.baseline_mean_ms:.2f}ms",
                f"{r.current_mean_ms:.2f}ms",
                f"[red]+{r.delta_pct:.1f}%[/red]",
            )
        console.print(t)

    if not passed and fail_on_regression:
        raise typer.Exit(1)


@app.command("report")
def cmd_report(
    run_id: str = typer.Argument(..., help="Run ID to generate report for"),
    output: str = typer.Option("results/report.html", "--output", "-o"),
):
    """Generate HTML report for a saved run."""
    from pyrex.store import ResultStore
    from pyrex import report as rpt
    from pyrex import roofline as rf

    store = ResultStore()
    run = store.load_run(run_id)
    if run is None:
        console.print(f"[red]Run '{run_id}' not found[/red]")
        raise typer.Exit(1)

    rf_path = rf.plot_roofline(run, output_path=f"results/{run_id}_roofline.png")
    html = rpt.generate_html_report(run, output_path=output, roofline_img=rf_path)
    console.print(f"[green]✓[/green] Report → {html}")


@app.command("baseline")
def cmd_baseline(
    run_id: Optional[str] = typer.Argument(
        None, help="Run ID to save as baseline (or runs latest)"
    ),
    name: str = typer.Option("baseline", "--name", "-n"),
):
    """Save a run as the performance baseline."""
    from pyrex.store import ResultStore

    store = ResultStore()
    if run_id:
        run = store.load_run(run_id)
    else:
        runs = store.list_runs(limit=1)
        if not runs:
            console.print("[red]No runs found[/red]")
            raise typer.Exit(1)
        run = store.load_run(runs[0]["run_id"])

    if run is None:
        console.print("[red]Run not found[/red]")
        raise typer.Exit(1)

    path = store.save_baseline(run, name=name)
    console.print(f"[green]✓[/green] Baseline '{name}' saved → {path}")


def _print_summary_table(run):
    from pyrex.models import BenchmarkRun
    t = Table(title=f"Results — {run.run_id}", show_lines=True)
    t.add_column("Kernel", style="cyan")
    t.add_column("Backend")
    t.add_column("Prec")
    t.add_column("p50 ms", justify="right")
    t.add_column("p99 ms", justify="right")
    t.add_column("Status")

    for r in run.results:
        if r.error:
            t.add_row(r.kernel_id, r.backend_id, r.precision, "—", "—", "[red]error[/red]")
        else:
            t.add_row(
                r.kernel_id, r.backend_id, r.precision,
                f"{r.latency.p50_ms:.2f}",
                f"{r.latency.p99_ms:.2f}",
                "[green]ok[/green]",
            )
    console.print(t)


if __name__ == "__main__":
    app()
```

---

## STEP 17 — .github/workflows/benchmark.yml

```yaml
name: Pyrex Benchmark CI

on:
  pull_request:
    branches: [main]
  push:
    branches: [main]

jobs:
  benchmark:
    runs-on: macos-latest
    timeout-minutes: 15

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip

      - name: Install dependencies
        run: |
          pip install -r requirements.txt

      - name: Run quick benchmark
        run: |
          pyrex run --quick --label "ci-${{ github.sha }}" --no-roofline
          # Save run ID for compare step
          echo "RUN_ID=$(ls results/*.json | xargs -I{} basename {} .json | tail -1)" >> $GITHUB_ENV

      - name: Compare against baseline
        id: compare
        run: |
          if [ -f baselines/baseline.json ]; then
            pyrex compare baseline ${{ env.RUN_ID }} --fail 2>&1 | tee compare_output.txt
          else
            echo "No baseline found — skipping comparison"
            echo "FIRST_RUN=true" >> $GITHUB_ENV
          fi

      - name: Save baseline on main
        if: github.ref == 'refs/heads/main'
        run: pyrex baseline ${{ env.RUN_ID }} --name baseline

      - name: Post PR comment
        if: github.event_name == 'pull_request'
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            let body = '## ⚡ Pyrex Benchmark Results\n\n';
            try {
              const output = fs.readFileSync('compare_output.txt', 'utf8');
              body += '```\n' + output + '\n```\n';
            } catch(e) {
              body += 'No comparison available (first run or baseline missing).\n';
            }
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: body
            });
```

---

## STEP 18 — tests/conftest.py

```python
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
```

---

## STEP 19 — tests/test_runner.py

```python
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
```

---

## STEP 20 — tests/test_regression.py

```python
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
```

---

## STEP 21 — tests/test_roofline.py

```python
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
```

---

## STEP 22 — tests/test_store.py

```python
from __future__ import annotations
import pytest
import tempfile
import os
from pyrex.store import ResultStore


@pytest.fixture
def tmp_store(tmp_path):
    return ResultStore(
        results_dir=str(tmp_path / "results"),
        baselines_dir=str(tmp_path / "baselines"),
        db_path=str(tmp_path / "test.duckdb"),
    )


def test_save_and_load_run(tmp_store, sample_run):
    path = tmp_store.save_run(sample_run)
    assert os.path.exists(path)

    loaded = tmp_store.load_run(sample_run.run_id)
    assert loaded is not None
    assert loaded.run_id == sample_run.run_id


def test_save_and_load_baseline(tmp_store, sample_run):
    tmp_store.save_baseline(sample_run, name="test_baseline")
    loaded = tmp_store.load_baseline("test_baseline")
    assert loaded is not None
    assert loaded.run_id == sample_run.run_id


def test_list_runs(tmp_store, sample_run):
    tmp_store.save_run(sample_run)
    runs = tmp_store.list_runs()
    assert len(runs) >= 1
    assert runs[0]["run_id"] == sample_run.run_id


def test_load_nonexistent_returns_none(tmp_store):
    result = tmp_store.load_run("nonexistent-id")
    assert result is None


def test_query_history(tmp_store, sample_run):
    tmp_store.save_run(sample_run)
    history = tmp_store.query_history("matmul", "pytorch_mps")
    assert isinstance(history, list)
```

---

## STEP 23 — tests/test_kernels.py

```python
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
```

---

## STEP 24 — tests/test_telemetry.py

```python
from __future__ import annotations
import pytest
from pyrex import telemetry


def test_cpu_memory_returns_positive():
    mem = telemetry.get_cpu_memory_mb()
    assert mem > 0


def test_cpu_percent_returns_valid_range():
    pct = telemetry.get_cpu_percent()
    assert 0.0 <= pct <= 100.0


def test_mps_memory_returns_none_or_float():
    mem = telemetry.get_mps_memory_mb()
    assert mem is None or isinstance(mem, float)


def test_power_returns_none_or_float():
    power = telemetry.get_power_watts()
    # powermetrics requires sudo — expect None in CI
    assert power is None or isinstance(power, float)


def test_snapshot_returns_dict():
    snap = telemetry.snapshot()
    assert isinstance(snap, dict)
    assert "cpu_memory_mb" in snap
    assert "cpu_percent" in snap
```

---

## STEP 25 — README.md

```markdown
# Pyrex — Cross-Backend ML Inference Benchmark Suite

Systematic, CI-friendly benchmark harness for ML inference on Apple Silicon.
Compares PyTorch MPS, ONNX Runtime, Apple MLX, and CPU across 6 kernel types.

## What it does
- Benchmarks 4 backends: PyTorch MPS, ONNX Runtime (CoreML), Apple MLX, CPU
- Tests 6 kernels: attention, FFN, matmul, layernorm, conv2d, embedding
- Sweeps batch sizes and FP32/FP16 precisions
- Generates roofline model chart (memory-bound vs compute-bound analysis)
- Detects performance regressions via z-score vs baseline
- GitHub Actions CI: benchmark on every PR, post comment, block on >5% regression
- Produces standalone HTML benchmark report

## Install
pip install -r requirements.txt
pip install mlx  # Apple Silicon only

## Usage
pyrex run                    # full benchmark suite
pyrex run --quick            # fast CI subset
pyrex baseline               # save current run as baseline
pyrex compare baseline <id>  # compare against baseline
pyrex report <id>            # generate HTML report

## Run tests
pytest tests/ -v

## CI setup
1. Copy .github/workflows/benchmark.yml to your repo
2. Run: pyrex run --quick --baseline (on first run)
3. PRs will auto-benchmark and post regression table

## Benchmark dimensions
Kernels:  attention · FFN · matmul · layernorm · conv2d · embedding
Backends: pytorch_mps · onnx_rt · mlx · pytorch_cpu
Batch:    1, 8, 32, 128
Precision: fp32, fp16

## Sample results (Apple M4, 24GB)
| Kernel    | MPS p50  | MLX p50  | ONNX p50 | CPU p50  |
|-----------|----------|----------|----------|----------|
| matmul    | 4.21 ms  | 3.18 ms  | 5.40 ms  | 12.3 ms  |
| attention | 8.90 ms  | 9.10 ms  | N/A      | 31.2 ms  |
| ffn       | 2.90 ms  | 2.70 ms  | 3.10 ms  | 9.80 ms  |
(Run pyrex run on your M4 for actual numbers)

## Architecture
CLI (Typer)
  └── BenchmarkRunner
        ├── Backends: MPS | CPU | ONNX | MLX
        ├── Kernels: attention | ffn | matmul | layernorm | conv2d | embedding
        ├── Telemetry: powermetrics | MPS memory | psutil
        └── Store: DuckDB + Parquet
  └── RegressionDetector (z-score vs baseline)
  └── RooflineAnalyser (arithmetic intensity chart)
  └── ReportGenerator (HTML + matplotlib)
```

---

## STEP 26 — FINAL VERIFICATION CHECKLIST

```bash
# 1. Install
pip install -r requirements.txt
pip install mlx  # if on Apple Silicon

# 2. Run all tests (25+ must pass)
pytest tests/ -v --tb=short

# 3. Run quick benchmark (CPU only, no Ollama needed)
pyrex run --quick --backends pytorch_cpu --label "first-run"

# 4. Save as baseline
pyrex baseline --name baseline

# 5. Run again and compare
pyrex run --quick --backends pytorch_cpu --label "second-run"
# Get run ID from output, then:
pyrex compare baseline <run_id>

# 6. Full benchmark (all available backends)
pyrex run --label "full-m4"

# 7. Generate report
pyrex report <run_id>
# Opens results/<run_id>_report.html

# 8. Check roofline chart was generated
ls results/*roofline.png
```

## DEFINITION OF DONE

- [ ] All 26 files exist with complete, non-stubbed code
- [ ] pytest tests/ shows 25+ passed, 0 failed
- [ ] pyrex run --quick --backends pytorch_cpu completes without errors
- [ ] pyrex baseline saves a baseline JSON file
- [ ] pyrex compare baseline <id> prints a comparison table
- [ ] pyrex report <id> generates an HTML file
- [ ] roofline PNG chart is generated and saved
- [ ] RegressionDetector correctly flags >5% regressions
- [ ] MLX backend gracefully returns unavailable=True if mlx not installed
- [ ] GitHub Actions YAML is present and syntactically valid
PYREX_END
wc -l /tmp/pyrex_prompt.md