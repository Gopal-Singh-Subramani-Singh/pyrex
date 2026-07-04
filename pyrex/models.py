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
