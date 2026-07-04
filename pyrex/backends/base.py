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
