"""Timing and benchmarking utilities."""

import time
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class TimingResult:
    """Result of a timed function call."""
    elapsed_ms: float
    result: Any


def time_call(fn: Callable, *args, **kwargs) -> TimingResult:
    """Time a function call with nanosecond precision.

    Returns TimingResult with elapsed_ms and the function's return value.
    """
    start = time.perf_counter_ns()
    result = fn(*args, **kwargs)
    end = time.perf_counter_ns()
    elapsed_ms = (end - start) / 1_000_000.0
    return TimingResult(elapsed_ms=elapsed_ms, result=result)


@dataclass
class BenchmarkAccumulator:
    """Accumulates timing and memory measurements across trials."""
    name: str
    times_ms: list[float] = field(default_factory=list)
    sizes_bytes: list[int] = field(default_factory=list)
    count: int = 0

    def record(self, time_ms: float, size_bytes: int = 0):
        self.times_ms.append(time_ms)
        self.sizes_bytes.append(size_bytes)
        self.count += 1

    def mean_time_ms(self) -> float:
        return sum(self.times_ms) / len(self.times_ms) if self.times_ms else 0.0

    def mean_size_bytes(self) -> float:
        return sum(self.sizes_bytes) / len(self.sizes_bytes) if self.sizes_bytes else 0.0

    def summary(self) -> dict:
        import numpy as np
        times = np.array(self.times_ms)
        sizes = np.array(self.sizes_bytes)
        return {
            "name": self.name,
            "count": self.count,
            "time_ms_mean": float(np.mean(times)) if len(times) > 0 else 0.0,
            "time_ms_std": float(np.std(times)) if len(times) > 0 else 0.0,
            "time_ms_median": float(np.median(times)) if len(times) > 0 else 0.0,
            "size_bytes_mean": float(np.mean(sizes)) if len(sizes) > 0 else 0.0,
        }
