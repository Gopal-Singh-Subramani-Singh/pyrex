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


def test_snapshot_has_all_keys():
    snap = telemetry.snapshot()
    expected_keys = {"mps_memory_mb", "cpu_memory_mb", "power_watts", "cpu_percent"}
    assert expected_keys == set(snap.keys())


def test_cpu_memory_is_float():
    mem = telemetry.get_cpu_memory_mb()
    assert isinstance(mem, float)


def test_cpu_percent_is_float():
    pct = telemetry.get_cpu_percent()
    assert isinstance(pct, float)
