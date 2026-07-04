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


def test_results_dir_created(tmp_path):
    store = ResultStore(
        results_dir=str(tmp_path / "new_results"),
        baselines_dir=str(tmp_path / "new_baselines"),
        db_path=str(tmp_path / "new.duckdb"),
    )
    assert (tmp_path / "new_results").exists()
    assert (tmp_path / "new_baselines").exists()


def test_save_run_stores_kernel_results(tmp_store):
    """Verify kernel results are queryable when run_id is consistent."""
    from tests.conftest import make_latency_stats
    from pyrex.models import BenchmarkRun, KernelResult

    run = BenchmarkRun(
        run_id="consistent-run",
        chip="Apple M4",
        platform="macOS",
        results=[
            KernelResult(
                run_id="consistent-run",
                kernel_id="matmul",
                backend_id="pytorch_mps",
                precision="fp32",
                params={"size": [512, 512, 512]},
                latency=make_latency_stats(),
                flops=1e9,
                bytes_transferred=1e7,
            )
        ],
        total_seconds=5.0,
    )
    tmp_store.save_run(run)
    history = tmp_store.query_history("matmul", "pytorch_mps")
    assert len(history) >= 1
    assert "mean_ms" in history[0]
