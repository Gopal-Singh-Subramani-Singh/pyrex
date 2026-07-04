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
