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
    quick: bool = typer.Option(False, "--quick", help="Run fast CI subset"),
    label: Optional[str] = typer.Option(None, "--label", "-l", help="Human-readable label"),
    backends: Optional[str] = typer.Option(
        None, "--backends", "-b", help="Comma-separated backend IDs"
    ),
    kernels: Optional[str] = typer.Option(
        None, "--kernels", "-k", help="Comma-separated kernel IDs"
    ),
    skip_roofline: bool = typer.Option(False, "--skip-roofline", help="Skip roofline chart"),
    no_save: bool = typer.Option(False, "--no-save", help="Skip saving results to store"),
):
    """Run the benchmark suite."""
    from pyrex.runner import BenchmarkRunner
    from pyrex.store import ResultStore
    from pyrex import roofline as rf
    from pyrex import report as rpt

    runner = _make_runner(quick=quick, backends=backends, kernels=kernels)

    if quick:
        run = runner.quick_run(label=label)
    else:
        run = runner.run(label=label)

    _print_summary_table(run)

    if not no_save:
        store = ResultStore()
        path = store.save_run(run)
        console.print(f"\n[green]✓[/green] Saved → {path}")
        console.print(f"[bold]Run ID:[/bold] {run.run_id}")

    if not skip_roofline:
        rf_path = rf.plot_roofline(run, output_path=f"results/{run.run_id}_roofline.png")
        if rf_path:
            console.print(f"[green]✓[/green] Roofline chart → {rf_path}")

    return run


@app.command("compare")
def cmd_compare(
    baseline_id: str = typer.Argument(..., help="Baseline run ID or 'baseline'"),
    current_id: str = typer.Argument(..., help="Current run ID to compare against"),
    threshold: float = typer.Option(5.0, "--threshold", "-t", help="Regression threshold %"),
    fail: bool = typer.Option(False, "--fail", help="Exit with code 1 on regressions"),
):
    """Compare two benchmark runs and detect regressions."""
    from pyrex.store import ResultStore
    from pyrex.regression import RegressionDetector

    store = ResultStore()

    # Load baseline: try by name first (e.g., "baseline"), then by run_id
    if baseline_id == "baseline" or not baseline_id.startswith("run"):
        base_run = store.load_baseline(baseline_id)
        if base_run is None:
            base_run = store.load_run(baseline_id)
    else:
        base_run = store.load_run(baseline_id)

    if base_run is None:
        console.print(f"[red]Baseline '{baseline_id}' not found[/red]")
        raise typer.Exit(1)

    cur_run = store.load_run(current_id)
    if cur_run is None:
        console.print(f"[red]Run '{current_id}' not found[/red]")
        raise typer.Exit(1)

    detector = RegressionDetector(threshold_pct=threshold)
    report = detector.compare(base_run, cur_run)

    # Print comparison table
    t = Table(title=f"Comparison: {base_run.run_id} → {cur_run.run_id}", show_lines=True)
    t.add_column("Kernel", style="cyan")
    t.add_column("Backend")
    t.add_column("Prec")
    t.add_column("Baseline ms", justify="right")
    t.add_column("Current ms", justify="right")
    t.add_column("Delta %", justify="right")
    t.add_column("z-score", justify="right")
    t.add_column("Status")

    for r in report.regressions + report.improvements + report.stable:
        if r.is_regression:
            status = "[red]⬆ regression[/red]"
            delta_str = f"[red]+{r.delta_pct:.1f}%[/red]"
        elif r.is_improvement:
            status = "[green]⬇ faster[/green]"
            delta_str = f"[green]{r.delta_pct:.1f}%[/green]"
        else:
            status = "[dim]stable[/dim]"
            delta_str = f"{r.delta_pct:+.1f}%"

        t.add_row(
            r.kernel_id, r.backend_id, r.precision,
            f"{r.baseline_mean_ms:.2f}",
            f"{r.current_mean_ms:.2f}",
            delta_str,
            f"{r.z_score:.1f}",
            status,
        )

    console.print(t)

    passed, msg = detector.check_ci(report)
    console.print(f"\n{msg}")

    if fail and not passed:
        raise typer.Exit(1)


@app.command("report")
def cmd_report(
    run_id: str = typer.Argument(..., help="Run ID to generate report for"),
    output: str = typer.Option("", "--output", "-o", help="Output HTML path"),
):
    """Generate an HTML benchmark report."""
    from pyrex.store import ResultStore
    from pyrex import roofline as rf
    from pyrex import report as rpt

    store = ResultStore()
    run = store.load_run(run_id)
    if run is None:
        console.print(f"[red]Run '{run_id}' not found[/red]")
        raise typer.Exit(1)

    if not output:
        output = f"results/{run_id}_report.html"

    rf_path = rf.plot_roofline(run, output_path=f"results/{run_id}_roofline.png")
    html = rpt.generate_html_report(run, output_path=output, roofline_img=rf_path)
    console.print(f"[green]✓[/green] Report → {html}")


@app.command("baseline")
def cmd_baseline(
    run_id: Optional[str] = typer.Argument(
        None, help="Run ID to save as baseline (or uses latest)"
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


@app.command("list")
def cmd_list(
    limit: int = typer.Option(20, "--limit", "-n", help="Number of recent runs to show"),
):
    """List recent benchmark runs."""
    from pyrex.store import ResultStore

    store = ResultStore()
    runs = store.list_runs(limit=limit)

    if not runs:
        console.print("[yellow]No runs found.[/yellow]")
        raise typer.Exit(0)

    t = Table(title="Recent Benchmark Runs", show_lines=False)
    t.add_column("Run ID", style="cyan")
    t.add_column("Label")
    t.add_column("Chip")
    t.add_column("Results", justify="right")
    t.add_column("Duration", justify="right")
    t.add_column("Started At")

    for r in runs:
        duration = f"{r['total_seconds']:.1f}s" if r["total_seconds"] else "—"
        t.add_row(
            r["run_id"],
            r["label"] or "—",
            r["chip"] or "—",
            str(r["result_count"] or 0),
            duration,
            r["started_at"],
        )

    console.print(t)


def _print_summary_table(run):
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
