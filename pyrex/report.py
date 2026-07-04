from __future__ import annotations
from pathlib import Path
from typing import Optional
import base64
from datetime import datetime
import structlog

from pyrex.models import BenchmarkRun, CompareReport

logger = structlog.get_logger(__name__)

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Pyrex — {{ run_id }}</title>
<style>
  :root {
    --bg:     #16161a;
    --bg2:    #1e1e24;
    --bg3:    #26262e;
    --bg4:    #2e2e38;
    --border: rgba(255,255,255,0.07);
    --text:   #e8e8e4;
    --text2:  #888899;
    --green:  #5dcaa5;
    --red:    #f0997b;
    --blue:   #7eb8f7;
    --amber:  #efb327;
    --purple: #c4a7e7;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, "SF Pro Text", "Segoe UI", sans-serif;
    background: var(--bg);
    color: var(--text);
    padding: 32px 40px;
    font-size: 13px;
    line-height: 1.5;
  }
  /* ── header ── */
  .header { margin-bottom: 28px; }
  .header h1 { font-size: 20px; font-weight: 600; letter-spacing: -0.3px; margin-bottom: 6px; }
  .header h1 span { color: var(--amber); }
  .meta { font-size: 11.5px; color: var(--text2); display: flex; flex-wrap: wrap; gap: 16px; }
  .meta strong { color: var(--text); }
  /* ── summary cards ── */
  .cards {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 32px;
  }
  .card {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px 20px;
  }
  .card-val { font-size: 28px; font-weight: 600; line-height: 1; }
  .card-lbl { font-size: 11px; color: var(--text2); margin-top: 5px; text-transform: uppercase; letter-spacing: .06em; }
  /* ── sections ── */
  .section-title {
    font-size: 10.5px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: .1em;
    color: var(--text2);
    margin: 28px 0 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
  }
  /* ── roofline ── */
  .roofline-wrap { margin-bottom: 8px; }
  .roofline-wrap img {
    max-width: 860px;
    width: 100%;
    border-radius: 10px;
    border: 1px solid var(--border);
    display: block;
  }
  /* ── table ── */
  .results-table { width: 100%; border-collapse: collapse; }
  .results-table thead tr { border-bottom: 1px solid rgba(255,255,255,0.12); }
  .results-table th {
    text-align: left;
    padding: 8px 12px;
    font-size: 10.5px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: .07em;
    color: var(--text2);
    white-space: nowrap;
  }
  .results-table td {
    padding: 7px 12px;
    border-bottom: 1px solid var(--border);
    font-variant-numeric: tabular-nums;
    white-space: nowrap;
  }
  .results-table tr:last-child td { border-bottom: none; }
  .results-table tbody tr:hover td { background: var(--bg2); }
  /* numeric columns */
  .num { text-align: right; }
  .p50 { color: var(--green); font-weight: 500; }
  .std { color: var(--amber); }
  .err-cell { color: var(--red); font-size: 11px; }
  /* badges */
  .badge {
    display: inline-block;
    font-size: 10.5px;
    padding: 2px 8px;
    border-radius: 5px;
    font-weight: 500;
    letter-spacing: .02em;
  }
  .badge-mps    { background: rgba(126,184,247,.13); color: var(--blue); }
  .badge-cpu    { background: rgba(136,136,153,.13); color: #aaa; }
  .badge-onnx   { background: rgba(239,179,39,.13);  color: var(--amber); }
  .badge-mlx    { background: rgba(93,202,165,.13);  color: var(--green); }
  .badge-ok     { background: rgba(93,202,165,.12);  color: var(--green); }
  .badge-error  { background: rgba(240,153,123,.12); color: var(--red); }
  /* precision pill */
  .prec {
    display: inline-block;
    font-size: 10px;
    padding: 1px 6px;
    border-radius: 4px;
    font-weight: 500;
    background: var(--bg4);
    color: var(--text2);
  }
  /* AI column colours */
  .ai-mem  { color: var(--amber); }
  .ai-comp { color: var(--purple); }
  .dash    { color: var(--text2); }
  /* kernel name */
  .kernel { font-weight: 500; }
  /* footer */
  .footer {
    margin-top: 40px;
    padding-top: 16px;
    border-top: 1px solid var(--border);
    font-size: 11px;
    color: var(--text2);
  }
</style>
</head>
<body>

<div class="header">
  <h1>⚡ Pyrex Benchmark Report &nbsp;<span>{{ run_id }}</span></h1>
  <div class="meta">
    <span>Chip: <strong>{{ chip }}</strong></span>
    <span>PyTorch: <strong>{{ torch_version }}</strong></span>
    <span>Platform: <strong>{{ platform }}</strong></span>
    <span>Generated: <strong>{{ generated_at }}</strong></span>
    {% if label %}<span>Label: <strong>{{ label }}</strong></span>{% endif %}
  </div>
</div>

<div class="cards">
  <div class="card">
    <div class="card-val">{{ total_results }}</div>
    <div class="card-lbl">Benchmarks run</div>
  </div>
  <div class="card">
    <div class="card-val">{{ backends }}</div>
    <div class="card-lbl">Backends tested</div>
  </div>
  <div class="card">
    <div class="card-val">{{ kernels }}</div>
    <div class="card-lbl">Kernel types</div>
  </div>
  <div class="card">
    <div class="card-val">{{ total_seconds }}s</div>
    <div class="card-lbl">Total runtime</div>
  </div>
</div>

{% if roofline_b64 %}
<div class="section-title">Roofline Analysis</div>
<div class="roofline-wrap">
  <img src="data:image/png;base64,{{ roofline_b64 }}" alt="Roofline Chart">
</div>
{% endif %}

<div class="section-title">Latency Results (p50 ms)</div>
<table class="results-table">
<thead>
  <tr>
    <th>Kernel</th>
    <th>Backend</th>
    <th>Prec</th>
    <th class="num">p50 ms</th>
    <th class="num">p95 ms</th>
    <th class="num">p99 ms</th>
    <th class="num">std ms</th>
    <th class="num">TFLOPS</th>
    <th class="num">AI (F/B)</th>
    <th>Bound</th>
    <th>Status</th>
  </tr>
</thead>
<tbody>
{% for r in results %}
<tr>
  <td class="kernel">{{ r.kernel_id }}</td>
  <td>
    {% if r.backend_id == 'pytorch_mps' %}<span class="badge badge-mps">mps</span>
    {% elif r.backend_id == 'pytorch_cpu' %}<span class="badge badge-cpu">cpu</span>
    {% elif r.backend_id == 'onnx_rt' %}<span class="badge badge-onnx">onnx</span>
    {% elif r.backend_id == 'mlx' %}<span class="badge badge-mlx">mlx</span>
    {% else %}<span class="badge">{{ r.backend_id }}</span>{% endif %}
  </td>
  <td><span class="prec">{{ r.precision }}</span></td>
  {% if r.error %}
  <td colspan="7" class="err-cell">{{ r.error[:80] }}</td>
  <td><span class="badge badge-error">error</span></td>
  {% else %}
  <td class="num p50">{{ "%.3f" | format(r.latency.p50_ms) }}</td>
  <td class="num">{{ "%.3f" | format(r.latency.p95_ms) }}</td>
  <td class="num">{{ "%.3f" | format(r.latency.p99_ms) }}</td>
  <td class="num std">{{ "%.3f" | format(r.latency.std_ms) }}</td>
  <td class="num">{% if r.throughput_ops_per_sec %}{{ "%.3f" | format(r.throughput_ops_per_sec) }}{% else %}<span class="dash">—</span>{% endif %}</td>
  <td class="num {% if r.arithmetic_intensity %}{% if r.arithmetic_intensity < 30 %}ai-mem{% else %}ai-comp{% endif %}{% endif %}">
    {% if r.arithmetic_intensity %}{{ "%.1f" | format(r.arithmetic_intensity) }}{% else %}<span class="dash">—</span>{% endif %}
  </td>
  <td>
    {% if r.arithmetic_intensity %}
      {% if r.arithmetic_intensity < 30 %}<span class="dash" style="color:var(--amber)">mem</span>
      {% else %}<span class="dash" style="color:var(--purple)">compute</span>{% endif %}
    {% else %}<span class="dash">—</span>{% endif %}
  </td>
  <td><span class="badge badge-ok">ok</span></td>
  {% endif %}
</tr>
{% endfor %}
</tbody>
</table>

<div class="footer">
  Pyrex v0.1.0 · Apple M4 · p50/p95/p99 from {{ repeat_runs }} timed repetitions after {{ warmup_runs }} warmup runs · outlier filtering at 3σ
</div>

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

    # Embed roofline PNG as base64 so the HTML is fully standalone
    roofline_b64 = None
    if roofline_img:
        try:
            img_bytes = Path(roofline_img).read_bytes()
            roofline_b64 = base64.b64encode(img_bytes).decode("utf-8")
        except Exception:
            pass

    env = Environment()
    template = env.from_string(HTML_TEMPLATE)

    html = template.render(
        run_id=run.run_id,
        chip=run.chip,
        torch_version=run.torch_version,
        platform=run.platform,
        label=run.label,
        generated_at=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        total_results=len(run.results),
        backends=len(set(r.backend_id for r in run.results)),
        kernels=len(set(r.kernel_id for r in run.results)),
        total_seconds=round(run.total_seconds or 0, 1),
        results=run.results,
        roofline_b64=roofline_b64,
        warmup_runs=3,
        repeat_runs=10,
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
    txt_path = output_path.replace(".html", ".txt")
    Path(txt_path).write_text("\n".join(lines))
    return txt_path
