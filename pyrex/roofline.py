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
    ax.set_title(
        f"Roofline Model — {run.chip}\n"
        f"(M4 peak: {M4_PEAK_TFLOPS_FP32} TFLOPS FP32, {M4_MEMORY_BANDWIDTH_GBS} GB/s)",
        fontsize=12,
    )

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
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    if show:
        plt.show()
    plt.close()
    logger.info("roofline.saved", path=output_path)
    return output_path
