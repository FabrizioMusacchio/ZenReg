"""Generate ZenReg preprint panels from benchmark outputs.

Run after ``run_preprint_benchmarks.py``:

    conda run -n zenreg2 python additional_scripts/plot_preprint_benchmarks.py

The script writes individual figure panels and draft composite PDFs into
``papers/preprint/figures``. It is intentionally centralized so panel sizes,
colors, limits, and labels can be adjusted in one place before final figure
assembly.
"""
# %% IMPORTS
from __future__ import annotations

import csv
import json
import os
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "zenreg-matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "zenreg-xdg-cache"))
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

from zenreg import z_project
# %% PATHS AND CONSTANTS
REPO_ROOT = Path(__file__).resolve().parents[1]
PREPRINT_DIR = REPO_ROOT / "papers" / "preprint"
BENCHMARK_DIR = PREPRINT_DIR / "benchmark_outputs"
FIGURES_DIR = PREPRINT_DIR / "figures"

CM_TO_INCH = 1.0 / 2.54
PANEL_DPI = 300
FONT_SIZE = 10
LEGEND_SIZE = 8
SHOW_LEGEND = True

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": FONT_SIZE,
        "axes.titlesize": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "legend.fontsize": LEGEND_SIZE,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

COLORS = {
    "phase": "#3b7cec",
    "pystackreg": "#d95f02",
    "normcorre": "#7EBA57",
    "normcorre_pw": "#7EBA57",
    "phase_median_filter": "#3b7cec",
    "pystackreg_median_filter": "#d95f02",
    "normcorre_gsig": "#7EBA57",
    "projection": "#3b7cec",
    "full_3d": "#7030a0",
    "dense_simpleitk": "#2f5597",
    "dense_points": "#7fb0ed",
    "sparse_simpleitk": "#863601",
    "sparse_points": "#d95f02",
}
LABELS = {
    "phase": "phase correlation",
    "pystackreg": "StackReg",
    "normcorre": "NoRMCorre rigid",
    "normcorre_pw": "NoRMCorre piecewise",
    "phase_median_filter": "phase correlation + median",
    "pystackreg_median_filter": "StackReg + median",
    "normcorre_gsig": "NoRMCorre + high-pass",
    "projection": "projection",
    "full_3d": "full volume",
    "dense_simpleitk": "dense/SimpleITK",
    "dense_points": "dense/points",
    "sparse_simpleitk": "puncta/SimpleITK",
    "sparse_points": "puncta/points",
}
LINESTYLES = {
    "phase": "-",
    "pystackreg": "-.",
    "normcorre": "--",
    "normcorre_pw": "-",
    "phase_median_filter": "-",
    "pystackreg_median_filter": "-.",
    "normcorre_gsig": "--",
    "projection": "-",
    "full_3d": "--",
}
MARKERS = {
    "phase": "o",
    "phase_median_filter": "o",
    "normcorre": "s",
    "normcorre_gsig": "s",
    "normcorre_pw": "s",
    "pystackreg": "^",
    "pystackreg_median_filter": "^",
    "projection": "D",
    "full_3d": "P",
    "dense_simpleitk": "o",
    "dense_points": "s",
    "sparse_simpleitk": "^",
    "sparse_points": "D",
}
PARALLEL_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")
METHOD_ORDER = [
    "phase",
    "phase_median_filter",
    "normcorre",
    "normcorre_gsig",
    "normcorre_pw",
    "pystackreg",
    "pystackreg_median_filter",
    "projection",
    "full_3d",
]
DEFAULT_SPINES = {"top": False, "right": False, "left": True, "bottom": True}


@dataclass(frozen=True)
class PanelConfig:
    output_subdir: str
    output_name: str
    figsize_cm: tuple[float, float]
    title: str
    xlabel: str | None = None
    ylabel: str | None = None
    xlim: tuple[float | None, float | None] | None = None
    ylim: tuple[float | None, float | None] | None = None
    legend_show: bool = False
    legend_loc: str = "best"
    spines: dict[str, bool] = field(default_factory=lambda: DEFAULT_SPINES.copy())
    grid_axis: str | None = "y"
    cmap: str | None = None

def image_panel(output_subdir: str, output_name: str, title: str, *, figsize_cm=(5.4, 5.1), cmap="gray") -> PanelConfig:
    return PanelConfig(output_subdir, output_name, figsize_cm, title, cmap=cmap, grid_axis=None)

def plot_panel(
    output_subdir: str,
    output_name: str,
    title: str,
    *,
    figsize_cm=(5.4, 4.6),
    xlabel=None,
    ylabel=None,
    xlim=None,
    ylim=None,
    show_legend=True,
    legend_show=None,
    legend_loc="best",
    grid_axis="y",
) -> PanelConfig:
    if legend_show is None:
        legend_show = show_legend
    return PanelConfig(
        output_subdir,
        output_name,
        figsize_cm,
        title,
        xlabel=xlabel,
        ylabel=ylabel,
        xlim=xlim,
        ylim=ylim,
        legend_show=legend_show,
        legend_loc=legend_loc,
        grid_axis=grid_axis,
    )


FIG2_PANELS = {
    "template": image_panel("figure2", "panel_a.png", "all-frame max template"),
    "raw_t0": image_panel("figure2", "panel_b.png", "raw t=0"),
    "raw_t1": image_panel("figure2", "panel_c.png", "raw t=25"),
    "raw_diff": image_panel("figure2", "panel_d.png", "raw difference", cmap="coolwarm"),
    "reg_t0": image_panel("figure2", "panel_e.png", "registered t=0"),
    "reg_t1": image_panel("figure2", "panel_f.png", "registered t=25"),
    "reg_diff": image_panel("figure2", "panel_g.png", "registered difference", cmap="coolwarm"),
    "shifts": plot_panel("figure2", 
                         "panel_h.pdf", "translation estimate", ylabel="correction (px)", 
                         xlabel="frame", 
                         legend_show=False,
                         figsize_cm=(6.0, 4.0)),
    "correlation": plot_panel("figure2", "panel_i.pdf", "template correlation", ylabel="Pearson r", 
                              ylim=(0, 1.05), legend_show=True,
                              xlabel="frame", 
                              figsize_cm=(6.0, 4.0)),
    "accuracy": plot_panel("figure2", "panel_j.pdf", "shift error", ylabel="absolute error (px)", 
                           ylim=(0, None), legend_show=True,
                           xlabel="frame", 
                           figsize_cm=(5.85, 4.0))}

FIG3_PANELS = {
    "snr_2d": plot_panel("figure3", "panel_a.pdf", "2D+t noise", 
                         xlabel="noise sigma", ylabel="mean abs. error (px)", 
                         legend_show=True,
                         figsize_cm=(4.3, 4.0)),
    "snr_2d_filter": plot_panel("figure3", "panel_b.pdf", "2D+t noise+filters", 
                                xlabel="noise sigma", ylabel="mean abs. error (px)", 
                                legend_show=False, figsize_cm=(4.3, 4.0)),
    "snr_3d": plot_panel("figure3", "panel_c.pdf", "3D+t projection noise", 
                         xlabel="noise sigma", ylabel="mean abs. error (px)", 
                         legend_show=False, 
                         figsize_cm=(4.3, 4.0)),
    "snr_3d_filter": plot_panel("figure3", "panel_d.pdf", "3D+t projection noise+filters", 
                                xlabel="noise sigma", ylabel="mean abs. error (px)", 
                                legend_show=False, 
                                figsize_cm=(4.3, 4.0)),
    "drift_2d": plot_panel("figure3", "panel_e.pdf", "drift amplitude", 
                           xlabel="drift scale", ylabel="mean abs. error (px)", 
                           legend_show=False, 
                           figsize_cm=(4.3, 4.0)),
    "drift_2d_filter": plot_panel("figure3", "panel_f.pdf", "drift amplitude+filters", 
                                  xlabel="drift scale", ylabel="mean abs. error (px)", 
                                  legend_show=False, 
                                  figsize_cm=(4.3, 4.0)),
    "piecewise": plot_panel("figure3", "panel_g.pdf", "piecewise motion", ylabel="mean Pearson r", 
                            figsize_cm=(3.9, 4.0)),
    "runtime": plot_panel("figure3", "panel_h.pdf", "runtime", ylabel="seconds", 
                          figsize_cm=(4.3, 4.0))}

FIG4_PANELS = {
    "dense_ref": image_panel("figure4", "panel_a.png", "dense reference"),
    "dense_raw": image_panel("figure4", "panel_b.png", "dense moving"),
    "dense_reg": image_panel("figure4", "panel_c.png", "dense registered"),
    "dense_raw_diff": image_panel("figure4", "panel_d.png", "dense ref - moving", cmap="bwr"),
    "dense_reg_diff": image_panel("figure4", "panel_e.png", "dense ref - registered", cmap="bwr"),
    "sparse_ref": image_panel("figure4", "panel_f.png", "puncta reference"),
    "sparse_raw": image_panel("figure4", "panel_g.png", "puncta moving"),
    "sparse_reg": image_panel("figure4", "panel_h.png", "puncta registered"),
    "sparse_raw_diff": image_panel("figure4", "panel_i.png", "puncta ref - moving", cmap="bwr"),
    "sparse_reg_diff": image_panel("figure4", "panel_j.png", "puncta ref - registered", cmap="bwr"),
    "shift_error": plot_panel("figure4", "panel_k.pdf", "3D translation error", 
                              xlabel="frame", ylabel="mean abs. error (px)", 
                              legend_show=True,
                              ylim=(0.0, 0.7),
                              figsize_cm=(5.0, 4.0)),
    "rotation_error": plot_panel("figure4", "panel_l.pdf", "3D rotation error", 
                                 xlabel="frame", ylabel="mean abs. error (°)", 
                                 legend_show=False,
                                 ylim=(0.0, 0.7),
                                 figsize_cm=(5.0, 4.0)),
    "dense_correlation": plot_panel("figure4", "panel_m.pdf", "dense correlation",
                                    xlabel="frame", ylabel="Pearson r", 
                                    legend_show=False, 
                                    ylim=(0,1.05),
                                    figsize_cm=(5.0, 4.0)),
    "sparse_correlation": plot_panel("figure4", "panel_n.pdf", "puncta correlation", 
                                     xlabel="frame", ylabel="Pearson r", 
                                     legend_show=False, 
                                     ylim=(0,1.05),
                                     figsize_cm=(5.0, 4.0)),
    "summary": plot_panel("figure4", "panel_o.pdf", "rigid backend summary", 
                          ylabel="mean abs. error", 
                          legend_show=True,
                          figsize_cm=(5.5, 4.0)),
}

FIG5_PANELS = {
    "trace_memmap": plot_panel("figure5", "panel_b.pdf", "memmap warm trace", 
                               figsize_cm=(6.2, 4.0), 
                               xlabel="elapsed time (s)", ylabel="additional memory\n(MB)", 
                               legend_show=False),
    "trace_in_memory": plot_panel("figure5", "panel_c.pdf", "in-memory trace", 
                                  figsize_cm=(6.2, 4.0), 
                                  xlabel="elapsed time (s)", ylabel="additional memory\n(MB)", 
                                  legend_show=False),
    "peak": plot_panel("figure5", "panel_d.pdf", "peak memory\nabove baseline", 
                       figsize_cm=(5.0, 4.0), 
                       ylabel="additional memory (MB)", 
                       legend_show=True),
    "duration": plot_panel("figure5", "panel_e.pdf", "workflow timing", 
                           ylabel="seconds",
                           legend_show=True,
                           figsize_cm=(5.0, 4.0)),
}

FIG6_PANELS = {
    "workflow_lateral": plot_panel("figure6", "panel_a.pdf", "lateral 3D+t\nmotion", 
                                   ylabel="mean abs. error (px)", 
                                   figsize_cm=(2.7, 4.0),
                                   ylim=(0, 0.04),
                                   legend_show=True),
    "workflow_axial": plot_panel("figure6", "panel_b.pdf", "axial 3D+t\nmotion", 
                                 ylabel="mean abs. error (px)", 
                                 figsize_cm=(2.7, 4.0),
                                 ylim=(0, 0.04),
                                 legend_show=True),
    "workflow_runtime": plot_panel("figure6", "panel_c.pdf", "projection vs\nfull-volume time", 
                                   ylabel="seconds", 
                                   figsize_cm=(4.0, 4.0),
                                   legend_show=True),
    "parallel": plot_panel("figure6", "panel_d.pdf", "parallel scaling", 
                           figsize_cm=(6.2, 4.3), 
                           xlabel="time points (log)", ylabel="runtime (s)", 
                           legend_show=True)}
# %% IO AND STYLE HELPERS
def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))

def _float(row: dict, key: str, default=np.nan) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return float(default)

def _save(fig, config: PanelConfig) -> Path:
    output = FIGURES_DIR / config.output_subdir / config.output_name
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=PANEL_DPI if output.suffix.lower() == ".png" else None, bbox_inches="tight", transparent=True)
    if output.suffix.lower() == ".pdf":
        fig.savefig(output.with_suffix(".png"), dpi=PANEL_DPI, bbox_inches="tight")
    plt.close(fig)
    return output

def _new_fig(config: PanelConfig):
    fig, ax = plt.subplots(figsize=(config.figsize_cm[0] * CM_TO_INCH, config.figsize_cm[1] * CM_TO_INCH))
    ax.set_title(config.title, fontsize=FONT_SIZE, pad=5)
    if config.xlabel:
        ax.set_xlabel(config.xlabel, fontsize=FONT_SIZE)
    if config.ylabel:
        ax.set_ylabel(config.ylabel, fontsize=FONT_SIZE)
    if config.xlim:
        ax.set_xlim(config.xlim)
    if config.ylim:
        ax.set_ylim(config.ylim)
    for spine, visible in config.spines.items():
        ax.spines[spine].set_visible(visible)
    if config.grid_axis:
        ax.grid(True, axis=config.grid_axis, alpha=0.25, linewidth=0.6)
    ax.tick_params(labelsize=FONT_SIZE)
    return fig, ax

def _maybe_legend(ax, config: PanelConfig, **kwargs) -> None:
    if SHOW_LEGEND and config.legend_show and ax.get_legend_handles_labels()[0]:
        legend_kwargs = {"loc": config.legend_loc, "fontsize": LEGEND_SIZE, "frameon": False}
        legend_kwargs.update(kwargs)
        ax.legend(**legend_kwargs)

def _placeholder(config: PanelConfig, message: str) -> None:
    fig, ax = _new_fig(config)
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=FONT_SIZE, color="#595959", wrap=True)
    _save(fig, config)

def _normalize_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    lo, hi = np.nanpercentile(image, [1, 99.5])
    if hi <= lo:
        return np.zeros_like(image)
    return np.clip((image - lo) / (hi - lo), 0, 1)

def _plot_image(
    config: PanelConfig,
    image: np.ndarray,
    *,
    diverging: bool = False,
    vmax_abs: float | None = None,
    colorbar: bool = True
) -> None:
    fig, ax = _new_fig(config)
    ax.axis("off")
    if diverging:
        vmax = float(vmax_abs) if vmax_abs is not None else float(np.nanmax(np.abs(image)))
        vmax = max(vmax, 1e-6)
        im = ax.imshow(image, cmap=config.cmap or "bwr", vmin=-vmax, vmax=vmax)
        if colorbar:
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cbar.ax.tick_params(labelsize=LEGEND_SIZE, length=2)
    else:
        ax.imshow(_normalize_image(image), cmap=config.cmap or "gray", vmin=0, vmax=1)
    _save(fig, config)

def _group_mean(rows: list[dict], *, scenario: str, value_key: str, group_keys: tuple[str, ...]) -> dict[tuple, float]:
    buckets: dict[tuple, list[float]] = defaultdict(list)
    for row in rows:
        if row.get("scenario") != scenario:
            continue
        key = tuple(row.get(group_key, "") for group_key in group_keys)
        value = _float(row, value_key)
        if np.isfinite(value):
            buckets[key].append(value)
    return {key: float(np.mean(values)) for key, values in buckets.items() if values}

def _line_plot_from_grouped(
    config: PanelConfig,
    grouped: dict[tuple, float],
    *,
    x_index: int,
    method_index: int,
    x_cast=float,
):
    if not grouped:
        _placeholder(config, "No benchmark data found.\nRun run_preprint_benchmarks.py first.")
        return
    fig, ax = _new_fig(config)
    found_methods = {key[method_index] for key in grouped}
    methods = [method for method in METHOD_ORDER if method in found_methods]
    methods.extend(sorted(found_methods.difference(methods)))
    for method in methods:
        points = sorted(
            [(x_cast(key[x_index]), value) for key, value in grouped.items() if key[method_index] == method],
            key=lambda item: item[0],
        )
        if not points:
            continue
        xs, ys = np.asarray(points).T
        ax.plot(
            xs,
            ys,
            marker=MARKERS.get(method, "o"),
            linewidth=1.3,
            markersize=3,
            color=COLORS.get(method, "#595959"),
            linestyle=LINESTYLES.get(method, "-"),
            zorder=3 if method in {"normcorre", "normcorre_gsig", "normcorre_pw"} else 2,
            label=LABELS.get(method, method),
        )
    _maybe_legend(ax, config)
    _save(fig, config)
# %% FIGURE 2
def plot_figure2() -> None:
    path = BENCHMARK_DIR / "representative_2d_phase.npz"
    if not path.exists():
        for panel in FIG2_PANELS.values():
            _placeholder(panel, "Representative benchmark missing.\nRun --case representative.")
        assemble_figure(
            "figure2",
            ["panel_a.png", "panel_b.png", "panel_c.png", "panel_d.png", None, "panel_e.png", "panel_f.png", "panel_g.png", None, "panel_h.png", "panel_i.png", "panel_j.png"],
            columns=4,
            title="Representative 2D+t registration",
        )
        return
    data = np.load(path)
    raw = data["raw"]
    registered = data["registered"]
    expected = data["expected"]
    estimated = data["estimated"]
    errors = data["errors"]
    before = data["pearson_before"]
    after = data["pearson_after"]

    moving_t = min(25, raw.shape[0] - 1)
    template = np.nanmax(raw[:, 0, 0], axis=0)
    raw_t0 = z_project(raw[0:1, :, 0:1], projection_method="max")[0, 0, 0]
    raw_t1 = z_project(raw[moving_t : moving_t + 1, :, 0:1], projection_method="max")[0, 0, 0]
    reg_t0 = z_project(registered[0:1, :, 0:1], projection_method="max")[0, 0, 0]
    reg_t1 = z_project(registered[moving_t : moving_t + 1, :, 0:1], projection_method="max")[0, 0, 0]
    raw_diff = raw_t1 - raw_t0
    reg_diff = reg_t1 - reg_t0
    max_abs = max(float(np.nanmax(np.abs(raw_diff))), float(np.nanmax(np.abs(reg_diff))), 1e-6)
    _plot_image(FIG2_PANELS["template"], template)
    _plot_image(FIG2_PANELS["raw_t0"], raw_t0)
    _plot_image(FIG2_PANELS["raw_t1"], raw_t1)
    _plot_image(FIG2_PANELS["raw_diff"], raw_diff, diverging=True, vmax_abs=max_abs)
    _plot_image(FIG2_PANELS["reg_t0"], reg_t0)
    _plot_image(FIG2_PANELS["reg_t1"], reg_t1)
    _plot_image(FIG2_PANELS["reg_diff"], reg_diff, diverging=True, vmax_abs=max_abs)

    frames = np.arange(estimated.shape[0])
    fig, ax = _new_fig(FIG2_PANELS["shifts"])
    ax.plot(frames, expected[:, 0], color=COLORS["phase"], linestyle="--", linewidth=1.0, label="GT y")
    ax.plot(frames, estimated[:, 0], color=COLORS["phase"], linewidth=1.5, label="estimated y")
    ax.plot(frames, expected[:, 1], color="#d95f02", linestyle="--", linewidth=1.0, label="GT x")
    ax.plot(frames, estimated[:, 1], color="#d95f02", linewidth=1.5, label="estimated x")
    _maybe_legend(ax, FIG2_PANELS["shifts"], ncol=2)
    _save(fig, FIG2_PANELS["shifts"])

    fig, ax = _new_fig(FIG2_PANELS["correlation"])
    ax.plot(frames, before, color="#a5a5a5", linewidth=1.2, label="before")
    ax.plot(frames, after, color=COLORS["phase"], linewidth=1.5, label="after")
    _maybe_legend(ax, FIG2_PANELS["correlation"])
    _save(fig, FIG2_PANELS["correlation"])

    fig, ax = _new_fig(FIG2_PANELS["accuracy"])
    ax.plot(frames, np.abs(errors[:, 0]), color=COLORS["phase"], linewidth=1.4, label="y")
    ax.plot(frames, np.abs(errors[:, 1]), color="#d95f02", linewidth=1.4, label="x")
    _maybe_legend(ax, FIG2_PANELS["accuracy"])
    _save(fig, FIG2_PANELS["accuracy"])

    assemble_figure(
        "figure2",
        ["panel_a.png", "panel_b.png", "panel_c.png", "panel_d.png", None, "panel_e.png", "panel_f.png", "panel_g.png", None, "panel_h.png", "panel_i.png", "panel_j.png"],
        columns=4,
        title="Representative 2D+t registration",
    )

# %% FIGURE 3
def plot_figure3() -> None:
    snr_rows = _read_rows(BENCHMARK_DIR / "snr_benchmark_metrics.csv")
    drift_rows = _read_rows(BENCHMARK_DIR / "drift_benchmark_metrics.csv")
    piecewise_rows = _read_rows(BENCHMARK_DIR / "piecewise_benchmark_metrics.csv")

    _line_plot_from_grouped(
        FIG3_PANELS["snr_2d"],
        _group_mean(snr_rows, scenario="snr_2d_translation", value_key="abs_error_mean", group_keys=("level", "method")),
        x_index=0,
        method_index=1,
    )
    _line_plot_from_grouped(
        FIG3_PANELS["snr_2d_filter"],
        _group_mean(snr_rows, scenario="snr_2d_translation_prefilter", value_key="abs_error_mean", group_keys=("level", "method")),
        x_index=0,
        method_index=1,
    )
    _line_plot_from_grouped(
        FIG3_PANELS["snr_3d"],
        _group_mean(snr_rows, scenario="snr_3d_projection_translation", value_key="abs_error_mean", group_keys=("level", "method")),
        x_index=0,
        method_index=1,
    )
    _line_plot_from_grouped(
        FIG3_PANELS["snr_3d_filter"],
        _group_mean(snr_rows, scenario="snr_3d_projection_translation_prefilter", value_key="abs_error_mean", group_keys=("level", "method")),
        x_index=0,
        method_index=1,
    )
    _line_plot_from_grouped(
        FIG3_PANELS["drift_2d"],
        _group_mean(drift_rows, scenario="drift_2d_translation", value_key="abs_error_mean", group_keys=("level", "method")),
        x_index=0,
        method_index=1,
    )
    _line_plot_from_grouped(
        FIG3_PANELS["drift_2d_filter"],
        _group_mean(drift_rows, scenario="drift_2d_translation_prefilter", value_key="abs_error_mean", group_keys=("level", "method")),
        x_index=0,
        method_index=1,
    )

    if piecewise_rows:
        fig, ax = _new_fig(FIG3_PANELS["piecewise"])
        labels = [LABELS.get(row["method"], row["method"]) for row in piecewise_rows]
        x = np.arange(len(piecewise_rows))
        before = [_float(row, "mean_corr_before") for row in piecewise_rows]
        after = [_float(row, "mean_corr_after") for row in piecewise_rows]
        ax.bar(x - 0.18, before, width=0.34, color="#bfbfbf", label="before")
        ax.bar(x + 0.18, after, width=0.34, color=[COLORS.get(row["method"], "#595959") for row in piecewise_rows], label="after")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=FONT_SIZE)
        ax.set_ylim(0, 1.05)
        _maybe_legend(ax, FIG3_PANELS["piecewise"])
        _save(fig, FIG3_PANELS["piecewise"])
    else:
        _placeholder(FIG3_PANELS["piecewise"], "Piecewise benchmark missing.\nRun --case piecewise.")

    runtime_source = snr_rows + drift_rows
    runtime_grouped = _group_mean(runtime_source, scenario="snr_2d_translation", value_key="runtime_s", group_keys=("method",))
    if runtime_grouped:
        fig, ax = _new_fig(FIG3_PANELS["runtime"])
        methods = [method for method in METHOD_ORDER if (method,) in runtime_grouped]
        methods.extend(key[0] for key in sorted(runtime_grouped) if key[0] not in methods)
        values = [runtime_grouped[(method,)] for method in methods]
        ax.bar(np.arange(len(methods)), values, color=[COLORS.get(method, "#595959") for method in methods])
        ax.set_xticks(np.arange(len(methods)))
        ax.set_xticklabels([LABELS.get(method, method) for method in methods], rotation=20, ha="right", fontsize=FONT_SIZE)
        _save(fig, FIG3_PANELS["runtime"])
    else:
        _placeholder(FIG3_PANELS["runtime"], "Runtime benchmark missing.\nRun --case snr.")

    assemble_figure("figure3", [f"panel_{letter}.png" for letter in "abcdefgh"], columns=4, title="Robustness benchmarks")
# %% FIGURE 4
def plot_figure4() -> None:
    rows = _read_rows(BENCHMARK_DIR / "rigid3d_benchmark_metrics.csv")
    def plot_rigid_image_row(path: Path, keys: tuple[str, str, str, str, str], missing: str) -> None:
        if not path.exists():
            for key in keys:
                _placeholder(FIG4_PANELS[key], missing)
            return
        data = np.load(path)
        raw = data["raw"]
        registered = data["registered"]
        moving_time = 1 if raw.shape[0] > 1 else 0
        ref_proj = np.max(raw[0, :, 0], axis=0)
        raw_proj = np.max(raw[moving_time, :, 0], axis=0)
        reg_proj = np.max(registered[moving_time, :, 0], axis=0)
        raw_diff = ref_proj - raw_proj
        reg_diff = ref_proj - reg_proj
        max_abs = max(float(np.nanmax(np.abs(raw_diff))), float(np.nanmax(np.abs(reg_diff))), 1e-6)
        _plot_image(FIG4_PANELS[keys[0]], ref_proj)
        _plot_image(FIG4_PANELS[keys[1]], raw_proj)
        _plot_image(FIG4_PANELS[keys[2]], reg_proj)
        _plot_image(FIG4_PANELS[keys[3]], raw_diff, diverging=True, vmax_abs=max_abs, colorbar=False)
        _plot_image(FIG4_PANELS[keys[4]], reg_diff, diverging=True, vmax_abs=max_abs, colorbar=True)

    dense_path = BENCHMARK_DIR / "rigid3d_dense_simpleitk.npz"
    plot_rigid_image_row(
        dense_path,
        ("dense_ref", "dense_raw", "dense_reg", "dense_raw_diff", "dense_reg_diff"),
        "Dense rigid 3D benchmark missing.\nRun --case rigid3d.",
    )
    sparse_path = BENCHMARK_DIR / "rigid3d_sparse_points.npz"
    plot_rigid_image_row(
        sparse_path,
        ("sparse_ref", "sparse_raw", "sparse_reg", "sparse_raw_diff", "sparse_reg_diff"),
        "Sparse rigid 3D benchmark missing.\nRun --case rigid3d.",
    )

    if rows:
        frames_by_backend = defaultdict(list)
        for row in rows:
            if row.get("error_message"):
                continue
            frames_by_backend[row.get("case_backend", row.get("backend", ""))].append(row)

        fig, ax = _new_fig(FIG4_PANELS["shift_error"])
        for backend, backend_rows in sorted(frames_by_backend.items()):
            backend_rows = sorted(backend_rows, key=lambda row: int(row["frame"]))
            ax.plot(
                [int(row["frame"]) for row in backend_rows],
                [_float(row, "abs_shift_error_mean") for row in backend_rows],
                marker=MARKERS.get(backend, "o"),
                linewidth=1.3,
                markersize=3,
                color=COLORS.get(backend, "#595959"),
                linestyle=LINESTYLES.get(backend.split("_", 1)[-1], "-"),
                label=LABELS.get(backend, backend),
            )
        _maybe_legend(ax, FIG4_PANELS["shift_error"])
        _save(fig, FIG4_PANELS["shift_error"])

        fig, ax = _new_fig(FIG4_PANELS["rotation_error"])
        for backend, backend_rows in sorted(frames_by_backend.items()):
            backend_rows = sorted(backend_rows, key=lambda row: int(row["frame"]))
            ax.plot(
                [int(row["frame"]) for row in backend_rows],
                [_float(row, "abs_rotation_error_mean") for row in backend_rows],
                marker=MARKERS.get(backend, "o"),
                linewidth=1.3,
                markersize=3,
                color=COLORS.get(backend, "#595959"),
                linestyle=LINESTYLES.get(backend.split("_", 1)[-1], "-"),
                label=LABELS.get(backend, backend),
            )
        _maybe_legend(ax, FIG4_PANELS["rotation_error"])
        _save(fig, FIG4_PANELS["rotation_error"])

        def plot_case_correlation(panel_key: str, case_name: str) -> None:
            fig, ax = _new_fig(FIG4_PANELS[panel_key])
            case_backends = {
                backend: [row for row in backend_rows if row.get("case") == case_name]
                for backend, backend_rows in frames_by_backend.items()
            }
            case_backends = {backend: values for backend, values in case_backends.items() if values}
            if not case_backends:
                ax.text(0.5, 0.5, f"{case_name} correlation missing", ha="center", va="center", transform=ax.transAxes)
                _save(fig, FIG4_PANELS[panel_key])
                return
            before_label_added = False
            for backend, backend_rows in sorted(case_backends.items()):
                backend_rows = sorted(backend_rows, key=lambda row: int(row["frame"]))
                frames = [int(row["frame"]) for row in backend_rows]
                ax.plot(
                    frames,
                    [_float(row, "pearson_before") for row in backend_rows],
                    color="#bfbfbf",
                    linewidth=1.0,
                    linestyle=":",
                    label="before" if not before_label_added else None,
                )
                before_label_added = True
                ax.plot(
                    frames,
                    [_float(row, "pearson_after") for row in backend_rows],
                    marker=MARKERS.get(backend, "o"),
                    linewidth=1.3,
                    markersize=3,
                    color=COLORS.get(backend, "#595959"),
                    linestyle=LINESTYLES.get(backend.split("_", 1)[-1], "-"),
                    label=f"{LABELS.get(backend, backend)} after",
                )
            _maybe_legend(ax, FIG4_PANELS[panel_key])
            _save(fig, FIG4_PANELS[panel_key])

        plot_case_correlation("dense_correlation", "dense")
        plot_case_correlation("sparse_correlation", "sparse")

        fig, ax = _new_fig(FIG4_PANELS["summary"])
        backends = sorted(frames_by_backend)
        shift_mae = [np.mean([_float(row, "abs_shift_error_mean") for row in frames_by_backend[backend]]) for backend in backends]
        rot_mae = [np.mean([_float(row, "abs_rotation_error_mean") for row in frames_by_backend[backend]]) for backend in backends]
        x = np.arange(len(backends))
        backend_colors = [COLORS.get(backend, "#595959") for backend in backends]
        bar_width = 0.30
        bar_offset = 0.19
        ax.bar(
            x - bar_offset,
            shift_mae,
            width=bar_width,
            color=backend_colors,
            edgecolor=backend_colors,
            linewidth=0.8,
            label="translation (px)",
        )
        ax.bar(
            x + bar_offset,
            rot_mae,
            width=bar_width,
            color=backend_colors,
            alpha=0.35,
            edgecolor=backend_colors,
            linewidth=0.0,
            label="rotation (°)",
        )
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS.get(backend, backend) for backend in backends], rotation=20, ha="right", fontsize=FONT_SIZE)
        if SHOW_LEGEND and FIG4_PANELS["summary"].legend_show:
            ax.legend(
                handles=[
                    Patch(facecolor="#8c8c8c", edgecolor="#8c8c8c", linewidth=0.8, label="translation (px)"),
                    Patch(facecolor="#8c8c8c", edgecolor="#8c8c8c", linewidth=0.0, alpha=0.35, label="rotation (°)"),
                ],
                loc=FIG4_PANELS["summary"].legend_loc,
                fontsize=LEGEND_SIZE,
                frameon=False)
        _save(fig, FIG4_PANELS["summary"])
    else:
        for key in ["shift_error", "rotation_error", "dense_correlation", "sparse_correlation", "summary"]:
            _placeholder(FIG4_PANELS[key], "Rigid 3D benchmark missing.\nRun --case rigid3d.")

    assemble_figure("figure4", [f"panel_{letter}.png" for letter in "abcdefghijklmno"], columns=5, title="Full 3D rigid registration")
# %% FIGURE 5
def _memory_step_intervals(rows: list[dict[str, str]]) -> list[tuple[str, float]]:
    marks = [(row.get("step", ""), _float(row, "elapsed_s")) for row in rows if row.get("event") == "mark"]
    keep = []
    for step, elapsed in marks:
        if np.isfinite(elapsed) and step not in {"tracker:start", "tracker:stop", "sample"}:
            keep.append((step, elapsed))
    return keep

def _clean_memory_trace_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).copy()
    finite_indices = np.flatnonzero(np.isfinite(values))
    for index in finite_indices:
        previous_candidates = finite_indices[finite_indices < index]
        next_candidates = finite_indices[finite_indices > index]
        if not previous_candidates.size or not next_candidates.size:
            continue
        reference = min(float(values[previous_candidates[-1]]), float(values[next_candidates[0]]))
        if reference > 0 and float(values[index]) < 0.25 * reference:
            values[index] = np.nan
    return values

def _delta_from_baseline(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).copy()
    finite = values[np.isfinite(values)]
    if not finite.size:
        return values
    return values - float(finite[0])


def _plot_single_memory_trace(
    rows: list[dict[str, str]],
    panel: PanelConfig,
    *,
    rss_color: str,
    uss_color: str,
    marker_color: str,
) -> None:
    if not rows:
        _placeholder(panel, "Memory trace missing.\nRun --case memory.")
        return
    fig, ax = _new_fig(panel)
    times = np.asarray([_float(row, "elapsed_s") for row in rows], dtype=np.float32)
    rss = np.asarray([_float(row, "rss_mb") for row in rows], dtype=np.float32)
    uss = _clean_memory_trace_values(np.asarray([_float(row, "uss_mb") for row in rows], dtype=np.float32))
    rss_delta = _delta_from_baseline(rss)
    uss_delta = _delta_from_baseline(uss)
    finite_rss = rss_delta[np.isfinite(rss_delta)]
    finite_uss = uss_delta[np.isfinite(uss_delta)]
    ax.plot(
        times,
        rss_delta,
        color=rss_color,
        linewidth=1.4,
        linestyle="-",
        label="delta RSS")
    if finite_uss.size:
        ax.plot(
            times,
            uss_delta,
            color=uss_color,
            linewidth=1.2,
            linestyle="--",
            alpha=0.90,
            label="delta USS")
    marker_rows = _memory_step_intervals(rows)
    grouped_counts: dict[float, int] = defaultdict(int)
    for index, (step, elapsed) in enumerate(marker_rows, start=1):
        rounded_elapsed = round(float(elapsed), 2)
        rank = grouped_counts[rounded_elapsed]
        grouped_counts[rounded_elapsed] += 1
        ax.axvline(elapsed, color=marker_color, linewidth=0.7, alpha=0.75)
        ax.text(
            elapsed,
            0.98 - rank * 0.075,
            str(index),
            transform=ax.get_xaxis_transform(),
            va="top",
            ha="center",
            fontsize=LEGEND_SIZE,
            color="#595959",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.65, "pad": 0.8},
        )
    finite_all = np.concatenate([finite_rss, finite_uss]) if finite_uss.size else finite_rss
    ax.set_ylim(0, float(np.max(finite_all)) * 1.12 if finite_all.size and np.max(finite_all) > 0 else 1)
    if not finite_uss.size:
        ax.text(
            0.99,
            0.82,
            "USS not available\nfor this run",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=LEGEND_SIZE,
            color="#595959",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 2},
        )
    legend_text = "\n".join(f"{idx}. {step}" for idx, (step, _) in enumerate(marker_rows, start=1))
    if legend_text:
        ax.text(
            0.99,
            0.03,
            legend_text,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=LEGEND_SIZE,
            color="#404040",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 2},
        )
    _maybe_legend(ax, panel, loc="upper left")
    _save(fig, panel)


def plot_figure5() -> None:
    memmap_rows = _read_rows(BENCHMARK_DIR / "memory_trace_memmap_warm.csv")
    if not memmap_rows:
        memmap_rows = _read_rows(BENCHMARK_DIR / "memory_trace_memmap_cold.csv")
    if not memmap_rows:
        memmap_rows = _read_rows(BENCHMARK_DIR / "memory_trace_memmap.csv")
    if not memmap_rows:
        memmap_rows = _read_rows(BENCHMARK_DIR / "memory_trace.csv")
    in_memory_rows = _read_rows(BENCHMARK_DIR / "memory_trace_in_memory.csv")
    timing_rows = _read_rows(BENCHMARK_DIR / "memory_timing_metrics.csv")
    summary_path = BENCHMARK_DIR / "memory_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    _plot_single_memory_trace(
        memmap_rows,
        FIG5_PANELS["trace_memmap"],
        rss_color="#2f5597",
        uss_color="#70ad47",
        marker_color="#a5a5a5",
    )
    _plot_single_memory_trace(
        in_memory_rows,
        FIG5_PANELS["trace_in_memory"],
        rss_color="#2f5597",
        uss_color="#70ad47",
        marker_color="#8c8c8c",
    )

    if summary:
        fig, ax = _new_fig(FIG5_PANELS["peak"])
        workflows = ["memmap_cold", "memmap_warm", "in_memory"]
        labels = ["MM cold", "MM warm", "RAM"]
        rss_values = [
            float(summary.get(f"{workflow}_peak_delta_rss_mb", np.nan))
            for workflow in workflows
        ]
        uss_values = [
            float(summary.get(f"{workflow}_peak_delta_uss_mb", np.nan))
            for workflow in workflows
        ]
        x = np.arange(len(labels))
        ax.bar(x - 0.16, rss_values, width=0.30, color="#2f5597", label="peak delta RSS")
        finite_uss_summary = np.asarray([value for value in uss_values if np.isfinite(value)], dtype=np.float32)
        if finite_uss_summary.size:
            ax.bar(x + 0.16, uss_values, width=0.30, color="#70ad47", alpha=0.82, label="peak delta USS")
        raw_size_mb = float(summary.get("raw_size_mb", np.nan))
        if np.isfinite(raw_size_mb):
            ax.axhline(raw_size_mb, color="#595959", linestyle=":", linewidth=1.1, label="raw stack size")
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=24, ha="right", fontsize=FONT_SIZE)
        finite_summary = np.asarray([value for value in [*rss_values, *uss_values, raw_size_mb] if np.isfinite(value)], dtype=np.float32)
        ax.set_ylim(0, float(np.max(finite_summary)) * 1.12 if finite_summary.size else 1)
        if not finite_uss_summary.size:
            ax.text(
                0.98,
                0.90,
                "USS not available\nfor this run",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=LEGEND_SIZE,
                color="#595959",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 2},
            )
        _maybe_legend(ax, FIG5_PANELS["peak"])
        _save(fig, FIG5_PANELS["peak"])

        if timing_rows:
            fig, ax = _new_fig(FIG5_PANELS["duration"])
            steps = ["load", "register", "save"]
            workflows = ["memmap_cold", "memmap_warm", "in_memory"]
            workflow_labels = {"memmap_cold": "MM cold", "memmap_warm": "MM warm", "in_memory": "RAM"}
            x = np.arange(len(steps))
            width = 0.25
            for offset, workflow, color in [
                (-0.27, "memmap_cold", "#5b9bd5"),
                (0.0, "memmap_warm", "#70ad47"),
                (0.27, "in_memory", "#d95f02"),
            ]:
                values = []
                for step in steps:
                    matches = [row for row in timing_rows if row.get("workflow") == workflow and row.get("step") == step]
                    values.append(_float(matches[0], "duration_s") if matches else np.nan)
                ax.bar(x + offset, values, width=width, color=color, label=workflow_labels.get(workflow, workflow))
            ax.set_xticks(x)
            ax.set_xticklabels(steps, fontsize=FONT_SIZE)
            _maybe_legend(ax, FIG5_PANELS["duration"])
            _save(fig, FIG5_PANELS["duration"])
        else:
            _placeholder(FIG5_PANELS["duration"], "Memory timing summary missing.\nRun --case memory.")
    else:
        _placeholder(FIG5_PANELS["peak"], "Memory summary missing.\nRun --case memory.")
        _placeholder(FIG5_PANELS["duration"], "Memory summary missing.\nRun --case memory.")

    assemble_figure("figure5", ["panel_a.png", "panel_b.png", "panel_c.png", "panel_d.png", "panel_e.png"], columns=3, title="Memory-efficient workflow")


# %% FIGURE 6
def plot_figure6() -> None:
    rows = _read_rows(BENCHMARK_DIR / "projection_parallel_benchmark_metrics.csv")
    if not rows:
        for panel in FIG6_PANELS.values():
            _placeholder(panel, "Projection/parallel benchmark missing.\nRun --case projection_parallel.")
        assemble_figure("figure6", [f"panel_{letter}.png" for letter in "abcd"], columns=2, title="Projection and parallelization benchmarks")
        return

    for scenario, panel_key in [("lateral_motion", "workflow_lateral"), ("axial_motion", "workflow_axial")]:
        grouped = _group_mean(rows, scenario=scenario, value_key="abs_error_mean", group_keys=("workflow",))
        fig, ax = _new_fig(FIG6_PANELS[panel_key])
        preferred_workflow_order = [("projection",), ("full_3d",)]
        workflows = [workflow for workflow in preferred_workflow_order if workflow in grouped]
        workflows.extend(workflow for workflow in sorted(grouped) if workflow not in workflows)
        values = [grouped[workflow] for workflow in workflows]
        ax.bar(
            np.arange(len(workflows)),
            values,
            color=[COLORS.get(workflow[0], "#595959") for workflow in workflows],
        )
        ax.set_xticks(np.arange(len(workflows)))
        ax.set_xticklabels([LABELS.get(workflow[0], workflow[0]) for workflow in workflows], rotation=15, ha="right", fontsize=FONT_SIZE)
        _save(fig, FIG6_PANELS[panel_key])

    runtime_grouped = _group_mean(
        [row for row in rows if row.get("scenario") in {"lateral_motion", "axial_motion"}],
        scenario="lateral_motion",
        value_key="runtime_s",
        group_keys=("workflow",),
    )
    runtime_grouped.update({
        ("axial:" + key[0],): value
        for key, value in _group_mean(rows, scenario="axial_motion", value_key="runtime_s", group_keys=("workflow",)).items()
    })
    fig, ax = _new_fig(FIG6_PANELS["workflow_runtime"])
    preferred_runtime_order = [("projection",), ("full_3d",), ("axial:projection",), ("axial:full_3d",)]
    labels = [label for label in preferred_runtime_order if label in runtime_grouped]
    labels.extend(label for label in runtime_grouped if label not in labels)
    values = [runtime_grouped[label] for label in labels]
    ax.bar(np.arange(len(labels)), values, color=["#2f5597", "#7030a0", "#6f9ed8", "#b095d6"][: len(labels)])
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels([label[0].replace("axial:", "axial ") for label in labels], rotation=25, ha="right", fontsize=FONT_SIZE)
    _save(fig, FIG6_PANELS["workflow_runtime"])

    parallel_rows = [row for row in rows if row.get("scenario") == "parallel_scaling"]
    fig, ax = _new_fig(FIG6_PANELS["parallel"])
    for marker_index, n_jobs in enumerate(sorted({int(row.get("n_jobs", 1)) for row in parallel_rows})):
        points = sorted(
            [(int(row["time_count"]), _float(row, "runtime_s")) for row in parallel_rows if int(row.get("n_jobs", 1)) == n_jobs],
            key=lambda item: item[0],
        )
        if not points:
            continue
        xs, ys = np.asarray(points).T
        ax.plot(
            xs,
            ys,
            marker=PARALLEL_MARKERS[marker_index % len(PARALLEL_MARKERS)],
            linewidth=1.3,
            markersize=3,
            label=f"n_jobs={n_jobs}")
    ax.set_xscale("log")
    time_ticks = sorted({int(row["time_count"]) for row in parallel_rows if row.get("time_count")})
    ax.set_xticks(time_ticks)
    ax.set_xticklabels([str(value) for value in time_ticks], rotation=45, ha="right", fontsize=LEGEND_SIZE)
    _maybe_legend(ax, FIG6_PANELS["parallel"])
    _save(fig, FIG6_PANELS["parallel"])

    assemble_figure("figure6", [f"panel_{letter}.png" for letter in "abcd"], columns=2, title="Projection and parallelization benchmarks")


# %% COMPOSITE HELPER
def assemble_figure(figure_dir_name: str, panel_names: list[str | None], *, columns: int, title: str) -> None:
    figure_dir = FIGURES_DIR / figure_dir_name
    rows = int(np.ceil(len(panel_names) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 3.2, rows * 2.65))
    axes = np.atleast_1d(axes).ravel()
    letter_index = 0
    letters = "abcdefghijklmnopqrstuvwxyz"
    for ax, panel_name in zip(axes, panel_names, strict=False):
        ax.axis("off")
        if panel_name is None:
            continue
        panel_path = figure_dir / panel_name
        if not panel_path.exists():
            panel_path = panel_path.with_suffix(".png")
        if panel_path.exists():
            ax.imshow(plt.imread(panel_path))
        else:
            ax.text(0.5, 0.5, panel_name, ha="center", va="center", fontsize=FONT_SIZE)
        letter = letters[letter_index]
        ax.text(0.01, 0.99, letter, transform=ax.transAxes, ha="left", va="top", fontsize=FONT_SIZE, weight="bold")
        letter_index += 1
    for ax in axes[len(panel_names) :]:
        ax.axis("off")
    fig.suptitle(title, fontsize=FONT_SIZE)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{figure_dir_name}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / f"{figure_dir_name}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


# %% MAIN FUNCTION
def main() -> None:
    plot_figure2()
    plot_figure3()
    plot_figure4()
    plot_figure5()
    plot_figure6()
    print(f"Wrote preprint figure panels to {FIGURES_DIR}")
#%% MAIN ENTRY POINT
if __name__ == "__main__":
    main()
# %% END
