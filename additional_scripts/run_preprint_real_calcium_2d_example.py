"""Real 2D+t calcium-imaging registration example for the ZenReg preprint.

This script is intentionally written like an interactive user script. In VS Code
or Spyder, run the ``# %%`` cells one by one. From the terminal, run the whole
script:

    conda run -n zenreg2 python additional_scripts/run_preprint_real_calcium_2d_example.py

The CaImAn demo movie itself is not shipped with ZenReg. Download instructions
are provided in ``example_data/CaImAn_example_Ca_file/README.md``.
"""
# %% IMPORTS
import csv
import json
import os
import tempfile
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "zenreg-matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "zenreg-xdg-cache"))

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from zenreg import create_stack_metadata, load_stack, register_stack, save_stack, update_stack_metadata
from zenreg.registration import _compute_registration_frame_correlations
# %% PATHS, STYLE AND REGISTRATION SETTINGS
REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = REPO_ROOT / "example_data" / "CaImAn_example_Ca_file" / "Sue_2x_3000_40_-46.tif"
PREPRINT_DIR = REPO_ROOT / "papers" / "preprint"
OUTPUT_DIR = PREPRINT_DIR / "benchmark_outputs" / "real_calcium_2d_t"
FIGURE_DIR = PREPRINT_DIR / "figures" / "figure7"
FIGURE_DRAFT = PREPRINT_DIR / "figures" / "figure7"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

CM_TO_INCH = 1.0 / 2.54
PANEL_DPI = 300
FONT_SIZE = 10
LEGEND_SIZE = 8

plt.rcParams.update(
    {   "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": FONT_SIZE,
        "axes.titlesize": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "legend.fontsize": LEGEND_SIZE,
        "pdf.fonttype": 42,
        "ps.fonttype": 42})


# Set MAX_FRAMES to a small integer for quick interactive tests, or None for the
# full 3000-frame CaImAn movie.
MAX_FRAMES = None
MOVING_TIME_FOR_PLOTS = 250
N_JOBS = 7

REGISTRATION_CHANNEL= 0
REGISTRATION_TEMPLATE_TIME_RANGE = "all"
REGISTRATION_METHOD = "phase_cross_correlation"
TIME_REGISTRATION_MODE = "projection"
TIME_REFERENCE_MODE = "template"
PROJECTION_METHOD   = "max"
MAX_XY_SHIFTS       = (5, 5)
ZERO_CLIP           = True
FILTER_SLICES       = True
FILTER_PROJECTIONS  = True
MEDIAN_KERNEL_SIZE  = 3

REGISTERED_OUTPUT_PATH  = OUTPUT_DIR / "caiman_sue_2d_t_zenreg_registered.ome.tif"
METRICS_CSV_PATH        = OUTPUT_DIR / "caiman_sue_2d_t_metrics.csv"
SUMMARY_JSON_PATH       = OUTPUT_DIR / "caiman_sue_2d_t_summary.json"
# %% HELPER FUNCTIONS
def normalize_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    lo, hi = np.nanpercentile(image, [1.0, 99.7])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image - lo) / (hi - lo), 0.0, 1.0)

def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    a = a - float(np.mean(a))
    b = b - float(np.mean(b))
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return float("nan")
    return float(np.dot(a, b) / denom)

def local_correlation_image(movie_tyx: np.ndarray) -> np.ndarray:
    """Compute a compact CaImAn-style local correlation image."""

    movie = np.asarray(movie_tyx, dtype=np.float32)
    movie = movie - np.nanmean(movie, axis=0, keepdims=True)
    std = np.nanstd(movie, axis=0, keepdims=True)
    normed = movie / np.maximum(std, 1e-6)
    corr = np.zeros(movie.shape[1:], dtype=np.float32)
    count = np.zeros(movie.shape[1:], dtype=np.float32)
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        y_src = slice(max(0, -dy), movie.shape[1] - max(0, dy))
        y_dst = slice(max(0, dy), movie.shape[1] - max(0, -dy))
        x_src = slice(max(0, -dx), movie.shape[2] - max(0, dx))
        x_dst = slice(max(0, dx), movie.shape[2] - max(0, -dx))
        values = np.nanmean(normed[:, y_src, x_src] * normed[:, y_dst, x_dst], axis=0)
        corr[y_src, x_src] += values
        corr[y_dst, x_dst] += values
        count[y_src, x_src] += 1.0
        count[y_dst, x_dst] += 1.0
    return np.divide(corr, np.maximum(count, 1.0), out=np.zeros_like(corr), where=count > 0)

def save_image_panel(
    name: str,
    image: np.ndarray,
    title: str,
    *,
    cmap: str = "gray",
    diverging: bool = False,
    vmax_abs: float | None = None,
    figsize_cm: tuple[float, float] = (5.2, 5.0)
) -> Path:
    fig, ax = plt.subplots(figsize=(figsize_cm[0] * CM_TO_INCH, figsize_cm[1] * CM_TO_INCH))
    ax.set_title(title, pad=5)
    ax.axis("off")
    if diverging:
        vmax = float(vmax_abs) if vmax_abs is not None else float(np.nanmax(np.abs(image)))
        vmax = max(vmax, 1e-6)
        ax.imshow(image, cmap=cmap, vmin=-vmax, vmax=vmax)
    else:
        ax.imshow(normalize_image(image), cmap=cmap, vmin=0, vmax=1)
    output = FIGURE_DIR / name
    fig.savefig(output, dpi=PANEL_DPI, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return output

def save_line_panel(
    name: str,
    title: str,
    xlabel: str,
    ylabel: str,
    plotter,
    *,
    figsize_cm: tuple[float, float] = (8.4, 4.0),
    ylim: tuple[float | None, float | None] | None = None
) -> Path:
    fig, ax = plt.subplots(figsize=(figsize_cm[0] * CM_TO_INCH, figsize_cm[1] * CM_TO_INCH))
    ax.set_title(title, pad=5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plotter(ax)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False, fontsize=LEGEND_SIZE)
    output = FIGURE_DIR / name
    fig.savefig(output, dpi=PANEL_DPI if output.suffix == ".png" else None, bbox_inches="tight", transparent=True)
    if output.suffix == ".pdf":
        fig.savefig(output.with_suffix(".png"), dpi=PANEL_DPI, bbox_inches="tight")
    plt.close(fig)
    return output

def write_metric_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def assemble_figure7() -> None:
    panel_names = [
        "panel_a.png",
        "panel_b.png",
        "panel_c.png",
        "panel_d.png",
        "panel_e.png",
        "panel_f.png",
        "panel_g.png",
        "panel_h.png",
        "panel_i.png",
        None,
        "panel_j.png",
        "panel_k.png",
    ]
    columns = 5
    rows = int(np.ceil(len(panel_names) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 3.1, rows * 2.65))
    axes = np.atleast_1d(axes).ravel()
    letters = "abcdefghijklmnopqrstuvwxyz"
    letter_index = 0
    for ax, panel_name in zip(axes, panel_names, strict=False):
        ax.axis("off")
        if panel_name is None:
            continue
        panel_path = FIGURE_DIR / panel_name
        if panel_path.exists():
            ax.imshow(plt.imread(panel_path))
        else:
            ax.text(0.5, 0.5, panel_name, ha="center", va="center")
        ax.text(0.01, 0.99, letters[letter_index], transform=ax.transAxes, ha="left", va="top", weight="bold")
        letter_index += 1
    for ax in axes[len(panel_names) :]:
        ax.axis("off")
    fig.suptitle("Real 2D+t calcium-imaging registration", fontsize=FONT_SIZE)
    fig.tight_layout()
    fig.savefig(FIGURE_DRAFT.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(FIGURE_DRAFT.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)
# %% LOAD AND ADJUST STACK
if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"Missing CaImAn example movie: {INPUT_PATH}. "
        "Download it using the instructions in example_data/CaImAn_example_Ca_file/README.md.")

stack_2d_t_xy, metadata_2d_t_xy = load_stack(
    INPUT_PATH,
    return_metadata=True,
    verbose=False)
print(f"Loaded CaImAn movie as {stack_2d_t_xy.shape} (TZCYX)")

# The CaImAn demo TIFF stores time as channels. OMIO preserves that faithfully,
# so we explicitly move C to T and keep a single registration channel.
if stack_2d_t_xy.shape[0] == 1 and stack_2d_t_xy.shape[1] == 1 and stack_2d_t_xy.shape[2] > 1:
    stack_2d_t_xy = np.swapaxes(stack_2d_t_xy, 2, 0)
    stack_2d_t_xy = np.ascontiguousarray(stack_2d_t_xy[:, :, 0:1, :, :])
elif not (stack_2d_t_xy.shape[1] == 1 and stack_2d_t_xy.shape[2] == 1):
    raise ValueError(f"Expected a single-channel 2D+t movie after axis normalization; got shape {stack_2d_t_xy.shape}.")

if MAX_FRAMES is not None:
    stack_2d_t_xy = stack_2d_t_xy[: int(MAX_FRAMES)].copy()

metadata_2d_t_xy = update_stack_metadata(
    create_stack_metadata(stack_2d_t_xy, input_metadata=metadata_2d_t_xy, verbose=False),
    stack_2d_t_xy,
    verbose=False)
print(f"Using CaImAn movie as {stack_2d_t_xy.shape} (TZCYX)")
# %% REGISTER STACK
registration_start = time.perf_counter()
registered_2d_t_xy, details_2d_t_xy = register_stack(
    stack_2d_t_xy,
    metadata=metadata_2d_t_xy,
    registration_channel=REGISTRATION_CHANNEL,
    registration_template_time_range=REGISTRATION_TEMPLATE_TIME_RANGE,
    method=REGISTRATION_METHOD,
    time_registration_mode=TIME_REGISTRATION_MODE,
    time_reference_mode=TIME_REFERENCE_MODE,
    projection_method=PROJECTION_METHOD,
    zreg=False,
    zero_clip=ZERO_CLIP,
    max_xy_shifts=MAX_XY_SHIFTS,
    transform_backend="skimage",
    transform_order=1,
    filter_slices=FILTER_SLICES,
    filter_projections=FILTER_PROJECTIONS,
    median_kernel_size=MEDIAN_KERNEL_SIZE,
    n_jobs=N_JOBS,
    calc_SNR=True,
    calc_CNR=True,
    verbose=True,
    return_shifts=True,
    return_details=True)
registration_runtime_s = time.perf_counter() - registration_start
print(f"Registration finished in {registration_runtime_s:.2f} s; output shape={registered_2d_t_xy.shape} (TZCYX)")

# save stack and zenreg report sidecars:
registered_output_path = save_stack(
    REGISTERED_OUTPUT_PATH,
    registered_2d_t_xy,
    metadata=metadata_2d_t_xy,
    registration_details=details_2d_t_xy,
    overwrite=True,
    verbose=False)
print(f"Wrote registered image and ZenReg sidecars to {registered_output_path}")
# %% PREPRINT ANALYSIS
moving_time = min(int(MOVING_TIME_FOR_PLOTS), stack_2d_t_xy.shape[0] - 1)
raw_movie = np.asarray(stack_2d_t_xy[:, 0, 0], dtype=np.float32)
registered_movie = np.asarray(registered_2d_t_xy[:, 0, 0], dtype=np.float32)

template_image = np.nanmax(raw_movie, axis=0)
raw_t0_image = raw_movie[0]
raw_t1_image = raw_movie[moving_time]
registered_t0_image = registered_movie[0]
registered_t1_image = registered_movie[moving_time]
raw_difference_image = raw_t1_image - raw_t0_image #raw_t0_image template_image
registered_difference_image = registered_t1_image - registered_t0_image #registered_t0_image template_image
difference_vmax_abs = max(
    float(np.nanmax(np.abs(raw_difference_image))),
    float(np.nanmax(np.abs(registered_difference_image))),
    1e-6)

raw_local_correlation_image = local_correlation_image(raw_movie)
registered_local_correlation_image = local_correlation_image(registered_movie)

detected_shifts_yx = np.asarray(details_2d_t_xy["time_shifts_yx"], dtype=np.float32)
frames = np.arange(detected_shifts_yx.shape[0])
pearson_before = np.asarray(details_2d_t_xy["pearson_correlations_before"], dtype=np.float32)
pearson_after = _compute_registration_frame_correlations(
    registered_2d_t_xy,
    registration_channel=REGISTRATION_CHANNEL,
    registration_stack=0,
    registration_template_time_range=(0, registered_2d_t_xy.shape[0]),
    projection_range=None,
    projection_method=PROJECTION_METHOD,
    effective_time_registration_mode=TIME_REGISTRATION_MODE)

metric_rows = []
snr_values = np.asarray(details_2d_t_xy.get("snr_before", np.full(frames.shape, np.nan)), dtype=np.float32)
cnr_values = np.asarray(details_2d_t_xy.get("cnr_before", np.full(frames.shape, np.nan)), dtype=np.float32)
for frame in frames:
    metric_rows.append(
        {
            "frame": int(frame),
            "shift_y": float(detected_shifts_yx[frame, 0]),
            "shift_x": float(detected_shifts_yx[frame, 1]),
            "pearson_before": float(pearson_before[frame]),
            "pearson_after": float(pearson_after[frame]),
            "snr": float(snr_values[frame]),
            "cnr": float(cnr_values[frame]),
        }
    )
write_metric_csv(METRICS_CSV_PATH, metric_rows)

summary = {
    "input": str(INPUT_PATH),
    "output_image": str(registered_output_path),
    "raw_shape_tzcyx": list(stack_2d_t_xy.shape),
    "registered_shape_tzcyx": list(registered_2d_t_xy.shape),
    "moving_time": int(moving_time),
    "runtime_s": float(registration_runtime_s),
    "max_abs_shift_y": float(np.nanmax(np.abs(detected_shifts_yx[:, 0]))),
    "max_abs_shift_x": float(np.nanmax(np.abs(detected_shifts_yx[:, 1]))),
    "mean_pearson_before": float(np.nanmean(pearson_before)),
    "mean_pearson_after": float(np.nanmean(pearson_after)),
}
SUMMARY_JSON_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(f"Wrote analysis metrics to {METRICS_CSV_PATH}")
# %% PREPRINT PLOT SETTINGS
def plot_translation_estimates(ax):
    ax.plot(frames, detected_shifts_yx[:, 0], color="#3b7cec", linewidth=1.2, label="shift y")
    ax.plot(frames, detected_shifts_yx[:, 1], color="#d95f02", linewidth=1.2, label="shift x")

def plot_template_correlation(ax):
    ax.plot(frames, pearson_before, color="#a5a5a5", linewidth=1.0, label="r before")
    ax.plot(frames, pearson_after, color="#7030a0", linewidth=1.2, label="r after")
# %% PREPRINT PANEL PLOTS
save_image_panel("panel_a.png", template_image, "all-frame max template")
save_image_panel("panel_b.png", raw_t0_image, "raw t=0")
save_image_panel("panel_c.png", raw_t1_image, f"raw t={moving_time}")
save_image_panel("panel_d.png", raw_difference_image, f"raw t$_{{{moving_time}}}$-t$_0$", cmap="coolwarm", diverging=True, vmax_abs=difference_vmax_abs)
save_image_panel("panel_e.png", raw_local_correlation_image, "raw local correlation", cmap="magma")
save_image_panel("panel_f.png", registered_t0_image, "registered t=0")
save_image_panel("panel_g.png", registered_t1_image, f"registered t={moving_time}")
save_image_panel("panel_h.png", registered_difference_image, f"registered t$_{{{moving_time}}}$-t$_0$", cmap="coolwarm", diverging=True, vmax_abs=difference_vmax_abs)
save_image_panel("panel_i.png", registered_local_correlation_image, "registered local correlation", cmap="magma")
save_line_panel("panel_j.pdf", "translation estimate", "frame", "correction (px)", plot_translation_estimates)
save_line_panel("panel_k.pdf", "template correlation", "frame", "Pearson r", plot_template_correlation, ylim=(0, 1.05))
# %% PREPRINT DRAFT FIGURE
assemble_figure7()
print(f"Wrote Figure 7 panels to {FIGURE_DIR}")
# %% END
