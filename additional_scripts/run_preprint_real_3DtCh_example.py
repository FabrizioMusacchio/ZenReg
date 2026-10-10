"""Real 2P 3D+t+Ch registration example for the ZenReg preprint/revision.

This script is intentionally written like an interactive user script. In VS Code
or Spyder, run the ``# %%`` cells one by one. From the terminal, run the whole
script:

    conda run -n zenreg2 python additional_scripts/run_preprint_real_3DtCh_example.py

The example stack is a true two-channel 3D+t microscopy time series. No
synthetic pseudo-time shifts are added here; ZenReg estimates the motion present
in the real data and applies the correction to both channels.

author: Fabrizio Musacchio
date:   October 2026
"""
# %% IMPORTS
import csv
import json
import os
import shutil
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
# %% PATHS AND STYLE
REPO_ROOT    = Path(__file__).resolve().parents[1]
INPUT_PATH   = REPO_ROOT / "example_data" / "MotilA_example_data" / "TLmerged_2" / "240321_21584_P17_3_merged_1_8.ome.tif"
PREPRINT_DIR = REPO_ROOT / "papers" / "preprint"
OUTPUT_DIR   = PREPRINT_DIR / "benchmark_outputs" / "real_3DtCh"
FIGURE_DIR   = PREPRINT_DIR / "figures" / "figure9"
FIGURE_DRAFT = PREPRINT_DIR / "figures" / "figure9"

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
# %% EXAMPLE SETTINGS
# First-look defaults use a central subvolume so the script can be run quickly
# during revision work. Set USE_SUBVOLUME=False once the settings are settled.
USE_MEMMAP_FOR_SOURCE = True
MEMMAP_FOLDER = OUTPUT_DIR / "omio_memmap_cache"
USE_SUBVOLUME = False
Z_RANGE: tuple[int, int] | None = None
YX_SIZE: int | None = 500

REGISTRATION_CHANNEL = 1
DISPLAY_CHANNELS = (0, 1)
DISPLAY_TIME = None  # None -> choose the frame with the largest estimated 3D shift
N_JOBS = 7

REGISTERED_OUTPUT_PATH = OUTPUT_DIR / "real_3DtCh_registered.ome.tif"
METRICS_CSV_PATH  = OUTPUT_DIR / "real_3DtCh_metrics.csv"
SUMMARY_JSON_PATH = OUTPUT_DIR / "real_3DtCh_summary.json"

REGISTRATION_OPTIONS = dict(
    registration_channel=REGISTRATION_CHANNEL,
    registration_stack=0,
    method="phase_cross_correlation", # phase_cross_correlation pystackreg
    time_registration_mode="full_3d", # projection full_3d
    time_reference_mode="template",
    registration_template_time_range=None, # all or None
    zreg=True,
    rotreg=False,
    #rigid_3d_backend="simpleitk"
    max_xy_shifts=(80, 80),
    max_z_shifts=15,
    max_rot_shifts=5,
    zero_clip=True,
    transform_backend="skimage",
    transform_order=1,
    filter_slices=True,
    filter_projections=True,
    median_kernel_size=3,
    n_jobs=N_JOBS,
    output_use_memmap=True,
    output_memmap_folder=OUTPUT_DIR / "zenreg_output_memmap",
    output_memmap_name="real_3DtCh_registered",
    output_dtype=np.float32,
    calc_SNR=True,
    calc_CNR=True,
    SNR_sampling_step=1,
    CNR_sampling_step=1,
    print_shifts=False,
    verbose=True,
    return_shifts=True,
    return_details=True)
# %% HELPER FUNCTIONS
def normalize_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    lo, hi = np.nanpercentile(image, [1.0, 99.7])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image - lo) / (hi - lo), 0.0, 1.0)

def robust_difference_vmax(*images: np.ndarray) -> float:
    values = []
    for image in images:
        arr = np.asarray(image, dtype=np.float32)
        if arr.size:
            values.append(float(np.nanpercentile(np.abs(arr), 99.5)))
    return max(values + [1e-6])

def pearson_volume(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    valid = np.isfinite(a) & np.isfinite(b)
    if int(np.count_nonzero(valid)) < 3:
        return float("nan")
    a = a[valid] - float(np.mean(a[valid]))
    b = b[valid] - float(np.mean(b[valid]))
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return float("nan")
    return float(np.dot(a, b) / denom)

def centered_bounds(size: int, count: int | None) -> tuple[int, int]:
    if count is None:
        return 0, int(size)
    count = min(int(count), int(size))
    start = max((int(size) - count) // 2, 0)
    return start, start + count

def select_subvolume(stack: np.ndarray, metadata: dict) -> tuple[np.ndarray, dict, dict]:
    if Z_RANGE is None:
        z_start, z_stop = 0, stack.shape[1]
    else:
        z_start = max(int(Z_RANGE[0]), 0)
        z_stop = min(int(Z_RANGE[1]), stack.shape[1])
    if USE_SUBVOLUME:
        y_start, y_stop = centered_bounds(stack.shape[3], YX_SIZE)
        x_start, x_stop = centered_bounds(stack.shape[4], YX_SIZE)
    else:
        y_start, y_stop = 0, stack.shape[3]
        x_start, x_stop = 0, stack.shape[4]
    selected = np.asarray(stack[:, z_start:z_stop, :, y_start:y_stop, x_start:x_stop])
    selected_metadata = update_stack_metadata(
        create_stack_metadata(selected, input_metadata=metadata, verbose=False),
        selected,
        verbose=False)
    crop = {
        "z_start": int(z_start),
        "z_stop": int(z_stop),
        "y_start": int(y_start),
        "y_stop": int(y_stop),
        "x_start": int(x_start),
        "x_stop": int(x_stop)}
    return selected, selected_metadata, crop

def frame_volume_correlations(stack: np.ndarray, *, channel: int, reference_time: int = 0) -> np.ndarray:
    reference = np.asarray(stack[int(reference_time), :, int(channel)], dtype=np.float32)
    return np.asarray([pearson_volume(reference, stack[t, :, int(channel)]) for t in range(stack.shape[0])], dtype=np.float32)

def max_projection(stack: np.ndarray, *, t: int, channel: int) -> np.ndarray:
    return np.max(np.asarray(stack[int(t), :, int(channel)], dtype=np.float32), axis=0)

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
        im = ax.imshow(image, cmap=cmap, vmin=-vmax, vmax=vmax)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
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
    figsize_cm: tuple[float, float] = (6.2, 4.2),
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

def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, type):
        return value.__name__
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value

def copy_summary_plot(details: dict) -> None:
    summary_path = details.get("summary_plot_path") or details.get("plot")
    if summary_path is None and REGISTERED_OUTPUT_PATH.exists():
        candidate = REGISTERED_OUTPUT_PATH.with_name(f"{REGISTERED_OUTPUT_PATH.stem}_registration_summary.png")
        if candidate.exists():
            summary_path = candidate
    if summary_path is None:
        return
    summary_path = Path(summary_path)
    if summary_path.exists():
        shutil.copy2(summary_path, FIGURE_DIR / "panel_m.png")

def assemble_figure9() -> None:
    panel_names = [f"panel_{letter}.png" for letter in "abcdefghijklm"]
    columns = 5
    rows = int(np.ceil(len(panel_names) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 3.1, rows * 2.65))
    axes = np.atleast_1d(axes).ravel()
    letters = "abcdefghijklmnopqrstuvwxyz"
    for index, (ax, panel_name) in enumerate(zip(axes, panel_names, strict=False)):
        ax.axis("off")
        panel_path = FIGURE_DIR / panel_name
        if panel_path.exists():
            ax.imshow(plt.imread(panel_path))
        else:
            ax.text(0.5, 0.5, panel_name, ha="center", va="center")
        ax.text(0.01, 0.99, letters[index], transform=ax.transAxes, ha="left", va="top", weight="bold")
    for ax in axes[len(panel_names):]:
        ax.axis("off")
    fig.suptitle("Real 2P 3D+t+Ch registration", fontsize=FONT_SIZE)
    fig.tight_layout()
    fig.savefig(FIGURE_DRAFT.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(FIGURE_DRAFT.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)
# %% LOAD TRUE 3D+t+Ch STACK
if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"Missing MotilA example stack: {INPUT_PATH}. "
        "Download or place the example data in example_data/MotilA_example_data/.")

stack_raw, metadata_raw = load_stack(
    INPUT_PATH,
    return_metadata=True,
    use_memmap=USE_MEMMAP_FOR_SOURCE,
    memmap_folder=MEMMAP_FOLDER,
    verbose=True)
print(f"Loaded true 3D+t+Ch stack: {stack_raw.shape} (TZCYX), dtype={stack_raw.dtype}")

stack_3dtch, metadata_3dtch, crop_bounds = select_subvolume(stack_raw, metadata_raw)
print(f"Using analysis stack: {stack_3dtch.shape} (TZCYX), crop={crop_bounds}, dtype={stack_3dtch.dtype}")
# %% REGISTER TRUE 3D+t+Ch STACK
registration_start = time.perf_counter()
registered_3dtch, details_3dtch = register_stack(stack_3dtch, metadata=metadata_3dtch, **REGISTRATION_OPTIONS)
runtime_s = time.perf_counter() - registration_start
print(f"Real 3D+t+Ch registration finished in {runtime_s:.2f} s")
print(f"Registered stack shape: {registered_3dtch.shape} (TZCYX)")

""" # open registered_3dtch in napari for visualization
import omio as om
om.open_in_napari(registered_3dtch, metadata_3dtch) """

registered_output_path = save_stack(
    REGISTERED_OUTPUT_PATH,
    registered_3dtch,
    metadata=metadata_3dtch,
    registration_details=details_3dtch,
    overwrite=True,
    verbose=True)
print(f"Wrote registered true 3D+t+Ch stack to {registered_output_path}")
# %% PREPRINT ANALYSIS
estimated_shifts_zyx = np.asarray(details_3dtch.get("time_shifts_zyx", []), dtype=np.float32)
if estimated_shifts_zyx.size == 0:
    estimated_shifts_zyx = np.zeros((stack_3dtch.shape[0], 3), dtype=np.float32)
frames = np.arange(stack_3dtch.shape[0])
shift_magnitude = np.linalg.norm(estimated_shifts_zyx, axis=1)
reference_time = int(REGISTRATION_OPTIONS.get("registration_stack", 0))
if DISPLAY_TIME is None:
    candidate_magnitudes = np.array(shift_magnitude, dtype=np.float32)
    candidate_magnitudes[int(reference_time)] = -np.inf
    moving_time = int(np.nanargmax(candidate_magnitudes))
else:
    moving_time = int(DISPLAY_TIME)
moving_time = int(np.clip(moving_time, 0, stack_3dtch.shape[0] - 1))

before_correlations = {}
after_correlations = {}
for channel in DISPLAY_CHANNELS:
    before_correlations[int(channel)] = frame_volume_correlations(stack_3dtch, channel=int(channel), reference_time=reference_time)
    after_correlations[int(channel)] = frame_volume_correlations(registered_3dtch, channel=int(channel), reference_time=reference_time)

image_panels = {}
for channel in DISPLAY_CHANNELS:
    channel = int(channel)
    reference_projection = max_projection(stack_3dtch, t=reference_time, channel=channel)
    moving_projection = max_projection(stack_3dtch, t=moving_time, channel=channel)
    registered_reference_projection = max_projection(registered_3dtch, t=reference_time, channel=channel)
    registered_projection = max_projection(registered_3dtch, t=moving_time, channel=channel)
    raw_difference = reference_projection - moving_projection
    registered_difference = registered_reference_projection - registered_projection
    difference_vmax_abs = robust_difference_vmax(raw_difference, registered_difference)
    image_panels[channel] = {
        "reference": reference_projection,
        "moving": moving_projection,
        "registered_reference": registered_reference_projection,
        "registered": registered_projection,
        "raw_difference": raw_difference,
        "registered_difference": registered_difference,
        "difference_vmax_abs": difference_vmax_abs}

metric_rows = []
for frame in frames:
    row = {
        "frame": int(frame),
        "runtime_s": float(runtime_s),
        "shift_magnitude_px": float(shift_magnitude[frame])}
    for axis_index, axis_name in enumerate(("z", "y", "x")):
        row[f"estimated_shift_{axis_name}"] = float(estimated_shifts_zyx[frame, axis_index])
    for channel in DISPLAY_CHANNELS:
        channel = int(channel)
        row[f"channel_{channel}_pearson_before"] = float(before_correlations[channel][frame])
        row[f"channel_{channel}_pearson_after"] = float(after_correlations[channel][frame])
    metric_rows.append(row)
write_csv(METRICS_CSV_PATH, metric_rows)

summary = {
    "input": str(INPUT_PATH),
    "raw_shape_tzcyx": list(stack_raw.shape),
    "analysis_shape_tzcyx": list(stack_3dtch.shape),
    "registered_shape_tzcyx": list(registered_3dtch.shape),
    "crop_bounds": crop_bounds,
    "registration_options": {
        key: json_safe(value)
        for key, value in REGISTRATION_OPTIONS.items()
        if key not in {"return_shifts", "return_details"}},
    "reference_time": int(reference_time),
    "representative_moving_time": int(moving_time),
    "runtime_s": float(runtime_s),
    "mean_shift_magnitude_px": float(np.nanmean(shift_magnitude)),
    "max_shift_magnitude_px": float(np.nanmax(shift_magnitude))}
for channel in DISPLAY_CHANNELS:
    channel = int(channel)
    summary[f"channel_{channel}_mean_correlation_before"] = float(np.nanmean(before_correlations[channel]))
    summary[f"channel_{channel}_mean_correlation_after"] = float(np.nanmean(after_correlations[channel]))
SUMMARY_JSON_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(f"Wrote analysis metrics to {METRICS_CSV_PATH}")
print(f"Representative moving frame: t={moving_time}; shift={estimated_shifts_zyx[moving_time].tolist()} (z,y,x)")
# %% PREPRINT PLOT SETTINGS
def plot_shifts(ax):
    ax.axhline(0, color="#7f7f7f", linewidth=0.7)
    ax.plot(frames, estimated_shifts_zyx[:, 2], marker="^", color="#31a354", linewidth=1.3, label="x")
    ax.plot(frames, estimated_shifts_zyx[:, 1], marker="s", color="#f03b20", linewidth=1.3, label="y")
    ax.plot(frames, estimated_shifts_zyx[:, 0], marker="o", color="#2c7fb8", linewidth=1.3, label="z")

def plot_volume_correlation(ax):
    colors = {0: "#7030a0", 1: "#008a9b"}
    for channel in DISPLAY_CHANNELS:
        channel = int(channel)
        color = colors.get(channel, "#333333")
        ax.plot(frames, before_correlations[channel], linestyle=":", marker="o", color=color, alpha=0.65, linewidth=1.0, label=f"ch{channel} before")
        ax.plot(frames, after_correlations[channel], linestyle="-", marker="s", color=color, linewidth=1.3, label=f"ch{channel} after")

def plot_summary(ax):
    channels = [int(channel) for channel in DISPLAY_CHANNELS]
    before = [float(np.nanmean(before_correlations[channel])) for channel in channels]
    after = [float(np.nanmean(after_correlations[channel])) for channel in channels]
    x = np.arange(len(channels))
    width = 0.34
    ax.bar(x - width / 2, before, width=width, color="#bdbdbd", label="before")
    ax.bar(x + width / 2, after, width=width, color="#4daf4a", label="after")
    ax.set_xticks(x)
    ax.set_xticklabels([f"ch{channel}" for channel in channels])
    ax.set_ylim(0, 1.05)
# %% PREPRINT PANEL PLOTS
channel0, channel1 = [int(channel) for channel in DISPLAY_CHANNELS[:2]]

t_after = moving_time

save_image_panel("panel_a.png", image_panels[channel0]["reference"], f"ch{channel0} reference t={reference_time}")
save_image_panel("panel_b.png", image_panels[channel0]["moving"], f"ch{channel0} raw t={t_after}")
save_image_panel("panel_c.png", image_panels[channel0]["registered"], f"ch{channel0} registered t={t_after}")
save_image_panel("panel_d.png", image_panels[channel0]["raw_difference"], f"ch{channel0} ref - raw", cmap="bwr", diverging=True, vmax_abs=image_panels[channel0]["difference_vmax_abs"])
save_image_panel("panel_e.png", image_panels[channel0]["registered_difference"], f"ch{channel0} registered ref - t", cmap="bwr", diverging=True, vmax_abs=image_panels[channel0]["difference_vmax_abs"])
save_image_panel("panel_f.png", image_panels[channel1]["reference"], f"ch{channel1} reference t={reference_time}")
save_image_panel("panel_g.png", image_panels[channel1]["moving"], f"ch{channel1} raw t={t_after}")
save_image_panel("panel_h.png", image_panels[channel1]["registered"], f"ch{channel1} registered t={t_after}")
save_image_panel("panel_i.png", image_panels[channel1]["raw_difference"], f"ch{channel1} ref - raw", cmap="bwr", diverging=True, vmax_abs=image_panels[channel1]["difference_vmax_abs"])
save_image_panel("panel_j.png", image_panels[channel1]["registered_difference"], f"ch{channel1} registered ref - t", cmap="bwr", diverging=True, vmax_abs=image_panels[channel1]["difference_vmax_abs"])
save_line_panel("panel_k.pdf", "estimated 3D shifts", "time frame", "correction shift (px)", plot_shifts)
save_line_panel("panel_l.pdf", "reference correlation", "time frame", "Pearson r", plot_volume_correlation, ylim=(0, 1.05))
save_line_panel("panel_m.pdf", "mean correlation", "", "Pearson r", plot_summary, ylim=(0, 1.05))
copy_summary_plot(details_3dtch)
# %% PREPRINT DRAFT FIGURE
assemble_figure9()
print(f"Wrote Figure 9 panels to {FIGURE_DIR}")
print(f"Wrote Figure 9 draft to {FIGURE_DRAFT.with_suffix('.pdf')}")
# % %END
# %% END
