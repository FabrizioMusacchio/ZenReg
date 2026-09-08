"""Real 3P pseudo-time 3D registration example for the ZenReg preprint.

This script is intentionally written like an interactive user script. In VS Code
or Spyder, run the ``# %%`` cells one by one. From the terminal, run the whole
script:

    conda run -n zenreg2 python additional_scripts/run_preprint_real_3p_pseudot_example.py

The source 3P stack itself is not shipped with ZenReg. Download instructions are
provided in ``example_data/3P_paper_Fuhrmann_Nebeling_Musacchio/README.md``.
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
from scipy.spatial.transform import Rotation

from zenreg import create_stack_metadata, load_stack, register_stack, save_stack, update_stack_metadata
from zenreg.synthetic import _apply_zyx_rigid_transform, _rotation_matrix_zyx
# %% PATHS AND STYLE
REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = REPO_ROOT / "example_data" / "3P_paper_Fuhrmann_Nebeling_Musacchio" / "Supplementary_Video_4.tif"
PREPRINT_DIR = REPO_ROOT / "papers" / "preprint"
OUTPUT_DIR = PREPRINT_DIR / "benchmark_outputs" / "real_3p_pseudot"
FIGURE_DIR = PREPRINT_DIR / "figures" / "figure8"
FIGURE_DRAFT = PREPRINT_DIR / "figures" / "figure8"

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
# Defaults use a central subvolume. Set FULL_VOLUME=True only if you deliberately
# want to create and register ten pseudo-time copies of the full 1-GB stack.
USE_MEMMAP_FOR_SOURCE = True
MEMMAP_FOLDER = None
FULL_VOLUME = False
TIME_COUNT = 10
Z_COUNT = 96
YX_SIZE = 384
N_JOBS = 7

TRANSLATION_MAX_Z = 2.0
TRANSLATION_MAX_Y = 5.0
TRANSLATION_MAX_X = 5.5
RIGID_MAX_Z = 1.5
RIGID_MAX_Y = 4.0
RIGID_MAX_X = 4.5
RIGID_MAX_ROTATION_Z_DEG = 3.0

ROT_INIT_ITERATIONS = 1
ROT_ITERATIONS = 80

TRANSLATION_REGISTERED_OUTPUT_PATH  = OUTPUT_DIR / "real_3p_pseudot_translation_registered.ome.tif"
RIGID_REGISTERED_OUTPUT_PATH        = OUTPUT_DIR / "real_3p_pseudot_translation_rotation_registered.ome.tif"
METRICS_CSV_PATH  = OUTPUT_DIR / "real_3p_pseudot_metrics.csv"
SUMMARY_JSON_PATH = OUTPUT_DIR / "real_3p_pseudot_summary.json"
# %% HELPER FUNCTIONS
def normalize_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    lo, hi = np.nanpercentile(image, [1.0, 99.7])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image - lo) / (hi - lo), 0.0, 1.0)

def pearson_volume(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    a = a - float(np.mean(a))
    b = b - float(np.mean(b))
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return float("nan")
    return float(np.dot(a, b) / denom)

def centered_bounds(size: int, count: int) -> tuple[int, int]:
    count = min(int(count), int(size))
    start = max((int(size) - count) // 2, 0)
    return start, start + count

def pseudo_time_shifts(
    time_count: int,
    *,
    max_z: float,
    max_y: float,
    max_x: float
) -> np.ndarray:
    shifts = np.zeros((int(time_count), 3), dtype=np.float32)
    denominator = max(int(time_count) - 1, 1)
    for t in range(int(time_count)):
        phase = 2.0 * np.pi * t / denominator
        shifts[t] = (
            max_z * np.sin(phase),
            max_y * np.sin(phase + 0.45),
            max_x * (np.cos(phase + 0.2) - np.cos(0.2)))
    shifts[0] = 0.0
    return shifts

def pseudo_time_rotations_z(time_count: int, *, max_z_deg: float) -> np.ndarray:
    rotations = np.zeros((int(time_count), 3), dtype=np.float32)
    denominator = max(int(time_count) - 1, 1)
    for t in range(int(time_count)):
        rotations[t, 0] = max_z_deg * np.sin(2.0 * np.pi * t / denominator)
    rotations[0] = 0.0
    return rotations

def expected_rigid_corrections(
    applied_shifts_zyx: np.ndarray,
    applied_rotations_zyx_deg: np.ndarray,
    *,
    registration_stack: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    applied_shifts_zyx = np.asarray(applied_shifts_zyx, dtype=np.float64)
    applied_rotations_zyx_deg = np.asarray(applied_rotations_zyx_deg, dtype=np.float64)
    reference_rotation = _rotation_matrix_zyx(
        rotation_z_deg=float(applied_rotations_zyx_deg[int(registration_stack), 0]),
        rotation_y_deg=float(applied_rotations_zyx_deg[int(registration_stack), 1]),
        rotation_x_deg=float(applied_rotations_zyx_deg[int(registration_stack), 2])).astype(np.float64)
    reference_shift = np.asarray(applied_shifts_zyx[int(registration_stack)], dtype=np.float64)
    expected_shifts = np.zeros_like(applied_shifts_zyx, dtype=np.float32)
    expected_rotations = np.zeros_like(applied_rotations_zyx_deg, dtype=np.float32)
    for t in range(applied_shifts_zyx.shape[0]):
        moving_rotation = _rotation_matrix_zyx(
            rotation_z_deg=float(applied_rotations_zyx_deg[t, 0]),
            rotation_y_deg=float(applied_rotations_zyx_deg[t, 1]),
            rotation_x_deg=float(applied_rotations_zyx_deg[t, 2])).astype(np.float64)
        correction_rotation = reference_rotation @ moving_rotation.T
        expected_rotations[t] = Rotation.from_matrix(correction_rotation).as_euler("ZYX", degrees=True)
        expected_shifts[t] = reference_shift - correction_rotation @ np.asarray(applied_shifts_zyx[t], dtype=np.float64)
    return expected_shifts, expected_rotations

def build_pseudo_time_stack(
    base_stack: np.ndarray,
    *,
    shifts_zyx: np.ndarray,
    rotations_zyx_deg: np.ndarray,
    center_zyx: tuple[float, float, float]
) -> np.ndarray:
    time_count = shifts_zyx.shape[0]
    _, z_count, channel_count, y_count, x_count = base_stack.shape
    output = np.zeros((time_count, z_count, channel_count, y_count, x_count), dtype=np.float32)
    base = np.asarray(base_stack[0], dtype=np.float32)
    for t in range(time_count):
        rot_z, rot_y, rot_x = [float(v) for v in rotations_zyx_deg[t]]
        shift_zyx = tuple(float(v) for v in shifts_zyx[t])
        for c in range(channel_count):
            output[t, :, c] = _apply_zyx_rigid_transform(
                base[:, c],
                shift_zyx=shift_zyx,
                rotation_z_deg=rot_z,
                rotation_y_deg=rot_y,
                rotation_x_deg=rot_x,
                center_zyx=center_zyx)
    return output

def frame_volume_correlations(stack: np.ndarray) -> np.ndarray:
    reference = np.asarray(stack[0, :, 0], dtype=np.float32)
    return np.asarray([pearson_volume(reference, stack[t, :, 0]) for t in range(stack.shape[0])], dtype=np.float32)

def max_projection(stack: np.ndarray, *, t: int, channel: int = 0) -> np.ndarray:
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
    figsize_cm: tuple[float, float] = (4., 4.0),
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

def assemble_figure8() -> None:
    panel_names = [f"panel_{letter}.png" for letter in "abcdefghijklmn"]
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
    for ax in axes[len(panel_names) :]:
        ax.axis("off")
    fig.suptitle("Real 3P pseudo-time 3D registration", fontsize=FONT_SIZE)
    fig.tight_layout()
    fig.savefig(FIGURE_DRAFT.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(FIGURE_DRAFT.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)
# %% LOAD REAL 3P STACK
if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"Missing 3P example stack: {INPUT_PATH}. "
        "Download it using the instructions in example_data/3P_paper_Fuhrmann_Nebeling_Musacchio/README.md.")

stack_3p, metadata_3p = load_stack(
    INPUT_PATH,
    return_metadata=True,
    use_memmap=USE_MEMMAP_FOR_SOURCE,
    memmap_folder=MEMMAP_FOLDER,
    verbose=False)
print(f"Loaded 3P stack as {stack_3p.shape} (TZCYX), dtype={stack_3p.dtype}")

if FULL_VOLUME:
    z_start, z_stop = 0, stack_3p.shape[1]
    y_start, y_stop = 0, stack_3p.shape[3]
    x_start, x_stop = 0, stack_3p.shape[4]
else:
    z_start, z_stop = centered_bounds(stack_3p.shape[1], Z_COUNT)
    y_start, y_stop = centered_bounds(stack_3p.shape[3], YX_SIZE)
    x_start, x_stop = centered_bounds(stack_3p.shape[4], YX_SIZE)

base_stack_3p = np.asarray(stack_3p[0:1, z_start:z_stop, :, y_start:y_stop, x_start:x_stop], dtype=np.float32)
metadata_base_3p = update_stack_metadata(
    create_stack_metadata(base_stack_3p, input_metadata=metadata_3p, verbose=False),
    base_stack_3p,
    verbose=False)
print(f"Using subvolume {base_stack_3p.shape} (TZCYX), z={z_start}:{z_stop}, y={y_start}:{y_stop}, x={x_start}:{x_stop}")
# %% CREATE PSEUDO-TIME DATASETS WITH KNOWN GROUND TRUTH
rotation_center_zyx = (
    (base_stack_3p.shape[1] - 1) / 2.0,
    (base_stack_3p.shape[3] - 1) / 2.0,
    (base_stack_3p.shape[4] - 1) / 2.0)

translation_applied_shifts_zyx = pseudo_time_shifts(
    TIME_COUNT,
    max_z=TRANSLATION_MAX_Z,
    max_y=TRANSLATION_MAX_Y,
    max_x=TRANSLATION_MAX_X)
translation_applied_rotations_zyx = np.zeros((TIME_COUNT, 3), dtype=np.float32)

rigid_applied_shifts_zyx = pseudo_time_shifts(
    TIME_COUNT,
    max_z=RIGID_MAX_Z,
    max_y=RIGID_MAX_Y,
    max_x=RIGID_MAX_X)
rigid_applied_rotations_zyx = pseudo_time_rotations_z(
    TIME_COUNT,
    max_z_deg=RIGID_MAX_ROTATION_Z_DEG)

stack_3p_translation = build_pseudo_time_stack(
    base_stack_3p,
    shifts_zyx=translation_applied_shifts_zyx,
    rotations_zyx_deg=translation_applied_rotations_zyx,
    center_zyx=rotation_center_zyx)
stack_3p_rigid = build_pseudo_time_stack(
    base_stack_3p,
    shifts_zyx=rigid_applied_shifts_zyx,
    rotations_zyx_deg=rigid_applied_rotations_zyx,
    center_zyx=rotation_center_zyx)

metadata_3p_translation = update_stack_metadata(
    create_stack_metadata(stack_3p_translation, input_metadata=metadata_base_3p, verbose=False),
    stack_3p_translation,
    verbose=False)
metadata_3p_rigid = update_stack_metadata(
    create_stack_metadata(stack_3p_rigid, input_metadata=metadata_base_3p, verbose=False),
    stack_3p_rigid,
    verbose=False)

translation_expected_shifts_zyx = -translation_applied_shifts_zyx
rigid_expected_shifts_zyx, rigid_expected_rotations_zyx = expected_rigid_corrections(
    rigid_applied_shifts_zyx,
    rigid_applied_rotations_zyx)

print(f"Translation pseudo-time stack: {stack_3p_translation.shape} (TZCYX)")
print(f"Translation + rotation pseudo-time stack: {stack_3p_rigid.shape} (TZCYX)")

# %% REGISTRATION 1: FULL-3D TRANSLATION ONLY
translation_registration_start = time.perf_counter()
registered_3p_translation, details_3p_translation = register_stack(
    stack_3p_translation,
    metadata=metadata_3p_translation,
    registration_channel=0,
    registration_stack=0,
    method="phase_cross_correlation",
    time_registration_mode="full_3d",
    time_reference_mode="template",
    zreg=True,
    rotreg=False,
    max_xy_shifts=(15, 15),
    max_z_shifts=4,
    zero_clip=False,
    transform_backend="skimage",
    transform_order=1,
    n_jobs=N_JOBS,
    verbose=True,
    return_shifts=True,
    return_details=True)
translation_runtime_s = time.perf_counter() - translation_registration_start
print(f"Translation registration finished in {translation_runtime_s:.2f} s")

# save registration 1:
translation_registered_output_path = save_stack(
    TRANSLATION_REGISTERED_OUTPUT_PATH,
    registered_3p_translation,
    metadata=metadata_3p_translation,
    registration_details=details_3p_translation,
    overwrite=True,
    verbose=False)
print(f"Wrote translation-only registered stack to {translation_registered_output_path}")
# %% REGISTRATION 2: FULL-3D TRANSLATION PLUS SIMPLEITK RIGID ROTATION
rigid_registration_start = time.perf_counter()
registered_3p_rigid, details_3p_rigid = register_stack(
    stack_3p_rigid,
    metadata=metadata_3p_rigid,
    registration_channel=0,
    registration_stack=0,
    method="phase_cross_correlation",
    time_registration_mode="full_3d",
    time_reference_mode="template",
    zreg=True,
    rotreg=True,
    rigid_3d_backend="simpleitk",
    rot_init_iterations=ROT_INIT_ITERATIONS,
    rot_metric="correlation",
    rot_shrink_factors=(4, 2, 1),
    rot_smoothing_sigmas=(2.0, 1.0, 0.0),
    rot_iterations=ROT_ITERATIONS,
    rot_n_jobs=N_JOBS,
    zero_clip=False,
    transform_order=1,
    verbose=True,
    return_shifts=True,
    return_details=True)
rigid_runtime_s = time.perf_counter() - rigid_registration_start
print(f"Translation + rotation registration finished in {rigid_runtime_s:.2f} s")

#  save registration 2:
rigid_registered_output_path = save_stack(
    RIGID_REGISTERED_OUTPUT_PATH,
    registered_3p_rigid,
    metadata=metadata_3p_rigid,
    registration_details=details_3p_rigid,
    overwrite=True,
    verbose=False)
print(f"Wrote translation + rotation registered stack to {rigid_registered_output_path}")
# %% PREPRINT ANALYSIS
moving_time = 4# min(1, TIME_COUNT - 1)

translation_before_correlation = frame_volume_correlations(stack_3p_translation)
translation_after_correlation = frame_volume_correlations(registered_3p_translation)
rigid_before_correlation = frame_volume_correlations(stack_3p_rigid)
rigid_after_correlation = frame_volume_correlations(registered_3p_rigid)

translation_estimated_shifts_zyx = np.asarray(details_3p_translation["time_shifts_zyx"], dtype=np.float32)
rigid_estimated_shifts_zyx = np.asarray(details_3p_rigid["time_shifts_zyx"], dtype=np.float32)
rigid_estimated_rotations_zyx = np.asarray(details_3p_rigid["rotation_shifts_zyx_deg"], dtype=np.float32)

translation_shift_error_zyx = np.abs(translation_estimated_shifts_zyx - translation_expected_shifts_zyx)
rigid_shift_error_zyx = np.abs(rigid_estimated_shifts_zyx - rigid_expected_shifts_zyx)
rigid_rotation_error_zyx = np.abs(rigid_estimated_rotations_zyx - rigid_expected_rotations_zyx)
frames = np.arange(TIME_COUNT)

translation_reference_projection = max_projection(stack_3p_translation, t=0)
translation_moving_projection = max_projection(stack_3p_translation, t=moving_time)
translation_registered_projection = max_projection(registered_3p_translation, t=moving_time)
rigid_reference_projection = max_projection(stack_3p_rigid, t=0)
rigid_moving_projection = max_projection(stack_3p_rigid, t=moving_time)
rigid_registered_projection = max_projection(registered_3p_rigid, t=moving_time)

translation_raw_difference = translation_reference_projection - translation_moving_projection
translation_registered_difference = translation_reference_projection - translation_registered_projection
rigid_raw_difference = rigid_reference_projection - rigid_moving_projection
rigid_registered_difference = rigid_reference_projection - rigid_registered_projection

translation_difference_vmax_abs = max(
    float(np.nanmax(np.abs(translation_raw_difference))),
    float(np.nanmax(np.abs(translation_registered_difference))),
    1e-6)
rigid_difference_vmax_abs = max(
    float(np.nanmax(np.abs(rigid_raw_difference))),
    float(np.nanmax(np.abs(rigid_registered_difference))),
    1e-6)

metric_rows = []
for frame in frames:
    row = {
        "frame": int(frame),
        "translation_runtime_s": float(translation_runtime_s),
        "rigid_runtime_s": float(rigid_runtime_s),
        "translation_pearson_before": float(translation_before_correlation[frame]),
        "translation_pearson_after": float(translation_after_correlation[frame]),
        "rigid_pearson_before": float(rigid_before_correlation[frame]),
        "rigid_pearson_after": float(rigid_after_correlation[frame]),
    }
    for axis_index, axis_name in enumerate(("z", "y", "x")):
        row[f"translation_expected_shift_{axis_name}"] = float(translation_expected_shifts_zyx[frame, axis_index])
        row[f"translation_estimated_shift_{axis_name}"] = float(translation_estimated_shifts_zyx[frame, axis_index])
        row[f"translation_abs_shift_error_{axis_name}"] = float(translation_shift_error_zyx[frame, axis_index])
        row[f"rigid_expected_shift_{axis_name}"] = float(rigid_expected_shifts_zyx[frame, axis_index])
        row[f"rigid_estimated_shift_{axis_name}"] = float(rigid_estimated_shifts_zyx[frame, axis_index])
        row[f"rigid_abs_shift_error_{axis_name}"] = float(rigid_shift_error_zyx[frame, axis_index])
        row[f"rigid_expected_rotation_{axis_name}"] = float(rigid_expected_rotations_zyx[frame, axis_index])
        row[f"rigid_estimated_rotation_{axis_name}"] = float(rigid_estimated_rotations_zyx[frame, axis_index])
        row[f"rigid_abs_rotation_error_{axis_name}"] = float(rigid_rotation_error_zyx[frame, axis_index])
    metric_rows.append(row)
write_csv(METRICS_CSV_PATH, metric_rows)

summary = {
    "input": str(INPUT_PATH),
    "base_shape_tzcyx": list(base_stack_3p.shape),
    "translation_stack_shape_tzcyx": list(stack_3p_translation.shape),
    "rigid_stack_shape_tzcyx": list(stack_3p_rigid.shape),
    "registered_translation_shape_tzcyx": list(registered_3p_translation.shape),
    "registered_rigid_shape_tzcyx": list(registered_3p_rigid.shape),
    "runtime_translation_s": float(translation_runtime_s),
    "runtime_rigid_s": float(rigid_runtime_s),
    "mean_translation_shift_error_px": float(np.nanmean(translation_shift_error_zyx)),
    "mean_rigid_shift_error_px": float(np.nanmean(rigid_shift_error_zyx)),
    "mean_rigid_rotation_error_deg": float(np.nanmean(rigid_rotation_error_zyx)),
    "mean_translation_correlation_before": float(np.nanmean(translation_before_correlation)),
    "mean_translation_correlation_after": float(np.nanmean(translation_after_correlation)),
    "mean_rigid_correlation_before": float(np.nanmean(rigid_before_correlation)),
    "mean_rigid_correlation_after": float(np.nanmean(rigid_after_correlation)),
}
SUMMARY_JSON_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(f"Wrote analysis metrics to {METRICS_CSV_PATH}")
# %% PREPRINT PLOT SETTINGS
def plot_shift_error(ax):
    ax.plot(frames, np.mean(translation_shift_error_zyx, axis=1), marker="o", color="#3b7cec", linewidth=1.3, label="translation only")
    ax.plot(frames, np.mean(rigid_shift_error_zyx, axis=1), marker="o", color="#7030a0", linewidth=1.3, label="translation + rotation")

def plot_rotation_error(ax):
    ax.plot(frames, rigid_rotation_error_zyx[:, 0], marker="o", color="#7030a0", linewidth=1.3, label="rot z")
    ax.plot(frames, rigid_rotation_error_zyx[:, 1], marker="o", color="#8f6bb5", linewidth=1.0, label="rot y")
    ax.plot(frames, rigid_rotation_error_zyx[:, 2], marker="o", color="#b095d6", linewidth=1.0, label="rot x")

def plot_volume_correlation(ax):
    ax.plot(frames, translation_before_correlation, color="#bfbfbf", linestyle=":", linewidth=1.0, label="translation before")
    ax.plot(frames, translation_after_correlation, color="#3b7cec", linewidth=1.3, label="translation after")
    ax.plot(frames, rigid_before_correlation, color="#8c8c8c", linestyle=":", linewidth=1.0, label="rigid before")
    ax.plot(frames, rigid_after_correlation, color="#7030a0", linewidth=1.3, label="rigid after")

def plot_summary(ax):
    labels = ["translation", "translation + rotation"]
    shift_values = [
        float(np.nanmean(translation_shift_error_zyx)),
        float(np.nanmean(rigid_shift_error_zyx))]
    rotation_values = [
        0.0,
        float(np.nanmean(rigid_rotation_error_zyx))]
    x = np.arange(len(labels))
    width = 0.32
    ax.bar(x - 0.18, shift_values, width=width, color=["#3b7cec", "#7030a0"], label="translation error (px)")
    ax.bar(x + 0.18, rotation_values, width=width, color=["#3b7cec", "#7030a0"], alpha=0.35, label="rotation error (deg)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
# %% PREPRINT PANEL PLOTS
save_image_panel("panel_a.png", translation_reference_projection, "translation reference")
save_image_panel("panel_b.png", translation_moving_projection, f"translation moving t={moving_time}")
save_image_panel("panel_c.png", translation_registered_projection, "translation registered")
save_image_panel("panel_d.png", translation_raw_difference, "translation ref - moving", cmap="bwr", diverging=True, vmax_abs=translation_difference_vmax_abs)
save_image_panel("panel_e.png", translation_registered_difference, "translation ref - registered", cmap="bwr", diverging=True, vmax_abs=translation_difference_vmax_abs)
save_image_panel("panel_f.png", rigid_reference_projection, "rigid reference")
save_image_panel("panel_g.png", rigid_moving_projection, f"rigid moving t={moving_time}")
save_image_panel("panel_h.png", rigid_registered_projection, "rigid registered")
save_image_panel("panel_i.png", rigid_raw_difference, "rigid ref - moving", cmap="bwr", diverging=True, vmax_abs=rigid_difference_vmax_abs)
save_image_panel("panel_j.png", rigid_registered_difference, "rigid ref - registered", cmap="bwr", diverging=True, vmax_abs=rigid_difference_vmax_abs)
save_line_panel("panel_k.pdf", "translation error", "pseudo-time frame", "mean abs. error (px)", plot_shift_error, ylim=(0, None))
save_line_panel("panel_l.pdf", "rotation error", "pseudo-time frame", "abs. error (deg)", plot_rotation_error, ylim=(0, None))
save_line_panel("panel_m.pdf", "volume correlation", "pseudo-time frame", "Pearson r", plot_volume_correlation, ylim=(0, 1.05))
save_line_panel("panel_n.pdf", "registration summary", "", "mean error", plot_summary, ylim=(0, None))
# %% PREPRINT DRAFT FIGURE
assemble_figure8()
print(f"Wrote Figure 8 panels to {FIGURE_DIR}")
# % %END
# %%
