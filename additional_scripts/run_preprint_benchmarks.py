"""Run synthetic benchmarks for the ZenReg preprint.

The script generates synthetic microscopy stacks with known ground truth,
registers them with selected ZenReg backends, and writes compact CSV/NPZ files
for the plotting script.

Examples
--------
Run a quick local benchmark:

    conda run -n zenreg2 python additional_scripts/run_preprint_benchmarks.py --quick

For a full benchmark, run:
    conda run -n zenreg2 python additional_scripts/run_preprint_benchmarks.py

Run only a specific benchmark:

    conda run -n zenreg2 python additional_scripts/run_preprint_benchmarks.py --case representative

    conda run -n zenreg2 python additional_scripts/run_preprint_benchmarks.py --case representative_random_walk
    
    conda run -n zenreg2 python additional_scripts/run_preprint_benchmarks.py --case snr
    
    conda run -n zenreg2 python additional_scripts/run_preprint_benchmarks.py --case drift
    
    conda run -n zenreg2 python additional_scripts/run_preprint_benchmarks.py --case piecewise
    
    conda run -n zenreg2 python additional_scripts/run_preprint_benchmarks.py --case rigid3d
    
    conda run -n zenreg2 python additional_scripts/run_preprint_benchmarks.py --case projection_parallel

    conda run -n zenreg2 python additional_scripts/run_preprint_benchmarks.py --case memory
"""
# %% IMPORTS
from __future__ import annotations

import argparse
import csv
import gc
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.ndimage import shift as ndi_shift
from scipy.spatial.transform import Rotation

from zenreg import (
    cleanup_omio_cache,
    create_empty_stack,
    create_stack_metadata,
    load_stack,
    load_expected_rigid_corrections,
    profile_memory,
    register_stack,
    save_stack,
    z_project,
)
from zenreg.synthetic import (
    _rotation_matrix_zyx,
    create_2d_motion_distorted_stack,
    create_3d_time_rigid_motion_distorted_stack,
    create_3d_time_sparse_puncta_rigid_motion_distorted_stack,
    create_3d_time_xy_motion_distorted_stack,
    create_3d_time_zyx_motion_distorted_stack,
)
# %% PATHS
REPO_ROOT               = Path(__file__).resolve().parents[1]
PREPRINT_DIR            = REPO_ROOT / "papers" / "preprint"
OUTPUT_DIR              = PREPRINT_DIR / "benchmark_outputs"
SYNTHETIC_DIR           = OUTPUT_DIR / "synthetic_inputs"
MEMMAP_DIR              = OUTPUT_DIR / "omio_memmap_cache"
EXAMPLE_SYNTHETIC_DIR   = REPO_ROOT / "example_data" / "synthetic_data"
# %% HELPERS
def _write_rows(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))

def _as_float(value) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")

def _project_registration_image(stack, *, t: int, channel: int = 0, projection_method: str = "max") -> np.ndarray:
    projected = z_project(
        stack[int(t) : int(t) + 1, :, int(channel) : int(channel) + 1, :, :],
        projection_method=projection_method,
    )
    return np.asarray(projected[0, 0, 0], dtype=np.float32)

def _normalize_template_time_range(registration_template_time_range, time_count: int):
    """Normalize benchmark template time-range shortcuts."""

    if registration_template_time_range is None:
        return None
    if isinstance(registration_template_time_range, str):
        if registration_template_time_range.strip().lower() == "all":
            return (0, int(time_count))
        raise ValueError(f"Unsupported registration_template_time_range={registration_template_time_range!r}")
    start, stop = registration_template_time_range
    return (int(start), int(stop))

def _aggregate_projection_images(projections: list[np.ndarray], *, projection_method: str) -> np.ndarray:
    """Aggregate 2D projections with the same projection keywords used by ZenReg."""

    stacked = np.stack([np.asarray(projection, dtype=np.float32) for projection in projections], axis=0)
    method = str(projection_method).lower()
    if method == "max":
        return np.max(stacked, axis=0).astype(np.float32, copy=False)
    if method == "mean":
        return np.mean(stacked, axis=0).astype(np.float32, copy=False)
    if method == "median":
        return np.median(stacked, axis=0).astype(np.float32, copy=False)
    if method == "var":
        return np.var(stacked, axis=0).astype(np.float32, copy=False)
    if method == "std":
        return np.std(stacked, axis=0).astype(np.float32, copy=False)
    raise ValueError(f"Unsupported projection_method={projection_method!r}")

def _registration_template_image(
    stack,
    *,
    registration_stack: int = 0,
    channel: int = 0,
    projection_method: str = "max",
    registration_template_time_range=None,
) -> np.ndarray:
    """Build the projected template used for benchmark correlations."""

    time_range = _normalize_template_time_range(registration_template_time_range, stack.shape[0])
    if time_range is None:
        return _project_registration_image(
            stack,
            t=int(registration_stack),
            channel=int(channel),
            projection_method=projection_method,
        )
    start, stop = time_range
    projections = [
        _project_registration_image(
            stack,
            t=t,
            channel=int(channel),
            projection_method=projection_method,
        )
        for t in range(start, stop)
    ]
    return _aggregate_projection_images(projections, projection_method=projection_method)

def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    a = a - float(np.mean(a))
    b = b - float(np.mean(b))
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return float("nan")
    return float(np.dot(a, b) / denom)

def _frame_correlations(
    stack,
    *,
    registration_stack: int = 0,
    channel: int = 0,
    projection_method: str = "max",
    registration_template_time_range=None,
) -> np.ndarray:
    reference = _registration_template_image(
        stack,
        registration_stack=int(registration_stack),
        channel=int(channel),
        projection_method=projection_method,
        registration_template_time_range=registration_template_time_range,
    )
    return np.asarray(
        [
            _pearson(
                reference,
                _project_registration_image(stack, t=t, channel=channel, projection_method=projection_method),
            )
            for t in range(stack.shape[0])
        ],
        dtype=np.float32,
    )

def _summary_from_errors(errors: np.ndarray) -> dict[str, float]:
    errors = np.asarray(errors, dtype=np.float32)
    return {
        "mae": float(np.mean(np.abs(errors))),
        "median_abs_error": float(np.median(np.abs(errors))),
        "max_abs_error": float(np.max(np.abs(errors))),
    }

def _raw_size_mb(stack) -> float:
    return float(np.prod(stack.shape) * np.dtype(stack.dtype).itemsize) / 1024.0 / 1024.0

def _expected_correction(applied_shift: np.ndarray) -> np.ndarray:
    return -np.asarray(applied_shift, dtype=np.float32)

def _expected_rigid_corrections_from_applied(
    applied_shifts_zyx: np.ndarray,
    applied_rotations_zyx_deg: np.ndarray,
    *,
    registration_stack: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute expected 3D rigid correction parameters from applied transforms."""

    applied_shifts_zyx = np.asarray(applied_shifts_zyx, dtype=np.float64)
    applied_rotations_zyx_deg = np.asarray(applied_rotations_zyx_deg, dtype=np.float64)
    reference_rotation = _rotation_matrix_zyx(
        rotation_z_deg=float(applied_rotations_zyx_deg[int(registration_stack), 0]),
        rotation_y_deg=float(applied_rotations_zyx_deg[int(registration_stack), 1]),
        rotation_x_deg=float(applied_rotations_zyx_deg[int(registration_stack), 2]),
    ).astype(np.float64)
    reference_shift = np.asarray(applied_shifts_zyx[int(registration_stack), :], dtype=np.float64)
    expected_shifts = np.zeros_like(applied_shifts_zyx, dtype=np.float32)
    expected_rotations = np.zeros_like(applied_rotations_zyx_deg, dtype=np.float32)
    for t in range(applied_shifts_zyx.shape[0]):
        moving_rotation = _rotation_matrix_zyx(
            rotation_z_deg=float(applied_rotations_zyx_deg[t, 0]),
            rotation_y_deg=float(applied_rotations_zyx_deg[t, 1]),
            rotation_x_deg=float(applied_rotations_zyx_deg[t, 2]),
        ).astype(np.float64)
        correction_rotation = reference_rotation @ moving_rotation.T
        expected_rotations[t, :] = Rotation.from_matrix(correction_rotation).as_euler("ZYX", degrees=True)
        expected_shifts[t, :] = reference_shift - correction_rotation @ np.asarray(applied_shifts_zyx[t, :], dtype=np.float64)
    return expected_shifts, expected_rotations

def _parallel_scaling_shifts_yx(time_count: int) -> np.ndarray:
    """Small deterministic shifts for large-T parallel scaling benchmarks."""

    shifts = np.zeros((int(time_count), 2), dtype=np.float32)
    denominator = max(int(time_count) - 1, 1)
    for t in range(int(time_count)):
        shifts[t] = (
            4.0 * np.sin(2 * np.pi * t / denominator),
            3.0 * (np.cos(2 * np.pi * t / denominator) - 1.0),
        )
    return shifts

def _random_walk_shifts_yx(
    time_count: int,
    *,
    step_sigma_y: float,
    step_sigma_x: float,
    max_abs_y: float,
    max_abs_x: float,
    random_state: int,
) -> np.ndarray:
    """Generate bounded Brownian-like YX shifts anchored at frame 0."""

    rng = np.random.default_rng(int(random_state))
    steps = rng.normal(
        loc=0.0,
        scale=(float(step_sigma_y), float(step_sigma_x)),
        size=(int(time_count), 2),
    ).astype(np.float32)
    steps[0] = 0.0
    shifts = np.cumsum(steps, axis=0).astype(np.float32)
    shifts -= shifts[0:1]
    for axis, max_abs in enumerate((float(max_abs_y), float(max_abs_x))):
        current = float(np.max(np.abs(shifts[:, axis])))
        if current > 1e-6:
            shifts[:, axis] *= max_abs / current
    shifts[0] = 0.0
    return shifts.astype(np.float32, copy=False)

def _parallel_scaling_base_channels(
    shape_yx: tuple[int, int],
    *,
    channel_count: int,
    random_state: int,
) -> list[np.ndarray]:
    """Generate reusable textured 2D channel templates for scaling benchmarks."""

    rng = np.random.default_rng(int(random_state))
    yy, xx = np.indices(shape_yx, dtype=np.float32)
    channels = []
    for channel in range(int(channel_count)):
        image = np.zeros(shape_yx, dtype=np.float32)
        spot_count = max(32, int(np.prod(shape_yx) // 6000))
        margin_y = max(8.0, shape_yx[0] * 0.05)
        margin_x = max(8.0, shape_yx[1] * 0.05)
        centers_y = rng.uniform(margin_y, shape_yx[0] - margin_y, spot_count)
        centers_x = rng.uniform(margin_x, shape_yx[1] - margin_x, spot_count)
        sigmas_y = rng.uniform(3.0, 9.0, spot_count)
        sigmas_x = rng.uniform(3.0, 9.0, spot_count)
        amplitudes = rng.uniform(0.35, 1.0, spot_count)
        for cy, cx, sy, sx, amp in zip(centers_y, centers_x, sigmas_y, sigmas_x, amplitudes, strict=True):
            image += float(amp) * np.exp(-(((yy - cy) ** 2) / (2 * sy**2) + ((xx - cx) ** 2) / (2 * sx**2)))
        image -= float(image.min())
        image /= max(float(image.max()), 1e-6)
        channels.append(image.astype(np.float32, copy=False))
    return channels

def _create_parallel_scaling_stack(
    *,
    time_count: int,
    channel_count: int,
    shape_yx: tuple[int, int],
    noise_sigma: float,
    random_state: int,
    use_memmap: bool,
    memmap_folder: Path,
    memmap_name: str,
):
    """Create a 2D+t scaling benchmark stack in RAM or as a disk-backed Zarr."""

    shifts = _parallel_scaling_shifts_yx(int(time_count))
    base_channels = _parallel_scaling_base_channels(
        shape_yx,
        channel_count=int(channel_count),
        random_state=int(random_state),
    )
    if use_memmap:
        stack = create_empty_stack(
            shape=(int(time_count), 1, int(channel_count), *shape_yx),
            dtype=np.float32,
            fill_value=0,
            use_memmap=True,
            memmap_folder=memmap_folder,
            memmap_name=memmap_name,
            return_metadata=False,
            verbose=False,
        )
    else:
        stack = np.empty((int(time_count), 1, int(channel_count), *shape_yx), dtype=np.float32)

    rng = np.random.default_rng(int(random_state) + 100_000)
    for t in range(int(time_count)):
        shift_yx = tuple(float(v) for v in shifts[t])
        for c, base_image in enumerate(base_channels):
            moved = ndi_shift(
                base_image,
                shift=shift_yx,
                order=1,
                mode="constant",
                cval=0.0,
                prefilter=True,
            ).astype(np.float32, copy=False)
            if noise_sigma > 0:
                moved = moved + rng.normal(0, float(noise_sigma), size=shape_yx).astype(np.float32)
            stack[t, 0, c, :, :] = np.clip(moved, 0, None).astype(np.float32, copy=False)
    return stack, shifts

def _create_2d_random_walk_motion_stack(
    *,
    time_count: int,
    channel_count: int,
    shape_yx: tuple[int, int],
    noise_sigma: float,
    random_state: int,
    step_sigma_y: float,
    step_sigma_x: float,
    max_abs_y: float,
    max_abs_x: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a 2D+t stack with cumulative random-walk translation."""

    shifts = _random_walk_shifts_yx(
        int(time_count),
        step_sigma_y=float(step_sigma_y),
        step_sigma_x=float(step_sigma_x),
        max_abs_y=float(max_abs_y),
        max_abs_x=float(max_abs_x),
        random_state=int(random_state),
    )
    base_channels = _parallel_scaling_base_channels(
        shape_yx,
        channel_count=int(channel_count),
        random_state=int(random_state) + 17,
    )
    stack = np.empty((int(time_count), 1, int(channel_count), *shape_yx), dtype=np.float32)
    rng = np.random.default_rng(int(random_state) + 100_000)
    for t in range(int(time_count)):
        shift_yx = tuple(float(v) for v in shifts[t])
        for c, base_image in enumerate(base_channels):
            moved = ndi_shift(
                base_image,
                shift=shift_yx,
                order=1,
                mode="constant",
                cval=0.0,
                prefilter=True,
            ).astype(np.float32, copy=False)
            if noise_sigma > 0:
                moved = moved + rng.normal(0, float(noise_sigma), size=shape_yx).astype(np.float32)
            stack[t, 0, c, :, :] = np.clip(moved, 0, None).astype(np.float32, copy=False)
    return stack, shifts

def _create_piecewise_wave_motion_stack(
    *,
    time_count: int,
    channel_count: int,
    shape_yx: tuple[int, int],
    amplitude_px: float,
    noise_sigma: float,
    random_state: int,
) -> np.ndarray:
    """Create a local-motion benchmark with no dominant global translation."""

    base_channels = _parallel_scaling_base_channels(
        shape_yx,
        channel_count=int(channel_count),
        random_state=int(random_state),
    )
    stack = np.empty((int(time_count), 1, int(channel_count), *shape_yx), dtype=np.float32)
    yy, xx = np.indices(shape_yx, dtype=np.float32)
    y_scale = max(float(shape_yx[0] - 1), 1.0)
    x_scale = max(float(shape_yx[1] - 1), 1.0)
    rng = np.random.default_rng(int(random_state) + 100_000)
    denominator = max(int(time_count) - 1, 1)
    for t in range(int(time_count)):
        temporal_envelope = np.sin(np.pi * t / denominator)
        phase = 2.0 * np.pi * t / denominator
        dy = (
            float(amplitude_px)
            * temporal_envelope
            * np.sin(2.0 * np.pi * xx / x_scale + phase)
            * np.sin(np.pi * yy / y_scale)
        )
        dx = (
            float(amplitude_px)
            * temporal_envelope
            * np.cos(2.0 * np.pi * yy / y_scale - 0.5 * phase)
            * np.sin(np.pi * xx / x_scale)
        )
        coordinates = np.asarray([yy - dy, xx - dx], dtype=np.float32)
        for c, base_image in enumerate(base_channels):
            moved = map_coordinates(
                base_image,
                coordinates,
                order=1,
                mode="constant",
                cval=0.0,
            ).astype(np.float32, copy=False)
            if noise_sigma > 0:
                moved = moved + rng.normal(0, float(noise_sigma), size=shape_yx).astype(np.float32)
            stack[t, 0, c, :, :] = np.clip(moved, 0, None).astype(np.float32, copy=False)
    return stack

def _run_parallel_scaling_registration(
    stack,
    *,
    applied_shift: np.ndarray,
    n_jobs: int,
    output_memmap_folder: Path | None,
    output_memmap_name: str,
) -> dict[str, float | int | str]:
    """Run only the registration path needed for large-T scaling benchmarks."""

    output_use_memmap = output_memmap_folder is not None
    start = time.perf_counter()
    registered, details = register_stack(
        stack,
        registration_channel=0,
        registration_stack=0,
        method="phase_cross_correlation",
        time_registration_mode="projection",
        projection_method="max",
        zreg=False,
        max_xy_shifts=(64, 64),
        n_jobs=int(n_jobs),
        output_use_memmap=output_use_memmap,
        output_memmap_folder=output_memmap_folder,
        output_memmap_name=output_memmap_name if output_use_memmap else "zenreg_registered",
        transform_backend="skimage",
        transform_order=1,
        verbose=False,
        return_shifts=True,
        return_details=True,
    )
    runtime_s = time.perf_counter() - start
    estimated = np.asarray(details["time_shifts_yx"], dtype=np.float32)
    expected = _expected_correction(applied_shift)
    errors = estimated - expected
    del registered, details
    gc.collect()
    return {
        "runtime_s": float(runtime_s),
        "mean_abs_error": float(np.nanmean(np.abs(errors))),
        "mean_corr_before": float("nan"),
        "mean_corr_after": float("nan"),
    }

def _normalize_method_name(method: str) -> tuple[str, dict]:
    method = str(method)
    if method == "phase":
        return "phase_cross_correlation", {}
    if method == "pystackreg":
        return "pystackreg", {}
    if method == "normcorre":
        return "normcorre", {
            "nc_pw_rigid": False,
            "nc_template_init_mode": "registration_stack",
            "nc_template_update_method": "caiman",
            "nc_n_iterations": 1,
        }
    if method == "normcorre_gsig":
        return "normcorre", {
            "nc_pw_rigid": False,
            "nc_template_init_mode": "registration_stack",
            "nc_template_update_method": "caiman",
            "nc_n_iterations": 1,
            "nc_gSig_filt": (3, 3),
        }
    if method == "normcorre_pw":
        return "normcorre", {
            "nc_pw_rigid": True,
            "nc_strides": (32, 32),
            "nc_overlaps": (16, 16),
            "nc_max_deviation_rigid": (8, 8),
            "nc_template_init_mode": "registration_stack",
            "nc_template_update_method": "caiman",
            "nc_n_iterations": 1,
        }
    raise ValueError(f"Unknown benchmark method: {method!r}")

def _run_translation_registration(
    stack,
    *,
    applied_shift: np.ndarray,
    method_label: str,
    dimension: str,
    time_registration_mode: str,
    zreg: bool,
    n_jobs: int,
    filter_projections: bool = False,
    registration_template_time_range=None,
):
    method, extra = _normalize_method_name(method_label)
    before = _frame_correlations(
        stack,
        registration_template_time_range=registration_template_time_range,
    )
    start = time.perf_counter()
    registered, details = register_stack(
        stack,
        registration_channel=0,
        registration_stack=0,
        registration_template_time_range=registration_template_time_range,
        method=method,
        time_registration_mode=time_registration_mode,
        projection_method="max",
        zreg=bool(zreg),
        filter_projections=bool(filter_projections),
        max_xy_shifts=(64, 64),
        max_z_shifts=8 if zreg else None,
        n_jobs=int(n_jobs),
        nc_n_jobs=int(n_jobs),
        transform_backend="skimage",
        transform_order=1,
        verbose=False,
        return_shifts=True,
        return_details=True,
        **extra,
    )
    runtime_s = time.perf_counter() - start
    after = _frame_correlations(
        registered,
        registration_template_time_range=registration_template_time_range,
    )

    if zreg or time_registration_mode == "full_3d":
        estimated = np.asarray(details["time_shifts_zyx"], dtype=np.float32)
    else:
        estimated = np.asarray(details["time_shifts_yx"], dtype=np.float32)
    expected = _expected_correction(applied_shift)
    errors = estimated - expected
    summary = _summary_from_errors(errors)
    summary.update(
        {
            "method": method_label,
            "dimension": dimension,
            "time_registration_mode": time_registration_mode,
            "zreg": bool(zreg),
            "filter_projections": bool(filter_projections),
            "runtime_s": float(runtime_s),
            "mean_corr_before": float(np.nanmean(before)),
            "mean_corr_after": float(np.nanmean(after)),
            "raw_size_mb": _raw_size_mb(stack),
        }
    )
    return registered, details, estimated, expected, errors, before, after, summary

def _per_frame_rows(
    *,
    scenario: str,
    variable: str,
    level: float,
    method_label: str,
    estimated: np.ndarray,
    expected: np.ndarray,
    errors: np.ndarray,
    before: np.ndarray,
    after: np.ndarray,
    runtime_s: float,
):
    estimated = np.asarray(estimated, dtype=np.float32)
    expected = np.asarray(expected, dtype=np.float32)
    errors = np.asarray(errors, dtype=np.float32)
    rows = []
    axis_names = ["z", "y", "x"] if estimated.shape[1] == 3 else ["y", "x"]
    for t in range(estimated.shape[0]):
        row = {
            "scenario": scenario,
            "variable": variable,
            "level": float(level),
            "method": method_label,
            "frame": int(t),
            "pearson_before": _as_float(before[t]),
            "pearson_after": _as_float(after[t]),
            "runtime_s": float(runtime_s),
        }
        for axis_index, axis_name in enumerate(axis_names):
            row[f"estimated_{axis_name}"] = float(estimated[t, axis_index])
            row[f"expected_{axis_name}"] = float(expected[t, axis_index])
            row[f"abs_error_{axis_name}"] = float(abs(errors[t, axis_index]))
        row["abs_error_mean"] = float(np.mean(np.abs(errors[t])))
        rows.append(row)
    return rows

def _write_snr_average_interpolation_table(path: Path, rows: list[dict]) -> None:
    """Write per-level SNR means plus 0.01-step interpolation for inspection."""
    buckets: dict[tuple[str, str, float], list[dict]] = {}
    for row in rows:
        key = (str(row["scenario"]), str(row["method"]), float(row["level"]))
        buckets.setdefault(key, []).append(row)

    mean_rows = []
    for (scenario, method, level), level_rows in sorted(buckets.items()):
        mean_rows.append(
            {
                "source": "measured",
                "scenario": scenario,
                "method": method,
                "noise_sigma": float(level),
                "n_frames": int(len(level_rows)),
                "abs_error_mean": float(np.nanmean([_as_float(row.get("abs_error_mean")) for row in level_rows])),
                "pearson_before_mean": float(np.nanmean([_as_float(row.get("pearson_before")) for row in level_rows])),
                "pearson_after_mean": float(np.nanmean([_as_float(row.get("pearson_after")) for row in level_rows])),
                "runtime_s_mean": float(np.nanmean([_as_float(row.get("runtime_s")) for row in level_rows])),
            }
        )

    interpolated_rows = []
    group_keys = sorted({(row["scenario"], row["method"]) for row in mean_rows})
    value_keys = ("abs_error_mean", "pearson_before_mean", "pearson_after_mean", "runtime_s_mean")
    for scenario, method in group_keys:
        measured = sorted(
            [row for row in mean_rows if row["scenario"] == scenario and row["method"] == method],
            key=lambda row: row["noise_sigma"],
        )
        levels = np.asarray([row["noise_sigma"] for row in measured], dtype=np.float32)
        if levels.size < 2:
            continue
        start = np.ceil(float(np.min(levels)) * 100.0) / 100.0
        stop = np.floor(float(np.max(levels)) * 100.0) / 100.0
        if stop < start:
            continue
        grid = np.round(np.arange(start, stop + 0.0001, 0.01), 2)
        for noise_sigma in grid:
            row = {
                "source": "interpolated",
                "scenario": scenario,
                "method": method,
                "noise_sigma": float(noise_sigma),
                "n_frames": "",
            }
            for value_key in value_keys:
                values = np.asarray([measured_row[value_key] for measured_row in measured], dtype=np.float32)
                valid = np.isfinite(values)
                row[value_key] = float(np.interp(float(noise_sigma), levels[valid], values[valid])) if np.sum(valid) >= 2 else float("nan")
            interpolated_rows.append(row)

    _write_rows(path, [*mean_rows, *interpolated_rows])

# %% BENCHMARK CASES
def run_representative_2d(args) -> None:
    print("Running representative 2D+t translation benchmark")
    
    registration_template_time_range = "all" # "all"
    stack, applied = create_2d_motion_distorted_stack(
        time_count=24 if args.quick else 64,
        channel_count=2,
        shape_yx=(128, 128) if args.quick else (256, 256),
        noise_sigma=0.03,
        random_state=7,
        shift_amplitude_y=8.0,
        shift_amplitude_x=5.0,
    )
    registered, details, estimated, expected, errors, before, after, summary = _run_translation_registration(
        stack,
        applied_shift=applied,
        method_label="phase",
        dimension="2d_t",
        time_registration_mode="projection",
        zreg=False,
        n_jobs=args.n_jobs,
        registration_template_time_range=registration_template_time_range,
    )
    # The registration itself uses an all-frame template. For the figure and GT
    # benchmark, re-reference the detected correction trace to t=0 so the
    # plotted values remain directly comparable to the synthetic ground truth.
    estimated = estimated - estimated[0:1, :]
    expected = _expected_correction(applied)
    errors = estimated - expected
    save_stack(
        OUTPUT_DIR / "representative_2d_phase_registered.ome.tif",
        registered,
        metadata=create_stack_metadata(stack, verbose=False),
        registration_details=details,
        overwrite=True,
        verbose=False,
    )
    np.savez_compressed(
        OUTPUT_DIR / "representative_2d_phase.npz",
        raw=np.asarray(stack, dtype=np.float32),
        registered=np.asarray(registered, dtype=np.float32),
        estimated=estimated,
        expected=expected,
        errors=errors,
        pearson_before=before,
        pearson_after=after,
    )
    _write_rows(
        OUTPUT_DIR / "representative_2d_phase_metrics.csv",
        _per_frame_rows(
            scenario="representative_2d_translation",
            variable="none",
            level=0.0,
            method_label="phase",
            estimated=estimated,
            expected=expected,
            errors=errors,
            before=before,
            after=after,
            runtime_s=summary["runtime_s"],
        ),
    )

def run_representative_random_walk_2d(args) -> None:
    print("Running representative 2D+t random-walk translation benchmark")

    registration_template_time_range = "all"
    stack, applied = _create_2d_random_walk_motion_stack(
        time_count=24 if args.quick else 64,
        channel_count=2,
        shape_yx=(128, 128) if args.quick else (256, 256),
        noise_sigma=0.03,
        random_state=37,
        step_sigma_y=1.2,
        step_sigma_x=0.9,
        max_abs_y=8.0,
        max_abs_x=6.0,
    )
    registered, details, estimated, expected, errors, before, after, summary = _run_translation_registration(
        stack,
        applied_shift=applied,
        method_label="phase",
        dimension="2d_t_random_walk",
        time_registration_mode="projection",
        zreg=False,
        n_jobs=args.n_jobs,
        registration_template_time_range=registration_template_time_range,
    )
    # Match the Figure 2 display convention: compare the correction trace after
    # re-referencing to t=0 against the known per-frame correction.
    estimated = estimated - estimated[0:1, :]
    expected = _expected_correction(applied)
    errors = estimated - expected
    save_stack(
        OUTPUT_DIR / "representative_2d_random_walk_phase_registered.ome.tif",
        registered,
        metadata=create_stack_metadata(stack, verbose=False),
        registration_details=details,
        overwrite=True,
        verbose=False,
    )
    np.savez_compressed(
        OUTPUT_DIR / "representative_2d_random_walk_phase.npz",
        raw=np.asarray(stack, dtype=np.float32),
        registered=np.asarray(registered, dtype=np.float32),
        estimated=estimated,
        expected=expected,
        errors=errors,
        pearson_before=before,
        pearson_after=after,
    )
    _write_rows(
        OUTPUT_DIR / "representative_2d_random_walk_phase_metrics.csv",
        _per_frame_rows(
            scenario="representative_2d_random_walk_translation",
            variable="none",
            level=0.0,
            method_label="phase",
            estimated=estimated,
            expected=expected,
            errors=errors,
            before=before,
            after=after,
            runtime_s=summary["runtime_s"],
        ),
    )

def run_snr_benchmarks(args) -> None:
    if args.quick:
        print("Running quick SNR benchmark (fewer levels, smaller stacks)")
    else:
        print("Running full SNR benchmark (more levels, larger stacks)")
    levels = [0.005, 0.03, 0.08, 0.16] if args.quick else [0.002, 0.006, 0.015, 0.03, 0.06, 0.10, 0.16, 0.24, 0.32, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.5, 2.0, 3.0, 4.0, 5.0]
    methods = ["phase", "pystackreg", "normcorre"]
    filtered_methods = ["phase_median_filter", "pystackreg_median_filter", "normcorre_gsig"]
    rows = []
    for noise_sigma in levels:
        stack_2d, applied_2d = create_2d_motion_distorted_stack(
            time_count=18 if args.quick else 48,
            channel_count=2,
            shape_yx=(128, 128),
            noise_sigma=float(noise_sigma),
            random_state=1200 + int(noise_sigma * 10000),
        )
        for method_label in methods:
            _, _, estimated, expected, errors, before, after, summary = _run_translation_registration(
                stack_2d,
                applied_shift=applied_2d,
                method_label=method_label,
                dimension="2d_t",
                time_registration_mode="projection",
                zreg=False,
                n_jobs=args.n_jobs,
            )
            rows.extend(
                _per_frame_rows(
                    scenario="snr_2d_translation",
                    variable="noise_sigma",
                    level=float(noise_sigma),
                    method_label=method_label,
                    estimated=estimated,
                    expected=expected,
                    errors=errors,
                    before=before,
                    after=after,
                    runtime_s=summary["runtime_s"],
                )
            )
            gc.collect()

        stack_3d, applied_3d = create_3d_time_xy_motion_distorted_stack(
            time_count=12 if args.quick else 32,
            z_count=8 if args.quick else 14,
            channel_count=2,
            shape_yx=(96, 96) if args.quick else (128, 128),
            noise_sigma=float(noise_sigma),
            random_state=1300 + int(noise_sigma * 10000),
        )
        for method_label in methods:
            _, _, estimated, expected, errors, before, after, summary = _run_translation_registration(
                stack_3d,
                applied_shift=applied_3d,
                method_label=method_label,
                dimension="3d_t_projection",
                time_registration_mode="projection",
                zreg=False,
                n_jobs=args.n_jobs,
            )
            rows.extend(
                _per_frame_rows(
                    scenario="snr_3d_projection_translation",
                    variable="noise_sigma",
                    level=float(noise_sigma),
                    method_label=method_label,
                    estimated=estimated,
                    expected=expected,
                    errors=errors,
                    before=before,
                    after=after,
                    runtime_s=summary["runtime_s"],
                )
            )
            gc.collect()

        for filtered_method in filtered_methods:
            if filtered_method == "phase_median_filter":
                method_label = "phase"
                output_label = "phase_median_filter"
                filter_projections = True
            elif filtered_method == "pystackreg_median_filter":
                method_label = "pystackreg"
                output_label = "pystackreg_median_filter"
                filter_projections = True
            else:
                method_label = "normcorre_gsig"
                output_label = "normcorre_gsig"
                filter_projections = False
            _, _, estimated, expected, errors, before, after, summary = _run_translation_registration(
                stack_2d,
                applied_shift=applied_2d,
                method_label=method_label,
                dimension="2d_t",
                time_registration_mode="projection",
                zreg=False,
                n_jobs=args.n_jobs,
                filter_projections=filter_projections,
            )
            rows.extend(
                _per_frame_rows(
                    scenario="snr_2d_translation_prefilter",
                    variable="noise_sigma",
                    level=float(noise_sigma),
                    method_label=output_label,
                    estimated=estimated,
                    expected=expected,
                    errors=errors,
                    before=before,
                    after=after,
                    runtime_s=summary["runtime_s"],
                )
            )
            _, _, estimated, expected, errors, before, after, summary = _run_translation_registration(
                stack_3d,
                applied_shift=applied_3d,
                method_label=method_label,
                dimension="3d_t_projection",
                time_registration_mode="projection",
                zreg=False,
                n_jobs=args.n_jobs,
                filter_projections=filter_projections,
            )
            rows.extend(
                _per_frame_rows(
                    scenario="snr_3d_projection_translation_prefilter",
                    variable="noise_sigma",
                    level=float(noise_sigma),
                    method_label=output_label,
                    estimated=estimated,
                    expected=expected,
                    errors=errors,
                    before=before,
                    after=after,
                    runtime_s=summary["runtime_s"],
                )
            )
    _write_rows(OUTPUT_DIR / "snr_benchmark_metrics.csv", rows)
    _write_snr_average_interpolation_table(
        OUTPUT_DIR / "snr_benchmark_average_interpolated_metrics.csv",
        rows,
    )

def run_drift_benchmarks(args) -> None:    
    if args.quick:
        print("Running quick drift benchmark (fewer levels, smaller stacks)")
    else:
        print("Running full drift benchmark (more levels, larger stacks)")
    levels = [0.5, 1.25, 2.5, 4.0] if args.quick else [0.25, 0.5, 1.0, 1.55, 2.0, 2.5, 3.5, 5.0, 6.5, 8.0, 10, 15, 20, 30, 50]
    methods = ["phase", "pystackreg", "normcorre"]
    filtered_methods = [
        ("phase", "phase_median_filter", True),
        ("pystackreg", "pystackreg_median_filter", True),
        ("normcorre_gsig", "normcorre_gsig", False),
    ]
    rows = []
    base_stack, applied = create_2d_motion_distorted_stack(
        time_count=20 if args.quick else 56,
        channel_count=2,
        shape_yx=(128, 128),
        noise_sigma=0.025,
        random_state=1401,
    )
    # The base generator already applies the requested shifts. To avoid
    # reimplementing private synthetic internals, drift-amplitude benchmarks are
    # generated by recursively registering progressively reshifted copies.
    from scipy.ndimage import shift as ndi_shift

    base_reference = np.asarray(base_stack[0], dtype=np.float32)
    for scale in levels:
        scaled = np.empty_like(base_stack)
        scaled[0] = base_reference
        scaled_applied = np.asarray(applied, dtype=np.float32) * float(scale)
        for t in range(base_stack.shape[0]):
            for z in range(base_stack.shape[1]):
                for c in range(base_stack.shape[2]):
                    scaled[t, z, c] = ndi_shift(
                        base_reference[z, c],
                        shift=tuple(float(v) for v in scaled_applied[t]),
                        order=1,
                        mode="constant",
                        cval=0.0,
                    )
        for method_label in methods:
            _, _, estimated, expected, errors, before, after, summary = _run_translation_registration(
                scaled,
                applied_shift=scaled_applied,
                method_label=method_label,
                dimension="2d_t",
                time_registration_mode="projection",
                zreg=False,
                n_jobs=args.n_jobs,
            )
            rows.extend(
                _per_frame_rows(
                    scenario="drift_2d_translation",
                    variable="drift_scale",
                    level=float(scale),
                    method_label=method_label,
                    estimated=estimated,
                    expected=expected,
                    errors=errors,
                    before=before,
                    after=after,
                    runtime_s=summary["runtime_s"],
                )
            )
        for method_label, output_label, filter_projections in filtered_methods:
            _, _, estimated, expected, errors, before, after, summary = _run_translation_registration(
                scaled,
                applied_shift=scaled_applied,
                method_label=method_label,
                dimension="2d_t",
                time_registration_mode="projection",
                zreg=False,
                n_jobs=args.n_jobs,
                filter_projections=filter_projections,
            )
            rows.extend(
                _per_frame_rows(
                    scenario="drift_2d_translation_prefilter",
                    variable="drift_scale",
                    level=float(scale),
                    method_label=output_label,
                    estimated=estimated,
                    expected=expected,
                    errors=errors,
                    before=before,
                    after=after,
                    runtime_s=summary["runtime_s"],
                )
            )
    _write_rows(OUTPUT_DIR / "drift_benchmark_metrics.csv", rows)

def run_piecewise_benchmark(args) -> None:
    print("Running piecewise 2D+t translation benchmark")
    
    stack = _create_piecewise_wave_motion_stack(
        time_count=40 if args.quick else 160,
        channel_count=2,
        shape_yx=(128, 128) if args.quick else (256, 256),
        amplitude_px=5.0 if args.quick else 9.0,
        noise_sigma=0.008,
        random_state=1501,
    )
    rows = []
    for method_label in ["phase", "normcorre_pw"]:
        method, extra = _normalize_method_name(method_label)
        if method_label == "normcorre_pw":
            extra = {
                **extra,
                "nc_strides": (24, 24) if args.quick else (40, 40),
                "nc_overlaps": (18, 18) if args.quick else (32, 32),
                "nc_max_deviation_rigid": (8, 8) if args.quick else (14, 14),
                "nc_n_iterations": 2,
            }
        before = _frame_correlations(stack)
        start = time.perf_counter()
        registered, details = register_stack(
            stack,
            registration_channel=0,
            registration_stack=0,
            method=method,
            time_registration_mode="projection",
            projection_method="max",
            nc_n_jobs=args.n_jobs,
            n_jobs=args.n_jobs,
            transform_order=1,
            verbose=False,
            return_shifts=True,
            return_details=True,
            **extra,
        )
        runtime_s = time.perf_counter() - start
        after = _frame_correlations(registered)
        rows.append(
            {
                "scenario": "piecewise_2d_translation",
                "method": method_label,
                "runtime_s": runtime_s,
                "mean_corr_before": float(np.nanmean(before)),
                "mean_corr_after": float(np.nanmean(after)),
                "delta_corr": float(np.nanmean(after) - np.nanmean(before)),
            }
        )
        gc.collect()
    _write_rows(OUTPUT_DIR / "piecewise_benchmark_metrics.csv", rows)

def run_rigid3d_benchmarks(args) -> None:
    print("Running 3D+t rigid motion benchmark")
    
    rows = []
    dense_image = EXAMPLE_SYNTHETIC_DIR / "synthetic_3d_t_rigid_simpleitk.ome.tif"
    dense_gt = EXAMPLE_SYNTHETIC_DIR / "synthetic_3d_t_rigid_simpleitk_rigid_transform_gt.csv"
    sparse_image = EXAMPLE_SYNTHETIC_DIR / "synthetic_3d_t_rigid_points.ome.tif"
    sparse_gt = EXAMPLE_SYNTHETIC_DIR / "synthetic_3d_t_rigid_points_rigid_transform_gt.csv"

    if dense_image.exists() and dense_gt.exists():
        print(f"   Loading dense synthetic 3D+t stack from {dense_image}")
        dense_stack = load_stack(dense_image, verbose=False)
        dense_shifts, dense_rotations = load_expected_rigid_corrections(dense_gt)
        dense_centers = np.repeat(
            np.asarray(
                [(dense_stack.shape[1] - 1) / 2, (dense_stack.shape[3] - 1) / 2, (dense_stack.shape[4] - 1) / 2],
                dtype=np.float32,
            )[None, :],
            dense_stack.shape[0],
            axis=0,
        )
        dense_payload = (dense_stack, np.asarray(dense_shifts, dtype=np.float32), np.asarray(dense_rotations, dtype=np.float32), dense_centers, True)
    else:
        print("   Creating dense synthetic 3D+t stack")
        dense_payload = (*create_3d_time_rigid_motion_distorted_stack(
            time_count=7,
            z_count=20,
            channel_count=2,
            shape_yx=(96, 96),
            rotation_mode="all",
            center_mode="middle",
            noise_sigma=0.01,
            random_state=71,
        ), False)

    if sparse_image.exists() and sparse_gt.exists():
        print(f"   Loading sparse synthetic 3D+t stack from {sparse_image}")
        sparse_stack = load_stack(sparse_image, verbose=False)
        sparse_shifts, sparse_rotations = load_expected_rigid_corrections(sparse_gt)
        sparse_centers = np.repeat(
            np.asarray(
                [(sparse_stack.shape[1] - 1) / 2, (sparse_stack.shape[3] - 1) / 2, (sparse_stack.shape[4] - 1) / 2],
                dtype=np.float32,
            )[None, :],
            sparse_stack.shape[0],
            axis=0,
        )
        sparse_payload = (sparse_stack, np.asarray(sparse_shifts, dtype=np.float32), np.asarray(sparse_rotations, dtype=np.float32), sparse_centers, True)
    else:
        print("   Creating sparse synthetic 3D+t stack")
        sparse_payload = (*create_3d_time_sparse_puncta_rigid_motion_distorted_stack(random_state=73), False)

    cases = [
        ("dense", dense_payload, "simpleitk", 1),
        ("dense", dense_payload, "points", 0),
        ("sparse", sparse_payload, "simpleitk", 1),
        ("sparse", sparse_payload, "points", 0),
    ]
    for case_name, payload, backend, order in cases:
        stack, shift_values, rotation_values, centers, values_are_expected = payload
        if values_are_expected:
            expected_shift = np.asarray(shift_values, dtype=np.float32)
            expected_rot = np.asarray(rotation_values, dtype=np.float32)
        else:
            expected_shift, expected_rot = _expected_rigid_corrections_from_applied(
                shift_values,
                rotation_values,
                registration_stack=0,
            )
        before = _frame_correlations(stack)
        start = time.perf_counter()
        registered = None
        details = None
        error_message = ""
        try:
            registered, details = register_stack(
                stack,
                metadata=create_stack_metadata(stack, verbose=False),
                registration_channel=0,
                registration_stack=0,
                method="phase_cross_correlation",
                time_registration_mode="full_3d",
                zreg=True,
                rotreg=True,
                rigid_3d_backend=backend,
                rot_init_iterations=0,
                rot_metric="correlation",
                rot_iterations=50 if args.quick else 120,
                rot_n_jobs=args.n_jobs,
                rot_points_max_points=140,
                rot_points_min_distance=2,
                rot_points_threshold_rel=0.15 if case_name == "sparse" else 0.25,
                rot_points_iterations=40,
                rot_points_max_match_distance=9.0,
                transform_order=order,
                zero_clip=False,
                verbose=False,
                return_shifts=True,
                return_details=True,
            )
        except Exception as exc:  # keep the benchmark table complete when a backend is unsuitable
            error_message = f"{type(exc).__name__}: {exc}"
        runtime_s = time.perf_counter() - start
        if registered is None or details is None:
            after = np.full(stack.shape[0], np.nan, dtype=np.float32)
            estimated_shift = np.full_like(expected_shift, np.nan, dtype=np.float32)
            estimated_rot = np.full_like(expected_rot, np.nan, dtype=np.float32)
        else:
            after = _frame_correlations(registered)
            estimated_shift = np.asarray(details["time_shifts_zyx"], dtype=np.float32)
            estimated_rot = np.asarray(details["rotation_shifts_zyx_deg"], dtype=np.float32)
        for t in range(stack.shape[0]):
            row = {
                "scenario": "rigid3d",
                "case": case_name,
                "backend": backend,
                "case_backend": f"{case_name}_{backend}",
                "frame": int(t),
                "runtime_s": float(runtime_s),
                "error_message": error_message,
                "pearson_before": _as_float(before[t]),
                "pearson_after": _as_float(after[t]),
            }
            for axis_index, axis_name in enumerate(["z", "y", "x"]):
                row[f"estimated_shift_{axis_name}"] = float(estimated_shift[t, axis_index])
                row[f"expected_shift_{axis_name}"] = float(expected_shift[t, axis_index])
                row[f"abs_shift_error_{axis_name}"] = float(abs(estimated_shift[t, axis_index] - expected_shift[t, axis_index]))
                row[f"estimated_rotation_{axis_name}"] = float(estimated_rot[t, axis_index])
                row[f"expected_rotation_{axis_name}"] = float(expected_rot[t, axis_index])
                row[f"abs_rotation_error_{axis_name}"] = float(abs(estimated_rot[t, axis_index] - expected_rot[t, axis_index]))
                row[f"center_{axis_name}"] = float(centers[t, axis_index])
            row["abs_shift_error_mean"] = float(np.mean(np.abs(estimated_shift[t] - expected_shift[t])))
            row["abs_rotation_error_mean"] = float(np.mean(np.abs(estimated_rot[t] - expected_rot[t])))
            rows.append(row)
        if registered is not None:
            np.savez_compressed(
                OUTPUT_DIR / f"rigid3d_{case_name}_{backend}.npz",
                raw=np.asarray(stack, dtype=np.float32),
                registered=np.asarray(registered, dtype=np.float32),
                estimated_shift=estimated_shift,
                estimated_rotation=estimated_rot,
                expected_shift=expected_shift,
                expected_rotation=expected_rot,
                pearson_before=before,
                pearson_after=after,
            )
        gc.collect()
    _write_rows(OUTPUT_DIR / "rigid3d_benchmark_metrics.csv", rows)

def run_projection_parallel_benchmarks(args) -> None:
    print("Running parallel scaling benchmark for 2D+t projection registration")
    
    rows = []
    lateral_stack, lateral_shift = create_3d_time_xy_motion_distorted_stack(
        time_count=10 if args.quick else 32,
        z_count=12 if args.quick else 20,
        channel_count=2,
        shape_yx=(96, 96) if args.quick else (512, 512),
        noise_sigma=0.02,
        random_state=1651,
    )
    axial_stack, axial_shift = create_3d_time_zyx_motion_distorted_stack(
        time_count=10 if args.quick else 32,
        z_count=12 if args.quick else 20,
        channel_count=2,
        shape_yx=(96, 96) if args.quick else (512, 512),
        noise_sigma=0.02,
        random_state=1652,
    )
    cases = [
        ("lateral_motion", lateral_stack, lateral_shift, "projection", False),
        ("lateral_motion", lateral_stack, np.column_stack([np.zeros(lateral_shift.shape[0]), lateral_shift]), "full_3d", True),
        ("axial_motion", axial_stack, axial_shift, "projection", True),
        ("axial_motion", axial_stack, axial_shift, "full_3d", True),
    ]
    for scenario, stack, applied_shift, mode, zreg in cases:
        _, _, estimated, expected, errors, before, after, summary = _run_translation_registration(
            stack,
            applied_shift=applied_shift,
            method_label="phase",
            dimension="3d_t",
            time_registration_mode=mode,
            zreg=zreg,
            n_jobs=args.n_jobs,
        )
        for row in _per_frame_rows(
            scenario=scenario,
            variable="workflow",
            level=0.0,
            method_label=mode,
            estimated=estimated,
            expected=expected,
            errors=errors,
            before=before,
            after=after,
            runtime_s=summary["runtime_s"],
        ):
            row["workflow"] = mode
            rows.append(row)

    time_counts = [1, 15, 25, 50, 100, 250, 500, 1000] if args.quick else [1, 15, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 20000]
    parallel_shape_yx = (96, 96) if args.quick else (512, 512)
    parallel_input_folder = MEMMAP_DIR / "parallel_scaling_inputs"
    parallel_output_folder = MEMMAP_DIR / "parallel_scaling_outputs"
    for time_count in time_counts:
        use_disk_stack = (not args.quick) and (
            int(time_count) * 1 * 2 * parallel_shape_yx[0] * parallel_shape_yx[1] * np.dtype(np.float32).itemsize
            > 512 * 1024 * 1024
        )
        cleanup_omio_cache(parallel_output_folder, full_cleanup=True, verbose=False)
        if use_disk_stack:
            cleanup_omio_cache(parallel_input_folder, full_cleanup=True, verbose=False)
        stack, applied = _create_parallel_scaling_stack(
            time_count=int(time_count),
            channel_count=2,
            shape_yx=parallel_shape_yx,
            noise_sigma=0.025,
            random_state=1660 + int(time_count),
            use_memmap=use_disk_stack,
            memmap_folder=parallel_input_folder,
            memmap_name=f"parallel_scaling_input_T{int(time_count)}",
        )
        try:
            for worker_count in [1, 8]:
                if time_count == 1:
                    rows.append(
                        {
                            "scenario": "parallel_scaling",
                            "time_count": int(time_count),
                            "n_jobs": int(worker_count),
                            "runtime_s": 0.0,
                            "mean_abs_error": 0.0,
                            "mean_corr_before": 1.0,
                            "mean_corr_after": 1.0,
                            "input_storage": "disk_zarr" if use_disk_stack else "ram",
                            "raw_size_mb": _raw_size_mb(stack),
                        }
                    )
                    continue
                summary = _run_parallel_scaling_registration(
                    stack,
                    applied_shift=applied,
                    n_jobs=worker_count,
                    output_memmap_folder=parallel_output_folder if use_disk_stack else None,
                    output_memmap_name=f"parallel_scaling_registered_T{int(time_count)}_n{int(worker_count)}",
                )
                row = {
                    "scenario": "parallel_scaling",
                    "time_count": int(time_count),
                    "n_jobs": int(worker_count),
                    "runtime_s": float(summary["runtime_s"]),
                    "mean_abs_error": float(summary["mean_abs_error"]),
                    "mean_corr_before": float(summary["mean_corr_before"]),
                    "mean_corr_after": float(summary["mean_corr_after"]),
                    "input_storage": "disk_zarr" if use_disk_stack else "ram",
                    "raw_size_mb": _raw_size_mb(stack),
                }
                rows.append(row)
                cleanup_omio_cache(parallel_output_folder, full_cleanup=True, verbose=False)
        finally:
            del stack, applied
            gc.collect()
            cleanup_omio_cache(parallel_output_folder, full_cleanup=True, verbose=False)
            if use_disk_stack:
                cleanup_omio_cache(parallel_input_folder, full_cleanup=True, verbose=False)
    _write_rows(OUTPUT_DIR / "projection_parallel_benchmark_metrics.csv", rows)

def _memory_trace_paths(workflow: str) -> tuple[Path, Path, Path, Path]:
    trace_csv = OUTPUT_DIR / f"memory_trace_{workflow}.csv"
    trace_plot = OUTPUT_DIR / f"memory_trace_{workflow}_raw.png"
    timing_csv = OUTPUT_DIR / f"memory_timing_{workflow}.csv"
    registered_path = OUTPUT_DIR / f"memory_{workflow}_registered.ome.tif"
    return trace_csv, trace_plot, timing_csv, registered_path

def _clean_trace_values(values: np.ndarray) -> np.ndarray:
    """Remove isolated implausible process-memory samples from a trace."""

    values = np.asarray(values, dtype=np.float32).copy()
    finite = np.isfinite(values)
    finite_indices = np.flatnonzero(finite)
    for index in finite_indices:
        previous_candidates = finite_indices[finite_indices < index]
        next_candidates = finite_indices[finite_indices > index]
        if not previous_candidates.size or not next_candidates.size:
            continue
        previous_value = values[previous_candidates[-1]]
        next_value = values[next_candidates[0]]
        local_reference = min(float(previous_value), float(next_value))
        if local_reference > 0 and float(values[index]) < 0.25 * local_reference:
            values[index] = np.nan
    return values

def _summarize_memory_trace(path: Path) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    def finite_column(key: str, *, clean: bool = False) -> np.ndarray:
        values = []
        for row in rows:
            text = str(row.get(key, "")).strip()
            if not text:
                values.append(np.nan)
                continue
            value = float(text)
            values.append(value if np.isfinite(value) else np.nan)
        array = np.asarray(values, dtype=np.float32)
        if clean:
            array = _clean_trace_values(array)
        return array

    rss = finite_column("rss_mb")
    uss = finite_column("uss_mb", clean=True)
    elapsed = finite_column("elapsed_s")

    def metrics(values: np.ndarray) -> tuple[float, float, float, float]:
        finite = values[np.isfinite(values)]
        if not finite.size:
            return (float("nan"),) * 4
        baseline = float(finite[0])
        peak = float(np.max(finite))
        final = float(finite[-1])
        return baseline, peak, final, peak - baseline

    baseline_rss, peak_rss, final_rss, peak_delta_rss = metrics(rss)
    baseline_uss, peak_uss, final_uss, peak_delta_uss = metrics(uss)
    finite_elapsed = elapsed[np.isfinite(elapsed)]
    return {
        "baseline_rss_mb": baseline_rss,
        "peak_rss_mb": peak_rss,
        "final_rss_mb": final_rss,
        "peak_delta_rss_mb": peak_delta_rss,
        "final_delta_rss_mb": final_rss - baseline_rss if np.isfinite(final_rss) and np.isfinite(baseline_rss) else float("nan"),
        "baseline_uss_mb": baseline_uss,
        "peak_uss_mb": peak_uss,
        "final_uss_mb": final_uss,
        "peak_delta_uss_mb": peak_delta_uss,
        "final_delta_uss_mb": final_uss - baseline_uss if np.isfinite(final_uss) and np.isfinite(baseline_uss) else float("nan"),
        "duration_s": float(finite_elapsed[-1]) if finite_elapsed.size else float("nan"),
    }

def run_memory_worker(args) -> None:
    workflow = str(args.memory_worker)
    if args.memory_input is None:
        raise ValueError("--memory-input is required when --memory-worker is used.")
    input_path = Path(args.memory_input)
    use_memmap = workflow in {"memmap_cold", "memmap_warm"}
    output_memmap_name = f"preprint_memory_registered_{workflow}" if use_memmap else None
    if workflow == "memmap_cold":
        cleanup_omio_cache(MEMMAP_DIR, full_cleanup=True, verbose=False)

    trace_csv, trace_plot, timing_csv, registered_path = _memory_trace_paths(workflow)
    timing_rows = []
    with profile_memory(csv_path=trace_csv, plot_path=trace_plot, interval_s=0.10, label=f"preprint_memory_{workflow}") as tracker:
        tracker.mark("load:start")
        step_start = time.perf_counter()
        image, metadata = load_stack(
            input_path,
            return_metadata=True,
            use_memmap=use_memmap,
            memmap_folder=MEMMAP_DIR if use_memmap else None,
            memmap_reuse=workflow == "memmap_warm",
            verbose=False,
        )
        timing_rows.append({"workflow": workflow, "step": "load", "duration_s": time.perf_counter() - step_start})
        tracker.mark("load:end")

        tracker.mark("register:start")
        step_start = time.perf_counter()
        registered, details = register_stack(
            image,
            registration_channel=0,
            registration_stack=0,
            method="phase_cross_correlation",
            time_registration_mode="projection",
            projection_method="max",
            n_jobs=args.n_jobs,
            output_use_memmap=use_memmap,
            output_memmap_folder=MEMMAP_DIR if use_memmap else None,
            output_memmap_name=output_memmap_name,
            verbose=False,
            return_shifts=True,
            return_details=True,
        )
        timing_rows.append({"workflow": workflow, "step": "register", "duration_s": time.perf_counter() - step_start})
        tracker.mark("register:end")

        tracker.mark("save:start")
        step_start = time.perf_counter()
        save_stack(
            registered_path,
            registered,
            metadata=metadata,
            registration_details=details,
            overwrite=True,
            verbose=False,
        )
        timing_rows.append({"workflow": workflow, "step": "save", "duration_s": time.perf_counter() - step_start})
        tracker.mark("save:end")
    _write_rows(timing_csv, timing_rows)

def run_memory_benchmark(args) -> None:
    print("Running memory usage benchmark for 2D+t projection registration")

    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
    MEMMAP_DIR.mkdir(parents=True, exist_ok=True)
    cleanup_omio_cache(MEMMAP_DIR, full_cleanup=True, verbose=False)
    stack, _ = create_2d_motion_distorted_stack(
        time_count=64 if args.quick else 512,
        channel_count=2,
        shape_yx=(256, 256) if args.quick else (512, 512),
        noise_sigma=0.02,
        random_state=1701,
    )
    raw_size_mb = _raw_size_mb(stack)
    input_path = SYNTHETIC_DIR / "preprint_memory_input.ome.tif"
    save_stack(
        input_path,
        stack,
        metadata=create_stack_metadata(stack, verbose=False),
        overwrite=True,
        verbose=False,
    )
    del stack
    gc.collect()

    workflows = ["memmap_cold", "memmap_warm", "in_memory"]
    for workflow in workflows:
        print(f"   Running memory workflow: {workflow}")
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--memory-worker",
                workflow,
                "--memory-input",
                str(input_path),
                "--memory-raw-size-mb",
                str(float(raw_size_mb)),
                "--n-jobs",
                str(int(args.n_jobs)),
            ],
            check=True,
        )

    timing_rows = []
    summary = {
        "input_path": str(input_path),
        "raw_size_mb": float(raw_size_mb),
        "workflows": workflows,
    }
    for workflow in workflows:
        trace_csv, trace_plot, timing_csv, registered_path = _memory_trace_paths(workflow)
        timing_rows.extend(_read_csv_rows(timing_csv) if timing_csv.exists() else [])
        workflow_summary = _summarize_memory_trace(trace_csv)
        summary[f"{workflow}_registered_path"] = str(registered_path)
        summary[f"{workflow}_trace_csv"] = str(trace_csv)
        summary[f"{workflow}_trace_plot"] = str(trace_plot)
        summary.update({f"{workflow}_{key}": value for key, value in workflow_summary.items()})
    (OUTPUT_DIR / "memory_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_rows(OUTPUT_DIR / "memory_timing_metrics.csv", timing_rows)

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=(
            "representative",
            "representative_random_walk",
            "snr",
            "drift",
            "piecewise",
            "rigid3d",
            "projection_parallel",
            "memory",
            "all",
        ),
        default="all",
    )
    parser.add_argument("--quick", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument("--memory-worker", choices=("memmap_cold", "memmap_warm", "in_memory"), default=None, help=argparse.SUPPRESS)
    parser.add_argument("--memory-input", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--memory-raw-size-mb", type=float, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()
# %% MAIN FUNCTION
def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.memory_worker is not None:
        run_memory_worker(args)
        return
    selected = args.case
    if selected in {"representative", "all"}:
        print("Running representative 2D+t registration benchmark")
        run_representative_2d(args)
    if selected in {"representative_random_walk", "all"}:
        print("Running representative 2D+t random-walk registration benchmark")
        run_representative_random_walk_2d(args)
    if selected in {"snr", "all"}:
        print("Running SNR robustness benchmarks")
        run_snr_benchmarks(args)
    if selected in {"drift", "all"}:
        print("Running drift-amplitude benchmarks")
        run_drift_benchmarks(args)
    if selected in {"piecewise", "all"}:
        print("Running piecewise translation benchmark")
        run_piecewise_benchmark(args)
    if selected in {"rigid3d", "all"}:
        print("Running full 3D rigid benchmarks")
        run_rigid3d_benchmarks(args)
    if selected in {"projection_parallel", "all"}:
        print("Running projection/full-volume and parallel scaling benchmarks")
        run_projection_parallel_benchmarks(args)
    if selected in {"memory", "all"}:
        print("Running memory benchmark")
        run_memory_benchmark(args)
    print(f"Wrote benchmark outputs to {OUTPUT_DIR}")
# %% MAIN ENTRYPOINT
if __name__ == "__main__":
    main()
# %% END
