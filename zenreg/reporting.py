"""
Registration report writers for ZenReg outputs.

Author: Fabrizio Musacchio
Date: July 2026
"""
# %% IMPORTS
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from ._axes import CANONICAL_AXIS_ORDER, ensure_tzcyx_stack, normalize_zrange
from .registration import _compute_registration_frame_correlations
# %% CONSTANTS
SETTING_KEYS = (
    "registration_channel",
    "registration_channel_requested",
    "registration_channel_used",
    "registration_channel_fallback",
    "registration_channel_fallback_reason",
    "registration_stack",
    "registration_template_time_range",
    "registration_range",
    "registration_range_requested",
    "registration_range_axis",
    "registration_range_ignored_reason",
    "method",
    "time_registration_mode",
    "effective_time_registration_mode",
    "time_reference_mode",
    "intra_stack",
    "zreg",
    "rotreg",
    "rotreg_iter",
    "rigid_3d_backend",
    "rot_spacing_zyx",
    "rot_init_iterations",
    "rot_metric",
    "rot_shrink_factors",
    "rot_smoothing_sigmas",
    "rot_iterations",
    "rot_learning_rate",
    "rot_min_step",
    "rot_sampling_percentage",
    "rot_cval",
    "rot_n_jobs",
    "rot_points_max_points",
    "rot_points_min_distance",
    "rot_points_threshold_rel",
    "rot_points_iterations",
    "rot_points_max_match_distance",
    "registration_z_range",
    "projection_range",
    "projection_method",
    "filter_slices",
    "filter_projections",
    "median_kernel_size",
    "calc_SNR",
    "calc_CNR",
    "SNR_sampling_step",
    "CNR_sampling_step",
    "quality_background_percentile",
    "quality_signal_percentile",
    "max_xy_shifts",
    "max_z_shifts",
    "max_rot_shifts",
    "max_shifts",
    "max_deviation_rigid",
    "strides",
    "overlaps",
    "patch_grid_shape",
    "upsample_factor",
    "nc_pw_rigid",
    "nc_strides",
    "nc_overlaps",
    "nc_max_deviation_rigid",
    "nc_n_iterations",
    "nc_correction_iterations",
    "nc_niter_rig",
    "nc_template_init_mode",
    "nc_template_update_method",
    "nc_splits",
    "nc_gSig_filt",
    "nc_shift_interpolation",
    "nc_border_nan",
    "nc_n_jobs",
    "gSig_filt",
    "min_mov",
    "add_to_movie",
    "nonneg_movie",
    "n_iterations",
    "correction_iterations",
    "niter_rig",
    "template_init_mode",
    "template_update_method",
    "splits",
    "shift_interpolation",
    "n_jobs",
    "output_use_memmap",
    "output_memmap_folder",
    "output_memmap_name",
    "phase_cross_correlation_upsample_factor",
    "phase_cross_correlation_normalization",
    "transform_backend",
    "transform_order",
    "transform_mode",
    "transform_cval",
    "border_nan",
    "zero_clip",
    "zero_clip_mode",
    "zero_clip_mask_threshold",
    "zero_clip_mask_strategy",
    "zero_clip_mask_min_fraction",
    "zero_clip_margin_zyx",
    "zero_clip_bounds",
    "zero_clip_failed_reason",
    "stack_shape_tzcyx",
)
_SUMMARY_MARKER_FRAME_LIMIT = 250

# %% HELPER FUNCTIONS
def _as_details_dict(registration_details: dict[str, Any] | np.ndarray) -> dict[str, Any]:
    """Normalize report input to a details dictionary."""

    if isinstance(registration_details, dict):
        return registration_details
    shifts_yx = np.asarray(registration_details, dtype=np.float32)
    shifts_zyx = np.zeros((shifts_yx.shape[0], 3), dtype=np.float32)
    shifts_zyx[:, 1:] = shifts_yx
    return {
        "time_shifts_zyx": shifts_zyx,
        "time_shifts_yx": shifts_yx,
    }

def _plain_value(value):
    """Convert NumPy-rich values to YAML/CSV-friendly Python values."""

    if isinstance(value, np.ndarray):
        return _plain_value(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _plain_value(val) for key, val in value.items()}
    if isinstance(value, tuple | list):
        return [_plain_value(item) for item in value]
    return value

def _format_yaml_scalar(value) -> str:
    """Return a scalar represented as a conservative YAML subset."""

    value = _plain_value(value)
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        if isinstance(value, float) and not np.isfinite(value):
            return "null"
        return repr(value)
    return json.dumps(str(value))

def _append_yaml_value(lines: list[str], key: str, value, *, indent: int = 0) -> None:
    """Append one YAML key/value block to ``lines``."""

    prefix = " " * indent
    value = _plain_value(value)
    if isinstance(value, dict):
        lines.append(f"{prefix}{key}:")
        if not value:
            lines.append(f"{prefix}  {{}}")
        for child_key, child_value in value.items():
            _append_yaml_value(lines, str(child_key), child_value, indent=indent + 2)
        return
    if isinstance(value, list):
        if not value:
            lines.append(f"{prefix}{key}: []")
            return
        if all(not isinstance(item, dict | list) for item in value):
            items = ", ".join(_format_yaml_scalar(item) for item in value)
            lines.append(f"{prefix}{key}: [{items}]")
            return
        lines.append(f"{prefix}{key}:")
        for item in value:
            if isinstance(item, dict):
                lines.append(f"{prefix}  -")
                for child_key, child_value in item.items():
                    _append_yaml_value(lines, str(child_key), child_value, indent=indent + 4)
            else:
                lines.append(f"{prefix}  - {_format_yaml_scalar(item)}")
        return
    lines.append(f"{prefix}{key}: {_format_yaml_scalar(value)}")

def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    """Write a small dependency-free YAML file."""

    lines: list[str] = []
    for key, value in payload.items():
        _append_yaml_value(lines, key, value)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def _report_prefix(output_image_path: Path, report_prefix: str | Path | None) -> Path:
    """Return the common report path prefix."""

    if report_prefix is not None:
        return Path(report_prefix)
    name = output_image_path.name
    lower_name = name.lower()
    for suffix in (".ome.tiff", ".ome.tif", ".tiff", ".tif"):
        if lower_name.endswith(suffix):
            return output_image_path.with_name(name[: -len(suffix)])
    return output_image_path.with_suffix("")

def _time_shifts_zyx(details: dict[str, Any], time_count: int) -> np.ndarray:
    """Return time shifts as ``T, 3`` with missing values filled by zeros."""

    shifts_zyx = details.get("time_shifts_zyx")
    if shifts_zyx is not None:
        shifts_zyx = np.asarray(shifts_zyx, dtype=np.float32)
        if shifts_zyx.shape == (time_count, 3):
            return shifts_zyx

    shifts_yx = details.get("time_shifts_yx")
    shifts = np.zeros((time_count, 3), dtype=np.float32)
    if shifts_yx is not None:
        shifts_yx = np.asarray(shifts_yx, dtype=np.float32)
        if shifts_yx.shape == (time_count, 2):
            shifts[:, 1:] = shifts_yx
    return shifts

def _time_shifts_zyx_raw(details: dict[str, Any], time_count: int) -> np.ndarray:
    """Return raw time shifts as ``T, 3``; fall back to applied shifts."""

    shifts_zyx_raw = details.get("time_shifts_zyx_raw")
    if shifts_zyx_raw is not None:
        shifts_zyx_raw = np.asarray(shifts_zyx_raw, dtype=np.float32)
        if shifts_zyx_raw.shape == (time_count, 3):
            return shifts_zyx_raw

    shifts_yx_raw = details.get("time_shifts_yx_raw")
    if shifts_yx_raw is not None:
        shifts_yx_raw = np.asarray(shifts_yx_raw, dtype=np.float32)
        if shifts_yx_raw.shape == (time_count, 2):
            shifts = np.zeros((time_count, 3), dtype=np.float32)
            shifts[:, 1:] = shifts_yx_raw
            return shifts
    return _time_shifts_zyx(details, time_count)

def _rotation_shift_series_deg(details: dict[str, Any], time_count: int) -> tuple[np.ndarray, list[str]]:
    """Return per-frame rotation corrections and axis labels."""

    rotations_zyx = details.get("rotation_shifts_zyx_deg")
    if rotations_zyx is not None:
        rotations_zyx = np.asarray(rotations_zyx, dtype=np.float32)
        if rotations_zyx.shape == (time_count, 3):
            return rotations_zyx, ["rotation_z", "rotation_y", "rotation_x"]

    rotations = details.get("rotation_shifts_deg")
    if rotations is None:
        return np.empty((time_count, 0), dtype=np.float32), []
    rotations = np.asarray(rotations, dtype=np.float32)
    if rotations.shape != (time_count,):
        return np.empty((time_count, 0), dtype=np.float32), []
    return rotations[:, None], ["rotation_z"]

def _rotation_shift_series_raw_deg(details: dict[str, Any], time_count: int) -> tuple[np.ndarray, list[str]]:
    """Return raw per-frame rotation corrections; fall back to applied rotations."""

    rotations_zyx = details.get("rotation_shifts_zyx_deg_raw")
    if rotations_zyx is not None:
        rotations_zyx = np.asarray(rotations_zyx, dtype=np.float32)
        if rotations_zyx.shape == (time_count, 3):
            return rotations_zyx, ["rotation_z", "rotation_y", "rotation_x"]

    rotations = details.get("rotation_shifts_deg_raw")
    if rotations is not None:
        rotations = np.asarray(rotations, dtype=np.float32)
        if rotations.shape == (time_count,):
            return rotations[:, None], ["rotation_z"]
    return _rotation_shift_series_deg(details, time_count)

def _frame_correlations(registered_stack: np.ndarray, details: dict[str, Any]) -> np.ndarray:
    """Compute template-vs-registered Pearson correlations per time frame."""

    return _compute_registration_frame_correlations(
        registered_stack,
        registration_channel=int(details.get("registration_channel", 0)),
        registration_stack=int(details.get("registration_stack", 0)),
        registration_template_time_range=details.get("registration_template_time_range"),
        projection_range=details.get("registration_z_range", details.get("projection_range")),
        projection_method=str(details.get("projection_method", "max")),
        effective_time_registration_mode=str(details.get("effective_time_registration_mode", "projection")),
    )

def _pre_frame_correlations(details: dict[str, Any], time_count: int) -> np.ndarray:
    """Return pre-registration correlations when stored in registration details."""

    correlations = details.get("pearson_correlations_before")
    if correlations is None:
        return np.full(time_count, np.nan, dtype=np.float32)
    correlations = np.asarray(correlations, dtype=np.float32)
    if correlations.shape != (time_count,):
        return np.full(time_count, np.nan, dtype=np.float32)
    return correlations

def _quality_series(details: dict[str, Any], key: str, time_count: int) -> np.ndarray:
    """Return one optional per-frame quality series from registration details."""

    values = details.get(key)
    if values is None:
        return np.full(time_count, np.nan, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    if values.shape != (time_count,):
        return np.full(time_count, np.nan, dtype=np.float32)
    return values

def _snr_plot_series(snr_values: np.ndarray, *, plot_SNR_log: bool) -> tuple[np.ndarray, str, str]:
    """Return SNR values transformed for plotting and their labels."""

    snr_values = np.asarray(snr_values, dtype=np.float32)
    if not plot_SNR_log:
        return snr_values, "SNR", "SNR"
    plotted = np.full_like(snr_values, np.nan, dtype=np.float32)
    positive = np.isfinite(snr_values) & (snr_values > 0)
    plotted[positive] = np.log10(snr_values[positive])
    return plotted, "log10(SNR)", "log10 SNR"

def _csv_value(value) -> str:
    """Format one CSV cell."""

    if value is None:
        return ""
    value = _plain_value(value)
    if isinstance(value, float) and not np.isfinite(value):
        return ""
    if isinstance(value, int | float):
        return f"{float(value):.8g}"
    return str(value)

def _nan_stat(values: np.ndarray, reducer) -> float:
    """Return a finite NaN-aware statistic or NaN for empty/all-NaN inputs."""

    values = np.asarray(values, dtype=np.float32)
    if values.size == 0 or not np.any(np.isfinite(values)):
        return float("nan")
    return float(reducer(values))

def _limit_exceeded(applied: float, raw: float, *, atol: float = 1e-5) -> bool:
    """Return True when a raw estimate was clipped before application."""

    if not (np.isfinite(applied) and np.isfinite(raw)):
        return False
    return bool(abs(float(applied) - float(raw)) > float(atol))

def _write_shift_csv(
    path: Path,
    details: dict[str, Any],
    correlations_after: np.ndarray,
    correlations_before: np.ndarray,
    *,
    time_count: int,
) -> None:
    """Write frame-wise and optional intra-stack shifts to CSV."""

    shifts_zyx = _time_shifts_zyx(details, time_count)
    raw_shifts_zyx = _time_shifts_zyx_raw(details, time_count)
    rotations, _ = _rotation_shift_series_deg(details, time_count)
    raw_rotations, _ = _rotation_shift_series_raw_deg(details, time_count)
    snr_before = _quality_series(details, "snr_before", time_count)
    cnr_before = _quality_series(details, "cnr_before", time_count)
    fieldnames = [
        "scope",
        "frame",
        "z",
        "shift_z",
        "shift_y",
        "shift_x",
        "shift_z_raw",
        "shift_y_raw",
        "shift_x_raw",
        "shift_z_limit_exceeded",
        "shift_y_limit_exceeded",
        "shift_x_limit_exceeded",
        "intra_shift_y",
        "intra_shift_x",
        "rotation_z_deg",
        "rotation_y_deg",
        "rotation_x_deg",
        "rotation_deg",
        "rotation_z_deg_raw",
        "rotation_y_deg_raw",
        "rotation_x_deg_raw",
        "rotation_deg_raw",
        "rotation_z_limit_exceeded",
        "rotation_y_limit_exceeded",
        "rotation_x_limit_exceeded",
        "snr_before",
        "cnr_before",
        "pearson_correlation_before",
        "pearson_correlation_after",
        "pearson_correlation",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for t in range(time_count):
            writer.writerow(
                {
                    "scope": "time",
                    "frame": t,
                    "z": "",
                    "shift_z": _csv_value(shifts_zyx[t, 0]),
                    "shift_y": _csv_value(shifts_zyx[t, 1]),
                    "shift_x": _csv_value(shifts_zyx[t, 2]),
                    "shift_z_raw": _csv_value(raw_shifts_zyx[t, 0]),
                    "shift_y_raw": _csv_value(raw_shifts_zyx[t, 1]),
                    "shift_x_raw": _csv_value(raw_shifts_zyx[t, 2]),
                    "shift_z_limit_exceeded": str(_limit_exceeded(shifts_zyx[t, 0], raw_shifts_zyx[t, 0])),
                    "shift_y_limit_exceeded": str(_limit_exceeded(shifts_zyx[t, 1], raw_shifts_zyx[t, 1])),
                    "shift_x_limit_exceeded": str(_limit_exceeded(shifts_zyx[t, 2], raw_shifts_zyx[t, 2])),
                    "intra_shift_y": "",
                    "intra_shift_x": "",
                    "rotation_z_deg": _csv_value(rotations[t, 0]) if rotations.shape[1] >= 1 else "",
                    "rotation_y_deg": _csv_value(rotations[t, 1]) if rotations.shape[1] >= 2 else "",
                    "rotation_x_deg": _csv_value(rotations[t, 2]) if rotations.shape[1] >= 3 else "",
                    "rotation_deg": _csv_value(rotations[t, 0]) if rotations.shape[1] >= 1 else "",
                    "rotation_z_deg_raw": _csv_value(raw_rotations[t, 0]) if raw_rotations.shape[1] >= 1 else "",
                    "rotation_y_deg_raw": _csv_value(raw_rotations[t, 1]) if raw_rotations.shape[1] >= 2 else "",
                    "rotation_x_deg_raw": _csv_value(raw_rotations[t, 2]) if raw_rotations.shape[1] >= 3 else "",
                    "rotation_deg_raw": _csv_value(raw_rotations[t, 0]) if raw_rotations.shape[1] >= 1 else "",
                    "rotation_z_limit_exceeded": str(_limit_exceeded(rotations[t, 0], raw_rotations[t, 0]))
                    if rotations.shape[1] >= 1 and raw_rotations.shape[1] >= 1
                    else "",
                    "rotation_y_limit_exceeded": str(_limit_exceeded(rotations[t, 1], raw_rotations[t, 1]))
                    if rotations.shape[1] >= 2 and raw_rotations.shape[1] >= 2
                    else "",
                    "rotation_x_limit_exceeded": str(_limit_exceeded(rotations[t, 2], raw_rotations[t, 2]))
                    if rotations.shape[1] >= 3 and raw_rotations.shape[1] >= 3
                    else "",
                    "snr_before": _csv_value(snr_before[t]),
                    "cnr_before": _csv_value(cnr_before[t]),
                    "pearson_correlation_before": _csv_value(correlations_before[t]),
                    "pearson_correlation_after": _csv_value(correlations_after[t]),
                    "pearson_correlation": _csv_value(correlations_after[t]),
                }
            )

        intra_shifts = details.get("intra_stack_shifts_yx")
        if intra_shifts is None:
            return
        intra_shifts = np.asarray(intra_shifts, dtype=np.float32)
        if intra_shifts.ndim != 3 or intra_shifts.shape[2] != 2:
            return
        for t in range(intra_shifts.shape[0]):
            for z in range(intra_shifts.shape[1]):
                writer.writerow(
                    {
                        "scope": "intra_stack",
                        "frame": t,
                        "z": z,
                        "shift_z": "",
                        "shift_y": "",
                        "shift_x": "",
                        "shift_z_raw": "",
                        "shift_y_raw": "",
                        "shift_x_raw": "",
                        "shift_z_limit_exceeded": "",
                        "shift_y_limit_exceeded": "",
                        "shift_x_limit_exceeded": "",
                        "intra_shift_y": _csv_value(intra_shifts[t, z, 0]),
                        "intra_shift_x": _csv_value(intra_shifts[t, z, 1]),
                        "rotation_z_deg": "",
                        "rotation_y_deg": "",
                        "rotation_x_deg": "",
                        "rotation_deg": "",
                        "rotation_z_deg_raw": "",
                        "rotation_y_deg_raw": "",
                        "rotation_x_deg_raw": "",
                        "rotation_deg_raw": "",
                        "rotation_z_limit_exceeded": "",
                        "rotation_y_limit_exceeded": "",
                        "rotation_x_limit_exceeded": "",
                        "snr_before": _csv_value(snr_before[t]),
                        "cnr_before": _csv_value(cnr_before[t]),
                        "pearson_correlation_before": _csv_value(correlations_before[t]),
                        "pearson_correlation_after": _csv_value(correlations_after[t]),
                        "pearson_correlation": _csv_value(correlations_after[t]),
                    }
                )

def _projection_range_label(details: dict[str, Any], z_count: int) -> str:
    """Return a compact projection-range label for annotations."""

    if int(z_count) <= 1:
        return f"Z_N={int(z_count)}"
    projection_range = details.get("registration_z_range", details.get("projection_range"))
    if projection_range is None:
        return f"all slices (0:{z_count})"
    z_start, z_stop = normalize_zrange(projection_range, z_count, strict=True)
    return f"{z_start}:{z_stop}"

def _registration_range_label(details: dict[str, Any]) -> str:
    """Return a compact processing-range label for annotations."""

    registration_range = details.get("registration_range")
    ignored_reason = details.get("registration_range_ignored_reason")
    if registration_range is None:
        if ignored_reason:
            return "ignored"
        return "all"
    axis = details.get("registration_range_axis")
    start, stop = (int(registration_range[0]), int(registration_range[1]))
    return f"{axis}_{start}:{stop}" if axis else f"{start}:{stop}"

def _template_time_label(details: dict[str, Any], time_count: int) -> str:
    """Return a compact registration-template time label for annotations."""

    template_time_range = details.get("registration_template_time_range")
    if template_time_range is None:
        return f"t={int(details.get('registration_stack', 0))}"
    start, stop = (int(template_time_range[0]), int(template_time_range[1]))
    if start == 0 and stop == int(time_count):
        return f"all frames (0:{int(time_count)})"
    return f"{start}:{stop}"

def _format_max_abs(values: np.ndarray) -> str:
    """Format the maximum absolute finite value in a compact way."""

    values = np.asarray(values, dtype=np.float32)
    if values.size == 0 or not np.any(np.isfinite(values)):
        return "n/a"
    return f"{float(np.nanmax(np.abs(values))):.3g}"

def _format_max_xy_limits(max_xy) -> str:
    """Format ``(max_y, max_x)`` limits with explicit axis labels."""

    if max_xy is None:
        return "None"
    max_y, max_x = [float(v) for v in max_xy]
    return f"max_y={max_y:g}, max_x={max_x:g}"

def _raw_estimate_label(details: dict[str, Any], time_count: int) -> str:
    """Return compact max-raw-shift and rotation labels."""

    labels = []
    raw_shifts = _time_shifts_zyx_raw(details, time_count)
    if raw_shifts.shape == (time_count, 3):
        shift_parts = [
            f"y={_format_max_abs(raw_shifts[:, 1])}",
            f"x={_format_max_abs(raw_shifts[:, 2])}",
        ]
        if bool(details.get("zreg")) or np.any(np.abs(raw_shifts[:, 0]) > 0):
            shift_parts.insert(0, f"z={_format_max_abs(raw_shifts[:, 0])}")
        labels.append("max_raw_shift[" + ", ".join(shift_parts) + "]")

    raw_rotations, rotation_labels = _rotation_shift_series_raw_deg(details, time_count)
    if raw_rotations.shape[1] > 0:
        rotation_parts = [
            f"{label.replace('rotation_', '')}={_format_max_abs(raw_rotations[:, axis_index])}"
            for axis_index, label in enumerate(rotation_labels)
        ]
        labels.append("max_raw_rot_deg[" + ", ".join(rotation_parts) + "]")
    return " | ".join(labels) if labels else "max_raw=n/a"

def _settings_annotation(details: dict[str, Any], registered_stack: np.ndarray) -> str:
    """Build the compact plot annotation."""

    shape_before = details.get("stack_shape_tzcyx")
    if shape_before is None:
        shape_before = details.get("input_shape_tzcyx")
    shape_after_label = str(tuple(int(v) for v in registered_stack.shape))
    if shape_before is None:
        shape_before_label = shape_after_label
    else:
        shape_before_label = str(tuple(int(v) for v in shape_before))
    max_xy = details.get("max_xy_shifts")
    max_z = details.get("max_z_shifts")
    max_shifts = details.get("max_shifts")
    if max_xy is None and max_shifts is not None:
        max_shifts_list = list(max_shifts)
        max_xy = max_shifts_list[-2:] if len(max_shifts_list) >= 2 else None
    if max_z is None and max_shifts is not None and len(list(max_shifts)) == 3:
        max_z = list(max_shifts)[0]
    max_rot = details.get("max_rot_shifts")
    return "\n".join(
        [
            f"shape_TZCYX before registration={shape_before_label} | after={shape_after_label}",
            (
                f"method={details.get('method')} | "
                f"time={details.get('time_registration_mode')}"
                f"->{details.get('effective_time_registration_mode')} | "
                f"reference={details.get('time_reference_mode')}"
            ),
            (
                f"backend={details.get('transform_backend')} order={details.get('transform_order')} | "
                f"intra={details.get('intra_stack')} zreg={details.get('zreg')} "
                f"rotreg={details.get('rotreg')}"
            ),
            (
                f"template_t={_template_time_label(details, registered_stack.shape[0])} | "
                f"projection={details.get('projection_method')} | "
                f"registration_z_range={_projection_range_label(details, registered_stack.shape[1])} | "
                f"registration_range={_registration_range_label(details)}"
            ),
            _raw_estimate_label(details, registered_stack.shape[0]),
            f"{_format_max_xy_limits(max_xy)} | max_z={max_z} | max_rot={max_rot}",
        ]
    )

def _add_shift_limits(ax_shift, details: dict[str, Any]) -> None:
    """Draw configured shift-limit guide lines."""

    max_xy = details.get("max_xy_shifts")
    max_shifts = details.get("max_shifts")
    if max_xy is None and max_shifts is not None:
        max_shifts_list = list(max_shifts)
        max_xy = max_shifts_list[-2:] if len(max_shifts_list) >= 2 else None
    if max_xy is not None:
        max_y, max_x = [float(v) for v in max_xy]
        ax_shift.axhline(max_y, color="tab:blue", linestyle="--", linewidth=0.8, alpha=0.45)
        ax_shift.axhline(-max_y, color="tab:blue", linestyle="--", linewidth=0.8, alpha=0.45)
        ax_shift.axhline(max_x, color="tab:orange", linestyle="--", linewidth=0.8, alpha=0.45)
        ax_shift.axhline(-max_x, color="tab:orange", linestyle="--", linewidth=0.8, alpha=0.45)

    max_z = details.get("max_z_shifts")
    if max_z is None and max_shifts is not None and len(list(max_shifts)) == 3:
        max_z = list(max_shifts)[0]
    if max_z is not None:
        max_z = float(max_z)
        ax_shift.axhline(max_z, color="tab:green", linestyle="--", linewidth=0.8, alpha=0.45)
        ax_shift.axhline(-max_z, color="tab:green", linestyle="--", linewidth=0.8, alpha=0.45)

def _summary_marker_for_frame_count(frame_count: int):
    """Return sparse markers only for summary plots with manageable frame counts."""

    return "o" if int(frame_count) <= _SUMMARY_MARKER_FRAME_LIMIT else None

def _add_limit_exceeded_markers(ax, frames: np.ndarray, applied: np.ndarray, raw: np.ndarray, *, label: str) -> None:
    """Mark frames for which a raw estimate was clipped."""

    applied = np.asarray(applied, dtype=np.float32)
    raw = np.asarray(raw, dtype=np.float32)
    if applied.shape != raw.shape:
        return
    exceeded = np.isfinite(applied) & np.isfinite(raw) & (np.abs(applied - raw) > 1e-5)
    if not np.any(exceeded):
        return
    existing_labels = {legend_label for legend_label in ax.get_legend_handles_labels()[1]}
    marker_label = label if label not in existing_labels else "_nolegend_"
    if len(frames) > _SUMMARY_MARKER_FRAME_LIMIT:
        ax.plot(
            frames[exceeded],
            applied[exceeded],
            linestyle="none",
            marker=".",
            markersize=5,
            color="red",
            zorder=6,
            label=marker_label,
        )
    else:
        ax.scatter(
            frames[exceeded],
            applied[exceeded],
            s=52,
            facecolor="none",
            edgecolor="red",
            linewidth=1.5,
            zorder=6,
            label=marker_label,
        )

def _write_summary_plot(
    path: Path,
    registered_stack: np.ndarray,
    details: dict[str, Any],
    correlations_after: np.ndarray,
    correlations_before: np.ndarray,
    *,
    plot_SNR_log: bool = True,
) -> None:
    """Write the shift/correlation summary plot."""

    import matplotlib.pyplot as plt

    time_count = registered_stack.shape[0]
    frames = np.arange(time_count)
    line_marker = _summary_marker_for_frame_count(time_count)
    shifts_zyx = _time_shifts_zyx(details, time_count)
    raw_shifts_zyx = _time_shifts_zyx_raw(details, time_count)
    rotations, rotation_labels = _rotation_shift_series_deg(details, time_count)
    raw_rotations, _ = _rotation_shift_series_raw_deg(details, time_count)

    fig = plt.figure(figsize=(11, 7.2), constrained_layout=True)
    grid = fig.add_gridspec(3, 1, height_ratios=[2.0, 1.2, 0.42])
    ax_shift = fig.add_subplot(grid[0])
    ax_corr = fig.add_subplot(grid[1], sharex=ax_shift)
    ax_note = fig.add_subplot(grid[2])
    ax_shift.tick_params(labelbottom=False)
    ax_shift.plot(frames, shifts_zyx[:, 1], marker=line_marker, label="shift_y", color="tab:blue")
    ax_shift.plot(frames, shifts_zyx[:, 2], marker=line_marker, label="shift_x", color="tab:orange")
    _add_limit_exceeded_markers(ax_shift, frames, shifts_zyx[:, 1], raw_shifts_zyx[:, 1], label="limit exceeded")
    _add_limit_exceeded_markers(ax_shift, frames, shifts_zyx[:, 2], raw_shifts_zyx[:, 2], label="limit exceeded")
    if bool(details.get("zreg")) or np.any(np.abs(shifts_zyx[:, 0]) > 0):
        ax_shift.plot(frames, shifts_zyx[:, 0], marker=line_marker, label="shift_z", color="tab:green")
        _add_limit_exceeded_markers(ax_shift, frames, shifts_zyx[:, 0], raw_shifts_zyx[:, 0], label="limit exceeded")
    _add_shift_limits(ax_shift, details)
    ax_shift.set_ylabel("Detected correction shift [px]")
    ax_shift.grid(True, alpha=0.25)
    ax_shift.legend(loc="upper left", ncols=3, fontsize=8)

    if rotations.shape[1] > 0 and np.any(np.isfinite(rotations)):
        ax_rot = ax_shift.twinx()
        rotation_colors = ("tab:red", "tab:pink", "tab:brown")
        for axis_index, label in enumerate(rotation_labels):
            ax_rot.plot(
                frames,
                rotations[:, axis_index],
                marker=line_marker,
                label=label,
                color=rotation_colors[axis_index % len(rotation_colors)],
                alpha=0.75,
            )
            if raw_rotations.shape[1] > axis_index:
                _add_limit_exceeded_markers(
                    ax_rot,
                    frames,
                    rotations[:, axis_index],
                    raw_rotations[:, axis_index],
                    label="limit exceeded",
                )
        max_rot = details.get("max_rot_shifts")
        if max_rot is not None:
            max_rot = float(max_rot)
            ax_rot.axhline(max_rot, color="tab:red", linestyle="--", linewidth=0.8, alpha=0.4)
            ax_rot.axhline(-max_rot, color="tab:red", linestyle="--", linewidth=0.8, alpha=0.4)
        ax_rot.set_ylabel("Rotation correction [deg]", color="tab:red")
        ax_rot.tick_params(axis="y", labelcolor="tab:red")
        ax_rot.legend(loc="upper right", fontsize=8)

    if np.any(np.isfinite(correlations_before)):
        ax_corr.plot(
            frames,
            correlations_before,
            marker=line_marker,
            color="0.45",
            alpha=0.75,
            label="r before",
        )
    ax_corr.plot(frames, correlations_after, marker=line_marker, color="tab:purple", label="r after")
    ax_corr.set_ylabel("Pearson r vs template")
    ax_corr.set_xlabel("Frame")
    ax_corr.set_ylim(-1.05, 1.05)
    ax_corr.grid(True, alpha=0.25)

    snr_before = _quality_series(details, "snr_before", time_count)
    cnr_before = _quality_series(details, "cnr_before", time_count)
    has_snr = np.any(np.isfinite(snr_before))
    has_cnr = np.any(np.isfinite(cnr_before))
    if has_snr or has_cnr:
        if has_snr:
            snr_plot_values, snr_ylabel, snr_label = _snr_plot_series(
                snr_before,
                plot_SNR_log=bool(plot_SNR_log),
            )
            ax_snr = ax_corr.twinx()
            ax_snr.plot(
                frames,
                snr_plot_values,
                marker=line_marker,
                color="tab:green",
                alpha=0.75,
                linewidth=1.2,
                label=snr_label,
            )
            ax_snr.set_ylabel(snr_ylabel, color="tab:green")
            ax_snr.tick_params(axis="y", labelcolor="tab:green")
            ax_snr.set_ylim(bottom=0)
            ax_snr.legend(loc="upper right", fontsize=8)
        if has_cnr:
            ax_cnr = ax_corr.twinx()
            if has_snr:
                ax_cnr.spines["right"].set_position(("axes", 1.12))
                ax_cnr.spines["right"].set_visible(True)
            ax_cnr.plot(
                frames,
                cnr_before,
                marker=line_marker,
                color="tab:brown",
                alpha=0.75,
                linewidth=1.2,
                label="CNR",
            )
            ax_cnr.set_ylabel("CNR", color="tab:brown")
            ax_cnr.tick_params(axis="y", labelcolor="tab:brown")
            ax_cnr.set_ylim(bottom=0)
            ax_cnr.legend(loc="center right" if has_snr else "upper right", fontsize=8)
        ax_corr.legend(loc="lower left", fontsize=8)
    else:
        ax_corr.legend(loc="lower right", fontsize=8)

    annotation = _settings_annotation(details, registered_stack)
    ax_note.axis("off")
    ax_note.text(
        0.0,
        1.0,
        annotation,
        transform=ax_note.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        family="monospace",
    )
    fig.suptitle("ZenReg registration report", fontsize=12)
    fig.savefig(path, dpi=180)
    plt.close(fig)

def _settings_payload(
    *,
    output_image_path: Path,
    registered_stack: np.ndarray,
    details: dict[str, Any],
    correlations_after: np.ndarray,
    correlations_before: np.ndarray,
    report_paths: dict[str, Path],
) -> dict[str, Any]:
    """Build the YAML settings payload."""

    settings = {key: _plain_value(details[key]) for key in SETTING_KEYS if key in details}
    template_time_range = details.get("registration_template_time_range")
    snr_before = _quality_series(details, "snr_before", registered_stack.shape[0])
    cnr_before = _quality_series(details, "cnr_before", registered_stack.shape[0])
    return {
        "zenreg_report": {
            "output_image": str(output_image_path),
            "axes": CANONICAL_AXIS_ORDER,
            "registered_shape_tzcyx": tuple(int(v) for v in registered_stack.shape),
            "correlation_reference_frame": int(details.get("registration_stack", 0)),
            "correlation_template_time_range": _plain_value(template_time_range),
            "correlation_mean": _nan_stat(correlations_after, np.nanmean),
            "correlation_min": _nan_stat(correlations_after, np.nanmin),
            "correlation_after_mean": _nan_stat(correlations_after, np.nanmean),
            "correlation_after_min": _nan_stat(correlations_after, np.nanmin),
            "correlation_before_mean": _nan_stat(correlations_before, np.nanmean),
            "correlation_before_min": _nan_stat(correlations_before, np.nanmin),
            "snr_before_mean": _nan_stat(snr_before, np.nanmean),
            "snr_before_min": _nan_stat(snr_before, np.nanmin),
            "cnr_before_mean": _nan_stat(cnr_before, np.nanmean),
            "cnr_before_min": _nan_stat(cnr_before, np.nanmin),
            "csv": str(report_paths["csv"]),
            "plot": str(report_paths["plot"]),
        },
        "registration_settings": settings,
    }

def write_registration_summary_plot(
    path: str | Path,
    registered_stack,
    registration_details: dict[str, Any] | np.ndarray,
    *,
    plot_SNR_log: bool = True,
) -> Path:
    """
    Write only the ZenReg registration summary plot.

    This is a lightweight preview helper for checking registration results after
    ``register_stack(...)`` and before committing time to saving a large
    registered image. It uses the same summary-plot implementation as
    ``save_stack(..., registration_details=...)`` but does not write an OME-TIFF,
    CSV table, or YAML settings sidecar.

    Parameters
    ----------
    path : str or pathlib.Path
        Output ``.png`` path for the summary plot.
    registered_stack : array-like
        Registered image stack in canonical ``TZCYX`` order.
    registration_details : dict or array-like
        Details returned by ``register_stack(..., return_shifts=True,
        return_details=True)``. A legacy ``T, 2`` shift array is accepted, but a
        full details dictionary gives richer annotations.
    plot_SNR_log : bool, optional
        If True, plot SNR as ``log10(SNR)`` in the summary figure while keeping
        raw ``snr_before`` values unchanged in details/CSV/YAML outputs.
        Default: ``True``.

    Returns
    -------
    pathlib.Path
        Path of the written summary plot.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    registered_stack = ensure_tzcyx_stack(registered_stack)
    details = _as_details_dict(registration_details)
    correlations_after = _frame_correlations(registered_stack, details)
    correlations_before = _pre_frame_correlations(details, registered_stack.shape[0])
    _write_summary_plot(
        path,
        registered_stack,
        details,
        correlations_after,
        correlations_before,
        plot_SNR_log=bool(plot_SNR_log),
    )
    return path

def write_registration_outputs(
    output_image_path: str | Path,
    registered_stack,
    registration_details: dict[str, Any] | np.ndarray,
    *,
    report_prefix: str | Path | None = None,
    plot_SNR_log: bool = True,
) -> dict[str, Path]:
    """
    Write ZenReg CSV/YAML/PNG report files next to a registered image.

    Parameters
    ----------
    output_image_path : str or pathlib.Path
        Registered image path used as anchor for report file names.
    registered_stack : array-like
        Registered image stack in canonical ``TZCYX`` order.
    registration_details : dict or array-like
        Details returned by ``register_stack(..., return_shifts=True,
        return_details=True)``. A legacy ``T, 2`` shift array is accepted for
        CSV-only-style compatibility, but settings annotations are richer with a
        full details dictionary.
    report_prefix : str, pathlib.Path, or None, optional
        Optional path prefix. By default ``image.ome.tif`` produces
        ``image_registration_shifts.csv``, ``image_registration_settings.yaml``,
        and ``image_registration_summary.png``.
    plot_SNR_log : bool, optional
        If True, plot SNR as ``log10(SNR)`` in the PNG summary while preserving
        raw SNR values in CSV and YAML sidecars. Default: ``True``.

    Returns
    -------
    dict[str, pathlib.Path]
        Paths for ``csv``, ``yaml``, and ``plot``.
    """

    output_image_path = Path(output_image_path)
    registered_stack = ensure_tzcyx_stack(registered_stack)
    details = _as_details_dict(registration_details)
    prefix = _report_prefix(output_image_path, report_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    report_paths = {
        "csv": prefix.with_name(prefix.name + "_registration_shifts.csv"),
        "yaml": prefix.with_name(prefix.name + "_registration_settings.yaml"),
        "plot": prefix.with_name(prefix.name + "_registration_summary.png"),
    }

    correlations_after = _frame_correlations(registered_stack, details)
    correlations_before = _pre_frame_correlations(details, registered_stack.shape[0])
    _write_shift_csv(
        report_paths["csv"],
        details,
        correlations_after,
        correlations_before,
        time_count=registered_stack.shape[0],
    )
    _write_summary_plot(
        report_paths["plot"],
        registered_stack,
        details,
        correlations_after,
        correlations_before,
        plot_SNR_log=bool(plot_SNR_log),
    )
    _write_yaml(
        report_paths["yaml"],
        _settings_payload(
            output_image_path=output_image_path,
            registered_stack=registered_stack,
            details=details,
            correlations_after=correlations_after,
            correlations_before=correlations_before,
            report_paths=report_paths,
        ),
    )
    return report_paths
# %% END
