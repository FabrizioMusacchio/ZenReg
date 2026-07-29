"""
Registration report writers for ZenReg outputs.

Author: Fabrizio Musacchio
Date: July 2026
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from ._axes import CANONICAL_AXIS_ORDER, ensure_tzcyx_stack, normalize_zrange
from .registration import _project_zyx_to_yx

SETTING_KEYS = (
    "registration_channel",
    "registration_stack",
    "method",
    "time_registration_mode",
    "effective_time_registration_mode",
    "time_reference_mode",
    "intra_stack",
    "zreg",
    "rotreg",
    "rotreg_iter",
    "projection_range",
    "projection_method",
    "filter_slices",
    "filter_projections",
    "median_kernel_size",
    "max_xy_shifts",
    "max_z_shifts",
    "max_rot_shifts",
    "max_shifts",
    "max_deviation_rigid",
    "strides",
    "overlaps",
    "patch_grid_shape",
    "upsample_factor",
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
    "zero_clip_margin_zyx",
    "zero_clip_bounds",
    "stack_shape_tzcyx",
)


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


def _rotation_shifts_deg(details: dict[str, Any], time_count: int) -> np.ndarray:
    """Return per-frame rotation corrections with missing values as NaN."""

    rotations = details.get("rotation_shifts_deg")
    if rotations is None:
        return np.full(time_count, np.nan, dtype=np.float32)
    rotations = np.asarray(rotations, dtype=np.float32)
    if rotations.shape != (time_count,):
        return np.full(time_count, np.nan, dtype=np.float32)
    return rotations


def _pearson_correlation(template: np.ndarray, image: np.ndarray) -> float:
    """Compute Pearson correlation robustly for flattened image data."""

    template = np.asarray(template, dtype=np.float64).ravel()
    image = np.asarray(image, dtype=np.float64).ravel()
    template = template - np.mean(template)
    image = image - np.mean(image)
    denominator = float(np.linalg.norm(template) * np.linalg.norm(image))
    if denominator == 0.0:
        return float("nan")
    return float(np.dot(template, image) / denominator)


def _registration_images_for_correlation(
    registered_stack: np.ndarray,
    details: dict[str, Any],
) -> tuple[np.ndarray, str]:
    """Extract per-frame registration images used for Pearson reporting."""

    channel = int(details.get("registration_channel", 0))
    projection_range = details.get("projection_range")
    z_start, z_stop = normalize_zrange(projection_range, registered_stack.shape[1], strict=True)
    volumes = np.asarray(registered_stack[:, z_start:z_stop, channel, :, :], dtype=np.float32)

    if details.get("effective_time_registration_mode") == "full_3d":
        return volumes.reshape(volumes.shape[0], -1), f"full_3d z={z_start}:{z_stop}"

    projection_method = str(details.get("projection_method", "max"))
    projections = np.empty((volumes.shape[0], volumes.shape[2], volumes.shape[3]), dtype=np.float32)
    for t in range(volumes.shape[0]):
        projections[t, :, :] = _project_zyx_to_yx(
            volumes[t, :, :, :],
            projection_method=projection_method,
        )
    return projections.reshape(projections.shape[0], -1), f"{projection_method} projection z={z_start}:{z_stop}"


def _frame_correlations(registered_stack: np.ndarray, details: dict[str, Any]) -> np.ndarray:
    """Compute template-vs-registered Pearson correlations per time frame."""

    series, _ = _registration_images_for_correlation(registered_stack, details)
    registration_stack = int(details.get("registration_stack", 0))
    registration_stack = int(np.clip(registration_stack, 0, series.shape[0] - 1))
    template = series[registration_stack, :]
    return np.asarray([_pearson_correlation(template, series[t, :]) for t in range(series.shape[0])])


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


def _write_shift_csv(
    path: Path,
    details: dict[str, Any],
    correlations: np.ndarray,
    *,
    time_count: int,
) -> None:
    """Write frame-wise and optional intra-stack shifts to CSV."""

    shifts_zyx = _time_shifts_zyx(details, time_count)
    rotations = _rotation_shifts_deg(details, time_count)
    fieldnames = [
        "scope",
        "frame",
        "z",
        "shift_z",
        "shift_y",
        "shift_x",
        "intra_shift_y",
        "intra_shift_x",
        "rotation_deg",
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
                    "intra_shift_y": "",
                    "intra_shift_x": "",
                    "rotation_deg": _csv_value(rotations[t]),
                    "pearson_correlation": _csv_value(correlations[t]),
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
                        "intra_shift_y": _csv_value(intra_shifts[t, z, 0]),
                        "intra_shift_x": _csv_value(intra_shifts[t, z, 1]),
                        "rotation_deg": "",
                        "pearson_correlation": _csv_value(correlations[t]),
                    }
                )


def _projection_range_label(details: dict[str, Any], z_count: int) -> str:
    """Return a compact projection-range label for annotations."""

    projection_range = details.get("projection_range")
    if projection_range is None:
        return f"all slices (0:{z_count})"
    z_start, z_stop = normalize_zrange(projection_range, z_count, strict=True)
    return f"{z_start}:{z_stop}"


def _settings_annotation(details: dict[str, Any], registered_stack: np.ndarray) -> str:
    """Build the compact plot annotation."""

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
                f"projection={details.get('projection_method')} | "
                f"projection_range={_projection_range_label(details, registered_stack.shape[1])}"
            ),
            f"max_xy={max_xy} | max_z={max_z} | max_rot={max_rot}",
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


def _write_summary_plot(
    path: Path,
    registered_stack: np.ndarray,
    details: dict[str, Any],
    correlations: np.ndarray,
) -> None:
    """Write the shift/correlation summary plot."""

    import matplotlib.pyplot as plt

    time_count = registered_stack.shape[0]
    frames = np.arange(time_count)
    shifts_zyx = _time_shifts_zyx(details, time_count)
    rotations = _rotation_shifts_deg(details, time_count)

    fig, (ax_shift, ax_corr) = plt.subplots(
        2,
        1,
        figsize=(9, 6.5),
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.2]},
    )
    ax_shift.plot(frames, shifts_zyx[:, 1], marker="o", label="shift_y", color="tab:blue")
    ax_shift.plot(frames, shifts_zyx[:, 2], marker="o", label="shift_x", color="tab:orange")
    if bool(details.get("zreg")) or np.any(np.abs(shifts_zyx[:, 0]) > 0):
        ax_shift.plot(frames, shifts_zyx[:, 0], marker="o", label="shift_z", color="tab:green")
    _add_shift_limits(ax_shift, details)
    ax_shift.set_ylabel("Detected correction shift [px]")
    ax_shift.grid(True, alpha=0.25)
    ax_shift.legend(loc="upper left", ncols=3, fontsize=8)

    if np.any(np.isfinite(rotations)):
        ax_rot = ax_shift.twinx()
        ax_rot.plot(frames, rotations, marker="s", label="rotation", color="tab:red", alpha=0.85)
        max_rot = details.get("max_rot_shifts")
        if max_rot is not None:
            max_rot = float(max_rot)
            ax_rot.axhline(max_rot, color="tab:red", linestyle="--", linewidth=0.8, alpha=0.4)
            ax_rot.axhline(-max_rot, color="tab:red", linestyle="--", linewidth=0.8, alpha=0.4)
        ax_rot.set_ylabel("Rotation correction [deg]", color="tab:red")
        ax_rot.tick_params(axis="y", labelcolor="tab:red")
        ax_rot.legend(loc="upper right", fontsize=8)

    ax_corr.plot(frames, correlations, marker="o", color="tab:purple")
    ax_corr.set_ylabel("Pearson r")
    ax_corr.set_xlabel("Frame")
    ax_corr.set_ylim(-1.05, 1.05)
    ax_corr.grid(True, alpha=0.25)

    annotation = _settings_annotation(details, registered_stack)
    fig.text(
        0.01,
        0.01,
        annotation,
        ha="left",
        va="bottom",
        fontsize=8,
        family="monospace",
    )
    fig.suptitle("ZenReg registration report", fontsize=12)
    fig.tight_layout(rect=(0, 0.16, 1, 0.95))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _settings_payload(
    *,
    output_image_path: Path,
    registered_stack: np.ndarray,
    details: dict[str, Any],
    correlations: np.ndarray,
    report_paths: dict[str, Path],
) -> dict[str, Any]:
    """Build the YAML settings payload."""

    settings = {key: _plain_value(details[key]) for key in SETTING_KEYS if key in details}
    return {
        "zenreg_report": {
            "output_image": str(output_image_path),
            "axes": CANONICAL_AXIS_ORDER,
            "registered_shape_tzcyx": tuple(int(v) for v in registered_stack.shape),
            "correlation_reference_frame": int(details.get("registration_stack", 0)),
            "correlation_mean": float(np.nanmean(correlations)),
            "correlation_min": float(np.nanmin(correlations)),
            "csv": str(report_paths["csv"]),
            "plot": str(report_paths["plot"]),
        },
        "registration_settings": settings,
    }


def write_registration_outputs(
    output_image_path: str | Path,
    registered_stack,
    registration_details: dict[str, Any] | np.ndarray,
    *,
    report_prefix: str | Path | None = None,
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

    correlations = _frame_correlations(registered_stack, details)
    _write_shift_csv(
        report_paths["csv"],
        details,
        correlations,
        time_count=registered_stack.shape[0],
    )
    _write_summary_plot(report_paths["plot"], registered_stack, details, correlations)
    _write_yaml(
        report_paths["yaml"],
        _settings_payload(
            output_image_path=output_image_path,
            registered_stack=registered_stack,
            details=details,
            correlations=correlations,
            report_paths=report_paths,
        ),
    )
    return report_paths
