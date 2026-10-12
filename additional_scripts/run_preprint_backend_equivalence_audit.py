"""Run direct-backend equivalence checks for the ZenReg revision.

This script is intended for the Biology Open revision response. It compares
the shifts reported by ZenReg's public ``register_stack`` workflow with direct
backend-level calls on the same synthetic inputs and references.

Outputs are written to ``papers/preprint/backend_equivalence``:

    conda run -n zenreg2 python additional_scripts/run_preprint_backend_equivalence_audit.py
"""
# %% IMPORTS
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
from skimage.registration import phase_cross_correlation

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from zenreg import register_stack, z_project
from zenreg.rigid3d import (
    Rigid3DTransform,
    _phase_correlation_shift_zyx,
    _register_simpleitk,
)
from zenreg.synthetic import (
    create_2d_motion_distorted_stack,
    create_3d_time_rigid_motion_distorted_stack,
    create_3d_time_zyx_motion_distorted_stack,
)

# %% PATHS AND SETTINGS
OUTPUT_DIR = PROJECT_ROOT / "papers" / "preprint" / "backend_equivalence"
UPSAMPLE_FACTOR = 20
NORMALIZATION = None


# %% HELPERS
def _write_rows(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _summarize(
    case: str,
    backend: str,
    axes: list[str],
    zenreg_values: np.ndarray,
    direct_values: np.ndarray,
    runtime_s: float,
    *,
    unit: str,
) -> dict:
    zenreg_values = np.asarray(zenreg_values, dtype=np.float64)
    direct_values = np.asarray(direct_values, dtype=np.float64)
    difference = zenreg_values - direct_values
    row = {
        "case": case,
        "backend": backend,
        "axes": ",".join(axes),
        "unit": unit,
        "n_frames": int(zenreg_values.shape[0]),
        "mean_abs_difference": float(np.mean(np.abs(difference))),
        "max_abs_difference": float(np.max(np.abs(difference))),
        "runtime_s": float(runtime_s),
    }
    for axis_index, axis_name in enumerate(axes):
        row[f"mean_abs_difference_{axis_name}"] = float(np.mean(np.abs(difference[:, axis_index])))
        row[f"max_abs_difference_{axis_name}"] = float(np.max(np.abs(difference[:, axis_index])))
    return row


def _per_frame_rows(case: str, backend: str, axes: list[str], zenreg_values: np.ndarray, direct_values: np.ndarray, *, unit: str) -> list[dict]:
    rows = []
    difference = np.asarray(zenreg_values, dtype=np.float64) - np.asarray(direct_values, dtype=np.float64)
    for frame in range(difference.shape[0]):
        row = {
            "case": case,
            "backend": backend,
            "frame": int(frame),
            "unit": unit,
            "mean_abs_difference": float(np.mean(np.abs(difference[frame]))),
            "max_abs_difference": float(np.max(np.abs(difference[frame]))),
        }
        for axis_index, axis_name in enumerate(axes):
            row[f"zenreg_{axis_name}"] = float(zenreg_values[frame, axis_index])
            row[f"direct_{axis_name}"] = float(direct_values[frame, axis_index])
            row[f"difference_{axis_name}"] = float(difference[frame, axis_index])
        rows.append(row)
    return rows


def _write_markdown_summary(path: Path, summary_rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Backend Equivalence Audit",
        "",
        "ZenReg outputs were compared with direct backend-level calls on the same synthetic inputs and fixed references.",
        "",
        "| Case | Backend-level comparison | Axes | Unit | Frames | Mean abs. difference | Max abs. difference |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {case} | {backend} | {axes} | {unit} | {n_frames} | {mean:.6g} | {maxv:.6g} |".format(
                case=row["case"],
                backend=row["backend"],
                axes=row["axes"],
                unit=row["unit"],
                n_frames=row["n_frames"],
                mean=row["mean_abs_difference"],
                maxv=row["max_abs_difference"],
            )
        )
    lines.extend(
        [
            "",
            "Notes:",
            "- The 2D+t and 3D+t translation checks compare ZenReg's reported phase-correlation shifts with direct `skimage.registration.phase_cross_correlation` calls.",
            "- The SimpleITK check compares ZenReg's public 3D rigid workflow with its backend-level SimpleITK rigid refinement on the same fixed/moving volumes and initialization.",
            "- These checks audit wrapper/reference/axis handling; they do not claim a new registration model.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _direct_phase_2d_shifts(stack: np.ndarray) -> np.ndarray:
    reference = z_project(stack[0:1, :, 0:1, :, :], projection_method="max")[0, 0, 0]
    shifts = []
    for frame in range(stack.shape[0]):
        moving = z_project(stack[frame : frame + 1, :, 0:1, :, :], projection_method="max")[0, 0, 0]
        shift, _, _ = phase_cross_correlation(
            reference,
            moving,
            upsample_factor=UPSAMPLE_FACTOR,
            normalization=NORMALIZATION,
        )
        shifts.append(np.asarray(shift, dtype=np.float32))
    return np.stack(shifts, axis=0)


def _direct_phase_3d_shifts(stack: np.ndarray) -> np.ndarray:
    reference = np.asarray(stack[0, :, 0, :, :], dtype=np.float32)
    shifts = []
    for frame in range(stack.shape[0]):
        moving = np.asarray(stack[frame, :, 0, :, :], dtype=np.float32)
        shift, _, _ = phase_cross_correlation(
            reference,
            moving,
            upsample_factor=UPSAMPLE_FACTOR,
            normalization=NORMALIZATION,
        )
        shifts.append(np.asarray(shift, dtype=np.float32))
    return np.stack(shifts, axis=0)


def _run_phase_2d() -> tuple[dict, list[dict]]:
    stack, _ = create_2d_motion_distorted_stack(
        time_count=12,
        channel_count=2,
        shape_yx=(128, 128),
        noise_sigma=0.0,
        random_state=17,
        shift_amplitude_y=5.0,
        shift_amplitude_x=4.0,
    )
    start = time.perf_counter()
    _, details = register_stack(
        stack,
        registration_channel=0,
        registration_stack=0,
        method="phase_cross_correlation",
        time_registration_mode="projection",
        projection_method="max",
        zreg=False,
        max_xy_shifts=None,
        phase_cross_correlation_upsample_factor=UPSAMPLE_FACTOR,
        phase_cross_correlation_normalization=NORMALIZATION,
        transform_backend="skimage",
        transform_order=1,
        verbose=False,
        return_shifts=True,
        return_details=True,
    )
    zenreg_values = np.asarray(details["time_shifts_yx"], dtype=np.float32)
    direct_values = _direct_phase_2d_shifts(stack)
    runtime_s = time.perf_counter() - start
    axes = ["y", "x"]
    return (
        _summarize("2D+t projection translation", "direct scikit-image phase_cross_correlation", axes, zenreg_values, direct_values, runtime_s, unit="px"),
        _per_frame_rows("2D+t projection translation", "direct scikit-image phase_cross_correlation", axes, zenreg_values, direct_values, unit="px"),
    )


def _run_phase_3d() -> tuple[dict, list[dict]]:
    stack, _ = create_3d_time_zyx_motion_distorted_stack(
        time_count=8,
        z_count=16,
        channel_count=2,
        shape_yx=(80, 80),
        noise_sigma=0.0,
        random_state=19,
    )
    start = time.perf_counter()
    _, details = register_stack(
        stack,
        registration_channel=0,
        registration_stack=0,
        method="phase_cross_correlation",
        time_registration_mode="full_3d",
        projection_method="max",
        zreg=True,
        rotreg=False,
        max_xy_shifts=None,
        max_z_shifts=None,
        phase_cross_correlation_upsample_factor=UPSAMPLE_FACTOR,
        phase_cross_correlation_normalization=NORMALIZATION,
        transform_backend="skimage",
        transform_order=1,
        verbose=False,
        return_shifts=True,
        return_details=True,
    )
    zenreg_values = np.asarray(details["time_shifts_zyx"], dtype=np.float32)
    direct_values = _direct_phase_3d_shifts(stack)
    runtime_s = time.perf_counter() - start
    axes = ["z", "y", "x"]
    return (
        _summarize("3D+t full-volume translation", "direct scikit-image phase_cross_correlation", axes, zenreg_values, direct_values, runtime_s, unit="px"),
        _per_frame_rows("3D+t full-volume translation", "direct scikit-image phase_cross_correlation", axes, zenreg_values, direct_values, unit="px"),
    )


def _run_simpleitk_dispatch() -> tuple[dict, list[dict]]:
    stack, _, _, _ = create_3d_time_rigid_motion_distorted_stack(
        time_count=5,
        z_count=14,
        channel_count=2,
        shape_yx=(64, 64),
        rotation_mode="z",
        center_mode="middle",
        noise_sigma=0.0,
        random_state=23,
    )
    fixed = np.asarray(stack[0, :, 0, :, :], dtype=np.float32)
    center = (np.asarray(fixed.shape, dtype=np.float64) - 1.0) / 2.0
    start = time.perf_counter()
    _, details = register_stack(
        stack,
        registration_channel=0,
        registration_stack=0,
        method="phase_cross_correlation",
        time_registration_mode="full_3d",
        zreg=True,
        rotreg=True,
        rigid_3d_backend="simpleitk",
        rot_init_iterations=0,
        rot_metric="correlation",
        rot_shrink_factors=(2, 1),
        rot_smoothing_sigmas=(1.0, 0.0),
        rot_iterations=50,
        phase_cross_correlation_upsample_factor=UPSAMPLE_FACTOR,
        phase_cross_correlation_normalization=NORMALIZATION,
        transform_order=1,
        zero_clip=False,
        verbose=False,
        return_shifts=True,
        return_details=True,
    )
    zenreg_shift = np.asarray(details["time_shifts_zyx"], dtype=np.float32)
    zenreg_rotation = np.asarray(details["rotation_shifts_zyx_deg"], dtype=np.float32)

    direct_shift = np.zeros_like(zenreg_shift)
    direct_rotation = np.zeros_like(zenreg_rotation)
    for frame in range(stack.shape[0]):
        moving = np.asarray(stack[frame, :, 0, :, :], dtype=np.float32)
        if frame == 0:
            continue
        initial_shift = _phase_correlation_shift_zyx(
            fixed,
            moving,
            upsample_factor=UPSAMPLE_FACTOR,
            normalization=NORMALIZATION,
        )
        initial_transform = Rigid3DTransform(
            matrix_zyx=np.eye(3, dtype=np.float64),
            translation_zyx=np.asarray(initial_shift, dtype=np.float64),
            center_zyx=center,
        )
        transform = _register_simpleitk(
            fixed,
            moving,
            initial_transform=initial_transform,
            spacing_zyx=(1.0, 1.0, 1.0),
            metric="correlation",
            shrink_factors=(2, 1),
            smoothing_sigmas=(1.0, 0.0),
            iterations=50,
            learning_rate=1.0,
            min_step=1e-4,
            sampling_percentage=None,
        )
        direct_shift[frame] = np.asarray(transform.translation_zyx, dtype=np.float32)
        # The synthetic motion uses a Z-dominant rigid rotation; Euler extraction
        # is intentionally kept consistent with the ZenReg details convention.
        from scipy.spatial.transform import Rotation

        direct_rotation[frame] = Rotation.from_matrix(transform.matrix_zyx).as_euler("ZYX", degrees=True).astype(np.float32)

    runtime_s = time.perf_counter() - start
    shift_axes = ["z", "y", "x"]
    rotation_axes = ["rot_z", "rot_y", "rot_x"]
    summary_rows = [
        _summarize(
            "3D rigid SimpleITK dispatch: translation",
            "backend-level SimpleITK rigid refinement",
            shift_axes,
            zenreg_shift,
            direct_shift,
            runtime_s,
            unit="px",
        ),
        _summarize(
            "3D rigid SimpleITK dispatch: rotation",
            "backend-level SimpleITK rigid refinement",
            rotation_axes,
            zenreg_rotation,
            direct_rotation,
            runtime_s,
            unit="degrees",
        ),
    ]
    rows = [
        *_per_frame_rows(
            "3D rigid SimpleITK dispatch: translation",
            "backend-level SimpleITK rigid refinement",
            shift_axes,
            zenreg_shift,
            direct_shift,
            unit="px",
        ),
        *_per_frame_rows(
            "3D rigid SimpleITK dispatch: rotation",
            "backend-level SimpleITK rigid refinement",
            rotation_axes,
            zenreg_rotation,
            direct_rotation,
            unit="degrees",
        ),
    ]
    return summary_rows, rows


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    per_frame_rows = []

    for runner in (_run_phase_2d, _run_phase_3d, _run_simpleitk_dispatch):
        try:
            summary, rows = runner()
        except Exception as exc:
            summary = {
                "case": runner.__name__.replace("_run_", ""),
                "backend": "not completed",
                "axes": "",
                "unit": "",
                "n_frames": 0,
                "mean_abs_difference": float("nan"),
                "max_abs_difference": float("nan"),
                "runtime_s": float("nan"),
                "error": f"{type(exc).__name__}: {exc}",
            }
            rows = []
        if isinstance(summary, list):
            summary_rows.extend(summary)
        else:
            summary_rows.append(summary)
        per_frame_rows.extend(rows)

    _write_rows(OUTPUT_DIR / "backend_equivalence_summary.csv", summary_rows)
    _write_rows(OUTPUT_DIR / "backend_equivalence_per_frame.csv", per_frame_rows)
    _write_markdown_summary(OUTPUT_DIR / "backend_equivalence_summary.md", summary_rows)
    (OUTPUT_DIR / "backend_equivalence_summary.json").write_text(
        json.dumps(summary_rows, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote backend equivalence audit to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
