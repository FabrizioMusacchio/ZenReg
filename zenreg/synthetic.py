"""
Synthetic motion-distorted example data for ZenReg.

Author: Fabrizio Musacchio
Date: June 2026
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import shift as ndi_shift

from .io import save_stack


def _gaussian_blob_grid(shape_yx: tuple[int, int], centers, sigmas) -> np.ndarray:
    """Create a sum of anisotropic Gaussian blobs."""

    yy, xx = np.indices(shape_yx, dtype=np.float32)
    image = np.zeros(shape_yx, dtype=np.float32)
    for (cy, cx), (sy, sx), amplitude in zip(centers, sigmas, np.linspace(0.7, 1.2, len(centers))):
        image += amplitude * np.exp(-(((yy - cy) ** 2) / (2 * sy**2) + ((xx - cx) ** 2) / (2 * sx**2)))
    image -= float(image.min())
    image /= max(float(image.max()), 1e-6)
    return image.astype(np.float32)


def _apply_yx_shift(image: np.ndarray, shift_yx: tuple[float, float]) -> np.ndarray:
    """Apply a 2D translation to one image plane."""

    return ndi_shift(
        image.astype(np.float32, copy=False),
        shift=shift_yx,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=True,
    ).astype(np.float32, copy=False)


def create_2d_motion_distorted_stack(
    *,
    time_count: int = 12,
    channel_count: int = 2,
    shape_yx: tuple[int, int] = (128, 128),
    noise_sigma: float = 0.035,
    random_state: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a two-channel ``TZCYX`` stack with time-wise 2D translation artifacts.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Motion-distorted stack and ground-truth ``T, 2`` shifts applied to each
        time point.
    """

    rng = np.random.default_rng(random_state)
    base0 = _gaussian_blob_grid(
        shape_yx,
        centers=[(34, 40), (70, 58), (94, 96), (42, 92)],
        sigmas=[(7, 10), (9, 7), (8, 9), (6, 6)],
    )
    base1 = _gaussian_blob_grid(
        shape_yx,
        centers=[(52, 28), (84, 44), (72, 102), (106, 72)],
        sigmas=[(8, 8), (6, 11), (9, 8), (7, 7)],
    )
    base_channels = [base0, base1]
    while len(base_channels) < channel_count:
        base_channels.append(rng.random(shape_yx, dtype=np.float32) * 0.2)

    stack = np.zeros((time_count, 1, channel_count, *shape_yx), dtype=np.float32)
    shifts = np.zeros((time_count, 2), dtype=np.float32)

    for t in range(time_count):
        shift_y = 4.0 * np.sin(2 * np.pi * t / max(time_count - 1, 1))
        shift_x = 3.0 * np.cos(2 * np.pi * t / max(time_count - 1, 1))
        shifts[t] = (shift_y, shift_x)
        for c in range(channel_count):
            image = _apply_yx_shift(base_channels[c], (shift_y, shift_x))
            image += rng.normal(0, noise_sigma, size=shape_yx).astype(np.float32)
            stack[t, 0, c, :, :] = np.clip(image, 0, None)

    return stack, shifts


def create_3d_motion_distorted_stack(
    *,
    time_count: int = 10,
    z_count: int = 14,
    channel_count: int = 2,
    shape_yx: tuple[int, int] = (128, 128),
    noise_sigma: float = 0.03,
    random_state: int = 11,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a two-channel ``TZCYX`` stack with time-wise and intra-Z motion artifacts.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        Motion-distorted stack, ground-truth time shifts ``T, 2``, and slice
        shifts ``T, Z, 2``.
    """

    rng = np.random.default_rng(random_state)
    stack = np.zeros((time_count, z_count, channel_count, *shape_yx), dtype=np.float32)
    time_shifts = np.zeros((time_count, 2), dtype=np.float32)
    z_shifts = np.zeros((time_count, z_count, 2), dtype=np.float32)

    z_positions = np.linspace(-1.0, 1.0, z_count, dtype=np.float32)
    for t in range(time_count):
        time_shift = (
            3.5 * np.sin(2 * np.pi * t / max(time_count - 1, 1)),
            3.0 * np.cos(2 * np.pi * t / max(time_count - 1, 1)),
        )
        time_shifts[t] = time_shift
        for z, z_pos in enumerate(z_positions):
            axial_scale = np.exp(-(z_pos**2) / 0.55)
            channel0 = axial_scale * _gaussian_blob_grid(
                shape_yx,
                centers=[(36 + 5 * z_pos, 42), (76, 64 + 8 * z_pos), (92, 96)],
                sigmas=[(7, 10), (9, 7), (8, 9)],
            )
            channel1 = axial_scale * _gaussian_blob_grid(
                shape_yx,
                centers=[(54, 30 + 5 * z_pos), (86 + 4 * z_pos, 48), (78, 104)],
                sigmas=[(8, 8), (6, 11), (9, 8)],
            )
            local_z_shift = (
                1.8 * np.sin(2 * np.pi * z / max(z_count - 1, 1) + 0.4 * t),
                1.5 * np.cos(2 * np.pi * z / max(z_count - 1, 1) + 0.3 * t),
            )
            total_shift = (time_shift[0] + local_z_shift[0], time_shift[1] + local_z_shift[1])
            z_shifts[t, z] = local_z_shift
            for c, image in enumerate([channel0, channel1]):
                moved = _apply_yx_shift(image, total_shift)
                moved += rng.normal(0, noise_sigma, size=shape_yx).astype(np.float32)
                stack[t, z, c, :, :] = np.clip(moved, 0, None)

    return stack, time_shifts, z_shifts


def _write_time_shift_table(
    path: Path,
    shifts_yx: np.ndarray,
    *,
    registration_stack: int = 0,
) -> Path:
    """Write applied and expected time-registration shifts to a CSV table."""

    expected_registration_shifts = shifts_yx[int(registration_stack), :] - shifts_yx
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "t",
                "applied_shift_y",
                "applied_shift_x",
                f"expected_registration_shift_y_ref_t{registration_stack}",
                f"expected_registration_shift_x_ref_t{registration_stack}",
            ]
        )
        for t, (applied_shift, expected_shift) in enumerate(
            zip(shifts_yx, expected_registration_shifts, strict=True)
        ):
            writer.writerow(
                [
                    t,
                    float(applied_shift[0]),
                    float(applied_shift[1]),
                    float(expected_shift[0]),
                    float(expected_shift[1]),
                ]
            )
    return path


def _write_3d_slice_shift_table(
    path: Path,
    *,
    time_shifts_yx: np.ndarray,
    z_shifts_yx: np.ndarray,
) -> Path:
    """Write per-slice local and total applied shifts to a CSV table."""

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "t",
                "z",
                "applied_time_shift_y",
                "applied_time_shift_x",
                "applied_local_z_shift_y",
                "applied_local_z_shift_x",
                "applied_total_shift_y",
                "applied_total_shift_x",
                "expected_local_z_correction_shift_y",
                "expected_local_z_correction_shift_x",
            ]
        )
        for t in range(z_shifts_yx.shape[0]):
            for z in range(z_shifts_yx.shape[1]):
                local_shift = z_shifts_yx[t, z, :]
                total_shift = time_shifts_yx[t, :] + local_shift
                writer.writerow(
                    [
                        t,
                        z,
                        float(time_shifts_yx[t, 0]),
                        float(time_shifts_yx[t, 1]),
                        float(local_shift[0]),
                        float(local_shift[1]),
                        float(total_shift[0]),
                        float(total_shift[1]),
                        float(-local_shift[0]),
                        float(-local_shift[1]),
                    ]
                )
    return path


def write_example_dataset(output_dir: str | Path) -> dict[str, str]:
    """
    Write the default ZenReg synthetic example datasets.

    Parameters
    ----------
    output_dir : str or pathlib.Path
        Directory where datasets and metadata should be written.

    Returns
    -------
    dict[str, str]
        Mapping from dataset labels to written file paths.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stack_2d, shifts_2d = create_2d_motion_distorted_stack()
    stack_3d, time_shifts_3d, z_shifts_3d = create_3d_motion_distorted_stack()

    paths = {
        "motion_2d_tif": str(save_stack(output_dir / "motion_distorted_2d_tzcyx.tif", stack_2d)),
        "motion_3d_tif": str(save_stack(output_dir / "motion_distorted_3d_tzcyx.tif", stack_3d)),
        "motion_2d_time_gt_csv": str(
            _write_time_shift_table(output_dir / "motion_distorted_2d_time_shifts_gt.csv", shifts_2d)
        ),
        "motion_3d_time_gt_csv": str(
            _write_time_shift_table(
                output_dir / "motion_distorted_3d_time_shifts_gt.csv",
                time_shifts_3d,
            )
        ),
        "motion_3d_slice_gt_csv": str(
            _write_3d_slice_shift_table(
                output_dir / "motion_distorted_3d_slice_shifts_gt.csv",
                time_shifts_yx=time_shifts_3d,
                z_shifts_yx=z_shifts_3d,
            )
        ),
    }

    metadata = {
        "axis_order": "TZCYX",
        "motion_2d_shape": list(stack_2d.shape),
        "motion_3d_shape": list(stack_3d.shape),
        "motion_2d_applied_time_shifts_yx": shifts_2d.tolist(),
        "motion_3d_applied_time_shifts_yx": time_shifts_3d.tolist(),
        "motion_3d_applied_slice_shifts_yx": z_shifts_3d.tolist(),
    }
    metadata_path = output_dir / "synthetic_motion_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    paths["metadata"] = str(metadata_path)
    return paths
