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

from .io import create_empty_stack, create_stack_metadata, save_stack


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


def _apply_zyx_shift(volume: np.ndarray, shift_zyx: tuple[float, float, float]) -> np.ndarray:
    """Apply a 3D translation to one ``ZYX`` volume."""

    return ndi_shift(
        volume.astype(np.float32, copy=False),
        shift=shift_zyx,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=True,
    ).astype(np.float32, copy=False)


def _base_2d_channels(
    shape_yx: tuple[int, int],
    *,
    channel_count: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Create deterministic, structurally distinct 2D channel templates."""

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
    return base_channels[:channel_count]


def _base_3d_channels(
    *,
    z_count: int,
    shape_yx: tuple[int, int],
    channel_count: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Create deterministic 3D channel templates with real Z structure."""

    z_positions = np.linspace(-1.0, 1.0, z_count, dtype=np.float32)
    channels = [
        np.zeros((z_count, *shape_yx), dtype=np.float32),
        np.zeros((z_count, *shape_yx), dtype=np.float32),
    ]
    for z, z_pos in enumerate(z_positions):
        axial_scale = 0.35 + 0.75 * np.exp(-(z_pos**2) / 0.55)
        channels[0][z] = axial_scale * _gaussian_blob_grid(
            shape_yx,
            centers=[(36 + 5 * z_pos, 42), (76, 64 + 8 * z_pos), (92, 96)],
            sigmas=[(7, 10), (9, 7), (8, 9)],
        )
        channels[1][z] = axial_scale * _gaussian_blob_grid(
            shape_yx,
            centers=[(54, 30 + 5 * z_pos), (86 + 4 * z_pos, 48), (78, 104)],
            sigmas=[(8, 8), (6, 11), (9, 8)],
        )

    while len(channels) < channel_count:
        random_volume = np.zeros((z_count, *shape_yx), dtype=np.float32)
        for z, z_pos in enumerate(z_positions):
            random_volume[z] = (0.2 + 0.6 * np.exp(-(z_pos**2) / 0.7)) * rng.random(
                shape_yx,
                dtype=np.float32,
            )
        channels.append(random_volume)
    return channels[:channel_count]


def _add_noise_and_clip(
    image: np.ndarray,
    *,
    rng: np.random.Generator,
    noise_sigma: float,
) -> np.ndarray:
    """Add Gaussian noise and clip negative values."""

    noisy = image.astype(np.float32, copy=False)
    if noise_sigma > 0:
        noisy = noisy + rng.normal(0, noise_sigma, size=image.shape).astype(np.float32)
    return np.clip(noisy, 0, None).astype(np.float32, copy=False)


def _time_shifts_yx(time_count: int, *, amplitude_y: float = 4.0, amplitude_x: float = 3.0) -> np.ndarray:
    """Create deterministic time-wise YX shifts with t=0 as the reference."""

    shifts = np.zeros((time_count, 2), dtype=np.float32)
    denominator = max(time_count - 1, 1)
    for t in range(time_count):
        shifts[t] = (
            amplitude_y * np.sin(2 * np.pi * t / denominator),
            amplitude_x * (np.cos(2 * np.pi * t / denominator) - 1.0),
        )
    return shifts


def _time_shifts_zyx(
    time_count: int,
    *,
    amplitude_z: float = 1.5,
    amplitude_y: float = 3.5,
    amplitude_x: float = 3.0,
) -> np.ndarray:
    """Create deterministic time-wise ZYX shifts with t=0 as the reference."""

    shifts = np.zeros((time_count, 3), dtype=np.float32)
    denominator = max(time_count - 1, 1)
    for t in range(time_count):
        shifts[t] = (
            amplitude_z * np.sin(2 * np.pi * t / denominator),
            amplitude_y * np.sin(2 * np.pi * t / denominator + 0.2),
            amplitude_x * (np.cos(2 * np.pi * t / denominator) - 1.0),
        )
    shifts[0, :] = 0.0
    return shifts


def _slice_shifts_yx(
    *,
    time_count: int,
    z_count: int,
    amplitude_y: float = 2.0,
    amplitude_x: float = 1.7,
    time_varying: bool = False,
) -> np.ndarray:
    """Create deterministic per-slice YX shifts with z=0 as the reference."""

    shifts = np.zeros((time_count, z_count, 2), dtype=np.float32)
    denominator = max(z_count - 1, 1)
    for t in range(time_count):
        time_phase = 0.45 * t if time_varying else 0.0
        for z in range(z_count):
            z_phase = 2 * np.pi * z / denominator
            shifts[t, z] = (
                amplitude_y * np.sin(z_phase + time_phase) - amplitude_y * np.sin(time_phase),
                amplitude_x * (np.cos(z_phase + time_phase) - np.cos(time_phase)),
            )
    return shifts


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
    base_channels = _base_2d_channels(shape_yx, channel_count=channel_count, rng=rng)
    stack = create_empty_stack(
        shape=(time_count, 1, channel_count, *shape_yx),
        dtype=np.float32,
        fill_value=0,
        verbose=False,
    )
    shifts = _time_shifts_yx(time_count)

    for t in range(time_count):
        shift_y, shift_x = shifts[t]
        for c in range(channel_count):
            image = _apply_yx_shift(base_channels[c], (shift_y, shift_x))
            stack[t, 0, c, :, :] = _add_noise_and_clip(image, rng=rng, noise_sigma=noise_sigma)

    return stack, shifts


def create_3d_slice_motion_distorted_stack(
    *,
    z_count: int = 14,
    channel_count: int = 2,
    shape_yx: tuple[int, int] = (128, 128),
    noise_sigma: float = 0.025,
    random_state: int = 13,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a ``TZCYX`` 3D stack with per-slice XY translation artifacts.

    The returned stack has ``T=1``. Slice ``z=0`` is the unshifted reference, so
    the expected correction shift for every slice is ``-applied_shift_yx``.
    """

    stack, slice_shifts = create_3d_time_intra_motion_distorted_stack(
        time_count=1,
        z_count=z_count,
        channel_count=channel_count,
        shape_yx=shape_yx,
        noise_sigma=noise_sigma,
        random_state=random_state,
        time_varying_slice_shifts=False,
    )
    return stack, slice_shifts


def create_3d_time_xy_motion_distorted_stack(
    *,
    time_count: int = 10,
    z_count: int = 14,
    channel_count: int = 2,
    shape_yx: tuple[int, int] = (128, 128),
    noise_sigma: float = 0.025,
    random_state: int = 17,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a ``TZCYX`` 3D+t stack with global time-wise XY translations only.

    The same ``(shift_y, shift_x)`` is applied to every Z slice and channel of a
    time point. ``t=0`` is unshifted.
    """

    rng = np.random.default_rng(random_state)
    base_channels = _base_3d_channels(
        z_count=z_count,
        shape_yx=shape_yx,
        channel_count=channel_count,
        rng=rng,
    )
    stack = create_empty_stack(
        shape=(time_count, z_count, channel_count, *shape_yx),
        dtype=np.float32,
        fill_value=0,
        verbose=False,
    )
    time_shifts = _time_shifts_yx(time_count, amplitude_y=3.5, amplitude_x=3.0)

    for t in range(time_count):
        shift_yx = tuple(float(v) for v in time_shifts[t])
        for c, base_volume in enumerate(base_channels):
            for z in range(z_count):
                moved = _apply_yx_shift(base_volume[z], shift_yx)
                stack[t, z, c, :, :] = _add_noise_and_clip(moved, rng=rng, noise_sigma=noise_sigma)

    return stack, time_shifts


def create_3d_time_intra_motion_distorted_stack(
    *,
    time_count: int = 8,
    z_count: int = 14,
    channel_count: int = 2,
    shape_yx: tuple[int, int] = (128, 128),
    noise_sigma: float = 0.025,
    random_state: int = 19,
    time_varying_slice_shifts: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a ``TZCYX`` 3D+t stack with per-slice XY translations only.

    No global time-wise translation is applied. This is the clean benchmark case
    for ``register_stack(..., time_registration_mode="none", intra_stack=True)``.
    Slice ``z=0`` is unshifted for every time point.
    """

    rng = np.random.default_rng(random_state)
    base_channels = _base_2d_channels(shape_yx, channel_count=channel_count, rng=rng)
    stack = create_empty_stack(
        shape=(time_count, z_count, channel_count, *shape_yx),
        dtype=np.float32,
        fill_value=0,
        verbose=False,
    )
    slice_shifts = _slice_shifts_yx(
        time_count=time_count,
        z_count=z_count,
        time_varying=time_varying_slice_shifts,
    )
    z_positions = np.linspace(-1.0, 1.0, z_count, dtype=np.float32)

    for t in range(time_count):
        for z, z_pos in enumerate(z_positions):
            axial_scale = 0.5 + 0.5 * np.exp(-(z_pos**2) / 0.7)
            shift_yx = tuple(float(v) for v in slice_shifts[t, z])
            for c, base_image in enumerate(base_channels):
                moved = _apply_yx_shift(base_image * axial_scale, shift_yx)
                stack[t, z, c, :, :] = _add_noise_and_clip(moved, rng=rng, noise_sigma=noise_sigma)

    return stack, slice_shifts


def create_3d_time_zyx_motion_distorted_stack(
    *,
    time_count: int = 10,
    z_count: int = 18,
    channel_count: int = 2,
    shape_yx: tuple[int, int] = (128, 128),
    noise_sigma: float = 0.02,
    random_state: int = 23,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a ``TZCYX`` 3D+t stack with global time-wise ZYX translations.

    This is the benchmark case for full-volume 3D time registration with
    ``zreg=True``. ``t=0`` is unshifted.
    """

    rng = np.random.default_rng(random_state)
    base_channels = _base_3d_channels(
        z_count=z_count,
        shape_yx=shape_yx,
        channel_count=channel_count,
        rng=rng,
    )
    stack = create_empty_stack(
        shape=(time_count, z_count, channel_count, *shape_yx),
        dtype=np.float32,
        fill_value=0,
        verbose=False,
    )
    time_shifts = _time_shifts_zyx(time_count)

    for t in range(time_count):
        shift_zyx = tuple(float(v) for v in time_shifts[t])
        for c, base_volume in enumerate(base_channels):
            moved = _apply_zyx_shift(base_volume, shift_zyx)
            stack[t, :, c, :, :] = _add_noise_and_clip(moved, rng=rng, noise_sigma=noise_sigma)

    return stack, time_shifts


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
    stack = create_empty_stack(
        shape=(time_count, z_count, channel_count, *shape_yx),
        dtype=np.float32,
        fill_value=0,
        verbose=False,
    )
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


def _write_time_shift_zyx_table(
    path: Path,
    shifts_zyx: np.ndarray,
    *,
    registration_stack: int = 0,
) -> Path:
    """Write applied and expected ZYX time-registration shifts to a CSV table."""

    expected_registration_shifts = shifts_zyx[int(registration_stack), :] - shifts_zyx
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "t",
                "applied_shift_z",
                "applied_shift_y",
                "applied_shift_x",
                f"expected_registration_shift_z_ref_t{registration_stack}",
                f"expected_registration_shift_y_ref_t{registration_stack}",
                f"expected_registration_shift_x_ref_t{registration_stack}",
            ]
        )
        for t, (applied_shift, expected_shift) in enumerate(
            zip(shifts_zyx, expected_registration_shifts, strict=True)
        ):
            writer.writerow(
                [
                    t,
                    float(applied_shift[0]),
                    float(applied_shift[1]),
                    float(applied_shift[2]),
                    float(expected_shift[0]),
                    float(expected_shift[1]),
                    float(expected_shift[2]),
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

    stack_2d_t_xy, shifts_2d_t_xy = create_2d_motion_distorted_stack()
    stack_3d_z_xy, shifts_3d_z_xy = create_3d_slice_motion_distorted_stack()
    stack_3d_t_xy, shifts_3d_t_xy = create_3d_time_xy_motion_distorted_stack()
    stack_3d_t_intra_xy, shifts_3d_t_intra_xy = create_3d_time_intra_motion_distorted_stack()
    stack_3d_t_zyx, shifts_3d_t_zyx = create_3d_time_zyx_motion_distorted_stack()

    zero_time_shifts_3d = np.zeros((stack_3d_z_xy.shape[0], 2), dtype=np.float32)
    zero_time_shifts_3d_t = np.zeros((stack_3d_t_intra_xy.shape[0], 2), dtype=np.float32)

    dataset_specs = {
        "synthetic_2d_t_xy": {
            "stack": stack_2d_t_xy,
            "image": "synthetic_2d_t_xy.ome.tif",
            "time_gt": "synthetic_2d_t_xy_time_shifts_gt.csv",
            "annotations": {
                "ZenReg_SyntheticDataset": "synthetic_2d_t_xy",
                "ZenReg_RegistrationTarget": "2D+t global XY time registration",
                "ZenReg_TimeShiftGT": "synthetic_2d_t_xy_time_shifts_gt.csv",
            },
        },
        "synthetic_3d_z_xy": {
            "stack": stack_3d_z_xy,
            "image": "synthetic_3d_z_xy.ome.tif",
            "slice_gt": "synthetic_3d_z_xy_slice_shifts_gt.csv",
            "annotations": {
                "ZenReg_SyntheticDataset": "synthetic_3d_z_xy",
                "ZenReg_RegistrationTarget": "3D intra-stack XY slice registration",
                "ZenReg_SliceShiftGT": "synthetic_3d_z_xy_slice_shifts_gt.csv",
            },
        },
        "synthetic_3d_t_xy": {
            "stack": stack_3d_t_xy,
            "image": "synthetic_3d_t_xy.ome.tif",
            "time_gt": "synthetic_3d_t_xy_time_shifts_gt.csv",
            "annotations": {
                "ZenReg_SyntheticDataset": "synthetic_3d_t_xy",
                "ZenReg_RegistrationTarget": "3D+t global XY time registration",
                "ZenReg_TimeShiftGT": "synthetic_3d_t_xy_time_shifts_gt.csv",
            },
        },
        "synthetic_3d_t_intra_xy": {
            "stack": stack_3d_t_intra_xy,
            "image": "synthetic_3d_t_intra_xy.ome.tif",
            "slice_gt": "synthetic_3d_t_intra_xy_slice_shifts_gt.csv",
            "annotations": {
                "ZenReg_SyntheticDataset": "synthetic_3d_t_intra_xy",
                "ZenReg_RegistrationTarget": "3D+t intra-stack-only XY slice registration",
                "ZenReg_SliceShiftGT": "synthetic_3d_t_intra_xy_slice_shifts_gt.csv",
            },
        },
        "synthetic_3d_t_zyx": {
            "stack": stack_3d_t_zyx,
            "image": "synthetic_3d_t_zyx.ome.tif",
            "time_gt": "synthetic_3d_t_zyx_time_shifts_gt.csv",
            "annotations": {
                "ZenReg_SyntheticDataset": "synthetic_3d_t_zyx",
                "ZenReg_RegistrationTarget": "3D+t global ZYX time registration",
                "ZenReg_TimeShiftGT": "synthetic_3d_t_zyx_time_shifts_gt.csv",
            },
        },
    }

    paths: dict[str, str] = {}
    for dataset_name, spec in dataset_specs.items():
        stack = spec["stack"]
        metadata = create_stack_metadata(
            stack,
            annotations=spec["annotations"],
            verbose=False,
        )
        paths[f"{dataset_name}_ome_tif"] = str(
            save_stack(
                output_dir / str(spec["image"]),
                stack,
                metadata=metadata,
            )
        )

    paths["synthetic_2d_t_xy_time_gt_csv"] = str(
        _write_time_shift_table(output_dir / "synthetic_2d_t_xy_time_shifts_gt.csv", shifts_2d_t_xy)
    )
    paths["synthetic_3d_z_xy_slice_gt_csv"] = str(
        _write_3d_slice_shift_table(
            output_dir / "synthetic_3d_z_xy_slice_shifts_gt.csv",
            time_shifts_yx=zero_time_shifts_3d,
            z_shifts_yx=shifts_3d_z_xy,
        )
    )
    paths["synthetic_3d_t_xy_time_gt_csv"] = str(
        _write_time_shift_table(output_dir / "synthetic_3d_t_xy_time_shifts_gt.csv", shifts_3d_t_xy)
    )
    paths["synthetic_3d_t_intra_xy_slice_gt_csv"] = str(
        _write_3d_slice_shift_table(
            output_dir / "synthetic_3d_t_intra_xy_slice_shifts_gt.csv",
            time_shifts_yx=zero_time_shifts_3d_t,
            z_shifts_yx=shifts_3d_t_intra_xy,
        )
    )
    paths["synthetic_3d_t_zyx_time_gt_csv"] = str(
        _write_time_shift_zyx_table(output_dir / "synthetic_3d_t_zyx_time_shifts_gt.csv", shifts_3d_t_zyx)
    )

    metadata = {
        "axis_order": "TZCYX",
        "synthetic_2d_t_xy_shape": list(stack_2d_t_xy.shape),
        "synthetic_3d_z_xy_shape": list(stack_3d_z_xy.shape),
        "synthetic_3d_t_xy_shape": list(stack_3d_t_xy.shape),
        "synthetic_3d_t_intra_xy_shape": list(stack_3d_t_intra_xy.shape),
        "synthetic_3d_t_zyx_shape": list(stack_3d_t_zyx.shape),
        "synthetic_2d_t_xy_applied_time_shifts_yx": shifts_2d_t_xy.tolist(),
        "synthetic_3d_z_xy_applied_slice_shifts_yx": shifts_3d_z_xy.tolist(),
        "synthetic_3d_t_xy_applied_time_shifts_yx": shifts_3d_t_xy.tolist(),
        "synthetic_3d_t_intra_xy_applied_slice_shifts_yx": shifts_3d_t_intra_xy.tolist(),
        "synthetic_3d_t_zyx_applied_time_shifts_zyx": shifts_3d_t_zyx.tolist(),
    }
    metadata_path = output_dir / "synthetic_motion_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    paths["metadata"] = str(metadata_path)
    return paths
