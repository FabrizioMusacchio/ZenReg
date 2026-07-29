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
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import affine_transform, map_coordinates
from scipy.ndimage import shift as ndi_shift
from skimage.transform import rotate

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


def _apply_yx_rotation(image: np.ndarray, angle_deg: float) -> np.ndarray:
    """Apply an in-plane XY rotation to one image plane."""

    return rotate(
        image.astype(np.float32, copy=False),
        float(angle_deg),
        resize=False,
        order=1,
        mode="constant",
        cval=0.0,
        preserve_range=True,
    ).astype(np.float32, copy=False)


def _rotation_matrix_zyx(
    *,
    rotation_z_deg: float = 0.0,
    rotation_y_deg: float = 0.0,
    rotation_x_deg: float = 0.0,
) -> np.ndarray:
    """Return a 3D rotation matrix in array-coordinate ``ZYX`` order."""

    rz = np.deg2rad(float(rotation_z_deg))
    ry = np.deg2rad(float(rotation_y_deg))
    rx = np.deg2rad(float(rotation_x_deg))
    cz, sz = np.cos(rz), np.sin(rz)
    cy, sy = np.cos(ry), np.sin(ry)
    cx, sx = np.cos(rx), np.sin(rx)

    rot_z = np.asarray(
        [
            [1, 0, 0],
            [0, cz, sz],
            [0, -sz, cz],
        ],
        dtype=np.float32,
    )
    rot_y = np.asarray(
        [
            [cy, 0, -sy],
            [0, 1, 0],
            [sy, 0, cy],
        ],
        dtype=np.float32,
    )
    rot_x = np.asarray(
        [
            [cx, sx, 0],
            [-sx, cx, 0],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )
    return (rot_z @ rot_y @ rot_x).astype(np.float32)


def _apply_zyx_rigid_transform(
    volume: np.ndarray,
    *,
    shift_zyx: tuple[float, float, float],
    rotation_z_deg: float = 0.0,
    rotation_y_deg: float = 0.0,
    rotation_x_deg: float = 0.0,
    center_zyx: tuple[float, float, float] | None = None,
) -> np.ndarray:
    """Apply a 3D rigid transform around an arbitrary center in ``ZYX`` order."""

    volume = volume.astype(np.float32, copy=False)
    if center_zyx is None:
        center = (np.asarray(volume.shape, dtype=np.float32) - 1) / 2.0
    else:
        center = np.asarray(center_zyx, dtype=np.float32)
    shift = np.asarray(shift_zyx, dtype=np.float32)
    rotation = _rotation_matrix_zyx(
        rotation_z_deg=rotation_z_deg,
        rotation_y_deg=rotation_y_deg,
        rotation_x_deg=rotation_x_deg,
    )
    inverse = rotation.T
    offset = center - inverse @ (center + shift)
    return affine_transform(
        volume,
        matrix=inverse,
        offset=offset,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=True,
    ).astype(np.float32, copy=False)


def _apply_yx_displacement(
    image: np.ndarray,
    *,
    shift_y: np.ndarray,
    shift_x: np.ndarray,
) -> np.ndarray:
    """Apply a dense 2D displacement field in correction-sign convention."""

    yy, xx = np.indices(image.shape, dtype=np.float32)
    coords = [yy - shift_y.astype(np.float32), xx - shift_x.astype(np.float32)]
    return map_coordinates(
        image.astype(np.float32, copy=False),
        coords,
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


def _base_2d_puncta_channels(
    shape_yx: tuple[int, int],
    *,
    channel_count: int,
    rng: np.random.Generator,
    spot_count: int = 45,
) -> list[np.ndarray]:
    """Create feature-rich 2D channels for local non-rigid benchmarks."""

    margin = 12
    centers0 = list(
        zip(
            rng.uniform(margin, shape_yx[0] - margin, spot_count),
            rng.uniform(margin, shape_yx[1] - margin, spot_count),
        )
    )
    sigmas0 = list(
        zip(
            rng.uniform(2.0, 5.0, spot_count),
            rng.uniform(2.0, 5.0, spot_count),
        )
    )
    base0 = _gaussian_blob_grid(shape_yx, centers=centers0, sigmas=sigmas0)

    centers1 = [
        (
            np.clip(cy + rng.normal(0, 4), margin, shape_yx[0] - margin),
            np.clip(cx + rng.normal(0, 4), margin, shape_yx[1] - margin),
        )
        for cy, cx in centers0
    ]
    sigmas1 = list(
        zip(
            rng.uniform(2.0, 5.0, spot_count),
            rng.uniform(2.0, 5.0, spot_count),
        )
    )
    base1 = _gaussian_blob_grid(shape_yx, centers=centers1, sigmas=sigmas1)

    base_channels = [base0, base1]
    while len(base_channels) < channel_count:
        extra_centers = list(
            zip(
                rng.uniform(margin, shape_yx[0] - margin, spot_count),
                rng.uniform(margin, shape_yx[1] - margin, spot_count),
            )
        )
        extra_sigmas = list(
            zip(
                rng.uniform(2.0, 5.0, spot_count),
                rng.uniform(2.0, 5.0, spot_count),
            )
        )
        base_channels.append(_gaussian_blob_grid(shape_yx, centers=extra_centers, sigmas=extra_sigmas))
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


def _time_rotations_deg(time_count: int, *, amplitude_deg: float = 8.0) -> np.ndarray:
    """Create deterministic time-wise in-plane rotations with t=0 as reference."""

    rotations = np.zeros(time_count, dtype=np.float32)
    denominator = max(time_count - 1, 1)
    for t in range(time_count):
        rotations[t] = amplitude_deg * np.sin(2 * np.pi * t / denominator)
    rotations[0] = 0.0
    return rotations


def _time_sparse_rotation_events_deg(time_count: int) -> np.ndarray:
    """Create sparse short rotation events on an otherwise stable baseline."""

    rotations = np.zeros(time_count, dtype=np.float32)
    if time_count <= 1:
        return rotations

    phase = np.linspace(0.0, 1.0, time_count, dtype=np.float32)
    rotations += 0.35 * np.sin(2 * np.pi * phase) + 0.2 * np.sin(6 * np.pi * phase + 0.4)
    event_specs = [
        (24, 13, 2.4),
        (57, 11, -2.1),
        (92, 15, 3.0),
        (128, 10, -2.6),
    ]
    for start, duration, amplitude in event_specs:
        for offset in range(duration):
            t = start + offset
            if t >= time_count:
                continue
            rel = offset / max(duration - 1, 1)
            rotations[t] += amplitude * np.sin(np.pi * rel)

    rotations -= rotations[0]
    rotations[0] = 0.0
    return rotations.astype(np.float32)


def _time_progressive_shifts_yx(time_count: int) -> np.ndarray:
    """Create smooth global translations with weak and stronger motion epochs."""

    shifts = np.zeros((time_count, 2), dtype=np.float32)
    if time_count <= 1:
        return shifts

    phase = np.linspace(0.0, 1.0, time_count, dtype=np.float32)
    strength = np.linspace(0.25, 1.0, time_count, dtype=np.float32)
    shifts[:, 0] = strength * (
        3.1 * np.sin(2 * np.pi * phase + 0.15)
        + 0.9 * np.sin(7 * np.pi * phase + 0.4)
    )
    shifts[:, 1] = strength * (
        2.7 * (np.cos(2 * np.pi * phase + 0.2) - np.cos(0.2))
        + 0.8 * np.sin(5 * np.pi * phase)
    )

    jitter_frames = {
        38: (0.8, -0.4),
        39: (0.4, -0.2),
        104: (-0.6, 0.7),
        105: (-0.3, 0.35),
    }
    for t, shift_yx in jitter_frames.items():
        if t < time_count:
            shifts[t, :] += np.asarray(shift_yx, dtype=np.float32)

    shifts -= shifts[0]
    shifts[0, :] = 0.0
    return shifts.astype(np.float32)


def _time_rotations_zyx_deg(
    time_count: int,
    *,
    amplitude_z_deg: float = 5.0,
    amplitude_y_deg: float = 3.0,
    amplitude_x_deg: float = 4.0,
) -> np.ndarray:
    """Create deterministic time-wise rotations around Z/Y/X axes in degrees."""

    rotations = np.zeros((time_count, 3), dtype=np.float32)
    denominator = max(time_count - 1, 1)
    for t in range(time_count):
        phase = 2 * np.pi * t / denominator
        rotations[t] = (
            amplitude_z_deg * np.sin(phase),
            amplitude_y_deg * np.sin(phase + 0.55),
            amplitude_x_deg * np.cos(phase + 0.25) - amplitude_x_deg * np.cos(0.25),
        )
    rotations[0, :] = 0.0
    return rotations


def _local_motion_parameters(time_count: int) -> np.ndarray:
    """Create sparse-in-time local motion bursts plus short global jitters."""

    params = np.zeros((time_count, 6), dtype=np.float32)
    burst_specs = [
        (42, 9, 2.0, -1.6, 0.0, 0.35),
        (104, 7, -1.7, 1.4, 1.1, -0.2),
        (154, 11, 2.3, 1.8, -0.7, 0.8),
    ]
    for start, duration, amp_y, amp_x, phase_y0, phase_x0 in burst_specs:
        for offset in range(duration):
            t = start + offset
            if t >= time_count:
                continue
            rel = offset / max(duration - 1, 1)
            envelope = np.sin(np.pi * rel)
            params[t, 0] += amp_y * envelope
            params[t, 1] += amp_x * envelope
            params[t, 2] = phase_y0 + 2 * np.pi * rel
            params[t, 3] = phase_x0 + np.pi * rel

    global_jitters = {
        73: (1.2, -0.8),
        74: (0.6, -0.4),
        136: (-1.0, 1.1),
        137: (-0.5, 0.5),
    }
    for t, shift_yx in global_jitters.items():
        if t < time_count:
            params[t, 4:6] = shift_yx
    return params


def _local_motion_fields_yx(shape_yx: tuple[int, int], params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Create smooth dense Y/X displacement fields from compact parameters."""

    yy, xx = np.indices(shape_yx, dtype=np.float32)
    height, width = shape_yx
    amp_y, amp_x, phase_y, phase_x = [float(v) for v in params[:4]]
    global_y = float(params[4]) if len(params) > 4 else 0.0
    global_x = float(params[5]) if len(params) > 5 else 0.0
    shift_y = amp_y * np.sin(2 * np.pi * xx / max(width, 1) + phase_y)
    shift_x = amp_x * np.sin(2 * np.pi * yy / max(height, 1) + phase_x)
    shift_y = shift_y + global_y
    shift_x = shift_x + global_x
    return shift_y.astype(np.float32), shift_x.astype(np.float32)


def _interpolate_anchor_shift_field_yx(
    shape_yx: tuple[int, int],
    anchor_shifts_yx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate sparse anchor shifts into dense Y/X displacement fields."""

    height, width = shape_yx
    anchor_shifts_yx = np.asarray(anchor_shifts_yx, dtype=np.float32)
    grid_y = np.linspace(0.0, height - 1.0, anchor_shifts_yx.shape[0], dtype=np.float32)
    grid_x = np.linspace(0.0, width - 1.0, anchor_shifts_yx.shape[1], dtype=np.float32)
    yy, xx = np.indices(shape_yx, dtype=np.float32)
    points = np.column_stack([yy.ravel(), xx.ravel()])
    interp_y = RegularGridInterpolator(
        (grid_y, grid_x),
        anchor_shifts_yx[..., 0],
        bounds_error=False,
        fill_value=None,
    )
    interp_x = RegularGridInterpolator(
        (grid_y, grid_x),
        anchor_shifts_yx[..., 1],
        bounds_error=False,
        fill_value=None,
    )
    shift_y = interp_y(points).reshape(shape_yx).astype(np.float32)
    shift_x = interp_x(points).reshape(shape_yx).astype(np.float32)
    return shift_y, shift_x


def _piecewise_anchor_shifts_yx(
    *,
    time_count: int,
    grid_shape_yx: tuple[int, int] = (4, 4),
    random_state: int = 71,
) -> np.ndarray:
    """Create deterministic local patch-translation GT without rotation."""

    rng = np.random.default_rng(random_state)
    pattern_y = rng.normal(0, 1, grid_shape_yx).astype(np.float32)
    pattern_x = rng.normal(0, 1, grid_shape_yx).astype(np.float32)
    pattern_y -= float(np.mean(pattern_y))
    pattern_x -= float(np.mean(pattern_x))
    pattern_y /= max(float(np.max(np.abs(pattern_y))), 1e-6)
    pattern_x /= max(float(np.max(np.abs(pattern_x))), 1e-6)

    global_shifts = _time_shifts_yx(time_count, amplitude_y=1.4, amplitude_x=1.2)
    shifts = np.zeros((time_count, *grid_shape_yx, 2), dtype=np.float32)
    denominator = max(time_count - 1, 1)
    for t in range(time_count):
        phase = 2 * np.pi * t / denominator
        amp_y = 2.6 * np.sin(phase) + 0.9 * np.sin(2.5 * phase + 0.4)
        amp_x = 2.3 * np.cos(phase + 0.25) - 2.3 * np.cos(0.25)
        amp_x += 0.8 * np.sin(3.0 * phase)
        shifts[t, :, :, 0] = global_shifts[t, 0] + amp_y * pattern_y
        shifts[t, :, :, 1] = global_shifts[t, 1] + amp_x * pattern_x

    shifts -= shifts[0:1]
    shifts[0, :, :, :] = 0.0
    return shifts.astype(np.float32)


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


def create_2d_time_rotation_motion_distorted_stack(
    *,
    time_count: int = 10,
    channel_count: int = 2,
    shape_yx: tuple[int, int] = (128, 128),
    noise_sigma: float = 0.02,
    random_state: int = 29,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a ``TZCYX`` 2D+t stack with global in-plane rotation.

    This is the benchmark case for ``rotreg=True``. ``t=0`` is unshifted and
    unrotated. Returns the stack, applied YX translations (zeros), and applied
    rotations in degrees.
    """

    rng = np.random.default_rng(random_state)
    base_channels = _base_2d_channels(shape_yx, channel_count=channel_count, rng=rng)
    stack = create_empty_stack(
        shape=(time_count, 1, channel_count, *shape_yx),
        dtype=np.float32,
        fill_value=0,
        verbose=False,
    )
    time_shifts = np.zeros((time_count, 2), dtype=np.float32)
    rotations_deg = _time_rotations_deg(time_count, amplitude_deg=8.0)

    for t in range(time_count):
        shift_yx = tuple(float(v) for v in time_shifts[t])
        angle_deg = float(rotations_deg[t])
        for c, base_image in enumerate(base_channels):
            moved = _apply_yx_rotation(base_image, angle_deg)
            moved = _apply_yx_shift(moved, shift_yx)
            stack[t, 0, c, :, :] = _add_noise_and_clip(moved, rng=rng, noise_sigma=noise_sigma)

    return stack, time_shifts, rotations_deg


def create_2d_local_motion_distorted_stack(
    *,
    time_count: int = 200,
    channel_count: int = 2,
    shape_yx: tuple[int, int] = (128, 128),
    noise_sigma: float = 0.004,
    random_state: int = 31,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a ``TZCYX`` 2D+t stack with smooth in-frame non-rigid motion.

    The field contains many local features so each NoRMCorre patch has enough
    structure for phase-correlation. Most frames are stable; short local motion
    bursts and a few global jitter frames mimic a more typical time-lapse motion
    correction problem. Returns the stack and a compact ``T, 6`` parameter table
    with local amplitudes/phases and global YX jitter.
    """

    rng = np.random.default_rng(random_state)
    base_channels = _base_2d_puncta_channels(shape_yx, channel_count=channel_count, rng=rng)
    stack = create_empty_stack(
        shape=(time_count, 1, channel_count, *shape_yx),
        dtype=np.float32,
        fill_value=0,
        verbose=False,
    )
    local_params = _local_motion_parameters(time_count)

    for t in range(time_count):
        shift_y, shift_x = _local_motion_fields_yx(shape_yx, local_params[t])
        for c, base_image in enumerate(base_channels):
            moved = _apply_yx_displacement(base_image, shift_y=shift_y, shift_x=shift_x)
            stack[t, 0, c, :, :] = _add_noise_and_clip(moved, rng=rng, noise_sigma=noise_sigma)

    return stack, local_params


def create_2d_time_translation_rotation_motion_distorted_stack(
    *,
    time_count: int = 160,
    channel_count: int = 2,
    shape_yx: tuple[int, int] = (128, 128),
    noise_sigma: float = 0.006,
    random_state: int = 37,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a ``TZCYX`` 2D+t stack with global XY translations and rotation.

    This is a NoRMCorre-oriented benchmark: many spatial landmarks are present,
    global translations vary from weak to stronger across the time series, and
    global in-plane rotations occur in short event windows rather than in every
    frame. The rotation is still only approximately representable by local
    translations, so this remains a useful stress test instead of a perfect
    rigid-rotation benchmark.
    """

    rng = np.random.default_rng(random_state)
    base_channels = _base_2d_puncta_channels(
        shape_yx,
        channel_count=channel_count,
        rng=rng,
        spot_count=120,
    )
    stack = create_empty_stack(
        shape=(time_count, 1, channel_count, *shape_yx),
        dtype=np.float32,
        fill_value=0,
        verbose=False,
    )
    time_shifts = _time_progressive_shifts_yx(time_count)
    rotations_deg = _time_sparse_rotation_events_deg(time_count)

    for t in range(time_count):
        shift_yx = tuple(float(v) for v in time_shifts[t])
        angle_deg = float(rotations_deg[t])
        for c, base_image in enumerate(base_channels):
            moved = _apply_yx_rotation(base_image, angle_deg)
            moved = _apply_yx_shift(moved, shift_yx)
            stack[t, 0, c, :, :] = _add_noise_and_clip(moved, rng=rng, noise_sigma=noise_sigma)

    return stack, time_shifts, rotations_deg


def create_2d_time_piecewise_xy_motion_distorted_stack(
    *,
    time_count: int = 80,
    channel_count: int = 2,
    shape_yx: tuple[int, int] = (128, 128),
    grid_shape_yx: tuple[int, int] = (4, 4),
    noise_sigma: float = 0.006,
    random_state: int = 73,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a ``TZCYX`` 2D+t stack with local piecewise XY translations.

    This benchmark contains no rotation. It is intentionally favorable to
    NoRMCorre-style patch-wise translation correction and unfavorable to a
    single global phase-cross-correlation shift: each time point has a smooth
    field of local translations defined by a small grid of anchor shifts.
    """

    rng = np.random.default_rng(random_state)
    base_channels = _base_2d_puncta_channels(
        shape_yx,
        channel_count=channel_count,
        rng=rng,
        spot_count=120,
    )
    stack = create_empty_stack(
        shape=(time_count, 1, channel_count, *shape_yx),
        dtype=np.float32,
        fill_value=0,
        verbose=False,
    )
    anchor_shifts_yx = _piecewise_anchor_shifts_yx(
        time_count=time_count,
        grid_shape_yx=grid_shape_yx,
        random_state=random_state + 1,
    )

    for t in range(time_count):
        shift_y, shift_x = _interpolate_anchor_shift_field_yx(shape_yx, anchor_shifts_yx[t])
        for c, base_image in enumerate(base_channels):
            moved = _apply_yx_displacement(base_image, shift_y=shift_y, shift_x=shift_x)
            stack[t, 0, c, :, :] = _add_noise_and_clip(moved, rng=rng, noise_sigma=noise_sigma)

    return stack, anchor_shifts_yx


def create_3d_time_rigid_motion_distorted_stack(
    *,
    time_count: int = 8,
    z_count: int = 18,
    channel_count: int = 2,
    shape_yx: tuple[int, int] = (128, 128),
    rotation_mode: str = "z",
    center_mode: str = "middle",
    noise_sigma: float = 0.018,
    random_state: int = 41,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a ``TZCYX`` 3D+t stack with global translations and 3D rotations.

    Parameters
    ----------
    rotation_mode : {"z", "x", "all"}
        Which rotational degrees of freedom are active.
    center_mode : {"middle", "inside_offset", "outside"}
        Rigid-transform center. Values are stored in the returned GT table.

    Returns
    -------
    tuple
        Stack, applied ``T, 3`` ZYX translations, applied ``T, 3`` rotations in
        ``(z, y, x)`` degrees, and the ``T, 3`` centers in ZYX coordinates.
    """

    rotation_mode = str(rotation_mode).strip().lower()
    center_mode = str(center_mode).strip().lower()
    if rotation_mode not in {"z", "x", "all"}:
        raise ValueError("rotation_mode must be 'z', 'x', or 'all'.")
    if center_mode not in {"middle", "inside_offset", "outside"}:
        raise ValueError("center_mode must be 'middle', 'inside_offset', or 'outside'.")

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
    time_shifts = _time_shifts_zyx(
        time_count,
        amplitude_z=1.0,
        amplitude_y=2.2,
        amplitude_x=2.0,
    )
    rotations = _time_rotations_zyx_deg(
        time_count,
        amplitude_z_deg=4.0,
        amplitude_y_deg=2.5,
        amplitude_x_deg=3.0,
    )
    if rotation_mode == "z":
        rotations[:, 1:] = 0.0
    elif rotation_mode == "x":
        rotations[:, :2] = 0.0

    middle = np.asarray([(z_count - 1) / 2.0, (shape_yx[0] - 1) / 2.0, (shape_yx[1] - 1) / 2.0], dtype=np.float32)
    if center_mode == "middle":
        center = middle
    elif center_mode == "inside_offset":
        center = middle + np.asarray([2.0, -14.0, 11.0], dtype=np.float32)
    else:
        center = middle + np.asarray([8.0, -0.5 * shape_yx[0], 0.65 * shape_yx[1]], dtype=np.float32)
    centers = np.repeat(center[None, :], time_count, axis=0).astype(np.float32)

    for t in range(time_count):
        shift_zyx = tuple(float(v) for v in time_shifts[t])
        rot_z, rot_y, rot_x = [float(v) for v in rotations[t]]
        center_zyx = tuple(float(v) for v in centers[t])
        for c, base_volume in enumerate(base_channels):
            moved = _apply_zyx_rigid_transform(
                base_volume,
                shift_zyx=shift_zyx,
                rotation_z_deg=rot_z,
                rotation_y_deg=rot_y,
                rotation_x_deg=rot_x,
                center_zyx=center_zyx,
            )
            stack[t, :, c, :, :] = _add_noise_and_clip(moved, rng=rng, noise_sigma=noise_sigma)

    return stack, time_shifts, rotations, centers


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


def _write_time_rotation_table(
    path: Path,
    rotations_deg: np.ndarray,
    *,
    registration_stack: int = 0,
) -> Path:
    """Write applied and expected time-wise rotation corrections to a CSV table."""

    expected_registration_rotations = rotations_deg[int(registration_stack)] - rotations_deg
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "t",
                "applied_rotation_deg",
                f"expected_registration_rotation_deg_ref_t{registration_stack}",
            ]
        )
        for t, (applied_rotation, expected_rotation) in enumerate(
            zip(rotations_deg, expected_registration_rotations, strict=True)
        ):
                writer.writerow([t, float(applied_rotation), float(expected_rotation)])
    return path


def _write_local_motion_table(path: Path, local_params: np.ndarray) -> Path:
    """Write compact GT parameters for smooth dense 2D local motion fields."""

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "t",
                "amplitude_y",
                "amplitude_x",
                "phase_y",
                "phase_x",
                "global_shift_y",
                "global_shift_x",
                "motion_magnitude",
            ]
        )
        for t, params in enumerate(local_params):
            params = np.asarray(params, dtype=np.float32)
            if params.shape[0] < 6:
                params = np.pad(params, (0, 6 - params.shape[0]))
            motion_magnitude = float(np.sum(np.abs(params[[0, 1, 4, 5]])))
            writer.writerow([t, *[float(v) for v in params[:6]], motion_magnitude])
    return path


def _write_piecewise_anchor_shift_table(path: Path, anchor_shifts_yx: np.ndarray) -> Path:
    """Write GT local anchor shifts for piecewise 2D+t translation data."""

    anchor_shifts_yx = np.asarray(anchor_shifts_yx, dtype=np.float32)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "t",
                "anchor_y_index",
                "anchor_x_index",
                "applied_anchor_shift_y",
                "applied_anchor_shift_x",
                "expected_anchor_correction_shift_y_ref_t0",
                "expected_anchor_correction_shift_x_ref_t0",
            ]
        )
        expected = anchor_shifts_yx[0:1, :, :, :] - anchor_shifts_yx
        for t in range(anchor_shifts_yx.shape[0]):
            for gy in range(anchor_shifts_yx.shape[1]):
                for gx in range(anchor_shifts_yx.shape[2]):
                    writer.writerow(
                        [
                            t,
                            gy,
                            gx,
                            float(anchor_shifts_yx[t, gy, gx, 0]),
                            float(anchor_shifts_yx[t, gy, gx, 1]),
                            float(expected[t, gy, gx, 0]),
                            float(expected[t, gy, gx, 1]),
                        ]
                    )
    return path


def _write_3d_rigid_transform_table(
    path: Path,
    *,
    shifts_zyx: np.ndarray,
    rotations_zyx_deg: np.ndarray,
    centers_zyx: np.ndarray,
    registration_stack: int = 0,
) -> Path:
    """Write applied 3D rigid-transform GT parameters to CSV."""

    expected_shifts = shifts_zyx[int(registration_stack), :] - shifts_zyx
    expected_rotations = rotations_zyx_deg[int(registration_stack), :] - rotations_zyx_deg
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "t",
                "applied_shift_z",
                "applied_shift_y",
                "applied_shift_x",
                "applied_rotation_z_deg",
                "applied_rotation_y_deg",
                "applied_rotation_x_deg",
                "rotation_center_z",
                "rotation_center_y",
                "rotation_center_x",
                f"expected_registration_shift_z_ref_t{registration_stack}",
                f"expected_registration_shift_y_ref_t{registration_stack}",
                f"expected_registration_shift_x_ref_t{registration_stack}",
                f"expected_registration_rotation_z_deg_ref_t{registration_stack}",
                f"expected_registration_rotation_y_deg_ref_t{registration_stack}",
                f"expected_registration_rotation_x_deg_ref_t{registration_stack}",
            ]
        )
        for t in range(shifts_zyx.shape[0]):
            writer.writerow(
                [
                    t,
                    float(shifts_zyx[t, 0]),
                    float(shifts_zyx[t, 1]),
                    float(shifts_zyx[t, 2]),
                    float(rotations_zyx_deg[t, 0]),
                    float(rotations_zyx_deg[t, 1]),
                    float(rotations_zyx_deg[t, 2]),
                    float(centers_zyx[t, 0]),
                    float(centers_zyx[t, 1]),
                    float(centers_zyx[t, 2]),
                    float(expected_shifts[t, 0]),
                    float(expected_shifts[t, 1]),
                    float(expected_shifts[t, 2]),
                    float(expected_rotations[t, 0]),
                    float(expected_rotations[t, 1]),
                    float(expected_rotations[t, 2]),
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
    stack_2d_t_rot_xy, shifts_2d_t_rot_xy, rotations_2d_t_rot = (
        create_2d_time_rotation_motion_distorted_stack()
    )
    stack_2d_t_local, local_params_2d_t = create_2d_local_motion_distorted_stack()
    stack_2d_t_trans_rot, shifts_2d_t_trans_rot, rotations_2d_t_trans_rot = (
        create_2d_time_translation_rotation_motion_distorted_stack()
    )
    stack_2d_t_piecewise_xy, anchor_shifts_2d_t_piecewise_xy = (
        create_2d_time_piecewise_xy_motion_distorted_stack()
    )
    stack_3d_t_trans_rot_z, shifts_3d_t_trans_rot_z, rotations_3d_t_trans_rot_z, centers_3d_t_trans_rot_z = (
        create_3d_time_rigid_motion_distorted_stack(rotation_mode="z", center_mode="middle", random_state=43)
    )
    stack_3d_t_trans_rot_x, shifts_3d_t_trans_rot_x, rotations_3d_t_trans_rot_x, centers_3d_t_trans_rot_x = (
        create_3d_time_rigid_motion_distorted_stack(rotation_mode="x", center_mode="middle", random_state=47)
    )
    (
        stack_3d_t_trans_rot_all_center,
        shifts_3d_t_trans_rot_all_center,
        rotations_3d_t_trans_rot_all_center,
        centers_3d_t_trans_rot_all_center,
    ) = create_3d_time_rigid_motion_distorted_stack(rotation_mode="all", center_mode="middle", random_state=53)
    (
        stack_3d_t_trans_rot_all_offcenter,
        shifts_3d_t_trans_rot_all_offcenter,
        rotations_3d_t_trans_rot_all_offcenter,
        centers_3d_t_trans_rot_all_offcenter,
    ) = create_3d_time_rigid_motion_distorted_stack(
        rotation_mode="all",
        center_mode="inside_offset",
        random_state=59,
    )
    (
        stack_3d_t_trans_rot_all_outside,
        shifts_3d_t_trans_rot_all_outside,
        rotations_3d_t_trans_rot_all_outside,
        centers_3d_t_trans_rot_all_outside,
    ) = create_3d_time_rigid_motion_distorted_stack(rotation_mode="all", center_mode="outside", random_state=61)

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
        "synthetic_2d_t_rot_xy": {
            "stack": stack_2d_t_rot_xy,
            "image": "synthetic_2d_t_rot_xy.ome.tif",
            "time_gt": "synthetic_2d_t_rot_xy_time_shifts_gt.csv",
            "rotation_gt": "synthetic_2d_t_rot_xy_time_rotations_gt.csv",
            "annotations": {
                "ZenReg_SyntheticDataset": "synthetic_2d_t_rot_xy",
                "ZenReg_RegistrationTarget": "2D+t global in-plane rotation registration",
                "ZenReg_TimeShiftGT": "synthetic_2d_t_rot_xy_time_shifts_gt.csv",
                "ZenReg_RotationGT": "synthetic_2d_t_rot_xy_time_rotations_gt.csv",
            },
        },
        "synthetic_2d_t_local": {
            "stack": stack_2d_t_local,
            "image": "synthetic_2d_t_local.ome.tif",
            "local_gt": "synthetic_2d_t_local_motion_gt.csv",
            "annotations": {
                "ZenReg_SyntheticDataset": "synthetic_2d_t_local",
                "ZenReg_RegistrationTarget": "2D+t NoRMCorre-style local in-frame motion",
                "ZenReg_LocalMotionGT": "synthetic_2d_t_local_motion_gt.csv",
            },
        },
        "synthetic_2d_t_trans_rot_xy": {
            "stack": stack_2d_t_trans_rot,
            "image": "synthetic_2d_t_trans_rot_xy.ome.tif",
            "time_gt": "synthetic_2d_t_trans_rot_xy_time_shifts_gt.csv",
            "rotation_gt": "synthetic_2d_t_trans_rot_xy_time_rotations_gt.csv",
            "annotations": {
                "ZenReg_SyntheticDataset": "synthetic_2d_t_trans_rot_xy",
                "ZenReg_RegistrationTarget": "2D+t global XY translation plus rotation",
                "ZenReg_TimeShiftGT": "synthetic_2d_t_trans_rot_xy_time_shifts_gt.csv",
                "ZenReg_RotationGT": "synthetic_2d_t_trans_rot_xy_time_rotations_gt.csv",
            },
        },
        "synthetic_2d_t_piecewise_xy": {
            "stack": stack_2d_t_piecewise_xy,
            "image": "synthetic_2d_t_piecewise_xy.ome.tif",
            "piecewise_gt": "synthetic_2d_t_piecewise_xy_anchor_shifts_gt.csv",
            "annotations": {
                "ZenReg_SyntheticDataset": "synthetic_2d_t_piecewise_xy",
                "ZenReg_RegistrationTarget": "2D+t local piecewise XY translation without rotation",
                "ZenReg_PiecewiseAnchorShiftGT": "synthetic_2d_t_piecewise_xy_anchor_shifts_gt.csv",
            },
        },
        "synthetic_3d_t_trans_rot_z": {
            "stack": stack_3d_t_trans_rot_z,
            "image": "synthetic_3d_t_trans_rot_z.ome.tif",
            "rigid_gt": "synthetic_3d_t_trans_rot_z_rigid_transform_gt.csv",
            "annotations": {
                "ZenReg_SyntheticDataset": "synthetic_3d_t_trans_rot_z",
                "ZenReg_RegistrationTarget": "3D+t translations plus rotation around Z",
                "ZenReg_RigidTransformGT": "synthetic_3d_t_trans_rot_z_rigid_transform_gt.csv",
            },
        },
        "synthetic_3d_t_trans_rot_x": {
            "stack": stack_3d_t_trans_rot_x,
            "image": "synthetic_3d_t_trans_rot_x.ome.tif",
            "rigid_gt": "synthetic_3d_t_trans_rot_x_rigid_transform_gt.csv",
            "annotations": {
                "ZenReg_SyntheticDataset": "synthetic_3d_t_trans_rot_x",
                "ZenReg_RegistrationTarget": "3D+t translations plus rotation around X",
                "ZenReg_RigidTransformGT": "synthetic_3d_t_trans_rot_x_rigid_transform_gt.csv",
            },
        },
        "synthetic_3d_t_trans_rot_all_center": {
            "stack": stack_3d_t_trans_rot_all_center,
            "image": "synthetic_3d_t_trans_rot_all_center.ome.tif",
            "rigid_gt": "synthetic_3d_t_trans_rot_all_center_rigid_transform_gt.csv",
            "annotations": {
                "ZenReg_SyntheticDataset": "synthetic_3d_t_trans_rot_all_center",
                "ZenReg_RegistrationTarget": "3D+t translations plus all-axis rotations around center",
                "ZenReg_RigidTransformGT": "synthetic_3d_t_trans_rot_all_center_rigid_transform_gt.csv",
            },
        },
        "synthetic_3d_t_trans_rot_all_offcenter": {
            "stack": stack_3d_t_trans_rot_all_offcenter,
            "image": "synthetic_3d_t_trans_rot_all_offcenter.ome.tif",
            "rigid_gt": "synthetic_3d_t_trans_rot_all_offcenter_rigid_transform_gt.csv",
            "annotations": {
                "ZenReg_SyntheticDataset": "synthetic_3d_t_trans_rot_all_offcenter",
                "ZenReg_RegistrationTarget": "3D+t translations plus all-axis rotations around off-center point",
                "ZenReg_RigidTransformGT": "synthetic_3d_t_trans_rot_all_offcenter_rigid_transform_gt.csv",
            },
        },
        "synthetic_3d_t_trans_rot_all_outside": {
            "stack": stack_3d_t_trans_rot_all_outside,
            "image": "synthetic_3d_t_trans_rot_all_outside.ome.tif",
            "rigid_gt": "synthetic_3d_t_trans_rot_all_outside_rigid_transform_gt.csv",
            "annotations": {
                "ZenReg_SyntheticDataset": "synthetic_3d_t_trans_rot_all_outside",
                "ZenReg_RegistrationTarget": "3D+t translations plus all-axis rotations around outside point",
                "ZenReg_RigidTransformGT": "synthetic_3d_t_trans_rot_all_outside_rigid_transform_gt.csv",
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
    paths["synthetic_2d_t_rot_xy_time_gt_csv"] = str(
        _write_time_shift_table(
            output_dir / "synthetic_2d_t_rot_xy_time_shifts_gt.csv",
            shifts_2d_t_rot_xy,
        )
    )
    paths["synthetic_2d_t_rot_xy_rotation_gt_csv"] = str(
        _write_time_rotation_table(
            output_dir / "synthetic_2d_t_rot_xy_time_rotations_gt.csv",
            rotations_2d_t_rot,
        )
    )
    paths["synthetic_2d_t_local_gt_csv"] = str(
        _write_local_motion_table(output_dir / "synthetic_2d_t_local_motion_gt.csv", local_params_2d_t)
    )
    paths["synthetic_2d_t_trans_rot_xy_time_gt_csv"] = str(
        _write_time_shift_table(
            output_dir / "synthetic_2d_t_trans_rot_xy_time_shifts_gt.csv",
            shifts_2d_t_trans_rot,
        )
    )
    paths["synthetic_2d_t_trans_rot_xy_rotation_gt_csv"] = str(
        _write_time_rotation_table(
            output_dir / "synthetic_2d_t_trans_rot_xy_time_rotations_gt.csv",
            rotations_2d_t_trans_rot,
        )
    )
    paths["synthetic_2d_t_piecewise_xy_anchor_gt_csv"] = str(
        _write_piecewise_anchor_shift_table(
            output_dir / "synthetic_2d_t_piecewise_xy_anchor_shifts_gt.csv",
            anchor_shifts_2d_t_piecewise_xy,
        )
    )
    paths["synthetic_3d_t_trans_rot_z_rigid_gt_csv"] = str(
        _write_3d_rigid_transform_table(
            output_dir / "synthetic_3d_t_trans_rot_z_rigid_transform_gt.csv",
            shifts_zyx=shifts_3d_t_trans_rot_z,
            rotations_zyx_deg=rotations_3d_t_trans_rot_z,
            centers_zyx=centers_3d_t_trans_rot_z,
        )
    )
    paths["synthetic_3d_t_trans_rot_x_rigid_gt_csv"] = str(
        _write_3d_rigid_transform_table(
            output_dir / "synthetic_3d_t_trans_rot_x_rigid_transform_gt.csv",
            shifts_zyx=shifts_3d_t_trans_rot_x,
            rotations_zyx_deg=rotations_3d_t_trans_rot_x,
            centers_zyx=centers_3d_t_trans_rot_x,
        )
    )
    paths["synthetic_3d_t_trans_rot_all_center_rigid_gt_csv"] = str(
        _write_3d_rigid_transform_table(
            output_dir / "synthetic_3d_t_trans_rot_all_center_rigid_transform_gt.csv",
            shifts_zyx=shifts_3d_t_trans_rot_all_center,
            rotations_zyx_deg=rotations_3d_t_trans_rot_all_center,
            centers_zyx=centers_3d_t_trans_rot_all_center,
        )
    )
    paths["synthetic_3d_t_trans_rot_all_offcenter_rigid_gt_csv"] = str(
        _write_3d_rigid_transform_table(
            output_dir / "synthetic_3d_t_trans_rot_all_offcenter_rigid_transform_gt.csv",
            shifts_zyx=shifts_3d_t_trans_rot_all_offcenter,
            rotations_zyx_deg=rotations_3d_t_trans_rot_all_offcenter,
            centers_zyx=centers_3d_t_trans_rot_all_offcenter,
        )
    )
    paths["synthetic_3d_t_trans_rot_all_outside_rigid_gt_csv"] = str(
        _write_3d_rigid_transform_table(
            output_dir / "synthetic_3d_t_trans_rot_all_outside_rigid_transform_gt.csv",
            shifts_zyx=shifts_3d_t_trans_rot_all_outside,
            rotations_zyx_deg=rotations_3d_t_trans_rot_all_outside,
            centers_zyx=centers_3d_t_trans_rot_all_outside,
        )
    )

    metadata = {
        "axis_order": "TZCYX",
        "synthetic_2d_t_xy_shape": list(stack_2d_t_xy.shape),
        "synthetic_3d_z_xy_shape": list(stack_3d_z_xy.shape),
        "synthetic_3d_t_xy_shape": list(stack_3d_t_xy.shape),
        "synthetic_3d_t_intra_xy_shape": list(stack_3d_t_intra_xy.shape),
        "synthetic_3d_t_zyx_shape": list(stack_3d_t_zyx.shape),
        "synthetic_2d_t_rot_xy_shape": list(stack_2d_t_rot_xy.shape),
        "synthetic_2d_t_local_shape": list(stack_2d_t_local.shape),
        "synthetic_2d_t_trans_rot_xy_shape": list(stack_2d_t_trans_rot.shape),
        "synthetic_2d_t_piecewise_xy_shape": list(stack_2d_t_piecewise_xy.shape),
        "synthetic_3d_t_trans_rot_z_shape": list(stack_3d_t_trans_rot_z.shape),
        "synthetic_3d_t_trans_rot_x_shape": list(stack_3d_t_trans_rot_x.shape),
        "synthetic_3d_t_trans_rot_all_center_shape": list(stack_3d_t_trans_rot_all_center.shape),
        "synthetic_3d_t_trans_rot_all_offcenter_shape": list(stack_3d_t_trans_rot_all_offcenter.shape),
        "synthetic_3d_t_trans_rot_all_outside_shape": list(stack_3d_t_trans_rot_all_outside.shape),
        "synthetic_2d_t_xy_applied_time_shifts_yx": shifts_2d_t_xy.tolist(),
        "synthetic_3d_z_xy_applied_slice_shifts_yx": shifts_3d_z_xy.tolist(),
        "synthetic_3d_t_xy_applied_time_shifts_yx": shifts_3d_t_xy.tolist(),
        "synthetic_3d_t_intra_xy_applied_slice_shifts_yx": shifts_3d_t_intra_xy.tolist(),
        "synthetic_3d_t_zyx_applied_time_shifts_zyx": shifts_3d_t_zyx.tolist(),
        "synthetic_2d_t_rot_xy_applied_time_shifts_yx": shifts_2d_t_rot_xy.tolist(),
        "synthetic_2d_t_rot_xy_applied_rotations_deg": rotations_2d_t_rot.tolist(),
        "synthetic_2d_t_local_params": local_params_2d_t.tolist(),
        "synthetic_2d_t_trans_rot_xy_applied_time_shifts_yx": shifts_2d_t_trans_rot.tolist(),
        "synthetic_2d_t_trans_rot_xy_applied_rotations_deg": rotations_2d_t_trans_rot.tolist(),
        "synthetic_2d_t_piecewise_xy_anchor_shifts_yx": anchor_shifts_2d_t_piecewise_xy.tolist(),
        "synthetic_3d_t_trans_rot_z_applied_time_shifts_zyx": shifts_3d_t_trans_rot_z.tolist(),
        "synthetic_3d_t_trans_rot_z_applied_rotations_zyx_deg": rotations_3d_t_trans_rot_z.tolist(),
        "synthetic_3d_t_trans_rot_z_centers_zyx": centers_3d_t_trans_rot_z.tolist(),
        "synthetic_3d_t_trans_rot_x_applied_time_shifts_zyx": shifts_3d_t_trans_rot_x.tolist(),
        "synthetic_3d_t_trans_rot_x_applied_rotations_zyx_deg": rotations_3d_t_trans_rot_x.tolist(),
        "synthetic_3d_t_trans_rot_x_centers_zyx": centers_3d_t_trans_rot_x.tolist(),
        "synthetic_3d_t_trans_rot_all_center_applied_time_shifts_zyx": shifts_3d_t_trans_rot_all_center.tolist(),
        "synthetic_3d_t_trans_rot_all_center_applied_rotations_zyx_deg": rotations_3d_t_trans_rot_all_center.tolist(),
        "synthetic_3d_t_trans_rot_all_center_centers_zyx": centers_3d_t_trans_rot_all_center.tolist(),
        "synthetic_3d_t_trans_rot_all_offcenter_applied_time_shifts_zyx": shifts_3d_t_trans_rot_all_offcenter.tolist(),
        "synthetic_3d_t_trans_rot_all_offcenter_applied_rotations_zyx_deg": rotations_3d_t_trans_rot_all_offcenter.tolist(),
        "synthetic_3d_t_trans_rot_all_offcenter_centers_zyx": centers_3d_t_trans_rot_all_offcenter.tolist(),
        "synthetic_3d_t_trans_rot_all_outside_applied_time_shifts_zyx": shifts_3d_t_trans_rot_all_outside.tolist(),
        "synthetic_3d_t_trans_rot_all_outside_applied_rotations_zyx_deg": rotations_3d_t_trans_rot_all_outside.tolist(),
        "synthetic_3d_t_trans_rot_all_outside_centers_zyx": centers_3d_t_trans_rot_all_outside.tolist(),
    }
    metadata_path = output_dir / "synthetic_motion_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    paths["metadata"] = str(metadata_path)
    return paths
