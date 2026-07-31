"""
Rigid 3D volume registration for canonical ``TZCYX`` stacks.

This module estimates and applies 6-DOF rigid transforms for full 3D volumes:
translation in ``ZYX`` and rotations around ``Z/Y/X``. Dense intensity
registration uses SimpleITK. A lightweight point-cloud backend is provided for
sparse puncta-style benchmarks and can be expanded later.

Author: Fabrizio Musacchio
Date: July 2026
"""
# %% IMPORTS
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.ndimage import affine_transform
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from skimage.feature import peak_local_max
from skimage.registration import phase_cross_correlation
from skimage.transform import warp_polar

from ._axes import CANONICAL_AXIS_ORDER, ensure_tzcyx_stack, normalize_zrange
# %% CONSTANTS
SUPPORTED_RIGID_3D_BACKENDS = {"phase_cross_correlation", "simpleitk", "points"}
SUPPORTED_RIGID_3D_METRICS = {"correlation", "mutual_information"}

# %% DATA CLASSES
@dataclass(frozen=True)
class Rigid3DTransform:
    """Rigid transform represented as moving-to-fixed correction in ZYX coordinates."""

    matrix_zyx: np.ndarray
    translation_zyx: np.ndarray
    center_zyx: np.ndarray

# %% HELPER FUNCTIONS
def normalize_rigid_3d_backend(rigid_3d_backend: str) -> str:
    """Normalize and validate the 3D rigid backend name."""

    normalized = str(rigid_3d_backend).strip().lower()
    if normalized not in SUPPORTED_RIGID_3D_BACKENDS:
        raise ValueError(
            f"Unsupported rigid_3d_backend {rigid_3d_backend!r}. "
            f"Supported backends: {sorted(SUPPORTED_RIGID_3D_BACKENDS)}."
        )
    return normalized

def normalize_rigid_3d_metric(metric: str) -> str:
    """Normalize and validate the dense 3D registration metric."""

    normalized = str(metric).strip().lower()
    if normalized not in SUPPORTED_RIGID_3D_METRICS:
        raise ValueError(
            f"Unsupported rot_metric {metric!r}. "
            f"Supported metrics: {sorted(SUPPORTED_RIGID_3D_METRICS)}."
        )
    return normalized

def _normalize_spacing_zyx(spacing_zyx) -> tuple[float, float, float]:
    """Normalize physical spacing in Z/Y/X order."""

    if spacing_zyx is None:
        return (1.0, 1.0, 1.0)
    if len(spacing_zyx) != 3:
        raise ValueError("rot_spacing_zyx must contain exactly three values: (z, y, x).")
    spacing = tuple(float(v) for v in spacing_zyx)
    if any(v <= 0 for v in spacing):
        raise ValueError(f"rot_spacing_zyx values must be > 0. Got {spacing_zyx!r}.")
    return spacing

def _normalize_vector_zyx(value, *, default: tuple[float, float, float], name: str) -> tuple[float, float, float]:
    """Normalize a scalar/sequence to a 3-value ZYX tuple."""

    if value is None:
        return default
    if np.isscalar(value):
        values = (float(value),) * 3
    else:
        if len(value) != 3:
            raise ValueError(f"{name} must be a scalar or contain exactly three values.")
        values = tuple(float(v) for v in value)
    return values

def _as_float32_stack(stack) -> np.ndarray:
    """Return a float32 ``TZCYX`` stack view/copy."""

    stack = ensure_tzcyx_stack(stack)
    try:
        return stack.astype(np.float32, copy=False)
    except (AttributeError, TypeError):
        return np.asarray(stack, dtype=np.float32)

def _create_registered_output(
    shape: tuple[int, int, int, int, int],
    *,
    dtype,
    output_use_memmap: bool,
    output_memmap_folder: str | os.PathLike | None,
    output_memmap_name: str | None,
):
    """Allocate a registered rigid-3D output in RAM or as an OMIO/Zarr array."""

    if output_use_memmap:
        from .io import create_empty_stack

        return create_empty_stack(
            shape=tuple(int(v) for v in shape),
            dtype=np.dtype(dtype),
            fill_value=0,
            use_memmap=True,
            memmap_folder=output_memmap_folder,
            memmap_name=output_memmap_name,
            return_metadata=False,
            verbose=False,
        )
    return np.empty(tuple(int(v) for v in shape), dtype=np.dtype(dtype))

def _project(volume_zyx: np.ndarray, *, axis: int, method: str = "max") -> np.ndarray:
    """Project a ``ZYX`` volume along one axis."""

    if method == "mean":
        return np.mean(volume_zyx, axis=axis)
    if method == "median":
        return np.median(volume_zyx, axis=axis)
    if method == "var":
        return np.var(volume_zyx, axis=axis)
    if method == "std":
        return np.std(volume_zyx, axis=axis)
    return np.max(volume_zyx, axis=axis)

def _normalize_rotation_image(image: np.ndarray) -> np.ndarray:
    """Normalize a 2D image for polar phase-correlation rotation estimation."""

    image = np.asarray(image, dtype=np.float32)
    image = image - float(np.nanmin(image))
    max_value = float(np.nanmax(image))
    if max_value > 0:
        image = image / max_value
    return image

def _estimate_rotation_deg_2d(
    fixed_projection: np.ndarray,
    moving_projection: np.ndarray,
    *,
    upsample_factor: int,
    normalization: str | None,
    max_angle_deg: float | None,
) -> float:
    """Estimate a 2D in-plane correction angle from polar phase correlation."""

    fixed_projection = _normalize_rotation_image(fixed_projection)
    moving_projection = _normalize_rotation_image(moving_projection)
    radius = max(8, min(fixed_projection.shape) // 2)
    fixed_polar = warp_polar(fixed_projection, radius=radius)
    moving_polar = warp_polar(moving_projection, radius=radius)
    shift, _, _ = phase_cross_correlation(
        fixed_polar,
        moving_polar,
        upsample_factor=int(upsample_factor),
        normalization=normalization,
    )
    angle_deg = -float(shift[0]) * 360.0 / float(fixed_polar.shape[0])
    angle_deg = ((angle_deg + 180.0) % 360.0) - 180.0
    if max_angle_deg is not None:
        angle_deg = float(np.clip(angle_deg, -float(max_angle_deg), float(max_angle_deg)))
    return float(angle_deg)

def _rotation_matrix_from_correction_zyx(rotation_zyx_deg: np.ndarray) -> np.ndarray:
    """Return an active ZYX correction rotation matrix in array coordinates."""

    return Rotation.from_euler(
        "ZYX",
        [float(rotation_zyx_deg[0]), float(rotation_zyx_deg[1]), float(rotation_zyx_deg[2])],
        degrees=True,
    ).as_matrix().astype(np.float64)

def _rotation_zyx_from_matrix(matrix_zyx: np.ndarray) -> np.ndarray:
    """Extract Z/Y/X Euler angles in degrees from an array-coordinate matrix."""

    return Rotation.from_matrix(np.asarray(matrix_zyx, dtype=np.float64)).as_euler("ZYX", degrees=True).astype(np.float32)

def _apply_rigid_transform_to_volume(
    volume_zyx: np.ndarray,
    transform: Rigid3DTransform,
    *,
    order: int,
    cval: float,
) -> np.ndarray:
    """Apply a moving-to-fixed correction transform to a ZYX volume."""

    matrix = np.asarray(transform.matrix_zyx, dtype=np.float64)
    translation = np.asarray(transform.translation_zyx, dtype=np.float64)
    center = np.asarray(transform.center_zyx, dtype=np.float64)
    inverse = matrix.T
    offset = center - inverse @ (center + translation)
    return affine_transform(
        np.asarray(volume_zyx, dtype=np.float32),
        inverse,
        offset=offset,
        order=int(order),
        mode="constant",
        cval=float(cval),
        prefilter=int(order) > 1,
    ).astype(np.float32, copy=False)

def _phase_correlation_shift_zyx(
    fixed: np.ndarray,
    moving: np.ndarray,
    *,
    upsample_factor: int,
    normalization: str | None,
) -> np.ndarray:
    """Estimate ZYX correction shift with phase cross-correlation."""

    shift, _, _ = phase_cross_correlation(
        np.asarray(fixed, dtype=np.float32),
        np.asarray(moving, dtype=np.float32),
        upsample_factor=int(upsample_factor),
        normalization=normalization,
    )
    return np.asarray(shift, dtype=np.float32)

def estimate_initial_rotation_from_projections(
    fixed_zyx: np.ndarray,
    moving_zyx: np.ndarray,
    *,
    projection_method: str,
    iterations: int,
    max_angle_deg: float | None,
    upsample_factor: int,
    normalization: str | None,
    transform_order: int,
) -> tuple[np.ndarray, Rigid3DTransform]:
    """Estimate an iterated coarse 3D rotation correction from orthogonal projections."""

    fixed_zyx = np.asarray(fixed_zyx, dtype=np.float32)
    moving_current = np.asarray(moving_zyx, dtype=np.float32)
    center = (np.asarray(fixed_zyx.shape, dtype=np.float64) - 1.0) / 2.0
    total_matrix = np.eye(3, dtype=np.float64)

    for _ in range(max(int(iterations), 0)):
        rot_z = _estimate_rotation_deg_2d(
            _project(fixed_zyx, axis=0, method=projection_method),
            _project(moving_current, axis=0, method=projection_method),
            upsample_factor=upsample_factor,
            normalization=normalization,
            max_angle_deg=max_angle_deg,
        )
        rot_x = _estimate_rotation_deg_2d(
            _project(fixed_zyx, axis=2, method=projection_method),
            _project(moving_current, axis=2, method=projection_method),
            upsample_factor=upsample_factor,
            normalization=normalization,
            max_angle_deg=max_angle_deg,
        )
        rot_y = _estimate_rotation_deg_2d(
            _project(fixed_zyx, axis=1, method=projection_method),
            _project(moving_current, axis=1, method=projection_method),
            upsample_factor=upsample_factor,
            normalization=normalization,
            max_angle_deg=max_angle_deg,
        )
        step_matrix = _rotation_matrix_from_correction_zyx(np.asarray([rot_z, rot_y, rot_x], dtype=np.float32))
        step_transform = Rigid3DTransform(
            matrix_zyx=step_matrix,
            translation_zyx=np.zeros(3, dtype=np.float64),
            center_zyx=center,
        )
        moving_current = _apply_rigid_transform_to_volume(
            moving_current,
            step_transform,
            order=transform_order,
            cval=0.0,
        )
        total_matrix = step_matrix @ total_matrix

    transform = Rigid3DTransform(
        matrix_zyx=total_matrix,
        translation_zyx=np.zeros(3, dtype=np.float64),
        center_zyx=center,
    )
    return _rotation_zyx_from_matrix(total_matrix), transform

def _compose_correction_transforms(first: Rigid3DTransform, second: Rigid3DTransform) -> Rigid3DTransform:
    """Compose two moving-to-fixed correction transforms as ``second(first(x))``."""

    center = np.asarray(first.center_zyx, dtype=np.float64)
    r1 = np.asarray(first.matrix_zyx, dtype=np.float64)
    t1 = np.asarray(first.translation_zyx, dtype=np.float64)
    r2 = np.asarray(second.matrix_zyx, dtype=np.float64)
    t2 = np.asarray(second.translation_zyx, dtype=np.float64)
    matrix = r2 @ r1
    translation = r2 @ t1 + t2
    return Rigid3DTransform(matrix_zyx=matrix, translation_zyx=translation, center_zyx=center)

def _sitk_image_from_zyx(volume_zyx: np.ndarray, spacing_zyx: tuple[float, float, float]):
    """Create a SimpleITK image from a ZYX NumPy volume."""

    import SimpleITK as sitk

    image = sitk.GetImageFromArray(np.asarray(volume_zyx, dtype=np.float32))
    image.SetSpacing((float(spacing_zyx[2]), float(spacing_zyx[1]), float(spacing_zyx[0])))
    return image

def _physical_xyz_from_zyx_shift(shift_zyx: np.ndarray, spacing_zyx: tuple[float, float, float]) -> np.ndarray:
    """Convert a ZYX pixel shift to XYZ physical units."""

    shift_zyx = np.asarray(shift_zyx, dtype=np.float64)
    return np.asarray(
        [
            shift_zyx[2] * float(spacing_zyx[2]),
            shift_zyx[1] * float(spacing_zyx[1]),
            shift_zyx[0] * float(spacing_zyx[0]),
        ],
        dtype=np.float64,
    )

def _zyx_shift_from_physical_xyz(translation_xyz: np.ndarray, spacing_zyx: tuple[float, float, float]) -> np.ndarray:
    """Convert an XYZ physical translation to ZYX pixels."""

    translation_xyz = np.asarray(translation_xyz, dtype=np.float64)
    return np.asarray(
        [
            translation_xyz[2] / float(spacing_zyx[0]),
            translation_xyz[1] / float(spacing_zyx[1]),
            translation_xyz[0] / float(spacing_zyx[2]),
        ],
        dtype=np.float32,
    )

def _permutation_zyx_to_xyz() -> np.ndarray:
    """Return permutation matrix mapping ZYX coordinate vectors to XYZ."""

    return np.asarray([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.float64)

def _zyx_matrix_to_xyz(matrix_zyx: np.ndarray) -> np.ndarray:
    """Convert an array-coordinate ZYX matrix to physical XYZ axis order."""

    p = _permutation_zyx_to_xyz()
    return p @ np.asarray(matrix_zyx, dtype=np.float64) @ p.T

def _xyz_matrix_to_zyx(matrix_xyz: np.ndarray) -> np.ndarray:
    """Convert a physical XYZ matrix to array-coordinate ZYX axis order."""

    p = _permutation_zyx_to_xyz()
    return p.T @ np.asarray(matrix_xyz, dtype=np.float64) @ p

def _effective_affine_from_sitk_transform(transform) -> tuple[np.ndarray, np.ndarray]:
    """Return matrix/translation for a SimpleITK transform in XYZ physical coordinates."""

    import SimpleITK as sitk

    if isinstance(transform, sitk.CompositeTransform):
        if transform.GetNumberOfTransforms() < 1:
            raise RuntimeError("SimpleITK returned an empty CompositeTransform.")
        transform = transform.GetNthTransform(transform.GetNumberOfTransforms() - 1)

    affine = sitk.AffineTransform(3)
    affine.SetCenter(transform.GetCenter())
    affine.SetMatrix(transform.GetMatrix())
    affine.SetTranslation(transform.GetTranslation())
    center = np.asarray(affine.GetCenter(), dtype=np.float64)
    matrix = np.asarray(affine.GetMatrix(), dtype=np.float64).reshape(3, 3)
    translation = center + np.asarray(affine.GetTranslation(), dtype=np.float64) - matrix @ center
    return matrix, translation

def _simpleitk_transform_from_correction(
    transform: Rigid3DTransform,
    *,
    spacing_zyx: tuple[float, float, float],
):
    """Create a SimpleITK fixed-to-moving transform from a ZYX correction transform."""

    import SimpleITK as sitk

    matrix_corr_xyz = _zyx_matrix_to_xyz(transform.matrix_zyx)
    translation_corr_xyz = _physical_xyz_from_zyx_shift(transform.translation_zyx, spacing_zyx)
    center_xyz = _physical_xyz_from_zyx_shift(transform.center_zyx, spacing_zyx)
    effective_translation_corr_xyz = center_xyz + translation_corr_xyz - matrix_corr_xyz @ center_xyz
    inv_matrix = matrix_corr_xyz.T
    inv_translation = -inv_matrix @ effective_translation_corr_xyz
    initial = sitk.VersorRigid3DTransform()
    initial.SetCenter(tuple(float(v) for v in center_xyz))
    initial.SetMatrix(tuple(float(v) for v in inv_matrix.ravel()))
    initial.SetTranslation(tuple(float(v) for v in (inv_translation - center_xyz + inv_matrix @ center_xyz)))
    return initial

def _correction_from_simpleitk_transform(
    sitk_transform,
    *,
    spacing_zyx: tuple[float, float, float],
    center_zyx: np.ndarray,
) -> Rigid3DTransform:
    """Convert a SimpleITK fixed-to-moving transform to moving-to-fixed correction."""

    matrix_map_xyz, translation_map_xyz = _effective_affine_from_sitk_transform(sitk_transform)
    matrix_corr_xyz = matrix_map_xyz.T
    effective_translation_corr_xyz = -matrix_corr_xyz @ translation_map_xyz
    center_xyz = _physical_xyz_from_zyx_shift(center_zyx, spacing_zyx)
    translation_corr_xyz = effective_translation_corr_xyz + matrix_corr_xyz @ center_xyz - center_xyz
    return Rigid3DTransform(
        matrix_zyx=_xyz_matrix_to_zyx(matrix_corr_xyz),
        translation_zyx=_zyx_shift_from_physical_xyz(translation_corr_xyz, spacing_zyx),
        center_zyx=np.asarray(center_zyx, dtype=np.float64),
    )

def _register_simpleitk(
    fixed_zyx: np.ndarray,
    moving_zyx: np.ndarray,
    *,
    initial_transform: Rigid3DTransform,
    spacing_zyx: tuple[float, float, float],
    metric: str,
    shrink_factors: tuple[int, ...],
    smoothing_sigmas: tuple[float, ...],
    iterations: int,
    learning_rate: float,
    min_step: float,
    sampling_percentage: float | None,
) -> Rigid3DTransform:
    """Refine a 6-DOF rigid transform with SimpleITK."""

    import SimpleITK as sitk

    fixed = _sitk_image_from_zyx(fixed_zyx, spacing_zyx)
    moving = _sitk_image_from_zyx(moving_zyx, spacing_zyx)
    initial = _simpleitk_transform_from_correction(initial_transform, spacing_zyx=spacing_zyx)

    registration = sitk.ImageRegistrationMethod()
    if metric == "mutual_information":
        registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    else:
        registration.SetMetricAsCorrelation()
    if sampling_percentage is not None and float(sampling_percentage) < 1.0:
        registration.SetMetricSamplingStrategy(registration.RANDOM)
        registration.SetMetricSamplingPercentage(float(sampling_percentage))
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsRegularStepGradientDescent(
        learningRate=float(learning_rate),
        minStep=float(min_step),
        numberOfIterations=int(iterations),
        gradientMagnitudeTolerance=1e-8,
    )
    registration.SetOptimizerScalesFromPhysicalShift()
    registration.SetShrinkFactorsPerLevel([int(v) for v in shrink_factors])
    registration.SetSmoothingSigmasPerLevel([float(v) for v in smoothing_sigmas])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration.SetInitialTransform(initial, inPlace=False)
    final_transform = registration.Execute(fixed, moving)
    return _correction_from_simpleitk_transform(
        final_transform,
        spacing_zyx=spacing_zyx,
        center_zyx=initial_transform.center_zyx,
    )

def _detect_points(volume_zyx: np.ndarray, *, max_points: int, min_distance: int, threshold_rel: float) -> np.ndarray:
    """Detect sparse bright points in a ZYX volume."""

    volume = np.asarray(volume_zyx, dtype=np.float32)
    threshold_abs = float(np.nanmin(volume) + float(threshold_rel) * (np.nanmax(volume) - np.nanmin(volume)))
    points = peak_local_max(
        volume,
        min_distance=max(int(min_distance), 1),
        threshold_abs=threshold_abs,
        num_peaks=int(max_points),
        exclude_border=False,
    )
    return np.asarray(points, dtype=np.float64)

def _rigid_kabsch(moving_points: np.ndarray, fixed_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Estimate rigid moving-to-fixed transform with Kabsch."""

    moving_centroid = np.mean(moving_points, axis=0)
    fixed_centroid = np.mean(fixed_points, axis=0)
    moving_centered = moving_points - moving_centroid
    fixed_centered = fixed_points - fixed_centroid
    h = moving_centered.T @ fixed_centered
    u, _, vt = np.linalg.svd(h)
    matrix = vt.T @ u.T
    if np.linalg.det(matrix) < 0:
        vt[-1, :] *= -1
        matrix = vt.T @ u.T
    translation = fixed_centroid - matrix @ moving_centroid
    return matrix, translation

def _register_points(
    fixed_zyx: np.ndarray,
    moving_zyx: np.ndarray,
    *,
    initial_transform: Rigid3DTransform,
    max_points: int,
    min_distance: int,
    threshold_rel: float,
    iterations: int,
    max_match_distance: float,
) -> Rigid3DTransform:
    """Estimate a rigid transform from sparse point clouds."""

    fixed_points = _detect_points(
        fixed_zyx,
        max_points=max_points,
        min_distance=min_distance,
        threshold_rel=threshold_rel,
    )
    moving_points = _detect_points(
        moving_zyx,
        max_points=max_points,
        min_distance=min_distance,
        threshold_rel=threshold_rel,
    )
    if len(fixed_points) < 4 or len(moving_points) < 4:
        raise RuntimeError("points backend needs at least four detected points in both fixed and moving volumes.")

    tree = cKDTree(fixed_points)
    current = initial_transform
    for _ in range(max(int(iterations), 1)):
        moved = (current.matrix_zyx @ (moving_points - current.center_zyx).T).T + current.center_zyx + current.translation_zyx
        distances, indices = tree.query(moved, k=1)
        keep = distances <= float(max_match_distance)
        if int(np.sum(keep)) < 4:
            keep = np.argsort(distances)[: min(len(distances), max(4, len(distances) // 2))]
            source = moving_points[keep]
            target = fixed_points[indices[keep]]
        else:
            source = moving_points[keep]
            target = fixed_points[indices[keep]]
        matrix, translation = _rigid_kabsch(source, target)
        current = Rigid3DTransform(
            matrix_zyx=matrix,
            translation_zyx=translation,
            center_zyx=np.zeros(3, dtype=np.float64),
        )
    center = (np.asarray(fixed_zyx.shape, dtype=np.float64) - 1.0) / 2.0
    center_translation = current.matrix_zyx @ center + current.translation_zyx - center
    return Rigid3DTransform(
        matrix_zyx=current.matrix_zyx,
        translation_zyx=center_translation,
        center_zyx=center,
    )

def _process_timepoint(
    t: int,
    stack: np.ndarray,
    *,
    fixed_zyx: np.ndarray,
    registration_channel: int,
    registration_stack: int,
    projection_range: tuple[int, int] | None,
    backend: str,
    projection_method: str,
    spacing_zyx: tuple[float, float, float],
    init_iterations: int,
    max_rot_shifts: float | None,
    upsample_factor: int,
    normalization: str | None,
    metric: str,
    shrink_factors: tuple[int, ...],
    smoothing_sigmas: tuple[float, ...],
    iterations: int,
    learning_rate: float,
    min_step: float,
    sampling_percentage: float | None,
    transform_order: int,
    cval: float,
    points_max_points: int,
    points_min_distance: int,
    points_threshold_rel: float,
    points_iterations: int,
    points_max_match_distance: float,
    return_valid_mask: bool,
) -> tuple[int, np.ndarray, np.ndarray | None, dict[str, Any]]:
    """Estimate and apply one 3D rigid transform for a time point."""

    moving_full = np.asarray(stack[t, :, registration_channel, :, :], dtype=np.float32)
    moving_for_estimation = moving_full
    if projection_range is not None:
        z0, z1 = projection_range
        moving_for_estimation = moving_full[int(z0) : int(z1), :, :]

    if t == int(registration_stack):
        center = (np.asarray(moving_for_estimation.shape, dtype=np.float64) - 1.0) / 2.0
        final_transform = Rigid3DTransform(np.eye(3), np.zeros(3), center)
        initial_shift = np.zeros(3, dtype=np.float32)
        initial_rotation = np.zeros(3, dtype=np.float32)
    else:
        initial_shift = _phase_correlation_shift_zyx(
            fixed_zyx,
            moving_for_estimation,
            upsample_factor=upsample_factor,
            normalization=normalization,
        )
        initial_rotation, rotation_transform = estimate_initial_rotation_from_projections(
            fixed_zyx,
            moving_for_estimation,
            projection_method=projection_method,
            iterations=init_iterations,
            max_angle_deg=max_rot_shifts,
            upsample_factor=upsample_factor,
            normalization=normalization,
            transform_order=transform_order,
        )
        translation_transform = Rigid3DTransform(
            matrix_zyx=np.eye(3),
            translation_zyx=np.asarray(initial_shift, dtype=np.float64),
            center_zyx=rotation_transform.center_zyx,
        )
        initial_transform = _compose_correction_transforms(rotation_transform, translation_transform)
        if backend == "simpleitk":
            final_transform = _register_simpleitk(
                fixed_zyx,
                moving_for_estimation,
                initial_transform=initial_transform,
                spacing_zyx=spacing_zyx,
                metric=metric,
                shrink_factors=shrink_factors,
                smoothing_sigmas=smoothing_sigmas,
                iterations=iterations,
                learning_rate=learning_rate,
                min_step=min_step,
                sampling_percentage=sampling_percentage,
            )
        elif backend == "points":
            final_transform = _register_points(
                fixed_zyx,
                moving_for_estimation,
                initial_transform=initial_transform,
                max_points=points_max_points,
                min_distance=points_min_distance,
                threshold_rel=points_threshold_rel,
                iterations=points_iterations,
                max_match_distance=points_max_match_distance,
            )
        else:
            final_transform = initial_transform

    registered_frame = np.empty(
        (stack.shape[1], stack.shape[2], stack.shape[3], stack.shape[4]),
        dtype=np.float32,
    )
    for c in range(stack.shape[2]):
        registered_frame[:, c, :, :] = _apply_rigid_transform_to_volume(
            stack[t, :, c, :, :],
            final_transform,
            order=transform_order,
            cval=cval,
        )
    valid_mask = None
    if return_valid_mask:
        valid_mask = _apply_rigid_transform_to_volume(
            np.ones((stack.shape[1], stack.shape[3], stack.shape[4]), dtype=np.float32),
            final_transform,
            order=1,
            cval=0.0,
        )

    rotation_zyx = _rotation_zyx_from_matrix(final_transform.matrix_zyx)
    details = {
        "initial_shift_zyx": np.asarray(initial_shift, dtype=np.float32),
        "initial_rotation_zyx_deg": np.asarray(initial_rotation, dtype=np.float32),
        "shift_zyx": np.asarray(final_transform.translation_zyx, dtype=np.float32),
        "rotation_zyx_deg": rotation_zyx,
        "matrix_zyx": np.asarray(final_transform.matrix_zyx, dtype=np.float32),
        "center_zyx": np.asarray(final_transform.center_zyx, dtype=np.float32),
    }
    return t, registered_frame, valid_mask, details

def register_stack_rigid_3d(
    stack,
    *,
    registration_channel: int = 0,
    registration_stack: int = 0,
    backend: str = "simpleitk",
    projection_range: tuple[int, int] | None = None,
    projection_method: str = "max",
    spacing_zyx=None,
    init_iterations: int = 1,
    max_rot_shifts: float | None = None,
    upsample_factor: int = 10,
    normalization: str | None = None,
    metric: str = "correlation",
    shrink_factors=(4, 2, 1),
    smoothing_sigmas=(2.0, 1.0, 0.0),
    iterations: int = 100,
    learning_rate: float = 1.0,
    min_step: float = 1e-4,
    sampling_percentage: float | None = None,
    transform_order: int = 1,
    cval: float = 0.0,
    points_max_points: int = 200,
    points_min_distance: int = 3,
    points_threshold_rel: float = 0.25,
    points_iterations: int = 20,
    points_max_match_distance: float = 8.0,
    n_jobs: int = 1,
    output_use_memmap: bool = False,
    output_memmap_folder: str | os.PathLike | None = None,
    output_memmap_name: str | None = "zenreg_rigid_3d_registered",
    output_dtype=np.float32,
    return_valid_mask: bool = False,
    verbose: bool = True,
):
    """Register a ``TZCYX`` stack with a full 6-DOF rigid 3D transform per time point."""

    stack = ensure_tzcyx_stack(stack)
    if stack.ndim != 5:
        raise ValueError(f"Expected a 5D {CANONICAL_AXIS_ORDER} stack. Got shape {stack.shape!r}.")
    output_dtype = np.dtype(output_dtype)
    if not np.issubdtype(output_dtype, np.floating):
        raise ValueError("Rigid 3D registration output_dtype must be a floating dtype.")
    if not output_use_memmap and output_memmap_folder is not None:
        raise ValueError("output_memmap_folder requires output_use_memmap=True.")
    if stack.shape[1] < 2:
        raise ValueError("Rigid 3D registration requires SizeZ >= 2.")
    backend = normalize_rigid_3d_backend(backend)
    metric = normalize_rigid_3d_metric(metric)
    spacing_zyx = _normalize_spacing_zyx(spacing_zyx)
    if not 0 <= int(registration_channel) < stack.shape[2]:
        raise ValueError(f"registration_channel must be between 0 and {stack.shape[2] - 1}.")
    registration_stack = int(registration_stack)
    if not 0 <= registration_stack < stack.shape[0]:
        raise ValueError(f"registration_stack must be between 0 and {stack.shape[0] - 1}.")
    projection_range = None if projection_range is None else normalize_zrange(projection_range, stack.shape[1], strict=True)
    if projection_range is None:
        fixed_zyx = np.asarray(stack[registration_stack, :, registration_channel, :, :], dtype=np.float32)
    else:
        z0, z1 = projection_range
        fixed_zyx = np.asarray(stack[registration_stack, int(z0) : int(z1), registration_channel, :, :], dtype=np.float32)
    shrink_factors = tuple(int(v) for v in shrink_factors)
    smoothing_sigmas = tuple(float(v) for v in smoothing_sigmas)
    if len(shrink_factors) != len(smoothing_sigmas):
        raise ValueError("rot_shrink_factors and rot_smoothing_sigmas must have the same length.")

    registered = _create_registered_output(
        tuple(int(v) for v in stack.shape),
        dtype=output_dtype,
        output_use_memmap=bool(output_use_memmap),
        output_memmap_folder=output_memmap_folder,
        output_memmap_name=output_memmap_name,
    )
    valid_mask_tzyx = (
        np.empty((stack.shape[0], stack.shape[1], stack.shape[3], stack.shape[4]), dtype=np.float32)
        if return_valid_mask
        else None
    )
    transform_details: list[dict[str, Any] | None] = [None] * stack.shape[0]

    if verbose:
        print(f"ZenReg rigid 3D registration: backend={backend}, n_jobs={int(n_jobs)}")

    worker_kwargs = {
        "fixed_zyx": fixed_zyx,
        "registration_channel": int(registration_channel),
        "registration_stack": int(registration_stack),
        "projection_range": projection_range,
        "backend": backend,
        "projection_method": projection_method,
        "spacing_zyx": spacing_zyx,
        "init_iterations": int(init_iterations),
        "max_rot_shifts": max_rot_shifts,
        "upsample_factor": int(upsample_factor),
        "normalization": normalization,
        "metric": metric,
        "shrink_factors": shrink_factors,
        "smoothing_sigmas": smoothing_sigmas,
        "iterations": int(iterations),
        "learning_rate": float(learning_rate),
        "min_step": float(min_step),
        "sampling_percentage": sampling_percentage,
        "transform_order": int(transform_order),
        "cval": float(cval),
        "points_max_points": int(points_max_points),
        "points_min_distance": int(points_min_distance),
        "points_threshold_rel": float(points_threshold_rel),
        "points_iterations": int(points_iterations),
        "points_max_match_distance": float(points_max_match_distance),
        "return_valid_mask": bool(return_valid_mask),
    }
    if int(n_jobs) <= 1:
        result_iter = (_process_timepoint(t, stack, **worker_kwargs) for t in range(stack.shape[0]))
        for t, registered_frame, valid_mask, details in result_iter:
            registered[int(t)] = registered_frame.astype(output_dtype, copy=False)
            if return_valid_mask and valid_mask_tzyx is not None:
                valid_mask_tzyx[int(t)] = valid_mask
            transform_details[int(t)] = details
            if verbose:
                shift = details["shift_zyx"]
                rotation = details["rotation_zyx_deg"]
                print(
                    f"t={int(t)} shift_zyx=({shift[0]:.3f}, {shift[1]:.3f}, {shift[2]:.3f}) "
                    f"rot_zyx_deg=({rotation[0]:.3f}, {rotation[1]:.3f}, {rotation[2]:.3f})"
                )
    else:
        with ThreadPoolExecutor(max_workers=int(n_jobs)) as executor:
            for t, registered_frame, valid_mask, details in executor.map(
                lambda index: _process_timepoint(index, stack, **worker_kwargs),
                range(stack.shape[0]),
            ):
                registered[int(t)] = registered_frame.astype(output_dtype, copy=False)
                if return_valid_mask and valid_mask_tzyx is not None:
                    valid_mask_tzyx[int(t)] = valid_mask
                transform_details[int(t)] = details
                if verbose:
                    shift = details["shift_zyx"]
                    rotation = details["rotation_zyx_deg"]
                    print(
                        f"t={int(t)} shift_zyx=({shift[0]:.3f}, {shift[1]:.3f}, {shift[2]:.3f}) "
                        f"rot_zyx_deg=({rotation[0]:.3f}, {rotation[1]:.3f}, {rotation[2]:.3f})"
                    )

    shifts_zyx = np.stack([details["shift_zyx"] for details in transform_details], axis=0).astype(np.float32)
    rotations_zyx_deg = np.stack(
        [details["rotation_zyx_deg"] for details in transform_details],
        axis=0,
    ).astype(np.float32)
    matrices_zyx = np.stack([details["matrix_zyx"] for details in transform_details], axis=0).astype(np.float32)
    centers_zyx = np.stack([details["center_zyx"] for details in transform_details], axis=0).astype(np.float32)
    initial_shifts_zyx = np.stack([details["initial_shift_zyx"] for details in transform_details], axis=0).astype(np.float32)
    initial_rotations_zyx_deg = np.stack(
        [details["initial_rotation_zyx_deg"] for details in transform_details],
        axis=0,
    ).astype(np.float32)

    details = {
        "method": "rigid_3d",
        "rigid_3d_backend": backend,
        "time_shifts_zyx": shifts_zyx,
        "time_shifts_yx": shifts_zyx[:, 1:],
        "rotation_shifts_zyx_deg": rotations_zyx_deg,
        "rotation_shifts_deg": rotations_zyx_deg[:, 0],
        "rigid_3d_matrices_zyx": matrices_zyx,
        "rigid_3d_centers_zyx": centers_zyx,
        "rigid_3d_initial_shifts_zyx": initial_shifts_zyx,
        "rigid_3d_initial_rotations_zyx_deg": initial_rotations_zyx_deg,
        "rot_spacing_zyx": spacing_zyx,
        "rot_metric": metric,
        "rot_init_iterations": int(init_iterations),
        "rot_shrink_factors": shrink_factors,
        "rot_smoothing_sigmas": smoothing_sigmas,
        "rot_iterations": int(iterations),
        "rot_learning_rate": float(learning_rate),
        "rot_min_step": float(min_step),
        "rot_sampling_percentage": sampling_percentage,
        "rot_points_max_points": int(points_max_points),
        "rot_points_min_distance": int(points_min_distance),
        "rot_points_threshold_rel": float(points_threshold_rel),
        "rot_points_iterations": int(points_iterations),
        "rot_points_max_match_distance": float(points_max_match_distance),
        "output_use_memmap": bool(output_use_memmap),
        "output_memmap_name": output_memmap_name if output_use_memmap else None,
        "output_dtype": str(output_dtype),
        "valid_mask_tzyx": valid_mask_tzyx,
    }
    return registered, details
# %% END