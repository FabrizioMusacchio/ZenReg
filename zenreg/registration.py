"""
Registration helpers for canonical ``TZCYX`` microscopy stacks.

Author: Fabrizio Musacchio
Date: June 2026
"""
# %% IMPORTS
from __future__ import annotations

import os
import warnings
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.ndimage import median_filter
from scipy.ndimage import shift as ndi_shift
from skimage import transform
from skimage.registration import phase_cross_correlation
from skimage.transform import rotate, warp_polar

from ._axes import CANONICAL_AXIS_ORDER, ensure_tzcyx_stack, normalize_zrange
# %% CONSTANTS
SUPPORTED_REGISTRATION_METHODS = {"phase_cross_correlation", "pystackreg", "normcorre"}
SUPPORTED_INTRA_STACK_REFERENCE_MODES = {"neighbor", "full_projection", "first_slice"}
SUPPORTED_PROJECTION_METHODS = {"max", "mean", "median", "var", "std"}
SUPPORTED_TIME_REGISTRATION_MODES = {"projection", "full_3d", "none"}
SUPPORTED_TIME_REFERENCE_MODES = {"template", "previous"}
SUPPORTED_ZERO_CLIP_MODES = {"auto", "shift", "mask"}
SUPPORTED_ZERO_CLIP_MASK_STRATEGIES = {"auto", "greedy", "relaxed", "max_volume"}
SUPPORTED_TRANSFORM_BACKENDS = {"skimage", "scipy"}
# %% FUNCTIONS
def _metadata_spacing_zyx(metadata: dict | None) -> tuple[tuple[float, float, float] | None, str]:
    """Return physical Z/Y/X spacing from an OMIO metadata dictionary if available."""

    if not isinstance(metadata, dict):
        return None, "default"
    try:
        spacing_zyx = (
            float(metadata["PhysicalSizeZ"]),
            float(metadata["PhysicalSizeY"]),
            float(metadata["PhysicalSizeX"]),
        )
    except (KeyError, TypeError, ValueError):
        return None, "default"
    if any(value <= 0 for value in spacing_zyx):
        return None, "default"
    units = [
        metadata.get("PhysicalSizeZUnit"),
        metadata.get("PhysicalSizeYUnit"),
        metadata.get("PhysicalSizeXUnit"),
    ]
    if len({unit for unit in units if unit is not None}) > 1:
        warnings.warn(
            "OMIO metadata contains different physical-size units for Z/Y/X. "
            "Using numeric PhysicalSizeZ/Y/X values as rot_spacing_zyx; please "
            "pass rot_spacing_zyx explicitly if unit conversion is required.",
            RuntimeWarning,
            stacklevel=3,
        )
    return spacing_zyx, "metadata"

def _resolve_rot_spacing_zyx(
    rot_spacing_zyx,
    metadata: dict | None,
) -> tuple[tuple[float, float, float], str]:
    """Resolve user-provided or OMIO-derived physical spacing in Z/Y/X order."""

    if rot_spacing_zyx is not None:
        if len(rot_spacing_zyx) != 3:
            raise ValueError("rot_spacing_zyx must contain exactly three values: (z, y, x).")
        spacing = tuple(float(v) for v in rot_spacing_zyx)
        if any(value <= 0 for value in spacing):
            raise ValueError(f"rot_spacing_zyx values must be > 0. Got {rot_spacing_zyx!r}.")
        return spacing, "user"
    metadata_spacing, source = _metadata_spacing_zyx(metadata)
    if metadata_spacing is not None:
        return metadata_spacing, source
    return (1.0, 1.0, 1.0), "default"

def _resolve_registration_channel(
    registration_channel: int,
    channel_count: int,
) -> tuple[int, int, bool, str | None]:
    """Resolve the requested registration channel against the available channels."""

    requested_channel = int(registration_channel)
    if int(channel_count) < 1:
        raise ValueError("Registration requires at least one channel.")
    if 0 <= requested_channel < int(channel_count):
        return requested_channel, requested_channel, False, None
    if int(channel_count) == 1:
        reason = (
            f"registration_channel={requested_channel} was requested, but the "
            "input stack has only one channel. Falling back to registration_channel=0."
        )
        warnings.warn(reason, RuntimeWarning, stacklevel=3)
        return requested_channel, 0, True, reason
    raise ValueError(
        f"registration_channel must be between 0 and {int(channel_count) - 1}. "
        f"Got {registration_channel!r}."
    )

def _normalize_registration_method(method: str) -> str:
    """Normalize and validate the requested registration backend."""

    normalized = str(method).strip().lower()
    if normalized not in SUPPORTED_REGISTRATION_METHODS:
        raise ValueError(
            f"Unsupported registration method {method!r}. "
            f"Supported methods: {sorted(SUPPORTED_REGISTRATION_METHODS)}."
        )
    return normalized

def _normalize_time_registration_mode(time_registration_mode: str) -> str:
    """Normalize and validate the time-registration strategy."""

    normalized = str(time_registration_mode).strip().lower()
    if normalized not in SUPPORTED_TIME_REGISTRATION_MODES:
        raise ValueError(
            f"Unsupported time_registration_mode {time_registration_mode!r}. "
            f"Supported modes: {sorted(SUPPORTED_TIME_REGISTRATION_MODES)}."
        )
    return normalized

def _normalize_time_reference_mode(time_reference_mode: str) -> str:
    """Normalize and validate the time-reference strategy."""

    normalized = str(time_reference_mode).strip().lower()
    if normalized not in SUPPORTED_TIME_REFERENCE_MODES:
        raise ValueError(
            f"Unsupported time_reference_mode {time_reference_mode!r}. "
            f"Supported modes: {sorted(SUPPORTED_TIME_REFERENCE_MODES)}."
        )
    return normalized

def _normalize_intra_stack_reference_mode(reference_mode: str) -> str:
    """Normalize and validate the intra-stack reference-image strategy."""

    normalized = str(reference_mode).strip().lower()
    if normalized not in SUPPORTED_INTRA_STACK_REFERENCE_MODES:
        raise ValueError(
            f"Unsupported intra-stack reference mode {reference_mode!r}. "
            f"Supported modes: {sorted(SUPPORTED_INTRA_STACK_REFERENCE_MODES)}."
        )
    return normalized

def _normalize_neighbor_window_size(neighbor_window_size: int) -> int:
    """Validate the odd-sized neighborhood used for local Z references."""

    neighbor_window_size = int(neighbor_window_size)
    if neighbor_window_size < 1:
        raise ValueError(f"neighbor_window_size must be >= 1. Got {neighbor_window_size!r}.")
    if neighbor_window_size % 2 == 0:
        raise ValueError("neighbor_window_size must be odd so that the current z-slice stays centered.")
    return neighbor_window_size

def _normalize_registration_stack(registration_stack: int, time_count: int) -> int:
    """Validate the time point used as the registration reference."""

    registration_stack = int(registration_stack)
    if not 0 <= registration_stack < time_count:
        raise ValueError(
            f"registration_stack must be between 0 and {time_count - 1}. "
            f"Got {registration_stack!r}."
        )
    return registration_stack

def _normalize_projection_method(projection_method: str) -> str:
    """Normalize and validate a Z-projection method."""

    normalized = str(projection_method).strip().lower()
    if normalized not in SUPPORTED_PROJECTION_METHODS:
        raise ValueError(
            f"Unsupported projection_method {projection_method!r}. "
            f"Supported methods: {sorted(SUPPORTED_PROJECTION_METHODS)}."
        )
    return normalized

def _normalize_max_xy_shifts(max_xy_shifts: tuple[float, float] | Sequence[float] | None):
    """Normalize optional absolute shift limits for Y and X."""

    if max_xy_shifts is None:
        return None
    if len(max_xy_shifts) != 2:
        raise ValueError("max_xy_shifts must be None or a tuple/list with exactly two values.")
    limits = np.asarray([float(max_xy_shifts[0]), float(max_xy_shifts[1])], dtype=np.float32)
    if np.any(limits < 0):
        raise ValueError(f"max_xy_shifts values must be >= 0. Got {max_xy_shifts!r}.")
    return limits

def _normalize_max_z_shifts(max_z_shifts: float | None):
    """Normalize an optional absolute shift limit for Z."""

    if max_z_shifts is None:
        return None
    limit = float(max_z_shifts)
    if limit < 0:
        raise ValueError(f"max_z_shifts must be >= 0. Got {max_z_shifts!r}.")
    return limit

def _normalize_max_rot_shifts(max_rot_shifts: float | None):
    """Normalize an optional absolute rotation limit in degrees."""

    if max_rot_shifts is None:
        return None
    limit = float(max_rot_shifts)
    if limit < 0:
        raise ValueError(f"max_rot_shifts must be >= 0. Got {max_rot_shifts!r}.")
    return limit

def _normalize_rotreg_iter(rotreg_iter: int) -> int:
    """Validate the number of translation-rotation refinement iterations."""

    rotreg_iter = int(rotreg_iter)
    if rotreg_iter < 1:
        raise ValueError(f"rotreg_iter must be >= 1. Got {rotreg_iter!r}.")
    return rotreg_iter

def _normalize_n_jobs(n_jobs: int | None) -> int:
    """Normalize worker-count arguments shared by CPU-parallel registration paths."""

    if n_jobs is None:
        return 1
    n_jobs = int(n_jobs)
    if n_jobs == 0:
        raise ValueError("n_jobs must be a positive integer, None, or -1 for all available CPUs.")
    if n_jobs < 0:
        cpu_count = os.cpu_count() or 1
        if n_jobs == -1:
            return cpu_count
        return max(cpu_count + 1 + n_jobs, 1)
    return n_jobs

def _progress_iter(iterable, *, total: int, enabled: bool, desc: str | None):
    """Wrap an iterable in tqdm when progress display is enabled and available."""

    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterable
    return tqdm(
        iterable,
        total=int(total),
        desc=desc,
        leave=False,
        dynamic_ncols=True,
    )

def _parallel_map_ordered(function, items, *, n_jobs: int, progress: bool = False, desc: str | None = None):
    """Map ``function`` over ``items`` while preserving input order."""

    items = list(items)
    if int(n_jobs) <= 1 or len(items) <= 1:
        return [function(item) for item in _progress_iter(items, total=len(items), enabled=progress, desc=desc)]
    with ThreadPoolExecutor(max_workers=int(n_jobs)) as executor:
        return list(
            _progress_iter(
                executor.map(function, items),
                total=len(items),
                enabled=progress,
                desc=desc,
            )
        )

def _iter_map_ordered(function, items, *, n_jobs: int, progress: bool = False, desc: str | None = None):
    """Yield mapped results in input order without storing all outputs first."""

    items = list(items)
    if int(n_jobs) <= 1 or len(items) <= 1:
        for item in _progress_iter(items, total=len(items), enabled=progress, desc=desc):
            yield function(item)
        return
    with ThreadPoolExecutor(max_workers=int(n_jobs)) as executor:
        yield from _progress_iter(
            executor.map(function, items),
            total=len(items),
            enabled=progress,
            desc=desc,
        )

def _normalize_zero_clip_mode(zero_clip_mode: str) -> str:
    """Normalize and validate the zero-clipping strategy."""

    normalized = str(zero_clip_mode).strip().lower()
    if normalized not in SUPPORTED_ZERO_CLIP_MODES:
        raise ValueError(
            f"Unsupported zero_clip_mode {zero_clip_mode!r}. "
            f"Supported modes: {sorted(SUPPORTED_ZERO_CLIP_MODES)}."
        )
    return normalized

def _normalize_zero_clip_mask_threshold(zero_clip_mask_threshold: float) -> float:
    """Validate the valid-mask threshold used by mask-based zero-clipping."""

    threshold = float(zero_clip_mask_threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "zero_clip_mask_threshold must be between 0 and 1. "
            f"Got {zero_clip_mask_threshold!r}."
        )
    return threshold

def _normalize_zero_clip_mask_strategy(zero_clip_mask_strategy: str) -> str:
    """Normalize and validate the crop strategy used for mask-based zero-clipping."""

    normalized = str(zero_clip_mask_strategy).strip().lower()
    if normalized not in SUPPORTED_ZERO_CLIP_MASK_STRATEGIES:
        raise ValueError(
            f"Unsupported zero_clip_mask_strategy {zero_clip_mask_strategy!r}. "
            f"Supported strategies: {sorted(SUPPORTED_ZERO_CLIP_MASK_STRATEGIES)}."
        )
    return normalized

def _normalize_zero_clip_mask_min_fraction(zero_clip_mask_min_fraction: float) -> float:
    """Validate the relaxed mask-crop minimum valid-plane fraction."""

    fraction = float(zero_clip_mask_min_fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError(
            "zero_clip_mask_min_fraction must be > 0 and <= 1. "
            f"Got {zero_clip_mask_min_fraction!r}."
        )
    return fraction

def _normalize_zero_clip_margin(zero_clip_margin: int | Sequence[int]) -> np.ndarray:
    """Normalize optional extra ``(z, y, x)`` crop margins."""

    if np.isscalar(zero_clip_margin):
        margins = np.asarray([int(zero_clip_margin)] * 3, dtype=np.int32)
    else:
        if len(zero_clip_margin) != 3:
            raise ValueError("zero_clip_margin must be an int or a tuple/list of three values: (z, y, x).")
        margins = np.asarray([int(v) for v in zero_clip_margin], dtype=np.int32)
    if np.any(margins < 0):
        raise ValueError(f"zero_clip_margin values must be >= 0. Got {zero_clip_margin!r}.")
    return margins

def _normalize_transform_backend(transform_backend: str) -> str:
    """Normalize and validate the image transformation backend."""

    normalized = str(transform_backend).strip().lower()
    if normalized not in SUPPORTED_TRANSFORM_BACKENDS:
        raise ValueError(
            f"Unsupported transform_backend {transform_backend!r}. "
            f"Supported backends: {sorted(SUPPORTED_TRANSFORM_BACKENDS)}."
        )
    return normalized

def _normalize_transform_order(transform_order: int) -> int:
    """Validate interpolation order for geometric transformations."""

    transform_order = int(transform_order)
    if not 0 <= transform_order <= 5:
        raise ValueError(f"transform_order must be between 0 and 5. Got {transform_order!r}.")
    return transform_order

def _as_float32_stack_copy(stack) -> np.ndarray:
    """Return a float32 ``TZCYX`` working copy from NumPy or disk-backed arrays."""

    stack = ensure_tzcyx_stack(stack)
    try:
        return stack.astype(np.float32, copy=True)
    except (AttributeError, TypeError):
        return np.asarray(stack, dtype=np.float32).copy()

def _as_float32_work_array(array) -> np.ndarray:
    """Return a local float32 working array for the currently processed chunk."""

    return np.asarray(array, dtype=np.float32)

def _normalize_output_dtype(output_dtype) -> np.dtype:
    """Normalize the registered-output dtype."""

    dtype = np.dtype(output_dtype)
    if not np.issubdtype(dtype, np.floating):
        warnings.warn(
            "Registration output is usually safest as a floating dtype because "
            "subpixel transforms create interpolated intensities. Continuing "
            f"with output_dtype={dtype}.",
            RuntimeWarning,
            stacklevel=3,
        )
    return dtype

def _compose_output_memmap_name(base_name: str | None, stage_name: str | None) -> str | None:
    """Build a deterministic Zarr store name for intermediate registration stages."""

    if base_name is None:
        return None
    if not stage_name:
        return str(base_name)
    return f"{base_name}_{stage_name}"

def _create_registered_output(
    shape: tuple[int, int, int, int, int],
    *,
    dtype,
    output_use_memmap: bool,
    output_memmap_folder: str | os.PathLike | None,
    output_memmap_name: str | None,
    stage_name: str | None = None,
):
    """Allocate a registered output stack in RAM or as an OMIO/Zarr array."""

    if output_use_memmap:
        from .io import create_empty_stack

        return create_empty_stack(
            shape=tuple(int(v) for v in shape),
            dtype=np.dtype(dtype),
            fill_value=0,
            use_memmap=True,
            memmap_folder=output_memmap_folder,
            memmap_name=_compose_output_memmap_name(output_memmap_name, stage_name),
            return_metadata=False,
            verbose=False,
        )
    return np.empty(tuple(int(v) for v in shape), dtype=np.dtype(dtype))

def _extract_registration_volume(
    stack,
    *,
    t: int,
    registration_channel: int,
    zrange: tuple[int, int] | Sequence[int] | None,
    filter_slices: bool,
    median_kernel_size: int,
) -> np.ndarray:
    """Extract one registration-channel ``ZYX`` volume as a local float32 chunk."""

    z_start, z_stop = normalize_zrange(zrange, stack.shape[1], strict=True)
    volume = _as_float32_work_array(stack[int(t), int(z_start) : int(z_stop), int(registration_channel), :, :])
    if filter_slices:
        return _apply_median_to_zyx(volume, int(median_kernel_size))
    return volume

def _extract_registration_template_volume(
    stack,
    *,
    registration_channel: int,
    registration_stack: int,
    registration_template_time_range: tuple[int, int] | None,
    zrange: tuple[int, int] | Sequence[int] | None,
    projection_method: str,
    filter_slices: bool,
    median_kernel_size: int,
) -> np.ndarray:
    """Build the reference registration-channel ``ZYX`` volume."""

    if registration_template_time_range is None:
        return _extract_registration_volume(
            stack,
            t=int(registration_stack),
            registration_channel=registration_channel,
            zrange=zrange,
            filter_slices=filter_slices,
            median_kernel_size=median_kernel_size,
        )
    start, stop = registration_template_time_range
    volumes = [
        _extract_registration_volume(
            stack,
            t=t,
            registration_channel=registration_channel,
            zrange=zrange,
            filter_slices=filter_slices,
            median_kernel_size=median_kernel_size,
        )
        for t in range(int(start), int(stop))
    ]
    template = _aggregate_along_axis(
        np.stack(volumes, axis=0),
        axis=0,
        method=projection_method,
    )
    return np.asarray(template, dtype=np.float32)

def _effective_zero_clip_mode(*, zero_clip: bool, zero_clip_mode: str, rotreg: bool) -> str:
    """Return the concrete zero-clipping mode for this registration run."""

    if not zero_clip:
        return "none"
    if zero_clip_mode == "auto":
        return "mask" if rotreg else "shift"
    return zero_clip_mode

def _effective_zero_clip_mask_strategy(
    *,
    zero_clip_mask_strategy: str,
    rigid_3d: bool,
) -> str:
    """Return the concrete mask-cropping strategy for this registration run."""

    if zero_clip_mask_strategy == "auto":
        return "relaxed" if rigid_3d else "greedy"
    return zero_clip_mask_strategy

def _resolve_registration_z_range_alias(
    *,
    registration_z_range: tuple[int, int] | Sequence[int] | None,
    zrange: tuple[int, int] | Sequence[int] | None,
    projection_range: tuple[int, int] | Sequence[int] | None,
):
    """Resolve preferred registration_z_range while keeping older aliases compatible."""

    provided = [
        ("registration_z_range", registration_z_range),
        ("projection_range", projection_range),
        ("zrange", zrange),
    ]
    active = [(name, value) for name, value in provided if value is not None]
    if not active:
        return None
    reference_name, reference_value = active[0]
    reference_tuple = tuple(reference_value)
    for name, value in active[1:]:
        if tuple(value) != reference_tuple:
            raise ValueError(
                "Use only one Z-range argument or provide matching values. "
                f"Got conflicting {reference_name}={reference_tuple!r} and {name}={tuple(value)!r}."
            )
    return reference_value

def _normalize_registration_template_time_range(
    registration_template_time_range: tuple[int, int] | Sequence[int] | str | None,
    time_count: int,
) -> tuple[int, int] | None:
    """Validate an optional half-open time range used to build a registration template."""

    if registration_template_time_range is None:
        return None
    if isinstance(registration_template_time_range, str):
        normalized = registration_template_time_range.strip().lower()
        if normalized == "all":
            return (0, int(time_count))
        raise ValueError(
            "registration_template_time_range must be None, 'all', or a "
            "two-element half-open time range (start, stop)."
        )
    if len(registration_template_time_range) != 2:
        raise ValueError(
            "registration_template_time_range must be None, 'all', or a two-element "
            "half-open time range (start, stop)."
        )
    start, stop = (int(registration_template_time_range[0]), int(registration_template_time_range[1]))
    if start < 0 or stop > int(time_count) or start >= stop:
        raise ValueError(
            "registration_template_time_range must satisfy "
            f"0 <= start < stop <= T. Got {(start, stop)!r} for T={int(time_count)}."
        )
    return (start, stop)

def _clip_shift_yx(shift_yx: np.ndarray, max_xy_shifts: np.ndarray | None) -> np.ndarray:
    """Clip a ``YX`` correction shift to configured absolute limits."""

    shift_yx = np.asarray(shift_yx, dtype=np.float32).copy()
    if max_xy_shifts is not None:
        shift_yx = np.clip(shift_yx, -max_xy_shifts, max_xy_shifts)
    return shift_yx.astype(np.float32, copy=False)

def _clip_shift_zyx(
    shift_zyx: np.ndarray,
    *,
    max_z_shifts: float | None,
    max_xy_shifts: np.ndarray | None,
) -> np.ndarray:
    """Clip a ``ZYX`` correction shift to configured absolute limits."""

    shift_zyx = np.asarray(shift_zyx, dtype=np.float32).copy()
    if max_z_shifts is not None:
        shift_zyx[0] = np.clip(shift_zyx[0], -max_z_shifts, max_z_shifts)
    if max_xy_shifts is not None:
        shift_zyx[1:] = np.clip(shift_zyx[1:], -max_xy_shifts, max_xy_shifts)
    return shift_zyx.astype(np.float32, copy=False)

def _clip_rotation_deg(angle_deg: float, max_rot_shifts: float | None) -> float:
    """Clip a rotation correction to an optional absolute degree limit."""

    angle_deg = float(angle_deg)
    if max_rot_shifts is not None:
        angle_deg = float(np.clip(angle_deg, -max_rot_shifts, max_rot_shifts))
    return angle_deg

def _crop_bounds_from_zyx_shifts(shifts_zyx: np.ndarray | None) -> dict[str, int]:
    """Compute directional crop widths from applied ``ZYX`` correction shifts."""

    bounds = {
        "z_top": 0,
        "z_bottom": 0,
        "y_top": 0,
        "y_bottom": 0,
        "x_left": 0,
        "x_right": 0,
    }
    if shifts_zyx is None:
        return bounds

    shifts = np.asarray(shifts_zyx, dtype=np.float32).reshape(-1, 3)
    if shifts.size == 0:
        return bounds
    bounds["z_top"] = int(np.ceil(max(0.0, float(np.max(shifts[:, 0])))))
    bounds["z_bottom"] = int(np.ceil(max(0.0, float(np.max(-shifts[:, 0])))))
    bounds["y_top"] = int(np.ceil(max(0.0, float(np.max(shifts[:, 1])))))
    bounds["y_bottom"] = int(np.ceil(max(0.0, float(np.max(-shifts[:, 1])))))
    bounds["x_left"] = int(np.ceil(max(0.0, float(np.max(shifts[:, 2])))))
    bounds["x_right"] = int(np.ceil(max(0.0, float(np.max(-shifts[:, 2])))))
    return bounds

def _crop_bounds_from_yx_shifts(shifts_yx: np.ndarray | None) -> dict[str, int]:
    """Compute directional Y/X crop widths from applied ``YX`` correction shifts."""

    bounds = {
        "z_top": 0,
        "z_bottom": 0,
        "y_top": 0,
        "y_bottom": 0,
        "x_left": 0,
        "x_right": 0,
    }
    if shifts_yx is None:
        return bounds

    shifts = np.asarray(shifts_yx, dtype=np.float32).reshape(-1, 2)
    if shifts.size == 0:
        return bounds
    bounds["y_top"] = int(np.ceil(max(0.0, float(np.max(shifts[:, 0])))))
    bounds["y_bottom"] = int(np.ceil(max(0.0, float(np.max(-shifts[:, 0])))))
    bounds["x_left"] = int(np.ceil(max(0.0, float(np.max(shifts[:, 1])))))
    bounds["x_right"] = int(np.ceil(max(0.0, float(np.max(-shifts[:, 1])))))
    return bounds

def _add_crop_bounds(*bounds_list: dict[str, int]) -> dict[str, int]:
    """Add crop bounds from sequential correction stages."""

    keys = ("z_top", "z_bottom", "y_top", "y_bottom", "x_left", "x_right")
    return {key: int(sum(bounds.get(key, 0) for bounds in bounds_list)) for key in keys}

def _apply_zero_clip_margin(crop_bounds: dict[str, int], margin_zyx: np.ndarray) -> dict[str, int]:
    """Add symmetric extra ``Z/Y/X`` margins to directional crop bounds."""

    margin_zyx = np.asarray(margin_zyx, dtype=np.int32)
    return {
        "z_top": int(crop_bounds.get("z_top", 0) + margin_zyx[0]),
        "z_bottom": int(crop_bounds.get("z_bottom", 0) + margin_zyx[0]),
        "y_top": int(crop_bounds.get("y_top", 0) + margin_zyx[1]),
        "y_bottom": int(crop_bounds.get("y_bottom", 0) + margin_zyx[1]),
        "x_left": int(crop_bounds.get("x_left", 0) + margin_zyx[2]),
        "x_right": int(crop_bounds.get("x_right", 0) + margin_zyx[2]),
    }

def _bounds_from_valid_starts_stops(valid: np.ndarray, starts: np.ndarray, stops: np.ndarray) -> dict[str, int]:
    """Convert Z/Y/X start-stop indices to directional crop widths."""

    if np.any(starts >= stops):
        raise ValueError("Mask-based zero_clip could not find a common valid image region.")
    return {
        "z_top": int(starts[0]),
        "z_bottom": int(valid.shape[0] - stops[0]),
        "y_top": int(starts[1]),
        "y_bottom": int(valid.shape[1] - stops[1]),
        "x_left": int(starts[2]),
        "x_right": int(valid.shape[2] - stops[2]),
    }

def _crop_bounds_from_valid_mask_greedy(valid: np.ndarray) -> dict[str, int]:
    """Compute strict crop bounds by greedily removing invalid border faces."""

    z_start = 0
    y_start = 0
    x_start = 0
    z_stop, y_stop, x_stop = valid.shape

    while z_start < z_stop and y_start < y_stop and x_start < x_stop:
        current = valid[z_start:z_stop, y_start:y_stop, x_start:x_stop]
        if np.all(current):
            break
        faces = {
            "z_top": valid[z_start, y_start:y_stop, x_start:x_stop] if z_stop - z_start > 1 else None,
            "z_bottom": valid[z_stop - 1, y_start:y_stop, x_start:x_stop] if z_stop - z_start > 1 else None,
            "y_top": valid[z_start:z_stop, y_start, x_start:x_stop],
            "y_bottom": valid[z_start:z_stop, y_stop - 1, x_start:x_stop],
            "x_left": valid[z_start:z_stop, y_start:y_stop, x_start],
            "x_right": valid[z_start:z_stop, y_start:y_stop, x_stop - 1],
        }
        candidates = []
        for name, face in faces.items():
            if face is None:
                continue
            invalid_count = int(face.size - np.count_nonzero(face))
            if invalid_count > 0:
                invalid_fraction = invalid_count / max(int(face.size), 1)
                candidates.append((invalid_fraction, invalid_count, name))
        if not candidates:
            break

        _, _, selected = max(candidates)
        if selected == "z_top":
            z_start += 1
        elif selected == "z_bottom":
            z_stop -= 1
        elif selected == "y_top":
            y_start += 1
        elif selected == "y_bottom":
            y_stop -= 1
        elif selected == "x_left":
            x_start += 1
        elif selected == "x_right":
            x_stop -= 1

    return _bounds_from_valid_starts_stops(
        valid,
        np.asarray([z_start, y_start, x_start], dtype=np.int64),
        np.asarray([z_stop, y_stop, x_stop], dtype=np.int64),
    )

def _crop_bounds_from_valid_mask_relaxed(valid: np.ndarray, *, min_fraction: float) -> dict[str, int]:
    """Crop border planes only while their valid fraction is below ``min_fraction``."""

    fallback_fractions = (0.90, 0.75, 0.50, 0.25, 0.01)
    fractions = [float(min_fraction)]
    fractions.extend(fraction for fraction in fallback_fractions if fraction < float(min_fraction))
    last_error: ValueError | None = None
    for fraction in fractions:
        z_start = 0
        y_start = 0
        x_start = 0
        z_stop, y_stop, x_stop = valid.shape
        while z_start < z_stop and y_start < y_stop and x_start < x_stop:
            faces = {
                "z_top": valid[z_start, y_start:y_stop, x_start:x_stop] if z_stop - z_start > 1 else None,
                "z_bottom": valid[z_stop - 1, y_start:y_stop, x_start:x_stop] if z_stop - z_start > 1 else None,
                "y_top": valid[z_start:z_stop, y_start, x_start:x_stop],
                "y_bottom": valid[z_start:z_stop, y_stop - 1, x_start:x_stop],
                "x_left": valid[z_start:z_stop, y_start:y_stop, x_start],
                "x_right": valid[z_start:z_stop, y_start:y_stop, x_stop - 1],
            }
            candidates = []
            for name, face in faces.items():
                if face is None:
                    continue
                invalid_count = int(face.size - np.count_nonzero(face))
                invalid_fraction = invalid_count / max(int(face.size), 1)
                if invalid_fraction > (1.0 - fraction):
                    candidates.append((invalid_fraction, invalid_count, name))
            if not candidates:
                break
            _, _, selected = max(candidates)
            if selected == "z_top":
                z_start += 1
            elif selected == "z_bottom":
                z_stop -= 1
            elif selected == "y_top":
                y_start += 1
            elif selected == "y_bottom":
                y_stop -= 1
            elif selected == "x_left":
                x_start += 1
            elif selected == "x_right":
                x_stop -= 1
        starts = np.asarray([z_start, y_start, x_start], dtype=np.int64)
        stops = np.asarray([z_stop, y_stop, x_stop], dtype=np.int64)
        try:
            return _bounds_from_valid_starts_stops(valid, starts, stops)
        except ValueError as exc:
            last_error = exc
    raise ValueError("Mask-based relaxed zero_clip could not find a valid image region.") from last_error

def _largest_rectangle_in_binary_mask(mask_yx: np.ndarray) -> tuple[int, int, int, int, int]:
    """Return area and Y/X bounds of the largest all-true rectangle."""

    mask_yx = np.asarray(mask_yx, dtype=bool)
    heights = np.zeros(mask_yx.shape[1], dtype=np.int64)
    best_area = 0
    best = (0, 0, 0, 0)
    for y, row in enumerate(mask_yx):
        heights[row] += 1
        heights[~row] = 0
        stack: list[int] = []
        extended = np.concatenate([heights, np.asarray([0], dtype=np.int64)])
        for x, height in enumerate(extended):
            while stack and int(extended[stack[-1]]) > int(height):
                h = int(extended[stack.pop()])
                left = int(stack[-1] + 1) if stack else 0
                width = int(x - left)
                area = h * width
                if area > best_area:
                    best_area = area
                    best = (int(y - h + 1), int(y + 1), left, int(x))
            stack.append(int(x))
    return int(best_area), best[0], best[1], best[2], best[3]

def _crop_bounds_from_valid_mask_max_volume(valid: np.ndarray) -> dict[str, int]:
    """Find the largest strict all-valid axis-aligned cuboid."""

    valid = np.asarray(valid, dtype=bool)
    z_size = int(valid.shape[0])
    best_volume = 0
    best_bounds: tuple[int, int, int, int, int, int] | None = None
    for z_start in range(z_size):
        slab = np.ones(valid.shape[1:], dtype=bool)
        for z_stop in range(z_start + 1, z_size + 1):
            slab &= valid[z_stop - 1]
            area, y_start, y_stop, x_start, x_stop = _largest_rectangle_in_binary_mask(slab)
            volume = int(area) * int(z_stop - z_start)
            if volume > best_volume:
                best_volume = volume
                best_bounds = (z_start, z_stop, y_start, y_stop, x_start, x_stop)
    if best_bounds is None or best_volume <= 0:
        raise ValueError("Mask-based zero_clip could not find a common valid image region.")
    z_start, z_stop, y_start, y_stop, x_start, x_stop = best_bounds
    return _bounds_from_valid_starts_stops(
        valid,
        np.asarray([z_start, y_start, x_start], dtype=np.int64),
        np.asarray([z_stop, y_stop, x_stop], dtype=np.int64),
    )

def _crop_bounds_from_valid_mask(
    mask_tzyx: np.ndarray,
    *,
    threshold: float,
    strategy: str = "greedy",
    min_fraction: float = 0.50,
) -> dict[str, int]:
    """Compute crop bounds from a transformed validity mask."""

    valid = np.all(np.asarray(mask_tzyx, dtype=np.float32) > float(threshold), axis=0)
    if strategy == "relaxed":
        return _crop_bounds_from_valid_mask_relaxed(valid, min_fraction=min_fraction)
    if strategy == "max_volume":
        return _crop_bounds_from_valid_mask_max_volume(valid)
    return _crop_bounds_from_valid_mask_greedy(valid)

def _zero_clip_stack(
    stack,
    crop_bounds: dict[str, int],
    *,
    output_use_memmap: bool = False,
    output_memmap_folder: str | os.PathLike | None = None,
    output_memmap_name: str | None = None,
    output_dtype=np.float32,
    n_jobs: int = 1,
    progress: bool = False,
):
    """Crop zero-fill borders from a registered ``TZCYX`` stack."""

    z_top = int(crop_bounds.get("z_top", 0))
    z_bottom = int(crop_bounds.get("z_bottom", 0))
    y_top = int(crop_bounds.get("y_top", 0))
    y_bottom = int(crop_bounds.get("y_bottom", 0))
    x_left = int(crop_bounds.get("x_left", 0))
    x_right = int(crop_bounds.get("x_right", 0))

    z_stop = stack.shape[1] - z_bottom
    y_stop = stack.shape[3] - y_bottom
    x_stop = stack.shape[4] - x_right
    if z_top >= z_stop or y_top >= y_stop or x_left >= x_stop:
        raise ValueError(
            "zero_clip would remove the complete image. "
            f"Shape={stack.shape}, crop_bounds={crop_bounds}."
        )
    cropped_shape = (
        int(stack.shape[0]),
        int(z_stop - z_top),
        int(stack.shape[2]),
        int(y_stop - y_top),
        int(x_stop - x_left),
    )
    if not output_use_memmap:
        return np.asarray(
            stack[:, z_top:z_stop, :, y_top:y_stop, x_left:x_stop],
            dtype=np.dtype(output_dtype),
        ).copy()

    cropped = _create_registered_output(
        cropped_shape,
        dtype=output_dtype,
        output_use_memmap=True,
        output_memmap_folder=output_memmap_folder,
        output_memmap_name=output_memmap_name,
        stage_name="zero_clipped",
    )

    def copy_timepoint(t: int) -> tuple[int, np.ndarray]:
        chunk = np.asarray(
            stack[int(t), z_top:z_stop, :, y_top:y_stop, x_left:x_stop],
            dtype=np.dtype(output_dtype),
        )
        return int(t), chunk

    for t, chunk in _iter_map_ordered(
        copy_timepoint,
        range(stack.shape[0]),
        n_jobs=n_jobs,
        progress=progress,
        desc="ZenReg zero-clip crop",
    ):
        cropped[t, :, :, :, :] = chunk
    return cropped

def _project_zyx_to_yx(volume_zyx: np.ndarray, *, projection_method: str) -> np.ndarray:
    """Project one ``ZYX`` volume to ``YX`` using a validated method."""

    return _aggregate_along_axis(volume_zyx, axis=0, method=projection_method)

def _aggregate_along_axis(array: np.ndarray, *, axis: int, method: str) -> np.ndarray:
    """Aggregate an array along one axis using ZenReg's projection methods."""

    if method == "max":
        return np.max(array, axis=axis)
    if method == "mean":
        return np.mean(array, axis=axis)
    if method == "median":
        return np.median(array, axis=axis)
    if method == "var":
        return np.var(array, axis=axis)
    return np.std(array, axis=axis)

def _pearson_correlation_flat(template: np.ndarray, image: np.ndarray) -> float:
    """Compute Pearson correlation for two registration-image vectors."""

    template = np.asarray(template, dtype=np.float64).ravel()
    image = np.asarray(image, dtype=np.float64).ravel()
    template = template - np.mean(template)
    image = image - np.mean(image)
    denominator = float(np.linalg.norm(template) * np.linalg.norm(image))
    if denominator == 0.0:
        return float("nan")
    return float(np.dot(template, image) / denominator)

def _registration_correlation_image_for_frame(
    stack,
    *,
    t: int,
    registration_channel: int,
    projection_range,
    projection_method: str,
    effective_time_registration_mode: str,
) -> np.ndarray:
    """Return one flattened frame image used for registration-correlation reporting."""

    z_start, z_stop = normalize_zrange(projection_range, stack.shape[1], strict=True)
    volume = np.asarray(stack[int(t), z_start:z_stop, int(registration_channel), :, :], dtype=np.float32)
    if effective_time_registration_mode == "full_3d":
        return volume.ravel()
    projection = _project_zyx_to_yx(volume, projection_method=projection_method)
    return np.asarray(projection, dtype=np.float32).ravel()

def _registration_correlation_template_image(
    stack,
    *,
    registration_channel: int,
    registration_stack: int,
    registration_template_time_range: tuple[int, int] | None,
    projection_range,
    projection_method: str,
    effective_time_registration_mode: str,
) -> np.ndarray:
    """Return the flattened template image used for registration-correlation reporting."""

    template_volume = _extract_registration_template_volume(
        stack,
        registration_channel=registration_channel,
        registration_stack=registration_stack,
        registration_template_time_range=registration_template_time_range,
        zrange=projection_range,
        projection_method=projection_method,
        filter_slices=False,
        median_kernel_size=1,
    )
    if effective_time_registration_mode == "full_3d":
        return template_volume.ravel()
    projection = _project_zyx_to_yx(template_volume, projection_method=projection_method)
    return np.asarray(projection, dtype=np.float32).ravel()

def _compute_registration_frame_correlations(
    stack,
    *,
    registration_channel: int,
    registration_stack: int,
    registration_template_time_range: tuple[int, int] | None,
    projection_range,
    projection_method: str,
    effective_time_registration_mode: str,
) -> np.ndarray:
    """Compute framewise template correlations without materializing all frames."""

    registration_stack = int(np.clip(int(registration_stack), 0, stack.shape[0] - 1))
    template = _registration_correlation_template_image(
        stack,
        registration_channel=registration_channel,
        registration_stack=registration_stack,
        registration_template_time_range=registration_template_time_range,
        projection_range=projection_range,
        projection_method=projection_method,
        effective_time_registration_mode=effective_time_registration_mode,
    )
    correlations = np.empty(stack.shape[0], dtype=np.float32)
    for t in range(stack.shape[0]):
        image = _registration_correlation_image_for_frame(
            stack,
            t=t,
            registration_channel=registration_channel,
            projection_range=projection_range,
            projection_method=projection_method,
            effective_time_registration_mode=effective_time_registration_mode,
        )
        correlations[t] = _pearson_correlation_flat(template, image)
    return correlations

def _project_zyx_along_axis(
    volume_zyx: np.ndarray,
    *,
    axis: int,
    projection_method: str,
) -> np.ndarray:
    """Project one ``ZYX`` volume along any axis using a validated method."""

    if projection_method == "max":
        return np.max(volume_zyx, axis=axis)
    if projection_method == "mean":
        return np.mean(volume_zyx, axis=axis)
    if projection_method == "median":
        return np.median(volume_zyx, axis=axis)
    if projection_method == "var":
        return np.var(volume_zyx, axis=axis)
    return np.std(volume_zyx, axis=axis)

def _resolve_filter_aliases(
    *,
    filter_slices: bool,
    filter_projections: bool,
    pre_median_filter: bool | None,
    post_median_filter: bool | None,
) -> tuple[bool, bool]:
    """Resolve new filter option names while accepting legacy aliases."""

    if pre_median_filter is not None:
        warnings.warn(
            "pre_median_filter is deprecated; use filter_slices instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        if filter_slices and not bool(pre_median_filter):
            raise ValueError("Conflicting filter_slices=True and pre_median_filter=False.")
        filter_slices = bool(pre_median_filter)

    if post_median_filter is not None:
        warnings.warn(
            "post_median_filter is deprecated; use filter_projections instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        if filter_projections and not bool(post_median_filter):
            raise ValueError("Conflicting filter_projections=True and post_median_filter=False.")
        filter_projections = bool(post_median_filter)

    return bool(filter_slices), bool(filter_projections)

def _normalize_phase_cross_correlation_normalization(normalization: str | None) -> str | None:
    """Normalize scikit-image's phase-cross-correlation normalization option."""

    if normalization is None:
        return None
    normalized = str(normalization).strip().lower()
    if normalized in {"none", "null"}:
        return None
    if normalized != "phase":
        raise ValueError(
            "phase_cross_correlation_normalization must be None, 'none', or 'phase'. "
            f"Got {normalization!r}."
        )
    return normalized

def _apply_median_to_zyx(volume_zyx: np.ndarray, kernel_size: int) -> np.ndarray:
    """Apply a 2D median filter independently to each Z plane of a ``ZYX`` volume."""

    filtered = np.empty_like(volume_zyx, dtype=np.float32)
    for z in range(volume_zyx.shape[0]):
        filtered[z, :, :] = median_filter(volume_zyx[z, :, :], size=(kernel_size, kernel_size))
    return filtered

def _build_intra_stack_reference_image(
    volume_zyx: np.ndarray,
    *,
    z_index: int,
    reference_mode: str,
    neighbor_window_size: int,
    projection_method: str,
) -> np.ndarray:
    """Build the per-slice registration reference used for Z-drift correction."""

    if reference_mode == "full_projection":
        return _project_zyx_to_yx(volume_zyx, projection_method=projection_method)
    if reference_mode == "first_slice":
        return volume_zyx[0, :, :]

    half_window = neighbor_window_size // 2
    start = max(0, z_index - half_window)
    stop = min(volume_zyx.shape[0], z_index + half_window + 1)
    return _project_zyx_to_yx(volume_zyx[start:stop, :, :], projection_method=projection_method)

def _build_registration_projections(
    stack: np.ndarray,
    *,
    registration_channel: int,
    zrange: tuple[int, int] | Sequence[int] | None,
    projection_method: str,
    filter_slices: bool,
    filter_projections: bool,
    median_kernel_size: int,
) -> np.ndarray:
    """Create per-time-point 2D registration projections from a ``TZCYX`` stack."""

    z_start, z_stop = normalize_zrange(zrange, stack.shape[1], strict=True)
    channel_stack = np.asarray(stack[:, z_start:z_stop, registration_channel, :, :], dtype=np.float32)
    working = channel_stack.copy()

    if filter_slices:
        for t in range(working.shape[0]):
            working[t, :, :, :] = _apply_median_to_zyx(working[t, :, :, :], median_kernel_size)

    projections = np.empty((working.shape[0], working.shape[2], working.shape[3]), dtype=np.float32)
    for t in range(working.shape[0]):
        projections[t, :, :] = _project_zyx_to_yx(
            working[t, :, :, :],
            projection_method=projection_method,
        )

    if filter_projections:
        for t in range(projections.shape[0]):
            projections[t, :, :] = median_filter(
                projections[t, :, :], size=(median_kernel_size, median_kernel_size)
            )
    return projections

def _phase_cross_correlation_shift(
    reference_projection: np.ndarray,
    moving_projection: np.ndarray,
    *,
    upsample_factor: int,
    normalization: str | None,
) -> np.ndarray:
    """Estimate a 2D translation with phase cross-correlation."""

    return _phase_cross_correlation_nd_shift(
        reference_projection,
        moving_projection,
        upsample_factor=upsample_factor,
        normalization=normalization,
    )

def _phase_cross_correlation_nd_shift(
    reference_image: np.ndarray,
    moving_image: np.ndarray,
    *,
    upsample_factor: int,
    normalization: str | None,
) -> np.ndarray:
    """Estimate an N-dimensional translation with phase cross-correlation."""

    shift_2d, _, _ = phase_cross_correlation(
        reference_image,
        moving_image,
        upsample_factor=int(upsample_factor),
        normalization=normalization,
    )
    return np.asarray(shift_2d, dtype=np.float32)

def _pystackreg_shift(reference_projection: np.ndarray, moving_projection: np.ndarray) -> np.ndarray:
    """Estimate a 2D translation with :mod:`pystackreg` in translation mode."""

    from pystackreg import StackReg

    sr = StackReg(StackReg.TRANSLATION)
    tmat = sr.register(reference_projection.astype(np.float32), moving_projection.astype(np.float32))
    return np.asarray([-tmat[1, 2], -tmat[0, 2]], dtype=np.float32)

def _estimate_shift(
    reference_projection: np.ndarray,
    moving_projection: np.ndarray,
    *,
    method: str,
    phase_cross_correlation_upsample_factor: int,
    phase_cross_correlation_normalization: str | None,
) -> np.ndarray:
    """Estimate a 2D registration shift using the selected backend."""

    if method == "phase_cross_correlation":
        return _phase_cross_correlation_shift(
            reference_projection,
            moving_projection,
            upsample_factor=phase_cross_correlation_upsample_factor,
            normalization=phase_cross_correlation_normalization,
        )
    if method == "pystackreg":
        return _pystackreg_shift(reference_projection, moving_projection)
    raise ValueError("method='normcorre' is available only through register_stack(), not pairwise shift helpers.")

def _estimate_full_3d_shift(
    reference_volume: np.ndarray,
    moving_volume: np.ndarray,
    *,
    phase_cross_correlation_upsample_factor: int,
    phase_cross_correlation_normalization: str | None,
) -> np.ndarray:
    """Estimate a ``ZYX`` translation from two 3D volumes."""

    return _phase_cross_correlation_nd_shift(
        reference_volume,
        moving_volume,
        upsample_factor=phase_cross_correlation_upsample_factor,
        normalization=phase_cross_correlation_normalization,
    )

def _estimate_z_shift_from_orthogonal_projections(
    reference_volume: np.ndarray,
    moving_volume: np.ndarray,
    *,
    method: str,
    projection_method: str,
    filter_projections: bool,
    median_kernel_size: int,
    phase_cross_correlation_upsample_factor: int,
    phase_cross_correlation_normalization: str | None,
) -> float:
    """Estimate Z translation from ``ZY`` and ``ZX`` projections."""

    if reference_volume.shape[0] <= 1:
        return 0.0

    z_shifts = []
    kernel = (int(median_kernel_size), int(median_kernel_size))
    for projection_axis in (1, 2):
        reference_projection = _project_zyx_along_axis(
            reference_volume,
            axis=projection_axis,
            projection_method=projection_method,
        ).astype(np.float32, copy=False)
        moving_projection = _project_zyx_along_axis(
            moving_volume,
            axis=projection_axis,
            projection_method=projection_method,
        ).astype(np.float32, copy=False)
        if filter_projections:
            reference_projection = median_filter(reference_projection, size=kernel)
            moving_projection = median_filter(moving_projection, size=kernel)
        shift = _estimate_shift(
            reference_projection,
            moving_projection,
            method=method,
            phase_cross_correlation_upsample_factor=phase_cross_correlation_upsample_factor,
            phase_cross_correlation_normalization=phase_cross_correlation_normalization,
        )
        z_shifts.append(float(shift[0]))
    return float(np.mean(z_shifts))

def _normalize_rotation_image(image: np.ndarray) -> np.ndarray:
    """Normalize a 2D image for polar phase-correlation rotation estimation."""

    image = np.asarray(image, dtype=np.float32)
    image = image - float(np.min(image))
    max_value = float(np.max(image))
    if max_value > 0:
        image = image / max_value
    return image.astype(np.float32, copy=False)

def _estimate_rotation_deg_from_projections(
    reference_projection: np.ndarray,
    moving_projection: np.ndarray,
    *,
    phase_cross_correlation_upsample_factor: int,
    phase_cross_correlation_normalization: str | None,
) -> float:
    """Estimate an in-plane rotation angle in degrees from two 2D projections."""

    reference_projection = _normalize_rotation_image(reference_projection)
    moving_projection = _normalize_rotation_image(moving_projection)
    radius = max(1, min(reference_projection.shape) // 2)
    output_shape = (360, radius)
    reference_polar = warp_polar(reference_projection, radius=radius, output_shape=output_shape)
    moving_polar = warp_polar(moving_projection, radius=radius, output_shape=output_shape)
    shift, _, _ = phase_cross_correlation(
        reference_polar,
        moving_polar,
        upsample_factor=int(phase_cross_correlation_upsample_factor),
        normalization=phase_cross_correlation_normalization,
    )
    angle_deg = float(shift[0]) * 360.0 / float(output_shape[0])
    if angle_deg > 180.0:
        angle_deg -= 360.0
    if angle_deg <= -180.0:
        angle_deg += 360.0
    return float(angle_deg)

def _apply_translation_to_yx(
    image_yx: np.ndarray,
    shift_yx: np.ndarray,
    *,
    transform_backend: str,
    transform_order: int,
) -> np.ndarray:
    """Apply one 2D translation with the selected transform backend."""

    image_yx = np.asarray(image_yx, dtype=np.float32)
    if transform_backend == "skimage":
        shift_y, shift_x = float(shift_yx[0]), float(shift_yx[1])
        tform = transform.SimilarityTransform(translation=(-shift_x, -shift_y))
        return transform.warp(
            image_yx,
            tform,
            order=int(transform_order),
            mode="constant",
            cval=0.0,
            preserve_range=True,
        ).astype(np.float32, copy=False)
    return ndi_shift(
        image_yx,
        shift=tuple(float(v) for v in shift_yx),
        order=int(transform_order),
        mode="constant",
        cval=0.0,
        prefilter=True,
    ).astype(np.float32, copy=False)

def _apply_rotation_to_yx(
    image_yx: np.ndarray,
    correction_angle_deg: float,
    *,
    transform_order: int,
) -> np.ndarray:
    """Apply one in-plane XY rotation to one image plane."""

    return rotate(
        np.asarray(image_yx, dtype=np.float32),
        float(correction_angle_deg),
        resize=False,
        order=int(transform_order),
        mode="constant",
        cval=0.0,
        preserve_range=True,
    ).astype(np.float32, copy=False)

def _apply_rotation_to_zcyx(
    stack_zcyx: np.ndarray,
    correction_angle_deg: float,
    *,
    transform_order: int,
) -> np.ndarray:
    """Apply one in-plane XY rotation to all Z slices and channels of one time point."""

    rotated = np.empty_like(stack_zcyx, dtype=np.float32)
    for z in range(stack_zcyx.shape[0]):
        for c in range(stack_zcyx.shape[1]):
            rotated[z, c, :, :] = _apply_rotation_to_yx(
                stack_zcyx[z, c, :, :],
                correction_angle_deg,
                transform_order=transform_order,
            )
    return rotated

def _apply_translation_to_tzyx(
    stack_tzyx: np.ndarray,
    shift_yx: np.ndarray,
    *,
    transform_backend: str,
    transform_order: int,
) -> np.ndarray:
    """Apply one XY translation to all channels and Z slices of one time point."""

    shifted = np.empty_like(stack_tzyx, dtype=np.float32)
    for z in range(stack_tzyx.shape[0]):
        for c in range(stack_tzyx.shape[1]):
            shifted[z, c, :, :] = _apply_translation_to_yx(
                stack_tzyx[z, c, :, :],
                shift_yx,
                transform_backend=transform_backend,
                transform_order=transform_order,
            )
    return shifted

def _apply_translation_to_zcyx(
    stack_zcyx: np.ndarray,
    shift_zyx: np.ndarray,
    *,
    transform_backend: str,
    transform_order: int,
) -> np.ndarray:
    """Apply one ZYX translation to all channels of one time point."""

    shifted = np.empty_like(stack_zcyx, dtype=np.float32)
    shift_zyx = np.asarray(shift_zyx, dtype=np.float32)
    if transform_backend == "skimage" and np.isclose(float(shift_zyx[0]), 0.0):
        for z in range(stack_zcyx.shape[0]):
            for c in range(stack_zcyx.shape[1]):
                shifted[z, c, :, :] = _apply_translation_to_yx(
                    stack_zcyx[z, c, :, :],
                    shift_zyx[1:],
                    transform_backend=transform_backend,
                    transform_order=transform_order,
                )
        return shifted

    for c in range(stack_zcyx.shape[1]):
        shifted[:, c, :, :] = ndi_shift(
            np.asarray(stack_zcyx[:, c, :, :], dtype=np.float32),
            shift=tuple(float(v) for v in shift_zyx),
            order=int(transform_order),
            mode="constant",
            cval=0.0,
            prefilter=True,
        )
    return shifted

def _apply_translation_to_cyx(
    slice_cyx: np.ndarray,
    shift_yx: np.ndarray,
    *,
    transform_backend: str,
    transform_order: int,
) -> np.ndarray:
    """Apply one XY translation to all channels of a single Z slice."""

    shifted = np.empty_like(slice_cyx, dtype=np.float32)
    for c in range(slice_cyx.shape[0]):
        shifted[c, :, :] = _apply_translation_to_yx(
            slice_cyx[c, :, :],
            shift_yx,
            transform_backend=transform_backend,
            transform_order=transform_order,
        )
    return shifted

def _apply_translation_to_mask_zyx(
    mask_zyx: np.ndarray,
    shift_zyx: np.ndarray,
    *,
    transform_backend: str,
) -> np.ndarray:
    """Apply one ZYX translation to one validity-mask volume."""

    shift_zyx = np.asarray(shift_zyx, dtype=np.float32)
    if transform_backend == "skimage" and np.isclose(float(shift_zyx[0]), 0.0):
        translated = np.empty_like(mask_zyx, dtype=np.float32)
        for z in range(mask_zyx.shape[0]):
            translated[z, :, :] = _apply_translation_to_yx(
                mask_zyx[z, :, :],
                shift_zyx[1:],
                transform_backend=transform_backend,
                transform_order=1,
            )
        return translated
    return ndi_shift(
        np.asarray(mask_zyx, dtype=np.float32),
        shift=tuple(float(v) for v in shift_zyx),
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=True,
    ).astype(np.float32, copy=False)

def _apply_translation_to_mask_yx(
    mask_yx: np.ndarray,
    shift_yx: np.ndarray,
    *,
    transform_backend: str,
) -> np.ndarray:
    """Apply one YX translation to one validity-mask plane."""

    return _apply_translation_to_yx(
        mask_yx,
        shift_yx,
        transform_backend=transform_backend,
        transform_order=1,
    )

def _apply_rotation_to_mask_zyx(mask_zyx: np.ndarray, correction_angle_deg: float) -> np.ndarray:
    """Apply one in-plane XY rotation to all Z slices of one validity-mask volume."""

    rotated = np.empty_like(mask_zyx, dtype=np.float32)
    for z in range(mask_zyx.shape[0]):
        rotated[z, :, :] = _apply_rotation_to_yx(
            mask_zyx[z, :, :],
            float(correction_angle_deg),
            transform_order=1,
        )
    return rotated

def _print_verbose(verbose: bool, message: str) -> None:
    """Print a progress message only when verbose mode is enabled."""

    if verbose:
        print(message, flush=True)

def _memory_mark(memory_tracker, step: str) -> None:
    """Record an optional memory profiling marker."""

    if memory_tracker is None:
        return
    mark = getattr(memory_tracker, "mark", None)
    if mark is not None:
        mark(step)

def _correct_intra_stack_z_drift_impl(
    stack,
    *,
    registration_channel: int = 0,
    method: str = "phase_cross_correlation",
    reference_mode: str = "neighbor",
    neighbor_window_size: int = 3,
    projection_method: str = "max",
    filter_slices: bool = False,
    filter_projections: bool = False,
    median_kernel_size: int = 3,
    phase_cross_correlation_upsample_factor: int = 20,
    phase_cross_correlation_normalization: str | None = None,
    transform_backend: str = "skimage",
    transform_order: int = 1,
    n_jobs: int = 1,
    output_use_memmap: bool = False,
    output_memmap_folder: str | os.PathLike | None = None,
    output_memmap_name: str | None = None,
    output_dtype=np.float32,
    output_stage_name: str | None = "intra_stack",
    memory_tracker=None,
    verbose: bool = True,
    return_shifts: bool = False,
    pre_median_filter: bool | None = None,
    post_median_filter: bool | None = None,
):
    """
    Correct XY drift between Z slices within each time point of a ``TZCYX`` stack.

    Parameters
    ----------
    stack : array-like
        Input stack in canonical ``TZCYX`` order.
    registration_channel : int, optional
        Channel used to estimate slice-wise XY shifts. Shifts are applied to all
        channels of the affected Z slice.
    method : {"phase_cross_correlation", "pystackreg", "normcorre"}, optional
        Backend used for shift estimation. ``"normcorre"`` dispatches to
        ZenReg's standalone CaImAn-compatible NoRMCorre port while reusing this
        wrapper's shared registration, projection, shift-limit, and transform
        settings where possible.
    reference_mode : {"neighbor", "full_projection", "first_slice"}, optional
        Strategy for constructing each per-slice reference image.
    neighbor_window_size : int, optional
        Odd number of slices used for ``reference_mode="neighbor"``.
    projection_method : {"max", "mean", "median", "var", "std"}, optional
        Z-projection method used for reference construction. ``"max"`` remains
        a good default for sparse spots or puncta. ``"mean"`` is often better
        for dense, spatially extended signal. ``"median"`` is robust to
        outliers, but can attenuate sparse spots. ``"std"`` and ``"var"`` can
        be useful when contrast-rich structure matters more than absolute
        intensity. A percentile projection, for example p95, would also be a
        useful microscopy-oriented future extension.
    filter_slices : bool, optional
        If True, apply a slice-wise median filter before reference construction.
        This affects only shift estimation.
    filter_projections : bool, optional
        If True, apply a 2D median filter to moving and reference images just
        before shift estimation.
    median_kernel_size : int, optional
        Median filter kernel size used by the optional filters.
    phase_cross_correlation_upsample_factor : int, optional
        Subpixel upsampling factor for ``method="phase_cross_correlation"``.
    phase_cross_correlation_normalization : {None, "phase"}, optional
        Normalization mode forwarded to scikit-image's phase cross-correlation.
        ``None`` is more robust for the smooth synthetic examples.
    nc_* : optional
        NoRMCorre-specific settings used only with ``method="normcorre"``.
        Important options are ``nc_pw_rigid``, ``nc_strides``, ``nc_overlaps``,
        ``nc_max_deviation_rigid``, ``nc_n_iterations``,
        ``nc_correction_iterations``, ``nc_template_init_mode``,
        ``nc_template_update_method``, ``nc_gSig_filt``,
        ``nc_shift_interpolation``, ``nc_border_nan``, ``nc_n_jobs``, and the
        ``nc_output_*`` memory-mapped output controls. Shared settings such as
        ``registration_channel``, ``registration_stack``, ``registration_z_range``,
        ``projection_method``, ``max_xy_shifts``, ``max_z_shifts``,
        ``phase_cross_correlation_upsample_factor``,
        ``phase_cross_correlation_normalization``, and ``transform_order`` are
        reused directly and are not duplicated with an ``nc_`` prefix.
    transform_backend : {"skimage", "scipy"}, optional
        Backend used to apply correction shifts. ``"skimage"`` is the default
        for XY transforms and matches the rotation-correction path. ``"scipy"``
        keeps the legacy ``scipy.ndimage.shift`` behavior.
    transform_order : int, optional
        Interpolation order used for geometric correction. ``1`` is recommended
        for intensity microscopy data because it gives smooth subpixel shifts.
        ``0`` uses nearest-neighbor interpolation, which can preserve sparse
        puncta or label-like images more sharply, but subpixel corrections become
        more quantized.
    n_jobs : int, optional
        Number of CPU worker threads used for independent slice registrations.
        ``1`` keeps serial execution. ``-1`` uses all available CPUs.
    verbose : bool, optional
        If True, print progress messages.
    return_shifts : bool, optional
        If True, return ``(corrected_stack, shifts_tzyx)`` where shifts has shape
        ``T, Z, 2`` and stores ``(shift_y, shift_x)``.

    Returns
    -------
    numpy.ndarray or tuple[numpy.ndarray, numpy.ndarray]
        Corrected stack, optionally with the estimated shifts.
    """

    stack = ensure_tzcyx_stack(stack)
    method = _normalize_registration_method(method)
    reference_mode = _normalize_intra_stack_reference_mode(reference_mode)
    neighbor_window_size = _normalize_neighbor_window_size(neighbor_window_size)
    projection_method = _normalize_projection_method(projection_method)
    transform_backend = _normalize_transform_backend(transform_backend)
    transform_order = _normalize_transform_order(transform_order)
    n_jobs = _normalize_n_jobs(n_jobs)
    output_dtype = _normalize_output_dtype(output_dtype)
    filter_slices, filter_projections = _resolve_filter_aliases(
        filter_slices=filter_slices,
        filter_projections=filter_projections,
        pre_median_filter=pre_median_filter,
        post_median_filter=post_median_filter,
    )
    phase_cross_correlation_normalization = _normalize_phase_cross_correlation_normalization(
        phase_cross_correlation_normalization
    )

    if not 0 <= int(registration_channel) < stack.shape[2]:
        raise ValueError(
            f"registration_channel must be between 0 and {stack.shape[2] - 1}. "
            f"Got {registration_channel!r}."
        )
    if int(median_kernel_size) < 1:
        raise ValueError(f"median_kernel_size must be >= 1. Got {median_kernel_size!r}.")
    if int(phase_cross_correlation_upsample_factor) < 1:
        raise ValueError(
            "phase_cross_correlation_upsample_factor must be >= 1. "
            f"Got {phase_cross_correlation_upsample_factor!r}."
        )

    shifts = np.zeros((stack.shape[0], stack.shape[1], 2), dtype=np.float32)
    _memory_mark(memory_tracker, "intra_stack:start")
    if stack.shape[1] <= 1:
        _print_verbose(verbose, "Skipping intra-stack Z drift correction because Z <= 1.")
        corrected = _create_registered_output(
            tuple(int(v) for v in stack.shape),
            dtype=output_dtype,
            output_use_memmap=output_use_memmap,
            output_memmap_folder=output_memmap_folder,
            output_memmap_name=output_memmap_name,
            stage_name=output_stage_name,
        )
        for t, frame in _iter_map_ordered(
            lambda index: (int(index), np.asarray(stack[int(index)], dtype=output_dtype)),
            range(stack.shape[0]),
            n_jobs=n_jobs,
            progress=verbose,
            desc="ZenReg intra-stack copy",
        ):
            corrected[t, :, :, :, :] = frame
        _memory_mark(memory_tracker, "intra_stack:end")
        return (corrected, shifts) if return_shifts else corrected

    def process_timepoint(t: int) -> tuple[int, np.ndarray, np.ndarray]:
        working_volume = _as_float32_work_array(stack[int(t), :, int(registration_channel), :, :])
        if filter_slices:
            working_volume = _apply_median_to_zyx(working_volume, int(median_kernel_size))
        corrected_frame = np.empty(stack.shape[1:], dtype=output_dtype)
        shifts_t = np.zeros((stack.shape[1], 2), dtype=np.float32)
        for z in range(stack.shape[1]):
            moving_image = _as_float32_work_array(working_volume[z, :, :])
            reference_image = _build_intra_stack_reference_image(
                working_volume,
                z_index=z,
                reference_mode=reference_mode,
                neighbor_window_size=neighbor_window_size,
                projection_method=projection_method,
            ).astype(np.float32, copy=False)

            if filter_projections:
                kernel = (int(median_kernel_size), int(median_kernel_size))
                moving_image = median_filter(moving_image, size=kernel)
                reference_image = median_filter(reference_image, size=kernel)

            shift_yx = _estimate_shift(
                reference_image,
                moving_image,
                method=method,
                phase_cross_correlation_upsample_factor=int(
                    phase_cross_correlation_upsample_factor
                ),
                phase_cross_correlation_normalization=phase_cross_correlation_normalization,
            )
            corrected_frame[z, :, :, :] = _apply_translation_to_cyx(
                stack[int(t), z, :, :, :],
                shift_yx,
                transform_backend=transform_backend,
                transform_order=transform_order,
            ).astype(output_dtype, copy=False)
            shifts_t[z, :] = shift_yx
        return int(t), corrected_frame, shifts_t

    corrected = _create_registered_output(
        tuple(int(v) for v in stack.shape),
        dtype=output_dtype,
        output_use_memmap=output_use_memmap,
        output_memmap_folder=output_memmap_folder,
        output_memmap_name=output_memmap_name,
        stage_name=output_stage_name,
    )
    for t, corrected_frame, shifts_t in _iter_map_ordered(
        process_timepoint,
        range(stack.shape[0]),
        n_jobs=n_jobs,
        progress=verbose,
        desc="ZenReg intra-stack correction",
    ):
        shifts[t, :, :] = shifts_t
        for z, shift_yx in enumerate(shifts_t):
            _print_verbose(
                verbose,
                f"t={t} z={z} shift_y={float(shift_yx[0]):.3f} shift_x={float(shift_yx[1]):.3f}",
            )
        corrected[t, :, :, :, :] = corrected_frame

    _memory_mark(memory_tracker, "intra_stack:end")
    return (corrected, shifts) if return_shifts else corrected

def correct_intra_stack_z_drift(
    stack,
    *,
    registration_channel: int = 0,
    method: str = "phase_cross_correlation",
    reference_mode: str = "neighbor",
    neighbor_window_size: int = 3,
    projection_method: str = "max",
    filter_slices: bool = False,
    filter_projections: bool = False,
    median_kernel_size: int = 3,
    phase_cross_correlation_upsample_factor: int = 20,
    phase_cross_correlation_normalization: str | None = None,
    max_xy_shifts: tuple[float, float] | Sequence[float] | None = None,
    transform_backend: str = "skimage",
    transform_order: int = 1,
    n_jobs: int = 1,
    verbose: bool = True,
    return_shifts: bool = False,
    pre_median_filter: bool | None = None,
    post_median_filter: bool | None = None,
):
    """
    Correct XY drift between Z slices within each time point of a ``TZCYX`` stack.

    This compatibility wrapper performs only intra-stack correction. New workflows
    can call :func:`register_stack` with ``intra_stack=True`` and
    ``time_registration_mode="none"`` for the same behavior.
    """

    max_xy_shifts = _normalize_max_xy_shifts(max_xy_shifts)
    transform_backend = _normalize_transform_backend(transform_backend)
    transform_order = _normalize_transform_order(transform_order)
    n_jobs = _normalize_n_jobs(n_jobs)
    corrected = _correct_intra_stack_z_drift_impl(
        stack,
        registration_channel=registration_channel,
        method=method,
        reference_mode=reference_mode,
        neighbor_window_size=neighbor_window_size,
        projection_method=projection_method,
        filter_slices=filter_slices,
        filter_projections=filter_projections,
        median_kernel_size=median_kernel_size,
        phase_cross_correlation_upsample_factor=phase_cross_correlation_upsample_factor,
        phase_cross_correlation_normalization=phase_cross_correlation_normalization,
        transform_backend=transform_backend,
        transform_order=transform_order,
        n_jobs=n_jobs,
        verbose=verbose,
        return_shifts=True,
        pre_median_filter=pre_median_filter,
        post_median_filter=post_median_filter,
    )
    corrected_stack, shifts = corrected
    if max_xy_shifts is not None:
        shifts = np.asarray(
            [[_clip_shift_yx(shift, max_xy_shifts) for shift in shifts_t] for shifts_t in shifts],
            dtype=np.float32,
        )
        canonical_stack = ensure_tzcyx_stack(stack)
        corrected_stack = _create_registered_output(
            tuple(int(v) for v in canonical_stack.shape),
            dtype=np.float32,
            output_use_memmap=False,
            output_memmap_folder=None,
            output_memmap_name=None,
            stage_name=None,
        )

        def apply_clipped_slice(index: tuple[int, int]) -> tuple[int, int, np.ndarray]:
            t, z = index
            return t, z, _apply_translation_to_cyx(
                canonical_stack[t, z, :, :, :],
                shifts[t, z, :],
                transform_backend=transform_backend,
                transform_order=transform_order,
            )

        tasks = [(t, z) for t in range(corrected_stack.shape[0]) for z in range(corrected_stack.shape[1])]
        for t, z, corrected_slice in _iter_map_ordered(
            apply_clipped_slice,
            tasks,
            n_jobs=n_jobs,
            progress=verbose,
            desc="ZenReg intra-stack clipped apply",
        ):
            corrected_stack[t, z, :, :, :] = corrected_slice
    return (corrected_stack, shifts) if return_shifts else corrected_stack

def _registration_channel_volumes(
    stack: np.ndarray,
    *,
    registration_channel: int,
    zrange: tuple[int, int] | Sequence[int] | None,
    filter_slices: bool,
    median_kernel_size: int,
) -> np.ndarray:
    """Extract optional-Z-range registration-channel volumes as ``TZYX``."""

    z_start, z_stop = normalize_zrange(zrange, stack.shape[1], strict=True)
    volumes = np.asarray(stack[:, z_start:z_stop, registration_channel, :, :], dtype=np.float32).copy()
    if filter_slices:
        for t in range(volumes.shape[0]):
            volumes[t, :, :, :] = _apply_median_to_zyx(volumes[t, :, :, :], median_kernel_size)
    return volumes

def _time_reference_index(
    *,
    t: int,
    registration_stack: int,
    time_reference_mode: str,
) -> int:
    """Return the reference index used for one time point."""

    if time_reference_mode == "previous":
        return t - 1
    return registration_stack

def _reference_shift_for_time(
    shifts_zyx: np.ndarray,
    *,
    reference_t: int,
    time_reference_mode: str,
) -> np.ndarray:
    """Return the cumulative reference shift for the selected reference mode."""

    if time_reference_mode == "previous":
        return shifts_zyx[reference_t, :]
    return np.zeros(3, dtype=np.float32)

def _estimate_time_shift_from_projections(
    reference_volume: np.ndarray,
    moving_volume: np.ndarray,
    *,
    method: str,
    projection_method: str,
    filter_projections: bool,
    median_kernel_size: int,
    zreg: bool,
    max_xy_shifts: np.ndarray | None,
    max_z_shifts: float | None,
    phase_cross_correlation_upsample_factor: int,
    phase_cross_correlation_normalization: str | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate raw and clipped ``ZYX`` time-registration shifts from projections."""

    reference_projection = _project_zyx_to_yx(
        reference_volume,
        projection_method=projection_method,
    ).astype(np.float32, copy=False)
    moving_projection = _project_zyx_to_yx(
        moving_volume,
        projection_method=projection_method,
    ).astype(np.float32, copy=False)
    if filter_projections:
        kernel = (int(median_kernel_size), int(median_kernel_size))
        reference_projection = median_filter(reference_projection, size=kernel)
        moving_projection = median_filter(moving_projection, size=kernel)

    shift_yx = _estimate_shift(
        reference_projection,
        moving_projection,
        method=method,
        phase_cross_correlation_upsample_factor=phase_cross_correlation_upsample_factor,
        phase_cross_correlation_normalization=phase_cross_correlation_normalization,
    )
    shift_z = 0.0
    if zreg:
        shift_z = _estimate_z_shift_from_orthogonal_projections(
            reference_volume,
            moving_volume,
            method=method,
            projection_method=projection_method,
            filter_projections=filter_projections,
            median_kernel_size=median_kernel_size,
            phase_cross_correlation_upsample_factor=phase_cross_correlation_upsample_factor,
            phase_cross_correlation_normalization=phase_cross_correlation_normalization,
        )

    raw_shift_zyx = np.asarray([shift_z, shift_yx[0], shift_yx[1]], dtype=np.float32)
    clipped_shift_zyx = _clip_shift_zyx(
        raw_shift_zyx,
        max_z_shifts=max_z_shifts,
        max_xy_shifts=max_xy_shifts,
    )
    return raw_shift_zyx, clipped_shift_zyx

def _estimate_time_shift_from_full_3d(
    reference_volume: np.ndarray,
    moving_volume: np.ndarray,
    *,
    zreg: bool,
    max_xy_shifts: np.ndarray | None,
    max_z_shifts: float | None,
    phase_cross_correlation_upsample_factor: int,
    phase_cross_correlation_normalization: str | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate raw and clipped ``ZYX`` time-registration shifts from full 3D volumes."""

    raw_shift_zyx = _estimate_full_3d_shift(
        reference_volume,
        moving_volume,
        phase_cross_correlation_upsample_factor=phase_cross_correlation_upsample_factor,
        phase_cross_correlation_normalization=phase_cross_correlation_normalization,
    )
    if not zreg:
        raw_shift_zyx[0] = 0.0
    clipped_shift_zyx = _clip_shift_zyx(
        raw_shift_zyx,
        max_z_shifts=max_z_shifts,
        max_xy_shifts=max_xy_shifts,
    )
    return raw_shift_zyx.astype(np.float32, copy=False), clipped_shift_zyx

def _register_stack_across_time(
    stack,
    *,
    registration_channel: int,
    registration_stack: int,
    method: str,
    time_registration_mode: str,
    time_reference_mode: str,
    registration_template_time_range: tuple[int, int] | None,
    zrange: tuple[int, int] | Sequence[int] | None,
    projection_method: str,
    filter_slices: bool,
    filter_projections: bool,
    median_kernel_size: int,
    zreg: bool,
    max_xy_shifts: np.ndarray | None,
    max_z_shifts: float | None,
    phase_cross_correlation_upsample_factor: int,
    phase_cross_correlation_normalization: str | None,
    transform_backend: str,
    transform_order: int,
    n_jobs: int,
    output_use_memmap: bool,
    output_memmap_folder: str | os.PathLike | None,
    output_memmap_name: str | None,
    output_dtype,
    output_stage_name: str | None,
    memory_tracker=None,
    verbose: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Register a ``TZCYX`` stack across time and return raw/applied ``TZYX`` shifts."""

    effective_mode = time_registration_mode
    if time_registration_mode == "full_3d" and method != "phase_cross_correlation":
        warnings.warn(
            "Full 3D time registration is only supported with "
            "method='phase_cross_correlation'. Falling back to projection mode.",
            RuntimeWarning,
            stacklevel=3,
        )
        _print_verbose(
            verbose,
            "Full 3D time registration requires method='phase_cross_correlation'; "
            "falling back to projection mode.",
        )
        effective_mode = "projection"

    output_dtype = _normalize_output_dtype(output_dtype)
    shifts_zyx = np.zeros((stack.shape[0], 3), dtype=np.float32)
    raw_shifts_zyx = np.zeros_like(shifts_zyx)
    _print_verbose(
        verbose,
        (
            f"Registering {CANONICAL_AXIS_ORDER} stack with method='{method}', "
            f"time_registration_mode='{effective_mode}', "
            f"time_reference_mode='{time_reference_mode}', zreg={bool(zreg)}, "
            f"registration_channel={int(registration_channel)}, "
            f"registration_stack={registration_stack}, projection_method='{projection_method}'"
        ),
    )
    reference_template_volume = None
    if registration_template_time_range is not None:
        _memory_mark(memory_tracker, f"{output_stage_name or 'time'}:build_time_template:start")
        reference_template_volume = _extract_registration_template_volume(
            stack,
            registration_channel=registration_channel,
            registration_stack=registration_stack,
            registration_template_time_range=registration_template_time_range,
            zrange=zrange,
            projection_method=projection_method,
            filter_slices=filter_slices,
            median_kernel_size=median_kernel_size,
        )
        _memory_mark(memory_tracker, f"{output_stage_name or 'time'}:build_time_template:end")

    def estimate_pair_shift(t: int) -> tuple[int, int, np.ndarray, np.ndarray]:
        if (
            time_reference_mode == "template"
            and registration_template_time_range is None
            and t == registration_stack
        ):
            return t, t, np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
        if time_reference_mode == "previous" and t == 0:
            return t, t, np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

        reference_t = _time_reference_index(
            t=t,
            registration_stack=registration_stack,
            time_reference_mode=time_reference_mode,
        )
        if reference_template_volume is None:
            reference_volume = _extract_registration_volume(
                stack,
                t=reference_t,
                registration_channel=registration_channel,
                zrange=zrange,
                filter_slices=filter_slices,
                median_kernel_size=median_kernel_size,
            )
        else:
            reference_volume = reference_template_volume
        moving_volume = _extract_registration_volume(
            stack,
            t=t,
            registration_channel=registration_channel,
            zrange=zrange,
            filter_slices=filter_slices,
            median_kernel_size=median_kernel_size,
        )
        if effective_mode == "full_3d":
            pair_raw_shift_zyx, pair_shift_zyx = _estimate_time_shift_from_full_3d(
                reference_volume,
                moving_volume,
                zreg=zreg,
                max_xy_shifts=max_xy_shifts,
                max_z_shifts=max_z_shifts,
                phase_cross_correlation_upsample_factor=phase_cross_correlation_upsample_factor,
                phase_cross_correlation_normalization=phase_cross_correlation_normalization,
            )
        else:
            pair_raw_shift_zyx, pair_shift_zyx = _estimate_time_shift_from_projections(
                reference_volume,
                moving_volume,
                method=method,
                projection_method=projection_method,
                filter_projections=filter_projections,
                median_kernel_size=median_kernel_size,
                zreg=zreg,
                max_xy_shifts=max_xy_shifts,
                max_z_shifts=max_z_shifts,
                phase_cross_correlation_upsample_factor=phase_cross_correlation_upsample_factor,
                phase_cross_correlation_normalization=phase_cross_correlation_normalization,
            )

        return t, reference_t, pair_raw_shift_zyx, pair_shift_zyx

    _memory_mark(memory_tracker, f"{output_stage_name or 'time'}:estimate_shifts:start")
    pair_results = _parallel_map_ordered(
        estimate_pair_shift,
        range(stack.shape[0]),
        n_jobs=n_jobs,
        progress=verbose,
        desc=f"ZenReg {output_stage_name or 'time'} estimate shifts",
    )
    _memory_mark(memory_tracker, f"{output_stage_name or 'time'}:estimate_shifts:end")
    pair_shifts = np.zeros_like(shifts_zyx)
    pair_raw_shifts = np.zeros_like(shifts_zyx)
    reference_indices = np.zeros(stack.shape[0], dtype=np.int32)
    for t, reference_t, pair_raw_shift_zyx, pair_shift_zyx in pair_results:
        pair_raw_shifts[t, :] = pair_raw_shift_zyx
        pair_shifts[t, :] = pair_shift_zyx
        reference_indices[t] = int(reference_t)

    for t in range(stack.shape[0]):
        reference_t = int(reference_indices[t])
        reference_shift_zyx = _reference_shift_for_time(
            shifts_zyx,
            reference_t=reference_t,
            time_reference_mode=time_reference_mode,
        )
        reference_raw_shift_zyx = _reference_shift_for_time(
            raw_shifts_zyx,
            reference_t=reference_t,
            time_reference_mode=time_reference_mode,
        )
        raw_shifts_zyx[t, :] = reference_raw_shift_zyx + pair_raw_shifts[t, :]
        shifts_zyx[t, :] = _clip_shift_zyx(
            reference_shift_zyx + pair_shifts[t, :],
            max_z_shifts=max_z_shifts,
            max_xy_shifts=max_xy_shifts,
        )
        _print_verbose(
            verbose,
            (
                f"t={t} shift_z={float(shifts_zyx[t, 0]):.3f} "
                f"shift_y={float(shifts_zyx[t, 1]):.3f} "
                f"shift_x={float(shifts_zyx[t, 2]):.3f}"
            ),
        )

    registered = _create_registered_output(
        tuple(int(v) for v in stack.shape),
        dtype=output_dtype,
        output_use_memmap=output_use_memmap,
        output_memmap_folder=output_memmap_folder,
        output_memmap_name=output_memmap_name,
        stage_name=output_stage_name,
    )

    def apply_timepoint(t: int) -> tuple[int, np.ndarray]:
        return int(t), _apply_translation_to_zcyx(
            stack[int(t), :, :, :, :],
            shifts_zyx[t, :],
            transform_backend=transform_backend,
            transform_order=transform_order,
        ).astype(output_dtype, copy=False)

    _memory_mark(memory_tracker, f"{output_stage_name or 'time'}:apply_transforms:start")
    for t, registered_timepoint in _iter_map_ordered(
        apply_timepoint,
        range(stack.shape[0]),
        n_jobs=n_jobs,
        progress=verbose,
        desc=f"ZenReg {output_stage_name or 'time'} apply transforms",
    ):
        registered[t, :, :, :, :] = registered_timepoint
    _memory_mark(memory_tracker, f"{output_stage_name or 'time'}:apply_transforms:end")

    return registered, shifts_zyx, raw_shifts_zyx, effective_mode

def _register_stack_rotations_across_time(
    stack,
    *,
    registration_channel: int,
    registration_stack: int,
    time_reference_mode: str,
    registration_template_time_range: tuple[int, int] | None,
    zrange: tuple[int, int] | Sequence[int] | None,
    projection_method: str,
    filter_slices: bool,
    filter_projections: bool,
    median_kernel_size: int,
    max_rot_shifts: float | None,
    phase_cross_correlation_upsample_factor: int,
    phase_cross_correlation_normalization: str | None,
    transform_order: int,
    n_jobs: int,
    output_use_memmap: bool,
    output_memmap_folder: str | os.PathLike | None,
    output_memmap_name: str | None,
    output_dtype,
    output_stage_name: str | None,
    memory_tracker=None,
    verbose: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Register in-plane XY rotations across time and return raw/applied correction angles."""

    output_dtype = _normalize_output_dtype(output_dtype)
    angles_deg = np.zeros(stack.shape[0], dtype=np.float32)
    raw_angles_deg = np.zeros_like(angles_deg)
    _print_verbose(
        verbose,
        (
            f"Registering XY rotations with time_reference_mode='{time_reference_mode}', "
            f"registration_stack={registration_stack}, projection_method='{projection_method}'"
        ),
    )

    def build_projection(t: int) -> np.ndarray:
        volume = _extract_registration_volume(
            stack,
            t=int(t),
            registration_channel=registration_channel,
            zrange=zrange,
            filter_slices=filter_slices,
            median_kernel_size=median_kernel_size,
        )
        projection = _project_zyx_to_yx(
            volume,
            projection_method=projection_method,
        ).astype(np.float32, copy=False)
        if filter_projections:
            projection = median_filter(
                projection,
                size=(int(median_kernel_size), int(median_kernel_size)),
            )
        return projection

    reference_template_projection = None
    if registration_template_time_range is not None:
        template_volume = _extract_registration_template_volume(
            stack,
            registration_channel=registration_channel,
            registration_stack=registration_stack,
            registration_template_time_range=registration_template_time_range,
            zrange=zrange,
            projection_method=projection_method,
            filter_slices=filter_slices,
            median_kernel_size=median_kernel_size,
        )
        reference_template_projection = _project_zyx_to_yx(
            template_volume,
            projection_method=projection_method,
        ).astype(np.float32, copy=False)
        if filter_projections:
            reference_template_projection = median_filter(
                reference_template_projection,
                size=(int(median_kernel_size), int(median_kernel_size)),
            )

    def estimate_pair_rotation(t: int) -> tuple[int, int, float]:
        if (
            time_reference_mode == "template"
            and registration_template_time_range is None
            and t == registration_stack
        ):
            return t, t, 0.0
        if time_reference_mode == "previous" and t == 0:
            return t, t, 0.0

        reference_t = _time_reference_index(
            t=t,
            registration_stack=registration_stack,
            time_reference_mode=time_reference_mode,
        )
        reference_projection = (
            build_projection(reference_t)
            if reference_template_projection is None
            else reference_template_projection
        )
        detected_rotation_deg = _estimate_rotation_deg_from_projections(
            reference_projection,
            build_projection(t),
            phase_cross_correlation_upsample_factor=phase_cross_correlation_upsample_factor,
            phase_cross_correlation_normalization=phase_cross_correlation_normalization,
        )
        return t, reference_t, detected_rotation_deg

    _memory_mark(memory_tracker, f"{output_stage_name or 'rotation'}:estimate_rotations:start")
    pair_results = _parallel_map_ordered(
        estimate_pair_rotation,
        range(stack.shape[0]),
        n_jobs=n_jobs,
        progress=verbose,
        desc=f"ZenReg {output_stage_name or 'rotation'} estimate rotations",
    )
    _memory_mark(memory_tracker, f"{output_stage_name or 'rotation'}:estimate_rotations:end")
    pair_rotations = np.zeros(stack.shape[0], dtype=np.float32)
    pair_raw_rotations = np.zeros(stack.shape[0], dtype=np.float32)
    reference_indices = np.zeros(stack.shape[0], dtype=np.int32)
    for t, reference_t, detected_rotation_deg in pair_results:
        pair_raw_rotations[t] = float(detected_rotation_deg)
        pair_rotations[t] = _clip_rotation_deg(float(detected_rotation_deg), max_rot_shifts)
        reference_indices[t] = int(reference_t)

    for t in range(stack.shape[0]):
        reference_t = int(reference_indices[t])
        reference_angle = float(angles_deg[reference_t]) if time_reference_mode == "previous" else 0.0
        reference_raw_angle = float(raw_angles_deg[reference_t]) if time_reference_mode == "previous" else 0.0
        raw_angles_deg[t] = reference_raw_angle - float(pair_raw_rotations[t])
        angles_deg[t] = _clip_rotation_deg(
            reference_angle - float(pair_rotations[t]),
            max_rot_shifts,
        )
        _print_verbose(verbose, f"t={t} rotation_correction_deg={float(angles_deg[t]):.3f}")

    registered = _create_registered_output(
        tuple(int(v) for v in stack.shape),
        dtype=output_dtype,
        output_use_memmap=output_use_memmap,
        output_memmap_folder=output_memmap_folder,
        output_memmap_name=output_memmap_name,
        stage_name=output_stage_name,
    )

    def apply_timepoint(t: int) -> tuple[int, np.ndarray]:
        return int(t), _apply_rotation_to_zcyx(
            stack[int(t), :, :, :, :],
            float(angles_deg[t]),
            transform_order=transform_order,
        ).astype(output_dtype, copy=False)

    _memory_mark(memory_tracker, f"{output_stage_name or 'rotation'}:apply_rotations:start")
    for t, registered_timepoint in _iter_map_ordered(
        apply_timepoint,
        range(stack.shape[0]),
        n_jobs=n_jobs,
        progress=verbose,
        desc=f"ZenReg {output_stage_name or 'rotation'} apply rotations",
    ):
        registered[t, :, :, :, :] = registered_timepoint
    _memory_mark(memory_tracker, f"{output_stage_name or 'rotation'}:apply_rotations:end")

    return registered, angles_deg, raw_angles_deg

def _return_registration_result(
    registered: np.ndarray,
    *,
    return_shifts: bool,
    return_details: bool,
    compatible_time_shifts_yx: bool,
    time_shifts_zyx: np.ndarray | None,
    time_shifts_zyx_raw: np.ndarray | None,
    intra_stack_shifts_yx: np.ndarray | None,
    rotation_shifts_deg: np.ndarray | None,
    rotation_shifts_deg_raw: np.ndarray | None,
    translation_pass_shifts_zyx: list[np.ndarray],
    translation_pass_shifts_zyx_raw: list[np.ndarray],
    rotation_pass_shifts_deg: list[np.ndarray],
    rotation_pass_shifts_deg_raw: list[np.ndarray],
    zero_clip_bounds: dict[str, int] | None,
    zero_clip_failed_reason: str | None,
    zero_clip_mode: str,
    zero_clip_mask_threshold: float,
    zero_clip_mask_strategy: str,
    zero_clip_mask_min_fraction: float,
    zero_clip_margin_zyx: np.ndarray,
    time_registration_mode: str,
    effective_time_registration_mode: str,
    time_reference_mode: str,
    transform_backend: str,
    transform_order: int,
    pearson_correlations_before: np.ndarray | None,
    registration_settings: dict,
):
    """Return a backwards-compatible shift object for simple cases."""

    if not return_shifts:
        return registered
    details = {
        "time_shifts_zyx": time_shifts_zyx,
        "time_shifts_yx": None if time_shifts_zyx is None else time_shifts_zyx[:, 1:],
        "time_shifts_zyx_raw": time_shifts_zyx_raw,
        "time_shifts_yx_raw": None if time_shifts_zyx_raw is None else time_shifts_zyx_raw[:, 1:],
        "intra_stack_shifts_yx": intra_stack_shifts_yx,
        "rotation_shifts_deg": rotation_shifts_deg,
        "rotation_shifts_deg_raw": rotation_shifts_deg_raw,
        "translation_pass_shifts_zyx": translation_pass_shifts_zyx,
        "translation_pass_shifts_zyx_raw": translation_pass_shifts_zyx_raw,
        "rotation_pass_shifts_deg": rotation_pass_shifts_deg,
        "rotation_pass_shifts_deg_raw": rotation_pass_shifts_deg_raw,
        "zero_clip_bounds": zero_clip_bounds,
        "zero_clip_failed_reason": zero_clip_failed_reason,
        "zero_clip_mode": zero_clip_mode,
        "zero_clip_mask_threshold": zero_clip_mask_threshold,
        "zero_clip_mask_strategy": zero_clip_mask_strategy,
        "zero_clip_mask_min_fraction": zero_clip_mask_min_fraction,
        "zero_clip_margin_zyx": tuple(int(v) for v in zero_clip_margin_zyx),
        "time_registration_mode": time_registration_mode,
        "effective_time_registration_mode": effective_time_registration_mode,
        "time_reference_mode": time_reference_mode,
        "transform_backend": transform_backend,
        "transform_order": transform_order,
        "pearson_correlations_before": pearson_correlations_before,
        **registration_settings,
    }
    if return_details:
        return registered, details
    if (
        intra_stack_shifts_yx is None
        and rotation_shifts_deg is None
        and zero_clip_bounds is None
        and compatible_time_shifts_yx
        and time_shifts_zyx is not None
    ):
        return registered, time_shifts_zyx[:, 1:]
    if time_shifts_zyx is None and rotation_shifts_deg is None and zero_clip_bounds is None and intra_stack_shifts_yx is not None:
        return registered, intra_stack_shifts_yx
    return registered, details

def _normcorre_max_shifts_from_common_limits(
    *,
    is3d: bool,
    max_xy_shifts,
    max_z_shifts,
):
    """Map register_stack shift-limit arguments to NoRMCorre spatial order."""

    if is3d:
        if max_xy_shifts is None and max_z_shifts is None:
            return None
        max_z = np.inf if max_z_shifts is None else float(max_z_shifts)
        if max_xy_shifts is None:
            max_y, max_x = np.inf, np.inf
        else:
            max_y, max_x = [float(v) for v in max_xy_shifts]
        return (max_z, max_y, max_x)
    return None if max_xy_shifts is None else tuple(float(v) for v in max_xy_shifts)

def _register_stack_normcorre_from_main_wrapper(
    stack: np.ndarray,
    *,
    registration_channel: int,
    registration_stack: int,
    time_registration_mode: str,
    time_reference_mode: str,
    intra_stack: bool,
    projection_range,
    projection_method: str,
    zreg: bool,
    zero_clip: bool,
    rotreg: bool,
    filter_slices: bool,
    filter_projections: bool,
    max_xy_shifts,
    max_z_shifts,
    phase_cross_correlation_upsample_factor: int,
    phase_cross_correlation_normalization: str | None,
    transform_order: int,
    nc_pw_rigid: bool,
    nc_strides,
    nc_overlaps,
    nc_max_deviation_rigid,
    nc_n_iterations: int,
    nc_correction_iterations: int,
    nc_niter_rig: int,
    nc_template_init_mode: str,
    nc_template_update_method: str,
    nc_splits: int,
    nc_gSig_filt,
    nc_add_to_movie,
    nc_nonneg_movie: bool,
    nc_shift_interpolation: str,
    nc_n_jobs: int,
    nc_transform_mode: str,
    nc_transform_cval: float,
    nc_border_nan,
    nc_block_size: int,
    nc_output_use_memmap: bool,
    nc_output_memmap_folder: str | None,
    nc_output_memmap_name: str | None,
    memory_tracker=None,
    verbose: bool,
    return_shifts: bool,
    return_details: bool,
    registration_settings: dict,
):
    """Dispatch ``register_stack(method='normcorre')`` to the NoRMCorre module."""

    if intra_stack:
        raise ValueError("method='normcorre' does not support intra_stack=True yet.")
    if time_reference_mode != "template":
        raise ValueError("method='normcorre' currently supports only time_reference_mode='template'.")
    if time_registration_mode == "none":
        raise ValueError("method='normcorre' requires time_registration_mode='projection' or 'full_3d'.")
    if rotreg:
        raise ValueError("method='normcorre' does not support rotreg=True; use register_stack rotation correction instead.")
    if zero_clip:
        raise ValueError("method='normcorre' does not support zero_clip=True yet.")
    if filter_slices or filter_projections:
        raise ValueError(
            "method='normcorre' does not use filter_slices/filter_projections. "
            "Use nc_gSig_filt for CaImAn-style high-pass filtering."
        )

    is3d = bool(time_registration_mode == "full_3d" or zreg)
    if is3d and stack.shape[1] < 2:
        raise ValueError("NoRMCorre full-3D/zreg mode requires SizeZ >= 2.")

    from .normcorre import register_stack_normcorre

    _memory_mark(memory_tracker, "normcorre:start")
    registered, details = register_stack_normcorre(
        stack,
        registration_channel=registration_channel,
        registration_stack=registration_stack,
        is3d=is3d,
        projection_range=projection_range,
        projection_method=projection_method,
        pw_rigid=nc_pw_rigid,
        strides=nc_strides,
        overlaps=nc_overlaps,
        max_shifts=_normcorre_max_shifts_from_common_limits(
            is3d=is3d,
            max_xy_shifts=max_xy_shifts,
            max_z_shifts=max_z_shifts,
        ),
        max_deviation_rigid=nc_max_deviation_rigid,
        n_iterations=nc_n_iterations,
        correction_iterations=nc_correction_iterations,
        niter_rig=nc_niter_rig,
        template_init_mode=nc_template_init_mode,
        template_update_method=nc_template_update_method,
        splits=nc_splits,
        upsample_factor=phase_cross_correlation_upsample_factor,
        normalization=phase_cross_correlation_normalization,
        gSig_filt=nc_gSig_filt,
        add_to_movie=nc_add_to_movie,
        nonneg_movie=nc_nonneg_movie,
        shift_interpolation=nc_shift_interpolation,
        n_jobs=nc_n_jobs,
        transform_order=transform_order,
        transform_mode=nc_transform_mode,
        transform_cval=nc_transform_cval,
        border_nan=nc_border_nan,
        block_size=nc_block_size,
        output_use_memmap=nc_output_use_memmap,
        output_memmap_folder=nc_output_memmap_folder,
        output_memmap_name=nc_output_memmap_name,
        verbose=verbose,
        return_details=True,
    )
    _memory_mark(memory_tracker, "normcorre:end")
    details.update(registration_settings)
    details["method"] = "normcorre"
    details["time_registration_mode"] = "full_3d" if is3d else "projection"
    details["effective_time_registration_mode"] = "full_3d" if is3d else "projection"
    details["zreg"] = bool(is3d)
    details["max_xy_shifts"] = None if max_xy_shifts is None else tuple(float(v) for v in max_xy_shifts)
    details["max_z_shifts"] = None if max_z_shifts is None else float(max_z_shifts)
    details["phase_cross_correlation_upsample_factor"] = int(phase_cross_correlation_upsample_factor)
    details["nc_pw_rigid"] = bool(nc_pw_rigid)
    details["nc_strides"] = details.get("strides")
    details["nc_overlaps"] = details.get("overlaps")
    details["nc_max_deviation_rigid"] = details.get("max_deviation_rigid")
    details["nc_n_iterations"] = int(nc_n_iterations)
    details["nc_correction_iterations"] = int(nc_correction_iterations)
    details["nc_niter_rig"] = int(nc_niter_rig)
    details["nc_template_init_mode"] = nc_template_init_mode
    details["nc_template_update_method"] = nc_template_update_method
    details["nc_splits"] = int(nc_splits)
    details["nc_gSig_filt"] = details.get("gSig_filt")
    details["nc_shift_interpolation"] = nc_shift_interpolation
    details["nc_border_nan"] = nc_border_nan
    details["nc_n_jobs"] = int(nc_n_jobs)
    if return_shifts or return_details:
        details["pearson_correlations_before"] = _compute_registration_frame_correlations(
            stack,
            registration_channel=int(registration_channel),
            registration_stack=int(registration_stack),
            registration_template_time_range=None,
            projection_range=projection_range,
            projection_method=projection_method,
            effective_time_registration_mode="full_3d" if is3d else "projection",
        )

    if return_details:
        return registered, details
    if return_shifts:
        if not is3d:
            return registered, details["time_shifts_yx"]
        return registered, details
    return registered

def _register_stack_rigid_3d_from_main_wrapper(
    stack: np.ndarray,
    *,
    registration_channel: int,
    registration_stack: int,
    method: str,
    projection_range,
    projection_method: str,
    rigid_3d_backend: str,
    zero_clip: bool,
    zero_clip_mode: str,
    zero_clip_mask_threshold: float,
    zero_clip_mask_strategy: str,
    zero_clip_mask_min_fraction: float,
    zero_clip_margin_zyx: np.ndarray,
    max_rot_shifts: float | None,
    phase_cross_correlation_upsample_factor: int,
    phase_cross_correlation_normalization: str | None,
    transform_order: int,
    rot_spacing_zyx,
    rot_init_iterations: int,
    rot_metric: str,
    rot_shrink_factors,
    rot_smoothing_sigmas,
    rot_iterations: int,
    rot_learning_rate: float,
    rot_min_step: float,
    rot_sampling_percentage: float | None,
    rot_cval: float,
    rot_n_jobs: int,
    rot_points_max_points: int,
    rot_points_min_distance: int,
    rot_points_threshold_rel: float,
    rot_points_iterations: int,
    rot_points_max_match_distance: float,
    output_use_memmap: bool,
    output_memmap_folder: str | os.PathLike | None,
    output_memmap_name: str | None,
    output_dtype,
    memory_tracker=None,
    verbose: bool,
    return_shifts: bool,
    return_details: bool,
    registration_settings: dict,
):
    """Dispatch full 3D rigid rotation registration from ``register_stack``."""

    from .rigid3d import register_stack_rigid_3d

    _memory_mark(memory_tracker, "rigid_3d:start")
    registered, rigid_details = register_stack_rigid_3d(
        stack,
        registration_channel=registration_channel,
        registration_stack=registration_stack,
        backend=rigid_3d_backend,
        projection_range=projection_range,
        projection_method=projection_method,
        spacing_zyx=rot_spacing_zyx,
        init_iterations=rot_init_iterations,
        max_rot_shifts=max_rot_shifts,
        upsample_factor=phase_cross_correlation_upsample_factor,
        normalization=phase_cross_correlation_normalization,
        metric=rot_metric,
        shrink_factors=rot_shrink_factors,
        smoothing_sigmas=rot_smoothing_sigmas,
        iterations=rot_iterations,
        learning_rate=rot_learning_rate,
        min_step=rot_min_step,
        sampling_percentage=rot_sampling_percentage,
        transform_order=transform_order,
        cval=rot_cval,
        points_max_points=rot_points_max_points,
        points_min_distance=rot_points_min_distance,
        points_threshold_rel=rot_points_threshold_rel,
        points_iterations=rot_points_iterations,
        points_max_match_distance=rot_points_max_match_distance,
        n_jobs=rot_n_jobs,
        output_use_memmap=output_use_memmap,
        output_memmap_folder=output_memmap_folder,
        output_memmap_name=output_memmap_name,
        output_dtype=output_dtype,
        return_valid_mask=bool(zero_clip),
        verbose=verbose,
    )
    _memory_mark(memory_tracker, "rigid_3d:end")
    valid_mask_tzyx = rigid_details.pop("valid_mask_tzyx", None)
    zero_clip_bounds = None
    zero_clip_failed_reason = None
    effective_zero_clip_mode = _effective_zero_clip_mode(
        zero_clip=bool(zero_clip),
        zero_clip_mode=zero_clip_mode,
        rotreg=True,
    )
    if zero_clip:
        if valid_mask_tzyx is None:
            raise RuntimeError("Rigid 3D zero clipping requires a valid-mask transform.")
        try:
            _memory_mark(memory_tracker, "zero_clip:compute_bounds:start")
            effective_mask_strategy = _effective_zero_clip_mask_strategy(
                zero_clip_mask_strategy=zero_clip_mask_strategy,
                rigid_3d=True,
            )
            zero_clip_bounds = _crop_bounds_from_valid_mask(
                valid_mask_tzyx,
                threshold=zero_clip_mask_threshold,
                strategy=effective_mask_strategy,
                min_fraction=zero_clip_mask_min_fraction,
            )
            zero_clip_bounds = _apply_zero_clip_margin(zero_clip_bounds, zero_clip_margin_zyx)
            _memory_mark(memory_tracker, "zero_clip:compute_bounds:end")
            _memory_mark(memory_tracker, "zero_clip:crop:start")
            registered = _zero_clip_stack(
                registered,
                zero_clip_bounds,
                output_use_memmap=output_use_memmap,
                output_memmap_folder=output_memmap_folder,
                output_memmap_name=output_memmap_name,
                output_dtype=output_dtype,
                n_jobs=rot_n_jobs,
                progress=verbose,
            )
            _memory_mark(memory_tracker, "zero_clip:crop:end")
        except ValueError as exc:
            zero_clip_failed_reason = str(exc)
            zero_clip_bounds = None
            warnings.warn(
                "zero_clip=True was requested, but no usable common valid "
                f"3D region could be found. Returning the registered stack without cropping. {exc}",
                RuntimeWarning,
                stacklevel=2,
            )

    details = {
        **rigid_details,
        **registration_settings,
        "method": method,
        "rigid_3d_backend": rigid_3d_backend,
        "time_registration_mode": "full_3d",
        "effective_time_registration_mode": "full_3d",
        "time_reference_mode": "template",
        "zreg": True,
        "rotreg": True,
        "rotation_shifts_deg": rigid_details["rotation_shifts_zyx_deg"][:, 0],
        "zero_clip": bool(zero_clip),
        "zero_clip_mode": effective_zero_clip_mode,
        "zero_clip_mask_strategy": _effective_zero_clip_mask_strategy(
            zero_clip_mask_strategy=zero_clip_mask_strategy,
            rigid_3d=True,
        ),
        "zero_clip_bounds": zero_clip_bounds,
        "zero_clip_failed_reason": zero_clip_failed_reason,
        "zero_clip_mask_threshold": zero_clip_mask_threshold,
        "zero_clip_mask_min_fraction": zero_clip_mask_min_fraction,
        "zero_clip_margin_zyx": tuple(int(v) for v in zero_clip_margin_zyx),
        "transform_backend": "rigid_3d",
        "transform_order": transform_order,
    }
    if return_shifts or return_details:
        details["pearson_correlations_before"] = _compute_registration_frame_correlations(
            stack,
            registration_channel=int(registration_channel),
            registration_stack=int(registration_stack),
            registration_template_time_range=None,
            projection_range=projection_range,
            projection_method=projection_method,
            effective_time_registration_mode="full_3d",
        )
    if return_details:
        return registered, details
    if return_shifts:
        return registered, details
    return registered

# %% MAIN REGISTRATION WRAPPER
def register_stack(
    stack,
    *,
    metadata: dict | None = None,
    registration_channel: int,
    registration_stack: int = 0,
    method: str = "phase_cross_correlation",
    time_registration_mode: str = "projection",
    time_reference_mode: str = "template",
    registration_template_time_range: tuple[int, int] | Sequence[int] | str | None = None,
    intra_stack: bool = False,
    registration_z_range: tuple[int, int] | Sequence[int] | None = None,
    zrange: tuple[int, int] | Sequence[int] | None = None,
    projection_range: tuple[int, int] | Sequence[int] | None = None,
    projection_method: str = "max",
    zreg: bool = False,
    rigid_3d_backend: str = "phase_cross_correlation",
    zero_clip: bool = False,
    zero_clip_mode: str = "auto",
    zero_clip_mask_threshold: float = 0.999,
    zero_clip_mask_strategy: str = "auto",
    zero_clip_mask_min_fraction: float = 0.50,
    zero_clip_margin: int | Sequence[int] = 0,
    max_xy_shifts: tuple[float, float] | Sequence[float] | None = None,
    max_z_shifts: float | None = None,
    rotreg: bool = False,
    max_rot_shifts: float | None = None,
    rotreg_iter: int = 1,
    rot_spacing_zyx: tuple[float, float, float] | Sequence[float] | None = None,
    rot_init_iterations: int = 1,
    rot_metric: str = "correlation",
    rot_shrink_factors: tuple[int, ...] | Sequence[int] = (4, 2, 1),
    rot_smoothing_sigmas: tuple[float, ...] | Sequence[float] = (2.0, 1.0, 0.0),
    rot_iterations: int = 100,
    rot_learning_rate: float = 1.0,
    rot_min_step: float = 1e-4,
    rot_sampling_percentage: float | None = None,
    rot_cval: float = 0.0,
    rot_n_jobs: int = 1,
    rot_points_max_points: int = 200,
    rot_points_min_distance: int = 3,
    rot_points_threshold_rel: float = 0.25,
    rot_points_iterations: int = 20,
    rot_points_max_match_distance: float = 8.0,
    transform_backend: str = "skimage",
    transform_order: int = 1,
    intra_stack_reference_mode: str = "neighbor",
    neighbor_window_size: int = 3,
    filter_slices: bool = False,
    filter_projections: bool = False,
    median_kernel_size: int = 3,
    phase_cross_correlation_upsample_factor: int = 20,
    phase_cross_correlation_normalization: str | None = None,
    nc_pw_rigid: bool = True,
    nc_strides: tuple[int, ...] | int | None = None,
    nc_overlaps: tuple[int, ...] | int | None = None,
    nc_max_deviation_rigid: tuple[float, ...] | float | None = None,
    nc_n_iterations: int = 1,
    nc_correction_iterations: int = 1,
    nc_niter_rig: int = 1,
    nc_template_init_mode: str = "registration_stack",
    nc_template_update_method: str = "caiman",
    nc_splits: int = 56,
    nc_gSig_filt=None,
    nc_add_to_movie: float | None = None,
    nc_nonneg_movie: bool = True,
    nc_shift_interpolation: str = "resize",
    nc_n_jobs: int = 1,
    nc_transform_mode: str = "constant",
    nc_transform_cval: float = 0.0,
    nc_border_nan=None,
    nc_block_size: int = 32,
    nc_output_use_memmap: bool = False,
    nc_output_memmap_folder: str | None = None,
    nc_output_memmap_name: str | None = "zenreg_normcorre_registered",
    output_use_memmap: bool = False,
    output_memmap_folder: str | None = None,
    output_memmap_name: str | None = "zenreg_registered",
    output_dtype=np.float32,
    n_jobs: int = 1,
    memory_tracker=None,
    verbose: bool = True,
    return_shifts: bool = False,
    return_details: bool = False,
    pre_median_filter: bool | None = None,
    post_median_filter: bool | None = None,
):
    """
    Main ZenReg registration wrapper for canonical ``TZCYX`` stacks.

    By default this performs the original behavior: time-wise XY registration
    from Z projections, with each estimated XY shift applied to every Z slice and
    channel at the corresponding time point. Additional switches can enable
    intra-stack slice correction, full-3D phase-cross-correlation time
    registration, sequential ``t`` to ``t-1`` registration, and optional Z-shift
    correction.

    Parameters
    ----------
    stack : array-like
        Input stack in canonical ``TZCYX`` order.
    metadata : dict or None, optional
        OMIO metadata dictionary returned by ``load_stack(..., return_metadata=True)``.
        When ``rotreg=True`` and ``rigid_3d_backend`` is ``"simpleitk"`` or
        ``"points"``, ZenReg uses ``PhysicalSizeZ``, ``PhysicalSizeY``, and
        ``PhysicalSizeX`` from this metadata as ``rot_spacing_zyx`` if
        ``rot_spacing_zyx`` is left as ``None``. Passing ``rot_spacing_zyx``
        explicitly overrides metadata-derived spacing.
    registration_channel : int
        Channel used to compute the time-wise registration shifts.
    registration_stack : int, optional
        Time point used as reference/template for registration. Defaults to 0.
    method : {"phase_cross_correlation", "pystackreg", "normcorre"}, optional
        Backend used for shift estimation. With ``rigid_3d_backend="simpleitk"``
        or ``"points"``, this mainly documents the translational/pre-estimation
        strategy while the full 3D rigid transform is refined by the selected
        3D backend.
    time_registration_mode : {"projection", "full_3d", "none"}, optional
        Time-registration strategy. ``"projection"`` registers per-time-point
        projections and applies the resulting shift to each Z slice. ``"full_3d"``
        registers full ZYX volumes with scikit-image phase cross-correlation and
        can estimate Z, Y, and X shifts. ``"none"`` disables time registration,
        useful for intra-stack-only correction.
    time_reference_mode : {"template", "previous"}, optional
        ``"template"`` registers every time point to ``registration_stack``.
        ``"previous"`` registers each ``t`` to ``t-1`` and accumulates the
        correction shifts through time.
    registration_template_time_range : tuple[int, int], "all", or None, optional
        Optional half-open time range ``(start, stop)`` used to build an
        averaged registration template for ``time_reference_mode="template"``.
        ``"all"`` expands to ``(0, T)``. When set, the template is aggregated
        from the selected time points using ``projection_method`` along the time
        axis. For 2D+t stacks this creates a more stable YX template from
        multiple frames; for 3D+t stacks it creates a ZYX template before
        optional Z projection. ``None`` preserves the default behavior of using
        ``registration_stack`` as a single reference frame.
    intra_stack : bool, optional
        If True, run intra-stack XY slice correction independently for each time
        point before optional time registration.
    registration_z_range : tuple[int, int] or None, optional
        Optional half-open Z range ``(start, stop)`` used to define the
        registration signal. This selects slices along the canonical ``Z`` axis
        for projection-based registration, full-3D translational registration,
        rotation pre-estimation, and correlation reporting. ``None`` uses all
        Z slices.
    zrange : tuple[int, int] or None, optional
        Deprecated alias for ``registration_z_range``.
    projection_range : tuple[int, int] or None, optional
        Deprecated alias for ``registration_z_range``. The old name is kept for
        compatibility, but ``registration_z_range`` is clearer because this
        range is also used by full-3D registration paths, not only projections.
    projection_method : {"max", "mean", "median", "var", "std"}, optional
        Z-projection method used for shift estimation. ``"max"`` remains a
        good default for sparse spots or puncta. ``"mean"`` is often better
        for dense, spatially extended signal. ``"median"`` is robust to
        outliers, but can attenuate sparse spots. ``"std"`` and ``"var"`` can
        be useful when contrast-rich structure matters more than absolute
        intensity. A percentile projection, for example p95, would also be a
        useful microscopy-oriented future extension.
    zreg : bool, optional
        If True, estimate and apply Z shifts during time registration. In
        ``time_registration_mode="full_3d"`` this uses the Z component returned
        by 3D phase cross-correlation. In projection mode this estimates Z shifts
        from orthogonal ``ZY`` and ``ZX`` projections.
    rigid_3d_backend : {"phase_cross_correlation", "simpleitk", "points"}, optional
        Backend used when ``rotreg=True``. The default keeps the previous
        projection-based XY rotation path. ``"simpleitk"`` runs full 6-DOF dense
        rigid-volume registration. ``"points"`` estimates a rigid transform from
        sparse detected 3D peaks and is intended for puncta/spot-like data.
    zero_clip : bool, optional
        If True, crop zero-fill borders introduced by registration correction.
    zero_clip_mode : {"auto", "shift", "mask"}, optional
        Strategy used for ``zero_clip=True``. ``"shift"`` derives crop widths
        direction-wise from the largest applied translation corrections.
        ``"mask"`` applies the same transformations to an internal all-ones
        validity mask and crops to the common valid ``Z/Y/X`` region, which is
        robust for rotation-induced angled borders. ``"auto"`` uses ``"mask"``
        when ``rotreg=True`` and ``"shift"`` otherwise.
    zero_clip_mask_threshold : float, optional
        Threshold for mask-based zero-clipping. Conservative values close to
        ``1`` remove interpolated mask-edge pixels.
    zero_clip_mask_strategy : {"auto", "greedy", "relaxed", "max_volume"}, optional
        Crop strategy for mask-based zero-clipping. ``"greedy"`` is strict and
        removes border faces until the remaining cuboid contains only valid
        pixels. ``"max_volume"`` is also strict, but searches for the largest
        all-valid cuboid. ``"relaxed"`` removes only border planes whose valid
        fraction is below ``zero_clip_mask_min_fraction``; this is usually more
        practical for full 3D rotations, where tiny angled corner wedges should
        not force a large crop. ``"auto"`` uses ``"relaxed"`` for SimpleITK or
        points-based full 3D rigid registration and ``"greedy"`` otherwise.
    zero_clip_mask_min_fraction : float, optional
        Minimum valid border-face fraction for
        ``zero_clip_mask_strategy="relaxed"``. The default keeps border faces
        once the majority of their pixels are valid. Lower values preserve more
        field of view while allowing more angled invalid corner pixels to remain;
        higher values crop more aggressively.
    zero_clip_margin : int or tuple[int, int, int], optional
        Extra symmetric crop margin in ``(z, y, x)`` after automatic crop-bound
        detection. A scalar applies the same margin to all three axes.
    max_xy_shifts : tuple[float, float] or None, optional
        Optional absolute ``(max_y, max_x)`` correction-shift limits. If None,
        XY shifts are not clipped.
    max_z_shifts : int, float, or None, optional
        Optional absolute Z correction-shift limit. If None, Z shifts are not
        clipped.
    rotreg : bool, optional
        If True, estimate and correct rotations across time. With the default
        ``rigid_3d_backend="phase_cross_correlation"``, this uses the existing
        projection-based in-plane XY rotation correction. With
        ``rigid_3d_backend="simpleitk"`` or ``"points"``, this switches to full
        3D rigid-volume registration with Z/Y/X translations and rotations.
    max_rot_shifts : int, float, or None, optional
        Optional absolute rotation correction limit in degrees. If None,
        rotation corrections are not clipped.
    rotreg_iter : int, optional
        Number of translation-rotation refinement iterations for ``rotreg=True``.
        ``1`` runs translation, rotation, translation. Values > 1 repeat the
        rotation and final translation pattern.
    rot_spacing_zyx : tuple[float, float, float] or None, optional
        Physical voxel spacing in ``(z, y, x)`` used by full 3D rigid
        registration. This is important for anisotropic stacks. If ``None``,
        ZenReg uses OMIO metadata physical sizes when available and otherwise
        falls back to unit spacing.
    rot_init_iterations : int, optional
        Number of orthogonal-projection initialization passes before full 3D
        rigid refinement. Higher values can improve the coarse rotation
        starting point but add runtime.
    rot_metric : {"correlation", "mutual_information"}, optional
        Similarity metric used by the SimpleITK full 3D rigid backend.
        ``"correlation"`` is suitable for same-modality volumes.
    rot_shrink_factors : sequence[int], optional
        Multi-resolution shrink factors for the SimpleITK backend, ordered from
        coarse to fine.
    rot_smoothing_sigmas : sequence[float], optional
        Gaussian smoothing sigmas for each SimpleITK pyramid level, ordered from
        coarse to fine.
    rot_iterations : int, optional
        Maximum SimpleITK optimizer iterations.
    rot_learning_rate : float, optional
        SimpleITK optimizer learning rate.
    rot_min_step : float, optional
        SimpleITK optimizer minimum step length / convergence scale.
    rot_sampling_percentage : float or None, optional
        Optional fraction of voxels sampled by the SimpleITK metric. ``None``
        uses dense metric evaluation.
    rot_cval : float, optional
        Constant fill value used outside the transformed volume for full 3D
        rigid correction.
    rot_n_jobs : int, optional
        Worker count for independent time points in full 3D rigid registration.
        ``1`` inherits the global ``n_jobs`` value.
    rot_points_max_points : int, optional
        Maximum number of detected 3D peaks used by the sparse point backend.
    rot_points_min_distance : int, optional
        Minimum distance in pixels between detected peaks for the sparse point
        backend.
    rot_points_threshold_rel : float, optional
        Relative peak-detection threshold for the sparse point backend.
    rot_points_iterations : int, optional
        Number of ICP/Kabsch refinement iterations for the sparse point backend.
    rot_points_max_match_distance : float, optional
        Maximum nearest-neighbor point-match distance in pixels for the sparse
        point backend.
    transform_backend : {"skimage", "scipy"}, optional
        Backend used to apply translation corrections. ``"skimage"`` is the
        default for XY transforms and keeps translation correction aligned with
        the scikit-image rotation-correction path. ``"scipy"`` uses
        ``scipy.ndimage.shift`` and is useful for legacy comparison. True
        subpixel Z translations are internally applied with SciPy even when
        ``transform_backend="skimage"``, because the scikit-image path here is
        intentionally restricted to explicit 2D XY transforms.
    transform_order : int, optional
        Interpolation order for correction transforms. ``1`` is recommended for
        most intensity microscopy data because it gives smooth subpixel shifts.
        ``0`` uses nearest-neighbor interpolation and is useful for sparse
        puncta, label-like images, or cases where preserving peak sharpness is
        more important than smooth subpixel interpolation. Higher orders are
        available but can introduce more smoothing or ringing.
    intra_stack_reference_mode : {"neighbor", "full_projection", "first_slice"}, optional
        Reference strategy for ``intra_stack=True``.
    neighbor_window_size : int, optional
        Odd number of slices used for ``intra_stack_reference_mode="neighbor"``.
    filter_slices : bool, optional
        If True, apply slice-wise median filtering to the registration channel
        before projection. This affects only shift estimation.
    filter_projections : bool, optional
        If True, apply 2D median filtering to each projection after projection.
        This affects only shift estimation.
    median_kernel_size : int, optional
        Median filter kernel size used by the optional filters.
    phase_cross_correlation_upsample_factor : int, optional
        Subpixel upsampling factor for ``method="phase_cross_correlation"``.
    phase_cross_correlation_normalization : {None, "phase"}, optional
        Normalization mode forwarded to scikit-image's phase cross-correlation.
        ``None`` is more robust for the smooth synthetic examples.
    nc_pw_rigid : bool, optional
        If ``method="normcorre"``, enable piecewise-rigid NoRMCorre-style patch
        correction. If False, NoRMCorre runs in rigid/global translation mode.
    nc_strides : int, tuple[int, ...], or None, optional
        NoRMCorre patch-grid stride. For 2D+t data this is interpreted as
        ``(stride_y, stride_x)``; for 3D+t data as ``(stride_z, stride_y,
        stride_x)``. ``None`` uses backend defaults.
    nc_overlaps : int, tuple[int, ...], or None, optional
        NoRMCorre patch overlap. The effective patch size is
        ``nc_strides + nc_overlaps``. Larger overlaps smooth transitions between
        neighboring local shifts but increase runtime.
    nc_max_deviation_rigid : float, tuple[float, ...], or None, optional
        Maximum allowed local patch deviation from the global rigid shift in
        NoRMCorre. This is not the same as ``max_xy_shifts``:
        ``max_xy_shifts`` limits the overall/global correction, whereas
        ``nc_max_deviation_rigid`` limits how far each piecewise-rigid local
        patch may move relative to the global rigid estimate. ``None`` leaves
        local deviations unconstrained.
    nc_n_iterations : int, optional
        Number of NoRMCorre template-update passes. Each pass estimates motion,
        applies correction, and updates the template according to
        ``nc_template_update_method``.
    nc_correction_iterations : int, optional
        Number of times the already corrected NoRMCorre output is fed back into
        the correction loop. This is useful for difficult data but increases
        runtime.
    nc_niter_rig : int, optional
        Number of rigid pre-alignment iterations before piecewise NoRMCorre
        patch correction.
    nc_template_init_mode : {"registration_stack", "median"}, optional
        Initial NoRMCorre template strategy. ``"registration_stack"`` uses the
        selected reference time point. ``"median"`` uses a CaImAn-like sparse
        temporal sample and median template.
    nc_template_update_method : {"caiman", "mean", "median", "none"}, optional
        NoRMCorre template update strategy. ``"caiman"`` computes chunk means
        and takes a median across chunks, following the batch NoRMCorre idea.
    nc_splits : int, optional
        Number of temporal chunks used by the CaImAn-style NoRMCorre template
        update.
    nc_gSig_filt : int, tuple[int, ...], or None, optional
        CaImAn-style spatial high-pass filter scale used by NoRMCorre for
        estimation. ``None`` disables this filter.
    nc_add_to_movie : float or None, optional
        Constant offset added internally before NoRMCorre processing. ``None``
        lets the backend choose an offset when required by non-negativity
        handling.
    nc_nonneg_movie : bool, optional
        If True, NoRMCorre processing keeps the internal movie non-negative by
        adding an offset when needed.
    nc_shift_interpolation : {"resize", "linear"}, optional
        Interpolation strategy used to expand patch shifts into a dense
        NoRMCorre displacement field.
    nc_n_jobs : int, optional
        Worker count used by the NoRMCorre backend. ``1`` inherits the global
        ``n_jobs`` value.
    nc_transform_mode : str, optional
        Boundary mode used when applying NoRMCorre transforms, forwarded to the
        internal interpolation routines.
    nc_transform_cval : float, optional
        Constant fill value used by NoRMCorre transforms when
        ``nc_transform_mode="constant"``.
    nc_border_nan : bool, "copy", or None, optional
        NoRMCorre border handling. ``None`` uses the backend default; other
        values follow CaImAn/NoRMCorre-style border handling conventions.
    nc_block_size : int, optional
        Temporal block size used by NoRMCorre for chunked processing and
        template updates.
    nc_output_use_memmap : bool, optional
        If True, write NoRMCorre registered output to an OMIO disk-backed Zarr
        store instead of a full in-memory NumPy array. The global
        ``output_use_memmap`` option acts as an alias for this setting when
        ``method="normcorre"``.
    nc_output_memmap_folder : str or None, optional
        Folder forwarded to OMIO as ``zarr_store_path`` for NoRMCorre output
        caches.
    nc_output_memmap_name : str or None, optional
        Base Zarr store name for NoRMCorre registered outputs.
    n_jobs : int, optional
        Number of CPU worker threads for the standard registration paths. The
        template-based time registration, intra-stack slice registration,
        rotation estimation/application, and zero-clip mask updates are
        parallelized where time points or slices are independent. ``1`` keeps
        serial execution; ``-1`` uses all available CPUs. With ``method="normcorre"``
        this value is used as ``nc_n_jobs`` unless ``nc_n_jobs`` is set
        explicitly. With full 3D rigid backends it is used as ``rot_n_jobs``
        unless ``rot_n_jobs`` is set explicitly.
    output_use_memmap : bool, optional
        If True, standard and full-3D rigid registration outputs are written to
        an OMIO disk-backed Zarr store instead of a full in-memory NumPy output.
        With ``method="normcorre"``, this is treated as a shared alias for
        ``nc_output_use_memmap`` unless the NoRMCorre-specific setting is
        already enabled. Shift estimation still reads only the currently needed
        registration-channel volume; full 3D rigid registration necessarily
        works on one complete ZYX volume per time point.
    output_memmap_folder : str or None, optional
        Folder forwarded to OMIO as ``zarr_store_path`` for registered outputs.
        Use local scratch storage for large input files on network volumes.
    output_memmap_name : str or None, optional
        Base Zarr store name for standard registered outputs. Sequential
        correction stages append a small suffix such as ``time_pass_1`` or
        ``zero_clipped``.
    output_dtype : dtype, optional
        dtype used for registered outputs. The default ``np.float32`` is
        recommended for intensity microscopy because subpixel transforms create
        interpolated values and avoids integer clipping/rounding. Use integer
        dtypes only when you intentionally want quantized output.
    memory_tracker : zenreg.profiling.MemoryTracker or None, optional
        Optional diagnostic memory tracker. If provided, ``register_stack`` and
        major internal steps add markers to the tracker's RSS trace. The default
        ``None`` keeps profiling fully disabled and has negligible overhead.
    verbose : bool, optional
        If True, print the estimated shifts.
    return_shifts : bool, optional
        If True, return shifts. For the default projection XY time-registration
        path, shifts remain a backwards-compatible ``T, 2`` array storing
        ``(shift_y, shift_x)``. Advanced modes return a dictionary containing
        ``time_shifts_zyx`` and/or ``intra_stack_shifts_yx``.
    return_details : bool, optional
        If True together with ``return_shifts=True``, always return the full
        registration details dictionary, including settings used for reports,
        instead of the backwards-compatible simple shift arrays.
    pre_median_filter : bool or None, optional
        Deprecated alias for ``filter_slices``. Kept for compatibility with
        early ZenReg scripts.
    post_median_filter : bool or None, optional
        Deprecated alias for ``filter_projections``. Kept for compatibility
        with early ZenReg scripts.

    Returns
    -------
    numpy.ndarray or tuple[numpy.ndarray, numpy.ndarray | dict]
        Registered stack, optionally with the estimated shifts.
    """

    stack = ensure_tzcyx_stack(stack)
    _memory_mark(memory_tracker, "register_stack:start")
    output_dtype = _normalize_output_dtype(output_dtype)
    if not output_use_memmap and output_memmap_folder is not None:
        raise ValueError("output_memmap_folder requires output_use_memmap=True.")
    if not output_use_memmap and output_memmap_name is not None and output_memmap_name != "zenreg_registered":
        raise ValueError("Custom output_memmap_name requires output_use_memmap=True.")
    method = _normalize_registration_method(method)
    from .rigid3d import normalize_rigid_3d_backend, normalize_rigid_3d_metric

    rigid_3d_backend = normalize_rigid_3d_backend(rigid_3d_backend)
    rot_metric = normalize_rigid_3d_metric(rot_metric)
    time_registration_mode = _normalize_time_registration_mode(time_registration_mode)
    time_reference_mode = _normalize_time_reference_mode(time_reference_mode)
    registration_template_time_range = _normalize_registration_template_time_range(
        registration_template_time_range,
        stack.shape[0],
    )
    zrange = _resolve_registration_z_range_alias(
        registration_z_range=registration_z_range,
        zrange=zrange,
        projection_range=projection_range,
    )
    projection_method = _normalize_projection_method(projection_method)
    registration_stack = _normalize_registration_stack(registration_stack, stack.shape[0])
    intra_stack_reference_mode = _normalize_intra_stack_reference_mode(intra_stack_reference_mode)
    neighbor_window_size = _normalize_neighbor_window_size(neighbor_window_size)
    zero_clip_mode = _normalize_zero_clip_mode(zero_clip_mode)
    zero_clip_mask_threshold = _normalize_zero_clip_mask_threshold(zero_clip_mask_threshold)
    zero_clip_mask_strategy = _normalize_zero_clip_mask_strategy(zero_clip_mask_strategy)
    zero_clip_mask_min_fraction = _normalize_zero_clip_mask_min_fraction(zero_clip_mask_min_fraction)
    zero_clip_margin_zyx = _normalize_zero_clip_margin(zero_clip_margin)
    max_xy_shifts = _normalize_max_xy_shifts(max_xy_shifts)
    max_z_shifts = _normalize_max_z_shifts(max_z_shifts)
    max_rot_shifts = _normalize_max_rot_shifts(max_rot_shifts)
    rotreg_iter = _normalize_rotreg_iter(rotreg_iter)
    effective_rot_spacing_zyx, rot_spacing_source = _resolve_rot_spacing_zyx(
        rot_spacing_zyx,
        metadata,
    )
    rot_init_iterations = int(rot_init_iterations)
    if rot_init_iterations < 0:
        raise ValueError(f"rot_init_iterations must be >= 0. Got {rot_init_iterations!r}.")
    rot_iterations = int(rot_iterations)
    if rot_iterations < 1:
        raise ValueError(f"rot_iterations must be >= 1. Got {rot_iterations!r}.")
    rot_n_jobs = max(int(rot_n_jobs), 1)
    n_jobs = _normalize_n_jobs(n_jobs)
    effective_rot_n_jobs = int(rot_n_jobs if int(rot_n_jobs) != 1 else n_jobs)
    effective_nc_n_jobs = int(nc_n_jobs if int(nc_n_jobs) != 1 else n_jobs)
    effective_nc_output_use_memmap = bool(nc_output_use_memmap or (output_use_memmap and method == "normcorre"))
    effective_nc_output_memmap_folder = nc_output_memmap_folder
    if effective_nc_output_use_memmap and effective_nc_output_memmap_folder is None:
        effective_nc_output_memmap_folder = output_memmap_folder
    effective_nc_output_memmap_name = nc_output_memmap_name
    if output_use_memmap and method == "normcorre" and nc_output_memmap_name == "zenreg_normcorre_registered":
        effective_nc_output_memmap_name = output_memmap_name
    transform_backend = _normalize_transform_backend(transform_backend)
    transform_order = _normalize_transform_order(transform_order)
    filter_slices, filter_projections = _resolve_filter_aliases(
        filter_slices=filter_slices,
        filter_projections=filter_projections,
        pre_median_filter=pre_median_filter,
        post_median_filter=post_median_filter,
    )
    phase_cross_correlation_normalization = _normalize_phase_cross_correlation_normalization(
        phase_cross_correlation_normalization
    )
    if stack.shape[0] <= 1 and time_registration_mode != "none":
        raise ValueError("Registration requires T > 1.")
    (
        registration_channel_requested,
        registration_channel,
        registration_channel_fallback,
        registration_channel_fallback_reason,
    ) = _resolve_registration_channel(registration_channel, stack.shape[2])
    if int(median_kernel_size) < 1:
        raise ValueError(f"median_kernel_size must be >= 1. Got {median_kernel_size!r}.")
    if int(phase_cross_correlation_upsample_factor) < 1:
        raise ValueError(
            "phase_cross_correlation_upsample_factor must be >= 1. "
            f"Got {phase_cross_correlation_upsample_factor!r}."
        )
    if rotreg and time_registration_mode == "none":
        warnings.warn(
            "rotreg=True requires time registration. Ignoring rotreg because "
            "time_registration_mode='none'.",
            RuntimeWarning,
            stacklevel=2,
        )
        rotreg = False
    if registration_template_time_range is not None and time_reference_mode != "template":
        raise ValueError(
            "registration_template_time_range requires time_reference_mode='template'. "
            "Frame-to-frame registration with time_reference_mode='previous' uses the "
            "previous time point as reference instead."
        )
    if registration_template_time_range is not None and method == "normcorre":
        raise ValueError(
            "registration_template_time_range is used by the standard registration backends. "
            "For method='normcorre', use nc_template_init_mode and "
            "nc_template_update_method instead."
        )
    if (
        registration_template_time_range is not None
        and rotreg
        and rigid_3d_backend in {"simpleitk", "points"}
    ):
        raise ValueError(
            "registration_template_time_range is not supported with full 3D rigid "
            "registration backends. Use registration_stack as the explicit rigid "
            "reference volume."
        )
    effective_zero_clip_mode = _effective_zero_clip_mode(
        zero_clip=bool(zero_clip),
        zero_clip_mode=zero_clip_mode,
        rotreg=bool(rotreg),
    )
    effective_zero_clip_mask_strategy = _effective_zero_clip_mask_strategy(
        zero_clip_mask_strategy=zero_clip_mask_strategy,
        rigid_3d=bool(rotreg and rigid_3d_backend in {"simpleitk", "points"}),
    )
    registration_z_range_setting = (
        None
        if zrange is None
        else tuple(int(v) for v in normalize_zrange(zrange, stack.shape[1], strict=True))
    )
    registration_settings = {
        "registration_channel": int(registration_channel),
        "registration_channel_requested": int(registration_channel_requested),
        "registration_channel_used": int(registration_channel),
        "registration_channel_fallback": bool(registration_channel_fallback),
        "registration_channel_fallback_reason": registration_channel_fallback_reason,
        "registration_stack": int(registration_stack),
        "registration_template_time_range": registration_template_time_range,
        "registration_z_range": registration_z_range_setting,
        "method": method,
        "intra_stack": bool(intra_stack),
        "zreg": bool(zreg),
        "zero_clip": bool(zero_clip),
        "zero_clip_mask_strategy": effective_zero_clip_mask_strategy,
        "zero_clip_mask_min_fraction": float(zero_clip_mask_min_fraction),
        "n_jobs": int(n_jobs),
        "output_use_memmap": bool(output_use_memmap),
        "output_memmap_folder": output_memmap_folder,
        "output_memmap_name": output_memmap_name if output_use_memmap else None,
        "output_dtype": str(output_dtype),
        "rotreg": bool(rotreg),
        "rotreg_iter": int(rotreg_iter),
        "rigid_3d_backend": rigid_3d_backend,
        "rot_spacing_zyx": tuple(float(v) for v in effective_rot_spacing_zyx),
        "rot_spacing_source": rot_spacing_source,
        "rot_init_iterations": int(rot_init_iterations),
        "rot_metric": rot_metric,
        "rot_shrink_factors": tuple(int(v) for v in rot_shrink_factors),
        "rot_smoothing_sigmas": tuple(float(v) for v in rot_smoothing_sigmas),
        "rot_iterations": int(rot_iterations),
        "rot_learning_rate": float(rot_learning_rate),
        "rot_min_step": float(rot_min_step),
        "rot_sampling_percentage": rot_sampling_percentage,
        "rot_cval": float(rot_cval),
        "rot_n_jobs": int(effective_rot_n_jobs),
        "rot_points_max_points": int(rot_points_max_points),
        "rot_points_min_distance": int(rot_points_min_distance),
        "rot_points_threshold_rel": float(rot_points_threshold_rel),
        "rot_points_iterations": int(rot_points_iterations),
        "rot_points_max_match_distance": float(rot_points_max_match_distance),
        "registration_z_range": registration_z_range_setting,
        "projection_range": registration_z_range_setting,
        "registration_template_time_range": registration_template_time_range,
        "projection_method": projection_method,
        "filter_slices": bool(filter_slices),
        "filter_projections": bool(filter_projections),
        "median_kernel_size": int(median_kernel_size),
        "max_xy_shifts": None
        if max_xy_shifts is None
        else tuple(float(v) for v in max_xy_shifts),
        "max_z_shifts": None if max_z_shifts is None else float(max_z_shifts),
        "max_rot_shifts": None if max_rot_shifts is None else float(max_rot_shifts),
        "phase_cross_correlation_upsample_factor": int(phase_cross_correlation_upsample_factor),
        "phase_cross_correlation_normalization": phase_cross_correlation_normalization,
        "stack_shape_tzcyx": tuple(int(v) for v in stack.shape),
    }
    if rotreg and rigid_3d_backend in {"simpleitk", "points"}:
        if stack.shape[1] < 2:
            raise ValueError("Full 3D rigid rotation registration requires SizeZ >= 2.")
        if time_reference_mode != "template":
            raise ValueError("Full 3D rigid rotation registration currently supports only time_reference_mode='template'.")
        if intra_stack:
            raise ValueError("Full 3D rigid rotation registration does not support intra_stack=True yet.")
        result = _register_stack_rigid_3d_from_main_wrapper(
            stack,
            registration_channel=int(registration_channel),
            registration_stack=int(registration_stack),
            method=method,
            projection_range=registration_z_range_setting,
            projection_method=projection_method,
            rigid_3d_backend=rigid_3d_backend,
            zero_clip=bool(zero_clip),
            zero_clip_mode=zero_clip_mode,
            zero_clip_mask_threshold=zero_clip_mask_threshold,
            zero_clip_mask_strategy=zero_clip_mask_strategy,
            zero_clip_mask_min_fraction=zero_clip_mask_min_fraction,
            zero_clip_margin_zyx=zero_clip_margin_zyx,
            max_rot_shifts=max_rot_shifts,
            phase_cross_correlation_upsample_factor=int(phase_cross_correlation_upsample_factor),
            phase_cross_correlation_normalization=phase_cross_correlation_normalization,
            transform_order=transform_order,
            rot_spacing_zyx=effective_rot_spacing_zyx,
            rot_init_iterations=rot_init_iterations,
            rot_metric=rot_metric,
            rot_shrink_factors=rot_shrink_factors,
            rot_smoothing_sigmas=rot_smoothing_sigmas,
            rot_iterations=rot_iterations,
            rot_learning_rate=float(rot_learning_rate),
            rot_min_step=float(rot_min_step),
            rot_sampling_percentage=rot_sampling_percentage,
            rot_cval=float(rot_cval),
            rot_n_jobs=effective_rot_n_jobs,
            rot_points_max_points=int(rot_points_max_points),
            rot_points_min_distance=int(rot_points_min_distance),
            rot_points_threshold_rel=float(rot_points_threshold_rel),
            rot_points_iterations=int(rot_points_iterations),
            rot_points_max_match_distance=float(rot_points_max_match_distance),
            output_use_memmap=bool(output_use_memmap),
            output_memmap_folder=output_memmap_folder,
            output_memmap_name=output_memmap_name,
            output_dtype=output_dtype,
            memory_tracker=memory_tracker,
            verbose=verbose,
            return_shifts=return_shifts,
            return_details=return_details,
            registration_settings=registration_settings,
        )
        _memory_mark(memory_tracker, "register_stack:end")
        return result
    if method == "normcorre":
        result = _register_stack_normcorre_from_main_wrapper(
            stack,
            registration_channel=int(registration_channel),
            registration_stack=int(registration_stack),
            time_registration_mode=time_registration_mode,
            time_reference_mode=time_reference_mode,
            intra_stack=bool(intra_stack),
            projection_range=registration_z_range_setting,
            projection_method=projection_method,
            zreg=bool(zreg),
            zero_clip=bool(zero_clip),
            rotreg=bool(rotreg),
            filter_slices=bool(filter_slices),
            filter_projections=bool(filter_projections),
            max_xy_shifts=max_xy_shifts,
            max_z_shifts=max_z_shifts,
            phase_cross_correlation_upsample_factor=int(phase_cross_correlation_upsample_factor),
            phase_cross_correlation_normalization=phase_cross_correlation_normalization,
            transform_order=transform_order,
            nc_pw_rigid=bool(nc_pw_rigid),
            nc_strides=nc_strides,
            nc_overlaps=nc_overlaps,
            nc_max_deviation_rigid=nc_max_deviation_rigid,
            nc_n_iterations=int(nc_n_iterations),
            nc_correction_iterations=int(nc_correction_iterations),
            nc_niter_rig=int(nc_niter_rig),
            nc_template_init_mode=nc_template_init_mode,
            nc_template_update_method=nc_template_update_method,
            nc_splits=int(nc_splits),
            nc_gSig_filt=nc_gSig_filt,
            nc_add_to_movie=nc_add_to_movie,
            nc_nonneg_movie=bool(nc_nonneg_movie),
            nc_shift_interpolation=nc_shift_interpolation,
            nc_n_jobs=int(effective_nc_n_jobs),
            nc_transform_mode=nc_transform_mode,
            nc_transform_cval=float(nc_transform_cval),
            nc_border_nan=nc_border_nan,
            nc_block_size=int(nc_block_size),
            nc_output_use_memmap=bool(effective_nc_output_use_memmap),
            nc_output_memmap_folder=effective_nc_output_memmap_folder,
            nc_output_memmap_name=effective_nc_output_memmap_name,
            memory_tracker=memory_tracker,
            verbose=verbose,
            return_shifts=return_shifts,
            return_details=return_details,
            registration_settings=registration_settings,
        )
        _memory_mark(memory_tracker, "register_stack:end")
        return result

    registered = stack
    intra_stack_shifts_yx = None
    zero_clip_stage_bounds = []
    zero_clip_mask_tzyx = None
    if effective_zero_clip_mode == "mask":
        _memory_mark(memory_tracker, "zero_clip_mask:allocate:start")
        zero_clip_mask_tzyx = np.ones(
            (stack.shape[0], stack.shape[1], stack.shape[3], stack.shape[4]),
            dtype=np.float32,
        )
        _memory_mark(memory_tracker, "zero_clip_mask:allocate:end")
    if intra_stack:
        registered, intra_stack_shifts_yx = _correct_intra_stack_z_drift_impl(
            registered,
            registration_channel=int(registration_channel),
            method=method,
            reference_mode=intra_stack_reference_mode,
            neighbor_window_size=neighbor_window_size,
            projection_method=projection_method,
            filter_slices=filter_slices,
            filter_projections=filter_projections,
            median_kernel_size=int(median_kernel_size),
            phase_cross_correlation_upsample_factor=int(phase_cross_correlation_upsample_factor),
            phase_cross_correlation_normalization=phase_cross_correlation_normalization,
            transform_backend=transform_backend,
            transform_order=transform_order,
            n_jobs=n_jobs,
            output_use_memmap=bool(output_use_memmap),
            output_memmap_folder=output_memmap_folder,
            output_memmap_name=output_memmap_name,
            output_dtype=output_dtype,
            output_stage_name="intra_stack",
            memory_tracker=memory_tracker,
            verbose=verbose,
            return_shifts=True,
        )
        if max_xy_shifts is not None:
            clipped_intra_stack_shifts_yx = np.asarray(
                [
                    [_clip_shift_yx(shift, max_xy_shifts) for shift in shifts_t]
                    for shifts_t in intra_stack_shifts_yx
                ],
                dtype=np.float32,
            )
            if not np.allclose(clipped_intra_stack_shifts_yx, intra_stack_shifts_yx):
                registered = _create_registered_output(
                    tuple(int(v) for v in stack.shape),
                    dtype=output_dtype,
                    output_use_memmap=bool(output_use_memmap),
                    output_memmap_folder=output_memmap_folder,
                    output_memmap_name=output_memmap_name,
                    stage_name="intra_stack_clipped",
                )

                def apply_clipped_intra_slice(index: tuple[int, int]) -> tuple[int, int, np.ndarray]:
                    t, z = index
                    return t, z, _apply_translation_to_cyx(
                        stack[t, z, :, :, :],
                        clipped_intra_stack_shifts_yx[t, z, :],
                        transform_backend=transform_backend,
                        transform_order=transform_order,
                    )

                tasks = [(t, z) for t in range(stack.shape[0]) for z in range(stack.shape[1])]
                for t, z, corrected_slice in _iter_map_ordered(
                    apply_clipped_intra_slice,
                    tasks,
                    n_jobs=n_jobs,
                    progress=verbose,
                    desc="ZenReg intra-stack clipped apply",
                ):
                    registered[t, z, :, :, :] = corrected_slice
                intra_stack_shifts_yx = clipped_intra_stack_shifts_yx
        if effective_zero_clip_mode == "mask":
            def apply_intra_mask(index: tuple[int, int]) -> tuple[int, int, np.ndarray]:
                t, z = index
                return t, z, _apply_translation_to_mask_yx(
                    zero_clip_mask_tzyx[t, z, :, :],
                    intra_stack_shifts_yx[t, z, :],
                    transform_backend=transform_backend,
                )

            tasks = [(t, z) for t in range(stack.shape[0]) for z in range(stack.shape[1])]
            for t, z, mask_plane in _parallel_map_ordered(
                apply_intra_mask,
                tasks,
                n_jobs=n_jobs,
                progress=verbose,
                desc="ZenReg zero-clip intra mask",
            ):
                zero_clip_mask_tzyx[t, z, :, :] = mask_plane
        elif effective_zero_clip_mode == "shift":
            zero_clip_stage_bounds.append(_crop_bounds_from_yx_shifts(intra_stack_shifts_yx))

    time_shifts_zyx = None
    time_shifts_zyx_raw = None
    rotation_shifts_deg = None
    rotation_shifts_deg_raw = None
    translation_pass_shifts_zyx = []
    translation_pass_shifts_zyx_raw = []
    rotation_pass_shifts_deg = []
    rotation_pass_shifts_deg_raw = []
    effective_time_registration_mode = time_registration_mode
    if time_registration_mode != "none":
        translation_pass_count = rotreg_iter + 1 if rotreg else 1
        for pass_index in range(translation_pass_count):
            registered, pass_shifts_zyx, pass_shifts_zyx_raw, effective_time_registration_mode = _register_stack_across_time(
                registered,
                registration_channel=int(registration_channel),
                registration_stack=registration_stack,
                method=method,
                time_registration_mode=time_registration_mode,
                time_reference_mode=time_reference_mode,
                registration_template_time_range=registration_template_time_range,
                zrange=zrange,
                projection_method=projection_method,
                filter_slices=filter_slices,
                filter_projections=filter_projections,
                median_kernel_size=int(median_kernel_size),
                zreg=bool(zreg),
                max_xy_shifts=max_xy_shifts,
                max_z_shifts=max_z_shifts,
                phase_cross_correlation_upsample_factor=int(phase_cross_correlation_upsample_factor),
                phase_cross_correlation_normalization=phase_cross_correlation_normalization,
                transform_backend=transform_backend,
                transform_order=transform_order,
                n_jobs=n_jobs,
                output_use_memmap=bool(output_use_memmap),
                output_memmap_folder=output_memmap_folder,
                output_memmap_name=output_memmap_name,
                output_dtype=output_dtype,
                output_stage_name=f"time_pass_{pass_index + 1}",
                memory_tracker=memory_tracker,
                verbose=verbose,
            )
            translation_pass_shifts_zyx.append(pass_shifts_zyx)
            translation_pass_shifts_zyx_raw.append(pass_shifts_zyx_raw)
            if effective_zero_clip_mode == "mask":
                def apply_time_mask(t: int) -> tuple[int, np.ndarray]:
                    return t, _apply_translation_to_mask_zyx(
                        zero_clip_mask_tzyx[t, :, :, :],
                        pass_shifts_zyx[t, :],
                        transform_backend=transform_backend,
                    )

                for t, mask_volume in _parallel_map_ordered(
                    apply_time_mask,
                    range(stack.shape[0]),
                    n_jobs=n_jobs,
                    progress=verbose,
                    desc="ZenReg zero-clip time mask",
                ):
                    zero_clip_mask_tzyx[t, :, :, :] = mask_volume
            elif effective_zero_clip_mode == "shift":
                zero_clip_stage_bounds.append(_crop_bounds_from_zyx_shifts(pass_shifts_zyx))

            if rotreg and pass_index < rotreg_iter:
                registered, pass_rotation_shifts_deg, pass_rotation_shifts_deg_raw = _register_stack_rotations_across_time(
                    registered,
                    registration_channel=int(registration_channel),
                    registration_stack=registration_stack,
                    time_reference_mode=time_reference_mode,
                    registration_template_time_range=registration_template_time_range,
                    zrange=zrange,
                    projection_method=projection_method,
                    filter_slices=filter_slices,
                    filter_projections=filter_projections,
                    median_kernel_size=int(median_kernel_size),
                    max_rot_shifts=max_rot_shifts,
                    phase_cross_correlation_upsample_factor=int(phase_cross_correlation_upsample_factor),
                    phase_cross_correlation_normalization=phase_cross_correlation_normalization,
                    transform_order=transform_order,
                    n_jobs=n_jobs,
                    output_use_memmap=bool(output_use_memmap),
                    output_memmap_folder=output_memmap_folder,
                    output_memmap_name=output_memmap_name,
                    output_dtype=output_dtype,
                    output_stage_name=f"rotation_pass_{pass_index + 1}",
                    memory_tracker=memory_tracker,
                    verbose=verbose,
                )
                rotation_pass_shifts_deg.append(pass_rotation_shifts_deg)
                rotation_pass_shifts_deg_raw.append(pass_rotation_shifts_deg_raw)
                if effective_zero_clip_mode == "mask":
                    def apply_rotation_mask(t: int) -> tuple[int, np.ndarray]:
                        return t, _apply_rotation_to_mask_zyx(
                            zero_clip_mask_tzyx[t, :, :, :],
                            float(pass_rotation_shifts_deg[t]),
                        )

                    for t, mask_volume in _parallel_map_ordered(
                        apply_rotation_mask,
                        range(stack.shape[0]),
                        n_jobs=n_jobs,
                        progress=verbose,
                        desc="ZenReg zero-clip rotation mask",
                    ):
                        zero_clip_mask_tzyx[t, :, :, :] = mask_volume

        time_shifts_zyx = np.sum(np.stack(translation_pass_shifts_zyx, axis=0), axis=0).astype(np.float32)
        time_shifts_zyx_raw = np.sum(np.stack(translation_pass_shifts_zyx_raw, axis=0), axis=0).astype(np.float32)
        if rotation_pass_shifts_deg:
            rotation_shifts_deg = np.sum(np.stack(rotation_pass_shifts_deg, axis=0), axis=0).astype(np.float32)
            rotation_shifts_deg_raw = np.sum(np.stack(rotation_pass_shifts_deg_raw, axis=0), axis=0).astype(np.float32)

    zero_clip_bounds = None
    zero_clip_failed_reason = None
    if zero_clip:
        try:
            _memory_mark(memory_tracker, "zero_clip:compute_bounds:start")
            if effective_zero_clip_mode == "mask":
                zero_clip_bounds = _crop_bounds_from_valid_mask(
                    zero_clip_mask_tzyx,
                    threshold=zero_clip_mask_threshold,
                    strategy=effective_zero_clip_mask_strategy,
                    min_fraction=zero_clip_mask_min_fraction,
                )
            else:
                zero_clip_bounds = _add_crop_bounds(*zero_clip_stage_bounds)
            zero_clip_bounds = _apply_zero_clip_margin(zero_clip_bounds, zero_clip_margin_zyx)
            _memory_mark(memory_tracker, "zero_clip:compute_bounds:end")
            _memory_mark(memory_tracker, "zero_clip:crop:start")
            registered = _zero_clip_stack(
                registered,
                zero_clip_bounds,
                output_use_memmap=bool(output_use_memmap),
                output_memmap_folder=output_memmap_folder,
                output_memmap_name=output_memmap_name,
                output_dtype=output_dtype,
                n_jobs=n_jobs,
                progress=verbose,
            )
            _memory_mark(memory_tracker, "zero_clip:crop:end")
        except ValueError as exc:
            zero_clip_failed_reason = str(exc)
            zero_clip_bounds = None
            warnings.warn(
                "zero_clip=True was requested, but no usable common valid "
                f"region could be found. Returning the registered stack without cropping. {exc}",
                RuntimeWarning,
                stacklevel=2,
            )

    compatible_time_shifts_yx = (
        time_registration_mode == "projection"
        and effective_time_registration_mode == "projection"
        and time_reference_mode == "template"
        and not zreg
        and not intra_stack
        and not rotreg
        and not zero_clip
    )
    pearson_correlations_before = None
    if return_shifts or return_details:
        _memory_mark(memory_tracker, "correlation_before:start")
        pearson_correlations_before = _compute_registration_frame_correlations(
            stack,
            registration_channel=int(registration_channel),
            registration_stack=int(registration_stack),
            registration_template_time_range=registration_template_time_range,
            projection_range=zrange,
            projection_method=projection_method,
            effective_time_registration_mode=effective_time_registration_mode,
        )
        _memory_mark(memory_tracker, "correlation_before:end")
    result = _return_registration_result(
        registered,
        return_shifts=return_shifts,
        return_details=return_details,
        compatible_time_shifts_yx=compatible_time_shifts_yx,
        time_shifts_zyx=time_shifts_zyx,
        time_shifts_zyx_raw=time_shifts_zyx_raw,
        intra_stack_shifts_yx=intra_stack_shifts_yx,
        rotation_shifts_deg=rotation_shifts_deg,
        rotation_shifts_deg_raw=rotation_shifts_deg_raw,
        translation_pass_shifts_zyx=translation_pass_shifts_zyx,
        translation_pass_shifts_zyx_raw=translation_pass_shifts_zyx_raw,
        rotation_pass_shifts_deg=rotation_pass_shifts_deg,
        rotation_pass_shifts_deg_raw=rotation_pass_shifts_deg_raw,
        zero_clip_bounds=zero_clip_bounds,
        zero_clip_failed_reason=zero_clip_failed_reason,
        zero_clip_mode=effective_zero_clip_mode,
        zero_clip_mask_threshold=zero_clip_mask_threshold,
        zero_clip_mask_strategy=effective_zero_clip_mask_strategy,
        zero_clip_mask_min_fraction=zero_clip_mask_min_fraction,
        zero_clip_margin_zyx=zero_clip_margin_zyx,
        time_registration_mode=time_registration_mode,
        effective_time_registration_mode=effective_time_registration_mode,
        time_reference_mode=time_reference_mode,
        transform_backend=transform_backend,
        transform_order=transform_order,
        pearson_correlations_before=pearson_correlations_before,
        registration_settings=registration_settings,
    )
    _memory_mark(memory_tracker, "register_stack:end")
    return result
# %% END
