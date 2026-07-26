"""
Registration helpers for canonical ``TZCYX`` microscopy stacks.

Author: Fabrizio Musacchio
Date: June 2026
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.ndimage import median_filter, shift as ndi_shift
from skimage.registration import phase_cross_correlation

from ._axes import CANONICAL_AXIS_ORDER, ensure_tzcyx_stack, normalize_zrange

SUPPORTED_REGISTRATION_METHODS = {"phase_cross_correlation", "pystackreg"}
SUPPORTED_INTRA_STACK_REFERENCE_MODES = {"neighbor", "full_projection"}


def _normalize_registration_method(method: str) -> str:
    """Normalize and validate the requested registration backend."""

    normalized = str(method).strip().lower()
    if normalized not in SUPPORTED_REGISTRATION_METHODS:
        raise ValueError(
            f"Unsupported registration method {method!r}. "
            f"Supported methods: {sorted(SUPPORTED_REGISTRATION_METHODS)}."
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
) -> np.ndarray:
    """Build the per-slice registration reference used for Z-drift correction."""

    if reference_mode == "full_projection":
        return np.max(volume_zyx, axis=0)

    half_window = neighbor_window_size // 2
    start = max(0, z_index - half_window)
    stop = min(volume_zyx.shape[0], z_index + half_window + 1)
    return np.max(volume_zyx[start:stop, :, :], axis=0)


def _build_registration_projections(
    stack: np.ndarray,
    *,
    registration_channel: int,
    zrange: tuple[int, int] | Sequence[int] | None,
    pre_median_filter: bool,
    post_median_filter: bool,
    median_kernel_size: int,
) -> np.ndarray:
    """Create per-time-point 2D registration projections from a ``TZCYX`` stack."""

    z_start, z_stop = normalize_zrange(zrange, stack.shape[1], strict=True)
    channel_stack = np.asarray(stack[:, z_start:z_stop, registration_channel, :, :], dtype=np.float32)
    working = channel_stack.copy()

    if pre_median_filter:
        for t in range(working.shape[0]):
            working[t, :, :, :] = _apply_median_to_zyx(working[t, :, :, :], median_kernel_size)

    projections = np.max(working, axis=1)

    if post_median_filter:
        for t in range(projections.shape[0]):
            projections[t, :, :] = median_filter(
                projections[t, :, :], size=(median_kernel_size, median_kernel_size)
            )
    return projections


def _phase_cross_correlation_shift(reference_projection: np.ndarray, moving_projection: np.ndarray) -> np.ndarray:
    """Estimate a 2D translation with phase cross-correlation."""

    shift_2d, _, _ = phase_cross_correlation(reference_projection, moving_projection)
    return np.asarray(shift_2d, dtype=np.float32)


def _pystackreg_shift(reference_projection: np.ndarray, moving_projection: np.ndarray) -> np.ndarray:
    """Estimate a 2D translation with :mod:`pystackreg` in translation mode."""

    from pystackreg import StackReg

    sr = StackReg(StackReg.TRANSLATION)
    tmat = sr.register(reference_projection.astype(np.float32), moving_projection.astype(np.float32))
    return np.asarray([-tmat[1, 2], -tmat[0, 2]], dtype=np.float32)


def _estimate_shift(reference_projection: np.ndarray, moving_projection: np.ndarray, *, method: str) -> np.ndarray:
    """Estimate a 2D registration shift using the selected backend."""

    if method == "phase_cross_correlation":
        return _phase_cross_correlation_shift(reference_projection, moving_projection)
    return _pystackreg_shift(reference_projection, moving_projection)


def _apply_translation_to_tzyx(stack_tzyx: np.ndarray, shift_yx: np.ndarray) -> np.ndarray:
    """Apply one XY translation to all channels and Z slices of one time point."""

    shifted = np.empty_like(stack_tzyx, dtype=np.float32)
    for z in range(stack_tzyx.shape[0]):
        for c in range(stack_tzyx.shape[1]):
            shifted[z, c, :, :] = ndi_shift(
                np.asarray(stack_tzyx[z, c, :, :], dtype=np.float32),
                shift=tuple(float(v) for v in shift_yx),
                order=1,
                mode="constant",
                cval=0.0,
                prefilter=True,
            )
    return shifted


def _apply_translation_to_cyx(slice_cyx: np.ndarray, shift_yx: np.ndarray) -> np.ndarray:
    """Apply one XY translation to all channels of a single Z slice."""

    shifted = np.empty_like(slice_cyx, dtype=np.float32)
    for c in range(slice_cyx.shape[0]):
        shifted[c, :, :] = ndi_shift(
            np.asarray(slice_cyx[c, :, :], dtype=np.float32),
            shift=tuple(float(v) for v in shift_yx),
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=True,
        )
    return shifted


def _print_verbose(verbose: bool, message: str) -> None:
    """Print a progress message only when verbose mode is enabled."""

    if verbose:
        print(message)


def correct_intra_stack_z_drift(
    stack,
    *,
    registration_channel: int = 0,
    method: str = "phase_cross_correlation",
    reference_mode: str = "neighbor",
    neighbor_window_size: int = 3,
    pre_median_filter: bool = False,
    post_median_filter: bool = False,
    median_kernel_size: int = 3,
    verbose: bool = True,
    return_shifts: bool = False,
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
    method : {"phase_cross_correlation", "pystackreg"}, optional
        Backend used for shift estimation.
    reference_mode : {"neighbor", "full_projection"}, optional
        Strategy for constructing each per-slice reference image.
    neighbor_window_size : int, optional
        Odd number of slices used for ``reference_mode="neighbor"``.
    pre_median_filter : bool, optional
        If True, apply a slice-wise median filter before reference construction.
        This affects only shift estimation.
    post_median_filter : bool, optional
        If True, apply a 2D median filter to moving and reference images just
        before shift estimation.
    median_kernel_size : int, optional
        Median filter kernel size used by the optional filters.
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

    stack = ensure_tzcyx_stack(stack).astype(np.float32, copy=True)
    method = _normalize_registration_method(method)
    reference_mode = _normalize_intra_stack_reference_mode(reference_mode)
    neighbor_window_size = _normalize_neighbor_window_size(neighbor_window_size)

    if not 0 <= int(registration_channel) < stack.shape[2]:
        raise ValueError(
            f"registration_channel must be between 0 and {stack.shape[2] - 1}. "
            f"Got {registration_channel!r}."
        )
    if int(median_kernel_size) < 1:
        raise ValueError(f"median_kernel_size must be >= 1. Got {median_kernel_size!r}.")

    shifts = np.zeros((stack.shape[0], stack.shape[1], 2), dtype=np.float32)
    if stack.shape[1] <= 1:
        _print_verbose(verbose, "Skipping intra-stack Z drift correction because Z <= 1.")
        return (stack.copy(), shifts) if return_shifts else stack.copy()

    corrected = stack.copy()
    for t in range(stack.shape[0]):
        volume_zyx = np.asarray(stack[t, :, int(registration_channel), :, :], dtype=np.float32)
        working_volume = volume_zyx.copy()

        if pre_median_filter:
            working_volume = _apply_median_to_zyx(working_volume, int(median_kernel_size))

        for z in range(stack.shape[1]):
            moving_image = np.asarray(working_volume[z, :, :], dtype=np.float32)
            reference_image = _build_intra_stack_reference_image(
                working_volume,
                z_index=z,
                reference_mode=reference_mode,
                neighbor_window_size=neighbor_window_size,
            ).astype(np.float32, copy=False)

            if post_median_filter:
                kernel = (int(median_kernel_size), int(median_kernel_size))
                moving_image = median_filter(moving_image, size=kernel)
                reference_image = median_filter(reference_image, size=kernel)

            shift_yx = _estimate_shift(reference_image, moving_image, method=method)
            shifts[t, z, :] = shift_yx
            _print_verbose(
                verbose,
                f"t={t} z={z} shift_y={float(shift_yx[0]):.3f} shift_x={float(shift_yx[1]):.3f}",
            )
            corrected[t, z, :, :, :] = _apply_translation_to_cyx(stack[t, z, :, :, :], shift_yx)

    return (corrected, shifts) if return_shifts else corrected


def register_stack(
    stack,
    *,
    registration_channel: int,
    method: str = "phase_cross_correlation",
    zrange: tuple[int, int] | Sequence[int] | None = None,
    pre_median_filter: bool = False,
    post_median_filter: bool = False,
    median_kernel_size: int = 3,
    verbose: bool = True,
    return_shifts: bool = False,
):
    """
    Register a ``TZCYX`` stack across time using shifts estimated from Z projections.

    Parameters
    ----------
    stack : array-like
        Input stack in canonical ``TZCYX`` order.
    registration_channel : int
        Channel used to compute the time-wise registration shifts.
    method : {"phase_cross_correlation", "pystackreg"}, optional
        Backend used for shift estimation.
    zrange : tuple[int, int] or None, optional
        Optional half-open Z range ``(start, stop)`` used for the max-Z
        registration projection.
    pre_median_filter : bool, optional
        If True, apply slice-wise median filtering to the registration channel
        before max-Z projection. This affects only shift estimation.
    post_median_filter : bool, optional
        If True, apply 2D median filtering to each projection after projection.
        This affects only shift estimation.
    median_kernel_size : int, optional
        Median filter kernel size used by the optional filters.
    verbose : bool, optional
        If True, print the estimated shifts.
    return_shifts : bool, optional
        If True, return ``(registered_stack, shifts_tyxc)`` where shifts has shape
        ``T, 2`` and stores ``(shift_y, shift_x)``.

    Returns
    -------
    numpy.ndarray or tuple[numpy.ndarray, numpy.ndarray]
        Registered stack, optionally with the estimated shifts.
    """

    stack = ensure_tzcyx_stack(stack).astype(np.float32, copy=True)
    method = _normalize_registration_method(method)

    if stack.shape[0] <= 1:
        raise ValueError("Registration requires T > 1.")
    if not 0 <= int(registration_channel) < stack.shape[2]:
        raise ValueError(
            f"registration_channel must be between 0 and {stack.shape[2] - 1}. "
            f"Got {registration_channel!r}."
        )
    if int(median_kernel_size) < 1:
        raise ValueError(f"median_kernel_size must be >= 1. Got {median_kernel_size!r}.")

    projections = _build_registration_projections(
        stack,
        registration_channel=int(registration_channel),
        zrange=zrange,
        pre_median_filter=pre_median_filter,
        post_median_filter=post_median_filter,
        median_kernel_size=int(median_kernel_size),
    )
    reference_projection = projections[0, :, :]
    registered = stack.copy()
    shifts = np.zeros((stack.shape[0], 2), dtype=np.float32)

    _print_verbose(
        verbose,
        (
            f"Registering {CANONICAL_AXIS_ORDER} stack with method='{method}', "
            f"registration_channel={int(registration_channel)}, reference_t=0"
        ),
    )
    _print_verbose(verbose, "t=0 shift_y=0.000 shift_x=0.000")

    for t in range(1, stack.shape[0]):
        shift_yx = _estimate_shift(reference_projection, projections[t, :, :], method=method)
        shifts[t, :] = shift_yx
        _print_verbose(
            verbose,
            f"t={t} shift_y={float(shift_yx[0]):.3f} shift_x={float(shift_yx[1]):.3f}",
        )
        registered[t, :, :, :, :] = _apply_translation_to_tzyx(stack[t, :, :, :, :], shift_yx)

    return (registered, shifts) if return_shifts else registered
