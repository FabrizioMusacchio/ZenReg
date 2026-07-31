"""
Filtering and projection helpers for ZenReg registration workflows.

Author: Fabrizio Musacchio
Date: June 2026
"""
# %% IMPORTS
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.ndimage import gaussian_filter, median_filter

from ._axes import normalize_zrange, promote_to_tzcyx, restore_promoted_shape
# %% CONSTANTS
SUPPORTED_FILTERS = {"median", "gaussian"}
SUPPORTED_PROJECTION_METHODS = {"max", "mean", "median", "var", "std"}
# %% HELPER FUNCTIONS
def _normalize_filter_sequence(filters: str | Sequence[str]) -> list[str]:
    """Normalize one or more filter names into a validated execution sequence."""

    filter_sequence = [filters] if isinstance(filters, str) else list(filters)
    if not filter_sequence:
        raise ValueError("filters must contain at least one filter name.")

    normalized = []
    for filter_name in filter_sequence:
        normalized_name = str(filter_name).strip().lower()
        if normalized_name not in SUPPORTED_FILTERS:
            raise ValueError(
                f"Unsupported filter {filter_name!r}. Supported filters: {sorted(SUPPORTED_FILTERS)}."
            )
        normalized.append(normalized_name)
    return normalized

def _normalize_time_dependent_parameter(value, *, time_count: int, name: str, cast):
    """Expand a scalar or time-matched sequence into one value per time point."""

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = list(value)
        if not values:
            raise ValueError(f"{name} must not be an empty list.")
        if len(values) == time_count:
            return [cast(item) for item in values]
        fallback = cast(values[0])
        return [fallback for _ in range(time_count)]
    scalar_value = cast(value)
    return [scalar_value for _ in range(time_count)]

def _apply_filter_sequence_to_volume(
    volume_zyx: np.ndarray,
    *,
    filter_sequence: Sequence[str],
    median_size: int,
    gaussian_sigma: float,
    apply_3d: bool,
) -> np.ndarray:
    """Apply a validated filter sequence to one ``ZYX`` volume."""

    working_volume = np.asarray(volume_zyx, dtype=np.float32).copy()

    for filter_name in filter_sequence:
        if apply_3d:
            if filter_name == "median":
                working_volume = median_filter(
                    working_volume,
                    size=(median_size, median_size, median_size),
                ).astype(np.float32, copy=False)
            else:
                working_volume = gaussian_filter(
                    working_volume,
                    sigma=(gaussian_sigma, gaussian_sigma, gaussian_sigma),
                ).astype(np.float32, copy=False)
        else:
            filtered = np.empty_like(working_volume, dtype=np.float32)
            for z in range(working_volume.shape[0]):
                plane = working_volume[z, :, :]
                if filter_name == "median":
                    filtered[z, :, :] = median_filter(plane, size=(median_size, median_size))
                else:
                    filtered[z, :, :] = gaussian_filter(plane, sigma=(gaussian_sigma, gaussian_sigma))
            working_volume = filtered
    return working_volume

def _normalize_projection_method(projection_method: str) -> str:
    """Normalize and validate a Z-projection method."""

    normalized = str(projection_method).strip().lower()
    if normalized not in SUPPORTED_PROJECTION_METHODS:
        raise ValueError(
            f"Unsupported projection_method {projection_method!r}. "
            f"Supported methods: {sorted(SUPPORTED_PROJECTION_METHODS)}."
        )
    return normalized

def _project_z(stack: np.ndarray, *, projection_method: str) -> np.ndarray:
    """Project a temporary ``TZCYX`` stack along Z while preserving Z as length 1."""

    if projection_method == "max":
        return np.max(stack, axis=1, keepdims=True)
    if projection_method == "mean":
        return np.mean(stack, axis=1, keepdims=True)
    if projection_method == "median":
        return np.median(stack, axis=1, keepdims=True)
    if projection_method == "var":
        return np.var(stack, axis=1, keepdims=True)
    return np.std(stack, axis=1, keepdims=True)

def apply_filters(
    stack,
    filters: str | Sequence[str],
    *,
    median_size: int | Sequence[int] = 3,
    gaussian_sigma: float | Sequence[float] = 1.0,
    apply_3d: bool = False,
) -> np.ndarray:
    """
    Apply one or more filters to a ``TZCYX``, ``ZYX``, or ``YX`` image stack.

    Parameters
    ----------
    stack : array-like
        Input image. ``TZCYX`` stacks are filtered per time point and channel.
        Simpler ``ZYX`` and ``YX`` inputs are temporarily promoted.
    filters : str or sequence of str
        Filter name or ordered filter sequence. Supported values are ``"median"``
        and ``"gaussian"``.
    median_size : int or sequence of int, optional
        Median kernel size. If a sequence with length ``T`` is provided, the
        value is applied per time point. Otherwise the first value is reused.
    gaussian_sigma : float or sequence of float, optional
        Gaussian sigma. If a sequence with length ``T`` is provided, the value
        is applied per time point. Otherwise the first value is reused.
    apply_3d : bool, optional
        If True, filters are applied in ``ZYX``. If False, filters are applied
        plane-wise in ``YX``.

    Returns
    -------
    numpy.ndarray
        Filtered image with the same dimensionality as the input.
    """

    filter_sequence = _normalize_filter_sequence(filters)
    original_stack = np.asarray(stack)
    working_stack, original_ndim = promote_to_tzcyx(original_stack)
    working_stack = working_stack.astype(np.float32, copy=True)
    time_count = int(working_stack.shape[0])

    median_sizes = _normalize_time_dependent_parameter(
        median_size,
        time_count=time_count,
        name="median_size",
        cast=int,
    )
    gaussian_sigmas = _normalize_time_dependent_parameter(
        gaussian_sigma,
        time_count=time_count,
        name="gaussian_sigma",
        cast=float,
    )

    filtered = np.empty_like(working_stack, dtype=np.float32)
    for t in range(time_count):
        if median_sizes[t] < 1:
            raise ValueError(f"median_size must be >= 1. Got {median_sizes[t]!r} at t={t}.")
        if gaussian_sigmas[t] <= 0:
            raise ValueError(f"gaussian_sigma must be > 0. Got {gaussian_sigmas[t]!r} at t={t}.")
        for c in range(working_stack.shape[2]):
            filtered[t, :, c, :, :] = _apply_filter_sequence_to_volume(
                working_stack[t, :, c, :, :],
                filter_sequence=filter_sequence,
                median_size=int(median_sizes[t]),
                gaussian_sigma=float(gaussian_sigmas[t]),
                apply_3d=apply_3d,
            )
    return restore_promoted_shape(filtered, original_ndim)

def z_project(
    stack,
    *,
    zrange: tuple[int, int] | Sequence[int] | None = None,
    projection_method: str = "max",
) -> np.ndarray:
    """
    Project over Z while preserving ``T`` and ``C``.

    Parameters
    ----------
    stack : array-like
        Input image with shape ``TZCYX``, ``ZYX``, or ``YX``.
    zrange : tuple[int, int] or None, optional
        Optional half-open Z range ``(start, stop)``. Out-of-bound values are
        clamped to the stack extent.
    projection_method : {"max", "mean", "median", "var", "std"}, optional
        Projection method used along Z. ``"max"`` is a good default for sparse
        spots or puncta. ``"mean"`` is often better for dense, spatially
        extended signal. ``"median"`` is robust to outliers, but can attenuate
        sparse spots. ``"std"`` and ``"var"`` can be useful when
        contrast-rich structure matters more than absolute intensity. A
        percentile projection, for example p95, would also be a useful
        microscopy-oriented future extension.

    Returns
    -------
    numpy.ndarray
        Projected image. A ``TZCYX`` input returns a ``T, 1, C, Y, X`` stack.
    """

    projection_method = _normalize_projection_method(projection_method)
    original_stack = np.asarray(stack)
    working_stack, original_ndim = promote_to_tzcyx(original_stack)
    z_start, z_stop = normalize_zrange(zrange, working_stack.shape[1])
    projected = _project_z(
        working_stack[:, z_start:z_stop, :, :, :],
        projection_method=projection_method,
    )
    return restore_promoted_shape(projected.astype(np.float32, copy=False), original_ndim)

def max_z_project(stack, *, zrange: tuple[int, int] | Sequence[int] | None = None) -> np.ndarray:
    """
    Compute a maximum-intensity projection over Z while preserving ``T`` and ``C``.

    This is a convenience wrapper around :func:`z_project`.
    """

    return z_project(stack, zrange=zrange, projection_method="max")
# %% END