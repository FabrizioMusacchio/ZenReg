"""
Axis and shape helpers used across ZenReg.

Author: Fabrizio Musacchio
Date: June 2026
"""

from __future__ import annotations

import numpy as np

CANONICAL_AXIS_ORDER = "TZCYX"


def ensure_tzcyx_stack(stack) -> np.ndarray:
    """
    Validate that an input array follows canonical ``TZCYX`` order.

    Parameters
    ----------
    stack : array-like
        Input stack expected to have shape ``T, Z, C, Y, X``.

    Returns
    -------
    numpy.ndarray
        View or array representation of the input.

    Raises
    ------
    ValueError
        If the input does not have exactly five dimensions.
    """

    stack = np.asarray(stack)
    if stack.ndim != 5:
        raise ValueError(
            f"Expected a {CANONICAL_AXIS_ORDER} stack with 5 dimensions. "
            f"Got shape {stack.shape!r}."
        )
    return stack


def promote_to_tzcyx(stack) -> tuple[np.ndarray, int]:
    """
    Promote simple ``YX`` or ``ZYX`` arrays to ``TZCYX``.

    This helper is intended for filters and projections that can naturally work
    on simpler inputs. Registration itself expects explicit ``TZCYX`` input.

    Parameters
    ----------
    stack : array-like
        Input array with shape ``YX``, ``ZYX``, or ``TZCYX``.

    Returns
    -------
    tuple[numpy.ndarray, int]
        Promoted stack and the original number of dimensions.
    """

    stack = np.asarray(stack)
    original_ndim = stack.ndim
    if stack.ndim == 2:
        return stack[np.newaxis, np.newaxis, np.newaxis, :, :], original_ndim
    if stack.ndim == 3:
        return stack[np.newaxis, :, np.newaxis, :, :], original_ndim
    if stack.ndim == 5:
        return stack, original_ndim
    raise ValueError(
        "Expected a stack with shape YX, ZYX, or TZCYX. "
        f"Got shape {stack.shape!r}."
    )


def restore_promoted_shape(stack: np.ndarray, original_ndim: int) -> np.ndarray:
    """
    Restore the shape of an array promoted by :func:`promote_to_tzcyx`.

    Parameters
    ----------
    stack : numpy.ndarray
        Array in temporary ``TZCYX`` representation.
    original_ndim : int
        Original dimensionality returned by :func:`promote_to_tzcyx`.

    Returns
    -------
    numpy.ndarray
        Array with the original dimensionality.
    """

    if original_ndim == 2:
        return stack[0, 0, 0, :, :]
    if original_ndim == 3:
        return stack[0, :, 0, :, :]
    return stack


def normalize_zrange(zrange, z_count: int, *, strict: bool = False) -> tuple[int, int]:
    """
    Normalize an optional half-open Z range against stack bounds.

    Parameters
    ----------
    zrange : tuple[int, int] or None
        Half-open range ``(start, stop)``. If ``None``, the full Z extent is used.
    z_count : int
        Number of available Z slices.
    strict : bool, optional
        If True, out-of-bound ranges raise an error. If False, ranges are clamped.

    Returns
    -------
    tuple[int, int]
        Sanitized half-open Z range.
    """

    if zrange is None:
        return 0, int(z_count)
    if len(zrange) != 2:
        raise ValueError("zrange must be None or a tuple/list with exactly two integers.")

    start = int(zrange[0])
    stop = int(zrange[1])

    if strict and not 0 <= start < stop <= z_count:
        raise ValueError(
            f"zrange must satisfy 0 <= start < stop <= {z_count}. Got {(start, stop)!r}."
        )

    start = max(0, min(start, z_count))
    stop = max(0, min(stop, z_count))
    if stop < start:
        start, stop = stop, start
    if start == stop:
        if start >= z_count:
            start = max(0, z_count - 1)
            stop = z_count
        else:
            stop = min(z_count, start + 1)
    return start, stop
