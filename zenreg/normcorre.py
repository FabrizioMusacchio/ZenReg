"""
NoRMCorre-style non-rigid motion correction for canonical ``TZCYX`` stacks.

This module implements a dependency-light ZenReg port of the main CaImAn
NoRMCorre batch path: estimate a rigid correction first, estimate local
patch-wise translation corrections around that rigid shift, interpolate the
patch shifts to a dense displacement field, update templates from corrected
time chunks, and apply the resulting field to every channel of the frame. It
supports 2D+t and full 3D+t stacks without requiring the full CaImAn suite.

TODO: implement NoRMCorre-compatible zero clipping for the registered output.
For now, ``register_stack(method="normcorre", zero_clip=True)`` warns and
continues without zero clipping.

Author: Fabrizio Musacchio
Date: July 2026
"""
# %% IMPORTS
from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import convolve, map_coordinates
from skimage.registration import phase_cross_correlation
from skimage.transform import resize as resize_image

from ._axes import CANONICAL_AXIS_ORDER, ensure_tzcyx_stack, normalize_zrange
from .filters import _normalize_projection_method, z_project
# %% CONSTANTS AND DATA CLASSES

@dataclass(frozen=True)
class _PatchGrid:
    """Patch slices and centers for one spatial dimensionality."""

    slices: list[tuple[slice, ...]]
    centers_by_axis: tuple[np.ndarray, ...]
    grid_shape: tuple[int, ...]

# %% HELPER FUNCTIONS
def _as_float32_stack(stack) -> np.ndarray:
    """Return a float32 ``TZCYX`` working view/copy."""

    stack = ensure_tzcyx_stack(stack)
    try:
        return stack.astype(np.float32, copy=False)
    except (AttributeError, TypeError):
        return np.asarray(stack, dtype=np.float32)

def _normalize_spatial_tuple(
    value,
    *,
    ndim: int,
    default: tuple[int, ...],
    name: str,
) -> tuple[int, ...]:
    """Normalize scalar or tuple parameters used along spatial axes."""

    if value is None:
        values = default
    elif np.isscalar(value):
        values = (int(value),) * ndim
    else:
        if len(value) != ndim:
            raise ValueError(f"{name} must have {ndim} values for this registration mode.")
        values = tuple(int(v) for v in value)
    if any(v < 1 for v in values):
        raise ValueError(f"{name} values must be >= 1. Got {value!r}.")
    return values

def _normalize_max_shifts(max_shifts, *, ndim: int) -> np.ndarray | None:
    """Normalize optional absolute shift limits in spatial axis order."""

    if max_shifts is None:
        return None
    if np.isscalar(max_shifts):
        limits = np.asarray([float(max_shifts)] * ndim, dtype=np.float32)
    else:
        if len(max_shifts) != ndim:
            raise ValueError(f"max_shifts must be None, scalar, or contain {ndim} values.")
        limits = np.asarray([float(v) for v in max_shifts], dtype=np.float32)
    if np.any(limits < 0):
        raise ValueError(f"max_shifts values must be >= 0. Got {max_shifts!r}.")
    return limits

def _normalize_max_deviation(max_deviation_rigid, *, ndim: int) -> np.ndarray | None:
    """Normalize optional patch-shift deviation limits around the rigid shift."""

    if max_deviation_rigid is None:
        return None
    if np.isscalar(max_deviation_rigid):
        limits = np.asarray([float(max_deviation_rigid)] * ndim, dtype=np.float32)
    else:
        if len(max_deviation_rigid) != ndim:
            raise ValueError(f"max_deviation_rigid must be scalar or contain {ndim} values.")
        limits = np.asarray([float(v) for v in max_deviation_rigid], dtype=np.float32)
    if np.any(limits < 0):
        raise ValueError(f"max_deviation_rigid values must be >= 0. Got {max_deviation_rigid!r}.")
    return limits

def _normalize_choice(value: str, *, allowed: set[str], name: str) -> str:
    """Normalize a small string option."""

    normalized = str(value).strip().lower()
    if normalized not in allowed:
        raise ValueError(f"{name} must be one of {sorted(allowed)}. Got {value!r}.")
    return normalized

def _normalize_gsig_filt(gSig_filt, *, ndim: int) -> tuple[float, float] | None:
    """Normalize CaImAn-style spatial high-pass filter sigma values."""

    if gSig_filt is None:
        return None
    if np.isscalar(gSig_filt):
        values = (float(gSig_filt), float(gSig_filt))
    else:
        if len(gSig_filt) < 2:
            raise ValueError("gSig_filt must be None, a scalar, or contain at least Y/X values.")
        values = tuple(float(v) for v in gSig_filt[-2:])
    if ndim not in (2, 3):
        raise ValueError(f"gSig_filt is only supported for 2D/3D registration images. Got ndim={ndim}.")
    if any(v <= 0 for v in values):
        raise ValueError(f"gSig_filt values must be > 0. Got {gSig_filt!r}.")
    return values

def _caiman_high_pass_kernel(gSig_filt: tuple[float, float]) -> np.ndarray:
    """Create CaImAn's centered Gaussian high-pass kernel."""

    ky, kx = [int((3 * sigma) // 2 * 2 + 1) for sigma in gSig_filt]
    yy = np.arange(ky, dtype=np.float32) - (ky - 1) / 2.0
    xx = np.arange(kx, dtype=np.float32) - (kx - 1) / 2.0
    gy = np.exp(-(yy**2) / (2 * float(gSig_filt[0]) ** 2))
    gx = np.exp(-(xx**2) / (2 * float(gSig_filt[1]) ** 2))
    gy /= np.sum(gy)
    gx /= np.sum(gx)
    kernel = np.outer(gy, gx).astype(np.float32)
    edge_threshold = float(kernel[:, 0].max())
    core_mask = kernel >= edge_threshold
    kernel[core_mask] -= float(kernel[core_mask].mean())
    kernel[~core_mask] = 0
    return kernel

def _high_pass_filter_space(image: np.ndarray, gSig_filt: tuple[float, float] | None) -> np.ndarray:
    """Apply CaImAn-style XY spatial high-pass filtering."""

    image = np.asarray(image, dtype=np.float32)
    if gSig_filt is None:
        return image
    kernel = _caiman_high_pass_kernel(gSig_filt)
    if image.ndim == 2:
        return convolve(image, kernel, mode="reflect").astype(np.float32, copy=False)
    if image.ndim == 3:
        filtered = np.empty_like(image, dtype=np.float32)
        for z in range(image.shape[0]):
            filtered[z] = convolve(image[z], kernel, mode="reflect")
        return filtered
    raise ValueError(f"Expected a 2D or 3D image for gSig_filt. Got ndim={image.ndim}.")

def _clip_shift(
    shift: np.ndarray,
    *,
    max_shifts: np.ndarray | None,
    rigid_shift: np.ndarray | None = None,
    max_deviation_rigid: np.ndarray | None = None,
) -> np.ndarray:
    """Clip a correction shift by absolute and rigid-relative limits."""

    shift = np.asarray(shift, dtype=np.float32).copy()
    if rigid_shift is not None and max_deviation_rigid is not None:
        rigid_shift = np.asarray(rigid_shift, dtype=np.float32)
        shift = np.clip(shift, rigid_shift - max_deviation_rigid, rigid_shift + max_deviation_rigid)
    if max_shifts is not None:
        shift = np.clip(shift, -max_shifts, max_shifts)
    return shift.astype(np.float32, copy=False)

def _patch_starts(dim: int, window: int, stride: int) -> list[int]:
    """Return CaImAn-style patch starts, including the final edge patch."""

    window = min(int(window), int(dim))
    stride = max(int(stride), 1)
    if dim <= window:
        return [0]
    starts = list(range(0, dim - window, stride))
    final_start = dim - window
    if not starts or starts[-1] != final_start:
        starts.append(final_start)
    return starts

def _build_patch_grid(
    spatial_shape: tuple[int, ...],
    *,
    strides: tuple[int, ...],
    overlaps: tuple[int, ...],
) -> _PatchGrid:
    """Build patch slices and center coordinates for a spatial shape."""

    windows = tuple(min(dim, stride + overlap) for dim, stride, overlap in zip(spatial_shape, strides, overlaps))
    starts_by_axis = tuple(
        _patch_starts(dim, window, stride)
        for dim, window, stride in zip(spatial_shape, windows, strides)
    )
    centers_by_axis = tuple(
        np.asarray([start + (window - 1) / 2.0 for start in starts], dtype=np.float32)
        for starts, window in zip(starts_by_axis, windows)
    )

    slices: list[tuple[slice, ...]] = []
    for index in np.ndindex(*(len(starts) for starts in starts_by_axis)):
        patch_slices = tuple(
            slice(starts_by_axis[axis][index[axis]], starts_by_axis[axis][index[axis]] + windows[axis])
            for axis in range(len(spatial_shape))
        )
        slices.append(patch_slices)

    return _PatchGrid(
        slices=slices,
        centers_by_axis=centers_by_axis,
        grid_shape=tuple(len(starts) for starts in starts_by_axis),
    )

def _project_zyx_for_overlay(volume_zyx: np.ndarray, *, projection_method: str) -> np.ndarray:
    """Project one ``ZYX`` volume to a ``YX`` image for patch-grid visualization."""

    if projection_method == "max":
        return np.max(volume_zyx, axis=0)
    if projection_method == "mean":
        return np.mean(volume_zyx, axis=0)
    if projection_method == "median":
        return np.median(volume_zyx, axis=0)
    if projection_method == "var":
        return np.var(volume_zyx, axis=0)
    return np.std(volume_zyx, axis=0)

def _metadata_source_parent(metadata: dict[str, Any] | None) -> Path:
    """Return the best available source folder from an OMIO metadata dictionary."""

    if metadata is None:
        return Path.cwd()
    annotations = metadata.get("Annotations", {}) if isinstance(metadata, dict) else {}
    for key in ("original_parentfolder", "omio_cache_folder", "omio_zarr_store_path"):
        value = annotations.get(key) if key in annotations else metadata.get(key)
        if value:
            path = Path(value)
            return path if path.suffix == "" else path.parent
    return Path.cwd()

def _metadata_source_stem(metadata: dict[str, Any] | None, *, fallback: str) -> str:
    """Return a compact source name from OMIO metadata."""

    if metadata is None:
        return fallback
    annotations = metadata.get("Annotations", {}) if isinstance(metadata, dict) else {}
    filename = annotations.get("original_filename") or metadata.get("original_filename")
    if not filename:
        return fallback
    name = Path(str(filename)).name
    lower_name = name.lower()
    for suffix in (".ome.tiff", ".ome.tif", ".tiff", ".tif"):
        if lower_name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem

def _normalize_overlay_yx_tuple(
    value,
    *,
    z_count: int,
    default_2d: tuple[int, int],
    default_3d: tuple[int, int, int],
    name: str,
) -> tuple[tuple[int, int], int | None]:
    """Normalize NoRMCorre patch settings for a YX projection overlay."""

    if value is None:
        values = default_3d if z_count > 1 else default_2d
    elif np.isscalar(value):
        values = (int(value), int(value))
    else:
        values = tuple(int(v) for v in value)
        if len(values) not in (2, 3):
            raise ValueError(f"{name} must be a scalar or contain 2 YX or 3 ZYX values.")
    if any(v < 1 for v in values):
        raise ValueError(f"{name} values must be >= 1. Got {value!r}.")
    if len(values) == 3:
        return (int(values[1]), int(values[2])), int(values[0])
    return (int(values[0]), int(values[1])), None

def plot_normcorre_patch_overlay(
    stack,
    metadata: dict[str, Any] | None = None,
    *,
    registration_channel: int = 0,
    registration_stack: int = 0,
    nc_strides: tuple[int, ...] | int | None = None,
    nc_overlaps: tuple[int, ...] | int | None = None,
    projection_method: str = "max",
    projection_range: tuple[int, int] | Sequence[int] | None = (1, 10),
    output_dir: str | Path | None = None,
    output_name: str | None = None,
    show: bool = False,
    dpi: int = 180,
) -> Path:
    """
    Plot and save a NoRMCorre patch/stride overlay on one reference projection.

    This helper is intended to be called after ``load_stack`` and before
    ``register_stack(method="normcorre")``. It reads only one time point, one
    channel, and the requested Z range, so OMIO/Zarr-backed large stacks are not
    materialized in full. For 3D NoRMCorre settings, the YX patch footprints are
    drawn on the Z projection and the Z stride/overlap values are noted in the
    plot annotation.

    Parameters
    ----------
    stack : array-like
        Input image in canonical ``TZCYX`` order.
    metadata : dict or None, optional
        OMIO metadata. Used only to infer a convenient default output folder and
        filename.
    registration_channel, registration_stack : int, optional
        Channel and time point shown in the overlay.
    nc_strides, nc_overlaps : tuple, int, or None, optional
        NoRMCorre patch-grid settings. The effective patch size is
        ``nc_strides + nc_overlaps``. 2D settings are interpreted as ``YX``;
        3D settings are interpreted as ``ZYX`` and projected to YX.
    projection_method : {"max", "mean", "median", "var", "std"}, optional
        Projection method for the selected reference volume.
    projection_range : tuple[int, int] or None, optional
        Half-open Z range for the projection. The default ``(1, 10)`` is
        clamped to the available Z extent; if the stack has fewer slices, all
        available slices in that range are used.
    output_dir : str, pathlib.Path, or None, optional
        Destination folder. If None, the plot is saved in
        ``<source_parent>/registered_normcorre`` when source metadata is
        available, otherwise in ``./registered_normcorre``.
    output_name : str or None, optional
        PNG filename. If None, a descriptive filename is generated.
    show : bool, optional
        If True, display the figure interactively after saving.
    dpi : int, optional
        Figure resolution.

    Returns
    -------
    pathlib.Path
        Path to the saved PNG overlay.
    """

    stack = ensure_tzcyx_stack(stack)
    time_count, z_count, channel_count, y_count, x_count = stack.shape
    registration_stack = int(registration_stack)
    registration_channel = int(registration_channel)
    if not 0 <= registration_stack < time_count:
        raise ValueError(f"registration_stack must be between 0 and {time_count - 1}.")
    if not 0 <= registration_channel < channel_count:
        raise ValueError(f"registration_channel must be between 0 and {channel_count - 1}.")

    projection_method = _normalize_projection_method(projection_method)
    z_start, z_stop = normalize_zrange(projection_range, z_count, strict=False)
    volume = np.asarray(
        stack[registration_stack, z_start:z_stop, registration_channel, :, :],
        dtype=np.float32,
    )
    projection = _project_zyx_for_overlay(volume, projection_method=projection_method)

    strides_yx, stride_z = _normalize_overlay_yx_tuple(
        nc_strides,
        z_count=z_count,
        default_2d=(48, 48),
        default_3d=(6, 48, 48),
        name="nc_strides",
    )
    overlaps_yx, overlap_z = _normalize_overlay_yx_tuple(
        nc_overlaps,
        z_count=z_count,
        default_2d=(24, 24),
        default_3d=(3, 24, 24),
        name="nc_overlaps",
    )
    patch_grid = _build_patch_grid(
        (int(y_count), int(x_count)),
        strides=strides_yx,
        overlaps=overlaps_yx,
    )

    if output_dir is None:
        output_dir = _metadata_source_parent(metadata) / "registered_normcorre"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_name is None:
        stem = _metadata_source_stem(metadata, fallback="zenreg_stack")
        output_name = (
            f"{stem}_normcorre_patch_overlay_"
            f"t{registration_stack}_c{registration_channel}_{projection_method}_z{z_start}-{z_stop}.png"
        )
    output_path = output_dir / output_name

    import matplotlib.lines as mlines
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    finite_projection = projection[np.isfinite(projection)]
    if finite_projection.size:
        vmin, vmax = np.percentile(finite_projection, [1, 99.5])
        if vmin == vmax:
            vmin, vmax = None, None
    else:
        vmin, vmax = None, None

    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    ax.imshow(projection, cmap="gray", vmin=vmin, vmax=vmax)

    y_intervals = sorted({(y_slice.start, y_slice.stop) for y_slice, _ in patch_grid.slices})
    x_intervals = sorted({(x_slice.start, x_slice.stop) for _, x_slice in patch_grid.slices})

    def _adjacent_overlaps(intervals):
        for previous, current in zip(intervals[:-1], intervals[1:]):
            start = max(previous[0], current[0])
            stop = min(previous[1], current[1])
            if stop > start:
                yield start, stop

    for overlap_start, overlap_stop in _adjacent_overlaps(x_intervals):
        ax.add_patch(
            patches.Rectangle(
                (overlap_start, -0.5),
                overlap_stop - overlap_start,
                y_count,
                facecolor="white",
                edgecolor="none",
                alpha=0.22,
                zorder=1,
            )
        )
    for overlap_start, overlap_stop in _adjacent_overlaps(y_intervals):
        ax.add_patch(
            patches.Rectangle(
                (-0.5, overlap_start),
                x_count,
                overlap_stop - overlap_start,
                facecolor="white",
                edgecolor="none",
                alpha=0.22,
                zorder=1,
            )
        )

    for y_slice, x_slice in patch_grid.slices:
        rect = patches.Rectangle(
            (x_slice.start, y_slice.start),
            x_slice.stop - x_slice.start,
            y_slice.stop - y_slice.start,
            fill=False,
            edgecolor="tab:orange",
            linewidth=1.35,
            alpha=0.9,
            zorder=3,
        )
        ax.add_patch(rect)
    for center_y in patch_grid.centers_by_axis[0]:
        ax.axhline(float(center_y), color="tab:cyan", linewidth=1.1, alpha=0.65, zorder=2)
    for center_x in patch_grid.centers_by_axis[1]:
        ax.axvline(float(center_x), color="tab:cyan", linewidth=1.1, alpha=0.65, zorder=2)
    ax.scatter(
        np.repeat(patch_grid.centers_by_axis[1], len(patch_grid.centers_by_axis[0])),
        np.tile(patch_grid.centers_by_axis[0], len(patch_grid.centers_by_axis[1])),
        s=18,
        color="tab:cyan",
        alpha=0.85,
        linewidths=0,
        zorder=4,
    )
    annotation = [
        f"strides_yx={strides_yx} overlaps_yx={overlaps_yx}",
        f"patch_size_yx=({strides_yx[0] + overlaps_yx[0]}, {strides_yx[1] + overlaps_yx[1]})",
        f"grid_yx={patch_grid.grid_shape} patches={len(patch_grid.slices)}",
    ]
    if stride_z is not None or overlap_z is not None:
        annotation.append(f"z stride/overlap={stride_z}/{overlap_z} projected z={z_start}:{z_stop}")
    else:
        annotation.append(f"projected z={z_start}:{z_stop}")
    ax.text(
        0.01,
        0.99,
        "\n".join(annotation),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="white",
        bbox={"facecolor": "black", "alpha": 0.55, "edgecolor": "none", "pad": 4},
    )
    ax.set_title(
        f"NoRMCorre patch overlay: t={registration_stack}, c={registration_channel}, {projection_method}"
    )
    ax.set_xlim(-0.5, x_count - 0.5)
    ax.set_ylim(y_count - 0.5, -0.5)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    legend_handles = [
        patches.Patch(facecolor="white", edgecolor="none", alpha=0.22, label="overlap region"),
        patches.Patch(facecolor="none", edgecolor="tab:orange", linewidth=1.35, label="patch footprint"),
        mlines.Line2D(
            [],
            [],
            color="tab:cyan",
            marker="o",
            linestyle="-",
            linewidth=1.1,
            markersize=4,
            alpha=0.85,
            label="patch centers",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower right",
        fontsize=8,
        framealpha=0.72,
        facecolor="black",
        edgecolor="none",
        labelcolor="white",
    )
    fig.savefig(output_path, dpi=int(dpi))
    if show:
        plt.show()
    else:
        plt.close(fig)
    return output_path

def _estimate_shift(
    reference: np.ndarray,
    moving: np.ndarray,
    *,
    upsample_factor: int,
    normalization: str | None,
) -> tuple[np.ndarray, float]:
    """Estimate the correction shift needed to align ``moving`` to ``reference``."""

    shift, error, _ = phase_cross_correlation(
        np.asarray(reference, dtype=np.float32),
        np.asarray(moving, dtype=np.float32),
        upsample_factor=int(upsample_factor),
        normalization=normalization,
    )
    return np.asarray(shift, dtype=np.float32), float(error)

def _estimate_patch_shifts(
    reference: np.ndarray,
    moving: np.ndarray,
    *,
    patch_grid: _PatchGrid,
    rigid_shift: np.ndarray,
    max_shifts: np.ndarray | None,
    max_deviation_rigid: np.ndarray | None,
    upsample_factor: int,
    normalization: str | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate local patch shifts and errors on the patch grid."""

    ndim = reference.ndim
    patch_shifts = np.zeros((*patch_grid.grid_shape, ndim), dtype=np.float32)
    patch_errors = np.zeros(patch_grid.grid_shape, dtype=np.float32)

    for flat_index, patch_slices in enumerate(patch_grid.slices):
        grid_index = np.unravel_index(flat_index, patch_grid.grid_shape)
        ref_patch = reference[patch_slices]
        mov_patch = moving[patch_slices]
        shift, error = _estimate_shift(
            ref_patch,
            mov_patch,
            upsample_factor=upsample_factor,
            normalization=normalization,
        )
        patch_shifts[grid_index] = _clip_shift(
            shift,
            max_shifts=max_shifts,
            rigid_shift=rigid_shift,
            max_deviation_rigid=max_deviation_rigid,
        )
        patch_errors[grid_index] = error

    return patch_shifts, patch_errors

def _uniform_shift_grid(
    spatial_shape: tuple[int, ...],
    shift: np.ndarray,
) -> tuple[tuple[np.ndarray, ...], np.ndarray]:
    """Return a one-point grid that applies the same shift everywhere."""

    centers = tuple(np.asarray([(dim - 1) / 2.0], dtype=np.float32) for dim in spatial_shape)
    shift_grid = np.zeros((*((1,) * len(spatial_shape)), len(spatial_shape)), dtype=np.float32)
    shift_grid[...] = np.asarray(shift, dtype=np.float32)
    return centers, shift_grid

def _regular_interpolators(
    centers_by_axis: tuple[np.ndarray, ...],
    shift_grid: np.ndarray,
    spatial_shape: tuple[int, ...],
) -> list[RegularGridInterpolator]:
    """Create one shift interpolator for each spatial axis."""

    ndim = len(centers_by_axis)
    expanded_centers: list[np.ndarray] = []
    expanded_shift_grid = np.asarray(shift_grid, dtype=np.float32)
    for axis, (centers, dim) in enumerate(zip(centers_by_axis, spatial_shape)):
        centers = np.asarray(centers, dtype=np.float32)
        target_centers = centers
        prepend_edge = centers[0] > 0
        append_edge = centers[-1] < dim - 1
        if prepend_edge:
            target_centers = np.concatenate([np.asarray([0.0], dtype=np.float32), target_centers])
            edge_values = np.take(expanded_shift_grid, [0], axis=axis)
            expanded_shift_grid = np.concatenate([edge_values, expanded_shift_grid], axis=axis)
        if append_edge:
            target_centers = np.concatenate([target_centers, np.asarray([dim - 1.0], dtype=np.float32)])
            edge_values = np.take(expanded_shift_grid, [-1], axis=axis)
            expanded_shift_grid = np.concatenate([expanded_shift_grid, edge_values], axis=axis)
        expanded_centers.append(target_centers)

    return [
        RegularGridInterpolator(
            tuple(expanded_centers),
            expanded_shift_grid[..., axis],
            bounds_error=False,
            fill_value=None,
        )
        for axis in range(ndim)
    ]

def _dense_shift_field_resize(
    shift_grid: np.ndarray,
    spatial_shape: tuple[int, ...],
) -> list[np.ndarray]:
    """Upsample patch shifts to image size like CaImAn's default remap path."""

    shift_grid = np.asarray(shift_grid, dtype=np.float32)
    return [
        resize_image(
            shift_grid[..., axis],
            spatial_shape,
            order=3,
            mode="edge",
            anti_aliasing=False,
            preserve_range=True,
        ).astype(np.float32, copy=False)
        for axis in range(len(spatial_shape))
    ]

def _evaluate_shift_block(
    coords: tuple[np.ndarray, ...],
    *,
    interpolators: list[RegularGridInterpolator],
) -> list[np.ndarray]:
    """Evaluate interpolated shifts for one block of output coordinates."""

    points = np.stack([coord.ravel() for coord in coords], axis=1)
    return [
        interpolator(points).reshape(coords[0].shape).astype(np.float32, copy=False)
        for interpolator in interpolators
    ]

def _map_border_settings(image: np.ndarray, *, border_nan, mode: str, cval: float) -> tuple[str, float]:
    """Map CaImAn-style border_nan values to scipy map_coordinates settings."""

    if border_nan is None:
        return mode, float(cval)
    if border_nan is False:
        return "constant", 0.0
    if border_nan is True:
        return "constant", np.nan
    if border_nan == "min":
        return "constant", float(np.nanmin(image))
    if border_nan == "copy":
        return "nearest", 0.0
    raise ValueError("border_nan must be None, False, True, 'min', or 'copy'.")

def _warp_with_shift_grid(
    image: np.ndarray,
    *,
    centers_by_axis: tuple[np.ndarray, ...],
    shift_grid: np.ndarray,
    shift_interpolation: str,
    order: int,
    mode: str,
    cval: float,
    border_nan,
    block_size: int,
) -> np.ndarray:
    """Apply an interpolated correction-shift grid to a 2D or 3D image."""

    image = np.asarray(image, dtype=np.float32)
    ndim = image.ndim
    output = np.empty_like(image, dtype=np.float32)
    mode, cval = _map_border_settings(image, border_nan=border_nan, mode=mode, cval=cval)
    if shift_interpolation == "resize":
        dense_shifts = _dense_shift_field_resize(shift_grid, image.shape)
        interpolators = None
    else:
        dense_shifts = None
        interpolators = _regular_interpolators(centers_by_axis, shift_grid, image.shape)
    block_size = max(int(block_size), 1)

    if ndim == 2:
        height, width = image.shape
        for y0 in range(0, height, block_size):
            y1 = min(y0 + block_size, height)
            yy, xx = np.mgrid[y0:y1, 0:width].astype(np.float32)
            shifts = (
                [dense_shifts[0][y0:y1, :], dense_shifts[1][y0:y1, :]]
                if dense_shifts is not None
                else _evaluate_shift_block((yy, xx), interpolators=interpolators)
            )
            sample_coords = [yy - shifts[0], xx - shifts[1]]
            output[y0:y1, :] = map_coordinates(
                image,
                sample_coords,
                order=int(order),
                mode=mode,
                cval=float(cval),
                prefilter=int(order) > 1,
            )
        return output

    if ndim == 3:
        depth, height, width = image.shape
        for z0 in range(0, depth, block_size):
            z1 = min(z0 + block_size, depth)
            zz, yy, xx = np.mgrid[z0:z1, 0:height, 0:width].astype(np.float32)
            shifts = (
                [
                    dense_shifts[0][z0:z1, :, :],
                    dense_shifts[1][z0:z1, :, :],
                    dense_shifts[2][z0:z1, :, :],
                ]
                if dense_shifts is not None
                else _evaluate_shift_block((zz, yy, xx), interpolators=interpolators)
            )
            sample_coords = [zz - shifts[0], yy - shifts[1], xx - shifts[2]]
            output[z0:z1, :, :] = map_coordinates(
                image,
                sample_coords,
                order=int(order),
                mode=mode,
                cval=float(cval),
                prefilter=int(order) > 1,
            )
        return output

    raise ValueError(f"Only 2D and 3D spatial images are supported. Got ndim={ndim}.")

def _registration_image(
    frame_tzcyx: np.ndarray,
    *,
    channel: int,
    is3d: bool,
    projection_range: tuple[int, int] | None,
    projection_method: str,
) -> np.ndarray:
    """Extract the image used for NoRMCorre shift estimation."""

    volume = np.asarray(frame_tzcyx[:, channel, :, :], dtype=np.float32)
    if projection_range is not None:
        z_start, z_stop = projection_range
        volume = volume[int(z_start) : int(z_stop), :, :]
    if is3d:
        return volume
    if volume.shape[0] == 1:
        return volume[0]
    projection_input = volume[np.newaxis, :, np.newaxis, :, :]
    return z_project(projection_input, projection_method=projection_method)[0, 0, 0].astype(np.float32, copy=False)

def _registration_image_for_estimation(
    frame_tzcyx: np.ndarray,
    *,
    channel: int,
    is3d: bool,
    projection_range: tuple[int, int] | None,
    projection_method: str,
    gSig_filt: tuple[float, float] | None,
) -> np.ndarray:
    """Extract and optionally high-pass filter the image used for shift estimation."""

    return _high_pass_filter_space(
        _registration_image(
            frame_tzcyx,
            channel=channel,
            is3d=is3d,
            projection_range=projection_range,
            projection_method=projection_method,
        ),
        gSig_filt,
    )

def _registration_images(
    stack: np.ndarray,
    *,
    channel: int,
    is3d: bool,
    projection_range: tuple[int, int] | None,
    projection_method: str,
    gSig_filt: tuple[float, float] | None = None,
    indices: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Extract registration images for selected frames."""

    if indices is None:
        indices = np.arange(stack.shape[0], dtype=np.int32)
    return [
        _registration_image_for_estimation(
            stack[int(t)],
            channel=channel,
            is3d=is3d,
            projection_range=projection_range,
            projection_method=projection_method,
            gSig_filt=gSig_filt,
        )
        for t in np.asarray(indices, dtype=np.int32)
    ]

def _frame_sample_indices(time_count: int, *, is3d: bool) -> np.ndarray:
    """Return CaImAn-like sparse frame indices used for initial templates."""

    step = time_count // 10 if is3d else time_count // 50
    return np.arange(time_count, dtype=np.int32)[slice(None, None, step + 1)]

def _update_template_from_registered(
    registered: np.ndarray,
    *,
    registration_channel: int,
    is3d: bool,
    projection_range: tuple[int, int] | None,
    projection_method: str,
    gSig_filt: tuple[float, float] | None,
    template_update_method: str,
    splits: int,
) -> np.ndarray:
    """Update template from corrected frames using CaImAn-style chunk logic."""

    time_count = registered.shape[0]
    indices = np.arange(time_count, dtype=np.int32)
    if template_update_method == "mean":
        images = _registration_images(
            registered,
            channel=registration_channel,
            is3d=is3d,
            projection_range=projection_range,
            projection_method=projection_method,
            indices=indices,
        )
        template = np.nanmean(np.stack(images, axis=0), axis=0).astype(np.float32)
        return _high_pass_filter_space(template, gSig_filt).astype(np.float32, copy=False)
    if template_update_method == "median":
        images = _registration_images(
            registered,
            channel=registration_channel,
            is3d=is3d,
            projection_range=projection_range,
            projection_method=projection_method,
            indices=indices,
        )
        template = np.nanmedian(np.stack(images, axis=0), axis=0).astype(np.float32)
        return _high_pass_filter_space(template, gSig_filt).astype(np.float32, copy=False)

    chunk_count = max(1, min(int(splits), time_count))
    chunk_templates = []
    for chunk_indices in np.array_split(indices, chunk_count):
        if len(chunk_indices) == 0:
            continue
        images = _registration_images(
            registered,
            channel=registration_channel,
            is3d=is3d,
            projection_range=projection_range,
            projection_method=projection_method,
            indices=chunk_indices,
        )
        chunk_template = np.nanmean(np.stack(images, axis=0), axis=0)
        chunk_template[np.isnan(chunk_template)] = np.nanmin(chunk_template)
        chunk_templates.append(chunk_template)
    template = np.nanmedian(np.stack(chunk_templates, axis=0), axis=0).astype(np.float32)
    return _high_pass_filter_space(template, gSig_filt).astype(np.float32, copy=False)

def _estimate_min_mov(
    stack: np.ndarray,
    *,
    registration_channel: int,
    is3d: bool,
    projection_range: tuple[int, int] | None,
    projection_method: str,
    gSig_filt: tuple[float, float] | None,
    max_frames: int = 400,
) -> float:
    """Estimate CaImAn's min_mov from the first frames used for registration."""

    time_count = min(int(max_frames), int(stack.shape[0]))
    minima = []
    for t in range(time_count):
        image = _registration_image_for_estimation(
            stack[t],
            channel=registration_channel,
            is3d=is3d,
            projection_range=projection_range,
            projection_method=projection_method,
            gSig_filt=gSig_filt,
        )
        minima.append(float(np.nanmin(image)))
    return float(np.nanmin(minima)) if minima else 0.0

def _initial_template(
    stack: np.ndarray,
    *,
    registration_channel: int,
    registration_stack: int,
    is3d: bool,
    projection_range: tuple[int, int] | None,
    projection_method: str,
    template_init_mode: str,
    niter_rig: int,
    max_shifts: np.ndarray | None,
    upsample_factor: int,
    normalization: str | None,
    gSig_filt: tuple[float, float] | None,
    add_to_movie: float,
    shift_interpolation: str,
    transform_mode: str,
    transform_cval: float,
    border_nan,
    block_size: int,
) -> np.ndarray:
    """Create the first template, including a CaImAn-like rigid median mode."""

    if template_init_mode == "registration_stack":
        template = _registration_image(
            stack[int(registration_stack)],
            channel=registration_channel,
            is3d=is3d,
            projection_range=projection_range,
            projection_method=projection_method,
        )
        return _high_pass_filter_space(template, gSig_filt).astype(np.float32, copy=True)

    sample_indices = _frame_sample_indices(stack.shape[0], is3d=is3d)
    sampled_images = _registration_images(
        stack,
        channel=registration_channel,
        is3d=is3d,
        projection_range=projection_range,
        projection_method=projection_method,
        indices=sample_indices,
    )
    template = np.nanmedian(np.stack(sampled_images, axis=0), axis=0).astype(np.float32)
    template = _high_pass_filter_space(template, gSig_filt).astype(np.float32, copy=False)
    if template_init_mode == "median":
        return template

    niter_rig = max(int(niter_rig), 1)
    for _ in range(niter_rig):
        corrected_images = []
        centers, _ = _uniform_shift_grid(template.shape, np.zeros(template.ndim, dtype=np.float32))
        for moving in sampled_images:
            moving_for_estimation = _high_pass_filter_space(moving, gSig_filt)
            shift, _ = _estimate_shift(
                template + float(add_to_movie),
                moving_for_estimation + float(add_to_movie),
                upsample_factor=upsample_factor,
                normalization=normalization,
            )
            shift = _clip_shift(shift, max_shifts=max_shifts)
            _, shift_grid = _uniform_shift_grid(template.shape, shift)
            corrected_images.append(
                _warp_with_shift_grid(
                    moving,
                    centers_by_axis=centers,
                    shift_grid=shift_grid,
                    shift_interpolation=shift_interpolation,
                    order=1,
                    mode=transform_mode,
                    cval=transform_cval,
                    border_nan=border_nan,
                    block_size=block_size,
                )
            )
        template = np.nanmedian(np.stack(corrected_images, axis=0), axis=0).astype(np.float32)
        template = _high_pass_filter_space(template, gSig_filt).astype(np.float32, copy=False)
        sampled_images = corrected_images
    return template

def _apply_shift_grid_to_frame(
    frame_tzcyx: np.ndarray,
    *,
    is3d: bool,
    centers_by_axis: tuple[np.ndarray, ...],
    shift_grid: np.ndarray,
    shift_interpolation: str,
    order: int,
    mode: str,
    cval: float,
    border_nan,
    block_size: int,
) -> np.ndarray:
    """Apply one 2D or 3D shift field to every channel in one time frame."""

    z_count, channel_count, y_count, x_count = frame_tzcyx.shape
    corrected = np.empty_like(frame_tzcyx, dtype=np.float32)

    if is3d:
        for c in range(channel_count):
            corrected[:, c, :, :] = _warp_with_shift_grid(
                frame_tzcyx[:, c, :, :],
                centers_by_axis=centers_by_axis,
                shift_grid=shift_grid,
                shift_interpolation=shift_interpolation,
                order=order,
                mode=mode,
                cval=cval,
                border_nan=border_nan,
                block_size=block_size,
            )
        return corrected

    for z in range(z_count):
        for c in range(channel_count):
            corrected[z, c, :, :] = _warp_with_shift_grid(
                frame_tzcyx[z, c, :, :],
                centers_by_axis=centers_by_axis,
                shift_grid=shift_grid,
                shift_interpolation=shift_interpolation,
                order=order,
                mode=mode,
                cval=cval,
                border_nan=border_nan,
                block_size=block_size,
            )
    return corrected

def _process_frame(
    t: int,
    stack: np.ndarray,
    *,
    template: np.ndarray,
    registration_channel: int,
    is3d: bool,
    projection_range: tuple[int, int] | None,
    projection_method: str,
    patch_grid: _PatchGrid,
    pw_rigid: bool,
    max_shifts: np.ndarray | None,
    max_deviation_rigid: np.ndarray | None,
    upsample_factor: int,
    normalization: str | None,
    gSig_filt: tuple[float, float] | None,
    add_to_movie: float,
    shift_interpolation: str,
    transform_order: int,
    transform_mode: str,
    transform_cval: float,
    border_nan,
    block_size: int,
) -> tuple[int, np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    """Estimate and apply NoRMCorre shifts for one frame."""

    moving = _registration_image(
        stack[t],
        channel=registration_channel,
        is3d=is3d,
        projection_range=projection_range,
        projection_method=projection_method,
    )
    moving_for_estimation = _high_pass_filter_space(moving, gSig_filt)
    template_for_estimation = np.asarray(template, dtype=np.float32) + float(add_to_movie)
    moving_for_estimation = moving_for_estimation + float(add_to_movie)
    rigid_shift, rigid_error = _estimate_shift(
        template_for_estimation,
        moving_for_estimation,
        upsample_factor=upsample_factor,
        normalization=normalization,
    )
    rigid_shift = _clip_shift(rigid_shift, max_shifts=max_shifts)

    if pw_rigid:
        patch_shifts, patch_errors = _estimate_patch_shifts(
            template_for_estimation,
            moving_for_estimation,
            patch_grid=patch_grid,
            rigid_shift=rigid_shift,
            max_shifts=max_shifts,
            max_deviation_rigid=max_deviation_rigid,
            upsample_factor=upsample_factor,
            normalization=normalization,
        )
    else:
        patch_shifts = np.zeros((*patch_grid.grid_shape, template.ndim), dtype=np.float32)
        patch_shifts[...] = rigid_shift
        patch_errors = np.full(patch_grid.grid_shape, rigid_error, dtype=np.float32)

    corrected_frame = _apply_shift_grid_to_frame(
        stack[t],
        is3d=is3d,
        centers_by_axis=patch_grid.centers_by_axis,
        shift_grid=patch_shifts,
        shift_interpolation=shift_interpolation,
        order=transform_order,
        mode=transform_mode,
        cval=transform_cval,
        border_nan=border_nan,
        block_size=block_size,
    )
    return t, corrected_frame, rigid_shift, rigid_error, patch_shifts, patch_errors

def register_stack_normcorre(
    stack,
    *,
    registration_channel: int = 0,
    registration_stack: int = 0,
    is3d: bool | None = None,
    projection_range: tuple[int, int] | Sequence[int] | None = None,
    projection_method: str = "max",
    pw_rigid: bool = True,
    strides: tuple[int, ...] | int | None = None,
    overlaps: tuple[int, ...] | int | None = None,
    max_shifts: tuple[float, ...] | float | None = None,
    max_deviation_rigid: tuple[float, ...] | float | None = None,
    n_iterations: int = 1,
    correction_iterations: int = 1,
    niter_rig: int = 1,
    template_init_mode: str = "registration_stack",
    template_update_method: str = "caiman",
    splits: int = 56,
    upsample_factor: int = 10,
    normalization: str | None = None,
    gSig_filt=None,
    add_to_movie: float | None = None,
    nonneg_movie: bool = True,
    shift_interpolation: str = "resize",
    n_jobs: int = 1,
    transform_order: int = 1,
    transform_mode: str = "constant",
    transform_cval: float = 0.0,
    border_nan=None,
    block_size: int = 32,
    output_use_memmap: bool = False,
    output_memmap_folder: str | None = None,
    output_memmap_name: str | None = "zenreg_normcorre_registered",
    verbose: bool = True,
    return_details: bool = True,
):
    """
    Run NoRMCorre-style piecewise-rigid registration on a ``TZCYX`` stack.

    Parameters
    ----------
    stack : array-like
        Input image in canonical OMIO/OME axis order ``TZCYX``.
    registration_channel : int, optional
        Channel used for motion estimation. The detected correction is applied
        to all channels.
    registration_stack : int, optional
        Time point used as the initial template. Default is ``0``.
    is3d : bool or None, optional
        If True, estimate local shifts on full ``ZYX`` volumes and return Z/Y/X
        correction fields. If False, estimate shifts on 2D ``YX`` frames. If
        None, ZenReg uses 3D mode when ``SizeZ > 1`` and 2D mode otherwise.
    projection_range : tuple[int, int] or None, optional
        Optional half-open Z range ``(start, stop)`` used for the NoRMCorre
        registration template/shift estimation. The detected correction is still
        applied to the full stack.
    projection_method : {"max", "mean", "median", "var", "std"}, optional
        Z-projection method used only when ``is3d=False`` and the input has more
        than one Z slice. ``"max"`` is a good default for sparse spots or
        puncta. ``"mean"`` is often better for dense, spatially extended signal.
        ``"median"`` is robust to outliers, but can attenuate sparse spots.
        ``"std"`` and ``"var"`` can be useful when contrast-rich structure
        matters more than absolute intensity.
    pw_rigid : bool, optional
        If True, estimate patch-wise local translations after the initial rigid
        shift. If False, only the rigid shift is used.
    strides, overlaps : tuple, int, or None, optional
        NoRMCorre patch-grid parameters. The effective patch size is
        ``strides + overlaps``, following CaImAn/NoRMCorre terminology. 2D
        defaults are ``strides=(48, 48)``, ``overlaps=(24, 24)``. 3D defaults are
        ``strides=(6, 48, 48)``, ``overlaps=(3, 24, 24)``.
    max_shifts : tuple, scalar, or None, optional
        Optional absolute correction-shift limits in ``YX`` order for 2D mode
        and ``ZYX`` order for 3D mode.
    max_deviation_rigid : tuple, scalar, or None, optional
        Optional limit for how far each local patch shift may deviate from the
        initial rigid correction. This mirrors CaImAn's ``max_deviation_rigid``
        idea and helps prevent single low-information patches from drifting.
    n_iterations : int, optional
        Number of template-refinement passes. Each pass registers all original
        frames of the current correction stage to the current template and
        updates the template from the mean of the corrected registration
        channel. ``1`` is a conservative default.
    correction_iterations : int, optional
        Number of outer correction stages. After each stage, the already
        corrected stack becomes the input for the next stage. This can help when
        the first pass removes large/global motion and exposes smaller residual
        local motion. It does not add a true rotation model; rotations are still
        approximated by local translation fields.
    niter_rig : int, optional
        Number of rigid initialization passes when
        ``template_init_mode="rigid_median"``.
    template_init_mode : {"registration_stack", "median", "rigid_median"}, optional
        Initial template strategy. ``"registration_stack"`` keeps ZenReg's
        explicit reference-frame semantics. ``"median"`` uses the median of a
        sparse CaImAn-like frame sample. ``"rigid_median"`` first rigid-aligns
        that sample before taking the median, matching CaImAn's NoRMCorre
        initialization idea.
    template_update_method : {"caiman", "mean", "median"}, optional
        Template update strategy after each template pass. ``"caiman"`` computes
        means for time chunks and then takes a median across chunk templates,
        mirroring CaImAn's batch update. ``"mean"`` is the earlier ZenReg
        behavior.
    splits : int, optional
        Number of time chunks used by ``template_update_method="caiman"``.
    upsample_factor : int, optional
        Subpixel precision used by ``skimage.registration.phase_cross_correlation``.
    normalization : {"phase", None}, optional
        Normalization passed to ``phase_cross_correlation``. ``None`` is often
        robust for microscopy intensity data; ``"phase"`` can be useful for
        sharper, high-contrast structures.
    gSig_filt : tuple, scalar, or None, optional
        CaImAn-compatible spatial high-pass filter for motion estimation. This
        is mainly useful for one-photon or high-background calcium imaging,
        where landmarks become clearer after suppressing broad background. The
        filter is applied in XY for shift estimation/templates only; the returned
        registered stack keeps the original image intensities.
    add_to_movie : float or None, optional
        CaImAn-style additive offset used during shift estimation. If None, it
        is estimated as ``-min_mov`` from the first registration frames.
    nonneg_movie : bool, optional
        Stored for CaImAn parameter parity. ZenReg returns the registered image
        in the original intensity scale; the additive offset is not baked into
        the returned stack.
    shift_interpolation : {"resize", "linear"}, optional
        How patch shifts are upsampled to dense displacement fields.
        ``"resize"`` follows CaImAn's default remap path using cubic image
        resizing of the patch-shift grid. ``"linear"`` uses coordinate-aware
        interpolation from patch centers to image coordinates.
    n_jobs : int, optional
        Number of worker threads over time frames. This parallelizes the natural
        NoRMCorre batch dimension while keeping each frame processed slice-wise,
        which plays well with OMIO/Zarr-backed inputs.
    transform_order : int, optional
        Interpolation order for applying the dense displacement field. Use
        ``1`` for most intensity images. Use ``0`` for sparse puncta, labels, or
        cases where preserving hard edges is more important than smooth
        interpolation. Use ``3`` for the closest match to CaImAn's default
        cubic OpenCV/skimage remapping.
    transform_mode, transform_cval : optional
        Boundary handling forwarded to ``scipy.ndimage.map_coordinates``.
    border_nan : {None, False, True, "min", "copy"}, optional
        CaImAn-style border handling. If None, ``transform_mode`` and
        ``transform_cval`` are used. ``"copy"`` replicates edge values.
    block_size : int, optional
        Number of Y rows in 2D mode, or Z slices in 3D mode, processed per warp
        block. Lower values reduce temporary memory use.
    output_use_memmap : bool, optional
        If True, create the registered output as an OMIO disk-backed Zarr array
        instead of a RAM-backed NumPy array. This is useful for large 20-40 GB
        lab stacks where the input may already be OMIO/Zarr-backed.
    output_memmap_folder : str or None, optional
        Optional folder passed to OMIO as ``zarr_store_path`` for the registered
        output cache.
    output_memmap_name : str or None, optional
        Optional Zarr store name for the registered output.
    verbose : bool, optional
        Print progress messages.
    return_details : bool, optional
        If True, return ``(registered, details)``. Otherwise return only the
        registered stack.

    Returns
    -------
    numpy.ndarray or tuple[numpy.ndarray, dict]
        Registered ``TZCYX`` stack, optionally with shifts, patch fields, errors,
        and settings useful for reporting/reproducibility.
    """

    stack = _as_float32_stack(stack)
    if stack.ndim != 5:
        raise ValueError(f"Expected a 5D {CANONICAL_AXIS_ORDER} stack. Got shape {stack.shape!r}.")

    time_count, z_count, channel_count, y_count, x_count = stack.shape
    registration_channel = int(registration_channel)
    registration_stack = int(registration_stack)
    if not 0 <= registration_channel < channel_count:
        raise ValueError(f"registration_channel must be between 0 and {channel_count - 1}.")
    if not 0 <= registration_stack < time_count:
        raise ValueError(f"registration_stack must be between 0 and {time_count - 1}.")

    is3d = bool(z_count > 1) if is3d is None else bool(is3d)
    ndim = 3 if is3d else 2
    if is3d and z_count < 2:
        raise ValueError("is3d=True requires SizeZ >= 2.")
    projection_range = None if projection_range is None else normalize_zrange(projection_range, z_count, strict=True)
    projection_method = _normalize_projection_method(projection_method)

    default_strides = (6, 48, 48) if is3d else (48, 48)
    default_overlaps = (3, 24, 24) if is3d else (24, 24)
    strides = _normalize_spatial_tuple(strides, ndim=ndim, default=default_strides, name="strides")
    overlaps = _normalize_spatial_tuple(overlaps, ndim=ndim, default=default_overlaps, name="overlaps")
    max_shifts_array = _normalize_max_shifts(max_shifts, ndim=ndim)
    max_deviation_array = _normalize_max_deviation(max_deviation_rigid, ndim=ndim)
    gSig_filt = _normalize_gsig_filt(gSig_filt, ndim=ndim)

    n_iterations = int(n_iterations)
    if n_iterations < 1:
        raise ValueError(f"n_iterations must be >= 1. Got {n_iterations!r}.")
    correction_iterations = int(correction_iterations)
    if correction_iterations < 1:
        raise ValueError(f"correction_iterations must be >= 1. Got {correction_iterations!r}.")
    niter_rig = int(niter_rig)
    if niter_rig < 1:
        raise ValueError(f"niter_rig must be >= 1. Got {niter_rig!r}.")
    template_init_mode = _normalize_choice(
        template_init_mode,
        allowed={"registration_stack", "median", "rigid_median"},
        name="template_init_mode",
    )
    template_update_method = _normalize_choice(
        template_update_method,
        allowed={"caiman", "mean", "median"},
        name="template_update_method",
    )
    shift_interpolation = _normalize_choice(
        shift_interpolation,
        allowed={"resize", "linear"},
        name="shift_interpolation",
    )
    splits = max(int(splits), 1)
    upsample_factor = int(upsample_factor)
    if upsample_factor < 1:
        raise ValueError(f"upsample_factor must be >= 1. Got {upsample_factor!r}.")
    n_jobs = max(int(n_jobs), 1)
    transform_order = int(transform_order)
    if not 0 <= transform_order <= 5:
        raise ValueError(f"transform_order must be between 0 and 5. Got {transform_order!r}.")

    min_mov = _estimate_min_mov(
        stack,
        registration_channel=registration_channel,
        is3d=is3d,
        projection_range=projection_range,
        projection_method=projection_method,
        gSig_filt=gSig_filt,
    )
    if add_to_movie is None:
        add_to_movie = -float(min_mov)
    else:
        add_to_movie = float(add_to_movie)

    spatial_shape = (z_count, y_count, x_count) if is3d else (y_count, x_count)
    patch_grid = _build_patch_grid(spatial_shape, strides=strides, overlaps=overlaps)

    def _create_registered_output(*, correction_index: int):
        if not output_use_memmap:
            return np.empty_like(stack, dtype=np.float32)

        from .io import create_empty_stack

        memmap_name = output_memmap_name
        is_final_correction = correction_index == correction_iterations - 1
        if correction_iterations > 1 and not is_final_correction and memmap_name is not None:
            memmap_name = f"{memmap_name}_correction_{correction_index + 1}"
        return create_empty_stack(
            shape=tuple(int(v) for v in stack.shape),
            dtype=np.float32,
            fill_value=0,
            use_memmap=True,
            memmap_folder=output_memmap_folder,
            memmap_name=memmap_name,
            verbose=False,
        )

    working_stack = stack
    registered = None
    cumulative_rigid_shifts = np.zeros((time_count, ndim), dtype=np.float32)
    rigid_shifts = np.zeros((time_count, ndim), dtype=np.float32)
    rigid_errors = np.zeros(time_count, dtype=np.float32)
    patch_shifts_all = np.zeros((time_count, *patch_grid.grid_shape, ndim), dtype=np.float32)
    patch_errors_all = np.zeros((time_count, *patch_grid.grid_shape), dtype=np.float32)
    rigid_shifts_by_correction: list[np.ndarray] = []
    rigid_errors_by_correction: list[np.ndarray] = []
    patch_shifts_by_correction: list[np.ndarray] = []
    patch_errors_by_correction: list[np.ndarray] = []

    for correction_index in range(correction_iterations):
        template = _initial_template(
            working_stack,
            registration_channel=registration_channel,
            registration_stack=registration_stack,
            is3d=is3d,
            projection_range=projection_range,
            projection_method=projection_method,
            template_init_mode=template_init_mode if correction_index == 0 else "registration_stack",
            niter_rig=niter_rig,
            max_shifts=max_shifts_array,
            upsample_factor=upsample_factor,
            normalization=normalization,
            gSig_filt=gSig_filt,
            add_to_movie=add_to_movie,
            shift_interpolation=shift_interpolation,
            transform_mode=transform_mode,
            transform_cval=transform_cval,
            border_nan=border_nan,
            block_size=block_size,
        )
        registered = _create_registered_output(correction_index=correction_index)

        for iteration in range(n_iterations):
            if verbose:
                mode_label = "3D+t" if is3d else "2D+t"
                print(
                    f"ZenReg NoRMCorre {mode_label}: correction "
                    f"{correction_index + 1}/{correction_iterations}, "
                    f"template pass {iteration + 1}/{n_iterations}, "
                    f"{len(patch_grid.slices)} patches, n_jobs={n_jobs}",
                    flush=True,
                )

            worker_kwargs = {
                "template": template,
                "registration_channel": registration_channel,
                "is3d": is3d,
                "projection_range": projection_range,
                "projection_method": projection_method,
                "patch_grid": patch_grid,
                "pw_rigid": bool(pw_rigid),
                "max_shifts": max_shifts_array,
                "max_deviation_rigid": max_deviation_array,
                "upsample_factor": upsample_factor,
                "normalization": normalization,
                "gSig_filt": gSig_filt,
                "add_to_movie": add_to_movie,
                "shift_interpolation": shift_interpolation,
                "transform_order": transform_order,
                "transform_mode": transform_mode,
                "transform_cval": transform_cval,
                "border_nan": border_nan,
                "block_size": block_size,
            }

            if n_jobs == 1:
                results = [_process_frame(t, working_stack, **worker_kwargs) for t in range(time_count)]
            else:
                with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                    futures = [
                        executor.submit(_process_frame, t, working_stack, **worker_kwargs)
                        for t in range(time_count)
                    ]
                    results = [future.result() for future in futures]

            for t, corrected_frame, rigid_shift, rigid_error, patch_shifts, patch_errors in results:
                registered[t] = corrected_frame
                rigid_shifts[t] = rigid_shift
                rigid_errors[t] = rigid_error
                patch_shifts_all[t] = patch_shifts
                patch_errors_all[t] = patch_errors

            if iteration < n_iterations - 1:
                template = _update_template_from_registered(
                    registered,
                    registration_channel=registration_channel,
                    is3d=is3d,
                    projection_range=projection_range,
                    projection_method=projection_method,
                    gSig_filt=gSig_filt,
                    template_update_method=template_update_method,
                    splits=splits,
                )

        cumulative_rigid_shifts += rigid_shifts
        rigid_shifts_by_correction.append(rigid_shifts.copy())
        rigid_errors_by_correction.append(rigid_errors.copy())
        patch_shifts_by_correction.append(patch_shifts_all.copy())
        patch_errors_by_correction.append(patch_errors_all.copy())

        if correction_index < correction_iterations - 1:
            working_stack = registered

    time_shifts_zyx = np.zeros((time_count, 3), dtype=np.float32)
    if is3d:
        time_shifts_zyx[:, :] = cumulative_rigid_shifts
        time_shifts_yx = cumulative_rigid_shifts[:, 1:].copy()
    else:
        time_shifts_zyx[:, 1:] = cumulative_rigid_shifts
        time_shifts_yx = cumulative_rigid_shifts.copy()

    details: dict[str, Any] = {
        "method": "normcorre",
        "normcorre_variant": "pw_rigid" if pw_rigid else "rigid",
        "registration_channel": registration_channel,
        "registration_stack": registration_stack,
        "time_registration_mode": "full_3d" if is3d else "projection",
        "effective_time_registration_mode": "full_3d" if is3d else "projection",
        "time_reference_mode": "template",
        "intra_stack": False,
        "zreg": bool(is3d),
        "rotreg": False,
        "projection_range": projection_range,
        "projection_method": projection_method,
        "filter_slices": False,
        "filter_projections": False,
        "median_kernel_size": None,
        "strides": strides,
        "overlaps": overlaps,
        "patch_grid_shape": patch_grid.grid_shape,
        "patch_centers_by_axis": [axis.tolist() for axis in patch_grid.centers_by_axis],
        "max_shifts": None if max_shifts_array is None else max_shifts_array.tolist(),
        "max_deviation_rigid": None if max_deviation_array is None else max_deviation_array.tolist(),
        "upsample_factor": upsample_factor,
        "phase_cross_correlation_normalization": normalization,
        "gSig_filt": None if gSig_filt is None else tuple(float(v) for v in gSig_filt),
        "min_mov": float(min_mov),
        "add_to_movie": float(add_to_movie),
        "nonneg_movie": bool(nonneg_movie),
        "n_iterations": n_iterations,
        "correction_iterations": correction_iterations,
        "niter_rig": niter_rig,
        "template_init_mode": template_init_mode,
        "template_update_method": template_update_method,
        "splits": splits,
        "shift_interpolation": shift_interpolation,
        "n_jobs": n_jobs,
        "output_use_memmap": bool(output_use_memmap),
        "output_memmap_folder": output_memmap_folder,
        "output_memmap_name": output_memmap_name if output_use_memmap else None,
        "transform_backend": "scipy_map_coordinates",
        "transform_order": transform_order,
        "transform_mode": transform_mode,
        "transform_cval": transform_cval,
        "border_nan": border_nan,
        "zero_clip": False,
        "zero_clip_mode": "none",
        "zero_clip_bounds": None,
        "stack_shape_tzcyx": tuple(int(v) for v in stack.shape),
        "time_shifts_zyx": time_shifts_zyx,
        "time_shifts_yx": time_shifts_yx,
        "time_shifts_zyx_raw": time_shifts_zyx.copy(),
        "time_shifts_yx_raw": time_shifts_yx.copy(),
        "rigid_shifts": cumulative_rigid_shifts,
        "rigid_errors": rigid_errors,
        "last_correction_rigid_shifts": rigid_shifts,
        "last_correction_rigid_errors": rigid_errors,
        "rigid_shifts_by_correction": np.stack(rigid_shifts_by_correction, axis=0),
        "rigid_errors_by_correction": np.stack(rigid_errors_by_correction, axis=0),
        "patch_shifts": patch_shifts_all,
        "patch_errors": patch_errors_all,
        "patch_shifts_by_correction": np.stack(patch_shifts_by_correction, axis=0),
        "patch_errors_by_correction": np.stack(patch_errors_by_correction, axis=0),
    }

    return (registered, details) if return_details else registered
# %% END
