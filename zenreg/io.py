"""
OMIO-backed stack I/O helpers for ZenReg.

Author: Fabrizio Musacchio
Date: June 2026
"""
# %% IMPORTS
from __future__ import annotations

import os
import tempfile
import warnings
from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np

from ._axes import CANONICAL_AXIS_ORDER, ensure_tzcyx_stack
from .reporting import write_registration_outputs
# %% HELPER FUNCTIONS

def _configure_omio_runtime() -> None:
    """Point optional OMIO/Napari caches at writable locations when needed."""

    cache_root = Path(
        os.environ.get(
            "ZENREG_OMIO_CACHE_DIR",
            Path(tempfile.gettempdir()) / "zenreg-omio-cache",
        )
    )
    cache_root.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache_root / "numba"))
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg-cache"))

    home = Path.home()
    if not os.access(home, os.W_OK):
        omio_home = cache_root / "home"
        omio_home.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HOME", str(omio_home))

def _import_omio():
    """Import OMIO lazily so array-only ZenReg workflows do not pay I/O import cost."""

    _configure_omio_runtime()
    try:
        import omio as om
    except ImportError as exc:
        raise ImportError(
            "ZenReg I/O requires OMIO. Install the 'omio-microscopy' dependency "
            "or use an environment that provides `import omio as om`."
        ) from exc
    return om

def _copy_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    """Return a detached metadata dictionary."""

    return deepcopy(metadata) if isinstance(metadata, dict) else {}

def create_stack_metadata(
    stack,
    *,
    input_metadata: dict[str, Any] | None = None,
    annotations: dict[str, Any] | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Create OMIO metadata for a canonical ``TZCYX`` ZenReg stack.

    Parameters
    ----------
    stack : array-like
        Image stack in canonical ``TZCYX`` order.
    input_metadata : dict or None, optional
        Existing OMIO metadata whose physical sizes, timing, and annotations are
        inherited.
    annotations : dict or None, optional
        Extra annotations merged into the output metadata.
    verbose : bool, optional
        If True, let OMIO print metadata normalization diagnostics.

    Returns
    -------
    dict
        OMIO metadata synchronized to the stack shape and canonical axes.
    """

    stack = ensure_tzcyx_stack(stack)
    om = _import_omio()
    metadata = om.create_empty_metadata(
        shape=tuple(int(v) for v in stack.shape),
        input_metadata=_copy_metadata(input_metadata),
        annotations=annotations,
        verbose=verbose,
    )
    return om.update_metadata_from_image(metadata, stack, verbose=verbose)

def update_stack_metadata(
    metadata: dict[str, Any] | None,
    stack,
    *,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Update inherited OMIO metadata from a modified canonical ``TZCYX`` stack.

    Use this after operations that change image shape, for example cropping or
    future zero-clipping steps. Physical pixel sizes, timing fields, and existing
    annotations are inherited from ``metadata``; shape, ``Size*`` fields, and
    axes are synchronized from ``stack`` via ``om.update_metadata_from_image``.

    Parameters
    ----------
    metadata : dict or None
        Existing OMIO metadata to update.
    stack : array-like
        Modified image stack in canonical ``TZCYX`` order.
    verbose : bool, optional
        If True, let OMIO print metadata normalization diagnostics.

    Returns
    -------
    dict
        Updated OMIO metadata for ``stack``.
    """

    stack = ensure_tzcyx_stack(stack)
    om = _import_omio()
    return om.update_metadata_from_image(
        _copy_metadata(metadata),
        stack,
        verbose=verbose,
    )

def _normalize_crop_value(crop: dict[str, int], key: str) -> int:
    """Return one non-negative integer crop value from a sparse crop dictionary."""

    value = int(crop.get(key, 0))
    if value < 0:
        raise ValueError(f"crop[{key!r}] must be >= 0. Got {value!r}.")
    return value

def crop_stack(
    stack,
    metadata: dict[str, Any] | None,
    crop: dict[str, int],
    *,
    verbose: bool = False,
):
    """
    Post-hoc crop a canonical ``TZCYX`` stack and update OMIO metadata.

    Parameters
    ----------
    stack : array-like
        Image stack in canonical ``TZCYX`` order.
    metadata : dict or None
        OMIO metadata inherited from the input or previous processing step.
    crop : dict
        Sparse crop dictionary. Supported keys are ``top`` and ``bottom`` for
        the Z axis, ``up`` and ``down`` for the Y axis, and ``left`` and
        ``right`` for the X axis. Omitted keys default to zero. For effectively
        2D or 2D+t stacks with ``SizeZ == 1``, requested Z cropping is ignored
        and a warning is emitted.
    verbose : bool, optional
        If True, let OMIO print metadata update diagnostics.

    Returns
    -------
    tuple[numpy.ndarray, dict]
        Cropped stack and OMIO metadata synchronized to the cropped shape.
    """

    if not isinstance(crop, dict):
        raise TypeError("crop must be a dictionary with optional top/bottom/left/right/up/down entries.")
    allowed = {"top", "bottom", "left", "right", "up", "down"}
    unknown = sorted(set(crop) - allowed)
    if unknown:
        raise ValueError(f"Unsupported crop keys: {unknown}. Supported keys: {sorted(allowed)}.")

    array = ensure_tzcyx_stack(stack)
    top = _normalize_crop_value(crop, "top")
    bottom = _normalize_crop_value(crop, "bottom")
    left = _normalize_crop_value(crop, "left")
    right = _normalize_crop_value(crop, "right")
    up = _normalize_crop_value(crop, "up")
    down = _normalize_crop_value(crop, "down")

    if array.shape[1] <= 1 and (top > 0 or bottom > 0):
        warnings.warn(
            "Requested top/bottom Z cropping for an effectively 2D stack "
            f"with SizeZ={array.shape[1]}. Ignoring Z crop values.",
            RuntimeWarning,
            stacklevel=2,
        )
        top = 0
        bottom = 0

    z_stop = array.shape[1] - bottom
    y_stop = array.shape[3] - down
    x_stop = array.shape[4] - right
    if top >= z_stop:
        raise ValueError(f"Z crop would remove the complete stack. SizeZ={array.shape[1]}, top={top}, bottom={bottom}.")
    if up >= y_stop:
        raise ValueError(f"Y crop would remove the complete stack. SizeY={array.shape[3]}, up={up}, down={down}.")
    if left >= x_stop:
        raise ValueError(f"X crop would remove the complete stack. SizeX={array.shape[4]}, left={left}, right={right}.")

    cropped = array[:, top:z_stop, :, up:y_stop, left:x_stop].copy()
    cropped_metadata = update_stack_metadata(metadata, cropped, verbose=verbose)
    annotations = cropped_metadata.setdefault("Annotations", {})
    if isinstance(annotations, dict):
        annotations["ZenReg_PosthocCrop"] = {
            "top": int(top),
            "bottom": int(bottom),
            "left": int(left),
            "right": int(right),
            "up": int(up),
            "down": int(down),
        }
    return cropped, cropped_metadata

def create_empty_stack(
    shape: tuple[int, int, int, int, int] = (1, 1, 1, 1, 1),
    *,
    dtype=np.uint16,
    fill_value=0,
    use_memmap: bool = False,
    memmap_folder: str | Path | None = None,
    memmap_name: str | None = None,
    return_metadata: bool = False,
    input_metadata: dict[str, Any] | None = None,
    verbose: bool = False,
):
    """
    Create an empty canonical ``TZCYX`` stack via OMIO.

    This is a thin ZenReg wrapper around ``om.create_empty_image``. OMIO creates
    the array and, optionally, a matching OME/OMIO metadata dictionary.

    Parameters
    ----------
    use_memmap : bool, optional
        If True, create a disk-backed Zarr array via ``zarr_store="disk"``.
    memmap_folder : str, pathlib.Path, or None, optional
        Optional folder forwarded to OMIO as ``zarr_store_path``. This is useful
        when temporary data should live on local scratch storage instead of next
        to a large input file.
    memmap_name : str or None, optional
        Optional Zarr store name forwarded to OMIO as ``zarr_store_name``.
    """

    if not use_memmap and memmap_folder is not None:
        raise ValueError("memmap_folder requires use_memmap=True.")
    if not use_memmap and memmap_name is not None:
        raise ValueError("memmap_name requires use_memmap=True.")

    om = _import_omio()
    return om.create_empty_image(
        shape=shape,
        dtype=dtype,
        fill_value=fill_value,
        zarr_store="disk" if use_memmap else None,
        zarr_store_path=None if memmap_folder is None else str(memmap_folder),
        zarr_store_name=memmap_name,
        return_metadata=return_metadata,
        input_metadata=input_metadata,
        verbose=verbose,
    )

def load_stack(
    path: str | Path,
    *,
    return_metadata: bool = False,
    use_memmap: bool = False,
    memmap_folder: str | Path | None = None,
    memmap_reuse: bool = True,
    on_error: str = "raise",
    **imread_kwargs,
):
    """
    Load a microscopy stack with OMIO.

    OMIO supports TIFF/OME-TIFF, CZI, LSM, and Thorlabs RAW inputs and normalizes
    returned images to canonical OME axis order ``TZCYX``. ZenReg therefore always
    receives a 5D stack, even when an input file stores singleton ``T``, ``Z``, or
    ``C`` dimensions implicitly.

    Parameters
    ----------
    path : str or pathlib.Path
        Input file, folder, or OMIO-supported source.
    return_metadata : bool, optional
        If True, return ``(stack, metadata)``. If False, return only ``stack``.
    use_memmap : bool, optional
        If True, ask OMIO to read through a disk-backed Zarr store by forwarding
        ``zarr_store="disk"``. This can reduce memory pressure and avoid repeated
        reads from network/server storage.
    memmap_folder : str, pathlib.Path, or None, optional
        Optional folder forwarded to OMIO as ``zarr_store_path``. Use a local
        scratch folder for large files stored on remote/network volumes.
    memmap_reuse : bool, optional
        If True with ``use_memmap=True``, forward ``reuse_disk_cache=True`` to
        OMIO so an existing validated ``.omio_cache`` Zarr store is reused. If
        no cache exists, OMIO builds it. If False, OMIO rebuilds the disk cache.
    on_error : {"raise", "return_none"}, optional
        Error handling mode forwarded to ``om.imread``. ``"raise"`` is the
        default and lets unreadable files fail loudly, which is safest for
        interactive work. ``"return_none"`` lets OMIO return ``(None, None)``
        for unrecoverable metadata problems, allowing batch workflows to skip
        that file deliberately.
    **imread_kwargs
        Extra keyword arguments forwarded to ``om.imread``.

    Returns
    -------
    numpy.ndarray, None, or tuple[numpy.ndarray | None, dict | None]
        Loaded canonical ``TZCYX`` stack, optionally with OMIO metadata. If
        ``on_error="return_none"`` and OMIO cannot read the file, returns
        ``None`` or ``(None, None)`` depending on ``return_metadata``.
    """

    on_error = str(on_error).strip().lower()
    if on_error not in {"raise", "return_none"}:
        raise ValueError("on_error must be 'raise' or 'return_none'.")
    if not use_memmap and memmap_folder is not None:
        raise ValueError("memmap_folder requires use_memmap=True.")
    if use_memmap:
        zarr_store = imread_kwargs.get("zarr_store")
        if zarr_store not in (None, "disk"):
            raise ValueError("use_memmap=True requires zarr_store=None or zarr_store='disk'.")
        imread_kwargs["zarr_store"] = "disk"
        imread_kwargs["reuse_disk_cache"] = bool(memmap_reuse)
        if memmap_folder is not None:
            imread_kwargs["zarr_store_path"] = str(memmap_folder)

    om = _import_omio()
    imread_kwargs["on_error"] = on_error
    stack, metadata = om.imread(path, **imread_kwargs)
    if stack is None or metadata is None:
        if on_error != "return_none":
            raise ValueError(f"OMIO returned None for {path!s} although on_error='raise'.")
        return (None, None) if return_metadata else None
    stack = ensure_tzcyx_stack(stack)
    if metadata.get("axes") != CANONICAL_AXIS_ORDER:
        raise ValueError(
            f"OMIO returned axes={metadata.get('axes')!r}, expected {CANONICAL_AXIS_ORDER!r}."
        )
    return (stack, metadata) if return_metadata else stack

def cleanup_omio_cache(
    target: str | Path,
    *,
    full_cleanup: bool = True,
    verbose: bool = False,
) -> None:
    """
    Clean OMIO disk-backed Zarr cache folders.

    Pass the original input filename when OMIO used its default ``.omio_cache``
    location next to the file, or pass the custom ``memmap_folder`` when
    ``load_stack(..., use_memmap=True, memmap_folder=...)`` was used. For large
    workflows it is usually sensible to call this once before loading, for a
    fresh start, and once after ``save_stack`` has written the registered result.
    """

    if not Path(target).exists():
        return
    cleanup = _import_omio().cleanup_omio_cache
    if verbose:
        cleanup(str(target), full_cleanup=full_cleanup, verbose=verbose)
        return
    with redirect_stdout(StringIO()):
        cleanup(str(target), full_cleanup=full_cleanup, verbose=verbose)

def _metadata_for_output_path(
    path: Path,
    stack: np.ndarray,
    metadata: dict[str, Any] | None,
    *,
    verbose: bool,
) -> dict[str, Any]:
    """Prepare inherited OMIO metadata for writing one registered output stack."""

    annotations = {
        "ZenReg_Axes": CANONICAL_AXIS_ORDER,
        "ZenReg_RegisteredOutput": True,
        "original_filename": path.name,
        "original_filetype": path.suffix.lower().lstrip("."),
        "original_parentfolder": str(path.parent),
    }
    output_metadata = create_stack_metadata(
        stack,
        input_metadata=metadata,
        annotations=annotations,
        verbose=verbose,
    )
    output_metadata["axes"] = CANONICAL_AXIS_ORDER
    output_metadata = update_stack_metadata(
        output_metadata,
        stack,
        verbose=verbose,
    )
    return output_metadata

def save_stack(
    path: str | Path,
    stack,
    *,
    metadata: dict[str, Any] | None = None,
    registration_details: dict[str, Any] | np.ndarray | None = None,
    report_prefix: str | Path | None = None,
    dtype: str | np.dtype | None = None,
    compression_level: int = 3,
    overwrite: bool = True,
    verbose: bool = False,
) -> Path:
    """
    Save a canonical ``TZCYX`` stack with OMIO as OME-TIFF.

    Parameters
    ----------
    path : str or pathlib.Path
        Output anchor path. OMIO writes OME-TIFF; the returned path is the actual
        written ``.ome.tif`` filename.
    stack : array-like
        Image data in canonical ``TZCYX`` order.
    metadata : dict or None, optional
        OMIO metadata to inherit from the input image. Size and axis fields are
        updated to match ``stack`` before writing.
    registration_details : dict, array-like, or None, optional
        If provided, write ZenReg report sidecars next to the registered image:
        ``*_registration_shifts.csv`` with detected shifts and Pearson
        correlations, ``*_registration_settings.yaml`` with reproducibility
        settings, and ``*_registration_summary.png`` with shift/correlation
        plots.
    report_prefix : str, pathlib.Path, or None, optional
        Optional prefix for report sidecars. If None, the registered image path
        is used as prefix.
    dtype : str, numpy dtype, or None, optional
        Optional output dtype conversion. If ``None``, the input dtype is kept.
    compression_level : int, optional
        zlib compression level forwarded to ``om.imwrite``.
    overwrite : bool, optional
        If True, allow replacing an existing OME-TIFF output.
    verbose : bool, optional
        If True, let OMIO print write diagnostics.

    Returns
    -------
    pathlib.Path
        Actual OME-TIFF path written by OMIO.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = ensure_tzcyx_stack(stack)
    if dtype is not None:
        try:
            array = array.astype(dtype, copy=False)
        except TypeError:
            array = array.astype(dtype)

    output_metadata = _metadata_for_output_path(path, array, metadata, verbose=verbose)
    written_paths = _import_omio().imwrite(
        str(path),
        array,
        output_metadata,
        compression_level=compression_level,
        overwrite=overwrite,
        return_fnames=True,
        verbose=verbose,
    )
    if not written_paths:
        raise RuntimeError(f"OMIO did not report a written path for {path!s}.")
    output_path = Path(written_paths[0])
    if registration_details is not None:
        write_registration_outputs(
            output_path,
            array,
            registration_details,
            report_prefix=report_prefix,
        )
    return output_path
# %% END
