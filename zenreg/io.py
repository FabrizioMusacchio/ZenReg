"""
OMIO-backed stack I/O helpers for ZenReg.

Author: Fabrizio Musacchio
Date: June 2026
"""

from __future__ import annotations

import os
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from ._axes import CANONICAL_AXIS_ORDER, ensure_tzcyx_stack
from .reporting import write_registration_outputs


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


def create_empty_stack(
    shape: tuple[int, int, int, int, int] = (1, 1, 1, 1, 1),
    *,
    dtype=np.uint16,
    fill_value=0,
    return_metadata: bool = False,
    input_metadata: dict[str, Any] | None = None,
    verbose: bool = False,
):
    """
    Create an empty canonical ``TZCYX`` stack via OMIO.

    This is a thin ZenReg wrapper around ``om.create_empty_image``. OMIO creates
    the array and, optionally, a matching OME/OMIO metadata dictionary.
    """

    om = _import_omio()
    return om.create_empty_image(
        shape=shape,
        dtype=dtype,
        fill_value=fill_value,
        return_metadata=return_metadata,
        input_metadata=input_metadata,
        verbose=verbose,
    )


def load_stack(path: str | Path, *, return_metadata: bool = False, **imread_kwargs):
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
    **imread_kwargs
        Extra keyword arguments forwarded to ``om.imread``.

    Returns
    -------
    numpy.ndarray or tuple[numpy.ndarray, dict]
        Loaded canonical ``TZCYX`` stack, optionally with OMIO metadata.
    """

    om = _import_omio()
    stack, metadata = om.imread(path, **imread_kwargs)
    stack = ensure_tzcyx_stack(stack)
    if metadata.get("axes") != CANONICAL_AXIS_ORDER:
        raise ValueError(
            f"OMIO returned axes={metadata.get('axes')!r}, expected {CANONICAL_AXIS_ORDER!r}."
        )
    return (stack, metadata) if return_metadata else stack


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
    array = ensure_tzcyx_stack(np.asarray(stack))
    if dtype is not None:
        array = array.astype(dtype, copy=False)

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
