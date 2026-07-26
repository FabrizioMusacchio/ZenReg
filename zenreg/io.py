"""
Minimal stack I/O helpers for ZenReg examples and quick scripts.

Author: Fabrizio Musacchio
Date: June 2026
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_stack(path: str | Path) -> np.ndarray:
    """
    Load a stack from ``.npy``, ``.npz``, ``.tif``, or ``.tiff``.

    Parameters
    ----------
    path : str or pathlib.Path
        Input file path.

    Returns
    -------
    numpy.ndarray
        Loaded image stack.
    """

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path)
    if suffix == ".npz":
        data = np.load(path)
        if "stack" not in data:
            raise ValueError(f"NPZ file {path!s} does not contain a 'stack' array.")
        return data["stack"]
    if suffix in {".tif", ".tiff"}:
        import tifffile

        return tifffile.imread(path)
    raise ValueError(f"Unsupported file suffix {suffix!r}.")


def save_stack(path: str | Path, stack, *, dtype: str | None = "float32") -> Path:
    """
    Save a stack to ``.npy``, ``.npz``, ``.tif``, or ``.tiff``.

    Parameters
    ----------
    path : str or pathlib.Path
        Output file path.
    stack : array-like
        Image data to save.
    dtype : str or None, optional
        Optional output dtype conversion. If ``None``, the input dtype is kept.

    Returns
    -------
    pathlib.Path
        Saved output path.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(stack)
    if dtype is not None:
        array = array.astype(dtype, copy=False)

    suffix = path.suffix.lower()
    if suffix == ".npy":
        np.save(path, array)
    elif suffix == ".npz":
        np.savez_compressed(path, stack=array)
    elif suffix in {".tif", ".tiff"}:
        import tifffile

        tifffile.imwrite(path, array, imagej=False)
    else:
        raise ValueError(f"Unsupported file suffix {suffix!r}.")
    return path
