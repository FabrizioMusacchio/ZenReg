"""
Small batch-processing helpers for ZenReg project folders.

Author: Fabrizio Musacchio
Date: July 2026
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_IMAGE_EXTENSIONS = (
    ".tif",
    ".tiff",
    ".ome.tif",
    ".ome.tiff",
    ".lsm",
    ".czi",
    ".raw",
)


@dataclass(frozen=True)
class BatchImageRecord:
    """One image discovered in a BIDS-like ZenReg batch project."""

    subject_id: str
    experiment_tag: str
    image_path: Path


def _normalize_requested_names(
    names: Iterable[str | Path] | None,
    *,
    prefix: str,
) -> set[str] | None:
    """Normalize optional requested subject or experiment names."""

    if names is None:
        return None
    normalized = set()
    for name in names:
        value = Path(name).name if isinstance(name, Path) else str(name)
        if not value:
            continue
        normalized.add(value)
    return normalized


def _has_supported_image_extension(
    path: Path,
    *,
    image_extensions: tuple[str, ...],
) -> bool:
    """Return True if ``path`` looks like an OMIO-readable microscopy image."""

    name = path.name.lower()
    return any(name.endswith(extension.lower()) for extension in image_extensions)


def iter_bids_like_image_files(
    project_root: str | Path,
    *,
    subject_ids: Iterable[str | Path] | None = None,
    experiment_tags: Iterable[str | Path] | None = None,
    subject_prefix: str = "ID",
    experiment_prefix: str = "TP",
    image_extensions: tuple[str, ...] = DEFAULT_IMAGE_EXTENSIONS,
    recursive: bool = False,
) -> list[BatchImageRecord]:
    """
    Find microscopy image files in a simple BIDS-like project tree.

    The expected project layout is::

        project_root/
          ID000001/
            TP000/
              image_01.ome.tif
            TP001/
              image_01.ome.tif
          ID000002/
            TP000/
              image_01.ome.tif

    Parameters
    ----------
    project_root : str or pathlib.Path
        Root folder that contains subject folders.
    subject_ids : iterable of str or None, optional
        If provided, only these subject folder names are used. If None, all
        folders whose names start with ``subject_prefix`` are considered.
    experiment_tags : iterable of str or None, optional
        If provided, only these experiment folder names are used. If None, all
        folders whose names start with ``experiment_prefix`` are considered.
    subject_prefix, experiment_prefix : str, optional
        Prefixes used for automatic subject/experiment discovery.
    image_extensions : tuple[str, ...], optional
        Filename endings considered image files. Defaults cover TIFF, OME-TIFF,
        CZI, LSM, and Thorlabs RAW.
    recursive : bool, optional
        If True, search recursively inside each experiment folder. The default
        searches only the immediate experiment folder.

    Returns
    -------
    list[BatchImageRecord]
        Sorted image records with subject ID, experiment tag, and image path.
    """

    root = Path(project_root)
    if not root.exists():
        raise FileNotFoundError(f"project_root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"project_root is not a directory: {root}")

    requested_subjects = _normalize_requested_names(subject_ids, prefix=subject_prefix)
    requested_experiments = _normalize_requested_names(experiment_tags, prefix=experiment_prefix)

    if requested_subjects is None:
        subject_dirs = sorted(
            path for path in root.iterdir()
            if path.is_dir() and path.name.startswith(subject_prefix)
        )
    else:
        subject_dirs = [root / subject for subject in sorted(requested_subjects)]

    records: list[BatchImageRecord] = []
    for subject_dir in subject_dirs:
        if not subject_dir.is_dir():
            continue
        if requested_experiments is None:
            experiment_dirs = sorted(
                path for path in subject_dir.iterdir()
                if path.is_dir() and path.name.startswith(experiment_prefix)
            )
        else:
            experiment_dirs = [subject_dir / experiment for experiment in sorted(requested_experiments)]

        for experiment_dir in experiment_dirs:
            if not experiment_dir.is_dir():
                continue
            candidates = experiment_dir.rglob("*") if recursive else experiment_dir.iterdir()
            for image_path in sorted(path for path in candidates if path.is_file()):
                if _has_supported_image_extension(image_path, image_extensions=image_extensions):
                    records.append(
                        BatchImageRecord(
                            subject_id=subject_dir.name,
                            experiment_tag=experiment_dir.name,
                            image_path=image_path,
                        )
                    )
    return records
