"""
Batch-processing helpers for BIDS-like ZenReg project folders.

Author: Fabrizio Musacchio
Date: July 2026
"""
# %% IMPORTS
from __future__ import annotations

import ast
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import re
from typing import Iterable, Sequence
import shutil

import numpy as np

from .io import cleanup_omio_cache, load_stack, save_stack
from .registration import register_stack
# %% CONSTANTS
DEFAULT_IMAGE_PATTERNS = (
    "*.tif",
    "*.tiff",
    "*.ome.tif",
    "*.ome.tiff",
    "*.lsm",
    "*.czi",
    "*.raw",
)

DEFAULT_RAW_TEMPLATE_METADATA = {
    "T": 1,
    "Z": 1,
    "C": 1,
    "Y": 1,
    "X": 1,
    "bits": 16,
    "pixelunit": "micron",
    "physicalsize_xyz": (0.5, 0.5, 1.0),
    "time_increment": 1.0,
    "time_increment_unit": "seconds",
}
# %% DATA CLASSES
@dataclass(frozen=True)
class BatchImageRecord:
    """One image discovered in a BIDS-like ZenReg batch project."""

    subject_id: str
    tag_folders: tuple[str, ...]
    image_path: Path
    output_scope_dir: Path

    @property
    def experiment_tag(self) -> str:
        """Backward-compatible first tag folder label."""

        return self.tag_folders[0] if self.tag_folders else ""

@dataclass(frozen=True)
class BatchProcessedRecord:
    """One successfully registered image in a ZenReg batch run."""

    input_path: Path
    output_path: Path
    subject_id: str
    tag_folders: tuple[str, ...]

@dataclass(frozen=True)
class BatchSkippedRecord:
    """One skipped or failed image in a ZenReg batch run."""

    input_path: Path
    reason: str
    subject_id: str
    tag_folders: tuple[str, ...] = ()
    stage: str = "unknown"

@dataclass(frozen=True)
class BatchRegistrationResult:
    """Summary returned by :func:`register_bids_like_batch`."""

    processed: tuple[BatchProcessedRecord, ...] = ()
    skipped: tuple[BatchSkippedRecord, ...] = ()
    root_error_report_path: Path | None = None
    tag_error_report_paths: tuple[Path, ...] = field(default_factory=tuple)

@dataclass(frozen=True)
class BatchRawYamlTemplateRecord:
    """One RAW file considered for OMIO YAML template creation."""

    raw_path: Path
    yaml_path: Path | None
    template_metadata: dict
    status: str
    reason: str = ""

@dataclass(frozen=True)
class BatchRawYamlTemplateResult:
    """Summary returned by :func:`create_thorlabs_raw_yaml_templates_from_batch_report`."""

    report_path: Path | None
    records: tuple[BatchRawYamlTemplateRecord, ...] = ()

    @property
    def created(self) -> tuple[BatchRawYamlTemplateRecord, ...]:
        """RAW files for which YAML template creation was attempted."""

        return tuple(record for record in self.records if record.status == "created")

    @property
    def skipped(self) -> tuple[BatchRawYamlTemplateRecord, ...]:
        """RAW files skipped during YAML template creation."""

        return tuple(record for record in self.records if record.status != "created")

# %% HELPER FUNCTIONS
def _normalize_subject_ids(subject_ids: Iterable[str | Path] | None) -> tuple[str, ...] | None:
    """Normalize optional requested subject names."""

    if subject_ids is None:
        return None
    normalized = tuple(str(Path(subject_id).name) for subject_id in subject_ids if str(subject_id))
    return normalized

def _select_child_dirs(
    parent: Path,
    requested_tokens: Iterable[str | Path] | None,
    *,
    prefix: str | None = None,
) -> list[Path]:
    """Return child folders matching requested name tokens or an optional prefix."""

    tokens = None if requested_tokens is None else tuple(requested_tokens)
    if tokens is None or tokens == ():
        return sorted(
            path
            for path in parent.iterdir()
            if path.is_dir() and (prefix is None or path.name.startswith(prefix))
        )
    tokens = tuple(str(Path(token).name if isinstance(token, Path) else token) for token in tokens)
    return sorted(
        path
        for path in parent.iterdir()
        if path.is_dir() and any(token in path.name for token in tokens)
    )

def _normalize_tag_folder_levels(
    tag_folder_levels: Sequence[Iterable[str | Path] | None] | None,
) -> tuple[Iterable[str | Path] | None, ...]:
    """Normalize folder-tag levels used below each subject."""

    if tag_folder_levels is None:
        return (("TP",),)
    return tuple(tag_folder_levels)

def _iter_tag_folder_chains(
    root_dir: Path,
    tag_folder_levels: Sequence[Iterable[str | Path] | None],
) -> list[list[Path]]:
    """Return matched folder chains for an arbitrary number of tag levels."""

    if not tag_folder_levels:
        return [[]]

    current_level = tag_folder_levels[0]
    remaining_levels = tag_folder_levels[1:]
    chains: list[list[Path]] = []

    for child_dir in _select_child_dirs(root_dir, current_level, prefix=None):
        for tail_chain in _iter_tag_folder_chains(child_dir, remaining_levels):
            chains.append([child_dir, *tail_chain])
    return chains

def _collect_image_paths(
    scan_dir: Path,
    image_patterns: str | Sequence[str],
    *,
    exclude_name_contains: Sequence[str] = (),
) -> list[Path]:
    """Collect images from one folder using one glob or multiple globs."""

    patterns = (image_patterns,) if isinstance(image_patterns, str) else tuple(image_patterns)
    matched_paths: dict[Path, None] = {}
    for pattern in patterns:
        for path in scan_dir.glob(str(pattern)):
            if path.is_file():
                matched_paths[path] = None
    excluded_tokens = tuple(str(token) for token in exclude_name_contains)
    return [
        path
        for path in sorted(matched_paths)
        if not any(token in path.name for token in excluded_tokens)
    ]

def _sanitize_name(value: str) -> str:
    """Return a filesystem-friendly name fragment."""

    return (
        str(value)
        .replace("\\", "_")
        .replace("/", "_")
        .replace(" ", "_")
        .replace(":", "_")
    )

def _output_path_for_image(output_dir: Path, image_path: Path) -> Path:
    """Return the default registered OME-TIFF output path for one image."""

    name_lower = image_path.name.lower()
    if name_lower.endswith(".ome.tif"):
        stem = image_path.name[:-8]
    elif name_lower.endswith(".ome.tiff"):
        stem = image_path.name[:-9]
    else:
        stem = image_path.stem
    return output_dir / f"{stem}_zenreg_registered.ome.tif"

def _metadata_for_batch_output(metadata: dict | None, output_path: Path) -> dict | None:
    """Return metadata whose OMIO output annotations point at ``output_path``."""

    if metadata is None:
        return None
    output_metadata = deepcopy(metadata)
    annotations = output_metadata.setdefault("Annotations", {})
    annotations["original_filename"] = output_path.name
    annotations["original_filetype"] = "ome.tif"
    annotations["original_parentfolder"] = str(output_path.parent)
    return output_metadata

def _import_omio():
    """Import OMIO lazily for optional RAW YAML helper functionality."""

    try:
        import omio as om
    except ImportError as exc:
        raise ImportError(
            "Creating Thorlabs RAW YAML templates requires OMIO. Install "
            "'omio-microscopy' or use an environment that provides `import omio`."
        ) from exc
    return om

def _output_scope_for_chain(subject_dir: Path, tag_folder_chain: Sequence[Path]) -> Path:
    """Choose the folder that receives the ZenReg output folder."""

    return tag_folder_chain[0] if tag_folder_chain else subject_dir

def _append_tag_error_report(
    report_scope_dir: Path,
    *,
    timestamp: str,
    image_path: Path,
    reason: str,
    raw_template_metadata: dict,
) -> Path:
    """Append one short ZenReg error report inside a subject/tag folder."""

    report_path = report_scope_dir / f"zenreg_batch_error_report_{timestamp}.txt"
    with report_path.open("a", encoding="utf-8") as handle:
        handle.write(
            "\n".join(
                [
                    f"ZenReg batch error report: {timestamp}",
                    f"Skipped image: {image_path}",
                    f"Reason: {reason}",
                    f"Template metadata defaults: {raw_template_metadata!r}",
                    "",
                ]
            )
        )
    return report_path

def _write_template_metadata_block(handle, metadata: dict) -> None:
    """Write one formatted ``template_metadata`` block."""

    handle.write("        'template_metadata': {\n")
    for key, value in metadata.items():
        handle.write(f"            {key!r}: {value!r},\n")
    handle.write("        },\n")

def _write_root_error_report(
    project_root: Path,
    *,
    timestamp: str,
    skipped_records: Sequence[BatchSkippedRecord],
    raw_template_metadata: dict,
) -> Path | None:
    """Write skipped image paths as a copy-pasteable Python dictionary."""

    if not skipped_records:
        return None
    report_path = project_root / f"zenreg_batch_error_report_{timestamp}.txt"
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write(f"# ZenReg batch error report: {timestamp}\n")
        handle.write("# Edit each 'template_metadata' block here, then create OMIO YAML templates.\n")
        handle.write("# The dictionary is valid Python and can also be copied into custom repair scripts.\n\n")
        handle.write("ZENREG_BATCH_SKIPPED_RAW_FILES = {\n")
        for record in skipped_records:
            handle.write(f"    {str(record.input_path)!r}: {{\n")
            handle.write(f"        'reason': {record.reason!r},\n")
            handle.write(f"        'stage': {record.stage!r},\n")
            handle.write(f"        'subject_id': {record.subject_id!r},\n")
            handle.write(f"        'tag_folders': {tuple(record.tag_folders)!r},\n")
            handle.write(f"        'reported_at': {timestamp!r},\n")
            _write_template_metadata_block(handle, raw_template_metadata)
            handle.write("    },\n")
        handle.write("}\n")
    return report_path

def _extract_skipped_raw_dict(report_text: str) -> dict:
    """Extract ``ZENREG_BATCH_SKIPPED_RAW_FILES`` from a root error report."""

    variable_name = "ZENREG_BATCH_SKIPPED_RAW_FILES"
    assignment_index = report_text.find(variable_name)
    if assignment_index < 0:
        return {}

    brace_start = report_text.find("{", assignment_index)
    if brace_start < 0:
        return {}

    depth = 0
    for index in range(brace_start, len(report_text)):
        character = report_text[index]
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                parsed = ast.literal_eval(report_text[brace_start : index + 1])
                if not isinstance(parsed, dict):
                    raise ValueError("ZENREG_BATCH_SKIPPED_RAW_FILES is not a dictionary.")
                return parsed

    raise ValueError("Could not find the end of ZENREG_BATCH_SKIPPED_RAW_FILES.")

def _extract_raw_paths_from_report_text(report_text: str) -> list[Path]:
    """Fallback parser for legacy plain-text reports containing RAW paths."""

    raw_path_pattern = re.compile(r"([A-Za-z]:\\[^\n\r'\"]+?\.raw|/[^\n\r'\"]+?\.raw)")
    return [Path(match.group(1).strip()) for match in raw_path_pattern.finditer(report_text)]

def _load_skipped_raw_entries_from_report(
    report_path: Path,
    *,
    raw_template_metadata: dict,
) -> list[dict]:
    """Load skipped RAW paths and template metadata from one ZenReg report."""

    report_text = report_path.read_text(encoding="utf-8")
    skipped_dict = _extract_skipped_raw_dict(report_text)
    if skipped_dict:
        entries = []
        for raw_path, details in skipped_dict.items():
            details = details if isinstance(details, dict) else {}
            entries.append(
                {
                    "path": Path(raw_path),
                    "template_metadata": dict(
                        details.get("template_metadata", raw_template_metadata)
                    ),
                }
            )
        return entries
    return [
        {
            "path": raw_path,
            "template_metadata": dict(raw_template_metadata),
        }
        for raw_path in _extract_raw_paths_from_report_text(report_text)
    ]

def _expected_raw_yaml_paths(raw_path: Path) -> tuple[Path, ...]:
    """Return likely OMIO Thorlabs RAW YAML sidecar paths."""

    return (
        raw_path.with_suffix(".yaml"),
        raw_path.with_suffix(".yml"),
        raw_path.with_name(raw_path.name + ".yaml"),
        raw_path.with_name(raw_path.name + ".yml"),
    )

def _find_latest_batch_error_report(project_root: Path) -> Path | None:
    """Return the latest root-level ZenReg batch error report if present."""

    reports = sorted(project_root.glob("zenreg_batch_error_report_*.txt"))
    return reports[-1] if reports else None

def _cleanup_fresh_failed_output_dir(output_dir: Path, *, was_created_for_file: bool) -> None:
    """Remove an output folder only if this failed file created it in this run."""

    if was_created_for_file and output_dir.exists():
        shutil.rmtree(output_dir)

def discover_bids_like_batch_images(
    project_root: str | Path,
    *,
    subject_ids: Iterable[str | Path] | None = None,
    subject_prefix: str = "ID",
    tag_folder_levels: Sequence[Iterable[str | Path] | None] | None = None,
    image_patterns: str | Sequence[str] = DEFAULT_IMAGE_PATTERNS,
    exclude_name_contains: Sequence[str] = ("ROIMask.raw",),
) -> list[BatchImageRecord]:
    """
    Discover microscopy image files in a flexible BIDS-like project tree.

    Parameters
    ----------
    project_root : str or pathlib.Path
        Root folder that contains subject folders.
    subject_ids : iterable of str or None, optional
        Explicit subject folders to process. If None, all child folders whose
        names start with ``subject_prefix`` are used.
    subject_prefix : str, optional
        Prefix used for automatic subject discovery. Default: ``"ID"``.
    tag_folder_levels : sequence, optional
        Folder-token levels below each subject. Each level can be ``None`` or
        ``()`` to include all child folders at that level, or a tuple/list of
        name tokens. Tokens are matched by containment, e.g.
        ``("DC000_FOV", "DA000_FOV")`` matches ``DC000_FOV1`` and
        ``DA000_FOV2``. The default ``(("TP",),)`` matches simple
        ``subject/TP.../image`` layouts.
    image_patterns : str or sequence[str], optional
        Glob pattern(s) used to find images in the final tag-folder level.
    exclude_name_contains : sequence[str], optional
        Filename tokens to exclude, for example ``("ROIMask.raw",)``.

    Returns
    -------
    list[BatchImageRecord]
        Sorted image records with subject ID, tag-folder chain, image path, and
        output scope folder.
    """

    root = Path(project_root)
    if not root.exists():
        raise FileNotFoundError(f"project_root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"project_root is not a directory: {root}")

    requested_subjects = _normalize_subject_ids(subject_ids)
    if requested_subjects is None:
        subject_dirs = sorted(
            path for path in root.iterdir()
            if path.is_dir() and path.name.startswith(subject_prefix)
        )
    else:
        subject_dirs = [root / subject_id for subject_id in requested_subjects]

    levels = _normalize_tag_folder_levels(tag_folder_levels)
    records: list[BatchImageRecord] = []
    for subject_dir in subject_dirs:
        if not subject_dir.is_dir():
            continue
        folder_chains = _iter_tag_folder_chains(subject_dir, levels)
        for folder_chain in folder_chains:
            scan_dir = folder_chain[-1] if folder_chain else subject_dir
            image_paths = _collect_image_paths(
                scan_dir,
                image_patterns,
                exclude_name_contains=exclude_name_contains,
            )
            output_scope_dir = _output_scope_for_chain(subject_dir, folder_chain)
            tag_folders = tuple(path.name for path in folder_chain)
            for image_path in image_paths:
                records.append(
                    BatchImageRecord(
                        subject_id=subject_dir.name,
                        tag_folders=tag_folders,
                        image_path=image_path,
                        output_scope_dir=output_scope_dir,
                    )
                )
    return records

def register_bids_like_batch(
    project_root: str | Path,
    *,
    subject_ids: Iterable[str | Path] | None = None,
    subject_prefix: str = "ID",
    tag_folder_levels: Sequence[Iterable[str | Path] | None] | None = None,
    image_patterns: str | Sequence[str] = DEFAULT_IMAGE_PATTERNS,
    exclude_name_contains: Sequence[str] = ("ROIMask.raw",),
    output_folder_name: str = "zenreg_output",
    skip_registered: bool = True,
    load_kwargs: dict | None = None,
    register_kwargs: dict | None = None,
    save_kwargs: dict | None = None,
    use_memmap: bool = False,
    memmap_folder_name: str | None = "omio_memmap_cache",
    memmap_reuse: bool = True,
    cleanup_cache_before_load: bool = False,
    cleanup_cache_after_save: bool = False,
    raw_template_metadata: dict | None = None,
    write_error_reports: bool = True,
    continue_on_error: bool = True,
    verbose: bool = True,
) -> BatchRegistrationResult:
    """
    Load, register, save, and report a BIDS-like microscopy image batch.

    The processor assumes a BIDS-like tree with subject folders and one or more
    folder-tag levels below each subject::

        project_root/
          <sub*>/
            <exp*>/
              image_01.tif / image_01.ome.tif / image_01.lsm / image_01.czi / image_01.raw
            <exp*>/
              <tagfolder*>01/
                image_02.raw

    Parameters
    ----------
    project_root : str or pathlib.Path
        Root folder containing subject folders.
    subject_ids : iterable of str or None, optional
        Explicit subject folders to process. If None, subjects are discovered
        by ``subject_prefix``.
    subject_prefix : str, optional
        Prefix used for subject discovery when ``subject_ids is None``.
    tag_folder_levels : sequence, optional
        Folder-token levels below each subject. Each level can be ``None`` or
        an empty tuple/list to include all child folders, or a tuple/list of
        tokens matched by containment.
    image_patterns : str or sequence[str], optional
        Glob pattern(s) used to find image files in the final tag-folder level.
    exclude_name_contains : sequence[str], optional
        Filename tokens to exclude from processing.
    output_folder_name : str, optional
        Name of the output folder created inside the first tag-folder level
        (or inside the subject folder when no tag folders are configured).
    skip_registered : bool, optional
        If True, skip an input image when its expected registered output already
        exists.
    load_kwargs, register_kwargs, save_kwargs : dict or None, optional
        Keyword arguments forwarded directly to ``load_stack``,
        ``register_stack``, and ``save_stack``. The processor injects required
        values such as ``return_metadata=True`` for loading and
        ``registration_details``/``metadata`` for saving.
    use_memmap : bool, optional
        If True, add memory-mapped loading and registered-output settings unless
        already provided in ``load_kwargs`` or ``register_kwargs``.
    memmap_folder_name : str or None, optional
        Name of the per-output-folder cache directory. If None, the output
        folder itself is used as cache root.
    memmap_reuse : bool, optional
        Forwarded to ``load_stack`` when memory mapping is enabled.
    cleanup_cache_before_load, cleanup_cache_after_save : bool, optional
        Clean the per-image OMIO cache before loading or after saving.
    raw_template_metadata : dict or None, optional
        Metadata defaults written to error reports for later Thorlabs RAW YAML
        repair. Users can edit these blocks in the root report before creating
        YAML templates.
    write_error_reports : bool, optional
        If True, write per-tag and root-level error reports for skipped/failed
        images.
    continue_on_error : bool, optional
        If True, record load/register/save errors and continue with the next
        image. If False, re-raise exceptions immediately.
    verbose : bool, optional
        If True, print batch progress.

    Returns
    -------
    BatchRegistrationResult
        Processed image records, skipped/failed records, and report paths.
    """

    root = Path(project_root)
    records = discover_bids_like_batch_images(
        root,
        subject_ids=subject_ids,
        subject_prefix=subject_prefix,
        tag_folder_levels=tag_folder_levels,
        image_patterns=image_patterns,
        exclude_name_contains=exclude_name_contains,
    )

    base_load_kwargs = dict(load_kwargs or {})
    base_register_kwargs = dict(register_kwargs or {})
    base_save_kwargs = dict(save_kwargs or {})
    base_raw_template_metadata = dict(raw_template_metadata or DEFAULT_RAW_TEMPLATE_METADATA)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    processed: list[BatchProcessedRecord] = []
    skipped: list[BatchSkippedRecord] = []
    tag_report_paths: set[Path] = set()

    def record_skip(record: BatchImageRecord, *, reason: str, stage: str, output_dir: Path) -> None:
        skipped_record = BatchSkippedRecord(
            input_path=record.image_path,
            reason=reason,
            subject_id=record.subject_id,
            tag_folders=record.tag_folders,
            stage=stage,
        )
        skipped.append(skipped_record)
        if write_error_reports and stage != "already_registered":
            report_path = _append_tag_error_report(
                record.output_scope_dir,
                timestamp=timestamp,
                image_path=record.image_path,
                reason=reason,
                raw_template_metadata=base_raw_template_metadata,
            )
            tag_report_paths.add(report_path)
        if verbose:
            print(f"  skipped [{stage}]: {record.image_path}")
            print(f"  reason: {reason}")

    for record in records:
        output_dir = record.output_scope_dir / output_folder_name
        output_dir_was_created_for_file = not output_dir.exists()
        output_dir.mkdir(parents=True, exist_ok=True)
        memmap_cache_dir = output_dir / memmap_folder_name if memmap_folder_name else output_dir
        output_path = _output_path_for_image(output_dir, record.image_path)

        if skip_registered and output_path.exists():
            record_skip(
                record,
                reason=f"Registered output already exists: {output_path}",
                stage="already_registered",
                output_dir=output_dir,
            )
            continue

        tag_text = "_".join(_sanitize_name(tag) for tag in record.tag_folders) or "subject_root"
        memmap_name = f"{_sanitize_name(record.subject_id)}_{tag_text}_{_sanitize_name(record.image_path.stem)}"
        if verbose:
            chain = "/".join(record.tag_folders) if record.tag_folders else "subject_root"
            print(f"Registering {record.subject_id}/{chain}/{record.image_path.name}")

        load_options = dict(base_load_kwargs)
        load_options["return_metadata"] = True
        load_options.setdefault("on_error", "return_none")
        if use_memmap:
            load_options.setdefault("use_memmap", True)
            load_options.setdefault("memmap_folder", memmap_cache_dir)
            load_options.setdefault("memmap_reuse", memmap_reuse)

        if cleanup_cache_before_load and use_memmap:
            cleanup_omio_cache(memmap_cache_dir, full_cleanup=True, verbose=False)

        try:
            stack, metadata = load_stack(record.image_path, **load_options)
        except Exception as exc:
            if not continue_on_error:
                raise
            reason = f"{type(exc).__name__} during load_stack: {exc}"
            _cleanup_fresh_failed_output_dir(output_dir, was_created_for_file=output_dir_was_created_for_file)
            record_skip(record, reason=reason, stage="load", output_dir=output_dir)
            continue

        if stack is None or metadata is None:
            reason = (
                "OMIO returned (None, None) during load_stack(..., "
                "on_error='return_none'), most likely because image metadata "
                "could not be resolved."
            )
            _cleanup_fresh_failed_output_dir(output_dir, was_created_for_file=output_dir_was_created_for_file)
            record_skip(record, reason=reason, stage="load", output_dir=output_dir)
            continue

        if verbose:
            print(f"  input shape: {stack.shape} (TZCYX)")

        registration_options = dict(base_register_kwargs)
        if use_memmap:
            registration_options.setdefault("output_use_memmap", True)
            registration_options.setdefault("output_memmap_folder", memmap_cache_dir)
            registration_options.setdefault("output_memmap_name", memmap_name)
            registration_options.setdefault("output_dtype", np.float32)
        registration_options.setdefault("return_shifts", True)
        registration_options.setdefault("return_details", True)

        try:
            registered, details = register_stack(stack, **registration_options)
        except Exception as exc:
            if not continue_on_error:
                raise
            reason = f"{type(exc).__name__} during register_stack: {exc}"
            record_skip(record, reason=reason, stage="register", output_dir=output_dir)
            if cleanup_cache_after_save and use_memmap:
                cleanup_omio_cache(memmap_cache_dir, full_cleanup=True, verbose=False)
            continue

        save_options = dict(base_save_kwargs)
        save_options.setdefault("metadata", _metadata_for_batch_output(metadata, output_path))
        save_options.setdefault("registration_details", details)

        try:
            written_path = save_stack(output_path, registered, **save_options)
        except Exception as exc:
            if not continue_on_error:
                raise
            reason = f"{type(exc).__name__} during save_stack: {exc}"
            record_skip(record, reason=reason, stage="save", output_dir=output_dir)
            if cleanup_cache_after_save and use_memmap:
                cleanup_omio_cache(memmap_cache_dir, full_cleanup=True, verbose=False)
            continue

        processed.append(
            BatchProcessedRecord(
                input_path=record.image_path,
                output_path=written_path,
                subject_id=record.subject_id,
                tag_folders=record.tag_folders,
            )
        )
        if verbose:
            print(f"  wrote: {written_path}")

        if cleanup_cache_after_save and use_memmap:
            cleanup_omio_cache(memmap_cache_dir, full_cleanup=True, verbose=False)

    reportable_skipped = tuple(record for record in skipped if record.stage != "already_registered")
    root_report_path = (
        _write_root_error_report(
            root,
            timestamp=timestamp,
            skipped_records=reportable_skipped,
            raw_template_metadata=base_raw_template_metadata,
        )
        if write_error_reports
        else None
    )

    if verbose:
        print(
            f"ZenReg batch finished: {len(processed)} processed, "
            f"{len(skipped)} skipped."
        )
        if reportable_skipped:
            print("ZenReg skipped/failed image files:")
            for record in reportable_skipped:
                print(str(record.input_path))
            print(f"ZenReg batch error report written to: {root_report_path}")

    return BatchRegistrationResult(
        processed=tuple(processed),
        skipped=tuple(skipped),
        root_error_report_path=root_report_path,
        tag_error_report_paths=tuple(sorted(tag_report_paths)),
    )

def create_thorlabs_raw_yaml_templates_from_batch_report(
    project_root: str | Path,
    *,
    report_name: str | Path | None = None,
    subject_ids: Iterable[str | Path] | None = None,
    subject_prefix: str = "ID",
    tag_folder_levels: Sequence[Iterable[str | Path] | None] | None = None,
    image_patterns: str | Sequence[str] = ("*.raw",),
    exclude_name_contains: Sequence[str] = ("ROIMask.raw",),
    restrict_to_discovered: bool = True,
    raw_template_metadata: dict | None = None,
    overwrite_existing: bool = False,
    verbose: bool = True,
) -> BatchRawYamlTemplateResult:
    """
    Create OMIO Thorlabs RAW YAML templates from a ZenReg batch error report.

    The root-level error report written by :func:`register_bids_like_batch`
    contains a copy-pasteable ``ZENREG_BATCH_SKIPPED_RAW_FILES`` dictionary with
    RAW paths and editable ``template_metadata`` blocks. This helper reads that
    report and calls ``omio.create_thorlabs_raw_yaml`` for each selected RAW
    file. By default, report entries are additionally restricted to files
    discoverable from the same BIDS-like folder settings used for batch
    registration.

    Parameters
    ----------
    project_root : str or pathlib.Path
        Root folder containing the ZenReg batch error report and subject
        folders.
    report_name : str, pathlib.Path, or None, optional
        Report filename or path. If None, the latest
        ``zenreg_batch_error_report_*.txt`` in ``project_root`` is used.
    subject_ids, subject_prefix, tag_folder_levels, image_patterns,
    exclude_name_contains
        Same discovery controls as :func:`register_bids_like_batch`. They are
        used only when ``restrict_to_discovered=True``.
    restrict_to_discovered : bool, optional
        If True, create YAML templates only for report entries that also match
        the provided BIDS-like discovery settings.
    raw_template_metadata : dict or None, optional
        Fallback metadata used for legacy reports without per-file
        ``template_metadata`` blocks.
    overwrite_existing : bool, optional
        If False, skip RAW files that already have a likely YAML/YML sidecar.
    verbose : bool, optional
        If True, print progress and skip reasons.

    Returns
    -------
    BatchRawYamlTemplateResult
        Per-RAW template creation records.
    """

    root = Path(project_root)
    if report_name is None:
        report_path = _find_latest_batch_error_report(root)
        if report_path is None:
            raise FileNotFoundError(
                f"No zenreg_batch_error_report_*.txt found in {root!s}."
            )
    else:
        report_path = Path(report_name)
        if not report_path.is_absolute():
            report_path = root / report_path
    if not report_path.exists():
        raise FileNotFoundError(f"ZenReg batch error report not found: {report_path}")

    fallback_metadata = dict(raw_template_metadata or DEFAULT_RAW_TEMPLATE_METADATA)
    entries = _load_skipped_raw_entries_from_report(
        report_path,
        raw_template_metadata=fallback_metadata,
    )

    allowed_paths: set[Path] | None = None
    if restrict_to_discovered:
        allowed_paths = {
            record.image_path.resolve()
            for record in discover_bids_like_batch_images(
                root,
                subject_ids=subject_ids,
                subject_prefix=subject_prefix,
                tag_folder_levels=tag_folder_levels,
                image_patterns=image_patterns,
                exclude_name_contains=exclude_name_contains,
            )
        }

    om = _import_omio()
    records: list[BatchRawYamlTemplateRecord] = []
    for entry in entries:
        raw_path = Path(entry["path"])
        template_metadata = dict(entry.get("template_metadata", fallback_metadata))
        yaml_paths = _expected_raw_yaml_paths(raw_path)
        existing_yaml_paths = [path for path in yaml_paths if path.exists()]

        if allowed_paths is not None and raw_path.resolve() not in allowed_paths:
            reason = "RAW file is not part of the selected BIDS-like batch folders."
            records.append(
                BatchRawYamlTemplateRecord(
                    raw_path=raw_path,
                    yaml_path=None,
                    template_metadata=template_metadata,
                    status="skipped",
                    reason=reason,
                )
            )
            if verbose:
                print(f"Skipping RAW outside selected folders: {raw_path}")
            continue

        if not raw_path.exists():
            reason = "RAW file does not exist."
            records.append(
                BatchRawYamlTemplateRecord(
                    raw_path=raw_path,
                    yaml_path=None,
                    template_metadata=template_metadata,
                    status="missing",
                    reason=reason,
                )
            )
            if verbose:
                print(f"Skipping missing RAW file: {raw_path}")
            continue

        if existing_yaml_paths and not overwrite_existing:
            reason = "YAML/YML sidecar already exists."
            records.append(
                BatchRawYamlTemplateRecord(
                    raw_path=raw_path,
                    yaml_path=existing_yaml_paths[0],
                    template_metadata=template_metadata,
                    status="exists",
                    reason=reason,
                )
            )
            if verbose:
                existing_names = ", ".join(str(path) for path in existing_yaml_paths)
                print(f"Skipping existing YAML for {raw_path}: {existing_names}")
            continue

        if verbose:
            print(f"Creating OMIO YAML template for: {raw_path}")
        om.create_thorlabs_raw_yaml(raw_path, **template_metadata)
        created_yaml = next((path for path in yaml_paths if path.exists()), yaml_paths[0])
        records.append(
            BatchRawYamlTemplateRecord(
                raw_path=raw_path,
                yaml_path=created_yaml,
                template_metadata=template_metadata,
                status="created",
                reason="",
            )
        )

    if verbose:
        created_count = sum(record.status == "created" for record in records)
        skipped_count = len(records) - created_count
        print(
            f"ZenReg RAW YAML template creation finished: "
            f"{created_count} created, {skipped_count} skipped."
        )

    return BatchRawYamlTemplateResult(
        report_path=report_path,
        records=tuple(records),
    )
# %% END
