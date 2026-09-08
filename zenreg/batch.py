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
import json
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

CANONICAL_AXIS_ORDER = "TZCYX"
BATCH_AXIS_TO_INDEX = {"T": 0, "Z": 1, "C": 2, "Y": 3, "X": 4}
# %% DATA CLASSES
@dataclass(frozen=True)
class BatchImageRecord:
    """One image discovered in a BIDS-like ZenReg batch project."""

    subject_id: str
    tag_folders: tuple[str, ...]
    image_path: Path
    output_scope_dir: Path
    input_kind: str = "file"
    stack_folder_tag: str | None = None
    stack_folder_paths: tuple[Path, ...] = ()
    stack_folder_merge_axis: str | None = None
    output_name_stem: str | None = None

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
    root_run_report_yaml_path: Path | None = None
    root_run_report_txt_path: Path | None = None

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
    """Summary returned by :func:`batch_create_thorlabs_raw_yaml_templates`."""

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
    image_patterns: str | Sequence[str] | None,
    *,
    exclude_name_contains: Sequence[str] = (),
) -> list[Path]:
    """Collect images from one folder using one glob or multiple globs."""

    if image_patterns is None:
        patterns = DEFAULT_IMAGE_PATTERNS
    elif isinstance(image_patterns, str):
        patterns = (image_patterns,)
    else:
        patterns = tuple(image_patterns)
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

def _natural_sort_key(path: Path) -> tuple:
    """Return a sort key that orders numeric suffixes naturally."""

    parts = re.split(r"(\d+)", path.name)
    return tuple(int(part) if part.isdigit() else part.lower() for part in parts)

def _normalize_stack_folder_tags(stack_folder_tag: str | Path | Sequence[str | Path] | None) -> tuple[str, ...]:
    """Normalize optional stack-folder tag(s)."""

    if stack_folder_tag is None:
        return ()
    if isinstance(stack_folder_tag, bool):
        raise ValueError("stack_folder_tag expects a tag string or sequence of strings, not a boolean.")
    if isinstance(stack_folder_tag, (str, Path)):
        tag = str(Path(stack_folder_tag).name if isinstance(stack_folder_tag, Path) else stack_folder_tag)
        return (tag,) if tag else ()
    return tuple(
        str(Path(tag).name if isinstance(tag, Path) else tag)
        for tag in stack_folder_tag
        if str(tag))

def _stack_folder_matches(name: str, tag: str, match_mode: str) -> bool:
    """Return True when one folder name matches a stack-folder tag."""

    if match_mode == "startswith":
        return name.startswith(tag)
    if match_mode == "contains":
        return tag in name
    raise ValueError("stack_folder_match must be 'startswith' or 'contains'.")

def _collect_stack_folder_groups(
    scan_dir: Path,
    stack_folder_tags: Sequence[str],
    *,
    stack_folder_match: str,
    image_patterns: str | Sequence[str] | None,
    exclude_name_contains: Sequence[str],
) -> list[tuple[str, tuple[Path, ...]]]:
    """Collect tagged child-folder groups that OMIO should merge as folder stacks."""

    if not stack_folder_tags or not scan_dir.is_dir():
        return []
    child_dirs = sorted(
        (
            path
            for path in scan_dir.iterdir()
            if path.is_dir()
            and not any(token in path.name for token in exclude_name_contains)
        ),
        key=_natural_sort_key)
    groups: list[tuple[str, tuple[Path, ...]]] = []
    seen_first_paths: set[Path] = set()
    for tag in stack_folder_tags:
        matching_dirs = tuple(
            path
            for path in child_dirs
            if _stack_folder_matches(path.name, tag, stack_folder_match)
            and "_" in path.name
            and _collect_image_paths(
                path,
                image_patterns,
                exclude_name_contains=exclude_name_contains))
        if not matching_dirs:
            continue
        first_path = matching_dirs[0]
        if first_path in seen_first_paths:
            continue
        seen_first_paths.add(first_path)
        groups.append((tag, matching_dirs))
    return groups

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

def _output_path_for_record(output_dir: Path, record: BatchImageRecord) -> Path:
    """Return the default registered OME-TIFF output path for one batch record."""

    if record.output_name_stem:
        return output_dir / f"{record.output_name_stem}_zenreg_registered.ome.tif"
    return _output_path_for_image(output_dir, record.image_path)

def _merged_stack_path_for_record(
    output_scope_dir: Path,
    record: BatchImageRecord,
    *,
    merged_stack_name: str | None,
    merged_stack_suffix: str,
) -> Path:
    """Return the optional intermediate merged-stack output path."""

    stem = merged_stack_name or record.output_name_stem
    if not stem:
        stem = f"{record.image_path.stem}{merged_stack_suffix}"
    return output_scope_dir / f"{stem}.ome.tif"

def _merged_stack_cache_path(record: BatchImageRecord, load_options: dict) -> Path | None:
    """Return the disk-backed Zarr path for a merged folder-stack input."""

    if not load_options.get("use_memmap", False):
        return None
    memmap_folder = load_options.get("memmap_folder")
    if memmap_folder is None:
        return None
    stem = record.output_name_stem or f"{record.image_path.stem}_merged"
    return Path(memmap_folder) / ".omio_cache" / f"{_sanitize_name(stem)}_input.zarr"

def _stack_folder_source_memmap_folder(
    base_memmap_folder: Path | None,
    source_path: Path,
    record: BatchImageRecord,
) -> Path | None:
    """Return a cache parent that is unique for one folder-stack source."""

    if base_memmap_folder is None:
        return None
    source_stem = _sanitize_name(source_path.parent.name)
    record_stem = _sanitize_name(record.output_name_stem or record.image_path.parent.name)
    return (
        base_memmap_folder
        / ".omio_cache"
        / "stack_folder_sources"
        / record_stem
        / source_stem
    )

def _stack_chunks_tzcyx(shape: Sequence[int]) -> tuple[int, int, int, int, int]:
    """Return conservative chunks for disk-backed merged folder-stack arrays."""

    t, z, c, y, x = (int(value) for value in shape)
    return (1, min(z, 4), min(c, 1), min(y, 512), min(x, 512))

def _first_stack_folder_image_path(
    folder: Path,
    *,
    image_patterns: str | Sequence[str] | None,
    exclude_name_contains: Sequence[str],
) -> Path | None:
    """Return the first naturally sorted image file that matches a stack folder."""

    image_paths = _collect_image_paths(
        folder,
        image_patterns,
        exclude_name_contains=exclude_name_contains)
    return sorted(image_paths, key=_natural_sort_key)[0] if image_paths else None

def _empty_merged_stack(
    shape: Sequence[int],
    dtype,
    *,
    cache_path: Path | None,
):
    """Allocate a merged folder-stack array in Zarr when requested."""

    shape_tuple = tuple(int(value) for value in shape)
    if cache_path is None:
        return np.zeros(shape_tuple, dtype=dtype)

    import zarr

    if cache_path.exists():
        shutil.rmtree(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    return zarr.open(
        str(cache_path),
        mode="w",
        shape=shape_tuple,
        dtype=np.dtype(dtype),
        chunks=_stack_chunks_tzcyx(shape_tuple))

def _merge_stack_folder_arrays(
    stacks: Sequence,
    metadata_items: Sequence[dict],
    *,
    merge_axis: str,
    cache_path: Path | None,
    source_paths: Sequence[Path],
):
    """Merge canonical TZCYX stacks along one axis with zero padding if needed."""

    if not stacks:
        raise ValueError("No readable stack-folder images found.")

    shapes = [tuple(int(value) for value in stack.shape) for stack in stacks]
    for index, shape in enumerate(shapes):
        if len(shape) != 5:
            raise ValueError(f"Stack-folder image {source_paths[index]} is not 5D TZCYX: {shape}.")
    for index, metadata in enumerate(metadata_items):
        if metadata.get("axes") != CANONICAL_AXIS_ORDER:
            raise ValueError(
                f"Stack-folder image {source_paths[index]} has axes={metadata.get('axes')!r}; "
                f"expected {CANONICAL_AXIS_ORDER!r}.")

    axis_index = BATCH_AXIS_TO_INDEX[merge_axis]
    output_shape = list(shapes[0])
    output_shape[axis_index] = sum(shape[axis_index] for shape in shapes)
    for axis in range(5):
        if axis != axis_index:
            output_shape[axis] = max(shape[axis] for shape in shapes)

    dtype = np.result_type(*[getattr(stack, "dtype", np.float32) for stack in stacks])
    merged = _empty_merged_stack(output_shape, dtype, cache_path=cache_path)
    destination_start = 0
    for stack, shape in zip(stacks, shapes):
        destination_stop = destination_start + shape[axis_index]
        destination_slices = [slice(0, size) for size in shape]
        destination_slices[axis_index] = slice(destination_start, destination_stop)
        merged[tuple(destination_slices)] = np.asarray(stack, dtype=dtype)
        destination_start = destination_stop

    metadata = deepcopy(metadata_items[0])
    metadata["axes"] = CANONICAL_AXIS_ORDER
    metadata["shape"] = tuple(output_shape)
    metadata["SizeT"] = int(output_shape[0])
    metadata["SizeZ"] = int(output_shape[1])
    metadata["SizeC"] = int(output_shape[2])
    metadata["SizeY"] = int(output_shape[3])
    metadata["SizeX"] = int(output_shape[4])
    annotations = metadata.setdefault("Annotations", {})
    annotations["zenreg_folder_stack_merge_axis"] = merge_axis
    annotations["zenreg_folder_stack_source_paths"] = [str(path) for path in source_paths]
    if cache_path is not None:
        annotations["zenreg_folder_stack_zarr_cache"] = str(cache_path)
    return merged, metadata

def _load_stack_folder_record(
    record: BatchImageRecord,
    *,
    load_options: dict,
    image_patterns: str | Sequence[str] | None,
    exclude_name_contains: Sequence[str],
    verbose: bool,
):
    """Load and naturally merge one stacked-folder batch record."""

    matched_source_paths = []
    stacks = []
    metadata_items = []
    base_memmap_folder = (
        Path(load_options["memmap_folder"])
        if load_options.get("use_memmap", False) and load_options.get("memmap_folder") is not None
        else None
    )
    for stack_folder in record.stack_folder_paths:
        source_path = _first_stack_folder_image_path(
            stack_folder,
            image_patterns=image_patterns,
            exclude_name_contains=exclude_name_contains)
        if source_path is None:
            raise FileNotFoundError(f"No matching image file found in stack folder: {stack_folder}")
        matched_source_paths.append(source_path)
        source_load_options = dict(load_options)
        source_load_options["return_metadata"] = True
        source_load_options.pop("folder_stacks", None)
        source_load_options.pop("merge_folder_stacks", None)
        source_load_options.pop("merge_along_axis", None)
        source_memmap_folder = _stack_folder_source_memmap_folder(
            base_memmap_folder,
            source_path,
            record)
        if source_memmap_folder is not None:
            source_load_options["memmap_folder"] = source_memmap_folder
        stack_i, metadata_i = load_stack(source_path, **source_load_options)
        if stack_i is None or metadata_i is None:
            raise ValueError(f"OMIO returned None while reading stack-folder source: {source_path}")
        stacks.append(stack_i)
        metadata_items.append(metadata_i)

    merge_axis = record.stack_folder_merge_axis or "T"
    stack, metadata = _merge_stack_folder_arrays(
        stacks,
        metadata_items,
        merge_axis=merge_axis,
        cache_path=_merged_stack_cache_path(record, load_options),
        source_paths=matched_source_paths)

    if metadata is not None:
        annotations = metadata.setdefault("Annotations", {})
        annotations["zenreg_stack_folder_tag"] = record.stack_folder_tag
        annotations["zenreg_stack_folder_merge_axis"] = merge_axis
        annotations["zenreg_stack_folder_paths"] = [str(path) for path in record.stack_folder_paths]
        annotations["zenreg_stack_folder_matched_files"] = [str(path) for path in matched_source_paths]
    if verbose:
        folder_names = ", ".join(path.name for path in record.stack_folder_paths)
        print(f"    merged stack folders along {merge_axis}: {folder_names}", flush=True)
    return stack, metadata

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

def _run_report_paths(project_root: Path, run_report_name: str) -> tuple[Path, Path]:
    """Return YAML and text report paths for one batch project."""

    report_base = project_root / run_report_name
    return report_base.with_suffix(".yaml"), report_base.with_suffix(".txt")

def _load_run_report(path: Path, project_root: Path) -> dict:
    """Load an existing run report or return an empty report payload."""

    if not path.exists():
        return {
            "zenreg_batch_run_report_version": 1,
            "project_root": str(project_root),
            "last_updated": None,
            "files": {},
        }
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return {
            "zenreg_batch_run_report_version": 1,
            "project_root": str(project_root),
            "last_updated": None,
            "files": {},
        }
    try:
        import yaml

        loaded = yaml.safe_load(text)
    except Exception:
        loaded = json.loads(text)
    if not isinstance(loaded, dict):
        raise ValueError(f"Run report is not a mapping: {path}")
    loaded.setdefault("zenreg_batch_run_report_version", 1)
    loaded.setdefault("project_root", str(project_root))
    loaded.setdefault("last_updated", None)
    loaded.setdefault("files", {})
    if not isinstance(loaded["files"], dict):
        raise ValueError(f"Run report 'files' entry is not a mapping: {path}")
    return loaded

def _write_run_report_yaml(path: Path, payload: dict) -> None:
    """Write the machine-readable run report."""

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml

        text = yaml.safe_dump(
            payload,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        )
    except Exception:
        text = json.dumps(payload, indent=2, ensure_ascii=False)
    path.write_text(text, encoding="utf-8")

def _relative_report_path(path: Path, root: Path) -> str:
    """Return a stable POSIX-style path relative to ``root`` when possible."""

    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()

def _status_symbol(status: str, style: str) -> str:
    """Return one status marker for the text report."""

    normalized_style = str(style).lower()
    if normalized_style in {"unicode", "symbol", "symbols"}:
        return {
            "processed": "✓ REGISTERED",
            "already_registered": "✓ REGISTERED",
            "load": "✗ FAILED",
            "register": "✗ FAILED",
            "save": "✗ FAILED",
            "failed": "✗ FAILED",
        }.get(status, "•")
    return {
        "processed": "REGISTERED",
        "already_registered": "REGISTERED",
        "load": "FAILED",
        "register": "FAILED",
        "save": "FAILED",
        "failed": "FAILED",
    }.get(status, status.upper())

def _format_run_summary(run: dict) -> str:
    """Return one compact run-history line."""

    status = str(run.get("status", "unknown"))
    if status == "already_registered":
        status_label = "skipped/already registered"
    elif status in {"load", "register", "save", "failed"}:
        status_label = f"failed/{status}"
    else:
        status_label = status
    parts = [str(run.get("timestamp", "unknown")), status_label]
    method = run.get("method")
    if method and status == "processed":
        parts.append(str(method))
    registration_channel = run.get("registration_channel")
    if registration_channel is not None and status == "processed":
        parts.append(f"c={registration_channel}")
    reason = run.get("reason")
    if reason and status != "already_registered":
        parts.append(str(reason))
    return " | ".join(parts)

def _add_tree_path(tree: dict, parts: Sequence[str], file_key: str) -> None:
    """Add one file key to a nested folder tree."""

    node = tree
    for part in parts:
        node = node.setdefault(part, {})
    node.setdefault("__files__", []).append(file_key)

def _render_tree_node(
    lines: list[str],
    tree: dict,
    files: dict,
    *,
    indent: str = "",
    status_symbol_style: str = "ascii",
) -> None:
    """Render one nested folder node into ``lines``."""

    folder_names = sorted(key for key in tree if key != "__files__")
    file_keys = sorted(tree.get("__files__", []))
    entries = [(name, "folder") for name in folder_names] + [(key, "file") for key in file_keys]
    for index, (name, entry_type) in enumerate(entries):
        is_last = index == len(entries) - 1
        branch = "└─ " if is_last else "├─ "
        child_indent = indent + ("   " if is_last else "│  ")
        if entry_type == "folder":
            lines.append(f"{indent}{branch}{name}/")
            _render_tree_node(
                lines,
                tree[name],
                files,
                indent=child_indent,
                status_symbol_style=status_symbol_style,
            )
            continue
        file_entry = files[name]
        runs = list(file_entry.get("runs", []))
        latest_run = runs[-1] if runs else {}
        status = str(latest_run.get("status", file_entry.get("latest_status", "unknown")))
        symbol = _status_symbol(status, status_symbol_style)
        display_name = file_entry.get("display_name") or Path(name).name
        lines.append(f"{indent}{branch}{display_name} [{symbol}]")
        output_path = latest_run.get("output_path") or file_entry.get("latest_output_path")
        if output_path:
            lines.append(f"{child_indent}output: {output_path}")
        if file_entry.get("input_kind") == "folder_stack":
            stack_tag = file_entry.get("stack_folder_tag")
            stack_paths = file_entry.get("stack_folder_paths", [])
            merge_axis = file_entry.get("merge_axis")
            if stack_tag:
                lines.append(f"{child_indent}input: tagged folder stack {stack_tag}_*")
            if merge_axis:
                lines.append(f"{child_indent}merge_axis: {merge_axis}")
            if stack_paths:
                lines.append(f"{child_indent}folders: {', '.join(stack_paths)}")
        if runs:
            lines.append(f"{child_indent}runs:")
            for run in runs:
                lines.append(f"{child_indent}  - {_format_run_summary(run)}")

def _render_run_report_text(
    payload: dict,
    path: Path,
    *,
    status_symbol_style: str,
) -> None:
    """Write the human-readable text run report."""

    files = payload.get("files", {})
    tree: dict = {}
    for file_key in files:
        parts = Path(file_key).parts
        if not parts:
            continue
        _add_tree_path(tree, parts[:-1], file_key)

    lines = [
        "ZenReg batch run report",
        f"Project root: {payload.get('project_root', '')}",
        f"Last updated: {payload.get('last_updated', '')}",
        "",
    ]
    if files:
        _render_tree_node(
            lines,
            tree,
            files,
            status_symbol_style=status_symbol_style,
        )
    else:
        lines.append("No batch image files have been recorded yet.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")

def _registration_setting(register_kwargs: dict, key: str):
    """Return one registration setting from batch ``register_kwargs``."""

    return register_kwargs.get(key)

def _append_run_report_entry(
    payload: dict,
    record: BatchImageRecord,
    *,
    root: Path,
    timestamp: str,
    status: str,
    register_kwargs: dict,
    output_path: Path | None = None,
    reason: str | None = None,
) -> None:
    """Append one run-history entry for a discovered image."""

    file_key = _relative_report_path(record.image_path, root)
    files = payload.setdefault("files", {})
    file_entry = files.setdefault(
        file_key,
        {
            "subject_id": record.subject_id,
            "tag_folders": list(record.tag_folders),
            "input_path": file_key,
            "input_kind": record.input_kind,
            "runs": [],
        },
    )
    file_entry["subject_id"] = record.subject_id
    file_entry["tag_folders"] = list(record.tag_folders)
    file_entry["input_path"] = file_key
    file_entry["input_kind"] = record.input_kind
    if record.input_kind == "folder_stack":
        file_entry["stack_folder_tag"] = record.stack_folder_tag
        file_entry["display_name"] = f"{record.stack_folder_tag}_* folder stack"
        file_entry["merge_axis"] = record.stack_folder_merge_axis
        file_entry["stack_folder_paths"] = [
            _relative_report_path(path, root) for path in record.stack_folder_paths
        ]
    run_entry = {
        "timestamp": timestamp,
        "status": status,
        "method": _registration_setting(register_kwargs, "method"),
        "registration_channel": _registration_setting(register_kwargs, "registration_channel"),
        "time_registration_mode": _registration_setting(register_kwargs, "time_registration_mode"),
        "time_reference_mode": _registration_setting(register_kwargs, "time_reference_mode"),
    }
    if output_path is not None:
        run_entry["output_path"] = _relative_report_path(output_path, root)
        file_entry["latest_output_path"] = run_entry["output_path"]
    if reason:
        run_entry["reason"] = reason
    file_entry.setdefault("runs", []).append(run_entry)
    file_entry["latest_status"] = status

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
    image_patterns: str | Sequence[str] | None = DEFAULT_IMAGE_PATTERNS,
    exclude_name_contains: Sequence[str] = ("ROIMask.raw",),
    stack_folder_tag: str | Path | Sequence[str | Path] | None = None,
    stack_folder_match: str = "startswith",
    stack_folder_merge_axis: str = "T",
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
    stack_folder_tag : str, sequence[str], or None, optional
        Optional tagged child-folder group(s) below the final tag-folder level.
        For example, ``stack_folder_tag="OV"`` detects ``OV_1``, ``OV_2``,
        etc. below a FOV folder and returns one folder-stack record per FOV.
    stack_folder_match : {"startswith", "contains"}, optional
        How ``stack_folder_tag`` is matched against child-folder names.
        Default: ``"startswith"``.
    stack_folder_merge_axis : {"T", "Z", "C"}, optional
        Logical axis along which OMIO should merge tagged folder stacks. Stored
        in returned records for reporting. Default: ``"T"``.

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
    stack_folder_tags = _normalize_stack_folder_tags(stack_folder_tag)
    merge_axis = str(stack_folder_merge_axis).upper()
    if merge_axis not in {"T", "Z", "C"}:
        raise ValueError("stack_folder_merge_axis must be one of 'T', 'Z', or 'C'.")

    records: list[BatchImageRecord] = []
    for subject_dir in subject_dirs:
        if not subject_dir.is_dir():
            continue
        folder_chains = _iter_tag_folder_chains(subject_dir, levels)
        for folder_chain in folder_chains:
            scan_dir = folder_chain[-1] if folder_chain else subject_dir
            output_scope_dir = _output_scope_for_chain(subject_dir, folder_chain)
            tag_folders = tuple(path.name for path in folder_chain)
            if stack_folder_tags:
                for tag, stack_folder_paths in _collect_stack_folder_groups(
                    scan_dir,
                    stack_folder_tags,
                    stack_folder_match=stack_folder_match,
                    image_patterns=image_patterns,
                    exclude_name_contains=exclude_name_contains,
                ):
                    output_name_stem = (
                        f"{_sanitize_name(output_scope_dir.name)}_"
                        f"{_sanitize_name(tag)}_merged_{merge_axis}"
                    )
                    records.append(
                        BatchImageRecord(
                            subject_id=subject_dir.name,
                            tag_folders=tag_folders,
                            image_path=stack_folder_paths[0],
                            output_scope_dir=output_scope_dir,
                            input_kind="folder_stack",
                            stack_folder_tag=tag,
                            stack_folder_paths=stack_folder_paths,
                            stack_folder_merge_axis=merge_axis,
                            output_name_stem=output_name_stem,
                        )
                    )
                continue

            image_paths = _collect_image_paths(
                scan_dir,
                image_patterns,
                exclude_name_contains=exclude_name_contains,
            )
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
    image_patterns: str | Sequence[str] | None = DEFAULT_IMAGE_PATTERNS,
    exclude_name_contains: Sequence[str] = ("ROIMask.raw",),
    stack_folder_tag: str | Path | Sequence[str | Path] | None = None,
    stack_folder_match: str = "startswith",
    stack_folder_merge_axis: str = "T",
    save_merged_stack: bool = False,
    merged_stack_name: str | None = None,
    merged_stack_suffix: str = "_merged",
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
    write_run_report: bool = True,
    run_report_name: str = "zenreg_batch_run_report",
    run_report_format: str | Sequence[str] = ("yaml", "txt"),
    run_report_status_symbol_style: str = "ascii",
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
    stack_folder_tag : str, sequence[str], or None, optional
        Optional tagged child-folder group below the final tag-folder level.
        When set, ZenReg creates one batch record per matching group instead of
        processing individual image files. For example, use
        ``tag_folder_levels=(("FOV",),)`` and ``stack_folder_tag="OV"`` for a
        tree such as ``ID25068/FOV1_pre/OV_1``, ``OV_2``, ...
    stack_folder_match : {"startswith", "contains"}, optional
        How ``stack_folder_tag`` is matched against child-folder names.
        Default: ``"startswith"``.
    stack_folder_merge_axis : {"T", "Z", "C"}, optional
        Axis passed to OMIO as ``merge_along_axis`` for folder-stack merging.
        Default: ``"T"``.
    save_merged_stack : bool, optional
        If True, save the OMIO-merged stack as an intermediate OME-TIFF in the
        output-scope folder before registration. Default: False.
    merged_stack_name : str or None, optional
        Optional explicit stem for intermediate merged stacks. If None, ZenReg
        builds one from the output-scope folder and stack-folder tag.
    merged_stack_suffix : str, optional
        Suffix used for the intermediate merged stack when no record-specific
        stem is available. Default: ``"_merged"``.
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
    write_run_report : bool, optional
        If True, update a root-level project run report after the batch. Unlike
        error reports, the run report is written for successful runs as well and
        preserves a per-image run history.
    run_report_name : str, optional
        Base filename for the project run report. ZenReg writes
        ``<run_report_name>.yaml`` and/or ``<run_report_name>.txt`` depending
        on ``run_report_format``.
    run_report_format : str or sequence[str], optional
        Report format(s) to write. Supported values are ``"yaml"`` and
        ``"txt"``. Default: ``("yaml", "txt")``.
    run_report_status_symbol_style : {"ascii", "unicode"}, optional
        Status marker style for the text report. ``"ascii"`` uses
        ``REGISTERED`` and ``FAILED`` for terminal-safe output.
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
        stack_folder_tag=stack_folder_tag,
        stack_folder_match=stack_folder_match,
        stack_folder_merge_axis=stack_folder_merge_axis,
    )

    base_load_kwargs = dict(load_kwargs or {})
    base_register_kwargs = dict(register_kwargs or {})
    base_save_kwargs = dict(save_kwargs or {})
    base_raw_template_metadata = dict(raw_template_metadata or DEFAULT_RAW_TEMPLATE_METADATA)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    processed: list[BatchProcessedRecord] = []
    skipped: list[BatchSkippedRecord] = []
    tag_report_paths: set[Path] = set()
    run_report_events: list[dict] = []

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
        status = "already_registered" if stage == "already_registered" else stage
        run_report_events.append(
            {
                "record": record,
                "status": status,
                "reason": reason,
                "output_path": _output_path_for_record(output_dir, record)
                if stage == "already_registered"
                else None,
            }
        )

    for record in records:
        output_dir = record.output_scope_dir / output_folder_name
        output_dir_was_created_for_file = not output_dir.exists()
        output_dir.mkdir(parents=True, exist_ok=True)
        memmap_cache_dir = output_dir / memmap_folder_name if memmap_folder_name else output_dir
        output_path = _output_path_for_record(output_dir, record)

        if skip_registered and output_path.exists():
            record_skip(
                record,
                reason=f"Registered output already exists: {output_path}",
                stage="already_registered",
                output_dir=output_dir,
            )
            continue

        tag_text = "_".join(_sanitize_name(tag) for tag in record.tag_folders) or "subject_root"
        input_stem = record.output_name_stem or record.image_path.stem
        memmap_name = f"{_sanitize_name(record.subject_id)}_{tag_text}_{_sanitize_name(input_stem)}"
        if verbose:
            chain = "/".join(record.tag_folders) if record.tag_folders else "subject_root"
            if record.input_kind == "folder_stack":
                folder_names = ", ".join(path.name for path in record.stack_folder_paths)
                print(
                    f"Registering {record.subject_id}/{chain}/"
                    f"{record.stack_folder_tag}_* folder stack ({folder_names})",
                    flush=True)
            else:
                print(
                    f"Registering {record.subject_id}/{chain}/{record.image_path.name}",
                    flush=True)

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
            if record.input_kind == "folder_stack":
                stack, metadata = _load_stack_folder_record(
                    record,
                    load_options=load_options,
                    image_patterns=image_patterns,
                    exclude_name_contains=exclude_name_contains,
                    verbose=verbose)
            else:
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
            print(f"  load done; input shape: {stack.shape} (TZCYX)", flush=True)

        if save_merged_stack and record.input_kind == "folder_stack":
            merged_output_path = _merged_stack_path_for_record(
                record.output_scope_dir,
                record,
                merged_stack_name=merged_stack_name,
                merged_stack_suffix=merged_stack_suffix)
            merged_save_options = dict(base_save_kwargs)
            merged_save_options.setdefault(
                "metadata",
                _metadata_for_batch_output(metadata, merged_output_path))
            merged_save_options.setdefault("overwrite", False)
            try:
                if verbose:
                    print(f"  writing merged folder-stack OME-TIFF: {merged_output_path}", flush=True)
                save_stack(merged_output_path, stack, **merged_save_options)
            except Exception as exc:
                if not continue_on_error:
                    raise
                reason = f"{type(exc).__name__} during merged-stack save_stack: {exc}"
                record_skip(record, reason=reason, stage="save", output_dir=output_dir)
                if cleanup_cache_after_save and use_memmap:
                    cleanup_omio_cache(memmap_cache_dir, full_cleanup=True, verbose=False)
                continue

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

        if verbose:
            print("  registration done.", flush=True)

        save_options = dict(base_save_kwargs)
        save_options.setdefault("metadata", _metadata_for_batch_output(metadata, output_path))
        save_options.setdefault("registration_details", details)

        try:
            if verbose:
                print(
                    "  writing registered OME-TIFF and ZenReg report sidecars...",
                    flush=True,
                )
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
            print(f"  save done; wrote: {written_path}", flush=True)
        run_report_events.append(
            {
                "record": record,
                "status": "processed",
                "reason": None,
                "output_path": written_path,
            }
        )

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

    run_report_yaml_path = None
    run_report_txt_path = None
    if write_run_report:
        requested_formats = (
            {str(run_report_format).lower()}
            if isinstance(run_report_format, str)
            else {str(item).lower() for item in run_report_format}
        )
        if not requested_formats <= {"yaml", "txt"}:
            raise ValueError(
                "run_report_format must contain only 'yaml' and/or 'txt'. "
                f"Got {run_report_format!r}."
            )
        run_report_yaml_path, run_report_txt_path = _run_report_paths(root, run_report_name)
        payload = _load_run_report(run_report_yaml_path, root)
        payload["project_root"] = str(root)
        payload["last_updated"] = timestamp
        for event in run_report_events:
            _append_run_report_entry(
                payload,
                event["record"],
                root=root,
                timestamp=timestamp,
                status=event["status"],
                register_kwargs=base_register_kwargs,
                output_path=event.get("output_path"),
                reason=event.get("reason"),
            )
        if "yaml" in requested_formats:
            _write_run_report_yaml(run_report_yaml_path, payload)
        else:
            run_report_yaml_path = None
        if "txt" in requested_formats:
            _render_run_report_text(
                payload,
                run_report_txt_path,
                status_symbol_style=run_report_status_symbol_style,
            )
        else:
            run_report_txt_path = None

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
        root_run_report_yaml_path=run_report_yaml_path,
        root_run_report_txt_path=run_report_txt_path,
    )

def batch_create_thorlabs_raw_yaml_templates(
    project_root: str | Path,
    *,
    report_name: str | Path | None = None,
    raw_template_metadata: dict | None = None,
    overwrite_existing: bool = False,
    verbose: bool = True,
) -> BatchRawYamlTemplateResult:
    """
    Create OMIO Thorlabs RAW YAML templates from a ZenReg batch error report.

    The root-level error report written by :func:`register_bids_like_batch`
    contains a copy-pasteable ``ZENREG_BATCH_SKIPPED_RAW_FILES`` dictionary with
    RAW paths and editable ``template_metadata`` blocks. This helper reads that
    report and calls ``omio.create_thorlabs_raw_yaml`` for each RAW file listed
    in the report.

    Parameters
    ----------
    project_root : str or pathlib.Path
        Root folder containing the ZenReg batch error report and subject
        folders.
    report_name : str, pathlib.Path, or None, optional
        Report filename or path. If None, the latest
        ``zenreg_batch_error_report_*.txt`` in ``project_root`` is used.
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

    om = _import_omio()
    records: list[BatchRawYamlTemplateRecord] = []
    for entry in entries:
        raw_path = Path(entry["path"])
        template_metadata = dict(entry.get("template_metadata", fallback_metadata))
        yaml_paths = _expected_raw_yaml_paths(raw_path)
        existing_yaml_paths = [path for path in yaml_paths if path.exists()]

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
