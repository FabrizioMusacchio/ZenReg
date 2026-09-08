"""
Interactive stacked-folder batch registration example for ZenReg.

Run this script cell-by-cell in VS Code's interactive window. It creates a
small synthetic BIDS-like project with repeated OV folders, each containing a
shifted 3D image stack with the same filename. The batch processor then reads
the OV folder family, merges it along T, registers the merged 3D+t stack, and
writes the registered output plus ZenReg reports.

author: Fabrizio Musacchio
date:   September 2026
"""
# %% IMPORTS
import os
import sys
import tempfile
from pathlib import Path

SCRIPT_CACHE_DIR = Path(
    os.environ.get("ZENREG_OMIO_CACHE_DIR", Path(tempfile.gettempdir()) / "zenreg-omio-cache"))
SCRIPT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_CACHE_DIR / "matplotlib"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(SCRIPT_CACHE_DIR / "numba"))

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from additional_scripts.create_synthetic_stacked_folder_batch_project import (
    write_synthetic_stacked_folder_batch_project)
from zenreg import (
    cleanup_omio_cache,
    discover_bids_like_batch_images,
    load_stack,
    open_in_napari,
    print_available_compute,
    register_bids_like_batch,
    show_timepoints)
# %% PATHS AND COMMON SETTINGS
BATCH_PROJECT_ROOT = PROJECT_ROOT / "example_data" / "synthetic_stacked_folder_batch_project"
AVAILABLE_CPUS = print_available_compute()
N_JOBS = max(1, min(AVAILABLE_CPUS, 4))

RUN_SYNTHETIC_DATA_GENERATOR = True
RUN_STACKED_FOLDER_BATCH = True
OPEN_IN_NAPARI = False
# %% CREATE SYNTHETIC STACKED-FOLDER PROJECT
if RUN_SYNTHETIC_DATA_GENERATOR:
    write_synthetic_stacked_folder_batch_project(
        BATCH_PROJECT_ROOT,
        subject_ids=("ID000001", "ID000002"),
        fov_names=("FOV1_pre", "FOV2_pre"),
        ov_shifts_zyx=((0, 0, 0), (1, 4, -3), (-2, -5, 5), (2, 2, 7)),
        shape_zyx=(16, 128, 128),
        channel_count=2,
        filename="image_001.ome.tif",
        noise_sigma=0.015,
        overwrite=True)
# %% DISCOVER STACKED-FOLDER RECORDS
records = discover_bids_like_batch_images(
    BATCH_PROJECT_ROOT,
    subject_ids=None,
    subject_prefix="ID",
    tag_folder_levels=(("FOV",),),
    stack_folder_tag="OV",
    stack_folder_match="startswith",
    stack_folder_merge_axis="T",
    image_patterns=("*.ome.tif",),
    exclude_name_contains=("ROIMask.raw",))

print("Stacked-folder records selected for registration:")
for record in records:
    folder_names = ", ".join(path.name for path in record.stack_folder_paths)
    print(f"  {record.subject_id}/{'/'.join(record.tag_folders)}/{record.stack_folder_tag}_*: {folder_names}")
# %% OPTIONAL: READ ONE SOURCE STACK FOR A QUICK SANITY CHECK
first_source_stack, first_source_metadata = load_stack(
    records[0].stack_folder_paths[0] / "image_001.ome.tif",
    return_metadata=True,
    verbose=False)
print(f"First OV source stack shape: {first_source_stack.shape} (TZCYX)")

if OPEN_IN_NAPARI:
    open_in_napari(
        first_source_stack,
        first_source_metadata,
        fname="Synthetic stacked-folder source OV_1",
        enabled=True)
# %% RUN STACKED-FOLDER BATCH REGISTRATION
if RUN_STACKED_FOLDER_BATCH:
    batch_result = register_bids_like_batch(
        BATCH_PROJECT_ROOT,
        subject_ids=None,
        subject_prefix="ID",
        tag_folder_levels=(("FOV",),),
        stack_folder_tag="OV",
        stack_folder_match="startswith",
        stack_folder_merge_axis="T",
        save_merged_stack=True,
        image_patterns=("*.ome.tif",),
        output_folder_name="zenreg_output",
        skip_registered=False,
        use_memmap=True,
        memmap_folder_name="omio_memmap_cache",
        memmap_reuse=True,
        cleanup_cache_before_load=True,
        cleanup_cache_after_save=True,
        load_kwargs={"on_error": "return_none",
                     "verbose": False},
        register_kwargs={
            "registration_channel": 0,
            "method": "phase_cross_correlation",
            "time_registration_mode": "full_3d",
            "time_reference_mode": "template",
            "registration_template_time_range": "all",
            "projection_method": "median",
            "zreg": True,
            "zero_clip": True,
            "zero_clip_margin": (0, 0, 0),
            "max_xy_shifts": (12, 12),
            "max_z_shifts": 4,
            "transform_backend": "skimage",
            "transform_order": 1,
            "filter_slices": False,
            "filter_projections": False,
            "median_kernel_size": 3,
            "n_jobs": N_JOBS,
            "verbose": True,
            "return_details": True},
        save_kwargs={"compression_level": 3,
                     "overwrite": True,
                     "verbose": False},
        write_error_reports=True,
        write_run_report=True,
        run_report_name="zenreg_batch_run_report",
        run_report_format=("yaml", "txt"),
        continue_on_error=True,
        verbose=True)

    print(f"Processed files: {len(batch_result.processed)}")
    print(f"Skipped files:   {len(batch_result.skipped)}")
    print(f"Run report YAML: {batch_result.root_run_report_yaml_path}")
    print(f"Run report TXT:  {batch_result.root_run_report_txt_path}")
# %% INSPECT ONE MERGED AND REGISTERED RESULT
merged_path = (
    BATCH_PROJECT_ROOT
    / "ID000001"
    / "FOV1_pre"
    / "FOV1_pre_OV_merged_T.ome.tif")
registered_path = (
    BATCH_PROJECT_ROOT
    / "ID000001"
    / "FOV1_pre"
    / "zenreg_output"
    / "FOV1_pre_OV_merged_T_zenreg_registered.ome.tif")

if merged_path.exists():
    merged_stack, merged_metadata = load_stack(
        merged_path,
        return_metadata=True,
        verbose=False)
    print(f"Merged pre-registration stack shape: {merged_stack.shape} (TZCYX)")
    show_timepoints(
        merged_stack,
        channel=0,
        reference_time=0,
        moving_time=min(2, merged_stack.shape[0] - 1),
        projection_method="max",
        projection_z_range="all",
        title="Synthetic stacked-folder merged input")

if registered_path.exists():
    registered_stack, registered_metadata = load_stack(
        registered_path,
        return_metadata=True,
        verbose=False)
    print(f"Registered stack shape: {registered_stack.shape} (TZCYX)")
    if OPEN_IN_NAPARI:
        open_in_napari(
            registered_stack,
            registered_metadata,
            fname="Synthetic stacked-folder registered",
            enabled=True)
# %% OPTIONAL CACHE CLEANUP
# cleanup_omio_cache(BATCH_PROJECT_ROOT, full_cleanup=True, verbose=False)
# %% END
