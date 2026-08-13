"""
Batch-register a small BIDS-like synthetic ZenReg project.

Run this script cell-by-cell in VS Code's interactive window. It creates a
minimal project tree in ``example_data/synthetic_batch_project`` and then shows
two batch-processing patterns:

1. a simple custom loop around ``load_stack -> register_stack -> save_stack``;
2. ZenReg's built-in ``register_bids_like_batch`` processor.

The built-in processor is the recommended starting point for real projects once
your folder tree follows the BIDS-like ``project_root / subject / experiment``
pattern.

Author: Fabrizio Musacchio
Date: August 2026
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

from zenreg import (
    cleanup_omio_cache,
    discover_bids_like_batch_images,
    load_stack,
    print_available_compute,
    register_bids_like_batch,
    register_stack,
    save_stack)
from zenreg.synthetic import write_batch_example_project
# %% PATHS AND PROJECT CREATION
BATCH_PROJECT_ROOT = PROJECT_ROOT / "example_data" / "synthetic_batch_project"
MEMMAP_CACHE_DIR   = BATCH_PROJECT_ROOT / "omio_memmap_cache"
AVAILABLE_CPUS     = print_available_compute()

write_batch_example_project(
    BATCH_PROJECT_ROOT,
    subject_ids=("ID000001", "ID000002"),
    experiment_tags=("TP000", "TP001"),)
# %% DISCOVER INPUT FILES
records = discover_bids_like_batch_images(
    BATCH_PROJECT_ROOT,
    subject_ids=None,  # None discovers all folders starting with subject_prefix
    subject_prefix="ID",
    tag_folder_levels=(("TP000", "TP001"),),
    image_patterns=("*.ome.tif",))

print("Files selected for batch registration:")
for record in records:
    tag_text = "/".join(record.tag_folders)
    print(f"  {record.subject_id}/{tag_text}: {record.image_path.name}")
# %% OPTION A: SIMPLE CUSTOM LOOP
RUN_CUSTOM_LOOP = False

if RUN_CUSTOM_LOOP:
    for record in records:
        output_dir = record.image_path.parent / "zenreg_custom_loop"
        output_path = output_dir / f"{record.image_path.stem}_zenreg_registered.ome.tif"

        cleanup_omio_cache(MEMMAP_CACHE_DIR, full_cleanup=True)
        stack, metadata = load_stack(
            record.image_path,
            return_metadata=True,
            use_memmap=True,
            memmap_folder=MEMMAP_CACHE_DIR,
            memmap_reuse=True)

        registered, details = register_stack(
            stack,
            registration_channel=0,
            method="phase_cross_correlation",
            time_registration_mode="projection",
            time_reference_mode="template",
            projection_method="max",
            zreg=False,
            zero_clip=True,
            max_xy_shifts=(8, 8),
            transform_backend="skimage",
            transform_order=1,
            output_use_memmap=True,
            output_memmap_folder=MEMMAP_CACHE_DIR,
            output_memmap_name=f"{record.subject_id}_{'_'.join(record.tag_folders)}_registered",
            n_jobs=max(1, min(AVAILABLE_CPUS, 4)),
            verbose=True,
            return_shifts=True,
            return_details=True)

        written_path = save_stack(
            output_path,
            registered,
            metadata=metadata,
            registration_details=details)
        print(f"Wrote {record.subject_id}/{record.experiment_tag}: {written_path}")
        cleanup_omio_cache(MEMMAP_CACHE_DIR, full_cleanup=True)
# %% OPTION B: BUILT-IN BIDS-LIKE BATCH PROCESSOR
RUN_ZENREG_BATCH_PROCESSOR = True

if RUN_ZENREG_BATCH_PROCESSOR:
    batch_result = register_bids_like_batch(
        BATCH_PROJECT_ROOT,
        subject_ids                 = None,  # None discovers all folders starting with "ID"
        subject_prefix              = "ID",
        tag_folder_levels           = (("TP000", "TP001"),),
        # image_patterns              = ("*.ome.tif",), # uncomment if you want to restrict to specific image files
        output_folder_name          = "zenreg_output",
        skip_registered             = False,
        use_memmap                  = True,
        memmap_folder_name          = "omio_memmap_cache",
        memmap_reuse                = True,
        cleanup_cache_before_load   = True,
        cleanup_cache_after_save    = True,
        load_kwargs = {"on_error": "return_none",
                       "verbose": False,},
        register_kwargs={
            "registration_channel":     0,
            "method":                   "phase_cross_correlation",
            "time_registration_mode":   "projection",
            "time_reference_mode":      "template",
            "projection_method":        "max",
            "zreg":                     False,
            "zero_clip":                True,
            "max_xy_shifts":            (8, 8),
            "transform_backend":        "skimage",
            "transform_order":          1,
            "n_jobs":                   max(1, min(AVAILABLE_CPUS, 4)),
            "verbose":                  True,
        },
        save_kwargs={"compression_level":   3,
                     "overwrite":           True,
                     "verbose":             False,},
        verbose=True,
    )

    print(f"Processed files: {len(batch_result.processed)}")
    print(f"Skipped files:   {len(batch_result.skipped)}")
# %% END
