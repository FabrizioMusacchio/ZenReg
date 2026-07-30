"""
Batch-register a small BIDS-like synthetic ZenReg project.

Run this script cell-by-cell in VS Code's interactive window. It creates a
minimal project tree in ``example_data/synthetic_batch_project`` and then shows
how to batch-process all matching images with the usual ZenReg sequence:

    load_stack -> register_stack -> save_stack

The same loop can be adapted to real projects with subject folders such as
``ID000001`` and experiment/time-point folders such as ``TP000`` or ``TP001``.

Author: Fabrizio Musacchio
Date: July 2026
"""
# %% IMPORTS
import os
import sys
import tempfile
from pathlib import Path

SCRIPT_CACHE_DIR = Path(
    os.environ.get(
        "ZENREG_OMIO_CACHE_DIR",
        Path(tempfile.gettempdir()) / "zenreg-omio-cache",
    )
)
SCRIPT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_CACHE_DIR / "matplotlib"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(SCRIPT_CACHE_DIR / "numba"))

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from zenreg import (
    cleanup_omio_cache,
    iter_bids_like_image_files,
    load_stack,
    print_available_compute,
    register_stack,
    save_stack,
)
from zenreg.synthetic import write_batch_example_project

# %% PATHS AND PROJECT CREATION
BATCH_PROJECT_ROOT = PROJECT_ROOT / "example_data" / "synthetic_batch_project"
MEMMAP_CACHE_DIR = BATCH_PROJECT_ROOT / "omio_memmap_cache"
AVAILABLE_CPUS = print_available_compute()

write_batch_example_project(
    BATCH_PROJECT_ROOT,
    subject_ids=("ID000001", "ID000002"),
    experiment_tags=("TP000", "TP001"),
)

# %% DISCOVER INPUT FILES
# Use subject_ids=None to process every folder starting with "ID".
# Alternatively, pass a concrete list such as ("ID000001",).
records = iter_bids_like_image_files(
    BATCH_PROJECT_ROOT,
    subject_ids=None,
    experiment_tags=("TP000", "TP001"),
    subject_prefix="ID",
    experiment_prefix="TP",
)

print("Files selected for batch registration:")
for record in records:
    print(f"  {record.subject_id}/{record.experiment_tag}: {record.image_path.name}")

# %% BATCH REGISTER
for record in records:
    output_dir = record.image_path.parent / "zenreg_registered"
    output_path = output_dir / record.image_path.name

    cleanup_omio_cache(MEMMAP_CACHE_DIR, full_cleanup=True)
    stack, metadata = load_stack(
        record.image_path,
        return_metadata=True,
        use_memmap=True,
        memmap_folder=MEMMAP_CACHE_DIR,
        memmap_reuse=True,
    )

    registered, details = register_stack(
        stack,
        registration_channel=0,  # channel used to estimate shifts
        registration_stack=0,  # reference time point
        method="phase_cross_correlation",  # "phase_cross_correlation", "pystackreg", or "normcorre"
        time_registration_mode="projection",  # register YX projections over time
        time_reference_mode="template",  # register every t to registration_stack
        projection_method="max",  # "max", "mean", "median", "var", or "std"
        zreg=False,  # 2D+t example: no Z shift estimation
        zero_clip=True,  # crop translation-introduced zero borders
        max_xy_shifts=(8, 8),  # guard rail against implausibly large shifts
        transform_backend="skimage",  # "skimage" or "scipy"
        transform_order=1,  # 1 for intensity data, 0 for sparse labels/puncta
        output_use_memmap=True,  # write the intermediate registered result to disk-backed Zarr
        output_memmap_folder=MEMMAP_CACHE_DIR,
        output_memmap_name=f"{record.subject_id}_{record.experiment_tag}_registered",
        n_jobs=max(1, min(AVAILABLE_CPUS, 4)),  # use a few local CPU workers
        verbose=True,
        return_shifts=True,
        return_details=True,
    )

    written_path = save_stack(
        output_path,
        registered,
        metadata=metadata,
        registration_details=details,
    )
    print(f"Wrote {record.subject_id}/{record.experiment_tag}: {written_path}")
    cleanup_omio_cache(MEMMAP_CACHE_DIR, full_cleanup=True)

# %% END
