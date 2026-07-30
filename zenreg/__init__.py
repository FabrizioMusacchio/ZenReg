"""
ZenReg: lightweight registration helpers for microscopy image stacks.

The package starts from the registration helpers developed in the
spectral-unmixing project and exposes them as an independent, reusable module.

Author: Fabrizio Musacchio
Date: June 2026
"""

from .batch import BatchImageRecord, DEFAULT_IMAGE_EXTENSIONS, iter_bids_like_image_files
from .compute import available_cpu_count, print_available_compute
from .filters import apply_filters, max_z_project, z_project
from .io import (
    cleanup_omio_cache,
    create_empty_stack,
    create_stack_metadata,
    crop_stack,
    load_stack,
    save_stack,
    update_stack_metadata,
)
from .normcorre import plot_normcorre_patch_overlay, register_stack_normcorre
from .profiling import MemoryTracker, profile_memory
from .rigid3d import register_stack_rigid_3d
from .registration import correct_intra_stack_z_drift, register_stack
from .reporting import write_registration_outputs
from .tutorial import (
    load_csv_table,
    load_expected_rigid_corrections,
    load_expected_rigid_z_rotation,
    load_expected_slice_registration_shifts,
    load_expected_time_registration_rotations,
    load_expected_time_registration_shifts,
    maybe_open_in_napari,
    print_caiman_patch_summary,
    print_local_patch_summary,
    print_residual_mae_summary,
    print_rigid_comparison,
    print_shift_comparison,
    show_before_after,
    show_residual_comparison,
    show_residual_comparison_multi,
    show_slices,
    show_timepoints,
)

__all__ = [
    "apply_filters",
    "available_cpu_count",
    "BatchImageRecord",
    "cleanup_omio_cache",
    "correct_intra_stack_z_drift",
    "create_empty_stack",
    "create_stack_metadata",
    "DEFAULT_IMAGE_EXTENSIONS",
    "crop_stack",
    "load_stack",
    "iter_bids_like_image_files",
    "load_csv_table",
    "load_expected_rigid_corrections",
    "load_expected_rigid_z_rotation",
    "load_expected_slice_registration_shifts",
    "load_expected_time_registration_rotations",
    "load_expected_time_registration_shifts",
    "max_z_project",
    "maybe_open_in_napari",
    "MemoryTracker",
    "print_available_compute",
    "print_caiman_patch_summary",
    "print_local_patch_summary",
    "print_residual_mae_summary",
    "print_rigid_comparison",
    "print_shift_comparison",
    "profile_memory",
    "plot_normcorre_patch_overlay",
    "register_stack_normcorre",
    "register_stack_rigid_3d",
    "register_stack",
    "save_stack",
    "show_before_after",
    "show_residual_comparison",
    "show_residual_comparison_multi",
    "show_slices",
    "show_timepoints",
    "update_stack_metadata",
    "write_registration_outputs",
    "z_project",
]

__version__ = "0.0.2"
