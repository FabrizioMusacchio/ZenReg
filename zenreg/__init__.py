"""
ZenReg: lightweight registration helpers for microscopy image stacks.

The package starts from the registration helpers developed in the
spectral-unmixing project and exposes them as an independent, reusable module.

Author: Fabrizio Musacchio
Date: June 2026
"""

from .filters import apply_filters, max_z_project, z_project
from .io import (
    cleanup_omio_cache,
    create_empty_stack,
    create_stack_metadata,
    load_stack,
    save_stack,
    update_stack_metadata,
)
from .normcorre import register_stack_normcorre
from .registration import correct_intra_stack_z_drift, register_stack
from .reporting import write_registration_outputs

__all__ = [
    "apply_filters",
    "cleanup_omio_cache",
    "correct_intra_stack_z_drift",
    "create_empty_stack",
    "create_stack_metadata",
    "load_stack",
    "max_z_project",
    "register_stack_normcorre",
    "register_stack",
    "save_stack",
    "update_stack_metadata",
    "write_registration_outputs",
    "z_project",
]

__version__ = "0.0.1"
