"""
ZenReg: lightweight registration helpers for microscopy image stacks.

The package starts from the registration helpers developed in the
spectral-unmixing project and exposes them as an independent, reusable module.

Author: Fabrizio Musacchio
Date: June 2026
"""

from .filters import apply_filters, max_z_project
from .io import load_stack, save_stack
from .registration import correct_intra_stack_z_drift, register_stack

__all__ = [
    "apply_filters",
    "correct_intra_stack_z_drift",
    "load_stack",
    "max_z_project",
    "register_stack",
    "save_stack",
]

__version__ = "0.0.1"
