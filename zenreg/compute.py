"""
Compute-resource helpers for ZenReg.

Author: Fabrizio Musacchio
Date: July 2026
"""
# %% IMPORTS
from __future__ import annotations

import os
# %% HELPER FUNCTIONS
def available_cpu_count() -> int:
    """
    Return the number of CPU workers visible to the current Python process.

    This mirrors the CPU count used by ``register_stack(..., n_jobs=-1)`` and is
    therefore the most direct helper for choosing a fixed ``n_jobs`` value in
    notebooks, scripts, containers, and HPC jobs.
    """

    return int(os.cpu_count() or 1)

def print_available_compute() -> int:
    """
    Print a compact CPU availability summary and return the CPU worker count.

    Examples
    --------
    >>> n_cpus = print_available_compute()
    >>> # register_stack(..., n_jobs=-1) uses all reported CPUs.
    >>> # register_stack(..., n_jobs=max(1, n_cpus // 2)) keeps some headroom.
    """

    n_cpus = available_cpu_count()
    print(f"ZenReg available CPU workers: {n_cpus}")
    print("Use n_jobs=-1 to use all available CPU workers.")
    return n_cpus
# %% END