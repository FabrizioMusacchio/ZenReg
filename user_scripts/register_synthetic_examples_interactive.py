"""
Interactive VS Code script for testing ZenReg on synthetic 2D and 3D stacks.

Run this script cell-by-cell in VS Code's interactive window. If the example
data do not exist yet, first run:

    python additional_scripts/create_synthetic_example_data.py

Author: Fabrizio Musacchio
Date: June 2026
"""
# %% IMPORTS
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# path setup (only used during development):
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from zenreg import correct_intra_stack_z_drift, load_stack, max_z_project, register_stack, save_stack
# %% DEFINE INPUT AND OUTPUT PATHS
EXAMPLE_DIR = PROJECT_ROOT / "example_data"
OUTPUT_DIR = EXAMPLE_DIR / "registered"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STACK_2D_PATH = EXAMPLE_DIR / "motion_distorted_2d_tzcyx.tif"
STACK_3D_PATH = EXAMPLE_DIR / "motion_distorted_3d_tzcyx.tif"

OUTPUT_2D_PATH = OUTPUT_DIR / "motion_distorted_2d_registered.tif"
OUTPUT_3D_PATH = OUTPUT_DIR / "motion_distorted_3d_zcorrected_registered.tif"
# %% QUICK VIEW HELPER
def show_timepoints(stack, *, title: str, channel: int = 0) -> None:
    """Show t=0, t=1, and their difference as max-Z projections."""

    stack = np.asarray(stack)
    if stack.shape[0] < 2:
        raise ValueError("show_timepoints requires at least two time points (T >= 2).")

    projection_t0 = np.max(stack[0, :, channel, :, :], axis=0)
    projection_t1 = np.max(stack[1, :, channel, :, :], axis=0)
    projection_diff = projection_t1 - projection_t0

    fig, axes = plt.subplots(1, 3, figsize=(10, 3), constrained_layout=True)
    axes[0].imshow(projection_t0, cmap="gray")
    axes[0].set_title("t=0")
    axes[0].axis("off")

    axes[1].imshow(projection_t1, cmap="gray")
    axes[1].set_title("t=1")
    axes[1].axis("off")

    max_abs_diff = float(np.max(np.abs(projection_diff)))
    if max_abs_diff == 0:
        max_abs_diff = 1.0
    axes[2].imshow(projection_diff, cmap="bwr", vmin=-max_abs_diff, vmax=max_abs_diff)
    axes[2].set_title("t1 - t0")
    axes[2].axis("off")

    fig.suptitle(title)
    plt.show()
# %% LOAD 2D SYNTHETIC STACK
stack_2d = load_stack(STACK_2D_PATH)
print(f"2D stack shape: {stack_2d.shape} (TZCYX)")
show_timepoints(stack_2d, title="2D synthetic stack before registration")
# %% REGISTER 2D STACK ACROSS TIME
registered_2d, shifts_2d = register_stack(
    stack_2d,
    registration_channel=0,
    method="pystackreg", # phase_cross_correlation or pystackreg
    pre_median_filter=True,
    post_median_filter=True,
    median_kernel_size=3,
    verbose=True,
    return_shifts=True)
print("Estimated 2D time shifts:")
print(shifts_2d)
show_timepoints(registered_2d, title="2D synthetic stack after registration")
save_stack(OUTPUT_2D_PATH, registered_2d)
# %% LOAD 3D SYNTHETIC STACK
stack_3d = load_stack(STACK_3D_PATH)
print(f"3D stack shape: {stack_3d.shape} (TZCYX)")
show_timepoints(stack_3d, title="3D synthetic stack before correction")
# %% CORRECT INTRA-STACK Z DRIFT
z_corrected_3d, z_shifts_3d = correct_intra_stack_z_drift(
    stack_3d,
    registration_channel=0,
    method="phase_cross_correlation",
    reference_mode="full_projection", # neighbor or full_projection
    neighbor_window_size=3,
    pre_median_filter=True,
    post_median_filter=True,
    median_kernel_size=3,
    verbose=True,
    return_shifts=True)
print(f"Estimated Z shifts shape: {z_shifts_3d.shape}")
show_timepoints(z_corrected_3d, title="3D synthetic stack after intra-stack Z correction")
# %% REGISTER 3D STACK ACROSS TIME
registered_3d, time_shifts_3d = register_stack(
    z_corrected_3d,
    registration_channel=0,
    method="phase_cross_correlation",
    zrange=None,
    pre_median_filter=True,
    post_median_filter=True,
    median_kernel_size=3,
    verbose=True,
    return_shifts=True)
print("Estimated 3D time shifts:")
print(time_shifts_3d)
show_timepoints(registered_3d, title="3D synthetic stack after Z correction and time registration")
save_stack(OUTPUT_3D_PATH, registered_3d)
# %% PROJECT REGISTERED 3D STACK FOR QUICK INSPECTION
projected_3d = max_z_project(registered_3d)
print(f"Projected 3D stack shape: {projected_3d.shape}")
show_timepoints(projected_3d, title="Projected registered 3D stack")
# %% END
