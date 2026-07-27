"""
Interactive VS Code script for testing ZenReg on synthetic 2D and 3D stacks.

Run this script cell-by-cell in VS Code's interactive window. If the example
data do not exist yet, first run:

    python additional_scripts/create_synthetic_example_data.py

Author: Fabrizio Musacchio
Date: June 2026
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
if not os.access(Path.home(), os.W_OK):
    script_home = SCRIPT_CACHE_DIR / "home"
    script_home.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HOME", str(script_home))

import matplotlib.pyplot as plt
import numpy as np

# path setup (only used during development):
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from zenreg import correct_intra_stack_z_drift, load_stack, register_stack, save_stack, z_project

# %% DEFINE INPUT AND OUTPUT PATHS
EXAMPLE_DIR = PROJECT_ROOT / "example_data" / "synthetic_data"
OUTPUT_DIR = EXAMPLE_DIR / "registered"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STACK_2D_PATH = EXAMPLE_DIR / "motion_distorted_2d_tzcyx.ome.tif"
STACK_3D_PATH = EXAMPLE_DIR / "motion_distorted_3d_tzcyx.ome.tif"
GT_2D_TIME_PATH = EXAMPLE_DIR / "motion_distorted_2d_time_shifts_gt.csv"
GT_3D_TIME_PATH = EXAMPLE_DIR / "motion_distorted_3d_time_shifts_gt.csv"
GT_3D_SLICE_PATH = EXAMPLE_DIR / "motion_distorted_3d_slice_shifts_gt.csv"

OUTPUT_2D_PATH = OUTPUT_DIR / "motion_distorted_2d_registered.ome.tif"
OUTPUT_3D_PATH = OUTPUT_DIR / "motion_distorted_3d_zcorrected_registered.ome.tif"
# %% REGISTRATION SETTINGS
REGISTRATION_STACK = 0
REGISTRATION_CHANNEL = 0
PROJECTION_METHOD = "max"  # max, mean, median, var, std
REGISTRATION_METHOD = "phase_cross_correlation"  # phase_cross_correlation or pystackreg
# %% QUICK VIEW HELPER
def show_timepoints(
    stack,
    *,
    title: str,
    channel: int = 0,
    projection_method: str = "max",
) -> None:
    """Show t=0, t=1, and their difference as Z projections."""

    stack = np.asarray(stack)
    if stack.shape[0] < 2:
        raise ValueError("show_timepoints requires at least two time points (T >= 2).")

    projection_t0 = z_project(
        stack[0:1, :, channel : channel + 1, :, :],
        projection_method=projection_method,
    )[0, 0, 0, :, :]
    projection_t1 = z_project(
        stack[1:2, :, channel : channel + 1, :, :],
        projection_method=projection_method,
    )[0, 0, 0, :, :]
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


def load_expected_time_registration_shifts(path: Path, *, registration_stack: int = 0) -> np.ndarray:
    """Load expected correction shifts from the synthetic GT time-shift table."""

    table = np.genfromtxt(path, delimiter=",", names=True)
    shift_y_column = f"expected_registration_shift_y_ref_t{registration_stack}"
    shift_x_column = f"expected_registration_shift_x_ref_t{registration_stack}"
    return np.column_stack([table[shift_y_column], table[shift_x_column]]).astype(np.float32)


def print_shift_comparison(name: str, estimated_shifts: np.ndarray, expected_shifts: np.ndarray) -> None:
    """Print a compact comparison between estimated correction shifts and GT."""

    delta = np.asarray(estimated_shifts, dtype=np.float32) - np.asarray(expected_shifts, dtype=np.float32)
    print(f"{name}:")
    print(f"  mean abs error y/x: {np.mean(np.abs(delta), axis=0)}")
    print(f"  max abs error y/x:  {np.max(np.abs(delta), axis=0)}")
    print("  first rows [estimated_y, estimated_x, expected_y, expected_x, delta_y, delta_x]:")
    print(np.column_stack([estimated_shifts, expected_shifts, delta])[:5])


def compare_time_registration_backends(stack, expected_shifts, *, title: str) -> None:
    """Compare both time-registration backends against the synthetic GT table."""

    for method in ("phase_cross_correlation", "pystackreg"):
        _, shifts = register_stack(
            stack,
            registration_channel=REGISTRATION_CHANNEL,  # channel used to estimate shifts
            registration_stack=REGISTRATION_STACK,  # reference time point/template
            method=method,  # "phase_cross_correlation" or "pystackreg"
            zrange=None,
            projection_method=PROJECTION_METHOD,  # "max", "mean", "median", "var", or "std"
            filter_slices=True,  # median-filter Z slices before projection
            filter_projections=True,  # median-filter projections before shift estimation
            median_kernel_size=3,  # median-filter kernel size in pixels
            verbose=False,
            return_shifts=True,
        )
        print_shift_comparison(f"{title} / {method}", shifts, expected_shifts)
# %% LOAD 2D SYNTHETIC STACK
stack_2d, metadata_2d = load_stack(STACK_2D_PATH, return_metadata=True, verbose=False)
expected_2d_time_shifts = load_expected_time_registration_shifts(
    GT_2D_TIME_PATH,
    registration_stack=REGISTRATION_STACK,
)
print(f"2D stack shape: {stack_2d.shape} (TZCYX)")
show_timepoints(
    stack_2d,
    title="2D synthetic stack before registration",
    channel=REGISTRATION_CHANNEL,
    projection_method=PROJECTION_METHOD,
)
# %% COMPARE 2D BACKENDS AGAINST GT
compare_time_registration_backends(stack_2d, expected_2d_time_shifts, title="2D time registration")
# %% REGISTER 2D STACK ACROSS TIME
registered_2d, shifts_2d = register_stack(
    stack_2d,
    registration_channel=REGISTRATION_CHANNEL,  # channel used to estimate shifts
    registration_stack=REGISTRATION_STACK,  # reference time point/template
    method=REGISTRATION_METHOD,  # "phase_cross_correlation" or "pystackreg"
    projection_method=PROJECTION_METHOD,  # "max", "mean", "median", "var", or "std"
    filter_slices=True,  # median-filter Z slices before projection
    filter_projections=True,  # median-filter projections before shift estimation
    median_kernel_size=3,  # median-filter kernel size in pixels
    verbose=True,
    return_shifts=True,
)
print("Estimated 2D time shifts:")
print(shifts_2d)
print_shift_comparison("2D selected backend", shifts_2d, expected_2d_time_shifts)
show_timepoints(
    registered_2d,
    title="2D synthetic stack after registration",
    channel=REGISTRATION_CHANNEL,
    projection_method=PROJECTION_METHOD,
)
written_2d_path = save_stack(OUTPUT_2D_PATH, registered_2d, metadata=metadata_2d)
print(f"Saved registered 2D stack to: {written_2d_path}")
# %% LOAD 3D SYNTHETIC STACK
stack_3d, metadata_3d = load_stack(STACK_3D_PATH, return_metadata=True, verbose=False)
expected_3d_time_shifts = load_expected_time_registration_shifts(
    GT_3D_TIME_PATH,
    registration_stack=REGISTRATION_STACK,
)
print(f"3D stack shape: {stack_3d.shape} (TZCYX)")
show_timepoints(
    stack_3d,
    title="3D synthetic stack before correction",
    channel=REGISTRATION_CHANNEL,
    projection_method=PROJECTION_METHOD,
)
# %% CORRECT INTRA-STACK Z DRIFT
z_corrected_3d, z_shifts_3d = correct_intra_stack_z_drift(
    stack_3d,
    registration_channel=REGISTRATION_CHANNEL,  # channel used to estimate shifts
    method="phase_cross_correlation",  # "phase_cross_correlation" or "pystackreg"
    reference_mode="full_projection",  # "neighbor" or "full_projection"
    neighbor_window_size=3,  # odd Z-window size for reference_mode="neighbor"
    projection_method=PROJECTION_METHOD,  # "max", "mean", "median", "var", or "std"
    filter_slices=True,  # median-filter Z slices before projection/reference creation
    filter_projections=True,  # median-filter images before shift estimation
    median_kernel_size=3,  # median-filter kernel size in pixels
    verbose=True,
    return_shifts=True,
)
print(f"Estimated Z shifts shape: {z_shifts_3d.shape}")
print(f"3D slice GT table: {GT_3D_SLICE_PATH}")
print(
    "Note: 3D slice GT stores the applied local shifts. The projection-based "
    "intra-stack reference is a diagnostic target, not a direct unshifted-slice GT."
)
show_timepoints(
    z_corrected_3d,
    title="3D synthetic stack after intra-stack Z correction",
    channel=REGISTRATION_CHANNEL,
    projection_method=PROJECTION_METHOD,
)
# %% COMPARE 3D TIME-REGISTRATION BACKENDS AGAINST GT
print(
    "Note: 3D time GT stores the global time shift. Time-dependent local Z drift "
    "can still deform the projection, so this comparison is a sanity check."
)
compare_time_registration_backends(z_corrected_3d, expected_3d_time_shifts, title="3D time registration")
# %% REGISTER 3D STACK ACROSS TIME
registered_3d, time_shifts_3d = register_stack(
    z_corrected_3d,
    registration_channel=REGISTRATION_CHANNEL,  # channel used to estimate shifts
    registration_stack=REGISTRATION_STACK,  # reference time point/template
    method=REGISTRATION_METHOD,  # "phase_cross_correlation" or "pystackreg"
    zrange=None,
    projection_method=PROJECTION_METHOD,  # "max", "mean", "median", "var", or "std"
    filter_slices=True,  # median-filter Z slices before projection
    filter_projections=True,  # median-filter projections before shift estimation
    median_kernel_size=3,  # median-filter kernel size in pixels
    verbose=True,
    return_shifts=True,
)
print("Estimated 3D time shifts:")
print(time_shifts_3d)
print_shift_comparison("3D selected backend", time_shifts_3d, expected_3d_time_shifts)
show_timepoints(
    registered_3d,
    title="3D synthetic stack after Z correction and time registration",
    channel=REGISTRATION_CHANNEL,
    projection_method=PROJECTION_METHOD,
)
written_3d_path = save_stack(OUTPUT_3D_PATH, registered_3d, metadata=metadata_3d)
print(f"Saved registered 3D stack to: {written_3d_path}")
# %% PROJECT REGISTERED 3D STACK FOR QUICK INSPECTION
projected_3d = z_project(registered_3d, projection_method=PROJECTION_METHOD)
print(f"Projected 3D stack shape: {projected_3d.shape}")
show_timepoints(
    projected_3d,
    title="Projected registered 3D stack",
    channel=REGISTRATION_CHANNEL,
    projection_method=PROJECTION_METHOD,
)
# %% END
