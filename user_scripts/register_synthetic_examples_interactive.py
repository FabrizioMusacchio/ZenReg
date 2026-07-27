"""
Interactive VS Code script for testing ZenReg on synthetic benchmark stacks.

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

from zenreg import load_stack, register_stack, save_stack, z_project
# %% DEFINE INPUT AND OUTPUT PATHS
EXAMPLE_DIR = PROJECT_ROOT / "example_data" / "synthetic_data"
OUTPUT_DIR = EXAMPLE_DIR / "registered"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STACK_2D_T_XY_PATH = EXAMPLE_DIR / "synthetic_2d_t_xy.ome.tif"
STACK_3D_Z_XY_PATH = EXAMPLE_DIR / "synthetic_3d_z_xy.ome.tif"
STACK_3D_T_XY_PATH = EXAMPLE_DIR / "synthetic_3d_t_xy.ome.tif"
STACK_3D_T_INTRA_XY_PATH = EXAMPLE_DIR / "synthetic_3d_t_intra_xy.ome.tif"
STACK_3D_T_ZYX_PATH = EXAMPLE_DIR / "synthetic_3d_t_zyx.ome.tif"

GT_2D_T_XY_PATH = EXAMPLE_DIR / "synthetic_2d_t_xy_time_shifts_gt.csv"
GT_3D_Z_XY_PATH = EXAMPLE_DIR / "synthetic_3d_z_xy_slice_shifts_gt.csv"
GT_3D_T_XY_PATH = EXAMPLE_DIR / "synthetic_3d_t_xy_time_shifts_gt.csv"
GT_3D_T_INTRA_XY_PATH = EXAMPLE_DIR / "synthetic_3d_t_intra_xy_slice_shifts_gt.csv"
GT_3D_T_ZYX_PATH = EXAMPLE_DIR / "synthetic_3d_t_zyx_time_shifts_gt.csv"
# %% QUICK VIEW AND GT HELPERS
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
        print(f"Skipping timepoint quick view for {title!r}: T < 2.")
        return

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

def show_slices(
    stack,
    *,
    title: str,
    channel: int = 0,
    z0: int = 0,
    z1: int = 6,
) -> None:
    """Show two slices from t=0 and their difference."""

    stack = np.asarray(stack)
    z1 = min(int(z1), stack.shape[1] - 1)
    image_z0 = stack[0, int(z0), int(channel), :, :]
    image_z1 = stack[0, z1, int(channel), :, :]
    diff = image_z1 - image_z0

    fig, axes = plt.subplots(1, 3, figsize=(10, 3), constrained_layout=True)
    axes[0].imshow(image_z0, cmap="gray")
    axes[0].set_title(f"z={z0}")
    axes[0].axis("off")

    axes[1].imshow(image_z1, cmap="gray")
    axes[1].set_title(f"z={z1}")
    axes[1].axis("off")

    max_abs_diff = float(np.max(np.abs(diff)))
    if max_abs_diff == 0:
        max_abs_diff = 1.0
    axes[2].imshow(diff, cmap="bwr", vmin=-max_abs_diff, vmax=max_abs_diff)
    axes[2].set_title(f"z{z1} - z{z0}")
    axes[2].axis("off")

    fig.suptitle(title)
    plt.show()

def _load_csv(path: Path) -> np.ndarray:
    """Load a GT CSV as a structured numpy table."""

    return np.genfromtxt(path, delimiter=",", names=True)

def load_expected_time_registration_shifts(
    path: Path,
    *,
    registration_stack: int = 0,
    axes: str = "yx",
) -> np.ndarray:
    """Load expected correction shifts from a synthetic GT time-shift table."""

    table = _load_csv(path)
    columns = [f"expected_registration_shift_{axis}_ref_t{registration_stack}" for axis in axes]
    return np.column_stack([table[column] for column in columns]).astype(np.float32)

def load_expected_slice_registration_shifts(path: Path) -> np.ndarray:
    """Load expected per-slice correction shifts as ``T, Z, 2``."""

    table = _load_csv(path)
    t_count = int(np.max(table["t"])) + 1
    z_count = int(np.max(table["z"])) + 1
    expected = np.zeros((t_count, z_count, 2), dtype=np.float32)
    for row in table:
        expected[int(row["t"]), int(row["z"]), :] = (
            row["expected_local_z_correction_shift_y"],
            row["expected_local_z_correction_shift_x"],
        )
    return expected


def print_shift_comparison(name: str, estimated_shifts: np.ndarray, expected_shifts: np.ndarray) -> None:
    """Print a compact comparison between estimated correction shifts and GT."""

    estimated_shifts = np.asarray(estimated_shifts, dtype=np.float32)
    expected_shifts = np.asarray(expected_shifts, dtype=np.float32)
    delta = estimated_shifts - expected_shifts
    flat_estimated = estimated_shifts.reshape(-1, estimated_shifts.shape[-1])
    flat_expected = expected_shifts.reshape(-1, expected_shifts.shape[-1])
    flat_delta = delta.reshape(-1, delta.shape[-1])
    print(f"{name}:")
    print(f"  mean abs error: {np.mean(np.abs(flat_delta), axis=0)}")
    print(f"  max abs error:  {np.max(np.abs(flat_delta), axis=0)}")
    print("  first rows [estimated..., expected..., delta...]:")
    print(np.column_stack([flat_estimated, flat_expected, flat_delta])[:5])

# %% 1) 2D+t: GLOBAL XY TIME REGISTRATION RELATIVE TO t=0
stack_2d_t_xy, metadata_2d_t_xy = load_stack(STACK_2D_T_XY_PATH, return_metadata=True, verbose=False)
expected_2d_t_xy = load_expected_time_registration_shifts(GT_2D_T_XY_PATH, registration_stack=0, axes="yx")
print(f"2D+t XY stack shape: {stack_2d_t_xy.shape} (TZCYX)")
show_timepoints(stack_2d_t_xy, title="2D+t XY before registration", channel=0, projection_method="max")

registered_2d_t_xy, shifts_2d_t_xy = register_stack(
    stack_2d_t_xy,
    registration_channel=0,  # channel used to estimate shifts
    registration_stack=0,  # reference time point/template
    method="phase_cross_correlation",  # "phase_cross_correlation" or "pystackreg"
    time_registration_mode="projection",  # "projection", "full_3d", or "none"
    time_reference_mode="template",  # "template" or "previous"
    projection_method="max",  # "max", "mean", "median", "var", or "std"
    zreg=False,  # estimate/apply Z shifts during time registration
    max_xy_shifts=None,  # None or (max_y, max_x)
    max_z_shifts=None,  # None or max_z
    filter_slices=False,  # median-filter Z slices before projection
    filter_projections=False,  # median-filter projections before shift estimation
    median_kernel_size=3,  # median-filter kernel size in pixels
    verbose=True,
    return_shifts=True,
)
print_shift_comparison("2D+t XY time registration", shifts_2d_t_xy, expected_2d_t_xy)
show_timepoints(registered_2d_t_xy, title="2D+t XY after registration", channel=0, projection_method="max")
save_stack(OUTPUT_DIR / "synthetic_2d_t_xy_registered.ome.tif", registered_2d_t_xy, metadata=metadata_2d_t_xy)
# %% 2) 3D: INTRA-STACK XY SLICE REGISTRATION
stack_3d_z_xy, metadata_3d_z_xy = load_stack(STACK_3D_Z_XY_PATH, return_metadata=True, verbose=False)
expected_3d_z_xy = load_expected_slice_registration_shifts(GT_3D_Z_XY_PATH)
print(f"3D Z-XY stack shape: {stack_3d_z_xy.shape} (TZCYX)")
show_slices(stack_3d_z_xy, title="3D intra-stack before correction", channel=0, z0=0, z1=6)

registered_3d_z_xy, shifts_3d_z_xy = register_stack(
    stack_3d_z_xy,
    registration_channel=0,  # channel used to estimate shifts
    method="phase_cross_correlation",  # "phase_cross_correlation" or "pystackreg"
    time_registration_mode="none",  # "projection", "full_3d", or "none"
    intra_stack=True,  # correct XY shifts within each 3D stack
    intra_stack_reference_mode="first_slice",  # "neighbor", "full_projection", or "first_slice"
    projection_method="max",  # "max", "mean", "median", "var", or "std"
    max_xy_shifts=None,  # None or (max_y, max_x)
    filter_slices=False,  # median-filter Z slices before reference creation
    filter_projections=False,  # median-filter images before shift estimation
    median_kernel_size=3,  # median-filter kernel size in pixels
    verbose=True,
    return_shifts=True)
print_shift_comparison("3D intra-stack XY correction", shifts_3d_z_xy, expected_3d_z_xy)
show_slices(registered_3d_z_xy, title="3D intra-stack after correction", channel=0, z0=0, z1=6)
save_stack(OUTPUT_DIR / "synthetic_3d_z_xy_registered.ome.tif", registered_3d_z_xy, metadata=metadata_3d_z_xy)
# %% 3) 3D+t: GLOBAL XY TIME REGISTRATION RELATIVE TO t=0
stack_3d_t_xy, metadata_3d_t_xy = load_stack(STACK_3D_T_XY_PATH, return_metadata=True, verbose=False)
expected_3d_t_xy = load_expected_time_registration_shifts(GT_3D_T_XY_PATH, registration_stack=0, axes="yx")
print(f"3D+t XY stack shape: {stack_3d_t_xy.shape} (TZCYX)")
show_timepoints(stack_3d_t_xy, title="3D+t XY before time registration", channel=0, projection_method="max")

registered_3d_t_xy, shifts_3d_t_xy = register_stack(
    stack_3d_t_xy,
    registration_channel=0,  # channel used to estimate shifts
    registration_stack=0,  # reference time point/template
    method="phase_cross_correlation",  # "phase_cross_correlation" or "pystackreg"
    time_registration_mode="projection",  # "projection", "full_3d", or "none"
    time_reference_mode="template",  # "template" or "previous"
    zrange=None,  # None or (z_start, z_stop)
    projection_method="max",  # "max", "mean", "median", "var", or "std"
    zreg=False,  # True also estimates/applies Z shifts
    max_xy_shifts=None,  # None or (max_y, max_x)
    max_z_shifts=None,  # None or max_z
    filter_slices=False,  # median-filter Z slices before projection
    filter_projections=False,  # median-filter projections before shift estimation
    median_kernel_size=3,  # median-filter kernel size in pixels
    verbose=True,
    return_shifts=True)
print_shift_comparison("3D+t global XY time registration", shifts_3d_t_xy, expected_3d_t_xy)
show_timepoints(registered_3d_t_xy, title="3D+t XY after time registration", channel=0, projection_method="max")
save_stack(OUTPUT_DIR / "synthetic_3d_t_xy_registered.ome.tif", registered_3d_t_xy, metadata=metadata_3d_t_xy)
# %% 4) 3D+t: INTRA-STACK ONLY, NO TIMEPOINT REGISTRATION
stack_3d_t_intra_xy, metadata_3d_t_intra_xy = load_stack(
    STACK_3D_T_INTRA_XY_PATH,
    return_metadata=True,
    verbose=False,
)
expected_3d_t_intra_xy = load_expected_slice_registration_shifts(GT_3D_T_INTRA_XY_PATH)
print(f"3D+t intra-only stack shape: {stack_3d_t_intra_xy.shape} (TZCYX)")
show_slices(stack_3d_t_intra_xy, title="3D+t intra-only before correction", channel=0, z0=0, z1=6)

registered_3d_t_intra_xy, shifts_3d_t_intra_xy = register_stack(
    stack_3d_t_intra_xy,
    registration_channel=0,  # channel used to estimate shifts
    method="phase_cross_correlation",  # "phase_cross_correlation" or "pystackreg"
    time_registration_mode="none",  # "projection", "full_3d", or "none"
    intra_stack=True,  # correct within each 3D stack only
    intra_stack_reference_mode="first_slice",  # "neighbor", "full_projection", or "first_slice"
    projection_method="max",  # "max", "mean", "median", "var", or "std"
    max_xy_shifts=None,  # None or (max_y, max_x)
    filter_slices=False,  # median-filter Z slices before reference creation
    filter_projections=False,  # median-filter images before shift estimation
    median_kernel_size=3,  # median-filter kernel size in pixels
    verbose=True,
    return_shifts=True,
)
print_shift_comparison("3D+t intra-stack-only XY correction", shifts_3d_t_intra_xy, expected_3d_t_intra_xy)
show_slices(registered_3d_t_intra_xy, title="3D+t intra-only after correction", channel=0, z0=0, z1=6)
save_stack(
    OUTPUT_DIR / "synthetic_3d_t_intra_xy_registered.ome.tif",
    registered_3d_t_intra_xy,
    metadata=metadata_3d_t_intra_xy,
)
# %% 5) 3D+t: GLOBAL ZYX TIME REGISTRATION RELATIVE TO t=0
stack_3d_t_zyx, metadata_3d_t_zyx = load_stack(STACK_3D_T_ZYX_PATH, return_metadata=True, verbose=False)
expected_3d_t_zyx = load_expected_time_registration_shifts(GT_3D_T_ZYX_PATH, registration_stack=0, axes="zyx")
print(f"3D+t ZYX stack shape: {stack_3d_t_zyx.shape} (TZCYX)")
show_timepoints(stack_3d_t_zyx, title="3D+t ZYX before full 3D registration", channel=0, projection_method="max")

registered_3d_t_zyx, details_3d_t_zyx = register_stack(
    stack_3d_t_zyx,
    registration_channel=0,  # channel used to estimate shifts
    registration_stack=0,  # reference time point/template
    method="phase_cross_correlation",  # full 3D requires "phase_cross_correlation"
    time_registration_mode="full_3d",  # "projection", "full_3d", or "none"
    time_reference_mode="template",  # "template" or "previous"
    zrange=None,  # None or (z_start, z_stop)
    projection_method="max",  # used by projection fallback/z-projection paths
    zreg=True,  # apply Z shifts from full 3D phase cross-correlation
    max_xy_shifts=None,  # None or (max_y, max_x)
    max_z_shifts=None,  # None or max_z
    filter_slices=False,  # median-filter Z slices before shift estimation
    filter_projections=False,  # median-filter projections before projection fallback
    median_kernel_size=3,  # median-filter kernel size in pixels
    verbose=True,
    return_shifts=True,
)
print_shift_comparison(
    "3D+t global ZYX full-volume time registration",
    details_3d_t_zyx["time_shifts_zyx"],
    expected_3d_t_zyx,
)
show_timepoints(
    registered_3d_t_zyx,
    title="3D+t ZYX after full 3D registration",
    channel=0,
    projection_method="max",
)
save_stack(OUTPUT_DIR / "synthetic_3d_t_zyx_registered.ome.tif", registered_3d_t_zyx, metadata=metadata_3d_t_zyx)
# %% END
