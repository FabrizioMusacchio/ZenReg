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

from zenreg import cleanup_omio_cache, load_stack, print_available_compute, register_stack, save_stack, z_project

# %% DEFINE INPUT AND OUTPUT PATHS
EXAMPLE_DIR = PROJECT_ROOT / "example_data" / "synthetic_data"
OUTPUT_DIR = EXAMPLE_DIR / "registered"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MEMMAP_CACHE_DIR = OUTPUT_DIR / "omio_memmap_cache"
OPEN_IN_NAPARI = False
AVAILABLE_CPUS = print_available_compute()

STACK_2D_T_XY_PATH = EXAMPLE_DIR / "synthetic_2d_t_xy.ome.tif"
STACK_3D_Z_XY_PATH = EXAMPLE_DIR / "synthetic_3d_z_xy.ome.tif"
STACK_3D_T_XY_PATH = EXAMPLE_DIR / "synthetic_3d_t_xy.ome.tif"
STACK_3D_T_INTRA_XY_PATH = EXAMPLE_DIR / "synthetic_3d_t_intra_xy.ome.tif"
STACK_3D_T_ZYX_PATH = EXAMPLE_DIR / "synthetic_3d_t_zyx.ome.tif"
STACK_2D_T_ROT_XY_PATH = EXAMPLE_DIR / "synthetic_2d_t_rot_xy.ome.tif"
STACK_3D_T_TRANS_ROT_Z_PATH = EXAMPLE_DIR / "synthetic_3d_t_trans_rot_z.ome.tif"

GT_2D_T_XY_PATH = EXAMPLE_DIR / "synthetic_2d_t_xy_time_shifts_gt.csv"
GT_3D_Z_XY_PATH = EXAMPLE_DIR / "synthetic_3d_z_xy_slice_shifts_gt.csv"
GT_3D_T_XY_PATH = EXAMPLE_DIR / "synthetic_3d_t_xy_time_shifts_gt.csv"
GT_3D_T_INTRA_XY_PATH = EXAMPLE_DIR / "synthetic_3d_t_intra_xy_slice_shifts_gt.csv"
GT_3D_T_ZYX_PATH = EXAMPLE_DIR / "synthetic_3d_t_zyx_time_shifts_gt.csv"
GT_2D_T_ROT_XY_PATH = EXAMPLE_DIR / "synthetic_2d_t_rot_xy_time_shifts_gt.csv"
GT_2D_T_ROT_DEG_PATH = EXAMPLE_DIR / "synthetic_2d_t_rot_xy_time_rotations_gt.csv"
GT_3D_T_TRANS_ROT_Z_PATH = EXAMPLE_DIR / "synthetic_3d_t_trans_rot_z_rigid_transform_gt.csv"
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


def load_expected_time_registration_rotations(path: Path, *, registration_stack: int = 0) -> np.ndarray:
    """Load expected rotation corrections from a synthetic GT table."""

    table = _load_csv(path)
    column = f"expected_registration_rotation_deg_ref_t{registration_stack}"
    return np.asarray(table[column], dtype=np.float32)


def load_expected_rigid_z_rotation(path: Path, *, registration_stack: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Load expected ZYX shifts and Z-axis rotation corrections from a 3D rigid GT table."""

    table = _load_csv(path)
    shift_columns = [
        f"expected_registration_shift_{axis}_ref_t{registration_stack}"
        for axis in ("z", "y", "x")
    ]
    rotation_column = f"expected_registration_rotation_z_deg_ref_t{registration_stack}"
    shifts_zyx = np.column_stack([table[column] for column in shift_columns]).astype(np.float32)
    rotations_z_deg = np.asarray(table[rotation_column], dtype=np.float32)
    return shifts_zyx, rotations_z_deg


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


def maybe_open_in_napari(stack, metadata, *, fname: str) -> None:
    """Open a stack in Napari only when the interactive flag is enabled."""

    if OPEN_IN_NAPARI:
        import omio as om

        om.open_in_napari(stack, metadata, fname=fname)

# %% 1) 2D+t: GLOBAL XY TIME REGISTRATION RELATIVE TO t=0
stack_2d_t_xy, metadata_2d_t_xy = load_stack(STACK_2D_T_XY_PATH, return_metadata=True, verbose=False)
expected_2d_t_xy = load_expected_time_registration_shifts(GT_2D_T_XY_PATH, registration_stack=0, axes="yx")
print(f"2D+t XY stack shape: {stack_2d_t_xy.shape} (TZCYX)")
show_timepoints(stack_2d_t_xy, title="2D+t XY before registration", channel=0, projection_method="max")

registered_2d_t_xy, details_2d_t_xy = register_stack(
    stack_2d_t_xy,
    registration_channel=0,  # channel used to estimate shifts
    registration_stack=0,  # reference time point/template
    method="phase_cross_correlation",  # "phase_cross_correlation" or "pystackreg"
    time_registration_mode="projection",  # "projection", "full_3d", or "none"
    time_reference_mode="template",  # "template" or "previous"
    projection_method="max",  # "max", "mean", "median", "var", or "std"
    zreg=False,  # estimate/apply Z shifts during time registration
    zero_clip=False,  # crop translation-introduced zero borders in Z/Y/X
    zero_clip_mode="auto",  # "auto", "shift", or "mask"
    zero_clip_mask_threshold=0.999,  # threshold for mask-based clipping
    zero_clip_margin=(0, 0, 0),  # extra crop margin as (z, y, x)
    max_xy_shifts=None,  # None or (max_y, max_x)
    max_z_shifts=None,  # None or max_z
    rotreg=False,  # estimate/apply in-plane XY rotations across time
    max_rot_shifts=None,  # None or max rotation in degrees
    rotreg_iter=1,  # 1 = translation, rotation, translation if rotreg=True
    transform_backend="skimage",  # "skimage" or "scipy"
    transform_order=1,  # 1 for intensity data, 0 for sparse puncta/labels
    filter_slices=False,  # median-filter Z slices before projection
    filter_projections=False,  # median-filter projections before shift estimation
    median_kernel_size=3,  # median-filter kernel size in pixels
    n_jobs=2,  # CPU worker threads for independent time points/slices
    verbose=True,
    return_shifts=True,
    return_details=True,
)
print_shift_comparison("2D+t XY time registration", details_2d_t_xy["time_shifts_yx"], expected_2d_t_xy)
show_timepoints(registered_2d_t_xy, title="2D+t XY after registration", channel=0, projection_method="max")
save_stack(
    OUTPUT_DIR / "synthetic_2d_t_xy_registered.ome.tif",
    registered_2d_t_xy,
    metadata=metadata_2d_t_xy,
    registration_details=details_2d_t_xy,
)
# %% 2) 3D: INTRA-STACK XY SLICE REGISTRATION
stack_3d_z_xy, metadata_3d_z_xy = load_stack(STACK_3D_Z_XY_PATH, return_metadata=True, verbose=False)
expected_3d_z_xy = load_expected_slice_registration_shifts(GT_3D_Z_XY_PATH)
print(f"3D Z-XY stack shape: {stack_3d_z_xy.shape} (TZCYX)")
show_slices(stack_3d_z_xy, title="3D intra-stack before correction", channel=0, z0=0, z1=6)

registered_3d_z_xy, details_3d_z_xy = register_stack(
    stack_3d_z_xy,
    registration_channel=0,  # channel used to estimate shifts
    method="phase_cross_correlation",  # "phase_cross_correlation" or "pystackreg"
    time_registration_mode="none",  # "projection", "full_3d", or "none"
    intra_stack=True,  # correct XY shifts within each 3D stack
    intra_stack_reference_mode="first_slice",  # "neighbor", "full_projection", or "first_slice"
    projection_method="max",  # "max", "mean", "median", "var", or "std"
    zero_clip=False,  # crop translation-introduced zero borders in Z/Y/X
    zero_clip_mode="auto",  # "auto", "shift", or "mask"
    zero_clip_mask_threshold=0.999,  # threshold for mask-based clipping
    zero_clip_margin=(0, 0, 0),  # extra crop margin as (z, y, x)
    max_xy_shifts=None,  # None or (max_y, max_x)
    transform_backend="skimage",  # "skimage" or "scipy"
    transform_order=1,  # 1 for intensity data, 0 for sparse puncta/labels
    filter_slices=False,  # median-filter Z slices before reference creation
    filter_projections=False,  # median-filter images before shift estimation
    median_kernel_size=3,  # median-filter kernel size in pixels
    n_jobs=2,  # CPU worker threads for independent time points/slices
    verbose=True,
    return_shifts=True,
    return_details=True)
print_shift_comparison(
    "3D intra-stack XY correction",
    details_3d_z_xy["intra_stack_shifts_yx"],
    expected_3d_z_xy)
show_slices(registered_3d_z_xy, title="3D intra-stack after correction", channel=0, z0=0, z1=6)
save_stack(
    OUTPUT_DIR / "synthetic_3d_z_xy_registered.ome.tif",
    registered_3d_z_xy,
    metadata=metadata_3d_z_xy,
    registration_details=details_3d_z_xy,
)
# %% 3) 3D+t: GLOBAL XY TIME REGISTRATION RELATIVE TO t=0
stack_3d_t_xy, metadata_3d_t_xy = load_stack(STACK_3D_T_XY_PATH, return_metadata=True, verbose=False)
expected_3d_t_xy = load_expected_time_registration_shifts(GT_3D_T_XY_PATH, registration_stack=0, axes="yx")
print(f"3D+t XY stack shape: {stack_3d_t_xy.shape} (TZCYX)")
show_timepoints(stack_3d_t_xy, title="3D+t XY before time registration", channel=0, projection_method="max")

registered_3d_t_xy, details_3d_t_xy = register_stack(
    stack_3d_t_xy,
    registration_channel=0,  # channel used to estimate shifts
    registration_stack=0,  # reference time point/template
    method="phase_cross_correlation",  # "phase_cross_correlation" or "pystackreg"
    time_registration_mode="projection",  # "projection", "full_3d", or "none"
    time_reference_mode="template",  # "template" or "previous"
    projection_range=None,  # None or (z_start, z_stop)
    projection_method="max",  # "max", "mean", "median", "var", or "std"
    zreg=False,  # True also estimates/applies Z shifts
    zero_clip=False,  # crop translation-introduced zero borders in Z/Y/X
    zero_clip_mode="auto",  # "auto", "shift", or "mask"
    zero_clip_mask_threshold=0.999,  # threshold for mask-based clipping
    zero_clip_margin=(0, 0, 0),  # extra crop margin as (z, y, x)
    max_xy_shifts=None,  # None or (max_y, max_x)
    max_z_shifts=None,  # None or max_z
    rotreg=False,  # estimate/apply in-plane XY rotations across time
    max_rot_shifts=None,  # None or max rotation in degrees
    rotreg_iter=1,  # 1 = translation, rotation, translation if rotreg=True
    transform_backend="skimage",  # "skimage" or "scipy"
    transform_order=1,  # 1 for intensity data, 0 for sparse puncta/labels
    filter_slices=False,  # median-filter Z slices before projection
    filter_projections=False,  # median-filter projections before shift estimation
    median_kernel_size=3,  # median-filter kernel size in pixels
    n_jobs=2,  # CPU worker threads for independent time points/slices
    verbose=True,
    return_shifts=True,
    return_details=True)
print_shift_comparison(
    "3D+t global XY time registration",
    details_3d_t_xy["time_shifts_yx"],
    expected_3d_t_xy)
show_timepoints(registered_3d_t_xy, title="3D+t XY after time registration", channel=0, projection_method="max")
save_stack(
    OUTPUT_DIR / "synthetic_3d_t_xy_registered.ome.tif",
    registered_3d_t_xy,
    metadata=metadata_3d_t_xy,
    registration_details=details_3d_t_xy,
)
# %% 4) 3D+t: INTRA-STACK ONLY, NO TIMEPOINT REGISTRATION
stack_3d_t_intra_xy, metadata_3d_t_intra_xy = load_stack(
    STACK_3D_T_INTRA_XY_PATH,
    return_metadata=True,
    verbose=False,
)
expected_3d_t_intra_xy = load_expected_slice_registration_shifts(GT_3D_T_INTRA_XY_PATH)
print(f"3D+t intra-only stack shape: {stack_3d_t_intra_xy.shape} (TZCYX)")
show_slices(stack_3d_t_intra_xy, title="3D+t intra-only before correction", channel=0, z0=0, z1=6)

registered_3d_t_intra_xy, details_3d_t_intra_xy = register_stack(
    stack_3d_t_intra_xy,
    registration_channel=0,  # channel used to estimate shifts
    method="phase_cross_correlation",  # "phase_cross_correlation" or "pystackreg"
    time_registration_mode="none",  # "projection", "full_3d", or "none"
    intra_stack=True,  # correct within each 3D stack only
    intra_stack_reference_mode="first_slice",  # "neighbor", "full_projection", or "first_slice"
    projection_method="max",  # "max", "mean", "median", "var", or "std"
    zero_clip=False,  # crop translation-introduced zero borders in Z/Y/X
    zero_clip_mode="auto",  # "auto", "shift", or "mask"
    zero_clip_mask_threshold=0.999,  # threshold for mask-based clipping
    zero_clip_margin=(0, 0, 0),  # extra crop margin as (z, y, x)
    max_xy_shifts=None,  # None or (max_y, max_x)
    transform_backend="skimage",  # "skimage" or "scipy"
    transform_order=1,  # 1 for intensity data, 0 for sparse puncta/labels
    filter_slices=False,  # median-filter Z slices before reference creation
    filter_projections=False,  # median-filter images before shift estimation
    median_kernel_size=3,  # median-filter kernel size in pixels
    n_jobs=2,  # CPU worker threads for independent time points/slices
    verbose=True,
    return_shifts=True,
    return_details=True,
)
print_shift_comparison(
    "3D+t intra-stack-only XY correction",
    details_3d_t_intra_xy["intra_stack_shifts_yx"],
    expected_3d_t_intra_xy)
show_slices(registered_3d_t_intra_xy, title="3D+t intra-only after correction", channel=0, z0=0, z1=6)
save_stack(
    OUTPUT_DIR / "synthetic_3d_t_intra_xy_registered.ome.tif",
    registered_3d_t_intra_xy,
    metadata=metadata_3d_t_intra_xy,
    registration_details=details_3d_t_intra_xy,
)
# %% 5) 3D+t: GLOBAL ZYX TIME REGISTRATION RELATIVE TO t=0
stack_3d_t_zyx, metadata_3d_t_zyx = load_stack(STACK_3D_T_ZYX_PATH, return_metadata=True, verbose=False)
expected_3d_t_zyx = load_expected_time_registration_shifts(GT_3D_T_ZYX_PATH, registration_stack=0, axes="zyx")
print(f"3D+t ZYX stack shape: {stack_3d_t_zyx.shape} (TZCYX)")
show_timepoints(stack_3d_t_zyx, title="3D+t ZYX before full 3D registration", channel=0, projection_method="max")

maybe_open_in_napari(stack_3d_t_zyx, metadata_3d_t_zyx, fname="3D+t ZYX before full 3D registration")

registered_3d_t_zyx, details_3d_t_zyx = register_stack(
    stack_3d_t_zyx,
    registration_channel=0,  # channel used to estimate shifts
    registration_stack=0,  # reference time point/template
    method="phase_cross_correlation",  # full 3D requires "phase_cross_correlation"
    time_registration_mode="full_3d",  # "projection", "full_3d", or "none"
    time_reference_mode="template",  # "template" or "previous"
    projection_range=None,  # None or (z_start, z_stop)
    projection_method="max",  # used by projection fallback/z-projection paths
    zreg=True,  # apply Z shifts from full 3D phase cross-correlation
    zero_clip=True,  # crop translation-introduced zero borders in Z/Y/X
    zero_clip_mode="auto",  # "auto", "shift", or "mask"
    zero_clip_mask_threshold=0.999,  # threshold for mask-based clipping
    zero_clip_margin=(0, 0, 0),  # extra crop margin as (z, y, x)
    max_xy_shifts=None,  # None or (max_y, max_x)
    max_z_shifts=None,  # None or max_z
    rotreg=False,  # estimate/apply in-plane XY rotations across time
    max_rot_shifts=None,  # None or max rotation in degrees
    rotreg_iter=1,  # 1 = translation, rotation, translation if rotreg=True
    transform_backend="skimage",  # "skimage" or "scipy"
    transform_order=1,  # 1 for intensity data, 0 for sparse puncta/labels
    filter_slices=False,  # median-filter Z slices before shift estimation
    filter_projections=False,  # median-filter projections before projection fallback
    median_kernel_size=3,  # median-filter kernel size in pixels
    n_jobs=2,  # CPU worker threads for independent time points/slices
    verbose=True,
    return_shifts=True,
    return_details=True)
print_shift_comparison(
    "3D+t global ZYX full-volume time registration",
    details_3d_t_zyx["time_shifts_zyx"],
    expected_3d_t_zyx)
show_timepoints(
    registered_3d_t_zyx,
    title="3D+t ZYX after full 3D registration",
    channel=0,
    projection_method="max",)
save_stack(
    OUTPUT_DIR / "synthetic_3d_t_zyx_registered.ome.tif",
    registered_3d_t_zyx,
    metadata=metadata_3d_t_zyx,
    registration_details=details_3d_t_zyx,
)

maybe_open_in_napari(registered_3d_t_zyx, metadata_3d_t_zyx, fname="3D+t ZYX after full 3D registration")
# %% 6) 3D+t: FULL 3D TRANSLATION PLUS XY-PLANE ROTATION ONLY
stack_3d_t_trans_rot_z, metadata_3d_t_trans_rot_z = load_stack(
    STACK_3D_T_TRANS_ROT_Z_PATH,
    return_metadata=True,
    verbose=False)
expected_3d_t_trans_rot_z_shifts, expected_3d_t_trans_rot_z_rot_deg = load_expected_rigid_z_rotation(
    GT_3D_T_TRANS_ROT_Z_PATH,
    registration_stack=0)
print(f"3D+t ZYX translation + Z rotation stack shape: {stack_3d_t_trans_rot_z.shape} (TZCYX)")
show_timepoints(
    stack_3d_t_trans_rot_z,
    title="3D+t ZYX translation + XY-plane rotation before registration",
    channel=0,
    projection_method="max")
maybe_open_in_napari(
    stack_3d_t_trans_rot_z,
    metadata_3d_t_trans_rot_z,
    fname="3D+t translation + XY rotation before registration")

registered_3d_t_trans_rot_z, details_3d_t_trans_rot_z = register_stack(
    stack_3d_t_trans_rot_z,
    registration_channel=0,  # channel used to estimate shifts and XY-plane rotation
    registration_stack=0,  # reference time point/template
    method="phase_cross_correlation",  # translation and projection-polar rotation estimator
    time_registration_mode="full_3d",  # full ZYX translation correction
    time_reference_mode="template",  # "template" or "previous"
    projection_range=None,  # None or (z_start, z_stop)
    projection_method="max",  # projection used for XY-plane rotation estimation
    zreg=True,  # estimate/apply Z shifts from full 3D phase cross-correlation
    zero_clip=True,  # crop zero borders from translation/rotation correction
    zero_clip_mode="auto",  # "auto", "shift", or "mask"; auto uses mask with rotreg=True
    zero_clip_mask_threshold=0.999,  # threshold for mask-based clipping
    zero_clip_mask_strategy="auto",  # "auto", "relaxed", "greedy", or "max_volume"
    zero_clip_mask_min_fraction=0.5,  # relaxed crop: lower keeps more FOV
    zero_clip_margin=(0, 0, 0),  # extra crop margin as (z, y, x)
    max_xy_shifts=None,  # None or (max_y, max_x)
    max_z_shifts=None,  # None or max_z
    rotreg=True,  # estimate/apply only in-plane XY rotations across time
    rigid_3d_backend="phase_cross_correlation",  # projection-polar Z-axis rotation, not full 6-DOF
    max_rot_shifts=12,  # None or max rotation in degrees
    rotreg_iter=2,  # 1 = translation, rotation, translation
    transform_backend="skimage",  # "skimage" or "scipy"
    transform_order=1,  # 1 for intensity data, 0 for sparse puncta/labels
    filter_slices=False,  # median-filter Z slices before projection
    filter_projections=False,  # median-filter projections before shift estimation
    median_kernel_size=3,  # median-filter kernel size in pixels
    n_jobs=2,  # CPU worker threads for independent time points/slices
    verbose=True,
    return_shifts=True,
    return_details=True)
print_shift_comparison(
    "3D+t full-volume translation during XY-plane rotation refinement",
    details_3d_t_trans_rot_z["time_shifts_zyx"],
    expected_3d_t_trans_rot_z_shifts)
print_shift_comparison(
    "3D+t XY-plane rotation correction [deg]",
    details_3d_t_trans_rot_z["rotation_shifts_deg"][:, None],
    expected_3d_t_trans_rot_z_rot_deg[:, None])
print(f"Zero-clip bounds: {details_3d_t_trans_rot_z['zero_clip_bounds']}")
show_timepoints(
    registered_3d_t_trans_rot_z,
    title="3D+t ZYX translation + XY-plane rotation after registration",
    channel=0,
    projection_method="max")
save_stack(
    OUTPUT_DIR / "synthetic_3d_t_trans_rot_z_projection_registered.ome.tif",
    registered_3d_t_trans_rot_z,
    metadata=metadata_3d_t_trans_rot_z,
    registration_details=details_3d_t_trans_rot_z)
maybe_open_in_napari(
    registered_3d_t_trans_rot_z,
    metadata_3d_t_trans_rot_z,
    fname="3D+t translation + XY rotation after registration")

# %% 7) 2D+t: GLOBAL XY ROTATION
stack_2d_t_rot_xy, metadata_2d_t_rot_xy = load_stack(
    STACK_2D_T_ROT_XY_PATH,
    return_metadata=True,
    verbose=False)
expected_2d_t_rot_xy = load_expected_time_registration_shifts(
    GT_2D_T_ROT_XY_PATH,
    registration_stack=0,
    axes="yx")
expected_2d_t_rot_deg = load_expected_time_registration_rotations(
    GT_2D_T_ROT_DEG_PATH,
    registration_stack=0)
print(f"2D+t rotation stack shape: {stack_2d_t_rot_xy.shape} (TZCYX)")
show_timepoints(
    stack_2d_t_rot_xy,
    title="2D+t rotation before registration",
    channel=0,
    projection_method="max")
maybe_open_in_napari(stack_2d_t_rot_xy, metadata_2d_t_rot_xy, fname="2D+t rotation before registration")


registered_2d_t_rot_xy, details_2d_t_rot_xy = register_stack(
    stack_2d_t_rot_xy,
    registration_channel=0,  # channel used to estimate shifts
    registration_stack=0,  # reference time point/template
    method="phase_cross_correlation",  # "phase_cross_correlation" or "pystackreg"
    time_registration_mode="projection",  # "projection", "full_3d", or "none"
    time_reference_mode="template",  # "template" or "previous"
    projection_range=None,  # None or (z_start, z_stop)
    projection_method="max",  # "max", "mean", "median", "var", or "std"
    zreg=False,  # True also estimates/applies Z shifts
    zero_clip=True,  # crop translation-introduced zero borders in Z/Y/X
    zero_clip_mode="auto",  # "auto", "shift", or "mask"; auto uses mask with rotreg=True
    zero_clip_mask_threshold=0.999,  # threshold for mask-based clipping
    zero_clip_margin=(0, 0, 0),  # extra crop margin as (z, y, x)
    max_xy_shifts=(0, 0),  # None or (max_y, max_x); here: isolate rotation
    max_z_shifts=None,  # None or max_z
    rotreg=True,  # estimate/apply in-plane XY rotations across time
    max_rot_shifts=12,  # None or max rotation in degrees
    rotreg_iter=1,  # 1 = translation, rotation, translation
    transform_backend="skimage",  # "skimage" or "scipy"
    transform_order=1,  # 1 for intensity data, 0 for sparse puncta/labels
    filter_slices=False,  # median-filter Z slices before projection
    filter_projections=False,  # median-filter projections before shift estimation
    median_kernel_size=3,  # median-filter kernel size in pixels
    n_jobs=2,  # CPU worker threads for independent time points/slices
    verbose=True,
    return_shifts=True,
    return_details=True)
print_shift_comparison(
    "2D+t translation passes during rotation refinement",
    details_2d_t_rot_xy["time_shifts_yx"],
    expected_2d_t_rot_xy)
print_shift_comparison(
    "2D+t XY rotation correction [deg]",
    details_2d_t_rot_xy["rotation_shifts_deg"][:, None],
    expected_2d_t_rot_deg[:, None])
print(f"Zero-clip bounds: {details_2d_t_rot_xy['zero_clip_bounds']}")
show_timepoints(
    registered_2d_t_rot_xy,
    title="2D+t rotation after registration",
    channel=0,
    projection_method="max")
save_stack(
    OUTPUT_DIR / "synthetic_2d_t_rot_xy_registered.ome.tif",
    registered_2d_t_rot_xy,
    metadata=metadata_2d_t_rot_xy,
    registration_details=details_2d_t_rot_xy)

maybe_open_in_napari(registered_2d_t_rot_xy, metadata_2d_t_rot_xy, fname="2D+t rotation after registration")
# %% 8) 2D+t: DISK-BACKED OMIO MEMMAP INPUT
# To force a fresh start, clear any pre-existing OMIO cache in the local scratch folder.
# Skip this cleanup line after a kernel restart if you want OMIO to reuse the existing cache.
cleanup_omio_cache(MEMMAP_CACHE_DIR, full_cleanup=True, verbose=False)

# load the stack:
stack_2d_t_xy_memmap, metadata_2d_t_xy_memmap = load_stack(
    STACK_2D_T_XY_PATH,
    return_metadata=True,
    use_memmap=True,  # read through OMIO disk-backed Zarr
    memmap_folder=MEMMAP_CACHE_DIR,  # local scratch/cache folder for the Zarr store
    memmap_reuse=True,  # reuse an existing valid .omio_cache if present
    verbose=False)
print(f"2D+t memmap stack shape: {stack_2d_t_xy_memmap.shape} (TZCYX)")
print(f"2D+t memmap stack type: {type(stack_2d_t_xy_memmap)}")
print(f"OMIO cache folder: {metadata_2d_t_xy_memmap.get('omio_cache_folder')}")
print(f"OMIO Zarr store path: {metadata_2d_t_xy_memmap.get('omio_zarr_store_path')}")

# register:
registered_2d_t_xy_memmap, details_2d_t_xy_memmap = register_stack(
    stack_2d_t_xy_memmap,
    registration_channel=0,  # channel used to estimate shifts
    registration_stack=0,  # reference time point/template
    method="phase_cross_correlation",  # "phase_cross_correlation" or "pystackreg"
    time_registration_mode="projection",  # "projection", "full_3d", or "none"
    time_reference_mode="template",  # "template" or "previous"
    projection_method="max",  # "max", "mean", "median", "var", or "std"
    zreg=False,  # estimate/apply Z shifts during time registration
    zero_clip=False,  # crop translation-introduced zero borders in Z/Y/X
    max_xy_shifts=None,  # None or (max_y, max_x)
    max_z_shifts=None,  # None or max_z
    transform_backend="skimage",  # "skimage" or "scipy"
    transform_order=1,  # 1 for intensity data, 0 for sparse puncta/labels
    filter_slices=False,  # median-filter Z slices before projection
    filter_projections=False,  # median-filter projections before shift estimation
    n_jobs=2,  # CPU worker threads for independent time points/slices
    output_use_memmap=True,  # write registered output into an OMIO/Zarr cache
    output_memmap_folder=MEMMAP_CACHE_DIR,  # local scratch/cache folder for registered results
    output_memmap_name="synthetic_2d_t_xy_registered",  # base Zarr store name for output stages
    output_dtype=np.float32,  # float32 preserves interpolated subpixel intensities
    verbose=True,
    return_shifts=True,
    return_details=True)
print(f"Registered memmap stack type: {type(registered_2d_t_xy_memmap)}")
print_shift_comparison(
    "2D+t memmap XY time registration",
    details_2d_t_xy_memmap["time_shifts_yx"],
    expected_2d_t_xy)

# save the registered memmap stack to a new OME-TIFF file:
save_stack(
    OUTPUT_DIR / "synthetic_2d_t_xy_memmap_registered.ome.tif",
    registered_2d_t_xy_memmap,
    metadata=metadata_2d_t_xy_memmap,
    registration_details=details_2d_t_xy_memmap)

# clean up the OMIO cache after saving and in case you don't need the cached memmap anymore:
cleanup_omio_cache(MEMMAP_CACHE_DIR, full_cleanup=True, verbose=False)
# %% END
