"""
Interactive VS Code script for testing ZenReg full 3D rigid registration.

Run this script cell-by-cell. It exercises the new 6-DOF registration path:

    register_stack(..., rotreg=True, rigid_3d_backend="simpleitk")
    register_stack(..., rotreg=True, rigid_3d_backend="points")

Both examples use OMIO-normalized ``TZCYX`` stacks and synthetic GT tables.

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
os.environ.setdefault("XDG_CACHE_HOME", str(SCRIPT_CACHE_DIR / "xdg-cache"))

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from zenreg import crop_stack, load_stack, register_stack, save_stack, z_project
from zenreg.synthetic import write_example_dataset

# %% DEFINE INPUT AND OUTPUT PATHS
EXAMPLE_DIR = PROJECT_ROOT / "example_data" / "synthetic_data"
OUTPUT_DIR = EXAMPLE_DIR / "registered_rigid3d"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OPEN_IN_NAPARI = False

STACK_3D_RIGID_SIMPLEITK_PATH = EXAMPLE_DIR / "synthetic_3d_t_rigid_simpleitk.ome.tif"
STACK_3D_RIGID_POINTS_PATH = EXAMPLE_DIR / "synthetic_3d_t_rigid_points.ome.tif"

GT_3D_RIGID_SIMPLEITK_PATH = EXAMPLE_DIR / "synthetic_3d_t_rigid_simpleitk_rigid_transform_gt.csv"
GT_3D_RIGID_POINTS_PATH = EXAMPLE_DIR / "synthetic_3d_t_rigid_points_rigid_transform_gt.csv"

if not STACK_3D_RIGID_SIMPLEITK_PATH.exists() or not STACK_3D_RIGID_POINTS_PATH.exists():
    write_example_dataset(EXAMPLE_DIR)

# %% SMALL HELPERS FOR GT COMPARISON AND VISUAL CHECKS
def _load_csv(path: Path) -> np.ndarray:
    """Load a GT CSV as a structured numpy table."""

    return np.genfromtxt(path, delimiter=",", names=True)


def load_expected_rigid_corrections(path: Path, *, registration_stack: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Load expected correction translations and rotations from a 3D rigid GT table."""

    table = _load_csv(path)
    expected_shifts_zyx = np.column_stack(
        [
            table[f"expected_registration_shift_z_ref_t{registration_stack}"],
            table[f"expected_registration_shift_y_ref_t{registration_stack}"],
            table[f"expected_registration_shift_x_ref_t{registration_stack}"],
        ]
    ).astype(np.float32)
    expected_rotations_zyx_deg = np.column_stack(
        [
            table[f"expected_registration_rotation_z_deg_ref_t{registration_stack}"],
            table[f"expected_registration_rotation_y_deg_ref_t{registration_stack}"],
            table[f"expected_registration_rotation_x_deg_ref_t{registration_stack}"],
        ]
    ).astype(np.float32)
    return expected_shifts_zyx, expected_rotations_zyx_deg


def print_rigid_comparison(
    title: str,
    details: dict,
    expected_shifts_zyx: np.ndarray,
    expected_rotations_zyx_deg: np.ndarray,
) -> None:
    """Print detected-vs-GT rigid registration summaries."""

    detected_shifts = np.asarray(details["time_shifts_zyx"], dtype=np.float32)
    detected_rotations = np.asarray(details["rotation_shifts_zyx_deg"], dtype=np.float32)
    shift_delta = detected_shifts - expected_shifts_zyx
    rotation_delta = detected_rotations - expected_rotations_zyx_deg

    print(title)
    print(f"  backend: {details['rigid_3d_backend']}")
    print(f"  shift mean abs error [z, y, x]: {np.mean(np.abs(shift_delta), axis=0)}")
    print(f"  shift max abs error  [z, y, x]: {np.max(np.abs(shift_delta), axis=0)}")
    print(f"  rot mean abs error   [z, y, x] deg: {np.mean(np.abs(rotation_delta), axis=0)}")
    print(f"  rot max abs error    [z, y, x] deg: {np.max(np.abs(rotation_delta), axis=0)}")
    print("  first rows [detected shifts..., expected shifts..., detected rotations..., expected rotations...]:")
    print(
        np.column_stack(
            [
                detected_shifts[:5],
                expected_shifts_zyx[:5],
                detected_rotations[:5],
                expected_rotations_zyx_deg[:5],
            ]
        )
    )


def show_before_after(
    raw_stack,
    registered_stack,
    *,
    title: str,
    channel: int = 0,
    moving_time: int = 1,
    projection_method: str = "max",
) -> None:
    """Show raw and registered projection residuals."""

    raw_t0 = z_project(raw_stack[0:1, :, channel : channel + 1, :, :], projection_method=projection_method)[0, 0, 0]
    raw_ti = z_project(
        raw_stack[moving_time : moving_time + 1, :, channel : channel + 1, :, :],
        projection_method=projection_method,
    )[0, 0, 0]
    reg_t0 = z_project(
        registered_stack[0:1, :, channel : channel + 1, :, :],
        projection_method=projection_method,
    )[0, 0, 0]
    reg_ti = z_project(
        registered_stack[moving_time : moving_time + 1, :, channel : channel + 1, :, :],
        projection_method=projection_method,
    )[0, 0, 0]

    raw_diff = raw_ti - raw_t0
    reg_diff = reg_ti - reg_t0
    max_abs = max(float(np.max(np.abs(raw_diff))), float(np.max(np.abs(reg_diff))), 1e-6)

    fig, axes = plt.subplots(2, 3, figsize=(10, 6), constrained_layout=True)
    for ax, image, label, cmap in [
        (axes[0, 0], raw_t0, "raw t=0", "gray"),
        (axes[0, 1], raw_ti, f"raw t={moving_time}", "gray"),
        (axes[0, 2], raw_diff, f"raw t{moving_time} - t0", "bwr"),
        (axes[1, 0], reg_t0, "registered t=0", "gray"),
        (axes[1, 1], reg_ti, f"registered t={moving_time}", "gray"),
        (axes[1, 2], reg_diff, f"registered t{moving_time} - t0", "bwr"),
    ]:
        if cmap == "bwr":
            ax.imshow(image, cmap=cmap, vmin=-max_abs, vmax=max_abs)
        else:
            ax.imshow(image, cmap=cmap)
        ax.set_title(label)
        ax.axis("off")
    fig.suptitle(title)
    if plt.get_backend().lower() == "agg":
        plt.close(fig)
    else:
        plt.show()


def maybe_open_in_napari(stack, metadata, *, fname: str) -> None:
    """Optionally open a stack in Napari for manual inspection."""

    if OPEN_IN_NAPARI:
        import omio as om

        om.open_in_napari(stack, metadata, fname=fname)


# %% 1) DENSE 3D+t: SIMPLEITK FULL 6-DOF RIGID REGISTRATION
stack_3d_rigid_simpleitk, metadata_3d_rigid_simpleitk = load_stack(
    STACK_3D_RIGID_SIMPLEITK_PATH,
    return_metadata=True,
)
expected_shifts_3d_rigid_simpleitk, expected_rotations_3d_rigid_simpleitk = load_expected_rigid_corrections(
    GT_3D_RIGID_SIMPLEITK_PATH,
    registration_stack=0,
)
print(f"Dense SimpleITK stack shape: {stack_3d_rigid_simpleitk.shape} (TZCYX)")
maybe_open_in_napari(stack_3d_rigid_simpleitk, metadata_3d_rigid_simpleitk, fname="Dense 3D rigid raw")

registered_3d_rigid_simpleitk, details_3d_rigid_simpleitk = register_stack(
    stack_3d_rigid_simpleitk,
    registration_channel=0,  # channel used to estimate the 3D rigid transform
    registration_stack=0,  # reference time point
    method="phase_cross_correlation",  # translational/coarse pre-estimation method
    time_registration_mode="full_3d",  # use full ZYX volumes instead of Z projections
    zreg=True,  # estimate and apply Z shifts
    rotreg=True,  # enable rotation correction
    rigid_3d_backend="simpleitk",  # "phase_cross_correlation", "simpleitk", or "points"
    rot_spacing_zyx=(1.0, 1.0, 1.0),  # physical spacing in Z/Y/X; set real microscope spacing here
    rot_init_iterations=2,  # iterations of orthogonal-projection rotation pre-estimation
    rot_metric="correlation",  # "correlation" or "mutual_information"
    rot_shrink_factors=(4, 2, 1),  # multi-resolution pyramid shrink factors
    rot_smoothing_sigmas=(2.0, 1.0, 0.0),  # multi-resolution smoothing sigmas
    rot_iterations=80,  # optimizer iterations per resolution schedule
    rot_learning_rate=1.0,  # SimpleITK regular-step optimizer learning rate
    rot_min_step=1e-4,  # SimpleITK regular-step optimizer convergence step
    rot_sampling_percentage=None,  # None/full metric; or e.g. 0.25 for random metric sampling
    rot_n_jobs=2,  # parallel time-point workers
    transform_order=1,  # 1 intensity data; 0 sparse puncta/labels
    zero_clip=True,  # keep full shape for 3D inspection; set True only for final common-valid cuboid export
    zero_clip_mode="auto",  # "auto", "shift", or "mask"
    zero_clip_mask_strategy="auto",  # "auto", "relaxed", "greedy", or "max_volume"
    zero_clip_mask_min_fraction=0.5,  # relaxed crop: lower keeps more FOV, higher removes more zero corners
    zero_clip_margin=(0, 0, 0),  # extra crop margin in Z/Y/X if zero_clip=True
    verbose=True,
    return_details=True,
)
print_rigid_comparison(
    "Dense SimpleITK 6-DOF rigid registration vs GT",
    details_3d_rigid_simpleitk,
    expected_shifts_3d_rigid_simpleitk,
    expected_rotations_3d_rigid_simpleitk,
)
show_before_after(
    stack_3d_rigid_simpleitk,
    registered_3d_rigid_simpleitk,
    title="Dense 3D+t SimpleITK 6-DOF registration",
    channel=0,
    moving_time=1,
)
# Optional post-hoc crop after visual inspection. Missing keys are treated as 0.
# registered_3d_rigid_simpleitk, metadata_3d_rigid_simpleitk = crop_stack(
#     registered_3d_rigid_simpleitk,
#     metadata_3d_rigid_simpleitk,
#     {"top": 1, "bottom": 1, "left": 2, "right": 2, "up": 2, "down": 2},
# )
save_stack(
    OUTPUT_DIR / "synthetic_3d_t_rigid_simpleitk_registered.ome.tif",
    registered_3d_rigid_simpleitk,
    metadata=metadata_3d_rigid_simpleitk,
    registration_details=details_3d_rigid_simpleitk,
)
maybe_open_in_napari(
    registered_3d_rigid_simpleitk,
    metadata_3d_rigid_simpleitk,
    fname="Dense 3D rigid registered SimpleITK",
)

print(f"Shape of raw dense stack:        {stack_3d_rigid_simpleitk.shape} (TZCYX)")
print(f"Shape of registered dense stack: {registered_3d_rigid_simpleitk.shape} (TZCYX)")
# %% 2) SPARSE 3D+t PUNCTA: POINTS FULL 6-DOF RIGID REGISTRATION
stack_3d_rigid_points, metadata_3d_rigid_points = load_stack(
    STACK_3D_RIGID_POINTS_PATH,
    return_metadata=True,
)
expected_shifts_3d_rigid_points, expected_rotations_3d_rigid_points = load_expected_rigid_corrections(
    GT_3D_RIGID_POINTS_PATH,
    registration_stack=0,
)
print(f"Sparse points stack shape: {stack_3d_rigid_points.shape} (TZCYX)")
maybe_open_in_napari(stack_3d_rigid_points, metadata_3d_rigid_points, fname="Sparse 3D puncta raw")

registered_3d_rigid_points, details_3d_rigid_points = register_stack(
    stack_3d_rigid_points,
    registration_channel=0,  # channel used to detect points and estimate the transform
    registration_stack=0,  # reference time point
    method="phase_cross_correlation",  # coarse translation pre-estimation
    time_registration_mode="full_3d",  # full volume registration
    zreg=True,  # estimate and apply Z shifts
    rotreg=True,  # enable 3D rotation correction
    rigid_3d_backend="points",  # "phase_cross_correlation", "simpleitk", or "points"
    rot_init_iterations=0,  # point ICP usually starts well from translation-only for this benchmark
    rot_points_max_points=120,  # maximum detected peaks per volume
    rot_points_min_distance=2,  # minimum peak distance in pixels
    rot_points_threshold_rel=0.2,  # relative peak threshold
    rot_points_iterations=10,  # ICP/Kabsch refinement iterations
    rot_points_max_match_distance=7.0,  # maximum nearest-neighbor match distance in pixels
    rot_n_jobs=2,  # parallel time-point workers
    transform_order=0,  # nearest-neighbor keeps sparse puncta sharp
    zero_clip=False,  # keep full shape for 3D inspection; set True only for final common-valid cuboid export
    zero_clip_mode="auto",  # "auto", "shift", or "mask"
    zero_clip_mask_strategy="auto",  # "auto", "relaxed", "greedy", or "max_volume"
    zero_clip_mask_min_fraction=0.5,  # relaxed crop: lower keeps more FOV, higher removes more zero corners
    zero_clip_margin=(0, 0, 0),  # extra crop margin in Z/Y/X if zero_clip=True
    verbose=True,
    return_details=True,
)
print_rigid_comparison(
    "Sparse points 6-DOF rigid registration vs GT",
    details_3d_rigid_points,
    expected_shifts_3d_rigid_points,
    expected_rotations_3d_rigid_points,
)
show_before_after(
    stack_3d_rigid_points,
    registered_3d_rigid_points,
    title="Sparse puncta 3D+t points 6-DOF registration",
    channel=0,
    moving_time=1,
)
# Optional post-hoc crop after visual inspection. Missing keys are treated as 0.
# registered_3d_rigid_points, metadata_3d_rigid_points = crop_stack(
#     registered_3d_rigid_points,
#     metadata_3d_rigid_points,
#     {"top": 1, "bottom": 1, "left": 2, "right": 2, "up": 2, "down": 2},
# )
save_stack(
    OUTPUT_DIR / "synthetic_3d_t_rigid_points_registered.ome.tif",
    registered_3d_rigid_points,
    metadata=metadata_3d_rigid_points,
    registration_details=details_3d_rigid_points,
)
maybe_open_in_napari(
    registered_3d_rigid_points,
    metadata_3d_rigid_points,
    fname="Sparse 3D puncta registered points",
)
# %% 3) SPARSE 3D+t PUNCTA: SIMPLEITK FULL 6-DOF RIGID REGISTRATION
stack_3d_rigid_points_simpleitk, metadata_3d_rigid_points_simpleitk = load_stack(
    STACK_3D_RIGID_POINTS_PATH,
    return_metadata=True,
)
expected_shifts_3d_rigid_points_simpleitk, expected_rotations_3d_rigid_points_simpleitk = load_expected_rigid_corrections(
    GT_3D_RIGID_POINTS_PATH,
    registration_stack=0,
)
print(f"Sparse points stack shape: {stack_3d_rigid_points_simpleitk.shape} (TZCYX)")
maybe_open_in_napari(
    stack_3d_rigid_points_simpleitk,
    metadata_3d_rigid_points_simpleitk,
    fname="Sparse 3D puncta raw",
)

registered_3d_rigid_points_simpleitk, details_3d_rigid_points_simpleitk = register_stack(
    stack_3d_rigid_points_simpleitk,
    registration_channel=0,  # channel used to estimate the dense SimpleITK transform
    registration_stack=0,  # reference time point
    method="phase_cross_correlation",  # coarse translation pre-estimation
    time_registration_mode="full_3d",  # full volume registration
    zreg=True,  # estimate and apply Z shifts
    rotreg=True,  # enable 3D rotation correction
    rigid_3d_backend="simpleitk",  # "phase_cross_correlation", "simpleitk", or "points"
    rot_spacing_zyx=(1.0, 1.0, 1.0),  # physical spacing in Z/Y/X; set real microscope spacing here
    rot_init_iterations=0,  # sparse puncta projections can give unstable polar-rotation pre-estimates
    rot_metric="correlation",  # "correlation" or "mutual_information"
    rot_shrink_factors=(2, 1),  # less aggressive pyramid works well for sparse puncta
    rot_smoothing_sigmas=(0.5, 0.0),  # keep sparse peaks crisp during refinement
    rot_iterations=60,  # optimizer iterations per resolution schedule
    rot_learning_rate=0.5,  # conservative optimizer step for sparse high-contrast peaks
    rot_min_step=1e-4,  # SimpleITK regular-step optimizer convergence step
    rot_sampling_percentage=None,  # full metric; sparse data are already cheap here
    rot_n_jobs=2,  # parallel time-point workers
    transform_order=0,  # nearest-neighbor keeps sparse puncta sharp
    zero_clip=False,  # keep full shape for 3D inspection; set True only for final common-valid cuboid export
    zero_clip_mode="auto",  # "auto", "shift", or "mask"
    zero_clip_mask_strategy="auto",  # "auto", "relaxed", "greedy", or "max_volume"
    zero_clip_mask_min_fraction=0.5,  # relaxed crop: lower keeps more FOV, higher removes more zero corners
    zero_clip_margin=(0, 0, 0),  # extra crop margin in Z/Y/X if zero_clip=True
    verbose=True,
    return_details=True,
)
print_rigid_comparison(
    "Sparse puncta SimpleITK 6-DOF rigid registration vs GT",
    details_3d_rigid_points_simpleitk,
    expected_shifts_3d_rigid_points_simpleitk,
    expected_rotations_3d_rigid_points_simpleitk,
)
show_before_after(
    stack_3d_rigid_points_simpleitk,
    registered_3d_rigid_points_simpleitk,
    title="Sparse puncta 3D+t SimpleITK 6-DOF registration",
    channel=0,
    moving_time=1,
)
# Optional post-hoc crop after visual inspection. Missing keys are treated as 0.
# registered_3d_rigid_points_simpleitk, metadata_3d_rigid_points_simpleitk = crop_stack(
#     registered_3d_rigid_points_simpleitk,
#     metadata_3d_rigid_points_simpleitk,
#     {"top": 1, "bottom": 1, "left": 2, "right": 2, "up": 2, "down": 2},
# )
save_stack(
    OUTPUT_DIR / "synthetic_3d_t_rigid_points_simpleitk_registered.ome.tif",
    registered_3d_rigid_points_simpleitk,
    metadata=metadata_3d_rigid_points_simpleitk,
    registration_details=details_3d_rigid_points_simpleitk,
)
maybe_open_in_napari(
    registered_3d_rigid_points_simpleitk,
    metadata_3d_rigid_points_simpleitk,
    fname="Sparse 3D puncta registered SimpleITK",
)
# %% END
