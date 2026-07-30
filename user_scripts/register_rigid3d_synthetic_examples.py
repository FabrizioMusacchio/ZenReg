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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from zenreg import (
    crop_stack,
    load_expected_rigid_corrections,
    load_stack,
    maybe_open_in_napari,
    print_available_compute,
    print_rigid_comparison,
    register_stack,
    save_stack,
    show_before_after,
)
from zenreg.synthetic import write_example_dataset

# %% DEFINE INPUT AND OUTPUT PATHS
EXAMPLE_DIR = PROJECT_ROOT / "example_data" / "synthetic_data"
OUTPUT_DIR = EXAMPLE_DIR / "registered_rigid3d"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OPEN_IN_NAPARI = False
AVAILABLE_CPUS = print_available_compute()

STACK_3D_RIGID_SIMPLEITK_PATH = EXAMPLE_DIR / "synthetic_3d_t_rigid_simpleitk.ome.tif"
STACK_3D_RIGID_POINTS_PATH = EXAMPLE_DIR / "synthetic_3d_t_rigid_points.ome.tif"

GT_3D_RIGID_SIMPLEITK_PATH = EXAMPLE_DIR / "synthetic_3d_t_rigid_simpleitk_rigid_transform_gt.csv"
GT_3D_RIGID_POINTS_PATH = EXAMPLE_DIR / "synthetic_3d_t_rigid_points_rigid_transform_gt.csv"

if not STACK_3D_RIGID_SIMPLEITK_PATH.exists() or not STACK_3D_RIGID_POINTS_PATH.exists():
    write_example_dataset(EXAMPLE_DIR)


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
maybe_open_in_napari(stack_3d_rigid_simpleitk, metadata_3d_rigid_simpleitk, fname="Dense 3D rigid raw", open_in_napari=OPEN_IN_NAPARI)

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
    open_in_napari=OPEN_IN_NAPARI,
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
maybe_open_in_napari(stack_3d_rigid_points, metadata_3d_rigid_points, fname="Sparse 3D puncta raw", open_in_napari=OPEN_IN_NAPARI)

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
    open_in_napari=OPEN_IN_NAPARI,
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
    open_in_napari=OPEN_IN_NAPARI,
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
    open_in_napari=OPEN_IN_NAPARI,
)
# %% END
