"""
Interactive VS Code script for ZenReg NoRMCorre-style synthetic benchmarks.

Run this script cell-by-cell in VS Code's interactive window. If synthetic data
are missing, this script creates them under ``example_data/synthetic_data``.
Optional CaImAn comparison blocks are commented out by default because CaImAn is
not a ZenReg dependency. To run those cells, install CaImAn separately, e.g.
``mamba install -y caiman``, then uncomment the marked CaImAn blocks.

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
CAIMAN_CACHE_DIR = SCRIPT_CACHE_DIR / "caiman"
CAIMAN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("CAIMAN_DATA", str(CAIMAN_CACHE_DIR / "data"))
os.environ.setdefault("CAIMAN_TEMP", str(CAIMAN_CACHE_DIR / "temp"))
Path(os.environ["CAIMAN_DATA"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["CAIMAN_TEMP"]).mkdir(parents=True, exist_ok=True)
if not os.access(Path.home(), os.W_OK):
    script_home = SCRIPT_CACHE_DIR / "home"
    script_home.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HOME", str(script_home))

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from zenreg import (
    load_stack,
    plot_normcorre_patch_overlay,
    print_available_compute,
    register_stack,
    save_stack,
    z_project,
)
from zenreg.synthetic import write_example_dataset

# %% PATHS
EXAMPLE_DIR = PROJECT_ROOT / "example_data" / "synthetic_data"
OUTPUT_DIR = EXAMPLE_DIR / "registered_normcorre"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
AVAILABLE_CPUS = print_available_compute()

STACK_2D_T_XY_PATH = EXAMPLE_DIR / "synthetic_2d_t_xy.ome.tif"
STACK_2D_T_LOCAL_PATH = EXAMPLE_DIR / "synthetic_2d_t_local.ome.tif"
STACK_2D_T_TRANS_ROT_PATH = EXAMPLE_DIR / "synthetic_2d_t_trans_rot_xy.ome.tif"
STACK_2D_T_PIECEWISE_XY_PATH = EXAMPLE_DIR / "synthetic_2d_t_piecewise_xy.ome.tif"
STACK_3D_T_ZYX_PATH = EXAMPLE_DIR / "synthetic_3d_t_zyx.ome.tif"
STACK_3D_T_TRANS_ROT_Z_PATH = EXAMPLE_DIR / "synthetic_3d_t_trans_rot_z.ome.tif"
STACK_3D_T_TRANS_ROT_X_PATH = EXAMPLE_DIR / "synthetic_3d_t_trans_rot_x.ome.tif"
STACK_3D_T_TRANS_ROT_ALL_CENTER_PATH = EXAMPLE_DIR / "synthetic_3d_t_trans_rot_all_center.ome.tif"
STACK_3D_T_TRANS_ROT_ALL_OFFCENTER_PATH = EXAMPLE_DIR / "synthetic_3d_t_trans_rot_all_offcenter.ome.tif"
STACK_3D_T_TRANS_ROT_ALL_OUTSIDE_PATH = EXAMPLE_DIR / "synthetic_3d_t_trans_rot_all_outside.ome.tif"

GT_2D_T_XY_PATH = EXAMPLE_DIR / "synthetic_2d_t_xy_time_shifts_gt.csv"
GT_2D_T_LOCAL_PATH = EXAMPLE_DIR / "synthetic_2d_t_local_motion_gt.csv"
GT_2D_T_TRANS_ROT_SHIFT_PATH = EXAMPLE_DIR / "synthetic_2d_t_trans_rot_xy_time_shifts_gt.csv"
GT_2D_T_TRANS_ROT_ROTATION_PATH = EXAMPLE_DIR / "synthetic_2d_t_trans_rot_xy_time_rotations_gt.csv"
GT_2D_T_PIECEWISE_XY_PATH = EXAMPLE_DIR / "synthetic_2d_t_piecewise_xy_anchor_shifts_gt.csv"
GT_3D_T_ZYX_PATH = EXAMPLE_DIR / "synthetic_3d_t_zyx_time_shifts_gt.csv"
GT_3D_T_TRANS_ROT_Z_PATH = EXAMPLE_DIR / "synthetic_3d_t_trans_rot_z_rigid_transform_gt.csv"
GT_3D_T_TRANS_ROT_X_PATH = EXAMPLE_DIR / "synthetic_3d_t_trans_rot_x_rigid_transform_gt.csv"
GT_3D_T_TRANS_ROT_ALL_CENTER_PATH = EXAMPLE_DIR / "synthetic_3d_t_trans_rot_all_center_rigid_transform_gt.csv"
GT_3D_T_TRANS_ROT_ALL_OFFCENTER_PATH = EXAMPLE_DIR / "synthetic_3d_t_trans_rot_all_offcenter_rigid_transform_gt.csv"
GT_3D_T_TRANS_ROT_ALL_OUTSIDE_PATH = EXAMPLE_DIR / "synthetic_3d_t_trans_rot_all_outside_rigid_transform_gt.csv"

OPEN_IN_NAPARI = False

if not STACK_2D_T_LOCAL_PATH.exists() or not STACK_2D_T_PIECEWISE_XY_PATH.exists():
    print("Synthetic benchmark data missing; creating them now.")
    write_example_dataset(EXAMPLE_DIR)

# %% HELPERS
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

def print_local_patch_summary(name: str, details: dict, *, t: int = 1) -> None:
    """Print patch-shift statistics for local NoRMCorre examples."""

    patch_shifts = np.asarray(details["patch_shifts"][int(t)], dtype=np.float32).reshape(-1, 2)
    print(f"{name}:")
    print(f"  rigid shift yx: {details['time_shifts_yx'][int(t)]}")
    print(f"  patch mean yx:  {patch_shifts.mean(axis=0)}")
    print(f"  patch std yx:   {patch_shifts.std(axis=0)}")
    print(f"  patch min yx:   {patch_shifts.min(axis=0)}")
    print(f"  patch max yx:   {patch_shifts.max(axis=0)}")

def run_caiman_normcorre_2d_t(
    stack,
    *,
    registration_channel: int = 0,
    pw_rigid: bool = True,
    strides: tuple[int, int] = (48, 48),
    overlaps: tuple[int, int] = (24, 24),
    max_shifts: tuple[int, int] = (6, 6),
    max_deviation_rigid: int = 3,
    niter_rig: int = 1,
    splits_rig: int = 10,
    splits_els: int = 10,
    upsample_factor_grid: int = 4,
    shifts_opencv: bool = True,
    nonneg_movie: bool = True,
    border_nan: bool | str = "copy",
    gSig_filt=None,
    shifts_interpolate: bool = False,
):
    """Run CaImAn's MotionCorrect/NoRMCorre on a 2D+t ZenReg stack.

    CaImAn estimates motion on ``registration_channel``. The resulting rigid or
    piecewise-rigid shifts are then applied to every channel, and the returned
    image is converted back to ZenReg's canonical ``TZCYX`` layout.
    """

    if stack.ndim != 5:
        raise ValueError(f"Expected a TZCYX stack, got shape {stack.shape}.")
    if stack.shape[1] != 1:
        raise ValueError("This CaImAn comparison helper currently expects 2D+t data with Z=1.")

    # import caiman as cm
    # from caiman.motion_correction import MotionCorrect
    if "cm" not in locals() or "MotionCorrect" not in locals():
        raise ImportError(
            "CaImAn is optional and not installed with ZenReg. Install it with "
            "`mamba install -y caiman`, uncomment the CaImAn imports in "
            "run_caiman_normcorre_2d_t(), and uncomment the CaImAn example cells."
        )

    movie_for_registration = np.asarray(stack[:, 0, registration_channel, :, :], dtype=np.float32)
    time_count = int(movie_for_registration.shape[0])
    if time_count < 2:
        raise ValueError("CaImAn NoRMCorre comparison requires at least two time frames.")
    effective_splits_rig = min(int(splits_rig), max(1, time_count // 2))
    effective_splits_els = min(int(splits_els), max(1, time_count // 2))
    if effective_splits_rig != splits_rig or effective_splits_els != splits_els:
        print(
            "CaImAn comparison: reduced splits to avoid single-frame chunks "
            f"(splits_rig {splits_rig}->{effective_splits_rig}, "
            f"splits_els {splits_els}->{effective_splits_els})."
        )
    mc = MotionCorrect(
        movie_for_registration,
        min_mov=float(np.nanmin(movie_for_registration)),
        max_shifts=max_shifts,
        niter_rig=niter_rig,
        splits_rig=effective_splits_rig,
        strides=strides,
        overlaps=overlaps,
        splits_els=effective_splits_els,
        upsample_factor_grid=upsample_factor_grid,
        max_deviation_rigid=max_deviation_rigid,
        shifts_opencv=shifts_opencv,
        nonneg_movie=nonneg_movie,
        gSig_filt=gSig_filt,
        use_cuda=False,
        border_nan=border_nan,
        pw_rigid=pw_rigid,
        is3D=False,
        shifts_interpolate=shifts_interpolate,
    )
    mc.motion_correct(save_movie=False)

    registered = np.asarray(stack, dtype=np.float32).copy()
    channel_files = []
    for channel in range(stack.shape[2]):
        channel_movie = np.asarray(stack[:, 0, channel, :, :], dtype=np.float32)
        channel_file = CAIMAN_CACHE_DIR / f"caiman_apply_channel_{channel}.hdf5"
        channel_files.append(channel_file)
        cm.movie(channel_movie).save(str(channel_file))
        corrected_channel = mc.apply_shifts_movie(
            str(channel_file),
            save_memmap=False,
            remove_min=False,
        )
        registered[:, 0, channel, :, :] = np.asarray(corrected_channel, dtype=np.float32)

    rigid_shifts = np.asarray(getattr(mc, "shifts_rig", []), dtype=np.float32)
    time_shifts_yx = np.zeros((stack.shape[0], 2), dtype=np.float32)
    if rigid_shifts.shape == time_shifts_yx.shape:
        time_shifts_yx = rigid_shifts
    time_shifts_zyx = np.zeros((stack.shape[0], 3), dtype=np.float32)
    time_shifts_zyx[:, 1:] = time_shifts_yx
    details = {
        "method": "caiman_normcorre",
        "pw_rigid": bool(pw_rigid),
        "registration_channel": int(registration_channel),
        "registration_stack": 0,
        "time_shifts_yx": time_shifts_yx,
        "time_shifts_zyx": time_shifts_zyx,
        "time_shifts_raw_caiman": rigid_shifts,
        "x_shifts_els": np.asarray(getattr(mc, "x_shifts_els", []), dtype=np.float32),
        "y_shifts_els": np.asarray(getattr(mc, "y_shifts_els", []), dtype=np.float32),
        "coord_shifts_els": getattr(mc, "coord_shifts_els", []),
        "border_to_0": int(getattr(mc, "border_to_0", 0)),
        "template": np.asarray(getattr(mc, "total_template_els", []), dtype=np.float32),
        "channel_files": [str(path) for path in channel_files],
        "settings": {
            "backend": "caiman.motion_correction.MotionCorrect",
            "pw_rigid": bool(pw_rigid),
            "strides": tuple(int(v) for v in strides),
            "overlaps": tuple(int(v) for v in overlaps),
            "max_shifts": tuple(int(v) for v in max_shifts),
            "max_deviation_rigid": int(max_deviation_rigid),
            "niter_rig": int(niter_rig),
            "splits_rig": int(effective_splits_rig),
            "splits_els": int(effective_splits_els),
            "requested_splits_rig": int(splits_rig),
            "requested_splits_els": int(splits_els),
            "upsample_factor_grid": int(upsample_factor_grid),
            "shifts_opencv": bool(shifts_opencv),
            "nonneg_movie": bool(nonneg_movie),
            "border_nan": border_nan,
            "gSig_filt": gSig_filt,
            "shifts_interpolate": bool(shifts_interpolate),
        },
    }
    return registered, details

def print_caiman_patch_summary(name: str, details: dict, *, t: int = 1) -> None:
    """Print CaImAn patch-shift statistics for one time frame."""

    x_shifts = np.asarray(details["x_shifts_els"][int(t)], dtype=np.float32).reshape(-1)
    y_shifts = np.asarray(details["y_shifts_els"][int(t)], dtype=np.float32).reshape(-1)
    patch_shifts_xy = np.column_stack([x_shifts, y_shifts])
    print(f"{name}:")
    print(f"  raw CaImAn rigid shift: {np.asarray(details['time_shifts_raw_caiman'])[int(t)]}")
    print(f"  patch mean xy:          {patch_shifts_xy.mean(axis=0)}")
    print(f"  patch std xy:           {patch_shifts_xy.std(axis=0)}")
    print(f"  patch min xy:           {patch_shifts_xy.min(axis=0)}")
    print(f"  patch max xy:           {patch_shifts_xy.max(axis=0)}")

def show_before_after(
    raw,
    registered,
    *,
    title: str,
    channel: int = 0,
    moving_time: int = 1,
    reference_time: int = 0,
    projection_method: str = "max",
) -> None:
    """Show two time points and their difference before/after registration."""

    raw_proj = z_project(raw[:, :, channel : channel + 1, :, :], projection_method=projection_method)[:, 0, 0]
    reg_proj = z_project(registered[:, :, channel : channel + 1, :, :], projection_method=projection_method)[:, 0, 0]
    reference_time = int(np.clip(reference_time, 0, raw_proj.shape[0] - 1))
    moving_time = int(np.clip(moving_time, 0, raw_proj.shape[0] - 1))
    raw_diff = raw_proj[moving_time] - raw_proj[reference_time]
    reg_diff = reg_proj[moving_time] - reg_proj[reference_time]
    vmax = max(float(np.max(np.abs(raw_diff))), float(np.max(np.abs(reg_diff))), 1e-6)

    fig, axes = plt.subplots(2, 3, figsize=(10, 6), constrained_layout=True)
    axes[0, 0].imshow(raw_proj[reference_time], cmap="gray")
    axes[0, 0].set_title(f"raw t={reference_time}")
    axes[0, 1].imshow(raw_proj[moving_time], cmap="gray")
    axes[0, 1].set_title(f"raw t={moving_time}")
    axes[0, 2].imshow(raw_diff, cmap="bwr", vmin=-vmax, vmax=vmax)
    axes[0, 2].set_title(f"raw t{moving_time} - t{reference_time}")
    axes[1, 0].imshow(reg_proj[reference_time], cmap="gray")
    axes[1, 0].set_title(f"registered t={reference_time}")
    axes[1, 1].imshow(reg_proj[moving_time], cmap="gray")
    axes[1, 1].set_title(f"registered t={moving_time}")
    axes[1, 2].imshow(reg_diff, cmap="bwr", vmin=-vmax, vmax=vmax)
    axes[1, 2].set_title(f"registered t{moving_time} - t{reference_time}")
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle(title)
    plt.show()

def show_residual_comparison(
    raw,
    registered_zenreg,
    registered_caiman,
    *,
    title: str,
    channel: int = 0,
    moving_time: int = 1,
    reference_time: int = 0,
    projection_method: str = "max",
    labels: tuple[str, str, str] = ("raw", "ZenReg NoRMCorre", "CaImAn NoRMCorre"),
) -> None:
    """Compare residual difference images for raw, ZenReg, and CaImAn outputs."""

    def _project(stack):
        return z_project(stack[:, :, channel : channel + 1, :, :], projection_method=projection_method)[:, 0, 0]

    raw_proj = _project(raw)
    zenreg_proj = _project(registered_zenreg)
    caiman_proj = _project(registered_caiman)
    reference_time = int(np.clip(reference_time, 0, raw_proj.shape[0] - 1))
    moving_time = int(np.clip(moving_time, 0, raw_proj.shape[0] - 1))
    residuals = [
        raw_proj[moving_time] - raw_proj[reference_time],
        zenreg_proj[moving_time] - zenreg_proj[reference_time],
        caiman_proj[moving_time] - caiman_proj[reference_time],
    ]
    labels = list(labels)
    vmax = max(float(np.max(np.abs(diff))) for diff in residuals)
    vmax = max(vmax, 1e-6)

    print(f"{title} residual MAE at t={moving_time} vs t={reference_time}:")
    for label, diff in zip(labels, residuals):
        print(f"  {label}: {float(np.mean(np.abs(diff))):.6f}")

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), constrained_layout=True)
    for ax, label, diff in zip(axes, labels, residuals):
        ax.imshow(diff, cmap="bwr", vmin=-vmax, vmax=vmax)
        ax.set_title(label)
        ax.axis("off")
    fig.suptitle(title)
    plt.show()

def show_residual_comparison_multi(
    raw,
    registered_stacks,
    *,
    title: str,
    labels: tuple[str, ...],
    channel: int = 0,
    moving_time: int = 1,
    reference_time: int = 0,
    projection_method: str = "max",
) -> None:
    """Compare residual difference images for raw and multiple registered outputs."""

    stacks = (raw, *tuple(registered_stacks))
    if len(labels) != len(stacks):
        raise ValueError("labels must contain one entry for raw and each registered stack.")

    residuals = []
    for stack in stacks:
        proj = z_project(stack[:, :, channel : channel + 1, :, :], projection_method=projection_method)[:, 0, 0]
        reference_time_clipped = int(np.clip(reference_time, 0, proj.shape[0] - 1))
        moving_time_clipped = int(np.clip(moving_time, 0, proj.shape[0] - 1))
        residuals.append(proj[moving_time_clipped] - proj[reference_time_clipped])

    vmax = max(float(np.max(np.abs(diff))) for diff in residuals)
    vmax = max(vmax, 1e-6)
    fig, axes = plt.subplots(1, len(stacks), figsize=(3.4 * len(stacks), 3.5), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, label, diff in zip(axes, labels, residuals):
        ax.imshow(diff, cmap="bwr", vmin=-vmax, vmax=vmax)
        ax.set_title(label)
        ax.axis("off")
    fig.suptitle(title)
    plt.show()

def print_residual_mae_summary(
    raw,
    *registered_stacks,
    labels: tuple[str, ...],
    channel: int = 0,
    reference_time: int = 0,
    projection_method: str = "max",
) -> None:
    """Print mean residual error to the reference frame for several stacks."""

    stacks = (raw, *registered_stacks)
    if len(labels) != len(stacks):
        raise ValueError("labels must contain one entry for raw and each registered stack.")
    print(f"Residual MAE to t={reference_time}:")
    for label, stack in zip(labels, stacks):
        proj = z_project(stack[:, :, channel : channel + 1, :, :], projection_method=projection_method)[:, 0, 0]
        reference = proj[int(reference_time)]
        residuals = np.mean(np.abs(proj - reference[None, :, :]), axis=(1, 2))
        print(
            f"  {label}: mean={float(np.mean(residuals)):.6f}, "
            f"max={float(np.max(residuals)):.6f}"
        )

def maybe_open_in_napari(stack, metadata, *, fname: str) -> None:
    """Open a stack in Napari when the interactive flag is enabled."""

    if OPEN_IN_NAPARI:
        import omio as om

        om.open_in_napari(stack, metadata, fname=fname)

# %% 1) 2D+t: GLOBAL XY TRANSLATION
stack_2d_t_xy, metadata_2d_t_xy = load_stack(
    STACK_2D_T_XY_PATH,
    return_metadata=True,
    verbose=False,
)
expected_2d_t_xy = load_expected_time_registration_shifts(GT_2D_T_XY_PATH, registration_stack=0, axes="yx")
print(f"2D+t global XY stack shape: {stack_2d_t_xy.shape} (TZCYX)")
maybe_open_in_napari(stack_2d_t_xy, metadata_2d_t_xy, fname="2D+t global XY")


registered_2d_t_xy_phase, details_2d_t_xy_phase = register_stack(
    stack_2d_t_xy,
    registration_channel=0,  # channel used for global phase-cross shift estimation
    registration_stack=0,  # reference time point/template
    method="phase_cross_correlation",  # "phase_cross_correlation" or "pystackreg"
    time_registration_mode="projection",  # register 2D YX projections over time
    time_reference_mode="template",  # register every t to registration_stack
    projection_method="max",  # "max", "mean", "median", "var", or "std"
    zreg=False,  # no Z motion in this 2D+t benchmark
    zero_clip=False,  # keep original shape for visual comparison
    max_xy_shifts=(6, 6),  # absolute correction shift limits in YX
    rotreg=False,  # no rotation correction; this dataset has no rotation
    transform_backend="skimage",  # "skimage" or "scipy"
    transform_order=1,  # 1 for intensity images; 0 for sparse puncta/labels
    filter_slices=False,  # median-filter Z slices before projection
    filter_projections=False,  # median-filter projections before shift estimation
    phase_cross_correlation_upsample_factor=20,  # subpixel precision
    phase_cross_correlation_normalization=None,  # None or "phase"
    n_jobs=2,  # CPU worker threads for independent time points/slices
    verbose=True,
    return_shifts=True,
    return_details=True)

plot_normcorre_patch_overlay(
    stack_2d_t_xy,
    metadata_2d_t_xy,
    registration_channel=0,
    registration_stack=0,
    nc_strides=(32, 32),
    nc_overlaps=(16, 16),
    projection_method="max",
    projection_range=(1, 10),
    output_dir=OUTPUT_DIR,
)

registered_2d_t_xy_normcorre, details_2d_t_xy_normcorre = register_stack(
    stack_2d_t_xy,
    registration_channel=0,  # channel used for NoRMCorre shift estimation
    registration_stack=0,  # reference time point/template
    method="normcorre",  # dispatches to ZenReg's CaImAn-compatible NoRMCorre port
    time_registration_mode="projection",  # "projection" for 2D/projection mode, "full_3d" for ZYX NoRMCorre
    projection_method="max",  # "max", "mean", "median", "var", or "std"
    zreg=False,  # full 3D/Z registration is disabled for this 2D+t example
    max_xy_shifts=(6, 6),  # absolute correction shift limits in YX
    phase_cross_correlation_upsample_factor=10,  # subpixel phase-correlation precision
    phase_cross_correlation_normalization=None,  # None or "phase"
    transform_order=3,  # 3 matches CaImAn's cubic remap; use 1 for gentler intensity interpolation
    nc_pw_rigid=True,  # True = patch-wise NoRMCorre, False = rigid-only baseline
    nc_strides=(32, 32),  # patch-grid stride in YX
    nc_overlaps=(16, 16),  # patch overlap in YX
    nc_max_deviation_rigid=3,  # local patch-shift deviation from rigid estimate
    nc_n_iterations=1,  # template-refinement passes
    nc_correction_iterations=1,  # re-run NoRMCorre on the already corrected result
    nc_n_jobs=2,  # parallelize over time frames
    nc_block_size=32,  # Y rows warped per block
    nc_output_use_memmap=False,  # True writes registered output into OMIO/Zarr
    nc_output_memmap_folder=None,  # local scratch folder for output Zarr if enabled
    nc_output_memmap_name="2d_t_xy_normcorre_registered_zarr",  # output Zarr store name
    verbose=True,
    return_details=True)

# Optional CaImAn comparison; requires ``mamba install -y caiman`` and uncommented imports above.
# registered_2d_t_xy_caiman, details_2d_t_xy_caiman = run_caiman_normcorre_2d_t(
#     stack_2d_t_xy,
#     registration_channel=0,  # channel used by CaImAn to estimate the motion field
#     pw_rigid=True,  # True = original CaImAn piecewise-rigid NoRMCorre
#     strides=(32, 32),  # patch-grid stride in YX
#     overlaps=(16, 16),  # patch overlap in YX
#     max_shifts=(6, 6),  # absolute correction shift limits in YX
#     max_deviation_rigid=3,  # local patch-shift deviation from rigid estimate
#     niter_rig=1,  # CaImAn rigid template initialization iterations
#     splits_rig=10,  # time chunks for rigid initialization
#     splits_els=10,  # time chunks for piecewise-rigid registration
#     upsample_factor_grid=4,  # CaImAn local-shift grid upsampling
#     shifts_opencv=True,  # CaImAn's default fast OpenCV transform backend
#     nonneg_movie=True,  # keep CaImAn's saved/intermediate movie non-negative
#     border_nan="copy",  # copy edge values instead of introducing NaNs
#     gSig_filt=None,  # optional spatial high-pass filter kernel, e.g. (3, 3)
#     shifts_interpolate=False,  # CaImAn default patch-field interpolation mode
# )

print_shift_comparison(
    "2D+t global XY phase-cross shifts",
    details_2d_t_xy_phase["time_shifts_yx"],
    expected_2d_t_xy,
)
print_shift_comparison(
    "2D+t global XY ZenReg NoRMCorre rigid shifts",
    details_2d_t_xy_normcorre["time_shifts_yx"],
    expected_2d_t_xy,
)
print_residual_mae_summary(
    stack_2d_t_xy,
    registered_2d_t_xy_phase,
    registered_2d_t_xy_normcorre,
    labels=("raw", "phase cross", "ZenReg NoRMCorre"),
    channel=0,
    reference_time=0,
)
show_before_after(
    stack_2d_t_xy,
    registered_2d_t_xy_phase,
    title="2D+t global XY translation (phase cross)",
    channel=0,
)
show_before_after(
    stack_2d_t_xy,
    registered_2d_t_xy_normcorre,
    title="2D+t global XY translation (ZenReg NoRMCorre)",
    channel=0,
)
# show_before_after(
#     stack_2d_t_xy,
#     registered_2d_t_xy_caiman,
#     title="2D+t global XY translation (CaImAn NoRMCorre)",
#     channel=0,
# )
show_residual_comparison_multi(
    stack_2d_t_xy,
    (
        registered_2d_t_xy_phase,
        registered_2d_t_xy_normcorre,
    ),
    title="2D+t global XY residual comparison",
    labels=("raw", "phase cross", "ZenReg NoRMCorre"),
    channel=0,
    moving_time=1,
)
save_stack(
    OUTPUT_DIR / "2d_t_global_xy_phase_cross_registered.ome.tif",
    registered_2d_t_xy_phase,
    metadata=metadata_2d_t_xy,
    registration_details=details_2d_t_xy_phase,
)
save_stack(
    OUTPUT_DIR / "2d_t_global_xy_normcorre_registered.ome.tif",
    registered_2d_t_xy_normcorre,
    metadata=metadata_2d_t_xy,
    registration_details=details_2d_t_xy_normcorre,
)
# save_stack(
#     OUTPUT_DIR / "2d_t_global_xy_caiman_normcorre_registered.ome.tif",
#     registered_2d_t_xy_caiman,
#     metadata=metadata_2d_t_xy,
#     registration_details=details_2d_t_xy_caiman,
# )

maybe_open_in_napari(registered_2d_t_xy_phase, metadata_2d_t_xy, fname="2D+t global XY phase cross")
maybe_open_in_napari(registered_2d_t_xy_normcorre, metadata_2d_t_xy, fname="2D+t global XY ZenReg NoRMCorre")
# maybe_open_in_napari(registered_2d_t_xy_caiman, metadata_2d_t_xy, fname="2D+t global XY CaImAn NoRMCorre")
# %% 2) 2D+t: LOCAL IN-FRAME MOTION
stack_2d_t_local, metadata_2d_t_local = load_stack(
    STACK_2D_T_LOCAL_PATH,
    return_metadata=True,
    verbose=False)
local_motion_gt_2d_t = _load_csv(GT_2D_T_LOCAL_PATH)
local_motion_frame_2d_t = int(local_motion_gt_2d_t["t"][np.argmax(local_motion_gt_2d_t["motion_magnitude"])])
print(f"2D+t local-motion stack shape: {stack_2d_t_local.shape} (TZCYX)")
print("Local GT columns:", local_motion_gt_2d_t.dtype.names)
print(f"Strongest local-motion frame: t={local_motion_frame_2d_t}")

plot_normcorre_patch_overlay(
    stack_2d_t_local,
    metadata_2d_t_local,
    registration_channel=0,
    registration_stack=0,
    nc_strides=(14, 14),
    nc_overlaps=(24, 24),
    projection_method="max",
    projection_range=(1, 10),
    output_dir=OUTPUT_DIR,
)

registered_2d_t_local, details_2d_t_local = register_stack(
    stack_2d_t_local,
    registration_channel=0,  # channel used for NoRMCorre shift estimation
    registration_stack=0,  # reference time point/template
    method="normcorre",  # dispatches to ZenReg's CaImAn-compatible NoRMCorre port
    time_registration_mode="projection",  # 2D+t/projection local shift fields
    projection_method="max",  # "max", "mean", "median", "var", or "std"
    zreg=False,  # full 3D/Z registration is disabled for this 2D+t example
    max_xy_shifts=(5, 5),  # absolute correction shift limits in YX
    phase_cross_correlation_upsample_factor=10,  # subpixel phase-correlation precision
    phase_cross_correlation_normalization=None,  # None or "phase"
    transform_order=3,  # 3 matches CaImAn's cubic remap; use 1 for gentler intensity interpolation
    nc_pw_rigid=True,  # patch-wise NoRMCorre local correction
    nc_strides=(14, 14),  # patch-grid stride in YX
    nc_overlaps=(24, 24),  # overlap smooths local patch transitions
    nc_max_deviation_rigid=3,  # local patch-shift deviation from rigid estimate
    nc_n_iterations=3,  # template-refinement passes
    nc_correction_iterations=1,  # re-run NoRMCorre on the already corrected result
    nc_n_jobs=2,  # parallelize over time frames
    nc_block_size=32,  # Y rows warped per block
    nc_output_use_memmap=False,  # True writes registered output into OMIO/Zarr
    nc_output_memmap_folder=None,  # local scratch folder for output Zarr if enabled
    nc_output_memmap_name="2d_t_local_normcorre_registered_zarr",  # output Zarr store name
    verbose=True,
    return_details=True,
)
print_local_patch_summary("2D+t local NoRMCorre patch shifts", details_2d_t_local, t=local_motion_frame_2d_t)
show_before_after(
    stack_2d_t_local,
    registered_2d_t_local,
    title="2D+t local in-frame motion",
    channel=0,
    moving_time=local_motion_frame_2d_t,
)
save_stack(
    OUTPUT_DIR / "2d_t_local_normcorre_registered.ome.tif",
    registered_2d_t_local,
    metadata=metadata_2d_t_local,
    registration_details=details_2d_t_local,
)

maybe_open_in_napari(stack_2d_t_local, metadata_2d_t_local, fname="2D+t local")
maybe_open_in_napari(registered_2d_t_local, metadata_2d_t_local, fname="2D+t local NoRMCorre")

# Optional CaImAn comparison; requires ``mamba install -y caiman`` and uncommented imports above.
# registered_2d_t_local_caiman, details_2d_t_local_caiman = run_caiman_normcorre_2d_t(
#     stack_2d_t_local,
#     registration_channel=0,  # channel used by CaImAn to estimate the motion field
#     pw_rigid=True,  # True = original CaImAn piecewise-rigid NoRMCorre
#     strides=(14, 14),  # patch-grid stride in YX
#     overlaps=(24, 24),  # patch overlap in YX
#     max_shifts=(5, 5),  # absolute correction shift limits in YX
#     max_deviation_rigid=3,  # local patch-shift deviation from rigid estimate
#     niter_rig=1,  # CaImAn rigid template initialization iterations
#     splits_rig=10,  # time chunks for rigid initialization
#     splits_els=10,  # time chunks for piecewise-rigid registration
#     upsample_factor_grid=4,  # CaImAn local-shift grid upsampling
#     shifts_opencv=True,  # CaImAn's default fast OpenCV transform backend
#     nonneg_movie=True,  # keep CaImAn's saved/intermediate movie non-negative
#     border_nan="copy",  # copy edge values instead of introducing NaNs
#     gSig_filt=None,  # optional spatial high-pass filter kernel, e.g. (3, 3)
#     shifts_interpolate=False,  # CaImAn default patch-field interpolation mode
# )
# print_caiman_patch_summary(
#     "2D+t local CaImAn NoRMCorre patch shifts",
#     details_2d_t_local_caiman,
#     t=local_motion_frame_2d_t,
# )
# show_before_after(
#     stack_2d_t_local,
#     registered_2d_t_local_caiman,
#     title="2D+t local in-frame motion (CaImAn NoRMCorre)",
#     channel=0,
#     moving_time=local_motion_frame_2d_t,
# )
# show_residual_comparison(
#     stack_2d_t_local,
#     registered_2d_t_local,
#     registered_2d_t_local_caiman,
#     title="2D+t local residual comparison",
#     channel=0,
#     moving_time=local_motion_frame_2d_t,
# )
# save_stack(
#     OUTPUT_DIR / "2d_t_local_caiman_normcorre_registered.ome.tif",
#     registered_2d_t_local_caiman,
#     metadata=metadata_2d_t_local,
#     registration_details=details_2d_t_local_caiman,
# )
# maybe_open_in_napari(
#     registered_2d_t_local_caiman,
#     metadata_2d_t_local,
#     fname="2D+t local CaImAn NoRMCorre",
# )
# %% 3) 2D+t: GLOBAL XY TRANSLATION PLUS LIGHT ROTATION
stack_2d_t_trans_rot, metadata_2d_t_trans_rot = load_stack(
    STACK_2D_T_TRANS_ROT_PATH,
    return_metadata=True,
    verbose=False,
)
expected_2d_t_trans_rot = load_expected_time_registration_shifts(
    GT_2D_T_TRANS_ROT_SHIFT_PATH,
    registration_stack=0,
    axes="yx",
)
expected_2d_t_trans_rot_deg = load_expected_time_registration_rotations(
    GT_2D_T_TRANS_ROT_ROTATION_PATH,
    registration_stack=0,
)
rotation_event_frame_2d_t = int(np.argmax(np.abs(expected_2d_t_trans_rot_deg)))
print(f"2D+t translation+rotation stack shape: {stack_2d_t_trans_rot.shape} (TZCYX)")
print("Applied rotation corrections [deg], first rows:", expected_2d_t_trans_rot_deg[:5])
print(
    f"Strongest rotation-correction frame: t={rotation_event_frame_2d_t}, "
    f"expected correction={expected_2d_t_trans_rot_deg[rotation_event_frame_2d_t]:.3f} deg"
)
maybe_open_in_napari(stack_2d_t_trans_rot, metadata_2d_t_trans_rot, fname="2D+t translation+rotation")

plot_normcorre_patch_overlay(
    stack_2d_t_trans_rot,
    metadata_2d_t_trans_rot,
    registration_channel=0,
    registration_stack=0,
    nc_strides=(16, 16),
    nc_overlaps=(24, 24),
    projection_method="max",
    projection_range=(1, 10),
    output_dir=OUTPUT_DIR,
)

registered_2d_t_trans_rot, details_2d_t_trans_rot = register_stack(
    stack_2d_t_trans_rot,
    registration_channel=0,  # channel used for NoRMCorre shift estimation
    registration_stack=0,  # reference time point/template
    method="normcorre",  # dispatches to ZenReg's CaImAn-compatible NoRMCorre port
    time_registration_mode="projection",  # 2D+t/projection local shift fields
    projection_method="max",  # "max", "mean", "median", "var", or "std"
    zreg=False,  # full 3D/Z registration is disabled for this 2D+t example
    max_xy_shifts=(6, 6),  # absolute correction shift limits in YX
    phase_cross_correlation_upsample_factor=10,  # subpixel phase-correlation precision
    phase_cross_correlation_normalization=None,  # None or "phase"
    transform_order=3,  # 3 matches CaImAn's cubic remap; use 1 for gentler intensity interpolation
    nc_pw_rigid=True,  # approximate small rotations via local translations
    nc_strides=(16, 16),  # patch-grid stride in YX
    nc_overlaps=(24, 24),  # patch overlap in YX
    nc_max_deviation_rigid=5,  # local patch-shift deviation from rigid estimate
    nc_n_iterations=2,  # template-refinement passes
    nc_correction_iterations=1,  # re-run NoRMCorre on the already corrected result
    nc_n_jobs=2,  # parallelize over time frames
    nc_block_size=32,  # Y rows warped per block
    nc_output_use_memmap=False,  # True writes registered output into OMIO/Zarr
    nc_output_memmap_folder=None,  # local scratch folder for output Zarr if enabled
    nc_output_memmap_name="2d_t_trans_rot_normcorre_registered_zarr",  # output Zarr store name
    verbose=True,
    return_details=True,
)
print_shift_comparison(
    "2D+t translation+rotation rigid shifts vs translation GT",
    details_2d_t_trans_rot["time_shifts_yx"],
    expected_2d_t_trans_rot,
)
print_local_patch_summary(
    "2D+t translation+rotation NoRMCorre patch shifts",
    details_2d_t_trans_rot,
    t=rotation_event_frame_2d_t,
)
show_before_after(
    stack_2d_t_trans_rot,
    registered_2d_t_trans_rot,
    title="2D+t translation plus rotation",
    channel=0,
    moving_time=rotation_event_frame_2d_t,
)
save_stack(
    OUTPUT_DIR / "2d_t_translation_rotation_normcorre_registered.ome.tif",
    registered_2d_t_trans_rot,
    metadata=metadata_2d_t_trans_rot,
    registration_details=details_2d_t_trans_rot,
)

maybe_open_in_napari(registered_2d_t_trans_rot, metadata_2d_t_trans_rot, fname="2D+t translation+rotation NoRMCorre")

# Optional CaImAn comparison; requires ``mamba install -y caiman`` and uncommented imports above.
# registered_2d_t_trans_rot_caiman, details_2d_t_trans_rot_caiman = run_caiman_normcorre_2d_t(
#     stack_2d_t_trans_rot,
#     registration_channel=0,  # channel used by CaImAn to estimate the motion field
#     pw_rigid=True,  # True = original CaImAn piecewise-rigid NoRMCorre
#     strides=(12, 12),  # patch-grid stride in YX
#     overlaps=(24, 24),  # patch overlap in YX
#     max_shifts=(6, 6),  # absolute correction shift limits in YX
#     max_deviation_rigid=10,  # local patch-shift deviation from rigid estimate
#     niter_rig=3,  # CaImAn rigid template initialization iterations
#     splits_rig=10,  # time chunks for rigid initialization
#     splits_els=10,  # time chunks for piecewise-rigid registration
#     upsample_factor_grid=4,  # CaImAn local-shift grid upsampling
#     shifts_opencv=True,  # CaImAn's default fast OpenCV transform backend
#     nonneg_movie=True,  # keep CaImAn's saved/intermediate movie non-negative
#     border_nan="copy",  # copy edge values instead of introducing NaNs
#     gSig_filt=None,  # optional spatial high-pass filter kernel, e.g. (3, 3)
#     shifts_interpolate=False,  # CaImAn default patch-field interpolation mode
# )
# print_shift_comparison(
#     "2D+t translation+rotation CaImAn rigid shifts vs translation GT",
#     details_2d_t_trans_rot_caiman["time_shifts_yx"],
#     expected_2d_t_trans_rot,
# )
# print_caiman_patch_summary(
#     "2D+t translation+rotation CaImAn NoRMCorre patch shifts",
#     details_2d_t_trans_rot_caiman,
#     t=rotation_event_frame_2d_t,
# )
# show_before_after(
#     stack_2d_t_trans_rot,
#     registered_2d_t_trans_rot_caiman,
#     title="2D+t translation plus rotation (CaImAn NoRMCorre)",
#     channel=0,
#     moving_time=rotation_event_frame_2d_t,
# )
# show_residual_comparison(
#     stack_2d_t_trans_rot,
#     registered_2d_t_trans_rot,
#     registered_2d_t_trans_rot_caiman,
#     title="2D+t translation+rotation residual comparison",
#     channel=0,
#     moving_time=rotation_event_frame_2d_t,
# )
# save_stack(
#     OUTPUT_DIR / "2d_t_translation_rotation_caiman_normcorre_registered.ome.tif",
#     registered_2d_t_trans_rot_caiman,
#     metadata=metadata_2d_t_trans_rot,
#     registration_details=details_2d_t_trans_rot_caiman,
# )
# maybe_open_in_napari(
#     registered_2d_t_trans_rot_caiman,
#     metadata_2d_t_trans_rot,
#     fname="2D+t translation+rotation CaImAn NoRMCorre",
# )
# %% 4) 2D+t: PIECEWISE XY TRANSLATION, PHASE CROSS VS NoRMCorre
stack_2d_t_piecewise_xy, metadata_2d_t_piecewise_xy = load_stack(
    STACK_2D_T_PIECEWISE_XY_PATH,
    return_metadata=True,
    verbose=False,
)
piecewise_gt_2d_t = _load_csv(GT_2D_T_PIECEWISE_XY_PATH)
piecewise_motion_by_t = np.zeros(stack_2d_t_piecewise_xy.shape[0], dtype=np.float32)
for t in range(stack_2d_t_piecewise_xy.shape[0]):
    rows_t = piecewise_gt_2d_t[piecewise_gt_2d_t["t"] == t]
    piecewise_motion_by_t[t] = float(
        np.mean(
            np.hypot(
                rows_t["expected_anchor_correction_shift_y_ref_t0"],
                rows_t["expected_anchor_correction_shift_x_ref_t0"],
            )
        )
    )
piecewise_event_frame_2d_t = int(np.argmax(piecewise_motion_by_t))
print(f"2D+t piecewise XY stack shape: {stack_2d_t_piecewise_xy.shape} (TZCYX)")
print(f"Strongest piecewise-translation frame: t={piecewise_event_frame_2d_t}")

plot_normcorre_patch_overlay(
    stack_2d_t_piecewise_xy,
    metadata_2d_t_piecewise_xy,
    registration_channel=0,
    registration_stack=0,
    nc_strides=(24, 24),
    nc_overlaps=(24, 24),
    projection_method="max",
    projection_range=(1, 10),
    output_dir=OUTPUT_DIR,
)

registered_2d_t_piecewise_phase, details_2d_t_piecewise_phase = register_stack(
    stack_2d_t_piecewise_xy,
    registration_channel=0,  # channel used for global phase-cross shift estimation
    registration_stack=0,  # reference time point/template
    method="phase_cross_correlation",  # global rigid translation baseline
    time_registration_mode="projection",  # register 2D YX projections over time
    time_reference_mode="template",  # register every t to registration_stack
    projection_method="max",  # "max", "mean", "median", "var", or "std"
    zreg=False,  # no Z motion in this 2D+t benchmark
    zero_clip=False,  # keep original shape for visual comparison
    max_xy_shifts=(6, 6),  # absolute correction shift limits in YX
    rotreg=False,  # no rotation correction; this dataset has no rotation
    transform_backend="skimage",  # "skimage" or "scipy"
    transform_order=3,  # 3 matches CaImAn's cubic remap; use 1 for gentler intensity interpolation
    filter_slices=False,  # median-filter Z slices before projection
    filter_projections=False,  # median-filter projections before shift estimation
    phase_cross_correlation_upsample_factor=20,  # subpixel precision
    phase_cross_correlation_normalization=None,  # None or "phase"
    n_jobs=2,  # CPU worker threads for independent time points/slices
    verbose=True,
    return_shifts=True,
    return_details=True,
)

registered_2d_t_piecewise_normcorre, details_2d_t_piecewise_normcorre = register_stack(
    stack_2d_t_piecewise_xy,
    registration_channel=0,  # channel used for NoRMCorre shift estimation
    registration_stack=0,  # reference time point/template
    method="normcorre",  # dispatches to ZenReg's CaImAn-compatible NoRMCorre port
    time_registration_mode="projection",  # 2D+t/projection local shift fields
    projection_method="max",  # "max", "mean", "median", "var", or "std"
    zreg=False,  # full 3D/Z registration is disabled for this 2D+t example
    max_xy_shifts=(6, 6),  # absolute correction shift limits in YX
    phase_cross_correlation_upsample_factor=10,  # subpixel phase-correlation precision
    phase_cross_correlation_normalization=None,  # None or "phase"
    transform_order=3,  # 3 matches CaImAn's cubic remap; use 1 for gentler intensity interpolation
    nc_pw_rigid=True,  # patch-wise local translations
    nc_strides=(24, 24),  # patch-grid stride in YX
    nc_overlaps=(24, 24),  # patch overlap in YX
    nc_max_deviation_rigid=5,  # local patch-shift deviation from rigid estimate
    nc_n_iterations=2,  # template-refinement passes
    nc_correction_iterations=1,  # re-run NoRMCorre on the already corrected result
    nc_n_jobs=2,  # parallelize over time frames
    nc_block_size=32,  # Y rows warped per block
    nc_output_use_memmap=False,  # True writes registered output into OMIO/Zarr
    nc_output_memmap_folder=None,  # local scratch folder for output Zarr if enabled
    nc_output_memmap_name="2d_t_piecewise_xy_normcorre_registered_zarr",  # output Zarr store name
    verbose=True,
    return_details=True,
)

# Optional CaImAn comparison; requires ``mamba install -y caiman`` and uncommented imports above.
# registered_2d_t_piecewise_caiman, details_2d_t_piecewise_caiman = run_caiman_normcorre_2d_t(
#     stack_2d_t_piecewise_xy,
#     registration_channel=0,  # channel used by CaImAn to estimate the motion field
#     pw_rigid=True,  # True = original CaImAn piecewise-rigid NoRMCorre
#     strides=(24, 24),  # patch-grid stride in YX
#     overlaps=(24, 24),  # patch overlap in YX
#     max_shifts=(6, 6),  # absolute correction shift limits in YX
#     max_deviation_rigid=5,  # local patch-shift deviation from rigid estimate
#     niter_rig=1,  # CaImAn rigid template initialization iterations
#     splits_rig=10,  # time chunks for rigid initialization
#     splits_els=10,  # time chunks for piecewise-rigid registration
#     upsample_factor_grid=4,  # CaImAn local-shift grid upsampling
#     shifts_opencv=True,  # CaImAn's default fast OpenCV transform backend
#     nonneg_movie=True,  # keep CaImAn's saved/intermediate movie non-negative
#     border_nan="copy",  # copy edge values instead of introducing NaNs
#     gSig_filt=None,  # optional spatial high-pass filter kernel, e.g. (3, 3)
#     shifts_interpolate=False,  # CaImAn default patch-field interpolation mode
# )

print_local_patch_summary(
    "2D+t piecewise XY NoRMCorre patch shifts",
    details_2d_t_piecewise_normcorre,
    t=piecewise_event_frame_2d_t,
)
# print_caiman_patch_summary(
#     "2D+t piecewise XY CaImAn NoRMCorre patch shifts",
#     details_2d_t_piecewise_caiman,
#     t=piecewise_event_frame_2d_t,
# )
print_residual_mae_summary(
    stack_2d_t_piecewise_xy,
    registered_2d_t_piecewise_phase,
    registered_2d_t_piecewise_normcorre,
    labels=("raw", "phase_cross_correlation", "ZenReg NoRMCorre"),
    channel=0,
)
show_before_after(
    stack_2d_t_piecewise_xy,
    registered_2d_t_piecewise_phase,
    title="2D+t piecewise XY translation (phase cross)",
    channel=0,
    moving_time=piecewise_event_frame_2d_t,
)
show_before_after(
    stack_2d_t_piecewise_xy,
    registered_2d_t_piecewise_normcorre,
    title="2D+t piecewise XY translation (NoRMCorre)",
    channel=0,
    moving_time=piecewise_event_frame_2d_t,
)
# show_before_after(
#     stack_2d_t_piecewise_xy,
#     registered_2d_t_piecewise_caiman,
#     title="2D+t piecewise XY translation (CaImAn NoRMCorre)",
#     channel=0,
#     moving_time=piecewise_event_frame_2d_t,
# )
show_residual_comparison_multi(
    stack_2d_t_piecewise_xy,
    (
        registered_2d_t_piecewise_phase,
        registered_2d_t_piecewise_normcorre,
    ),
    title="2D+t piecewise XY residual comparison",
    channel=0,
    moving_time=piecewise_event_frame_2d_t,
    labels=("raw", "phase_cross_correlation", "ZenReg NoRMCorre"),
)
save_stack(
    OUTPUT_DIR / "2d_t_piecewise_xy_phase_cross_registered.ome.tif",
    registered_2d_t_piecewise_phase,
    metadata=metadata_2d_t_piecewise_xy,
    registration_details=details_2d_t_piecewise_phase,
)
save_stack(
    OUTPUT_DIR / "2d_t_piecewise_xy_normcorre_registered.ome.tif",
    registered_2d_t_piecewise_normcorre,
    metadata=metadata_2d_t_piecewise_xy,
    registration_details=details_2d_t_piecewise_normcorre,
)
# save_stack(
#     OUTPUT_DIR / "2d_t_piecewise_xy_caiman_normcorre_registered.ome.tif",
#     registered_2d_t_piecewise_caiman,
#     metadata=metadata_2d_t_piecewise_xy,
#     registration_details=details_2d_t_piecewise_caiman,
# )
maybe_open_in_napari(stack_2d_t_piecewise_xy, metadata_2d_t_piecewise_xy, fname="2D+t piecewise XY")
maybe_open_in_napari(
    registered_2d_t_piecewise_phase,
    metadata_2d_t_piecewise_xy,
    fname="2D+t piecewise XY phase cross",
)
maybe_open_in_napari(
    registered_2d_t_piecewise_normcorre,
    metadata_2d_t_piecewise_xy,
    fname="2D+t piecewise XY NoRMCorre",
)
# maybe_open_in_napari(
#     registered_2d_t_piecewise_caiman,
#     metadata_2d_t_piecewise_xy,
#     fname="2D+t piecewise XY CaImAn NoRMCorre",
# )
# %% 5) 3D+t: GLOBAL ZYX TRANSLATION
stack_3d_t_zyx, metadata_3d_t_zyx = load_stack(
    STACK_3D_T_ZYX_PATH,
    return_metadata=True,
    verbose=False,
)
expected_3d_t_zyx = load_expected_time_registration_shifts(GT_3D_T_ZYX_PATH, registration_stack=0, axes="zyx")
print(f"3D+t global ZYX stack shape: {stack_3d_t_zyx.shape} (TZCYX)")

plot_normcorre_patch_overlay(
    stack_3d_t_zyx,
    metadata_3d_t_zyx,
    registration_channel=0,
    registration_stack=0,
    nc_strides=(6, 48, 48),
    nc_overlaps=(3, 24, 24),
    projection_method="max",
    projection_range=(1, 10),
    output_dir=OUTPUT_DIR,
)

registered_3d_t_zyx, details_3d_t_zyx = register_stack(
    stack_3d_t_zyx,
    registration_channel=0,  # channel used for NoRMCorre shift estimation
    registration_stack=0,  # reference time point/template
    method="normcorre",  # dispatches to ZenReg's CaImAn-compatible NoRMCorre port
    time_registration_mode="full_3d",  # full 3D+t local shift fields in ZYX
    zreg=True,  # estimate and apply Z shifts via full 3D NoRMCorre
    max_z_shifts=3,  # absolute correction shift limit in Z
    max_xy_shifts=(6, 6),  # absolute correction shift limits in YX
    phase_cross_correlation_upsample_factor=10,  # subpixel phase-correlation precision
    phase_cross_correlation_normalization=None,  # None or "phase"
    transform_order=3,  # 3 matches CaImAn's cubic remap; use 1 for gentler intensity interpolation
    nc_pw_rigid=True,  # patch-wise NoRMCorre local correction
    nc_strides=(6, 48, 48),  # patch-grid stride in ZYX
    nc_overlaps=(3, 24, 24),  # patch overlap in ZYX
    nc_max_deviation_rigid=(1.5, 3, 3),  # local deviation from rigid estimate in ZYX
    nc_n_iterations=1,  # template-refinement passes
    nc_correction_iterations=1,  # re-run NoRMCorre on the already corrected result
    nc_n_jobs=2,  # parallelize over time frames
    nc_block_size=4,  # Z slices warped per block
    nc_output_use_memmap=False,  # True writes registered output into OMIO/Zarr
    nc_output_memmap_folder=None,  # local scratch folder for output Zarr if enabled
    nc_output_memmap_name="3d_t_zyx_normcorre_registered_zarr",  # output Zarr store name
    verbose=True,
    return_details=True,
)
print_shift_comparison("3D+t global ZYX NoRMCorre rigid shifts", details_3d_t_zyx["time_shifts_zyx"], expected_3d_t_zyx)
show_before_after(stack_3d_t_zyx, registered_3d_t_zyx, title="3D+t global ZYX translation", channel=0)
save_stack(
    OUTPUT_DIR / "3d_t_global_zyx_normcorre_registered.ome.tif",
    registered_3d_t_zyx,
    metadata=metadata_3d_t_zyx,
    registration_details=details_3d_t_zyx,
)

maybe_open_in_napari(stack_3d_t_zyx, metadata_3d_t_zyx, fname="3D+t global ZYX NoRMCorre")
maybe_open_in_napari(registered_3d_t_zyx, metadata_3d_t_zyx, fname="3D+t global ZYX NoRMCorre")
# %% 6) 3D+t: TRANSLATION PLUS ROTATION AROUND Z
stack_3d_t_trans_rot_z, metadata_3d_t_trans_rot_z = load_stack(
    STACK_3D_T_TRANS_ROT_Z_PATH,
    return_metadata=True,
    verbose=False,
)
expected_3d_t_trans_rot_z = load_expected_time_registration_shifts(
    GT_3D_T_TRANS_ROT_Z_PATH,
    registration_stack=0,
    axes="zyx",
)
rigid_gt_3d_t_trans_rot_z = _load_csv(GT_3D_T_TRANS_ROT_Z_PATH)
print(f"3D+t translation+Z-rotation stack shape: {stack_3d_t_trans_rot_z.shape} (TZCYX)")
print("Applied rotations [z, y, x] deg, first rows:")
print(
    np.column_stack(
        [
            rigid_gt_3d_t_trans_rot_z["applied_rotation_z_deg"],
            rigid_gt_3d_t_trans_rot_z["applied_rotation_y_deg"],
            rigid_gt_3d_t_trans_rot_z["applied_rotation_x_deg"],
        ]
    )[:5]
)

plot_normcorre_patch_overlay(
    stack_3d_t_trans_rot_z,
    metadata_3d_t_trans_rot_z,
    registration_channel=0,
    registration_stack=0,
    nc_strides=(6, 48, 48),
    nc_overlaps=(3, 24, 24),
    projection_method="max",
    projection_range=(1, 10),
    output_dir=OUTPUT_DIR,
)

registered_3d_t_trans_rot_z, details_3d_t_trans_rot_z = register_stack(
    stack_3d_t_trans_rot_z,
    registration_channel=0,  # channel used for NoRMCorre shift estimation
    registration_stack=0,  # reference time point/template
    method="normcorre",  # dispatches to ZenReg's CaImAn-compatible NoRMCorre port
    time_registration_mode="full_3d",  # full 3D+t local shift fields in ZYX
    zreg=True,  # estimate and apply Z shifts via full 3D NoRMCorre
    max_z_shifts=3,  # absolute correction shift limit in Z
    max_xy_shifts=(6, 6),  # absolute correction shift limits in YX
    phase_cross_correlation_upsample_factor=10,  # subpixel phase-correlation precision
    phase_cross_correlation_normalization=None,  # None or "phase"
    transform_order=3,  # 3 matches CaImAn's cubic remap; use 1 for gentler intensity interpolation
    nc_pw_rigid=True,  # approximate small rotations via local translations
    nc_strides=(6, 48, 48),  # patch-grid stride in ZYX
    nc_overlaps=(3, 24, 24),  # patch overlap in ZYX
    nc_max_deviation_rigid=(1.5, 4, 4),  # local deviation from rigid estimate in ZYX
    nc_n_iterations=1,  # template-refinement passes
    nc_correction_iterations=1,  # re-run NoRMCorre on the already corrected result
    nc_n_jobs=2,  # parallelize over time frames
    nc_block_size=4,  # Z slices warped per block
    nc_output_use_memmap=False,  # True writes registered output into OMIO/Zarr
    nc_output_memmap_folder=None,  # local scratch folder for output Zarr if enabled
    nc_output_memmap_name="3d_t_trans_rot_z_normcorre_registered_zarr",  # output Zarr store name
    verbose=True,
    return_details=True,
)
print_shift_comparison(
    "3D+t Z-rotation rigid shifts vs translation component GT",
    details_3d_t_trans_rot_z["time_shifts_zyx"],
    expected_3d_t_trans_rot_z,
)
show_before_after(stack_3d_t_trans_rot_z, registered_3d_t_trans_rot_z, title="3D+t translation plus Z rotation", channel=0)
save_stack(
    OUTPUT_DIR / "3d_t_translation_rotation_z_normcorre_registered.ome.tif",
    registered_3d_t_trans_rot_z,
    metadata=metadata_3d_t_trans_rot_z,
    registration_details=details_3d_t_trans_rot_z,
)

maybe_open_in_napari(stack_3d_t_trans_rot_z, metadata_3d_t_trans_rot_z, fname="3D+t Z-rotation")
maybe_open_in_napari(registered_3d_t_trans_rot_z, metadata_3d_t_trans_rot_z, fname="3D+t Z-rotation NoRMCorre")
# %% 7) 3D+t: TRANSLATION PLUS ROTATION AROUND X
stack_3d_t_trans_rot_x, metadata_3d_t_trans_rot_x = load_stack(
    STACK_3D_T_TRANS_ROT_X_PATH,
    return_metadata=True,
    verbose=False,
)
expected_3d_t_trans_rot_x = load_expected_time_registration_shifts(
    GT_3D_T_TRANS_ROT_X_PATH,
    registration_stack=0,
    axes="zyx",
)
rigid_gt_3d_t_trans_rot_x = _load_csv(GT_3D_T_TRANS_ROT_X_PATH)
print(f"3D+t translation+X-rotation stack shape: {stack_3d_t_trans_rot_x.shape} (TZCYX)")
print("Applied rotations [z, y, x] deg, first rows:")
print(
    np.column_stack(
        [
            rigid_gt_3d_t_trans_rot_x["applied_rotation_z_deg"],
            rigid_gt_3d_t_trans_rot_x["applied_rotation_y_deg"],
            rigid_gt_3d_t_trans_rot_x["applied_rotation_x_deg"],
        ]
    )[:5]
)

plot_normcorre_patch_overlay(
    stack_3d_t_trans_rot_x,
    metadata_3d_t_trans_rot_x,
    registration_channel=0,
    registration_stack=0,
    nc_strides=(6, 48, 48),
    nc_overlaps=(3, 24, 24),
    projection_method="max",
    projection_range=(1, 10),
    output_dir=OUTPUT_DIR,
)

registered_3d_t_trans_rot_x, details_3d_t_trans_rot_x = register_stack(
    stack_3d_t_trans_rot_x,
    registration_channel=0,  # channel used for NoRMCorre shift estimation
    registration_stack=0,  # reference time point/template
    method="normcorre",  # dispatches to ZenReg's CaImAn-compatible NoRMCorre port
    time_registration_mode="full_3d",  # full 3D+t local shift fields in ZYX
    zreg=True,  # estimate and apply Z shifts via full 3D NoRMCorre
    max_z_shifts=3,  # absolute correction shift limit in Z
    max_xy_shifts=(6, 6),  # absolute correction shift limits in YX
    phase_cross_correlation_upsample_factor=10,  # subpixel phase-correlation precision
    phase_cross_correlation_normalization=None,  # None or "phase"
    transform_order=3,  # 3 matches CaImAn's cubic remap; use 1 for gentler intensity interpolation
    nc_pw_rigid=True,  # approximate small rotations via local translations
    nc_strides=(6, 48, 48),  # patch-grid stride in ZYX
    nc_overlaps=(3, 24, 24),  # patch overlap in ZYX
    nc_max_deviation_rigid=(1.5, 4, 4),  # local deviation from rigid estimate in ZYX
    nc_n_iterations=1,  # template-refinement passes
    nc_correction_iterations=1,  # re-run NoRMCorre on the already corrected result
    nc_n_jobs=2,  # parallelize over time frames
    nc_block_size=4,  # Z slices warped per block
    nc_output_use_memmap=False,  # True writes registered output into OMIO/Zarr
    nc_output_memmap_folder=None,  # local scratch folder for output Zarr if enabled
    nc_output_memmap_name="3d_t_trans_rot_x_normcorre_registered_zarr",  # output Zarr store name
    verbose=True,
    return_details=True,
)
print_shift_comparison(
    "3D+t X-rotation rigid shifts vs translation component GT",
    details_3d_t_trans_rot_x["time_shifts_zyx"],
    expected_3d_t_trans_rot_x,
)
show_before_after(stack_3d_t_trans_rot_x, registered_3d_t_trans_rot_x, title="3D+t translation plus X rotation", channel=0)
save_stack(
    OUTPUT_DIR / "3d_t_translation_rotation_x_normcorre_registered.ome.tif",
    registered_3d_t_trans_rot_x,
    metadata=metadata_3d_t_trans_rot_x,
    registration_details=details_3d_t_trans_rot_x,
)
maybe_open_in_napari(registered_3d_t_trans_rot_x, metadata_3d_t_trans_rot_x, fname="3D+t X-rotation NoRMCorre")

# %% 8) 3D+t: ALL-AXIS ROTATION AROUND STACK CENTER
stack_3d_t_trans_rot_all_center, metadata_3d_t_trans_rot_all_center = load_stack(
    STACK_3D_T_TRANS_ROT_ALL_CENTER_PATH,
    return_metadata=True,
    verbose=False,
)
expected_3d_t_trans_rot_all_center = load_expected_time_registration_shifts(
    GT_3D_T_TRANS_ROT_ALL_CENTER_PATH,
    registration_stack=0,
    axes="zyx",
)
rigid_gt_3d_t_trans_rot_all_center = _load_csv(GT_3D_T_TRANS_ROT_ALL_CENTER_PATH)
print(f"3D+t all-axis center-rotation stack shape: {stack_3d_t_trans_rot_all_center.shape} (TZCYX)")
print("Rotation centers [z, y, x], first rows:")
print(
    np.column_stack(
        [
            rigid_gt_3d_t_trans_rot_all_center["rotation_center_z"],
            rigid_gt_3d_t_trans_rot_all_center["rotation_center_y"],
            rigid_gt_3d_t_trans_rot_all_center["rotation_center_x"],
        ]
    )[:5]
)

plot_normcorre_patch_overlay(
    stack_3d_t_trans_rot_all_center,
    metadata_3d_t_trans_rot_all_center,
    registration_channel=0,
    registration_stack=0,
    nc_strides=(6, 48, 48),
    nc_overlaps=(3, 24, 24),
    projection_method="max",
    projection_range=(1, 10),
    output_dir=OUTPUT_DIR,
)

registered_3d_t_trans_rot_all_center, details_3d_t_trans_rot_all_center = register_stack(
    stack_3d_t_trans_rot_all_center,
    registration_channel=0,  # channel used for NoRMCorre shift estimation
    registration_stack=0,  # reference time point/template
    method="normcorre",  # dispatches to ZenReg's CaImAn-compatible NoRMCorre port
    time_registration_mode="full_3d",  # full 3D+t local shift fields in ZYX
    zreg=True,  # estimate and apply Z shifts via full 3D NoRMCorre
    max_z_shifts=3,  # absolute correction shift limit in Z
    max_xy_shifts=(6, 6),  # absolute correction shift limits in YX
    phase_cross_correlation_upsample_factor=10,  # subpixel phase-correlation precision
    phase_cross_correlation_normalization=None,  # None or "phase"
    transform_order=3,  # 3 matches CaImAn's cubic remap; use 1 for gentler intensity interpolation
    nc_pw_rigid=True,  # approximate small rotations via local translations
    nc_strides=(6, 48, 48),  # patch-grid stride in ZYX
    nc_overlaps=(3, 24, 24),  # patch overlap in ZYX
    nc_max_deviation_rigid=(1.5, 4, 4),  # local deviation from rigid estimate in ZYX
    nc_n_iterations=1,  # template-refinement passes
    nc_correction_iterations=1,  # re-run NoRMCorre on the already corrected result
    nc_n_jobs=2,  # parallelize over time frames
    nc_block_size=4,  # Z slices warped per block
    nc_output_use_memmap=False,  # True writes registered output into OMIO/Zarr
    nc_output_memmap_folder=None,  # local scratch folder for output Zarr if enabled
    nc_output_memmap_name="3d_t_trans_rot_all_center_normcorre_registered_zarr",  # output Zarr store name
    verbose=True,
    return_details=True,
)
print_shift_comparison(
    "3D+t all-axis center rigid shifts vs translation component GT",
    details_3d_t_trans_rot_all_center["time_shifts_zyx"],
    expected_3d_t_trans_rot_all_center,
)
show_before_after(
    stack_3d_t_trans_rot_all_center,
    registered_3d_t_trans_rot_all_center,
    title="3D+t all-axis rotation around stack center",
    channel=0,
)
save_stack(
    OUTPUT_DIR / "3d_t_translation_rotation_all_center_normcorre_registered.ome.tif",
    registered_3d_t_trans_rot_all_center,
    metadata=metadata_3d_t_trans_rot_all_center,
    registration_details=details_3d_t_trans_rot_all_center,
)
maybe_open_in_napari(
    registered_3d_t_trans_rot_all_center,
    metadata_3d_t_trans_rot_all_center,
    fname="3D+t all-axis center NoRMCorre",
)

# %% 9) 3D+t: ALL-AXIS ROTATION AROUND OFF-CENTER POINT
stack_3d_t_trans_rot_all_offcenter, metadata_3d_t_trans_rot_all_offcenter = load_stack(
    STACK_3D_T_TRANS_ROT_ALL_OFFCENTER_PATH,
    return_metadata=True,
    verbose=False,
)
expected_3d_t_trans_rot_all_offcenter = load_expected_time_registration_shifts(
    GT_3D_T_TRANS_ROT_ALL_OFFCENTER_PATH,
    registration_stack=0,
    axes="zyx",
)
rigid_gt_3d_t_trans_rot_all_offcenter = _load_csv(GT_3D_T_TRANS_ROT_ALL_OFFCENTER_PATH)
print(f"3D+t all-axis off-center-rotation stack shape: {stack_3d_t_trans_rot_all_offcenter.shape} (TZCYX)")
print("Rotation centers [z, y, x], first rows:")
print(
    np.column_stack(
        [
            rigid_gt_3d_t_trans_rot_all_offcenter["rotation_center_z"],
            rigid_gt_3d_t_trans_rot_all_offcenter["rotation_center_y"],
            rigid_gt_3d_t_trans_rot_all_offcenter["rotation_center_x"],
        ]
    )[:5]
)

plot_normcorre_patch_overlay(
    stack_3d_t_trans_rot_all_offcenter,
    metadata_3d_t_trans_rot_all_offcenter,
    registration_channel=0,
    registration_stack=0,
    nc_strides=(6, 48, 48),
    nc_overlaps=(3, 24, 24),
    projection_method="max",
    projection_range=(1, 10),
    output_dir=OUTPUT_DIR,
)

registered_3d_t_trans_rot_all_offcenter, details_3d_t_trans_rot_all_offcenter = register_stack(
    stack_3d_t_trans_rot_all_offcenter,
    registration_channel=0,  # channel used for NoRMCorre shift estimation
    registration_stack=0,  # reference time point/template
    method="normcorre",  # dispatches to ZenReg's CaImAn-compatible NoRMCorre port
    time_registration_mode="full_3d",  # full 3D+t local shift fields in ZYX
    zreg=True,  # estimate and apply Z shifts via full 3D NoRMCorre
    max_z_shifts=4,  # absolute correction shift limit in Z
    max_xy_shifts=(8, 8),  # outside/offset centers can induce larger apparent shifts
    phase_cross_correlation_upsample_factor=10,  # subpixel phase-correlation precision
    phase_cross_correlation_normalization=None,  # None or "phase"
    transform_order=3,  # 3 matches CaImAn's cubic remap; use 1 for gentler intensity interpolation
    nc_pw_rigid=True,  # approximate small rotations via local translations
    nc_strides=(6, 48, 48),  # patch-grid stride in ZYX
    nc_overlaps=(3, 24, 24),  # patch overlap in ZYX
    nc_max_deviation_rigid=(2, 5, 5),  # local deviation from rigid estimate in ZYX
    nc_n_iterations=1,  # template-refinement passes
    nc_correction_iterations=1,  # re-run NoRMCorre on the already corrected result
    nc_n_jobs=2,  # parallelize over time frames
    nc_block_size=4,  # Z slices warped per block
    nc_output_use_memmap=False,  # True writes registered output into OMIO/Zarr
    nc_output_memmap_folder=None,  # local scratch folder for output Zarr if enabled
    nc_output_memmap_name="3d_t_trans_rot_all_offcenter_normcorre_registered_zarr",  # output Zarr store name
    verbose=True,
    return_details=True,
)
print_shift_comparison(
    "3D+t all-axis off-center rigid shifts vs translation component GT",
    details_3d_t_trans_rot_all_offcenter["time_shifts_zyx"],
    expected_3d_t_trans_rot_all_offcenter,
)
show_before_after(
    stack_3d_t_trans_rot_all_offcenter,
    registered_3d_t_trans_rot_all_offcenter,
    title="3D+t all-axis rotation around off-center point",
    channel=0,
)
save_stack(
    OUTPUT_DIR / "3d_t_translation_rotation_all_offcenter_normcorre_registered.ome.tif",
    registered_3d_t_trans_rot_all_offcenter,
    metadata=metadata_3d_t_trans_rot_all_offcenter,
    registration_details=details_3d_t_trans_rot_all_offcenter,
)
maybe_open_in_napari(
    registered_3d_t_trans_rot_all_offcenter,
    metadata_3d_t_trans_rot_all_offcenter,
    fname="3D+t all-axis off-center NoRMCorre",
)

# %% 10) 3D+t: ALL-AXIS ROTATION AROUND OUTSIDE POINT
stack_3d_t_trans_rot_all_outside, metadata_3d_t_trans_rot_all_outside = load_stack(
    STACK_3D_T_TRANS_ROT_ALL_OUTSIDE_PATH,
    return_metadata=True,
    verbose=False,
)
expected_3d_t_trans_rot_all_outside = load_expected_time_registration_shifts(
    GT_3D_T_TRANS_ROT_ALL_OUTSIDE_PATH,
    registration_stack=0,
    axes="zyx",
)
rigid_gt_3d_t_trans_rot_all_outside = _load_csv(GT_3D_T_TRANS_ROT_ALL_OUTSIDE_PATH)
print(f"3D+t all-axis outside-rotation stack shape: {stack_3d_t_trans_rot_all_outside.shape} (TZCYX)")
print("Rotation centers [z, y, x], first rows:")
print(
    np.column_stack(
        [
            rigid_gt_3d_t_trans_rot_all_outside["rotation_center_z"],
            rigid_gt_3d_t_trans_rot_all_outside["rotation_center_y"],
            rigid_gt_3d_t_trans_rot_all_outside["rotation_center_x"],
        ]
    )[:5]
)

plot_normcorre_patch_overlay(
    stack_3d_t_trans_rot_all_outside,
    metadata_3d_t_trans_rot_all_outside,
    registration_channel=0,
    registration_stack=0,
    nc_strides=(6, 48, 48),
    nc_overlaps=(3, 24, 24),
    projection_method="max",
    projection_range=(1, 10),
    output_dir=OUTPUT_DIR,
)

registered_3d_t_trans_rot_all_outside, details_3d_t_trans_rot_all_outside = register_stack(
    stack_3d_t_trans_rot_all_outside,
    registration_channel=0,  # channel used for NoRMCorre shift estimation
    registration_stack=0,  # reference time point/template
    method="normcorre",  # dispatches to ZenReg's CaImAn-compatible NoRMCorre port
    time_registration_mode="full_3d",  # full 3D+t local shift fields in ZYX
    zreg=True,  # estimate and apply Z shifts via full 3D NoRMCorre
    max_z_shifts=5,  # absolute correction shift limit in Z
    max_xy_shifts=(10, 10),  # outside centers can induce larger apparent shifts
    phase_cross_correlation_upsample_factor=10,  # subpixel phase-correlation precision
    phase_cross_correlation_normalization=None,  # None or "phase"
    transform_order=3,  # 3 matches CaImAn's cubic remap; use 1 for gentler intensity interpolation
    nc_pw_rigid=True,  # approximate small rotations via local translations
    nc_strides=(6, 48, 48),  # patch-grid stride in ZYX
    nc_overlaps=(3, 24, 24),  # patch overlap in ZYX
    nc_max_deviation_rigid=(2, 6, 6),  # local deviation from rigid estimate in ZYX
    nc_n_iterations=1,  # template-refinement passes
    nc_correction_iterations=1,  # re-run NoRMCorre on the already corrected result
    nc_n_jobs=2,  # parallelize over time frames
    nc_block_size=4,  # Z slices warped per block
    nc_output_use_memmap=False,  # True writes registered output into OMIO/Zarr
    nc_output_memmap_folder=None,  # local scratch folder for output Zarr if enabled
    nc_output_memmap_name="3d_t_trans_rot_all_outside_normcorre_registered_zarr",  # output Zarr store name
    verbose=True,
    return_details=True,
)
print_shift_comparison(
    "3D+t all-axis outside rigid shifts vs translation component GT",
    details_3d_t_trans_rot_all_outside["time_shifts_zyx"],
    expected_3d_t_trans_rot_all_outside,
)
show_before_after(
    stack_3d_t_trans_rot_all_outside,
    registered_3d_t_trans_rot_all_outside,
    title="3D+t all-axis rotation around outside point",
    channel=0,
)
save_stack(
    OUTPUT_DIR / "3d_t_translation_rotation_all_outside_normcorre_registered.ome.tif",
    registered_3d_t_trans_rot_all_outside,
    metadata=metadata_3d_t_trans_rot_all_outside,
    registration_details=details_3d_t_trans_rot_all_outside,
)
maybe_open_in_napari(
    registered_3d_t_trans_rot_all_outside,
    metadata_3d_t_trans_rot_all_outside,
    fname="3D+t all-axis outside NoRMCorre",
)

# %% END
