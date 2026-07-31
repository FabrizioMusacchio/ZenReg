"""
Small tutorial and documentation helpers for ZenReg examples.

These functions keep the user scripts compact and are intentionally oriented
toward readable examples: loading synthetic GT tables, printing compact
detected-vs-GT summaries, quick residual plots, and optional Napari display.

Author: Fabrizio Musacchio
Date: July 2026
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .filters import z_project


def load_csv_table(path: str | Path) -> np.ndarray:
    """Load a CSV file as a structured NumPy table."""

    return np.genfromtxt(Path(path), delimiter=",", names=True)


def load_expected_time_registration_shifts(
    path: str | Path,
    *,
    registration_stack: int = 0,
    axes: str = "yx",
) -> np.ndarray:
    """Load expected correction shifts from a synthetic GT time-shift table."""

    table = load_csv_table(path)
    columns = [f"expected_registration_shift_{axis}_ref_t{registration_stack}" for axis in axes]
    return np.column_stack([table[column] for column in columns]).astype(np.float32)


def load_expected_time_registration_rotations(
    path: str | Path,
    *,
    registration_stack: int = 0,
) -> np.ndarray:
    """Load expected in-plane rotation corrections from a synthetic GT table."""

    table = load_csv_table(path)
    column = f"expected_registration_rotation_deg_ref_t{registration_stack}"
    return np.asarray(table[column], dtype=np.float32)


def load_expected_rigid_z_rotation(
    path: str | Path,
    *,
    registration_stack: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Load expected ZYX shifts and Z-axis rotation corrections from a 3D rigid GT table."""

    table = load_csv_table(path)
    shift_columns = [
        f"expected_registration_shift_{axis}_ref_t{registration_stack}"
        for axis in ("z", "y", "x")
    ]
    rotation_column = f"expected_registration_rotation_z_deg_ref_t{registration_stack}"
    shifts_zyx = np.column_stack([table[column] for column in shift_columns]).astype(np.float32)
    rotations_z_deg = np.asarray(table[rotation_column], dtype=np.float32)
    return shifts_zyx, rotations_z_deg


def load_expected_rigid_corrections(
    path: str | Path,
    *,
    registration_stack: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Load expected correction translations and rotations from a 3D rigid GT table."""

    table = load_csv_table(path)
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


def load_expected_slice_registration_shifts(path: str | Path) -> np.ndarray:
    """Load expected intra-stack slice correction shifts from a synthetic GT table."""

    table = load_csv_table(path)
    time_count = int(np.max(table["t"])) + 1
    z_count = int(np.max(table["z"])) + 1
    shifts = np.zeros((time_count, z_count, 2), dtype=np.float32)
    if "expected_registration_shift_y" in table.dtype.names:
        y_column = "expected_registration_shift_y"
        x_column = "expected_registration_shift_x"
    else:
        y_column = "expected_local_z_correction_shift_y"
        x_column = "expected_local_z_correction_shift_x"
    for row in table:
        shifts[int(row["t"]), int(row["z"]), :] = (
            row[y_column],
            row[x_column],
        )
    return shifts


def print_shift_comparison(
    name: str,
    estimated_shifts: np.ndarray,
    expected_shifts: np.ndarray,
) -> None:
    """Print a compact detected-vs-GT shift comparison."""

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


def print_rigid_comparison(
    title: str,
    details: dict,
    expected_shifts_zyx: np.ndarray,
    expected_rotations_zyx_deg: np.ndarray,
) -> None:
    """Print detected-vs-GT full rigid registration summaries."""

    detected_shifts = np.asarray(details["time_shifts_zyx"], dtype=np.float32)
    detected_rotations = np.asarray(details["rotation_shifts_zyx_deg"], dtype=np.float32)
    shift_delta = detected_shifts - expected_shifts_zyx
    rotation_delta = detected_rotations - expected_rotations_zyx_deg

    print(title)
    if "rigid_3d_backend" in details:
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


def print_local_patch_summary(name: str, details: dict, *, t: int = 1) -> None:
    """Print patch-shift statistics for local NoRMCorre examples."""

    patch_shifts = np.asarray(details["patch_shifts"][int(t)], dtype=np.float32).reshape(-1, 2)
    print(f"{name}:")
    if "time_shifts_yx" in details:
        print(f"  rigid shift yx: {details['time_shifts_yx'][int(t)]}")
    print(f"  patch mean yx:  {patch_shifts.mean(axis=0)}")
    print(f"  patch std yx:   {patch_shifts.std(axis=0)}")
    print(f"  patch min yx:   {patch_shifts.min(axis=0)}")
    print(f"  patch max yx:   {patch_shifts.max(axis=0)}")


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


def _project_one_time(stack, *, t: int, channel: int, projection_method: str) -> np.ndarray:
    """Project one time point/channel to YX without reading all time points."""

    return z_project(
        stack[int(t) : int(t) + 1, :, int(channel) : int(channel) + 1, :, :],
        projection_method=projection_method,
    )[0, 0, 0]


def _slugify_for_filename(text: str) -> str:
    """Return a compact ASCII-ish filename slug for tutorial figures."""

    slug_chars = []
    previous_was_separator = False
    for character in str(text).strip().lower():
        if character.isalnum():
            slug_chars.append(character)
            previous_was_separator = False
        elif not previous_was_separator:
            slug_chars.append("_")
            previous_was_separator = True
    slug = "".join(slug_chars).strip("_")
    return slug or "zenreg_figure"


def show_timepoints(
    stack,
    *,
    title: str,
    channel: int = 0,
    projection_method: str = "max",
) -> None:
    """Show t=0, t=1, and their difference as Z projections."""

    if stack.shape[0] < 2:
        print(f"Skipping timepoint quick view for {title!r}: T < 2.")
        return

    projection_t0 = _project_one_time(stack, t=0, channel=channel, projection_method=projection_method)
    projection_t1 = _project_one_time(stack, t=1, channel=channel, projection_method=projection_method)
    projection_diff = projection_t1 - projection_t0

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(10, 3), constrained_layout=True)
    axes[0].imshow(projection_t0, cmap="gray")
    axes[0].set_title("t=0")
    axes[1].imshow(projection_t1, cmap="gray")
    axes[1].set_title("t=1")
    max_abs_diff = max(float(np.max(np.abs(projection_diff))), 1e-6)
    axes[2].imshow(projection_diff, cmap="bwr", vmin=-max_abs_diff, vmax=max_abs_diff)
    axes[2].set_title("t1 - t0")
    for ax in axes:
        ax.axis("off")
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

    z1 = min(int(z1), stack.shape[1] - 1)
    image_z0 = np.asarray(stack[0, int(z0), int(channel), :, :], dtype=np.float32)
    image_z1 = np.asarray(stack[0, z1, int(channel), :, :], dtype=np.float32)
    diff = image_z1 - image_z0

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(10, 3), constrained_layout=True)
    axes[0].imshow(image_z0, cmap="gray")
    axes[0].set_title(f"z={z0}")
    axes[1].imshow(image_z1, cmap="gray")
    axes[1].set_title(f"z={z1}")
    max_abs_diff = max(float(np.max(np.abs(diff))), 1e-6)
    axes[2].imshow(diff, cmap="bwr", vmin=-max_abs_diff, vmax=max_abs_diff)
    axes[2].set_title(f"z{z1} - z{z0}")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title)
    plt.show()


def show_before_after(
    raw_stack,
    registered_stack,
    *,
    title: str,
    channel: int = 0,
    moving_time: int = 1,
    reference_time: int = 0,
    projection_method: str = "max",
    save_path: str | Path | None = None,
    save_dir: str | Path | None = None,
    dpi: int = 200,
) -> None:
    """Show and optionally save raw/registered projection residuals.

    Parameters
    ----------
    raw_stack, registered_stack : array-like
        Canonical ``TZCYX`` stacks before and after registration.
    title : str
        Figure title. Used to generate a filename when ``save_dir`` is given.
    channel : int, optional
        Channel to project.
    moving_time, reference_time : int, optional
        Time points to compare.
    projection_method : str, optional
        Z-projection method used for the quick-look images.
    save_path : str, pathlib.Path, or None, optional
        Explicit output path for the figure. If provided, this takes precedence
        over ``save_dir``.
    save_dir : str, pathlib.Path, or None, optional
        Output directory. When provided without ``save_path``, ZenReg creates a
        deterministic PNG filename from the title and comparison settings.
    dpi : int, optional
        Figure resolution used for saving.
    """

    reference_time = int(np.clip(reference_time, 0, raw_stack.shape[0] - 1))
    moving_time = int(np.clip(moving_time, 0, raw_stack.shape[0] - 1))
    raw_t0 = _project_one_time(
        raw_stack,
        t=reference_time,
        channel=channel,
        projection_method=projection_method,
    )
    raw_ti = _project_one_time(
        raw_stack,
        t=moving_time,
        channel=channel,
        projection_method=projection_method,
    )
    reg_t0 = _project_one_time(
        registered_stack,
        t=reference_time,
        channel=channel,
        projection_method=projection_method,
    )
    reg_ti = _project_one_time(
        registered_stack,
        t=moving_time,
        channel=channel,
        projection_method=projection_method,
    )

    raw_diff = raw_ti - raw_t0
    reg_diff = reg_ti - reg_t0
    max_abs = max(float(np.max(np.abs(raw_diff))), float(np.max(np.abs(reg_diff))), 1e-6)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(10, 6), constrained_layout=True)
    for ax, image, label, cmap in [
        (axes[0, 0], raw_t0, f"raw t={reference_time}", "gray"),
        (axes[0, 1], raw_ti, f"raw t={moving_time}", "gray"),
        (axes[0, 2], raw_diff, f"raw t{moving_time} - t{reference_time}", "bwr"),
        (axes[1, 0], reg_t0, f"registered t={reference_time}", "gray"),
        (axes[1, 1], reg_ti, f"registered t={moving_time}", "gray"),
        (axes[1, 2], reg_diff, f"registered t{moving_time} - t{reference_time}", "bwr"),
    ]:
        if cmap == "bwr":
            ax.imshow(image, cmap=cmap, vmin=-max_abs, vmax=max_abs)
        else:
            ax.imshow(image, cmap=cmap)
        ax.set_title(label)
        ax.axis("off")
    fig.suptitle(title)
    if save_path is not None or save_dir is not None:
        if save_path is None:
            slug = _slugify_for_filename(title)
            save_path = (
                Path(save_dir)
                / f"{slug}_c{int(channel)}_t{reference_time}_t{moving_time}_{projection_method}.png"
            )
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=int(dpi), bbox_inches="tight")
    if plt.get_backend().lower() == "agg":
        plt.close(fig)
    else:
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

    show_residual_comparison_multi(
        raw,
        (registered_zenreg, registered_caiman),
        title=title,
        labels=labels,
        channel=channel,
        moving_time=moving_time,
        reference_time=reference_time,
        projection_method=projection_method,
    )


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
        projection_reference = _project_one_time(
            stack,
            t=reference_time,
            channel=channel,
            projection_method=projection_method,
        )
        projection_moving = _project_one_time(
            stack,
            t=moving_time,
            channel=channel,
            projection_method=projection_method,
        )
        residuals.append(projection_moving - projection_reference)

    vmax = max(float(np.max(np.abs(diff))) for diff in residuals)
    vmax = max(vmax, 1e-6)

    import matplotlib.pyplot as plt

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
        reference = _project_one_time(
            stack,
            t=reference_time,
            channel=channel,
            projection_method=projection_method,
        )
        residuals = []
        for t in range(stack.shape[0]):
            projection = _project_one_time(
                stack,
                t=t,
                channel=channel,
                projection_method=projection_method,
            )
            residuals.append(float(np.mean(np.abs(projection - reference))))
        residuals = np.asarray(residuals, dtype=np.float32)
        print(
            f"  {label}: mean={float(np.mean(residuals)):.6f}, "
            f"max={float(np.max(residuals)):.6f}"
        )


def open_in_napari(
    stack,
    metadata,
    *,
    fname: str,
    enabled: bool = True,
) -> None:
    """Open a stack in Napari when enabled by an example script."""

    if not enabled:
        return

    import omio as om

    om.open_in_napari(stack, metadata, fname=fname)
