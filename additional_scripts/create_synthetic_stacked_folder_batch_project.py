"""
Create a small synthetic stacked-folder batch project for ZenReg.

The generated tree mimics acquisition layouts such as:

    synthetic_stacked_folder_batch_project/
    └─ ID000001/
       └─ FOV1_pre/
          ├─ OV_1/
          │  └─ image_001.ome.tif
          ├─ OV_2/
          │  └─ image_001.ome.tif
          └─ OV_3/
             └─ image_001.ome.tif

Each OV folder contains a small 3D image stack with the same filename but a
different integer Z/Y/X shift. This deliberately tests that stacked-folder batch
processing does not reuse the same disk cache for repeated filenames in
different folders and can be used to exercise 3D+t registration.

Author: Fabrizio Musacchio
Date: September 2026
"""
# %% IMPORTS
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from zenreg import create_stack_metadata, save_stack
# %% DEFAULTS
DEFAULT_PROJECT_ROOT = PROJECT_ROOT / "example_data" / "synthetic_stacked_folder_batch_project"

DEFAULT_OV_SHIFTS_ZYX = (
    (0, 0, 0),
    (1, 4, -3),
    (-2, -5, 5),
    (2, 2, 7))
# %% HELPERS
def _spot_volume(shape_zyx: tuple[int, int, int], *, seed: int) -> np.ndarray:
    """Return one normalized synthetic microscopy-like 3D volume."""

    rng = np.random.default_rng(seed)
    z_size, y_size, x_size = shape_zyx
    zz, yy, xx = np.mgrid[:z_size, :y_size, :x_size]
    volume = np.zeros(shape_zyx, dtype=np.float32)

    for _ in range(55):
        cz = rng.uniform(3, z_size - 3)
        cy = rng.uniform(8, y_size - 8)
        cx = rng.uniform(8, x_size - 8)
        sigma_z = rng.uniform(0.9, 2.2)
        sigma_yx = rng.uniform(1.6, 4.5)
        amplitude = rng.uniform(0.35, 1.0)
        volume += amplitude * np.exp(
            -(
                ((zz - cz) ** 2) / (2 * sigma_z ** 2)
                + ((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma_yx ** 2)
            )
        )

    volume += 0.05 * rng.random(shape_zyx, dtype=np.float32)
    volume -= np.min(volume)
    volume /= max(float(np.max(volume)), np.finfo(np.float32).eps)
    return volume.astype(np.float32)

def _shift_3d_with_zeros(volume: np.ndarray, shift_z: int, shift_y: int, shift_x: int) -> np.ndarray:
    """Shift one 3D volume by integer pixels and fill uncovered borders with zeros."""

    out = np.zeros_like(volume)
    src_z0 = max(0, -shift_z)
    src_z1 = volume.shape[0] - max(0, shift_z)
    dst_z0 = max(0, shift_z)
    dst_z1 = dst_z0 + max(0, src_z1 - src_z0)
    src_y0 = max(0, -shift_y)
    src_y1 = volume.shape[1] - max(0, shift_y)
    dst_y0 = max(0, shift_y)
    dst_y1 = dst_y0 + max(0, src_y1 - src_y0)
    src_x0 = max(0, -shift_x)
    src_x1 = volume.shape[2] - max(0, shift_x)
    dst_x0 = max(0, shift_x)
    dst_x1 = dst_x0 + max(0, src_x1 - src_x0)
    if src_z1 <= src_z0 or src_y1 <= src_y0 or src_x1 <= src_x0:
        return out
    out[dst_z0:dst_z1, dst_y0:dst_y1, dst_x0:dst_x1] = volume[
        src_z0:src_z1,
        src_y0:src_y1,
        src_x0:src_x1]
    return out

def _make_ov_stack(
    base_channels: np.ndarray,
    *,
    shift_zyx: tuple[int, int, int],
    noise_sigma: float,
    seed: int,
) -> np.ndarray:
    """Return one single-timepoint TZCYX stack for one OV folder."""

    rng = np.random.default_rng(seed)
    shift_z, shift_y, shift_x = shift_zyx
    shifted_channels = [
        _shift_3d_with_zeros(channel, shift_z, shift_y, shift_x)
        for channel in base_channels
    ]
    stack = np.stack(shifted_channels, axis=1)
    stack += rng.normal(0.0, noise_sigma, size=stack.shape).astype(np.float32)
    stack = np.clip(stack, 0.0, 1.0)
    return stack[np.newaxis, :, :, :, :].astype(np.float32)

def write_synthetic_stacked_folder_batch_project(
    project_root: str | Path = DEFAULT_PROJECT_ROOT,
    *,
    subject_ids: tuple[str, ...] = ("ID000001",),
    fov_names: tuple[str, ...] = ("FOV1_pre",),
    ov_shifts_zyx: tuple[tuple[int, int, int], ...] = DEFAULT_OV_SHIFTS_ZYX,
    shape_zyx: tuple[int, int, int] = (16, 128, 128),
    channel_count: int = 2,
    filename: str = "image_001.ome.tif",
    noise_sigma: float = 0.015,
    overwrite: bool = True,
) -> Path:
    """Create or replace a synthetic stacked-folder ZenReg batch project."""

    project_root = Path(project_root)
    if overwrite and project_root.exists():
        shutil.rmtree(project_root)
    project_root.mkdir(parents=True, exist_ok=True)

    base = _spot_volume(shape_zyx, seed=90210)
    base_channels = [base]
    for channel_index in range(1, channel_count):
        base_channels.append(np.roll(base, shift=3 * channel_index, axis=1) * (1.0 - 0.08 * channel_index))
    base_channels = np.stack(base_channels, axis=0).astype(np.float32)

    rows = ["subject_id,fov,ov_folder,filename,applied_shift_z,applied_shift_y,applied_shift_x"]
    for subject_index, subject_id in enumerate(subject_ids):
        for fov_index, fov_name in enumerate(fov_names):
            fov_dir = project_root / subject_id / fov_name
            for ov_index, shift_zyx in enumerate(ov_shifts_zyx, start=1):
                ov_dir = fov_dir / f"OV_{ov_index}"
                ov_dir.mkdir(parents=True, exist_ok=True)
                stack = _make_ov_stack(
                    base_channels,
                    shift_zyx=shift_zyx,
                    noise_sigma=noise_sigma,
                    seed=subject_index * 1000 + fov_index * 100 + ov_index)
                metadata = create_stack_metadata(
                    stack,
                    annotations={
                        "zenreg_synthetic_stacked_folder": True,
                        "applied_shift_z": int(shift_zyx[0]),
                        "applied_shift_y": int(shift_zyx[1]),
                        "applied_shift_x": int(shift_zyx[2]),
                        "ov_folder": ov_dir.name},
                    verbose=False)
                written = save_stack(
                    ov_dir / filename,
                    stack,
                    metadata=metadata,
                    overwrite=True,
                    verbose=False)
                rows.append(
                    f"{subject_id},{fov_name},{ov_dir.name},{written.name},{shift_zyx[0]},{shift_zyx[1]},{shift_zyx[2]}")

    (project_root / "synthetic_stacked_folder_ground_truth.csv").write_text(
        "\n".join(rows) + "\n",
        encoding="utf-8")
    (project_root / "README.md").write_text(
        "\n".join(
            [
                "# Synthetic stacked-folder ZenReg batch project",
                "",
                "This folder is generated by `additional_scripts/create_synthetic_stacked_folder_batch_project.py`.",
                "It mimics an `ID/FOV/OV_*` acquisition layout in which repeated OV folders",
                "belong to one logical time series and should be merged along `T` before registration.",
                "",
                "Each OV folder contains the same filename but a different synthetic Z/Y/X shift.",
                "This makes the project useful for testing cache-safe stacked-folder batch processing and 3D+t registration.",
                "",
            ]
        ),
        encoding="utf-8")
    print(f"Wrote synthetic stacked-folder batch project: {project_root}")
    return project_root
# %% SCRIPT ENTRY POINT
if __name__ == "__main__":
    write_synthetic_stacked_folder_batch_project()
