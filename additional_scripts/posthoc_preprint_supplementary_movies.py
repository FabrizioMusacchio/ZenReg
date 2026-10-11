"""Create supplementary raw-vs-registered movies for ZenReg preprint figures.

Run after the corresponding preprint/example scripts have generated their raw
and registered outputs:

    conda run -n zenreg2 python additional_scripts/posthoc_preprint_supplementary_movies.py

The script writes short side-by-side movies into the matching figure folders.
MP4 output is used when ffmpeg is available; otherwise GIF files are written.
All movies use fixed raw/registered contrast limits per channel, magenta/green
two-channel overlays where applicable, frame/time overlays, and scale bars when
pixel-size metadata are available.
"""
# %% IMPORTS
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "zenreg-matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "zenreg-xdg-cache"))

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from zenreg import create_stack_metadata, load_stack, update_stack_metadata
from zenreg.synthetic import _apply_zyx_rigid_transform

# %% PATHS AND SETTINGS
REPO_ROOT = Path(__file__).resolve().parents[1]
PREPRINT_DIR = REPO_ROOT / "papers" / "preprint"
BENCHMARK_DIR = PREPRINT_DIR / "benchmark_outputs"
FIGURE_DIR = PREPRINT_DIR / "figures"

CREATE_REPRESENTATIVE_2D = True
CREATE_REAL_CALCIUM_2D = True
CREATE_SYNTHETIC_RIGID_3D = True
CREATE_REAL_3P_PSEUDOT = True
CREATE_REAL_3DTCH = True

FPS = 2
TIME_STRIDE = 1
MOVIE_MODE_3D = "max"  # "max", "mean", or "middle_slice"
INTENSITY_PERCENTILES = (1.0, 99.7)
SCALE_BAR_UM = 20.0
SCALE_BAR_FALLBACK_UM_PER_PX: float | None = None

CM_TO_INCH = 1.0 / 2.54
FONT_SIZE = 9
LEGEND_SIZE = 8
PANEL_DPI = 220

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": FONT_SIZE,
        "axes.titlesize": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "legend.fontsize": LEGEND_SIZE,
        "pdf.fonttype": 42,
        "ps.fonttype": 42})

# %% 3P PSEUDO-TIME SETTINGS, MATCHING run_preprint_real_3p_pseudot_example.py
THREEP_INPUT_PATH = REPO_ROOT / "example_data" / "3P_paper_Fuhrmann_Nebeling_Musacchio" / "Supplementary_Video_4.tif"
THREEP_OUTPUT_DIR = BENCHMARK_DIR / "real_3p_pseudot"
THREEP_FULL_VOLUME = False
THREEP_TIME_COUNT = 10
THREEP_Z_COUNT = 96
THREEP_YX_SIZE = 384
THREEP_TRANSLATION_MAX_Z = 2.0
THREEP_TRANSLATION_MAX_Y = 5.0
THREEP_TRANSLATION_MAX_X = 5.5
THREEP_RIGID_MAX_Z = 1.5
THREEP_RIGID_MAX_Y = 4.0
THREEP_RIGID_MAX_X = 4.5
THREEP_RIGID_MAX_ROTATION_Z_DEG = 3.0

# %% DATA STRUCTURES
@dataclass
class MovieSpec:
    name: str
    figure_subdir: str
    raw: np.ndarray
    registered: np.ndarray
    metadata: dict
    channels: tuple[int, ...] = (0,)
    projection_mode: str = "max"
    title: str = ""
    time_label: str = "t"
    scale_bar: bool = True

# %% GENERIC HELPERS
def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))

def metadata_get(metadata: dict, *keys: str):
    for key in keys:
        if key in metadata and metadata[key] is not None:
            return metadata[key]
    lower = {str(key).lower(): value for key, value in metadata.items()}
    for key in keys:
        value = lower.get(key.lower())
        if value is not None:
            return value
    return None

def pixel_size_x_um(metadata: dict) -> float | None:
    physical_xyz = metadata_get(metadata, "physicalsize_xyz", "PhysicalSizeXYZ", "physical_size_xyz")
    if physical_xyz is not None and len(physical_xyz) >= 1:
        try:
            return float(physical_xyz[0])
        except (TypeError, ValueError):
            pass
    for key in ("physicalsize_x", "PhysicalSizeX", "physical_size_x", "pixelsize_x", "PixelSizeX"):
        value = metadata_get(metadata, key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return SCALE_BAR_FALLBACK_UM_PER_PX

def centered_bounds(size: int, count: int) -> tuple[int, int]:
    count = min(int(count), int(size))
    start = max((int(size) - count) // 2, 0)
    return start, start + count

def crop_raw_to_analysis_stack(stack: np.ndarray, summary: dict) -> np.ndarray:
    crop = summary.get("crop_bounds") or {}
    z_start = int(crop.get("z_start", 0))
    z_stop = int(crop.get("z_stop", stack.shape[1]))
    y_start = int(crop.get("y_start", 0))
    y_stop = int(crop.get("y_stop", stack.shape[3]))
    x_start = int(crop.get("x_start", 0))
    x_stop = int(crop.get("x_stop", stack.shape[4]))
    return np.asarray(stack[:, z_start:z_stop, :, y_start:y_stop, x_start:x_stop])

def project_frame(frame_zcyx: np.ndarray, *, channel: int, mode: str) -> np.ndarray:
    volume = np.asarray(frame_zcyx[:, int(channel)], dtype=np.float32)
    if mode == "max":
        return np.nanmax(volume, axis=0)
    if mode == "mean":
        return np.nanmean(volume, axis=0)
    if mode == "middle_slice":
        return volume[volume.shape[0] // 2]
    raise ValueError(f"Unsupported projection mode {mode!r}.")

def contrast_limits(raw: np.ndarray, registered: np.ndarray, *, channels: tuple[int, ...], mode: str) -> dict[int, tuple[float, float]]:
    limits = {}
    for channel in channels:
        values = []
        for stack in (raw, registered):
            if int(channel) >= stack.shape[2]:
                continue
            for t in range(0, stack.shape[0], max(int(TIME_STRIDE), 1)):
                values.append(project_frame(stack[t], channel=int(channel), mode=mode).ravel())
        joined = np.concatenate(values) if values else np.asarray([0, 1], dtype=np.float32)
        lo, hi = np.nanpercentile(joined, INTENSITY_PERCENTILES)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = 0.0, max(float(np.nanmax(joined)), 1.0)
        limits[int(channel)] = (float(lo), float(hi))
    return limits

def normalize_channel(image: np.ndarray, limits: tuple[float, float]) -> np.ndarray:
    lo, hi = limits
    return np.clip((np.asarray(image, dtype=np.float32) - lo) / max(hi - lo, 1e-6), 0.0, 1.0)

def rgb_overlay(frame_zcyx: np.ndarray, *, channels: tuple[int, ...], mode: str, limits: dict[int, tuple[float, float]]) -> np.ndarray:
    rgb = np.zeros((*frame_zcyx.shape[-2:], 3), dtype=np.float32)
    if len(channels) == 1:
        channel = int(channels[0])
        gray = normalize_channel(project_frame(frame_zcyx, channel=channel, mode=mode), limits[channel])
        rgb[..., :] = gray[..., None]
        return rgb
    if 0 in channels and frame_zcyx.shape[1] > 0:
        ch0 = normalize_channel(project_frame(frame_zcyx, channel=0, mode=mode), limits[0])
        rgb[..., 0] = np.maximum(rgb[..., 0], ch0)
        rgb[..., 2] = np.maximum(rgb[..., 2], ch0)
    if 1 in channels and frame_zcyx.shape[1] > 1:
        ch1 = normalize_channel(project_frame(frame_zcyx, channel=1, mode=mode), limits[1])
        rgb[..., 1] = np.maximum(rgb[..., 1], ch1)
    return np.clip(rgb, 0.0, 1.0)

def add_scale_bar(ax, metadata: dict) -> None:
    px_um = pixel_size_x_um(metadata)
    if px_um is None or px_um <= 0:
        return
    bar_px = SCALE_BAR_UM / px_um
    scalebar = AnchoredSizeBar(
        ax.transData,
        bar_px,
        f"{SCALE_BAR_UM:g} um",
        loc="lower right",
        pad=0.25,
        borderpad=0.35,
        sep=2,
        frameon=False,
        color="white",
        size_vertical=max(bar_px * 0.025, 2.0),
        fontproperties={"size": LEGEND_SIZE})
    ax.add_artist(scalebar)

def add_panel_label(ax, text: str):
    return ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=FONT_SIZE,
        color="white",
        bbox={"facecolor": "black", "edgecolor": "none", "alpha": 0.55, "pad": 2.0})

def save_movie(spec: MovieSpec) -> Path:
    output_dir = FIGURE_DIR / spec.figure_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = np.asarray(spec.raw)
    registered = np.asarray(spec.registered)
    frame_indices = list(range(0, min(raw.shape[0], registered.shape[0]), max(int(TIME_STRIDE), 1)))
    if not frame_indices:
        raise ValueError(f"No movie frames available for {spec.name}.")

    limits = contrast_limits(raw, registered, channels=spec.channels, mode=spec.projection_mode)
    fig, axes = plt.subplots(1, 2, figsize=(16.0 * CM_TO_INCH, 6.5 * CM_TO_INCH))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.03, top=0.90, wspace=0.06)
    for ax in axes:
        ax.axis("off")
    raw_im = axes[0].imshow(rgb_overlay(raw[frame_indices[0]], channels=spec.channels, mode=spec.projection_mode, limits=limits), interpolation="nearest")
    reg_im = axes[1].imshow(rgb_overlay(registered[frame_indices[0]], channels=spec.channels, mode=spec.projection_mode, limits=limits), interpolation="nearest")
    if spec.scale_bar:
        add_scale_bar(axes[0], spec.metadata)
        add_scale_bar(axes[1], spec.metadata)

    channel_text = "ch0=magenta, ch1=green" if len(spec.channels) > 1 else f"ch{spec.channels[0]}=gray"
    title_text = spec.title or spec.name.replace("_", " ")
    fig.suptitle(title_text, fontsize=FONT_SIZE, y=0.985)
    raw_label = add_panel_label(axes[0], "")
    registered_label = add_panel_label(axes[1], "")

    def update(frame_number: int):
        t = frame_indices[frame_number]
        raw_im.set_data(rgb_overlay(raw[t], channels=spec.channels, mode=spec.projection_mode, limits=limits))
        reg_im.set_data(rgb_overlay(registered[t], channels=spec.channels, mode=spec.projection_mode, limits=limits))
        raw_label.set_text(f"raw {spec.time_label}={t}\n{channel_text}")
        registered_label.set_text(f"registered {spec.time_label}={t}\n{channel_text}")
        return raw_im, reg_im, raw_label, registered_label

    update(0)
    animation = FuncAnimation(fig, update, frames=len(frame_indices), interval=1000 / max(FPS, 1), blit=False)
    mp4_path = output_dir / f"{spec.name}.mp4"
    gif_path = output_dir / f"{spec.name}.gif"
    if FFMpegWriter.isAvailable():
        animation.save(mp4_path, writer=FFMpegWriter(fps=FPS, codec="libx264", bitrate=2400))
        output = mp4_path
    else:
        animation.save(gif_path, writer=PillowWriter(fps=FPS))
        output = gif_path
    plt.close(fig)
    return output

# %% DATASET LOADERS
def spec_representative_2d() -> MovieSpec | None:
    path = BENCHMARK_DIR / "representative_2d_phase.npz"
    if not path.exists():
        print(f"Skipping representative 2D+t movie; missing {path}")
        return None
    data = np.load(path)
    raw = np.asarray(data["raw"], dtype=np.float32)
    registered = np.asarray(data["registered"], dtype=np.float32)
    return MovieSpec(
        name="figure2_representative_2d_t_raw_vs_registered",
        figure_subdir="figure2",
        raw=raw,
        registered=registered,
        metadata={},
        channels=(0, 1) if raw.shape[2] > 1 else (0,),
        projection_mode="max",
        title="Representative 2D+t registration with known ground truth",
        scale_bar=False)

def spec_real_calcium() -> MovieSpec | None:
    input_path = REPO_ROOT / "example_data" / "CaImAn_example_Ca_file" / "Sue_2x_3000_40_-46.tif"
    registered_path = BENCHMARK_DIR / "real_calcium_2d_t" / "caiman_sue_2d_t_zenreg_registered.ome.tif"
    if not input_path.exists() or not registered_path.exists():
        print(f"Skipping real calcium movie; missing {input_path} or {registered_path}")
        return None
    raw, metadata = load_stack(input_path, return_metadata=True, verbose=False)
    if raw.shape[0] == 1 and raw.shape[1] == 1 and raw.shape[2] > 1:
        raw = np.swapaxes(raw, 2, 0)
        raw = np.ascontiguousarray(raw[:, :, 0:1, :, :])
    registered, _ = load_stack(registered_path, return_metadata=True, verbose=False)
    return MovieSpec(
        name="figure7_real_calcium_2d_t_raw_vs_registered",
        figure_subdir="figure7",
        raw=np.asarray(raw, dtype=np.float32),
        registered=np.asarray(registered, dtype=np.float32),
        metadata=metadata,
        channels=(0,),
        projection_mode="max",
        title="Real calcium-imaging 2D+t registration")

def specs_rigid3d() -> list[MovieSpec]:
    specs = []
    for stem, title in [
        ("rigid3d_dense_simpleitk", "Full 3D rigid registration: dense structural benchmark"),
        ("rigid3d_sparse_points", "Full 3D rigid registration: sparse puncta benchmark")]:
        path = BENCHMARK_DIR / f"{stem}.npz"
        if not path.exists():
            print(f"Skipping rigid 3D movie; missing {path}")
            continue
        data = np.load(path)
        raw = np.asarray(data["raw"], dtype=np.float32)
        registered = np.asarray(data["registered"], dtype=np.float32)
        specs.append(
            MovieSpec(
                name=f"figure4_{stem}_raw_vs_registered",
                figure_subdir="figure4",
                raw=raw,
                registered=registered,
                metadata={},
                channels=(0, 1) if raw.shape[2] > 1 else (0,),
                projection_mode=MOVIE_MODE_3D,
                title=title,
                scale_bar=False))
    return specs

def pseudo_time_shifts(time_count: int, *, max_z: float, max_y: float, max_x: float) -> np.ndarray:
    shifts = np.zeros((int(time_count), 3), dtype=np.float32)
    denominator = max(int(time_count) - 1, 1)
    for t in range(int(time_count)):
        phase = 2.0 * np.pi * t / denominator
        shifts[t] = (
            max_z * np.sin(phase),
            max_y * np.sin(phase + 0.45),
            max_x * (np.cos(phase + 0.2) - np.cos(0.2)))
    shifts[0] = 0.0
    return shifts

def pseudo_time_rotations_z(time_count: int, *, max_z_deg: float) -> np.ndarray:
    rotations = np.zeros((int(time_count), 3), dtype=np.float32)
    denominator = max(int(time_count) - 1, 1)
    for t in range(int(time_count)):
        rotations[t, 0] = max_z_deg * np.sin(2.0 * np.pi * t / denominator)
    rotations[0] = 0.0
    return rotations

def build_pseudo_time_stack(base_stack: np.ndarray, *, shifts_zyx: np.ndarray, rotations_zyx_deg: np.ndarray, center_zyx: tuple[float, float, float]) -> np.ndarray:
    time_count = shifts_zyx.shape[0]
    _, z_count, channel_count, y_count, x_count = base_stack.shape
    output = np.zeros((time_count, z_count, channel_count, y_count, x_count), dtype=np.float32)
    base = np.asarray(base_stack[0], dtype=np.float32)
    for t in range(time_count):
        rot_z, rot_y, rot_x = [float(v) for v in rotations_zyx_deg[t]]
        shift_zyx = tuple(float(v) for v in shifts_zyx[t])
        for c in range(channel_count):
            output[t, :, c] = _apply_zyx_rigid_transform(
                base[:, c],
                shift_zyx=shift_zyx,
                rotation_z_deg=rot_z,
                rotation_y_deg=rot_y,
                rotation_x_deg=rot_x,
                center_zyx=center_zyx)
    return output

def specs_real_3p_pseudot() -> list[MovieSpec]:
    if not THREEP_INPUT_PATH.exists():
        print(f"Skipping real 3P pseudo-time movies; missing {THREEP_INPUT_PATH}")
        return []
    translation_registered_path = THREEP_OUTPUT_DIR / "real_3p_pseudot_translation_registered.ome.tif"
    rigid_registered_path = THREEP_OUTPUT_DIR / "real_3p_pseudot_translation_rotation_registered.ome.tif"
    if not translation_registered_path.exists() or not rigid_registered_path.exists():
        print("Skipping real 3P pseudo-time movies; registered outputs are missing.")
        return []

    stack_3p, metadata_3p = load_stack(THREEP_INPUT_PATH, return_metadata=True, use_memmap=True, memmap_folder=None, verbose=False)
    if THREEP_FULL_VOLUME:
        z_start, z_stop = 0, stack_3p.shape[1]
        y_start, y_stop = 0, stack_3p.shape[3]
        x_start, x_stop = 0, stack_3p.shape[4]
    else:
        z_start, z_stop = centered_bounds(stack_3p.shape[1], THREEP_Z_COUNT)
        y_start, y_stop = centered_bounds(stack_3p.shape[3], THREEP_YX_SIZE)
        x_start, x_stop = centered_bounds(stack_3p.shape[4], THREEP_YX_SIZE)
    base_stack = np.asarray(stack_3p[0:1, z_start:z_stop, :, y_start:y_stop, x_start:x_stop], dtype=np.float32)
    metadata_base = update_stack_metadata(create_stack_metadata(base_stack, input_metadata=metadata_3p, verbose=False), base_stack, verbose=False)
    center_zyx = ((base_stack.shape[1] - 1) / 2.0, (base_stack.shape[3] - 1) / 2.0, (base_stack.shape[4] - 1) / 2.0)

    translation_shifts = pseudo_time_shifts(THREEP_TIME_COUNT, max_z=THREEP_TRANSLATION_MAX_Z, max_y=THREEP_TRANSLATION_MAX_Y, max_x=THREEP_TRANSLATION_MAX_X)
    translation_rotations = np.zeros((THREEP_TIME_COUNT, 3), dtype=np.float32)
    translation_raw = build_pseudo_time_stack(base_stack, shifts_zyx=translation_shifts, rotations_zyx_deg=translation_rotations, center_zyx=center_zyx)
    translation_registered, _ = load_stack(translation_registered_path, return_metadata=True, verbose=False)

    rigid_shifts = pseudo_time_shifts(THREEP_TIME_COUNT, max_z=THREEP_RIGID_MAX_Z, max_y=THREEP_RIGID_MAX_Y, max_x=THREEP_RIGID_MAX_X)
    rigid_rotations = pseudo_time_rotations_z(THREEP_TIME_COUNT, max_z_deg=THREEP_RIGID_MAX_ROTATION_Z_DEG)
    rigid_raw = build_pseudo_time_stack(base_stack, shifts_zyx=rigid_shifts, rotations_zyx_deg=rigid_rotations, center_zyx=center_zyx)
    rigid_registered, _ = load_stack(rigid_registered_path, return_metadata=True, verbose=False)

    return [
        MovieSpec(
            name="figure8_real_3p_pseudot_translation_raw_vs_registered",
            figure_subdir="figure8",
            raw=translation_raw,
            registered=np.asarray(translation_registered, dtype=np.float32),
            metadata=metadata_base,
            channels=(0, 1) if translation_raw.shape[2] > 1 else (0,),
            projection_mode=MOVIE_MODE_3D,
            title="Real three-photon pseudo-time 3D registration: translation"),
        MovieSpec(
            name="figure8_real_3p_pseudot_rigid_raw_vs_registered",
            figure_subdir="figure8",
            raw=rigid_raw,
            registered=np.asarray(rigid_registered, dtype=np.float32),
            metadata=metadata_base,
            channels=(0, 1) if rigid_raw.shape[2] > 1 else (0,),
            projection_mode=MOVIE_MODE_3D,
            title="Real three-photon pseudo-time 3D registration: rigid")]

def spec_real_3dtch() -> MovieSpec | None:
    input_path = REPO_ROOT / "example_data" / "MotilA_example_data" / "TLmerged_2" / "240321_21584_P17_3_merged_1_8.ome.tif"
    registered_path = BENCHMARK_DIR / "real_3DtCh" / "real_3DtCh_registered.ome.tif"
    summary_path = BENCHMARK_DIR / "real_3DtCh" / "real_3DtCh_summary.json"
    if not input_path.exists() or not registered_path.exists():
        print(f"Skipping real 3D+t+Ch movie; missing {input_path} or {registered_path}")
        return None
    raw, metadata = load_stack(
        input_path,
        return_metadata=True,
        use_memmap=True,
        memmap_folder=BENCHMARK_DIR / "real_3DtCh" / "omio_memmap_cache",
        verbose=False)
    raw = crop_raw_to_analysis_stack(raw, load_json(summary_path))
    registered, _ = load_stack(
        registered_path,
        return_metadata=True,
        use_memmap=True,
        memmap_folder=BENCHMARK_DIR / "real_3DtCh" / "registered_movie_memmap_cache",
        verbose=False)
    return MovieSpec(
        name="figure9_real_3DtCh_raw_vs_registered",
        figure_subdir="figure9",
        raw=np.asarray(raw, dtype=np.float32),
        registered=np.asarray(registered, dtype=np.float32),
        metadata=metadata,
        channels=(0, 1) if raw.shape[2] > 1 else (0,),
        projection_mode=MOVIE_MODE_3D,
        title="Real 2P 3D+t+Ch registration")

# %% RUN
movie_specs: list[MovieSpec] = []
if CREATE_REPRESENTATIVE_2D:
    spec = spec_representative_2d()
    if spec is not None:
        movie_specs.append(spec)
if CREATE_REAL_CALCIUM_2D:
    spec = spec_real_calcium()
    if spec is not None:
        movie_specs.append(spec)
if CREATE_SYNTHETIC_RIGID_3D:
    movie_specs.extend(specs_rigid3d())
if CREATE_REAL_3P_PSEUDOT:
    movie_specs.extend(specs_real_3p_pseudot())
if CREATE_REAL_3DTCH:
    spec = spec_real_3dtch()
    if spec is not None:
        movie_specs.append(spec)

for spec in movie_specs:
    output = save_movie(spec)
    print(f"Wrote movie: {output}")

# %% END
