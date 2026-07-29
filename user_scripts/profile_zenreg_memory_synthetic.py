"""
Profile ZenReg memory usage on synthetic OMIO/Zarr-backed registration workflows.

Run from the repository root, for example:

    conda run -n zenreg2 python user_scripts/profile_zenreg_memory_synthetic.py

The script writes one CSV memory trace and one PNG plot per case into
``profiling_outputs/``. Synthetic input generation happens before memory
profiling starts so the traces focus on load/register/save behavior.
"""
# %% IMPORTS
import argparse
import gc
from pathlib import Path

import numpy as np

from zenreg import (
    cleanup_omio_cache,
    create_stack_metadata,
    load_stack,
    profile_memory,
    register_stack,
    save_stack,
)
from zenreg.synthetic import (
    create_2d_motion_distorted_stack,
    create_2d_time_translation_rotation_motion_distorted_stack,
)
# %% SETUP
REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "profiling_outputs"
INPUT_DIR = OUTPUT_DIR / "synthetic_inputs"
MEMMAP_DIR = OUTPUT_DIR / "omio_memmap_cache"
# %% HELPER FUNCTIONS
def stack_size_mb(stack) -> float:
    """Return approximate raw stack size in MB."""

    return float(np.prod(stack.shape) * np.dtype(stack.dtype).itemsize) / 1024.0 / 1024.0


def prepare_translation_input(*, time_count: int, shape_yx: tuple[int, int], reuse: bool) -> Path:
    """Create or reuse a synthetic 2D+t translation input OME-TIFF."""

    path = INPUT_DIR / f"synthetic_2d_t_translation_T{time_count}_Y{shape_yx[0]}_X{shape_yx[1]}.ome.tif"
    if reuse and path.exists():
        return path
    stack, _ = create_2d_motion_distorted_stack(
        time_count=time_count,
        channel_count=2,
        shape_yx=shape_yx,
        noise_sigma=0.02,
        random_state=2701,
    )
    print(f"Created translation input: shape={stack.shape}, dtype={stack.dtype}, size={stack_size_mb(stack):.1f} MB")
    written = save_stack(
        path,
        stack,
        metadata=create_stack_metadata(stack, verbose=False),
        overwrite=True,
        verbose=False,
    )
    del stack
    gc.collect()
    return written


def prepare_rotation_input(*, time_count: int, shape_yx: tuple[int, int], reuse: bool) -> Path:
    """Create or reuse a feature-rich synthetic 2D+t translation+rotation input OME-TIFF."""

    path = INPUT_DIR / f"synthetic_2d_t_puncta_translation_rotation_T{time_count}_Y{shape_yx[0]}_X{shape_yx[1]}.ome.tif"
    if reuse and path.exists():
        return path
    stack, _, _ = create_2d_time_translation_rotation_motion_distorted_stack(
        time_count=time_count,
        channel_count=2,
        shape_yx=shape_yx,
        noise_sigma=0.006,
        random_state=2702,
    )
    print(f"Created rotation input: shape={stack.shape}, dtype={stack.dtype}, size={stack_size_mb(stack):.1f} MB")
    written = save_stack(
        path,
        stack,
        metadata=create_stack_metadata(stack, verbose=False),
        overwrite=True,
        verbose=False,
    )
    del stack
    gc.collect()
    return written


def run_profiled_case(
    *,
    case_name: str,
    input_path: Path,
    interval_s: float,
    n_jobs: int,
    zero_clip: bool,
    rotreg: bool,
) -> None:
    """Run one load-register-save workflow under MemoryTracker."""

    csv_path = OUTPUT_DIR / f"{case_name}_memory_trace.csv"
    plot_path = OUTPUT_DIR / f"{case_name}_memory_trace.png"
    registered_path = OUTPUT_DIR / f"{case_name}_registered.ome.tif"

    print(f"\nProfiling case: {case_name}")
    print(f"Input: {input_path}")
    with profile_memory(
        csv_path=csv_path,
        plot_path=plot_path,
        interval_s=interval_s,
        label=case_name,
    ) as tracker:
        tracker.mark("load_stack:start")
        stack, metadata = load_stack(
            input_path,
            return_metadata=True,
            use_memmap=True,
            memmap_folder=MEMMAP_DIR,
            memmap_reuse=True,
            verbose=False,
        )
        tracker.mark("load_stack:end")
        print(f"Loaded stack: shape={stack.shape}, type={type(stack)}")

        tracker.mark("register_stack:call:start")
        registered, details = register_stack(
            stack,
            registration_channel=0,
            registration_stack=0,
            method="phase_cross_correlation",
            time_registration_mode="projection",
            time_reference_mode="template",
            projection_method="max",
            zreg=False,
            rotreg=rotreg,
            zero_clip=zero_clip,
            zero_clip_mode="auto",
            zero_clip_mask_strategy="relaxed" if rotreg else "auto",
            zero_clip_mask_min_fraction=0.5,
            max_xy_shifts=(16, 16) if rotreg else None,
            max_rot_shifts=12 if rotreg else None,
            transform_backend="skimage",
            transform_order=1,
            n_jobs=n_jobs,
            output_use_memmap=True,
            output_memmap_folder=MEMMAP_DIR,
            output_memmap_name=case_name,
            output_dtype=np.float32,
            memory_tracker=tracker,
            verbose=False,
            return_shifts=True,
            return_details=True,
        )
        tracker.mark("register_stack:call:end")
        print(f"Registered stack: shape={registered.shape}, type={type(registered)}")

        tracker.mark("save_stack:start")
        save_stack(
            registered_path,
            registered,
            metadata=metadata,
            registration_details=details,
            overwrite=True,
            verbose=False,
        )
        tracker.mark("save_stack:end")

    summary = tracker.summary()
    print(f"Wrote CSV:  {csv_path}")
    print(f"Wrote plot: {plot_path}")
    print(
        "Memory summary: "
        f"peak RSS={summary.get('peak_rss_mb', float('nan')):.1f} MB, "
        f"final RSS={summary.get('final_rss_mb', float('nan')):.1f} MB, "
        f"duration={summary.get('duration_s', float('nan')):.2f} s"
    )
    del stack, registered
    gc.collect()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--time-count", type=int, default=48)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument("--interval-s", type=float, default=0.10)
    parser.add_argument(
        "--case",
        choices=("translation", "rotation", "all"),
        default="all",
        help="Which profiling case to run.",
    )
    parser.add_argument(
        "--reuse-input",
        action="store_true",
        help="Reuse existing synthetic OME-TIFF inputs instead of regenerating them.",
    )
    parser.add_argument(
        "--fresh-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clear the OMIO/Zarr cache folder before profiling.",
    )
    parser.add_argument(
        "--cleanup-caches",
        action="store_true",
        help="Remove OMIO/Zarr caches after profiling. Leave disabled if you want to inspect them.",
    )
    return parser.parse_args()

# %% MAIN FUNCTION
def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    MEMMAP_DIR.mkdir(parents=True, exist_ok=True)
    if args.fresh_cache:
        cleanup_omio_cache(MEMMAP_DIR, full_cleanup=True, verbose=False)
        MEMMAP_DIR.mkdir(parents=True, exist_ok=True)

    shape_yx = (int(args.height), int(args.width))
    cases = []
    if args.case in {"translation", "all"}:
        input_path = prepare_translation_input(
            time_count=int(args.time_count),
            shape_yx=shape_yx,
            reuse=bool(args.reuse_input),
        )
        cases.append(("standard_translation_memmap", input_path, False, False))
    if args.case in {"rotation", "all"}:
        input_path = prepare_rotation_input(
            time_count=int(args.time_count),
            shape_yx=shape_yx,
            reuse=bool(args.reuse_input),
        )
        cases.append(("rotation_mask_zero_clip_memmap", input_path, True, True))

    for case_name, input_path, zero_clip, rotreg in cases:
        run_profiled_case(
            case_name=case_name,
            input_path=input_path,
            interval_s=float(args.interval_s),
            n_jobs=int(args.n_jobs),
            zero_clip=bool(zero_clip),
            rotreg=bool(rotreg),
        )

    if args.cleanup_caches:
        cleanup_omio_cache(MEMMAP_DIR, full_cleanup=True, verbose=False)

# %% ENTRY POINT
if __name__ == "__main__":
    main()
# %% END
