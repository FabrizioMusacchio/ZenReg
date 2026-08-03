## Changelog

See here for a detailed list of changes made in each release of *ZenReg*. Please, also refer to the Repository [Releases page](https://github.com/FabrizioMusacchio/zenreg/releases).

Each release is also archived on Zenodo for long-term preservation and citation purposes:

[![Zenodo Archive](https://img.shields.io/badge/Zenodo%20Archive-10.5281%2Fzenodo.21727826-blue)](https://doi.org/10.5281/zenodo.21727826)

--- 

## 🚧 Upcoming release

Unreleased

### 🧩 Changes
#### Synthetic data and benchmarks

- Improved synthetic full 3D rigid benchmark data generation for dense
  structural volumes by adding richer volumetric texture and distributed
  features across the field of view. This makes the SimpleITK dense backend
  and point-based backend comparisons better constrained and more
  representative for six-degree-of-freedom registration validation.

#### Registration controls

- Added `registration_template_time_range` to `register_stack` for building
  multi-frame registration templates from a half-open time range. This is
  especially useful for noisy 2D+t stacks where a template aggregated over
  several frames is more stable than a single reference frame. The value
  `"all"` now expands to all available time points while `None` keeps the
  default single-reference-frame behavior.
- Added `registration_z_range` as the clearer primary name for selecting the
  Z-slice range used to build the registration signal. The older
  `projection_range` and `zrange` names remain supported as compatibility
  aliases.
- Added raw, pre-clipping shift and rotation estimates to registration details,
  CSV reports, and summary plots. Summary plots now mark limit-clipped estimates
  with red open markers and report maximum raw detected shifts/rotations.
- Added a single-channel safeguard for `register_stack`: if a stack has only
  one channel and a non-zero or out-of-range `registration_channel` is requested,
  ZenReg warns and falls back to channel `0`. Registration details and YAML
  reports now record the requested channel, used channel, and fallback reason.
- Summary plots now suppress per-frame line markers for long time series to
  keep large-frame reports readable; clipped estimates are still highlighted.
- Documented user control over OMIO OME-TIFF compression via
  `save_stack(compression_level=...)`.

#### Tutorial and preview helpers

- Extended `show_timepoints` with configurable `reference_time`,
  `moving_time`, and `projection_z_range` arguments.
- Added `show_projection` for previewing registration-style Z/time projections
  and all-frame template images before running `register_stack`. Projection
  arrays are returned only when `return_projection=True`.


--- 

## 🚀 v0.0.3 - Dummy release for Zenodo

July 31, 2026

This is a dummy release to trigger Zenodo archiving. 

--- 

## 🚀 v0.0.2 - First functional ZenReg release

July 31, 2026

This is the first real ZenReg release. Version 0.0.1 only established the
package skeleton; v0.0.2 introduces the functional microscopy registration
platform.

### Core workflow

- Added the main ZenReg workflow: `load_stack -> register_stack -> save_stack`.
- Standardized internal image handling on OMIO-normalized `TZCYX` arrays.
- Added metadata-aware OME-TIFF writing with OMIO metadata inheritance and
  metadata updates after cropped outputs.
- Added support for TIFF, OME-TIFF, CZI, LSM, and Thorlabs RAW input through
  OMIO.

### Registration methods and modes

- Added global translational registration with scikit-image
  `phase_cross_correlation`.
- Added `pystackreg` as a StackReg-style projection-registration backend.
- Added a ZenReg-native NoRMCorre-style backend with rigid and piecewise-rigid
  correction for 2D+t and 3D+t workflows.
- Added 2D+t registration for global XY motion and optional in-plane rotation.
- Added 3D+t registration on Z projections for fast XY correction.
- Added 3D+t Z-shift estimation via orthogonal projections.
- Added full-volume ZYX translational registration with phase
  cross-correlation.
- Added intra-stack correction for 3D and 3D+t stacks, including intra-stack
  only workflows and combined intra-stack plus time registration.
- Added full 3D 6-DOF rigid registration with a dense SimpleITK backend.
- Added a sparse puncta/spot-oriented 3D rigid backend based on point detection
  and point matching.

### Registration controls

- Added explicit `registration_channel` and `registration_stack` controls.
- Added configurable projection methods: `max`, `mean`, `median`, `var`, and
  `std`.
- Added Z-range control for selecting which slices contribute to registration
  templates and projection-based estimates.
- Added optional Z registration with `zreg`.
- Added optional rotation registration with `rotreg`.
- Added optional correction limits with `max_xy_shifts`, `max_z_shifts`, and
  `max_rot_shifts`.
- Added configurable transform backend and interpolation order through
  `transform_backend` and `transform_order`.
- Added optional zero clipping in Z, Y, and X, including mask-based clipping for
  rotation workflows.
- Added post hoc cropping utilities for manual inspection-based cropping.

### Memory efficiency and performance

- Added OMIO/Zarr memory-mapped loading via `use_memmap`, `memmap_folder`, and
  `memmap_reuse`.
- Added disk-backed registered-output caches through `output_use_memmap` and
  `output_memmap_folder`.
- Added OMIO cache cleanup helper exposure for explicit cache management.
- Added CPU parallelization controls with `n_jobs` and backend-specific worker
  settings.
- Added utility helpers for available compute-unit inspection.

### Reporting and reproducibility

- Added registration sidecar outputs next to registered images:
  shift/rotation/correlation CSV files, settings YAML files, and summary plots.
- Added Pearson correlation reporting before and after registration where
  available.
- Added rotation-aware summary plotting for one-axis and full three-axis
  rotation workflows.
- Added transparent settings capture for reproducibility.

### Synthetic data and tutorials

- Added synthetic OME-TIFF benchmark datasets with two channels and
  ground-truth tables.
- Added 2D+t, 3D, 3D+t, intra-stack, NoRMCorre, full 3D rigid, and batch
  synthetic examples.
- Added NoRMCorre patch/stride overlay plotting to help choose patch geometry.
- Added BIDS-like synthetic batch project generation and batch-processing
  tutorial code.
- Added dedicated user scripts for synthetic registration, NoRMCorre
  comparison, full 3D rigid registration, profiling, and batch processing.

### Documentation

- Added the first Read the Docs documentation tree with overview,
  installation, usage tutorials, API reference, changelog, and contribution
  guidance.
- Added tutorial-style explanations of important registration arguments and
  defaults.
- Added project logo assets for the README and documentation.

--- 

## 🚀 v0.0.0

February 3, 2026

Hollow project structure created and registered as a Python package.
