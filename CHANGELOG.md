## Changelog

See here for a detailed list of changes made in each release of *ZenReg*. Please, also refer to the Repository [Releases page](https://github.com/FabrizioMusacchio/zenreg/releases).

Each release is also archived on Zenodo for long-term preservation and citation purposes:

[![Zenodo Archive](https://img.shields.io/badge/Zenodo%20Archive-10.5281%2Fzenodo.21727826-blue)](https://doi.org/10.5281/zenodo.21727826)

---

### 🔜 v0.0.13 - UNRELEASED

#### 🧩 Changes and improvements
##### Registration summary plots
- Summary plot annotations now report both the input and output stack shape as
  `shape_TZCYX before registration=(...) | after=(...)`, making zero-clipping
  and other shape-changing registration steps immediately visible in reports.

##### Interactive execution
- `register_stack(..., zero_clip=True, verbose=True)` now prints an explicit
  message when zero clipping is skipped because no common valid image region can
  be retained. The detailed reason remains stored in `zero_clip_failed_reason`.


---

### 🚀 v0.0.12

August 12, 2026

This release adds explicit progress messages to the BIDS-like batch processor and reworks the API reference documentation for cleaner function entries.

#### 🧩 Changes and improvements
##### Batch progress output
- Added explicit flushed progress messages to `register_bids_like_batch` after loading and registration, and before/after saving registered OME-TIFF images and ZenReg report sidecars. This makes long interactive batch runs easier to follow when tqdm bars are cleared by the terminal frontend.

##### Batch run reports
- Added optional root-level batch run reports to `register_bids_like_batch`. ZenReg now writes `zenreg_batch_run_report.yaml` and `zenreg_batch_run_report.txt` by default, preserving a per-image run history across multiple batch runs.
- Added `write_run_report`, `run_report_name`, `run_report_format`, and `run_report_status_symbol_style` options to control project-level batch run reporting.

##### Documentation
- Reworked the API reference from autosummary-generated fully qualified names to explicit autodoc entries. Function entries now render in the cleaner OMIO-style format, for example `load_stack()` instead of `zenreg.load_stack`.

---

### 🚀 v0.0.11

August 11, 2026

This release adds explicit progress messages to the BIDS-like batch processor and reworks the API reference documentation for cleaner function entries.

#### 🧩 Changes and improvements
##### Batch RAW metadata repair
- Renamed and simplified the Thorlabs RAW YAML repair helper from `create_thorlabs_raw_yaml_templates_from_batch_report` to `batch_create_thorlabs_raw_yaml_templates`.
- Reduced `batch_create_thorlabs_raw_yaml_templates` to its specific task: reading RAW paths and editable `template_metadata` blocks from a ZenReg root error report and creating OMIO YAML sidecars for those paths.
- Removed BIDS-like rediscovery arguments from the RAW YAML template helper (`subject_ids`, `subject_prefix`, `tag_folder_levels`, `image_patterns`, `exclude_name_contains`, and `restrict_to_discovered`); the function now deliberately trusts the paths already recorded in the ZenReg error report.
- Simplified the public and private OMIO YAML template creator scripts so they expose only the report path, fallback metadata, overwrite behavior, and verbosity settings.

---

### 🚀 v0.0.10

August 10, 2026

Introducing: BIDS-like batch processing. This release adds a new batch processor that discovers microscopy image files in nested BIDS-like project trees, loads them with OMIO, runs `register_stack`, saves registered OME-TIFF outputs, manages optional disk-backed caches, skips already registered files, and returns a structured batch result. The batch processor also writes a root-level error report for skipped files, which can be used to create OMIO Thorlabs RAW YAML bypass templates for selected skipped RAW files.

#### ✨ New features
##### Batch processing
- Added `register_bids_like_batch`, a true BIDS-like batch processor that discovers image files, loads them with OMIO, runs `register_stack`, saves registered OME-TIFF outputs, manages optional disk-backed caches, skips already registered files, and returns a structured batch result.
- Added flexible nested tag-folder discovery via `discover_bids_like_batch_images`, including token-based matching such as `("DC000_FOV", "DA000_FOV")` for folders like `DC000_FOV1` and `DA000_FOV2`.
- Added batch-level skipped-file reporting with per-folder reports and a root-level, copy-pasteable Python dictionary for later OMIO Thorlabs RAW YAML template creation.
- Added `create_thorlabs_raw_yaml_templates_from_batch_report`, a batch helper that reads ZenReg root error reports and creates OMIO Thorlabs RAW YAML bypass templates for selected skipped RAW files using BIDS-like folder selection logic.

#### 🧩 Changes and improvements
##### Batch processing
- Reworked the synthetic batch user script to demonstrate both a simple custom loop and the new `register_bids_like_batch` workflow.
- Added a private Katharina batch template script that maps the new batch processor to nested `ID*/DC000_FOV*/TL_000/*.raw` style project trees.
- Added public and private OMIO YAML template creator scripts for creating Thorlabs RAW YAML bypass files from ZenReg batch error reports.
- Removed the older public `iter_bids_like_image_files` helper from the ZenReg top-level API in favor of the more general discovery and full batch registration functions.

##### Registration robustness
- `registration_template_time_range` values whose stop exceeds the available
  number of time points are now clipped to `T` with a warning instead of
  aborting the registration.

##### Documentation
- Rebuilt the RTD batch-processing page around the BIDS-like project tree,
  synthetic batch project creation, custom loops, and the new full batch
  processor.

##### Interactive execution
- Progress bars now use the standard text-based `tqdm` backend instead of
  `tqdm.auto`, avoiding optional `ipywidgets` warnings in VS Code interactive
  windows.

--- 

### 🚀 v0.0.9

August 10, 2026

This release updates ZenReg's OMIO dependency to use the latest Thorlabs RAW metadata fallback behavior introduced in OMIO v0.2.8.

#### 🧩 Changes and improvements
##### I/O robustness

- Updated the required OMIO dependency to `omio-microscopy>=0.2.8`.
- ZenReg batch workflows now benefit from OMIO's improved Thorlabs RAW fallback logic, where XML metadata that is parseable but inconsistent with the RAW file size triggers YAML fallback instead of deriving invalid dimensions such as `Z=0`.
- ZenReg's existing `load_stack(..., on_error="return_none")` batch-skip path now also covers this additional RAW metadata inconsistency case when no usable YAML fallback is available.

--- 

### 🚀 v0.0.8

August 7, 2026

This release adds batch-safe image loading and error handling, which is especially useful for large-scale batch processing of microscopy datasets where some files may be unreadable or corrupted.

#### ✨ New features
##### Batch-safe image loading

- Added `load_stack(..., on_error="raise"|"return_none")`, forwarding OMIO v0.2.7's batch-safe read-error handling. The default remains `"raise"` for interactive workflows; batch scripts can request `"return_none"` and skip unreadable files (OMIO then returns `(None, None)`) deliberately.

--- 

### 🚀 v0.0.7

August 6, 2026

Quieter interactive registration output: This minor release adds a new `print_shifts` argument to `register_stack()` and `correct_intra_stack_z_drift()` for optional per-frame/per-slice shift/rotation output, while keeping the default `verbose=True` behavior clean and focused on progress bars and high-level status messages.

#### ✨ New features
##### Interactive execution

- Added `print_shifts` to `register_stack()` and
  `correct_intra_stack_z_drift()`. Progress bars and high-level status messages
  remain controlled by `verbose`, while detailed per-frame or per-slice
  shift/rotation lines can be enabled separately with `print_shifts=True`.

#### 🧩 Changes and improvements
##### Interactive execution

- `print_shifts` defaults to `False`, so `verbose=True` now keeps the useful
  tqdm progress bars without flooding interactive terminals with lines such as
  `t=..., shift_y=..., shift_x=...`.

--- 

### 🚀 v0.0.6

August 6, 2026

This release improves the documentation for NoRMCorre usage and reduces peak RAM use when running NoRMCorre with parallel workers and disk-backed registered-output caches.

#### 🧩 Changes and improvements
##### NoRMCorre wrapper behavior

- `register_stack(method="normcorre")` now warns and ignores `registration_template_time_range` instead of raising an error, because NoRMCorre uses its own `nc_template_init_mode` and `nc_template_update_method` template workflow.
- `register_stack(method="normcorre", zero_clip=True)` now warns and continues with zero clipping disabled until NoRMCorre-compatible zero clipping is implemented.
- Registration details now record requested/effective NoRMCorre behavior for ignored `registration_template_time_range` and `zero_clip` settings.
- NoRMCorre parallel execution now streams completed worker results directly into the registered output array instead of collecting all corrected frames in an intermediate list first. This reduces peak RAM use, especially with disk-backed output caches and larger `nc_n_jobs` values.

##### Documentation

- Expanded the `register_stack()` docstring so every public parameter is documented explicitly, including NoRMCorre-specific `nc_*` options and full 3D rigid `rot_*` options.
- Added NoRMCorre guidance to the 2D+t RTD usage example for choosing `nc_strides`, `nc_overlaps`, and `nc_max_deviation_rigid`, including how these settings differ from global `max_xy_shifts`.
- Clarified NoRMCorre template behavior in the 2D+t RTD usage example, including CaImAn-like `nc_template_init_mode="median"` / `"rigid_median"` and `nc_template_update_method="caiman"` workflows.
- Added memory-efficient napari inspection guidance to the RTD memory workflow,
  Napari workflow, and assessing-results pages, including
  `open_in_napari(..., zarr_mode="zarr_nodask")`.

##### Tutorial helpers

- `open_in_napari` now forwards additional keyword arguments to
  `omio.open_in_napari`, enabling OMIO-specific options such as
  `zarr_mode="zarr_nodask"`.

--- 

### 🚀 v0.0.5

August 4, 2026

New registration reports and interactive progress bar: This release adds a new `write_registration_summary_plot` helper for writing only the ZenReg registration summary PNG after `register_stack`, without saving the full registered image or CSV/YAML sidecars. It also adds `tqdm`-based progress bars for long `register_stack` loops when `verbose=True`, including shift estimation, transform application, rotation correction, and zero-clip mask/crop steps.

#### ✨ New features
##### Registration reports

- Added `write_registration_summary_plot` for writing only the ZenReg registration summary PNG after `register_stack`, without saving the full registered image or CSV/YAML sidecars.

#### 🧩 Changes and improvements
##### Registration reports

- Summary plot annotations now list the registered image dimensions in canonical `TZCYX` order as the first entry.

##### Interactive execution

- Added `tqdm`-based progress bars for long `register_stack` loops when `verbose=True`, including shift estimation, transform application, rotation correction, and zero-clip mask/crop steps. Verbose messages are now flushed immediately to improve feedback in VS Code and Jupyter-style interactive sessions.

--- 

### 🚀 v0.0.4

August 3, 2026

Template controls, reporting refinements, and tutorial helpers. This release adds new controls for building multi-frame registration templates, clearer Z-range selection, raw shift/rotation reporting, single-channel fallback behavior, and OMIO OME-TIFF compression control. It also improves registration summary plots, extends tutorial helpers, and improves synthetic benchmark data generation.

#### ✨ New features
##### Registration controls and outputs

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
- Added user control over OMIO OME-TIFF compression via
  `save_stack(compression_level=...)`.

##### Tutorial and preview helpers

- Added `show_projection` for previewing registration-style Z/time projections
  and all-frame template images before running `register_stack`. Projection
  arrays are returned only when `return_projection=True`.

#### 🧩 Changes and improvements
##### Registration reports

- Summary plots now suppress per-frame line markers for long time series to
  keep large-frame reports readable; clipped estimates are still highlighted.

##### Tutorial and preview helpers

- Extended `show_timepoints` with configurable `reference_time`,
  `moving_time`, and `projection_z_range` arguments.

##### Synthetic data and benchmarks

- Improved synthetic full 3D rigid benchmark data generation for dense
  structural volumes by adding richer volumetric texture and distributed
  features across the field of view. This makes the SimpleITK dense backend
  and point-based backend comparisons better constrained and more
  representative for six-degree-of-freedom registration validation.

#### 📚 Documentation

- Expanded the multi-channel usage guide with the new single-channel fallback
  behavior and the distinction between tolerant single-channel handling and
  strict multi-channel validation.
- Expanded the contributor documentation and project Code of Conduct using the
  more complete OMIO community documentation as a template. The contribution
  guide now includes ZenReg-specific expectations for tests, documentation,
  memory-aware development, and the API contract for adding new registration
  backends.

--- 

### 🚀 v0.0.3

July 31, 2026

This is a dummy release to trigger Zenodo archiving. 

--- 

### 🚀 v0.0.2

July 31, 2026

This is the first real ZenReg release. Version 0.0.1 only established the
package skeleton; v0.0.2 introduces the functional microscopy registration
platform.

#### Core workflow

- Added the main ZenReg workflow: `load_stack -> register_stack -> save_stack`.
- Standardized internal image handling on OMIO-normalized `TZCYX` arrays.
- Added metadata-aware OME-TIFF writing with OMIO metadata inheritance and
  metadata updates after cropped outputs.
- Added support for TIFF, OME-TIFF, CZI, LSM, and Thorlabs RAW input through
  OMIO.

#### Registration methods and modes

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

#### Registration controls

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

#### Memory efficiency and performance

- Added OMIO/Zarr memory-mapped loading via `use_memmap`, `memmap_folder`, and
  `memmap_reuse`.
- Added disk-backed registered-output caches through `output_use_memmap` and
  `output_memmap_folder`.
- Added OMIO cache cleanup helper exposure for explicit cache management.
- Added CPU parallelization controls with `n_jobs` and backend-specific worker
  settings.
- Added utility helpers for available compute-unit inspection.

#### Reporting and reproducibility

- Added registration sidecar outputs next to registered images:
  shift/rotation/correlation CSV files, settings YAML files, and summary plots.
- Added Pearson correlation reporting before and after registration where
  available.
- Added rotation-aware summary plotting for one-axis and full three-axis
  rotation workflows.
- Added transparent settings capture for reproducibility.

#### Synthetic data and tutorials

- Added synthetic OME-TIFF benchmark datasets with two channels and
  ground-truth tables.
- Added 2D+t, 3D, 3D+t, intra-stack, NoRMCorre, full 3D rigid, and batch
  synthetic examples.
- Added NoRMCorre patch/stride overlay plotting to help choose patch geometry.
- Added BIDS-like synthetic batch project generation and batch-processing
  tutorial code.
- Added dedicated user scripts for synthetic registration, NoRMCorre
  comparison, full 3D rigid registration, profiling, and batch processing.

#### Documentation

- Added the first Read the Docs documentation tree with overview,
  installation, usage tutorials, API reference, changelog, and contribution
  guidance.
- Added tutorial-style explanations of important registration arguments and
  defaults.
- Added project logo assets for the README and documentation.

--- 

### 🚀 v0.0.0

February 3, 2026

Hollow project structure created and registered as a Python package.
