# How to Contribute

Thank you for your interest in contributing to ZenReg. ZenReg is a modular
Python platform for fast, memory-efficient microscopy image registration. The
project welcomes improvements to registration methods, memory-aware processing,
metadata handling, documentation, tests, tutorials, and benchmark datasets.

The goal of ZenReg is not only to provide registration algorithms, but to make
microscopy registration workflows reproducible: inputs are normalized to
canonical `TZCYX` arrays via OMIO, registration settings are explicit, outputs
can include CSV/YAML/plot sidecars, and large image files can be processed
through disk-backed Zarr stores.

## Before you start

Please check the GitHub issue tracker to see whether your idea, bug report, or
enhancement has already been discussed:

[https://github.com/FabrizioMusacchio/zenreg/issues](https://github.com/FabrizioMusacchio/zenreg/issues)

If a related issue exists, comment there to indicate your interest or to add
relevant technical details. If no issue exists, open a new one with a short
description of:

- what you would like to change or add
- why it is useful in the context of microscopy registration
- whether it affects public API, output files, metadata, or reproducibility
- any thoughts on implementation, edge cases, and testing

For small fixes such as typos or minor documentation improvements, opening a
pull request directly is fine.

## Development environment

ZenReg currently targets Python 3.12. A typical development setup using `conda`
looks like this:

```sh
git clone https://github.com/FabrizioMusacchio/ZenReg.git
cd ZenReg

conda create -n zenreg-dev -c conda-forge python=3.12
conda activate zenreg-dev

pip install -e ".[dev,docs]"
```

If you only need the package itself in editable mode:

```sh
pip install -e .
```

## Making changes and opening pull requests

All code contributions should be submitted as pull requests against the `main`
branch of the repository.

A recommended workflow:

1. Create a new feature branch:

   ```sh
   git checkout -b feature/my-feature
   ```

2. Implement your changes. New public functions or modules should include
   NumPy-style docstrings explaining:
   - their purpose
   - expected inputs and outputs
   - axis assumptions
   - memory behavior
   - limitations or unsupported cases

3. Add tests for new functionality or bug fixes where appropriate.

4. Push your branch and open a pull request that includes:
   - a concise and descriptive title
   - a brief explanation of what changed and why
   - references to related issues, for example `Closes #12`
   - notes about memory use, output sidecars, or reproducibility changes

Draft pull requests are welcome if you would like feedback during development.

## Commit conventions

Clear and consistent commit messages help keep the project history readable.
Prefixes inspired by Conventional Commits are encouraged:

- `feat:` new functionality
- `fix:` bug fixes
- `docs:` documentation changes
- `refactor:` internal code restructuring without behavior changes
- `test:` adding or modifying tests
- `chore:` maintenance tasks or tooling updates

Example:

```text
feat: add template-time-range support for 2D+t registration
```

## Testing

ZenReg uses `pytest` for automated testing. To run the full test suite locally:

```sh
pytest
```

If you add new features or fix bugs, please extend the test suite accordingly.
Tests should remain small and self-contained. Large microscopy datasets should
not be added to the repository. Whenever possible, use synthetic arrays or
minimal example files generated during the test run.

For registration changes, tests should usually cover at least one of:

- known ground-truth translation or rotation recovery
- shape and metadata preservation
- multi-channel transform application
- memory-mapped input or output behavior
- generated report fields and sidecars
- failure modes and warnings for unsupported settings

## Documentation

Build the documentation locally with:

```sh
sphinx-build -b html docs/source docs/build/html
```

Documentation examples should prefer short, complete ZenReg calls. If a helper
is useful across tutorials, add it to ZenReg rather than redefining it in every
example script.

User-facing changes should be reflected in the relevant Read the Docs page,
especially if they affect `load_stack`, `register_stack`, `save_stack`, memory
mapping, reporting, or tutorial helper functions.

## Notes for reviewable scientific software

ZenReg is developed with reproducible scientific use in mind. Contributions
should therefore respect the following principles:

- **Reproducibility**
  Behavior should be deterministic given identical inputs and parameters. Any
  nondeterministic behavior must be explicitly documented.
- **Test coverage**
  New functionality should be accompanied by tests that fail without the change
  and pass with it. Tests should target observable behavior rather than internal
  implementation details.
- **Documentation consistency**
  Public-facing functions must be documented in a way that is consistent with
  their actual behavior. Silent assumptions or undocumented side effects are
  discouraged.
- **Minimal scope changes**
  Pull requests should focus on a well-defined change. Large refactors or
  conceptual redesigns should be discussed in an issue before implementation.
- **Explicit limitations**
  Known limitations or unsupported cases should be documented rather than
  implicitly ignored.

## ZenReg design constraints

ZenReg makes several explicit design decisions. Contributions should preserve
these unless a design change has been discussed first:

- **Canonical axis order**
  Core functions operate on canonical `TZCYX` arrays. Missing dimensions are
  represented with length `1`; axes should not be guessed from raw array shape
  inside registration modules.
- **OMIO I/O boundary**
  File reading and writing should go through OMIO-facing helpers in `zenreg.io`
  whenever possible. Registration code should not implement separate readers for
  microscopy formats.
- **Registration-channel estimation**
  Motion is estimated from one `registration_channel` and the resulting
  transform is applied to every channel. This preserves cross-channel geometry.
- **Explicit settings**
  Parameters that affect registration behavior should be explicit and should
  appear in returned `details` and, where applicable, YAML reports.
- **Memory-aware processing**
  Avoid materializing full stacks when a time-point-wise, slice-wise, or
  chunk-wise implementation is practical. Disk-backed Zarr arrays returned by
  OMIO should be treated as lazy/chunked data sources.
- **Reproducible outputs**
  Registered images, CSV shift tables, YAML settings, and summary plots should
  remain mutually consistent.

## Registration backend extension interface

ZenReg is intended to grow as a registration platform. New registration methods
should be implemented as focused backend modules and exposed through the main
`register_stack(...)` wrapper when they are ready for user-facing use.

### Input contract

Registration backends should accept images in canonical `TZCYX` order:

```text
T: time
Z: z slices, length 1 for 2D/2D+t data
C: channels, length 1 for single-channel data
Y: image rows
X: image columns
```

The input may be a NumPy array or a disk-backed array-like object such as an
OMIO/Zarr-backed store. Backend code should therefore avoid assumptions that
require the whole stack to be a contiguous in-memory NumPy array.

Backends should estimate motion from:

```text
stack[:, :, registration_channel, :, :]
```

and apply the resulting transform to all channels:

```text
stack[t, z, :, y, x]
```

This rule is important for multi-channel microscopy data because registering
channels independently can destroy real spatial relationships.

### Output contract

A backend should return either:

```python
registered
```

or, when details are requested:

```python
registered, details
```

where `registered` is a canonical `TZCYX` array-like object with the same axis
order as the input. Unless `zero_clip=True` or another documented cropping mode
is used, the output shape should match the input shape.

The `details` dictionary should use ZenReg-compatible keys whenever possible:

```python
{
    "time_shifts_zyx": array_or_none,          # shape (T, 3), correction shifts
    "time_shifts_yx": array_or_none,           # shape (T, 2), convenience view
    "time_shifts_zyx_raw": array_or_none,      # pre-clipping estimates
    "time_shifts_yx_raw": array_or_none,
    "intra_stack_shifts_yx": array_or_none,    # shape (T, Z, 2)
    "rotation_shifts_deg": array_or_none,      # shape (T,) or compatible
    "rotation_shifts_zyx_deg": array_or_none,  # shape (T, 3) for 3D rigid
    "pearson_correlations_before": array_or_none,
    "pearson_correlations": array_or_none,
    "registration_channel": int,               # actual channel used
    "registration_channel_requested": int,
    "registration_channel_used": int,
    "registration_channel_fallback": bool,
    "registration_stack": int,
    "method": str,
    "effective_time_registration_mode": str,
    "projection_method": str_or_none,
    "registration_z_range": tuple_or_none,
    "registration_template_time_range": tuple_or_none,
    "zero_clip_bounds": dict_or_none,
}
```

Not every backend needs to populate every key. Unsupported quantities should be
omitted or set to `None`; they should not be silently encoded under incompatible
names.

Shift arrays should store **correction shifts**, not the originally applied
motion. For example, if a synthetic frame was shifted by `(dy, dx)`, the
registration correction is typically `(-dy, -dx)` relative to the reference.

### Integration with `register_stack`

User-facing backends should be reachable through:

```python
registered, details = register_stack(
    stack,
    registration_channel=0,
    method="your_backend",
    return_details=True,
)
```

When adding a new backend:

1. Add the method name to the supported method list.
2. Implement a focused internal dispatch function or module.
3. Convert common `register_stack` arguments to backend-specific arguments.
4. Preserve common output keys in `details`.
5. Make sure `save_stack(..., registration_details=details)` can write useful
   CSV/YAML/plot sidecars.
6. Add tests for the direct backend and the `register_stack` integration path.
7. Document the method in the API reference and at least one usage page.

Backend-specific options should have clear prefixes when exposed through
`register_stack`, for example `nc_...` for NoRMCorre-like settings or `rot_...`
for rotation-specific settings. This avoids collisions with common registration
settings.

### Memory and parallelization requirements

New backends should document whether they support:

- disk-backed input arrays
- disk-backed output arrays
- time-point-wise or slice-wise streaming
- `n_jobs` or backend-specific worker controls
- full-volume operations that necessarily materialize a `ZYX` volume

If a backend must materialize data in memory, document the scope explicitly, for
example "one time point as a `ZYX` volume" rather than "the full `TZCYX` stack".

### Reporting requirements

If a backend estimates shifts, rotations, correlations, or quality metrics, these
should be returned in `details` so that ZenReg can write:

- a CSV table of detected transforms and correlations
- a YAML settings file
- a summary plot

If a metric cannot be computed reliably for a backend, document why and return
`None` rather than a misleading placeholder.

## Reporting bugs

Please report bugs via the GitHub issue tracker:

[https://github.com/FabrizioMusacchio/zenreg/issues](https://github.com/FabrizioMusacchio/zenreg/issues)

Include the following information if possible:

- ZenReg version (`python -c "import zenreg; print(zenreg.__version__)"`)
- Python version
- operating system
- input shape in `TZCYX` order
- registration settings
- backend and version information
- whether OMIO memory mapping/Zarr loading was used
- a small crop or synthetic reproducer if possible
- relevant registration sidecars generated by `save_stack`

## Requests for new registration methods

Users are encouraged to request support for additional registration approaches,
including rigid, non-rigid, feature-based, optical-flow, point-cloud, or
modality-specific methods.

Such requests should include:

- a clear description of the microscopy use case
- the expected input dimensionality, for example 2D+t, 3D+t, or intra-stack 3D
- whether the method should estimate translation, rotation, full rigid motion,
  non-rigid deformation, or another transform model
- whether ground-truth or benchmark data are available
- references to papers, implementations, or established tools if applicable

Representative synthetic or cropped example data are highly useful. Large raw
datasets should not be committed to the repository; please use temporary links,
public archives, or minimal reproducible examples instead.

## License and contributions

By submitting a pull request, you agree that your contributions will be released
under the project's license as specified in the repository.

All interactions in the ZenReg project are governed by the
[Code of Conduct](CODE_OF_CONDUCT.md). If you are unsure how to begin or would
like to discuss a potential contribution, feel free to open an issue to start a
conversation.
