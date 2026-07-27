# ZenReg: A Python-based high-throughput, memory-efficient N-dimensional image registration pipeline

ZenReg is a Python-based high-throughput, memory-efficient pipeline for N-dimensional image registration, designed for large microscopy and imaging datasets.

The project targets volumetric and time-resolved data with arbitrary dimensionality (3D, 4D, and full 5D stacks such as TZCYX), with a strong focus on scalable, out-of-core processing using chunked array storage and parallel execution.

**ZenReg is currently under active development**. This repository hosts an initial  release while the core registration pipeline is being refactored, cleaned up, and prepared for open-source publication.

## Scope and design goals
ZenReg is designed with the following principles in mind:

* support for true N-dimensional image data without special-casing (e.g. T=1, C=1, or Z=1)
* memory-efficient, out-of-core processing via chunked array storage
* scalable and parallel execution for high-throughput workloads
* suitability for large microscopy datasets that do not fit into memory
* clean and explicit handling of dimensional metadata

## What ZenReg currently does
ZenReg currently focuses on translation-based registration for microscopy image
stacks represented in canonical `TZCYX` order:

- `T`: time
- `Z`: z-slices
- `C`: channels
- `Y, X`: image plane coordinates

The current core includes:

- time-wise stack registration with shifts estimated from configurable Z projections
  or full-3D phase cross-correlation;
- optional intra-stack Z-drift correction, where individual z-slices are aligned
  to local or full-stack projections;
- optional Z-shift estimation and correction for 3D+t time registration, either
  from full 3D phase cross-correlation or from orthogonal Z projections;
- template-based or sequential previous-frame time registration;
- optional median filtering on slices before projection and on projections before
  shift estimation;
- two shift-estimation backends: `phase_cross_correlation` from scikit-image and
  `pystackreg`;
- small filtering and Z-projection helpers;
- OMIO-backed microscopy I/O for TIFF/OME-TIFF, CZI, LSM, and Thorlabs RAW,
  normalized to canonical `TZCYX` with metadata inheritance on OME-TIFF output.

## Installation for local development
We recommend using a Python virtual environment for local development. You can create one with:

```bash
conda create -n zenreg -y python=3.12
conda activate zenreg
```


You can install ZenReg from PyPI with:

```bash
pip install zenreg
```

From inside this folder (dev mode), you can install ZenReg with:

```bash
python -m pip install -e ".[dev]"
```

In case you want to upgrade ZenReg to the latest version from this repository, you can run:

```bash
pip install --upgrade zenreg
```

or

```bash
python -m pip install -e ".[dev]" --upgrade
```


The examples use only local synthetic data. Generate them with:

```bash
python additional_scripts/create_synthetic_example_data.py
```

Then run the interactive VS Code script:

```text
user_scripts/register_synthetic_examples_interactive.py
```

The script is structured with `# %%` cells so it can be executed step by step in
VS Code's interactive window.

## Minimal usage

```python
from zenreg import load_stack, register_stack, save_stack

stack, metadata = load_stack(
    "example_data/synthetic_data/motion_distorted_2d_tzcyx.ome.tif",
    return_metadata=True)
registered, shifts = register_stack(
    stack,
    registration_channel=0,
    registration_stack=0,
    projection_method="max",
    method="phase_cross_correlation",
    return_shifts=True)
save_stack(
    "example_data/synthetic_data/registered/motion_distorted_2d_registered.ome.tif",
    registered,
    metadata=metadata)
print(shifts)
```

For a 3D time-lapse stack with intra-stack Z motion:

```python
from zenreg import register_stack

z_corrected, intra_shifts = register_stack(
    stack,
    registration_channel=0,
    time_registration_mode="none",
    intra_stack=True,
    intra_stack_reference_mode="neighbor",
    neighbor_window_size=3,
    return_shifts=True)
registered, shift_details = register_stack(
    z_corrected,
    registration_channel=0,
    registration_stack=0,
    time_registration_mode="full_3d",
    time_reference_mode="template",
    method="phase_cross_correlation",
    projection_method="max",
    zreg=True,
    max_xy_shifts=None,
    max_z_shifts=None,
    zrange=None,
    return_shifts=True)
```

## Project status
No stable public API is provided yet.

## License
ZenReg is released under an open-source license (GPL-3.0). See the LICENSE file for details.

## Citation
If you use ZenReg in your research, please cite it as:

```
Fabrizio Musacchio (2026). ZenReg: A Python-based high-throughput, memory-efficient N-dimensional image registration pipeline. Zenodo. https://doi.org/10.5281/zenodo.XXXXXXX
``` 
