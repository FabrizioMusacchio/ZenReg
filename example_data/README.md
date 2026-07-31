# ZenReg Example Data

This folder is the default location for ZenReg tutorial and benchmark example
datasets.

The repository keeps this folder so tutorial paths are stable, but generated
image data are not meant to be committed. Create the synthetic datasets locally
when you want to run the tutorials:

```bash
python additional_scripts/create_synthetic_example_data.py
```

The script writes OME-TIFF stacks and matching ground-truth CSV tables into:

```text
example_data/synthetic_data/
```

It also creates a small BIDS-like synthetic project for the batch-processing
tutorial:

```text
example_data/synthetic_batch_project/
```

The generated datasets cover 2D+t, 3D, 3D+t, intra-stack correction, NoRMCorre,
rotation, full 3D rigid registration, memory-mapped workflows, and batch
processing examples used throughout the documentation and `user_scripts/`.

Typical workflow:

```bash
python additional_scripts/create_synthetic_example_data.py
python user_scripts/register_synthetic_examples_interactive.py
```

Registered outputs, QC figures, registration CSV files, settings YAML files,
and summary plots are written into subfolders below `example_data/synthetic_data/`.

