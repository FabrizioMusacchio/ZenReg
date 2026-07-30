Synthetic Example Datasets
==========================

ZenReg ships a synthetic dataset generator that writes small OME-TIFF stacks
and ground-truth CSV tables into ``example_data/synthetic_data``. These files
are used by the tutorial scripts and by the examples in this documentation.

Create the datasets
-------------------

From the repository root:

.. code-block:: bash

   python additional_scripts/create_synthetic_example_data.py

Or from Python:

.. code-block:: python

   from pathlib import Path
   from zenreg.synthetic import write_example_dataset

   paths = write_example_dataset(Path("example_data") / "synthetic_data")

What is written?
----------------

The generator creates canonical ``TZCYX`` OME-TIFF files and matching GT CSV
tables. Important datasets include:

.. list-table::
   :header-rows: 1

   * - Dataset
     - Purpose
   * - ``synthetic_2d_t_xy.ome.tif``
     - 2D+t global XY translation relative to ``t=0``.
   * - ``synthetic_2d_t_trans_rot_xy.ome.tif``
     - 2D+t global XY translation plus light in-plane rotation.
   * - ``synthetic_3d_t_xy.ome.tif``
     - 3D+t global XY time registration using Z projections.
   * - ``synthetic_3d_t_zyx.ome.tif``
     - 3D+t global ZYX translation for full-volume registration.
   * - ``synthetic_3d_t_trans_rot_z.ome.tif``
     - 3D+t translation plus rotation around the Z axis.
   * - ``synthetic_3d_z_xy.ome.tif``
     - 3D intra-stack XY slice motion with ``T=1``.
   * - ``synthetic_3d_t_intra_xy.ome.tif``
     - 3D+t intra-stack-only XY slice motion.
   * - ``synthetic_3d_t_rigid_simpleitk.ome.tif``
     - Dense 3D+t full 6-DOF rigid benchmark for SimpleITK.
   * - ``synthetic_3d_t_rigid_points.ome.tif``
     - Sparse puncta 3D+t full 6-DOF rigid benchmark for the point backend.

Every image is accompanied by one or more GT tables, for example
``*_time_shifts_gt.csv`` or ``*_rigid_transform_gt.csv``. These tables contain
the applied synthetic motion and the expected correction relative to the
chosen reference time point.

Interactive scripts
-------------------

The main scripts using these datasets are:

- ``user_scripts/register_synthetic_examples_interactive.py``
- ``user_scripts/register_normcorre_synthetic_examples.py``
- ``user_scripts/register_rigid3d_synthetic_examples.py``
- ``user_scripts/register_batch_bids_like_synthetic.py``

They are designed to be run cell by cell. The scripts intentionally expose the
actual ``register_stack`` calls and settings so they can be copied into new
projects.
