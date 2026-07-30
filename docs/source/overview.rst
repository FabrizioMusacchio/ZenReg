Overview
========

ZenReg is a registration platform for microscopy stacks. It focuses on
transparent, scriptable workflows that can be inspected frame by frame in
interactive environments such as VS Code's interactive window, Jupyter, or
napari.

Core ideas
----------

ZenReg separates the workflow into three explicit steps:

- ``load_stack`` reads microscopy data via OMIO and returns canonical ``TZCYX``
  image data plus metadata.
- ``register_stack`` estimates and applies motion correction using a selected
  backend and registration mode.
- ``save_stack`` writes a registered OME-TIFF and optional report sidecars.

This keeps project scripts short while still making every important setting
visible and reproducible.

Supported data model
--------------------

Internally, ZenReg expects image arrays in OME-compliant ``TZCYX`` order:

.. code-block:: text

   T = time
   Z = z slices
   C = channels
   Y = image rows
   X = image columns

OMIO normalizes supported inputs to this order, even when an input file has
singleton or implicit dimensions. This makes channel, time, and z handling
consistent across TIFF, OME-TIFF, CZI, LSM, and Thorlabs RAW files.

Registration modes
------------------

ZenReg currently supports:

- 2D+t global translational registration on YX projections.
- 2D+t in-plane rotation correction.
- 3D+t translational registration on Z projections or full ZYX volumes.
- 3D and 3D+t intra-stack XY slice correction.
- NoRMCorre-style rigid and piecewise-rigid motion correction for 2D+t and
  3D+t.
- Full 3D rigid 6-DOF registration with dense SimpleITK or sparse point-based
  backends.
- Optional memory-efficient workflows with OMIO disk-backed Zarr arrays.

Output philosophy
-----------------

When ``registration_details`` are passed to ``save_stack``, ZenReg writes:

- the registered OME-TIFF,
- a CSV table with detected shifts, rotations, and correlations where
  available,
- a YAML settings file for reproducibility,
- a summary plot for quick quality control.

License
-------

ZenReg is distributed under the terms of the GNU General Public License v3.0
or later.
