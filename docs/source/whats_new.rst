What's New?
===========

ZenReg is currently in early development. This page summarizes user-visible
changes as the package evolves.

Version 0.0.1
-------------

Initial development release.

Highlights
~~~~~~~~~~

- OMIO-based microscopy I/O with canonical ``TZCYX`` handling.
- Main ``register_stack`` wrapper for phase cross-correlation, pystackreg, and
  NoRMCorre-style registration.
- Projection-based and full-volume 3D translational registration.
- Optional Z-shift estimation.
- Optional 2D in-plane rotation correction.
- Intra-stack XY slice correction for 3D and 3D+t stacks.
- NoRMCorre-style rigid and piecewise-rigid correction for 2D+t and 3D+t.
- Full 3D rigid 6-DOF registration via SimpleITK or sparse points.
- Optional OMIO disk-backed Zarr workflows for large stacks.
- Registration report sidecars: CSV shifts/correlations, YAML settings, and
  summary plots.
- Synthetic benchmark datasets with ground-truth tables.
- Interactive tutorial scripts for synthetic examples, NoRMCorre comparisons,
  full 3D rigid registration, profiling, and batch processing.
