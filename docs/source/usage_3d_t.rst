3D+t registration
=================

The 3D+t examples use canonical ``T, Z, C, Y, X`` stacks. ZenReg can register
3D+t time series either on 2D projections or on full ZYX volumes.

Projection-based XY registration
--------------------------------

For global XY motion, register Z projections and apply the detected XY shift
to every Z slice and channel:

.. code-block:: python

   from pathlib import Path
   from zenreg import load_stack, register_stack, save_stack

   path = Path("example_data/synthetic_data/synthetic_3d_t_xy.ome.tif")
   image, metadata = load_stack(path, return_metadata=True)

   registered, details = register_stack(
       image,
       registration_channel   = 0,
       registration_stack     = 0,
       registration_range     = None,
       method                 = "phase_cross_correlation",
       time_registration_mode = "projection",
       projection_method      = "max",
       zreg                   = False,
       n_jobs                 = 4,
       return_shifts          = True,
       return_details         = True)

This is usually the fastest 3D+t option because ZenReg estimates motion on one
2D image per time point instead of on full 3D volumes (which would also require 
full stack-materialization in memory). It works well when the
dominant motion is lateral XY drift and the axial structure remains stable. Do
not use it as the only correction when there is real Z drift, strong axial
tilting, or true 3D rotation; use projection-based Z registration, full-volume
translation, or full 3D rigid registration instead.

Options introduced here:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Argument
     - Meaning
   * - ``time_registration_mode="projection"``
     - Register YX projections over time and apply the resulting correction to
       all Z slices. Default: ``"projection"``. Set to ``full_3d`` for full-volume 
       ZYX registration or ``none`` to disable time registration.
   * - ``registration_range``
     - Optional half-open processing range for quick trial runs. In 3D+t time
       registration this refers to time points, for example ``(20, 60)``.
       Frames outside the range are copied unchanged and the returned stack
       keeps its original shape. ``None`` (default) registers all frames.
   * - ``projection_method``
     - Projection used to build the registration image. Default: ``"max"``.
   * - ``zreg=False``
     - Estimate XY shifts only. Default: ``False``.
   * - ``n_jobs``
     - Worker count for independent time points/slices. ``-1`` uses all
       available CPU workers. Default: ``1``.

Projection-based Z registration
-------------------------------

If the stack also drifts in Z, but you want to avoid full-volume registration,
set ``zreg=True`` in projection mode. ZenReg estimates XY motion from YX
projections and Z motion from orthogonal ZY/ZX projections:

.. code-block:: python

   registered, details = register_stack(
       image,
       registration_channel   = 0,
       registration_stack     = 0,
       method                 = "phase_cross_correlation",
       time_registration_mode = "projection",
       projection_method      = "max",
       zreg                   = True,
       max_z_shifts           = 3,
       return_shifts          = True,
       return_details         = True)

This is faster than full 3D registration and can be useful for moderate Z
drift.

New options in this block:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Argument
     - Meaning
   * - ``zreg=True``
     - Estimate and apply Z shifts in addition to XY shifts. Default: ``False``.
   * - ``max_z_shifts``
     - Optional absolute Z correction-shift limit. ``None`` (default) means no maximum Z-shift clipping.

Full-volume ZYX translation
---------------------------

For true 3D translational registration, use ``time_registration_mode="full_3d"``
and ``zreg=True``. In this mode, scikit-image's phase cross-correlation runs
on the selected ZYX volume and returns Z, Y, and X correction shifts.

.. code-block:: python

   path = Path("example_data/synthetic_data/synthetic_3d_t_zyx.ome.tif")
   image, metadata = load_stack(path, return_metadata=True)

   registered, details = register_stack(
       image,
       registration_channel   = 0,
       registration_stack     = 0,
       method                 = "phase_cross_correlation",
       time_registration_mode = "full_3d",
       registration_z_range   = None,
       zreg                   = True,
       max_xy_shifts          = (8, 8),
       max_z_shifts           = 4,
       n_jobs                 = 4,
       return_shifts          = True,
       return_details         = True)

New options in this block:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Argument
     - Meaning
   * - ``time_registration_mode="full_3d"``
     - Register full ZYX volumes instead of YX projections. Default: ``"projection"``.
   * - ``registration_z_range``
     - Optional half-open Z range used for registration. ``None`` (default) uses all
       slices.
   * - ``max_xy_shifts``
     - Optional absolute XY correction-shift limit as ``(max_y, max_x)``.  Default: ``None``.

Note that full-volume registration requires the entire ZYX volume to be materialized in memory.
Also note that full 3D translational registration is only available for the 
``phase_cross_correlation`` method. 

Rotation around the Z-axis
--------------------------

ZenReg's simple ``rotreg`` path estimates an in-plane rotation from YX
projections. In 3D+t this means rotation around the Z axis. Translation can
still be estimated in different ways.

Full 3D translation plus Z-axis rotation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use this when the stack drifts in ZYX and also rotates slightly around Z:

.. code-block:: python

   path = Path("example_data/synthetic_data/synthetic_3d_t_trans_rot_z.ome.tif")
   image, metadata = load_stack(path, return_metadata=True)

   registered, details = register_stack(
       image,
       registration_channel   = 0,
       registration_stack     = 0,
       method                 = "phase_cross_correlation",
       time_registration_mode = "full_3d",
       projection_method      = "max",
       zreg                   = True,
       rotreg                 = True,
       rigid_3d_backend       = "phase_cross_correlation",
       max_xy_shifts          = (8, 8),
       max_z_shifts           = 4,
       max_rot_shifts         = 10,
       return_shifts          = True,
       return_details         = True)

Projection-based translation plus Z-axis rotation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use this faster variant when the main translation can be estimated from
projections:

.. code-block:: python

   registered, details = register_stack(
       image,
       registration_channel   = 0,
       registration_stack     = 0,
       method                 = "phase_cross_correlation",
       time_registration_mode = "projection",
       projection_method      = "max",
       zreg                   = True,
       rotreg                 = True,
       rigid_3d_backend       = "phase_cross_correlation",
       max_z_shifts           = 3,
       max_rot_shifts         = 10,
       return_shifts          = True,
       return_details         = True)

Projection-only XY translation plus Z-axis rotation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use this when there is no meaningful Z drift and all corrections should be
estimated from YX projections:

.. code-block:: python

   registered, details = register_stack(
       image,
       registration_channel   = 0,
       registration_stack     = 0,
       method                 = "phase_cross_correlation",
       time_registration_mode = "projection",
       projection_method      = "max",
       zreg                   = False,
       rotreg                 = True,
       rigid_3d_backend       = "phase_cross_correlation",
       max_rot_shifts         = 10,
       return_shifts          = True,
       return_details         = True)

New options in this section
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Argument
     - Meaning
   * - ``rotreg``
     - Enable rotation correction. With
       ``rigid_3d_backend="phase_cross_correlation"``, this is Z-axis
       rotation from YX projections. Default: ``False``.
   * - ``rigid_3d_backend="phase_cross_correlation"``
     - Use the simple projection-based rotation path rather than SimpleITK or
       point-based full 3D rigid registration. Default: ``"phase_cross_correlation"``.
   * - ``max_rot_shifts``
     - Optional absolute rotation limit in degrees. Default: ``None``.

What about X/Y-axis rotations?
------------------------------

The simple ``rotreg`` path above only corrects in-plane rotation around Z. It
does not estimate full all-axis 3D rotations. For true ``rotation_z``,
``rotation_y``, and ``rotation_x`` correction, use the full 3D rigid workflow
(see :doc:`usage_full_3d_rigid`) with ``rigid_3d_backend="simpleitk"`` or ``"points"``.
