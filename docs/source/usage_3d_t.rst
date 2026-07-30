3D+t Registration
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
       registration_channel=0,
       registration_stack=0,
       method="phase_cross_correlation",
       time_registration_mode="projection",
       projection_method="max",
       zreg=False,
       n_jobs=4,
       return_shifts=True,
       return_details=True,
   )

Projection-based Z registration
-------------------------------

If the stack also drifts in Z, but you want to avoid full-volume registration,
set ``zreg=True`` in projection mode. ZenReg estimates XY motion from YX
projections and Z motion from orthogonal ZY/ZX projections:

.. code-block:: python

   registered, details = register_stack(
       image,
       registration_channel=0,
       registration_stack=0,
       method="phase_cross_correlation",
       time_registration_mode="projection",
       projection_method="max",
       zreg=True,
       max_z_shifts=3,
       return_shifts=True,
       return_details=True,
   )

This is faster than full 3D registration and can be useful for moderate Z
drift.

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
       registration_channel=0,
       registration_stack=0,
       method="phase_cross_correlation",
       time_registration_mode="full_3d",
       projection_range=None,
       zreg=True,
       max_xy_shifts=(8, 8),
       max_z_shifts=4,
       n_jobs=4,
       return_shifts=True,
       return_details=True,
   )

XY-plane rotation only
----------------------

For a 3D+t stack that shifts in ZYX but only rotates in the XY plane, use the
default projection-based rotation backend:

.. code-block:: python

   path = Path("example_data/synthetic_data/synthetic_3d_t_trans_rot_z.ome.tif")
   image, metadata = load_stack(path, return_metadata=True)

   registered, details = register_stack(
       image,
       registration_channel=0,
       registration_stack=0,
       method="phase_cross_correlation",
       time_registration_mode="projection",
       projection_method="max",
       zreg=True,
       rotreg=True,
       rigid_3d_backend="phase_cross_correlation",
       max_rot_shifts=10,
       return_shifts=True,
       return_details=True,
   )

Projection-based XYZ rotation estimates
---------------------------------------

The simple projection-based rotation path can also estimate rotations from
orthogonal projections. It is useful as a quick approximation, but it is not a
full 6-DOF rigid-volume optimization. For true 3D rotations, use the full 3D
rigid-registration workflow described in the next sections.

.. code-block:: python

   registered, details = register_stack(
       image,
       registration_channel=0,
       registration_stack=0,
       method="phase_cross_correlation",
       time_registration_mode="full_3d",
       zreg=True,
       rotreg=True,
       rigid_3d_backend="phase_cross_correlation",
       max_rot_shifts=10,
       return_shifts=True,
       return_details=True,
   )
