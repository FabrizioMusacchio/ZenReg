3D intra-stack registration
===========================

Intra-stack registration corrects XY motion between Z slices inside one 3D
stack. This is different from time registration: the goal is not to align
``t=1`` to ``t=0``, but to align slices within each Z stack.

3D stack with T=1
-----------------

Use ``time_registration_mode="none"`` and ``intra_stack=True``:

.. code-block:: python

   from pathlib import Path
   from zenreg import load_stack, register_stack, save_stack

   path = Path("example_data/synthetic_data/synthetic_3d_z_xy.ome.tif")
   image, metadata = load_stack(path, return_metadata=True)

   registered, details = register_stack(
       image,
       registration_channel=0,
       registration_stack=0,
       method="phase_cross_correlation",
       time_registration_mode="none",
       intra_stack=True,
       intra_stack_reference_mode="neighbor",
       neighbor_window_size=3,
       projection_method="max",
       zreg=False,
       return_shifts=True,
       return_details=True,
   )

   save_stack(
       "example_data/synthetic_data/registered/3d_intra_stack_registered.ome.tif",
       registered,
       metadata=metadata,
       registration_details=details,
   )

Options introduced here
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Argument
     - Meaning
     - Default behavior
   * - ``time_registration_mode="none"``
     - Disable time-point registration. Only the requested intra-stack
       correction is applied.
     - ``"projection"``.
   * - ``intra_stack=True``
     - Correct XY motion between Z slices inside each 3D stack.
     - ``False``.
   * - ``intra_stack_reference_mode``
     - Reference strategy for slice alignment. ``"neighbor"`` uses a local
       neighborhood, while ``"first_slice"`` aligns slices to z=0.
     - ``"neighbor"``.
   * - ``neighbor_window_size``
     - Number of neighboring slices used for local intra-stack references.
     - ``3``.

3D+t intra-stack only
---------------------

For a time series of 3D stacks where each time point should be corrected
internally but not aligned to other time points, keep
``time_registration_mode="none"``:

.. code-block:: python

   path = Path("example_data/synthetic_data/synthetic_3d_t_intra_xy.ome.tif")
   image, metadata = load_stack(path, return_metadata=True)

   registered, details = register_stack(
       image,
       registration_channel=0,
       registration_stack=0,
       method="phase_cross_correlation",
       time_registration_mode="none",
       intra_stack=True,
       intra_stack_reference_mode="neighbor",
       neighbor_window_size=3,
       n_jobs=4,
       return_shifts=True,
       return_details=True,
   )

Combining intra-stack and time registration
-------------------------------------------

If both slice-wise and time-wise correction are needed, set
``intra_stack=True`` and choose a time-registration mode:

.. code-block:: python

   registered, details = register_stack(
       image,
       registration_channel=0,
       method="phase_cross_correlation",
       intra_stack=True,
       time_registration_mode="projection",
       projection_method="max",
       zreg=False,
       return_shifts=True,
       return_details=True,
   )

ZenReg first corrects intra-stack slice motion and then applies the requested
time registration.

Full 3D time registration after intra-stack correction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The second stage can also be full-volume translational registration. This is
useful when slices wobble within each stack and the corrected stacks then drift
globally in ZYX over time:

.. code-block:: python

   registered, details = register_stack(
       image,
       registration_channel=0,
       registration_stack=0,
       method="phase_cross_correlation",
       intra_stack=True,
       intra_stack_reference_mode="neighbor",
       neighbor_window_size=3,
       time_registration_mode="full_3d",
       zreg=True,
       max_xy_shifts=(8, 8),
       max_z_shifts=4,
       return_shifts=True,
       return_details=True,
   )

This combination uses phase cross-correlation for full 3D translational time
registration. Full 3D rigid rotation backends, such as
``rigid_3d_backend="simpleitk"`` and ``"points"``, currently do not support
``intra_stack=True`` in the same call.

New options in this block
~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Argument
     - Meaning
     - Default behavior
   * - ``time_registration_mode="full_3d"``
     - Register full ZYX volumes over time after intra-stack correction.
     - ``"projection"``.
   * - ``zreg=True``
     - Estimate and apply Z shifts during the time-registration stage.
     - ``False``.
   * - ``max_xy_shifts`` / ``max_z_shifts``
     - Optional correction-shift limits for the time-registration stage.
     - ``None`` means no limit.
