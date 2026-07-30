3D Intra-stack Registration
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
