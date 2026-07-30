2D+t Registration
=================

The 2D+t examples use stacks with shape ``T, 1, C, Y, X``. The registration
channel is used to estimate motion; the detected correction is then applied to
all channels.

Phase Cross-Correlation
-----------------------

Use ``phase_cross_correlation`` for fast global translation correction:

.. code-block:: python

   from pathlib import Path
   from zenreg import load_stack, register_stack, save_stack

   path = Path("example_data/synthetic_data/synthetic_2d_t_xy.ome.tif")
   image, metadata = load_stack(path, return_metadata=True)

   registered, details = register_stack(
       image,
       registration_channel=0,
       registration_stack=0,
       method="phase_cross_correlation",
       time_registration_mode="projection",
       time_reference_mode="template",
       projection_method="max",
       zreg=False,
       max_xy_shifts=(8, 8),
       n_jobs=4,
       return_shifts=True,
       return_details=True,
   )

   save_stack(
       "example_data/synthetic_data/registered/2d_t_xy_phase_registered.ome.tif",
       registered,
       metadata=metadata,
       registration_details=details,
   )

``projection_method="max"`` is a good default for sparse spots or puncta.
``mean`` can be better for dense signals, while ``median`` is robust to
outliers but can attenuate sparse spots.

pystackreg
----------

``pystackreg`` provides a StackReg-style 2D backend. In ZenReg, it operates on
the same YX projections:

.. code-block:: python

   registered, details = register_stack(
       image,
       registration_channel=0,
       registration_stack=0,
       method="pystackreg",
       time_registration_mode="projection",
       projection_method="max",
       zreg=False,
       return_shifts=True,
       return_details=True,
   )

NoRMCorre
---------

NoRMCorre is available through the same main wrapper:

.. code-block:: python

   registered, details = register_stack(
       image,
       registration_channel=0,
       registration_stack=0,
       method="normcorre",
       time_registration_mode="projection",
       projection_method="max",
       nc_pw_rigid=True,
       nc_strides=(32, 32),
       nc_overlaps=(16, 16),
       nc_max_deviation_rigid=3,
       nc_n_jobs=4,
       return_shifts=True,
       return_details=True,
   )

Before choosing ``nc_strides`` and ``nc_overlaps``, it is useful to draw the
patch layout:

.. code-block:: python

   from zenreg import plot_normcorre_patch_overlay

   plot_normcorre_patch_overlay(
       image,
       metadata,
       registration_channel=0,
       registration_stack=0,
       nc_strides=(32, 32),
       nc_overlaps=(16, 16),
       projection_method="max",
       projection_range=(0, 1),
   )

Memory-efficient workflow
-------------------------

For large 2D+t files, use OMIO disk-backed Zarr loading and ask ZenReg to write
intermediate registered output to disk-backed Zarr as well:

.. code-block:: python

   from zenreg import cleanup_omio_cache, load_stack, register_stack, save_stack

   cache_dir = "local_omio_cache"
   cleanup_omio_cache(cache_dir, full_cleanup=True)

   image, metadata = load_stack(
       "large_timeseries.ome.tif",
       return_metadata=True,
       use_memmap=True,
       memmap_folder=cache_dir,
       memmap_reuse=True,
   )

   registered, details = register_stack(
       image,
       registration_channel=0,
       method="phase_cross_correlation",
       time_registration_mode="projection",
       output_use_memmap=True,
       output_memmap_folder=cache_dir,
       n_jobs=8,
       return_shifts=True,
       return_details=True,
   )

   save_stack("large_timeseries_registered.ome.tif", registered, metadata=metadata, registration_details=details)
   cleanup_omio_cache(cache_dir, full_cleanup=True)

Rigid rotation correction
-------------------------

For 2D+t translation plus in-plane rotation, enable ``rotreg``. ZenReg first
estimates translation, then rotation from polar-transformed projections, then
runs a final translation refinement. Increase ``rotreg_iter`` to alternate
rotation and translation refinement more than once.

.. code-block:: python

   path = Path("example_data/synthetic_data/synthetic_2d_t_trans_rot_xy.ome.tif")
   image, metadata = load_stack(path, return_metadata=True)

   registered, details = register_stack(
       image,
       registration_channel=0,
       registration_stack=0,
       method="phase_cross_correlation",
       time_registration_mode="projection",
       projection_method="max",
       rotreg=True,
       max_rot_shifts=12,
       rotreg_iter=1,
       transform_backend="skimage",
       transform_order=1,
       return_shifts=True,
       return_details=True,
   )
