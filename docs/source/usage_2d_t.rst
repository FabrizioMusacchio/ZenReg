2D+t registration
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
       return_details=True)

   save_stack(
       "example_data/synthetic_data/registered/2d_t_xy_phase_registered.ome.tif",
       registered,
       metadata=metadata,
       registration_details=details)

``projection_method="max"`` is a good default for sparse spots or puncta.
``mean`` can be better for dense signals, while ``median`` is robust to
outliers but can attenuate sparse spots.

Options used here:

.. list-table::
   :header-rows: 1
   :widths: 20 45 15
   
   * - Argument
     - Meaning
     - Default behavior
   * - ``registration_channel``
     - Channel used to estimate shifts. The correction is applied to all
       channels.
     - Required.
   * - ``registration_stack``
     - Reference time point/template.
     - ``0``.
   * - ``method``
     - Registration backend.
     - ``"phase_cross_correlation"``.
   * - ``time_registration_mode``
     - How time points are aligned. ``"projection"`` means that one YX
       projection is registered per time point.
     - ``"projection"``.
   * - ``time_reference_mode``
     - ``"template"`` aligns all frames to ``registration_stack``;
       ``"previous"`` accumulates frame-to-frame corrections.
     - ``"template"``.
   * - ``projection_method``
     - Z projection used for registration. For 2D+t data, Z is usually a
       singleton axis.
     - ``"max"``.
   * - ``zreg``
     - Whether to estimate Z shifts.
     - ``False``.
   * - ``max_xy_shifts``
     - Optional absolute correction-shift limit as ``(max_y, max_x)``.
     - ``None`` means no XY clipping.
   * - ``n_jobs``
     - Number of CPU workers for independent frames/slices. ``-1`` uses all
       available workers.
     - ``1``.
   * - ``return_shifts`` / ``return_details``
     - Return detected shifts or the full reproducibility/details dictionary.
     - Both are ``False`` unless requested.

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
       max_xy_shifts=(8, 8),
       return_shifts=True,
       return_details=True,
   )

New option in this block
~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Argument
     - Meaning
     - Default behavior
   * - ``method="pystackreg"``
     - Uses the pystackreg backend for 2D projection registration.
     - Without this setting, ZenReg uses ``"phase_cross_correlation"``.

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
       max_xy_shifts=(8, 8),
       nc_n_jobs=4,
       return_shifts=True,
       return_details=True,
   )

New options in this block
~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Argument
     - Meaning
     - Default behavior
   * - ``method="normcorre"``
     - Dispatches to ZenReg's NoRMCorre-style backend.
     - Without this setting, ZenReg uses phase cross-correlation.
   * - ``nc_pw_rigid``
     - Enables piecewise-rigid patch correction instead of rigid-only
       NoRMCorre.
     - ``True``.
   * - ``nc_strides``
     - Patch-grid stride in YX.
     - ``(48, 48)`` for 2D.
   * - ``nc_overlaps``
     - Patch overlap in YX. Effective patch size is
       ``nc_strides + nc_overlaps``.
     - ``(24, 24)`` for 2D.
   * - ``nc_max_deviation_rigid``
     - Maximum local patch deviation around the global rigid shift.
     - ``None`` means not limited.
   * - ``nc_n_jobs``
     - Worker count used by the NoRMCorre backend.
     - ``1``.

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

New options in this block
~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Argument
     - Meaning
     - Default behavior
   * - ``use_memmap``
     - Ask OMIO to read through a disk-backed Zarr cache.
     - ``False``.
   * - ``memmap_folder``
     - Location for the OMIO disk cache. Use local scratch storage for remote
       input data.
     - ``None`` means OMIO chooses its default cache location.
   * - ``memmap_reuse``
     - Reuse a validated existing OMIO cache instead of rebuilding it.
     - ``True``.
   * - ``output_use_memmap``
     - Store ZenReg's intermediate registered result in disk-backed Zarr.
     - ``False``.
   * - ``output_memmap_folder``
     - Folder for ZenReg's disk-backed output cache.
     - ``None``.

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

New options in this block
~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Argument
     - Meaning
     - Default behavior
   * - ``rotreg``
     - Enables in-plane XY/Z-axis rotation estimation from polar-transformed
       projections.
     - ``False``.
   * - ``max_rot_shifts``
     - Optional absolute rotation limit in degrees.
     - ``None`` means no rotation clipping.
   * - ``rotreg_iter``
     - Number of translation/rotation refinement rounds. ``1`` runs
       translation, rotation, translation.
     - ``1``.
   * - ``transform_backend``
     - Backend used to apply XY translations.
     - ``"skimage"``.
   * - ``transform_order``
     - Interpolation order. ``1`` is good for intensity data; ``0`` preserves
       sparse puncta or labels.
     - ``1``.
