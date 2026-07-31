Optional preprocessing
======================

Phase cross-correlation and projection-based registration can sometimes benefit
from light preprocessing of the images used for shift estimation. This is
especially true for noisy microscopy data, sparse bright outliers, detector
speckles, or weak structures where a small median filter stabilizes the
registration image.

ZenReg distinguishes between two concepts:

- registration-only preprocessing inside ``register_stack``; this affects shift
  estimation but does not filter the registered output image,
- explicit preprocessing with ``apply_filters``; this creates a filtered image
  that you can inspect, register, or save yourself.

Registration-only median filtering
----------------------------------

Use ``filter_slices=True`` to median-filter the registration channel before
projection. For 3D data, this means each Z slice is filtered before ZenReg
creates a projection for registration.

.. code-block:: python

   from pathlib import Path
   from zenreg import load_stack, register_stack

   path = Path("example_data/synthetic_data/synthetic_3d_t_xy.ome.tif")
   image, metadata = load_stack(path, return_metadata=True)

   registered, details = register_stack(
       image,
       registration_channel   = 0,
       registration_stack     = 0,
       method                 = "phase_cross_correlation",
       time_registration_mode = "projection",
       projection_method      = "max",
       filter_slices          = True,
       median_kernel_size     = 3,
       return_shifts          = True,
       return_details         = True)

This can help when individual planes contain salt-and-pepper noise or small
isolated hot pixels. It is usually not necessary for clean synthetic data.

Filtering projections before shift estimation
---------------------------------------------

Use ``filter_projections=True`` to median-filter the projected registration
images after projection but before shift estimation.

.. code-block:: python

   registered, details = register_stack(
       image,
       registration_channel   = 0,
       registration_stack     = 0,
       method                 = "phase_cross_correlation",
       time_registration_mode = "projection",
       projection_method      = "mean",
       filter_projections     = True,
       median_kernel_size     = 3,
       return_shifts          = True,
       return_details         = True)

This is useful when the projection itself is noisy or contains isolated
outliers. It is also cheaper than filtering every Z slice because it runs on one
2D projection per time point.

Combining both filters
----------------------

The two switches can be combined. This is the strongest built-in preprocessing
path and should be used deliberately, because too much smoothing can broaden or
attenuate small puncta.

.. code-block:: python

   registered, details = register_stack(
       image,
       registration_channel   = 0,
       registration_stack     = 0,
       method                 = "phase_cross_correlation",
       time_registration_mode = "projection",
       projection_method      = "max",
       filter_slices          = True,
       filter_projections     = True,
       median_kernel_size     = 3,
       max_xy_shifts          = (8, 8),
       return_shifts          = True,
       return_details         = True)

For sparse spots or puncta, start with ``median_kernel_size=3``. Larger kernels
can make registration more stable in very noisy data, but they can also remove
the fine structures that should drive registration.

Options overview
-----------------

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - Argument
     - Meaning
   * - ``filter_slices``
     - Median-filter Z slices in the registration channel before projection or
       full-3D registration-volume preparation. Default: ``False``.
   * - ``filter_projections``
     - Median-filter 2D registration projections before shift or rotation
       estimation. Default: ``False``.
   * - ``median_kernel_size``
     - Median filter kernel size in pixels. Default: ``3``.
   * - ``pre_median_filter`` / ``post_median_filter``
     - Deprecated aliases kept for older scripts. Use ``filter_slices`` and
       ``filter_projections`` instead.

Notes for different backends
----------------------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Backend
     - Preprocessing behavior
   * - ``phase_cross_correlation``
     - Supports ``filter_slices`` and ``filter_projections``. This is the main
       use case for the built-in median preprocessing switches.
   * - ``pystackreg``
     - Supports the same projection preprocessing path for projection-based
       registration.
   * - ``method="normcorre"``
     - Does not use ``filter_slices`` or ``filter_projections``. Use
       ``nc_gSig_filt`` for CaImAn-style high-pass filtering in the NoRMCorre
       backend.
   * - Full 3D rigid ``simpleitk`` / ``points``
     - Uses its own registration path. Prefer backend-specific settings and
       inspect results carefully before adding explicit image preprocessing.

NoRMCorre high-pass filtering
-----------------------------

For NoRMCorre-style workflows, use ``nc_gSig_filt`` instead of the median
filter switches:

.. code-block:: python

   registered, details = register_stack(
       image,
       registration_channel   = 0,
       registration_stack     = 0,
       method                 = "normcorre",
       time_registration_mode = "projection",
       nc_pw_rigid            = True,
       nc_strides             = (32, 32),
       nc_overlaps            = (16, 16),
       nc_gSig_filt           = (3, 3),
       return_shifts          = True,
       return_details         = True)

``nc_gSig_filt`` follows the NoRMCorre/CaImAn convention and is useful when
slow background variations should contribute less to the motion estimate.

Explicit preprocessing with apply_filters
-----------------------------------------

If you want to inspect or save a filtered image yourself, use ``apply_filters``.
This is not limited to registration-estimation images; it creates a real
filtered array.

.. code-block:: python

   from zenreg import apply_filters

   filtered_image = apply_filters(
       image,
       filters       = "median",
       median_size   = 3,
       apply_3d      = False)

   registered, details = register_stack(
       filtered_image,
       registration_channel   = 0,
       registration_stack     = 0,
       method                 = "phase_cross_correlation",
       time_registration_mode = "projection",
       projection_method      = "max",
       return_shifts          = True,
       return_details         = True)

Use this explicit route only when you really want to register the filtered
image. For most workflows, ``filter_slices`` and ``filter_projections`` are
safer because the registered output remains based on the original intensities.
