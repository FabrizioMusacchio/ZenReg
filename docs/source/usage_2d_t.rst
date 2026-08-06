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
   from zenreg import load_stack, register_stack, save_stack, show_projection

   path = Path("example_data/synthetic_data/synthetic_2d_t_xy.ome.tif")
   image, metadata = load_stack(path, return_metadata=True)

   show_projection(
       image,
       title                            = "2D+t all-frame template preview",
       registration_channel             = 0,
       registration_template_time_range = "all",
       registration_z_range             = "all",
       projection_method                = "max",
       save_dir                         = "example_data/synthetic_data/registered/figures",
       return_projection                = False)

   registered, details = register_stack(
       image,
       registration_channel   = 0,
       registration_template_time_range = "all",
       method                 = "phase_cross_correlation",
       time_registration_mode = "projection",
       projection_method      = "max",
       zreg                   = False,
       zero_clip              = False,
       max_xy_shifts          = (8, 8),
       transform_backend      = "skimage",
       transform_order        = 1,
       filter_slices          = False,
       filter_projections     = False,
       median_kernel_size     = 3,
       n_jobs                 = 4,
       return_shifts          = True,
       return_details         = True)

   save_stack(
       "example_data/synthetic_data/registered/2d_t_xy_phase_registered.ome.tif",
       registered,
       metadata             = metadata,
       registration_details = details,
       compression_level    = 3)

``projection_method="max"`` is a good default for sparse spots or puncta.
``mean`` can be better for dense signals, while ``median`` is robust to
outliers but can attenuate sparse spots.

Options used here:

.. list-table::
   :header-rows: 1
   :widths: 36 64
   
   * - Argument
     - Meaning
   * - ``registration_channel``
     - Channel used to estimate shifts. The correction is applied to all
       channels.
   * - ``registration_template_time_range``
     - Optional half-open time range ``(start, stop)`` used to build a
       multi-frame registration template. Use ``"all"`` to aggregate all time
       points. ``None`` uses ``registration_stack`` as one reference frame.
       Default: ``None``.
   * - ``method``
     - Registration backend. Default:  ``"phase_cross_correlation"``.
   * - ``time_registration_mode``
     - How time points are aligned. ``"projection"`` means that one YX
       projection is registered per time point. Default: ``"projection"``.
   * - ``time_reference_mode``
     - ``"template"`` aligns all frames to ``registration_stack``;
       ``"previous"`` accumulates frame-to-frame corrections. Default: ``"template"``.
   * - ``projection_method``
     - Z projection used for registration. For 2D+t data, Z is usually a
       singleton axis. If ``registration_template_time_range`` is set, the
       same method is also used to aggregate the selected time points into the
       template. Default:  ``"max"``.
   * - ``zreg``
     - Whether to estimate Z shifts. Default: ``False``.
   * - ``max_xy_shifts``
     - Optional absolute correction-shift limit as ``(max_y, max_x)``.
       ``None`` means no XY clipping. Use a value above the expected motion;
       for very small drift, ``(2, 2)`` can be appropriate.
   * - ``zero_clip``
     - If ``True``, crop zero borders introduced by correction. ``False`` keeps
       the original shape, which is useful for visual before/after comparison.
       Default: ``False``.
   * - ``transform_backend``
     - Backend used to apply translations. Default: ``"skimage"``. Alternative:
       ``"scipy"``.
   * - ``transform_order``
     - Interpolation order. ``1`` is a good default for intensity data;
       ``0`` preserves sparse puncta or label-like images.
   * - ``filter_slices``
     - Median-filter Z slices before projection. For 2D+t data with ``Z=1``,
       this is usually equivalent to filtering each frame before estimation.
       Default: ``False``.
   * - ``filter_projections``
     - Median-filter projected registration images before shift estimation.
       Default: ``False``.
   * - ``median_kernel_size``
     - Median-filter kernel size in pixels for ``filter_slices`` and
       ``filter_projections``. Default: ``3``.
   * - ``n_jobs``
     - Number of CPU workers for independent frames/slices. ``-1`` uses all
       available workers. Default:  ``1``.
   * - ``compression_level``
     - OME-TIFF compression level forwarded to OMIO in ``save_stack``.
       Default: ``3``.
   * - ``return_shifts`` / ``return_details``
     - Return detected shifts or the full reproducibility/details dictionary. Both are ``False`` unless requested.


.. note::

   With ZenReg's helper function ``zenreg.available_cpu_count()``, you can 
   check how many CPU workers are available for your system.

ZenReg comes with a useful helper function ``show_projection`` to preview an
intended registration template. This can help you decide whether to use a
single reference frame or an aggregated multi-frame template, which projection
method to use, and which registration channel to select:

.. code-block:: python

   from zenreg import show_projection

   show_projection(
       image,
       title                            = "Template preview",
       registration_channel             = 0,
       registration_stack               = 0,
       registration_template_time_range = "all",
       registration_z_range             = "all",
       projection_method                = "max",
       save_dir                         = "example_data/synthetic_data/registered/figures",
       return_projection                = False)

Options used by ``show_projection``:

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - Argument
     - Meaning
   * - ``registration_channel``
     - Channel used for the preview image. Match this to the channel you plan
       to use for registration.
   * - ``registration_stack``
     - Single time point to preview when
       ``registration_template_time_range=None``. Default: ``0``.
   * - ``registration_template_time_range``
     - Time range used to aggregate a template preview. Use ``"all"`` for all
       frames, ``(start, stop)`` for a half-open range, or ``None`` for only
       ``registration_stack``.
   * - ``registration_z_range``
     - Z range used before projection. Use ``"all"`` for all available Z
       slices or ``(z_start, z_stop)`` for a half-open range.
   * - ``projection_method``
     - Projection and aggregation method. Same options as registration:
       ``"max"``, ``"mean"``, ``"median"``, ``"var"``, or ``"std"``.
   * - ``save_dir`` / ``save_path``
     - Save the preview figure. ``save_path`` uses an explicit filename;
       ``save_dir`` lets ZenReg generate one from the settings.
   * - ``return_projection``
     - If ``True``, return the projected ``YX`` image. Default: ``False``, so
       the helper only shows/saves the preview.

pystackreg
----------

``pystackreg`` provides a StackReg-style 2D backend. In ZenReg, it operates on
the same YX projections:

.. code-block:: python

   registered, details = register_stack(
       image,
       registration_channel   = 0,
       registration_template_time_range = "all",
       method                 = "pystackreg",
       time_registration_mode = "projection",
       projection_method      = "max",
       zreg                   = False,
       max_xy_shifts          = (8, 8),
       zero_clip              = False,
       transform_backend      = "skimage",
       transform_order        = 1,
       filter_slices          = False,
       filter_projections     = False,
       median_kernel_size     = 3,
       return_shifts          = True,
       return_details         = True)


NoRMCorre
---------

NoRMCorre is available through the same main wrapper:

.. code-block:: python

   registered, details = register_stack(
       image,
       registration_channel   = 0,
       method                 = "normcorre",
       time_registration_mode = "projection",
       projection_method      = "max",
       zreg                   = False,
       max_xy_shifts          = (8, 8),
       zero_clip              = False,
       transform_backend      = "skimage",
       transform_order        = 1,
       nc_pw_rigid            = True,
       nc_strides             = (32, 32),
       nc_overlaps            = (16, 16),
       nc_max_deviation_rigid = 3,
       nc_template_init_mode  = "median",
       nc_template_update_method = "caiman",
       nc_n_jobs              = 4,
       return_shifts          = True,
       return_details         = True)

``registration_template_time_range`` is not used with ``method="normcorre"``.
NoRMCorre builds and updates its own template through ``nc_template_init_mode``
and ``nc_template_update_method``.

New options in this block:

.. list-table::
   :header-rows: 1
   :widths: 40 60
   
   * - Argument
     - Meaning
   * - ``nc_pw_rigid``
     - Enables piecewise-rigid patch correction instead of rigid-only
       NoRMCorre. Default: ``True``.
   * - ``nc_strides``
     - Patch-grid stride in YX. Default:  ``(48, 48)`` for 2D.
   * - ``nc_overlaps``
     - Patch overlap in YX. Effective patch size is
       ``nc_strides + nc_overlaps``. Default: ``(24, 24)`` for 2D.
   * - ``nc_max_deviation_rigid``
     - Maximum local patch deviation around the global rigid shift. This is
       different from ``max_xy_shifts``: ``max_xy_shifts`` limits the overall
       correction, while ``nc_max_deviation_rigid`` limits how far an
       individual patch may deviate from the global estimate. ``None``
       (default) means not limited.
   * - ``nc_n_jobs``
     - Worker count used by the NoRMCorre backend.  Default:  ``1``.
   * - ``nc_template_init_mode``
     - Initial NoRMCorre template strategy. ``"median"`` uses a sparse
       CaImAn-like sample across time; ``"registration_stack"`` uses one
       explicit reference frame. Default: ``"registration_stack"``.
   * - ``nc_template_update_method``
     - Template update strategy after each NoRMCorre pass. ``"caiman"`` uses
       chunk means followed by a median across chunks. Default: ``"caiman"``.

Before choosing ``nc_strides`` and ``nc_overlaps``, it is useful to draw the
patch layout. ZenReg provides a helper function for this:

.. code-block:: python

   from zenreg import plot_normcorre_patch_overlay

   plot_normcorre_patch_overlay(
       image,
       metadata,
       registration_channel = 0,
       registration_stack   = 0,
       nc_strides           = (32, 32),
       nc_overlaps          = (16, 16),
       projection_method    = "max",
       projection_range     = (0, 1))


.. figure:: _static/synthetic_2d_t_xy_normcorre_patch_overlay_t0_c0_max_z0-1.png
   :alt: Example NoRMCorre patch layout for synthetic 2D+t data.
   :align: center
   :figwidth: 100%

   Example NoRMCorre patch layout for synthetic 2D+t data. The background is a
   maximum-intensity Z projection of the first time point. The orange grid shows
   the patch layout with stride 32 and overlap 16. The effective patch size is
   48x48 pixels. 

.. tip::

   Choosing ``nc_strides`` and ``nc_overlaps`` is a model choice, not only a
   speed setting. The effective patch size is ``nc_strides + nc_overlaps``.
   A useful patch should contain enough stable image structure to estimate a
   local translation, for example several puncta, vessels, neuropil features,
   or cell bodies. A single isolated bright object per patch is often
   ambiguous, while empty or mostly noisy patches cannot constrain motion.

   As a practical starting point, use patches large enough to contain multiple
   informative structures, with overlaps of about one third to one half of the
   stride. Smaller strides provide denser local correction fields but increase
   runtime and can overfit noisy or weakly textured data. Larger strides behave
   more like a global rigid correction and are usually more stable when the
   signal is sparse. If motion is mostly global, start with rigid NoRMCorre or
   phase correlation; use piecewise NoRMCorre when different regions of the
   field of view visibly move differently.

.. tip::

   ``max_xy_shifts`` and ``nc_max_deviation_rigid`` constrain different parts
   of a NoRMCorre run. ``max_xy_shifts=(max_y, max_x)`` is a global safety
   limit for the final correction. ``nc_max_deviation_rigid`` applies only to
   piecewise-rigid NoRMCorre and limits how much each local patch may move
   relative to the global rigid shift. If this value is too small, local motion
   will be suppressed; if it is too large, noisy or weakly textured patches can
   jump to implausible local shifts.



Memory-efficient workflow
-------------------------

Memory mapping is a central ZenReg concept and works beyond 2D+t. See
:doc:`usage_memory_efficient` for OMIO cache behavior, cache cleanup and reuse,
server/local-cache strategies, and backend support.


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
       registration_channel   = 0,
       registration_stack     = 0,
       method                 = "phase_cross_correlation",
       time_registration_mode = "projection",
       projection_method      = "max",
       rotreg                 = True,
       max_rot_shifts         = 12,
       rotreg_iter            = 1,
       transform_backend      = "skimage",
       transform_order        = 1,
       return_shifts          = True,
       return_details         = True)

New options in this block:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Argument
     - Meaning
   * - ``method``
     - Backend used for the translational registration passes. Supported here:
       ``"phase_cross_correlation"`` and ``"pystackreg"``. Rotation estimation
       still uses internal polar-transform phase cross-correlation.
       ``"normcorre"`` currently does not support ``rotreg=True``.
   * - ``rotreg``
     - Enables in-plane XY/Z-axis rotation estimation from polar-transformed
       projections. Default:  ``False``.
   * - ``max_rot_shifts``
     - Optional absolute rotation limit in degrees. ``None`` (default) means no rotation clipping.
   * - ``rotreg_iter``
     - Number of translation/rotation refinement rounds. ``1`` runs
       translation, rotation, translation. Default:  ``1``.
   * - ``transform_backend``
     - Backend used to apply XY translations. Default:  ``"skimage"``. Alternative backend is ``"scipy"``.
   * - ``transform_order``
     - Interpolation order. ``1`` is good for intensity data; ``0`` preserves
       sparse puncta or labels. Default:  ``1``.

``method`` can be ``"phase_cross_correlation"`` or ``"pystackreg"`` in this
workflow. The selected method is used for the translational registration passes.
The rotation estimate itself is always computed internally from polar-transformed
projections using phase cross-correlation. ``method="normcorre"`` is currently
not supported together with ``rotreg=True``.
