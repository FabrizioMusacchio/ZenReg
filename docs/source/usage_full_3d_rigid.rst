Full 3D rigid registration
===========================

Full 3D rigid registration estimates a 6-DOF transform for each time point:

.. code-block:: text

   translation: z, y, x
   rotation:    rotation_z, rotation_y, rotation_x

The transform is estimated on one ``registration_channel`` and then applied to
all channels.

Dense SimpleITK backend
-----------------------

Use ``rigid_3d_backend="simpleitk"`` for dense structural volumes:

.. code-block:: python

   from pathlib import Path
   from zenreg import load_stack, register_stack, save_stack

   path = Path("example_data/synthetic_data/synthetic_3d_t_rigid_simpleitk.ome.tif")
   image, metadata = load_stack(path, return_metadata=True)

   registered, details = register_stack(
       image,
       metadata                 = metadata,
       registration_channel     = 0,
       registration_stack       = 0,
       method                   = "phase_cross_correlation",
       time_registration_mode   = "full_3d",
       zreg                     = True,
       rotreg                   = True,
       rigid_3d_backend         = "simpleitk",
       rot_init_iterations      = 2,
       rot_metric               = "correlation",
       rot_shrink_factors       = (4, 2, 1),
       rot_smoothing_sigmas     = (2.0, 1.0, 0.0),
       rot_iterations           = 100,
       rot_n_jobs               = 2,
       zero_clip                = True,
       zero_clip_mode           = "auto",
       zero_clip_mask_strategy  = "relaxed",
       return_shifts            = True,
       return_details           = True)

When ``metadata`` is passed and ``rot_spacing_zyx`` is left as ``None``, ZenReg
uses OMIO's ``PhysicalSizeZ``, ``PhysicalSizeY``, and ``PhysicalSizeX`` values
as physical spacing in Z/Y/X order. This matters for anisotropic data because
rotations in anisotropic Z stacks are otherwise geometrically mis-scaled.
Pass ``rot_spacing_zyx=(z, y, x)`` explicitly only when you want to override the
metadata-derived spacing.

Options introduced here:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Argument
     - Meaning
   * - ``metadata``
     - OMIO metadata from ``load_stack``. Used to infer physical
       ``PhysicalSizeZ/Y/X`` spacing for full 3D rigid registration. Default: ``None``.
   * - ``rotreg=True``
     - Enable rotation correction. Default:  ``False``.
   * - ``rigid_3d_backend="simpleitk"``
     - Use intensity-based SimpleITK full 3D rigid registration. Default:  ``"phase_cross_correlation"`` for the simpler projection path.
   * - ``rot_spacing_zyx``
     - Optional physical voxel spacing override in Z, Y, X order. Default:  ``None`` (default) uses OMIO metadata spacing when ``metadata`` is passed, or unit
       spacing otherwise.
   * - ``rot_init_iterations``
     - Number of projection-based initialization passes before SimpleITK
       refinement. Default:  ``1``.
   * - ``rot_metric``
     - SimpleITK similarity metric. ``"correlation"`` is recommended for
       same-modality microscopy stacks. Default:  ``"correlation"``.
   * - ``rot_shrink_factors`` / ``rot_smoothing_sigmas``
     - Multi-resolution pyramid settings for coarse-to-fine optimization. Uses backend defaults.
   * - ``rot_iterations``
     - Maximum SimpleITK optimizer iterations. Uses backend default.
   * - ``zero_clip_mask_strategy="relaxed"``
     - Use a less aggressive valid-region crop after rotations. Default:  ``"strict"``.

Sparse points backend
---------------------

Use ``rigid_3d_backend="points"`` for sparse puncta or spot-like data:

.. code-block:: python

   path = Path("example_data/synthetic_data/synthetic_3d_t_rigid_points.ome.tif")
   image, metadata = load_stack(path, return_metadata=True)

   registered, details = register_stack(
       image,
       metadata=metadata,
       registration_channel          = 0,
       registration_stack            = 0,
       method                        = "phase_cross_correlation",
       time_registration_mode        = "full_3d",
       zreg                          = True,
       rotreg                        = True,
       rigid_3d_backend              = "points",
       rot_points_max_points         = 120,
       rot_points_min_distance       = 2,
       rot_points_threshold_rel      = 0.2,
       rot_points_iterations         = 40,
       rot_points_max_match_distance = 7.0,
       transform_order               = 0,
       rot_n_jobs                    = 2,
       zero_clip                     = True,
       zero_clip_mask_strategy       = "relaxed",
       return_shifts                 = True,
       return_details                = True)

Sparse puncta often benefit from ``transform_order=0`` because nearest-neighbor
resampling keeps spots sharp. Dense intensity data usually benefits from
``transform_order=1``.

New options in this block:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Argument
     - Meaning
   * - ``rigid_3d_backend="points"``
     - Use peak detection plus point matching/RANSAC-style rigid estimation. Default:  ``"phase_cross_correlation"``.
   * - ``rot_points_max_points``
     - Maximum number of detected points used per volume. Uses backend default.
   * - ``rot_points_min_distance``
     - Minimum distance between detected local maxima. Uses backend default.
   * - ``rot_points_threshold_rel``
     - Relative intensity threshold for peak detection. Uses backend default.
   * - ``rot_points_iterations``
     - Maximum point-matching refinement iterations. Uses backend default.
   * - ``rot_points_max_match_distance``
     - Maximum nearest-neighbor distance accepted as a point match. Uses backend default.
   * - ``transform_order=0``
     - Nearest-neighbor interpolation during transform application. Default:  ``1`` for linear interpolation.

SimpleITK on sparse puncta
--------------------------

The sparse puncta synthetic dataset can of course also be run with SimpleITK 
for comparison. This is useful for understanding whether intensity-based or
point-based rigid registration is better for a specific acquisition:

.. code-block:: python

   registered, details = register_stack(
       image,
       metadata=metadata,
       registration_channel   = 0,
       registration_stack     = 0,
       method                 = "phase_cross_correlation",
       time_registration_mode = "full_3d",
       zreg                   = True,
       rotreg                 = True,
       rigid_3d_backend       = "simpleitk",
       rot_metric             = "correlation",
       transform_order        = 0,
       zero_clip              = False,
       return_shifts          = True,
       return_details         = True)

Post hoc cropping
-----------------

For full 3D rotations, automatic zero clipping can be conservative. A practical
workflow is to first run with ``zero_clip=False``, inspect the result, and then
crop manually:

.. code-block:: python

   from zenreg import crop_stack

   cropped, cropped_metadata = crop_stack(
       registered,
       metadata,
       crop={"top":    1, 
             "bottom": 1, 
             "left":   4, 
             "right":  4, 
             "up":     3, 
             "down":   3})
