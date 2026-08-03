Projection strategies
=====================

Many ZenReg workflows estimate motion from 2D projections, then apply the
detected correction to the full ``TZCYX`` stack. Projection-based registration
is often much faster than full-volume registration and can be very robust when
the dominant motion is lateral XY drift.

Projection method
-----------------

Use ``projection_method`` to choose how ZenReg collapses Z into a registration
image:

.. code-block:: python

   registered, details = register_stack(
       image,
       registration_channel   = 0,
       registration_stack     = 0,
       method                 = "phase_cross_correlation",
       time_registration_mode = "projection",
       projection_method      = "max",
       return_shifts          = True,
       return_details         = True)

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Method
     - When to use it
   * - ``"max"``
     - Good default for sparse spots, puncta, beads, nuclei, and other bright
       compact structures.
   * - ``"mean"``
     - Often better for dense, smooth, or spatially extended signals.
   * - ``"median"``
     - Robust to outliers, but can attenuate sparse spots.
   * - ``"std"`` / ``"var"``
     - Useful when contrast-rich structure matters more than absolute
       intensity.

Restricting the projection range
--------------------------------

Use ``registration_z_range`` to restrict registration to a half-open Z interval
``(z_start, z_stop)``:

.. code-block:: python

   registered, details = register_stack(
       image,
       registration_channel   = 0,
       registration_stack     = 0,
       method                 = "phase_cross_correlation",
       time_registration_mode = "projection",
       projection_method      = "max",
       registration_z_range   = (5, 25),
       return_shifts          = True,
       return_details         = True)

This is useful when only part of the stack contains stable registration signal,
or when top/bottom slices are noisy, empty, saturated, or outside the specimen.
``registration_z_range=None`` uses all available Z slices. Older examples may
use ``projection_range`` or ``zrange`` for the same setting; both names remain
accepted as compatibility aliases, but ``registration_z_range`` is the preferred
name.

Projection-based vs full 3D registration
----------------------------------------

Projection-based registration is the right first choice when motion is mostly
XY and the axial structure stays stable. It is computationally cheaper because
ZenReg estimates one 2D registration image per time point instead of registering
full ZYX volumes.

Use full 3D registration instead when:

- there is real Z drift that cannot be estimated reliably from orthogonal
  projections,
- the structure changes strongly with Z,
- the stack undergoes full 3D rotations,
- the projection hides important local structure.

Quick projection previews
-------------------------

Before choosing a projection strategy, inspect the registration image or
template that ZenReg would use for estimation:

.. code-block:: python

   from zenreg import show_projection, show_timepoints, z_project

   show_projection(
       image,
       title                            = "Template preview",
       registration_channel             = 0,
       registration_stack               = 0,
       registration_template_time_range = "all",
       registration_z_range             = (5, 25),
       projection_method                = "max",
       save_dir                         = "example_data/synthetic_data/registered/figures",
       return_projection                = False)

   projected_template = show_projection(
       image,
       title                            = "Returned template preview",
       registration_channel             = 0,
       registration_template_time_range = "all",
       registration_z_range             = (5, 25),
       projection_method                = "mean",
       return_projection                = True)

   show_timepoints(
       image,
       title             = "Projection preview",
       channel           = 0,
       reference_time    = 0,
       moving_time       = 25,
       projection_method = "max",
       projection_z_range = (5, 25),
       save_dir          = "example_data/synthetic_data/registered/figures")

   projected = z_project(
       image,
       zrange            = (5, 25),
       projection_method = "mean")

``show_projection`` is useful for previewing a single reference-frame
projection or a time-aggregated template. By default it only shows or saves the
preview figure. Set ``return_projection=True`` when you also want the projected
``YX`` image in Python. ``show_timepoints`` compares two projected time points
and their residual. ``z_project`` returns projected arrays for custom scripts.

Projection settings in reports
------------------------------

When ``return_details=True`` and ``registration_details`` are passed to
``save_stack``, ZenReg records the selected ``projection_method`` and
``registration_z_range`` in the YAML settings and annotates them in summary plots.
This makes projection choices visible for later quality control and
reproducibility.
