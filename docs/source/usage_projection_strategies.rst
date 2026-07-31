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

Use ``projection_range`` to restrict registration to a half-open Z interval
``(z_start, z_stop)``:

.. code-block:: python

   registered, details = register_stack(
       image,
       registration_channel   = 0,
       registration_stack     = 0,
       method                 = "phase_cross_correlation",
       time_registration_mode = "projection",
       projection_method      = "max",
       projection_range       = (5, 25),
       return_shifts          = True,
       return_details         = True)

This is useful when only part of the stack contains stable registration signal,
or when top/bottom slices are noisy, empty, saturated, or outside the specimen.
``projection_range=None`` uses all available Z slices.

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

Before choosing a projection strategy, inspect a few projected time points:

.. code-block:: python

   from zenreg import show_timepoints, z_project

   show_timepoints(
       image,
       title             = "Projection preview",
       channel           = 0,
       projection_method = "max",
       save_dir          = "example_data/synthetic_data/registered/figures")

   projected = z_project(
       image,
       zrange            = (5, 25),
       projection_method = "mean")

``show_timepoints`` is a tutorial helper for quick visual inspection and can
save the preview figure when ``save_dir`` or ``save_path`` is provided.
``z_project`` returns the projected array and can be used in custom scripts.

Projection settings in reports
------------------------------

When ``return_details=True`` and ``registration_details`` are passed to
``save_stack``, ZenReg records the selected ``projection_method`` and
``projection_range`` in the YAML settings and annotates them in summary plots.
This makes projection choices visible for later quality control and
reproducibility.
