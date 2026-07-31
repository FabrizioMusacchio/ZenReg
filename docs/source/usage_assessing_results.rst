Assessing the results
=====================

Registration should always be checked before downstream analysis. ZenReg writes
registered images together with machine-readable sidecars and quick-look plots
so that visual inspection, quantitative quality control, and reproducibility
stay connected.

The minimal pattern is:

.. code-block:: python

   from pathlib import Path
   from zenreg import load_stack, register_stack, save_stack

   image, metadata = load_stack(
       Path("example_data/synthetic_data/synthetic_2d_t_xy.ome.tif"),
       return_metadata = True)

   registered, details = register_stack(
       image,
       registration_channel = 0,
       registration_stack   = 0,
       method               = "phase_cross_correlation",
       return_details       = True)

   written_path = save_stack(
       "example_data/synthetic_data/registered/synthetic_2d_t_xy_registered.ome.tif",
       registered,
       metadata             = metadata,
       registration_details = details)

Passing ``registration_details`` to ``save_stack`` activates ZenReg's report
writer. For an output called ``image_registered.ome.tif``, ZenReg writes:

.. list-table::
   :header-rows: 1

   * - Output
     - Purpose
   * - ``image_registered.ome.tif``
     - The registered image, written through OMIO with canonical ``TZCYX`` axes
       and updated metadata.
   * - ``image_registered_registration_shifts.csv``
     - Frame-wise shifts, optional intra-stack shifts, optional rotations, and
       Pearson correlations before and after registration.
   * - ``image_registered_registration_settings.yaml``
     - Reproducibility metadata: selected methods, backends, projection
       settings, limits, zero-clipping settings, and registered output shape.
   * - ``image_registered_registration_summary.png``
     - A compact summary plot for fast quality control.

Visual before/after checks
--------------------------

Use ``show_before_after`` for a quick residual plot. The raw difference image
should usually become flatter after registration, but it does not need to become
perfectly zero when the sample changes over time, photobleaches, has biological
activity, or contains noise.

.. code-block:: python

   from zenreg import show_before_after

   show_before_after(
       image,
       registered,
       title             = "synthetic 2D+t XY before/after registration",
       channel           = 0,
       reference_time    = 0,
       moving_time       = 1,
       projection_method = "max",
       save_dir          = "example_data/synthetic_data/registered/figures")

.. figure:: _static/synthetic_2d_t_xy_before_after_registration_c0_t0_t1_max.png
   :alt: Example result of a 2D+t registration using ZenReg's phase_cross_correlation backend
   :align: center
   :figwidth: 100%

   Example result of a 2D+t registration using ZenReg's
   ``phase_cross_correlation`` backend. Top panels show from left to right the
   first and the second time point of the raw data, along with the difference
   image. Bottom panels show the same for the registered data.

Summary plots
-------------

The registration summary plot is meant as a first quantitative health check.
The upper panel shows detected correction shifts as a function of frame. If
``max_xy_shifts`` or ``max_z_shifts`` were set, dashed limit lines are drawn so
that clipped estimates are easy to spot. When rotation registration is enabled,
rotation estimates are shown on a secondary axis as ``rotation_z``,
``rotation_y``, and/or ``rotation_x`` depending on the selected backend.

The lower panel shows Pearson correlation between the registration template and
each frame. When available, ZenReg plots both the pre-registration and
post-registration correlations. A useful registration usually increases or
stabilizes these values, but the plot should be interpreted together with the
images: high correlation can still hide local errors, projection artefacts, or
unwanted cropping.

.. figure:: _static/synthetic_2d_t_xy_registered_registration_summary.png
   :alt: ZenReg registration summary plot with detected shifts and correlations
   :align: center
   :figwidth: 100%

   Example ZenReg summary plot. The plot records detected shifts, correlation
   before and after registration, and the most important settings used to
   create the registered image.

CSV reports
-----------

The shift CSV is useful for plotting, debugging, and benchmarking. It contains
one ``scope="time"`` row per time point and, when intra-stack correction was
used, additional ``scope="intra_stack"`` rows per time point and Z slice.

Important columns are:

.. list-table::
   :header-rows: 1

   * - Column
     - Meaning
   * - ``frame`` and ``z``
     - Time point and, for intra-stack rows, Z slice.
   * - ``shift_z``, ``shift_y``, ``shift_x``
     - Detected time-registration correction shifts in pixels.
   * - ``intra_shift_y``, ``intra_shift_x``
     - Detected within-stack slice correction shifts.
   * - ``rotation_z_deg``, ``rotation_y_deg``, ``rotation_x_deg``
     - Detected correction rotations in degrees when rotation registration was
       enabled.
   * - ``pearson_correlation_before``
     - Template-vs-frame Pearson correlation before applying the detected
       registration, when available.
   * - ``pearson_correlation_after``
     - Template-vs-frame Pearson correlation after registration.

Load it like any other CSV table:

.. code-block:: python

   import pandas as pd

   shifts = pd.read_csv(
       "example_data/synthetic_data/registered/"
       "synthetic_2d_t_xy_registered_registration_shifts.csv")

   time_rows = shifts[shifts["scope"] == "time"]
   print(time_rows[["frame", "shift_y", "shift_x", "pearson_correlation_after"]])

YAML settings
-------------

The settings YAML is the reproducibility record for the registration run. It
contains the registered shape, canonical axes, summary correlation statistics,
paths to the report files, and the registration settings that affected the
result.

Use it to answer questions such as:

- Which channel and reference time point were used?
- Was registration projection-based or full 3D?
- Which projection method and projection range created the template?
- Were Z shifts, rotations, zero clipping, or maximum shift limits enabled?
- Which transform backend and interpolation order were used?

Practical interpretation
------------------------

For most datasets, use several checks together:

- Inspect raw and registered data in napari.
- Check the before/after residual plots for obvious remaining motion.
- Check whether shifts and rotations stay within biologically and technically
  plausible ranges.
- Compare correlation before and after registration.
- Inspect zero borders before enabling aggressive ``zero_clip`` settings.
- For synthetic examples, compare detected shifts and rotations against the
  ground-truth CSV tables.

Small residuals do not guarantee perfect registration, and perfect-looking
summary curves do not replace visual inspection. The goal is agreement between
the image, the detected motion parameters, and the correlation trend.
