Multi-channel processing
========================

ZenReg treats the channel axis as part of the image that should be preserved,
not as an axis that is independently registered by default. Motion is estimated
from one selected ``registration_channel`` and the resulting correction is then
applied unchanged to every channel in the stack.

This is the usual microscopy workflow: choose the channel with the most stable
registration signal, for example a structural marker, a bright anatomical
channel, or a channel with sparse but reliable landmarks. Other channels are
carried along with exactly the same transform so that cross-channel spatial
relationships remain intact.

Canonical channel handling
--------------------------

All images loaded through OMIO are represented as ``T, Z, C, Y, X``. Even files
that look like 2D, 3D, or single-channel data still have an explicit channel
axis. This means that ``registration_channel=0`` always refers to the first
channel along the canonical ``C`` axis.

.. code-block:: python

   from pathlib import Path
   from zenreg import load_stack, register_stack, save_stack

   path = Path("example_data/synthetic_data/synthetic_2d_t_xy.ome.tif")
   image, metadata = load_stack(path, return_metadata=True)

   print(image.shape)  # T, Z, C, Y, X

   registered, details = register_stack(
       image,
       registration_channel   = 0,
       registration_stack     = 0,
       method                 = "phase_cross_correlation",
       time_registration_mode = "projection",
       projection_method      = "max",
       max_xy_shifts          = (8, 8),
       return_shifts          = True,
       return_details         = True)

   save_stack(
       "example_data/synthetic_data/registered/multichannel_registered.ome.tif",
       registered,
       metadata             = metadata,
       registration_details = details)

In this example, ZenReg estimates the motion from channel ``0``. If the input
has two or more channels, channels ``1``, ``2``, and so on are shifted, rotated,
or rigidly transformed using the same correction.

For single-channel images, ZenReg falls back to channel ``0`` if a higher
``registration_channel`` is requested accidentally. A runtime warning is shown,
and the returned details record both ``registration_channel_requested`` and
``registration_channel_used``. For true multi-channel stacks, invalid channel
indices still raise an error because silently switching channels could register
against the wrong biological signal.

Choosing a registration channel
-------------------------------

The best registration channel is not necessarily the brightest channel. Prefer
the channel that is most spatially stable and most informative for alignment:

- For sparse puncta, a channel with many reliable spots is usually better than
  a channel with only one or two very bright objects.
- For dense structural signals, a channel with broad anatomical texture can be
  more stable than a sparse activity channel.
- For calcium imaging, a structural or averaged signal is often better suited
  than a highly dynamic activity channel, because biological activity can look
  like apparent motion to an image-registration algorithm.
- For multi-color experiments, estimate motion from the channel that best
  represents sample or microscope motion, then apply that transform to all
  colors to preserve colocalization geometry.

Inspect the selected channel before registration, for example in napari, and
check the registered result for every channel afterwards.

2D+t and 3D+t behavior
----------------------

For 2D+t data, the detected YX correction is applied to every channel in each
time point. For projection-based 3D+t registration, ZenReg estimates motion
from a projection of the selected channel and applies the resulting correction
to every Z slice and every channel of the corresponding time point. When
``zreg=True`` or ``time_registration_mode="full_3d"`` is used, the detected ZYX
correction is likewise applied to all channels.

The same rule applies to rotation and full 3D rigid registration. Rotations or
6-DOF rigid transforms are estimated from ``registration_channel`` and then
applied to all channels. This keeps the channel geometry coherent and avoids
introducing artificial channel-to-channel offsets.

Options overview
----------------

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - Argument
     - Meaning
   * - ``registration_channel``
     - Channel used to estimate the transform. Default: ``0``.
   * - ``registration_stack``
     - Reference time point used to build or select the registration template.
       Default: ``0``.
   * - ``projection_method``
     - Projection statistic used when a 3D volume is reduced to a 2D
       registration image. The projection is computed only from the selected
       registration channel. Default: ``"max"``.
   * - ``filter_slices`` / ``filter_projections``
     - Optional preprocessing for shift estimation. Filtering is applied to the
       selected registration channel used for estimation, not as a general
       image-restoration step for all output channels.

When to run channels separately
-------------------------------

Register channels separately only when they truly contain independent motion,
for example when channels were acquired in separate passes with different
mechanical drift. In that case, run ZenReg separately for each channel or split
the image before registration. This is not the default because independent
per-channel registration can destroy real spatial relationships between
channels and can bias downstream colocalization or morphology analyses.
