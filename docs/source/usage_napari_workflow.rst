Napari workflow
===============

ZenReg is designed to work well in interactive sessions. For microscopy data,
it is often useful to inspect raw and registered stacks directly in napari while
tuning registration settings.

Open raw and registered stacks
------------------------------

The tutorial helper ``maybe_open_in_napari`` opens a stack only when
``open_in_napari=True``. This keeps scripts safe for headless runs while making
interactive inspection one switch away.

.. code-block:: python

   from pathlib import Path
   from zenreg import load_stack, maybe_open_in_napari, register_stack, save_stack

   OPEN_IN_NAPARI = True

   path = Path("example_data/synthetic_data/synthetic_3d_t_zyx.ome.tif")
   image, metadata = load_stack(path, return_metadata=True)

   maybe_open_in_napari(
       image,
       metadata,
       fname          = "raw 3D+t stack",
       open_in_napari = OPEN_IN_NAPARI)

   registered, details = register_stack(
       image,
       registration_channel   = 0,
       registration_stack     = 0,
       method                 = "phase_cross_correlation",
       time_registration_mode = "full_3d",
       zreg                   = True,
       return_shifts          = True,
       return_details         = True)

   maybe_open_in_napari(
       registered,
       metadata,
       fname          = "registered 3D+t stack",
       open_in_napari = OPEN_IN_NAPARI)

   save_stack(
       "example_data/synthetic_data/registered/napari_checked_registered.ome.tif",
       registered,
       metadata             = metadata,
       registration_details = details)

Metadata matters here: OMIO provides canonical ``TZCYX`` image data and a
matching metadata dictionary. Passing both to napari preserves the intended
axis interpretation and physical metadata whenever OMIO can provide it.

Recommended inspection pattern
------------------------------

A practical interactive loop is:

1. Load with ``load_stack(..., return_metadata=True)``.
2. Open the raw stack in napari.
3. Run ``register_stack`` with an initial conservative setting.
4. Open the registered stack in the same session.
5. Compare raw and registered layers by toggling visibility, blending, and
   stepping through time/Z.
6. Save only once the result looks plausible and the summary plot/CSV agrees.

This is especially useful for:

- choosing ``projection_method`` and ``projection_range``,
- deciding whether ``zreg=True`` is needed,
- inspecting zero borders before enabling ``zero_clip`` or manual
  ``crop_stack``,
- checking whether full 3D rigid registration preserves the expected field of
  view,
- comparing raw, phase-cross-correlation, NoRMCorre, and rigid-3D outputs.

Headless and RTD-safe scripts
-----------------------------

Keep ``OPEN_IN_NAPARI = False`` in scripts that should run on servers,
continuous integration, Read the Docs, or HPC nodes:

.. code-block:: python

   OPEN_IN_NAPARI = False

   maybe_open_in_napari(
       registered,
       metadata,
       fname          = "registered stack",
       open_in_napari = OPEN_IN_NAPARI)

The function simply returns without opening napari when the switch is false.

Quick figures without napari
----------------------------

For documentation figures or lightweight checks, use the tutorial plotting
helpers instead of opening napari:

.. code-block:: python

   from zenreg import show_before_after, show_slices, show_timepoints

   show_timepoints(
       image,
       title             = "raw timepoints",
       channel           = 0,
       projection_method = "max")

   show_before_after(
       image,
       registered,
       title             = "before/after registration",
       channel           = 0,
       save_dir          = "example_data/synthetic_data/registered/figures")

   show_slices(
       registered,
       title   = "registered slices",
       channel = 0,
       z0      = 0,
       z1      = 6)

``show_before_after`` can save PNGs directly, which makes it convenient for
building documentation figures from the same tutorial scripts used for testing.
