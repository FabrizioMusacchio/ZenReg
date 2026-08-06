Memory-efficient workflows
==========================

ZenReg can use `OMIO <https://github.com/FabrizioMusacchio/omio>`_
disk-backed Zarr arrays for memory-efficient loading and intermediate output.
This is useful for 2D+t, 3D+t, intra-stack, NoRMCorre, and full 3D rigid
workflows. The example below uses a 2D+t time series for compactness, but the
same pattern applies to other ``register_stack`` calls.

Why use the OMIO cache?
-----------------------

The OMIO cache has two practical roles:

- It provides a disk-backed, chunked representation for memory-mapped
  processing. The cached Zarr array is chunked, so ZenReg can read only the
  slices or volumes needed by the current processing step instead of loading the
  full image into RAM.
- It can move a remote/server-hosted image into a local cache once, so repeated
  registration attempts read mostly from local storage instead of continuously
  hitting a network file system.

If ``memmap_folder=None``, OMIO creates its cache relative to the input image.
That is convenient for local data, but can be undesirable when the input image
lives on a server. For server or NAS data, prefer a local scratch folder such as
``/local_scratch/omio_cache`` or a project-local cache directory on a fast local
disk. A central local cache folder also makes it easier to monitor disk usage
and clean up old cached data.

Fresh cache start
-----------------

If you want to make sure a run starts from a clean cache, clear the cache folder
before loading:

.. code-block:: python

   from zenreg import cleanup_omio_cache

   cache_dir = "local_omio_cache"
   cleanup_omio_cache(cache_dir, full_cleanup=True)

This is useful after changing input files, debugging suspicious cache state, or
before a benchmark where you want the first read to include cache creation time.
ZenReg intentionally does not delete caches automatically, because the
registered image may still live in a disk-backed output cache until
``save_stack`` has finished.

Reusing an existing cache
-------------------------

For interactive work, keeping the cache is often exactly what you want. If a
Python session crashes, a registration fails, or you simply want to try
different registration settings, load again with ``memmap_reuse=True``:

.. code-block:: python

   from zenreg import load_stack

   image, metadata = load_stack(
       "large_timeseries.ome.tif",
       return_metadata = True,
       use_memmap      = True,
       memmap_folder   = cache_dir,
       memmap_reuse    = True)

OMIO validates and reuses the existing disk cache when possible. This avoids
rebuilding the Zarr cache and, for server-hosted images, avoids copying the
large source file over the network again.

Full memory-mapped registration workflow
----------------------------------------

The complete pattern is:

.. code-block:: python

   from zenreg import cleanup_omio_cache, load_stack, register_stack, save_stack

   cache_dir = "local_omio_cache"
   cleanup_omio_cache(cache_dir, full_cleanup=True)  # optional fresh start

   image, metadata = load_stack(
       "large_timeseries.ome.tif",
       return_metadata = True,
       use_memmap      = True,
       memmap_folder   = cache_dir,
       memmap_reuse    = True)

   registered, details = register_stack(
       image,
       registration_channel   = 0,
       method                 = "phase_cross_correlation",
       time_registration_mode = "projection",
       output_use_memmap      = True,
       output_memmap_folder   = cache_dir,
       n_jobs                 = 8,
       return_shifts          = True,
       return_details         = True)

   save_stack("large_timeseries_registered.ome.tif",
              registered,
              metadata             = metadata,
              registration_details = details)

   cleanup_omio_cache(cache_dir, full_cleanup=True)

The final cleanup is optional. Keep the cache if you expect to run another
registration on the same input soon; clean it when you are done or when local
scratch space is limited.

Memory-efficient napari inspection
----------------------------------

For large disk-backed results, opening the registered stack in napari should
also avoid materializing the full array in RAM. ZenReg's ``open_in_napari``
helper forwards additional keyword arguments to OMIO's napari bridge, so you
can choose OMIO's Zarr opening mode explicitly:

.. code-block:: python

   from zenreg import open_in_napari

   open_in_napari(
       registered,
       metadata,
       fname      = "large_timeseries_registered",
       enabled    = True,
       zarr_mode  = "zarr_nodask")

Use this after a memory-mapped ``register_stack`` call when ``registered`` is a
disk-backed Zarr array. The exact available ``zarr_mode`` values are provided
by OMIO, but ``"zarr_nodask"`` is useful when you want napari to read directly
from the Zarr-backed array without an additional Dask layer. This keeps the
inspection workflow aligned with the registration workflow: data are pulled
from disk in chunks as napari needs them, rather than copied into a dense NumPy
array before display.

For scripts that may run on servers or headless machines, keep an explicit
switch:

.. code-block:: python

   OPEN_IN_NAPARI = False

   open_in_napari(
       registered,
       metadata,
       fname      = "large_timeseries_registered",
       enabled    = OPEN_IN_NAPARI,
       zarr_mode  = "zarr_nodask")

Options used here:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Argument
     - Meaning
   * - ``use_memmap``
     - Ask OMIO to read through a disk-backed Zarr cache. Default:
       ``False``.
   * - ``memmap_folder``
     - Location for the OMIO disk cache. Use local scratch storage for remote
       input data. ``None`` (default) means OMIO chooses its default cache
       location.
   * - ``memmap_reuse``
     - Reuse a validated existing OMIO cache instead of rebuilding it. Default:
       ``True``.
   * - ``output_use_memmap``
     - Store ZenReg's intermediate registered result in disk-backed Zarr.
       Default: ``False``.
   * - ``output_memmap_folder``
     - Folder for ZenReg's disk-backed output cache. ``None`` (default) means
       ZenReg chooses its default cache location.

Backend support
---------------

.. list-table::
   :header-rows: 1
   :widths: 35 20 45

   * - Workflow/backend
     - Memmap support
     - Notes
   * - ``load_stack(..., use_memmap=True)``
     - Yes
     - Uses OMIO's disk-backed Zarr cache and optional
       ``memmap_reuse=True``.
   * - Standard ``phase_cross_correlation`` workflows
     - Yes
     - Supports disk-backed registered output with ``output_use_memmap=True``.
   * - ``pystackreg`` workflows
     - Yes
     - Uses the same standard output memmap path for projection-based
       registration.
   * - ``method="normcorre"``
     - Yes
     - ``output_use_memmap=True`` is forwarded to the NoRMCorre output cache;
       NoRMCorre-specific names are also available as ``nc_output_*`` options.
   * - Full 3D rigid ``rigid_3d_backend="simpleitk"`` / ``"points"``
     - Yes, with care
     - Registered output can be disk-backed. Transform estimation still works
       on one full ZYX volume per time point, so memory use depends on volume
       size.
   * - ``save_stack``
     - Yes
     - OMIO writes from disk-backed arrays without requiring the full registered
       stack to be duplicated in memory.
