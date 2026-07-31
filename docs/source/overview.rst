Overview
========

ZenReg is a registration platform for microscopy stacks. It focuses on
transparent, scriptable, fast and memory-efficient workflows that can be 
inspected frame by frame in interactive environments such as VS Code's 
interactive window, Jupyter, or
napari.

ZenReg philosophy
-----------------

ZenReg aims to make common microscopy registration tasks as simple as possible
without hiding the scientific settings that matter. The public workflow is
centered on one convenient wrapper, ``register_stack``, while the internal
implementation remains modular so additional registration methods can be added
over time.

.. image:: _static/ZenReg_logo_wide.jpg
   :alt: ZenReg logo
   :align: center
   :width: 100%

.. add some html empty space to get a clear line break:

.. raw:: html

   <br>

The project is intentionally contribution-friendly: If a backend or workflow is
missing, please feel free to request it through `GitHub issues <https://github.com/FabrizioMusacchio/ZenReg/issues>`_ 
or contribute a focused module that plugs into the same ``TZCYX`` data model 
(see below) and report-writing logic. Please refer to  
:doc:`contributing` for details.



Core ideas
----------

ZenReg separates the workflow into three explicit steps:

- ``load_stack`` reads microscopy data via `OMIO <https://github.com/FabrizioMusacchio/omio>`_
  which ensures consistent return of canonical ``TZCYX`` ordered image data plus metadata
  for a wide range of image formats.
- ``register_stack`` estimates and applies motion correction using a selected
  backend and registration mode.
- ``save_stack`` writes a registered OME-TIFF and optional report sidecars.

This keeps project scripts short while still making every important setting
visible and reproducible.


.. admonition:: OME-compliant ``TZCYX`` order

  .. code-block:: text

    T = time
    Z = z slices
    C = channels
    Y = image rows
    X = image columns

  `OMIO <https://github.com/FabrizioMusacchio/omio>`_ normalizes supported inputs 
  to this order, even when an input file has singleton or implicit dimensions. 
  This makes channel, time, and z handling consistent across TIFF, OME-TIFF, CZI,
  LSM, and Thorlabs RAW files.


Memory-efficient processing
---------------------------

ZenReg is designed for large microscopy files that may not fit comfortably into
RAM. Through `OMIO <https://github.com/FabrizioMusacchio/omio>`_, input images can be converted to disk-backed, chunked Zarr
caches. ZenReg can then read only the slices, projections, or volumes needed by
the current processing step instead of duplicating the full stack in memory.

This is especially useful when raw data live on a server or network-attached
storage. A local ``memmap_folder`` can cache the image once on fast local disk,
so repeated registration attempts, parameter tuning, or restarted Python
sessions reuse the local cache instead of repeatedly reading the full file over
the network. ZenReg exposes cache cleanup explicitly via
``cleanup_omio_cache`` and intentionally does not delete caches automatically,
so users can decide when reuse or cleanup is preferable.

See :doc:`usage_memory_efficient` for the full workflow and backend support.

Performance and parallel processing
-----------------------------------

ZenReg uses several measures to keep processing fast and scalable:

- projection-based registration for fast XY/ZYX estimates when full-volume
  registration is unnecessary,
- optional full-volume processing only for workflows that need it,
- disk-backed output caches for large registered results,
- CPU worker controls through ``n_jobs`` and backend-specific worker settings,
- parallel execution across independent time points or Z slices where possible,
- reusable OMIO caches to avoid repeated server reads or repeated cache builds.

The helper ``available_cpu_count()`` reports the number of CPU workers visible
to the current machine, workstation, or compute node. Passing ``n_jobs=-1`` uses
all available workers for ZenReg paths that can be parallelized.



Registration modes
------------------

ZenReg currently supports:

- 2D+t global translational registration on YX projections.
- 2D+t in-plane rotation correction.
- 3D+t translational registration on Z projections or full ZYX volumes.
- 3D and 3D+t intra-stack XY slice correction.
- NoRMCorre-style rigid and piecewise-rigid motion correction for 2D+t and
  3D+t.
- Full 3D rigid 6-DOF registration with dense SimpleITK or sparse point-based
  backends.
- Optional memory-efficient workflows with
  `OMIO <https://github.com/FabrizioMusacchio/omio>`_ disk-backed Zarr arrays.

Supported methods
-----------------

Currently, ZenReg supports the following registration methods and backends:

.. list-table::
   :header-rows: 1

   * - Method/backend
     - Main use
     - Notes
   * - ``phase_cross_correlation``
     - Fast translational registration in 2D, projection-based 3D+t, and full
       ZYX volumes.
     - Uses scikit-image phase cross-correlation.
   * - ``pystackreg``
     - StackReg-style 2D registration on projections.
     - Useful as a familiar alternative for projection-based workflows.
   * - ``normcorre``
     - NoRMCorre-style rigid and piecewise-rigid motion correction.
     - Implemented in ZenReg without requiring the full CaImAn suite.
   * - ``rigid_3d_backend="simpleitk"``
     - Dense full 3D 6-DOF rigid-volume registration.
     - Uses SimpleITK, supports physical Z/Y/X spacing and multiresolution
       optimization.
   * - ``rigid_3d_backend="points"``
     - Sparse puncta/spot-like full 3D rigid registration.
     - Uses detected 3D peaks, nearest-neighbor matching, and RANSAC/ICP-style
       refinement.

Output philosophy
-----------------

When ``registration_details`` are passed to ``save_stack``, ZenReg writes:

- the registered OME-TIFF,
- a CSV table with detected shifts, rotations, and correlations where
  available,
- a YAML settings file for reproducibility,
- a summary plot for quick quality control.

This ensures that ZenReg outputs are fully reproducible and shareable, e.g.,
for publication or collaboration. 

License
-------

ZenReg is distributed under the terms of the GNU General Public License v3.0
or later.

ZenReg is distributed under the terms of the 
`GNU General Public License v3.0 (GPL-3.0) <https://github.com/FabrizioMusacchio/ZenReg?tab=GPL-3.0-1-ov-file>`_.

In summary, users are permitted to

* **use** the software for any purpose  
* **modify** the source code and adapt it to their needs
* **redistribute** the original or modified code

Under the following conditions:

* **Copyleft** applies. Modifications must be released under the same GPL-3.0 license.  
* The **original copyright notice and license** must be preserved.

Not permitted:

* Use of ZenReg in **proprietary or closed-source** applications  
* Redistribution of modified versions under more restrictive terms  

ZenReg is provided **without any warranty**, including implied warranties of merchantability
or fitness for a particular purpose.

For full license terms, see the ``LICENSE`` file in the 
`repository <https://github.com/FabrizioMusacchio/ZenReg?tab=GPL-3.0-1-ov-file>`_ or  
`https://www.gnu.org/licenses/gpl-3.0.html <https://www.gnu.org/licenses/gpl-3.0.html>`_.



Citation
--------

If you use ZenReg in scientific work, please cite the ZenReg release archive
and the software repository.

Suggested citation:


  Musacchio, F. (2026). ZenReg: Fast and memory-efficient microscopy image
  registration for Python. https://doi.org/10.5281/zenodo.21727826

.. raw:: html

   <hr>

For questions, suggestions or bug reports, please refer to the
`GitHub issue tracker <https://github.com/FabrizioMusacchio/ZenReg/issues>`_ of 
the `ZenReg repository <https://github.com/FabrizioMusacchio/ZenReg>`_ or contact the maintainer 
directly:

| **Fabrizio Musacchio**: `Email <mailto:fabrizio.musacchio@dzne.de>`_ | `GitHub <https://github.com/FabrizioMusacchio>`_ | `Website <https://www.fabriziomusacchio.com>`_
