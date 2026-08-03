Contributing and community guidelines
=====================================

ZenReg is an open source project for modular, memory-efficient microscopy image
registration. Contributions range from bug reports and documentation
improvements to new registration backends, synthetic benchmark datasets,
workflow helpers, and improvements to reporting or memory-mapped processing.

The goal of ZenReg is to make microscopy registration workflows explicit and
reproducible. Contributions are therefore evaluated not only by functionality,
but also by clarity, testability, memory behavior, and long-term
maintainability.

How to contribute
-----------------

If you are interested in contributing to ZenReg, the recommended entry points
are:

* reporting bugs or unexpected behavior
* suggesting improvements to the documentation or tutorials
* requesting support for additional registration methods
* improving synthetic benchmark datasets
* submitting pull requests with code changes

Bug reports and feature requests should be submitted via the
`GitHub issue tracker <https://github.com/FabrizioMusacchio/ZenReg/issues>`_.
For code changes and larger contributions, please open a pull request against
the main repository.

Contribution guide
------------------

The repository contains a dedicated contribution guide in
`CONTRIBUTING.md <https://github.com/FabrizioMusacchio/ZenReg?tab=contributing-ov-file>`_.
It describes in more detail:

* how to set up a local development environment
* the preferred workflow for branching and pull requests
* conventions for commit messages and code style
* expectations regarding tests and documentation
* ZenReg's canonical ``TZCYX`` axis contract
* the expected interface for adding new registration backends

Before opening a pull request, please make sure that:

* the code is formatted consistently with the existing code base
* existing tests pass locally, and new functionality is covered by tests where applicable
* public functions and modules are documented via docstrings
* user-facing changes are reflected in the documentation pages
* changes that affect registration outputs also update returned details and reports

Development setup
-----------------

Install ZenReg in editable mode:

.. code-block:: bash

   git clone https://github.com/FabrizioMusacchio/ZenReg.git
   cd ZenReg

   conda create -n zenreg-dev -c conda-forge python=3.12
   conda activate zenreg-dev

   pip install -e ".[dev,docs]"

Run tests with:

.. code-block:: bash

   pytest

Build the documentation locally with:

.. code-block:: bash

   sphinx-build -b html docs/source docs/build/html

Registration backend interface
------------------------------

ZenReg is designed to grow as a registration platform. New registration methods
should be implemented as focused backend modules and exposed through
``register_stack`` when they are ready for user-facing use.

Core input contract:

* backends receive image data in canonical ``TZCYX`` order
* missing dimensions are represented with length ``1``
* motion is estimated from one ``registration_channel``
* the estimated transform is applied to all channels
* disk-backed OMIO/Zarr arrays should not be unnecessarily materialized

Core output contract:

* the registered image remains in canonical ``TZCYX`` order
* the output shape should match the input shape unless cropping is explicitly requested
* returned ``details`` should use ZenReg-compatible keys such as
  ``time_shifts_zyx``, ``time_shifts_yx``, ``rotation_shifts_deg``,
  ``rotation_shifts_zyx_deg``, ``pearson_correlations``,
  ``registration_channel``, ``method``, ``registration_z_range``, and
  ``zero_clip_bounds`` where applicable
* shift arrays should store correction shifts, not the originally applied motion
* unsupported quantities should be omitted or set to ``None`` rather than
  encoded under incompatible names

When a new backend is exposed through ``register_stack``, it should preserve the
main ZenReg workflow:

.. code-block:: python

   image, metadata = load_stack(path, return_metadata=True)
   registered, details = register_stack(
       image,
       registration_channel=0,
       method="your_backend",
       return_details=True)
   save_stack(output_path, registered, metadata=metadata, registration_details=details)

Backend-specific options should use clear prefixes when exposed through
``register_stack``. Examples in the current API include ``nc_...`` for
NoRMCorre-like settings and ``rot_...`` for rotation-specific settings.

Memory-aware contributions
--------------------------

ZenReg is intended to work with large microscopy stacks. Contributions should
avoid loading a full ``TZCYX`` stack into memory unless the algorithm truly
requires it. Prefer time-point-wise, slice-wise, or chunk-wise processing when
possible.

If a backend must materialize data in memory, document the scope explicitly, for
example "one time point as a ``ZYX`` volume" rather than "the full stack".

Reporting issues
----------------

When reporting a registration issue, please include:

* ZenReg version
* Python version and operating system
* input shape in ``TZCYX`` order
* registration settings
* backend and version information
* whether OMIO memmap/Zarr loading was used
* a small crop or synthetic reproducer if possible
* the relevant registration sidecars generated by ``save_stack``

Requests for new registration methods
-------------------------------------

Requests for new registration methods are welcome. Please include:

* the microscopy use case
* expected input dimensionality, for example 2D+t, 3D+t, or intra-stack 3D
* the transform model, for example translation, rotation, rigid, non-rigid,
  optical flow, or point-cloud registration
* relevant papers, implementations, or existing tools
* representative synthetic or cropped example data if available

Large raw datasets should not be committed to the repository. Please use
temporary links, public archives, or minimal reproducible examples instead.

Code of conduct
---------------

All interactions in the ZenReg project are governed by a
`Code of Conduct <https://github.com/FabrizioMusacchio/ZenReg?tab=coc-ov-file>`_
based on the `Contributor Covenant <https://www.contributor-covenant.org>`_.
By participating in the project, you agree to abide by these guidelines.

If you experience or observe behavior that violates the Code of Conduct, please
report it via email to the maintainer.

Where to start
--------------

If you are looking for a first contribution, the issue tracker may contain
issues suitable for documentation improvements, focused bug fixes, tests, or
small registration-workflow refinements.

You are also welcome to open an issue to discuss ideas for new features or
registration backends before starting an implementation.
