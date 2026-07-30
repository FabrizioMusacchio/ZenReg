Installation
============

ZenReg is intended to be used in a scientific Python environment with OMIO and
the registration backends installed.

Development installation
------------------------

From a local checkout:

.. code-block:: bash

   git clone https://github.com/FabrizioMusacchio/ZenReg.git
   cd ZenReg
   pip install -e .

For development and documentation work:

.. code-block:: bash

   pip install -e ".[dev,docs]"

Core dependencies
-----------------

ZenReg depends on:

- ``omio-microscopy`` for microscopy I/O and metadata handling,
- ``scikit-image`` and ``scipy`` for phase-correlation and transformations,
- ``pystackreg`` for StackReg-style 2D registration,
- ``SimpleITK`` for full 3D dense rigid registration,
- ``matplotlib`` for report and tutorial plots.

Optional CaImAn comparison
--------------------------

ZenReg's NoRMCorre implementation does not require CaImAn. Some tutorial
scripts contain optional, commented CaImAn comparison cells. To run those
comparison cells, install CaImAn separately, for example:

.. code-block:: bash

   mamba install -y caiman

Then uncomment the CaImAn imports and example cells in the relevant user
script.
