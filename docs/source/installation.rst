Installation
============

ZenReg is intended to be used in a scientific Python environment with OMIO and
the registration backends installed. The package has currently been tested with
Python 3.12. If you run into problems with newer Python versions, please open 
`GitHub issue <https://github.com/FabrizioMusacchio/ZenReg/issues>`_
with the Python version, operating system, and error message.

Create an environment
---------------------

We recommend creating a fresh conda environment first:

.. code-block:: bash

   conda create -n zenreg -y python=3.12
   conda activate zenreg

Install from PyPI
-----------------

ZenReg can be installed from PyPI:

.. code-block:: bash

   pip install zenreg

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

Verify installation
-------------------

After installation, verify that ZenReg imports correctly and reports its
version:

.. code-block:: bash

   python -c "import zenreg; print(f'ZenReg {zenreg.__version__} imported successfully; available CPUs: {zenreg.available_cpu_count()}')"

This one-liner should print the installed ZenReg version and the number of CPU
workers visible to Python. If it fails, check that the active terminal session
uses the environment in which ZenReg was installed.

Upgrade ZenReg
---------------

To upgrade ZenReg to the latest version, run:

.. code-block:: bash

   pip install --upgrade zenreg

or, if installed from a local checkout:

.. code-block:: bash

   git pull
   pip install --upgrade -e .

or

.. code-block:: bash

   pip install --upgrade git+https://github.com/FabrizioMusacchio/ZenReg.git

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
