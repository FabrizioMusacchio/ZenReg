ZenReg documentation
====================

.. image:: _static/ZenReg_logo_wide.jpg
   :alt: ZenReg logo
   :align: center
   :width: 100%

|

.. image:: https://badgen.net/badge/icon/GitHub%20repository?icon=github&label
   :target: https://github.com/FabrizioMusacchio/ZenReg/
   :alt: GitHub repository

.. image:: https://img.shields.io/github/v/release/FabrizioMusacchio/ZenReg
   :alt: GitHub Release

.. image:: https://img.shields.io/pypi/v/zenreg.svg
   :target: https://pypi.org/project/zenreg/
   :alt: PyPI version

.. image:: https://img.shields.io/badge/License-GPL%20v3-green.svg
   :target: https://github.com/FabrizioMusacchio/ZenReg
   :alt: GPLv3 License

.. image:: https://github.com/FabrizioMusacchio/ZenReg/actions/workflows/zenreg_tests.yml/badge.svg
   :alt: Tests

.. image:: https://codecov.io/gh/FabrizioMusacchio/ZenReg/graph/badge.svg?token=OYTRL4WO0U 
   :target: https://codecov.io/gh/FabrizioMusacchio/ZenReg
   :alt: Codecov

.. image:: https://img.shields.io/github/last-commit/FabrizioMusacchio/ZenReg
   :target: https://github.com/FabrizioMusacchio/ZenReg/commits/main/
   :alt: GitHub last commit

.. image:: https://img.shields.io/github/issues/FabrizioMusacchio/ZenReg
   :target: https://github.com/FabrizioMusacchio/ZenReg/issues
   :alt: GitHub Issues Open

.. image:: https://img.shields.io/github/issues-closed/FabrizioMusacchio/ZenReg?color=53c92e
   :target: https://github.com/FabrizioMusacchio/ZenReg/issues?q=is%3Aissue%20state%3Aclosed
   :alt: GitHub Issues Closed

.. image:: https://img.shields.io/github/issues-pr/FabrizioMusacchio/ZenReg
   :target: https://github.com/FabrizioMusacchio/ZenReg/pulls
   :alt: GitHub Issues or Pull Requests

.. image:: https://readthedocs.org/projects/zenreg/badge/?version=latest
   :target: https://zenreg.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. image:: https://img.shields.io/github/languages/code-size/fabriziomusacchio/ZenReg
   :alt: GitHub code size in bytes

.. image:: https://img.shields.io/pypi/dm/zenreg?logo=pypy&label=PiPY%20downloads&color=blue
   :target: https://pypistats.org/packages/zenreg
   :alt: PyPI Downloads

.. image:: https://static.pepy.tech/personalized-badge/zenreg?period=total&units=INTERNATIONAL_SYSTEM&left_color=GRAY&right_color=BLUE&left_text=PiPY+total+downloads
   :target: https://pepy.tech/projects/zenreg
   :alt: PyPI Total Downloads

.. image:: https://img.shields.io/badge/Zenodo%20Archive-10.5281%2Fzenodo.21727826-blue
   :target: https://doi.org/10.5281/zenodo.21727826
   :alt: Zenodo Archive

.. image:: https://img.shields.io/badge/bioRxiv-10.64898%2F2026.08.07.743572-red
   :target: https://doi.org/10.64898/2026.08.07.743572
   :alt: bioRxiv preprint


ZenReg is a Python package for fast, memory-efficient, and modular microscopy
image registration. 

ZenReg's philosophy is to make common microscopy registration tasks available
through one convenient wrapper while keeping the internals modular enough for
new backends and contributions. The package is therefore designed around one main workflow
and a set of modular backends that can be used:

.. code-block:: python

   from zenreg import load_stack, register_stack, save_stack

   image, metadata = load_stack("image.ome.tif", return_metadata=True)

   registered, details = register_stack(
       image,
       registration_channel=0,
       method="phase_cross_correlation",
       return_shifts=True,
       return_details=True)

   save_stack("image_registered.ome.tif", registered, metadata=metadata, registration_details=details)

At the same time, the package is built for reproducible outputs:
Registered OME-TIFFs can be saved together with shift tables, settings YAML
files, and summary plots. This enables users to easily share their results and 
reproduce them later.

.. toctree::
   :maxdepth: 3
   :caption: Contents

   overview
   installation
   usage
   api
   changelog
   contributing
