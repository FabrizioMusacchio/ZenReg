# ZenReg: A Python-based high-throughput, memory-efficient N-dimensional image registration pipeline

ZenReg is a Python-based high-throughput, memory-efficient pipeline for N-dimensional image registration, designed for large microscopy and imaging datasets.

The project targets volumetric and time-resolved data with arbitrary dimensionality (3D, 4D, and full 5D stacks such as TZCYX), with a strong focus on scalable, out-of-core processing using chunked array storage and parallel execution.

**ZenReg is currently under active development**. This repository hosts an initial  release while the core registration pipeline is being refactored, cleaned up, and prepared for open-source publication.

## Scope and design goals

ZenReg is designed with the following principles in mind:

* support for true N-dimensional image data without special-casing (e.g. T=1, C=1, or Z=1)
* memory-efficient, out-of-core processing via chunked array storage
* scalable and parallel execution for high-throughput workloads
* suitability for large microscopy datasets that do not fit into memory
* clean and explicit handling of dimensional metadata

## Current status
No stable public API is provided yet.


## License
ZenReg is released under an open-source license (GPL-3.0). See the LICENSE file for details.
