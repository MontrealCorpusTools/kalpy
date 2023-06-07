# kalpy
Pybind11 bindings for Kaldi for use in [Montreal Forced Aligner](montreal-forced-aligner.readthedocs.io/).

## Installation

Kalpy depends on Kaldi being built as shared libraries, and the easiest way to install is via conda-forge:

```
conda install -c conda-forge kalpy
```

Kalpy is also available on pip via the `kalpy-kaldi` package, but as this is only a binding library, it relies on Kaldi shared libraries being available. The `KALDI_ROOT` environment variable must be set to locate the shared libraries and header files.  The easiest way to install the appropriately built kaldi libraries is via `conda install -c conda-forge kaldi`.

```
export KALDI_ROOT=/path/to/conda/enviornment
pip install kalpy-kaldi
```

## Usage

Two libraries are installed, `_kalpy` which contains low level bindings conforming to the original C++ style, and `kalpy` which is a more pythonic interface for higher level operations.  The `kalpy` package is under heavy development and expansion to expose more functionality.
