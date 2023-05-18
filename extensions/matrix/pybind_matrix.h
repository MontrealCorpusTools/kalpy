
#ifndef KALPY_PYBIND_MATRIX_H_
#define KALPY_PYBIND_MATRIX_H_

#include "pybind/kaldi_pybind.h"


void init_matrix(py::module &);
void pybind_matrix_common(py::module &);
void pybind_kaldi_vector(py::module &);
void pybind_sparse_matrix(py::module &);
void pybind_kaldi_matrix(py::module &);
void pybind_compressed_matrix(py::module &);
#endif  // KALPY_PYBIND_MATRIX_H_
