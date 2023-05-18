
#ifndef KALPY_PYBIND_GMM_H_
#define KALPY_PYBIND_GMM_H_

#include "pybind/kaldi_pybind.h"

void pybind_am_diag_gmm(py::module &);
void pybind_decodable_am_diag_gmm(py::module &);
void pybind_diag_gmm_normal(py::module &);
void pybind_diag_gmm(py::module &);
void pybind_ebw_diag_gmm(py::module &);
void pybind_full_gmm_normal(py::module &);
void pybind_full_gmm(py::module &);
void pybind_indirect_diff_diag_gmm(py::module &);
void pybind_mle_am_diag_gmm(py::module &);
void pybind_mle_diag_gmm(py::module &);
void pybind_mle_full_gmm(py::module &);
void pybind_model_common(py::module &);
void init_gmm(py::module &);
#endif  // KALPY_PYBIND_GMM_H_
