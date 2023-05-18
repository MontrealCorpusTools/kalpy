
#ifndef KALPY_PYBIND_LM_H_
#define KALPY_PYBIND_LM_H_

#include "pybind/kaldi_pybind.h"

void pybind_lm_arpa_file_parser(py::module &);
void pybind_lm_arpa_lm_compiler(py::module &);
void pybind_lm_const_arpa_lm(py::module &);
void pybind_lm_kaldi_rnnlm(py::module &);
void pybind_lm_mikolov_rnnlm_lib(py::module &);
void init_lm(py::module &);
#endif  // KALPY_PYBIND_LM_H_
