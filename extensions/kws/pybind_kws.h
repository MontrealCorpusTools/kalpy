
#ifndef KALPY_PYBIND_KWS_H_
#define KALPY_PYBIND_KWS_H_

#include "pybind/kaldi_pybind.h"

void pybind_kws_functions(py::module &);
void pybind_kws_scoring(py::module &);
void init_kws(py::module &);
#endif  // KALPY_PYBIND_KWS_H_
