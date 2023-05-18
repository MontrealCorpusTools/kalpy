
#ifndef KALPY_PYBIND_ITF_H_
#define KALPY_PYBIND_ITF_H_

#include "pybind/kaldi_pybind.h"

void pybind_context_dep_itf(py::module &);
void pybind_decodable_itf(py::module &);
void pybind_options_itf(py::module &);
void pybind_clusterable_itf(py::module &);
void pybind_transition_information_itf(py::module &);
void init_itf(py::module &);
#endif  // KALPY_PYBIND_ITF_H_
