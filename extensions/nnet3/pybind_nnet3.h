
#ifndef KALPY_PYBIND_NNET3_H_
#define KALPY_PYBIND_NNET3_H_

#include "pybind/kaldi_pybind.h"

void pybind_nnet_common(py::module &);
void pybind_nnet_component_itf(py::module &);
void pybind_nnet_convolutional_component(py::module &);
void pybind_nnet_example(py::module &);
void pybind_nnet_chain_example(py::module &);
void pybind_nnet_nnet(py::module &);
void pybind_nnet_normalize_component(py::module &);
void pybind_nnet_simple_component(py::module &);
void init_nnet3(py::module &);
#endif  // KALPY_PYBIND_NNET3_H_
