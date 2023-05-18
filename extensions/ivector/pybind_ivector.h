
#ifndef KALPY_PYBIND_IVECTOR_H_
#define KALPY_PYBIND_IVECTOR_H_

#include "pybind/kaldi_pybind.h"

void pybind_agglomerative_clustering(py::module &);
void pybind_ivector_extractor(py::module &);
void pybind_logistic_regression(py::module &);
void pybind_plda(py::module &);
void pybind_voice_activity_detection(py::module &);
void init_ivector(py::module &);
#endif  // KALPY_PYBIND_IVECTOR_H_
