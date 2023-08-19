
#ifndef KALPY_PYBIND_HMM_H_
#define KALPY_PYBIND_HMM_H_

#include "pybind/kaldi_pybind.h"
#include "hmm/hmm-topology.h"
#include "hmm/hmm-utils.h"
#include "hmm/posterior.h"
#include "hmm/transition-model.h"
#include "hmm/tree-accu.h"

void pybind_hmm_topology(py::module &);
void pybind_hmm_utils(py::module &);
void pybind_posterior(py::module &);
void pybind_transition_model(py::module &);
void pybind_tree_accu(py::module &);
void init_hmm(py::module &);
#endif  // KALPY_PYBIND_HMM_H_
