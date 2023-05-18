
#include "rnnlm/pybind_rnnlm.h"
#include "rnnlm/rnnlm-compute-state.h"
#include "rnnlm/rnnlm-core-compute.h"
#include "rnnlm/rnnlm-core-training.h"
#include "rnnlm/rnnlm-embedding-training.h"
#include "rnnlm/rnnlm-example-utils.h"
#include "rnnlm/rnnlm-example.h"
#include "rnnlm/rnnlm-lattice-rescoring.h"
#include "rnnlm/rnnlm-test-utils.h"
#include "rnnlm/rnnlm-training.h"
#include "rnnlm/rnnlm-utils.h"
#include "rnnlm/sampler.h"
#include "rnnlm/sampling-lm-estimate.h"
#include "rnnlm/sampling-lm.h"

using namespace kaldi;

void pybind_rnnlm_compute_state(py::module &m) {

}

void pybind_rnnlm_core_compute(py::module &m) {

}

void pybind_rnnlm_core_training(py::module &m) {

}

void pybind_rnnlm_embedding_training(py::module &m) {

}

void pybind_rnnlm_example_utils(py::module &m) {

}

void pybind_rnnlm_example(py::module &m) {

}

void pybind_rnnlm_lattice_rescoring(py::module &m) {

}

void pybind_rnnlm_test_util(py::module &m) {

}

void pybind_rnnlm_training(py::module &m) {

}

void pybind_rnnlm_utils(py::module &m) {

}

void pybind_rnnlm_sampler(py::module &m) {

}

void pybind_rnnlm_sampling_lm_estimate(py::module &m) {

}

void pybind_rnnlm_sampling_lm(py::module &m) {

}

void init_rnnlm(py::module &_m) {
  py::module m = _m.def_submodule("rnnlm", "rnnlm pybind for Kaldi");
  pybind_rnnlm_compute_state(m);
  pybind_rnnlm_core_compute(m);
  pybind_rnnlm_core_training(m);
  pybind_rnnlm_embedding_training(m);
  pybind_rnnlm_example_utils(m);
  pybind_rnnlm_example(m);
  pybind_rnnlm_lattice_rescoring(m);
  pybind_rnnlm_test_util(m);
  pybind_rnnlm_training(m);
  pybind_rnnlm_utils(m);
  pybind_rnnlm_sampler(m);
  pybind_rnnlm_sampling_lm_estimate(m);
  pybind_rnnlm_sampling_lm(m);
}
