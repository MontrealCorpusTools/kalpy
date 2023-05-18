
#ifndef KALPY_PYBIND_NNET2_H_
#define KALPY_PYBIND_NNET2_H_

#include "pybind/kaldi_pybind.h"

void pybind_nnet2_am_nnet(py::module &);
void pybind_nnet2_combine_nnet_a(py::module &);
void pybind_nnet2_combine_nnet_fast(py::module &);
void pybind_nnet2_combine_nnet(py::module &);
void pybind_nnet2_decodable_am_net(py::module &);
void pybind_nnet2_get_feature_transform(py::module &);
void pybind_nnet2_mixup_nnet(py::module &);
void pybind_nnet2_nnet_component(py::module &);
void pybind_nnet2_nnet_compute_discriminative_parallel(py::module &);
void pybind_nnet2_nnet_compute_discriminative(py::module &);
void pybind_nnet2_nnet_compute_online(py::module &);
void pybind_nnet2_nnet_compute(py::module &);
void pybind_nnet2_nnet_example_functions(py::module &);
void pybind_nnet2_nnet_example(py::module &);
void pybind_nnet2_nnet_fix(py::module &);
void pybind_nnet2_nnet_functions(py::module &);
void pybind_nnet2_nnet_limit_rank(py::module &);
void pybind_nnet2_nnet_nnet(py::module &);
void pybind_nnet2_nnet_precondition_online(py::module &);
void pybind_nnet2_nnet_precondition(py::module &);
void pybind_nnet2_nnet_stats(py::module &);
void pybind_nnet2_nnet_update_parallel(py::module &);
void pybind_nnet2_nnet_update(py::module &);
void pybind_nnet2_online_nnet2_decodable(py::module &);
void pybind_nnet2_rescale_nnet(py::module &);
void pybind_nnet2_shrink_nnet(py::module &);
void pybind_nnet2_train_nnet_ensemble(py::module &);
void pybind_nnet2_train_nnet(py::module &);
void pybind_nnet2_widen_nnet(py::module &);

void init_nnet2(py::module &);
#endif  // KALPY_PYBIND_NNET2_H_
