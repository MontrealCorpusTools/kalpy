
#ifndef KALPY_PYBIND_NNET_H_
#define KALPY_PYBIND_NNET_H_

#include "pybind/kaldi_pybind.h"

void pybind_nnet_nnet_activation(py::module &);
void pybind_nnet_nnet_affine_transform(py::module &);
void pybind_nnet_nnet_average_pooling_component(py::module &);
void pybind_nnet_nnet_blstm_projected(py::module &);
void pybind_nnet_nnet_component(py::module &);
void pybind_nnet_nnet_convolutional_component(py::module &);
void pybind_nnet_nnet_frame_pooling_component(py::module &);
void pybind_nnet_nnet_kl_hmm(py::module &);
void pybind_nnet_nnet_linear_transform(py::module &);
void pybind_nnet_nnet_loss(py::module &);
void pybind_nnet_nnet_lstm_projected(py::module &);
void pybind_nnet_nnet_matrix_buffer(py::module &);
void pybind_nnet_nnet_max_pooling_component(py::module &);
void pybind_nnet_nnet_multibasis_component(py::module &);
void pybind_nnet_nnet_nnet(py::module &);
void pybind_nnet_nnet_parallel_component(py::module &);
void pybind_nnet_nnet_parametric_relu(py::module &);
void pybind_nnet_nnet_pdf_prior(py::module &);
void pybind_nnet_nnet_randomizer(py::module &);
void pybind_nnet_nnet_rbm(py::module &);
void pybind_nnet_nnet_recurrent(py::module &);
void pybind_nnet_nnet_sentence_averaging_component(py::module &);
void pybind_nnet_nnet_trnopts(py::module &);
void pybind_nnet_nnet_various(py::module &);

void init_nnet(py::module &);
#endif  // KALPY_PYBIND_NNET_H_
