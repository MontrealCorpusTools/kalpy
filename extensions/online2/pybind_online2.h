
#ifndef KALPY_PYBIND_ONLINE2_H_
#define KALPY_PYBIND_ONLINE2_H_

#include "pybind/kaldi_pybind.h"

void pybind_online_endpoint(py::module &);
void pybind_online_feature_pipeline(py::module &);
void pybind_online_gmm_decodable(py::module &);
void pybind_online_gmm_decoding(py::module &);
void pybind_online_ivector_feature(py::module &);
void pybind_online_nnet2_decoding_threaded(py::module &);
void pybind_online_nnet2_decoding(py::module &);
void pybind_online_nnet2_feature_pipeline(py::module &);
void pybind_online_nnet3_decoding(py::module &);
void pybind_online_nnet3_incremental_decoding(py::module &);
void pybind_online_nnet3_wake_word_faster_decoder(py::module &);
void pybind_online_speex_wrapper(py::module &);
void pybind_online_timing(py::module &);
void pybind_online2bin_util(py::module &);
void init_online2(py::module &);
#endif  // KALPY_PYBIND_ONLINE2_H_
