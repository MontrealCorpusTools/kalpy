
#ifndef KALPY_PYBIND_ONLINE_H_
#define KALPY_PYBIND_ONLINE_H_

#include "pybind/kaldi_pybind.h"

void pybind_online_audio_source(py::module &);
void pybind_online_decodable(py::module &);
void pybind_online_faster_decoder(py::module &);
void pybind_online_feat_input(py::module &);
void pybind_online_tcp_source(py::module &);
void pybind_onlinebin_util(py::module &);
void init_online(py::module &);
#endif  // KALPY_PYBIND_ONLINE_H_
