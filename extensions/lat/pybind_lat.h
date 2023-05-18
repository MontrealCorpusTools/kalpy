
#ifndef KALPY_PYBIND_LAT_H_
#define KALPY_PYBIND_LAT_H_

#include "pybind/kaldi_pybind.h"

void pybind_arctic_weight(py::module &);
void pybind_compose_lattice_pruned(py::module &);
void pybind_confidence(py::module &);
void pybind_kaldi_lattice(py::module &);
void pybind_determinize_lattice_pruned(py::module &);
void pybind_kaldi_functions_transition_model(py::module &);
void pybind_lat_kaldi_functions(py::module &);
void pybind_lat_minimize_lattice(py::module &);
void pybind_lat_phone_align_lattice(py::module &);
void pybind_lat_push_lattice(py::module &);
void pybind_lat_sausages(py::module &);
void pybind_lat_word_align_lattice_lexicon(py::module &);
void pybind_lat_word_align_lattice(py::module &);
void init_lat(py::module &);
#endif  // KALPY_PYBIND_LAT_H_
