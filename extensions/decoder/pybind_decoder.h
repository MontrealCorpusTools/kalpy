
#ifndef KALPY_PYBIND_DECODER_H_
#define KALPY_PYBIND_DECODER_H_

#include "pybind/kaldi_pybind.h"

void pybind_decoder_biglm_faster_decoder(py::module &);
void pybind_decoder_decodable_mapped(py::module &);
void pybind_decoder_decodable_sum(py::module &);
void pybind_decoder_grammar_fst(py::module &);
void pybind_decoder_faster_decoder(py::module &);
void pybind_decoder_lattice_biglm_faster_decoder(py::module &);
void pybind_decoder_lattice_faster_online_decoder(py::module &);
void pybind_decoder_lattice_incremental_decoder(py::module &);
void pybind_decoder_lattice_incremental_online_decoder(py::module &);
void pybind_decoder_lattice_simple_decoder(py::module &);
void pybind_decoder_wrappers(py::module &);
void pybind_decoder_simple_decoder(py::module &);
void pybind_lattice_faster_decoder(py::module &);
void pybind_lattice_faster_decoder_config(py::module &);
void pybind_decodable_matrix_mapped_offset(py::module &);
void pybind_decodable_matrix_mapped(py::module &);
void pybind_decodable_matrix_scale_mapped(py::module &);
void pybind_training_graph_compiler(py::module &);
void init_decoder(py::module &);
#endif  // KALPY_PYBIND_DECODER_H_
