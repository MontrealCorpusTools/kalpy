
#include "chain/pybind_chain.h"
#include "fstext/pybind_fstext.h"

#include "chain/chain-den-graph.h"
#include "chain/chain-training.h"
#include "chain/chain-supervision.h"
#include "cudamatrix/cu-matrix.h"

using namespace kaldi;
using namespace kaldi::chain;
using namespace fst;

void pybind_chain_den_graph(py::module& _m) {
  py::module m = _m.def_submodule("chain", "chain pybind for Kaldi");
  using PyClass = DenominatorGraph;
  py::class_<PyClass>(m, "DenominatorGraph",
                      "This class is responsible for storing the FST that we use as the"
                      "'anti-model' or 'denominator-model', that models all possible phone"
                      "sequences (or most possible phone sequences, depending how we built it).."
                      "It stores the FST in a format where we can access both the transitions out"
                      "of each state, and the transitions into each state.")
    .def(py::init<const StdVectorFst&, int>(),
         "Initialize from epsilon-free acceptor FST with pdf-ids plus one as the"
         "labels.  'num_pdfs' is only needeed for checking.",
         py::arg("fst"), py::arg("num_pdfs"));
}


void pybind_chain_supervision(py::module& m) {
  {
    using PyClass = Supervision;
    py::class_<PyClass>(m, "Supervision",
                        "struct Supervision is the fully-processed supervision "
                        "information for a whole utterance or (after "
                        "splitting) part of an utterance.  It contains the "
                        "time limits on phones encoded into the FST.")
        .def(py::init<>())
        .def(py::init<const PyClass&>(), py::arg("other"))
        .def("Swap", &PyClass::Swap)
        .def_readwrite("weight", &PyClass::weight,
                       "The weight of this example (will usually be 1.0).")
        .def_readwrite("num_sequences", &PyClass::num_sequences,
                       "num_sequences will be 1 if you create a Supervision "
                       "object from a single lattice or alignment, but if you "
                       "combine multiple Supevision objects the "
                       "'num_sequences' is the number of objects that were "
                       "combined (the FSTs get appended).")
        .def_readwrite("frames_per_sequence", &PyClass::frames_per_sequence,
                       "the number of frames in each sequence of appended "
                       "objects.  num_frames * num_sequences must equal the "
                       "path length of any path in the FST. Technically this "
                       "information is redundant with the FST, but it's "
                       "convenient to have it separately.")
        .def_readwrite("label_dim", &PyClass::label_dim,
                       "the maximum possible value of the labels in 'fst' "
                       "(which go from 1 to label_dim).  For fully-processed "
                       "examples this will equal the NumPdfs() in the "
                       "TransitionModel object, but for newer-style "
                       "'unconstrained' examples that have been output by "
                       "chain-get-supervision but not yet processed by "
                       "nnet3-chain-get-egs, it will be the NumTransitionIds() "
                       "of the TransitionModel object.")
        .def_readwrite(
            "fst", &PyClass::fst,
            "This is an epsilon-free unweighted acceptor that is sorted in "
            "increasing order of frame index (this implies it's topologically "
            "sorted but it's a stronger condition).  The labels will normally "
            "be pdf-ids plus one (to avoid epsilons, since pdf-ids are "
            "zero-based), but for newer-style 'unconstrained' examples that "
            "have been output by chain-get-supervision but not yet processed "
            "by nnet3-chain-get-egs, they will be transition-ids. Each "
            "successful path in 'fst' has exactly 'frames_per_sequence * "
            "num_sequences' arcs on it (first 'frames_per_sequence' arcs for "
            "the first sequence; then 'frames_per_sequence' arcs for the "
            "second sequence, and so on).")
        .def_readwrite(
            "e2e_fsts", &PyClass::e2e_fsts,
            "'e2e_fsts' may be set as an alternative to 'fst'.  These FSTs are "
            "used when the numerator computation will be done with 'full "
            "forward_backward' instead of constrained in time.  (The "
            "'constrained in time' fsts are how we described it in the "
            "original LF-MMI paper, where each phone can only occur at the "
            "same time it occurred in the lattice, extended by a tolerance)."
            "\n"
            "This 'e2e_fsts' is an array of FSTs, one per sequence, that are "
            "acceptors with (pdf_id + 1) on the labels, just like 'fst', but "
            "which are cyclic FSTs. Unlike with 'fst', it is not the case with "
            "'e2e_fsts' that each arc corresponds to a specific frame)."
            "\n"
            "There are two situations 'e2e_fsts' might be set. The first is in "
            "'end-to-end' training, where we train without a tree from a flat "
            "start.  The function responsible for creating this object in that "
            "case is TrainingGraphToSupervision(); to find out more about "
            "end-to-end training, see chain-generic-numerator.h The second "
            "situation is where we create the supervision from lattices, and "
            "split them into chunks using the time marks in the lattice, but "
            "then make a cyclic FST, and don't enforce the times on the "
            "lattice inside the chunk.  [Code location TBD].")
        .def_readwrite("alignment_pdfs", &PyClass::alignment_pdfs,
                       "This member is only set to a nonempty value if we are "
                       "creating 'unconstrained' egs.  These are egs that are "
                       "split into chunks using the lattice alignments, but "
                       "then within the chunks we remove the frame-level "
                       "constraints on which phones can appear when, and use "
                       "the 'e2e_fsts' member."
                       "\n"
                       "It is only required in order to accumulate the LDA "
                       "stats using `nnet3-chain-acc-lda-stats`, and it is not "
                       "merged by nnet3-chain-merge-egs; it will only be "
                       "present for un-merged egs.")
        .def("__str__",
             [](const PyClass& sup) {
               std::ostringstream os;
               os << "weight: " << sup.weight << "\n"
                  << "num_sequences: " << sup.num_sequences << "\n"
                  << "frames_per_sequence: " << sup.frames_per_sequence << "\n"
                  << "label_dim: " << sup.label_dim << "\n";
               return os.str();
             })
        // TODO(fangjun): Check, Write and Read are not wrapped
        ;
  }
}

void pybind_chain_training(py::module& m) {
  py::class_<ChainTrainingOptions>(m, "ChainTrainingOptions")
      .def(py::init<>())
      .def_readwrite(
          "l2_regularize", &ChainTrainingOptions::l2_regularize,
          "l2 regularization constant on the 'chain' output; the actual term "
          "added to the objf will be -0.5 times this constant times the "
          "squared l2 norm.(squared so it's additive across the dimensions). "
          "e.g. try 0.0005.")
      .def_readwrite(
          "out_of_range_regularize",
          &ChainTrainingOptions::out_of_range_regularize,
          "This is similar to an l2 regularization constant (like "
          "l2-regularize) but applied on the part of the nnet output matrix "
          "that exceeds the range [-30,30]... this is necessary to avoid "
          "things regularly going out of the range that we can do exp() on, "
          "since the denominator computation is not in log space and to avoid "
          "NaNs we limit the outputs to the range [-30,30].")
      .def_readwrite(
          "leaky_hmm_coefficient", &ChainTrainingOptions::leaky_hmm_coefficient,
          "Coefficient for 'leaky hmm'.  This means we have an "
          "epsilon-transition from each state to a special state with "
          "probability one, and then another epsilon-transition from that "
          "special state to each state, with probability leaky_hmm_coefficient "
          "times [initial-prob of destination state]. Imagine"
          "we make two copies of each state prior to doing this, version A and "
          "version B, with transition from A to B, so we don't have to "
          "consider epsilon loops- or just imagine the coefficient is small "
          "enough that we can ignore the epsilon loops. Note: we generally set "
          "leaky_hmm_coefficient to 0.1.")
      .def_readwrite("xent_regularize", &ChainTrainingOptions::xent_regularize,
                     "Cross-entropy regularization constant.  (e.g. try 0.1).  "
                     "If nonzero, the network is expected to have an output "
                     "named 'output-xent', which should have a softmax as its "
                     "final nonlinearity.");

  m.def(
      "ComputeChainObjfAndDeriv",
      [](const ChainTrainingOptions& opts, const DenominatorGraph& den_graph,
         const Supervision& supervision, const CuMatrixBase<float>& nnet_output,
         VectorBase<float>* objf_l2_term_weight,
         CuMatrixBase<float>* nnet_output_deriv,
         CuMatrixBase<float>* xent_output_deriv = nullptr) {
        // Note that we have changed `CuMatrix<float>*`
        // to `CuMatrixBase<float>*` for xent_output_deriv

        float* objf = objf_l2_term_weight->Data();
        float* l2_term = objf_l2_term_weight->Data() + 1;
        float* weight = objf_l2_term_weight->Data() + 2;

        ComputeChainObjfAndDeriv(
            opts, den_graph, supervision, nnet_output, objf, l2_term, weight,
            nnet_output_deriv,
            reinterpret_cast<CuMatrix<float>*>(xent_output_deriv));
      },
      py::arg("opts"), py::arg("den_graph"), py::arg("supervision"),
      py::arg("nnet_output"), py::arg("objf_l2_term_weight"),
      py::arg("nnet_output_deriv"), py::arg("xent_output_deriv") = nullptr);
}

void init_chain(py::module &m) {
    pybind_chain_den_graph(m);

  pybind_chain_supervision(m);
  pybind_chain_training(m);

}
