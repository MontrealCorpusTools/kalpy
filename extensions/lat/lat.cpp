
#include "lat/pybind_lat.h"
#include "base/kaldi-types.h"
#include "fstext/pybind_fstext.h"
#include "fstext/lattice-weight.h"
#include "util/pybind_util.h"

#include "lat/arctic-weight.h"
#include "lat/compose-lattice-pruned.h"
#include "lat/confidence.h"
#include "lat/determinize-lattice-pruned.h"
#include "lat/kaldi-lattice.h"
#include "lat/lattice-functions-transition-model.h"
#include "lat/lattice-functions.h"
#include "lat/minimize-lattice.h"
#include "lat/phone-align-lattice.h"
#include "lat/push-lattice.h"
#include "lat/sausages.h"
#include "lat/word-align-lattice-lexicon.h"
#include "lat/word-align-lattice.h"
#include "lm/const-arpa-lm.h"

using namespace kaldi;

void pybind_arctic_weight(py::module& m) {
  using namespace fst;
  {
    using PyClass = ArcticWeight;

    py::class_<PyClass>(m, "ArcticWeight")
        .def(py::init<>())
        .def(py::init<float>(),
          py::arg("f"))
        .def(py::init<PyClass>(),
          py::arg("w"))
        .def_static("Zero", &PyClass::Zero)
        .def_static("One", &PyClass::One)
        .def_static("Type", &PyClass::Type)
        .def_static("NoWeight", &PyClass::NoWeight)
        .def("Member", &PyClass::Member)
        .def("Quantize", &PyClass::Quantize,
          py::arg("delta"))
        .def_static("Properties", &PyClass::Properties)
        .def("Reverse", &PyClass::Reverse);
  }

}

void pybind_compose_lattice_pruned(py::module& m) {

  {
    using PyClass = ComposeLatticePrunedOptions;

    auto compose_lattice_pruned_options = py::class_<PyClass>(
        m, "ComposeLatticePrunedOptions");
    compose_lattice_pruned_options.def(py::init<>())
      .def_readwrite("lattice_compose_beam", &PyClass::lattice_compose_beam,
                      "'lattice_compose_beam' is a beam that determines "
                    "how much of a given composition space we will expand (at least, "
                    "until we hit the limit imposed by 'max_arcs'..  This "
                    "beam is applied using heuristically-estimated expected costs "
                    "to the end of the lattice, so if you specify, for example, "
                    "beam=5.0, it doesn't guarantee that all paths with best-cost "
                    "within 5.0 of the best path in the composed output will be "
                    "retained (However, this would be exact if the LM we were "
                    "rescoring with had zero costs).")
      .def_readwrite("max_arcs", &PyClass::max_arcs,
                      "'max_arcs' is the maximum number of arcs that we are willing to expand per "
                    "lattice; once this limit is reached, we terminate the composition (however, "
                    "this limit is not applied until at least one path to a final-state has been "
                    "produced).")
      .def_readwrite("initial_num_arcs", &PyClass::initial_num_arcs,
                      "'initial_num_arcs' is the number of arcs we use on the first outer "
                      "iteration of the algorithm.  This is so unimportant that we do not expose "
                      "it on the command line.")
      .def_readwrite("growth_ratio", &PyClass::growth_ratio,
                      "'growth_ratio' determines how much we allow the num-arcs to grow on each "
                      "outer iteration of the algorithm.  1.5 is a reasonable value; if it is set "
                      "too small, too much time will be taken in RecomputePruningInfo(), and if "
                      "too large, the paths searched may be less optimal than they could be (the "
                      "heuristics will be less accurate)..");
  }
  m.def("ComposeCompactLatticePruned",
        &ComposeCompactLatticePruned,
        "Does pruned composition of a lattice 'clat' with a DeterministicOnDemandFst "
        "'det_fst'; implements LM rescoring. "
        "\n"
        "@param [in] opts Class containing options "
        "@param [in] clat   The input lattice, which is expected to already have a "
        "                reasonable acoustic scale applied (e.g. 0.1 if it's a normal "
        "                cross-entropy system, but 1.0 for a chain system); this scale "
        "                affects the pruning. "
        "@param [in] det_fst   The on-demand FST that we are composing with; its "
        "                ilabels will correspond to words and it should be an acceptor "
        "                in practice (ilabel == olabel).  Will often contain a "
        "                weighted difference of language model scores, with scores "
        "                of the form alpha * new - alpha * old, where alpha "
        "                is the interpolation weight for the 'new' language model "
        "                (e.g. 0.5 or 0.8).  It's non-const because 'det_fst' is "
        "                on-demand. "
        "@param [out] composed_clat  The output, which is a result of composing "
        "                'clat' with '*det_fst'.  Notionally, '*det_fst' is on the "
        "                right, although both are acceptors so it doesn't really "
        "                matter in practice. "
        "                Although the two FSTs are of different types, the code "
        "                manually does the conversion.  The weights in '*det_fst' "
        "                will be interpreted as graph weights (Value1()) in the "
        "                lattice semiring.",
        py::arg("opts"),
        py::arg("clat"),
        py::arg("det_fst"),
        py::arg("composed_clat"));
}

void pybind_confidence(py::module& m) {

  m.def("SentenceLevelConfidence",
        py::overload_cast<const CompactLattice &,
                                  int32 *,
                                  std::vector<int32> *,
                                  std::vector<int32> *>(&SentenceLevelConfidence),
        "Caution: this function is not the only way to get confidences in Kaldi. "
        "This only gives you sentence-level (utterance-level) confidence.  You can "
        "get word-by-word confidence within a sentence, along with Minimum Bayes Risk "
        "decoding, by looking at sausages.h. "
        "Caution: confidences estimated using this type of method are not very "
        "accurate. "
        "This function will return the difference between the best path in clat and "
        "the second-best path in clat (a positive number), or zero if clat was "
        "equivalent to the empty FST (no successful paths), or infinity if there "
        "was only one path in \"clat\".  It will output to \"num_paths\" (if non-NULL) "
        "a number n = 0, 1 or 2 saying how many n-best paths (up to two) were found. "
        "If n >= 1 it outputs to \"best_sentence\" (if non-NULL) the best word-sequence; "
        "if n == 2 it outputs to \"second_best_sentence\" (if non-NULL) the second best "
        "word-sequence (this may be useful for testing whether the two best word "
        "sequences are somehow equivalent for the task at hand).  If you need more "
        "information than this or want to look deeper inside the n-best list, then "
        "look at the implementation of this function. "
        "This function requires that distinct paths in \"lat\" have distinct word "
        "sequences; this will automatically be the case if you generated \"clat\" "
        "in any normal way, such as from a decoder, because a deterministic FST "
        "has this property. "
        "This function assumes that any acoustic scaling you want to apply, "
        "has already been applied.",
        py::arg("clat"),
        py::arg("num_paths"),
        py::arg("best_sentence"),
        py::arg("second_best_sentence"));
  m.def("SentenceLevelConfidence",
        py::overload_cast<const Lattice &,
                                  int32 *,
                                  std::vector<int32> *,
                                  std::vector<int32> *>(&SentenceLevelConfidence),
        "Caution: this function is not the only way to get confidences in Kaldi. "
        "This only gives you sentence-level (utterance-level) confidence.  You can "
        "get word-by-word confidence within a sentence, along with Minimum Bayes Risk "
        "decoding, by looking at sausages.h. "
        "Caution: confidences estimated using this type of method are not very "
        "accurate. "
        "This function will return the difference between the best path in clat and "
        "the second-best path in clat (a positive number), or zero if clat was "
        "equivalent to the empty FST (no successful paths), or infinity if there "
        "was only one path in \"clat\".  It will output to \"num_paths\" (if non-NULL) "
        "a number n = 0, 1 or 2 saying how many n-best paths (up to two) were found. "
        "If n >= 1 it outputs to \"best_sentence\" (if non-NULL) the best word-sequence; "
        "if n == 2 it outputs to \"second_best_sentence\" (if non-NULL) the second best "
        "word-sequence (this may be useful for testing whether the two best word "
        "sequences are somehow equivalent for the task at hand).  If you need more "
        "information than this or want to look deeper inside the n-best list, then "
        "look at the implementation of this function. "
        "This function requires that distinct paths in \"lat\" have distinct word "
        "sequences; this will automatically be the case if you generated \"clat\" "
        "in any normal way, such as from a decoder, because a deterministic FST "
        "has this property. "
        "This function assumes that any acoustic scaling you want to apply, "
        "has already been applied.",
        py::arg("clat"),
        py::arg("num_paths"),
        py::arg("best_sentence"),
        py::arg("second_best_sentence"));
}

void pybind_kaldi_functions_transition_model(py::module& m) {

  m.def("LatticeForwardBackwardMmi",
        &LatticeForwardBackwardMmi,
        "This function can be used to compute posteriors for MMI, with a positive contribution "
        "for the numerator and a negative one for the denominator.  This function is not actually "
        "used in our normal MMI training recipes, where it's instead done using various command "
        "line programs that each do a part of the job.  This function was written for use in "
        "neural-net MMI training. "
        "\n"
        "@param [in] trans    The transition model. Used to map the "
        "                      transition-ids to phones or pdfs. "
        "@param [in] lat      The denominator lattice "
        "@param [in] num_ali  The numerator alignment "
        "@param [in] drop_frames   If \"drop_frames\" is true, it will not compute any "
        "                      posteriors on frames where the num and den have disjoint "
        "                      pdf-ids. "
        "@param [in] convert_to_pdf_ids   If \"convert_to_pdfs_ids\" is true, it will "
        "                      convert the output to be at the level of pdf-ids, not "
        "                      transition-ids. "
        "@param [in] cancel   If \"cancel\" is true, it will cancel out any positive and "
        "                      negative parts from the same transition-id (or pdf-id, "
        "                      if convert_to_pdf_ids == true). "
        "@param [out] arc_post   The output MMI posteriors of transition-ids (or "
        "                      pdf-ids if convert_to_pdf_ids == true) at each frame "
        "                      i.e. the difference between the numerator "
        "                      and denominator posteriors. "
        "\n"
        "It returns the forward-backward likelihood of the lattice.",
        py::arg("trans"),
        py::arg("lat"),
        py::arg("num_ali"),
        py::arg("drop_frames"),
        py::arg("convert_to_pdf_ids"),
        py::arg("cancel"),
        py::arg("arc_post"));
  m.def("CompactLatticeToWordProns",
        &CompactLatticeToWordProns,
        "This function takes a CompactLattice that should only contain a single "
        "linear sequence (e.g. derived from lattice-1best), and that should have been "
        "processed so that the arcs in the CompactLattice align correctly with the "
        "word boundaries (e.g. by lattice-align-words).  It outputs 4 vectors of the "
        "same size, which give, for each word in the lattice (in sequence), the word "
        "label, the begin time and length in frames, and the pronunciation (sequence "
        "of phones).  This is done even for zero words, corresponding to optional "
        "silences -- if you don't want them, just ignore them in the output. "
        "This function will print a warning and return false, if the lattice "
        "did not have the correct format (e.g. if it is empty or it is not "
        "linear).",
        py::arg("tmodel"),
        py::arg("clat"),
        py::arg("words"),
        py::arg("begin_times"),
        py::arg("lengths"),
        py::arg("prons"),
        py::arg("phone_lengths"));
}

void pybind_lat_kaldi_functions(py::module& m) {

  m.def("GetPerFrameAcousticCosts",
           [](const fst::VectorFst<LatticeArc> &decoded) -> Vector<BaseFloat> {
          py::gil_scoped_release release;
            Vector<BaseFloat> per_frame_loglikes;
             GetPerFrameAcousticCosts(decoded, &per_frame_loglikes);
             return per_frame_loglikes;
           },
        "This function extracts the per-frame log likelihoods from a linear "
        "lattice (which we refer to as an 'nbest' lattice elsewhere in Kaldi code). "
        "The dimension of *per_frame_loglikes will be set to the "
        "number of input symbols in 'nbest'.  The elements of "
        "'*per_frame_loglikes' will be set to the .Value2() elements of the lattice "
        "weights, which represent the acoustic costs; you may want to scale this "
        "vector afterward by -1/acoustic_scale to get the original loglikes. "
        "If there are acoustic costs on input-epsilon arcs or the final-prob in 'nbest' "
        "(and this should not normally be the case in situations where it makes "
        "sense to call this function), they will be included to the cost of the "
        "preceding input symbol, or the following input symbol for input-epsilons "
        "encountered prior to any input symbol.  If 'nbest' has no input symbols, "
        "'per_frame_loglikes' will be set to the empty vector.",
        py::arg("nbest"));

  m.def("LatticeStateTimes",
        &LatticeStateTimes,
        "This function iterates over the states of a topologically sorted lattice and "
        "counts the time instance corresponding to each state. The times are returned "
        "in a vector of integers 'times' which is resized to have a size equal to the "
        "number of states in the lattice. The function also returns the maximum time "
        "in the lattice (this will equal the number of frames in the file).",
        py::arg("lat"),
        py::arg("times"));

  m.def("CompactLatticeStateTimes",
        &CompactLatticeStateTimes,
        "As LatticeStateTimes, but in the CompactLattice format.  Note: must "
        "be topologically sorted.  Returns length of the utterance in frames, which "
        "might not be the same as the maximum time in the lattice, due to frames "
        "in the final-prob.",
        py::arg("clat"),
        py::arg("times"));

  m.def("LatticeForwardBackward",
        &LatticeForwardBackward,
        "This function does the forward-backward over lattices and computes the "
        "posterior probabilities of the arcs. It returns the total log-probability "
        "of the lattice.  The Posterior quantities contain pairs of (transition-id, weight) "
        "on each frame. "
        "If the pointer \"acoustic_like_sum\" is provided, this value is set to "
        "the sum over the arcs, of the posterior of the arc times the "
        "acoustic likelihood [i.e. negated acoustic score] on that link. "
        "This is used in combination with other quantities to work out "
        "the objective function in MMI discriminative training.",
        py::arg("lat"),
        py::arg("arc_post"),
        py::arg("acoustic_like_sum") = NULL);

  m.def("ComputeCompactLatticeAlphas",
        &ComputeCompactLatticeAlphas,
        "This function is something similar to LatticeForwardBackward(), but it is on "
        "the CompactLattice lattice format. Also we only need the alpha in the forward "
        "path, not the posteriors.",
        py::arg("lat"),
        py::arg("alpha"));

  m.def("ComputeCompactLatticeBetas",
        &ComputeCompactLatticeBetas,
        "A sibling of the function CompactLatticeAlphas()... We compute the beta from "
        "the backward path here.",
        py::arg("lat"),
        py::arg("beta"));
  m.def("ComputeLatticeAlphasAndBetas",
        py::overload_cast<const kaldi::Lattice &,
                                    bool,
                                    std::vector<double> *,
                                    std::vector<double> *>(&ComputeLatticeAlphasAndBetas<kaldi::Lattice>),
        "Computes (normal or Viterbi) alphas and betas; returns (total-prob, or "
        "best-path negated cost) Note: in either case, the alphas and betas are "
        "negated costs.  Requires that lat be topologically sorted.  This code "
        "will work for either CompactLattice or Lattice.",
        py::arg("lat"),
        py::arg("viterbi"),
        py::arg("alpha"),
        py::arg("beta"));
  m.def("ComputeLatticeAlphasAndBetas",
        py::overload_cast<const kaldi::CompactLattice &,
                                    bool,
                                    std::vector<double> *,
                                    std::vector<double> *>(&ComputeLatticeAlphasAndBetas<kaldi::CompactLattice>),
        "Computes (normal or Viterbi) alphas and betas; returns (total-prob, or "
        "best-path negated cost) Note: in either case, the alphas and betas are "
        "negated costs.  Requires that lat be topologically sorted.  This code "
        "will work for either CompactLattice or Lattice.",
        py::arg("lat"),
        py::arg("viterbi"),
        py::arg("alpha"),
        py::arg("beta"));
  m.def("TopSortCompactLatticeIfNeeded",
        &TopSortCompactLatticeIfNeeded,
        "Topologically sort the compact lattice if not already topologically sorted. "
        "Will crash if the lattice cannot be topologically sorted.",
        py::arg("clat"));
  m.def("TopSortLatticeIfNeeded",
        &TopSortLatticeIfNeeded,
        "Topologically sort the compact lattice if not already topologically sorted. "
        "Will crash if the lattice cannot be topologically sorted.",
        py::arg("clat"));
  m.def("CompactLatticeDepth",
        &CompactLatticeDepth,
        "Returns the depth of the lattice, defined as the average number of arcs (or "
        "final-prob strings) crossing any given frame.  Returns 1 for empty lattices. "
        "Requires that clat is topologically sorted!",
        py::arg("clat"),
        py::arg("num_frames") = NULL);
  m.def("CompactLatticeDepthPerFrame",
        &CompactLatticeDepthPerFrame,
        "This function returns, for each frame, the number of arcs crossing that frame.",
        py::arg("clat"),
        py::arg("depth_per_frame"));
  m.def("CompactLatticeLimitDepth",
        &CompactLatticeLimitDepth,
        "This function limits the depth of the lattice, per frame: that means, it "
        "does not allow more than a specified number of arcs active on any given "
        "frame.  This can be used to reduce the size of the \"very deep\" portions of "
        "the lattice.",
        py::arg("max_arcs_per_frame"),
        py::arg("clat"));
  m.def("LatticeActivePhones",
        &LatticeActivePhones,
        "Given a lattice, and a transition model to map pdf-ids to phones, "
        "outputs for each frame the set of phones active on that frame.  If "
        "sil_phones (which must be sorted and uniq) is nonempty, it excludes "
        "phones in this list.",
        py::arg("lat"),
        py::arg("trans"),
        py::arg("sil_phones"),
        py::arg("active_phones"));
  m.def("ConvertLatticeToPhones",
        &ConvertLatticeToPhones,
        "Given a lattice, and a transition model to map pdf-ids to phones, "
        "replace the output symbols (presumably words), with phones; we "
        "use the TransitionModel to work out the phone sequence.  Note "
        "that the phone labels are not exactly aligned with the phone "
        "boundaries.  We put a phone label to coincide with any transition "
        "to the final, nonemitting state of a phone (this state always exists, "
        "we ensure this in HmmTopology::Check()).  This would be the last "
        "transition-id in the phone if reordering is not done (but typically "
        "we do reorder). "
        "Also see PhoneAlignLattice, in phone-align-lattice.h.",
        py::arg("trans_model"),
        py::arg("lat"));
  m.def("PruneLattice",
        py::overload_cast<BaseFloat, kaldi::Lattice *>(&PruneLattice<kaldi::Lattice>),
        "Prunes a lattice or compact lattice.  Returns true on success, false if "
        "there was some kind of failure.",
        py::arg("beam"),
        py::arg("lat"));
  m.def("PruneLattice",
        py::overload_cast<BaseFloat, kaldi::CompactLattice *>(&PruneLattice<kaldi::CompactLattice>),
        "Prunes a lattice or compact lattice.  Returns true on success, false if "
        "there was some kind of failure.",
        py::arg("beam"),
        py::arg("lat"));
  m.def("ConvertCompactLatticeToPhones",
        &ConvertCompactLatticeToPhones,
        "Given a lattice, and a transition model to map pdf-ids to phones, "
        "replace the sequences of transition-ids with sequences of phones. "
        "Note that this is different from ConvertLatticeToPhones, in that "
        "we replace the transition-ids not the words.",
        py::arg("trans_model"),
        py::arg("clat"));
  m.def("LatticeBoost",
        &LatticeBoost,
        "Boosts LM probabilities by b * [number of frame errors]; equivalently, adds "
        "-b*[number of frame errors] to the graph-component of the cost of each arc/path. "
        "There is a frame error if a particular transition-id on a particular frame "
        "corresponds to a phone not matching transcription's alignment for that frame. "
        "This is used in \"margin-inspired\" discriminative training, esp. Boosted MMI. "
        "The TransitionInformation is used to map transition-ids in the lattice "
        "input-side to phones; the phones appearing in "
        "\"silence_phones\" are treated specially in that we replace the frame error f "
        "(either zero or 1) for a frame, with the minimum of f or max_silence_error. "
        "For the normal recipe, max_silence_error would be zero. "
        "Returns true on success, false if there was some kind of mismatch. "
        "At input, silence_phones must be sorted and unique.",
        py::arg("trans"),
        py::arg("alignment"),
        py::arg("silence_phones"),
        py::arg("b"),
        py::arg("max_silence_error"),
        py::arg("lat"));
  m.def("LatticeForwardBackwardMpeVariants",
        &LatticeForwardBackwardMpeVariants,
        "This function implements either the MPFE (minimum phone frame error) or SMBR "
        "(state-level minimum bayes risk) forward-backward, depending on whether "
        "\"criterion\" is \"mpfe\" or \"smbr\".  It returns the MPFE "
        "criterion of SMBR criterion for this utterance, and outputs the posteriors (which "
        "may be positive or negative) into \"post\". "
        "\n"
        "@param [in] trans    The transition model. Used to map the "
        "                      transition-ids to phones or pdfs. "
        "@param [in] silence_phones   A list of integer ids of silence phones. The "
        "                      silence frames i.e. the frames where num_ali "
        "                      corresponds to a silence phones are treated specially. "
        "                      The behavior is determined by 'one_silence_class' "
        "                      being false (traditional behavior) or true. "
        "                      Usually in our setup, several phones including "
        "                      the silence, vocalized noise, non-spoken noise "
        "                      and unk are treated as \"silence phones\" "
        "@param [in] lat      The denominator lattice "
        "@param [in] num_ali  The numerator alignment "
        "@param [in] criterion    The objective function. Must be \"mpfe\" or \"smbr\" "
        "                      for MPFE (minimum phone frame error) or sMBR "
        "                      (state minimum bayes risk) training. "
        "@param [in] one_silence_class   Determines how the silence frames are treated. "
        "                      Setting this to false gives the old traditional behavior, "
        "                      where the silence frames (according to num_ali) are "
        "                      treated as incorrect. However, this means that the "
        "                      insertions are not penalized by the objective. "
        "                      Setting this to true gives the new behaviour, where we "
        "                      treat silence as any other phone, except that all pdfs "
        "                      of silence phones are collapsed into a single class for "
        "                      the frame-error computation. This can possible reduce "
        "                      the insertions in the trained model. This is closer to "
        "                      the WER metric that we actually care about, since WER is "
        "                      generally computed after filtering out noises, but "
        "                      does penalize insertions. "
        "  @param [out] post   The \"MBR posteriors\" i.e. derivatives w.r.t to the "
        "                      pseudo log-likelihoods of states at each frame.",
        py::arg("trans"),
        py::arg("silence_phones"),
        py::arg("lat"),
        py::arg("num_ali"),
        py::arg("criterion"),
        py::arg("one_silence_class"),
        py::arg("post"));

  m.def("CompactLatticeToWordAlignment",
        [](const CompactLattice& clat) {
            std::vector<int32> words, times, lengths;
            bool ans = CompactLatticeToWordAlignment(clat, &words, &times, &lengths);
             return py::make_tuple(ans, words, times, lengths);
           },
        "This function takes a CompactLattice that should only contain a single "
        "linear sequence (e.g. derived from lattice-1best), and that should have been "
        "processed so that the arcs in the CompactLattice align correctly with the "
        "word boundaries (e.g. by lattice-align-words).  It outputs 3 vectors of the "
        "same size, which give, for each word in the lattice (in sequence), the word "
        "label and the begin time and length in frames.  This is done even for zero "
        "(epsilon) words, generally corresponding to optional silence-- if you don't "
        "want them, just ignore them in the output. "
        "This function will print a warning and return false, if the lattice "
        "did not have the correct format (e.g. if it is empty or it is not "
        "linear).",
        py::arg("clat"));
  m.def("CompactLatticeShortestPath",
        &CompactLatticeShortestPath,
        "A form of the shortest-path/best-path algorithm that's specially coded for "
        "CompactLattice.  Requires that clat be acyclic.",
        py::arg("clat"),
        py::arg("shortest_path"));
  m.def("ExpandCompactLattice",
        &ExpandCompactLattice,
        "This function expands a CompactLattice to ensure high-probability paths "
        "have unique histories. Arcs with posteriors larger than epsilon get splitted.",
        py::arg("clat"),
        py::arg("epsilon"),
        py::arg("expand_clat"));
  m.def("CompactLatticeBestCostsAndTracebacks",
        &CompactLatticeBestCostsAndTracebacks,
        "For each state, compute forward and backward best (viterbi) costs and its "
        "traceback states (for generating best paths later). The forward best cost "
        "for a state is the cost of the best path from the start state to the state. "
        "The traceback state of this state is its predecessor state in the best path. "
        "The backward best cost for a state is the cost of the best path from the "
        "state to a final one. Its traceback state is the successor state in the best "
        "path in the forward direction. "
        "Note: final weights of states are in backward_best_cost_and_pred. "
        "Requires the input CompactLattice clat be acyclic.",
        py::arg("clat"),
        py::arg("forward_best_cost_and_pred"),
        py::arg("backward_best_cost_and_pred"));
  m.def("AddNnlmScoreToCompactLattice",
        &AddNnlmScoreToCompactLattice,
        "This function adds estimated neural language model scores of words in a "
        "minimal list of hypotheses that covers a lattice, to the graph scores on the "
        "arcs. The list of hypotheses are generated by latbin/lattice-path-cover.",
        py::arg("nnlm_scores"),
        py::arg("clat"));
  m.def("AddWordInsPenToCompactLattice",
        &AddWordInsPenToCompactLattice,
        "This function add the word insertion penalty to graph score of each word "
        "in the compact lattice",
        py::arg("word_ins_penalty"),
        py::arg("clat"));
  m.def("RescoreCompactLattice",
        &RescoreCompactLattice,
        "This function *adds* the negated scores obtained from the Decodable object, "
        "to the acoustic scores on the arcs.  If you want to replace them, you should "
        "use ScaleCompactLattice to first set the acoustic scores to zero.  Returns "
        "true on success, false on error (typically some kind of mismatched inputs).",
        py::arg("decodable"),
        py::arg("clat"));
  m.def("LongestSentenceLength",
        py::overload_cast<const Lattice &>(&LongestSentenceLength),
        "This function returns the number of words in the longest sentence in a "
        "CompactLattice (i.e. the the maximum of any path, of the count of "
        "olabels on that path).",
        py::arg("lat"));
  m.def("LongestSentenceLength",
        py::overload_cast<const CompactLattice &>(&LongestSentenceLength),
        "This function returns the number of words in the longest sentence in a "
        "CompactLattice (i.e. the the maximum of any path, of the count of "
        "olabels on that path).",
        py::arg("lat"));
  m.def("RescoreCompactLatticeSpeedup",
        &RescoreCompactLatticeSpeedup,
        "This function is like RescoreCompactLattice, but it is modified to avoid "
        "computing probabilities on most frames where all the pdf-ids are the same. "
        "(it needs the transition-model to work out whether two transition-ids map to "
        "the same pdf-id, and it assumes that the lattice has transition-ids on it). "
        "The naive thing would be to just set all probabilities to zero on frames "
        "where all the pdf-ids are the same (because this value won't affect the "
        "lattice posterior).  But this would become confusing when we compute "
        "corpus-level diagnostics such as the MMI objective function.  Instead, "
        "imagine speedup_factor = 100 (it must be >= 1.0)... with probability (1.0 / "
        "speedup_factor) we compute those likelihoods and multiply them by "
        "speedup_factor; otherwise we set them to zero.  This gives the right "
        "expected probability so our corpus-level diagnostics will be about right.",
        py::arg("tmodel"),
        py::arg("speedup_factor"),
        py::arg("decodable"),
        py::arg("clat"));
  m.def("RescoreLattice",
        &RescoreLattice,
        "This function *adds* the negated scores obtained from the Decodable object, "
        "to the acoustic scores on the arcs.  If you want to replace them, you should "
        "use ScaleCompactLattice to first set the acoustic scores to zero.  Returns "
        "true on success, false on error (e.g. some kind of mismatched inputs). "
        "The input labels, if nonzero, are interpreted as transition-ids or whatever "
        "other index the Decodable object expects.",
        py::arg("decodable"),
        py::arg("lat"));
  m.def("ComposeCompactLatticeDeterministic",
        &ComposeCompactLatticeDeterministic,
        "This function Composes a CompactLattice format lattice with a "
        "DeterministicOnDemandFst<fst::StdFst> format fst, and outputs another "
        "CompactLattice format lattice. The first element (the one that corresponds "
        "to LM weight) in CompactLatticeWeight is used for composition. "
        "\n"
        "Note that the DeterministicOnDemandFst interface is not \"const\", therefore "
        "we cannot use \"const\" for <det_fst>.",
        py::arg("clat"),
        py::arg("det_fst"),
        py::arg("composed_clat"));
  m.def("ComputeAcousticScoresMap",
        &ComputeAcousticScoresMap,
        "This function computes the mapping from the pair "
        "(frame-index, transition-id) to the pair "
        "(sum-of-acoustic-scores, num-of-occurences) over all occurences of the "
        "transition-id in that frame. "
        "frame-index in the lattice. "
        "This function is useful for retaining the acoustic scores in a "
        "non-compact lattice after a process like determinization where the "
        "frame-level acoustic scores are typically lost. "
        "The function ReplaceAcousticScoresFromMap is used to restore the "
        "acoustic scores computed by this function. "
        "\n"
        "  @param [in] lat   Input lattice. Expected to be top-sorted. Otherwise the "
        "                    function will crash. "
        "  @param [out] acoustic_scores   "
        "                    Pointer to a map from the pair (frame-index, "
        "                    transition-id) to a pair (sum-of-acoustic-scores, "
        "                    num-of-occurences). "
        "                    Usually the acoustic scores for a pdf-id (and hence "
        "                    transition-id) on a frame will be the same for all the "
        "                    occurences of the pdf-id in that frame. "
        "                    But if not, we will take the average of the acoustic "
        "                    scores. Hence, we store both the sum-of-acoustic-scores "
        "                    and the num-of-occurences of the transition-id in that "
        "                    frame.",
        py::arg("lat"),
        py::arg("acoustic_scores"));
  m.def("ReplaceAcousticScoresFromMap",
        &ReplaceAcousticScoresFromMap,
        "This function restores acoustic scores computed using the function "
        "ComputeAcousticScoresMap into the lattice. "
        "\n"
        "  @param [in] acoustic_scores  "
        "                     A map from the pair (frame-index, transition-id) to a "
        "                     pair (sum-of-acoustic-scores, num-of-occurences) of  "
        "                     the occurences of the transition-id in that frame. "
        "                     See the comments for ComputeAcousticScoresMap for  "
        "                     details. "
        "  @param [out] lat   Pointer to the output lattice.",
        py::arg("acoustic_scores"),
        py::arg("lat"));
}

void pybind_lat_minimize_lattice(py::module& m) {
  using namespace fst;
  m.def("MinimizeCompactLattice",
        &MinimizeCompactLattice<kaldi::LatticeWeight, int32>,
        "This function minimizes the compact lattice.  It is to be called after "
        "determinization (see ./determinize-lattice-pruned.h) and pushing "
        "(see ./push-lattice.h).  If the lattice is not determinized and pushed this "
        "function will not combine as many states as it could, but it won't crash. "
        "Returns true on success, and false if it failed due to topological sorting "
        "failing. "
        "The output will be topologically sorted.",
        py::arg("clat"),
        py::arg("delta") = fst::kDelta);
}

void pybind_lat_phone_align_lattice(py::module& m) {

  {
    using PyClass = PhoneAlignLatticeOptions;

    auto phone_align_lattice_options = py::class_<PyClass>(
        m, "PhoneAlignLatticeOptions");
    phone_align_lattice_options.def(py::init<>())
      .def_readwrite("reorder", &PyClass::reorder)
      .def_readwrite("remove_epsilon", &PyClass::remove_epsilon)
      .def_readwrite("replace_output_symbols", &PyClass::replace_output_symbols);
  }
  m.def("PhoneAlignLattice",
        &PhoneAlignLattice,
        "Outputs a lattice in which the arcs correspond exactly to sequences of "
        "phones, so the boundaries between the arcs correspond to the boundaries "
        "between phones If remove-epsilon == false and replace-output-symbols == "
        "false, but an arc may have >1 phone on it, but the boundaries will still "
        "correspond with the boundaries between phones.  Note: it's possible "
        "to have arcs with words on them but no transition-ids at all.  Returns true if "
        "everything was OK, false if some kind of error was detected (e.g. the "
        "\"reorder\" option was incorrectly specified.)",
        py::arg("lat"),
        py::arg("tmodel"),
        py::arg("opts"),
        py::arg("lat_out"));
}

void pybind_lat_push_lattice(py::module& m) {

  using namespace fst;
  m.def("PushCompactLatticeStrings",
        &PushCompactLatticeStrings<kaldi::LatticeWeight, int32>,
        "This function pushes the transition-ids as far towards the start as they "
        "will go.  It can be useful prior to lattice-align-words (for non-linear "
        "lattices).  We can't use the generic OpenFst \"push\" function because "
        "it uses the sum as the divisor, which is not appropriate in this case "
        "(a+b generally won't divide a or b in this semiring). "
        "It returns true on success, false if it failed due to TopSort failing, "
        "which should never happen, but we handle it gracefully by just leaving the "
        "lattice the same. "
        "This function used to be called just PushCompactLattice.",
        py::arg("clat"));
  m.def("PushCompactLatticeWeights",
        &PushCompactLatticeWeights<kaldi::LatticeWeight, int32>,
        "This function pushes the weights in the CompactLattice so that all states "
        "except possibly the start state, have Weight components (of type "
        "LatticeWeight) that \"sum to one\" in the LatticeWeight (i.e. interpreting the "
        "weights as negated log-probs).  It returns true on success, false if it "
        "failed due to TopSort failing, which should never happen, but we handle it "
        "gracefully by just leaving the lattice the same.",
        py::arg("clat"));
}

void pybind_lat_sausages(py::module& m) {

  {
    using PyClass = MinimumBayesRiskOptions;

    auto minimum_bayes_risk_options = py::class_<PyClass>(
        m, "MinimumBayesRiskOptions");
    minimum_bayes_risk_options.def(py::init<>())

      .def_readwrite("decode_mbr", &PyClass::decode_mbr,
                      "Boolean configuration parameter: if true, we actually update the hypothesis "
                      "to do MBR decoding (if false, our output is the MAP decoded output, but we "
                      "output the stats too (i.e. the confidences)).")
      .def_readwrite("print_silence", &PyClass::print_silence,
                      "Boolean configuration parameter: if true, the 1-best path will 'keep' the <eps> bins,");
  }
  {
    using PyClass = MinimumBayesRisk;

    auto minimum_bayes_risk = py::class_<PyClass>(
        m, "MinimumBayesRisk");
    minimum_bayes_risk.def(py::init<const CompactLattice &,
                   MinimumBayesRiskOptions>(),
        py::arg("clat"),
        py::arg("opts") = MinimumBayesRiskOptions())
      .def(py::init<const CompactLattice &,
                   const std::vector<int32> &,
                   MinimumBayesRiskOptions>(),
        py::arg("clat"),
        py::arg("words"),
        py::arg("opts") = MinimumBayesRiskOptions())
      .def(py::init<const CompactLattice &,
                   const std::vector<int32> &,
                   const std::vector<std::pair<BaseFloat,BaseFloat> > &,
                   MinimumBayesRiskOptions>(),
        py::arg("clat"),
        py::arg("words"),
        py::arg("times"),
        py::arg("opts") = MinimumBayesRiskOptions())
      .def("GetOneBest", &PyClass::GetOneBest)
      .def("GetTimes", &PyClass::GetTimes)
      .def("GetSausageTimes", &PyClass::GetSausageTimes)
      .def("GetOneBestTimes", &PyClass::GetOneBestTimes)
      .def("GetOneBestConfidences", &PyClass::GetOneBestConfidences)
      .def("GetBayesRisk", &PyClass::GetBayesRisk)
      .def("GetSausageStats", &PyClass::GetSausageStats);
  }
}

void pybind_lat_word_align_lattice_lexicon(py::module& m) {

  m.def("ReadLexiconForWordAlign",
        &ReadLexiconForWordAlign,
        "Read the lexicon in the special format required for word alignment.  Each line has "
        "a series of integers on it (at least two on each line), representing: "
        "\n"
        "<old-word-id> <new-word-id> [<phone-id-1> [<phone-id-2> ... ] ] "
        "\n"
        "Here, <old-word-id> is the word-id that appears in the lattice before alignment, and "
        "<new-word-id> is the word-is that should appear in the lattice after alignment.  This "
        "is mainly useful when the lattice may have no symbol for the optional-silence arcs "
        "(so <old-word-id> would equal zero), but we want it to be output with a symbol on those "
        "arcs (so <new-word-id> would be nonzero). "
        "If the silence should not be added to the lattice, both <old-word-id> and <new-word-id> "
        "may be zero. "
        "\n"
        "This function is very simple: it just reads in a series of lines from a text file, "
        "each with at least two integers on them.",
        py::arg("is"),
        py::arg("lexicon"));
  {
    using PyClass = WordAlignLatticeLexiconInfo;

    auto word_align_lattice_lexicon_info = py::class_<PyClass>(
        m, "WordAlignLatticeLexiconInfo");
    word_align_lattice_lexicon_info.def(py::init<const std::vector<std::vector<int32> > &>())
      .def("IsValidEntry", &PyClass::IsValidEntry,
      "Returns true if this lexicon-entry can appear, intepreted as "
      "(output-word phone1 phone2 ...).  This is just used in testing code.",
        py::arg("entry"))
      .def("EquivalenceClassOf", &PyClass::EquivalenceClassOf,
      "Purely for the testing code, we map words into equivalence classes derived "
      "from the mappings in the first two fields of each line in the lexicon.  This "
      "function maps from each word-id to the lowest member of its equivalence class.",
        py::arg("word"));
  }
  {
    using PyClass = WordAlignLatticeLexiconOpts;

    auto word_align_lattice_lexicon_opts = py::class_<PyClass>(
        m, "WordAlignLatticeLexiconOpts");
    word_align_lattice_lexicon_opts.def(py::init<>())
      .def_readwrite("partial_word_label", &PyClass::partial_word_label)
      .def_readwrite("reorder", &PyClass::reorder)
      .def_readwrite("max_expand", &PyClass::max_expand);
  }
  m.def("WordAlignLatticeLexicon",
        &WordAlignLatticeLexicon,
        "Align lattice so that each arc has the transition-ids on it "
        "that correspond to the word that is on that arc.  [May also have "
        "epsilon arcs for optional silences.] "
        "Returns true if everything was OK, false if there was any kind of "
        "error including when the the lattice seems to have been \"forced out\" "
        "(did not reach end state, resulting in partial word at end).",
        py::arg("lat"),
        py::arg("tmodel"),
        py::arg("lexicon_info"),
        py::arg("opts"),
        py::arg("lat_out"));
}

void pybind_lat_word_align_lattice(py::module& m) {

  {
    using PyClass = WordBoundaryInfoOpts;

    auto word_boundary_info_opts = py::class_<PyClass>(
        m, "WordBoundaryInfoOpts",
        "Note: use of this structure "
        "is deprecated, see WordBoundaryInfoNewOpts. "
        "\n"
        "Note: this structure (and the code in word-align-lattice.{h,cc} "
        "makes stronger assumptions than the rest of the Kaldi toolkit: "
        "that is, it assumes you have word-position-dependent phones, "
        "with disjoint subsets of phones for (word-begin, word-end, "
        "word-internal, word-begin-and-end), and of course silence, "
        "which is assumed not to be inside a word [it will just print "
        "a warning if it is, though, and should give the right output "
        "as long as it's not at the beginning or end of a word].");
    word_boundary_info_opts.def(py::init<>())
      .def_readwrite("wbegin_phones", &PyClass::wbegin_phones)
      .def_readwrite("wend_phones", &PyClass::wend_phones)
      .def_readwrite("wbegin_and_end_phones", &PyClass::wbegin_and_end_phones)
      .def_readwrite("winternal_phones", &PyClass::winternal_phones)
      .def_readwrite("silence_phones", &PyClass::silence_phones)
      .def_readwrite("silence_label", &PyClass::silence_label)
      .def_readwrite("partial_word_label", &PyClass::partial_word_label)
      .def_readwrite("reorder", &PyClass::reorder)
      .def_readwrite("silence_may_be_word_internal", &PyClass::silence_may_be_word_internal)
      .def_readwrite("silence_has_olabels", &PyClass::silence_has_olabels);
  }
  {
    using PyClass = WordBoundaryInfoNewOpts;

    auto word_boundary_info_new_opts = py::class_<PyClass>(
        m, "WordBoundaryInfoNewOpts",
        "This structure is to be used for newer code, from s5 scripts on.");
    word_boundary_info_new_opts.def(py::init<>())
      .def_readwrite("silence_label", &PyClass::silence_label)
      .def_readwrite("partial_word_label", &PyClass::partial_word_label)
      .def_readwrite("reorder", &PyClass::reorder);
  }
  {
    using PyClass = WordBoundaryInfo;

    auto word_boundary_info = py::class_<PyClass>(
        m, "WordBoundaryInfo");
    word_boundary_info.def(py::init<const WordBoundaryInfoNewOpts &>(),
        py::arg("opts"))
      .def(py::init<const WordBoundaryInfoNewOpts &, std::string>(),
        py::arg("opts"),
        py::arg("word_boundary_file"))
      .def("Init", &PyClass::Init,
        py::arg("stream"))
      .def("TypeOfPhone", &PyClass::TypeOfPhone,
        py::arg("p"));
  py::enum_<WordBoundaryInfo::PhoneType>(word_boundary_info, "PhoneType")
    .value("kNoPhone", WordBoundaryInfo::PhoneType::kNoPhone)
    .value("kWordBeginPhone", WordBoundaryInfo::PhoneType::kWordBeginPhone)
    .value("kWordEndPhone", WordBoundaryInfo::PhoneType::kWordEndPhone)
    .value("kSkipNGkWordBeginAndEndPhoneram", WordBoundaryInfo::PhoneType::kWordBeginAndEndPhone)
    .value("kWordInternalPhone", WordBoundaryInfo::PhoneType::kWordInternalPhone)
    .value("kNonWordPhone", WordBoundaryInfo::PhoneType::kNonWordPhone)
    .export_values();
  }
  m.def("WordAlignLattice",
        &WordAlignLattice,
        "Align lattice so that each arc has the transition-ids on it "
        "that correspond to the word that is on that arc.  [May also have "
        "epsilon arcs for optional silences.] "
        "Returns true if everything was OK, false if some kind of "
        "error was detected (e.g. the words didn't have the kinds of "
        "sequences we would expect if the WordBoundaryInfo was "
        "correct).  Note: we don't expect silence inside words, "
        "or empty words (words with no phones), and we expect "
        "the word to start with a wbegin_phone, to end with "
        "a wend_phone, and to possibly have winternal_phones "
        "inside (or to consist of just one wbegin_and_end_phone). "
        "Note: if it returns false, it doesn't mean the lattice "
        "that the output is necessarily bad: it might just be that "
        "the lattice was \"forced out\" as the end-state was not "
        "reached during decoding, and in this case the output might "
        "be usable. "
        " If max_states > 0, if this code detects that the #states "
        "of the output will be greater than max_states, it will "
        "abort the computation, return false and produce an empty "
        "lattice out.",
        py::arg("lat"),
        py::arg("tmodel"),
        py::arg("info"),
        py::arg("max_states"),
        py::arg("lat_out"));
}

void pybind_determinize_lattice_pruned(py::module& m) {
  using namespace fst;
  {
    using PyClass = fst::DeterminizeLatticePrunedOptions;

    py::class_<PyClass>(m, "DeterminizeLatticePrunedOptions")
        .def(py::init<>())
        .def_readwrite("delta", &PyClass::delta,
                       "A small offset used to measure equality of weights.")
        .def_readwrite("max_mem", &PyClass::max_mem,
                       "If >0, determinization will fail and return false when "
                       "the algorithm's (approximate) memory consumption "
                       "crosses this threshold.")
        .def_readwrite("max_loop", &PyClass::max_loop,
                       "If >0, can be used to detect non-determinizable input "
                       "(a case that wouldn't be caught by max_mem).")
        .def_readwrite("max_states", &PyClass::max_states)
        .def_readwrite("max_arcs", &PyClass::max_arcs)
        .def_readwrite("retry_cutoff", &PyClass::retry_cutoff)
        .def("__str__", [](const PyClass& opt) {
          std::ostringstream os;
          os << "delta: " << opt.delta << "\n";
          os << "max_mem: " << opt.max_mem << "\n";
          os << "max_loop: " << opt.max_loop << "\n";
          os << "max_states: " << opt.max_states << "\n";
          os << "max_arcs: " << opt.max_arcs << "\n";
          os << "retry_cutoff: " << opt.retry_cutoff << "\n";
          return os.str();
        });
  }

  {
    using PyClass = fst::DeterminizeLatticePhonePrunedOptions;

    py::class_<PyClass>(m, "DeterminizeLatticePhonePrunedOptions")
        .def(py::init<>())
        .def_readwrite("delta", &PyClass::delta,
                       "A small offset used to measure equality of weights.")
        .def_readwrite("max_mem", &PyClass::max_mem,
                       "If >0, determinization will fail and return false when "
                       "the algorithm's (approximate) memory consumption "
                       "crosses this threshold.")
        .def_readwrite("phone_determinize", &PyClass::phone_determinize,
                       "phone_determinize: if true, do a first pass "
                       "determinization on both phones and words.")
        .def_readwrite("word_determinize", &PyClass::word_determinize,
                       "word_determinize: if true, do a second pass "
                       "determinization on words only.")
        .def_readwrite(
            "minimize", &PyClass::minimize,
            "minimize: if true, push and minimize after determinization.")
        .def("__str__", [](const PyClass& opts) {
          std::ostringstream os;
          os << "delta: " << opts.delta << "\n";
          os << "max_mem: " << opts.max_mem << "\n";
          os << "phone_determinize: " << opts.phone_determinize << "\n";
          os << "word_determinize: " << opts.word_determinize << "\n";
          os << "minimize: " << opts.minimize << "\n";
          return os.str();
        });
  }
  m.def("DeterminizeLatticePruned",
        py::overload_cast<const ExpandedFst<kaldi::LatticeArc> &,
    double,
    MutableFst<kaldi::LatticeArc> *,
    DeterminizeLatticePrunedOptions>(&DeterminizeLatticePruned<kaldi::LatticeWeight>),
        "This function implements the normal version of DeterminizeLattice, in which the "
        "output strings are represented using sequences of arcs, where all but the "
        "first one has an epsilon on the input side.  It also prunes using the beam "
        "in the \"prune\" parameter.  The input FST must be topologically sorted in order "
        "for the algorithm to work. For efficiency it is recommended to sort ilabel as well. "
        "Returns true on success, and false if it had to terminate the determinization "
        "earlier than specified by the \"prune\" beam-- that is, if it terminated because "
        "of the max_mem, max_loop or max_arcs constraints in the options. "
        "CAUTION: you may want to use the version below which outputs to CompactLattice.",
        py::arg("ifst"),
        py::arg("prune"),
        py::arg("ofst"),
        py::arg("opts") = DeterminizeLatticePrunedOptions());
  m.def("DeterminizeLatticePruned",
        py::overload_cast<const ExpandedFst<kaldi::LatticeArc> &,
                      double,
                      MutableFst<kaldi::LatticeArc> *,
                      DeterminizeLatticePrunedOptions>(&DeterminizeLatticePruned<kaldi::LatticeWeight>),
        "This function implements the normal version of DeterminizeLattice, in which the "
        "output strings are represented using sequences of arcs, where all but the "
        "first one has an epsilon on the input side.  It also prunes using the beam "
        "in the \"prune\" parameter.  The input FST must be topologically sorted in order "
        "for the algorithm to work. For efficiency it is recommended to sort ilabel as well. "
        "Returns true on success, and false if it had to terminate the determinization "
        "earlier than specified by the \"prune\" beam-- that is, if it terminated because "
        "of the max_mem, max_loop or max_arcs constraints in the options. "
        "CAUTION: you may want to use the version below which outputs to CompactLattice.",
        py::arg("ifst"),
        py::arg("prune"),
        py::arg("ofst"),
        py::arg("opts") = DeterminizeLatticePrunedOptions());
  m.def("DeterminizeLatticePhonePruned",
        py::overload_cast<const kaldi::TransitionInformation &,
    const ExpandedFst<kaldi::LatticeArc> &,
    double,
    MutableFst<kaldi::CompactLatticeArc> *,
    DeterminizeLatticePhonePrunedOptions >(&DeterminizeLatticePhonePruned<kaldi::LatticeWeight, int32>),
        "This function is a wrapper of DeterminizeLatticePhonePrunedFirstPass() and "
        "DeterminizeLatticePruned(). If --phone-determinize is set to true, it first "
        "calls DeterminizeLatticePhonePrunedFirstPass() to do the initial pass of "
        "determinization on the phone + word lattices. If --word-determinize is set "
        "true, it then does a second pass of determinization on the word lattices by "
        "calling DeterminizeLatticePruned(). If both are set to false, then it gives "
        "a warning and copying the lattices without determinization. "
        "\n"
        "Note: the point of doing first a phone-level determinization pass and then "
        "a word-level determinization pass is that it allows us to determinize "
        "deeper lattices without \"failing early\" and returning a too-small lattice "
        "due to the max-mem constraint.  The result should be the same as word-level "
        "determinization in general, but for deeper lattices it is a bit faster, "
        "despite the fact that we now have two passes of determinization by default.",
        py::arg("trans_model"),
        py::arg("ifst"),
        py::arg("prune"),
        py::arg("ofst"),
        py::arg("opts")= DeterminizeLatticePhonePrunedOptions());
  m.def("DeterminizeLatticePhonePruned",
        py::overload_cast<const kaldi::TransitionInformation &,
    MutableFst<kaldi::LatticeArc> *,
    double,
    MutableFst<kaldi::CompactLatticeArc> *,
    DeterminizeLatticePhonePrunedOptions >(&DeterminizeLatticePhonePruned<kaldi::LatticeWeight, int32>),
        "This function is a wrapper of DeterminizeLatticePhonePrunedFirstPass() and "
        "DeterminizeLatticePruned(). If --phone-determinize is set to true, it first "
        "calls DeterminizeLatticePhonePrunedFirstPass() to do the initial pass of "
        "determinization on the phone + word lattices. If --word-determinize is set "
        "true, it then does a second pass of determinization on the word lattices by "
        "calling DeterminizeLatticePruned(). If both are set to false, then it gives "
        "a warning and copying the lattices without determinization. "
        "\n"
        "Note: the point of doing first a phone-level determinization pass and then "
        "a word-level determinization pass is that it allows us to determinize "
        "deeper lattices without \"failing early\" and returning a too-small lattice "
        "due to the max-mem constraint.  The result should be the same as word-level "
        "determinization in general, but for deeper lattices it is a bit faster, "
        "despite the fact that we now have two passes of determinization by default.",
        py::arg("trans_model"),
        py::arg("ifst"),
        py::arg("prune"),
        py::arg("ofst"),
        py::arg("opts")= DeterminizeLatticePhonePrunedOptions());
  /*m.def("DeterminizeLatticeInsertPhones",
        &DeterminizeLatticeInsertPhones<kaldi::LatticeWeight>,
        "This function takes in lattices and inserts phones at phone boundaries. It "
        "uses the transition model to work out the transition_id to phone map. The "
        "returning value is the starting index of the phone label. Typically we pick "
        "(maximum_output_label_index + 1) as this value. The inserted phones are then "
        "mapped to (returning_value + original_phone_label) in the new lattice. The "
        "returning value will be used by DeterminizeLatticeDeletePhones() where it "
        "works out the phones according to this value.",
        py::arg("trans_model"),
        py::arg("fst"));
      */
  m.def("DeterminizeLatticeDeletePhones",
        &DeterminizeLatticeDeletePhones<kaldi::LatticeWeight>,
        "This function takes in lattices and deletes \"phones\" from them. The \"phones\" "
        "here are actually any label that is larger than first_phone_label because "
        "when we insert phones into the lattice, we map the original phone label to "
        "(first_phone_label + original_phone_label). It is supposed to be used "
        "together with DeterminizeLatticeInsertPhones()",
        py::arg("first_phone_label"),
        py::arg("fst"));
  m.def("DeterminizeLatticePhonePrunedWrapper",
        &DeterminizeLatticePhonePrunedWrapper,
        "This function is a wrapper of DeterminizeLatticePhonePruned() that works for "
        "Lattice type FSTs.  It simplifies the calling process by calling "
        "TopSort() Invert() and ArcSort() for you. "
        "Unlike other determinization routines, the function "
        "requires \"ifst\" to have transition-id's on the input side and words on the "
        "output side. "
        "This function can be used as the top-level interface to all the determinization "
        "code.",
        py::arg("trans_model"),
        py::arg("ifst"),
        py::arg("prune"),
        py::arg("ofst"),
        py::arg("opts") = DeterminizeLatticePhonePrunedOptions(),
      py::call_guard<py::gil_scoped_release>());
}

void pybind_kaldi_lattice(py::module& m) {
  pybind_arc_impl<LatticeWeight>(m, "LatticeArc");
  pybind_fst_impl<LatticeArc>(m, "LatticeBase");
  pybind_expanded_fst_impl<LatticeArc>(m, "LatticeExpandedBase");
  pybind_mutable_fst_impl<LatticeArc>(m, "LatticeMutableBase");
  pybind_lattice_fst_impl<LatticeArc>(m, "Lattice");
  pybind_state_iterator_impl<Lattice>(m, "LatticeStateIterator");
  pybind_arc_iterator_impl<Lattice>(m, "LatticeArcIterator");
  pybind_mutable_arc_iterator_impl<Lattice>(m, "LatticeMutableArcIterator");

  pybind_arc_impl<CompactLatticeWeight>(m, "CompactLatticeArc");
  pybind_fst_impl<CompactLatticeArc>(m, "CompactLatticeBase");
  pybind_expanded_fst_impl<CompactLatticeArc>(m, "CompactLatticeExpandedBase");
  pybind_mutable_fst_impl<CompactLatticeArc>(m, "CompactLatticeMutableBase");
  pybind_lattice_fst_impl<CompactLatticeArc>(m, "CompactLattice");
  pybind_state_iterator_impl<CompactLattice>(m, "CompactLatticeStateIterator");
  pybind_arc_iterator_impl<CompactLattice>(m, "CompactLatticeArcIterator");
  pybind_mutable_arc_iterator_impl<CompactLattice>(
      m, "CompactLatticeMutableArcIterator");

  m.def("WriteLattice", &WriteLattice, py::arg("os"), py::arg("binary"),
        py::arg("lat"));

  m.def("WriteCompactLattice", &WriteCompactLattice, py::arg("os"),
        py::arg("binary"), py::arg("clat"));

  m.def("ReadLattice",
        [](std::istream& is, bool binary) -> Lattice* {
          Lattice* p = nullptr;
          bool ret = ReadLattice(is, binary, &p);
          if (!ret) {
            KALDI_ERR << "Failed to read lattice";
          }
          return p;
          // NOTE(fangjun): p points to a memory area allocated by `operator
          // new`; we ask python to take the ownership of the allocated memory
          // which will finally invokes `operator delete`.
          //
          // Refer to
          // https://pybind11-rtdtest.readthedocs.io/en/stable/advanced.html
          // for the explanation of `return_value_policy::take_ownership`.
        },
        py::arg("is"), py::arg("binary"),
        py::return_value_policy::take_ownership);

  m.def("ReadCompactLattice",
        [](std::istream& is, bool binary) -> CompactLattice* {
          CompactLattice* p = nullptr;
          bool ret = ReadCompactLattice(is, binary, &p);
          if (!ret) {
            KALDI_ERR << "Failed to read compact lattice";
          }
          return p;
        },
        py::arg("is"), py::arg("binary"),
        py::return_value_policy::take_ownership);

  {
    using PyClass = LatticeHolder;
    py::class_<PyClass>(m, "LatticeHolder")
        .def(py::init<>())
        .def_static("Write", &PyClass::Write, py::arg("os"), py::arg("binary"),
                    py::arg("t"),
      py::call_guard<py::gil_scoped_release>())
        .def("Read", &PyClass::Read, py::arg("is"),
      py::call_guard<py::gil_scoped_release>())
        .def_static("IsReadInBinary", &PyClass::IsReadInBinary)
        .def("Value", &PyClass::Value, py::return_value_policy::reference,
      py::call_guard<py::gil_scoped_release>())
        .def("Clear", &PyClass::Clear);
    // TODO(fangjun): other methods can be wrapped when needed
  }
  {
    using PyClass = CompactLatticeHolder;
    py::class_<PyClass>(m, "CompactLatticeHolder")
        .def(py::init<>())
        .def_static("Write", &PyClass::Write, py::arg("os"), py::arg("binary"),
                    py::arg("t"),
      py::call_guard<py::gil_scoped_release>())
        .def("Read", &PyClass::Read, py::arg("is"),
      py::call_guard<py::gil_scoped_release>())
        .def_static("IsReadInBinary", &PyClass::IsReadInBinary)
        .def("Value", &PyClass::Value, py::return_value_policy::reference,
      py::call_guard<py::gil_scoped_release>())
        .def("Clear", &PyClass::Clear);
    // TODO(fangjun): other methods can be wrapped when needed
  }

  pybind_sequential_table_reader<LatticeHolder>(m, "SequentialLatticeReader");

  pybind_random_access_table_reader<LatticeHolder>(
      m, "RandomAccessLatticeReader");

  pybind_table_writer<LatticeHolder>(m, "LatticeWriter");

  pybind_sequential_table_reader<CompactLatticeHolder>(
      m, "SequentialCompactLatticeReader");

  pybind_random_access_table_reader<CompactLatticeHolder>(
      m, "RandomAccessCompactLatticeReader");

  pybind_table_writer<CompactLatticeHolder>(m, "CompactLatticeWriter");
}


void init_lat(py::module &_m) {
  py::module m = _m.def_submodule("lat", "lat pybind for Kaldi");
    pybind_arctic_weight(m);
    pybind_compose_lattice_pruned(m);
    pybind_confidence(m);
    pybind_kaldi_lattice(m);
    pybind_determinize_lattice_pruned(m);
    pybind_kaldi_functions_transition_model(m);
    pybind_lat_kaldi_functions(m);
    pybind_lat_minimize_lattice(m);
    pybind_lat_phone_align_lattice(m);
    pybind_lat_push_lattice(m);
    pybind_lat_sausages(m);
    pybind_lat_word_align_lattice_lexicon(m);
    pybind_lat_word_align_lattice(m);

    m.def("linear_to_lattice",
            [](
            const std::vector<int32> &ali,
                           const std::vector<int32> &words,
                           BaseFloat lm_cost = 0.0,
                           BaseFloat ac_cost = 0.0
            ) -> CompactLattice {
                  Lattice lat_out;
                  CompactLattice clat;
            typedef LatticeArc::StateId StateId;
            typedef LatticeArc::Weight Weight;
            typedef LatticeArc::Label Label;
            lat_out.DeleteStates();
            StateId cur_state = lat_out.AddState(); // will be 0.
            lat_out.SetStart(cur_state);
            for (size_t i = 0; i < ali.size() || i < words.size(); i++) {
            Label ilabel = (i < ali.size()  ? ali[i] : 0);
            Label olabel = (i < words.size()  ? words[i] : 0);
            StateId next_state = lat_out.AddState();
            lat_out.AddArc(cur_state,
                              LatticeArc(ilabel, olabel, Weight::One(), next_state));
            cur_state = next_state;
            }
            lat_out.SetFinal(cur_state, Weight(lm_cost, ac_cost));
            ConvertLattice(lat_out, &clat);
            return clat;
            },
                  py::arg("ali"),
                  py::arg("words"),
                  py::arg("lm_cost") = 0.0,
                  py::arg("ac_cost") = 0.0);

    m.def("word_align_lattice_lexicon",
            [](
            const CompactLattice &clat,
            const TransitionModel &tmodel,
            const WordAlignLatticeLexiconInfo &lexicon_info,
            const WordAlignLatticeLexiconOpts &opts
            ) {

      CompactLattice aligned_clat;
      bool ok = WordAlignLatticeLexicon(clat, tmodel, lexicon_info, opts,
                                        &aligned_clat);
            return py::make_tuple(ok, aligned_clat);
            },
                  py::arg("clat"),
                  py::arg("tmodel"),
                  py::arg("lexicon_info"),
                  py::arg("opts"));

    m.def("lattice_best_path",
            [](
            CompactLattice &clat,
            BaseFloat lm_scale = 1.0,
            BaseFloat acoustic_scale = 1.0
            ) {

      fst::ScaleLattice(fst::LatticeScale(lm_scale, acoustic_scale), &clat);
      CompactLattice clat_best_path;
      CompactLatticeShortestPath(clat, &clat_best_path);  // A specialized
      // implementation of shortest-path for CompactLattice.
      Lattice best_path;
      ConvertLattice(clat_best_path, &best_path);
      return best_path;

            },
                  py::arg("clat"),
                  py::arg("lm_scale") = 1.0,
                  py::arg("acoustic_scale") = 1.0);

    m.def("lattice_to_post",
            [](
            CompactLattice &clat,
            BaseFloat beam = 10.0,
            BaseFloat acoustic_scale = 1.0,
            bool minimize = false
            ) {
                  fst::DeterminizeLatticePrunedOptions opts;
                  opts.max_mem = 50000000;
                  opts.max_loop = 0; // was 500000;
                  Lattice lat;
                  ConvertLattice(clat, &lat);
                  Invert(&lat);
                  fst::ScaleLattice(fst::AcousticLatticeScale(acoustic_scale), &lat);
                  TopSort(&lat);
                  fst::ArcSort(&lat, fst::ILabelCompare<LatticeArc>());
                  CompactLattice det_clat;
                  DeterminizeLatticePruned(lat, beam, &det_clat, opts);
                  fst::Connect(&det_clat);
                  if (minimize) {
                  PushCompactLatticeStrings(&det_clat);
                  PushCompactLatticeWeights(&det_clat);
                  MinimizeCompactLattice(&det_clat);
                  }
                  TopSortCompactLatticeIfNeeded(&det_clat);
                  fst::ScaleLattice(fst::AcousticLatticeScale(1.0/acoustic_scale), &det_clat);
                  return det_clat;
            },
                  py::arg("clat"),
                  py::arg("beam") = 10.0,
                  py::arg("acoustic_scale") = 1.0,
                  py::arg("minimize") = false,
           py::return_value_policy::take_ownership);


    m.def("lattice_determinize_pruned",
            [](
            const Lattice &lat
            ) {
            Posterior post;
            double lat_like = LatticeForwardBackward(lat, &post);
            return py::make_tuple(post, lat_like);
            },
                  py::arg("lat"),
           py::return_value_policy::take_ownership);

    m.def("lm_rescore",
            [](
                  CompactLattice &clat,
                  VectorFst<StdArc> *lm_to_subtract_fst,
                  VectorFst<StdArc> *lm_to_add_fst,
            const ComposeLatticePrunedOptions &compose_opts,
                  BaseFloat lm_scale = 1.0,
             BaseFloat acoustic_scale = 1.0
            ) {
                  if (lm_to_subtract_fst->Properties(fst::kAcceptor, true) == 0) {
                  // If it's not already an acceptor, project on the output, i.e. copy olabels
                  // to ilabels.  Generally the G.fst's on disk will have the disambiguation
                  // symbol #0 on the input symbols of the backoff arc, and projection will
                  // replace them with epsilons which is what is on the output symbols of
                  // those arcs.
                  fst::Project(lm_to_subtract_fst, fst::PROJECT_OUTPUT);
                  }
                  if (lm_to_subtract_fst->Properties(fst::kILabelSorted, true) == 0) {
                  // Make sure LM is sorted on ilabel.
                  fst::ILabelCompare<fst::StdArc> ilabel_comp;
                  fst::ArcSort(lm_to_subtract_fst, ilabel_comp);
                  }
                  fst::BackoffDeterministicOnDemandFst<StdArc> lm_to_subtract_det_backoff(
                        *lm_to_subtract_fst);
                  fst::ScaleDeterministicOnDemandFst lm_to_subtract_det_scale(
                        -lm_scale, &lm_to_subtract_det_backoff);


                  if (lm_to_add_fst->Properties(fst::kAcceptor, true) == 0) {
                  // If it's not already an acceptor, project on the output, i.e. copy olabels
                  // to ilabels.  Generally the G.fst's on disk will have the disambiguation
                  // symbol #0 on the input symbols of the backoff arc, and projection will
                  // replace them with epsilons which is what is on the output symbols of
                  // those arcs.
                  fst::Project(lm_to_add_fst, fst::PROJECT_OUTPUT);
                  }
                  if (lm_to_add_fst->Properties(fst::kILabelSorted, true) == 0) {
                  // Make sure LM is sorted on ilabel.
                  fst::ILabelCompare<fst::StdArc> ilabel_comp;
                  fst::ArcSort(lm_to_add_fst, ilabel_comp);
                  }

                  fst::BackoffDeterministicOnDemandFst<StdArc>
                        lm_to_add(
                        *lm_to_add_fst);

                  if (acoustic_scale != 1.0) {
                  fst::ScaleLattice(fst::AcousticLatticeScale(acoustic_scale), &clat);
                  }
                  TopSortCompactLatticeIfNeeded(&clat);
                  fst::ComposeDeterministicOnDemandFst<StdArc> combined_lms(
                  &lm_to_subtract_det_scale, &lm_to_add);
                  CompactLattice composed_clat;
                  ComposeCompactLatticePruned(compose_opts,
                                          clat,
                                          &combined_lms,
                                          &composed_clat);

                  if (composed_clat.NumStates() > 0) {

                        if (acoustic_scale != 1.0) {
                              fst::ScaleLattice(fst::AcousticLatticeScale(1.0 / acoustic_scale),
                                          &composed_clat);
                        }
                  }
                  return composed_clat;

            },
                  py::arg("clat"),
                  py::arg("lm_to_subtract_fst"),
                  py::arg("lm_to_add_fst"),
                  py::arg("compose_opts"),
                  py::arg("lm_scale") = 1.0,
                  py::arg("acoustic_scale") = 1.0);

    m.def("lm_rescore_carpa",
            [](
                  CompactLattice &clat,
                  VectorFst<StdArc> *lm_to_subtract_fst,
                  ConstArpaLm &const_arpa,
            const ComposeLatticePrunedOptions &compose_opts,
                  BaseFloat lm_scale = 1.0,
             BaseFloat acoustic_scale = 1.0
            ) {
                  if (lm_to_subtract_fst->Properties(fst::kAcceptor, true) == 0) {
                  // If it's not already an acceptor, project on the output, i.e. copy olabels
                  // to ilabels.  Generally the G.fst's on disk will have the disambiguation
                  // symbol #0 on the input symbols of the backoff arc, and projection will
                  // replace them with epsilons which is what is on the output symbols of
                  // those arcs.
                  fst::Project(lm_to_subtract_fst, fst::PROJECT_OUTPUT);
                  }
                  if (lm_to_subtract_fst->Properties(fst::kILabelSorted, true) == 0) {
                  // Make sure LM is sorted on ilabel.
                  fst::ILabelCompare<fst::StdArc> ilabel_comp;
                  fst::ArcSort(lm_to_subtract_fst, ilabel_comp);
                  }
                  fst::BackoffDeterministicOnDemandFst<StdArc> lm_to_subtract_det_backoff(
                        *lm_to_subtract_fst);
                  fst::ScaleDeterministicOnDemandFst lm_to_subtract_det_scale(
                        -lm_scale, &lm_to_subtract_det_backoff);


                  ConstArpaLmDeterministicFst lm_to_add(const_arpa);

                  if (acoustic_scale != 1.0) {
                  fst::ScaleLattice(fst::AcousticLatticeScale(acoustic_scale), &clat);
                  }
                  TopSortCompactLatticeIfNeeded(&clat);
                  fst::ComposeDeterministicOnDemandFst<StdArc> combined_lms(
                  &lm_to_subtract_det_scale, &lm_to_add);
                  CompactLattice composed_clat;
                  ComposeCompactLatticePruned(compose_opts,
                                          clat,
                                          &combined_lms,
                                          &composed_clat);

                  if (composed_clat.NumStates() > 0) {

                        if (acoustic_scale != 1.0) {
                              fst::ScaleLattice(fst::AcousticLatticeScale(1.0 / acoustic_scale),
                                          &composed_clat);
                        }
                  }
                  return composed_clat;

            },
                  py::arg("clat"),
                  py::arg("lm_to_subtract_fst"),
                  py::arg("const_arpa"),
                  py::arg("compose_opts"),
                  py::arg("lm_scale") = 1.0,
                  py::arg("acoustic_scale") = 1.0);

    m.def("load_fst_for_subtract",
            [](
                  VectorFst<StdArc> *lm_to_subtract_fst,
                  BaseFloat lm_scale = 1.0
            ) {
                  if (lm_to_subtract_fst->Properties(fst::kAcceptor, true) == 0) {
                  // If it's not already an acceptor, project on the output, i.e. copy olabels
                  // to ilabels.  Generally the G.fst's on disk will have the disambiguation
                  // symbol #0 on the input symbols of the backoff arc, and projection will
                  // replace them with epsilons which is what is on the output symbols of
                  // those arcs.
                  fst::Project(lm_to_subtract_fst, fst::PROJECT_OUTPUT);
                  }
                  if (lm_to_subtract_fst->Properties(fst::kILabelSorted, true) == 0) {
                  // Make sure LM is sorted on ilabel.
                  fst::ILabelCompare<fst::StdArc> ilabel_comp;
                  fst::ArcSort(lm_to_subtract_fst, ilabel_comp);
                  }
                  fst::BackoffDeterministicOnDemandFst<StdArc> lm_to_subtract_det_backoff(
                        *lm_to_subtract_fst);
                  fst::ScaleDeterministicOnDemandFst lm_to_subtract_det_scale(
                        -lm_scale, &lm_to_subtract_det_backoff);
                  return lm_to_subtract_det_scale;
            },
                  py::arg("lm_to_subtract_fst"),
                  py::arg("lm_scale") = 1.0);

    m.def("load_fst_for_add",
            [](
                  VectorFst<StdArc> *lm_to_add_fst
            ) {
                  BaseFloat lm_scale = 1.0;
                  if (lm_to_add_fst->Properties(fst::kAcceptor, true) == 0) {
                  // If it's not already an acceptor, project on the output, i.e. copy olabels
                  // to ilabels.  Generally the G.fst's on disk will have the disambiguation
                  // symbol #0 on the input symbols of the backoff arc, and projection will
                  // replace them with epsilons which is what is on the output symbols of
                  // those arcs.
                  fst::Project(lm_to_add_fst, fst::PROJECT_OUTPUT);
                  }
                  if (lm_to_add_fst->Properties(fst::kILabelSorted, true) == 0) {
                  // Make sure LM is sorted on ilabel.
                  fst::ILabelCompare<fst::StdArc> ilabel_comp;
                  fst::ArcSort(lm_to_add_fst, ilabel_comp);
                  }
                  fst::BackoffDeterministicOnDemandFst<StdArc>
                        lm_to_add(
                        *lm_to_add_fst);
                  return lm_to_add;
            },
                  py::arg("lm_to_add_fst"));
}
