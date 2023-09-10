
#include "decoder/pybind_decoder.h"

#include "decoder/biglm-faster-decoder.h"
#include "decoder/decodable-mapped.h"
#include "decoder/decodable-matrix.h"
#include "decoder/decodable-sum.h"
#include "decoder/faster-decoder.h"
#include "decoder/grammar-fst.h"
#include "decoder/lattice-biglm-faster-decoder.h"
#include "decoder/decoder-wrappers.h"
#include "decoder/lattice-faster-decoder.h"
#include "decoder/lattice-faster-online-decoder.h"
#include "decoder/lattice-incremental-decoder.h"
#include "decoder/lattice-incremental-online-decoder.h"
#include "decoder/lattice-simple-decoder.h"
#include "decoder/simple-decoder.h"
#include "decoder/training-graph-compiler.h"
#include "fst/fst.h"
#include "hmm/transition-model.h"
#include "matrix/kaldi-matrix.h"
#include "gmm/diag-gmm.h"
#include "gmm/decodable-am-diag-gmm.h"
#include "lat/lattice-functions.h"

using namespace kaldi;
using namespace fst;


void pybind_decoder_biglm_faster_decoder(py::module& m) {

  {
    using PyClass = BiglmFasterDecoderOptions;

    auto biglm_faster_decoder_options = py::class_<BiglmFasterDecoderOptions, FasterDecoderOptions>(
        m, "BiglmFasterDecoderOptions");
    biglm_faster_decoder_options.def(py::init<>());
  }
  {
    using PyClass = BiglmFasterDecoder;

    auto biglm_faster_decoder = py::class_<PyClass>(
        m, "BiglmFasterDecoder");
    biglm_faster_decoder
      .def(py::init<const fst::Fst<fst::StdArc> &,
                     const BiglmFasterDecoderOptions &,
                     fst::DeterministicOnDemandFst<fst::StdArc> *>(),
                       py::arg("fst"),
                       py::arg("opts"),
                       py::arg("lm_diff_fst"))
      .def("SetOptions",
        &PyClass::SetOptions,
                       py::arg("opts"))
      .def("Decode",
        &PyClass::Decode,
                       py::arg("decodable"))
      .def("ReachedFinal",
        &PyClass::ReachedFinal)
      .def("GetBestPath",
        &PyClass::GetBestPath,
        "GetBestPath gets the decoding output.  If \"use_final_probs\" is true "
          "AND we reached a final state, it limits itself to final states; "
          "otherwise it gets the most likely token not taking into "
          "account final-probs.  fst_out will be empty (Start() == kNoStateId) if "
          "nothing was available.  It returns true if it got output (thus, fst_out "
          "will be nonempty).",
                       py::arg("fst_out"),
                       py::arg("use_final_probs") = true);
  }
}

void pybind_decoder_decodable_mapped(py::module& m) {

  {
    using PyClass = DecodableMapped;

    auto decodable_mapped = py::class_<DecodableMapped, DecodableInterface>(
        m, "DecodableMapped");
    decodable_mapped
      .def(py::init<const std::vector<int32> &, DecodableInterface *>(),
                       py::arg("index_map"),
                       py::arg("d"));
  }
}

void pybind_decoder_decodable_sum(py::module& m) {

  {
    using PyClass = DecodableSum;

    auto decodable_sum = py::class_<DecodableSum, DecodableInterface>(
        m, "DecodableSum");
    decodable_sum
      .def(py::init<DecodableInterface *, BaseFloat ,
               DecodableInterface *, BaseFloat >(),
                       py::arg("d1"),
                       py::arg("w1"),
                       py::arg("d2"),
                       py::arg("w2"))
      .def(py::init<const std::vector<std::pair<DecodableInterface*, BaseFloat> > &>(),
                       py::arg("decodables"));
  }
  {
    using PyClass = DecodableSumScaled;

    auto decodable_sum_scaled = py::class_<DecodableSumScaled, DecodableSum>(
        m, "DecodableSumScaled");
    decodable_sum_scaled
      .def(py::init<DecodableInterface *, BaseFloat ,
               DecodableInterface *, BaseFloat,
                     BaseFloat  >(),
                       py::arg("d1"),
                       py::arg("w1"),
                       py::arg("d2"),
                       py::arg("w2"),
                       py::arg("scale"))
      .def(py::init<const std::vector<std::pair<DecodableInterface*, BaseFloat> > &,
                     BaseFloat  >(),
                       py::arg("decodables"),
                       py::arg("scale"));
  }
}

void pybind_decoder_wrappers(py::module& m) {

  {
    using PyClass = AlignConfig;

    auto align_config = py::class_<PyClass>(
        m, "AlignConfig");
    align_config.def(py::init<>())
      .def_readwrite("beam", &PyClass::beam)
      .def_readwrite("retry_beam", &PyClass::retry_beam)
      .def_readwrite("careful", &PyClass::careful)
      .def(py::pickle(
        [](const PyClass &p) { // __getstate__
            /* Return a tuple that fully encodes the state of the object */
            return py::make_tuple(
                p.beam,
                p.retry_beam,
                p.careful);
        },
        [](py::tuple t) { // __setstate__
            if (t.size() != 3)
                throw std::runtime_error("Invalid state!");

            /* Create a new C++ instance */
            PyClass opts;

            /* Assign any additional state */
            opts.beam = t[0].cast<BaseFloat>();
            opts.retry_beam = t[1].cast<BaseFloat>();
            opts.careful = t[2].cast<bool>();

            return opts;
        }
    ));
  }

  m.def("AlignUtteranceWrapper",
        &AlignUtteranceWrapper,
        "AlignUtteranceWapper is a wrapper for alignment code used in training, that "
          "is called from many different binaries, e.g. gmm-align, gmm-align-compiled, "
          "sgmm-align, etc.  The writers for alignments and words will only be written "
          "to if they are open.  The num_done, num_error, num_retried, tot_like and "
          "frame_count pointers will (if non-NULL) be incremented or added to, not set, "
          "by this function.",
        py::arg("config"),
        py::arg("utt"),
        py::arg("acoustic_scale"),
        py::arg("fst"),
        py::arg("decodable"),
        py::arg("alignment_writer"),
        py::arg("scores_writer"),
        py::arg("num_done"),
        py::arg("num_error"),
        py::arg("num_retried"),
        py::arg("tot_like"),
        py::arg("frame_count"),
        py::arg("per_frame_acwt_writer")=NULL);

  m.def("ModifyGraphForCarefulAlignment",
        &ModifyGraphForCarefulAlignment,
        "This function modifies the decoding graph for what we call \"careful "
          "alignment\".  The problem we are trying to solve is that if the decoding eats "
          "up the words in the graph too fast, it can get stuck at the end, and produce "
          "what looks like a valid alignment even though there was really a failure. "
          "So what we want to do is to introduce, after the final-states of the graph, "
          "a \"blind alley\" with no final-probs reachable, where the decoding can go to "
          "get lost.  Our basic idea is to append the decoding-graph to itself using "
          "the fst Concat operation; but in order that there should be final-probs at the end of "
          "the first but not the second FST, we modify the right-hand argument to the "
          "Concat operation so that it has none of the original final-probs, and add "
          "a \"pre-initial\" state that is final.",
        py::arg("fst"),
      py::call_guard<py::gil_scoped_release>());
  {
    using PyClass = DecodeUtteranceLatticeFasterClass;

    auto decode_utterance_lattice_faster_class = py::class_<PyClass>(
        m, "DecodeUtteranceLatticeFasterClass",
        "This class basically does the same job as the function "
          "DecodeUtteranceLatticeFaster, but in a way that allows us "
          "to build a multi-threaded command line program more easily. "
          "The main computation takes place in operator (), and the output "
          "happens in the destructor.");
    decode_utterance_lattice_faster_class.def(py::init<LatticeFasterDecoder *,
      DecodableInterface *,
      const TransitionInformation &,
      const fst::SymbolTable *,
      const std::string &,
      BaseFloat ,
      bool ,
      bool ,
      Int32VectorWriter *,
      Int32VectorWriter *,
      CompactLatticeWriter *,
      LatticeWriter *,
      double *,
      int64 *,
      int32 *,
      int32 *,
      int32 *>(),
        py::arg("decoder"),
        py::arg("decodable"),
        py::arg("trans_model"),
        py::arg("word_syms"),
        py::arg("utt"),
        py::arg("acoustic_scale"),
        py::arg("determinize"),
        py::arg("allow_partial"),
        py::arg("alignments_writer"),
        py::arg("words_writer"),
        py::arg("compact_lattice_writer"),
        py::arg("lattice_writer"),
        py::arg("like_sum"),
        py::arg("frame_sum"),
        py::arg("num_done"),
        py::arg("num_err"),
        py::arg("num_partial"))
      .def("__call__", &PyClass::operator());
  }

  m.def("DecodeUtteranceLatticeSimple",
        &DecodeUtteranceLatticeSimple,
        "This function DecodeUtteranceLatticeSimple is used in several decoders, and "
          "we have moved it here.  Note: this is really \"binary-level\" code as it "
          "involves table readers and writers; we've just put it here as there is no "
          "other obvious place to put it.  If determinize == false, it writes to "
          "lattice_writer, else to compact_lattice_writer.  The writers for "
          "alignments and words will only be written to if they are open.",
        py::arg("decoder"),
        py::arg("decodable"),
        py::arg("trans_model"),
        py::arg("word_syms"),
        py::arg("utt"),
        py::arg("acoustic_scale"),
        py::arg("determinize"),
        py::arg("allow_partial"),
        py::arg("alignments_writer"),
        py::arg("words_writer"),
        py::arg("compact_lattice_writer"),
        py::arg("lattice_writer"),
        py::arg("like_ptr"));
}

void pybind_decoder_grammar_fst(py::module& m) {

  {
    using PyClass = GrammarFstArc;

    auto grammar_fst_arc = py::class_<PyClass>(
        m, "GrammarFstArc");
    grammar_fst_arc.def(py::init<>())
     .def(py::init<PyClass::Label ,
                    PyClass::Label ,
                    PyClass::Weight ,
                    PyClass::StateId >(),
        py::arg("ilabel"),
        py::arg("olabel"),
        py::arg("weight"),
        py::arg("nextstate"))
      .def_readwrite("ilabel", &PyClass::ilabel)
      .def_readwrite("olabel", &PyClass::olabel)
      .def_readwrite("weight", &PyClass::weight)
      .def_readwrite("nextstate", &PyClass::nextstate);
  }
  {
    using PyClass = VectorGrammarFst;

    auto grammar_fst = py::class_<PyClass>(
        m, "VectorGrammarFst");
    grammar_fst.def(py::init<>())
     .def(py::init<int32 ,
      std::shared_ptr<StdVectorFst> ,
      const std::vector<std::pair<int32, std::shared_ptr<StdVectorFst> > > &>(),
        py::arg("nonterm_phones_offset"),
        py::arg("top_fst"),
        py::arg("ifsts"))
     .def(py::init<const VectorGrammarFst &>(),
        py::arg("other"))
      .def("Write",
        &PyClass::Write,
        "This Write function allows you to dump a GrammarFst to disk as a single "
          "object.  It only supports binary mode, but the option is allowed for "
          "compatibility with other Kaldi read/write functions (it will crash if "
          "binary == false).",
          py::arg("os"),
          py::arg("binary"),
      py::call_guard<py::gil_scoped_release>())
      .def("Read",
        &PyClass::Read,
        "Reads the format that Write() outputs.  Will crash if binary == false.",
          py::arg("os"),
          py::arg("binary"),
      py::call_guard<py::gil_scoped_release>())
      .def("Start",
        &PyClass::Start)
      .def("Final",
        &PyClass::Final,
          py::arg("s"))
      .def("NumInputEpsilons",
        &PyClass::NumInputEpsilons,
        "This is called in LatticeFasterDecoder.  As an implementation shortcut, if "
          "the state is an expanded state, we return 1, meaning 'yes, there are input "
          "epsilons'; the calling code doesn't actually care about the exact number.",
          py::arg("s"))
      .def("Type",
        &PyClass::Type)
      .def_readwrite("instances_", &PyClass::instances_,
          "The FST instances.  Initially it is a vector with just one element "
          "representing top_fst_, and it will be populated with more elements on "
          "demand.  An instance_id refers to an index into this vector.")
      .def_readwrite("nonterm_phones_offset_", &PyClass::nonterm_phones_offset_,
          "The integer id of the symbol #nonterm_bos in phones.txt.")
      .def_readwrite("top_fst_", &PyClass::top_fst_,
          "The top-level FST passed in by the user; contains the start state and "
          "final-states, and may invoke FSTs in 'ifsts_' (which can also invoke "
          "each other recursively).")
      .def_readwrite("ifsts_", &PyClass::ifsts_,
          "A list of pairs (nonterm, fst), where 'nonterm' is a user-defined "
          "nonterminal symbol as numbered in phones.txt (e.g. #nonterm:foo), and "
          "'fst' is the corresponding FST.")
      .def_readwrite("nonterminal_map_", &PyClass::nonterminal_map_,
          "Maps from the user-defined nonterminals like #nonterm:foo as numbered in "
          "phones.txt, to the corresponding index into 'ifsts_', i.e. the ifst_index.")
      .def_readwrite("entry_arcs_", &PyClass::entry_arcs_,
          "entry_arcs_ will have the same dimension as ifsts_.  Each entry_arcs_[i] "
          "is a map from left-context phone (i.e. either a phone-index or "
          "#nonterm_bos) to the corresponding arc-index leaving the start-state in "
          "the FST 'ifsts_[i].second'. "
          "We populate this only on demand as each one is needed (except for the "
          "first one, which we populate immediately as a kind of sanity check). "
          "Doing it on-demand prevents this object's initialization from being "
          "nontrivial in the case where there are a lot of nonterminals.");

    py::class_<VectorGrammarFst::ExpandedState, std::shared_ptr<VectorGrammarFst::ExpandedState>>(
        grammar_fst, "ExpandedState",
        "Represents an expanded state in an FstInstance.  We expand states whenever "
          "we encounter states with a final-cost equal to "
          "KALDI_GRAMMAR_FST_SPECIAL_WEIGHT (4096.0).  The function "
          "PrepareGrammarFst() makes sure to add this special final-cost on states "
          "that have special arcs leaving them.")
     .def(py::init<>())
      .def_readwrite("dest_fst_instance", &PyClass::ExpandedState::dest_fst_instance,
          "The final-prob for expanded states is always zero; to avoid "
          "corner cases, we ensure this via adding epsilon arcs where "
          "needed. "
          "\n"
          "fst-instance index of destination state (we will have ensured previously "
          "that this is the same for all outgoing arcs).")
      .def_readwrite("arcs", &PyClass::ExpandedState::arcs,
          "List of arcs out of this state, where the 'nextstate' element will be the "
          "lower-order 32 bits of the destination state and the higher order bits "
          "will be given by 'dest_fst_instance'.  We do it this way, instead of "
          "constructing a vector<Arc>, in order to simplify the ArcIterator code and "
          "avoid unnecessary branches in loops over arcs. "
          "We guarantee that this 'arcs' array will always be nonempty; this "
          "is to avoid certain hassles on Windows with automated bounds-checking.");

    py::class_<VectorGrammarFst::FstInstance>(
        grammar_fst, "FstInstance")
     .def(py::init<>())
      .def_readwrite("ifst_index",
          &PyClass::FstInstance::ifst_index,
          "ifst_index is the index into the ifsts_ vector that corresponds to this "
          "FST instance, or -1 if this is the top-level instance.")
      .def_readwrite("fst",
          &PyClass::FstInstance::fst,
          "Pointer to the FST corresponding to this instance: it will equal top_fst_ "
          "if ifst_index == -1, or ifsts_[ifst_index].second otherwise.")
      .def_readwrite("expanded_states",
          &PyClass::FstInstance::expanded_states,
          "'expanded_states', which will be populated on demand as states in this "
          "FST instance are accessed, will only contain entries for states in this "
          "FST that the final-prob's value equal to "
          "KALDI_GRAMMAR_FST_SPECIAL_WEIGHT.  (That final-prob value is used as a "
          "kind of signal to this code that the state needs expansion).")
      .def_readwrite("child_instances",
          &PyClass::FstInstance::child_instances,
          "'child_instances', which is populated on demand as states in this FST "
          "instance are accessed, is logically a map from pair (nonterminal_index, "
          "return_state) to instance_id.  When we encounter an arc in our FST with a "
          "user-defined nonterminal indexed 'nonterminal_index' on its ilabel, and "
          "with 'return_state' as its nextstate, we look up that pair "
          "(nonterminal_index, return_state) in this map to see whether there "
          "already exists an FST instance for that.  If it exists then the "
          "transition goes to that FST instance; if not, then we create a new one. "
          "The 'return_state' that's part of the key in this map would be the same "
          "as the 'parent_state' in that child FST instance, and of course the "
          "'parent_instance' in that child FST instance would be the instance_id of "
          "this instance. "
          "\n"
          "In most cases each return_state would only have a single "
          "nonterminal_index, making the 'nonterminal_index' in the key *usually* "
          "redundant, but in principle it could happen that two user-defined "
          "nonterminals might share the same return-state.")
      .def_readwrite("parent_instance",
          &PyClass::FstInstance::parent_instance,
          "The instance-id of the FST we return to when we are done with this one "
          "(or -1 if this is the top-level FstInstance so there is nowhere to "
          "return).")
      .def_readwrite("parent_state",
          &PyClass::FstInstance::parent_state,
          "The state in the FST of 'parent_instance' at which we expanded this FST "
          "instance, and to which we return (actually we return to the next-states "
          "of arcs out of 'parent_state').")
      .def_readwrite("parent_reentry_arcs",
          &PyClass::FstInstance::parent_reentry_arcs,
          "'parent_reentry_arcs' is a map from left-context-phone (i.e. either a "
          "phone index or #nonterm_bos), to an arc-index, which we could use to "
          "Seek() in an arc-iterator for state parent_state in the FST-instance "
          "'parent_instance'.  It's set up when we create this FST instance.  (The "
          "arcs used to enter this instance are not located here, they can be "
          "located in entry_arcs_[instance_id]).  We make use of reentry_arcs when "
          "we expand states in this FST that have #nonterm_end on their arcs, "
          "leading to final-states, which signal a return to the parent "
          "FST-instance.");
  }
  {
    using PyClass = ConstGrammarFst;

    auto grammar_fst = py::class_<PyClass>(
        m, "ConstGrammarFst");
    grammar_fst.def(py::init<>())
     .def(py::init<int32 ,
      std::shared_ptr<const ConstFst<StdArc> > ,
      const std::vector<std::pair<int32, std::shared_ptr<const ConstFst<StdArc> > > > &>(),
        py::arg("nonterm_phones_offset"),
        py::arg("top_fst"),
        py::arg("ifsts"))
     .def(py::init<const PyClass &>(),
        py::arg("other"))
      .def("Write",
        &PyClass::Write,
        "This Write function allows you to dump a GrammarFst to disk as a single "
          "object.  It only supports binary mode, but the option is allowed for "
          "compatibility with other Kaldi read/write functions (it will crash if "
          "binary == false).",
          py::arg("os"),
          py::arg("binary"),
      py::call_guard<py::gil_scoped_release>())
      .def("Read",
        &PyClass::Read,
        "Reads the format that Write() outputs.  Will crash if binary == false.",
          py::arg("os"),
          py::arg("binary"),
      py::call_guard<py::gil_scoped_release>())
      .def("Start",
        &PyClass::Start)
      .def("Final",
        &PyClass::Final,
          py::arg("s"))
      .def("NumInputEpsilons",
        &PyClass::NumInputEpsilons,
        "This is called in LatticeFasterDecoder.  As an implementation shortcut, if "
          "the state is an expanded state, we return 1, meaning 'yes, there are input "
          "epsilons'; the calling code doesn't actually care about the exact number.",
          py::arg("s"))
      .def("Type",
        &PyClass::Type)
      .def_readwrite("instances_", &PyClass::instances_,
          "The FST instances.  Initially it is a vector with just one element "
          "representing top_fst_, and it will be populated with more elements on "
          "demand.  An instance_id refers to an index into this vector.")
      .def_readwrite("nonterm_phones_offset_", &PyClass::nonterm_phones_offset_,
          "The integer id of the symbol #nonterm_bos in phones.txt.")
      .def_readwrite("top_fst_", &PyClass::top_fst_,
          "The top-level FST passed in by the user; contains the start state and "
          "final-states, and may invoke FSTs in 'ifsts_' (which can also invoke "
          "each other recursively).")
      .def_readwrite("ifsts_", &PyClass::ifsts_,
          "A list of pairs (nonterm, fst), where 'nonterm' is a user-defined "
          "nonterminal symbol as numbered in phones.txt (e.g. #nonterm:foo), and "
          "'fst' is the corresponding FST.")
      .def_readwrite("nonterminal_map_", &PyClass::nonterminal_map_,
          "Maps from the user-defined nonterminals like #nonterm:foo as numbered in "
          "phones.txt, to the corresponding index into 'ifsts_', i.e. the ifst_index.")
      .def_readwrite("entry_arcs_", &PyClass::entry_arcs_,
          "entry_arcs_ will have the same dimension as ifsts_.  Each entry_arcs_[i] "
          "is a map from left-context phone (i.e. either a phone-index or "
          "#nonterm_bos) to the corresponding arc-index leaving the start-state in "
          "the FST 'ifsts_[i].second'. "
          "We populate this only on demand as each one is needed (except for the "
          "first one, which we populate immediately as a kind of sanity check). "
          "Doing it on-demand prevents this object's initialization from being "
          "nontrivial in the case where there are a lot of nonterminals.");

    py::class_<PyClass::ExpandedState, std::shared_ptr<PyClass::ExpandedState>>(
        grammar_fst, "ExpandedState",
        "Represents an expanded state in an FstInstance.  We expand states whenever "
          "we encounter states with a final-cost equal to "
          "KALDI_GRAMMAR_FST_SPECIAL_WEIGHT (4096.0).  The function "
          "PrepareGrammarFst() makes sure to add this special final-cost on states "
          "that have special arcs leaving them.")
     .def(py::init<>())
      .def_readwrite("dest_fst_instance", &PyClass::ExpandedState::dest_fst_instance,
          "The final-prob for expanded states is always zero; to avoid "
          "corner cases, we ensure this via adding epsilon arcs where "
          "needed. "
          "\n"
          "fst-instance index of destination state (we will have ensured previously "
          "that this is the same for all outgoing arcs).")
      .def_readwrite("arcs", &PyClass::ExpandedState::arcs,
          "List of arcs out of this state, where the 'nextstate' element will be the "
          "lower-order 32 bits of the destination state and the higher order bits "
          "will be given by 'dest_fst_instance'.  We do it this way, instead of "
          "constructing a vector<Arc>, in order to simplify the ArcIterator code and "
          "avoid unnecessary branches in loops over arcs. "
          "We guarantee that this 'arcs' array will always be nonempty; this "
          "is to avoid certain hassles on Windows with automated bounds-checking.");

    py::class_<PyClass::FstInstance>(
        grammar_fst, "FstInstance")
     .def(py::init<>())
      .def_readwrite("ifst_index",
          &PyClass::FstInstance::ifst_index,
          "ifst_index is the index into the ifsts_ vector that corresponds to this "
          "FST instance, or -1 if this is the top-level instance.")
      .def_readwrite("fst",
          &PyClass::FstInstance::fst,
          "Pointer to the FST corresponding to this instance: it will equal top_fst_ "
          "if ifst_index == -1, or ifsts_[ifst_index].second otherwise.")
      .def_readwrite("expanded_states",
          &PyClass::FstInstance::expanded_states,
          "'expanded_states', which will be populated on demand as states in this "
          "FST instance are accessed, will only contain entries for states in this "
          "FST that the final-prob's value equal to "
          "KALDI_GRAMMAR_FST_SPECIAL_WEIGHT.  (That final-prob value is used as a "
          "kind of signal to this code that the state needs expansion).")
      .def_readwrite("child_instances",
          &PyClass::FstInstance::child_instances,
          "'child_instances', which is populated on demand as states in this FST "
          "instance are accessed, is logically a map from pair (nonterminal_index, "
          "return_state) to instance_id.  When we encounter an arc in our FST with a "
          "user-defined nonterminal indexed 'nonterminal_index' on its ilabel, and "
          "with 'return_state' as its nextstate, we look up that pair "
          "(nonterminal_index, return_state) in this map to see whether there "
          "already exists an FST instance for that.  If it exists then the "
          "transition goes to that FST instance; if not, then we create a new one. "
          "The 'return_state' that's part of the key in this map would be the same "
          "as the 'parent_state' in that child FST instance, and of course the "
          "'parent_instance' in that child FST instance would be the instance_id of "
          "this instance. "
          "\n"
          "In most cases each return_state would only have a single "
          "nonterminal_index, making the 'nonterminal_index' in the key *usually* "
          "redundant, but in principle it could happen that two user-defined "
          "nonterminals might share the same return-state.")
      .def_readwrite("parent_instance",
          &PyClass::FstInstance::parent_instance,
          "The instance-id of the FST we return to when we are done with this one "
          "(or -1 if this is the top-level FstInstance so there is nowhere to "
          "return).")
      .def_readwrite("parent_state",
          &PyClass::FstInstance::parent_state,
          "The state in the FST of 'parent_instance' at which we expanded this FST "
          "instance, and to which we return (actually we return to the next-states "
          "of arcs out of 'parent_state').")
      .def_readwrite("parent_reentry_arcs",
          &PyClass::FstInstance::parent_reentry_arcs,
          "'parent_reentry_arcs' is a map from left-context-phone (i.e. either a "
          "phone index or #nonterm_bos), to an arc-index, which we could use to "
          "Seek() in an arc-iterator for state parent_state in the FST-instance "
          "'parent_instance'.  It's set up when we create this FST instance.  (The "
          "arcs used to enter this instance are not located here, they can be "
          "located in entry_arcs_[instance_id]).  We make use of reentry_arcs when "
          "we expand states in this FST that have #nonterm_end on their arcs, "
          "leading to final-states, which signal a return to the parent "
          "FST-instance.");
  }
  m.def("PrepareForGrammarFst",
        &PrepareForGrammarFst,
        "This function prepares 'ifst' for use in GrammarFst: it ensures that it has "
          "the expected properties, changing it slightly as needed.  'ifst' is expected "
          "to be a fully compiled HCLG graph that is intended to be used in GrammarFst. "
          "The user will most likely want to copy it to the ConstFst type after calling "
          "this function. "
          "\n"
          "The following describes what this function does, and the reasons why "
          "it has to do these things: "
          "\n"
          "     - To keep the ArcIterator code simple (to avoid branches in loops), even "
          "     for expanded states we store the destination fst-instance index "
          "     separately per state, not per arc.  This requires that any transitions "
          "     across FST boundaries from a single FST must be to a single destination "
          "     FST (for a given source state).  We fix this problem by introducing "
          "     epsilon arcs and new states whenever we find a state that would cause a "
          "     problem for the above. "
          "     - In order to signal to the GrammarFst code that a particular state has "
          "     cross-FST-boundary transitions, we set the final-prob to a nonzero value "
          "     on that state.  Specifically, we use a weight with Value() == 4096.0. "
          "     When the GrammarFst code sees that value it knows that it was not a "
          "     'real' final-prob.  Prior to doing this we ensure, by adding epsilon "
          "     transitions as needed, that the state did not previously have a "
          "     final-prob. "
          "     - For arcs that are final arcs in an FST that represents a nonterminal "
          "     (these arcs would have #nonterm_exit on them), we ensure that the "
          "     states that they transition to have unit final-prob (i.e. final-prob "
          "     equal to One()), by incorporating any final-prob into the arc itself. "
          "     This avoids the GrammarFst code having to inspect those final-probs "
          "     when expanding states. "
          "\n"
          "     @param [in] nonterm_phones_offset   The integer id of "
          "               the symbols #nonterm_bos in the phones.txt file. "
          "     @param [in,out] fst  The FST to be (slightly) modified.",
        py::arg("nonterm_phones_offset"),
        py::arg("fst"));
}

void pybind_decoder_faster_decoder(py::module& m) {

  {
    using PyClass = FasterDecoderOptions;

    auto faster_decoder_options = py::class_<PyClass>(
        m, "FasterDecoderOptions");
    faster_decoder_options.def(py::init<>())
      .def_readwrite("beam", &PyClass::beam)
      .def_readwrite("max_active", &PyClass::max_active)
      .def_readwrite("min_active", &PyClass::min_active)
      .def_readwrite("beam_delta", &PyClass::beam_delta)
      .def_readwrite("hash_ratio", &PyClass::hash_ratio);
  }
  {
    using PyClass = FasterDecoder;

    auto faster_decoder = py::class_<PyClass>(
        m, "FasterDecoder");
    faster_decoder.def(py::init<const fst::Fst<fst::StdArc> &,
                const FasterDecoderOptions &>(),
                       py::arg("fst"),
                       py::arg("config"))
      .def("SetOptions",
        &PyClass::SetOptions,
                       py::arg("config"))
      .def("Decode",
        &PyClass::Decode,
                       py::arg("decodable"),
      py::call_guard<py::gil_scoped_release>())
      .def("ReachedFinal",
        &PyClass::ReachedFinal)
      .def("GetBestPath",
           [](PyClass& decoder,
              bool use_final_probs = true) -> std::pair<bool, Lattice> {
          py::gil_scoped_release gil_release;
             Lattice ofst;
             bool is_succeeded = decoder.GetBestPath(&ofst, use_final_probs);
              py::gil_scoped_acquire acquire;
             return std::make_pair(is_succeeded, ofst);
           },
        "GetBestPath gets the decoding traceback. If \"use_final_probs\" is true "
          "AND we reached a final state, it limits itself to final states; "
          "otherwise it gets the most likely token not taking into account "
          "final-probs. Returns true if the output best path was not the empty "
          "FST (will only return false in unusual circumstances where "
          "no tokens survived).",
                       py::arg("use_final_probs") = true,
                       py::return_value_policy::reference)
      .def("InitDecoding",
        &PyClass::InitDecoding,
        "As a new alternative to Decode(), you can call InitDecoding "
          "and then (possibly multiple times) AdvanceDecoding().")
      .def("AdvanceDecoding",
        &PyClass::AdvanceDecoding,
        "This will decode until there are no more frames ready in the decodable "
          "object, but if max_num_frames is >= 0 it will decode no more than "
          "that many frames.",
                       py::arg("decodable"),
                       py::arg("max_num_frames") = -1)
      .def("NumFramesDecoded",
        &PyClass::NumFramesDecoded,
        "Returns the number of frames already decoded.");
  }
}

void pybind_decoder_lattice_biglm_faster_decoder(py::module& m) {

  {
    using PyClass = LatticeBiglmFasterDecoder;

    auto lattice_biglm_faster_decoder = py::class_<PyClass>(
        m, "LatticeBiglmFasterDecoder",
        "This is as LatticeFasterDecoder, but does online composition between "
          "HCLG and the \"difference language model\", which is a deterministic "
          "FST that represents the difference between the language model you want "
          "and the language model you compiled HCLG with.  The class "
          "DeterministicOnDemandFst follows through the epsilons in G for you "
          "(assuming G is a standard backoff language model) and makes it look "
          "like a determinized FST.");
    lattice_biglm_faster_decoder.def(py::init<const fst::Fst<fst::StdArc> &,
      const LatticeBiglmFasterDecoderConfig &,
      fst::DeterministicOnDemandFst<fst::StdArc> *>(),
                       py::arg("fst"),
                       py::arg("config"),
                       py::arg("lm_diff_fst"))
      .def("SetOptions",
        &PyClass::SetOptions,
                       py::arg("config"))
      .def("GetOptions",
        &PyClass::GetOptions)
      .def("Decode",
        &PyClass::Decode,
        "Returns true if any kind of traceback is available (not necessarily from "
          "a final state).",
                       py::arg("decodable"))
      .def("ReachedFinal",
        &PyClass::ReachedFinal,
        "says whether a final-state was active on the last frame.  If it was not, the "
          "lattice (or traceback) will end with states that are not final-states.")
      .def("GetBestPath",
           [](const PyClass& decoder,
              bool use_final_probs = true) -> std::pair<bool, Lattice> {
             Lattice ofst;
             bool is_succeeded = decoder.GetBestPath(&ofst, use_final_probs);
             return std::make_pair(is_succeeded, ofst);
           },
        "Outputs an FST corresponding to the single best path "
          "through the lattice.",
                       py::arg("use_final_probs") = true)
      .def("GetRawLattice",
           [](const PyClass& decoder,
              bool use_final_probs = true) -> std::pair<bool, Lattice> {
             Lattice ofst;
             bool is_succeeded = decoder.GetRawLattice(&ofst, use_final_probs);
             return std::make_pair(is_succeeded, ofst);
           },
        "Outputs an FST corresponding to the raw, state-level "
          "tracebacks.",
                       py::arg("use_final_probs") = true)
      .def("GetLattice",
        &PyClass::GetLattice,
        "This function is now deprecated, since now we do determinization from "
          "outside the LatticeBiglmFasterDecoder class. "
          "Outputs an FST corresponding to the lattice-determinized"
          "lattice (one path per word sequence).",
                       py::arg("ofst"),
                       py::arg("use_final_probs") = true);
  }
}

void pybind_decoder_lattice_faster_online_decoder(py::module& m) {

  {
    using PyClass = LatticeFasterOnlineDecoder;

    auto lattice_faster_online_decoder = py::class_<PyClass>(
        m, "LatticeFasterOnlineDecoder",
        "LatticeFasterOnlineDecoderTpl is as LatticeFasterDecoderTpl but also "
          "supports an efficient way to get the best path (see the function "
          "BestPathEnd()), which is useful in endpointing and in situations where you "
          "might want to frequently access the best path. "
          "\n"
          "This is only templated on the FST type, since the Token type is required to "
          "be BackpointerToken.  Actually it only makes sense to instantiate "
          "LatticeFasterDecoderTpl with Token == BackpointerToken if you do so indirectly via "
          "this child class.");
    lattice_faster_online_decoder.def(py::init<const fst::StdFst &,
                                const LatticeFasterDecoderConfig &>(),
                       py::arg("fst"),
                       py::arg("config"))
          .def(py::init<const LatticeFasterDecoderConfig &,
                                fst::StdFst *>(),
                       py::arg("config"),
                       py::arg("fst"))
      .def("GetBestPath",
           [](const PyClass& decoder,
              bool use_final_probs = true) -> std::pair<bool, Lattice> {
             Lattice ofst;
             bool is_succeeded = decoder.GetBestPath(&ofst, use_final_probs);
             return std::make_pair(is_succeeded, ofst);
           },
        "Outputs an FST corresponding to the single best path through the lattice. "
          "This is quite efficient because it doesn't get the entire raw lattice and find "
          "the best path through it; instead, it uses the BestPathEnd and BestPathIterator "
          "so it basically traces it back through the lattice. "
          "Returns true if result is nonempty (using the return status is deprecated, "
          "it will become void).  If \"use_final_probs\" is true AND we reached the "
          "final-state of the graph then it will include those as final-probs, else "
          "it will treat all final-probs as one.",
                       py::arg("use_final_probs") = true)
      .def("TestGetBestPath",
        &PyClass::TestGetBestPath,
        "This function does a self-test of GetBestPath().  Returns true on "
       "success; returns false and prints a warning on failure.",
                       py::arg("use_final_probs") = true)
      .def("BestPathEnd",
        &PyClass::BestPathEnd,
        "This function returns an iterator that can be used to trace back "
          "the best path.  If use_final_probs == true and at least one final state "
          "survived till the end, it will use the final-probs in working out the best "
          "final Token, and will output the final cost to *final_cost (if non-NULL), "
          "else it will use only the forward likelihood, and will put zero in "
          "*final_cost (if non-NULL). "
          "Requires that NumFramesDecoded() > 0.",
                       py::arg("use_final_probs"),
                       py::arg("final_cost") = NULL)
      .def("TraceBackBestPath",
        &PyClass::TraceBackBestPath,
        "This function can be used in conjunction with BestPathEnd() to trace back "
          "the best path one link at a time (e.g. this can be useful in endpoint "
          "detection).  By \"link\" we mean a link in the graph; not all links cross "
          "frame boundaries, but each time you see a nonzero ilabel you can interpret "
          "that as a frame.  The return value is the updated iterator.  It outputs "
          "the ilabel and olabel, and the (graph and acoustic) weight to the \"arc\" pointer, "
          "while leaving its \"nextstate\" variable unchanged.",
                       py::arg("iter"),
                       py::arg("arc"))
      .def("GetRawLatticePruned",
        &PyClass::GetRawLatticePruned,
        "Behaves the same as GetRawLattice but only processes tokens whose "
          "extra_cost is smaller than the best-cost plus the specified beam. "
          "It is only worthwhile to call this function if beam is less than "
          "the lattice_beam specified in the config; otherwise, it would "
          "return essentially the same thing as GetRawLattice, but more slowly.",
                       py::arg("ofst"),
                       py::arg("use_final_probs"),
                       py::arg("beam"));

    py::class_<LatticeFasterOnlineDecoder::BestPathIterator>(
        lattice_faster_online_decoder, "BestPathIterator")
     .def(py::init<void *, int32 >(),
                       py::arg("t"),
                       py::arg("f"))
      .def_readwrite("tok",
          &PyClass::BestPathIterator::tok)
      .def_readwrite("frame",
          &PyClass::BestPathIterator::frame,
          "note, \"frame\" is the frame-index of the frame you'll get the "
          "transition-id for next time, if you call TraceBackBestPath on this "
          "iterator (assuming it's not an epsilon transition).  Note that this "
          "is one less than you might reasonably expect, e.g. it's -1 for "
          "the nonemitting transitions before the first frame.")
      .def("Done",
        &PyClass::BestPathIterator::Done);
  }
}

void pybind_decoder_lattice_incremental_decoder(py::module& m) {


  {
    using PyClass = LatticeIncrementalDecoderConfig;

    auto lattice_incremental_decoder_config = py::class_<PyClass>(
        m, "LatticeIncrementalDecoderConfig");
    lattice_incremental_decoder_config.def(py::init<>())
      .def_readwrite("beam", &PyClass::beam)
      .def_readwrite("max_active", &PyClass::max_active)
      .def_readwrite("min_active", &PyClass::min_active)
      .def_readwrite("lattice_beam", &PyClass::lattice_beam)
      .def_readwrite("prune_interval", &PyClass::prune_interval)
      .def_readwrite("beam_delta", &PyClass::beam_delta)
      .def_readwrite("hash_ratio", &PyClass::hash_ratio)
      .def_readwrite("prune_scale", &PyClass::prune_scale)
      .def_readwrite("det_opts", &PyClass::det_opts)
      .def_readwrite("determinize_max_delay", &PyClass::determinize_max_delay)
      .def_readwrite("determinize_min_chunk_size", &PyClass::determinize_min_chunk_size)
      .def_readwrite("determinize_max_active", &PyClass::determinize_max_active)
      .def("Check",
        &PyClass::Check);
  }
  {
    using PyClass = LatticeIncrementalDeterminizer;

    auto lattice_incremental_determinizer = py::class_<PyClass>(
        m, "LatticeIncrementalDeterminizer");
    lattice_incremental_determinizer.def(py::init<const TransitionInformation &,
      const LatticeIncrementalDecoderConfig &>(),
          py::arg("trans_model"),
          py::arg("config"))
      .def("Init",
        &PyClass::Init)
      .def("GetDeterminizedLattice",
        &PyClass::GetDeterminizedLattice)
      .def("InitializeRawLatticeChunk",
        &PyClass::InitializeRawLatticeChunk,
          "Starts the process of creating a raw lattice chunk.  (Search the glossary "
          "for \"raw lattice chunk\").  This just sets up the initial states and "
          "redeterminized-states in the chunk.  Relates to sec. 5.2 in the paper, "
          "specifically the initial-state i and the redeterminized-states. "
          "\n"
          "After calling this, the caller would add the remaining arcs and states "
          "to `olat` and then call AcceptRawLatticeChunk() with the result. "
          "\n"
          "@param [out] olat    The lattice to be (partially) created "
          "\n"
          "@param [out] token_label2state  This function outputs to here "
          "          a map from `token-label` to the state we created for "
          "          it in *olat.  See glossary for `token-label`. "
          "          The keys actually correspond to the .nextstate fields "
          "          in the arcs in final_arcs_; values are states in `olat`. "
          "          See the last bullet point before Sec. 5.3 in the paper.",
          py::arg("olat"),
          py::arg("token_label2state"))
      .def("AcceptRawLatticeChunk",
        &PyClass::AcceptRawLatticeChunk,
          "This function accepts the raw FST (state-level lattice) corresponding to a "
          "single chunk of the lattice, determinizes it and appends it to this->clat_. "
          "Unless this was the "
          "\n"
          "Note: final-probs in `raw_fst` are treated specially: they are used to "
          "guide the pruned determinization, but when you call GetLattice() it will be "
          "-- except for pruning effects-- as if all nonzero final-probs in `raw_fst` "
          "were: One() if final_costs == NULL; else the value present in `final_costs`. "
          "\n"
          "@param [in] raw_fst  (Consumed destructively).  The input "
          "          raw (state-level) lattice.  Would correspond to the "
          "          FST A in the paper if first_frame == 0, and B "
          "          otherwise. "
          "\n"
          "@return returns false if determinization finished earlier than the beam "
          "or the determinized lattice was empty; true otherwise. "
          "\n"
          "NOTE: if this is not the final chunk, you will probably want to call "
          "SetFinalCosts() directly after calling this.",
          py::arg("raw_fst"))
      .def("SetFinalCosts",
        &PyClass::SetFinalCosts,
          "Sets final-probs in `clat_`.  Must only be called if the final chunk "
          "has not been processed.  (The final chunk is whenever GetLattice() is "
          "called with finalize == true). "
          "\n"
          "The reason this is a separate function from AcceptRawLatticeChunk() is that "
          "there may be situations where a user wants to get the latice with "
          "final-probs in it, after previously getting it without final-probs; or "
          "vice versa.  By final-probs, we mean the Final() probabilities in the "
          "HCLG (decoding graph; this->fst_). "
          "\n"
          "     @param [in] token_label2final_cost   A map from the token-label "
          "          corresponding to Tokens active on the final frame of the "
          "          lattice in the object, to the final-cost we want to use for "
          "          those tokens.  If NULL, it means all Tokens should be treated "
          "          as final with probability One().  If non-NULL, and a particular "
          "          token-label is not a key of this map, it means that Token "
          "          corresponded to a state that was not final in HCLG; and "
          "          such tokens will be treated as non-final.  However, "
          "          if this would result in no states in the lattice being final, "
          "          we will treat all Tokens as final with probability One(), "
          "          a warning will be printed (this should not happen.)",
          py::arg("token_label2final_cost") = NULL)
      .def("GetLattice",
        &PyClass::GetLattice);
  }
  {
    using PyClass = LatticeIncrementalDecoder;

    auto lattice_incremental_decoder = py::class_<PyClass>(
        m, "LatticeIncrementalDecoder");
    lattice_incremental_decoder.def(py::init<const fst::StdFst &, const TransitionInformation &,
                               const LatticeIncrementalDecoderConfig &>(),
          py::arg("fst"),
          py::arg("trans_model"),
          py::arg("config"))
          .def(py::init<const LatticeIncrementalDecoderConfig &,
                               fst::StdFst *, const TransitionInformation &>(),
          py::arg("config"),
          py::arg("fst"),
          py::arg("trans_model"))
      .def("SetOptions",
        &PyClass::SetOptions,
          py::arg("config"))
      .def("GetOptions",
        &PyClass::GetOptions)
      .def("Decode",
        &PyClass::Decode,
        "CAUTION: it's unlikely that you will ever want to call this function.  In a "
          "scenario where you have the entire file and just want to decode it, there "
          "is no point using this decoder. "
          "\n"
          "An example of how to do decoding together with incremental "
          "determinization. It decodes until there are no more frames left in the "
          "\"decodable\" object. "
          "\n"
          "In this example, config_.determinize_max_delay, config_.determinize_min_chunk_size "
          "and config_.determinize_max_active are used to determine the time to "
          "call GetLattice(). "
          "\n"
          "Users will probably want to use appropriate combinations of "
          "AdvanceDecoding() and GetLattice() to build their application; this just "
          "gives you some idea how. "
          "\n"
          "The function returns true if any kind of traceback is available (not "
          "necessarily from a final state).",
          py::arg("decodable"))
      .def("ReachedFinal",
        &PyClass::ReachedFinal)
      .def("GetLattice",
        &PyClass::GetLattice,
        "This GetLattice() function returns the lattice containing "
          "`num_frames_to_decode` frames; this will be all frames decoded so "
          "far, if you let num_frames_to_decode == NumFramesDecoded(), "
          "but it will generally be better to make it a few frames less than "
          "that to avoid the lattice having too many active states at "
          "the end. "
          "\n"
          "@param [in] num_frames_to_include  The number of frames that you want "
          "               to be included in the lattice.  Must be >= "
          "               NumFramesInLattice() and <= NumFramesDecoded(). "
          "\n"
          "@param [in] use_final_probs  True if you want the final-probs "
          "               of HCLG to be included in the output lattice.  Must not "
          "               be set to true if num_frames_to_include != "
          "               NumFramesDecoded().  Must be set to true if you have "
          "               previously called FinalizeDecoding(). "

          "               (If no state was final on frame `num_frames_to_include`, the "
          "               final-probs won't be included regardless of "
          "               `use_final_probs`; you can test whether this "
          "               was the case by calling ReachedFinal(). "
          "\n"
          "@return clat   The CompactLattice representing what has been decoded "
          "               up until `num_frames_to_include` (e.g., LatticeStateTimes() "
          "               on this lattice would return `num_frames_to_include`). "
          "\n"
          "See also UpdateLatticeDeterminizaton().  Caution: this const ref "
          "is only valid until the next time you call AdvanceDecoding() or "
          "GetLattice(). "
          "\n"
          "CAUTION: the lattice may contain disconnnected states; you should "
          "call Connect() on the output before writing it out.",
          py::arg("num_frames_to_include"),
          py::arg("use_final_probs") = false)
      .def("NumFramesInLattice",
        &PyClass::NumFramesInLattice,
        "Returns the number of frames in the currently-determinized part of the "
          "lattice which will be a number in [0, NumFramesDecoded()].  It will "
          "be the largest number that GetLattice() was called with, but note "
          "that GetLattice() may be called from UpdateLatticeDeterminization(). "
          "\n"
          "Made available in case the user wants to give that same number to "
          "GetLattice().")
      .def("InitDecoding",
        &PyClass::InitDecoding,
        "InitDecoding initializes the decoding, and should only be used if you "
          "intend to call AdvanceDecoding().  If you call Decode(), you don't need to "
          "call this.  You can also call InitDecoding if you have already decoded an "
          "utterance and want to start with a new utterance.")
      .def("AdvanceDecoding",
        &PyClass::AdvanceDecoding,
        "This will decode until there are no more frames ready in the decodable "
          "object.  You can keep calling it each time more frames become available "
          "(this is the normal pattern in a real-time/online decoding scenario). "
          "If max_num_frames is specified, it specifies the maximum number of frames "
          "the function will decode before returning.",
          py::arg("decodable"),
          py::arg("max_num_frames") = -1)
      .def("FinalRelativeCost",
        &PyClass::FinalRelativeCost,
        "FinalRelativeCost() serves the same purpose as ReachedFinal(), but gives "
          "more information.  It returns the difference between the best (final-cost "
          "plus cost) of any token on the final frame, and the best cost of any token "
          "on the final frame.  If it is infinity it means no final-states were "
          "present on the final frame.  It will usually be nonnegative.  If it not "
          "too positive (e.g. < 5 is my first guess, but this is not tested) you can "
          "take it as a good indication that we reached the final-state with "
          "reasonable likelihood.")
      .def("NumFramesDecoded",
        &PyClass::NumFramesDecoded,
        "Returns the number of frames decoded so far.")
      .def("FinalizeDecoding",
        &PyClass::FinalizeDecoding,
        "Finalizes the decoding, doing an extra pruning step on the last frame "
          "that uses the final-probs.  May be called only once.");
  }
}

void pybind_decoder_lattice_incremental_online_decoder(py::module& m) {

  {
    using PyClass = LatticeIncrementalOnlineDecoder;

    auto lattice_incremental_online_decoder = py::class_<PyClass>(
        m, "LatticeIncrementalOnlineDecoder",
        "LatticeIncrementalOnlineDecoderTpl is as LatticeIncrementalDecoderTpl but also "
          "supports an efficient way to get the best path (see the function "
          "BestPathEnd()), which is useful in endpointing and in situations where you "
          "might want to frequently access the best path. "
          "\n"
          "This is only templated on the FST type, since the Token type is required to "
          "be BackpointerToken.  Actually it only makes sense to instantiate "
          "LatticeIncrementalDecoderTpl with Token == BackpointerToken if you do so indirectly via "
          "this child class.");
    lattice_incremental_online_decoder.def(py::init<const fst::StdFst &, const TransitionInformation &,
                               const LatticeIncrementalDecoderConfig &>(),
          py::arg("fst"),
          py::arg("trans_model"),
          py::arg("config"))
          .def(py::init<const LatticeIncrementalDecoderConfig &,
                               fst::StdFst *, const TransitionInformation &>(),
          py::arg("config"),
          py::arg("fst"),
          py::arg("trans_model"))
      .def("GetBestPath",
           [](const PyClass& decoder,
              bool use_final_probs = true) -> std::pair<bool, Lattice> {
             Lattice ofst;
             bool is_succeeded = decoder.GetBestPath(&ofst, use_final_probs);
             return std::make_pair(is_succeeded, ofst);
           },
        "GetBestPath gets the decoding output.  If \"use_final_probs\" is true "
          "AND we reached a final state, it limits itself to final states; "
          "otherwise it gets the most likely token not taking into "
          "account final-probs.  fst_out will be empty (Start() == kNoStateId) if "
          "nothing was available.  It returns true if it got output (thus, fst_out "
          "will be nonempty).",
                       py::arg("use_final_probs") = true)
      .def("BestPathEnd",
        &PyClass::BestPathEnd,
        "This function returns an iterator that can be used to trace back "
          "the best path.  If use_final_probs == true and at least one final state "
          "survived till the end, it will use the final-probs in working out the best "
          "final Token, and will output the final cost to *final_cost (if non-NULL), "
          "else it will use only the forward likelihood, and will put zero in "
          "*final_cost (if non-NULL). "
          "Requires that NumFramesDecoded() > 0.",
                       py::arg("use_final_probs"),
                       py::arg("final_cost") = NULL)
      .def("TraceBackBestPath",
        &PyClass::TraceBackBestPath,
        "This function can be used in conjunction with BestPathEnd() to trace back "
          "the best path one link at a time (e.g. this can be useful in endpoint "
          "detection).  By \"link\" we mean a link in the graph; not all links cross "
          "frame boundaries, but each time you see a nonzero ilabel you can interpret "
          "that as a frame.  The return value is the updated iterator.  It outputs "
          "the ilabel and olabel, and the (graph and acoustic) weight to the \"arc\" pointer, "
          "while leaving its \"nextstate\" variable unchanged.",
                       py::arg("iter"),
                       py::arg("final_arccost"));

    py::class_<LatticeIncrementalOnlineDecoder::BestPathIterator>(
        lattice_incremental_online_decoder, "BestPathIterator")
     .def(py::init<void *, int32 >(),
                       py::arg("t"),
                       py::arg("f"))
      .def_readwrite("tok",
          &PyClass::BestPathIterator::tok)
      .def_readwrite("frame",
          &PyClass::BestPathIterator::frame,
          "note, \"frame\" is the frame-index of the frame you'll get the "
          "transition-id for next time, if you call TraceBackBestPath on this "
          "iterator (assuming it's not an epsilon transition).  Note that this "
          "is one less than you might reasonably expect, e.g. it's -1 for "
          "the nonemitting transitions before the first frame.")
      .def("Done",
        &PyClass::BestPathIterator::Done);
  }
}

void pybind_decoder_lattice_simple_decoder(py::module& m) {
  {
    using PyClass = LatticeSimpleDecoderConfig;

    auto lattice_simple_decoder_config = py::class_<PyClass>(
        m, "LatticeSimpleDecoderConfig");
    lattice_simple_decoder_config.def(py::init<>())
      .def_readwrite("beam", &PyClass::beam)
      .def_readwrite("lattice_beam", &PyClass::lattice_beam)
      .def_readwrite("prune_interval", &PyClass::prune_interval)
      .def_readwrite("determinize_lattice", &PyClass::determinize_lattice)
      .def_readwrite("prune_lattice", &PyClass::prune_lattice)
      .def_readwrite("beam_ratio", &PyClass::beam_ratio)
      .def_readwrite("prune_scale", &PyClass::prune_scale)
      .def_readwrite("det_opts", &PyClass::det_opts)
      .def("Check",
        &PyClass::Check);
  }
  {
    using PyClass = LatticeSimpleDecoder;

    auto lattice_simple_decoder = py::class_<PyClass>(
        m, "LatticeSimpleDecoder");
    lattice_simple_decoder.def(py::init<const fst::Fst<fst::StdArc> &,
                       const LatticeSimpleDecoderConfig &>(),
          py::arg("fst"),
          py::arg("c"))
      .def("GetOptions",
        &PyClass::GetOptions)
      .def("Decode",
        &PyClass::Decode,
        "Returns true if any kind of traceback is available (not necessarily from "
          "a final state).",
          py::arg("decodable"))
      .def("ReachedFinal",
        &PyClass::ReachedFinal,
        "says whether a final-state was active on the last frame.  If it was not, the "
  "lattice (or traceback) will end with states that are not final-states.")
      .def("InitDecoding",
        &PyClass::InitDecoding,
        "InitDecoding initializes the decoding, and should only be used if you "
          "intend to call AdvanceDecoding().  If you call Decode(), you don't need "
          "to call this.  You can call InitDecoding if you have already decoded an "
          "utterance and want to start with a new utterance.")
      .def("FinalizeDecoding",
        &PyClass::FinalizeDecoding,
        "This function may be optionally called after AdvanceDecoding(), when you "
          "do not plan to decode any further.  It does an extra pruning step that "
          "will help to prune the lattices output by GetLattice and (particularly) "
          "GetRawLattice more accurately, particularly toward the end of the "
          "utterance.  It does this by using the final-probs in pruning (if any "
          "final-state survived); it also does a final pruning step that visits all "
          "states (the pruning that is done during decoding may fail to prune states "
          "that are within kPruningScale = 0.1 outside of the beam).  If you call "
          "this, you cannot call AdvanceDecoding again (it will fail), and you "
          "cannot call GetLattice() and related functions with use_final_probs = "
          "false. "
          "Used to be called PruneActiveTokensFinal().")
      .def("FinalRelativeCost",
        &PyClass::FinalRelativeCost,
        "FinalRelativeCost() serves the same purpose as ReachedFinal(), but gives "
          "more information.  It returns the difference between the best (final-cost "
          "plus cost) of any token on the final frame, and the best cost of any token "
          "on the final frame.  If it is infinity it means no final-states were "
          "present on the final frame.  It will usually be nonnegative.  If it not "
          "too positive (e.g. < 5 is my first guess, but this is not tested) you can "
          "take it as a good indication that we reached the final-state with "
          "reasonable likelihood.")
      .def("GetBestPath",
           [](const PyClass& decoder,
              bool use_final_probs = true) -> std::pair<bool, Lattice> {
             Lattice ofst;
             bool is_succeeded = decoder.GetBestPath(&ofst, use_final_probs);
             return std::make_pair(is_succeeded, ofst);
           },
        "Outputs an FST corresponding to the single best path "
          "through the lattice.  Returns true if result is nonempty "
          "(using the return status is deprecated, it will become void). "
          "If \"use_final_probs\" is true AND we reached the final-state "
          "of the graph then it will include those as final-probs, else "
          "it will treat all final-probs as one.",
          py::arg("use_final_probs") = true)
      .def("GetRawLattice",
           [](const PyClass& decoder,
              bool use_final_probs = true) -> std::pair<bool, Lattice> {
             Lattice ofst;
             bool is_succeeded = decoder.GetRawLattice(&ofst, use_final_probs);
             return std::make_pair(is_succeeded, ofst);
           },
        "Outputs an FST corresponding to the raw, state-level "
          "tracebacks.  Returns true if result is nonempty "
          "(using the return status is deprecated, it will become void). "
          "If \"use_final_probs\" is true AND we reached the final-state "
          "of the graph then it will include those as final-probs, else "
          "it will treat all final-probs as one.",
          py::arg("use_final_probs") = true)
      .def("GetLattice",
        &PyClass::GetLattice,
        "This function is now deprecated, since now we do determinization from "
          "outside the LatticeTrackingDecoder class. "
          "Outputs an FST corresponding to the lattice-determinized "
          "lattice (one path per word sequence).  [will become deprecated, "
          "users should determinize themselves.]",
          py::arg("clat"),
          py::arg("use_final_probs") = true)
      .def("NumFramesDecoded",
        &PyClass::NumFramesDecoded);
  }
}

void pybind_decoder_simple_decoder(py::module& m) {

  {
    using PyClass = SimpleDecoder;

    auto simple_decoder = py::class_<PyClass>(
        m, "SimpleDecoder",
        "Simplest possible decoder, included largely for didactic purposes and as a "
          "means to debug more highly optimized decoders.  See \\ref decoders_simple "
          "for more information.");
    simple_decoder.def(py::init<const fst::Fst<fst::StdArc> &, BaseFloat >(),
          py::arg("fst"),
          py::arg("beam"))
      .def("Decode",
        &PyClass::Decode,
        "Decode this utterance. "
          "Returns true if any tokens reached the end of the file (regardless of "
          "whether they are in a final state); query ReachedFinal() after Decode() "
          "to see whether we reached a final state.",
          py::arg("decodable"))
      .def("ReachedFinal",
        &PyClass::ReachedFinal)
      .def("GetBestPath",
           [](const PyClass& decoder,
              bool use_final_probs = true) -> std::pair<bool, Lattice> {
             Lattice ofst;
             bool is_succeeded = decoder.GetBestPath(&ofst, use_final_probs);
             return std::make_pair(is_succeeded, ofst);
           },
        "GetBestPath gets the decoding traceback. If \"use_final_probs\" is true "
          "AND we reached a final state, it limits itself to final states; "
          "otherwise it gets the most likely token not taking into account final-probs. "
          "fst_out will be empty (Start() == kNoStateId) if nothing was available due to "
          "search error. "
          "If Decode() returned true, it is safe to assume GetBestPath will return true. "
          "It returns true if the output lattice was nonempty (i.e. had states in it); "
          "using the return value is deprecated.",
          py::arg("use_final_probs") = true)
      .def("FinalRelativeCost",
        &PyClass::FinalRelativeCost,
        "FinalRelativeCost() serves the same function as ReachedFinal(), but gives "
          "more information.  It returns the difference between the best (final-cost plus "
          "cost) of any token on the final frame, and the best cost of any token "
          "on the final frame.  If it is infinity it means no final-states were present "
          "on the final frame.  It will usually be nonnegative.")
      .def("InitDecoding",
        &PyClass::InitDecoding,
        "InitDecoding initializes the decoding, and should only be used if you "
          "intend to call AdvanceDecoding().  If you call Decode(), you don't need "
          "to call this.  You can call InitDecoding if you have already decoded an "
          "utterance and want to start with a new utterance.")
      .def("AdvanceDecoding",
        &PyClass::AdvanceDecoding,
        "This will decode until there are no more frames ready in the decodable "
          "object, but if max_num_frames is >= 0 it will decode no more than "
          "that many frames.  If it returns false, then no tokens are alive, "
          "which is a kind of error state.",
          py::arg("decodable"),
          py::arg("max_num_frames") = -1)
      .def("NumFramesDecoded",
        &PyClass::NumFramesDecoded);
  }
}

void pybind_decodable_matrix_scale_mapped(py::module& m) {
  using PyClass = DecodableMatrixScaledMapped;
  py::class_<PyClass, DecodableInterface>(m, "DecodableMatrixScaledMapped")
      .def(py::init<const TransitionModel&, const Matrix<BaseFloat>&,
                    BaseFloat>(),
           "This constructor creates an object that will not delete 'likes' "
           "when done.",
           py::arg("tm"), py::arg("likes"), py::arg("scale"))
      // TODO(fangjun): how to wrap the constructor taking the ownership of
      // likes??
      ;
}

void pybind_decodable_matrix_mapped(py::module& m) {
  using PyClass = DecodableMatrixMapped;
  py::class_<PyClass, DecodableInterface>(m, "DecodableMatrixMapped",
                                          R"doc(
   This is like DecodableMatrixScaledMapped, but it doesn't support an acoustic
   scale, and it does support a frame offset, whereby you can state that the
   first row of 'likes' is actually the n'th row of the matrix of available
   log-likelihoods.  It's useful if the neural net output comes in chunks for
   different frame ranges.

   Note: DecodableMatrixMappedOffset solves the same problem in a slightly
   different way, where you use the same decodable object.  This one, unlike
   DecodableMatrixMappedOffset, is compatible with when the loglikes are in a
   SubMatrix.
      )doc")
      .def(py::init<const TransitionModel&, const Matrix<BaseFloat>&, int>(),
           R"doc(
  This constructor creates an object that will not delete "likes" when done.
  the frame_offset is the frame the row 0 of 'likes' corresponds to, would be
  greater than one if this is not the first chunk of likelihoods.
                    )doc",
           py::arg("tm"), py::arg("likes"), py::arg("frame_offset") = 0)
      // TODO(fangjun): how to wrap the constructor taking the ownership of
      // the likes??
      ;
}

void pybind_decodable_matrix_mapped_offset(py::module& m) {
  using PyClass = DecodableMatrixMappedOffset;
  py::class_<PyClass, DecodableInterface>(m, "DecodableMatrixMappedOffset",
                                          R"doc(
   This decodable class returns log-likes stored in a matrix; it supports
   repeatedly writing to the matrix and setting a time-offset representing the
   frame-index of the first row of the matrix.  It's intended for use in
   multi-threaded decoding; mutex and semaphores are not included.  External
   code will call SetLoglikes() each time more log-likelihods are available.
   If you try to access a log-likelihood that's no longer available because
   the frame index is less than the current offset, it is of course an error.

   See also DecodableMatrixMapped, which supports the same type of thing but
   with a different interface where you are expected to re-construct the
   object each time you want to decode.
      )doc")
      .def(py::init<const TransitionModel&>(), py::arg("tm"))
      .def("FirstAvailableFrame", &PyClass::FirstAvailableFrame,
           "this is not part of the generic Decodable interface.")
      .def("AcceptLoglikes", &PyClass::AcceptLoglikes,
           R"doc(
  Logically, this function appends 'loglikes' (interpreted as newly available
  frames) to the log-likelihoods stored in the class.

  This function is destructive of the input "loglikes" because it may
  under some circumstances do a shallow copy using Swap().  This function
  appends loglikes to any existing likelihoods you've previously supplied.
          )doc",
           py::arg("loglikes"), py::arg("frames_to_discard"))
      .def("InputIsFinished", &PyClass::InputIsFinished);
}

void pybind_decodable_matrix_scaled(py::module& m) {
  using PyClass = DecodableMatrixScaled;
  py::class_<PyClass, DecodableInterface>(m, "DecodableMatrixScaled")
      .def(py::init<const Matrix<BaseFloat>&, BaseFloat>(), py::arg("likes"),
           py::arg("scale"));
}

template <typename FST>
void pybind_decode_utterance_lattice_faster_impl(
    py::module& m, const std::string& func_name,
    const std::string& func_help_doc = "") {
  m.def(
      func_name.c_str(),
      [](LatticeFasterDecoderTpl<FST>& decoder, DecodableInterface& decodable,
         const TransitionModel& trans_model, const fst::SymbolTable* word_syms,
         std::string utt, double acoustic_scale, bool determinize,
         bool allow_partial, Int32VectorWriter* alignments_writer,
         Int32VectorWriter* words_writer,
         CompactLatticeWriter* compact_lattice_writer,
         LatticeWriter* lattice_writer) -> std::pair<bool, double> {
        // return a pair [is_succeeded, likelihood]
        double likelihood = 0;
        bool is_succeeded = DecodeUtteranceLatticeFaster(
            decoder, decodable, trans_model, word_syms, utt, acoustic_scale,
            determinize, allow_partial, alignments_writer, words_writer,
            compact_lattice_writer, lattice_writer, &likelihood);
        return std::make_pair(is_succeeded, likelihood);
      },
      func_help_doc.c_str(), py::arg("decoder"), py::arg("decodable"),
      py::arg("trans_model"), py::arg("word_syms"), py::arg("utt"),
      py::arg("acoustic_scale"), py::arg("determinize"),
      py::arg("allow_partial"), py::arg("alignments_writer"),
      py::arg("words_writer"), py::arg("compact_lattice_writer"),
      py::arg("lattice_writer"));
}

void pybind_lattice_faster_decoder_config(py::module& m) {
  using PyClass = LatticeFasterDecoderConfig;

  py::class_<PyClass>(m, "LatticeFasterDecoderConfig")
      .def(py::init<>())
      .def_readwrite("beam", &PyClass::beam,
                     "Decoding beam.  Larger->slower, more accurate.")
      .def_readwrite("max_active", &PyClass::max_active,
                     "Decoder max active states. Larger->slower; more accurate")
      .def_readwrite("min_active", &PyClass::min_active,
                     "Decoder minimum #active states.")
      .def_readwrite(
          "lattice_beam", &PyClass::lattice_beam,
          "Lattice generation beam.  Larger->slower, and deeper lattices",
          "Interval (in frames) at which to prune tokens")
      .def_readwrite("prune_interval", &PyClass::prune_interval,
                     "Interval (in frames) at which to prune tokens")
      .def_readwrite("determinize_lattice", &PyClass::determinize_lattice,
                     "If true, determinize the lattice "
                     "(lattice-determinization, keeping only best pdf-sequence "
                     "for each word-sequence).")
      .def_readwrite("beam_delta", &PyClass::beam_delta,
                     "Increment used in decoding-- this parameter is obscure "
                     "and relates to a speedup in the way the max-active "
                     "constraint is applied.  Larger is more accurate.")
      .def_readwrite("hash_ratio", &PyClass::hash_ratio,
                     "Setting used in decoder to control hash behavior")
      .def_readwrite("prune_scale", &PyClass::prune_scale)
      .def_readwrite("det_opts", &PyClass::det_opts)
      .def("Check", &PyClass::Check)
      .def("__str__",
           [](const PyClass& opts) {
             std::ostringstream os;
             os << "beam: " << opts.beam << "\n";
             os << "max_active: " << opts.max_active << "\n";
             os << "lattice_beam: " << opts.lattice_beam << "\n";
             os << "prune_interval: " << opts.prune_interval << "\n";
             os << "determinize_lattice: " << opts.determinize_lattice << "\n";
             os << "beam_delta: " << opts.beam_delta << "\n";
             os << "hash_ratio: " << opts.hash_ratio << "\n";
             os << "prune_scale: " << opts.prune_scale << "\n";

             os << "det_opts:\n";
             os << "  delta: " << opts.det_opts.delta << "\n";
             os << "  max_mem: " << opts.det_opts.max_mem << "\n";
             os << "  phone_determinize: " << opts.det_opts.phone_determinize
                << "\n";
             os << "  word_determinize: " << opts.det_opts.word_determinize
                << "\n";
             os << "  minimize: " << opts.det_opts.minimize << "\n";
             return os.str();
           })
      .def("Register", &PyClass::Register, py::arg("opts"));
}

template <typename FST, typename Token = decoder::StdToken>
void pybind_lattice_faster_decoder_impl(
    py::module& m, const std::string& class_name,
    const std::string& class_help_doc = "") {
  using PyClass = LatticeFasterDecoderTpl<FST, Token>;
  py::class_<PyClass>(m, class_name.c_str(), class_help_doc.c_str())
      .def(py::init<const FST&, const LatticeFasterDecoderConfig&>(),
           "Instantiate this class once for each thing you have to decode. "
           "This version of the constructor does not take ownership of 'fst'.",
           py::arg("fst"), py::arg("config"))
      // TODO(fangjun): how to wrap the constructor taking the ownership of fst
      .def("SetOptions", &PyClass::SetOptions, py::arg("config"))
      .def("GetOptions", &PyClass::GetOptions,
           py::return_value_policy::reference)
      .def("Decode", &PyClass::Decode,
           "Decodes until there are no more frames left in the 'decodable' "
           "object..\n"
           "note, this may block waiting for input if the 'decodable' object "
           "blocks. Returns true if any kind of traceback is available (not "
           "necessarily from a final state).",
           py::arg("decodable"),
      py::call_guard<py::gil_scoped_release>())
      .def("ReachedFinal", &PyClass::ReachedFinal,
           "says whether a final-state was active on the last frame.  If it "
           "was not, the lattice (or traceback) will end with states that are "
           "not final-states.",
      py::call_guard<py::gil_scoped_release>())
      .def("GetBestPath",
           [](const PyClass& decoder,
              bool use_final_probs = true) -> std::pair<bool, Lattice> {
          py::gil_scoped_release release;
             Lattice ofst;
             bool is_succeeded = decoder.GetBestPath(&ofst, use_final_probs);
             return std::make_pair(is_succeeded, ofst);
           },
           "Return a pair [is_succeeded, lattice], where is_succeeded is true "
           "if lattice is NOT empty.\n"
           "If the lattice is not empty, it contains the single best path "
           "through the lattice."
           "\n"
           "If 'use_final_probs' is true AND we reached the final-state of the "
           "graph then it will include those as final-probs, else it will "
           "treat all final-probs as one.  Note: this just calls "
           "GetRawLattice() and figures out the shortest path."
           "\n"
           "Note: Using the return status `is_succeeded` is deprecated, it "
           "will be removed",
           py::arg("use_final_probs") = true)
      .def("GetRawLattice",
           [](const PyClass& decoder,
              bool use_final_probs = true) -> std::pair<bool, Lattice> {
          py::gil_scoped_release release;
             Lattice ofst;
             bool is_succeeded = decoder.GetRawLattice(&ofst, use_final_probs);
             return std::make_pair(is_succeeded, ofst);
           },
           R"doc(
  Return a pair [is_succeeded, lattice], where is_succeeded is true
  if lattice is not empty.

  Outputs an FST corresponding to the raw, state-level
  tracebacks.  Returns true if result is nonempty.
  If "use_final_probs" is true AND we reached the final-state
  of the graph then it will include those as final-probs, else
  it will treat all final-probs as one.
  The raw lattice will be topologically sorted.

  See also GetRawLatticePruned in lattice-faster-online-decoder.h,
  which also supports a pruning beam, in case for some reason
  you want it pruned tighter than the regular lattice beam.
  We could put that here in future needed.
           )doc",
           py::arg("use_final_probs") = true)
      // NOTE(fangjun): we do not wrap the deprecated `GetLattice` method
      .def("InitDecoding", &PyClass::InitDecoding,
           "InitDecoding initializes the decoding, and should only be used if "
           "you intend to call AdvanceDecoding().  If you call Decode(), you "
           "don't need to call this.  You can also call InitDecoding if you "
           "have already decoded an utterance and want to start with a new "
           "utterance.")
      .def("AdvanceDecoding", &PyClass::AdvanceDecoding,
           "This will decode until there are no more frames ready in the "
           "decodable object.  You can keep calling it each time more frames "
           "become available. If max_num_frames is specified, it specifies the "
           "maximum number of frames the function will decode before "
           "returning.",
           py::arg("decodable"), py::arg("max_num_frames") = -1)
      .def("FinalizeDecoding", &PyClass::FinalizeDecoding,
           "This function may be optionally called after AdvanceDecoding(), "
           "when you do not plan to decode any further.  It does an extra "
           "pruning step that will help to prune the lattices output by "
           "GetLattice and (particularly) GetRawLattice more completely, "
           "particularly toward the end of the utterance.  If you call this, "
           "you cannot call AdvanceDecoding again (it will fail), and you "
           "cannot call GetLattice() and related functions with "
           "use_final_probs = false.  Used to be called "
           "PruneActiveTokensFinal().")
      .def("FinalRelativeCost", &PyClass::FinalRelativeCost,
           "FinalRelativeCost() serves the same purpose as ReachedFinal(), but "
           "gives more information.  It returns the difference between the "
           "best (final-cost plus cost) of any token on the final frame, and "
           "the best cost of any token on the final frame.  If it is infinity "
           "it means no final-states were present on the final frame.  It will "
           "usually be nonnegative.  If it not too positive (e.g. < 5 is my "
           "first guess, but this is not tested) you can take it as a good "
           "indication that we reached the final-state with reasonable "
           "likelihood.")
      .def("NumFramesDecoded", &PyClass::NumFramesDecoded,
           "Returns the number of frames decoded so far.  The value returned "
           "changes whenever we call ProcessEmitting().");
}


void pybind_lattice_faster_decoder(py::module& m) {
  pybind_lattice_faster_decoder_config(m);

  using namespace decoder;
  {
    // You are not supposed to use the following classes directly in Python
    auto std_token = py::class_<StdToken>(m, "_StdToken");
    auto backpointer_token =
        py::class_<BackpointerToken>(m, "_BackpointerToken");
    auto forward_link_std_token =
        py::class_<ForwardLink<StdToken>>(m, "_ForwardLinkStdToken");
    auto forward_link_backpointer_token =
        py::class_<ForwardLink<BackpointerToken>>(
            m, "_ForwardLinkBackpointerToken");
  }

  pybind_lattice_faster_decoder_impl<fst::StdFst, StdToken>(
      m, "LatticeFasterDecoder",
      R"doc(This is the "normal" lattice-generating decoder.
See `lattices_generation` `decoders_faster` and `decoders_simple`
for more information.

The decoder is templated on the FST type and the token type.  The token type
will normally be StdToken, but also may be BackpointerToken which is to support
quick lookup of the current best path (see lattice-faster-online-decoder.h)

The FST you invoke this decoder which is expected to equal
Fst::Fst<fst::StdArc>, a.k.a. StdFst, or GrammarFst.  If you invoke it with
FST == StdFst and it notices that the actual FST type is
fst::VectorFst<fst::StdArc> or fst::ConstFst<fst::StdArc>, the decoder object
will internally cast itself to one that is templated on those more specific
types; this is an optimization for speed.)doc");
}


void pybind_training_graph_compiler(py::module &m) {

  {
    using PyClass = TrainingGraphCompilerOptions;

    auto training_graph_compiler_options = py::class_<PyClass>(
        m, "TrainingGraphCompilerOptions");
    training_graph_compiler_options.def(py::init<>())
      .def(py::init<BaseFloat, BaseFloat, bool>(),
                       py::arg("transition_scale") = 1.0,
                       py::arg("self_loop_scale") = 1.0,
                       py::arg("b") = true)
      .def_readwrite("transition_scale", &PyClass::transition_scale)
      .def_readwrite("self_loop_scale", &PyClass::self_loop_scale)
      .def_readwrite("rm_eps", &PyClass::rm_eps)
      .def_readwrite("reorder", &PyClass::reorder)
      .def(py::pickle(
        [](const TrainingGraphCompilerOptions &p) { // __getstate__
            /* Return a tuple that fully encodes the state of the object */
            return py::make_tuple(
                p.transition_scale,
                p.self_loop_scale,
                p.rm_eps,
                p.reorder);
        },
        [](py::tuple t) { // __setstate__
            if (t.size() != 4)
                throw std::runtime_error("Invalid state!");

            /* Create a new C++ instance */
            TrainingGraphCompilerOptions opts;

            /* Assign any additional state */
            opts.transition_scale = t[0].cast<BaseFloat>();
            opts.self_loop_scale = t[1].cast<BaseFloat>();
            opts.rm_eps = t[2].cast<bool>();
            opts.reorder = t[3].cast<bool>();

            return opts;
        }
    ));
  }
  {
     using PyClass = kaldi::TrainingGraphCompiler;
     py::class_<PyClass>(
      m, "TrainingGraphCompiler",
      "TrainingGraphCompiler")
        .def(py::init<const TransitionModel &, const ContextDependency &,
                fst::VectorFst<fst::StdArc> *, const std::vector<int32> &,
                const TrainingGraphCompilerOptions &>(),
                       py::arg("trans_model"),
                       py::arg("ctx_dep"),
                       py::arg("lex_fst"),
                       py::arg("disambig_syms"),
                       py::arg("opts"))
          .def(py::init([](const TransitionModel &trans_model, const ContextDependency &ctx_dep,
                py::object fst, const std::vector<int32> &disambig_syms,
                const TrainingGraphCompilerOptions &opts){
                  auto pywrapfst_mod = py::module_::import("pywrapfst");
                  auto ptr = reinterpret_cast<VectorFstStruct*>(fst.ptr());
                  VectorFst<StdArc>* mf = down_cast<VectorFst<StdArc> *>(ptr->__pyx_base._mfst->GetMutableFst<StdArc>());
                    TrainingGraphCompiler gc(trans_model, ctx_dep, mf, disambig_syms, opts);
                    return gc;
          }))
        .def("CompileGraph",
               &PyClass::CompileGraph,
               "CompileGraph compiles a single training graph its input is a "
               "weighted acceptor (G) at the word level, its output is HCLG. "
               "Note: G could actually be a transducer, it would also work. "
               "This function is not const for technical reasons involving the cache. "
               "if not for \"table_compose\" we could make it const.",
               py::arg("word_grammar"),
               py::arg("out_fst"))
        .def("CompileGraphFromLG",
               [](PyClass& gc, const fst::VectorFst<fst::StdArc> &phone2word_fst){

                    VectorFst<StdArc> decode_fst;

                    if (!gc.CompileGraphFromLG(phone2word_fst, &decode_fst)) {
                         decode_fst.DeleteStates();  // Just make it empty.
                    }
                    return decode_fst;
               },
               "Same as `CompileGraph`, but uses an external LG fst.",
               py::arg("phone2word_fst"))
        .def("CompileGraphFromLG",
               [](PyClass& gc, py::object fst){
                  auto pywrapfst_mod = py::module_::import("pywrapfst");
                  auto ptr = reinterpret_cast<VectorFstStruct*>(fst.ptr());
                  auto vf = ptr->__pyx_base._mfst->GetMutableFst<StdArc>();
                  VectorFst<StdArc> phone2word_fst(*vf);
                    VectorFst<StdArc> decode_fst;

                    if (!gc.CompileGraphFromLG(phone2word_fst, &decode_fst)) {
                         decode_fst.DeleteStates();  // Just make it empty.
                    }
                    return decode_fst;
               },
               "Same as `CompileGraph`, but uses an external LG fst.",
               py::arg("phone2word_fst"))
        .def("CompileGraphFromLG",
               &PyClass::CompileGraphFromLG,
               "Same as `CompileGraph`, but uses an external LG fst.",
               py::arg("phone2word_fst"), py::arg("out_fst"))
        .def("CompileGraphs",
               &PyClass::CompileGraphs,
               "CompileGraphs allows you to compile a number of graphs at the same "
               "time.  This consumes more memory but is faster.",
               py::arg("word_fsts"), py::arg("out_fst"))
        .def("CompileGraphFromText",
               &PyClass::CompileGraphFromText,
               "This version creates an FST from the text and calls CompileGraph.",
               py::arg("transcript"), py::arg("out_fst"))
        .def("CompileGraphFromText",
               [](PyClass& gc, const std::vector<int32> &transcript){

                    VectorFst<StdArc> decode_fst;

                    if (!gc.CompileGraphFromText(transcript, &decode_fst)) {
                         decode_fst.DeleteStates();  // Just make it empty.
                    }
                    return decode_fst;
               },
               py::arg("transcript"))
        .def("CompileGraphsFromText",
               &PyClass::CompileGraphsFromText,
               "This function creates FSTs from the text and calls CompileGraphs.",
               py::arg("transcripts"), py::arg("out_fsts"))
        .def("CompileGraphsFromText",

               [](PyClass& gc, const std::vector<std::vector<int32> > &transcripts){

          py::gil_scoped_release gil_release;
                    std::vector<fst::VectorFst<fst::StdArc>* > fsts;

                    bool ans = gc.CompileGraphsFromText(transcripts, &fsts);
                    return fsts;
               },
               "This function creates FSTs from the text and calls CompileGraphs.",
               py::arg("transcripts"),
                  py::return_value_policy::reference);
        }
}

void init_decoder(py::module &_m) {
py::module m = _m.def_submodule("decoder", "pybind for decoder");

  pybind_decode_utterance_lattice_faster_impl<fst::StdFst>(
      m, "DecodeUtteranceLatticeFaster",
      "Return a pair [is_succeeded, likelihood], where is_succeeded is true if "
      "it decoded successfully."
      "\n"
      "This function DecodeUtteranceLatticeFaster is used in several decoders, "
      "and we have moved it here.  Note: this is really 'binary-level' code as "
      "it involves table readers and writers; we've just put it here as there "
      "is no other obvious place to put it.  If determinize == false, it "
      "writes to lattice_writer, else to compact_lattice_writer. The writers "
      "for alignments and words will only be written to if they are open.");
  // TODO(fangjun): add wrapper for fst::GrammarFst
  // Add wrappers for other functions when needed


  pybind_decoder_faster_decoder(m);
  pybind_decoder_biglm_faster_decoder(m);
  pybind_decoder_decodable_mapped(m);
  pybind_decoder_decodable_sum(m);
  pybind_decoder_wrappers(m);
  pybind_decoder_grammar_fst(m);
  pybind_decoder_lattice_faster_online_decoder(m);
  pybind_decoder_lattice_biglm_faster_decoder(m);
  pybind_decoder_lattice_incremental_decoder(m);
  pybind_decoder_lattice_incremental_online_decoder(m);
  pybind_decoder_lattice_simple_decoder(m);
  pybind_decoder_simple_decoder(m);
  pybind_lattice_faster_decoder(m);
  pybind_decodable_matrix_scale_mapped(m);
  pybind_decodable_matrix_mapped(m);
  pybind_decodable_matrix_mapped_offset(m);
  pybind_decodable_matrix_scaled(m);
  pybind_training_graph_compiler(m);

  m.def("gmm_latgen_faster",
          [](LatticeFasterDecoder &decoder, // not const but is really an input.
    DecodableInterface &decodable, // not const but is really an input.
    const TransitionInformation &trans_model,
    double acoustic_scale,
    bool determinize,
    bool allow_partial){

  double likelihood;
  LatticeWeight weight;
  int32 num_frames;
    VectorFst<LatticeArc> decoded;
  Lattice lat;
    std::vector<int32> alignment;
    std::vector<int32> words;
      bool ans = decoder.Decode(&decodable);
      if (!ans){
         return py::make_tuple(false, alignment, words, lat);
      }
      ans = decoder.ReachedFinal();
      if (!ans){
            if (!allow_partial){

                  return py::make_tuple(false, alignment, words, lat);
            }
      }
      decoder.GetBestPath(&decoded);
    GetLinearSymbolSequence(decoded, &alignment, &words, &weight);
    likelihood = -(weight.Value1() + weight.Value2());
  decoder.GetRawLattice(&lat);
  fst::Connect(&lat);
  if (determinize) {
    CompactLattice clat;
    if (!DeterminizeLatticePhonePrunedWrapper(
            trans_model,
            &lat,
            decoder.GetOptions().lattice_beam,
            &clat,
            decoder.GetOptions().det_opts))
      KALDI_WARN << "Determinization finished earlier than the beam";
    // We'll write the lattice without acoustic scaling.
    if (acoustic_scale != 0.0)
      fst::ScaleLattice(fst::AcousticLatticeScale(1.0 / acoustic_scale), &clat);
      return py::make_tuple(true, alignment, words, clat);
  } else {
    // We'll write the lattice without acoustic scaling.
    if (acoustic_scale != 0.0)
      fst::ScaleLattice(fst::AcousticLatticeScale(1.0 / acoustic_scale), &lat);
      return py::make_tuple(true, alignment, words, lat);
  }

          },
        py::arg("decoder"),
        py::arg("decodable"),
        py::arg("trans_model"),
        py::arg("acoustic_scale"),
        py::arg("determinize"),
        py::arg("allow_partial")
        );


}
