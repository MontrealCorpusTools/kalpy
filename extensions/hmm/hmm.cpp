#include "hmm/pybind_hmm.h"
#include "hmm/hmm-topology.h"
#include "hmm/hmm-utils.h"
#include "hmm/posterior.h"
#include "hmm/transition-model.h"
#include "hmm/tree-accu.h"
#include "util/pybind_util.h"
#include "tree/context-dep.h"
#include "fst/fstlib.h"
#include "fstext/table-matcher.h"
#include "fstext/fstext-utils.h"
#include "fstext/context-fst.h"
#include <pybind11/stl_bind.h>

using namespace kaldi;
using namespace fst;

//PYBIND11_MAKE_OPAQUE(Posterior);

class KalpyGaussPostHolder : public GaussPostHolder {
 public:
    //Inherit the constructors
    using GaussPostHolder::GaussPostHolder;
  T &Value() { return const_cast<T&>(Value()); }

};

void pybind_hmm_topology(py::module &m) {

    {
  using PyClass = HmmTopology;
  auto hmm = py::class_<PyClass>(
      m, "HmmTopology",
      "A class for storing topology information for phones. See `hmm` for "
      "context. This object is sometimes accessed in a file by itself, but "
      "more often as a class member of the Transition class (this is for "
      "convenience to reduce the number of files programs have to access).");

  using State = HmmTopology::HmmState;
  py::class_<State>(
      hmm, "HmmState",
      "A structure defined inside HmmTopology to represent a HMM state.")
      .def(py::init<>())
      .def(py::init<int>(), py::arg("pdf_class"))
      .def(py::init<int, int>(), py::arg("forward_pdf_class"),
           py::arg("self_loop_pdf_class"))
      .def_readwrite("forward_pdf_class", &State::forward_pdf_class)
      .def_readwrite("self_loop_pdf_class", &State::self_loop_pdf_class)
      .def_readwrite("transitions", &State::transitions)
      .def("__eq__", [](const State& s1, const State& s2) { return s1 == s2; })
      .def("__str__", [](const State& s) {
        std::ostringstream os;
        os << "forward_pdf_class: " << s.forward_pdf_class << "\n";
        os << "self_loop_pdf_class: " << s.self_loop_pdf_class << "\n";
        return os.str();
      });

  hmm.def(py::init<>())
      .def("Read", &PyClass::Read, py::arg("is"), py::arg("binary"),
      py::call_guard<py::gil_scoped_release>())
      .def("Write", &PyClass::Write, py::arg("os"), py::arg("binary"),
      py::call_guard<py::gil_scoped_release>())
      .def("Check", &PyClass::Check,
           "Checks that the object is valid, and throw exception otherwise.")
      .def("IsHmm", &PyClass::IsHmm,
           "Returns true if this HmmTopology is really 'hmm-like', i.e. the "
           "pdf-class on the self-loops and forward transitions of all states "
           "are identical. [note: in HMMs, the densities are associated with "
           "the states.] We have extended this to support 'non-hmm-like' "
           "topologies (where those pdf-classes are different), in order to "
           "make for more compact decoding graphs in our so-called 'chain "
           "models' (AKA lattice-free MMI), where we use 1-state topologies "
           "that have different pdf-classes for the self-loop and the forward "
           "transition. Note that we always use the 'reorder=true' option so "
           "the 'forward transition' actually comes before the self-loop.")
      .def("TopologyForPhone", &PyClass::TopologyForPhone,
           "Returns the topology entry (i.e. vector of HmmState) for this "
           "phone; will throw exception if phone not covered by the topology.",
           py::arg("phone"), py::return_value_policy::reference)
      .def("NumPdfClasses", &PyClass::NumPdfClasses,
           "Returns the number of 'pdf-classes' for this phone; throws "
           "exception if phone not covered by this topology.",
           py::arg("phone"))
      .def("GetPhones", &PyClass::GetPhones,
           "Returns a reference to a sorted, unique list of phones covered by "
           "the topology (these phones will be positive integers, and usually "
           "contiguous and starting from one but the toolkit doesn't assume "
           "they are contiguous).",
           py::return_value_policy::reference)
      .def("GetPhoneToNumPdfClasses",
           [](const PyClass& topo) -> std::vector<int> {
             std::vector<int> phone2num_pdf_classes;
             topo.GetPhoneToNumPdfClasses(&phone2num_pdf_classes);
             return phone2num_pdf_classes;
           },
           "Outputs a vector of int32, indexed by phone, that gives the number "
           "of \\ref pdf_class pdf-classes for the phones; this is used by "
           "tree-building code such as BuildTree().")
      .def("MinLength", &PyClass::MinLength,
           "Returns the minimum number of frames it takes to traverse this "
           "model for this phone: e.g. 3 for the normal HMM topology.",
           py::arg("phone"))
      .def("__eq__",
           [](const PyClass& t1, const PyClass& t2) { return t1 == t2; })
      .def("__str__", [](const PyClass& topo) {
        std::ostringstream os;
        bool binary = false;
        topo.Write(os, binary);
        return os.str();
      });
}
}


void pybind_hmm_utils(py::module &m) {

  py::class_<HTransducerConfig>(m, "HTransducerConfig")
      .def(py::init<>())
      .def_readwrite("transition_scale", &HTransducerConfig::transition_scale,
      "Scale of transition probs (relative to LM)")
      .def_readwrite("nonterm_phones_offset", &HTransducerConfig::nonterm_phones_offset,
      "The integer id of #nonterm_bos in phones.txt, if present. "
                   "Only needs to be set if you are doing grammar decoding, "
                   "see doc/grammar.dox.");

  m.def("GetHmmAsFsa",
        &GetHmmAsFsa,
        "Called by GetHTransducer() and probably will not need to be called directly; "
          "it creates and returns the FST corresponding to the phone.  It's actually an "
          "acceptor (ilabels equal to olabels), which is why this is called \"Fsa\" not "
          "\"Fst\".  This acceptor does not include self-loops; you have to call "
          "AddSelfLoops() for that.  (We do that at a later graph compilation phase, "
          "for efficiency).  The labels on the FSA correspond to transition-ids. "
          "\n"
          "as the symbols. "
          "For documentation in context, see \\ref hmm_graph_get_hmm_as_fst "
          "  @param context_window  A vector representing the phonetic context; see "
          "           \\ref tree_window \"here\" for explanation. "
          "  @param ctx_dep The object that contains the phonetic decision-tree "
          "  @param trans_model The transition-model object, which provides "
          "        the mappings to transition-ids and also the transition "
          "        probabilities. "
          "  @param config Configuration object, see \\ref HTransducerConfig. "
          "  @param cache Object used as a lookaside buffer to save computation; "
          "      if it finds that the object it needs is already there, it will "
          "      just return a pointer value from \"cache\"-- not that this means "
          "      you have to be careful not to delete things twice.",
        py::arg("context_window"),
        py::arg("ctx_dep"),
        py::arg("trans_model"),
        py::arg("config"),
        py::arg("cache") = NULL);

  m.def("GetHmmAsFsaSimple",
        &GetHmmAsFsaSimple,
        "Included mainly as a form of documentation, not used in any other code "
     "currently.  Creates the acceptor FST with self-loops, and with fewer "
     "options.",
        py::arg("context_window"),
        py::arg("ctx_dep"),
        py::arg("trans_model"),
        py::arg("prob_scale"));

  m.def("GetHTransducer",
        &GetHTransducer,
        "Returns the H tranducer; result owned by caller.  Caution: our version of "
          "the H transducer does not include self-loops; you have to add those later. "
          "See \\ref hmm_graph_get_h_transducer.  The H transducer has on the "
          "input transition-ids, and also possibly some disambiguation symbols, which "
          "will be put in disambig_syms.  The output side contains the identifiers that "
          "are indexes into \"ilabel_info\" (these represent phones-in-context or "
          "disambiguation symbols).  The ilabel_info vector allows GetHTransducer to map "
          "from symbols to phones-in-context (i.e. phonetic context windows).  Any "
          "singleton symbols in the ilabel_info vector which are not phones, will be "
          "treated as disambiguation symbols.  [Not all recipes use these].  The output "
          "\"disambig_syms_left\" will be set to a list of the disambiguation symbols on "
          "the input of the transducer (i.e. same symbol type as whatever is on the "
          "input of the transducer",
        py::arg("ilabel_info"),
        py::arg("ctx_dep"),
        py::arg("trans_model"),
        py::arg("config"),
        py::arg("disambig_syms_left"));

  m.def("GetIlabelMapping",
        &GetIlabelMapping,
        "GetIlabelMapping produces a mapping that's similar to HTK's logical-to-physical "
          "model mapping (i.e. the xwrd.clustered.mlist files).   It groups together "
          "\"logical HMMs\" (i.e. in our world, phonetic context windows) that share the "
          "same sequence of transition-ids.   This can be used in an "
          "optional graph-creation step that produces a remapped form of CLG that can be "
          "more productively determinized and minimized.  This is used in the command-line program "
          "make-ilabel-transducer.cc. "
          "@param ilabel_info_old [in] The original \\ref tree_ilabel \"ilabel_info\" vector "
          "@param ctx_dep [in] The tree "
          "@param trans_model [in] The transition-model object "
          "@param old2new_map [out] The output; this vector, which is of size equal to the "
          "      number of new labels, is a mapping to the old labels such that we could "
          "      create a vector ilabel_info_new such that "
          "      ilabel_info_new[i] == ilabel_info_old[old2new_map[i]]",
        py::arg("ilabel_info"),
        py::arg("ctx_dep"),
        py::arg("trans_model"),
        py::arg("disambig_syms_left"));

  /*m.def("AddSelfLoops",
        py::overload_cast<const TransitionModel &,
                  const std::vector<int32> &,  // used as a check only.
                  BaseFloat,
                  bool ,
                  bool ,
                  fst::VectorFst<fst::StdArc> *>(&AddSelfLoops),
        "For context, see \\ref hmm_graph_add_self_loops.  Expands an FST that has been "
          "built without self-loops, and adds the self-loops (it also needs to modify "
          "the probability of the non-self-loop ones, as the graph without self-loops "
          "was created in such a way that it was stochastic).  Note that the "
          "disambig_syms will be empty in some recipes (e.g.  if you already removed "
          "the disambiguation symbols). "
          "This function will treat numbers over 10000000 (kNontermBigNumber) the "
          "same as disambiguation symbols, assuming they are special symbols for "
          "grammar decoding. "
          "\n"
          "@param trans_model [in] Transition model "
          "@param disambig_syms [in] Sorted, uniq list of disambiguation symbols, required "
          "      if the graph contains disambiguation symbols but only needed for sanity checks. "
          "@param self_loop_scale [in] Transition-probability scale for self-loops; c.f. "
          "                   \\ref hmm_scale "
          "@param reorder [in] If true, reorders the transitions (see \\ref hmm_reorder). "
          "                    You'll normally want this to be true. "
          "@param check_no_self_loops [in]  If true, it will check that there are no "
          "                     self-loops in the original graph; you'll normally want "
          "                     this to be true.  If false, it will allow them, and "
          "                     will add self-loops after the original self-loop "
          "                     transitions, assuming reorder==true... this happens to "
          "                     be what we want when converting normal to unconstrained "
          "                     chain examples.  WARNING: this was added in 2018; "
          "                     if you get a compilation error, add this as 'true', "
          "                     which emulates the behavior of older code. "
          "@param  fst [in, out] The FST to be modified.",
        py::arg("trans_model"),
        py::arg("disambig_syms"),
        py::arg("self_loop_scale"),
        py::arg("reorder"),
        py::arg("check_no_self_loops"),
        py::arg("fst"));*/

  m.def("AddTransitionProbs",
        py::overload_cast<const TransitionModel &,
                        const std::vector<int32> &,
                        BaseFloat ,
                        BaseFloat ,
                        fst::VectorFst<fst::StdArc> *>(&AddTransitionProbs),
        "Adds transition-probs, with the supplied "
          "scales (see \\ref hmm_scale), to the graph. "
          "Useful if you want to create a graph without transition probs, then possibly "
          "train the model (including the transition probs) but keep the graph fixed, "
          "and add back in the transition probs.  It assumes the fst has transition-ids "
          "on it.  It is not an error if the FST has no states (nothing will be done). "
          "@param trans_model [in] The transition model "
          "@param disambig_syms [in] A list of disambiguation symbols, required if the "
          "                      graph has disambiguation symbols on its input but only "
          "                      used for checks. "
          "@param transition_scale [in] A scale on transition-probabilities apart from "
          "                     those involving self-loops; see \\ref hmm_scale. "
          "@param self_loop_scale [in] A scale on self-loop transition probabilities; "
          "                     see \\ref hmm_scale. "
          "@param  fst [in, out] The FST to be modified.",
        py::arg("trans_model"),
        py::arg("disambig_syms"),
        py::arg("transition_scale"),
        py::arg("self_loop_scale"),
        py::arg("fst"),
                       py::call_guard<py::gil_scoped_release>());

  m.def("AddTransitionProbs",
        py::overload_cast<const TransitionModel &,
                        BaseFloat ,
                        BaseFloat ,
                        Lattice *>(&AddTransitionProbs),
        "This is as AddSelfLoops(), but operates on a Lattice, where "
   "it affects the graph part of the weight (the first element "
   "of the pair).",
        py::arg("trans_model"),
        py::arg("transition_scale"),
        py::arg("self_loop_scale"),
        py::arg("lat"),
                       py::call_guard<py::gil_scoped_release>());

  m.def("GetPdfToTransitionIdTransducer",
        &GetPdfToTransitionIdTransducer,
        "Returns a transducer from pdfs plus one (input) to  transition-ids (output). "
     "Currenly of use only for testing.",
        py::arg("trans_model"));

  m.def("SplitToPhones",
        [](const TransitionModel &trans_model, const std::vector<int32> &alignment){

          py::gil_scoped_release release;
      std::vector<std::vector<int32> > split;
      SplitToPhones(trans_model, alignment, &split);
      return split;
        },
        "SplitToPhones splits up the TransitionIds in \"alignment\" into their "
          "individual phones (one vector per instance of a phone).  At output, "
          "the sum of the sizes of the vectors in split_alignment will be the same "
          "as the corresponding sum for \"alignment\".  The function returns "
          "true on success.  If the alignment appears to be incomplete, e.g. "
          "not ending at the end-state of a phone, it will still break it up into "
          "phones but it will return false.  For more serious errors it will "
          "die or throw an exception. "
          "This function works out by itself whether the graph was created "
          "with \"reordering\", and just does the right thing.",
        py::arg("trans_model"),
        py::arg("alignment"));

  m.def("ConvertAlignment",
        &ConvertAlignment,
        "ConvertAlignment converts an alignment that was created using one model, to "
   "another model.  Returns false if it could not be split to phones (e.g. "
   "because the alignment was partial), or because some other error happened, "
   "such as we couldn't convert the alignment because there were too few frames "
   "for the new topology. "
     "\n"
   "@param old_trans_model [in]  The transition model that the original alignment "
   "                             used. "
   "@param new_trans_model [in]  The transition model that we want to use for the "
   "                             new alignment. "
   "@param new_ctx_dep     [in]  The new tree "
   "@param old_alignment   [in]  The alignment we want to convert "
   "@param subsample_factor [in] The frame subsampling factor... normally 1, but "
   "                             might be > 1 if we're converting to a reduced-frame-rate "
   "                             system. "
   "@param repeat_frames [in]    Only relevant when subsample_factor != 1 "
   "                             If true, repeat frames of alignment by "
   "                             'subsample_factor' after alignment "
   "                             conversion, to keep the alignment the same "
   "                             length as the input alignment. "
   "                             [note: we actually do this by interpolating "
   "                             'subsample_factor' separately generated "
   "                             alignments, to keep the phone boundaries "
   "                             the same as the input where possible.] "
   "@param reorder [in]          True if you want the pdf-ids on the new alignment to "
   "                             be 'reordered'. (vs. the way they appear in "
   "                             the HmmTopology object) "
   "@param phone_map [in]        If non-NULL, map from old to new phones. "
   "@param new_alignment [out]   The converted alignment.",
        py::arg("old_trans_model"),
        py::arg("new_trans_model"),
        py::arg("new_ctx_dep"),
        py::arg("old_alignment"),
        py::arg("subsample_factor"),
        py::arg("repeat_frames"),
        py::arg("reorder"),
        py::arg("phone_map"),
        py::arg("new_alignment"));

  m.def("ConvertPhnxToProns",
        &ConvertPhnxToProns,
        "ConvertPhnxToProns is only needed in bin/phones-to-prons.cc and "
     "isn't closely related with HMMs, but we put it here as there isn't "
     "any other obvious place for it and it needs to be tested. "
     "This function takes a phone-sequence with word-start and word-end "
     "markers in it, and a word-sequence, and outputs the pronunciations "
     "\"prons\"... the format of \"prons\" is, each element is a vector, "
     "where the first element is the word (or zero meaning no word, e.g. "
     "for optional silence introduced by the lexicon), and the remaining "
     "elements are the phones in the word's pronunciation. "
     "It returns false if it encounters a problem of some kind, e.g. "
     "if the phone-sequence doesn't seem to have the right number of "
     "words in it.",
        py::arg("phnx"),
        py::arg("words"),
        py::arg("word_start_sym"),
        py::arg("word_end_sym"),
        py::arg("prons"));

  m.def("GetRandomAlignmentForPhone",
        &GetRandomAlignmentForPhone,
        "Generates a random alignment for this phone, of length equal to "
          "alignment->size(), which is required to be at least the MinLength() of the "
          "topology for this phone, or this function will crash. "
          "The alignment will be without 'reordering'.",
        py::arg("ctx_dep"),
        py::arg("trans_model"),
        py::arg("phone_window"),
        py::arg("alignment"));

  m.def("ChangeReorderingOfAlignment",
        &ChangeReorderingOfAlignment,
        "If the alignment was non-reordered makes it reordered, and vice versa.",
        py::arg("trans_model"),
        py::arg("alignment"));

  m.def("GetPdfToPhonesMap",
        &GetPdfToPhonesMap,
        "GetPdfToPhonesMap creates a map which maps each pdf-id into its "
     "corresponding monophones.",
        py::arg("trans_model"),
        py::arg("pdf2phones"));
}

void pybind_posterior(py::module &m) {

  /*{
     using PyClass = Posterior;

     auto posterior = py::class_<PyClass>(
        m, "Posterior");

     posterior.def(py::init<>())
     .def("weight_silence_post",
          [](Posterior &post,
                         const TransitionModel &trans_model,
                         const ConstIntegerSet<int32> &silence_set,
                         BaseFloat silence_scale,
                         bool distributed = false){
                    if (distributed)
                    WeightSilencePostDistributed(trans_model, silence_set,
                                                  silence_scale, &post);
                    else
                    WeightSilencePost(trans_model, silence_set,
                         silence_scale, &post);
                         },
          "Weight any silence phones in the posterior (i.e. any phones "
     "in the set \"silence_set\" by scale \"silence_scale\". "
     "The interface was changed in Feb 2014 to do the modification "
     "\"in-place\" rather than having separate input and output.",
          py::arg("trans_model"),
          py::arg("silence_set"),
          py::arg("silence_scale"),
          py::arg("distributed") = false
          );
  }*/

  {
     using PyClass = PosteriorHolder;

     auto posterior_holder = py::class_<PyClass>(
        m, "PosteriorHolder");

     posterior_holder.def(py::init<>())
        .def_static("Write", &PyClass::Write,
               py::arg("os"),
               py::arg("binary"),
               py::arg("t"),
      py::call_guard<py::gil_scoped_release>())
        .def("Clear", &PyClass::Clear)
        .def("Read", &PyClass::Read,
               py::arg("is"),
      py::call_guard<py::gil_scoped_release>())
        .def_static("IsReadInBinary", &PyClass::IsReadInBinary)
        .def("Value", &PyClass::Value,
      py::call_guard<py::gil_scoped_release>())
        .def("Swap", &PyClass::Swap,
               py::arg("other"))
        .def("ExtractRange", &PyClass::ExtractRange,
               py::arg("other"),
               py::arg("range"));
  }

  m.def("WritePosterior",
        &WritePosterior,
        "stand-alone function for writing a Posterior.",
        py::arg("os"),
        py::arg("binary"),
        py::arg("post"));

  m.def("ReadPosterior",
        &ReadPosterior,
        "stand-alone function for reading a Posterior.",
        py::arg("os"),
        py::arg("binary"),
        py::arg("post"));

  {
     using PyClass = GaussPostHolder;

     auto gauss_posterior_holder = py::class_<PyClass>(
        m, "GaussPostHolder");

     gauss_posterior_holder.def(py::init<>())
        .def_static("Write", &PyClass::Write,
               py::arg("os"),
               py::arg("binary"),
               py::arg("t"))
        .def("Clear", &PyClass::Clear)
        .def("Read", &PyClass::Read,
               py::arg("is"),
      py::call_guard<py::gil_scoped_release>())
        .def_static("IsReadInBinary", &PyClass::IsReadInBinary)
        .def("Value", &PyClass::Value,
      py::call_guard<py::gil_scoped_release>())
        .def("Swap", &PyClass::Swap,
               py::arg("other"))
        .def("ExtractRange", &PyClass::ExtractRange,
               py::arg("other"),
               py::arg("range"));
        ;
  }

  pybind_table_writer<PosteriorHolder>(m, "PosteriorWriter");
  pybind_sequential_table_reader<PosteriorHolder>(m, "SequentialPosteriorReader");
  pybind_random_access_table_reader<PosteriorHolder>(m, "RandomAccessPosteriorReader");

  pybind_table_writer<GaussPostHolder>(m, "GaussPostWriter");
  pybind_sequential_table_reader<KalpyGaussPostHolder>(m, "SequentialGaussPostReader");
  pybind_random_access_table_reader<GaussPostHolder>(m, "RandomAccessGaussPostReader");


  m.def("ScalePosterior",
        &ScalePosterior,
        "Scales the BaseFloat (weight) element in the posterior entries.",
        py::arg("scale"),
        py::arg("post"),
      py::call_guard<py::gil_scoped_release>());

  m.def("TotalPosterior",
        &TotalPosterior,
        "Returns the total of all the weights in \"post\".",
        py::arg("post"));

  m.def("PosteriorEntriesAreDisjoint",
        &PosteriorEntriesAreDisjoint,
        "Returns true if the two lists of pairs have no common .first element.",
        py::arg("post_elem1"),
        py::arg("post_elem2")
        );

  m.def("MergePosteriors",
        &MergePosteriors,
        "Merge two sets of posteriors, which must have the same length.  If \"merge\" "
          "is true, it will make a common entry whenever there are duplicated entries, "
          "adding up the weights.  If \"drop_frames\" is true, for frames where the "
          "two sets of posteriors were originally disjoint, makes no entries for that "
          "frame (relates to frame dropping, or drop_frames, see Vesely et al, ICASSP "
          "2013).  Returns the number of frames for which the two posteriors were "
          "disjoint (i.e. no common transition-ids or whatever index we are using).",
        py::arg("post1"),
        py::arg("post2"),
        py::arg("merge"),
        py::arg("drop_frames"),
        py::arg("post")
        );

  m.def("VectorToPosteriorEntry",
        &VectorToPosteriorEntry,
        "Given a vector of log-likelihoods (typically of Gaussians in a GMM "
          "but could be of pdf-ids), a number gselect >= 1 and a minimum posterior "
          "0 <= min_post < 1, it gets the posterior for each element of log-likes "
          "by applying Softmax(), then prunes the posteriors using \"gselect\" and "
          "\"min_post\" (keeping at least one), and outputs the result into "
          "\"post_entry\", sorted from greatest to least posterior. "
          "\n"
          "It returns the log of the sum of the selected log-likes that contributed "
          "to the posterior.",
        py::arg("log_likes"),
        py::arg("num_gselect"),
        py::arg("min_post"),
        py::arg("post_entry")
        );

  m.def("AlignmentToPosterior",
          [](const std::vector<int32> &ali){

          py::gil_scoped_release release;
               Posterior post;
               AlignmentToPosterior(ali, &post);
               return post;
          },
        "Convert an alignment to a posterior (with a scale of 1.0 on "
          "each entry).",
        py::arg("ali"),
        py::return_value_policy::reference
        );

  m.def("SortPosteriorByPdfs",
        &SortPosteriorByPdfs,
        "Sorts posterior entries so that transition-ids with same pdf-id are next to "
"each other.",
        py::arg("tmodel"),
        py::arg("post")
        );

  m.def("convert_posterior_to_pdfs",
          [](const TransitionModel &tmodel,
                            const Posterior &post_in){

          py::gil_scoped_release release;
               Posterior post_out;
               ConvertPosteriorToPdfs(tmodel, post_in, &post_out);
               return post_out;
          },
        "Converts a posterior over transition-ids to be a posterior "
"over pdf-ids.",
        py::arg("tmodel"),
        py::arg("post_in")
        );

  m.def("ConvertPosteriorToPdfs",
          &ConvertPosteriorToPdfs,
        "Converts a posterior over transition-ids to be a posterior "
"over pdf-ids.",
        py::arg("tmodel"),
        py::arg("post_in"),
        py::arg("post_out")
        );

  m.def("ConvertPosteriorToPhones",
        &ConvertPosteriorToPhones,
        "Converts a posterior over transition-ids to be a posterior "
"over phones.",
        py::arg("tmodel"),
        py::arg("post_in"),
        py::arg("post_out")
        );

  m.def("WeightSilencePost",
        &WeightSilencePost,
        "Weight any silence phones in the posterior (i.e. any phones "
"in the set \"silence_set\" by scale \"silence_scale\". "
"The interface was changed in Feb 2014 to do the modification "
"\"in-place\" rather than having separate input and output.",
        py::arg("trans_model"),
        py::arg("silence_set"),
        py::arg("silence_scale"),
        py::arg("post")
        );

  m.def("weight_silence_post",
        [](const TransitionModel &trans_model,
                       const ConstIntegerSet<int32> &silence_set,
                       BaseFloat silence_scale,
                       Posterior &post,
                       bool distributed = false){
          py::gil_scoped_release release;
                         if (distributed)
                         WeightSilencePostDistributed(trans_model, silence_set,
                                                       silence_scale, &post);
                         else
                         WeightSilencePost(trans_model, silence_set,
                                   silence_scale, &post);
                          return post;
                       },
        "Weight any silence phones in the posterior (i.e. any phones "
"in the set \"silence_set\" by scale \"silence_scale\". "
"The interface was changed in Feb 2014 to do the modification "
"\"in-place\" rather than having separate input and output.",
        py::arg("trans_model"),
        py::arg("silence_set"),
        py::arg("silence_scale"),
        py::arg("post"),
        py::arg("distributed") = false
        );

  m.def("ali_to_pdf_post",
        [](const std::vector<int32> &ali,
                         const TransitionModel &trans_model,
                       const ConstIntegerSet<int32> &silence_set,
                       BaseFloat silence_scale,
                       bool distributed = false){
          py::gil_scoped_release release;
                    Posterior post;
                    AlignmentToPosterior(ali, &post);
                         if (distributed)
                         WeightSilencePostDistributed(trans_model, silence_set,
                                                       silence_scale, &post);
                         else
                         WeightSilencePost(trans_model, silence_set,
                                   silence_scale, &post);
                         Posterior pdf_post;
                         ConvertPosteriorToPdfs(trans_model, post, &pdf_post);
                          return pdf_post;
                       },
        py::arg("ali"),
        py::arg("trans_model"),
        py::arg("silence_set"),
        py::arg("silence_scale"),
        py::arg("distributed") = false
        );

  m.def("WeightSilencePostDistributed",
        &WeightSilencePostDistributed,
        "This is similar to WeightSilencePost, except that on each frame it "
"works out the amount by which the overall posterior would be reduced, "
"and scales down everything on that frame by the same amount.  It "
"has the effect that frames that are mostly silence get down-weighted. "
"The interface was changed in Feb 2014 to do the modification "
"\"in-place\" rather than having separate input and output.",
        py::arg("trans_model"),
        py::arg("silence_set"),
        py::arg("silence_scale"),
        py::arg("post")
        );

  m.def("PosteriorToMatrix",
        &PosteriorToMatrix<double>,
        "This converts a Posterior to a Matrix. The number of matrix-rows is the same "
     "as the 'post.size()', the number of matrix-columns is defined by 'post_dim'. "
     "The elements which are not specified in 'Posterior' are equal to zero.",
        py::arg("post"),
        py::arg("post_dim"),
        py::arg("mat")
        );

  m.def("PosteriorToMatrix",
        &PosteriorToMatrix<float>,
        "This converts a Posterior to a Matrix. The number of matrix-rows is the same "
     "as the 'post.size()', the number of matrix-columns is defined by 'post_dim'. "
     "The elements which are not specified in 'Posterior' are equal to zero.",
        py::arg("post"),
        py::arg("post_dim"),
        py::arg("mat")
        );

  m.def("PosteriorToPdfMatrix",
        &PosteriorToPdfMatrix<double>,
        "This converts a Posterior to a Matrix. The number of matrix-rows is the same "
"as the 'post.size()', the number of matrix-columns is defined by 'NumPdfs' "
"in the TransitionModel. "
"The elements which are not specified in 'Posterior' are equal to zero.",
        py::arg("post"),
        py::arg("model"),
        py::arg("mat")
        );

  m.def("PosteriorToPdfMatrix",
        &PosteriorToPdfMatrix<float>,
        "This converts a Posterior to a Matrix. The number of matrix-rows is the same "
"as the 'post.size()', the number of matrix-columns is defined by 'NumPdfs' "
"in the TransitionModel. "
"The elements which are not specified in 'Posterior' are equal to zero.",
        py::arg("post"),
        py::arg("model"),
        py::arg("mat")
        );
}

void pybind_transition_model(py::module &m) {

{
  using PyClass = TransitionModel;
  py::class_<PyClass, TransitionInformation>(m, "TransitionModel")
      .def(py::init<>())
      .def(py::init<const ContextDependencyInterface&, const HmmTopology&>(),
           "Initialize the object [e.g. at the start of training]. The class "
           "keeps a copy of the HmmTopology object, but not the "
           "ContextDependency object.",
           py::arg("ctx_dep"), py::arg("hmm_topo"))
     .def_static("read_from_file", [](std::string file_path) {

               static TransitionModel trans_model;
               ReadKaldiObject(file_path, &trans_model);
               return &trans_model;
          }, py::return_value_policy::reference)
      .def("Read", &PyClass::Read, py::arg("is"), py::arg("binary"),
      py::call_guard<py::gil_scoped_release>())
      .def("Write", &PyClass::Write, py::arg("os"), py::arg("binary"),
      py::call_guard<py::gil_scoped_release>())
      .def("GetTopo", &PyClass::GetTopo, py::return_value_policy::reference)
      .def("TupleToTransitionState", &PyClass::TupleToTransitionState,
           py::arg("phone"), py::arg("hmm_state"), py::arg("pdf"),
           py::arg("self_loop_pdf"))
      .def("PairToTransitionId", &PyClass::PairToTransitionId,
           py::arg("trans_state"), py::arg("trans_index"))
      .def("TransitionIdToTransitionState",
           &PyClass::TransitionIdToTransitionState, py::arg("trans_id"))
      .def("TransitionIdToTransitionIndex",
           &PyClass::TransitionIdToTransitionIndex, py::arg("trans_id"))
      .def("TransitionStateToPhone", &PyClass::TransitionStateToPhone,
           py::arg("trans_state"))
      .def("TransitionStateToHmmState", &PyClass::TransitionStateToHmmState,
           py::arg("trans_state"))
      .def("TransitionStateToForwardPdfClass",
           &PyClass::TransitionStateToForwardPdfClass, py::arg("trans_state"))
      .def("TransitionStateToSelfLoopPdfClass",
           &PyClass::TransitionStateToSelfLoopPdfClass, py::arg("trans_state"))
      .def("TransitionStateToForwardPdf", &PyClass::TransitionStateToForwardPdf,
           py::arg("trans_state"))
      .def("TransitionStateToSelfLoopPdf",
           &PyClass::TransitionStateToSelfLoopPdf, py::arg("trans_state"))
      .def("SelfLoopOf", &PyClass::SelfLoopOf,
           "returns the self-loop transition-id, or zero if this state "
           "doesn't have a self-loop.",
           py::arg("trans_state"))
      .def("TransitionIdToPdf", &PyClass::TransitionIdToPdf,
           py::arg("trans_id"))
      .def("TransitionIdToPdfFast", &PyClass::TransitionIdToPdfFast,
           "TransitionIdToPdfFast is as TransitionIdToPdf but skips an "
           "assertion (unless we're in paranoid mode).",
           py::arg("trans_id"))
      .def("TransitionIdToPdfClass", &PyClass::TransitionIdToPdfClass,
           py::arg("trans_id"))
      .def("TransitionIdToHmmState", &PyClass::TransitionIdToHmmState,
           py::arg("trans_id"))
      .def("IsFinal", &PyClass::IsFinal,
           "returns true if this trans_id goes to the final state (which is "
           "bound to be nonemitting).",
           py::arg("trans_id"))
      .def("IsSelfLoop", &PyClass::IsSelfLoop,
           "return true if this trans_id corresponds to a self-loop.",
           py::arg("trans_id"))
      .def("NumTransitionIds", &PyClass::NumTransitionIds,
           "Returns the total number of transition-ids (note, these are "
           "one-based).")
      .def("NumTransitionIndices", &PyClass::NumTransitionIndices,
           "Returns the number of transition-indices for a particular "
           "transition-state. Note: 'Indices' is the plural of 'index'. "
           "Index is not the same as 'id', here. A transition-index is a "
           "zero-based offset into the transitions out of a particular "
           "transition state.",
           py::arg("trans_state"))
      .def("NumTransitionStates", &PyClass::NumTransitionStates,
           "Returns the total number of transition-states (note, these are "
           "one-based).")
      .def("NumPdfs", &PyClass::NumPdfs,
           "NumPdfs() actually returns the highest-numbered pdf we ever saw, "
           "plus one. In normal cases this should equal the number of pdfs "
           "in the system, but if you initialized this object with fewer "
           "than all the phones, and it happens that an unseen phone has the "
           "highest-numbered pdf, this might be different.")
      .def("NumPhones", &PyClass::NumPhones,
           "This loops over the tuples and finds the highest phone index "
           "present. If the FST symbol table for the phones is created in "
           "the expected way, i.e.: starting from 1 (<eps> is 0) and "
           "numbered contiguously till the last phone, this will be the "
           "total number of phones.")
      .def("GetPhones", &PyClass::GetPhones,
           "Returns a sorted, unique list of phones.",
           py::return_value_policy::reference)
      .def("GetTransitionProb", &PyClass::GetTransitionProb,
           py::arg("trans_id"))
      .def("GetTransitionLogProb", &PyClass::GetTransitionLogProb,
           py::arg("trans_id"))
      .def("GetTransitionLogProbIgnoringSelfLoops",
           &PyClass::GetTransitionLogProbIgnoringSelfLoops,
           "Returns the log-probability of a particular non-self-loop "
           "transition after subtracting the probability mass of the "
           "self-loop and renormalizing; will crash if called on a "
           "self-loop.  Specifically: for non-self-loops it returns the log "
           "of (that prob divided by (1 minus "
           "self-loop-prob-for-that-state)).",
           py::arg("trans_id"))
      .def("GetNonSelfLoopLogProb", &PyClass::GetNonSelfLoopLogProb,
           "Returns the log-prob of the non-self-loop probability mass for "
           "this transition state. (you can get the self-loop prob, if a "
           "self-loop exists, by calling "
           "GetTransitionLogProb(SelfLoopOf(trans_state)).",
           py::arg("trans_id"))
      .def("MleUpdate", &PyClass::MleUpdate,
           "Does Maximum Likelihood estimation.  The stats are counts/weights, indexed "
  "by transition-id.  This was previously called Update().",
           py::arg("stats"),
           py::arg("cfg"),
           py::arg("objf_impr_out"),
           py::arg("count_out"))
      .def("mle_update",
          [](PyClass&trans_model, const Vector<double> &stats){
          BaseFloat objf_impr;
          BaseFloat count;
          MleTransitionUpdateConfig tcfg;
          trans_model.MleUpdate(stats, tcfg, &objf_impr, &count);
          return py::make_tuple(objf_impr, count);
          },
           "Does Maximum Likelihood estimation.  The stats are counts/weights, indexed "
  "by transition-id.  This was previously called Update().",
           py::arg("stats"))
      .def("MapUpdate", &PyClass::MapUpdate,
           "Does Maximum A Posteriori (MAP) estimation.  The stats are counts/weights, "
  "indexed by transition-id.",
           py::arg("stats"),
           py::arg("cfg"),
           py::arg("objf_impr_out"),
           py::arg("count_out"))
      .def("Print", &PyClass::Print, py::arg("os"), py::arg("phone_names"),
           py::arg("occs") = nullptr)
      .def("InitStats", &PyClass::InitStats,
           py::arg("stats"))
      .def("Accumulate", &PyClass::Accumulate,
           py::arg("prob"),
           py::arg("trans_id"),
           py::arg("stats"))
      .def("Compatible", &PyClass::Compatible,
           "returns true if all the integer class members are identical (but "
           "does not compare the transition probabilities.")
      .def("Print", &PyClass::Print,
           "Print will print the transition model in a human-readable way, "
           "for purposes of human inspection.  The 'occs' are optional (they "
           "are indexed by pdf-id).",
           py::arg("os"), py::arg("phone_names"), py::arg("occs") = nullptr)
        .def("acc_stats",
                  [](PyClass &trans_model,
                  const std::vector<int32> &alignment,
                                  Vector<double>* transition_accs){
          py::gil_scoped_release gil_release;

            for (size_t i = 0; i < alignment.size(); i++) {
               int32 tid = alignment[i],  // transition identifier.
               pdf_id = trans_model.TransitionIdToPdf(tid);
                    trans_model.Accumulate(1.0, tid, transition_accs);
               }
                  },
              py::arg("alignment"),
              py::arg("transition_accs")
              )
        .def("acc_twofeats",
                  [](PyClass &trans_model,
                                  const Posterior &posterior,
                                  const Matrix<BaseFloat> &mat1,
                                  const Matrix<BaseFloat> &mat2,
                                  Vector<double>* transition_accs){

          py::gil_scoped_release release;
              for (size_t i = 0; i < posterior.size(); i++) {
                    // Accumulates for transitions.
                    for (size_t j = 0; j < posterior[i].size(); j++) {
                    int32 tid = posterior[i][j].first;
                    BaseFloat weight = posterior[i][j].second;
                    trans_model.Accumulate(weight, tid, transition_accs);
                    }
                }

                  },
              py::arg("posterior"),
              py::arg("mat1"),
              py::arg("mat2"),
              py::arg("transition_accs")
              )
      .def("__str__",
           [](const PyClass& mdl) {
             std::ostringstream os;
             bool binary = false;
             mdl.Write(os, binary);
             return os.str();
           })
      .def(py::pickle(
        [](const PyClass &p) { // __getstate__
            /* Return a tuple that fully encodes the state of the object */
             std::ostringstream os;
             bool binary = true;
             p.Write(os, binary);
            return py::make_tuple(
                    py::bytes(os.str()));
        },
        [](py::tuple t) { // __setstate__
            if (t.size() != 1)
                throw std::runtime_error("Invalid state!");

            /* Create a new C++ instance */
            PyClass *p = new PyClass();

            /* Assign any additional state */
            std::istringstream str(t[0].cast<std::string>());
               p->Read(str, true);

            return p;
        }
    ));

  m.def("GetPdfsForPhones",
        [](const TransitionModel& trans_model, const std::vector<int32>& phones)
            -> std::pair<bool, std::vector<int>> {
              std::vector<int> pdfs;
              bool is_succeeded = GetPdfsForPhones(trans_model, phones, &pdfs);
              return std::make_pair(is_succeeded, pdfs);
            },
        "Return a pair of [is_succeeded, pdfs]"
        "\n"
        "Works out which pdfs might correspond to the given phones. Will "
        "return true if these pdfs correspond *just* to these phones, false if "
        "these pdfs are also used by other phones."
        "\n"
        "trans_model [in] Transition-model used to work out this information"
        "\n"
        "phones [in] A sorted, uniq vector that represents a set of phones"
        "\n"
        "pdfs [out] Will be set to a sorted, uniq list of pdf-ids that "
        "correspond to one of this set of phones."
        "\n"
        "is_succeeded is true if all of the pdfs output to 'pdfs' correspond "
        "to phones from just this set (false if they may be shared with phones "
        "outside this set).",
        py::arg("trans_model"), py::arg("phones"));

  m.def(
      "GetPhonesForPdfs",
      [](const TransitionModel& trans_model,
         const std::vector<int32>& pdfs) -> std::pair<bool, std::vector<int>> {
        std::vector<int> phones;
        bool is_succeeded = GetPhonesForPdfs(trans_model, pdfs, &phones);
        return std::make_pair(is_succeeded, phones);
      },
      "Return a pair of [is_succeeded, phones]",
      "\n"
      "Works out which phones might correspond to the given pdfs. Similar "
      "to GetPdfsForPhones(, ,)",
      py::arg("trans_model"), py::arg("pdfs"));
}
}

void pybind_tree_accu(py::module &m) {

  {
    using PyClass = AccumulateTreeStatsOptions;

    auto accumulate_tree_stats_options = py::class_<PyClass>(
        m, "AccumulateTreeStatsOptions");
    accumulate_tree_stats_options.def(py::init<>())
      .def(py::init<>())
      .def_readwrite("var_floor", &PyClass::var_floor)
      .def_readwrite("ci_phones_str", &PyClass::ci_phones_str)
      .def_readwrite("phone_map_rxfilename", &PyClass::phone_map_rxfilename)
      .def_readwrite("collapse_pdf_classes", &PyClass::collapse_pdf_classes)
      .def_readwrite("context_width", &PyClass::context_width)
      .def_readwrite("central_position", &PyClass::central_position);
  }

  {
    using PyClass = AccumulateTreeStatsInfo;

    auto accumulate_tree_stats_info = py::class_<PyClass>(
        m, "AccumulateTreeStatsInfo");
    accumulate_tree_stats_info.def(py::init<const AccumulateTreeStatsOptions &>(),
          py::arg("opts"))
      .def_readwrite("var_floor", &PyClass::var_floor)
      .def_readwrite("ci_phones", &PyClass::ci_phones)
      .def_readwrite("phone_map", &PyClass::phone_map)
      .def_readwrite("context_width", &PyClass::context_width)
      .def_readwrite("central_position", &PyClass::central_position);
  }
  m.def("AccumulateTreeStats",
        &AccumulateTreeStats,
        "Accumulates the stats needed for training context-dependency trees (in the "
          "\"normal\" way).  It adds to 'stats' the stats obtained from this file.  Any "
          "new GaussClusterable* pointers in \"stats\" will be allocated with \"new\".",
        py::arg("trans_model"),
        py::arg("info"),
        py::arg("alignment"),
        py::arg("features"),
        py::arg("stats"),
      py::call_guard<py::gil_scoped_release>());
  m.def("accumulate_tree_stats",
        [](const TransitionModel &trans_model,
                         const AccumulateTreeStatsInfo &info,
                         const std::vector<int32> &alignment,
                         const Matrix<BaseFloat> &features){
          py::gil_scoped_release gil_release;
          std::map<EventType, GaussClusterable*> tree_stats;
          AccumulateTreeStats(trans_model,
                              info,
                              alignment,
                              features,
                              &tree_stats);
          BuildTreeStatsType stats;  // vectorized form.

          for (std::map<EventType, GaussClusterable*>::const_iterator iter = tree_stats.begin();
               iter != tree_stats.end();
               ++iter) {
               stats.push_back(std::make_pair(iter->first, iter->second));
          }
          tree_stats.clear();
          return stats;
        },
        "Accumulates the stats needed for training context-dependency trees (in the "
          "\"normal\" way).  It adds to 'stats' the stats obtained from this file.  Any "
          "new GaussClusterable* pointers in \"stats\" will be allocated with \"new\".",
        py::arg("trans_model"),
        py::arg("info"),
        py::arg("alignment"),
        py::arg("features"));
  m.def("ReadPhoneMap",
        &ReadPhoneMap,
        "Read a mapping from one phone set to another.  The phone map file has lines "
          "of the form <old-phone> <new-phone>, where both entries are integers, usually "
          "nonzero (but this is not enforced).  This program will crash if the input is "
          "invalid, e.g. there are multiple inconsistent entries for the same old phone. "
          "The output vector \"phone_map\" will be indexed by old-phone and will contain "
          "the corresponding new-phone, or -1 for any entry that was not defined.",
        py::arg("phone_map_rxfilename"),
        py::arg("phone_map"));
}


void init_hmm(py::module &_m) {
  py::module m = _m.def_submodule("hmm", "hmm pybind for Kaldi");
  //py::bind_vector<Posterior>(m, "Posterior");
    pybind_hmm_topology(m);
    pybind_hmm_utils(m);
    pybind_transition_model(m);
    pybind_posterior(m);
    pybind_tree_accu(m);

    m.def("make_h_transducer",
          [](
          const ContextDependency &ctx_dep,
          const TransitionModel &trans_model,
               const std::vector<std::vector<int32> > ilabel_info
          ){
          py::gil_scoped_release release;

          HTransducerConfig hcfg;
          std::vector<int32> disambig_syms_out;
          fst::VectorFst<fst::StdArc> *H = GetHTransducer (ilabel_info,
                                                            ctx_dep,
                                                            trans_model,
                                                            hcfg,
                                                            &disambig_syms_out);
          py::gil_scoped_acquire acquire;
               return py::make_tuple(H, disambig_syms_out);
          },
               py::arg("ctx_dep"),
               py::arg("trans_model"),
               py::arg("ilabel_info"));

    m.def("convert_alignments",
          [](
          const TransitionModel &old_trans_model,
                      const TransitionModel &new_trans_model,
                      const ContextDependencyInterface &new_ctx_dep,
                      const std::vector<int32> &old_alignment,
                      int32 frame_subsampling_factor = 1,
                      bool repeat_frames = false,
                      bool reorder = true
          ){
          py::gil_scoped_release gil_release;
          std::vector<int32> new_alignment;

          ConvertAlignment(old_trans_model,
                           new_trans_model,
                           new_ctx_dep,
                           old_alignment,
                           frame_subsampling_factor,
                           repeat_frames,
                           reorder,
                           NULL,
                           &new_alignment);
          return new_alignment;
          },
               py::arg("old_trans_model"),
               py::arg("new_trans_model"),
               py::arg("new_ctx_dep"),
               py::arg("old_alignment"),
               py::arg("frame_subsampling_factor") = 1,
               py::arg("repeat_frames") = false,
               py::arg("reorder") = true);

    m.def("phones_to_prons",
          [](
          const std::vector<int32> &phones,
          const std::vector<int32> &words,
          VectorFst<StdArc> *L,
          int32 word_start_sym, int32 word_end_sym
          ){

          py::gil_scoped_release gil_release;
    {
      // Make sure that L is sorted on output symbol (words).
      fst::OLabelCompare<StdArc> olabel_comp;
      ArcSort(L, olabel_comp);
    }
          VectorFst<StdArc> phn2word;
          {
          VectorFst<StdArc> words_acceptor;
          MakeLinearAcceptor(words, &words_acceptor);
          Compose(*L, words_acceptor, &phn2word);
          }

      VectorFst<StdArc> phones_alt_fst;
      {
          using fst::StdArc;
          typedef fst::StdArc::StateId StateId;
          typedef fst::StdArc::Weight Weight;

          phones_alt_fst.DeleteStates();
          StateId cur_s = phones_alt_fst.AddState();
          phones_alt_fst.SetStart(cur_s); // will be 0.
          for (size_t i = 0; i < phones.size(); i++) {
          StateId next_s = phones_alt_fst.AddState();
          // add arc to next state.
          phones_alt_fst.AddArc(cur_s, StdArc(phones[i], phones[i], Weight::One(),
                                        next_s));
          cur_s = next_s;
          }
          for (StateId s = 0; s <= cur_s; s++) {
          phones_alt_fst.AddArc(s, StdArc(word_end_sym, word_end_sym,
                                             Weight::One(), s));
          phones_alt_fst.AddArc(s, StdArc(word_start_sym, word_start_sym,
                                             Weight::One(), s));
          }
          phones_alt_fst.SetFinal(cur_s, Weight::One());
          {
          fst::OLabelCompare<StdArc> olabel_comp;
          ArcSort(&phones_alt_fst, olabel_comp);
          }
     }

      // phnx2word will have phones and word-start and word-end symbols
      // on the input side, and words on the output side.
      VectorFst<StdArc> phnx2word;
      Compose(phones_alt_fst, phn2word, &phnx2word);
      std::vector<std::vector<int32> > prons;

      if (phnx2word.Start() == fst::kNoStateId) {
        KALDI_WARN << "phnx2word FST "
                   << "is empty (either decoding for this utterance did "
                   << "not reach end-state, or mismatched lexicon.)";
          KALDI_LOG << "phn2word FST is below:";
        return prons;
      }

      // Now get the best path in phnx2word.
      VectorFst<StdArc> phnx2word_best;
      ShortestPath(phnx2word, &phnx2word_best);

      // Now get seqs of phones and words.
      std::vector<int32> phnx, words2;
      StdArc::Weight garbage;
      if (!fst::GetLinearSymbolSequence(phnx2word_best,
                                        &phnx, &words2, &garbage))
        KALDI_ERR << "phnx2word is not a linear transducer (code error?)";
      if (words2 != words)
        KALDI_ERR << "words have changed! (code error?)";

      // Now, "phnx" should be the phone sequence with start and end
      // symbols included.  At this point we break it up into segments,
      // and try to match it up with words.
      if (!ConvertPhnxToProns(phnx, words,
                              word_start_sym, word_end_sym,
                              &prons)) {
        KALDI_WARN << "Error converting phones and words to prons "
                   << " (mismatched or non-marked lexicon or partial "
                   << " alignment?)";
      }
      return prons;
          },
               py::arg("phones"),
               py::arg("words"),
               py::arg("L"),
               py::arg("word_start_sym"),
               py::arg("word_end_sym"));

}
