
#include "fstext/pybind_fstext.h"
#include "util/pybind_util.h"
#include "fst/fst.h"
#include "fstext/context-fst.h"
#include "fstext/deterministic-fst.h"
#include "fstext/determinize-star.h"
#include "fstext/determinize-lattice.h"
#include "fstext/epsilon-property.h"
#include "fstext/factor.h"
#include "fstext/fstext-utils.h"
#include "fstext/grammar-context-fst.h"
#include "fstext/kaldi-fst-io.h"
#include "fstext/lattice-utils.h"
#include "fstext/lattice-weight.h"
#include "fstext/pre-determinize.h"
#include "fstext/prune-special.h"
#include "fstext/push-special.h"
#include "fstext/rand-fst.h"
#include "fstext/remove-eps-local.h"
#include "fstext/table-matcher.h"
#include "fstext/trivial-factor-weight.h"
#include "fst/fstlib.h"
#include "fst/fst-decl.h"

using namespace kaldi;
using namespace fst;

namespace {

template <typename FloatType>
void pybind_lattice_weight_impl(py::module& m, const std::string& class_name,
                                const std::string& class_help_doc = "") {
  using PyClass = fst::LatticeWeightTpl<FloatType>;
  py::class_<PyClass>(m, class_name.c_str(), class_help_doc.c_str())
      .def(py::init<>())
      .def(py::init<FloatType, FloatType>(), py::arg("a"), py::arg("b"))
      .def(py::init<const PyClass&>(), py::arg("other"))
      .def("Value1", &PyClass::Value1)
      .def("Value2", &PyClass::Value2)
      .def("SetValue1", &PyClass::SetValue1, py::arg("f"))
      .def("SetValue2", &PyClass::SetValue2, py::arg("f"))
      .def("Reverse", &PyClass::Reverse)
      .def_static("Zero", &PyClass::Zero)
      .def_static("One", &PyClass::One)
      .def_static("Type", &PyClass::Type)
      .def_static("NoWeight", &PyClass::NoWeight)
      .def("Member", &PyClass::Member)
      .def("Quantize", &PyClass::Quantize, py::arg("delta") = fst::kDelta)
      .def("Properties", &PyClass::Properties)
      .def("Hash", &PyClass::Hash)
      .def("__eq__",
           [](const PyClass& wa, const PyClass& wb) { return wa == wb; })
      .def("__ne__",
           [](const PyClass& wa, const PyClass& wb) { return wa != wb; })
      .def("__str__", [](const PyClass& lat_weight) {
        std::ostringstream os;
        os << "Value1 (lm cost): " << lat_weight.Value1() << "\n";
        os << "Value2 (acoustic cost): " << lat_weight.Value2() << "\n";
        return os.str();
      });

  m.def(
      "ScaleTupleWeight",
      (PyClass(*)(const PyClass&, const std::vector<std::vector<FloatType>>&))(
          &fst::ScaleTupleWeight<FloatType, FloatType>),
      "ScaleTupleWeight is a function defined for LatticeWeightTpl and "
      "CompactLatticeWeightTpl that mutliplies the pair (value1_, value2_) "
      "by a 2x2 matrix.  Used, for example, in applying acoustic scaling.",
      py::arg("w"), py::arg("scale"));
}

template <typename WeightType, typename IntType>
void pybind_compact_lattice_weight_impl(
    py::module& m, const std::string& class_name,
    const std::string& class_help_doc = "") {
  using PyClass = fst::CompactLatticeWeightTpl<WeightType, IntType>;
  py::class_<PyClass>(m, class_name.c_str(), class_help_doc.c_str())
      .def(py::init<>())
      .def(py::init<const WeightType&, const std::vector<IntType>&>(),
           py::arg("w"), py::arg("s"))
      .def("Weight", &PyClass::Weight, py::return_value_policy::reference)
      .def("String", &PyClass::String, py::return_value_policy::reference)
      .def("SetWeight", &PyClass::SetWeight, py::arg("w"))
      .def("SetString", &PyClass::SetString, py::arg("s"))
      .def_static("Zero", &PyClass::Zero)
      .def_static("One", &PyClass::One)
      .def_static("GetIntSizeString", &PyClass::GetIntSizeString)
      .def_static("Type", &PyClass::Type)
      .def_static("NoWeight", &PyClass::NoWeight)
      .def("Reverse", &PyClass::Reverse)
      .def("Member", &PyClass::Member)
      .def("Quantize", &PyClass::Quantize, py::arg("delta") = fst::kDelta)
      .def("Properties", &PyClass::Properties)
      .def("__eq__",
           [](const PyClass& w1, const PyClass& w2) { return w1 == w2; })
      .def("__ne__",
           [](const PyClass& w1, const PyClass& w2) { return w1 != w2; })
      .def("__str__", [](const PyClass& lat_weight) {
        std::ostringstream os;
        os << lat_weight;
        return os.str();
      });
}

}  // namespace



void pybind_kaldi_fst_io(py::module& m) {
  m.def("ReadFstKaldi", (fst::StdVectorFst * (*)(std::string))fst::ReadFstKaldi,
        "Read a binary FST using Kaldi I/O mechanisms (pipes, etc.) On error, "
        "throws using KALDI_ERR.  Note: this doesn't support the text-mode "
        "option that we generally like to support.",
        py::arg("rxfilename"), py::return_value_policy::reference);

  m.def("ReadFstKaldiGeneric", fst::ReadFstKaldiGeneric,
        "Read a binary FST using Kaldi I/O mechanisms (pipes, etc.) If it "
        "can't read the FST, if throw_on_err == true it throws using "
        "KALDI_ERR; otherwise it prints a warning and returns. Note:this "
        "doesn't support the text-mode option that we generally like to "
        "support. This version currently supports ConstFst<StdArc> or "
        "VectorFst<StdArc> (const-fst can give better performance for "
        "decoding).",
        py::arg("rxfilename"), py::arg("throw_on_err") = true,
        py::return_value_policy::reference);

  m.def("CastOrConvertToVectorFst", &fst::CastOrConvertToVectorFst,
        "This function attempts to dynamic_cast the pointer 'fst' (which will "
        "likely have been returned by ReadFstGeneric()), to the more derived "
        "type VectorFst<StdArc>. If this succeeds, it returns the same "
        "pointer; if it fails, it converts the FST type (by creating a new "
        "VectorFst<stdArc> initialized by 'fst'), prints a warning, and "
        "deletes 'fst'.",
        py::arg("fst"), py::return_value_policy::reference);

  m.def("ReadFstKaldi",
        (void (*)(std::string, fst::StdVectorFst*)) & fst::ReadFstKaldi,
        "Version of ReadFstKaldi() that writes to a pointer.  Assumes the FST "
        "is binary with no binary marker.  Crashes on error.",
        py::arg("rxfilename"), py::arg("ofst"));

  m.def("WriteFstKaldi",
        (void (*)(const fst::StdVectorFst&, std::string)) & fst::WriteFstKaldi,
        "Write an FST using Kaldi I/O mechanisms (pipes, etc.) On error, "
        "throws using KALDI_ERR.  For use only in code in fstbin/, as it "
        "doesn't support the text-mode option.",
        py::arg("fst"), py::arg("wxfilename"));

  m.def("WriteFstKaldi",
        (void (*)(std::ostream&, bool, const fst::StdVectorFst&)) &
            fst::WriteFstKaldi,
        "This is a more general Kaldi-type-IO mechanism of writing FSTs to "
        "streams, supporting binary or text-mode writing.  (note: we just "
        "write the integers, symbol tables are not supported). On error, "
        "throws using KALDI_ERR.",
        py::arg("os"), py::arg("binary"), py::arg("fst"));

  m.def("ReadFstKaldi",
        (void (*)(std::istream&, bool, fst::StdVectorFst*)) & fst::ReadFstKaldi,
        "A generic Kaldi-type-IO mechanism of reading FSTs from streams, "
        "supporting binary or text-mode reading/writing.",
        py::arg("is"), py::arg("binary"), py::arg("fst"));
  m.def("ReadAndPrepareLmFst", &fst::ReadAndPrepareLmFst,
        "Read an FST file for LM (G.fst) and make it an acceptor, and make "
        "sure it is sorted on labels",
        py::arg("rxfilename"), py::return_value_policy::reference);

  {
    // fangjun: it should be called StdVectorFstHolder to match the naming
    // convention in OpenFst but kaldi uses only StdArc so there is no confusion
    // here.
    using PyClass = fst::VectorFstHolder;
    py::class_<PyClass>(m, "VectorFstHolder")
        .def(py::init<>())
        .def_static("Write", &PyClass::Write, py::arg("os"), py::arg("binary"),
                    py::arg("t"))
        .def("Copy", &PyClass::Copy)
        .def("Read", &PyClass::Read, "Reads into the holder.", py::arg("is"));
  }
}

template<class F>
void pybind_table_matcher(py::module &m){
{
    py::class_<MatcherBase<typename F::Arc>>(m, "MatcherBase", "MatcherBase");
  auto tm = py::class_<TableMatcher<F>, MatcherBase<typename F::Arc>>(
      m, "TableMatcher",
      "TableMatcher");
    tm.def(py::init<const F &, MatchType,
                const TableMatcherOptions &>())
        .def("SetState", &TableMatcher<F>::SetState, py::arg("s"))
        .def("Find", &TableMatcher<F>::Find, py::arg("match_label"))
        .def("Next", &TableMatcher<F>::Next)
        .def("Done", &TableMatcher<F>::Done);
  auto tcc = py::class_<TableComposeCache<F>>(m, "TableComposeCache");
    tcc.def(py::init<const TableComposeOptions &>());
        }
}
template<class Arc>
void pybind_table_compose(py::module &m){

    m.def("TableCompose", (void (*)(const Fst<Arc> &,
    const Fst<Arc> &, MutableFst<Arc> *, const TableComposeOptions &))(&TableCompose<Arc>));
    m.def("TableCompose", (void (*)(const Fst<Arc> &,
    const Fst<Arc> &, MutableFst<Arc> *, TableComposeCache<Fst<Arc>> *))(&TableCompose<Arc>));
}

void pybind_fstext_context_fst(py::module &m) {
      using namespace fst;
  m.def("WriteILabelInfo",
        &WriteILabelInfo,
        "Utility function for writing ilabel-info vectors to disk.",
        py::arg("os"),
        py::arg("binary"),
        py::arg("ilabel_info"));
  m.def("ReadILabelInfo",
        &ReadILabelInfo,
        "Utility function for reading ilabel-info vectors from disk.",
        py::arg("is"),
        py::arg("binary"),
        py::arg("ilabel_info"));
  m.def("CreateILabelInfoSymbolTable",
        &CreateILabelInfoSymbolTable,
        "The following function is mainly of use for printing and debugging.",
        py::arg("ilabel_info"),
        py::arg("phones_symtab"),
        py::arg("separator"),
        py::arg("disambig_prefix"));
  m.def("ComposeContext",
        py::overload_cast<const std::vector<int32> &,
                    int32, int32,
                    VectorFst<StdArc> *,
                    VectorFst<StdArc> *,
                    std::vector<std::vector<int32> > *,
                    bool>(&ComposeContext),
        "Used in the command-line tool fstcomposecontext.  It creates a context FST and "
        "composes it on the left with \"ifst\" to make \"ofst\".  It outputs the label "
        "information to ilabels_out.  \"ifst\" is mutable because we need to add the "
        "subsequential loop. "
        "\n"
        "@param [in] disambig_syms  List of disambiguation symbols, e.g. the integer "
        "            ids of #0, #1, #2 ... in the phones.txt. "
        "@param [in] context_width  Size of context window, e.g. 3 for triphone. "
        "@param [in] central_position  Central position in phonetic context window "
        "               (zero-based index), e.g. 1 for triphone. "
        "@param [in,out] ifst   The FST we are composing with C (e.g. LG.fst), mustable because "
        "                  we need to add the subsequential loop to it. "
        "@param [out] ofst   Composed output FST (would be CLG.fst). "
        "@param [out] ilabels_out  Vector, indexed by ilabel of CLG.fst, providing information "
        "                  about the meaning of that ilabel; see "
        "                \"http://kaldi-asr.org/doc/tree_externals.html#tree_ilabel\". "
        "@param [in] project_ifst  This is intended only to be set to true "
        "                  in the program 'fstmakecontextfst'... if true, it will "
        "                  project on the input after adding the subsequential loop "
        "                  to 'ifst', which allows us to reconstruct the context "
        "                  fst C.fst.",
        py::arg("disambig_syms"),
        py::arg("context_width"),
        py::arg("central_position"),
        py::arg("ifst"),
        py::arg("ofst"),
        py::arg("ilabels_out"),
        py::arg("project_ifst") = false);
  m.def("AddSubsequentialLoop",
      &AddSubsequentialLoop,
      "Modifies an FST so that it transuces the same paths, but the input side of the "
      "paths can all have the subsequential symbol '$' appended to them any number of "
      "times (we could easily specify the number of times, but accepting any number of "
      "repetitions is just more convenient).  The actual way we do this is for each "
      "final state, we add a transition with weight equal to the final-weight of that "
      "state, with input-symbol '$' and output-symbols <eps>, and ending in a new "
      "super-final state that has unit final-probability and a unit-weight self-loop "
      "with '$' on its input and <eps> on its output.  The reason we don't just "
      "add a loop to each final-state has to do with preserving stochasticity "
      "(see \ref fst_algo_stochastic).  We keep the final-probability in all the "
      "original final-states rather than setting them to zero, so the resulting FST "
      "can accept zero '$' symbols at the end (in case we had no right context).",
      py::arg("subseq_symbol"),
      py::arg("fst"));
  {
    using PyClass = InverseContextFst;

    auto inverse_context_fst = py::class_<InverseContextFst, DeterministicOnDemandFst<StdArc>>(
        m, "InverseContextFst");
    inverse_context_fst.def(py::init<PyClass::Label,
                    const std::vector<int32>&,
                    const std::vector<int32>&,
                    int32,
                    int32>(),
            py::arg("subsequential_symbol"),
            py::arg("phones"),
            py::arg("disambig_syms"),
            py::arg("context_width"),
            py::arg("central_position"))
      .def("Start", &PyClass::Start)
      .def("Final", &PyClass::Final,
            py::arg("s"))
      .def("GetArc", &PyClass::GetArc,
            py::arg("s"),
            py::arg("ilabel"),
            py::arg("arc"))
      .def("IlabelInfo", &PyClass::IlabelInfo)
      .def("SwapIlabelInfo", &PyClass::SwapIlabelInfo);
  }
}

class PyDeterministicOnDemandFst : public DeterministicOnDemandFst<StdArc> {
public:
    //Inherit the constructors
    using DeterministicOnDemandFst::DeterministicOnDemandFst;

};

void pybind_fstext_deterministic_fst(py::module &m) {
  {
    using PyClass = DeterministicOnDemandFst<StdArc>;

    auto deterministic_on_demand_fst = py::class_<DeterministicOnDemandFst<StdArc>, PyDeterministicOnDemandFst>(
        m, "DeterministicOnDemandFst");
    deterministic_on_demand_fst
      .def("Start", &PyClass::Start)
      .def("Final", &PyClass::Final,
            py::arg("s"))
      .def("GetArc", &PyClass::GetArc,
            py::arg("s"),
            py::arg("ilabel"),
            py::arg("oarc"));
  }
  {
    using PyClass = BackoffDeterministicOnDemandFst<StdArc>;

    auto backoff_deterministic_on_demand_fst = py::class_<BackoffDeterministicOnDemandFst<StdArc>, DeterministicOnDemandFst<StdArc>>(
        m, "BackoffDeterministicOnDemandFst");
    backoff_deterministic_on_demand_fst.def(py::init<const Fst<StdArc> &>(),
            py::arg("fst"))
      .def("Start", &PyClass::Start)
      .def("Final", &PyClass::Final,
            py::arg("s"))
      .def("GetArc", &PyClass::GetArc,
            py::arg("s"),
            py::arg("ilabel"),
            py::arg("oarc"));
  }
  {
    using PyClass = ScaleDeterministicOnDemandFst;

    auto scaled_deterministic_on_demand_fst = py::class_<ScaleDeterministicOnDemandFst, DeterministicOnDemandFst<StdArc>>(
        m, "ScaleDeterministicOnDemandFst");
    scaled_deterministic_on_demand_fst.def(py::init<float,
                                DeterministicOnDemandFst<StdArc> *>(),
            py::arg("scale"),
            py::arg("det_fst"))
      .def("Start", &PyClass::Start)
      .def("Final", &PyClass::Final,
            py::arg("s"))
      .def("GetArc", &PyClass::GetArc,
            py::arg("s"),
            py::arg("ilabel"),
            py::arg("oarc"));
  }
  {
    using PyClass = UnweightedNgramFst<StdArc>;

    auto unweighted_ngram_fst = py::class_<UnweightedNgramFst<StdArc>, DeterministicOnDemandFst<StdArc>>(
        m, "UnweightedNgramFst");
    unweighted_ngram_fst.def(py::init<int>(),
            py::arg("n"))
      .def("Start", &PyClass::Start)
      .def("Final", &PyClass::Final,
            py::arg("s"))
      .def("GetArc", &PyClass::GetArc,
            py::arg("s"),
            py::arg("ilabel"),
            py::arg("oarc"));
  }
  {
    using PyClass = ComposeDeterministicOnDemandFst<StdArc>;

    auto compose_deterministic_on_demand_fst = py::class_<ComposeDeterministicOnDemandFst<StdArc>, DeterministicOnDemandFst<StdArc>>(
        m, "ComposeDeterministicOnDemandFst");
    compose_deterministic_on_demand_fst.def(py::init<DeterministicOnDemandFst<StdArc> *,
                                  DeterministicOnDemandFst<StdArc> *>(),
            py::arg("fst1"),
            py::arg("fst2"))
      .def("Start", &PyClass::Start)
      .def("Final", &PyClass::Final,
            py::arg("s"))
      .def("GetArc", &PyClass::GetArc,
            py::arg("s"),
            py::arg("ilabel"),
            py::arg("oarc"));
  }
  {
    using PyClass = CacheDeterministicOnDemandFst<StdArc>;

    auto cache_deterministic_on_demand_fst = py::class_<CacheDeterministicOnDemandFst<StdArc>, DeterministicOnDemandFst<StdArc>>(
        m, "CacheDeterministicOnDemandFst");
    cache_deterministic_on_demand_fst.def(py::init<DeterministicOnDemandFst<StdArc> *,
                                PyClass::StateId>(),
            py::arg("fst"),
            py::arg("num_cached_arcs") = 100000)
      .def("Start", &PyClass::Start)
      .def("Final", &PyClass::Final,
            py::arg("s"))
      .def("GetArc", &PyClass::GetArc,
            py::arg("s"),
            py::arg("ilabel"),
            py::arg("oarc"));
  }
  {
    using PyClass = LmExampleDeterministicOnDemandFst<StdArc>;

    auto lm_example_deterministic_on_demand_fst = py::class_<LmExampleDeterministicOnDemandFst<StdArc>, DeterministicOnDemandFst<StdArc>>(
        m, "LmExampleDeterministicOnDemandFst");
    lm_example_deterministic_on_demand_fst.def(py::init<void *,
                                    PyClass::Label,
                                    PyClass::Label>(),
            py::arg("lm"),
            py::arg("bos_symbol"),
            py::arg("eos_symbol"))
      .def("Start", &PyClass::Start)
      .def("Final", &PyClass::Final,
            py::arg("s"))
      .def("GetArc", &PyClass::GetArc,
            py::arg("s"),
            py::arg("ilabel"),
            py::arg("oarc"));
  }
  m.def("ComposeDeterministicOnDemand",
        &ComposeDeterministicOnDemand<StdArc>,
        "Compose an FST (which may be a lattice) with a DeterministicOnDemandFst and "
            "store the result in fst_composed.  This is mainly used for expanding lattice "
            "n-gram histories, where fst1 is a lattice and fst2 is an UnweightedNgramFst. "
            "This does not call Connect.",
        py::arg("fst1"),
        py::arg("fst2"),
        py::arg("fst_composed"));
  m.def("ComposeDeterministicOnDemandInverse",
        &ComposeDeterministicOnDemandInverse<StdArc>,
        "This function does "
            "'*fst_composed = Compose(Inverse(*fst2), fst1)' "
            "Note that the arguments are reversed; this is unfortunate but it's "
            "because the fst2 argument needs to be non-const and non-const arguments "
            "must follow const ones. "
            "This is the counterpart to ComposeDeterministicOnDemand, used for "
            "the case where the DeterministicOnDemandFst is on the left.  The "
            "reason why we need to make the left-hand argument to compose the "
            "inverse of 'fst2' (i.e. with the input and output symbols swapped), "
            "is that the DeterministicOnDemandFst interface only supports lookup "
            "by ilabel (see its function GetArc). "
            "This does not call Connect().",
        py::arg("fst1"),
        py::arg("fst2"),
        py::arg("fst_composed"));

}

void pybind_fstext_deterministic_lattice(py::module &m) {

  py::class_<DeterminizeLatticeOptions>(m, "DeterminizeLatticeOptions")
      .def(py::init<>())
      .def_readwrite("delta", &DeterminizeLatticeOptions::delta)
      .def_readwrite("max_mem", &DeterminizeLatticeOptions::max_mem)
      .def_readwrite("max_loop", &DeterminizeLatticeOptions::max_loop);

  m.def("DeterminizeLattice",
        py::overload_cast<const Fst<ArcTpl<TropicalWeight> > &,
    MutableFst<ArcTpl<TropicalWeight> > *,
    DeterminizeLatticeOptions,
    bool *>(&DeterminizeLattice<TropicalWeight, int32>),
        "This function implements the normal version of DeterminizeLattice, in which "
            "the output strings are represented using sequences of arcs, where all but "
            "the first one has an epsilon on the input side.  The debug_ptr argument is "
            "an optional pointer to a bool that, if it becomes true while the algorithm "
            "is executing, the algorithm will print a traceback and terminate (used in "
            "fstdeterminizestar.cc debug non-terminating determinization).  More "
            "efficient if ifst is arc-sorted on input label.  If the number of arcs gets "
            "more than max_states, it will throw std::runtime_error (otherwise this code "
            "does not use exceptions).  This is mainly useful for debug.",
        py::arg("ifst"),
        py::arg("ofst"),
        py::arg("opts") = DeterminizeLatticeOptions(),
        py::arg("debug_ptr") = NULL);
  m.def("DeterminizeLattice",
        py::overload_cast<const Fst<ArcTpl<TropicalWeight> >&,
    MutableFst<ArcTpl<CompactLatticeWeightTpl<TropicalWeight, int32> > > *,
    DeterminizeLatticeOptions,
    bool *>(&DeterminizeLattice<TropicalWeight, int32>),
        "This is a version of DeterminizeLattice with a slightly more \"natural\" output format, "
            "where the output sequences are encoded using the CompactLatticeArcTpl template "
            "(i.e. the sequences of output symbols are represented directly as strings) "
            "More efficient if ifst is arc-sorted on input label. "
            "If the #arcs gets more than max_arcs, it will throw std::runtime_error (otherwise "
            "this code does not use exceptions).  This is mainly useful for debug.",
        py::arg("ifst"),
        py::arg("ofst"),
        py::arg("opts") = DeterminizeLatticeOptions(),
        py::arg("debug_ptr") = NULL);
}

void pybind_fstext_determinize_star(py::module &m) {
  m.def("DeterminizeStar",
        py::overload_cast<VectorFst<StdArc> &, MutableFst<StdArc> *,
                     float,
                     bool *,
                     int ,
                     bool >(&DeterminizeStar<VectorFst<StdArc>>),
        "This function implements the normal version of DeterminizeStar, in which the "
            "output strings are represented using sequences of arcs, where all but the "
            "first one has an epsilon on the input side.  The debug_ptr argument is an "
            "optional pointer to a bool that, if it becomes true while the algorithm is "
            "executing, the algorithm will print a traceback and terminate (used in "
            "fstdeterminizestar.cc debug non-terminating determinization). "
            "If max_states is positive, it will stop determinization and throw an "
            "exception as soon as the max-states is reached. This can be useful in test. "
            "If allow_partial is true, the algorithm will output partial results when the "
            "specified max_states is reached (when larger than zero), instead of throwing "
            "out an error. "
            "\n"
            "Caution, the return status is un-intuitive: this function will return false if "
            "determinization completed normally, and true if it was stopped early by "
            "reaching the 'max-states' limit, and a partial FST was generated.",
        py::arg("ifst"),
        py::arg("ofst"),
        py::arg("delta") = kDelta,
        py::arg("debug_ptr") = NULL,
        py::arg("max_states") = -1,
        py::arg("allow_partial") = false);

  m.def("DeterminizeStar",
        py::overload_cast<VectorFst<StdArc> &, MutableFst<GallicArc<StdArc>> *,
                     float,
                     bool *,
                     int ,
                     bool >(&DeterminizeStar<VectorFst<StdArc>>),
        "This is a version of DeterminizeStar with a slightly more \"natural\" output format, "
            "where the output sequences are encoded using the GallicArc (i.e. the output symbols "
            "are strings. "
            "If max_states is positive, it will stop determinization and throw an "
            "exception as soon as the max-states is reached.  This can be useful in test. "
            "If allow_partial is true, the algorithm will output partial results when the "
            "specified max_states is reached (when larger than zero), instead of throwing "
            "out an error. "
            "\n"
            "Caution, the return status is un-intuitive: this function will return false if "
            "determinization completed normally, and true if it was stopped early by "
            "reaching the 'max-states' limit, and a partial FST was generated.",
        py::arg("ifst"),
        py::arg("ofst"),
        py::arg("delta") = kDelta,
        py::arg("debug_ptr") = NULL,
        py::arg("max_states") = -1,
        py::arg("allow_partial") = false);
}
void pybind_fstext_epsilon_property(py::module &m) {

        enum EpsilonInfo {
            kStateHasEpsilonArcsEntering = 0x1,
            kStateHasNonEpsilonArcsEntering = 0x2,
            kStateHasEpsilonArcsLeaving = 0x4,
            kStateHasNonEpsilonArcsLeaving = 0x8
            };
  py::enum_<EpsilonInfo>(m, "EpsilonInfo")
    .value("kStateHasEpsilonArcsEntering", EpsilonInfo::kStateHasEpsilonArcsEntering)
    .value("kStateHasNonEpsilonArcsEntering", EpsilonInfo::kStateHasNonEpsilonArcsEntering)
    .value("kStateHasEpsilonArcsLeaving", EpsilonInfo::kStateHasEpsilonArcsLeaving)
    .value("kStateHasNonEpsilonArcsLeaving", EpsilonInfo::kStateHasNonEpsilonArcsLeaving)
    .export_values();

  m.def("ComputeStateInfo",
        py::overload_cast<const VectorFst<StdArc> &,
                      std::vector<char> *>(&ComputeStateInfo<StdArc>),
        "This function will set epsilon_info to have size equal to the "
      "NumStates() of the FST, containing a logical-or of the enum "
      "values kStateHasEpsilonArcsEntering, kStateHasNonEpsilonArcsEntering, "
      "kStateHasEpsilonArcsLeaving, and kStateHasNonEpsilonArcsLeaving. "
      "The meaning should be obvious.  Note: an epsilon arc is defined "
      "as an arc where ilabel == olabel == 0.",
        py::arg("fst"),
        py::arg("epsilon_info"));

  m.def("EnsureEpsilonProperty",
        py::overload_cast<VectorFst<StdArc> *>(&EnsureEpsilonProperty<StdArc>),
        "This function modifies the fst (while maintaining equivalence) in such a way "
      "that, after the modification, all states of the FST which have epsilon-arcs "
      "entering them, have no non-epsilon arcs entering them, and all states which "
      "have epsilon-arcs leaving them, have no non-epsilon arcs leaving them.  It does "
      "this by creating extra states and adding extra epsilon transitions.  An epsilon "
      "arc is defined as an arc where both the ilabel and the olabel are epsilons. "
      "This function may fail with KALDI_ASSERT for certain cyclic FSTs, but is safe "
      "in the acyclic case.",
        py::arg("fst"));


}
void pybind_fstext_factor(py::module &m) {

  m.def("Factor",
        py::overload_cast<const Fst<StdArc> &, MutableFst<StdArc> *,
            std::vector<std::vector<int32> > *>(&Factor<StdArc, int32>),
        "Factor identifies linear chains of states with an olabel (if any) "
            "only on the first arc of the chain, and possibly a sequence of "
            "ilabels; it outputs an FST with different symbols on the input "
            "that represent sequences of the original input symbols; it outputs "
            "the mapping from the new symbol to sequences of original symbols, "
            "as \"symbols\" [zero is reserved for epsilon]. "
            "\n"
            "As a side effect it also sorts the FST in depth-first order.  Factor will "
            "usually do the best job when the olabels have been pushed to the left, "
            "i.e. if you make a call like "
            "\n"
            "Push<Arc, REWEIGHT_TO_INITIAL>(fsta, &fstb, kPushLabels); "
            "\n"
            "This is because it only creates a chain with olabels on the first arc of the "
            "chain (or a chain with no olabels). [it's possible to construct cases where "
            "pushing makes things worse, though].  After Factor, the composition of *ofst "
            "with the result of calling CreateFactorFst(*symbols) should be equivalent to "
            "fst.  Alternatively, calling ExpandInputSequences with ofst and *symbols "
            "would produce something equivalent to fst.",
        py::arg("fst"),
        py::arg("ofst"),
        py::arg("symbols"));

  m.def("Factor",
        py::overload_cast<const Fst<StdArc> &, MutableFst<StdArc> *,
            MutableFst<StdArc> *>(&Factor<StdArc>),
        "This is a more conventional interface of Factor that outputs "
            "the result as two FSTs.",
        py::arg("fst"),
        py::arg("ofst1"),
        py::arg("ofst2"));

  m.def("ExpandInputSequences",
        py::overload_cast<const std::vector<std::vector<int32> > &,
                          MutableFst<StdArc> *>(&ExpandInputSequences<StdArc, int32>),
        "ExpandInputSequences expands out the input symbols into sequences of input "
            "symbols.  It creates linear chains of states for each arc that had >1 "
            "augmented symbol on it.  It also sets the input symbol table to NULL, since "
            "in case you did have a symbol table there it would no longer be valid.  It "
            "leaves any weight and output symbols on the first arc of the chain.",
        py::arg("sequences"),
        py::arg("fst"));

  m.def("CreateFactorFst",
        &CreateFactorFst<StdArc, int32>,
        "The function CreateFactorFst will create an FST that expands out the "
            "\"factors\" that are the indices of the \"sequences\" array, into linear sequences "
            "of symbols.  There is a single start and end state (state 0), and for each "
            "nonzero index i into the array \"sequences\", there is an arc from state 0 that "
            "has output-label i, and enters a chain of states with output epsilons and input "
            "labels corresponding to the remaining elements of the sequences, terminating "
            "again in state 0.  This FST is output-deterministic and sorted on olabel. "
            "Composing an FST on the left with the output of this function, should be the "
            "same as calling \"ExpandInputSequences\".  Use TableCompose (see table-matcher.h) "
            "for efficiency.",
        py::arg("sequences"),
        py::arg("fst"));

  m.def("CreateMapFst",
        &CreateMapFst<StdArc, int32>,
        "CreateMapFst will create an FST representing this symbol_map.  The "
            "FST has a single loop state with single-arc loops with "
            "isymbol = symbol_map[i], osymbol = i.  The resulting FST applies this "
            "map to the input symbols of something we compose with it on the right. "
            "Must have symbol_map[0] == 0.",
        py::arg("symbol_map"),
        py::arg("fst"));

  py::enum_<StatePropertiesEnum>(m, "StatePropertiesEnum")
    .value("kStateFinal", StatePropertiesEnum::kStateFinal)
    .value("kStateInitial", StatePropertiesEnum::kStateInitial)
    .value("kStateArcsIn", StatePropertiesEnum::kStateArcsIn)
    .value("kStateMultipleArcsIn", StatePropertiesEnum::kStateMultipleArcsIn)
    .value("kStateArcsOut", StatePropertiesEnum::kStateArcsOut)
    .value("kStateMultipleArcsOut", StatePropertiesEnum::kStateMultipleArcsOut)
    .value("kStateOlabelsOut", StatePropertiesEnum::kStateOlabelsOut)
    .value("kStateIlabelsOut", StatePropertiesEnum::kStateIlabelsOut)
    .export_values();

  m.def("GetStateProperties",
        &GetStateProperties<StdArc>,
        "This function works out various properties of the states in the "
            "FST, using the bit properties defined in StatePropertiesEnum.",
        py::arg("fst"),
        py::arg("max_state"),
        py::arg("props"));
  {
    using PyClass = DfsOrderVisitor<StdArc>;

    auto dfs_order_vistor = py::class_<PyClass>(
        m, "DfsOrderVisitor");
    dfs_order_vistor.def(py::init<std::vector<StdArc::StateId> *>(),
            py::arg("order"));
  }
}

void pybind_fstext_fstext_utils(py::module &m) {

  m.def("HighestNumberedOutputSymbol",
        &HighestNumberedOutputSymbol<StdArc>,
        "Returns the highest numbered output symbol id of the FST (or zero "
      "for an empty FST.",
        py::arg("fst"));

  m.def("HighestNumberedInputSymbol",
        &HighestNumberedInputSymbol<StdArc>,
        "Returns the highest numbered input symbol id of the FST (or zero "
      "for an empty FST.",
        py::arg("fst"));

  m.def("NumArcs",
        &NumArcs<StdArc>,
        "Returns the total number of arcs in an FST.",
        py::arg("fst"));

  m.def("GetInputSymbols",
        &GetInputSymbols<StdArc, int32>,
        "GetInputSymbols gets the list of symbols on the input of fst "
            "(including epsilon, if include_eps == true), as a sorted, unique "
            "list.",
        py::arg("fst"),
        py::arg("include_eps"),
        py::arg("symbols"));

  m.def("GetOutputSymbols",
        &GetOutputSymbols<StdArc, int32>,
        "GetOutputSymbols gets the list of symbols on the output of fst "
"(including epsilon, if include_eps == true)",
        py::arg("fst"),
        py::arg("include_eps"),
        py::arg("symbols"));

  m.def("ClearSymbols",
        &ClearSymbols<StdArc>,
        "ClearSymbols sets all the symbols on the input and/or "
      "output side of the FST to zero, as specified. "
      "It does not alter the symbol tables.",
        py::arg("clear_input"),
        py::arg("clear_output"),
        py::arg("fst"));

  m.def("GetSymbols",
        &GetSymbols<int32>,
        py::arg("symtab"),
        py::arg("include_eps"),
        py::arg("syms_out"));

  m.def("DeterminizeStarInLog",
        &DeterminizeStarInLog,
        py::arg("fst"),
        py::arg("delta") = kDelta,
        py::arg("debug_ptr") = NULL,
        py::arg("max_states") = -1);

  m.def("PushInLogInitial",
        &PushInLog<REWEIGHT_TO_INITIAL>,
        py::arg("fst"),
        py::arg("ptype"),
        py::arg("delta") = kDelta);

  m.def("PushInLogFinal",
        &PushInLog<REWEIGHT_TO_FINAL>,
        py::arg("fst"),
        py::arg("ptype"),
        py::arg("delta") = kDelta);

  m.def("MinimizeEncoded",
        &MinimizeEncoded<StdArc>,
        "Minimizes after encoding; applicable to all FSTs.  It is like what you get "
      "from the Minimize() function, except it will not push the weights, or the "
      "symbols.  This is better for our recipes, as we avoid ever pushing the "
      "weights.  However, it will only minimize optimally if your graphs are such "
      "that the symbols are as far to the left as they can go, and the weights "
      "in combinable paths are the same... hard to formalize this, but it's something "
      "that is satisified by our normal FSTs.",
        py::arg("fst"),
        py::arg("delta") = kDelta);

  m.def("GetLinearSymbolSequence",
        &GetLinearSymbolSequence<StdArc, int32>,
        "GetLinearSymbolSequence gets the symbol sequence from a linear FST. "
      "If the FST is not just a linear sequence, it returns false.   If it is "
      "a linear sequence (including the empty FST), it returns true.  In this "
      "case it outputs the symbol "
      "sequences as \"isymbols_out\" and \"osymbols_out\" (removing epsilons), and "
      "the total weight as \"tot_weight\". The total weight will be Weight::Zero() "
      "if the FST is empty.  If any of the output pointers are NULL, it does not "
      "create that output.",
        py::arg("fst"),
        py::arg("isymbols_out"),
        py::arg("osymbols_out"),
        py::arg("tot_weight_out"));

  m.def("ConvertNbestToVector",
        &ConvertNbestToVector<StdArc>,
        "This function converts an FST with a special structure, which is "
            "output by the OpenFst functions ShortestPath and RandGen, and converts "
            "them into a std::vector of separate FSTs.  This special structure is that "
            "the only state that has more than one (arcs-out or final-prob) is the "
            "start state.  fsts_out is resized to the appropriate size.",
        py::arg("fst"),
        py::arg("fsts_out"));

  m.def("NbestAsFsts",
        &NbestAsFsts<StdArc>,
        "Takes the n-shortest-paths (using ShortestPath), but outputs "
            "the result as a vector of up to n fsts.  This function will "
            "size the \"fsts_out\" vector to however many paths it got "
            "(which will not exceed n).  n must be >= 1.",
        py::arg("fst"),
        py::arg("n"),
        py::arg("fsts_out"));

  m.def("MakeLinearAcceptor",
        &MakeLinearAcceptor<StdArc, int32>,
        "Creates unweighted linear acceptor from symbol sequence.",
        py::arg("labels"),
        py::arg("ofst"));

  m.def("MakeLinearAcceptorWithAlternatives",
        &MakeLinearAcceptorWithAlternatives<StdArc, int32>,
        "Creates an unweighted acceptor with a linear structure, with alternatives "
      "at each position.  Epsilon is treated like a normal symbol here. "
      "Each position in \"labels\" must have at least one alternative.",
        py::arg("labels"),
        py::arg("ofst"));

  m.def("SafeDeterminizeWrapper",
        &SafeDeterminizeWrapper<StdArc>,
        "Does PreDeterminize and DeterminizeStar and then removes the disambiguation symbols. "
            "This is a form of determinization that will never blow up. "
            "Note that ifst is non-const and can be considered to be destroyed by this "
            "operation. "
            "Does not do epsilon removal (RemoveEpsLocal)-- this is so it's safe to cast to "
            "log and do this, and maintain equivalence in tropical.",
        py::arg("ifst"),
        py::arg("ofst"),
        py::arg("delta") = kDelta);

  m.def("SafeDeterminizeMinimizeWrapper",
        &SafeDeterminizeMinimizeWrapper<StdArc>,
        "SafeDeterminizeMinimizeWapper is as SafeDeterminizeWrapper except that it also "
            "minimizes (encoded minimization, which is safe).  This algorithm will destroy \"ifst\".",
        py::arg("ifst"),
        py::arg("ofst"),
        py::arg("delta") = kDelta);

  m.def("SafeDeterminizeMinimizeWrapperInLog",
        &SafeDeterminizeMinimizeWrapperInLog,
        "SafeDeterminizeMinimizeWapperInLog is as SafeDeterminizeMinimizeWrapper except "
"it first casts tothe log semiring.",
        py::arg("ifst"),
        py::arg("ofst"),
        py::arg("delta") = kDelta);

  m.def("RemoveSomeInputSymbols",
        &RemoveSomeInputSymbols<StdArc, int32>,
        "RemoveSomeInputSymbols removes any symbol that appears in \"to_remove\", from "
"the input side of the FST, replacing them with epsilon.",
        py::arg("to_remove"),
        py::arg("fst"));

  /*
  m.def("MapInputSymbols",
        py::overload_cast<const std::vector<int32> &,
                     MutableFst<StdArc> *>(&MapInputSymbols<StdArc, int32>),
        "MapInputSymbols will replace any input symbol i that is between 0 and "
      "symbol_map.size()-1, with symbol_map[i].  It removes the input symbol "
      "table of the FST.",
        py::arg("symbol_map"),
        py::arg("fst"));
        */

  m.def("RemoveWeights",
        &RemoveWeights<StdArc>,
        py::arg("fst"));

  m.def("PrecedingInputSymbolsAreSame",
        &PrecedingInputSymbolsAreSame<StdArc>,
        "Returns true if and only if the FST is such that the input symbols "
      "on arcs entering any given state all have the same value. "
      "if \"start_is_epsilon\", treat start-state as an epsilon input arc "
      "[i.e. ensure only epsilon can enter start-state].",
        py::arg("start_is_epsilon"),
        py::arg("fst"));

  m.def("PrecedingInputSymbolsAreSameClass",
        &PrecedingInputSymbolsAreSameClass<StdArc, IdentityFunction<typename StdArc::Label>>,
        "This is as PrecedingInputSymbolsAreSame, but with a functor f that maps labels to classes. "
            "The function tests whether the symbols preceding any given state are in the same "
            "class. "
            "Formally, f is of a type F that has an operator of type "
            "F::Result F::operator() (F::Arg a) const; "
            "where F::Result is an integer type and F::Arc can be constructed from Arc::Label. "
            "this must apply to valid labels and also to kNoLabel (so we can have a marker for "
            "the invalid labels.",
        py::arg("start_is_epsilon"),
        py::arg("fst"),
        py::arg("f"));

  m.def("FollowingInputSymbolsAreSame",
        &FollowingInputSymbolsAreSame<StdArc>,
        "Returns true if and only if the FST is such that the input symbols "
      "on arcs exiting any given state all have the same value. "
      "If end_is_epsilon, treat end-state as an epsilon output arc [i.e. ensure "
      "end-states cannot have non-epsilon output transitions.]",
        py::arg("end_is_epsilon"),
        py::arg("fst"));

  m.def("FollowingInputSymbolsAreSameClass",
        &FollowingInputSymbolsAreSameClass<StdArc, IdentityFunction<typename StdArc::Label>>,
        py::arg("end_is_epsilon"),
        py::arg("fst"),
        py::arg("f"));

  m.def("MakePrecedingInputSymbolsSame",
        &MakePrecedingInputSymbolsSame<StdArc>,
        "MakePrecedingInputSymbolsSame ensures that all arcs entering any given fst "
      "state have the same input symbol.  It does this by detecting states "
      "that have differing input symbols going in, and inserting, for each of "
      "the preceding arcs with non-epsilon input symbol, a new dummy state that "
      "has an epsilon link to the fst state. "
      "If \"start_is_epsilon\", ensure that start-state can have only epsilon-links "
      "into it.",
        py::arg("start_is_epsilon"),
        py::arg("fst"));

  m.def("MakePrecedingInputSymbolsSameClass",
        &MakePrecedingInputSymbolsSameClass<StdArc, IdentityFunction<typename StdArc::Label>>,
        "As MakePrecedingInputSymbolsSame, but takes a functor object that maps labels to classes.",
        py::arg("start_is_epsilon"),
        py::arg("fst"),
        py::arg("f"));

  m.def("MakeFollowingInputSymbolsSame",
        &MakeFollowingInputSymbolsSame<StdArc>,
        "MakeFollowingInputSymbolsSame ensures that all arcs exiting any given fst "
"state have the same input symbol.  It does this by detecting states that have "
"differing input symbols on arcs that exit it, and inserting, for each of the "
"following arcs with non-epsilon input symbol, a new dummy state that has an "
"input-epsilon link from the fst state.  The output symbol and weight stay on the "
"link to the dummy state (in order to keep the FST output-deterministic and "
"stochastic, if it already was). "
"If end_is_epsilon, treat \"being a final-state\" like having an epsilon output "
"link.",
        py::arg("end_is_epsilon"),
        py::arg("fst"));

  m.def("MakeFollowingInputSymbolsSameClass",
        &MakeFollowingInputSymbolsSameClass<StdArc, IdentityFunction<typename StdArc::Label>>,
        "As MakeFollowingInputSymbolsSame, but takes a functor object that maps labels to classes.",
        py::arg("end_is_epsilon"),
        py::arg("fst"),
        py::arg("f"));

  m.def("MakeLoopFst",
        &MakeLoopFst<StdArc>,
        "MakeLoopFst creates an FST that has a state that is both initial and "
"final (weight == Weight::One()), and for each non-NULL pointer fsts[i], "
"it has an arc out whose output-symbol is i and which goes to a "
"sub-graph whose input language is equivalent to fsts[i], where the "
"final-state becomes a transition to the loop-state.  Each fst in \"fsts\" "
"should be an acceptor.  The fst MakeLoopFst returns is output-deterministic, "
"but not output-epsilon free necessarily, and arcs are sorted on output label. "
"Note: if some of the pointers in the input vector \"fsts\" have the same "
"value, \"MakeLoopFst\" uses this to speed up the computation. "

"Formally: suppose I is the set of indexes i such that fsts[i] != NULL. "
"Let L[i] be the language that the acceptor fsts[i] accepts. "
"Let the language K be the set of input-output pairs i:l such "
"that i in I and l in L[i].  Then the FST returned by MakeLoopFst "
"accepts the language K*, where * is the Kleene closure (CLOSURE_STAR) "
"of K. "

"We could have implemented this via a combination of \"project\", "
"\"concat\", \"union\" and \"closure\".  But that FST would have been "
"less well optimized and would have a lot of final-states.",
        py::arg("fsts"));

  m.def("ApplyProbabilityScale",
        &ApplyProbabilityScale<StdArc>,
        "ApplyProbabilityScale is applicable to FSTs in the log or tropical semiring. "
      "It multiplies the arc and final weights by \"scale\" [this is not the Mul "
      "operation of the semiring, it's actual multiplication, which is equivalent "
      "to taking a power in the semiring].",
        py::arg("scale"),
        py::arg("fst"));

  m.def("EqualAlign",
        &EqualAlign<StdArc>,
        "EqualAlign is similar to RandGen, but it generates a sequence with exactly \"length\" "
      "input symbols.  It returns true on success, false on failure (failure is partly "
      "random but should never happen in practice for normal speech models.) "
      "It generates a random path through the input FST, finds out which subset of the "
      "states it visits along the way have self-loops with inupt symbols on them, and "
      "outputs a path with exactly enough self-loops to have the requested number "
      "of input symbols. "
      "Note that EqualAlign does not use the probabilities on the FST.  It just uses "
      "equal probabilities in the first stage of selection (since the output will anyway "
      "not be a truly random sample from the FST). "
      "The input fst \"ifst\" must be connected or this may enter an infinite loop.",
        py::arg("ifst"),
        py::arg("length"),
        py::arg("rand_seed"),
        py::arg("ofst"),
        py::arg("fsnum_retriest") = 10);

  m.def("RemoveUselessArcs",
        &RemoveUselessArcs<StdArc>,
        "RemoveUselessArcs removes arcs such that there is no input symbol "
      "sequence for which the best path through the FST would contain "
      "those arcs [for these purposes, epsilon is not treated as a real symbol]. "
      "This is mainly geared towards decoding-graph FSTs which may contain "
      "transitions that have less likely words on them that would never be "
      "taken.  We do not claim that this algorithm removes all such arcs; "
      "it just does the best job it can. "
      "Only works for tropical (not log) semiring as it uses "
      "NaturalLess.",
        py::arg("fst"));

  m.def("PhiCompose",
        &PhiCompose<StdArc>,
        "PhiCompose is a version of composition where "
"the right hand FST (fst2) is treated as a backoff "
"LM, with the phi symbol (e.g. #0) treated as a "
"\"failure transition\", only taken when we don't "
"have a match for the requested symbol.",
        py::arg("fst1"),
        py::arg("fst2"),
        py::arg("phi_label"),
        py::arg("fst"));

  m.def("PropagateFinal",
        &PropagateFinal<StdArc>,
        "PropagateFinal propagates final-probs through "
"\"phi\" transitions (note that here, phi_label may "
"be epsilon if you want).  If you have a backoff LM "
"with special symbols (\"phi\") on the backoff arcs "
"instead of epsilon, you may use PhiCompose to compose "
"with it, but this won't do the right thing w.r.t. "
"final probabilities.  You should first call PropagateFinal "
"on the FST with phi's i it (fst2 in PhiCompose above), "
"to fix this.  If a state does not have a final-prob, "
"but has a phi transition, it makes the state's final-prob "
"(phi-prob * final-prob-of-dest-state), and does this "
"recursively i.e. follows phi transitions on the dest state "
"first.  It behaves as if there were a super-final state "
"with a special symbol leading to it, from each currently "
"final state.  Note that this may not behave as desired "
"if there are epsilons in your FST; it might be better "
"to remove those before calling this function.",
        py::arg("phi_label"),
        py::arg("fst"));

  m.def("RhoCompose",
        &RhoCompose<StdArc>,
        "RhoCompose is a version of composition where "
      "the right hand FST (fst2) has speciall \"rho transitions\" "
      "which are taken whenever no normal transition matches; these "
      "transitions will be rewritten with whatever symbol was on "
      "the first FST.",
        py::arg("fst1"),
        py::arg("fst2"),
        py::arg("rho_label"),
        py::arg("fst"));

  m.def("IsStochasticFst",
        &IsStochasticFst<StdArc>,
        "This function returns true if, in the semiring of the FST, the sum (within "
      "the semiring) of all the arcs out of each state in the FST is one, to within "
      "delta.  After MakeStochasticFst, this should be true (for a connected FST). "
      "\n"
      "@param fst [in] the FST that we are testing. "
      "@param delta [in] the tolerance to within which we test equality to 1. "
      "@param min_sum [out] if non, NULL, contents will be set to the minimum sum of weights. "
      "@param max_sum [out] if non, NULL, contents will be set to the maximum sum of weights. "
      "@return Returns true if the FST is stochastic, and false otherwise.",
        py::arg("fst"),
        py::arg("delta") = kDelta,
        py::arg("min_sum") = NULL,
        py::arg("max_sum") = NULL);

  m.def("IsStochasticFstInLog",
        &IsStochasticFstInLog,
        "IsStochasticFstInLog makes sure it's stochastic after casting to log.",
        py::arg("fst"),
        py::arg("delta") = kDelta,
        py::arg("min_sum") = NULL,
        py::arg("max_sum") = NULL);

}
void pybind_fstext_grammar_context_fst(py::module &m) {
  py::enum_<NonterminalValues>(m, "NonterminalValues")
    .value("kNontermBos", NonterminalValues::kNontermBos)
    .value("kNontermBegin", NonterminalValues::kNontermBegin)
    .value("kNontermEnd", NonterminalValues::kNontermEnd)
    .value("kNontermReenter", NonterminalValues::kNontermReenter)
    .value("kNontermUserDefined", NonterminalValues::kNontermUserDefined)
    .value("kNontermMediumNumber", NonterminalValues::kNontermMediumNumber)
    .value("kNontermBigNumber", NonterminalValues::kNontermBigNumber)
    .export_values();

  m.def("GetEncodingMultiple",
        &GetEncodingMultiple,
        "Returns the smallest multiple of 1000 that is strictly greater than "
      "nonterm_phones_offset.  Used in the encoding of special symbol in HCLG; "
      "they are encoded as "
      " special_symbol = "
      "    kNontermBigNumber + (nonterminal * encoding_multiple) + phone_index",
        py::arg("nonterm_phones_offset"));

  m.def("ComposeContextLeftBiphone",
        &ComposeContextLeftBiphone,
        "This is a variant of the function ComposeContext() which is to be used "
            "with our \"grammar FST\" framework (see \ref graph_context, i.e. "
            "../doc/grammar.dox, for more details).  This does not take "
            "the 'context_width' and 'central_position' arguments because they are "
            "assumed to be 2 and 1 respectively (meaning, left-biphone phonetic context). "

            "This function creates a context FST and composes it on the left with \"ifst\" "
            "to make \"ofst\". "

            "@param [in] nonterm_phones_offset  The integer id of the symbol "
            "                  #nonterm_bos in the phones.txt file.  You can just set this "
            "                  to a large value (like 1 million) if you are not actually using "
            "                  nonterminals (e.g. for testing purposes). "
            "@param [in] disambig_syms  List of disambiguation symbols, e.g. the integer "
            "            ids of #0, #1, #2 ... in the phones.txt. "
            "@param [in,out] ifst   The FST we are composing with C (e.g. LG.fst). "
            "@param [out] ofst   Composed output FST (would be CLG.fst). "
            "@param [out] ilabels  Vector, indexed by ilabel of CLG.fst, providing information "
            "                  about the meaning of that ilabel; see \ref tree_ilabel "
            "                  (http://kaldi-asr.org/doc/tree_externals.html#tree_ilabel) "
            "                  and also \ref grammar_special_clg "
            "                  (http://kaldi-asr.org/doc/grammar#grammar_special_clg).",
        py::arg("nonterm_phones_offset"),
        py::arg("disambig_syms"),
        py::arg("ifst"),
        py::arg("ofst"),
        py::arg("ilabels"));
  {
    using PyClass = InverseLeftBiphoneContextFst;

    auto inverse_left_biphone_context_fst = py::class_<PyClass, DeterministicOnDemandFst<StdArc>>(
        m, "InverseLeftBiphoneContextFst");
    inverse_left_biphone_context_fst.def(py::init<PyClass::Label ,
                               const std::vector<int32>& ,
                               const std::vector<int32>& >(),
            py::arg("nonterm_phones_offset"),
            py::arg("phones"),
            py::arg("disambig_syms"))
      .def("Start", &PyClass::Start)
      .def("Final", &PyClass::Final,
        py::arg("s"))
      .def("GetArc", &PyClass::GetArc,
        py::arg("s"),
        py::arg("ilabel"),
        py::arg("arc"))
      .def("IlabelInfo", &PyClass::IlabelInfo,
      "Returns a reference to a vector<vector<int32> > with information about all "
  "the input symbols of C (i.e. all the output symbols of this "
  "InverseContextFst).  See "
  "\"http://kaldi-asr.org/doc/tree_externals.html#tree_ilabel\".")
      .def("SwapIlabelInfo", &PyClass::SwapIlabelInfo,
            "A way to destructively obtain the ilabel-info.  Only do this if you "
  "are just about to destroy this object.",
        py::arg("vec"));
  }
}
void pybind_fstext_lattice_utils(py::module &m) {

  m.def("ConvertLattice",
        py::overload_cast<const ExpandedFst<ArcTpl<TropicalWeight> > &,
    MutableFst<ArcTpl<CompactLatticeWeightTpl<TropicalWeight, int32> > > *,
    bool >(&ConvertLattice<TropicalWeight, int32>),
        "Convert from FST with arc-type Weight, to one with arc-type "
            "CompactLatticeWeight.  Uses FactorFst to identify chains "
            "of states which can be turned into a single output arc.",
        py::arg("ifst"),
        py::arg("ofst"),
        py::arg("invert") = true);

  m.def("ConvertLattice",
        py::overload_cast<const ExpandedFst<ArcTpl<LatticeWeightTpl<float>> > &,
    MutableFst<ArcTpl<CompactLatticeWeightTpl<LatticeWeightTpl<float>, int32> > > *,
    bool >(&ConvertLattice<LatticeWeightTpl<float>, int32>),
        "Convert from FST with arc-type Weight, to one with arc-type "
            "CompactLatticeWeight.  Uses FactorFst to identify chains "
            "of states which can be turned into a single output arc.",
        py::arg("ifst"),
        py::arg("ofst"),
        py::arg("invert") = true);

  m.def("ConvertLattice",
        py::overload_cast<const ExpandedFst<ArcTpl<CompactLatticeWeightTpl<TropicalWeight, int32> > > &,
    MutableFst<ArcTpl<TropicalWeight> > *,
    bool >(&ConvertLattice<TropicalWeight, int32>),
        "Convert lattice CompactLattice  format to Lattice.  This is a bit "
      "like converting from the Gallic semiring.  As for any CompactLattice, \"ifst\" "
      "must be an acceptor (i.e., ilabels and olabels should be identical).  If "
      "invert=false, the labels on \"ifst\" become the ilabels on \"ofst\" and the "
      "strings in the weights of \"ifst\" becomes the olabels.  If invert=true "
      "[default], this is reversed (useful for speech recognition lattices; our "
      "standard non-compact format has the words on the output side to match HCLG).",
        py::arg("ifst"),
        py::arg("ofst"),
        py::arg("invert") = true);

  m.def("ConvertLattice",
        py::overload_cast<const ExpandedFst<ArcTpl<CompactLatticeWeightTpl<LatticeWeightTpl<float>, int32> > > &,
    MutableFst<ArcTpl<LatticeWeightTpl<float>> > *,
    bool >(&ConvertLattice<LatticeWeightTpl<float>, int32>),
        "Convert lattice CompactLattice  format to Lattice.  This is a bit "
      "like converting from the Gallic semiring.  As for any CompactLattice, \"ifst\" "
      "must be an acceptor (i.e., ilabels and olabels should be identical).  If "
      "invert=false, the labels on \"ifst\" become the ilabels on \"ofst\" and the "
      "strings in the weights of \"ifst\" becomes the olabels.  If invert=true "
      "[default], this is reversed (useful for speech recognition lattices; our "
      "standard non-compact format has the words on the output side to match HCLG).",
        py::arg("ifst"),
        py::arg("ofst"),
        py::arg("invert") = true);

  m.def("ConvertLattice",
        py::overload_cast<const ExpandedFst<ArcTpl<LatticeWeightTpl<float>> > &,
    MutableFst<ArcTpl<TropicalWeight> > *>(&ConvertLattice<LatticeWeightTpl<float>, TropicalWeight>),
        "Convert between CompactLattices and Lattices of different floating point types... "
            "this works between any pair of weight types for which ConvertLatticeWeight "
            "is defined (c.f. lattice-weight.h), and also includes conversion from "
            "LatticeWeight to TropicalWeight.",
        py::arg("ifst"),
        py::arg("ofst"));
      /*
  m.def("ConvertLattice",
        py::overload_cast<const ExpandedFst<ArcTpl<TropicalWeight> > &,
    MutableFst<ArcTpl<LatticeWeightTpl<float>> > *>(&ConvertLattice<TropicalWeight, LatticeWeightTpl<float>>),
        "Convert between CompactLattices and Lattices of different floating point types... "
            "this works between any pair of weight types for which ConvertLatticeWeight "
            "is defined (c.f. lattice-weight.h), and also includes conversion from "
            "LatticeWeight to TropicalWeight.",
        py::arg("ifst"),
        py::arg("ofst"));
  m.def("ConvertLattice",
        py::overload_cast<const ExpandedFst<ArcTpl<LatticeWeightTpl<float> > > &,
                    MutableFst<ArcTpl<CompactLatticeWeightTpl<LatticeWeightTpl<double>, int32> > > *>(&ConvertLattice<int32>),
        "Lattice with float to CompactLattice with double.",
        py::arg("ifst"),
        py::arg("ofst"));

      */
  m.def("ConvertLattice",
        py::overload_cast<const ExpandedFst<ArcTpl<LatticeWeightTpl<double> > > &,
                    MutableFst<ArcTpl<CompactLatticeWeightTpl<LatticeWeightTpl<float>, int32> > > *>(&ConvertLattice<int32>),
        "Lattice with double to CompactLattice with float.",
        py::arg("ifst"),
        py::arg("ofst"));

  m.def("ConvertLattice",
        py::overload_cast<const ExpandedFst<ArcTpl<CompactLatticeWeightTpl<LatticeWeightTpl<double>, int32> > > &,
                    MutableFst<ArcTpl<LatticeWeightTpl<float> > > *>(&ConvertLattice<int32>),
        "Converts CompactLattice with double to Lattice with float.",
        py::arg("ifst"),
        py::arg("ofst"));

  m.def("ConvertLattice",
        py::overload_cast<const ExpandedFst<ArcTpl<CompactLatticeWeightTpl<LatticeWeightTpl<float>, int32> > > &,
                    MutableFst<ArcTpl<LatticeWeightTpl<double> > > *>(&ConvertLattice<int32>),
        "Converts CompactLattice with float to Lattice with double.",
        py::arg("ifst"),
        py::arg("ofst"));

  m.def("ConvertFstToLattice",
        py::overload_cast<const ExpandedFst<ArcTpl<TropicalWeight> > &,
    MutableFst<ArcTpl<LatticeWeightTpl<float> > > *>(&ConvertFstToLattice<float>),
        "Converts TropicalWeight to LatticeWeight (puts all the weight on "
      "the first float in the lattice's pair).",
        py::arg("ifst"),
        py::arg("ofst"));

  m.def("DefaultLatticeScale",
        &DefaultLatticeScale,
        "Returns a default 2x2 matrix scaling factor for LatticeWeight");

  m.def("AcousticLatticeScale",
        &AcousticLatticeScale,
        py::arg("acwt"));

  m.def("GraphLatticeScale",
        &GraphLatticeScale,
        py::arg("lmwt"));

  m.def("LatticeScale",
        &LatticeScale,
        py::arg("lmwt"),
        py::arg("acwt"));

      /*
  m.def("ScaleLattice",
        py::overload_cast<const std::vector<std::vector<float> > &,
    MutableFst<ArcTpl<LatticeWeightTpl<float>> > *>(&ScaleLattice<LatticeWeightTpl<float>, float>),
        "Scales the pairs of weights in LatticeWeight or CompactLatticeWeight by "
            "viewing the pair (a, b) as a 2-vector and pre-multiplying by the 2x2 matrix "
            "in \"scale\".  E.g. typically scale would equal "
            "[ 1   0; "
            "      0  acwt ] "
            "if we want to scale the acoustics by \"acwt\".",
        py::arg("scale"),
        py::arg("fst"));
*/
  m.def("RemoveAlignmentsFromCompactLattice",
        &RemoveAlignmentsFromCompactLattice<TropicalWeight, int32>,
        "Removes state-level alignments (the strings that are "
            "part of the weights).",
        py::arg("fst"));

  m.def("RemoveAlignmentsFromCompactLattice",
        &RemoveAlignmentsFromCompactLattice<LatticeWeightTpl<float>, int32>,
        "Removes state-level alignments (the strings that are "
            "part of the weights).",
        py::arg("fst"));

  m.def("CompactLatticeHasAlignment",
        &CompactLatticeHasAlignment<TropicalWeight, int32>,
        "Returns true if lattice has alignments, i.e. it has "
      "any nonempty strings inside its weights.",
        py::arg("fst"));

  m.def("CompactLatticeHasAlignment",
        &CompactLatticeHasAlignment<LatticeWeightTpl<float>, int32>,
        "Returns true if lattice has alignments, i.e. it has "
      "any nonempty strings inside its weights.",
        py::arg("fst"));
  {
    using PyClass = StdToLatticeMapper<float>;

    auto std_to_lattice_mapper = py::class_<PyClass>(
        m, "StdToLatticeMapper");
    std_to_lattice_mapper.def(py::init<>())
      .def("FinalAction",
            &PyClass::FinalAction)
      .def("InputSymbolsAction",
            &PyClass::InputSymbolsAction)
      .def("OutputSymbolsAction",
            &PyClass::OutputSymbolsAction)
      .def("Properties",
            &PyClass::Properties,
        py::arg("props"));
  }
  {
    using PyClass = LatticeToStdMapper<float>;

    auto lattice_to_std_mapper = py::class_<PyClass>(
        m, "LatticeToStdMapper");
    lattice_to_std_mapper.def(py::init<>())
      .def("FinalAction",
            &PyClass::FinalAction)
      .def("InputSymbolsAction",
            &PyClass::InputSymbolsAction)
      .def("OutputSymbolsAction",
            &PyClass::OutputSymbolsAction)
      .def("Properties",
            &PyClass::Properties,
        py::arg("props"));
  }
      /*
  m.def("PruneCompactLattice",
        &PruneCompactLattice<TropicalWeight, int32>,
        py::arg("beam"),
        py::arg("fst"));

  m.def("PruneCompactLattice",
        &PruneCompactLattice<LatticeWeightTpl<float>, int32>,
        py::arg("beam"),
        py::arg("fst"));
*/
}
void pybind_fstext_pre_determinize(py::module &m) {

  m.def("PreDeterminize",
        &PreDeterminize<StdArc, int32>,
        "PreDeterminize inserts extra symbols on the input side of an FST as necessary to "
   "ensure that, after epsilon removal, it will be compactly determinizable by the "
   "determinize* algorithm.  By compactly determinizable we mean that "
   "no original FST state is represented in more than one determinized state). "

   "Caution: this code is now only used in testing. "

   "The new symbols start from the value \"first_new_symbol\", which should be "
   "higher than the largest-numbered symbol currently in the FST.  The new "
   "symbols added are put in the array syms_out, which should be empty at start.",
        py::arg("fst"),
        py::arg("first_new_symbol"),
        py::arg("syms_out"));

  m.def("CreateNewSymbols",
        &CreateNewSymbols<int32>,
        "CreateNewSymbols is a helper function used inside PreDeterminize, and is also useful "
   "when you need to add a number of extra symbols to a different vocabulary from the one "
   "modified by PreDeterminize.",
        py::arg("inputSymTable"),
        py::arg("nSym"),
        py::arg("prefix"),
        py::arg("syms_out"));

  m.def("AddSelfLoops",
        &AddSelfLoops<StdArc>,
        "AddSelfLoops is a function you will probably want to use alongside PreDeterminize, "
    "to add self-loops to any FSTs that you compose on the left hand side of the one "
    "modified by PreDeterminize. "

    "This function inserts loops with \"special symbols\" [e.g. #0, #1] into an FST. "
    "This is done at each final state and each state with non-epsilon output symbols on "
    "at least one arc out of it.  This is to ensure that these symbols, when inserted into "
    "the input side of an FST we will compose with on the right, can \"pass through\" this "
    "FST. "

    "At input, isyms and osyms must be vectors of the same size n, corresponding "
    "to symbols that currently do not exist in 'fst'.  For each state in n that has "
    "non-epsilon symbols on the output side of arcs leaving it, or which is a final state, "
    "this function inserts n self-loops with unit weight and one of the n pairs "
    "of symbols on its input and output.",
        py::arg("fst"),
        py::arg("isyms"),
        py::arg("osyms"));

  m.def("DeleteISymbols",
        &DeleteISymbols<StdArc>,
        "DeleteSymbols replaces any instances of symbols in the vector symsIn, appearing "
   "on the input side, with epsilon. "
      "It returns the number of instances of symbols deleted.",
        py::arg("fst"),
        py::arg("symsIn"));

  m.def("CreateSuperFinal",
        &CreateSuperFinal<StdArc>,
        "CreateSuperFinal takes an FST, and creates an equivalent FST with a single final "
   "state with no transitions out and unit final weight, by inserting epsilon transitions "
   "as necessary.",
        py::arg("fst"));
}
void pybind_fstext_prune_special(py::module &m) {

  m.def("PruneSpecial",
        &PruneSpecial<StdArc>,
        "The function PruneSpecial is like the standard OpenFst function \"prune\", "
   "except it does not expand the entire \"ifst\"- this is useful for cases where "
   "ifst is an on-demand FST such as a ComposeFst and we don't want to visit "
   "it all.  It supports pruning either to a specified beam (if beam is "
   "not One()), or to a specified max_states (if max_states is > 0).  One of the "
   "two must be specified. "

   "Requirements: "
   "  - Costs must be non-negative (equivalently, weights must not be greater than One()). "
   "  - There must be a Compare(a, b) function that compares two weights and returns (-1,0,1) "
   "    if (a<b, a=b, a>b).  We define this in Kaldi, for TropicalWeight, LogWeight (I think), "
   "    and LatticeWeight... also CompactLatticeWeight, but we doubt that will be used here; "
   "    better to use PruneCompactLattice().",
        py::arg("ifst"),
        py::arg("ofst"),
        py::arg("beam"),
        py::arg("max_states") =  0);

  m.def("PushSpecial",
        &PushSpecial,
        "This function does weight-pushing, in the log semiring, "
  "but in a special way, such that any \"leftover weight\" after pushing "
  "gets distributed evenly along the FST, and doesn't end up either "
  "at the start or at the end.  Basically it pushes the weights such "
  "that the total weight of each state (i.e. the sum of the arc "
  "probabilities plus the final-prob) is the same for all states.",
        py::arg("fst"),
        py::arg("delta") = kDelta);
}
void pybind_fstext_rand_fst(py::module &m) {
  py::class_<RandFstOptions>(m, "RandFstOptions")
      .def(py::init<>())
      .def_readwrite("n_syms", &RandFstOptions::n_syms)
      .def_readwrite("n_states", &RandFstOptions::n_states)
      .def_readwrite("n_arcs", &RandFstOptions::n_arcs)
      .def_readwrite("n_final", &RandFstOptions::n_final)
      .def_readwrite("allow_empty", &RandFstOptions::allow_empty)
      .def_readwrite("acyclic", &RandFstOptions::acyclic)
      .def_readwrite("weight_multiplier", &RandFstOptions::weight_multiplier);

  m.def("RandFst",
        &RandFst<StdArc>,
        "Returns a random FST.  Useful for randomized algorithm testing. "
"Only works if weight can be constructed from float.",
        py::arg("opts") = RandFstOptions());
        /*
  m.def("RandPairFst",
        &RandPairFst<StdArc>,
        "Returns a random FST.  Useful for randomized algorithm testing. "
      "Only works if weight can be constructed from a pair of floats",
        py::arg("opts") = RandFstOptions());
        */
}
void pybind_fstext_remove_eps_local(py::module &m) {
  m.def("RemoveEpsLocal",
        &RemoveEpsLocal<StdArc>,
        "RemoveEpsLocal remove some (but not necessarily all) epsilons in an FST, "
      "using an algorithm that is guaranteed to never increase the number of arcs "
      "in the FST (and will also never increase the number of states).  The "
      "algorithm is not optimal but is reasonably clever.  It does not just remove "
      "epsilon arcs;it also combines pairs of input-epsilon and output-epsilon arcs "
      "into one. "
      "The algorithm preserves equivalence and stochasticity in the given semiring. "
      "If you want to preserve stochasticity in a different semiring (e.g. log), "
      "then use RemoveEpsLocalSpecial, which only works for StdArc but which "
      "preserves stochasticity, where possible (*) in the LogArc sense.  The reason that we can't "
      "just cast to a different semiring is that in that case we would no longer "
      "be able to guarantee equivalence in the original semiring (this arises from "
      "what happens when we combine identical arcs). "
      "(*) by \"where possible\".. there are situations where we wouldn't be able to "
      "preserve stochasticity in the LogArc sense while maintaining equivalence in "
      "the StdArc sense, so in these situations we maintain equivalence.",
        py::arg("fst"));

  m.def("RemoveEpsLocalSpecial",
        &RemoveEpsLocalSpecial,
        "As RemoveEpsLocal but takes care to preserve stochasticity "
"when cast to LogArc.",
        py::arg("fst"));
}
void pybind_fst_types(py::module &m) {

  {
    using PyClass = VectorFst<StdArc>;

    auto vector_fst = py::class_<PyClass>(
        m, "VectorFst");
    vector_fst.def(py::init<>())
      .def("Start", &PyClass::Start)
      .def("Final", &PyClass::Final,
            py::arg("s"))
      .def("write_to_string", [](const PyClass& f){
             std::ostringstream os;
             fst::FstWriteOptions opts;
             opts.stream_write = true;
             f.Write(os, opts);
            return py::bytes(os.str());
      })
      .def_static("from_string", [](const std::string &bytes){
            std::istringstream str(bytes);

            fst::FstHeader hdr;
            if (!hdr.Read(str, "<unspecified>"))
            KALDI_ERR << "Reading FST: error reading FST header from "
                        << kaldi::PrintableRxfilename("<unspecified>");
              FstReadOptions ropts("<unspecified>", &hdr);
            VectorFst<StdArc> *f = VectorFst<StdArc>::Read(str, ropts);
            return f;
      },
            py::arg("bytes"),
           py::return_value_policy::reference);
  }
}


void init_fstext(py::module &_m) {
  py::module m = _m.def_submodule("fstext", "fstext pybind for Kaldi");
    pybind_kaldi_fst_io(m);

  pybind_lattice_weight_impl<float>(m, "LatticeWeight",
                                    "Contain two values: value1 is the lm cost "
                                    "and value2 is the acoustic cost.");
  pybind_compact_lattice_weight_impl<fst::LatticeWeightTpl<float>, int>(
      m, "CompactLatticeWeight",
      "Contain two members: fst::LatticeWeight and std::vector<int>");

  py::class_<TableMatcherOptions>(m, "TableMatcherOptions")
      .def(py::init<>())
      .def_readwrite("table_ratio", &TableMatcherOptions::table_ratio)
      .def_readwrite("min_table_size", &TableMatcherOptions::min_table_size);

    pybind_table_matcher<StdVectorFst>(m);
    pybind_table_compose<StdArc>(m);
    pybind_fst_types(m);
    pybind_fstext_deterministic_fst(m);
    pybind_fstext_deterministic_lattice(m);
    pybind_fstext_context_fst(m);
    pybind_fstext_determinize_star(m);
    pybind_fstext_epsilon_property(m);
    pybind_fstext_factor(m);
    pybind_fstext_fstext_utils(m);
    pybind_fstext_grammar_context_fst(m);
    pybind_fstext_lattice_utils(m);
    pybind_fstext_pre_determinize(m);
    pybind_fstext_prune_special(m);
    pybind_fstext_rand_fst(m);
    pybind_fstext_remove_eps_local(m);

  pybind_table_writer<fst::VectorFstHolder>(m, "VectorFstWriter");
}
