
#include "fstext/pybind_fstext.h"
#include "fst/fst.h"
#include "fstext/context-fst.h"
#include "fstext/deterministic-fst.h"
#include "fstext/determinize-star.h"
#include "fstext/epsilon-property.h"
#include "fstext/factor.h"
#include "fstext/fstext-utils.h"
#include "fstext/grammar-context-fst.h"
#include "fstext/kaldi-fst-io.h"
#include "fstext/lattice-utils.h"
#include "fstext/lattice-weight.h"
#include "fstext/pre-determinize.h"
#include "fstext/prune-special.h"
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

}

void pybind_fstext_determinize_star(py::module &m) {
}
void pybind_fstext_epsilon_property(py::module &m) {
}
void pybind_fstext_factor(py::module &m) {
}
void pybind_fstext_fstext_lib(py::module &m) {
}
void pybind_fstext_fstext_utils(py::module &m) {
}
void pybind_fstext_grammar_context_fst(py::module &m) {
}
void pybind_fstext_lattice_utils(py::module &m) {
}
void pybind_fstext_lattice_weight(py::module &m) {
}
void pybind_fstext_pre_determinize(py::module &m) {
}
void pybind_fstext_prune_special(py::module &m) {
}
void pybind_fstext_rand_fst(py::module &m) {
}
void pybind_fstext_remove_eps_local(py::module &m) {
}
void pybind_fstext_trivial_factor_weight(py::module &m) {
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
    pybind_table_matcher<StdVectorFst>(m);
    pybind_table_compose<StdArc>(m);
}
