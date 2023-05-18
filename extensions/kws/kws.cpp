
#include "kws/pybind_kws.h"
#include "base/kaldi-types.h"
#include "kws/kaldi-kws.h"
#include "kws/kws-functions.h"
#include "kws/kws-scoring.h"

using namespace kaldi;

void pybind_kws_functions(py::module &m) {

  {
    using PyClass = Interval;

    auto interval = py::class_<PyClass>(
        m, "Interval");
    interval.def(py::init<>())
      .def(py::init<int32, int32>(),
        py::arg("start"),
        py::arg("end"))
      .def(py::init<const Interval &>(),
        py::arg("interval"))
      .def("Overlap",
        &PyClass::Overlap,
        py::arg("interval"))
      .def("Start",
        &PyClass::Start)
      .def("End",
        &PyClass::End);
  }
  m.def("CompareInterval",
        &CompareInterval,
        "We define a function bool CompareInterval(const Interval &i1, const Interval "
        "&i2) to compare the Interval defined above. If interval i1 is in front of "
        "interval i2, then return true; otherwise return false.",
        py::arg("i1"),
        py::arg("i2"));
  m.def("ClusterLattice",
        &ClusterLattice,
        "This function clusters the arcs with same word id and overlapping time-spans. "
        "Examples of clusters: "
        "0 1 a a (0.1s ~ 0.5s) and 2 3 a a (0.2s ~ 0.4s) are within the same cluster; "
        "0 1 a a (0.1s ~ 0.5s) and 5 6 b b (0.2s ~ 0.4s) are in different clusters; "
        "0 1 a a (0.1s ~ 0.5s) and 7 8 a a (0.9s ~ 1.4s) are also in different clusters. "
        "It puts disambiguating symbols in the olabels, leaving the words on the "
        "ilabels.",
        py::arg("clat"),
        py::arg("state_times"));
  m.def("CreateFactorTransducer",
        &CreateFactorTransducer,
        "This function contains two steps: weight pushing and factor generation. The "
        "original ShortestDistance() is not very efficient, so we do the weight "
        "pushing and shortest path manually by computing the alphas and betas. The "
        "factor generation step expand the lattice to the LXTXT' semiring, with "
        "additional start state and end state (and corresponding arcs) added.",
        py::arg("clat"),
        py::arg("state_times"),
        py::arg("utterance_id"),
        py::arg("factor_transducer"));
  m.def("RemoveLongSilences",
        &RemoveLongSilences,
        "This function removes the arcs with long silence. By \"long\" we mean arcs with "
        "#frames exceeding the given max_silence_frames. We do this filtering because "
        "the gap between adjacent words in a keyword must be <= 0.5 second. "
        "Note that we should not remove the arcs created in the factor generation "
        "step, so the \"search area\" is limited to the original arcs before factor "
        "generation.",
        py::arg("max_silence_frames"),
        py::arg("state_times"),
        py::arg("factor_transducer"));
  m.def("DoFactorMerging",
        &DoFactorMerging,
        "Do the factor merging part: encode input and output, and apply weighted "
        "epsilon removal, determinization and minimization.  Modifies factor_transducer.",
        py::arg("factor_transducer"),
        py::arg("index_transducer"));
  m.def("DoFactorDisambiguation",
        &DoFactorDisambiguation,
        "Do the factor disambiguation step: remove the cluster id's for the non-final "
        "arcs and insert disambiguation symbols for the final arcs",
        py::arg("index_transducer"));
  m.def("OptimizeFactorTransducer",
        &OptimizeFactorTransducer,
        "Do the optimization: do encoded determinization, minimization",
        py::arg("index_transducer"),
        py::arg("max_states"),
        py::arg("allow_partial"));
  m.def("MaybeDoSanityCheck",
        py::overload_cast<const KwsProductFst &>(&MaybeDoSanityCheck),
        "the following two functions will, if GetVerboseLevel() >= 2, check that the "
        "cost of the second-best path in the transducers is not negative, and print "
        "out some associated debugging info if GetVerboseLevel() >= 3.  The best path "
        "in the transducers will typically be for the empty word sequence, and it may "
        "have negative cost (i.e. probability more than one), but the second-best one "
        "should not have negative cost.  A warning will be printed if "
        "GetVerboseLevel() >= 2 and a substantially negative cost is found.",
        py::arg("factor_transducer"));
  m.def("MaybeDoSanityCheck",
        py::overload_cast<const KwsLexicographicFst &>(&MaybeDoSanityCheck),
        "the following two functions will, if GetVerboseLevel() >= 2, check that the "
        "cost of the second-best path in the transducers is not negative, and print "
        "out some associated debugging info if GetVerboseLevel() >= 3.  The best path "
        "in the transducers will typically be for the empty word sequence, and it may "
        "have negative cost (i.e. probability more than one), but the second-best one "
        "should not have negative cost.  A warning will be printed if "
        "GetVerboseLevel() >= 2 and a substantially negative cost is found.",
        py::arg("index_transducer"));
  {
    using PyClass = KwsProductFstToKwsLexicographicFstMapper;

    auto kws_product_fst_to_kws_lexicographic_fst_mapper = py::class_<PyClass>(
        m, "KwsProductFstToKwsLexicographicFstMapper",
        "this Mapper class is used in some of the the internals; we have to declare it "
        "in the header because, for the sake of compilation time, we split up the "
        "implementation into two .cc files.");

    kws_product_fst_to_kws_lexicographic_fst_mapper.def(py::init<>())
        .def("FinalAction", &PyClass::FinalAction)
        .def("InputSymbolsAction", &PyClass::InputSymbolsAction)
        .def("OutputSymbolsAction", &PyClass::OutputSymbolsAction)
        .def("Properties", &PyClass::Properties,
          py::arg("props"));
  }
}

void pybind_kws_scoring(py::module &m) {

  {
    using PyClass = KwsTerm;

    auto kws_term = py::class_<PyClass>(
        m, "KwsTerm");

    kws_term.def(py::init<>())
      .def(py::init<const std::string &, const std::vector<double> &>(),
          py::arg("kw_id"), py::arg("vec"))
        .def("valid", &PyClass::valid)
        .def("utt_id", &PyClass::utt_id)
        .def("set_utt_id", &PyClass::set_utt_id,
          py::arg("utt_id"))
        .def("kw_id", &PyClass::kw_id)
        .def("set_kw_id", &PyClass::set_kw_id,
          py::arg("kw_id"))
        .def("start_time", &PyClass::start_time)
        .def("set_start_time", &PyClass::set_start_time,
          py::arg("start_time"))
        .def("end_time", &PyClass::end_time)
        .def("set_end_time", &PyClass::set_end_time,
          py::arg("end_time"))
        .def("score", &PyClass::score)
        .def("set_score", &PyClass::set_score,
          py::arg("score"));
  }
  py::enum_<DetectionDecision>(m, "DetectionDecision")
    .value("kKwsFalseAlarm", DetectionDecision::kKwsFalseAlarm)
    .value("kKwsMiss", DetectionDecision::kKwsMiss)
    .value("kKwsCorr", DetectionDecision::kKwsCorr)
    .value("kKwsCorrUndetected", DetectionDecision::kKwsCorrUndetected)
    .value("kKwsUnseen", DetectionDecision::kKwsUnseen)
    .export_values();
  {
    using PyClass = AlignedTermsPair;

    auto aligned_terms_pair = py::class_<PyClass>(
        m, "AlignedTermsPair");
    aligned_terms_pair.def(py::init<>())
      .def_readwrite("ref", &PyClass::ref)
      .def_readwrite("hyp", &PyClass::hyp)
      .def_readwrite("aligner_score", &PyClass::aligner_score);
  }
  {
    using PyClass = KwsAlignment;

    auto kws_alignment = py::class_<PyClass>(
        m, "KwsAlignment");
    kws_alignment.def(py::init<>())
        .def("WriteCsv", &PyClass::WriteCsv,
          py::arg("os"),
          py::arg("frames_per_sec"))
        .def("begin", &PyClass::begin)
        .def("end", &PyClass::end)
        .def("size", &PyClass::size);
  }
  {
    using PyClass = KwsTermsAlignerOptions;

    auto kws_terms_aligner_options = py::class_<PyClass>(
        m, "KwsTermsAlignerOptions");
    kws_terms_aligner_options.def(py::init<>())
      .def_readwrite("max_distance", &PyClass::max_distance,
                      "Maximum distance (in frames) of the centers of "
                     "the ref and and the hyp to be considered as a potential "
                     "match during alignment process "
                     "Default: 50 frames (usually 0.5 seconds)");
  }
  {
    using PyClass = KwsTermsAligner;

    auto kws_terms_aligner = py::class_<PyClass>(
        m, "KwsTermsAligner");
    kws_terms_aligner.def(py::init<const KwsTermsAlignerOptions &>(),
          py::arg("opts"))
        .def("AddRef", &PyClass::AddRef,
          py::arg("ref"))
        .def("AddHyp", &PyClass::AddRef,
          py::arg("hyp"))
        .def("nof_hyps", &PyClass::nof_hyps)
        .def("nof_refs", &PyClass::nof_refs)
        .def("AlignTerms", &PyClass::AlignTerms)
        .def("AlignerScore", &PyClass::AlignerScore,
          py::arg("ref"),
          py::arg("hyp"));
  }
  {
    using PyClass = TwvMetricsOptions;

    auto twv_metrics_options = py::class_<PyClass>(
        m, "TwvMetricsOptions");
    twv_metrics_options.def(py::init<>())
        .def("beta", &PyClass::beta);
  }
  {
    using PyClass = TwvMetrics;

    auto twv_metrics = py::class_<PyClass>(
        m, "TwvMetrics");
    twv_metrics.def(py::init<const TwvMetricsOptions &>(),
          py::arg("opts"))
        .def("AddAlignment", &PyClass::AddAlignment,
          py::arg("ali"))
        .def("Reset", &PyClass::Reset)
        .def("Atwv", &PyClass::Atwv)
        .def("Stwv", &PyClass::Stwv)
        .def("GetOracleMeasures", &PyClass::GetOracleMeasures,
          py::arg("final_mtwv"),
          py::arg("final_mtwv_threshold"),
          py::arg("final_otwv"));
  }
}

void init_kws(py::module &_m) {
  py::module m = _m.def_submodule("kws", "kws pybind for Kaldi");

  pybind_kws_functions(m);
  pybind_kws_scoring(m);
}
