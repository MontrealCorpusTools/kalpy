
#include "lm/pybind_lm.h"
#include "lm/arpa-file-parser.h"
#include "lm/arpa-lm-compiler.h"
#include "lm/const-arpa-lm.h"
#include "lm/kaldi-rnnlm.h"
#include "lm/mikolov-rnnlm-lib.h"

using namespace kaldi;
using namespace fst;



class PyArpaFileParser : public ArpaFileParser {
public:
    //Inherit the constructors
    using ArpaFileParser::ArpaFileParser;
protected:

    void ConsumeNGram(const NGram& ngram) override {
        PYBIND11_OVERRIDE_PURE(
            void, //Return type (ret_type)
            ArpaFileParser,      //Parent class (cname)
            ConsumeNGram          //Name of function in C++ (must match Python name) (fn)
            ngram      //Argument(s) (...)
        );
    }
};

void pybind_lm_arpa_file_parser(py::module &m) {

  {
    using PyClass = ArpaParseOptions;

    auto arpa_parse_options = py::class_<PyClass>(
        m, "ArpaParseOptions");

  py::enum_<ArpaParseOptions::OovHandling>(arpa_parse_options, "OovHandling")
    .value("kRaiseError", ArpaParseOptions::OovHandling::kRaiseError)
    .value("kAddToSymbols", ArpaParseOptions::OovHandling::kAddToSymbols)
    .value("kReplaceWithUnk", ArpaParseOptions::OovHandling::kReplaceWithUnk)
    .value("kSkipNGram", ArpaParseOptions::OovHandling::kSkipNGram)
    .export_values();
  arpa_parse_options.def(py::init<>())
      .def_readwrite("bos_symbol", &PyClass::bos_symbol)
      .def_readwrite("eos_symbol", &PyClass::eos_symbol)
      .def_readwrite("unk_symbol", &PyClass::unk_symbol)
      .def_readwrite("oov_handling", &PyClass::oov_handling)
      .def_readwrite("max_warnings", &PyClass::max_warnings);
  }
  {
    using PyClass = NGram;

    auto ngram = py::class_<PyClass>(
        m, "NGram");
    ngram.def(py::init<>())
      .def_readwrite("words", &PyClass::words)
      .def_readwrite("logprob", &PyClass::logprob)
      .def_readwrite("backoff", &PyClass::backoff);
  }
  {
    using PyClass = ArpaFileParser;

    auto arpa_file_parser = py::class_<ArpaFileParser, PyArpaFileParser>(
        m, "ArpaFileParser",
        "ArpaFileParser is an abstract base class for ARPA LM file conversion. "
        "See ConstArpaLmBuilder and ArpaLmCompiler for usage examples");
    arpa_file_parser.def(py::init<const ArpaParseOptions&, fst::SymbolTable* >(),
                        py::arg("options"),
                        py::arg("symbols"))
      .def("Read", &PyClass::Read, py::arg("is"),
      py::call_guard<py::gil_scoped_release>())
      .def("Options", &PyClass::Options);
  }
}

void pybind_lm_arpa_lm_compiler(py::module &m) {

  {
    using PyClass = ArpaLmCompiler;

    auto arpa_lm_parser = py::class_<ArpaLmCompiler, ArpaFileParser>(
        m, "ArpaLmCompiler");
    arpa_lm_parser.def(py::init<const ArpaParseOptions&, int,
                 fst::SymbolTable*>(),
                        py::arg("options"),
                        py::arg("sub_eps"),
                        py::arg("symbols"))
      .def("Fst", &PyClass::Fst)
      .def("MutableFst", &PyClass::MutableFst);
  }
}

void pybind_lm_const_arpa_lm(py::module &m) {
  {
    using PyClass = ConstArpaLm;

    auto const_arpa_lm = py::class_<PyClass>(
        m, "ConstArpaLm");
    const_arpa_lm.def(py::init<>())
      .def("Read",
        &PyClass::Read,
        "Reads the ConstArpaLm format language model. It calls ReadInternal() or "
        "ReadInternalOldFormat() to do the actual reading.",
        py::arg("is"), py::arg("binary"),
      py::call_guard<py::gil_scoped_release>())
      .def_static("read_from_file",
        [](const std::string &filename){
          static ConstArpaLm const_arpa;
        ReadKaldiObject(filename, &const_arpa);
        return const_arpa;
        },
        "Reads the ConstArpaLm format language model. It calls ReadInternal() or "
        "ReadInternalOldFormat() to do the actual reading.",
        py::arg("filename"), py::return_value_policy::take_ownership,
      py::call_guard<py::gil_scoped_release>())
      .def("Write",
        &PyClass::Write,
        "Writes the language model in ConstArpaLm format.",
        py::arg("os"), py::arg("binary"),
      py::call_guard<py::gil_scoped_release>())
      .def("WriteArpa",
        &PyClass::WriteArpa,
        "Creates Arpa format language model from ConstArpaLm format, and writes it "
        "to output stream. This will be useful in testing.",
        py::arg("os"),
      py::call_guard<py::gil_scoped_release>())
      .def("GetNgramLogprob",
        &PyClass::GetNgramLogprob,
        "Wrapper of GetNgramLogprobRecurse. It first maps possible out-of-vocabulary "
        "words to <unk>, if <unk> is defined, and then calls GetNgramLogprobRecurse.",
        py::arg("word"),
        py::arg("hist"))
      .def("HistoryStateExists",
        &PyClass::HistoryStateExists,
        "Returns true if the history word sequence <hist> has successor, which means "
        "<hist> will be a state in the FST format language model.",
        py::arg("hist"))
      .def("BosSymbol",
        &PyClass::BosSymbol)
      .def("EosSymbol",
        &PyClass::EosSymbol)
      .def("UnkSymbol",
        &PyClass::UnkSymbol)
      .def("NgramOrder",
        &PyClass::NgramOrder)
      .def("Initialized",
        &PyClass::Initialized);
  }
  {
    using PyClass = ConstArpaLmDeterministicFst;

    auto const_arpa_lm_deterministic = py::class_<ConstArpaLmDeterministicFst, DeterministicOnDemandFst<fst::StdArc>>(
        m, "ConstArpaLmDeterministicFst");
    const_arpa_lm_deterministic.def(py::init<const ConstArpaLm&>(),
        py::arg("lm"))
      .def("Start",
        &PyClass::Start)
      .def("Final",
        &PyClass::Final,
        py::arg("s"))
      .def("GetArc",
        &PyClass::GetArc,
        py::arg("s"),
        py::arg("ilabel"),
        py::arg("oarc"));
  }
  m.def("BuildConstArpaLm",
        &BuildConstArpaLm,
        "Reads in an Arpa format language model and converts it into ConstArpaLm "
        "format. We assume that the words in the input Arpa format language model have "
        "been converted into integers.",
        py::arg("options"),
        py::arg("arpa_rxfilename"),
        py::arg("const_arpa_wxfilename"));
}

void pybind_lm_kaldi_rnnlm(py::module &m) {

  {
    using PyClass = KaldiRnnlmWrapperOpts;

    auto kaldi_rnnlm_wrapper_opts = py::class_<PyClass>(
        m, "KaldiRnnlmWrapperOpts");
    kaldi_rnnlm_wrapper_opts.def(py::init<>())
      .def_readwrite("unk_symbol", &PyClass::unk_symbol)
      .def_readwrite("eos_symbol", &PyClass::eos_symbol);
  }
  {
    using PyClass = KaldiRnnlmWrapper;

    auto kaldi_rnnlm_wrapper = py::class_<PyClass>(
        m, "KaldiRnnlmWrapper");
    kaldi_rnnlm_wrapper.def(py::init<const KaldiRnnlmWrapperOpts &,
                    const std::string &,
                    const std::string &,
                    const std::string &>(),
        py::arg("opts"),
        py::arg("unk_prob_rspecifier"),
        py::arg("word_symbol_table_rxfilename"),
        py::arg("rnnlm_rxfilename"))
      .def("GetHiddenLayerSize",
        &PyClass::GetHiddenLayerSize)
      .def("GetEos",
        &PyClass::GetEos)
      .def("GetLogProb",
        &PyClass::GetLogProb,
        py::arg("word"),
        py::arg("wseq"),
        py::arg("context_in"),
        py::arg("context_out"));
  }
  {
    using PyClass = RnnlmDeterministicFst;

    auto rnnlm_deterministic_fst = py::class_<RnnlmDeterministicFst>(
        m, "RnnlmDeterministicFst");
    rnnlm_deterministic_fst.def(py::init<int32, KaldiRnnlmWrapper *>(),
        py::arg("max_ngram_order"),
        py::arg("rnnlm"))
      .def("Start",
        &PyClass::Start)
      .def("Final",
        &PyClass::Final,
        py::arg("s"))
      .def("GetArc",
        &PyClass::GetArc,
        py::arg("s"),
        py::arg("ilabel"),
        py::arg("oarc"));
  }
}

void pybind_lm_mikolov_rnnlm_lib(py::module &m) {

  using namespace rnnlm;

  {
    using PyClass = neuron;

    auto neuron = py::class_<PyClass>(
        m, "neuron");
  }

  {
    using PyClass = synapse;

    auto synapse = py::class_<PyClass>(
        m, "synapse");
  }

  {
    using PyClass = vocab_word;

    auto vocab_word = py::class_<PyClass>(
        m, "vocab_word");
  }
  py::enum_<FileTypeEnum>(m, "FileTypeEnum")
    .value("TEXT", FileTypeEnum::TEXT)
    .value("BINARY", FileTypeEnum::BINARY)
    .value("COMPRESSED", FileTypeEnum::COMPRESSED)
    .export_values();


  {
    using PyClass = CRnnLM;

    auto crnn_lm = py::class_<PyClass>(
        m, "CRnnLM");
    crnn_lm.def(py::init<>())
      .def("random",
        &PyClass::random,
        py::arg("min"),
        py::arg("max"))
      .def("setRnnLMFile",
        &PyClass::setRnnLMFile,
        py::arg("str"))
      .def("getHiddenLayerSize",
        &PyClass::getHiddenLayerSize)
      .def("setRandSeed",
        &PyClass::setRandSeed,
        py::arg("newSeed"))
      .def("getWordHash",
        &PyClass::getWordHash,
        py::arg("word"))
      .def("readWord",
        &PyClass::readWord,
        py::arg("word"),
        py::arg("fin"))
      .def("searchVocab",
        &PyClass::searchVocab,
        py::arg("word"))
      .def("saveWeights",
        &PyClass::saveWeights)
      .def("initNet",
        &PyClass::initNet)
      .def("goToDelimiter",
        &PyClass::goToDelimiter)
      .def("restoreNet",
        &PyClass::restoreNet)
      .def("netReset",
        &PyClass::netReset)
      .def("computeNet",
        &PyClass::computeNet,
        py::arg("last_word"),
        py::arg("word"))
      .def("copyHiddenLayerToInput",
        &PyClass::copyHiddenLayerToInput)
      .def("matrixXvector",
        &PyClass::matrixXvector,
        py::arg("dest"),
        py::arg("srcvec"),
        py::arg("srcmatrix"),
        py::arg("matrix_width"),
        py::arg("from"),
        py::arg("to"),
        py::arg("from2"),
        py::arg("to2"),
        py::arg("type"))
      .def("restoreContextFromVector",
        &PyClass::restoreContextFromVector,
        py::arg("context_in"))
      .def("saveContextToVector",
        &PyClass::saveContextToVector,
        py::arg("context_out"))
      .def("computeConditionalLogprob",
        &PyClass::computeConditionalLogprob,
        py::arg("current_word"),
        py::arg("history_words"),
        py::arg("context_in"),
        py::arg("context_out"))
      .def("setUnkSym",
        &PyClass::setUnkSym,
        py::arg("unk"))
      .def("setUnkPenalty",
        &PyClass::setUnkPenalty,
        py::arg("filename"))
      .def("getUnkPenalty",
        &PyClass::getUnkPenalty,
        py::arg("word"))
      .def("isUnk",
        &PyClass::isUnk,
        py::arg("word"));
  }
}

void init_lm(py::module &_m) {
  py::module m = _m.def_submodule("lm", "lm pybind for Kaldi");
  pybind_lm_arpa_file_parser(m);
  pybind_lm_arpa_lm_compiler(m);
  pybind_lm_const_arpa_lm(m);
  pybind_lm_kaldi_rnnlm(m);
  pybind_lm_mikolov_rnnlm_lib(m);

    m.def("arpa_to_fst",
    [](
      std::string arpa_rxfilename,
      py::handle symbol_table,
    std::string disambig_symbol = "#0",
    std::string bos_symbol = "<s>",
    std::string eos_symbol = "</s>",
    bool ilabel_sort = true
    ){
        auto pywrapfst_mod = py::module_::import("pywrapfst");
        auto ptr = reinterpret_cast<SymbolTableStruct*>(symbol_table.ptr());
      fst::SymbolTable* symbols = ptr->_smart_table.get();
    int64 disambig_symbol_id = 0;

    ArpaParseOptions options;
    options.max_warnings = -1;
      options.oov_handling = ArpaParseOptions::kSkipNGram;
      if (!disambig_symbol.empty()) {
        disambig_symbol_id = symbols->Find(disambig_symbol);
        if (disambig_symbol_id == -1) // fst::kNoSymbol
          KALDI_ERR << "Symbol table has no symbol for " << disambig_symbol;
      }
    // Add or use existing BOS and EOS.
    options.bos_symbol = symbols->AddSymbol(bos_symbol);
    options.eos_symbol = symbols->AddSymbol(eos_symbol);

    ArpaLmCompiler lm_compiler(options, disambig_symbol_id, symbols);
    {
      Input ki(arpa_rxfilename);
      lm_compiler.Read(ki.Stream());
    }
    if (ilabel_sort) {
      fst::ArcSort(lm_compiler.MutableFst(), fst::StdILabelCompare());
    }
    return lm_compiler.Fst();
    },
        py::arg("arpa_rxfilename"),
        py::arg("symbols"),
        py::arg("disambig_symbol") = "#0",
        py::arg("bos_symbol") = "<s>",
        py::arg("eos_symbol") = "</s>",
        py::arg("ilabel_sort") = true);

    m.def("arpa_to_const_arpa",
    [](
      std::string arpa_rxfilename,
      std::string const_arpa_wxfilename,
      fst::SymbolTable* symbols,
    int32 unk_symbol,
    int32 bos_symbol,
    int32 eos_symbol
    ){

    ArpaParseOptions options;
    options.unk_symbol = unk_symbol;
    options.bos_symbol = bos_symbol;
    options.eos_symbol = eos_symbol;
    bool ans = BuildConstArpaLm(options, arpa_rxfilename,
                                const_arpa_wxfilename);
    return ans;
    },
        py::arg("arpa_rxfilename"),
        py::arg("const_arpa_wxfilename"),
        py::arg("symbols"),
        py::arg("unk_symbol"),
        py::arg("bos_symbol"),
        py::arg("eos_symbol"));

}
