
#include "online/pybind_online.h"
#include "online/online-decodable.h"
#include "online/online-faster-decoder.h"
#include "online/online-feat-input.h"
#ifndef KALDI_NO_PORTAUDIO
  #include "online/online-tcp-source.h"
#endif
#include "online/onlinebin-util.h"

using namespace kaldi;

void pybind_online_audio_source(py::module &m) {
  {
    using PyClass = OnlineAudioSourceItf;
    py::class_<PyClass>(m, "OnlineAudioSourceItf")
        .def("Read", &PyClass::Read,
                        "Reads from the audio source, and writes the samples converted to BaseFloat "
                        "into the vector pointed by \"data\". "
                        "The user sets data->Dim() as a way of requesting that many samples. "
                        "The function returns true if there may be more data, and false if it "
                        "knows we are at the end of the stream. "
                        "In case an unexpected and unrecoverable error occurs the function throws "
                        "an exception of type KaldiFatalError (by using KALDI_ERR macro). "
                        "\n"
                        "NOTE: The older version of this interface had a second parameter - \"timeout\". "
                        "      We decided to remove it, because we don't envision usage scenarios, "
                        "      where \"timeout\" will need to be changed dynamically from call to call. "
                        "      If the particular audio source can experience timeouts for some reason "
                        "      (e.g. the samples are received over a network connection) "
                        "      we encourage the implementors to configure timeout using a "
                        "      constructor parameter. "
                        "      The suggested semantics are: if timeout is used and is greater than 0, "
                        "      this method has to wait no longer than \"timeout\" milliseconds before "
                        "      returning data-- by that time, it will return as much data as it has.",
             py::arg("data"));
  }
#ifndef KALDI_NO_PORTAUDIO
  {
    using PyClass = OnlinePaSource;
    py::class_<OnlinePaSource, OnlineAudioSourceItf>(m, "OnlineAudioSourceItf")
        .def(py::init<const uint32,
                 const uint32,
                 const uint32,
                 const uint32 >(),
              py::arg("timeout"),
              py::arg("sample_rate"),
              py::arg("rb_size"),
              py::arg("report_interval"))
        .def("Read", &PyClass::Read,
                        "Reads from the audio source, and writes the samples converted to BaseFloat "
                        "into the vector pointed by \"data\". "
                        "The user sets data->Dim() as a way of requesting that many samples. "
                        "The function returns true if there may be more data, and false if it "
                        "knows we are at the end of the stream. "
                        "In case an unexpected and unrecoverable error occurs the function throws "
                        "an exception of type KaldiFatalError (by using KALDI_ERR macro). "
                        "\n"
                        "NOTE: The older version of this interface had a second parameter - \"timeout\". "
                        "      We decided to remove it, because we don't envision usage scenarios, "
                        "      where \"timeout\" will need to be changed dynamically from call to call. "
                        "      If the particular audio source can experience timeouts for some reason "
                        "      (e.g. the samples are received over a network connection) "
                        "      we encourage the implementors to configure timeout using a "
                        "      constructor parameter. "
                        "      The suggested semantics are: if timeout is used and is greater than 0, "
                        "      this method has to wait no longer than \"timeout\" milliseconds before "
                        "      returning data-- by that time, it will return as much data as it has.",
             py::arg("data"))
        .def("TimedOut", &PyClass::TimedOut,
                        "Returns True if the last call to Read() failed to read the requested "
                        "number of samples due to timeout.");
  }
  m.def("PaCallback",
        &PaCallback,
        "The actual PortAudio callback - delegates to OnlinePaSource->Callback()",
        py::arg("input"),
        py::arg("output"),
        py::arg("frame_count"),
        py::arg("time_info"),
        py::arg("status_flags"),
        py::arg("user_data"));

#endif
  {
    using PyClass = OnlineVectorSource;
    py::class_<OnlineVectorSource, OnlineAudioSourceItf>(m,
        "OnlineVectorSource",
        "Simulates audio input, by returning data from a Vector. "
        "This class is mostly meant to be used for online decoder testing using "
        "pre-recorded audio")
        .def(py::init<const VectorBase<BaseFloat> &>(),
              py::arg("input"))
        .def("Read", &PyClass::Read,
                        "Reads from the audio source, and writes the samples converted to BaseFloat "
                        "into the vector pointed by \"data\". "
                        "The user sets data->Dim() as a way of requesting that many samples. "
                        "The function returns true if there may be more data, and false if it "
                        "knows we are at the end of the stream. "
                        "In case an unexpected and unrecoverable error occurs the function throws "
                        "an exception of type KaldiFatalError (by using KALDI_ERR macro). "
                        "\n"
                        "NOTE: The older version of this interface had a second parameter - \"timeout\". "
                        "      We decided to remove it, because we don't envision usage scenarios, "
                        "      where \"timeout\" will need to be changed dynamically from call to call. "
                        "      If the particular audio source can experience timeouts for some reason "
                        "      (e.g. the samples are received over a network connection) "
                        "      we encourage the implementors to configure timeout using a "
                        "      constructor parameter. "
                        "      The suggested semantics are: if timeout is used and is greater than 0, "
                        "      this method has to wait no longer than \"timeout\" milliseconds before "
                        "      returning data-- by that time, it will return as much data as it has.",
             py::arg("data"));
  }

}

void pybind_online_decodable(py::module &m) {

  {
    using PyClass = OnlineDecodableDiagGmmScaled;
    py::class_<OnlineDecodableDiagGmmScaled, DecodableInterface>(m,
        "OnlineDecodableDiagGmmScaled",
        "A decodable, taking input from an OnlineFeatureInput object on-demand")
        .def(py::init<const AmDiagGmm &,
                               const TransitionModel &,
                               const BaseFloat ,
                               OnlineFeatureMatrix *>(),
              py::arg("am"),
              py::arg("trans_model"),
              py::arg("scale"),
              py::arg("input_feats"))
        .def("LogLikelihood", &PyClass::LogLikelihood,
              "Returns the log likelihood, which will be negated in the decoder.",
             py::arg("frame"),
             py::arg("index"))
        .def("IsLastFrame", &PyClass::IsLastFrame,
             py::arg("frame"))
        .def("NumIndices", &PyClass::NumIndices,
              "Indices are one-based!  This is for compatibility with OpenFst.");
  }
}

void pybind_online_faster_decoder(py::module &m) {

  {
    using PyClass = OnlineFasterDecoderOpts;
    py::class_<OnlineFasterDecoderOpts, FasterDecoderOptions>(m,
        "OnlineFasterDecoderOpts",
        "Extends the definition of FasterDecoder's options to include additional "
        "parameters. The meaning of the \"beam\" option is also redefined as "
        "the _maximum_ beam value allowed.")
        .def(py::init<>())
        .def_readwrite("rt_min", &PyClass::rt_min,
          "minimum decoding runtime factor")
        .def_readwrite("rt_max", &PyClass::rt_max,
          "maximum decoding runtime factor")
        .def_readwrite("batch_size", &PyClass::batch_size,
          "number of features decoded in one go")
        .def_readwrite("inter_utt_sil", &PyClass::inter_utt_sil,
          "minimum silence (#frames) to trigger end of utterance")
        .def_readwrite("max_utt_len_", &PyClass::max_utt_len_,
          "if utt. is longer, we accept shorter silence as utt. separators")
        .def_readwrite("update_interval", &PyClass::update_interval,
          "beam update period in # of frames")
        .def_readwrite("beam_update", &PyClass::beam_update,
          "rate of adjustment of the beam")
        .def_readwrite("max_beam_update", &PyClass::max_beam_update,
          "maximum rate of beam adjustment");
  }
  {
    using PyClass = OnlineFasterDecoder;
    auto online_faster_decoder = py::class_<OnlineFasterDecoder, FasterDecoder>(m,
        "OnlineFasterDecoder");
    online_faster_decoder.def(py::init<const fst::Fst<fst::StdArc> &,
                      const OnlineFasterDecoderOpts &,
                      const std::vector<int32> &,
                      const TransitionModel &>(),
        py::arg("fst"),
        py::arg("opts"),
        py::arg("sil_phones"),
        py::arg("trans_model"))
      .def("Decode", &PyClass::Decode,
        py::arg("decodable"))
      .def("PartialTraceback", &PyClass::PartialTraceback,
        "Makes a linear graph, by tracing back from the last \"immortal\" token "
        "to the previous one",
        py::arg("out_fst"))
      .def("FinishTraceBack", &PyClass::FinishTraceBack,
        "Makes a linear graph, by tracing back from the best currently active token "
        "to the last immortal token. This method is meant to be invoked at the end "
        "of an utterance in order to get the last chunk of the hypothesis",
        py::arg("fst_out"))
      .def("EndOfUtterance", &PyClass::EndOfUtterance,
        "Returns \"true\" if the best current hypothesis ends with long enough silence")
      .def("frame", &PyClass::frame);
  py::enum_<OnlineFasterDecoder::DecodeState>(online_faster_decoder, "DecodeState")
    .value("kEndFeats", OnlineFasterDecoder::DecodeState::kEndFeats)
    .value("kEndUtt", OnlineFasterDecoder::DecodeState::kEndUtt)
    .value("kEndBatch", OnlineFasterDecoder::DecodeState::kEndBatch)
    .export_values();
  }
}

void pybind_online_feat_input(py::module &m) {

  {
    using PyClass = OnlineFeatInputItf;
    py::class_<PyClass>(m, "OnlineFeatInputItf")
        .def("Compute", &PyClass::Compute,
              "Produces feature vectors in some way. "
              "The features may be e.g. extracted from an audio samples, received and/or "
              "transformed from another OnlineFeatInput class etc. "
              "\n"
              "\"output\" - a matrix to store the extracted feature vectors in its rows. "
              "           The number of rows (NumRows()) of \"output\" when the function is "
              "           called, is treated as a hint of how many frames the user wants, "
              "           but this function does not promise to produce exactly that many: "
              "           it may be slightly more, less, or even zero, on a given call. "
              "           Zero frames may be returned because we timed out or because "
              "           we're at the beginning of the file and some buffering is going on. "
              "           In that case you should try again.  The function will return \"false\" "
              "           when it knows the stream is finished, but if it returns nothing "
              "           several times in a row you may want to terminate processing the "
              "           stream. "
              "\n"
              "Note: similar to the OnlineAudioSourceItf::Read(), Compute() previously "
              "      had a second argument - \"timeout\". Again we decided against including "
              "      this parameter in the interface specification. Instead we are "
              "      considering time out handling to be implementation detail, and if needed "
              "      it should be configured, through the descendant class' constructor, "
              "      or by other means. "
              "      For consistency, we recommend 'timeout' values greater than zero "
              "      to mean that Compute() should not block for more than that number "
              "      of milliseconds, and to return whatever data it has, when the timeout "
              "      period is exceeded. "
              "\n"
              "Returns \"false\" if we know the underlying data source has no more data, and "
              "true if there may be more data.",
             py::arg("output"))
        .def("Dim", &PyClass::Dim);
  }

  {
    using PyClass = OnlineCmnInput;
    py::class_<OnlineCmnInput, OnlineFeatInputItf>(m, "OnlineCmnInput")
        .def(py::init<OnlineFeatInputItf *, int32 , int32 >(),
            py::arg("input"),
            py::arg("cmn_window"),
            py::arg("min_window"))
        .def("Compute", &PyClass::Compute,
              "Produces feature vectors in some way. "
              "The features may be e.g. extracted from an audio samples, received and/or "
              "transformed from another OnlineFeatInput class etc. "
              "\n"
              "\"output\" - a matrix to store the extracted feature vectors in its rows. "
              "           The number of rows (NumRows()) of \"output\" when the function is "
              "           called, is treated as a hint of how many frames the user wants, "
              "           but this function does not promise to produce exactly that many: "
              "           it may be slightly more, less, or even zero, on a given call. "
              "           Zero frames may be returned because we timed out or because "
              "           we're at the beginning of the file and some buffering is going on. "
              "           In that case you should try again.  The function will return \"false\" "
              "           when it knows the stream is finished, but if it returns nothing "
              "           several times in a row you may want to terminate processing the "
              "           stream. "
              "\n"
              "Note: similar to the OnlineAudioSourceItf::Read(), Compute() previously "
              "      had a second argument - \"timeout\". Again we decided against including "
              "      this parameter in the interface specification. Instead we are "
              "      considering time out handling to be implementation detail, and if needed "
              "      it should be configured, through the descendant class' constructor, "
              "      or by other means. "
              "      For consistency, we recommend 'timeout' values greater than zero "
              "      to mean that Compute() should not block for more than that number "
              "      of milliseconds, and to return whatever data it has, when the timeout "
              "      period is exceeded. "
              "\n"
              "Returns \"false\" if we know the underlying data source has no more data, and "
              "true if there may be more data.",
             py::arg("output"))
        .def("Dim", &PyClass::Dim);
  }

  {
    using PyClass = OnlineCacheInput;
    py::class_<OnlineCacheInput, OnlineFeatInputItf>(m, "OnlineCacheInput")
        .def(py::init<OnlineFeatInputItf *>(),
            py::arg("input"))
        .def("Compute", &PyClass::Compute,
              "The Compute function just forwards to the previous member of the "
              "chain, except that we locally accumulate the result, and "
              "GetCachedData() will return the entire input up to the current time.",
             py::arg("output"))
        .def("GetCachedData", &PyClass::GetCachedData,
             py::arg("output"))
        .def("Dim", &PyClass::Dim)
        .def("Deallocate", &PyClass::Deallocate);
  }
#ifndef KALDI_NO_PORTAUDIO

  {
    using PyClass = OnlineUdpInput;
    py::class_<OnlineUdpInput, OnlineFeatInputItf>(m, "OnlineUdpInput")
        .def(py::init<int32 , int32 feature_dim>(),
            py::arg("port"),
            py::arg("feature_dim"))
        .def("Compute", &PyClass::Compute,
             py::arg("output"))
        .def("Dim", &PyClass::Dim)
        .def("client_addr", &PyClass::client_addr)
        .def("descriptor", &PyClass::descriptor);
  }
#endif

  {
    using PyClass = OnlineLdaInput;
    py::class_<OnlineLdaInput, OnlineFeatInputItf>(m, "OnlineLdaInput")
        .def(py::init<OnlineFeatInputItf *,
                 const Matrix<BaseFloat> &,
                 int32 ,
                 int32>(),
            py::arg("input"),
            py::arg("transform"),
            py::arg("left_context"),
            py::arg("right_context"))
        .def("Compute", &PyClass::Compute,
              "The Compute function just forwards to the previous member of the "
              "chain, except that we locally accumulate the result, and "
              "GetCachedData() will return the entire input up to the current time.",
             py::arg("output"))
        .def("Dim", &PyClass::Dim);
  }

  {
    using PyClass = OnlineDeltaInput;
    py::class_<OnlineDeltaInput, OnlineFeatInputItf>(m, "OnlineDeltaInput")
        .def(py::init<const DeltaFeaturesOptions &,
                   OnlineFeatInputItf *>(),
            py::arg("delta_opts"),
            py::arg("input"))
        .def("Compute", &PyClass::Compute,
              "The Compute function just forwards to the previous member of the "
              "chain, except that we locally accumulate the result, and "
              "GetCachedData() will return the entire input up to the current time.",
             py::arg("output"))
        .def("Dim", &PyClass::Dim);
  }
  {
    using PyClass = OnlineFeatureMatrixOptions;

    auto online_feature_matrix_options = py::class_<PyClass>(
        m, "OnlineFeatureMatrixOptions");
    online_feature_matrix_options.def(py::init<>())
      .def_readwrite("batch_size", &PyClass::batch_size,
            "number of frames to request each time.")
      .def_readwrite("num_tries", &PyClass::num_tries,
            "number of tries of getting no output and timing out, "
            "before we give up.");
  }
  {
    using PyClass = OnlineFeatureMatrix;

    auto online_feature_matrix = py::class_<PyClass>(
        m, "OnlineFeatureMatrix");
    online_feature_matrix.def(py::init<const OnlineFeatureMatrixOptions &,
                      OnlineFeatInputItf *>(),
            py::arg("opts"),
            py::arg("input"))
      .def("IsValidFrame", &PyClass::IsValidFrame,
          py::arg("frame"))
      .def("Dim", &PyClass::Dim)
      .def("GetFrame", &PyClass::GetFrame,
          "GetFrame() will die if it's not a valid frame; you have to "
          "call IsValidFrame() for this frame, to see whether it "
          "is valid.",
          py::arg("frame"));
  }
}

void pybind_online_tcp_source(py::module &m) {

#ifndef KALDI_NO_PORTAUDIO
  {
    using PyClass = OnlineTcpVectorSource;

    auto online_tcp_vector_source = py::class_<PyClass, OnlineAudioSourceItf>(
        m, "OnlineTcpVectorSource");
    online_tcp_vector_source.def(py::init<int32 >(),
            py::arg("socket"))
      .def("Read", &PyClass::Read,
          py::arg("data"))
      .def("IsConnected", &PyClass::IsConnected)
      .def("SamplesProcessed", &PyClass::SamplesProcessed)
      .def("ResetSamples", &PyClass::ResetSamples);
  }
#endif // !defined(KALDI_NO_PORTAUDIO)
}

void pybind_onlinebin_util(py::module &m) {

  m.def("ReadDecodeGraph",
        &ReadDecodeGraph,
        "Reads a decoding graph from a file",
        py::arg("filename"));
  m.def("PrintPartialResult",
        &PrintPartialResult,
        "Prints a string corresponding to (a possibly partial) decode result as "
        "and adds a \"new line\" character if \"line_break\" argument is true",
        py::arg("words"),
        py::arg("word_syms"),
        py::arg("line_break"));
}

void init_online(py::module &_m) {
  py::module m = _m.def_submodule("online", "online pybind for Kaldi");
  pybind_online_audio_source(m);
  pybind_online_decodable(m);
  pybind_online_faster_decoder(m);
  pybind_online_feat_input(m);
  pybind_online_tcp_source(m);
  pybind_onlinebin_util(m);
}
