
#include "online2/pybind_online2.h"
#include "online2/online-endpoint.h"
#include "online2/online-feature-pipeline.h"
#include "online2/online-gmm-decodable.h"
#include "online2/online-gmm-decoding.h"
#include "online2/online-ivector-feature.h"
#include "online2/online-nnet2-decoding-threaded.h"
#include "online2/online-nnet2-decoding.h"
#include "online2/online-nnet2-feature-pipeline.h"
#include "online2/online-nnet3-decoding.h"
#include "online2/online-nnet3-incremental-decoding.h"
#include "online2/online-nnet3-wake-word-faster-decoder.h"
#include "online2/online-speex-wrapper.h"
#include "online2/online-timing.h"
#include "online2/onlinebin-util.h"

using namespace kaldi;

void pybind_online_endpoint(py::module &m) {

  {
    using PyClass = OnlineEndpointRule;

    auto online_endpoint_rule = py::class_<PyClass>(
        m, "OnlineEndpointRule");
    online_endpoint_rule
      .def(py::init<bool ,
                     BaseFloat ,
                     BaseFloat ,
                     BaseFloat >(),
                       py::arg("must_contain_nonsilence") = true,
                       py::arg("min_trailing_silence") = 1.0,
                       py::arg("max_relative_cost") = std::numeric_limits<BaseFloat>::infinity(),
                       py::arg("min_utterance_length") = 0.0)
      .def_readwrite("must_contain_nonsilence", &PyClass::must_contain_nonsilence)
      .def_readwrite("min_trailing_silence", &PyClass::min_trailing_silence)
      .def_readwrite("max_relative_cost", &PyClass::max_relative_cost)
      .def_readwrite("min_utterance_length", &PyClass::min_utterance_length);
  }

  {
    using PyClass = OnlineEndpointConfig;

    auto online_endpoint_config = py::class_<PyClass>(
        m, "OnlineEndpointConfig");
    online_endpoint_config.def(py::init<>())
      .def_readwrite("silence_phones", &PyClass::silence_phones,
              "e.g. 1:2:3:4, colon separated list of phones "
                              "that we consider as silence for purposes of "
                              "endpointing.")
      .def_readwrite("rule1", &PyClass::rule1,
          "rule1 times out after 5 seconds of silence, even if we decoded nothing.")
      .def_readwrite("rule2", &PyClass::rule2,
          "rule2 times out after 0.5 seconds of silence if we reached the final-state "
          "with good probability (relative_cost < 2.0) after decoding something.")
      .def_readwrite("rule3", &PyClass::rule3,
          "rule3 times out after 1.0 seconds of silence if we reached the final-state "
          "with OK probability (relative_cost < 8.0) after decoding something")
      .def_readwrite("rule4", &PyClass::rule4,
          "rule4 times out after 2.0 seconds of silence after decoding something, "
          "even if we did not reach a final-state at all.")
      .def_readwrite("rule5", &PyClass::rule5,
          "rule5 times out after the utterance is 20 seconds long, regardless of "
        "anything else.");
  }
  //m.def("EndpointDetected",
  //      py::overload_cast<const OnlineEndpointConfig &,
  //                    int32,
  //                    int32,
  //                    BaseFloat,
  //                    BaseFloat>(&EndpointDetected),
  //      "This function returns true if this set of endpointing "
  //      "rules thinks we should terminate decoding.  Note: in verbose "
  //      "mode it will print logging information when returning true.",
  //      py::arg("config"),
  //      py::arg("num_frames_decoded"),
  //      py::arg("trailing_silence_frames"),
  //      py::arg("frame_shift_in_seconds"),
  //      py::arg("final_relative_cost"));

}

void pybind_online_feature_pipeline(py::module &m) {

  {
    using PyClass = OnlineFeaturePipelineCommandLineConfig;

    py::class_<PyClass>(
        m, "OnlineFeaturePipelineCommandLineConfig",
        "This configuration class is to set up OnlineFeaturePipelineConfig, which "
        "in turn is the configuration class for OnlineFeaturePipeline. "
        "Instead of taking the options for the parts of the feature pipeline "
        "directly, it reads in the names of configuration classes. "
        "I'm conflicted about whether this is a wise thing to do, but I think "
        "for ease of scripting it's probably better to do it like this.")
        .def(py::init<>())
      .def_readwrite("feature_type", &PyClass::feature_type)
      .def_readwrite("mfcc_config", &PyClass::mfcc_config)
      .def_readwrite("plp_config", &PyClass::plp_config)
      .def_readwrite("fbank_config", &PyClass::fbank_config)
      .def_readwrite("add_pitch", &PyClass::add_pitch)
      .def_readwrite("pitch_config", &PyClass::pitch_config)
      .def_readwrite("pitch_process_config", &PyClass::pitch_process_config)
      .def_readwrite("cmvn_config", &PyClass::cmvn_config)
      .def_readwrite("global_cmvn_stats_rxfilename", &PyClass::global_cmvn_stats_rxfilename)
      .def_readwrite("add_deltas", &PyClass::add_deltas)
      .def_readwrite("delta_config", &PyClass::delta_config)
      .def_readwrite("splice_feats", &PyClass::splice_feats)
      .def_readwrite("splice_config", &PyClass::splice_config)
      .def_readwrite("lda_rxfilename", &PyClass::lda_rxfilename);
  }
  {
    using PyClass = OnlineFeaturePipelineConfig;

    py::class_<PyClass>(
        m, "OnlineFeaturePipelineConfig",
        "This configuration class is to set up OnlineFeaturePipelineConfig, which "
        "in turn is the configuration class for OnlineFeaturePipeline. "
        "Instead of taking the options for the parts of the feature pipeline "
        "directly, it reads in the names of configuration classes. "
        "I'm conflicted about whether this is a wise thing to do, but I think "
        "for ease of scripting it's probably better to do it like this.")
        .def(py::init<>())
        .def(py::init<const OnlineFeaturePipelineCommandLineConfig>(),
                       py::arg("cmdline_config"))
      .def("FrameShiftInSeconds", &PyClass::FrameShiftInSeconds)
      .def_readwrite("feature_type", &PyClass::feature_type)
      .def_readwrite("mfcc_opts", &PyClass::mfcc_opts)
      .def_readwrite("plp_opts", &PyClass::plp_opts)
      .def_readwrite("fbank_opts", &PyClass::fbank_opts)
      .def_readwrite("add_pitch", &PyClass::add_pitch)
      .def_readwrite("pitch_opts", &PyClass::pitch_opts)
      .def_readwrite("pitch_process_opts", &PyClass::pitch_process_opts)
      .def_readwrite("cmvn_opts", &PyClass::cmvn_opts)
      .def_readwrite("add_deltas", &PyClass::add_deltas)
      .def_readwrite("delta_opts", &PyClass::delta_opts)
      .def_readwrite("splice_feats", &PyClass::splice_feats)
      .def_readwrite("splice_opts", &PyClass::splice_opts)
      .def_readwrite("lda_rxfilename", &PyClass::lda_rxfilename)
      .def_readwrite("global_cmvn_stats_rxfilename", &PyClass::global_cmvn_stats_rxfilename);
  }
}

void pybind_online_gmm_decodable(py::module &m) {

  {
    using PyClass = DecodableDiagGmmScaledOnline;

    auto decodable_diag_gmm_scaled_online = py::class_<PyClass>(
        m, "DecodableDiagGmmScaledOnline");
    decodable_diag_gmm_scaled_online
      .def(py::init<const AmDiagGmm &,
                               const TransitionModel &,
                               const BaseFloat ,
                               OnlineFeatureInterface *>(),
                       py::arg("am"),
                       py::arg("trans_model"),
                       py::arg("scale"),
                       py::arg("input_feats"))
      .def("LogLikelihood", &PyClass::LogLikelihood,
                       py::arg("frame"),
                       py::arg("index"))
      .def("IsLastFrame", &PyClass::IsLastFrame,
                       py::arg("frame"))
      .def("NumFramesReady", &PyClass::NumFramesReady)
      .def("NumIndices", &PyClass::NumIndices);
  }
}

void pybind_online_gmm_decoding(py::module &m) {

  {
    using PyClass = OnlineGmmDecodingAdaptationPolicyConfig;

    auto online_gmm_decoding_adaptation_policy_config = py::class_<PyClass>(
        m, "OnlineGmmDecodingAdaptationPolicyConfig");
    online_gmm_decoding_adaptation_policy_config
      .def(py::init<>())
      .def_readwrite("adaptation_first_utt_delay", &PyClass::adaptation_first_utt_delay)
      .def_readwrite("adaptation_first_utt_ratio", &PyClass::adaptation_first_utt_ratio)
      .def_readwrite("adaptation_delay", &PyClass::adaptation_delay)
      .def_readwrite("adaptation_ratio", &PyClass::adaptation_ratio)
      .def("Check", &PyClass::Check)
      .def("DoAdapt", &PyClass::DoAdapt,
          "This function returns true if we are scheduled "
          "to re-estimate fMLLR somewhere in the interval "
          "[ chunk_begin_secs, chunk_end_secs ).",
                       py::arg("chunk_begin_secs"),
                       py::arg("chunk_end_secs"),
                       py::arg("is_first_utterance"));
  }

  {
    using PyClass = OnlineGmmDecodingConfig;

    auto online_gmm_decoding_config = py::class_<PyClass>(
        m, "OnlineGmmDecodingConfig");
    online_gmm_decoding_config
      .def(py::init<>())
      .def_readwrite("fmllr_lattice_beam", &PyClass::fmllr_lattice_beam)
      .def_readwrite("basis_opts", &PyClass::basis_opts,
        "options for basis-fMLLR adaptation.")
      .def_readwrite("faster_decoder_opts", &PyClass::faster_decoder_opts)
      .def_readwrite("adaptation_policy_opts", &PyClass::adaptation_policy_opts)
      .def_readwrite("online_alimdl_rxfilename", &PyClass::online_alimdl_rxfilename,
            "rxfilename for model trained with online-CMN features "
            "(only needed if different from model_rxfilename)")
      .def_readwrite("model_rxfilename", &PyClass::model_rxfilename,
            "rxfilename for model used for estimating fMLLR transforms")
      .def_readwrite("rescore_model_rxfilename", &PyClass::rescore_model_rxfilename,
            "rxfilename for possible discriminatively trained model"
            "(only needed if different from model_rxfilename)")
      .def_readwrite("fmllr_basis_rxfilename", &PyClass::fmllr_basis_rxfilename,
            "rxfilename for the BasisFmllrEstimate object containing the basis "
            "used for basis-fMLLR.")
      .def_readwrite("acoustic_scale", &PyClass::acoustic_scale)
      .def_readwrite("silence_phones", &PyClass::silence_phones)
      .def_readwrite("silence_weight", &PyClass::silence_weight);
  }
  {
    using PyClass = OnlineGmmDecodingModels;

    auto online_gmm_decoding_models = py::class_<PyClass>(
        m, "OnlineGmmDecodingModels");
    online_gmm_decoding_models.def(py::init<const OnlineGmmDecodingConfig &>(),
        py::arg("config"))
      .def("GetTransitionModel",
        &PyClass::GetTransitionModel)
      .def("GetOnlineAlignmentModel",
        &PyClass::GetOnlineAlignmentModel)
      .def("GetModel",
        &PyClass::GetModel)
      .def("GetFinalModel",
        &PyClass::GetFinalModel)
      .def("GetFmllrBasis",
        &PyClass::GetFmllrBasis);
  }
  {
    using PyClass = OnlineGmmAdaptationState;

    auto online_gmm_adaptation_state = py::class_<PyClass>(
        m, "OnlineGmmAdaptationState");
    online_gmm_adaptation_state.def(py::init<>())
      .def_readwrite("cmvn_state", &PyClass::cmvn_state)
      .def_readwrite("spk_stats", &PyClass::spk_stats)
      .def_readwrite("transform", &PyClass::transform)
      .def("Write",
        &PyClass::Write,
                       py::arg("out_stream"),
                       py::arg("binary"))
      .def("Read",
        &PyClass::Read,
                       py::arg("in_stream"),
                       py::arg("binary"));
  }
  {
    using PyClass = SingleUtteranceGmmDecoder;

    auto single_utterance_gmm_decoder = py::class_<PyClass>(
        m, "SingleUtteranceGmmDecoder",
        "You will instantiate this class when you want to decode a single "
        "utterance using the online-decoding setup.  This is an alternative "
        "to manually putting things together yourself.");
    single_utterance_gmm_decoder.def(py::init<const OnlineGmmDecodingConfig &,
                            const OnlineGmmDecodingModels &,
                            const OnlineFeaturePipeline &,
                            const fst::Fst<fst::StdArc> &,
                            const OnlineGmmAdaptationState &>(),
                       py::arg("config"),
                       py::arg("models"),
                       py::arg("feature_prototype"),
                       py::arg("fst"),
                       py::arg("adaptation_state"))
      .def("FeaturePipeline",
        &PyClass::FeaturePipeline)
      .def("AdvanceDecoding",
        &PyClass::AdvanceDecoding,
        "advance the decoding as far as we can.  May also estimate fMLLR after "
        "advancing the decoding, depending on the configuration values in "
        "config_.adaptation_policy_opts.  [Note: we expect the user will also call "
        "EstimateFmllr() at utterance end, which should generally improve the "
        "quality of the estimated transforms, although we don't rely on this].")
      .def("FinalizeDecoding",
        &PyClass::FinalizeDecoding,
        "Finalize the decoding. Cleanups and prunes remaining tokens, so the final result "
        "is faster to obtain.")
      .def("HaveTransform",
        &PyClass::HaveTransform,
        "Returns true if we already have an fMLLR transform.  The user will "
        "already know this; the call is for convenience.")
      .def("EstimateFmllr",
        &PyClass::EstimateFmllr,
        "Estimate the [basis-]fMLLR transform and apply it to the features. "
        "This will get used if you call RescoreLattice() or if you just "
        "continue decoding; however to get it applied retroactively "
        "you'd have to call RescoreLattice(). "
        "\"end_of_utterance\" just affects how we interpret the final-probs in the "
        "lattice.  This should generally be true if you think you've reached "
        "the end of the grammar, and false otherwise.",
                       py::arg("end_of_utterance"))
      .def("GetAdaptationState",
        &PyClass::GetAdaptationState,
                       py::arg("adaptation_state"))
      .def("GetLattice",
        &PyClass::GetLattice,
        "Gets the lattice.  If rescore_if_needed is true, and if there is any point "
        "in rescoring the state-level lattice (see RescoringIsNeeded()), it will "
        "rescore the lattice.  The output lattice has any acoustic scaling in it "
        "(which will typically be desirable in an online-decoding context); if you "
        "want an un-scaled lattice, scale it using ScaleLattice() with the inverse "
        "of the acoustic weight.  \"end_of_utterance\" will be true if you want the "
        "final-probs to be included.",
                       py::arg("rescore_if_needed"),
                       py::arg("end_of_utterance"),
                       py::arg("clat"))
      .def("GetBestPath",
        &PyClass::GetBestPath,
        "Outputs an FST corresponding to the single best path through the current "
        "lattice. If \"use_final_probs\" is true AND we reached the final-state of "
        "the graph then it will include those as final-probs, else it will treat "
        "all final-probs as one.",
                       py::arg("end_of_utterance"),
                       py::arg("best_path"))
      .def("FinalRelativeCost",
        &PyClass::FinalRelativeCost,
        "This function outputs to \"final_relative_cost\", if non-NULL, a number >= 0 "
        "that will be close to zero if the final-probs were close to the best probs "
        "active on the final frame.  (the output to final_relative_cost is based on "
        "the first-pass decoding).  If it's close to zero (e.g. < 5, as a guess), "
        "it means you reached the end of the grammar with good probability, which "
        "can be taken as a good sign that the input was OK.")
      .def("EndpointDetected",
        &PyClass::EndpointDetected,
        "This function calls EndpointDetected from online-endpoint.h, "
        "with the required arguments.",
                       py::arg("config"));
  }
}

void pybind_online_ivector_feature(py::module &m) {

  {
    using PyClass = OnlineIvectorExtractionConfig;

    auto online_ivector_extraction_config = py::class_<PyClass>(
        m, "OnlineIvectorExtractionConfig",
        "This class includes configuration variables relating to the online iVector "
      "extraction, but not including configuration for the \"base feature\", "
      "i.e. MFCC/PLP/filterbank, which is an input to this feature.  This "
      "configuration class can be used from the command line, but before giving it "
      "to the code we create a config class called "
      "OnlineIvectorExtractionInfo which contains the actual configuration "
      "classes as well as various objects that are needed.  The principle is that "
      "any code should be callable from other code, so we didn't want to force "
      "configuration classes to be read from disk.");
    online_ivector_extraction_config.def(py::init<>())
      .def_readwrite("lda_mat_rxfilename", &PyClass::lda_mat_rxfilename,
            "to read the LDA+MLLT matrix")
      .def_readwrite("global_cmvn_stats_rxfilename", &PyClass::global_cmvn_stats_rxfilename,
            "to read matrix of global CMVN stats")
      .def_readwrite("splice_config_rxfilename", &PyClass::splice_config_rxfilename,
            "to read OnlineSpliceOptions")
      .def_readwrite("cmvn_config_rxfilename", &PyClass::cmvn_config_rxfilename,
            "to read in OnlineCmvnOptions")
      .def_readwrite("online_cmvn_iextractor", &PyClass::online_cmvn_iextractor,
            "flag activating online-cmvn in iextractor feature pipeline")
      .def_readwrite("diag_ubm_rxfilename", &PyClass::diag_ubm_rxfilename,
            "reads type DiagGmm.")
      .def_readwrite("ivector_extractor_rxfilename", &PyClass::ivector_extractor_rxfilename,
            "reads type IvectorExtractor")
      .def_readwrite("ivector_period", &PyClass::ivector_period,
            "How frequently we re-estimate iVectors.")
      .def_readwrite("num_gselect", &PyClass::num_gselect,
            "maximum number of posteriors to use per frame for "
                      "iVector extractor.")
      .def_readwrite("min_post", &PyClass::min_post,
            "pruning threshold for posteriors for the iVector "
                       "extractor.")
      .def_readwrite("posterior_scale", &PyClass::posterior_scale,
                  "Scale on posteriors used for iVector "
                              "extraction; can be interpreted as the inverse "
                              "of a scale on the log-prior.")
      .def_readwrite("max_count", &PyClass::max_count,
                  "Maximum stats count we allow before we start scaling "
                        "down stats (if nonzero).. this prevents us getting "
                        "atypical-looking iVectors for very long utterances. "
                        "Interpret this as a number of frames times "
                        "posterior_scale, typically 1/10 of a frame count.")
      .def_readwrite("num_cg_iters", &PyClass::num_cg_iters,
        "set to 15.  I don't believe this is very important, so it's "
                       "not configurable from the command line for now.")
      .def_readwrite("use_most_recent_ivector", &PyClass::use_most_recent_ivector,
              "If use_most_recent_ivector is true, we always return the most recent "
              "available iVector rather than the one for the current frame.  This means "
              "that if audio is coming in faster than we can process it, we will return a "
              "more accurate iVector.")
      .def_readwrite("greedy_ivector_extractor", &PyClass::greedy_ivector_extractor,
            "If true, always read ahead to NumFramesReady() when getting iVector stats.")
      .def_readwrite("max_remembered_frames", &PyClass::max_remembered_frames,
            "max_remembered_frames is the largest number of frames it will remember "
            "between utterances of the same speaker; this affects the output of "
            "GetAdaptationState(), and has the effect of limiting the number of frames "
            "of both the CMVN stats and the iVector stats.  Setting this to a smaller "
            "value means the adaptation is less constrained by previous utterances "
            "(assuming you provided info from a previous utterance of the same speaker "
            "by calling SetAdaptationState()).");
  }
  {
    using PyClass = OnlineIvectorExtractionInfo;

    auto online_ivector_extraction_info = py::class_<PyClass>(
        m, "OnlineIvectorExtractionInfo",
          "This struct contains various things that are needed (as const references) "
          "by class OnlineIvectorExtractor.");
    online_ivector_extraction_info.def(py::init<>())
      .def(py::init<const OnlineIvectorExtractionConfig &>(),
                       py::arg("config"))
      .def_readwrite("lda_mat", &PyClass::lda_mat, "LDA+MLLT matrix.")
      .def_readwrite("global_cmvn_stats", &PyClass::global_cmvn_stats,
            "Global CMVN stats.")
      .def_readwrite("cmvn_opts", &PyClass::cmvn_opts,
              "Options for online CMN/CMVN computation.")
      .def_readwrite("online_cmvn_iextractor", &PyClass::online_cmvn_iextractor,
              "flag activating online CMN/CMVN for iextractor input.")
      .def_readwrite("splice_opts", &PyClass::splice_opts,
          "Options for frame splicing (--left-context,--right-context)")
      .def_readonly("diag_ubm", &PyClass::diag_ubm)
      .def_readwrite("extractor", &PyClass::extractor)
      .def_readwrite("ivector_period", &PyClass::ivector_period)
      .def_readwrite("num_gselect", &PyClass::num_gselect)
      .def_readwrite("min_post", &PyClass::min_post)
      .def_readwrite("posterior_scale", &PyClass::posterior_scale)
      .def_readwrite("max_count", &PyClass::max_count)
      .def_readwrite("num_cg_iters", &PyClass::num_cg_iters)
      .def_readwrite("use_most_recent_ivector",
          &PyClass::use_most_recent_ivector)
      .def_readwrite("greedy_ivector_extractor",
          &PyClass::greedy_ivector_extractor)
      .def_readwrite("max_remembered_frames",
          &PyClass::max_remembered_frames)
      .def("Init",
        &PyClass::Init,
                       py::arg("config"))
      .def("ExpectedFeatureDim",
        &PyClass::ExpectedFeatureDim)
      .def("Check",
        &PyClass::Check);
  }
  {
    using PyClass = OnlineIvectorExtractorAdaptationState;

    auto online_ivector_extractor_adaptation_state = py::class_<PyClass>(
        m, "OnlineIvectorExtractorAdaptationState");
    online_ivector_extractor_adaptation_state.def(py::init<const OnlineIvectorExtractorAdaptationState &>(),
                       py::arg("other"))
      .def(py::init<const OnlineIvectorExtractionInfo &>(),
                       py::arg("info"))
      .def_readwrite("cmvn_state", &PyClass::cmvn_state)
      .def_readwrite("ivector_stats", &PyClass::ivector_stats)
      .def("LimitFrames",
        &PyClass::LimitFrames,
        "Scales down the stats if needed to ensure the number of frames in the "
        "speaker-specific CMVN stats does not exceed max_remembered_frames "
        "and the data-count in the iVector stats does not exceed "
        "max_remembered_frames * posterior_scale.  [the posterior_scale "
        "factor is necessary because those stats have already been scaled "
        "by that factor.]",
                       py::arg("max_remembered_frames"),
                       py::arg("posterior_scale"))
      .def("Write",
        &PyClass::Write,
                       py::arg("os"),
                       py::arg("binary"))
      .def("Read",
        &PyClass::Read,
                       py::arg("is"),
                       py::arg("binary"));
  }
  {
    using PyClass = OnlineIvectorFeature;

    auto online_ivector_feature = py::class_<OnlineIvectorFeature, OnlineFeatureInterface>(
        m, "OnlineIvectorFeature");
    online_ivector_feature.def(py::init<const OnlineIvectorExtractionInfo &,
                                OnlineFeatureInterface *>(),
                       py::arg("info"),
                       py::arg("base_feature"))
      .def("Dim",
        &PyClass::Dim)
      .def("IsLastFrame",
        &PyClass::IsLastFrame,
                       py::arg("frame"))
      .def("NumFramesReady",
        &PyClass::NumFramesReady)
      .def("FrameShiftInSeconds",
        &PyClass::FrameShiftInSeconds)
      .def("GetFrame",
        &PyClass::GetFrame,
                       py::arg("frame"),
                       py::arg("feat"))
      .def("SetAdaptationState",
        &PyClass::SetAdaptationState,
        "Set the adaptation state to a particular value, e.g. reflecting previous "
        "utterances of the same speaker; this will generally be called after "
        "constructing a new instance of this class.",
                       py::arg("adaptation_state"))
      .def("GetAdaptationState",
        &PyClass::GetAdaptationState,
        "Get the adaptation state; you may want to call this before destroying this "
        "object, to get adaptation state that can be used to improve decoding of "
        "later utterances of this speaker.",
                       py::arg("adaptation_state"))
      .def("UbmLogLikePerFrame",
        &PyClass::UbmLogLikePerFrame)
      .def("ObjfImprPerFrame",
        &PyClass::ObjfImprPerFrame)
      .def("NumFrames",
        &PyClass::NumFrames)
      .def("UpdateFrameWeights",
        &PyClass::UpdateFrameWeights,
        "If you are downweighting silence, you can call "
        "OnlineSilenceWeighting::GetDeltaWeights and supply the output to this class "
        "using UpdateFrameWeights().  The reason why this call happens outside this "
        "class, rather than this class pulling in the data weights, relates to "
        "multi-threaded operation and also from not wanting this class to have "
        "excessive dependencies. "
        "\n"
        "You must either always call this as soon as new data becomes available "
        "(ideally just after calling AcceptWaveform), or never call it for the "
        "lifetime of this object.",
                       py::arg("delta_weights"));
  }
  {
    using PyClass = OnlineSilenceWeightingConfig;

    auto online_silence_weighting_config = py::class_<PyClass>(
        m, "OnlineSilenceWeightingConfig");
    online_silence_weighting_config.def(py::init<>())
      .def(py::init<>())
      .def_readwrite("silence_phones_str", &PyClass::silence_phones_str)
      .def_readwrite("silence_weight", &PyClass::silence_weight)
      .def_readwrite("max_state_duration", &PyClass::max_state_duration)
      .def_readwrite("new_data_weight", &PyClass::new_data_weight)
      .def("Active",
        &PyClass::Active);
  }
  {
    using PyClass = OnlineSilenceWeighting;

    auto online_silence_weighting = py::class_<PyClass>(
        m, "OnlineSilenceWeighting",
        "This class is responsible for keeping track of the best-path traceback from "
        "the decoder (efficiently) and computing a weighting of the data based on the "
        "classification of frames as silence (or not silence)... also with a duration "
        "limitation, so data from a very long run of the same transition-id will get "
        "weighted down.  (this is often associated with misrecognition or silence).");
    online_silence_weighting.def(py::init<const TransitionModel &,
                         const OnlineSilenceWeightingConfig &,
                         int32 >(),
                       py::arg("trans_model"),
                       py::arg("config"),
                       py::arg("frame_subsampling_factor") = 1)
      .def("Active",
        &PyClass::Active)
      .def("GetDeltaWeights",
        py::overload_cast<int32, int32,
      std::vector<std::pair<int32, BaseFloat> > *>(&PyClass::GetDeltaWeights),
        "Calling this function gets the changes in weight that require us to modify "
        "the stats... the output format is (frame-index, delta-weight). "
        "\n"
        "The num_frames_ready argument is the number of frames available at "
        "the input (or equivalently, output) of the online iVector feature in the "
        "feature pipeline from the stream start. It may be more than the currently "
        "available decoder traceback. "
        "\n"
        "The first_decoder_frame is the offset from the start of the stream in "
        "pipeline frames when decoder was restarted last time. We do not change "
        "weight for the frames earlier than first_decoder_frame. Set it to 0 in "
        "case of compilation error to reproduce the previous behavior or for a "
        "single utterance decoding. "
        "\n"
        "How many frames of weights it outputs depends on how much \"num_frames_ready\" "
        "increased since last time we called this function, and whether the decoder "
        "traceback changed.  Negative delta_weights might occur if frames previously "
        "classified as non-silence become classified as silence if the decoder's "
        "traceback changes.  You must call this function with \"num_frames_ready\" "
        "arguments that only increase, not decrease, with time.  You would provide "
        "this output to class OnlineIvectorFeature by calling its function "
        "UpdateFrameWeights with the output. "
        "\n"
        "Returned frame-index is in pipeline frames from the pipeline start.",
                       py::arg("num_frames_ready"),
                       py::arg("first_decoder_frame"),
                       py::arg("delta_weights"))
      .def("GetDeltaWeights",
        py::overload_cast<int32,
      std::vector<std::pair<int32, BaseFloat> > *>(&PyClass::GetDeltaWeights),
        "Calling this function gets the changes in weight that require us to modify "
        "the stats... the output format is (frame-index, delta-weight). "
        "\n"
        "The num_frames_ready argument is the number of frames available at "
        "the input (or equivalently, output) of the online iVector feature in the "
        "feature pipeline from the stream start. It may be more than the currently "
        "available decoder traceback. "
        "\n"
        "The first_decoder_frame is the offset from the start of the stream in "
        "pipeline frames when decoder was restarted last time. We do not change "
        "weight for the frames earlier than first_decoder_frame. Set it to 0 in "
        "case of compilation error to reproduce the previous behavior or for a "
        "single utterance decoding. "
        "\n"
        "How many frames of weights it outputs depends on how much \"num_frames_ready\" "
        "increased since last time we called this function, and whether the decoder "
        "traceback changed.  Negative delta_weights might occur if frames previously "
        "classified as non-silence become classified as silence if the decoder's "
        "traceback changes.  You must call this function with \"num_frames_ready\" "
        "arguments that only increase, not decrease, with time.  You would provide "
        "this output to class OnlineIvectorFeature by calling its function "
        "UpdateFrameWeights with the output. "
        "\n"
        "Returned frame-index is in pipeline frames from the pipeline start.",
                       py::arg("num_frames_ready"),
                       py::arg("delta_weights"))
      .def("GetNonsilenceFrames",
        &PyClass::GetNonsilenceFrames,
        "Gets a list of nonsilence frames collected on traceback. Useful "
        "for algorithms to extract speaker properties like speaker identification "
        "vectors.",
                       py::arg("num_frames_ready"),
                       py::arg("first_decoder_frame"),
                       py::arg("frames"));
  }
}

void pybind_online_nnet2_decoding_threaded(py::module &m) {

  {
    using PyClass = ThreadSynchronizer;

    auto thread_synchronizer = py::class_<PyClass>(
        m, "ThreadSynchronizer",
        "class ThreadSynchronizer acts to guard an arbitrary type of buffer between a "
        "producing and a consuming thread (note: it's all symmetric between the two "
        "thread types).  It has a similar interface to a mutex, except that instead of "
        "just Lock and Unlock, it has Lock, UnlockSuccess and UnlockFailure, and each "
        "function takes an argument kProducer or kConsumer to identify whether the "
        "producing or consuming thread is waiting. "
        "\n"
        "The basic concept is that you lock the object; and if you discover the you're "
        "blocked because you're either trying to read an empty buffer or trying to "
        "write to a full buffer, you unlock with UnlockFailure; and this will cause "
        "your next call to Lock to block until the *other* thread has called Lock and "
        "then UnlockSuccess.  However, if at that point the other thread calls Lock "
        "and then UnlockFailure, it is an error because you can't have both producing "
        "and consuming threads claiming that the buffer is full/empty.  If you lock "
        "the object and were successful you call UnlockSuccess; and you call "
        "UnlockSuccess even if, for your own reasons, you ended up not changing the "
        "state of the buffer.");
    thread_synchronizer.def(py::init<>())
      .def("Lock",
        &PyClass::Lock,
        "call this to lock the object being guarded.",
        py::arg("t"))
      .def("UnlockSuccess",
        &PyClass::UnlockSuccess,
        "Call this to unlock the object being guarded, if you don't want the next call to "
        "Lock to stall.",
        py::arg("t"))
      .def("UnlockFailure",
        &PyClass::UnlockFailure,
        "Call this if you want the next call to Lock() to stall until the other "
        "(producer/consumer) thread has locked and then unlocked the mutex.  Note "
        "that, if the other thread then calls Lock and then UnlockFailure, this will "
        "generate a printed warning (and if repeated too many times, an exception).",
        py::arg("t"))
      .def("SetAbort",
        &PyClass::SetAbort,
        "Sets abort_ flag so future calls will return false, and future calls to "
        "Lock() won't lock the mutex but will immediately return false.");
  py::enum_<ThreadSynchronizer::ThreadType>(thread_synchronizer, "ThreadType")
    .value("kProducer", ThreadSynchronizer::ThreadType::kProducer)
    .value("kConsumer", ThreadSynchronizer::ThreadType::kConsumer)
    .export_values();
  }

  {
    using PyClass = OnlineNnet2DecodingThreadedConfig;

    auto online_nnet2_decoding_threaded_config = py::class_<PyClass>(
        m, "OnlineNnet2DecodingThreadedConfig",
        "This is the configuration class for SingleUtteranceNnet2DecoderThreaded.  The "
        "actual command line program requires other configs that it creates "
        "separately, and which are not included here: namely, "
        "OnlineNnet2FeaturePipelineConfig and OnlineEndpointConfig.");
    online_nnet2_decoding_threaded_config.def(py::init<>())
      .def_readwrite("decoder_opts", &PyClass::decoder_opts)
      .def_readwrite("acoustic_scale", &PyClass::acoustic_scale)
      .def_readwrite("max_buffered_features", &PyClass::max_buffered_features,
          "maximum frames of features we allow to be "
                                "held in the feature buffer before we block "
                                "the feature-processing thread.")
      .def_readwrite("feature_batch_size", &PyClass::feature_batch_size,
          "maximum number of frames at a time that we decode "
                             "before unlocking the mutex.  The only real cost "
                             "here is a mutex lock/unlock, so it's OK to make "
                             "this fairly small.")
      .def_readwrite("max_loglikes_copy", &PyClass::max_loglikes_copy,
          "maximum unused frames of log-likelihoods we will "
                      "copy from the decodable object back into another "
                      "matrix to be supplied to the decodable object. "
                      "make this too large-> will block the "
                      "decoder-search thread while copying; too small "
                      "-> the nnet-evaluation thread may get blocked "
                      "for too long while waiting for the decodable "
                      "thread to be ready.")
      .def_readwrite("nnet_batch_size", &PyClass::nnet_batch_size,
          "batch size (number of frames) we evaluate in the "
                      "neural net, if this many is available.  To take "
                      "best advantage of BLAS, you may want to set this "
                      "fairly large, e.g. 32 or 64 frames.  It probably "
                      "makes sense to tune this a bit.")
      .def_readwrite("decode_batch_size", &PyClass::decode_batch_size,
          "maximum number of frames at a time that we decode "
                            "before unlocking the mutex.  The only real cost "
                            "here is a mutex lock/unlock, so it's OK to make "
                            "this fairly small.")
      .def("Check",
        &PyClass::Check);
  }
  {
    using PyClass = SingleUtteranceNnet2DecoderThreaded;

    auto single_utterance_nnet2_decoder_threaded = py::class_<PyClass>(
        m, "SingleUtteranceNnet2DecoderThreaded",
        "You will instantiate this class when you want to decode a single "
        "utterance using the online-decoding setup for neural nets.  Each time this "
        "class is created, it creates three background threads, and the feature "
        "extraction, neural net evaluation, and search aspects of decoding all "
        "happen in different threads. "
        "Note: we assume that all calls to its public interface happen from a single "
        "thread.");
    single_utterance_nnet2_decoder_threaded.def(py::init<
      const OnlineNnet2DecodingThreadedConfig &,
      const TransitionModel &,
      const nnet2::AmNnet &,
      const fst::Fst<fst::StdArc> &,
      const OnlineNnet2FeaturePipelineInfo &,
      const OnlineIvectorExtractorAdaptationState &,
      const OnlineCmvnState &>(),
                       py::arg("config"),
                       py::arg("tmodel"),
                       py::arg("am_nnet"),
                       py::arg("fst"),
                       py::arg("feature_info"),
                       py::arg("adaptation_state"),
                       py::arg("cmvn_state"))
      .def("AcceptWaveform",
        &PyClass::AcceptWaveform,
        "You call this to provide this class with more waveform to decode.  This "
        "call is, for all practical purposes, non-blocking.",
                       py::arg("samp_freq"),
                       py::arg("wave_part"))
      .def("NumWaveformPiecesPending",
        &PyClass::NumWaveformPiecesPending,
        "Returns the number of pieces of waveform that are still waiting to be "
        "processed.  This may be useful for calling code to judge whether to supply "
        "more waveform or to wait.")
      .def("InputFinished",
        &PyClass::InputFinished,
        "You call this to inform the class that no more waveform will be provided; "
        "this allows it to flush out the last few frames of features, and is "
        "necessary if you want to call Wait() to wait until all decoding is done. "
        "After calling InputFinished() you cannot call AcceptWaveform any more.")
      .def("TerminateDecoding",
        &PyClass::TerminateDecoding,
        "You can call this if you don't want the decoding to proceed further with "
        "this utterance.  It just won't do any more processing, but you can still "
        "use the lattice from the decoding that it's already done.  Note: it may "
        "still continue decoding up to decode_batch_size (default: 2) frames of "
        "data before the decoding thread exits.  You can call Wait() after calling "
        "this, if you want to wait for that.")
      .def("Wait",
        &PyClass::Wait,
        "This call will block until all the data has been decoded; it must only be "
        "called after either InputFinished() has been called or TerminateDecoding() has "
        "been called; otherwise, to call it is an error.")
      .def("FinalizeDecoding",
        &PyClass::FinalizeDecoding,
        "Finalizes the decoding. Cleans up and prunes remaining tokens, so the final "
        "lattice is faster to obtain.  May not be called unless either InputFinished() "
        "or TerminateDecoding() has been called.  If InputFinished() was called, it "
        "calls Wait() to ensure that the decoding has finished (it's not an error "
        "if you already called Wait()).")
      .def("NumFramesReceivedApprox",
        &PyClass::NumFramesReceivedApprox,
        "Returns *approximately* (ignoring end effects), the number of frames of "
        "data that we expect given the amount of data that the pipeline has "
        "received via AcceptWaveform().  (ignores small end effects).  This might "
        "be useful in application code to compare with NumFramesDecoded() and gauge "
        "how much latency there is.")
      .def("NumFramesDecoded",
        &PyClass::NumFramesDecoded,
        "Returns the number of frames currently decoded.  Caution: don't rely on "
        "the lattice having exactly this number if you get it after this call, as "
        "it may increase after this-- unless you've already called either "
        "TerminateDecoding() or InputFinished(), followed by Wait().")
      .def("GetLattice",
        &PyClass::GetLattice,
        "Gets the lattice.  The output lattice has any acoustic scaling in it "
        "(which will typically be desirable in an online-decoding context); if you "
        "want an un-scaled lattice, scale it using ScaleLattice() with the inverse "
        "of the acoustic weight.  \"end_of_utterance\" will be true if you want the "
        "final-probs to be included.  If this is at the end of the utterance, "
        "you might want to first call FinalizeDecoding() first; this will make this "
        "call return faster. "
        "If no frames have been decoded yet, it will set clat to a lattice with "
        "a single state that is final and with unit weight (no cost or alignment). "
        "The output to final_relative_cost (if non-NULL) is a number >= 0 that's "
        "closer to 0 if a final-state was close to the best-likelihood state "
        "active on the last frame, at the time we obtained the lattice.",
                       py::arg("end_of_utterance"),
                       py::arg("clat"),
                       py::arg("final_relative_cost"))
      .def("GetBestPath",
        &PyClass::GetBestPath,
        "Outputs an FST corresponding to the single best path through the current "
        "lattice. If \"use_final_probs\" is true AND we reached the final-state of "
        "the graph then it will include those as final-probs, else it will treat "
        "all final-probs as one. "
        "If no frames have been decoded yet, it will set best_path to a lattice with "
        "a single state that is final and with unit weight (no cost). "
        "The output to final_relative_cost (if non-NULL) is a number >= 0 that's "
        "closer to 0 if a final-state were close to the best-likelihood state "
        "active on the last frame, at the time we got the best path.",
                       py::arg("end_of_utterance"),
                       py::arg("best_path"),
                       py::arg("final_relative_cost"))
      .def("EndpointDetected",
        &PyClass::EndpointDetected,
        "This function calls EndpointDetected from online-endpoint.h, "
        "with the required arguments.",
                       py::arg("config"))
      .def("GetAdaptationState",
        &PyClass::GetAdaptationState,
        "Outputs the adaptation state of the feature pipeline to \"adaptation_state\".  This "
        "mostly stores stats for iVector estimation, and will generally be called at the "
        "end of an utterance, assuming it's a scenario where each speaker is seen for "
        "more than one utterance. "
        "You may only call this function after either calling TerminateDecoding() or "
        "InputFinished, and then Wait().  Otherwise it is an error.",
                       py::arg("adaptation_state"))
      .def("GetCmvnState",
        &PyClass::GetCmvnState,
        "Outputs the OnlineCmvnState of the feature pipeline to \"cmvn_stat\".  This "
        "stores cmvn stats for the non-iVector features, and will be called at the "
        "end of an utterance, assuming it's a scenario where each speaker is seen for "
        "more than one utterance. "
        "You may only call this function after either calling TerminateDecoding() or "
        "InputFinished, and then Wait().  Otherwise it is an error.",
                       py::arg("cmvn_state"))
      .def("GetRemainingWaveform",
        &PyClass::GetRemainingWaveform,
        "Gets the remaining, un-decoded part of the waveform and returns the sample "
        "rate.  May only be called after Wait(), and it only makes sense to call "
        "this if you called TerminateDecoding() before Wait().  The idea is that "
        "you can then provide this un-decoded piece of waveform to another decoder.",
                       py::arg("waveform_out"));
  }
}

void pybind_online_nnet2_decoding(py::module &m) {

  {
    using PyClass = OnlineNnet2DecodingConfig;

    auto online_nnet2_decoding_config = py::class_<PyClass>(
        m, "OnlineNnet2DecodingConfig");
    online_nnet2_decoding_config.def(py::init<>())
      .def_readwrite("decoder_opts", &PyClass::decodable_opts)
      .def_readwrite("decodable_opts", &PyClass::decodable_opts);
  }
  {
    using PyClass = SingleUtteranceNnet2Decoder;

    auto single_utterance_nnet2_decoder = py::class_<PyClass>(
        m, "SingleUtteranceNnet2Decoder");
    single_utterance_nnet2_decoder
      .def(py::init<const OnlineNnet2DecodingConfig &,
                              const TransitionModel &,
                              const nnet2::AmNnet &,
                              const fst::Fst<fst::StdArc> &,
                              OnlineFeatureInterface *>(),
                       py::arg("config"),
                       py::arg("tmodel"),
                       py::arg("model"),
                       py::arg("fst"),
                       py::arg("feature_pipeline"))
      .def("AdvanceDecoding",
        &PyClass::AdvanceDecoding,
        "advance the decoding as far as we can.")
      .def("FinalizeDecoding",
        &PyClass::FinalizeDecoding,
        "Finalizes the decoding. Cleans up and prunes remaining tokens, so the "
        "GetLattice() call will return faster.  You must not call this before "
        "calling (TerminateDecoding() or InputIsFinished()) and then Wait().")
      .def("NumFramesDecoded",
        &PyClass::NumFramesDecoded)
      .def("GetLattice",
        &PyClass::GetLattice,
        "Gets the lattice.  The output lattice has any acoustic scaling in it "
        "(which will typically be desirable in an online-decoding context); if you "
        "want an un-scaled lattice, scale it using ScaleLattice() with the inverse "
        "of the acoustic weight.  \"end_of_utterance\" will be true if you want the "
        "final-probs to be included.",
                       py::arg("end_of_utterance"),
                       py::arg("clat"))
      .def("GetBestPath",
        &PyClass::GetBestPath,
        "Outputs an FST corresponding to the single best path through the current "
        "lattice. If \"use_final_probs\" is true AND we reached the final-state of "
        "the graph then it will include those as final-probs, else it will treat "
        "all final-probs as one.",
                       py::arg("end_of_utterance"),
                       py::arg("best_path"))
      .def("EndpointDetected",
        &PyClass::EndpointDetected,
        "This function calls EndpointDetected from online-endpoint.h, "
        "with the required arguments.",
                       py::arg("config"))
      .def("Decoder",
        &PyClass::Decoder);
  }

}

void pybind_online_nnet2_feature_pipeline(py::module &m) {

  {
    using PyClass = OnlineNnet2FeaturePipelineConfig;

    auto online_nnet2_feature_pipeline_config = py::class_<PyClass>(
        m, "OnlineNnet2FeaturePipelineConfig");
    online_nnet2_feature_pipeline_config.def(py::init<>())
      .def_readwrite("feature_type", &PyClass::feature_type,
            "\"plp\" or \"mfcc\" or \"fbank\"")
      .def_readwrite("mfcc_config", &PyClass::mfcc_config)
      .def_readwrite("plp_config", &PyClass::plp_config)
      .def_readwrite("fbank_config", &PyClass::fbank_config)
      .def_readwrite("cmvn_config", &PyClass::cmvn_config)
      .def_readwrite("global_cmvn_stats_rxfilename", &PyClass::global_cmvn_stats_rxfilename)
      .def_readwrite("add_pitch", &PyClass::add_pitch,
            "Note: if we do add pitch, it will not be added to the features we give to "
            "the iVector extractor but only to the features we give to the neural "
            "network, after the base features but before the iVector.  We don't think "
            "the iVector will be particularly helpful in normalizing the pitch features.")
      .def_readwrite("online_pitch_config", &PyClass::online_pitch_config,
            "the following contains the type of options that you could give to "
            "compute-and-process-kaldi-pitch-feats.")
      .def_readwrite("ivector_extraction_config", &PyClass::ivector_extraction_config,
            "The configuration variables in ivector_extraction_config relate to the "
            "iVector extractor and options related to it, see type "
            "OnlineIvectorExtractionConfig.")
      .def_readwrite("silence_weighting_config", &PyClass::silence_weighting_config,
          "Config that relates to how we weight silence for (ivector) adaptation "
        "this is registered directly to the command line as you might want to "
        "play with it in test time.");
  }
  {
    using PyClass = OnlineNnet2FeaturePipelineInfo;

    auto online_nnet2_feature_pipeline_info = py::class_<PyClass>(
        m, "OnlineNnet2FeaturePipelineInfo",
          "This class is responsible for storing configuration variables, objects and "
"options for OnlineNnet2FeaturePipeline (including the actual LDA and "
"CMVN-stats matrices, and the iVector extractor, which is a member of "
"ivector_extractor_info.  This class does not register options on the command "
"line; instead, it is initialized from class OnlineNnet2FeaturePipelineConfig "
"which reads the options from the command line.  The reason for structuring "
"it this way is to make it easier to configure from code as well as from the "
"command line, as well as for easier multithreaded operation.");
    online_nnet2_feature_pipeline_info.def(py::init<>())
      .def(py::init<const OnlineNnet2FeaturePipelineConfig &>(),
                       py::arg("config"))
      .def_readwrite("feature_type", &PyClass::feature_type,
            "\"mfcc\" or \"plp\" or \"fbank\"")
      .def_readwrite("mfcc_opts", &PyClass::mfcc_opts,
            "options for MFCC computation, if feature_type == \"mfcc\"")
      .def_readwrite("plp_opts", &PyClass::plp_opts,
            "Options for PLP computation, if feature_type == \"plp\"")
      .def_readwrite("fbank_opts", &PyClass::fbank_opts,
            "Options for filterbank computation, if feature_type == \"fbank\"")
      .def_readwrite("add_pitch", &PyClass::add_pitch)
      .def_readwrite("pitch_opts", &PyClass::pitch_opts,
            "Options for pitch extraction, if done.")
      .def_readwrite("pitch_process_opts", &PyClass::pitch_process_opts,
          "Options for pitch post-processing")
      .def_readwrite("use_cmvn", &PyClass::use_cmvn,
            "If the user specified --cmvn-config, we set 'use_cmvn' to true, "
          "and the OnlineCmvn is added to the feature preparation pipeline.")
      .def_readwrite("cmvn_opts", &PyClass::cmvn_opts,
            "Options for online cmvn, read from config file.")
      .def_readwrite("global_cmvn_stats", &PyClass::global_cmvn_stats,
            "Matrix with global cmvn stats in OnlineCmvn.")
      .def_readwrite("use_ivectors", &PyClass::use_ivectors,
            "If the user specified --ivector-extraction-config, we assume we're using "
          "iVectors as an extra input to the neural net.  Actually, we don't "
          "anticipate running this setup without iVectors.")
      .def_readonly("ivector_extractor_info", &PyClass::ivector_extractor_info)
      .def_readwrite("silence_weighting_config", &PyClass::silence_weighting_config,
            "Config for weighting silence in iVector adaptation. "
            "We declare this outside of ivector_extractor_info... it was "
            "just easier to set up the code that way; and also we think "
            "it's the kind of thing you might want to play with directly "
            "on the command line instead of inside sub-config-files.")
      .def("GetSamplingFrequency",
        &PyClass::GetSamplingFrequency,
        "Returns the frequency expected by the model")
      .def("IvectorDim",
        &PyClass::IvectorDim);
  }
  {
    using PyClass = OnlineNnet2FeaturePipeline;

    auto online_nnet2_feature_pipeline = py::class_<OnlineNnet2FeaturePipeline, OnlineFeatureInterface>(
        m, "OnlineNnet2FeaturePipeline");
    online_nnet2_feature_pipeline.def(py::init<const OnlineNnet2FeaturePipelineInfo &>(),
                       py::arg("info"))
      .def("Dim",
        &PyClass::Dim,
        "Dim() will return the base-feature dimension (e.g. 13 for normal MFCC); "
        "plus the pitch-feature dimension (e.g. 3), if used; plus the iVector "
        "dimension, if used.  Any frame-splicing happens inside the neural-network "
        "code.")
      .def("IsLastFrame",
        &PyClass::IsLastFrame,
                       py::arg("info"))
      .def("NumFramesReady",
        &PyClass::NumFramesReady)
      .def("GetFrame",
        &PyClass::GetFrame,
                       py::arg("frame"),
                       py::arg("feat"))
      .def("UpdateFrameWeights",
        &PyClass::UpdateFrameWeights,
        "If you are downweighting silence, you can call "
          "OnlineSilenceWeighting::GetDeltaWeights and supply the output to this "
          "class using UpdateFrameWeights().  The reason why this call happens "
          "outside this class, rather than this class pulling in the data weights, "
          "relates to multi-threaded operation and also from not wanting this class "
          "to have excessive dependencies. "
          "\n"
          "You must either always call this as soon as new data becomes available, "
          "ideally just after calling AcceptWaveform(), or never call it for the "
          "lifetime of this object.",
                       py::arg("delta_weights"))
      .def("SetAdaptationState",
        &PyClass::SetAdaptationState,
        "Set the adaptation state to a particular value, e.g. reflecting previous "
        "utterances of the same speaker; this will generally be called after "
        "Copy().",
                       py::arg("adaptation_state"))
      .def("GetAdaptationState",
        &PyClass::GetAdaptationState,
        "Get the adaptation state; you may want to call this before destroying this "
        "object, to get adaptation state that can be used to improve decoding of "
        "later utterances of this speaker.  You might not want to do this, though, "
        "if you have reason to believe that something went wrong in the recognition "
        "(e.g., low confidence).",
                       py::arg("adaptation_state"))
      .def("SetCmvnState",
        &PyClass::SetCmvnState,
        "Set the CMVN state to a particular value. "
  "(for features on nnet3 input, not the i-vector input).",
                       py::arg("cmvn_state"))
      .def("GetCmvnState",
        &PyClass::GetCmvnState,
                       py::arg("cmvn_state"))
      .def("AcceptWaveform",
        &PyClass::AcceptWaveform,
        "Accept more data to process.  It won't actually process it until you call "
  "GetFrame() [probably indirectly via (decoder).AdvanceDecoding()], when you "
  "call this function it will just copy it).  sampling_rate is necessary just "
  "to assert it equals what's in the config.",
                       py::arg("sampling_rate"),
                       py::arg("waveform"))
      .def("FrameShiftInSeconds",
        &PyClass::FrameShiftInSeconds)
      .def("InputFinished",
        &PyClass::InputFinished,
        "If you call InputFinished(), it tells the class you won't be providing any "
  "more waveform.  This will help flush out the last few frames of delta or "
  "LDA features, and finalize the pitch features (making them more "
  "accurate)... although since in neural-net decoding we don't anticipate "
  "rescoring the lattices, this may not be much of an issue.")
      .def("IvectorFeature",
        py::overload_cast<>(&PyClass::IvectorFeature),
        "This function returns the iVector-extracting part of the feature pipeline "
  "(or NULL if iVectors are not being used); the pointer ownership is retained "
  "by this object and not transferred to the caller.  This function is used in "
  "nnet3, and also in the silence-weighting code used to exclude silence from "
  "the iVector estimation.")
      .def("InputFeature",
        &PyClass::InputFeature,
        "This function returns the part of the feature pipeline that would be given "
        "as the primary (non-iVector) input to the neural network in nnet3 "
        "applications.");
  }
}

void pybind_online_nnet3_decoding(py::module &m) {

  {
    using PyClass = SingleUtteranceNnet3Decoder;

    auto single_utterance_nnet3_decoder = py::class_<PyClass>(
        m, "SingleUtteranceNnet3Decoder");
    single_utterance_nnet3_decoder.def(py::init<const LatticeFasterDecoderConfig &,
                                 const TransitionModel &,
                                 const nnet3::DecodableNnetSimpleLoopedInfo &,
                                 const fst::Fst<fst::StdArc> &,
                                 OnlineNnet2FeaturePipeline *>(),
                       py::arg("decoder_opts"),
                       py::arg("trans_model"),
                       py::arg("info"),
                       py::arg("fst"),
                       py::arg("features"))
      .def("InitDecoding",
        &PyClass::InitDecoding,
        "Initializes the decoding and sets the frame offset of the underlying "
  "decodable object. This method is called by the constructor. You can also "
  "call this method when you want to reset the decoder state, but want to "
  "keep using the same decodable object, e.g. in case of an endpoint.",
                       py::arg("frame_offset") = 0)
      .def("AdvanceDecoding",
        &PyClass::AdvanceDecoding,
        "Advances the decoding as far as we can.")
      .def("FinalizeDecoding",
        &PyClass::FinalizeDecoding,
        "Finalizes the decoding. Cleans up and prunes remaining tokens, so the "
  "GetLattice() call will return faster.  You must not call this before "
  "calling (TerminateDecoding() or InputIsFinished()) and then Wait().")
      .def("NumFramesDecoded",
        &PyClass::NumFramesDecoded)
      .def("GetLattice",
        &PyClass::GetLattice,
        "Gets the lattice.  The output lattice has any acoustic scaling in it "
  "(which will typically be desirable in an online-decoding context); if you "
  "want an un-scaled lattice, scale it using ScaleLattice() with the inverse "
  "of the acoustic weight.  \"end_of_utterance\" will be true if you want the "
  "final-probs to be included.",
                       py::arg("end_of_utterance"),
                       py::arg("clat"))
      .def("GetBestPath",
        &PyClass::GetBestPath,
        "Outputs an FST corresponding to the single best path through the current "
  "lattice. If \"use_final_probs\" is true AND we reached the final-state of "
  "the graph then it will include those as final-probs, else it will treat "
  "all final-probs as one.",
                       py::arg("end_of_utterance"),
                       py::arg("best_path"))
      .def("EndpointDetected",
        &PyClass::EndpointDetected,
        "This function calls EndpointDetected from online-endpoint.h,"
  "with the required arguments.",
                       py::arg("config"))
      .def("Decoder",
        &PyClass::Decoder);
  }
}

void pybind_online_nnet3_incremental_decoding(py::module &m) {

  {
    using PyClass = SingleUtteranceNnet3IncrementalDecoder;

    auto single_utterance_nnet3_incremental_decoder = py::class_<PyClass>(
        m, "SingleUtteranceNnet3IncrementalDecoder");
    single_utterance_nnet3_incremental_decoder.def(py::init<const LatticeIncrementalDecoderConfig &,
                                 const TransitionModel &,
                                 const nnet3::DecodableNnetSimpleLoopedInfo &,
                                 const fst::Fst<fst::StdArc> &,
                                 OnlineNnet2FeaturePipeline *>(),
                       py::arg("decoder_opts"),
                       py::arg("trans_model"),
                       py::arg("info"),
                       py::arg("fst"),
                       py::arg("features"))
      .def("InitDecoding",
        &PyClass::InitDecoding,
        "Initializes the decoding and sets the frame offset of the underlying "
  "decodable object. This method is called by the constructor. You can also "
  "call this method when you want to reset the decoder state, but want to "
  "keep using the same decodable object, e.g. in case of an endpoint.",
                       py::arg("frame_offset") = 0)
      .def("AdvanceDecoding",
        &PyClass::AdvanceDecoding,
        "Advances the decoding as far as we can.")
      .def("FinalizeDecoding",
        &PyClass::FinalizeDecoding,
        "Finalizes the decoding. Cleans up and prunes remaining tokens, so the "
  "GetLattice() call will return faster.  You must not call this before "
  "calling (TerminateDecoding() or InputIsFinished()) and then Wait().")
      .def("NumFramesDecoded",
        &PyClass::NumFramesDecoded)
      .def("NumFramesInLattice",
        &PyClass::NumFramesInLattice)
      .def("GetLattice",
        &PyClass::GetLattice,
        "Gets the lattice.  The output lattice has any acoustic scaling in it "
        "(which will typically be desirable in an online-decoding context); if you "
        "want an un-scaled lattice, scale it using ScaleLattice() with the inverse "
        "of the acoustic weight. "
        "\n"
        "    @param [in] num_frames_to_include  The number of frames you want "
        "              to be included in the lattice.  Must be in the range "
        "              [NumFramesInLattice().. NumFramesDecoded()].  If you "
        "              make it a few frames less than NumFramesDecoded(), it "
        "              will save significant computation. "
        "    @param [in] use_final_probs   True if you want the lattice to "
        "              contain final-probs (if at least one state was final "
        "              on the most recently decoded frame).  Must be false "
        "              if num_frames_to_include < NumFramesDecoded(). "
        "              Must be true if you have previously called "
        "              FinalizeDecoding().",
                       py::arg("num_frames_to_include"),
                       py::arg("use_final_probs") = false)
      .def("GetBestPath",
        &PyClass::GetBestPath,
        "Outputs an FST corresponding to the single best path through the current "
  "lattice. If \"use_final_probs\" is true AND we reached the final-state of "
  "the graph then it will include those as final-probs, else it will treat "
  "all final-probs as one.",
                       py::arg("end_of_utterance"),
                       py::arg("best_path"))
      .def("EndpointDetected",
        &PyClass::EndpointDetected,
        "This function calls EndpointDetected from online-endpoint.h,"
  "with the required arguments.",
                       py::arg("config"))
      .def("Decoder",
        &PyClass::Decoder);
  }
}

void pybind_online_nnet3_wake_word_faster_decoder(py::module &m) {

  {
    using PyClass = OnlineWakeWordFasterDecoderOpts;

    py::class_<OnlineWakeWordFasterDecoderOpts, FasterDecoderOptions>(
        m, "OnlineWakeWordFasterDecoderOpts");
  }
  {
    using PyClass = OnlineWakeWordFasterDecoder;

    auto online_wake_word_faster_decoder = py::class_<PyClass>(
        m, "OnlineWakeWordFasterDecoder");
    online_wake_word_faster_decoder.def(py::init<const fst::Fst<fst::StdArc> &,
                              const OnlineWakeWordFasterDecoderOpts &>(),
                       py::arg("fst"),
                       py::arg("opts"))
      .def("PartialTraceback",
        &PyClass::PartialTraceback,
        "Makes a linear graph, by tracing back from the last \"immortal\" token "
        "to the previous one",
                       py::arg("out_fst"))
      .def("FinishTraceBack",
        &PyClass::FinishTraceBack,
        "Makes a linear graph, by tracing back from the best currently active token "
        "to the last immortal token. This method is meant to be invoked at the end "
        "of an utterance in order to get the last chunk of the hypothesis",
                       py::arg("fst_out"))
      .def("InitDecoding",
        &PyClass::InitDecoding,
        "As a new alternative to Decode(), you can call InitDecoding "
          "and then (possibly multiple times) AdvanceDecoding().");
  }
}

void pybind_online_speex_wrapper(py::module &m) {

  {
    using PyClass = SpeexOptions;

    auto speex_options = py::class_<PyClass>(
        m, "SpeexOptions");
    speex_options.def(py::init<>())
      .def_readwrite("sample_rate", &PyClass::sample_rate,
                      "The sample frequency of the waveform, it decides which Speex mode "
                    "to use. Often 8kHz---> narrow band, 16kHz---> wide band and 32kHz "
                    "---> ultra wide band")
      .def_readwrite("speex_quality", &PyClass::speex_quality,
                      "Ranges from 0 to 10, the higher the quality is better. In my preliminary "
                    "tests with the RM recipe, if set it to 8, I observed the WER increased by "
                    "0.1%; while set it to 10, the WER almost kept unchanged.")
      .def_readwrite("speex_bits_frame_size", &PyClass::speex_bits_frame_size,
                      "In bytes. "
                      "Should be set according to speex_quality. Just name a few here(wideband): "
                      "    quality            size(in bytes) "
                      "       8                  70 "
                      "       9                  86 "
                      "       10                 106")
      .def_readwrite("speex_wave_frame_size", &PyClass::speex_wave_frame_size,
                      "In samples. The Speex toolkit uses a 20ms long window by default");
  }
  {
    using PyClass = OnlineSpeexEncoder;

    auto online_speex_encoder = py::class_<PyClass>(
        m, "OnlineSpeexEncoder");
    online_speex_encoder
      .def(py::init<const SpeexOptions &>(),
        py::arg("config"))
      .def("AcceptWaveform", &PyClass::AcceptWaveform,
        py::arg("sample_rate"),
        py::arg("waveform"))
      .def("InputFinished", &PyClass::InputFinished)
      .def("GetSpeexBits", &PyClass::GetSpeexBits,
        py::arg("spx_bits"));
  }
  {
    using PyClass = OnlineSpeexDecoder;

    auto online_speex_decoder = py::class_<PyClass>(
        m, "OnlineSpeexDecoder");
    online_speex_decoder
      .def(py::init<const SpeexOptions &>(),
        py::arg("config"))
      .def("AcceptSpeexBits", &PyClass::AcceptSpeexBits,
        py::arg("spx_enc_bits"))
      .def("GetWaveform", &PyClass::GetWaveform,
        py::arg("waveform"));
  }
}

void pybind_online_timing(py::module &m) {

  {
    using PyClass = OnlineTimingStats;

    auto online_timing_stats = py::class_<PyClass>(
        m, "OnlineTimingStats",
        "class OnlineTimingStats stores statistics from timing of online decoding, "
"which will enable the Print() function to print out the average real-time "
"factor and average delay per utterance.  See class OnlineTimer.");
    online_timing_stats
      .def(py::init<>())
      .def("Print", &PyClass::Print,
        py::arg("online") = true);
  }
  {
    using PyClass = OnlineTimer;

    auto online_timer = py::class_<PyClass>(
        m, "OnlineTimer",
        "class OnlineTimer is used to test real-time decoding algorithms and evaluate "
        "how long the decoding of a particular utterance would take.  The 'obvious' "
        "way to evaluate this would be to measure the wall-clock time, and if we're "
        "processing the data in chunks, to sleep() until a given chunk would become "
        "available in a real-time application-- e.g. say we need to process a chunk "
        "that ends half a second into the utterance, we would sleep until half a "
        "second had elapsed since the start of the utterance.  In this code we "
        "have the option to not actually sleep: we can simulate the effect of "
        "sleeping by just incrementing "
        "a variable that says how long we would have slept; and we add this to "
        "wall-clock times obtained from Timer::Elapsed(). "
        "The usage of this class will be something like as follows: "
        "\\code "
        "OnlineTimingStats stats; "
        "while (.. process different utterances..) { "
        "  OnlineTimer this_utt_timer(utterance_id); "
        "  while (...process chunks of this utterance..) { "
        "     double num_secs_elapsed = 0.01 * num_frames; "
        "     this_utt_timer.WaitUntil(num_secs_elapsed); "
        "  } "
        "  this_utt_timer.OutputStats(&stats); "
        "\\endcode "
        "This assumes that the value provided to the last WaitUntil() "
        "call was the length of the utterance.");
    online_timer
      .def(py::init<const std::string &>(),
        py::arg("utterance_id"))
      .def("SleepUntil", &PyClass::SleepUntil,
        "The call to SleepUntil(t) will sleep until cur_utterance_length seconds "
      "after this object was initialized, or return immediately if we've "
      "already passed that time." ,
        py::arg("cur_utterance_length"))
      .def("WaitUntil", &PyClass::WaitUntil,
        "The call to WaitUntil(t) simulates the effect of sleeping until "
  "cur_utterance_length seconds after this object was initialized;  "
  "but instead of actually sleeping, it increases a counter." ,
        py::arg("cur_utterance_length"))
      .def("OutputStats", &PyClass::OutputStats,
        "This call, which should be made after decoding is done, "
  "writes the stats to the object that accumulates them.",
        py::arg("stats"))
      .def("Elapsed", &PyClass::Elapsed,
        "Returns the simulated time elapsed in seconds since the timer was started; "
  "this equals waited_ plus the real time elapsed.");
  }
}

void pybind_online2bin_util(py::module &m) {

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

void init_online2(py::module &_m) {
  py::module m = _m.def_submodule("online2", "online2 pybind for Kaldi");

  pybind_online_endpoint(m);
  pybind_online_feature_pipeline(m);
  pybind_online_gmm_decodable(m);
  pybind_online_gmm_decoding(m);
  pybind_online_ivector_feature(m);
  pybind_online_nnet2_decoding_threaded(m);
  pybind_online_nnet2_decoding(m);
  pybind_online_nnet2_feature_pipeline(m);
  pybind_online_nnet3_decoding(m);
  pybind_online_nnet3_incremental_decoding(m);
  pybind_online_nnet3_wake_word_faster_decoder(m);
  pybind_online_speex_wrapper(m);
  pybind_online_timing(m);
  pybind_online2bin_util(m);
}
