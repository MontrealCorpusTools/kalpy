
#include "feat/pybind_feat.h"

#include "feat/feature-mfcc.h"
#include "feat/feature-spectrogram.h"
#include "feat/feature-plp.h"
#include "feat/pitch-functions.h"
#include "feat/feature-fbank.h"
#include "feat/online-feature.h"
#include "feat/wave-reader.h"
#include "feat/signal.h"
#include "util/pybind_util.h"

using namespace kaldi;


template <class Feature>
void offline_feature(py::module& m, const std::string& feat_type) {
  py::class_<OfflineFeatureTpl<Feature>>(m, feat_type.c_str())
      .def(py::init<const typename Feature::Options&>())
      .def("ComputeFeatures", &OfflineFeatureTpl<Feature>::ComputeFeatures)
      .def("compute", [](OfflineFeatureTpl<Feature>& v, py::array_t<float> x){
        auto r = x.unchecked<1>();
        auto vector = Vector<float>(r.shape(0));
        float vtln_warp = 1.0;
        Matrix<float> features;
        for (py::size_t i = 0; i < r.shape(0); i++)
          vector(i) = r(i);
        v.Compute(vector, vtln_warp, &features);
        return features;
      })
      .def("Dim", &OfflineFeatureTpl<Feature>::Dim);
}


void feat_signal(py::module& m){

  m.def("ConvolveSignals",
        &ConvolveSignals,
        "This function implements a simple non-FFT-based convolution of two signals. "
        "It is suggested to use the FFT-based convolution function which is more "
        "efficient.",
        py::arg("filter"),
        py::arg("signal"));

  m.def("FFTbasedConvolveSignals",
        &FFTbasedConvolveSignals,
        "This function implements FFT-based convolution of two signals. "
        "However this should be an inefficient version of BlockConvolveSignals() "
        "as it processes the entire signal with a single FFT.",
        py::arg("filter"),
        py::arg("signal"));

  m.def("FFTbasedBlockConvolveSignals",
        &FFTbasedBlockConvolveSignals,
        "This function implements FFT-based block convolution of two signals using "
        "overlap-add method. This is an efficient way to evaluate the discrete "
        "convolution of a long signal with a finite impulse response filter.",
        py::arg("filter"),
        py::arg("signal"));
}

void feat_pitch_functions(py::module& m){


  py::class_<PitchExtractionOptions>(m, "PitchExtractionOptions")
      .def(py::init<>())
      .def_readwrite("samp_freq", &PitchExtractionOptions::samp_freq)
      .def_readwrite("frame_shift_ms", &PitchExtractionOptions::frame_shift_ms)
      .def_readwrite("frame_length_ms", &PitchExtractionOptions::frame_length_ms)
      .def_readwrite("preemph_coeff", &PitchExtractionOptions::preemph_coeff)
      .def_readwrite("min_f0", &PitchExtractionOptions::min_f0)
      .def_readwrite("max_f0", &PitchExtractionOptions::max_f0)
      .def_readwrite("soft_min_f0", &PitchExtractionOptions::soft_min_f0)
      .def_readwrite("penalty_factor", &PitchExtractionOptions::penalty_factor)
      .def_readwrite("resample_freq", &PitchExtractionOptions::resample_freq)
      .def_readwrite("delta_pitch", &PitchExtractionOptions::delta_pitch)
      .def_readwrite("nccf_ballast", &PitchExtractionOptions::nccf_ballast)
      .def_readwrite("lowpass_filter_width", &PitchExtractionOptions::lowpass_filter_width)
      .def_readwrite("upsample_filter_width", &PitchExtractionOptions::upsample_filter_width)
      .def_readwrite("max_frames_latency", &PitchExtractionOptions::max_frames_latency)
      .def_readwrite("frames_per_chunk", &PitchExtractionOptions::frames_per_chunk)
      .def_readwrite("simulate_first_pass_online", &PitchExtractionOptions::simulate_first_pass_online)
      .def_readwrite("recompute_frame", &PitchExtractionOptions::recompute_frame)
      .def_readwrite("nccf_ballast_online", &PitchExtractionOptions::nccf_ballast_online)
      .def_readwrite("snip_edges", &PitchExtractionOptions::snip_edges)
      .def("NccfWindowSize", &PitchExtractionOptions::NccfWindowSize)
      .def("NccfWindowShift", &PitchExtractionOptions::NccfWindowShift);

  py::class_<ProcessPitchOptions>(m, "ProcessPitchOptions")
      .def(py::init<>())
      .def_readwrite("pitch_scale", &ProcessPitchOptions::pitch_scale)
      .def_readwrite("pov_scale", &ProcessPitchOptions::pov_scale)
      .def_readwrite("pov_offset", &ProcessPitchOptions::pov_offset)
      .def_readwrite("delta_pitch_scale", &ProcessPitchOptions::delta_pitch_scale)
      .def_readwrite("delta_pitch_noise_stddev", &ProcessPitchOptions::delta_pitch_noise_stddev)
      .def_readwrite("normalization_left_context", &ProcessPitchOptions::normalization_left_context)
      .def_readwrite("normalization_right_context", &ProcessPitchOptions::normalization_right_context)
      .def_readwrite("delta_window", &ProcessPitchOptions::delta_window)
      .def_readwrite("delay", &ProcessPitchOptions::delay)
      .def_readwrite("add_pov_feature", &ProcessPitchOptions::add_pov_feature)
      .def_readwrite("add_normalized_log_pitch", &ProcessPitchOptions::add_normalized_log_pitch)
      .def_readwrite("add_delta_pitch", &ProcessPitchOptions::add_delta_pitch)
      .def_readwrite("add_raw_log_pitch", &ProcessPitchOptions::add_raw_log_pitch);
    {

      using PyClass = OnlinePitchFeature;
      auto online_pitch_feature = py::class_<PyClass>(
          m, "OnlinePitchFeature");
      online_pitch_feature.def(py::init<const PitchExtractionOptions &>(), py::arg("opts"))
        .def("Dim", &PyClass::Dim)
        .def("NumFramesReady", &PyClass::NumFramesReady)
        .def("FrameShiftInSeconds", &PyClass::FrameShiftInSeconds)
        .def("IsLastFrame", &PyClass::IsLastFrame, py::arg("frame"))
        .def("GetFrame", &PyClass::GetFrame, py::arg("frame"), py::arg("feat"))
        .def("AcceptWaveform", &PyClass::AcceptWaveform, py::arg("sampling_rate"), py::arg("waveform"))
        .def("InputFinished", &PyClass::InputFinished);
    }
    {

      using PyClass = OnlineProcessPitch;
      auto online_process_pitch = py::class_<PyClass, OnlineFeatureInterface>(
          m, "OnlineProcessPitch");
      online_process_pitch.def(py::init<const ProcessPitchOptions &, OnlineFeatureInterface *>(),
                    py::arg("opts"),
                    py::arg("src"))
        .def("Dim", &PyClass::Dim)
        .def("IsLastFrame", &PyClass::IsLastFrame, py::arg("frame"))
        .def("FrameShiftInSeconds", &PyClass::FrameShiftInSeconds)
        .def("NumFramesReady", &PyClass::NumFramesReady)
        .def("GetFrame", &PyClass::GetFrame, py::arg("frame"), py::arg("feat"));
    }
  m.def("ComputeKaldiPitch",
        &ComputeKaldiPitch,
        "This function extracts (pitch, NCCF) per frame, using the pitch extraction "
        "method described in \"A Pitch Extraction Algorithm Tuned for Automatic Speech "
        "Recognition\", Pegah Ghahremani, Bagher BabaAli, Daniel Povey, Korbinian "
        "Riedhammer, Jan Trmal and Sanjeev Khudanpur, ICASSP 2014.  The output will "
        "have as many rows as there are frames, and two columns corresponding to "
        "(NCCF, pitch)",
        py::arg("opts"),
        py::arg("wave"),
        py::arg("output"));
  m.def("ProcessPitch",
        &ProcessPitch,
        "This function processes the raw (NCCF, pitch) quantities computed by "
        "ComputeKaldiPitch, and processes them into features.  By default it will "
        "output three-dimensional features, (POV-feature, mean-subtracted-log-pitch, "
        "delta-of-raw-pitch), but this is configurable in the options.  The number of "
        "rows of \"output\" will be the number of frames (rows) in \"input\", and the "
        "number of columns will be the number of different types of features "
        "requested (by default, 3; 4 is the max).  The four config variables "
        "--add-pov-feature, --add-normalized-log-pitch, --add-delta-pitch, "
        "--add-raw-log-pitch determine which features we create; by default we create "
        "the first three.",
        py::arg("opts"),
        py::arg("input"),
        py::arg("output"));
  m.def("ComputeAndProcessKaldiPitch",
        &ComputeAndProcessKaldiPitch,
        "This function combines ComputeKaldiPitch and ProcessPitch.  The reason "
        "why we need a separate function to do this is in order to be able to "
        "accurately simulate the online pitch-processing, for testing and for "
        "training models matched to the \"first-pass\" features.  It is sensitive to "
        "the variables in pitch_opts that relate to online processing, "
        "i.e. max_frames_latency, frames_per_chunk, simulate_first_pass_online, "
        "recompute_frame.",
        py::arg("pitch_opts"),
        py::arg("process_opts"),
        py::arg("wave"),
        py::arg("output"));
}

template <class Feature>
void online_base_feature(py::module& m, const std::string& feat_type) {
  py::class_<OnlineGenericBaseFeature<Feature>, OnlineBaseFeature>(m, feat_type.c_str())
      .def(py::init<const typename Feature::Options&>(), py::arg("opts"));
}

void init_feat(py::module &_m) {
  py::module m = _m.def_submodule("feat", "feat pybind for Kaldi");
  py::class_<FrameExtractionOptions>(m, "FrameExtractionOptions")
      .def(py::init<>())
      .def_readwrite("samp_freq", &FrameExtractionOptions::samp_freq)
      .def_readwrite("frame_shift_ms", &FrameExtractionOptions::frame_shift_ms)
      .def_readwrite("frame_length_ms", &FrameExtractionOptions::frame_length_ms)
      .def_readwrite("dither", &FrameExtractionOptions::dither)
      .def_readwrite("preemph_coeff", &FrameExtractionOptions::preemph_coeff)
      .def_readwrite("remove_dc_offset", &FrameExtractionOptions::remove_dc_offset)
      .def_readwrite("window_type", &FrameExtractionOptions::window_type)
      .def_readwrite("round_to_power_of_two", &FrameExtractionOptions::round_to_power_of_two)
      .def_readwrite("blackman_coeff", &FrameExtractionOptions::blackman_coeff)
      .def_readwrite("snip_edges", &FrameExtractionOptions::snip_edges)
      .def_readwrite("allow_downsample", &FrameExtractionOptions::allow_downsample)
      .def_readwrite("allow_upsample", &FrameExtractionOptions::allow_upsample)
      .def_readwrite("max_feature_vectors", &FrameExtractionOptions::max_feature_vectors);

  py::class_<MelBanksOptions>(m, "MelBanksOptions")
      .def(py::init<const int&>())
      .def_readwrite("num_bins", &MelBanksOptions::num_bins)
      .def_readwrite("low_freq", &MelBanksOptions::low_freq)
      .def_readwrite("high_freq", &MelBanksOptions::high_freq)
      .def_readwrite("vtln_low", &MelBanksOptions::vtln_low)
      .def_readwrite("vtln_high", &MelBanksOptions::vtln_high)
      .def_readwrite("debug_mel", &MelBanksOptions::debug_mel)
      .def_readwrite("htk_mode", &MelBanksOptions::htk_mode);

  py::class_<MfccOptions>(m, "MfccOptions")
      .def(py::init<>())
      .def_readwrite("frame_opts", &MfccOptions::frame_opts)
      .def_readwrite("mel_opts", &MfccOptions::mel_opts)
      .def_readwrite("num_ceps", &MfccOptions::num_ceps)
      .def_readwrite("use_energy", &MfccOptions::use_energy)
      .def_readwrite("energy_floor", &MfccOptions::energy_floor)
      .def_readwrite("raw_energy", &MfccOptions::raw_energy)
      .def_readwrite("cepstral_lifter", &MfccOptions::cepstral_lifter)
      .def_readwrite("htk_compat", &MfccOptions::htk_compat);

  py::class_<PlpOptions>(m, "PlpOptions")
      .def(py::init<>())
      .def_readwrite("frame_opts", &PlpOptions::frame_opts)
      .def_readwrite("mel_opts", &PlpOptions::mel_opts)
      .def_readwrite("lpc_order", &PlpOptions::lpc_order)
      .def_readwrite("num_ceps", &PlpOptions::num_ceps)
      .def_readwrite("use_energy", &PlpOptions::use_energy)
      .def_readwrite("energy_floor", &PlpOptions::energy_floor)
      .def_readwrite("raw_energy", &PlpOptions::raw_energy)
      .def_readwrite("compress_factor", &PlpOptions::compress_factor)
      .def_readwrite("cepstral_lifter", &PlpOptions::cepstral_lifter)
      .def_readwrite("cepstral_scale", &PlpOptions::cepstral_scale)
      .def_readwrite("htk_compat", &PlpOptions::htk_compat);

  py::class_<FbankOptions>(m, "FbankOptions")
      .def(py::init<>())
      .def_readwrite("frame_opts", &FbankOptions::frame_opts)
      .def_readwrite("mel_opts", &FbankOptions::mel_opts)
      .def_readwrite("use_energy", &FbankOptions::use_energy)
      .def_readwrite("energy_floor", &FbankOptions::energy_floor)
      .def_readwrite("raw_energy", &FbankOptions::raw_energy)
      .def_readwrite("use_log_fbank", &FbankOptions::use_log_fbank)
      .def_readwrite("use_power", &FbankOptions::use_power)
      .def_readwrite("htk_compat", &FbankOptions::htk_compat);

  offline_feature<MfccComputer>(m, "Mfcc");
  offline_feature<PlpComputer>(m, "Plp");
  offline_feature<FbankComputer>(m, "Fbank");

  py::class_<OnlineFeatureInterface>(m, "OnlineFeatureInterface",
      "OnlineFeatureInterface is an interface for online feature processing."
      "This interface only specifies how the object *outputs* the features."
      "How it obtains the features, e.g. from a previous object or objects of type"
      "OnlineFeatureInterface, is not specified in the interface and you will"
      "likely define new constructors or methods in the derived type to do that.")
      .def("Dim", &OnlineFeatureInterface::Dim)
      .def("NumFramesReady", &OnlineFeatureInterface::NumFramesReady)
      .def("IsLastFrame", &OnlineFeatureInterface::IsLastFrame)
      .def("GetFrame", &OnlineFeatureInterface::GetFrame)
      .def("GetFrames", &OnlineFeatureInterface::GetFrames)
      .def("FrameShiftInSeconds", &OnlineFeatureInterface::FrameShiftInSeconds);

  py::class_<OnlineBaseFeature, OnlineFeatureInterface>(m, "OnlineBaseFeature")
      .def("AcceptWaveform", &OnlineBaseFeature::AcceptWaveform,
           "This would be called from the application, when you get more wave data."
           "Note: the sampling_rate is typically only provided so the code can assert"
           "that it matches the sampling rate expected in the options.")
      .def("InputFinished", &OnlineBaseFeature::InputFinished,
           "InputFinished() tells the class you won't be providing any"
           "more waveform.  This will help flush out the last few frames"
           "of delta or LDA features (it will typically affect the return value"
           "of IsLastFrame.");

  online_base_feature<MfccComputer>(m, "OnlineMfcc");
  online_base_feature<PlpComputer>(m, "OnlinePlp");
  online_base_feature<FbankComputer>(m, "OnlineFbank");

  py::class_<OnlineCmvnOptions>(m, "OnlineCmvnOptions")
      .def(py::init<>())
      .def_readwrite("cmn_window", &OnlineCmvnOptions::cmn_window)
      .def_readwrite("speaker_frames", &OnlineCmvnOptions::speaker_frames)
      .def_readwrite("global_frames", &OnlineCmvnOptions::global_frames)
      .def_readwrite("normalize_mean", &OnlineCmvnOptions::normalize_mean)
      .def_readwrite("normalize_variance", &OnlineCmvnOptions::normalize_variance)
      .def_readwrite("modulus", &OnlineCmvnOptions::modulus)
      .def_readwrite("ring_buffer_size", &OnlineCmvnOptions::ring_buffer_size)
      .def_readwrite("skip_dims", &OnlineCmvnOptions::skip_dims)
      .def("Check", &OnlineCmvnOptions::Check);

  py::class_<OnlineCmvnState>(m, "OnlineCmvnState",
      "This bind is only used internally, so members are not exposed.")
      .def(py::init<>())
      .def(py::init<const Matrix<double>&>())
      .def(py::init<const OnlineCmvnState&>());

  py::class_<OnlineCmvn, OnlineFeatureInterface>(m, "OnlineCmvn")
      .def(py::init<const OnlineCmvnOptions&, OnlineFeatureInterface*>(),
           py::arg("opts"), py::arg("src"))
      .def(py::init<const OnlineCmvnOptions&, const OnlineCmvnState&,
           OnlineFeatureInterface*>(),
           py::arg("opts"), py::arg("stat"), py::arg("src"))
      .def("GetState", &OnlineCmvn::GetState)
      .def("SetState", &OnlineCmvn::SetState)
      .def("Freeze", &OnlineCmvn::Freeze);

  py::class_<OnlineSpliceOptions>(m, "OnlineSpliceOptions")
      .def(py::init<>())
      .def_readwrite("left_context", &OnlineSpliceOptions::left_context)
      .def_readwrite("right_context", &OnlineSpliceOptions::right_context);

  py::class_<OnlineSpliceFrames, OnlineFeatureInterface>(m, "OnlineSpliceFrames")
      .def(py::init<const OnlineSpliceOptions&, OnlineFeatureInterface*>(),
           py::arg("opts"), py::arg("src"));

  py::class_<OnlineTransform, OnlineFeatureInterface>(m, "OnlineTransform")
      .def(py::init<const Matrix<float>&, OnlineFeatureInterface*>());

  py::class_<OnlineCacheFeature, OnlineFeatureInterface>(m, "OnlineCacheFeature")
      .def(py::init<OnlineFeatureInterface*>(), py::arg("src"))
      .def("ClearCache", &OnlineCacheFeature::ClearCache);

  py::class_<OnlineAppendFeature, OnlineFeatureInterface>(m, "OnlineAppendFeature")
      .def(py::init<OnlineFeatureInterface*, OnlineFeatureInterface*>(),
           py::arg("src1"), py::arg("src2"));

  m.attr("kWaveSampleMax") = py::cast(kWaveSampleMax);

  py::class_<WaveInfo>(m, "WaveInfo")
      .def(py::init<>())
      .def("IsStreamed", &WaveInfo::IsStreamed,
           "Is stream size unknown? Duration and SampleCount not valid if true.")
      .def("SampFreq", &WaveInfo::SampFreq,
           "Sample frequency, Hz.")
      .def("SampleCount", &WaveInfo::SampleCount,
           "Number of samples in stream. Invalid if IsStreamed() is true.")
      .def("Duration", &WaveInfo::Duration,
           "Approximate duration, seconds. Invalid if IsStreamed() is true.")
      .def("NumChannels", &WaveInfo::NumChannels,
           "Number of channels, 1 to 16.")
      .def("BlockAlign", &WaveInfo::BlockAlign,
           "Bytes per sample.")
      .def("DataBytes", &WaveInfo::DataBytes,
           "Wave data bytes. Invalid if IsStreamed() is true.")
      .def("ReverseBytes", &WaveInfo::ReverseBytes,
           "Is data file byte order different from machine byte order?");

  py::class_<WaveData>(m, "WaveData")
      .def(py::init<>())
      .def(py::init<const float, const Matrix<float>>(),
           py::arg("samp_freq"), py::arg("data"))
      .def("Duration", &WaveData::Duration,
           "Returns the duration in seconds")
      .def("Data", &WaveData::Data, py::return_value_policy::reference)
      .def("SampFreq", &WaveData::SampFreq)
      .def("Clear", &WaveData::Clear)
      .def("CopyFrom", &WaveData::CopyFrom)
      .def("Read", &WaveData::Read)
      .def("Swap", &WaveData::Swap);

  pybind_sequential_table_reader<WaveHolder>(m, "SequentialWaveReader");
  pybind_sequential_table_reader<WaveInfoHolder>(m, "SequentialWaveInfoReader");
  pybind_random_access_table_reader<WaveHolder>(m, "RandomAccessWaveReader");
  pybind_random_access_table_reader<WaveInfoHolder>(m, "RandomAccessWaveInfoReader");
  feat_pitch_functions(m);
  feat_signal(m);
}
