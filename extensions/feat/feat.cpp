
#include "feat/pybind_feat.h"

#include "feat/feature-mfcc.h"
#include "feat/feature-functions.h"
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
      .def("ComputeFeatures",
      &OfflineFeatureTpl<Feature>::ComputeFeatures,
      py::call_guard<py::gil_scoped_release>())
      .def("compute", [](
            const OfflineFeatureTpl<Feature>& v,
            py::array_t<float> x
            ) -> Matrix<float> {
          py::gil_scoped_release gil_release;
        auto r = x.unchecked<1>();
        auto vector = Vector<float>(r.shape(0));
        float vtln_warp = 1.0;
        Matrix<float> features;
        for (py::size_t i = 0; i < r.shape(0); i++)
          vector(i) = r(i);
        v.Compute(vector, vtln_warp, &features);
        return features;
      })
      .def("compute", [](
            const OfflineFeatureTpl<Feature>& v,
            const Vector<float> &vector
            ) -> Matrix<float> {
          py::gil_scoped_release gil_release;
        float vtln_warp = 1.0;
        Matrix<float> features;
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
        py::arg("signal"),
      py::call_guard<py::gil_scoped_release>());

  m.def("FFTbasedConvolveSignals",
        &FFTbasedConvolveSignals,
        "This function implements FFT-based convolution of two signals. "
        "However this should be an inefficient version of BlockConvolveSignals() "
        "as it processes the entire signal with a single FFT.",
        py::arg("filter"),
        py::arg("signal"),
      py::call_guard<py::gil_scoped_release>());

  m.def("FFTbasedBlockConvolveSignals",
        &FFTbasedBlockConvolveSignals,
        "This function implements FFT-based block convolution of two signals using "
        "overlap-add method. This is an efficient way to evaluate the discrete "
        "convolution of a long signal with a finite impulse response filter.",
        py::arg("filter"),
        py::arg("signal"),
      py::call_guard<py::gil_scoped_release>());
}

void feat_feat_functions(py::module& m){

  m.def("ComputePowerSpectrum",
        &ComputePowerSpectrum,
        "ComputePowerSpectrum converts a complex FFT (as produced by the FFT "
        "functions in matrix/matrix-functions.h), and converts it into "
        "a power spectrum.  If the complex FFT is a vector of size n (representing "
        "half the complex FFT of a real signal of size n, as described there), "
        "this function computes in the first (n/2) + 1 elements of it, the "
        "energies of the fft bins from zero to the Nyquist frequency.  Contents of the "
        "remaining (n/2) - 1 elements are undefined at output.",
        py::arg("complex_fft"),
      py::call_guard<py::gil_scoped_release>());

  py::class_<DeltaFeaturesOptions>(m, "DeltaFeaturesOptions")
      .def(py::init<>())
      .def_readwrite("order", &DeltaFeaturesOptions::order,
                   "Order of delta computation")
      .def_readwrite("window", &DeltaFeaturesOptions::window, "Parameter controlling window for delta computation (actual window"
                   " size for each delta order is 1 + 2*delta-window-size)");

  py::class_<DeltaFeatures>(m, "DeltaFeatures",
    "This class provides a low-level function to compute delta features. "
    "The function takes as input a matrix of features and a frame index "
    "that it should compute the deltas on.  It puts its output in an object "
    "of type VectorBase, of size (original-feature-dimension) * (opts.order+1). "
    "This is not the most efficient way to do the computation, but it's "
    "state-free and thus easier to understand")
      .def(py::init<const DeltaFeaturesOptions &>(),
        py::arg("opts"))
      .def("Process", &DeltaFeatures::Process,
        py::arg("input_feats"),
        py::arg("frame"),
        py::arg("output_frame"),
      py::call_guard<py::gil_scoped_release>());

  py::class_<ShiftedDeltaFeaturesOptions>(m, "ShiftedDeltaFeaturesOptions")
      .def(py::init<>())
      .def_readwrite("window", &ShiftedDeltaFeaturesOptions::window,
                   "Size of delta advance and delay.")
      .def_readwrite("num_blocks", &ShiftedDeltaFeaturesOptions::num_blocks,
            "Number of delta blocks in advance"
                   " of each frame to be concatenated")
      .def_readwrite("block_shift", &ShiftedDeltaFeaturesOptions::block_shift,
                    "Distance between each block");

  py::class_<ShiftedDeltaFeatures>(m, "ShiftedDeltaFeatures",
            "This class provides a low-level function to compute shifted "
            "delta cesptra (SDC). "
            "The function takes as input a matrix of features and a frame index "
            "that it should compute the deltas on.  It puts its output in an object "
            "of type VectorBase, of size original-feature-dimension + (1  * num_blocks).")
      .def(py::init<const ShiftedDeltaFeaturesOptions &>(),
        py::arg("opts"))
      .def("Process", &ShiftedDeltaFeatures::Process,
        py::arg("input_feats"),
        py::arg("frame"),
        py::arg("output_frame"));

  m.def("ComputeDeltas",
        &ComputeDeltas,
        "ComputeDeltas is a convenience function that computes deltas on a feature "
        "file.  If you want to deal with features coming in bit by bit you would have "
        "to use the DeltaFeatures class directly, and do the computation frame by "
        "frame.  Later we will have to come up with a nice mechanism to do this for "
        "features coming in.",
        py::arg("delta_opts"),
        py::arg("input_features"),
        py::arg("output_features"),
      py::call_guard<py::gil_scoped_release>());

  m.def("compute_deltas",
        [](
            const DeltaFeaturesOptions &delta_opts,
            const MatrixBase<BaseFloat> &input_features){
          py::gil_scoped_release gil_release;
                Matrix<BaseFloat> output_features;
            ComputeDeltas(delta_opts,
                            input_features,
                            &output_features);
            return output_features;

        },
        "ComputeDeltas is a convenience function that computes deltas on a feature "
        "file.  If you want to deal with features coming in bit by bit you would have "
        "to use the DeltaFeatures class directly, and do the computation frame by "
        "frame.  Later we will have to come up with a nice mechanism to do this for "
        "features coming in.",
        py::arg("delta_opts"),
        py::arg("input_features"));

  m.def("ComputeShiftedDeltas",
        &ComputeShiftedDeltas,
        "ComputeShiftedDeltas computes deltas from a feature file by applying "
        "ShiftedDeltaFeatures over the frames. This function is provided for "
        "convenience, however, ShiftedDeltaFeatures can be used directly.",
        py::arg("delta_opts"),
        py::arg("input_features"),
        py::arg("output_features"),
      py::call_guard<py::gil_scoped_release>());

  m.def("SpliceFrames",
        &SpliceFrames,
        "SpliceFrames will normally be used together with LDA. "
        "It splices frames together to make a window.  At the "
        "start and end of an utterance, it duplicates the first "
        "and last frames. "
        "Will throw if input features are empty. "
        "left_context and right_context must be nonnegative. "
        "these both represent a number of frames (e.g. 4, 4 is "
        "a good choice).",
        py::arg("input_features"),
        py::arg("left_context"),
        py::arg("right_context"),
        py::arg("output_features"),
      py::call_guard<py::gil_scoped_release>());

  m.def("splice_frames",

        [](
            const MatrixBase<BaseFloat> &input_features,
                  int32 left_context,
                  int32 right_context){
          py::gil_scoped_release gil_release;
                Matrix<BaseFloat> output_features;
            SpliceFrames(input_features,
                   left_context,
                   right_context,
                            &output_features);
            return output_features;

        },
        "SpliceFrames will normally be used together with LDA. "
        "It splices frames together to make a window.  At the "
        "start and end of an utterance, it duplicates the first "
        "and last frames. "
        "Will throw if input features are empty. "
        "left_context and right_context must be nonnegative. "
        "these both represent a number of frames (e.g. 4, 4 is "
        "a good choice).",
        py::arg("input_features"),
        py::arg("left_context"),
        py::arg("right_context"));

  m.def("ReverseFrames",
        &ReverseFrames,
        "ReverseFrames reverses the frames in time (used for backwards decoding)",
        py::arg("input_features"),
        py::arg("output_features"));

  m.def("InitIdftBases",
        &InitIdftBases,
        py::arg("n_bases"),
        py::arg("dimension"),
        py::arg("mat_out"));

  m.def("paste_feats",
        [](const std::vector<Matrix<BaseFloat> > &in,
                 int32 tolerance
                 ){
          py::gil_scoped_release gil_release;
                    Matrix<BaseFloat> out;
            int32 min_len = in[0].NumRows(),
            max_len = in[0].NumRows(),
            tot_dim = in[0].NumCols();
        for (int32 i = 1; i < in.size(); i++) {
            int32 len = in[i].NumRows(), dim = in[i].NumCols();
            tot_dim += dim;
            if(len < min_len) min_len = len;
            if(len > max_len) max_len = len;
        }
        if (max_len - min_len > tolerance || min_len == 0) {
            out.Resize(0, 0);
            return out;
        }
        out.Resize(min_len, tot_dim);
        int32 dim_offset = 0;
        for (int32 i = 0; i < in.size(); i++) {
            int32 this_dim = in[i].NumCols();
            out.Range(0, min_len, dim_offset, this_dim).CopyFromMat(
                in[i].Range(0, min_len, 0, this_dim));
            dim_offset += this_dim;
        }
        return out;
        },
        py::arg("in"),
        py::arg("tolerance"));

  py::class_<SlidingWindowCmnOptions>(m, "SlidingWindowCmnOptions")
      .def(py::init<>())
      .def_readwrite("cmn_window", &SlidingWindowCmnOptions::cmn_window,
                   "Window in frames for running "
                   "average CMN computation")
      .def_readwrite("min_window", &SlidingWindowCmnOptions::min_window,
            "Minimum CMN window "
                   "used at start of decoding (adds latency only at start). "
                   "Only applicable if center == false, ignored if center==true")
      .def_readwrite("max_warnings", &SlidingWindowCmnOptions::max_warnings,
                    "Maximum warnings to report "
                   "per utterance. 0 to disable, -1 to show all.")
      .def_readwrite("normalize_variance", &SlidingWindowCmnOptions::normalize_variance,
                    "If true, normalize "
                   "variance to one.")
      .def_readwrite("center", &SlidingWindowCmnOptions::center,
                    "If true, use a window centered on the "
                   "current frame (to the extent possible, modulo end effects). "
                   "If false, window is to the left.");

  m.def("SlidingWindowCmn",
        &SlidingWindowCmn,
        "Applies sliding-window cepstral mean and/or variance normalization.  See the "
        "strings registering the options in the options class for information on how "
        "this works and what the options are.  input and output must have the same "
        "dimension.",
        py::arg("opts"),
        py::arg("input"),
        py::arg("output"));

  m.def("sliding_window_cmn",
        [](const SlidingWindowCmnOptions &opts,
                      const MatrixBase<BaseFloat> &input){
          py::gil_scoped_release gil_release;
                        Matrix<BaseFloat> output(input.NumRows(),
                                  input.NumCols(), kUndefined);
                        SlidingWindowCmn(opts, input, &output);
                        return output;
                      },
        "Applies sliding-window cepstral mean and/or variance normalization.  See the "
        "strings registering the options in the options class for information on how "
        "this works and what the options are.  input and output must have the same "
        "dimension.",
        py::arg("opts"),
        py::arg("input"));
}

void feat_pitch_functions(py::module& m){


  py::class_<PitchExtractionOptions>(m, "PitchExtractionOptions")
      .def(py::init<>())
      .def_readwrite("samp_freq", &PitchExtractionOptions::samp_freq,
                   "Waveform data sample frequency (must match the waveform "
                   "file, if specified there)")
      .def_readwrite("frame_shift_ms", &PitchExtractionOptions::frame_shift_ms, "Frame length in "
                   "milliseconds")
      .def_readwrite("frame_length_ms", &PitchExtractionOptions::frame_length_ms, "Frame shift in "
                   "milliseconds")
      .def_readwrite("preemph_coeff", &PitchExtractionOptions::preemph_coeff,
                   "Coefficient for use in signal preemphasis (deprecated)")
      .def_readwrite("min_f0", &PitchExtractionOptions::min_f0,
                   "min. F0 to search for (Hz)")
      .def_readwrite("max_f0", &PitchExtractionOptions::max_f0,
                   "max. F0 to search for (Hz)")
      .def_readwrite("soft_min_f0", &PitchExtractionOptions::soft_min_f0,
                   "Minimum f0, applied in soft way, must not exceed min-f0")
      .def_readwrite("penalty_factor", &PitchExtractionOptions::penalty_factor,
                   "cost factor for FO change.")
      .def_readwrite("lowpass_cutoff", &PitchExtractionOptions::lowpass_cutoff,
                   "cutoff frequency for LowPass filter (Hz) ")
      .def_readwrite("resample_freq", &PitchExtractionOptions::resample_freq,
                   "Frequency that we down-sample the signal to.  Must be "
                   "more than twice lowpass-cutoff")
      .def_readwrite("delta_pitch", &PitchExtractionOptions::delta_pitch,
                   "Smallest relative change in pitch that our algorithm "
                   "measures")
      .def_readwrite("nccf_ballast", &PitchExtractionOptions::nccf_ballast,
                   "Increasing this factor reduces NCCF for quiet frames")
      .def_readwrite("lowpass_filter_width", &PitchExtractionOptions::lowpass_filter_width,
                   "Integer that determines filter width of "
                   "lowpass filter, more gives sharper filter")
      .def_readwrite("upsample_filter_width", &PitchExtractionOptions::upsample_filter_width,
                   "Integer that determines filter width when upsampling NCCF")
      .def_readwrite("max_frames_latency", &PitchExtractionOptions::max_frames_latency, "Maximum number "
                   "of frames of latency that we allow pitch tracking to "
                   "introduce into the feature processing (affects output only "
                   "if --frames-per-chunk > 0 and "
                   "--simulate-first-pass-online=true")
      .def_readwrite("frames_per_chunk", &PitchExtractionOptions::frames_per_chunk, "Only relevant for "
                   "offline pitch extraction (e.g. compute-kaldi-pitch-feats), "
                   "you can set it to a small nonzero value, such as 10, for "
                   "better feature compatibility with online decoding (affects "
                   "energy normalization in the algorithm)")
      .def_readwrite("simulate_first_pass_online", &PitchExtractionOptions::simulate_first_pass_online,
                   "If true, compute-kaldi-pitch-feats will output features "
                   "that correspond to what an online decoder would see in the "
                   "first pass of decoding-- not the final version of the "
                   "features, which is the default.  Relevant if "
                   "--frames-per-chunk > 0")
      .def_readwrite("recompute_frame", &PitchExtractionOptions::recompute_frame, "Only relevant for "
                   "online pitch extraction, or for compatibility with online "
                   "pitch extraction.  A non-critical parameter; the frame at "
                   "which we recompute some of the forward pointers, after "
                   "revising our estimate of the signal energy.  Relevant if"
                   "--frames-per-chunk > 0")
      .def_readwrite("nccf_ballast_online", &PitchExtractionOptions::nccf_ballast_online,
                   "This is useful mainly for debug; it affects how the NCCF "
                   "ballast is computed.")
      .def_readwrite("snip_edges", &PitchExtractionOptions::snip_edges, "If this is set to false, the "
                   "incomplete frames near the ending edge won't be snipped, "
                   "so that the number of frames is the file size divided by "
                   "the frame-shift. This makes different types of features "
                   "give the same number of frames.")
      .def("NccfWindowSize", &PitchExtractionOptions::NccfWindowSize,
            "Returns the window-size in samples, after resampling.  This is the "
            "\"basic window size\", not the full window size after extending by max-lag. "
            "Because of floating point representation, it is more reliable to divide "
            "by 1000 instead of multiplying by 0.001, but it is a bit slower.")
      .def("NccfWindowShift", &PitchExtractionOptions::NccfWindowShift,
            "Returns the window-shift in samples, after resampling.")
      .def(py::pickle(
        [](const PitchExtractionOptions &p) { // __getstate__
            /* Return a tuple that fully encodes the state of the object */
            return py::make_tuple(
                p.samp_freq,
                p.frame_shift_ms,
                p.frame_length_ms,
                p.preemph_coeff,
                p.min_f0,
                p.max_f0,
                p.soft_min_f0,
                p.penalty_factor,
                p.lowpass_cutoff,
                p.resample_freq,
                p.delta_pitch,
                p.nccf_ballast,
                p.lowpass_filter_width,
                p.upsample_filter_width,
                p.max_frames_latency,
                p.frames_per_chunk,
                p.simulate_first_pass_online,
                p.recompute_frame,
                p.nccf_ballast_online,
                p.snip_edges);
        },
        [](py::tuple t) { // __setstate__
            if (t.size() != 20)
                throw std::runtime_error("Invalid state!");

            /* Create a new C++ instance */
            PitchExtractionOptions opts;

            /* Assign any additional state */
            opts.samp_freq = t[0].cast<BaseFloat>();
            opts.frame_shift_ms = t[1].cast<BaseFloat>();
            opts.frame_length_ms = t[2].cast<BaseFloat>();
            opts.preemph_coeff = t[3].cast<BaseFloat>();
            opts.min_f0 = t[4].cast<BaseFloat>();
            opts.max_f0 = t[5].cast<BaseFloat>();
            opts.soft_min_f0 = t[6].cast<BaseFloat>();
            opts.penalty_factor = t[7].cast<BaseFloat>();
            opts.lowpass_cutoff = t[8].cast<BaseFloat>();
            opts.resample_freq = t[9].cast<BaseFloat>();
            opts.delta_pitch = t[10].cast<BaseFloat>();
            opts.nccf_ballast = t[11].cast<BaseFloat>();
            opts.lowpass_filter_width = t[12].cast<int32>();
            opts.upsample_filter_width = t[13].cast<int32>();
            opts.max_frames_latency = t[14].cast<int32>();
            opts.frames_per_chunk = t[15].cast<int32>();
            opts.simulate_first_pass_online = t[16].cast<bool>();
            opts.recompute_frame = t[17].cast<int32>();
            opts.nccf_ballast_online = t[18].cast<bool>();
            opts.snip_edges = t[19].cast<bool>();

            return opts;
        }
    ));

  py::class_<ProcessPitchOptions>(m, "ProcessPitchOptions")
      .def(py::init<>())
      .def_readwrite("pitch_scale", &ProcessPitchOptions::pitch_scale,
                   "Scaling factor for the final normalized log-pitch value")
      .def_readwrite("pov_scale", &ProcessPitchOptions::pov_scale,
                   "Scaling factor for final POV (probability of voicing) "
                   "feature")
      .def_readwrite("pov_offset", &ProcessPitchOptions::pov_offset,
                   "This can be used to add an offset to the POV feature. "
                   "Intended for use in online decoding as a substitute for "
                   " CMN.")
      .def_readwrite("delta_pitch_scale", &ProcessPitchOptions::delta_pitch_scale,
                   "Term to scale the final delta log-pitch feature")
      .def_readwrite("delta_pitch_noise_stddev", &ProcessPitchOptions::delta_pitch_noise_stddev,
                   "Standard deviation for noise we add to the delta log-pitch "
                   "(before scaling); should be about the same as delta-pitch "
                   "option to pitch creation.  The purpose is to get rid of "
                   "peaks in the delta-pitch caused by discretization of pitch "
                   "values.")
      .def_readwrite("normalization_left_context", &ProcessPitchOptions::normalization_left_context,
                   "Left-context (in frames) for moving window normalization")
      .def_readwrite("normalization_right_context", &ProcessPitchOptions::normalization_right_context,
                   "Right-context (in frames) for moving window normalization")
      .def_readwrite("delta_window", &ProcessPitchOptions::delta_window,
                   "Number of frames on each side of central frame, to use for "
                   "delta window.")
      .def_readwrite("delay", &ProcessPitchOptions::delay,
                   "Number of frames by which the pitch information is "
                   "delayed.")
      .def_readwrite("add_pov_feature", &ProcessPitchOptions::add_pov_feature,
                   "If true, the warped NCCF is added to output features")
      .def_readwrite("add_normalized_log_pitch", &ProcessPitchOptions::add_normalized_log_pitch,
                   "If true, the log-pitch with POV-weighted mean subtraction "
                   "over 1.5 second window is added to output features")
      .def_readwrite("add_delta_pitch", &ProcessPitchOptions::add_delta_pitch,
                   "If true, time derivative of log-pitch is added to output "
                   "features")
      .def_readwrite("add_raw_log_pitch", &ProcessPitchOptions::add_raw_log_pitch,
                   "If true, log(pitch) is added to output features")
      .def(py::pickle(
        [](const ProcessPitchOptions &p) { // __getstate__
            /* Return a tuple that fully encodes the state of the object */
            return py::make_tuple(
                p.pitch_scale,
                p.pov_scale,
                p.pov_offset,
                p.delta_pitch_scale,
                p.delta_pitch_noise_stddev,
                p.normalization_left_context,
                p.normalization_right_context,
                p.delta_window,
                p.delay,
                p.add_pov_feature,
                p.add_normalized_log_pitch,
                p.add_delta_pitch,
                p.add_raw_log_pitch);
        },
        [](py::tuple t) { // __setstate__
            if (t.size() != 13)
                throw std::runtime_error("Invalid state!");

            /* Create a new C++ instance */
            ProcessPitchOptions opts;

            /* Assign any additional state */
            opts.pitch_scale = t[0].cast<BaseFloat>();
            opts.pov_scale = t[1].cast<BaseFloat>();
            opts.pov_offset = t[2].cast<BaseFloat>();
            opts.delta_pitch_scale = t[3].cast<BaseFloat>();
            opts.delta_pitch_noise_stddev = t[4].cast<BaseFloat>();
            opts.normalization_left_context = t[5].cast<int32>();
            opts.normalization_right_context = t[6].cast<int32>();
            opts.delta_window = t[7].cast<int32>();
            opts.delay = t[8].cast<int32>();
            opts.add_pov_feature = t[9].cast<bool>();
            opts.add_normalized_log_pitch = t[10].cast<bool>();
            opts.add_delta_pitch = t[11].cast<bool>();
            opts.add_raw_log_pitch = t[12].cast<bool>();

            return opts;
        }
    ));
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
  m.def("compute_pitch",
        [](py::array_t<float> wave, const PitchExtractionOptions & pitch_opts, const ProcessPitchOptions & process_opts){
            auto r = wave.unchecked<1>();
            auto vector = Vector<float>(r.shape(0));
            Matrix<float> features;
            for (py::size_t i = 0; i < r.shape(0); i++)
                vector(i) = r(i);
            ComputeAndProcessKaldiPitch(pitch_opts, process_opts, vector, &features);
            return features;
        },
        "This function combines ComputeKaldiPitch and ProcessPitch.  The reason "
        "why we need a separate function to do this is in order to be able to "
        "accurately simulate the online pitch-processing, for testing and for "
        "training models matched to the \"first-pass\" features.  It is sensitive to "
        "the variables in pitch_opts that relate to online processing, "
        "i.e. max_frames_latency, frames_per_chunk, simulate_first_pass_online, "
        "recompute_frame.",
        py::arg("wave"),
        py::arg("pitch_opts"),
        py::arg("process_opts"));
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
      .def_readwrite("max_feature_vectors", &FrameExtractionOptions::max_feature_vectors)
      .def(py::pickle(
        [](const FrameExtractionOptions &p) { // __getstate__
            /* Return a tuple that fully encodes the state of the object */
            return py::make_tuple(
                p.samp_freq,
                p.frame_shift_ms,
                p.frame_length_ms,
                p.dither,
                p.preemph_coeff,
                p.remove_dc_offset,
                p.window_type,
                p.round_to_power_of_two,
                p.blackman_coeff,
                p.snip_edges,
                p.allow_downsample,
                p.allow_upsample,
            p.max_feature_vectors);
        },
        [](py::tuple t) { // __setstate__
            if (t.size() != 13)
                throw std::runtime_error("Invalid state!");

            /* Create a new C++ instance */
            FrameExtractionOptions opts;

            /* Assign any additional state */
            opts.samp_freq = t[0].cast<BaseFloat>();
            opts.frame_shift_ms = t[1].cast<BaseFloat>();
            opts.frame_length_ms = t[2].cast<BaseFloat>();
            opts.dither = t[3].cast<BaseFloat>();
            opts.preemph_coeff = t[4].cast<BaseFloat>();
            opts.remove_dc_offset = t[5].cast<bool>();
            opts.window_type = t[6].cast<std::string>();
            opts.round_to_power_of_two = t[7].cast<bool>();
            opts.blackman_coeff = t[8].cast<BaseFloat>();
            opts.snip_edges = t[9].cast<bool>();
            opts.allow_downsample = t[10].cast<bool>();
            opts.allow_upsample = t[11].cast<bool>();
            opts.max_feature_vectors = t[12].cast<int>();

            return opts;
        }
    ));

  py::class_<MelBanksOptions>(m, "MelBanksOptions")
      .def(py::init<const int&>())
      .def_readwrite("num_bins", &MelBanksOptions::num_bins)
      .def_readwrite("low_freq", &MelBanksOptions::low_freq)
      .def_readwrite("high_freq", &MelBanksOptions::high_freq)
      .def_readwrite("vtln_low", &MelBanksOptions::vtln_low)
      .def_readwrite("vtln_high", &MelBanksOptions::vtln_high)
      .def_readwrite("debug_mel", &MelBanksOptions::debug_mel)
      .def_readwrite("htk_mode", &MelBanksOptions::htk_mode)
      .def(py::pickle(
        [](const MelBanksOptions &p) { // __getstate__
            /* Return a tuple that fully encodes the state of the object */
            return py::make_tuple(
                p.num_bins,
                p.low_freq,
                p.high_freq,
                p.vtln_low,
                p.vtln_high,
                p.debug_mel,
                p.htk_mode);
        },
        [](py::tuple t) { // __setstate__
            if (t.size() != 7)
                throw std::runtime_error("Invalid state!");

            /* Create a new C++ instance */
            MelBanksOptions opts;

            /* Assign any additional state */
            opts.num_bins = t[0].cast<int32>();
            opts.low_freq = t[1].cast<BaseFloat>();
            opts.high_freq = t[2].cast<BaseFloat>();
            opts.vtln_low = t[3].cast<BaseFloat>();
            opts.vtln_high = t[4].cast<BaseFloat>();
            opts.debug_mel = t[5].cast<bool>();
            opts.htk_mode = t[6].cast<bool>();

            return opts;
        }
    ));

  py::class_<MfccOptions>(m, "MfccOptions")
      .def(py::init<>())
      .def_readwrite("frame_opts", &MfccOptions::frame_opts)
      .def_readwrite("mel_opts", &MfccOptions::mel_opts)
      .def_readwrite("num_ceps", &MfccOptions::num_ceps)
      .def_readwrite("use_energy", &MfccOptions::use_energy)
      .def_readwrite("energy_floor", &MfccOptions::energy_floor)
      .def_readwrite("raw_energy", &MfccOptions::raw_energy)
      .def_readwrite("cepstral_lifter", &MfccOptions::cepstral_lifter)
      .def_readwrite("htk_compat", &MfccOptions::htk_compat)
      .def(py::pickle(
        [](const MfccOptions &p) { // __getstate__
            /* Return a tuple that fully encodes the state of the object */
            return py::make_tuple(
                p.frame_opts,
                p.mel_opts,
                p.num_ceps,
                p.use_energy,
                p.energy_floor,
                p.raw_energy,
                p.cepstral_lifter,
                p.htk_compat);
        },
        [](py::tuple t) { // __setstate__
            if (t.size() != 8)
                throw std::runtime_error("Invalid state!");

            /* Create a new C++ instance */
            MfccOptions opts;

            /* Assign any additional state */
            opts.frame_opts = t[0].cast<FrameExtractionOptions>();
            opts.mel_opts = t[1].cast<MelBanksOptions>();
            opts.num_ceps = t[2].cast<int32>();
            opts.use_energy = t[3].cast<bool>();
            opts.energy_floor = t[4].cast<BaseFloat>();
            opts.raw_energy = t[5].cast<bool>();
            opts.cepstral_lifter = t[6].cast<BaseFloat>();
            opts.htk_compat = t[7].cast<bool>();

            return opts;
        }
    ));

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
      .def_readwrite("cmn_window", &OnlineCmvnOptions::cmn_window, "Number of frames of sliding "
                 "context for cepstral mean normalization.")
      .def_readwrite("speaker_frames", &OnlineCmvnOptions::speaker_frames, "Number of frames of "
                 "previous utterance(s) from this speaker to use in cepstral "
                 "mean normalization")
      .def_readwrite("global_frames", &OnlineCmvnOptions::global_frames, "Number of frames of "
                 "global-average cepstral mean normalization stats to use for "
                 "first utterance of a speaker")
      .def_readwrite("normalize_mean", &OnlineCmvnOptions::normalize_mean, "If true, do mean normalization "
                 "(note: you cannot normalize the variance but not the mean)")
      .def_readwrite("normalize_variance", &OnlineCmvnOptions::normalize_variance, "If true, do "
                 "cepstral variance normalization in addition to cepstral mean "
                 "normalization ")
      .def_readwrite("modulus", &OnlineCmvnOptions::modulus)
      .def_readwrite("ring_buffer_size", &OnlineCmvnOptions::ring_buffer_size)
      .def_readwrite("skip_dims", &OnlineCmvnOptions::skip_dims, "Dimensions to skip normalization of "
                 "(colon-separated list of integers)")
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
      .def("Read", &WaveData::Read,
      py::call_guard<py::gil_scoped_release>())
      .def("Swap", &WaveData::Swap);

  pybind_sequential_table_reader<WaveHolder>(m, "SequentialWaveReader");
  pybind_sequential_table_reader<WaveInfoHolder>(m, "SequentialWaveInfoReader");
  pybind_random_access_table_reader<WaveHolder>(m, "RandomAccessWaveReader");
  pybind_random_access_table_reader<WaveInfoHolder>(m, "RandomAccessWaveInfoReader");
  feat_pitch_functions(m);
  feat_feat_functions(m);
  feat_signal(m);

    m.def("select_voiced_frames",
        [](
        const Matrix<BaseFloat> &feat,
            const Vector<BaseFloat> &voiced
        ){
          py::gil_scoped_release gil_release;

            int32 dim = 0;
        for (int32 i = 0; i < voiced.Dim(); i++)
            if (voiced(i) != 0.0)
            dim++;
        Matrix<BaseFloat> voiced_feat(dim, feat.NumCols());
        int32 index = 0;
        for (int32 i = 0; i < feat.NumRows(); i++) {
            if (voiced(i) != 0.0) {
            voiced_feat.Row(index).CopyFromVec(feat.Row(i));
            index++;
            }
        }
        return voiced_feat;
        },
        py::arg("feat"),
        py::arg("voiced"));

    m.def("subsample_feats",
        [](
        const Matrix<BaseFloat> &feats,
            int32 n = 1, int32 offset = 0
        ){
          py::gil_scoped_release gil_release;


      if (n > 0) {
        // This code could, of course, be much more efficient; I'm just
        // keeping it simple.
        int32 num_indexes = 0;
        for (int32 k = offset; k < feats.NumRows(); k += n)
          num_indexes++; // k is the index.

        if (num_indexes == 0) {
            Matrix<BaseFloat> output(0, feats.NumCols());
          return output;
        }
        Matrix<BaseFloat> output(num_indexes, feats.NumCols());
        int32 i = 0;
        for (int32 k = offset; k < feats.NumRows(); k += n, i++) {
          SubVector<BaseFloat> src(feats, k), dest(output, i);
          dest.CopyFromVec(src);
        }
        return output;
      } else {
        int32 repeat = -n;
        Matrix<BaseFloat> output(feats.NumRows() * repeat, feats.NumCols());
        for (int32 i = 0; i < output.NumRows(); i++)
          output.Row(i).CopyFromVec(feats.Row(i / repeat));
      return output;
      }
        },
        py::arg("feats"),
        py::arg("n") = 1,
        py::arg("offset") = 0);
}
