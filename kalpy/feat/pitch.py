"""Classes for computing pitch features"""
from __future__ import annotations

import pathlib
import typing

import librosa
import numpy as np

from _kalpy import feat
from _kalpy.matrix import CompressedMatrix, FloatMatrixBase
from _kalpy.util import BaseFloatMatrixWriter, CompressedMatrixWriter
from kalpy.data import Segment
from kalpy.utils import generate_write_specifier


class PitchComputer:
    """
    Class for computing pitch features

    Parameters
    ----------
    sample_frequency: float
        Sample rate to use in generating features, audio will be resampled as necessary,
        defaults to 16000
    frame_length: int
        Frame length in milliseconds, defaults to 25
    frame_shift: int
        Frame shift in milliseconds, defaults to 10
    min_f0: float
        Minimum f0 to search (Hz)
    max_f0: float
        Maximum f0 to search (Hz)
    soft_min_f0: float
        Minimum f0, applied in a soft way, must not exceed min_f0
    penalty_factor: float
        Cost factor for f0 change
    lowpass_cutoff: float
        Cutoff frequency for lowpass filter
    resample_frequency: float
        Frequency that we down-sample the signal to.  Must be more than twice lowpass-cutoff
    delta_pitch: float
        Smallest relative change in pitch that our algorithm measures
    nccf_ballast: float
        Increasing this factor reduces NCCF for quiet frames, helping ensure pitch
        continuity in unvoiced region
    lowpass_filter_width: int
        Integer that determines filter width of lowpass filter
    upsample_filter_width: int
        Integer that determines filter width when upsampling NCCF
    max_frames_latency: int
        The maximum number of frames of latency that we allow the pitch-processing
        to introduce, for online operation. If you set this to a large value,
        there would be no inaccuracy from the Viterbi traceback (but it might make
        you wait to see the pitch). This is not very relevant for the online
        operation: normalization-right-context is more relevant, you
        can just leave this value at zero.
    frames_per_chunk: int
        Only relevant for the function ComputeKaldiPitch which is called by
        compute-kaldi-pitch-feats. If nonzero, we provide the input as chunks of
        this size. This affects the energy normalization which has a small effect
        on the resulting features, especially at the beginning of a file. For best
        compatibility with online operation (e.g. if you plan to train models for
        the online-decoding setup), you might want to set this to a small value,
        like one frame.
    simulate_first_pass_online: bool
        If true, compute-kaldi-pitch-feats will output features that correspond to
        what an online decoder would see in the first pass of decoding-- not the final
        version of the features, which is the default.  Relevant if --frames-per-chunk > 0
    recompute_frame: int
        Only relevant for online operation or when emulating online operation
        (e.g. when setting frames_per_chunk). This is the frame-index on which we
        recompute the NCCF (e.g. frame-index 500 = after 5 seconds); if the
        segment ends before this we do it when the segment ends. We do this by
        re-computing the signal average energy, which affects the NCCF via the
        "ballast term", scaling the resampled NCCF by a factor derived from the
        average change in the "ballast term", and re-doing the backtrace
        computation. Making this infinity would be the most exact, but would
        introduce unwanted latency at the end of long utterances, for little
        benefit.
    snip_edges: bool
        Flag for whether edges of segments should be cutoff to ensure no zero padding in frames,
        defaults to True
    pitch_scale: float
        Scaling factor for the final normalized log-pitch valu
    pov_scale: float
        Scaling factor for final POV (probability of voicing) feature
    pov_offset: float
        This can be used to add an offset to the POV feature. Intended for use in
        online decoding as a substitute for  CMN.
    delta_pitch_scale: float
        Term to scale the final delta log-pitch feature
    delta_pitch_noise_stddev: float
        Standard deviation for noise we add to the delta log-pitch (before scaling);
        should be about the same as delta-pitch option to pitch creation.
        The purpose is to get rid of peaks in the delta-pitch caused by discretization of
        pitch values.
    normalization_left_context: int
        Left-context (in frames) for moving window normalization
    normalization_right_context: int
        Right-context (in frames) for moving window normalization
    delta_window: int
        Number of frames on each side of central frame, to use for delta window.
    delay: int
        Number of frames by which the pitch information is delayed.
    add_pov_feature: bool
        If true, the warped NCCF is added to output features
    add_normalized_log_pitch: bool
        If true, the log-pitch with POV-weighted mean subtraction over 1.5 second window
        is added to output features
    add_delta_pitch: bool
        If true, time derivative of log-pitch is added to output features
    add_raw_log_pitch: bool
        If true, log(pitch) is added to output features
    """

    def __init__(
        self,
        sample_frequency: float = 16000,
        frame_length: int = 25,
        frame_shift: int = 10,
        min_f0: float = 50,
        max_f0: float = 400,
        soft_min_f0: float = 10.0,
        penalty_factor: float = 0.1,
        lowpass_cutoff: float = 1000,
        resample_frequency: float = 4000,
        delta_pitch: float = 0.005,
        nccf_ballast: float = 7000,
        lowpass_filter_width: int = 1,
        upsample_filter_width: int = 5,
        max_frames_latency: int = 0,
        frames_per_chunk: int = 0,
        simulate_first_pass_online: bool = False,
        recompute_frame: int = 500,
        snip_edges: bool = True,
        pitch_scale: float = 2.0,
        pov_scale: float = 2.0,
        pov_offset: float = 0.0,
        delta_pitch_scale: float = 10.0,
        delta_pitch_noise_stddev: float = 0.005,
        normalization_left_context: int = 75,
        normalization_right_context: int = 75,
        delta_window: int = 2,
        delay: int = 0,
        add_pov_feature: bool = True,
        add_normalized_log_pitch: bool = True,
        add_delta_pitch: bool = True,
        add_raw_log_pitch: bool = False,
    ):
        self.extraction_opts = feat.PitchExtractionOptions()
        self.extraction_opts.samp_freq = sample_frequency
        self.extraction_opts.frame_length_ms = frame_length
        self.extraction_opts.frame_shift_ms = frame_shift
        self.extraction_opts.min_f0 = min_f0
        self.extraction_opts.max_f0 = max_f0
        self.extraction_opts.soft_min_f0 = soft_min_f0
        self.extraction_opts.penalty_factor = penalty_factor
        self.extraction_opts.lowpass_cutoff = lowpass_cutoff
        self.extraction_opts.resample_freq = resample_frequency
        self.extraction_opts.delta_pitch = delta_pitch
        self.extraction_opts.nccf_ballast = nccf_ballast
        self.extraction_opts.lowpass_filter_width = lowpass_filter_width
        self.extraction_opts.upsample_filter_width = upsample_filter_width
        self.extraction_opts.max_frames_latency = max_frames_latency
        self.extraction_opts.frames_per_chunk = frames_per_chunk
        self.extraction_opts.simulate_first_pass_online = simulate_first_pass_online
        self.extraction_opts.recompute_frame = recompute_frame
        self.extraction_opts.snip_edges = snip_edges

        self.process_opts = feat.ProcessPitchOptions()
        self.process_opts.pitch_scale = pitch_scale
        self.process_opts.pov_scale = pov_scale
        self.process_opts.pov_offset = pov_offset
        self.process_opts.delta_pitch_scale = delta_pitch_scale
        self.process_opts.delta_pitch_noise_stddev = delta_pitch_noise_stddev
        self.process_opts.normalization_left_context = normalization_left_context
        self.process_opts.normalization_right_context = normalization_right_context
        self.process_opts.delta_window = delta_window
        self.process_opts.delay = delay
        self.process_opts.add_pov_feature = add_pov_feature
        self.process_opts.add_normalized_log_pitch = add_normalized_log_pitch
        self.process_opts.add_delta_pitch = add_delta_pitch
        self.process_opts.add_raw_log_pitch = add_raw_log_pitch

    def compute_pitch(
        self,
        segment: Segment,
    ) -> np.ndarray:
        """
        Compute pitch features for a segment

        Parameters
        ----------
        segment: :class:`~kalpy.feat.mfcc.Segment`
            Acoustic segment to generate pitch features

        Returns
        -------
        :class:`numpy.ndarray`
            Feature matrix for the segment
        """
        pitch = self.compute_pitch_for_export(segment, compress=False)
        return pitch.numpy()

    def compute_pitch_for_export(
        self,
        segment: Segment,
        compress: bool = True,
    ) -> FloatMatrixBase:
        """
        Generate pitch features for exporting to a kaldi archive

        Parameters
        ----------
        segment: :class:`~kalpy.feat.mfcc.Segment`
            Acoustic segment to generate pitch features

        Returns
        -------
        :class:`_kalpy.matrix.FloatMatrixBase`
            Feature matrix for the segment
        """
        duration = None
        if segment.end is not None and segment.begin is not None:
            duration = segment.end - segment.begin
        wave, sr = librosa.load(
            segment.file_path,
            sr=16000,
            offset=segment.begin,
            duration=duration,
            mono=False,
        )
        wave = np.round(wave * 32768)
        if len(wave.shape) == 2:
            channel = 0 if segment.channel is None else segment.channel
            wave = wave[channel, :]
        pitch = feat.compute_pitch(wave, self.extraction_opts, self.process_opts)
        if compress:
            pitch = CompressedMatrix(pitch)
        return pitch

    def export_feats(
        self,
        file_name: typing.Union[pathlib.Path, str],
        segments: typing.Iterable[typing.Tuple[str, Segment]],
        write_scp: bool = False,
        compress: bool = True,
    ) -> None:
        """
        Export features to a kaldi archive file (i.e., pitch.ark)

        Parameters
        ----------
        file_name: :class:`~pathlib.Path` or str
            Archive file path to export to
        segments: dict[str, :class:`kalpy.feat.mfcc.Segment`]
            Mapping of utterance IDs to Segment objects
        write_scp: bool
            Flag for whether an SCP file should be generated as well
        compress: bool
            Flag for whether to export features as a compressed archive
        """
        write_specifier = generate_write_specifier(file_name, write_scp)
        if compress:
            writer = CompressedMatrixWriter(write_specifier)
        else:
            writer = BaseFloatMatrixWriter(write_specifier)
        for key, segment in segments:
            feats = self.compute_pitch_for_export(segment, compress=compress)
            writer.Write(str(key), feats)
        writer.Close()
