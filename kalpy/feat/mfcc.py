"""Classes for computing MFCC features"""
from __future__ import annotations

import logging
import pathlib
import typing

import librosa
import numpy as np

from _kalpy import feat
from _kalpy.matrix import CompressedMatrix, FloatMatrix, FloatVector
from _kalpy.util import BaseFloatMatrixWriter, CompressedMatrixWriter
from kalpy.data import Segment
from kalpy.utils import generate_write_specifier

logger = logging.getLogger("kalpy.mfcc")
logger.setLevel(logging.DEBUG)
logger.write = lambda msg: logger.info(msg) if msg != "\n" else None
logger.flush = lambda: None


class MfccComputer:
    """
    Class for computing MFCC features

    Parameters
    ----------
    sample_frequency: float
        Sample rate to use in generating features, audio will be resampled as necessary,
        defaults to 16000
    frame_length: int
        Frame length in milliseconds, defaults to 25
    frame_shift: int
        Frame shift in milliseconds, defaults to 10
    dither: float
        Dithering to use for generating deterministic features while avoiding numerical zeros,
        defaults to -1
    preemphasis_coefficient: float
        Pre-emphasis coefficient to use prior to feature calculation, defaults to 0.97
    remove_dc_offset: bool
        Flag for removing DC offset, defaults to True
    window_type: str
        Type of window to use in generating frames, defaults to "povey"
    round_to_power_of_two: bool
        Flag for using a window based on the power of two in FFT calculation for efficiency,
        defaults to True
    blackman_coeff: float
        Coefficient to use when `window_type` is "blackman", defaults to 0.42
    snip_edges: bool
        Flag for whether edges of segments should be cutoff to ensure no zero padding in frames,
        defaults to True
    max_feature_vectors: int
        Maximum number of vectors to store in memory for VTLN calculation, defaults to -1
    num_mel_bins: int
        Number of mel frequency bins to use in calculating MFCCs, defaults to 23
    low_frequency: float
        Lowest frequency for the mel spectrum, defaults to 20
    high_frequency: float
        Highest frequency for the mel spectrum, defaults to 7800
    vtln_low: float
        VTLN lower cutoff of warping function, defaults to 100
    vtln_high: float
        VTLN upper cutoff of warping function if negative,
        added to the Nyquist frequency to get the cutoff, defaults to -500
    num_coefficients: int
        Number of MFCC coefficients, defaults to 13
    use_energy: bool
        Use energy of frame in place of the zeroth MFCC coefficient, defaults to True
    energy_floor: float
        Energy floor for MFCC, defaults to 0, set to 1.0 or 0.1 if dithering is disabled
    raw_energy: bool
        Flag for computing energy before pre-emphasis and windowing, defaults to True
    cepstral_lifter: float
        Scaling factor on cepstra for HTK compatibility, defaults to 22.0
    htk_compatibility: bool
        Flag for generating features in HTK format
    """

    def __init__(
        self,
        sample_frequency: float = 16000,
        frame_length: int = 25,
        frame_shift: int = 10,
        dither: float = 1.0,
        preemphasis_coefficient: float = 0.97,
        remove_dc_offset: bool = True,
        window_type: str = "povey",
        round_to_power_of_two: bool = True,
        blackman_coeff: float = 0.42,
        snip_edges: bool = True,
        max_feature_vectors: int = -1,
        num_mel_bins: int = 25,
        low_frequency: float = 20,
        high_frequency: float = 7800,
        vtln_low: float = 100,
        vtln_high: float = -500,
        num_coefficients: int = 13,
        use_energy: bool = True,
        energy_floor: float = 0.0,
        raw_energy: bool = True,
        cepstral_lifter: float = 22.0,
        htk_compatibility: bool = False,
        allow_downsample: bool = True,
        allow_upsample: bool = True,
    ):
        self.frame_length = frame_length
        self._frame_shift = frame_shift
        self.dither = dither
        self.preemphasis_coefficient = preemphasis_coefficient
        self.sample_frequency = sample_frequency
        self.remove_dc_offset = remove_dc_offset
        self.window_type = window_type
        self.round_to_power_of_two = round_to_power_of_two
        self.blackman_coeff = blackman_coeff
        self.snip_edges = snip_edges
        self.max_feature_vectors = max_feature_vectors
        self.allow_downsample = allow_downsample
        self.allow_upsample = allow_upsample
        self.low_frequency = low_frequency
        self.high_frequency = high_frequency
        self.vtln_low = vtln_low
        self.vtln_high = vtln_high
        self.num_mel_bins = num_mel_bins
        self.cepstral_lifter = cepstral_lifter
        self.num_coefficients = num_coefficients
        self.energy_floor = energy_floor
        self.use_energy = use_energy
        self.raw_energy = raw_energy
        self.htk_compatibility = htk_compatibility

    @property
    def frame_shift(self):
        return round(self._frame_shift / 1000, 3)

    @property
    def mfcc_obj(self):

        frame_opts = feat.FrameExtractionOptions()
        frame_opts.frame_length_ms = self.frame_length
        frame_opts.frame_shift_ms = self._frame_shift
        frame_opts.dither = self.dither
        frame_opts.preemph_coeff = self.preemphasis_coefficient
        frame_opts.samp_freq = self.sample_frequency
        frame_opts.remove_dc_offset = self.remove_dc_offset
        frame_opts.window_type = self.window_type
        frame_opts.round_to_power_of_two = self.round_to_power_of_two
        frame_opts.blackman_coeff = self.blackman_coeff
        frame_opts.snip_edges = self.snip_edges
        frame_opts.max_feature_vectors = self.max_feature_vectors
        frame_opts.allow_downsample = self.allow_downsample
        frame_opts.allow_upsample = self.allow_upsample

        mel_opts = feat.MelBanksOptions(self.num_mel_bins)
        mel_opts.low_freq = self.low_frequency
        mel_opts.high_freq = self.high_frequency
        mel_opts.vtln_low = self.vtln_low
        mel_opts.vtln_high = self.vtln_high

        mfcc_opts = feat.MfccOptions()
        mfcc_opts.frame_opts = frame_opts
        mfcc_opts.mel_opts = mel_opts
        mfcc_opts.cepstral_lifter = self.cepstral_lifter
        mfcc_opts.num_ceps = self.num_coefficients
        mfcc_opts.use_energy = self.use_energy
        mfcc_opts.energy_floor = self.energy_floor
        mfcc_opts.raw_energy = self.raw_energy
        mfcc_opts.htk_compat = self.htk_compatibility
        return feat.Mfcc(mfcc_opts)

    def compute_mfccs(
        self,
        segment: typing.Union[Segment, np.ndarray],
    ) -> np.ndarray:
        """
        Compute MFCC features for a segment

        Parameters
        ----------
        segment: :class:`~kalpy.feat.mfcc.Segment`
            Acoustic segment to generate MFCCs

        Returns
        -------
        :class:`numpy.ndarray`
            Feature matrix for the segment
        """
        mfccs = self.compute_mfccs_for_export(segment, compress=False)
        return mfccs.numpy()

    def compute_mfccs_for_export(
        self, segment: typing.Union[Segment, np.ndarray, FloatVector], compress: bool = True
    ) -> FloatMatrix:
        """
        Generate MFCCs for exporting to a kaldi archive

        Parameters
        ----------
        segment: :class:`~kalpy.feat.mfcc.Segment`
            Acoustic segment to generate MFCCs
        compress: bool, defaults to True
            Flag for whether returned matrix should be compressed

        Returns
        -------
        :class:`_kalpy.matrix.FloatMatrix`
            Feature matrix for the segment
        """
        if isinstance(segment, Segment):
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
        else:
            wave = segment
            if isinstance(wave, np.ndarray) and np.max(wave) < 1.0:
                wave = np.round(wave * 32768)

        mfccs = self.mfcc_obj.compute(wave)
        if compress:
            mfccs = CompressedMatrix(mfccs)
        return mfccs

    def export_feats(
        self,
        file_name: typing.Union[pathlib.Path, str],
        segments: typing.Iterable[typing.Tuple[str, Segment]],
        write_scp: bool = False,
        compress: bool = True,
    ) -> None:
        """
        Export features to a kaldi archive file (i.e., mfccs.ark)

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
        logger.debug(f"Writing to: {write_specifier}")
        if compress:
            writer = CompressedMatrixWriter(write_specifier)
        else:
            writer = BaseFloatMatrixWriter(write_specifier)
        for key, segment in segments:
            feats = self.compute_mfccs_for_export(segment, compress)
            writer.Write(str(key), feats)
        writer.Close()
