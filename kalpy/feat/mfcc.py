from __future__ import annotations

import dataclasses
import pathlib
import typing

import librosa
import numpy as np

from _kalpy import feat
from _kalpy.matrix import CompressedMatrix, FloatMatrixBase
from _kalpy.util import (
    BaseFloatMatrixWriter,
    CompressedMatrixWriter,
    RandomAccessBaseFloatMatrixReader,
    SequentialBaseFloatMatrixReader,
)


@dataclasses.dataclass
class Segment:
    """
    Data class for information about acoustic segments
    """

    file_path: str
    begin: typing.Optional[float] = None
    end: typing.Optional[float] = None
    channel: typing.Optional[int] = 0


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
    ):
        self.frame_opts = feat.FrameExtractionOptions()
        self.frame_opts.frame_length_ms = frame_length
        self.frame_opts.frame_shift_ms = frame_shift
        self.frame_opts.dither = dither
        self.frame_opts.preemph_coeff = preemphasis_coefficient
        self.frame_opts.samp_freq = sample_frequency
        self.frame_opts.remove_dc_offset = remove_dc_offset
        self.frame_opts.window_type = window_type
        self.frame_opts.round_to_power_of_two = round_to_power_of_two
        self.frame_opts.blackman_coeff = blackman_coeff
        self.frame_opts.snip_edges = snip_edges
        self.frame_opts.max_feature_vectors = max_feature_vectors

        self.mel_opts = feat.MelBanksOptions(num_mel_bins)
        self.mel_opts.low_freq = low_frequency
        self.mel_opts.high_freq = high_frequency
        self.mel_opts.vtln_low = vtln_low
        self.mel_opts.vtln_high = vtln_high
        self.mfcc_opts = feat.MfccOptions()
        self.mfcc_opts.frame_opts = self.frame_opts
        self.mfcc_opts.mel_opts = self.mel_opts
        self.mfcc_opts.cepstral_lifter = cepstral_lifter
        self.mfcc_opts.num_ceps = num_coefficients
        self.mfcc_opts.use_energy = use_energy
        self.mfcc_opts.energy_floor = energy_floor
        self.mfcc_opts.raw_energy = raw_energy
        self.mfcc_opts.htk_compat = htk_compatibility
        self.mfcc_obj = feat.Mfcc(self.mfcc_opts)

    def compute_mfccs(
        self,
        segment: Segment,
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
        mfccs = self.mfcc_obj.compute(wave)
        return mfccs.numpy()

    def _compute_mfccs_for_export(
        self,
        segment: Segment,
    ) -> FloatMatrixBase:
        """
        Generate MFCCs for exporting to a kaldi archive

        Parameters
        ----------
        segment: :class:`~kalpy.feat.mfcc.Segment`
            Acoustic segment to generate MFCCs

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
        mfccs = self.mfcc_obj.compute(wave)
        return mfccs

    def export_feats(
        self,
        file_name: str,
        segments: typing.Dict[str, Segment],
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
        file_name = str(file_name)
        if not file_name.endswith(".ark"):
            file_name += ".ark"
        if write_scp:
            write_specifier = f"ark,scp:{file_name},{file_name.replace('.ark', '.scp')}"
        else:
            write_specifier = f"ark:{file_name}"
        if compress:
            writer = CompressedMatrixWriter(write_specifier)
        else:
            writer = BaseFloatMatrixWriter(write_specifier)
        for key, segment in segments.items():
            feats = self._compute_mfccs_for_export(segment)
            if compress:
                feats = CompressedMatrix(feats)
            writer.Write(str(key), feats)
        writer.Close()


class FeatureArchive:
    """
    Class for reading an archive or SCP of features

    Parameters
    ----------
    file_name: :class:`~pathlib.Path` or str
        Path to archive or SCP file to read from
    """

    def __init__(self, file_name: typing.Union[pathlib.Path, str]):
        self.file_name = str(file_name)
        self.read_identifier = "ark"
        if self.file_name.endswith(".scp"):
            self.read_identifier = "scp"

    def __iter__(self) -> typing.Generator[typing.Tuple[str, np.ndarray]]:
        """Iterate over the utterance features in the archive"""
        reader = SequentialBaseFloatMatrixReader(f"{self.read_identifier}:{self.file_name}")
        try:
            while not reader.Done():
                utt = reader.Key()
                feats = reader.Value().numpy()
                yield utt, feats
                reader.Next()
        finally:
            reader.Close()

    def __getitem__(self, item: str) -> np.ndarray:
        """Get features for a particular key from the archive file"""
        item = str(item)
        reader = RandomAccessBaseFloatMatrixReader(f"{self.read_identifier}:{self.file_name}")
        try:
            if not reader.HasKey(item):
                raise Exception(f"No key {item} found in {self.file_name}")
            return reader.Value(item).numpy()
        finally:
            reader.Close()
