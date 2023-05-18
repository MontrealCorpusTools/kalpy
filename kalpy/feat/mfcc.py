from __future__ import annotations

import typing

import librosa
import numpy as np

from _kalpy import feat


class MfccComputer:
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
        blackman_coeff=0.42,
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
        filepath: str,
        begin: typing.Optional[float] = None,
        end: typing.Optional[float] = None,
        channel: int = 0,
    ):
        duration = None
        if end is not None and begin is not None:
            duration = end - begin
        wave, sr = librosa.load(
            filepath,
            sr=16000,
            offset=begin,
            duration=duration,
            mono=False,
        )
        wave = np.round(wave * 32768)
        if len(wave.shape) == 2:
            wave = wave[channel, :]
        mfccs = self.mfcc_obj.compute(wave)
        return mfccs.numpy()
