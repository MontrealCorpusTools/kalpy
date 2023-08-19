import typing

import dataclassy

from _kalpy.feat import (
    DeltaFeaturesOptions,
    SlidingWindowCmnOptions,
    compute_deltas,
    paste_feats,
    sliding_window_cmn,
    splice_frames,
)
from _kalpy.matrix import DoubleMatrix, FloatMatrix
from _kalpy.transform import ApplyCmvn, apply_transform
from kalpy.data import Segment
from kalpy.feat.mfcc import MfccComputer
from kalpy.feat.pitch import PitchComputer


@dataclassy.dataclass
class Utterance:
    segment: Segment
    transcript: str
    cmvn_string: typing.Optional[str] = None
    fmllr_string: typing.Optional[str] = None
    mfccs: typing.Optional[FloatMatrix] = None

    def generate_mfccs(self, mfcc_computer: MfccComputer):
        if self.mfccs is None:
            self.mfccs = mfcc_computer.compute_mfccs_for_export(self.segment, compress=False)

    def apply_cmvn(
        self, cmvn: DoubleMatrix, sliding_cmvn: bool = False, cmvn_norm_vars: bool = False
    ):

        if not sliding_cmvn:
            ApplyCmvn(cmvn, cmvn_norm_vars, self.mfccs)
        else:
            sliding_cmvn_options = SlidingWindowCmnOptions()
            sliding_cmvn_options.cmn_window = 300
            sliding_cmvn_options.center = True
            sliding_cmvn_options.normalize_variance = cmvn_norm_vars
            self.mfccs = sliding_window_cmn(sliding_cmvn_options, self.mfccs)

    def generate_features(
        self,
        mfcc_computer: MfccComputer,
        pitch_computer: PitchComputer = None,
        lda_mat: FloatMatrix = None,
        fmllr_trans: FloatMatrix = None,
        uses_speaker_adaptation: bool = True,
        uses_splices: bool = False,
        uses_deltas: bool = True,
        splice_context: int = 3,
    ):
        if lda_mat is not None:
            uses_splices = True
            uses_deltas = False
        if self.mfccs is None:
            self.generate_mfccs(mfcc_computer)
        feats = self.mfccs

        if pitch_computer is not None:
            pitch = pitch_computer.compute_pitch_for_export(self.segment, compress=False)
            feats = paste_feats([feats, pitch], 0)
        if uses_splices:
            feats = splice_frames(feats, splice_context, splice_context)
            if lda_mat is not None:
                feats = apply_transform(feats, lda_mat)
        elif uses_deltas:
            delta_options = DeltaFeaturesOptions()
            feats = compute_deltas(delta_options, feats)
        if uses_speaker_adaptation and fmllr_trans is not None:
            feats = apply_transform(feats, fmllr_trans)
        return feats
