import pathlib
import typing

import numpy as np

from _kalpy.ivector import Plda
from _kalpy.matrix import DoubleVector, FloatVector
from kalpy.utils import read_kaldi_object


class PldaScorer:
    def __init__(
        self,
        plda_path: typing.Union[str, pathlib.Path],
        normalize_length: bool = True,
        simple_length_norm: bool = True,
    ):
        self.plda_path = str(plda_path)
        self.plda = read_kaldi_object(Plda, self.plda_path)
        self.normalize_length = normalize_length
        self.simple_length_norm = simple_length_norm

    def score_ivectors(
        self,
        speaker_ivector: typing.Union[np.ndarray, FloatVector, DoubleVector],
        utterance_ivector: typing.Union[np.ndarray, FloatVector, DoubleVector],
        num_speaker_examples: int = 1,
    ):
        if isinstance(speaker_ivector, np.ndarray):
            v = DoubleVector()
            v.from_numpy(speaker_ivector)
            speaker_ivector = v
        elif isinstance(speaker_ivector, FloatVector):
            speaker_ivector = DoubleVector(speaker_ivector)

        if isinstance(utterance_ivector, np.ndarray):
            v = DoubleVector()
            v.from_numpy(utterance_ivector)
            utterance_ivector = v
        elif isinstance(utterance_ivector, FloatVector):
            utterance_ivector = DoubleVector(utterance_ivector)

        score = self.plda.LogLikelihoodRatio(
            speaker_ivector, num_speaker_examples, utterance_ivector
        )
        return score
