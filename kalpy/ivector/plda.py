import pathlib
import typing

import numpy as np

from _kalpy.ivector import Plda, ivector_normalize_length, ivector_subtract_mean
from _kalpy.matrix import DoubleVector, FloatVector
from kalpy.utils import read_kaldi_object
from kalpy.ivector.data import IvectorArchive


class PldaScorer:
    def __init__(
        self,
        plda_path: typing.Union[str, pathlib.Path],
        normalize_length: bool = True,
        simple_length_norm: bool = True,
    ):
        self.plda_path = str(plda_path)
        self.plda: Plda = read_kaldi_object(Plda, self.plda_path)
        self.normalize_length = normalize_length
        self.simple_length_norm = simple_length_norm
        self.speaker_ids = None
        self.speaker_ivectors = None
        self.num_speaker_examples = None

    def load_speaker_ivectors(self, speaker_archive_path, num_utts_path=None):
        ivector_archive = IvectorArchive(
            speaker_archive_path, num_utterances_file_name=num_utts_path
        )
        speaker_ivectors = []
        self.speaker_ids = []
        self.num_speaker_examples = []
        for speaker_id, ivector, utts in ivector_archive:
            self.speaker_ids.append(speaker_id)
            self.num_speaker_examples.append(utts)
            if self.normalize_length:
                ivector_normalize_length(ivector)
            speaker_ivectors.append(DoubleVector(ivector))
        ivector_subtract_mean(speaker_ivectors,normalize=self.normalize_length)
        self.speaker_ivectors = self.plda.transform_ivectors(speaker_ivectors, self.num_speaker_examples)

    def transform_ivector(self, ivector: np.ndarray, num_examples: int = 1):
        return self.plda.transform_ivector(ivector, num_examples)

    def transform_ivectors(self, ivectors: np.ndarray, num_examples: np.ndarray = None):
        if num_examples is None:
            num_examples = np.ones((ivectors.shape[0]))
        return self.plda.transform_ivectors(ivectors, num_examples)

    def score_ivectors(
        self,
        speaker_ivector: typing.Union[np.ndarray, FloatVector, DoubleVector],
        utterance_ivector: typing.Union[np.ndarray, FloatVector, DoubleVector],
        num_speaker_examples: int = 1,
    ):
        score = self.plda.LogLikelihoodRatio(
            speaker_ivector, num_speaker_examples, utterance_ivector
        )
        return score

    def classify_speaker(
        self,
        utterance_ivector: typing.Union[np.ndarray, FloatVector, DoubleVector],
    ):
        if self.num_speaker_examples is None:
            self.num_speaker_examples = [1 for _ in (self.speaker_ivectors.shape[0])]
        if isinstance(utterance_ivector, np.ndarray):
            utterance_ivector = DoubleVector()
            utterance_ivector.from_numpy(utterance_ivector)
        ind, score = self.plda.classify_utterance(utterance_ivector, self.speaker_ivectors, self.num_speaker_examples)
        speaker = self.speaker_ids[ind]
        return speaker, score
