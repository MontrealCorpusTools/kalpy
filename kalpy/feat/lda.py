"""Classes for computing LDA transforms"""
from __future__ import annotations

import logging
import pathlib
import typing

from _kalpy import transform
from _kalpy.util import ConstIntegerSet, Output
from kalpy.feat.data import FeatureArchive
from kalpy.gmm.data import AlignmentArchive
from kalpy.gmm.utils import read_gmm_model

logger = logging.getLogger("kalpy.lda")
logger.setLevel(logging.DEBUG)
logger.write = lambda msg: logger.info(msg) if msg != "\n" else None
logger.flush = lambda: None


class LdaStatsAccumulator:
    def __init__(
        self,
        acoustic_model_path: typing.Union[pathlib.Path, str],
        silence_phones: typing.List[int],
        weight_distribute: bool = False,
        rand_prune: float = 0.0,
    ):
        self.acoustic_model_path = str(acoustic_model_path)
        self.transition_model, self.acoustic_model = read_gmm_model(self.acoustic_model_path)
        self.silence_phones = silence_phones
        self.rand_prune = rand_prune
        self.weight_distribute = weight_distribute
        self.lda = transform.LdaEstimate()

    def accumulate_stats(
        self,
        feature_archive: FeatureArchive,
        alignment_archive: AlignmentArchive,
        callback: typing.Callable = None,
    ):
        silence_weight = 0.0
        silence_set = ConstIntegerSet(self.silence_phones)
        num_done = 0
        for alignment in alignment_archive:
            feats = feature_archive[alignment.utterance_id]
            if feats.NumRows() == 0:
                logger.warning(f"Skipping {alignment.utterance_id} due to zero-length features")
                continue
            if self.lda.Dim() == 0:
                self.lda.Init(self.transition_model.NumPdfs(), feats.NumCols())
            if callback:
                callback(alignment.utterance_id)
            self.lda.acc_lda(
                self.transition_model,
                alignment.alignment,
                feats,
                silence_set,
                rand_prune=self.rand_prune,
                silence_weight=silence_weight,
            )
            num_done += 1
            if num_done % 100 == 0:
                logger.info(f"Done {num_done} utterances.")
        logger.info(f"Done {num_done} files.")

    def export_transform(
        self, file_name: str, feature_archive: FeatureArchive, alignment_archive: AlignmentArchive
    ):
        file_name = str(file_name)
        self.accumulate_stats(feature_archive, alignment_archive)
        ko = Output(file_name, True)
        self.lda.Write(ko.Steam(), True)
        logger.info("Written statistics.")


class MlltStatsAccumulator:
    def __init__(
        self,
        acoustic_model_path: typing.Union[pathlib.Path, str],
        silence_phones: typing.List[int],
        weight_distribute: bool = False,
        rand_prune: float = 0.0,
    ):
        self.acoustic_model_path = str(acoustic_model_path)
        self.transition_model, self.acoustic_model = read_gmm_model(self.acoustic_model_path)
        self.silence_phones = silence_phones
        self.rand_prune = rand_prune
        self.weight_distribute = weight_distribute
        self.mllt_accs = transform.MlltAccs(self.acoustic_model.Dim(), rand_prune)

    def accumulate_stats(
        self,
        feature_archive: FeatureArchive,
        alignment_archive: AlignmentArchive,
        callback: typing.Callable = None,
    ):
        silence_weight = 0.0
        silence_set = ConstIntegerSet(self.silence_phones)
        num_done = 0
        tot_like = 0.0
        tot_t = 0.0
        for alignment in alignment_archive:
            feats = feature_archive[alignment.utterance_id]
            if feats.NumRows() == 0:
                logger.warning(f"Skipping {alignment.utterance_id} due to zero-length features")
                continue
            if callback:
                callback(alignment.utterance_id)
            tot_like_this_file, tot_weight_this_file = self.mllt_accs.gmm_acc_mllt(
                self.acoustic_model,
                self.transition_model,
                alignment.alignment,
                feats,
                silence_set,
                silence_weight=silence_weight,
            )
            num_done += 1
            tot_like += tot_like_this_file
            tot_t += tot_weight_this_file
            logger.info(
                f"Average like for this file is {tot_like_this_file/tot_weight_this_file} "
                f"over {tot_weight_this_file} frames."
            )
            if num_done % 10 == 0:
                logger.info(f"Average per frame so far is {tot_like/tot_t}")
        logger.info(f"Done {num_done} files.")
        logger.info(
            f"Overall avg like per frame (Gaussian only) = {tot_like/tot_t} over {tot_t} frames."
        )

    def export_stats(
        self, file_name: str, feature_archive: FeatureArchive, alignment_archive: AlignmentArchive
    ):
        file_name = str(file_name)
        self.accumulate_stats(feature_archive, alignment_archive)
        ko = Output(file_name, True)
        self.mllt_accs.Write(ko.Steam(), True)
        logger.info(f"Written accs.")
