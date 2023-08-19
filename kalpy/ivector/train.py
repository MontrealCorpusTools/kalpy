"""Classes for training ivector models"""
from __future__ import annotations

import logging
import pathlib
import typing

from _kalpy import gmm, ivector
from _kalpy.hmm import RandomAccessPosteriorReader
from kalpy.feat.data import FeatureArchive
from kalpy.ivector.data import GselectArchive
from kalpy.utils import read_kaldi_object

logger = logging.getLogger("kalpy.ivector")
logger.setLevel(logging.DEBUG)
logger.write = lambda msg: logger.info(msg) if msg != "\n" else None
logger.flush = lambda: None


class GlobalGmmStatsAccumulator:
    def __init__(self, model_path: typing.Union[pathlib.Path, str]):
        self.model_path = str(model_path)
        self.model: gmm.DiagGmm = read_kaldi_object(gmm.DiagGmm, self.model_path)
        self.gmm_accs = gmm.AccumDiagGmm()
        self.gmm_accs.Resize(self.model, gmm.StringToGmmFlags("mvw"))
        self.num_done = 0
        self.num_skipped = 0

    def accumulate_stats(
        self,
        feature_archive: FeatureArchive,
        gselect_archive: GselectArchive,
        callback: typing.Callable = None,
    ):
        tot_like = 0.0
        tot_t = 0.0
        for utt_id, feats in feature_archive:
            if feats.NumRows() == 0:
                logger.warning(f"Skipping {utt_id} due to zero-length features")
                self.num_skipped += 1
                continue
            try:
                gselect = gselect_archive[utt_id]
            except KeyError:
                logger.warning(f"Skipping {utt_id} due to missing gselect")
                self.num_skipped += 1
                continue

            if callback:
                callback(utt_id)
            tot_like_this_file = self.gmm_accs.accumulate_from_gselect(self.model, gselect, feats)
            tot_t += feats.NumRows()
            tot_like += tot_like_this_file
            self.num_done += 1
        logger.info(f"Done {self.num_done} files, skipped {self.num_skipped}.")
        logger.info(
            f"Overall avg like per frame (Gaussian only) = {tot_like/tot_t} over {tot_t} frames."
        )


class IvectorExtractorStatsAccumulator:
    def __init__(self, ivector_extractor_path: typing.Union[pathlib.Path, str]):
        self.ivector_extractor_path = str(ivector_extractor_path)

        self.model: ivector.IvectorExtractor = read_kaldi_object(
            ivector.IvectorExtractor, self.ivector_extractor_path
        )
        self.options = ivector.IvectorExtractorStatsOptions()
        self.ivector_stats = ivector.IvectorExtractorStats(self.model, self.options)
        self.num_done = 0
        self.num_skipped = 0

    def accumulate_stats(
        self,
        feature_archive: FeatureArchive,
        post_reader: RandomAccessPosteriorReader,
        callback: typing.Callable = None,
    ):
        for utt_id, feats in feature_archive:
            if feats.NumRows() == 0:
                logger.warning(f"Skipping {utt_id} due to zero-length features")
                self.num_skipped += 1
                continue
            if not post_reader.HasKey(utt_id):
                logger.warning(f"Skipping {utt_id} due to missing posterior")
                self.num_skipped += 1
                continue
            post = post_reader.Value(utt_id)
            if len(post) != feats.NumRows():
                logger.warning(f"Skipping {utt_id} due to mismatch in lengths")
                self.num_skipped += 1
                continue
            self.ivector_stats.AccStatsForUtterance(self.model, feats, post)
            self.num_done += 1
            if callback:
                callback(utt_id)
        logger.info(f"Done {self.num_done} files, skipped {self.num_skipped}.")
