"""Classes for training GMM models"""
from __future__ import annotations

import logging
import pathlib
import typing

from _kalpy import gmm, hmm, tree
from _kalpy.matrix import DoubleVector
from _kalpy.util import Output
from kalpy.feat.data import FeatureArchive
from kalpy.gmm.data import AlignmentArchive
from kalpy.gmm.utils import read_gmm_model

logger = logging.getLogger("kalpy.train")
logger.setLevel(logging.DEBUG)
logger.write = lambda msg: logger.info(msg) if msg != "\n" else None
logger.flush = lambda: None


class GmmStatsAccumulator:
    def __init__(self, acoustic_model_path: typing.Union[pathlib.Path, str]):
        self.acoustic_model_path = str(acoustic_model_path)
        self.transition_accs = DoubleVector()
        self.transition_model, self.acoustic_model = read_gmm_model(self.acoustic_model_path)
        self.gmm_accs = gmm.AccumAmDiagGmm()
        self.transition_model.InitStats(self.transition_accs)
        self.gmm_accs.Init(self.acoustic_model, gmm.kGmmAll)
        self.num_done = 0

    def accumulate_stats(
        self,
        feature_archive: FeatureArchive,
        alignment_archive: AlignmentArchive,
        callback: typing.Callable = None,
    ):
        tot_like = 0.0
        tot_t = 0.0
        for alignment in alignment_archive:
            feats = feature_archive[alignment.utterance_id]
            if feats.NumRows() == 0:
                logger.warning(f"Skipping {alignment.utterance_id} due to zero-length features")
                continue
            if callback:
                callback(alignment.utterance_id)
            tot_like_this_file = self.gmm_accs.acc_stats(
                self.acoustic_model, self.transition_model, alignment.alignment, feats
            )
            self.transition_model.acc_stats(alignment.alignment, self.transition_accs)
            self.num_done += 1
            tot_like += tot_like_this_file
            tot_t += len(alignment.alignment)
            if self.num_done % 50 == 0:
                logger.info(
                    f"Processed {self.num_done} utterances; for utterance "
                    f"{alignment.utterance_id} avg. like is "
                    f"{tot_like_this_file/len(alignment.alignment)} "
                    f"over {len(alignment.alignment)} frames."
                )
        logger.info(f"Done {self.num_done} files.")
        if tot_t:
            logger.info(
                f"Overall avg like per frame (Gaussian only) = {tot_like/tot_t} over {tot_t} frames."
            )

    def export_stats(
        self,
        file_name: typing.Union[pathlib.Path, str],
        feature_archive: FeatureArchive,
        alignment_archive: AlignmentArchive,
    ):
        file_name = str(file_name)
        self.accumulate_stats(feature_archive, alignment_archive)
        ko = Output(file_name, True)
        self.transition_accs.Write(ko.Steam(), True)
        self.gmm_accs.Write(ko.Steam(), True)
        ko.Close()
        logger.info("Written accs.")


class TreeStatsAccumulator:
    def __init__(
        self,
        acoustic_model_path: typing.Union[pathlib.Path, str],
        var_floor: float = 0.01,
        context_width: int = 3,
        central_position: int = 1,
        context_independent_symbols: typing.List[int] = None,
        phone_map: typing.List[int] = None,
    ):
        self.acoustic_model_path = str(acoustic_model_path)
        self.transition_model, self.acoustic_model = read_gmm_model(self.acoustic_model_path)
        self.tree_stats_opts = hmm.AccumulateTreeStatsOptions()
        self.tree_stats_opts.var_floor = var_floor
        self.tree_stats_opts.context_width = context_width
        self.tree_stats_opts.central_position = central_position
        self.tree_stats_info = hmm.AccumulateTreeStatsInfo(self.tree_stats_opts)
        self.tree_stats_info.ci_phones = context_independent_symbols
        if phone_map:
            self.tree_stats_info.ci_phones = phone_map
        self.tree_stats = {}

    def accumulate_stats(
        self,
        feature_archive: FeatureArchive,
        alignment_archive: AlignmentArchive,
        callback: typing.Callable = None,
    ):
        num_done = 0
        for alignment in alignment_archive:
            feats = feature_archive[alignment.utterance_id]
            if feats.NumRows() == 0:
                logger.warning(f"Skipping {alignment.utterance_id} due to zero-length features")
                continue
            if callback:
                callback(alignment.utterance_id)
            stats = hmm.accumulate_tree_stats(
                self.transition_model, self.tree_stats_info, alignment.alignment, feats
            )
            for e, c in stats:
                e = tuple(e)
                if e not in self.tree_stats:
                    self.tree_stats[e] = c
                else:
                    self.tree_stats[e].Add(c)
            num_done += 1
        logger.info(f"Done {num_done} files.")

    def export_stats(
        self,
        file_name: typing.Union[pathlib.Path, str],
        feature_archive: FeatureArchive,
        alignment_archive: AlignmentArchive,
    ):
        file_name = str(file_name)
        self.accumulate_stats(feature_archive, alignment_archive)
        ko = Output(file_name, True)
        tree.WriteBuildTreeStats(ko.Steam(), True, [x for x in self.tree_stats.items()])
        ko.Close()
        logger.info("Written tree stats.")


class TwoFeatsStatsAccumulator:
    def __init__(self, acoustic_model_path: typing.Union[pathlib.Path, str]):
        self.acoustic_model_path = str(acoustic_model_path)
        self.transition_accs = DoubleVector()
        self.transition_model, self.acoustic_model = read_gmm_model(self.acoustic_model_path)
        self.gmm_accs = gmm.AccumAmDiagGmm()
        self.transition_model.InitStats(self.transition_accs)
        self.gmm_accs.Init(self.acoustic_model, gmm.kGmmAll)

    def accumulate_stats(
        self,
        first_feature_archive: FeatureArchive,
        second_feature_archive: FeatureArchive,
        alignment_archive: AlignmentArchive,
        callback: typing.Callable = None,
    ):
        num_done = 0
        tot_like = 0.0
        for alignment in alignment_archive:
            first_feats = first_feature_archive[alignment.utterance_id]
            second_feats = second_feature_archive[alignment.utterance_id]
            if first_feats.NumRows() == 0:
                logger.warning(
                    f"Skipping {alignment.utterance_id} due to zero-length features in first archive"
                )
                continue
            if second_feats.NumRows() == 0:
                logger.warning(
                    f"Skipping {alignment.utterance_id} due to zero-length features in second archive"
                )
                continue
            if callback:
                callback(alignment.utterance_id)
            post = hmm.AlignmentToPosterior(alignment.alignment)
            pdf_post = hmm.convert_posterior_to_pdfs(self.transition_model, post)
            tot_like_this_file = self.gmm_accs.acc_twofeats(
                self.acoustic_model, pdf_post, first_feats, second_feats
            )
            self.transition_model.acc_twofeats(
                post, first_feats, second_feats, self.transition_accs
            )
            num_done += 1
            tot_like += tot_like_this_file
        logger.info(f"Done {num_done} files.")

    def export_stats(
        self,
        file_name: str,
        first_feature_archive: FeatureArchive,
        second_feature_archive: FeatureArchive,
        alignment_archive: AlignmentArchive,
    ):
        file_name = str(file_name)
        self.accumulate_stats(first_feature_archive, second_feature_archive, alignment_archive)
        ko = Output(file_name, True)
        self.transition_accs.Write(ko.Steam(), True)
        self.gmm_accs.Write(ko.Steam(), True)
        ko.Close()
        logger.info("Written accs.")
