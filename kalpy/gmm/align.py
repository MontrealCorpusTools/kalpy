"""Classes for GMM alignment"""
from __future__ import annotations

import logging
import pathlib
import typing

from _kalpy.fstext import VectorFst
from _kalpy.gmm import gmm_align_compiled
from _kalpy.matrix import FloatMatrix
from _kalpy.util import BaseFloatVectorWriter, Int32VectorWriter
from kalpy.decoder.data import FstArchive
from kalpy.feat.data import FeatureArchive
from kalpy.gmm.data import Alignment
from kalpy.gmm.utils import read_gmm_model
from kalpy.utils import generate_write_specifier

logger = logging.getLogger("kalpy.align")
logger.setLevel(logging.DEBUG)
logger.write = lambda msg: logger.info(msg) if msg != "\n" else None
logger.flush = lambda: None


class GmmAligner:
    def __init__(
        self,
        acoustic_model_path: typing.Union[pathlib.Path, str],
        acoustic_scale: float = 1.0,
        transition_scale: float = 1.0,
        self_loop_scale: float = 1.0,
        beam: float = 10,
        retry_beam: float = 40,
        careful: bool = False,
        disambiguation_symbols: typing.List[int] = None,
    ):
        self.acoustic_model_path = str(acoustic_model_path)
        self.transition_model, self.acoustic_model = read_gmm_model(self.acoustic_model_path)
        self.acoustic_scale = acoustic_scale
        self.transition_scale = transition_scale
        self.self_loop_scale = self_loop_scale
        self.beam = beam
        self.retry_beam = retry_beam
        self.careful = careful

        self.num_done = 0
        self.num_error = 0
        self.num_retry = 0
        self.total_likelihood = 0
        self.total_frames = 0
        self.disambiguation_symbols = (
            disambiguation_symbols if disambiguation_symbols is not None else []
        )
        if self.beam >= self.retry_beam:
            self.retry_beam = 4 * self.beam

    def boost_silence(self, silence_weight: float, silence_phones: typing.List[int]):
        if silence_weight != 1.0:
            self.acoustic_model.boost_silence(
                self.transition_model, silence_phones, silence_weight
            )

    def align_utterance(
        self, training_graph: VectorFst, features: FloatMatrix, utterance_id: str = None
    ) -> typing.Optional[Alignment]:
        (
            alignment,
            words,
            likelihood,
            per_frame_log_likelihoods,
            successful,
            retried,
        ) = gmm_align_compiled(
            self.transition_model,
            self.acoustic_model,
            training_graph,
            features,
            acoustic_scale=self.acoustic_scale,
            transition_scale=self.transition_scale,
            self_loop_scale=self.self_loop_scale,
            beam=self.beam,
            retry_beam=self.retry_beam,
            careful=self.careful,
        )
        if not successful:
            return None
        return Alignment(
            utterance_id, alignment, words, likelihood / len(alignment), per_frame_log_likelihoods
        )

    def align_utterances(
        self, training_graph_archive: FstArchive, feature_archive: FeatureArchive
    ) -> typing.Generator[Alignment]:
        logger.debug(f"Aligning with {self.acoustic_model_path}")
        num_done = 0
        num_error = 0
        total_frames = 0
        total_likelihood = 0
        for utterance_id, training_graph in training_graph_archive:
            try:
                feats = feature_archive[utterance_id]
            except KeyError:
                logger.warning(f"Skipping {utterance_id} not in feature archive.")
                num_error += 1
                continue

            if feats.NumRows() == 0:
                logger.warning(f"Skipping {utterance_id} due to zero-length features")
                num_error += 1
                continue
            alignment = self.align_utterance(training_graph, feats, utterance_id)
            if alignment is None:
                num_error += 1
                continue
            yield alignment
            total_likelihood += alignment.likelihood
            total_frames += len(alignment.alignment)
            num_done += 1
        if total_frames:
            logger.info(
                f"Overall log-likelihood per frame is {total_likelihood / total_frames} over {total_frames} frames."
            )
        logger.info(f"Done {num_done}, errors on {num_error}")

    def export_alignments(
        self,
        file_name: typing.Union[pathlib.Path, str],
        training_graph_archive: FstArchive,
        feature_archive: FeatureArchive,
        word_file_name: typing.Union[pathlib.Path, str] = None,
        likelihood_file_name: typing.Union[pathlib.Path, str] = None,
        write_scp: bool = False,
        callback: typing.Callable = None,
    ):
        write_specifier = generate_write_specifier(file_name, write_scp)
        writer = Int32VectorWriter(write_specifier)
        word_writer = None
        if word_file_name:
            word_write_specifier = generate_write_specifier(word_file_name, write_scp)
            word_writer = Int32VectorWriter(word_write_specifier)
        likelihood_writer = None
        if likelihood_file_name:
            likelihood_write_specifier = generate_write_specifier(likelihood_file_name, write_scp)
            likelihood_writer = BaseFloatVectorWriter(likelihood_write_specifier)
        try:
            for alignment in self.align_utterances(training_graph_archive, feature_archive):
                if alignment is None:
                    continue
                if callback:
                    callback((alignment.utterance_id, alignment.likelihood))
                writer.Write(str(alignment.utterance_id), alignment.alignment)
                if word_writer is not None:
                    word_writer.Write(str(alignment.utterance_id), alignment.words)
                if likelihood_writer is not None:
                    likelihood_writer.Write(
                        str(alignment.utterance_id), alignment.per_frame_likelihoods
                    )
        finally:
            writer.Close()
            if word_writer is not None:
                word_writer.Close()
            if likelihood_writer is not None:
                likelihood_writer.Close()
