"""Classes for GMM alignment"""
from __future__ import annotations

import logging
import pathlib
import typing

from _kalpy.decoder import LatticeFasterDecoder, LatticeFasterDecoderConfig
from _kalpy.fstext import ConstFst, GetLinearSymbolSequence
from _kalpy.gmm import DecodableAmDiagGmmScaled, gmm_rescore_lattice
from _kalpy.lat import (
    CompactLattice,
    CompactLatticeWriter,
    DeterminizeLatticePhonePrunedWrapper,
    GetPerFrameAcousticCosts,
    LatticeWriter,
    lattice_to_post,
)
from _kalpy.matrix import FloatMatrix
from _kalpy.util import Int32VectorWriter
from kalpy.feat.data import FeatureArchive
from kalpy.gmm.data import Alignment, LatticeArchive
from kalpy.gmm.utils import read_gmm_model
from kalpy.utils import generate_write_specifier

logger = logging.getLogger("kalpy.decode")
logger.setLevel(logging.DEBUG)
logger.write = lambda msg: logger.info(msg) if msg != "\n" else None
logger.flush = lambda: None


class GmmDecoder:
    def __init__(
        self,
        acoustic_model_path: typing.Union[pathlib.Path, str],
        hclg_fst: ConstFst,
        acoustic_scale: float = 0.1,
        beam: float = 16.0,
        lattice_beam: float = 10.0,
        max_active: int = 7000,
        min_active: int = 200,
        prune_interval: int = 25,
        determinize_lattice: bool = True,
        beam_delta: float = 0.5,
        hash_ratio: float = 2.0,
        prune_scale: float = 0.1,
        allow_partial: bool = True,
        fast: bool = False,
    ):
        self.acoustic_model_path = acoustic_model_path
        self.transition_model, self.acoustic_model = read_gmm_model(self.acoustic_model_path)
        self.hclg_fst = hclg_fst
        self.acoustic_scale = acoustic_scale
        self.allow_partial = allow_partial
        self.beam = beam
        self.lattice_beam = lattice_beam
        self.max_active = max_active
        self.min_active = min_active
        self.prune_interval = prune_interval
        self.determinize_lattice = determinize_lattice
        self.beam_delta = beam_delta
        self.hash_ratio = hash_ratio
        self.prune_scale = prune_scale
        self.fast = fast

        self.config = LatticeFasterDecoderConfig()
        self.config.beam = beam
        self.config.lattice_beam = lattice_beam
        self.config.max_active = max_active
        self.config.min_active = min_active
        self.config.prune_interval = prune_interval
        self.config.determinize_lattice = determinize_lattice
        self.config.beam_delta = beam_delta
        self.config.hash_ratio = hash_ratio
        self.config.prune_scale = prune_scale
        self.num_done = 0
        self.num_error = 0
        self.num_retry = 0
        self.total_likelihood = 0
        self.total_frames = 0

    def boost_silence(self, silence_weight: float, silence_phones: typing.List[int]):
        if silence_weight != 1.0:
            self.acoustic_model.boost_silence(
                self.transition_model, silence_phones, silence_weight
            )

    def decode_utterance(
        self, features: FloatMatrix, utterance_id: str = None
    ) -> typing.Optional[Alignment]:
        decodable = DecodableAmDiagGmmScaled(
            self.acoustic_model, self.transition_model, features, self.acoustic_scale
        )

        d = LatticeFasterDecoder(self.hclg_fst, self.config)
        ans = d.Decode(decodable)
        if not ans:
            logger.warning(f"Did not successfully decode {utterance_id}")
            self.num_error += 1
            return None

        ans = d.ReachedFinal()
        if not ans:
            if self.allow_partial:
                logger.warning(
                    f"Outputting partial output for utterance {utterance_id} "
                    "since no final-state reached"
                )
            else:
                logger.warning(
                    f"Not producing output for utterance {utterance_id} "
                    f"since no final-state reached and allow_partial==False"
                )

        ans, decoded = d.GetBestPath()
        if not ans:
            logger.error(f"Failed to get traceback for utterance {utterance_id}")
            return None
        if decoded.NumStates() == 0:
            logger.warning(f"Error getting best path from decoder for utterance {utterance_id}")
        alignment, words, weight = GetLinearSymbolSequence(decoded)
        likelihood = -(weight.Value1() + weight.Value2()) / self.acoustic_scale
        if self.fast:
            return Alignment(utterance_id, alignment, words, likelihood)

        self.num_done += 1
        self.total_likelihood += likelihood
        self.total_frames += len(alignment)
        per_frame_log_likelihoods = GetPerFrameAcousticCosts(decoded)
        per_frame_log_likelihoods.Scale(-1 / self.acoustic_scale)
        ans, lat = d.GetRawLattice()
        lat.Connect()
        if self.config.determinize_lattice:
            clat = CompactLattice()
            ans = DeterminizeLatticePhonePrunedWrapper(
                self.transition_model,
                lat,
                self.config.lattice_beam,
                clat,
                self.config.det_opts,
            )
            if not ans:
                logger.warning(
                    f"Determinization finished earlier than the beam for utterance {utterance_id}"
                )
            if self.acoustic_scale != 0.0:
                clat.ScaleLattice(acoustic_scale=self.acoustic_scale)
            data = Alignment(
                utterance_id,
                alignment,
                words,
                likelihood,
                per_frame_log_likelihoods,
                lattice=clat,
            )
        else:
            if self.acoustic_scale != 0.0:
                lat.ScaleLattice(acoustic_scale=self.acoustic_scale)
            data = Alignment(
                utterance_id,
                alignment,
                words,
                likelihood,
                per_frame_log_likelihoods,
                lattice=lat,
            )
        logger.info(
            f"Log-like per frame for utterance {utterance_id} is "
            f"{likelihood/len(alignment)} over {len(alignment)} frames."
        )
        logger.debug(
            f"Cost for utterance {utterance_id} is " f"{weight.Value1()} + {weight.Value2()}"
        )
        return data

    def decode_utterances(self, feature_archive: FeatureArchive) -> typing.Generator[Alignment]:
        self.num_done = 0
        self.num_error = 0
        self.num_retry = 0
        self.total_likelihood = 0
        self.total_frames = 0
        for (utterance_id, feats) in feature_archive:
            if feats.NumRows() == 0:
                logger.warning(f"Skipping {utterance_id} due to zero-length features")
                continue
            yield self.decode_utterance(feats, utterance_id)
        logger.info(
            f"Overall log-likelihood per frame is {self.total_likelihood / self.total_frames} "
            f"over {self.total_frames} frames."
        )
        logger.info(
            f"Retried {self.num_retry} out of {self.num_done + self.num_error} utterances."
        )
        logger.info(f"Done {self.num_done}, errors on {self.num_error}")

    def export_lattices(
        self,
        file_name: typing.Union[str, pathlib.Path],
        feature_archive: FeatureArchive,
        write_scp: bool = False,
        alignment_file_name: typing.Union[str, pathlib.Path] = None,
        word_file_name: typing.Union[str, pathlib.Path] = None,
        callback: typing.Callable = None,
    ):
        write_specifier = generate_write_specifier(file_name, write_scp)
        alignment_writer = None
        if alignment_file_name:
            alignment_write_specifier = generate_write_specifier(alignment_file_name, write_scp)
            alignment_writer = Int32VectorWriter(alignment_write_specifier)
        word_writer = None
        if word_file_name:
            word_write_specifier = generate_write_specifier(word_file_name, write_scp)
            word_writer = Int32VectorWriter(word_write_specifier)
        if self.config.determinize_lattice:
            writer = CompactLatticeWriter(write_specifier)
        else:
            writer = LatticeWriter(write_specifier)
        for transcription in self.decode_utterances(feature_archive):
            if transcription is None:
                continue
            if callback:
                callback((transcription.utterance_id, transcription.likelihood))
            writer.Write(str(transcription.utterance_id), transcription.lattice)
            if alignment_writer is not None:
                alignment_writer.Write(str(transcription.utterance_id), transcription.alignment)
            if word_writer is not None:
                word_writer.Write(str(transcription.utterance_id), transcription.words)
        writer.Close()
        if alignment_writer is not None:
            alignment_writer.Close()
        if word_writer is not None:
            word_writer.Close()


class GmmRescorer:
    def __init__(
        self,
        acoustic_model_path: typing.Union[pathlib.Path, str],
        acoustic_scale: float = 0.1,
        lattice_beam: float = 6.0,
    ):
        self.acoustic_model_path = acoustic_model_path
        self.transition_model, self.acoustic_model = read_gmm_model(self.acoustic_model_path)
        self.acoustic_scale = acoustic_scale
        self.lattice_beam = lattice_beam
        self.num_done = 0
        self.num_error = 0
        self.config = LatticeFasterDecoderConfig()

    def rescore_utterance(
        self, lattice: CompactLattice, feats: FloatMatrix, utterance_id: str = None
    ):

        try:
            ans, lattice = gmm_rescore_lattice(
                lattice, feats, self.acoustic_model, self.transition_model
            )
            if not ans:
                raise Exception(f"Error in rescoring {utterance_id}")

            clat = lattice_to_post(lattice, self.lattice_beam, self.acoustic_scale)
            self.num_done += 1
        except Exception:
            self.num_error += 1
            logger.warning(f"Error in rescoring {utterance_id}")
            raise
        return clat

    def rescore_utterances(
        self, lattice_archive: LatticeArchive, feature_archive: FeatureArchive
    ) -> typing.Generator[CompactLattice]:
        for (utterance_id, lattice) in lattice_archive:
            feats = feature_archive[utterance_id]
            yield utterance_id, self.rescore_utterance(lattice, feats, utterance_id)

    def export_lattices(
        self,
        file_name: typing.Union[str, pathlib.Path],
        lattice_archive: LatticeArchive,
        feature_archive: FeatureArchive,
        write_scp: bool = False,
        callback: typing.Callable = None,
    ):
        write_specifier = generate_write_specifier(file_name, write_scp)
        writer = CompactLatticeWriter(write_specifier)
        for utterance_id, lattice in self.rescore_utterances(lattice_archive, feature_archive):
            if callback:
                callback(utterance_id)
            writer.Write(str(utterance_id), lattice)
        writer.Close()
