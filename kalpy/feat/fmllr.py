"""Classes for computing fMLLR transforms"""
from __future__ import annotations

import logging
import pathlib
import threading
import typing

from _kalpy import gmm, hmm, transform
from _kalpy.util import BaseFloatMatrixWriter, ConstIntegerSet
from kalpy.data import KaldiMapping, MatrixArchive
from kalpy.feat.data import FeatureArchive
from kalpy.gmm.data import AlignmentArchive, LatticeArchive
from kalpy.gmm.utils import read_gmm_model
from kalpy.utils import generate_write_specifier

logger = logging.getLogger("kalpy.fmllr")
logger.setLevel(logging.DEBUG)
logger.write = lambda msg: logger.info(msg) if msg != "\n" else None
logger.flush = lambda: None


class FmllrComputer:
    def __init__(
        self,
        alignment_acoustic_model_path: typing.Union[pathlib.Path, str],
        acoustic_model_path: typing.Union[pathlib.Path, str],
        silence_phones: typing.List[int],
        spk2utt: KaldiMapping = None,
        weight_distribute: bool = False,
        fmllr_update_type: str = "full",
        silence_weight: float = 0.0,
        acoustic_scale: float = 1.0,
        fmllr_min_count: float = 500.0,
        fmllr_num_iters: int = 40,
        thread_lock: typing.Optional[threading.Lock] = None,
    ):
        self.acoustic_model_path = acoustic_model_path
        self.alignment_acoustic_model_path = alignment_acoustic_model_path
        self.transition_model, self.acoustic_model = read_gmm_model(self.acoustic_model_path)
        self.two_models = self.alignment_acoustic_model_path != self.acoustic_model_path
        if self.two_models:
            self.alignment_transition_model, self.alignment_acoustic_model = read_gmm_model(
                self.alignment_acoustic_model_path
            )
        else:
            self.alignment_transition_model, self.alignment_acoustic_model = (
                self.transition_model,
                self.acoustic_model,
            )
        self.spk2utt = spk2utt
        self.silence_weight = silence_weight
        self.acoustic_scale = acoustic_scale
        self.silence_phones = silence_phones
        self.weight_distribute = weight_distribute
        self.fmllr_update_type = fmllr_update_type
        self.fmllr_min_count = fmllr_min_count
        self.fmllr_num_iters = fmllr_num_iters
        self.thread_lock = thread_lock
        self.callback_frequency = 100

    def compute_fmllr(
        self,
        feature_archive: FeatureArchive,
        alignment_archive: typing.Union[AlignmentArchive, LatticeArchive],
        callback: typing.Callable = None,
    ):
        fmllr_options = transform.FmllrOptions()
        fmllr_options.update_type = self.fmllr_update_type
        fmllr_options.min_count = self.fmllr_min_count
        fmllr_options.num_iters = self.fmllr_num_iters
        use_alignment = isinstance(alignment_archive, AlignmentArchive)
        rand_prune = 0.0
        tot_impr = 0.0
        tot_t = 0.0
        num_done = 0
        num_skipped = 0
        am_dim = self.acoustic_model.Dim()
        silence_set = ConstIntegerSet(self.silence_phones)
        if self.spk2utt is not None:
            for spk, utt_list in self.spk2utt.items():
                spk_stats = transform.FmllrDiagGmmAccs(am_dim)
                logger.info(f"Processing speaker {spk}...")
                for utterance_id in utt_list:
                    try:
                        alignment = alignment_archive[utterance_id]
                    except KeyError:
                        logger.info(f"Skipping {utterance_id} due to missing alignment.")
                        num_skipped += 1
                        continue
                    if use_alignment:
                        alignment = alignment_archive[utterance_id].alignment
                    try:
                        feats = feature_archive[utterance_id]
                    except KeyError:
                        logger.info(f"Skipping {utterance_id} due to missing features.")
                        num_skipped += 1
                        continue
                    if feats.NumRows() == 0:
                        logger.warning(f"Skipping {utterance_id} due to zero-length features")
                        num_skipped += 1
                        continue
                    num_done += 1
                    if use_alignment:
                        spk_stats.accumulate_from_alignment(
                            self.alignment_transition_model,
                            self.alignment_acoustic_model,
                            self.transition_model,
                            self.acoustic_model,
                            feats,
                            alignment,
                            silence_set,
                            self.silence_weight,
                            rand_prune=rand_prune,
                            distributed=self.weight_distribute,
                            two_models=self.two_models,
                        )
                    else:
                        spk_stats.accumulate_from_lattice(
                            self.alignment_transition_model,
                            self.alignment_acoustic_model,
                            self.transition_model,
                            self.acoustic_model,
                            feats,
                            alignment,
                            silence_set,
                            self.silence_weight,
                            acoustic_scale=self.acoustic_scale,
                            rand_prune=rand_prune,
                            distributed=self.weight_distribute,
                            two_models=self.two_models,
                        )
                    if callback is not None and num_done % self.callback_frequency == 0:
                        callback(self.callback_frequency)
                if self.thread_lock is not None:
                    self.thread_lock.acquire()
                trans, impr, spk_tot_t = spk_stats.compute_transform(
                    self.acoustic_model, fmllr_options
                )
                if self.thread_lock is not None:
                    self.thread_lock.release()
                if spk_tot_t:
                    logger.debug(
                        f"For speaker {spk}, auxf-impr from fMLLR is {impr/spk_tot_t}, over {spk_tot_t} frames."
                    )
                    tot_impr += impr
                    tot_t += spk_tot_t
                    yield spk, trans
                else:
                    logger.debug(f"Skipping speaker {spk} due to no data")
        else:
            for utterance_id, feats in feature_archive:

                try:
                    alignment = alignment_archive[utterance_id]
                except KeyError:
                    logger.info(f"Skipping {utterance_id} due to missing alignment.")
                    num_skipped += 1
                    continue

                if feats.NumRows() == 0:
                    logger.warning(f"Skipping {utterance_id} due to zero-length features")
                    num_skipped += 1
                    continue
                spk_stats = transform.FmllrDiagGmmAccs(am_dim)
                if use_alignment:
                    spk_stats.accumulate_from_alignment(
                        self.transition_model,
                        self.acoustic_model,
                        feats,
                        alignment,
                        silence_set,
                        self.silence_weight,
                        rand_prune=rand_prune,
                        distributed=self.weight_distribute,
                        two_models=self.two_models,
                    )
                else:
                    spk_stats.accumulate_from_lattice(
                        self.transition_model,
                        self.acoustic_model,
                        feats,
                        alignment,
                        silence_set,
                        self.silence_weight,
                        acoustic_scale=self.acoustic_scale,
                        rand_prune=rand_prune,
                        distributed=self.weight_distribute,
                        two_models=self.two_models,
                    )
                trans = transform.compute_fmllr_transform(
                    spk_stats, self.acoustic_model.Dim(), fmllr_options
                )
                num_done += 1

                yield utterance_id, trans
                if callback is not None and num_done % self.callback_frequency == 0:
                    callback(self.callback_frequency)
        if callback is not None and num_done % self.callback_frequency:
            callback(num_done % self.callback_frequency)
        logger.info(f"Done {num_done} utterances.")
        logger.info(f"Skipped {num_skipped} utterances.")
        if tot_t:
            logger.info(
                f"Overall fMLLR auxf impr per frame is {tot_impr / tot_t} over {tot_t} frames."
            )

    def export_transforms(
        self,
        file_name: typing.Union[pathlib.Path, str],
        feature_archive: FeatureArchive,
        alignment_archive: typing.Union[AlignmentArchive, LatticeArchive],
        previous_transform_archive: MatrixArchive = None,
        write_scp: bool = False,
        callback: typing.Callable = None,
    ):
        write_specifier = generate_write_specifier(file_name, write_scp)
        writer = BaseFloatMatrixWriter(write_specifier)
        try:
            prev_reader = None
            if previous_transform_archive is not None:
                prev_reader = previous_transform_archive.random_reader
            for speaker, trans in self.compute_fmllr(
                feature_archive, alignment_archive, callback=callback
            ):
                if previous_transform_archive is not None:
                    if prev_reader.HasKey(speaker):
                        prev_trans = prev_reader.Value(speaker)
                        new_trans = transform.compose_transforms(trans, prev_trans, True)
                        trans = new_trans
                writer.Write(str(speaker), trans)
        finally:
            writer.Close()
