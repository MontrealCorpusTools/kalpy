"""Classes for computing fMLLR transforms"""
from __future__ import annotations

import logging
import pathlib
import threading
import typing

from _kalpy import transform
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
        acoustic_model_path: typing.Union[pathlib.Path, str],
        silence_phones: typing.List[int],
        spk2utt: KaldiMapping = None,
        two_models: bool = True,
        weight_distribute: bool = False,
        fmllr_update_type: str = "full",
        silence_weight: float = 0.0,
        acoustic_scale: float = 1.0,
        fmllr_min_count: float = 500.0,
        fmllr_num_iters: int = 40,
        thread_lock: typing.Optional[threading.Lock] = None,
    ):
        self.acoustic_model_path = acoustic_model_path
        self.transition_model, self.acoustic_model = read_gmm_model(self.acoustic_model_path)
        self.spk2utt = spk2utt
        self.silence_weight = silence_weight
        self.acoustic_scale = acoustic_scale
        self.two_models = two_models
        self.silence_phones = silence_phones
        self.weight_distribute = weight_distribute
        self.fmllr_update_type = fmllr_update_type
        self.fmllr_min_count = fmllr_min_count
        self.fmllr_num_iters = fmllr_num_iters
        self.thread_lock = thread_lock

    def compute_fmllr(
        self,
        feature_archive: FeatureArchive,
        alignment_archive: typing.Union[AlignmentArchive, LatticeArchive],
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
                spk_stats = transform.FmllrDiagGmmAccs(am_dim, fmllr_options)
                logger.info(f"Processing speaker {spk}...")
                for utterance_id in utt_list:
                    try:
                        alignment = alignment_archive[utterance_id]
                    except KeyError:
                        logger.info(f"Skipping {utterance_id} due to missing lattice.")
                        continue
                    if use_alignment:
                        alignment = alignment_archive[utterance_id].alignment
                    feats = feature_archive[utterance_id]
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
                if self.thread_lock is not None:
                    self.thread_lock.acquire()
                trans = transform.compute_fmllr_transform(
                    spk_stats, self.acoustic_model.Dim(), fmllr_options
                )
                if self.thread_lock is not None:
                    self.thread_lock.release()
                num_done += 1

                yield spk, trans
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
        logger.info(f"Done {num_done} speakers.")
        logger.info(f"Skipped {num_skipped} speakers.")
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
            for speaker, trans in self.compute_fmllr(feature_archive, alignment_archive):
                if callback:
                    callback(speaker)
                if previous_transform_archive is not None:
                    if prev_reader.HasKey(speaker):
                        prev_trans = prev_reader.Value(speaker)
                        new_trans = transform.compose_transforms(prev_trans, trans, True)
                        trans = new_trans
                writer.Write(str(speaker), trans)
        finally:
            writer.Close()
