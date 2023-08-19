"""Classes for extracting ivectors"""
from __future__ import annotations

import logging
import pathlib
import typing

from _kalpy.gmm import DiagGmm
from _kalpy.ivector import IvectorExtractor as KaldiIvectorExtractor
from _kalpy.ivector import ivector_extract
from _kalpy.util import BaseDoubleVectorWriter
from kalpy.feat.data import FeatureArchive
from kalpy.utils import generate_write_specifier, read_kaldi_object

logger = logging.getLogger("kalpy.ivector")
logger.setLevel(logging.DEBUG)
logger.write = lambda msg: logger.info(msg) if msg != "\n" else None
logger.flush = lambda: None


class IvectorExtractor:
    def __init__(
        self,
        dubm_path: typing.Union[str, pathlib.Path],
        ivector_extractor_path: typing.Union[str, pathlib.Path],
        acoustic_weight: float = 1.0,
        max_count: float = 0.0,
        num_gselect: int = 50,
        min_post: float = 0.0,
    ):
        self.dubm_path = str(dubm_path)
        self.ivector_extractor_path = str(ivector_extractor_path)
        self.ivector_extractor = read_kaldi_object(
            KaldiIvectorExtractor, self.ivector_extractor_path
        )
        self.dubm = read_kaldi_object(DiagGmm, self.dubm_path)

        self.acoustic_weight = acoustic_weight
        self.max_count = max_count
        self.num_gselect = num_gselect
        self.min_post = min_post

    def extract_ivectors(self, feature_archive: FeatureArchive):
        num_done = 0
        num_error = 0
        for utt_id, feats in feature_archive:
            if feats.NumRows() == 0:
                logger.warning(f"Skipping {utt_id} due to zero-length features")
                num_error += 1
                continue
            try:
                ivector = ivector_extract(
                    self.dubm,
                    self.ivector_extractor,
                    feats,
                    acoustic_weight=self.acoustic_weight,
                    max_count=self.max_count,
                    num_post=self.num_gselect,
                    min_post=self.min_post,
                )
            except Exception:
                num_error += 1
                continue
            yield utt_id, ivector
            num_done += 1
        logger.info(f"Done {num_done} utterances, errors on {num_error}.")

    def export_ivectors(
        self,
        file_name: typing.Union[pathlib.Path, str],
        feature_archive: FeatureArchive,
        write_scp: bool = False,
        callback: typing.Callable = None,
    ):
        write_specifier = generate_write_specifier(file_name, write_scp)
        writer = BaseDoubleVectorWriter(write_specifier)
        try:
            for utt_id, ivector in self.extract_ivectors(feature_archive):
                if callback:
                    callback(utt_id)
                writer.Write(str(utt_id), ivector)
        finally:
            writer.Close()
