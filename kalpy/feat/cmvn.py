"""Classes for computing MFCC features"""
from __future__ import annotations

import logging
import pathlib
import typing

import numpy as np

from _kalpy import transform
from _kalpy.matrix import CompressedMatrix, DoubleMatrix, FloatMatrix
from _kalpy.util import (
    BaseDoubleMatrixWriter,
    CompressedMatrixWriter,
    RandomAccessBaseFloatMatrixReader,
)
from kalpy.data import KaldiMapping
from kalpy.feat.data import FeatureArchive
from kalpy.utils import generate_write_specifier

logger = logging.getLogger("kalpy.cmvn")
logger.setLevel(logging.DEBUG)
logger.write = lambda msg: logger.info(msg) if msg != "\n" else None
logger.flush = lambda: None


class CmvnComputer:
    """
    Class for computing CMVN for features

    Parameters
    ----------
    online: bool

    """

    def __init__(
        self,
        online: bool = False,
    ):
        self.online = online
        self.num_done = 0
        self.num_error = 0

    def compute_cmvn(
        self,
        utterance_list: typing.List[str],
        feature_reader: RandomAccessBaseFloatMatrixReader,
    ) -> np.ndarray:
        """
        Calculate CMVN for a set of utterances

        Parameters
        ----------
        utterance_list: list[str]
            List of utterances to compute CMVN for
        feature_reader: :class:`~_kalpy.util.RandomAccessBaseFloatMatrixReader`
            Reader object for feature file

        Returns
        -------
        :class:`numpy.ndarray`
            Feature matrix for the segment
        """
        cmvn = self.compute_cmvn_for_export(utterance_list, feature_reader)
        return cmvn.numpy()

    def compute_cmvn_from_features(self, features):
        cmvn_stats = DoubleMatrix()
        for i, feats in enumerate(features):
            if i == 0:
                transform.InitCmvnStats(feats.NumCols(), cmvn_stats)
            transform.AccCmvnStats(feats, None, cmvn_stats)
        return cmvn_stats

    def compute_cmvn_for_export(
        self,
        utterance_list: typing.List[str],
        feature_reader: RandomAccessBaseFloatMatrixReader,
    ) -> FloatMatrix:
        """
        Generate MFCCs for exporting to a kaldi archive

        Parameters
        ----------
        utterance_list: list[str]
            List of utterances to compute CMVN for
        feature_reader: :class:`~_kalpy.util.RandomAccessBaseFloatMatrixReader`
            Reader object for feature file

        Returns
        -------
        :class:`_kalpy.matrix.FloatMatrixBase`
            Feature matrix for the segment
        """
        cmvn, num_done, num_error = transform.calculate_cmvn(utterance_list, feature_reader)
        self.num_done += num_done
        self.num_error += num_error
        return cmvn

    def export_cmvn(
        self,
        file_name: typing.Union[pathlib.Path, str],
        feature_archive: FeatureArchive,
        spk2utt: KaldiMapping,
        write_scp: bool = False,
        compress: bool = False,
    ) -> None:
        """
        Export features to a kaldi archive file (i.e., cmvn.ark)

        Parameters
        ----------
        file_name: :class:`~pathlib.Path` or str
            Archive file path to export to
        feature_archive: :class:`~kalpy.feat.data.FeatureArchive`
            Archive of features
        spk2utt: :class:`~kalpy.data.KaldiMapping`
            Mapping of speaker ids to utterance ids
        write_scp: bool
            Flag for whether an SCP file should be generated as well
        compress: bool
            Flag for whether to export features as a compressed archive
        """
        write_specifier = generate_write_specifier(file_name, write_scp)
        logger.debug(f"Writing to: {write_specifier}")
        if compress:
            writer = CompressedMatrixWriter(write_specifier)
        else:
            writer = BaseDoubleMatrixWriter(write_specifier)
        feat_reader = feature_archive.archive.random_reader
        for key, utterance_list in spk2utt.items():
            logger.info(f"Processing speaker {key}: {len(utterance_list)} utterances")
            feats = self.compute_cmvn_for_export(utterance_list, feat_reader)
            if compress:
                feats = CompressedMatrix(feats)
            writer.Write(str(key), feats)
        writer.Close()
