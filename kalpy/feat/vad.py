"""Classes for computing MFCC features"""
from __future__ import annotations

import logging
import pathlib
import typing

from _kalpy.ivector import ComputeVadEnergy, VadEnergyOptions
from _kalpy.matrix import FloatMatrix, FloatVector
from _kalpy.util import BaseFloatVectorWriter
from kalpy.feat.data import FeatureArchive
from kalpy.utils import generate_write_specifier

logger = logging.getLogger("kalpy.vad")
logger.setLevel(logging.DEBUG)
logger.write = lambda msg: logger.info(msg) if msg != "\n" else None
logger.flush = lambda: None


class VadComputer:
    """
    Class for computing VAD features

    Parameters
    ----------
    energy_threshold: float
        Constant term in energy threshold for MFCC0 for VAD
    energy_mean_scale: float
        If this is set to s, to get the actual threshold we let m be the mean log-energy of the file,
        and use s*m + energy_threshold
    frames_context: int
        Number of frames of context on each side of central frame, in window for which energy is monitored
    proportion_threshold: float
        Parameter controlling the proportion of frames within the window that need to have more energy than the threshold
    """

    def __init__(
        self,
        energy_threshold: float = 5.0,
        energy_mean_scale: float = 0.5,
        frames_context: int = 0,
        proportion_threshold: float = 0.6,
    ):
        self.energy_threshold = energy_threshold
        self.energy_mean_scale = energy_mean_scale
        self.frames_context = frames_context
        self.proportion_threshold = proportion_threshold
        self.num_done = 0
        self.num_skipped = 0
        self.options = VadEnergyOptions()
        self.options.vad_energy_threshold = self.energy_threshold
        self.options.vad_energy_mean_scale = self.energy_mean_scale
        self.options.vad_frames_context = self.frames_context
        self.options.vad_proportion_threshold = self.proportion_threshold

    def compute_vad(
        self,
        features: FloatMatrix,
    ) -> FloatVector:
        """
        Compute VAD features for a segment

        Parameters
        ----------
        features: :class:`~_kalpy.matrix.FloatMatrix`
            Feature matrix to compute VAD

        Returns
        -------
        :class:`~_kalpy.matrix.FloatVector`
            VAD for each
        """
        return ComputeVadEnergy(self.options, features)

    def compute_vads(
        self,
        feature_archive: FeatureArchive,
    ) -> typing.Tuple[str, FloatVector]:
        """
        Compute VAD features for a segment

        Parameters
        ----------
        feature_archive: :class:`~kalpy.feat.data.FeatureArchive`
            Archive of features to compute VAD

        Yields
        -------
        str, :class:`~_kalpy.matrix.FloatVector`
            VAD for each
        """
        for utterance_id, feats in feature_archive:
            if feats.NumRows() == 0:
                logger.debug(f"Skipping {utterance_id} due to empty features.")
                self.num_skipped += 1
                continue
            vad = self.compute_vad(feats)
            self.num_done += 1
            yield utterance_id, vad
        logger.info(f"Done {self.num_done} utterances, skipped {self.num_skipped}.")

    def export_vad(
        self,
        file_name: typing.Union[pathlib.Path, str],
        feature_archive: FeatureArchive,
        write_scp: bool = False,
        callback: typing.Callable = None,
    ):
        """
        Export features to a kaldi archive file (i.e., mfccs.ark)

        Parameters
        ----------
        file_name: :class:`~pathlib.Path` or str
            Archive file path to export to
        feature_archive: :class:`~kalpy.feat.data.FeatureArchive`
            Archive of features to compute VAD
        write_scp: bool
            Flag for whether an SCP file should be generated as well
        callback: typing.Callable
            Flag for yielding processed utterance IDs
        """
        write_specifier = generate_write_specifier(file_name, write_scp)
        logger.debug(f"Writing to: {write_specifier}")
        writer = BaseFloatVectorWriter(write_specifier)
        for key, vad in self.compute_vads(feature_archive):
            if callback:
                callback(key)
            writer.Write(str(key), vad)
        writer.Close()
