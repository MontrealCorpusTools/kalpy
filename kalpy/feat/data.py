"""Classes for storing and processing features"""
from __future__ import annotations

import os
import pathlib
import typing

from _kalpy import feat, transform
from _kalpy.matrix import FloatMatrix
from _kalpy.util import (
    RandomAccessBaseDoubleMatrixReader,
    RandomAccessBaseFloatMatrixReader,
    RandomAccessBaseFloatVectorReader,
)
from kalpy.data import KaldiMapping, MatrixArchive
from kalpy.utils import generate_read_specifier, read_kaldi_object


class FeatureArchive:
    def __init__(
        self,
        file_name: typing.Union[pathlib.Path, str],
        utt2spk: KaldiMapping = None,
        cmvn_file_name: typing.Union[pathlib.Path, str] = None,
        lda_mat_file_name: typing.Union[pathlib.Path, str] = None,
        transform_file_name: typing.Union[pathlib.Path, str] = None,
        vad_file_name: typing.Union[pathlib.Path, str] = None,
        use_sliding_cmvn: bool = False,
        cmvn_norm_vars: bool = False,
        cmvn_reverse: bool = False,
        deltas: bool = False,
        splices: bool = False,
        splice_frames: int = 3,
        subsample_n: int = 0,
        sliding_cmvn_window: int = 300,
        sliding_cmvn_center_window: bool = True,
        double: bool = False,
    ):
        self.cmvn_reader = None
        self.transform_reader = None
        self.vad_reader = None
        if not os.path.exists(file_name):
            raise OSError(f"Specified file does not exist: {file_name}")
        self.archive = MatrixArchive(file_name, double=double)
        self.utt2spk = utt2spk
        self.subsample_n = subsample_n

        self.use_sliding_cmvn = use_sliding_cmvn
        self.cmvn_norm_vars = cmvn_norm_vars
        self.cmvn_reverse = cmvn_reverse

        self.sliding_cmvn_options = feat.SlidingWindowCmnOptions()
        self.sliding_cmvn_options.cmn_window = sliding_cmvn_window
        self.sliding_cmvn_options.center = sliding_cmvn_center_window
        self.sliding_cmvn_options.normalize_variance = cmvn_norm_vars

        self.delta_options = feat.DeltaFeaturesOptions()
        self.splice_frames = splice_frames
        self.use_deltas = deltas
        self.use_splices = splices
        self.cmvn_file_name = cmvn_file_name
        if cmvn_file_name:
            cmvn_read_specifier = generate_read_specifier(cmvn_file_name)
            self.cmvn_reader = RandomAccessBaseDoubleMatrixReader(cmvn_read_specifier)

        self.lda_mat_file_name = lda_mat_file_name
        self.lda_mat = None
        if lda_mat_file_name:
            self.use_splices = True
            self.use_deltas = False
            self.lda_mat_file_name = str(lda_mat_file_name)
            self.lda_mat = read_kaldi_object(FloatMatrix, self.lda_mat_file_name)
        self.transform_file_name = transform_file_name
        if transform_file_name:
            transform_read_specifier = generate_read_specifier(transform_file_name)
            self.transform_reader = RandomAccessBaseFloatMatrixReader(transform_read_specifier)

        self.vad_file_name = vad_file_name
        if vad_file_name:
            vad_read_specifier = generate_read_specifier(vad_file_name)
            self.vad_reader = RandomAccessBaseFloatVectorReader(vad_read_specifier)
        self.current_speaker = None
        self.trans = None
        self.cmvn_stats = None

    def __del__(self):
        self.close()

    def close(self):
        if getattr(self, "archive", None) is not None and self.archive.random_reader.IsOpen():
            self.archive.random_reader.Close()
        if self.cmvn_reader is not None and self.cmvn_reader.IsOpen():
            self.cmvn_reader.Close()
        if self.transform_reader is not None and self.transform_reader.IsOpen():
            self.transform_reader.Close()
        if self.vad_reader is not None and self.vad_reader.IsOpen():
            self.vad_reader.Close()

    def __iter__(self) -> typing.Generator[typing.Tuple[str, FloatMatrix]]:
        """Iterate over the utterance features in the archive"""
        reader = self.archive.sequential_reader
        self.current_speaker = None
        self.trans = None
        self.cmvn_stats = None
        try:
            while not reader.Done():
                utt = reader.Key()
                feats = reader.Value()
                if self.utt2spk is not None:
                    speaker = self.utt2spk[utt]
                else:
                    speaker = None
                # Apply CMVN
                if self.cmvn_file_name and speaker is not None:
                    if self.current_speaker != speaker:
                        if not self.cmvn_reader.HasKey(speaker):
                            raise Exception(
                                f"Could not find key {speaker} in {self.cmvn_file_name}"
                            )
                        self.cmvn_stats = self.cmvn_reader.Value(speaker)
                        if self.transform_reader is not None and self.transform_reader.HasKey(
                            speaker
                        ):
                            self.trans = self.transform_reader.Value(speaker)
                        self.current_speaker = speaker
                    if self.cmvn_reverse:
                        transform.ApplyCmvnReverse(self.cmvn_stats, self.cmvn_norm_vars, feats)
                    else:
                        transform.ApplyCmvn(self.cmvn_stats, self.cmvn_norm_vars, feats)
                elif self.use_sliding_cmvn:
                    feats = feat.sliding_window_cmn(self.sliding_cmvn_options, feats)

                # Deltas or splices
                if self.use_deltas:
                    feats = feat.compute_deltas(self.delta_options, feats)
                elif self.use_splices:
                    feats = feat.splice_frames(feats, self.splice_frames, self.splice_frames)
                    if self.lda_mat is not None:
                        feats = transform.apply_transform(feats, self.lda_mat)

                # Speaker adapted features
                if self.trans is not None:
                    feats = transform.apply_transform(feats, self.trans)

                # Subsampling
                if self.vad_reader is not None and self.vad_reader.HasKey(utt):
                    vad = self.vad_reader.Value(utt)
                    feats = feat.select_voiced_frames(feats, vad)
                if self.subsample_n and self.subsample_n > 0:
                    feats = feat.subsample_feats(feats, n=self.subsample_n)
                yield utt, feats
                reader.Next()
        finally:
            reader.Close()

    def __getitem__(self, item: str) -> FloatMatrix:
        """Get features for a particular key from the archive file"""
        item = str(item)
        if not self.archive.random_reader.HasKey(item):
            raise KeyError(f"No key {item} found in {self.archive.file_name}")
        feats = self.archive.random_reader.Value(item)
        if self.utt2spk is not None:
            speaker = self.utt2spk[item]
        else:
            speaker = None
        # Apply CMVN
        if self.cmvn_reader is not None and speaker is not None:
            if self.current_speaker != speaker:
                if not self.cmvn_reader.HasKey(speaker):
                    raise Exception(f"Could not find key {speaker} in {self.cmvn_file_name}")
                self.cmvn_stats = self.cmvn_reader.Value(speaker)
                if self.transform_reader is not None and self.transform_reader.HasKey(speaker):
                    self.trans = self.transform_reader.Value(speaker)
                self.current_speaker = speaker
            if self.cmvn_reverse:
                transform.ApplyCmvnReverse(self.cmvn_stats, self.cmvn_norm_vars, feats)
            else:
                transform.ApplyCmvn(self.cmvn_stats, self.cmvn_norm_vars, feats)
        elif self.use_sliding_cmvn:
            feats = feat.sliding_window_cmn(self.sliding_cmvn_options, feats)

        # Deltas or splices
        if self.use_deltas:
            feats = feat.compute_deltas(self.delta_options, feats)
        elif self.use_splices:
            feats = feat.splice_frames(feats, self.splice_frames, self.splice_frames)
            if self.lda_mat_file_name:
                feats = transform.apply_transform(feats, self.lda_mat)

        # Speaker adapted features
        if self.trans is not None:
            feats = transform.apply_transform(feats, self.trans)

        # Subsampling
        if self.vad_reader is not None and self.vad_reader.HasKey(item):
            vad = self.vad_reader.Value(item)
            feats = feat.select_voiced_frames(feats, vad)
        if self.subsample_n and self.subsample_n > 0:
            feats = feat.subsample_feats(feats, n=self.subsample_n)
        return feats
