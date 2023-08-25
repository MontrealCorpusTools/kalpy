"""Data classes for GMM"""
from __future__ import annotations

import os
import typing

from _kalpy.matrix import FloatVector
from _kalpy.util import (
    RandomAccessBaseFloatVectorReader,
    RandomAccessInt32VectorVectorReader,
    SequentialBaseFloatVectorReader,
    SequentialInt32VectorVectorReader,
)
from kalpy.data import PathLike
from kalpy.utils import generate_read_specifier, read_kaldi_object


class IvectorArchive:
    """
    Class for reading an archive or SCP of ivector

    Parameters
    ----------
    file_name: :class:`~pathlib.Path` or str
        Path to archive or SCP file to read from
    """

    def __init__(self, file_name: PathLike, num_utterances_file_name: PathLike = None):
        if not os.path.exists(file_name):
            raise OSError(f"Specified file does not exist: {file_name}")
        self.file_name = str(file_name)
        self.num_utterances_file_name = num_utterances_file_name
        self.read_specifier = generate_read_specifier(file_name)
        self.random_reader = RandomAccessBaseFloatVectorReader(self.read_specifier)
        self.num_utterances_mapping = {}
        if self.num_utterances_file_name is not None:
            with open(self.num_utterances_file_name) as f:
                for line in f:
                    line = line.strip()
                    speaker_id, num_utts = line.split()
                    self.num_utterances_mapping[speaker_id] = int(num_utts)

    def close(self):
        if self.random_reader.IsOpen():
            self.random_reader.Close()

    @property
    def sequential_reader(self) -> SequentialBaseFloatVectorReader:
        """Sequential reader for lattices"""
        return SequentialBaseFloatVectorReader(self.read_specifier)

    def __iter__(self) -> typing.Generator[typing.Tuple[str, FloatVector]]:
        """Iterate over the utterance lattices in the archive"""
        if self.read_specifier.startswith("scp"):
            with open(self.file_name, encoding="utf8") as f:
                for line in f:
                    line = line.strip()
                    key, ark_path = line.split(maxsplit=1)
                    ivector = read_kaldi_object(FloatVector, ark_path)
                    num_utterances = self.num_utterances_mapping.get(key, 1)
                    yield key, ivector, num_utterances
        else:
            reader = self.sequential_reader
            try:
                while not reader.Done():
                    key = reader.Key()
                    ivector = reader.Value()
                    num_utterances = self.num_utterances_mapping.get(key, 1)
                    yield key, ivector, num_utterances
                    reader.Next()
            finally:
                reader.Close()

    def __del__(self):
        self.close()

    def __getitem__(self, item: str) -> FloatVector:
        """Get lattice for a particular key from the archive file"""
        item = str(item)
        if not self.random_reader.HasKey(item):
            raise KeyError(f"No key {item} found in {self.file_name}")
        return self.random_reader.Value(item)


class GselectArchive:
    """
    Class for reading an archive or SCP of gaussian selections

    Parameters
    ----------
    file_name: :class:`~pathlib.Path` or str
        Path to archive or SCP file to read from
    """

    def __init__(self, file_name: PathLike):
        if not os.path.exists(file_name):
            raise OSError(f"Specified file does not exist: {file_name}")
        self.file_name = str(file_name)
        self.read_specifier = generate_read_specifier(file_name)
        self.random_reader = RandomAccessInt32VectorVectorReader(self.read_specifier)

    def close(self):
        if self.random_reader.IsOpen():
            self.random_reader.Close()

    @property
    def sequential_reader(self) -> SequentialInt32VectorVectorReader:
        """Sequential reader for lattices"""
        return SequentialInt32VectorVectorReader(self.read_specifier)

    def __iter__(self) -> typing.Generator[typing.Tuple[str, typing.List[typing.List[int]]]]:
        """Iterate over the utterance lattices in the archive"""
        reader = self.sequential_reader
        try:
            while not reader.Done():
                utt = reader.Key()
                value = reader.Value()
                yield utt, value
                reader.Next()
        finally:
            reader.Close()

    def __del__(self):
        self.close()

    def __getitem__(self, item: str) -> typing.List[typing.List[int]]:
        """Get lattice for a particular key from the archive file"""
        item = str(item)
        if not self.random_reader.HasKey(item):
            raise KeyError(f"No key {item} found in {self.file_name}")
        return self.random_reader.Value(item)
