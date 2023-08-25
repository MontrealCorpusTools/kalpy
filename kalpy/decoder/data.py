"""Classes for storing graph archives"""
from __future__ import annotations

import os.path
import pathlib
import typing

from _kalpy.fstext import RandomAccessVectorFstReader, SequentialVectorFstReader, VectorFst
from kalpy.utils import generate_read_specifier


class FstArchive:
    """
    Class for reading an archive or SCP of FSTs

    Parameters
    ----------
    file_name: :class:`~pathlib.Path` or str
        Path to archive or SCP file to read from
    """

    def __init__(self, file_name: typing.Union[pathlib.Path, str]):
        if not os.path.exists(file_name):
            raise OSError(f"Specified file does not exist: {file_name}")
        self.file_name = str(file_name)
        self.read_specifier = generate_read_specifier(file_name)
        self.random_reader = RandomAccessVectorFstReader(self.read_specifier)

    @property
    def sequential_reader(self) -> SequentialVectorFstReader:
        return SequentialVectorFstReader(self.read_specifier)

    def __iter__(self) -> typing.Generator[typing.Tuple[str, VectorFst]]:
        """Iterate over the utterance FSTs in the archive"""
        reader = self.sequential_reader
        try:
            while not reader.Done():
                utt = reader.Key()
                fst = reader.Value()
                decode_fst = VectorFst(fst)
                yield utt, decode_fst
                reader.Next()
        finally:
            reader.Close()

    def __del__(self):
        if self.random_reader.IsOpen():
            self.random_reader.Close()

    def __getitem__(self, item: str) -> VectorFst:
        """Get FST for a particular key from the archive file"""
        item = str(item)
        if not self.random_reader.HasKey(item):
            raise KeyError(f"No key {item} found in {self.file_name}")
        return self.random_reader.Value(item)
