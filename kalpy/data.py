"""Data classes for kalpy"""
from __future__ import annotations

import os.path
import pathlib
import typing

import dataclassy
import librosa
import numpy as np

from _kalpy.matrix import FloatMatrix
from _kalpy.util import (
    RandomAccessBaseDoubleMatrixReader,
    RandomAccessBaseFloatMatrixReader,
    SequentialBaseDoubleMatrixReader,
    SequentialBaseFloatMatrixReader,
)
from kalpy.utils import generate_read_specifier

PathLike = typing.Union[str, pathlib.Path]


@dataclassy.dataclass
class Segment:
    """
    Data class for information about acoustic segments
    """

    file_path: str
    begin: typing.Optional[float] = None
    end: typing.Optional[float] = None
    channel: typing.Optional[int] = 0

    def load_audio(self):
        duration = self.end - self.begin
        y, _ = librosa.load(
            self.file_path,
            sr=16000,
            offset=self.begin,
            duration=duration,
            mono=False,
        )
        if len(y.shape) > 1:
            channel = 0 if self.channel is None else self.channel
            y = y[channel, :]
        return y

    @property
    def wave(self):
        if getattr(self, "_wave", None) is None:
            self._wave = self.load_audio()
        return self._wave

    @property
    def kaldi_wave(self):
        return np.round(self.wave * 32768)


def make_scp_safe(string: str) -> str:
    """
    Helper function to make a string safe for saving in Kaldi scp files.  They use space as a delimiter, so
    any spaces in the string will be converted to "_KALDISPACE_" to preserve them

    Parameters
    ----------
    string: str
        Text to escape

    Returns
    -------
    str
        Escaped text
    """
    return str(string).replace(" ", "_KALDISPACE_")


def load_scp_safe(string: str) -> str:
    """
    Helper function to load previously made safe text.  All instances of "_KALDISPACE_" will be converted to a
    regular space character

    Parameters
    ----------
    string: str
        String to convert

    Returns
    -------
    str
        Converted string
    """
    return string.replace("_KALDISPACE_", " ")


class KaldiMapping(dict):
    def __init__(self, *args, list_mapping: bool = False, **kwargs):
        self.list_mapping = list_mapping
        super().__init__(*args, **kwargs)

    def export(self, file_path: typing.Union[str, pathlib.Path], skip_safe: bool = False) -> None:
        with open(file_path, "w", encoding="utf8") as f:
            for k in sorted(self.keys()):
                v = self[k]
                if isinstance(v, (list, set, tuple)):
                    v = " ".join(map(str, v))
                elif not skip_safe:
                    v = make_scp_safe(v)
                f.write(f"{make_scp_safe(k)} {v}\n")

    def load(self, file_path, data_type: typing.Optional[typing.Type] = str):
        self.clear()
        with open(file_path, "r", encoding="utf8") as f:
            for line in f:
                line = line.strip()
                if line == "":
                    continue
                line_list = line.split()
                key = load_scp_safe(line_list.pop(0))
                if len(line_list) == 1 and not self.list_mapping:
                    value = data_type(line_list[0])
                    if isinstance(value, str):
                        value = load_scp_safe(value)
                else:
                    value = [data_type(x) for x in line_list if x not in ["[", "]"]]
                self[key] = value


class MatrixArchive:
    """
    Class for reading an archive or SCP of matrices

    Parameters
    ----------
    file_name: :class:`~pathlib.Path` or str
        Path to archive or SCP file to read from
    """

    def __init__(self, file_name: typing.Union[pathlib.Path, str], double: bool = False):
        if not os.path.exists(file_name):
            raise OSError(f"Specified file does not exist: {file_name}")
        self.file_name = str(file_name)
        self.double = double
        self.read_specifier = generate_read_specifier(file_name)
        if self.double:
            self.random_reader = RandomAccessBaseDoubleMatrixReader(self.read_specifier)
        else:
            self.random_reader = RandomAccessBaseFloatMatrixReader(self.read_specifier)

    def close(self):
        if getattr(self, "random_reader", None) is not None and self.random_reader.IsOpen():
            self.random_reader.Close()

    def __del__(self):
        self.close()

    @property
    def sequential_reader(self) -> SequentialBaseFloatMatrixReader:
        if self.double:
            return SequentialBaseDoubleMatrixReader(self.read_specifier)
        return SequentialBaseFloatMatrixReader(self.read_specifier)

    def __iter__(self) -> typing.Generator[typing.Tuple[str, FloatMatrix]]:
        """Iterate over the utterance features in the archive"""
        reader = self.sequential_reader
        try:
            while not reader.Done():
                utt = reader.Key()
                feats = reader.Value()
                yield utt, feats
                reader.Next()
        finally:
            reader.Close()

    def __getitem__(self, item: str) -> FloatMatrix:
        """Get features for a particular key from the archive file"""
        item = str(item)
        if not self.random_reader.HasKey(item):
            raise KeyError(f"No key {item} found in {self.file_name}")
        return self.random_reader.Value(item)
