"""Data classes for GMM"""
from __future__ import annotations

import os
import pathlib
import sys
import typing

import dataclassy
import numpy as np
import pywrapfst
from praatio import textgrid as tgio
from praatio.data_classes.textgrid import _tgToDictionary
from praatio.utilities.constants import Interval, TextgridFormats

from _kalpy.fstext import GetLinearSymbolSequence, VectorFst
from _kalpy.hmm import SplitToPhones, TransitionModel
from _kalpy.lat import (
    CompactLattice,
    CompactLatticeToWordAlignment,
    Lattice,
    RandomAccessCompactLatticeReader,
    RandomAccessLatticeReader,
    SequentialCompactLatticeReader,
    SequentialLatticeReader,
    WordAlignLatticeLexiconInfo,
    WordAlignLatticeLexiconOpts,
    lattice_best_path,
    linear_to_lattice,
    word_align_lattice_lexicon,
)
from _kalpy.matrix import FloatVector
from _kalpy.util import (
    RandomAccessBaseFloatVectorReader,
    RandomAccessInt32VectorReader,
    SequentialBaseFloatVectorReader,
    SequentialInt32VectorReader,
)
from kalpy.exceptions import CtmError
from kalpy.utils import generate_read_specifier


# noinspection PyUnresolvedReferences
@dataclassy.dataclass(slots=True)
class CtmInterval:
    """
    Data class for intervals derived from CTM files

    Parameters
    ----------
    begin: float
        Start time of interval
    end: float
        End time of interval
    label: str
        Text of interval
    confidence: float, optional
        Confidence score of the interval
    """

    begin: float
    end: float
    label: str
    symbol: int
    confidence: typing.Optional[float] = None

    def __lt__(self, other: CtmInterval):
        """Sorting function for CtmIntervals"""
        return self.begin < other.begin

    def __add__(self, other):
        if isinstance(other, str):
            return self.label + other
        else:
            self.begin += other
            self.end += other

    def __post_init__(self) -> None:
        """
        Check on data validity

        Raises
        ------
        :class:`~montreal_forced_aligner.exceptions.CtmError`
            If begin or end are not valid
        """
        if self.end < -1 or self.begin == 1000000:
            raise CtmError(self)

    def to_tg_interval(self, file_duration=None) -> Interval:
        """
        Converts the CTMInterval to
        `PraatIO's Interval class <http://timmahrt.github.io/praatIO/praatio/utilities/constants.html#Interval>`_

        Returns
        -------
        :class:`praatio.utilities.constants.Interval`
            Derived PraatIO Interval
        """
        if self.end < -1 or self.begin == 1000000:
            raise CtmError(self)
        end = round(self.end, 6)
        begin = round(self.begin, 6)
        if file_duration is not None and end > file_duration:
            end = round(file_duration, 6)
        assert begin < end
        return Interval(round(self.begin, 6), end, self.label)


@dataclassy.dataclass
class WordCtmInterval:
    label: str
    symbol: int
    phones: typing.List[CtmInterval]

    @property
    def begin(self):
        if len(self.phones) > 0:
            return self.phones[0].begin
        return 0.0

    @property
    def end(self):
        if len(self.phones) > 0:
            return self.phones[-1].end
        return 0.0

    @property
    def pronunciation(self):
        return " ".join(x.label for x in self.phones)

    def to_tg_interval(self, file_duration=None) -> Interval:
        """
        Converts the CTMInterval to
        `PraatIO's Interval class <http://timmahrt.github.io/praatIO/praatio/utilities/constants.html#Interval>`_

        Returns
        -------
        :class:`praatio.utilities.constants.Interval`
            Derived PraatIO Interval
        """
        if self.end < -1 or self.begin == 1000000:
            raise CtmError(self)
        end = round(self.end, 6)
        begin = round(self.begin, 6)
        if file_duration is not None and end > file_duration:
            end = file_duration
        assert begin < end
        return Interval(begin, end, self.label)


@dataclassy.dataclass
class HierarchicalCtm:
    word_intervals: typing.List[WordCtmInterval]
    text: str = None
    likelihood: float = None

    def export_textgrid(
        self,
        file_name: typing.Union[str, pathlib.Path],
        file_duration: float = None,
        output_format: str = TextgridFormats.LONG_TEXTGRID,
    ):
        # Create initial textgrid
        tg = tgio.Textgrid()
        tg.minTimestamp = 0
        tg.maxTimestamp = file_duration

        word_tier, phone_tier = self.to_textgrid_tiers(file_duration)
        tg.addTier(word_tier)
        tg.addTier(phone_tier)
        if str(file_name) == "-":

            tgAsDict = _tgToDictionary(tg)

            textgridStr = tgio.textgrid_io.getTextgridAsStr(
                tgAsDict,
                output_format,
                True,
                0,
                file_duration,
            )
            sys.stdout.write(textgridStr)
            sys.stdout.flush()
        else:
            tg.save(
                str(file_name),
                includeBlankSpaces=True,
                format=output_format,
                reportingMode="error",
            )

    def to_textgrid_tiers(
        self, file_duration: float = None
    ) -> typing.Tuple[tgio.IntervalTier, tgio.IntervalTier]:
        if file_duration is not None:
            file_duration = round(file_duration, 6)
        word_tier = tgio.IntervalTier("words", [], minT=0.0, maxT=file_duration)
        phone_tier = tgio.IntervalTier("phones", [], minT=0.0, maxT=file_duration)
        for w in self.word_intervals:
            word_tier.insertEntry(w.to_tg_interval(file_duration))
            for p in w.phones:
                phone_tier.insertEntry(p.to_tg_interval())
        return word_tier, phone_tier

    def update_utterance_boundaries(self, begin, end=None):
        for w in self.word_intervals:
            for p in w.phones:
                p.begin += begin
                p.end += begin
        if len(self.word_intervals):
            self.word_intervals[-1].phones[-1].end = end


@dataclassy.dataclass
class Alignment:
    utterance_id: str
    alignment: typing.List[int]
    words: typing.List[int]
    likelihood: float = None
    per_frame_likelihoods: FloatVector = None
    lattice: Lattice = None

    @property
    def num_frames(self):
        return len(self.alignment)

    def generate_word_ctm(
        self,
        transition_model: TransitionModel,
        align_lexicon: WordAlignLatticeLexiconInfo,
        word_table: pywrapfst.SymbolTable,
        frame_shift: float = 0.01,
    ):
        lattice = linear_to_lattice(self.alignment, self.words)
        success, lattice = word_align_lattice_lexicon(
            lattice, transition_model, align_lexicon, WordAlignLatticeLexiconOpts()
        )
        if not success:
            raise Exception("Error aligning lattice")
        success, words, times, lengths = CompactLatticeToWordAlignment(lattice)
        if not success:
            raise Exception("Error generating word CTM")
        ctm_output = []
        for w, time, length in zip(words, times, lengths):
            word_start = round(time * frame_shift, 3)
            word_end = round((time * frame_shift) + (length * frame_shift), 3)
            label = word_table.find(w)
            ctm_output.append(CtmInterval(word_start, word_end, label, w))
        return ctm_output

    def generate_ctm(
        self,
        transition_model: TransitionModel,
        phone_table: pywrapfst.SymbolTable,
        frame_shift: float = 0.01,
    ):
        split = SplitToPhones(transition_model, self.alignment)
        ctm_output = []
        phone_start = 0.0
        current_phone_index = 0
        likelihoods = None
        if self.per_frame_likelihoods:
            likelihoods = self.per_frame_likelihoods.numpy()
        for s in split:
            phone_id = transition_model.TransitionIdToPhone(s[0])
            num_repeats = len(s)
            duration = frame_shift * num_repeats
            phone_end = phone_start + duration
            label = phone_table.find(phone_id)
            confidence = None
            if likelihoods is not None:
                confidence = float(
                    np.mean(likelihoods[current_phone_index : current_phone_index + num_repeats])
                )
            ctm_output.append(
                CtmInterval(
                    round(phone_start, 3),
                    round(phone_end, 3),
                    label,
                    phone_id,
                    confidence=confidence,
                )
            )
            phone_start += duration
            current_phone_index += num_repeats

        return ctm_output

    def export_textgrid(
        self,
        file_name: str,
        transition_model: TransitionModel,
        align_lexicon: VectorFst,
        word_table: pywrapfst.SymbolTable,
        phone_table: pywrapfst.SymbolTable,
        frame_shift: float = 0.01,
        duration: float = None,
        output_format: str = TextgridFormats.LONG_TEXTGRID,
    ):
        # Create initial textgrid
        tg = tgio.Textgrid()
        tg.minTimestamp = 0
        tg.maxTimestamp = duration

        intervals = self.generate_ctm(transition_model, phone_table, frame_shift=frame_shift)
        word_intervals = self.generate_word_ctm(
            transition_model, align_lexicon, word_table, frame_shift=frame_shift
        )
        if "words" not in tg.tierNames:
            tg.addTier(tgio.IntervalTier("words", [], minT=0, maxT=duration))
        if "phones" not in tg.tierNames:
            tg.addTier(tgio.IntervalTier("phones", [], minT=0, maxT=duration))
        phone_tier = tg.getTier("phones")
        word_tier = tg.getTier("words")
        for i, a in enumerate(sorted(intervals, key=lambda x: x.begin)):
            if i == len(intervals) - 1:
                a.end = duration
            if i > 0 and phone_tier.entries[-1].end > a.to_tg_interval().start:
                a.begin = phone_tier.entries[-1].end
            phone_tier.insertEntry(a.to_tg_interval(duration))
        for i, a in enumerate(sorted(word_intervals, key=lambda x: x.begin)):
            if i == len(word_intervals) - 1:
                a.end = duration
            if i > 0 and word_tier.entries[-1].end > a.to_tg_interval().start:
                a.begin = word_tier.entries[-1].end
            word_tier.insertEntry(a.to_tg_interval(duration))
        for tier in tg.tiers:
            if len(tier.entries) > 0 and tier.entries[-1][1] > tg.maxTimestamp:
                tier.insertEntry(
                    Interval(tier.entries[-1].start, tg.maxTimestamp, tier.entries[-1].label),
                    collisionMode="replace",
                )
        tg.save(
            str(file_name),
            includeBlankSpaces=True,
            format=output_format,
            reportingMode="error",
        )


class AlignmentArchive:
    """
    Class for reading an archive or SCP of alignments

    Parameters
    ----------
    file_name: :class:`~pathlib.Path` or str
        Path to archive or SCP file to read from
    """

    def __init__(
        self,
        file_name: typing.Union[pathlib.Path, str],
        words_file_name: typing.Union[pathlib.Path, str] = None,
        likelihood_file_name: typing.Union[pathlib.Path, str] = None,
    ):
        if not os.path.exists(file_name):
            raise OSError(f"Specified file does not exist: {file_name}")
        self.file_name = str(file_name)
        self.read_specifier = generate_read_specifier(file_name)

        self.words_reader = None
        self.words_file_name = words_file_name
        if words_file_name:
            words_read_specifier = generate_read_specifier(words_file_name)
            self.words_reader = RandomAccessInt32VectorReader(words_read_specifier)

        self.likelihood_reader = None
        self.likelihood_file_name = likelihood_file_name
        if likelihood_file_name:
            likelihood_read_specifier = generate_read_specifier(likelihood_file_name)
            self.likelihood_reader = RandomAccessBaseFloatVectorReader(likelihood_read_specifier)

        self.random_reader = RandomAccessInt32VectorReader(self.read_specifier)

    def __del__(self):
        self.close()

    def close(self):
        if self.random_reader.IsOpen():
            self.random_reader.Close()
        if self.words_reader is not None and self.words_reader.IsOpen():
            self.words_reader.Close()
        if self.likelihood_reader is not None and self.likelihood_reader.IsOpen():
            self.likelihood_reader.Close()

    @property
    def sequential_reader(self) -> SequentialInt32VectorReader:
        """Sequential reader for alignments"""
        return SequentialInt32VectorReader(self.read_specifier)

    def __iter__(self) -> typing.Generator[Alignment]:
        """Iterate over the utterance alignments in the archive"""
        reader = self.sequential_reader
        words_reader = None
        likelihood_reader = None
        if self.words_file_name is not None:
            words_read_specifier = generate_read_specifier(self.words_file_name)
            words_reader = SequentialInt32VectorReader(words_read_specifier)
        if self.likelihood_file_name is not None:
            likelihood_read_specifier = generate_read_specifier(self.likelihood_file_name)
            likelihood_reader = SequentialBaseFloatVectorReader(likelihood_read_specifier)
        try:
            while not reader.Done():
                utt = reader.Key()
                alignment = reader.Value()
                words = []
                if words_reader is not None:
                    words_utt = words_reader.Key()
                    if words_utt != utt:
                        raise Exception(
                            f"Mismatch in keys between {self.file_name} and {self.words_file_name}"
                        )
                    words = words_reader.Value()
                likelihoods = []
                if likelihood_reader is not None:
                    likelihood_utt = likelihood_reader.Key()
                    if likelihood_utt != utt:
                        raise Exception(
                            f"Mismatch in keys between {self.file_name} and {self.likelihood_file_name}"
                        )
                    likelihoods = likelihood_reader.Value()
                alignment = Alignment(
                    utt, alignment, words=words, per_frame_likelihoods=likelihoods
                )
                yield alignment
                reader.Next()
                if words_reader is not None:
                    words_reader.Next()
                if likelihood_reader is not None:
                    likelihood_reader.Next()
        finally:
            reader.Close()

    def __getitem__(self, item: str) -> Alignment:
        """Get alignment for a particular key from the archive file"""
        item = str(item)
        if not self.random_reader.HasKey(item):
            raise KeyError(f"No key {item} found in {self.file_name}")
        words = []
        likelihoods = []
        if self.words_reader is not None:
            words = self.words_reader.Value(item)
        if self.likelihood_reader is not None:
            likelihoods = self.likelihood_reader.Value(item)
        return Alignment(
            item, self.random_reader.Value(item), words=words, per_frame_likelihoods=likelihoods
        )


class TranscriptionArchive:
    """
    Class for reading an archive or SCP of transcriptions

    Parameters
    ----------
    file_name: :class:`~pathlib.Path` or str
        Path to archive or SCP file to read from
    """

    def __init__(
        self,
        file_name: typing.Union[pathlib.Path, str],
        lm_scale: float = 1.0,
        acoustic_scale: float = 1.0,
    ):
        self.lattice_archive = LatticeArchive(file_name, determinized=True)
        self.lm_scale = lm_scale
        self.acoustic_scale = acoustic_scale

    def __del__(self):
        self.lattice_archive.close()

    def __iter__(self) -> typing.Generator[Alignment]:
        """Iterate over the utterance alignments in the archive"""
        reader = self.lattice_archive.sequential_reader
        try:
            while not reader.Done():
                utt = reader.Key()
                lattice = reader.Value()
                decoded = lattice_best_path(lattice, self.lm_scale, self.acoustic_scale)
                alignment, words, weight = GetLinearSymbolSequence(decoded)
                likelihood = -(weight.Value1() + weight.Value2()) / self.acoustic_scale
                transcription = Alignment(
                    utt, alignment, words, likelihood=likelihood, lattice=lattice
                )
                yield transcription
                reader.Next()
        finally:
            reader.Close()

    def __getitem__(self, item: str) -> Alignment:
        """Get alignment for a particular key from the archive file"""
        item = str(item)
        if not self.lattice_archive.random_reader.HasKey(item):
            raise KeyError(f"No key {item} found in {self.lattice_archive.file_name}")

        lattice = self.lattice_archive.random_reader.Value(item)
        decoded = lattice_best_path(lattice, self.lm_scale, self.acoustic_scale)
        alignment, words, weight = GetLinearSymbolSequence(decoded)
        likelihood = -(weight.Value1() + weight.Value2()) / self.acoustic_scale
        transcription = Alignment(item, alignment, words, likelihood=likelihood, lattice=lattice)
        return transcription


class LatticeArchive:
    """
    Class for reading an archive or SCP of lattices

    Parameters
    ----------
    file_name: :class:`~pathlib.Path` or str
        Path to archive or SCP file to read from
    """

    def __init__(self, file_name: typing.Union[pathlib.Path, str], determinized: bool = True):
        if not os.path.exists(file_name):
            raise OSError(f"Specified file does not exist: {file_name}")
        self.file_name = str(file_name)
        self.determinized = determinized
        self.read_specifier = generate_read_specifier(file_name)
        if self.determinized:
            self.random_reader = RandomAccessCompactLatticeReader(self.read_specifier)
        else:
            self.random_reader = RandomAccessLatticeReader(self.read_specifier)

    def close(self):
        if self.random_reader.IsOpen():
            self.random_reader.Close()

    @property
    def sequential_reader(
        self,
    ) -> typing.Union[SequentialCompactLatticeReader, SequentialLatticeReader]:
        """Sequential reader for lattices"""
        if self.determinized:
            return SequentialCompactLatticeReader(self.read_specifier)
        return SequentialLatticeReader(self.read_specifier)

    def __iter__(
        self,
    ) -> typing.Generator[typing.Tuple[str, typing.Union[Lattice, CompactLattice]]]:
        """Iterate over the utterance lattices in the archive"""
        reader = self.sequential_reader
        try:
            while not reader.Done():
                utt = reader.Key()
                lattice = reader.Value()
                yield utt, lattice
                reader.Next()
        finally:
            reader.Close()

    def __del__(self):
        self.close()

    def __getitem__(self, item: str) -> typing.Union[Lattice, CompactLattice]:
        """Get lattice for a particular key from the archive file"""
        item = str(item)
        if not self.random_reader.HasKey(item):
            raise KeyError(f"No key {item} found in {self.file_name}")
        return self.random_reader.Value(item)
