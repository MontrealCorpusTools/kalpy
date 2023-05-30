"""Classes for working with lexicons"""
from __future__ import annotations

import collections
import dataclasses
import math
import pathlib
import re
import typing

import pynini
import pywrapfst


@dataclasses.dataclass
class Pronunciation:
    """
    Data class for storing information about a particular pronunciation
    """

    orthography: str
    pronunciation: str
    probability: typing.Optional[float]
    silence_after_probability: typing.Optional[float]
    silence_before_correct: typing.Optional[float]
    non_silence_before_correct: typing.Optional[float]
    disambiguation: typing.Optional[int]


def parse_dictionary_file(
    path: typing.Union[pathlib.Path, str],
) -> typing.Generator[Pronunciation]:
    """
    Parses a lexicon file and yields parsed pronunciation lines

    Parameters
    ----------
    path: :class:`~pathlib.Path` or str
        Path to lexicon file

    Yields
    ------
    str
        :class:`~kalpy.fstext.lexicon.Pronunciation`
    """
    prob_pattern = re.compile(r"\b\d+\.\d+\b")
    with open(path, encoding="utf8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            line = line.split()
            if len(line) <= 1:
                raise Exception(
                    f'Error parsing line {i} of {path}: "{line}" did not have a pronunciation'
                )
            word = line.pop(0)
            prob = None
            silence_after_prob = None
            silence_before_correct = None
            non_silence_before_correct = None
            if prob_pattern.match(line[0]):
                prob = float(line.pop(0))
                if prob_pattern.match(line[0]):
                    silence_after_prob = float(line.pop(0))
                    if prob_pattern.match(line[0]):
                        silence_before_correct = float(line.pop(0))
                        if prob_pattern.match(line[0]):
                            non_silence_before_correct = float(line.pop(0))
            pron = " ".join(line)
            yield Pronunciation(
                word,
                pron,
                prob,
                silence_after_prob,
                silence_before_correct,
                non_silence_before_correct,
                None,
            )


class LexiconCompiler:
    """
    Class for compiling pronunciation dictionary files to lexicon FSTs

    Parameters
    ----------
    silence_disambiguation_symbol: str
        Disambiguation symbol for use in transcription decoding versus alignment decoding,
        leave empty to generate a lexicon for training/alignment
    silence_probability: float
        Probability of silence following words
    initial_silence_probability: float
        Probability of silence at the beginning of utterances
    final_silence_correction: float
        Correction factor for utterances ending in silence
    final_non_silence_correction: float
        Correction factor for utterances not ending in silence
    silence_word: str
        Word symbol to use for silence
    oov_word: str
        Word symbol to use for out of vocabulary items
    silence_phone: str
        Phone symbol to use for silence
    oov_phone: str
        Phone symbol to use for out of vocabulary items
    position_dependent_phones: bool
        Flag for using phones based on word position,
        i.e. "AA_S" (for words with a pronunciation of only "AA"),
        "AA_B" ("AA" at the beginning of a word),
        "AA_I" ("AA" internal to  a word),
        "AA_E" ("AA" at the end of a word)
        instead of using a single "AA" symbol across all word positions
    ignore_case: bool
        Flag for whether word orthographies should be transformed to lower case

    Attributes
    ----------
    word_table: :class:`pywrapfst.SymbolTable`
        Word symbol table
    phone_table: :class:`pywrapfst.SymbolTable`
        Phone symbol table
    pronunciations: list[:class:`~kalpy.fstext.lexicon.Pronunciation`]
        List of pronunciations loaded from dictionary file
    """

    def __init__(
        self,
        silence_disambiguation_symbol: typing.Optional[str] = None,
        silence_probability: float = 0.5,
        initial_silence_probability: float = 0.5,
        final_silence_correction: typing.Optional[float] = None,
        final_non_silence_correction: typing.Optional[float] = None,
        silence_word: str = "<eps>",
        oov_word: str = "<unk>",
        silence_phone: str = "sil",
        oov_phone: str = "spn",
        position_dependent_phones: bool = False,
        ignore_case: bool = True,
    ):
        self.silence_disambiguation_symbol = silence_disambiguation_symbol
        self.silence_probability = silence_probability
        self.initial_silence_probability = initial_silence_probability
        self.final_silence_correction = final_silence_correction
        self.final_non_silence_correction = final_non_silence_correction
        self.silence_word = silence_word
        self.oov_word = oov_word
        self.silence_phone = silence_phone
        self.oov_phone = oov_phone
        self.max_disambiguation_symbol = 1
        self.position_dependent_phones = position_dependent_phones
        self.ignore_case = ignore_case
        if self.silence_disambiguation_symbol is None:
            self.silence_disambiguation_symbol = "<eps>"
            self.disambiguation = False
        else:
            self.disambiguation = True
        self.word_table = pywrapfst.SymbolTable()
        self.word_table.add_symbol(silence_word)
        self.word_table.add_symbol(oov_word)
        self.phone_table = pywrapfst.SymbolTable()
        self.phone_table.add_symbol("<eps>")
        self.phone_table.add_symbol(silence_phone)
        if self.position_dependent_phones:
            for pos in ["_S", "_B", "_E", "_I"]:
                self.phone_table.add_symbol(silence_phone + pos)
        self.phone_table.add_symbol(oov_phone)
        if self.position_dependent_phones:
            for pos in ["_S", "_B", "_E", "_I"]:
                self.phone_table.add_symbol(oov_phone + pos)
        self.pronunciations: typing.List[Pronunciation] = []
        self._fst = None
        self._align_fst = None

    def to_int(self, word: str) -> int:
        """
        Look up a word in the word symbol table

        Parameters
        ----------
        word: str
            Word to look up

        Returns
        -------
        int
            Integer ID of word in symbol table
        """
        if self.word_table.member(word):
            return self.word_table.find(word)
        return self.word_table.find(self.oov_word)

    @property
    def specials_set(self) -> typing.Set[str]:
        """Special words, like the ``oov_word`` ``silence_word``, ``<s>``, and ``</s>``"""
        return {
            "#0",
            self.silence_word,
            "<s>",
            "</s>",
        }

    def load_pronunciations(self, file_name: typing.Union[pathlib.Path, str]) -> None:
        """
        Load pronunciations from a dictionary file and calculate necessary disambiguation symbols

        Parameters
        ----------
        file_name: :class:`~pathlib.Path` or str
            Path to lexicon file
        """
        non_silence_phones = set()
        words = set()
        oov_found = False
        for pron in parse_dictionary_file(file_name):
            if self.ignore_case:
                pron.orthography = pron.orthography.lower()
            if pron.orthography in self.specials_set:
                continue
            if pron.orthography == self.oov_word:
                oov_found = True
            phones = pron.pronunciation.split()
            non_silence_phones.update(phones)
            self.pronunciations.append(pron)
            words.add(pron.orthography)
        if not oov_found:
            self.pronunciations.append(
                Pronunciation(
                    self.oov_word,
                    self.oov_phone,
                    1.0,
                    None,
                    None,
                    None,
                    None,
                )
            )
        for s in sorted(words):
            self.word_table.add_symbol(s)
        for s in sorted(non_silence_phones):
            if self.position_dependent_phones:
                for pos in ["_S", "_B", "_E", "_I"]:
                    self.phone_table.add_symbol(s + pos)
            else:
                self.phone_table.add_symbol(s)

        self.compute_disambiguation_symbols()
        self.word_table.add_symbol("#0")
        self.word_table.add_symbol("<s>")
        self.word_table.add_symbol("</s>")

    @property
    def disambiguation_symbols(self) -> typing.List[int]:
        """List of integer IDs for disambiguation symbols in the phone symbol table"""
        return [
            i
            for i in range(self.phone_table.num_symbols())
            if self.phone_table.find(i).startswith("#")
        ]

    def compute_disambiguation_symbols(self):
        """Calculate the necessary disambiguation symbols for the lexicon"""
        subsequences = set()
        for pron in self.pronunciations:

            phones = pron.pronunciation.split()
            while phones:
                subsequences.add(" ".join(phones))
                phones = phones[:-1]
        last_used = collections.defaultdict(int)

        for pron in self.pronunciations:
            if pron.pronunciation in subsequences:
                last_used[pron.pronunciation] += 1
                pron.disambiguation = last_used[pron.pronunciation]

        self.max_disambiguation_symbol = max(
            self.max_disambiguation_symbol, max(last_used.values())
        )
        for x in range(self.max_disambiguation_symbol + 3):
            p = f"#{x}"
            self.phone_table.add_symbol(p)
        return self.pronunciations

    @property
    def fst(self) -> pynini.Fst:
        """Compiled lexicon FST"""
        if self._fst is not None:
            return self._fst
        initial_silence_cost = -1 * math.log(self.initial_silence_probability)
        initial_non_silence_cost = -1 * math.log(1.0 - self.initial_silence_probability)
        if self.final_silence_correction is None or self.final_non_silence_correction is None:
            final_silence_cost = 0
            final_non_silence_cost = 0
        else:
            final_silence_cost = str(-math.log(self.final_silence_correction))
            final_non_silence_cost = str(-math.log(self.final_non_silence_correction))
        base_silence_following_cost = -math.log(self.silence_probability)
        base_non_silence_following_cost = -math.log(1 - self.silence_probability)
        self.phone_table.find(self.silence_disambiguation_symbol)
        self.word_table.find("<eps>")
        self.word_table.find(self.silence_word)
        fst = pynini.Fst()
        start_state = fst.add_state()
        fst.set_start(start_state)
        non_silence_state = fst.add_state()  # Also loop state
        silence_state = fst.add_state()
        # initial no silence
        fst.add_arc(
            start_state,
            pywrapfst.Arc(
                self.phone_table.find(self.silence_disambiguation_symbol),
                self.word_table.find(self.silence_word),
                pywrapfst.Weight(fst.weight_type(), initial_non_silence_cost),
                non_silence_state,
            ),
        )
        # initial silence
        fst.add_arc(
            start_state,
            pywrapfst.Arc(
                self.phone_table.find(self.silence_phone),
                self.word_table.find(self.silence_word),
                pywrapfst.Weight(fst.weight_type(), initial_silence_cost),
                silence_state,
            ),
        )
        for pron in self.pronunciations:
            word_symbol = self.word_table.find(pron.orthography)
            phones = pron.pronunciation.split()
            silence_before_cost = (
                -math.log(pron.silence_before_correct)
                if pron.silence_before_correct is not None
                else 0.0
            )
            non_silence_before_cost = (
                -math.log(pron.non_silence_before_correct)
                if pron.non_silence_before_correct is not None
                else 0.0
            )
            silence_following_cost = (
                -math.log(pron.silence_after_probability)
                if pron.silence_after_probability is not None
                else base_silence_following_cost
            )
            non_silence_following_cost = (
                -math.log(1 - pron.silence_after_probability)
                if pron.silence_after_probability is not None
                else base_non_silence_following_cost
            )
            if self.position_dependent_phones:
                if len(phones) == 1:
                    phones[0] += "_S"
                else:
                    phones[0] += "_B"
                    phones[-1] += "_E"
                    for i in range(1, len(phones) - 1):
                        phones[i] += "_I"
            probability = pron.probability
            if probability is None:
                probability = 1
            elif probability < 0.01:
                probability = 0.01  # Dithering to ensure low probability entries
            pron_cost = abs(math.log(probability))
            if self.disambiguation and pron.disambiguation is not None:
                phones += [f"#{pron.disambiguation}"]

            new_state = fst.add_state()
            phone_symbol = self.phone_table.find(phones[0])
            # No silence before the pronunciation
            fst.add_arc(
                non_silence_state,
                pywrapfst.Arc(
                    phone_symbol,
                    word_symbol,
                    pywrapfst.Weight(fst.weight_type(), pron_cost + non_silence_before_cost),
                    new_state,
                ),
            )
            # Silence before the pronunciation
            fst.add_arc(
                silence_state,
                pywrapfst.Arc(
                    phone_symbol,
                    word_symbol,
                    pywrapfst.Weight(fst.weight_type(), pron_cost + silence_before_cost),
                    new_state,
                ),
            )
            current_state = new_state
            for i in range(1, len(phones)):
                next_state = fst.add_state()
                phone_symbol = self.phone_table.find(phones[i])
                fst.add_arc(
                    current_state,
                    pywrapfst.Arc(
                        phone_symbol,
                        self.word_table.find("<eps>"),
                        pywrapfst.Weight.one(fst.weight_type()),
                        next_state,
                    ),
                )
                current_state = next_state
            # No silence following the pronunciation
            fst.add_arc(
                current_state,
                pywrapfst.Arc(
                    self.phone_table.find(self.silence_disambiguation_symbol),
                    self.word_table.find("<eps>"),
                    pywrapfst.Weight(fst.weight_type(), non_silence_following_cost),
                    non_silence_state,
                ),
            )
            # Silence following the pronunciation
            fst.add_arc(
                current_state,
                pywrapfst.Arc(
                    self.phone_table.find(self.silence_phone),
                    self.word_table.find("<eps>"),
                    pywrapfst.Weight(fst.weight_type(), silence_following_cost),
                    silence_state,
                ),
            )
        if final_silence_cost > 0:
            fst.set_final(silence_state, pywrapfst.Weight(fst.weight_type(), final_silence_cost))
        else:
            fst.set_final(silence_state, pywrapfst.Weight.one(fst.weight_type()))
        if final_non_silence_cost > 0:
            fst.set_final(
                non_silence_state, pywrapfst.Weight(fst.weight_type(), final_non_silence_cost)
            )
        else:
            fst.set_final(non_silence_state, pywrapfst.Weight.one(fst.weight_type()))

        # fst.set_input_symbols(self.phone_table)
        # fst.set_output_symbols(self.word_table)
        fst.arcsort("olabel")
        self._fst = fst
        return self._fst

    @property
    def align_fst(self) -> pynini.Fst:
        """Compiled FST for aligning lattices when `position_dependent_phones` is False"""
        if self._align_fst is not None:
            return self._align_fst
        fst = pynini.Fst()
        start_state = fst.add_state()
        loop_state = fst.add_state()
        sil_state = fst.add_state()
        next_state = fst.add_state()
        fst.set_start(start_state)
        word_eps_symbol = self.word_table.find("<eps>")
        phone_eps_symbol = self.phone_table.find("<eps>")
        sil_cost = -math.log(0.5)
        non_sil_cost = sil_cost
        fst.add_arc(
            start_state,
            pywrapfst.Arc(
                phone_eps_symbol,
                word_eps_symbol,
                pywrapfst.Weight(fst.weight_type(), non_sil_cost),
                loop_state,
            ),
        )
        fst.add_arc(
            start_state,
            pywrapfst.Arc(
                phone_eps_symbol,
                word_eps_symbol,
                pywrapfst.Weight(fst.weight_type(), sil_cost),
                sil_state,
            ),
        )

        for pron in self.pronunciations:
            phones = pron.pronunciation.split()
            if self.position_dependent_phones:
                if phones[0] != self.silence_phone:
                    if len(phones) == 1:
                        phones[0] += "_S"
                    else:
                        phones[0] += "_B"
                        phones[-1] += "_E"
                        for i in range(1, len(phones) - 1):
                            phones[i] += "_I"
            phones = ["#1"] + phones + ["#2"]
            current_state = loop_state
            for i in range(len(phones) - 1):
                p_s = self.phone_table.find(phones[i])
                if i == 0:
                    w_s = self.word_table.find(pron.orthography)
                else:
                    w_s = word_eps_symbol
                fst.add_arc(
                    current_state,
                    pywrapfst.Arc(p_s, w_s, pywrapfst.Weight.one(fst.weight_type()), next_state),
                )
                current_state = next_state
                next_state = fst.add_state()
            i = len(phones) - 1
            if i >= 0:
                p_s = self.phone_table.find(phones[i])
            else:
                p_s = phone_eps_symbol
            if i <= 0:
                w_s = self.word_table.find(pron.orthography)
            else:
                w_s = word_eps_symbol
            fst.add_arc(
                current_state,
                pywrapfst.Arc(
                    p_s, w_s, pywrapfst.Weight(fst.weight_type(), non_sil_cost), loop_state
                ),
            )
            fst.add_arc(
                current_state,
                pywrapfst.Arc(p_s, w_s, pywrapfst.Weight(fst.weight_type(), sil_cost), sil_state),
            )
        fst.delete_states([next_state])
        fst.set_final(loop_state, pywrapfst.Weight.one(fst.weight_type()))
        fst.arcsort("olabel")
        self._align_fst = fst
        return self._align_fst
