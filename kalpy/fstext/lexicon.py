"""Classes for working with lexicons"""
from __future__ import annotations

import collections
import math
import pathlib
import re
import threading
import typing

import dataclassy
import pynini
import pywrapfst

from _kalpy.fstext import VectorFst
from _kalpy.lat import WordAlignLatticeLexiconInfo
from kalpy.exceptions import LexiconError, PhonesToPronunciationsError
from kalpy.gmm.data import CtmInterval, HierarchicalCtm, WordCtmInterval


@dataclassy.dataclass
class Pronunciation:
    """
    Data class for storing information about a particular pronunciation
    """

    orthography: str
    pronunciation: str
    probability: typing.Optional[float]
    silence_after_probability: typing.Optional[float]
    silence_before_correction: typing.Optional[float]
    non_silence_before_correction: typing.Optional[float]
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
    found_set = set()
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
            silence_before_correction = None
            non_silence_before_correction = None
            if prob_pattern.match(line[0]):
                prob = float(line.pop(0))
                if prob_pattern.match(line[0]):
                    silence_after_prob = float(line.pop(0))
                    if prob_pattern.match(line[0]):
                        silence_before_correction = float(line.pop(0))
                        if prob_pattern.match(line[0]):
                            non_silence_before_correction = float(line.pop(0))
            pron = " ".join(line)
            if (word, pron) in found_set:
                continue
            found_set.add((word, pron))
            yield Pronunciation(
                word,
                pron,
                prob,
                silence_after_prob,
                silence_before_correction,
                non_silence_before_correction,
                None,
            )


class LexiconCompiler:
    """
    Class for compiling pronunciation dictionary files to lexicon FSTs

    Parameters
    ----------
    disambiguation: bool
        Flag for compiling a disambiguated lexicon for decoding instead of alignment
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

    use_g2p = False

    def __init__(
        self,
        disambiguation: bool = False,
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
        phones: typing.Collection[str] = None,
        word_begin_label: str = "#1",
        word_end_label: str = "#2",
    ):
        self.disambiguation = disambiguation
        self.silence_disambiguation_symbol = "<eps>"
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
        if phones is not None:
            for p in sorted(phones):
                if self.position_dependent_phones:
                    for pos in ["_S", "_B", "_E", "_I"]:
                        self.phone_table.add_symbol(p + pos)
                else:
                    self.phone_table.add_symbol(p)
        self.pronunciations: typing.List[Pronunciation] = []
        self._cached_pronunciations: typing.Set[typing.Tuple[str, str]] = set()
        self._fst = None
        self._align_fst = None
        self._align_lexicon = None
        self.word_begin_label = word_begin_label
        self.word_end_label = word_end_label
        self.start_state = None
        self.loop_state = None
        self.silence_state = None
        self.non_silence_state = None

    def clear(self):
        self.pronunciations = []

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

    @property
    def disambiguation_symbols(self) -> typing.List[int]:
        """List of integer IDs for disambiguation symbols in the phone symbol table"""
        return [
            i
            for i in range(self.phone_table.num_symbols())
            if self.phone_table.find(i).startswith("#")
        ]

    @property
    def silence_symbols(self) -> typing.List[int]:
        """List of integer IDs for silence symbols in the phone symbol table"""
        return [
            i
            for i in range(self.phone_table.num_symbols())
            if self.phone_table.find(i) in {self.silence_phone, self.oov_phone}
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
        if self.disambiguation:
            self.silence_disambiguation_symbol = f"#{self.max_disambiguation_symbol + 1}"
        self.word_table.add_symbol("#0")
        self.word_table.add_symbol("<s>")
        self.word_table.add_symbol("</s>")

    @property
    def align_lexicon(self):
        if self._align_lexicon is None:
            lex = []
            word_symbol = self.to_int(self.silence_word)
            lex.append([word_symbol, word_symbol, self.phone_table.find(self.silence_phone)])
            for pron in self.pronunciations:
                word_symbol = self.to_int(pron.orthography)
                phones = pron.pronunciation.split()
                if self.position_dependent_phones:
                    if len(phones) == 1:
                        phones[0] += "_S"
                    else:
                        phones[0] += "_B"
                        for i in range(1, len(phones) - 1):
                            phones[i] += "_I"
                        phones[-1] += "_E"

                lex.append([word_symbol, word_symbol, *(self.phone_table.find(x) for x in phones)])

            self._align_lexicon = WordAlignLatticeLexiconInfo(lex)
        return self._align_lexicon

    def create_fsts(self, phonological_rule_fst: pynini.Fst = None):
        if self._fst is not None and self._align_fst is not None:
            return

        initial_silence_cost = 0
        initial_non_silence_cost = 0
        if self.initial_silence_probability:
            initial_silence_cost = -1 * math.log(self.initial_silence_probability)
            initial_non_silence_cost = -1 * math.log(1.0 - self.initial_silence_probability)

        final_silence_cost = 0
        final_non_silence_cost = 0
        if self.final_silence_correction:
            final_silence_cost = -math.log(self.final_silence_correction)
            final_non_silence_cost = -math.log(self.final_non_silence_correction)

        self.phone_table.find(self.silence_disambiguation_symbol)
        phone_eps_symbol = self.phone_table.find("<eps>")
        self.word_table.find(self.silence_word)
        self._fst = pynini.Fst()
        self._align_fst = pynini.Fst()
        self.start_state = self._fst.add_state()
        self._align_fst.add_state()
        self._fst.set_start(self.start_state)
        self.non_silence_state = self._fst.add_state()  # Also loop state
        self._align_fst.add_state()
        self.silence_state = self._fst.add_state()
        self._align_fst.add_state()

        self._align_fst.set_start(self.start_state)
        # initial no silence
        self._fst.add_arc(
            self.start_state,
            pywrapfst.Arc(
                phone_eps_symbol,
                self.word_table.find(self.silence_word),
                pywrapfst.Weight(self._fst.weight_type(), initial_non_silence_cost),
                self.non_silence_state,
            ),
        )
        self._align_fst.add_arc(
            self.start_state,
            pywrapfst.Arc(
                phone_eps_symbol,
                self.word_table.find(self.silence_word),
                pywrapfst.Weight(self._align_fst.weight_type(), initial_non_silence_cost),
                self.non_silence_state,
            ),
        )
        # initial silence
        self._fst.add_arc(
            self.start_state,
            pywrapfst.Arc(
                self.phone_table.find(self.silence_phone),
                self.word_table.find(self.silence_word),
                pywrapfst.Weight(self._fst.weight_type(), initial_silence_cost),
                self.silence_state,
            ),
        )
        self._align_fst.add_arc(
            self.start_state,
            pywrapfst.Arc(
                self.phone_table.find(self.silence_phone),
                self.word_table.find(self.silence_word),
                pywrapfst.Weight(self._align_fst.weight_type(), initial_silence_cost),
                self.silence_state,
            ),
        )

        for pron in self.pronunciations:
            self.add_pronunciation(pron, phonological_rule_fst)
        if final_silence_cost > 0:
            self._fst.set_final(
                self.silence_state, pywrapfst.Weight(self._fst.weight_type(), final_silence_cost)
            )
            self._align_fst.set_final(
                self.silence_state,
                pywrapfst.Weight(self._align_fst.weight_type(), final_silence_cost),
            )
        else:
            self._fst.set_final(self.silence_state, pywrapfst.Weight.one(self._fst.weight_type()))
            self._align_fst.set_final(
                self.silence_state, pywrapfst.Weight.one(self._align_fst.weight_type())
            )
        if final_non_silence_cost > 0:
            self._fst.set_final(
                self.non_silence_state,
                pywrapfst.Weight(self._fst.weight_type(), final_non_silence_cost),
            )
            self._align_fst.set_final(
                self.non_silence_state,
                pywrapfst.Weight(self._align_fst.weight_type(), final_non_silence_cost),
            )
        else:
            self._fst.set_final(
                self.non_silence_state, pywrapfst.Weight.one(self._fst.weight_type())
            )
            self._align_fst.set_final(
                self.non_silence_state, pywrapfst.Weight.one(self._align_fst.weight_type())
            )

        if (
            self._fst.num_states() <= self.silence_state + 1
            or self._fst.start() == pywrapfst.NO_STATE_ID
        ):
            num_words = self.word_table.num_symbols()
            num_phones = self.phone_table.num_symbols()
            num_pronunciations = len(self.pronunciations)
            raise LexiconError(
                f"There was an error compiling the lexicon "
                f"({num_words} words, {num_pronunciations} pronunciations, "
                f"{num_phones} phones)."
            )
        self._align_fst.arcsort("olabel")
        self._fst.arcsort("olabel")

    @property
    def base_silence_following_cost(self):
        base_silence_following_cost = 0
        if self.silence_probability:
            base_silence_following_cost = -math.log(self.silence_probability)
        return base_silence_following_cost

    @property
    def base_non_silence_following_cost(self):
        base_non_silence_following_cost = 0
        if self.silence_probability:
            base_non_silence_following_cost = -math.log(1 - self.silence_probability)
        return base_non_silence_following_cost

    @property
    def fst(self) -> pynini.Fst:
        """Compiled lexicon FST"""
        if self._fst is None:
            self.create_fsts()
        return self._fst

    def _create_word_fst(
        self, pronunciation: Pronunciation, phonological_rule_fst: pynini.Fst = None
    ):

        pron = pronunciation.pronunciation
        if self.position_dependent_phones:
            phones = pronunciation.pronunciation.split()
            if len(phones) == 1:
                phones[0] += "_S"
            else:
                phones[0] += "_B"
                phones[-1] += "_E"
                for i in range(1, len(phones) - 1):
                    phones[i] += "_I"
            pron = " ".join(phones)
        if not self.word_table.member(pronunciation.orthography):
            self.word_table.add_symbol(pronunciation.orthography)
        word_symbol = self.word_table.find(pronunciation.orthography)
        if self.disambiguation and pronunciation.disambiguation is not None:
            pron += f" #{pronunciation.disambiguation}"
        probability = pronunciation.probability
        weight = pywrapfst.Weight.one("tropical")
        if probability is not None:
            if probability < 0.01:
                probability = 0.01  # Dithering to ensure low probability entries
            weight = pywrapfst.Weight("tropical", abs(math.log(probability)))
        fst = pynini.accep(pron, weight=weight, token_type=self.phone_table)
        if phonological_rule_fst:
            fst = pynini.compose(phonological_rule_fst, fst)
        fst = pynini.arcmap(fst, map_type="output_epsilon")
        arcs = []
        for arc in fst.mutable_arcs(fst.start()):
            arc = arc.copy()
            arc.olabel = word_symbol
            arcs.append(arc)
        fst.delete_arcs(fst.start())
        for arc in arcs:
            fst.add_arc(fst.start(), arc)

        silence_before_cost = (
            -math.log(pronunciation.silence_before_correction)
            if pronunciation.silence_before_correction
            else 0.0
        )
        non_silence_before_cost = (
            -math.log(pronunciation.non_silence_before_correction)
            if pronunciation.non_silence_before_correction
            else 0.0
        )
        silence_following_cost = (
            -math.log(pronunciation.silence_after_probability)
            if pronunciation.silence_after_probability
            else self.base_silence_following_cost
        )
        non_silence_following_cost = (
            -math.log(1 - pronunciation.silence_after_probability)
            if pronunciation.silence_after_probability
            else self.base_non_silence_following_cost
        )
        initial_silence_fst = pynini.union(
            pynini.accep(
                self.silence_phone,
                weight=pywrapfst.Weight("tropical", silence_before_cost),
                token_type=self.phone_table,
            ),
            pynini.accep(
                "<eps>",
                weight=pywrapfst.Weight("tropical", non_silence_before_cost),
                token_type=self.phone_table,
            ),
        )

        initial_silence_fst = pynini.arcmap(initial_silence_fst, map_type="output_epsilon")
        final_silence_fst = pynini.union(
            pynini.accep(
                self.silence_phone,
                weight=pywrapfst.Weight("tropical", silence_following_cost),
                token_type=self.phone_table,
            ),
            pynini.accep(
                "<eps>",
                weight=pywrapfst.Weight("tropical", non_silence_following_cost),
                token_type=self.phone_table,
            ),
        )
        final_silence_fst = pynini.arcmap(final_silence_fst, map_type="output_epsilon")
        fst = initial_silence_fst + fst + final_silence_fst
        fst.optimize()
        return fst

    def add_pronunciation(
        self, pronunciation: Pronunciation, phonological_rule_fst: pynini.Fst = None
    ):
        if (pronunciation.orthography, pronunciation.pronunciation) in self._cached_pronunciations:
            return
        phones = pronunciation.pronunciation.split()
        if self.position_dependent_phones:
            if len(phones) == 1:
                phones[0] += "_S"
            else:
                phones[0] += "_B"
                phones[-1] += "_E"
                for i in range(1, len(phones) - 1):
                    phones[i] += "_I"
        new_phones = ", ".join(sorted({x for x in phones if not self.phone_table.member(x)}))
        if new_phones:
            raise Exception(
                f"The pronunciation '{pronunciation}' had the following phones not in the symbol table: {new_phones}"
            )
        pron = " ".join(phones)
        fst = pynini.accep(pron, token_type=self.phone_table)
        if phonological_rule_fst:
            fst = pynini.compose(phonological_rule_fst, fst)
        fst.rmepsilon()
        self._cached_pronunciations.add((pronunciation.orthography, pronunciation.pronunciation))
        if not self.word_table.member(pronunciation.orthography):
            self.word_table.add_symbol(pronunciation.orthography)
        word_symbol = self.word_table.find(pronunciation.orthography)
        word_eps_symbol = self.word_table.find("<eps>")
        phone_eps_symbol = self.phone_table.find("<eps>")
        silence_before_cost = (
            -math.log(pronunciation.silence_before_correction)
            if pronunciation.silence_before_correction
            else 0.0
        )
        non_silence_before_cost = (
            -math.log(pronunciation.non_silence_before_correction)
            if pronunciation.non_silence_before_correction
            else 0.0
        )
        silence_following_cost = (
            -math.log(pronunciation.silence_after_probability)
            if pronunciation.silence_after_probability
            else self.base_silence_following_cost
        )
        non_silence_following_cost = (
            -math.log(1 - pronunciation.silence_after_probability)
            if pronunciation.silence_after_probability
            else self.base_non_silence_following_cost
        )
        probability = pronunciation.probability
        if probability is None:
            probability = 1
        elif probability < 0.01:
            probability = 0.01  # Dithering to ensure low probability entries
        pron_cost = abs(math.log(probability))
        start_index = self._fst.num_states() - 1
        align_start_index = self._align_fst.num_states()
        num_new_states = fst.num_states() - 1
        self._fst.add_states(num_new_states)
        self._align_fst.add_states(num_new_states + 2)

        # FST arcs
        for state in fst.states():
            for arc in fst.arcs(state):
                if state == fst.start():
                    # No silence before the pronunciation
                    self._fst.add_arc(
                        self.non_silence_state,
                        pywrapfst.Arc(
                            arc.ilabel,
                            word_symbol,
                            pywrapfst.Weight(
                                self._fst.weight_type(), pron_cost + non_silence_before_cost
                            ),
                            arc.nextstate + start_index,
                        ),
                    )
                    # Silence before the pronunciation
                    self._fst.add_arc(
                        self.silence_state,
                        pywrapfst.Arc(
                            arc.ilabel,
                            word_symbol,
                            pywrapfst.Weight(
                                self._fst.weight_type(), pron_cost + silence_before_cost
                            ),
                            arc.nextstate + start_index,
                        ),
                    )

                    # No silence before the pronunciation
                    self._align_fst.add_arc(
                        self.non_silence_state,
                        pywrapfst.Arc(
                            self.phone_table.find(self.word_begin_label),
                            word_symbol,
                            pywrapfst.Weight(
                                self._fst.weight_type(), pron_cost + non_silence_before_cost
                            ),
                            arc.nextstate + align_start_index - 1,
                        ),
                    )
                    # Silence before the pronunciation
                    self._align_fst.add_arc(
                        self.silence_state,
                        pywrapfst.Arc(
                            self.phone_table.find(self.word_begin_label),
                            word_symbol,
                            pywrapfst.Weight(
                                self._fst.weight_type(), pron_cost + silence_before_cost
                            ),
                            arc.nextstate + align_start_index - 1,
                        ),
                    )
                else:
                    self._fst.add_arc(
                        state + start_index,
                        pywrapfst.Arc(
                            arc.ilabel,
                            word_eps_symbol,
                            arc.weight,
                            arc.nextstate + start_index,
                        ),
                    )
                self._align_fst.add_arc(
                    state + align_start_index,
                    pywrapfst.Arc(
                        arc.ilabel,
                        word_eps_symbol,
                        arc.weight,
                        arc.nextstate + align_start_index,
                    ),
                )

        if self.disambiguation and pronunciation.disambiguation is not None:
            self._fst.add_state()
            self._fst.add_arc(
                num_new_states + start_index,
                pywrapfst.Arc(
                    self.phone_table.find(f"#{pronunciation.disambiguation}"),
                    word_eps_symbol,
                    pywrapfst.Weight(self._fst.weight_type(), non_silence_following_cost),
                    num_new_states + start_index + 1,
                ),
            )
            start_index += 1

        # No silence following the pronunciation
        self._fst.add_arc(
            num_new_states + start_index,
            pywrapfst.Arc(
                self.phone_table.find(self.silence_disambiguation_symbol),
                word_eps_symbol,
                pywrapfst.Weight(self._fst.weight_type(), non_silence_following_cost),
                self.non_silence_state,
            ),
        )
        # Silence following the pronunciation
        self._fst.add_arc(
            num_new_states + start_index,
            pywrapfst.Arc(
                self.phone_table.find(self.silence_phone),
                word_eps_symbol,
                pywrapfst.Weight(self._fst.weight_type(), silence_following_cost),
                self.silence_state,
            ),
        )
        self._align_fst.add_arc(
            num_new_states + align_start_index,
            pywrapfst.Arc(
                self.phone_table.find(self.word_end_label),
                word_eps_symbol,
                pywrapfst.Weight.one(self._align_fst.weight_type()),
                num_new_states + align_start_index + 1,
            ),
        )

        # No silence following the pronunciation
        self._align_fst.add_arc(
            num_new_states + align_start_index + 1,
            pywrapfst.Arc(
                phone_eps_symbol,
                word_eps_symbol,
                pywrapfst.Weight(self._fst.weight_type(), non_silence_following_cost),
                self.non_silence_state,
            ),
        )
        # Silence following the pronunciation
        self._align_fst.add_arc(
            num_new_states + align_start_index + 1,
            pywrapfst.Arc(
                self.phone_table.find(self.silence_phone),
                word_eps_symbol,
                pywrapfst.Weight(self._fst.weight_type(), silence_following_cost),
                self.silence_state,
            ),
        )

    @property
    def kaldi_fst(self) -> VectorFst:
        return VectorFst.from_pynini(self.fst)

    def load_l_from_file(
        self,
        l_fst_path: typing.Union[pathlib.Path, str],
    ) -> None:
        """
        Read g.fst from file

        Parameters
        ----------
        l_fst_path: :class:`~pathlib.Path` or str
            Path to read HCLG.fst
        """
        self._fst = pynini.Fst.read(str(l_fst_path))

    def load_l_align_from_file(
        self,
        l_fst_path: typing.Union[pathlib.Path, str],
    ) -> None:
        """
        Read g.fst from file

        Parameters
        ----------
        l_fst_path: :class:`~pathlib.Path` or str
            Path to read HCLG.fst
        """
        self._align_fst = pynini.Fst.read(str(l_fst_path))

    @property
    def align_fst(self) -> pynini.Fst:
        """Compiled FST for aligning lattices when `position_dependent_phones` is False"""
        if self._align_fst is None:
            self.create_fsts()
        return self._align_fst

    def _create_pronunciation_string(
        self,
        word_symbols: typing.List[int],
        phone_symbols: typing.List[int],
        transcription: bool = False,
    ):
        word_begin_symbol = self.phone_table.find(self.word_begin_label)
        word_end_symbol = self.phone_table.find(self.word_end_label)
        text = " ".join(self.word_table.find(x) for x in word_symbols)
        acceptor = pynini.accep(text, token_type=self.word_table)
        phone_to_word = pynini.compose(self.align_fst, acceptor)
        phone_fst = pynini.Fst()
        current_state = phone_fst.add_state()
        phone_fst.set_start(current_state)
        for symbol in phone_symbols:
            next_state = phone_fst.add_state()
            phone_fst.add_arc(
                current_state,
                pywrapfst.Arc(
                    symbol, symbol, pywrapfst.Weight.one(phone_fst.weight_type()), next_state
                ),
            )
            current_state = next_state
        if transcription:
            if phone_symbols[-1] == self.phone_table.find(self.silence_phone):
                state = current_state - 1
            else:
                state = current_state
            phone_to_word_state = phone_to_word.num_states() - 1
            for i in range(self.phone_table.num_symbols()):
                if self.phone_table.find(i) == "<eps>":
                    continue
                if self.phone_table.find(i).startswith("#"):
                    continue
                phone_fst.add_arc(
                    state,
                    pywrapfst.Arc(
                        self.phone_table.find("<eps>"),
                        i,
                        pywrapfst.Weight.one(phone_fst.weight_type()),
                        state,
                    ),
                )

                phone_to_word.add_arc(
                    phone_to_word_state,
                    pywrapfst.Arc(
                        i,
                        self.phone_table.find("<eps>"),
                        pywrapfst.Weight.one(phone_fst.weight_type()),
                        phone_to_word_state,
                    ),
                )
        for s in range(current_state + 1):
            phone_fst.add_arc(
                s,
                pywrapfst.Arc(
                    word_end_symbol,
                    word_end_symbol,
                    pywrapfst.Weight.one(phone_fst.weight_type()),
                    s,
                ),
            )
            phone_fst.add_arc(
                s,
                pywrapfst.Arc(
                    word_begin_symbol,
                    word_begin_symbol,
                    pywrapfst.Weight.one(phone_fst.weight_type()),
                    s,
                ),
            )

        phone_fst.set_final(current_state, pywrapfst.Weight.one(phone_fst.weight_type()))
        phone_fst.arcsort("olabel")

        lattice = pynini.compose(phone_fst, phone_to_word)

        projection = pynini.shortestpath(lattice).project("input")
        if projection.start() == pywrapfst.NO_SYMBOL:
            phone_fst.set_input_symbols(self.phone_table)
            phone_fst.set_output_symbols(self.phone_table)
            phone_to_word.set_input_symbols(self.phone_table)
            phone_to_word.set_output_symbols(self.word_table)
            raise PhonesToPronunciationsError(
                text,
                " ".join(self.phone_table.find(x) for x in phone_symbols),
                phone_fst,
                phone_to_word,
            )
        path_string = projection.string(self.phone_table)
        if self.position_dependent_phones:
            path_string = re.sub(r"_[SIBE]\b", "", path_string)

        path_string = re.sub(f" {self.word_end_label}$", "", path_string)
        path_string = path_string.replace(
            f"{self.word_end_label} {self.word_end_label}", self.word_end_label
        )
        path_string = path_string.replace(
            f"{self.word_end_label} {self.word_begin_label}", self.word_begin_label
        )
        path_string = path_string.replace(f"{self.word_end_label}", self.word_begin_label)
        path_string = re.sub(f"^{self.word_begin_label} ", "", path_string)
        word_splits = [x for x in re.split(rf" ?{self.word_begin_label} ?", path_string)]
        return word_splits

    def phones_to_pronunciations(
        self,
        word_symbols: typing.List[int],
        intervals: typing.List[CtmInterval],
        transcription: bool = False,
        text: str = None,
    ) -> HierarchicalCtm:

        phones = [x.symbol for x in intervals]
        word_splits = self._create_pronunciation_string(
            word_symbols,
            phones,
            transcription=transcription,
        )

        actual_words = [self.word_table.find(x) for x in word_symbols]
        if not text:
            text = " ".join(actual_words)
        word_intervals = []
        current_phone_index = 0
        current_word_index = 0
        for i, w in enumerate(actual_words):
            pron = word_splits[current_word_index]
            word_symbol = word_symbols[i]
            if pron == self.silence_phone:
                word_intervals.append(
                    WordCtmInterval(
                        self.silence_word,
                        self.word_table.find(self.silence_word),
                        intervals[current_phone_index : current_phone_index + 1],
                    )
                )
                current_word_index += 1
                current_phone_index += 1
                pron = word_splits[current_word_index]

            phones = pron.split()
            word_intervals.append(
                WordCtmInterval(
                    w,
                    word_symbol,
                    intervals[current_phone_index : current_phone_index + len(phones)],
                )
            )
            current_phone_index += len(phones)
            current_word_index += 1
        if current_word_index != len(word_splits):
            pron = word_splits[current_word_index]
            if pron == self.silence_phone:
                word_intervals.append(
                    WordCtmInterval(
                        self.silence_word,
                        self.word_table.find(self.silence_word),
                        intervals[current_phone_index : current_phone_index + 1],
                    )
                )
        return HierarchicalCtm(word_intervals, text=text)


class G2PCompiler(LexiconCompiler):
    use_g2p = True

    def __init__(
        self,
        fst: pynini.Fst,
        grapheme_table: pywrapfst.SymbolTable,
        phone_table: pywrapfst.SymbolTable,
        silence_phone: str = "sil",
        silence_word: str = "<eps>",
        align_fst: typing.Optional[pynini.Fst] = None,
        position_dependent_phones: bool = False,
    ):
        self._fst = fst
        self._align_fst = align_fst
        self._align_fst.invert()
        self.word_table = grapheme_table
        self.phone_table = phone_table
        self.silence_phone = silence_phone
        self.silence_word = silence_word
        self.word_begin_label = "#1"
        self.word_end_label = "#2"
        self.position_dependent_phones = position_dependent_phones
        self.disambiguation = False

    def phones_to_pronunciations(
        self,
        word_symbols: typing.List[int],
        intervals: typing.List[CtmInterval],
        transcription: bool = False,
        text: str = None,
    ) -> HierarchicalCtm:
        phone_symbols = [x.symbol for x in intervals]
        word_symbols = [self.word_table.find(x) for x in text.split()]
        word_splits = self._create_pronunciation_string(
            word_symbols,
            phone_symbols,
            transcription=transcription,
        )

        # Might need some better logic
        actual_words = [x.replace(" ", "") for x in text.split("<space>")]
        word_intervals = []
        current_phone_index = 0
        current_word_index = 0
        for w in actual_words:
            pron = word_splits[current_word_index]
            if pron == self.silence_phone:
                word_intervals.append(
                    WordCtmInterval(
                        self.silence_word,
                        0,
                        intervals[current_phone_index : current_phone_index + 1],
                    )
                )
                current_word_index += 1
                current_phone_index += 1
                pron = word_splits[current_word_index]

            phones = pron.split()
            word_intervals.append(
                WordCtmInterval(
                    w, 0, intervals[current_phone_index : current_phone_index + len(phones)]
                )
            )
            current_phone_index += len(phones)
            current_word_index += 1
        if current_word_index != len(word_splits):
            pron = word_splits[current_word_index]
            if pron == self.silence_phone:
                word_intervals.append(
                    WordCtmInterval(
                        self.silence_word,
                        0,
                        intervals[current_phone_index : current_phone_index + 1],
                    )
                )
        return HierarchicalCtm(word_intervals, text=text)
