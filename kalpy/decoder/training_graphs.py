"""Classes for generating training graphs"""
import logging
import pathlib
import typing

import pynini
import pywrapfst

from _kalpy.decoder import TrainingGraphCompiler as _TrainingGraphCompiler
from _kalpy.decoder import TrainingGraphCompilerOptions
from _kalpy.fstext import (
    VectorFst,
    VectorFstWriter,
    fst_add_self_loops,
    fst_arc_sort,
    fst_compose_context,
    fst_determinize_star,
    fst_minimize_encoded,
    fst_push_special,
    fst_rm_eps_local,
    fst_rm_symbols,
    fst_table_compose,
)
from _kalpy.hmm import TransitionModel, make_h_transducer
from _kalpy.tree import ContextDependency
from _kalpy.util import ReadKaldiObject
from kalpy.fstext.lexicon import LexiconCompiler
from kalpy.fstext.utils import kaldi_to_pynini, pynini_to_kaldi
from kalpy.utils import generate_write_specifier

logger = logging.getLogger("kalpy.graphs")
logger.setLevel(logging.DEBUG)
logger.write = lambda msg: logger.info(msg) if msg != "\n" else None
logger.flush = lambda: None


class TrainingGraphCompiler:
    """
    Parameters
    ----------
    model_path: str
        Path to model file
    tree_path: str
        Path to tree file
    lexicon_compiler: :class:`~kalpy.fstext.lexicon.LexiconCompiler`
        Lexicon compiler to use in generating training graphs
    transition_scale: float
        Scale on transitions, typically set to 0 as it will be defined during alignment
    self_loop_scale: float
        Scale on transitions, typically set to 0 as it will be defined during alignment
    batch_size: int
        Batch size for compilation, larger batches will be faster but use more memory

    Attributes
    ----------
    transition_model: :class:`~_kalpy.hmm.TransitionModel`
        Transition model to use in compiling the graphs
    tree: :class:`~_kalpy.hmm.TransitionModel`
        Context dependency tree to use in compiling the graphs
    """

    def __init__(
        self,
        model_path: typing.Union[pathlib.Path, str],
        tree_path: typing.Union[pathlib.Path, str],
        lexicon_compiler: LexiconCompiler,
        transition_scale: float = 0.0,
        self_loop_scale: float = 0.0,
        batch_size: int = 1000,
        use_g2p: bool = False,
        disambiguation_symbols: typing.List[int] = None,
        oov_word: str = "<unk>",
    ):
        self.transition_model = TransitionModel()
        ReadKaldiObject(str(model_path), self.transition_model)
        self.tree = ContextDependency()
        ReadKaldiObject(str(tree_path), self.tree)
        self.batch_size = batch_size
        self.options = TrainingGraphCompilerOptions(transition_scale, self_loop_scale)
        self._compiler = None
        self.use_g2p = use_g2p
        self.lexicon_path = None
        self.lexicon_compiler = lexicon_compiler
        self.oov_word = oov_word
        if disambiguation_symbols is None:
            disambiguation_symbols = []
        self.disambiguation_symbols = disambiguation_symbols
        self._kaldi_fst = self.lexicon_compiler.fst
        if not isinstance(self._kaldi_fst, VectorFst):
            self._kaldi_fst = VectorFst.from_pynini(self._kaldi_fst)

    def __del__(self):
        del self._compiler
        del self._kaldi_fst

    @property
    def word_table(self):
        return self.lexicon_compiler.word_table

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
    def compiler(self):
        if self._compiler is None:
            disambiguation_symbols = []
            if self.lexicon_compiler is not None and self.lexicon_compiler.disambiguation:
                disambiguation_symbols = self.lexicon_compiler.disambiguation_symbols
            self._compiler = _TrainingGraphCompiler(
                self.transition_model,
                self.tree,
                self._kaldi_fst,
                disambiguation_symbols,
                self.options,
            )
        return self._compiler

    def export_graphs(
        self,
        file_name: typing.Union[pathlib.Path, str],
        transcripts: typing.Iterable[typing.Tuple[str, str]],
        write_scp: bool = False,
        callback: typing.Callable = None,
        interjection_words: typing.List[str] = None,
        cutoff_pattern: str = None,
    ):
        """
        Export training graphs to a kaldi archive file (i.e., fsts.ark)

        Parameters
        ----------
        file_name: :class:`~pathlib.Path` or str
            Archive file path to export to
        transcripts: iterable[tuple[str, str]]
            Dictionary of utterance IDs to transcripts
        write_scp: bool
            Flag for whether an SCP file should be generated as well
        callback: callable, optional
            Optional callback function for progress updates
        interjection_words: list[str], optional
            List of words to add as interjections to the transcripts
        cutoff_pattern: str, optional
            Cutoff symbol to use for inserting cutoffs before words
        """
        write_specifier = generate_write_specifier(file_name, write_scp)
        writer = VectorFstWriter(write_specifier)
        keys = []
        transcript_batch = []
        original_transcripts = []
        num_done = 0
        num_error = 0
        logger.debug(f"DISAMBIGUATION: {self.lexicon_compiler.disambiguation}")
        for key, transcript in transcripts:
            keys.append(key)
            original_transcripts.append(transcript)
            if self.use_g2p:
                transcript_batch.append(transcript)
            elif interjection_words:
                transcript_batch.append(
                    self.generate_utterance_graph(transcript, interjection_words, cutoff_pattern)
                )
            else:
                transcript_batch.append([self.to_int(x) for x in transcript.split()])
            if len(keys) >= self.batch_size:
                if self.use_g2p:
                    fsts = []
                    for t in transcript_batch:
                        fsts.append(self.compile_fst(t))
                elif interjection_words:
                    fsts = self.compiler.CompileGraphs(transcript_batch)
                else:
                    fsts = self.compiler.CompileGraphsFromText(transcript_batch)
                assert len(fsts) == len(keys)
                batch_done = 0
                batch_error = 0
                for i, key in enumerate(keys):
                    fst = fsts[i]
                    if fst.Start() == pywrapfst.NO_STATE_ID:
                        original_transcript = original_transcripts[i]
                        transcript = transcript_batch[i]
                        logger.warning(
                            f"Skipping {key}, empty FST for {transcript} ({original_transcript})"
                        )
                        batch_error += 1
                        continue
                    writer.Write(str(key), fst)
                    del fst
                    batch_done += 1
                num_done += batch_done
                num_error += batch_error
                logger.debug(f"Done {num_done} utterances, errors on {num_error}.")
                if callback:
                    callback(batch_done)
                keys = []
                transcript_batch = []
                original_transcripts = []
                del fsts
        if keys:
            if self.use_g2p:
                fsts = []
                for t in transcript_batch:
                    fsts.append(self.compile_fst(t))
            elif interjection_words:
                fsts = self.compiler.CompileGraphs(transcript_batch)
            else:
                fsts = self.compiler.CompileGraphsFromText(transcript_batch)
            del transcript_batch
            assert len(fsts) == len(keys)
            batch_done = 0
            batch_error = 0
            for i, key in enumerate(keys):
                fst = fsts[i]
                if fst.Start() == pywrapfst.NO_STATE_ID:
                    logger.warning(f"Skipping {key}, empty FST")
                    batch_error += 1
                    continue
                writer.Write(str(key), fst)
                batch_done += 1
                del fst
            num_done += batch_done
            num_error += batch_error
            del fsts
            if callback:
                callback(batch_done)
        writer.Close()
        logger.info(f"Done {num_done} utterances, errors on {num_error}.")

    def generate_utterance_graph(
        self,
        transcript: str,
        interjection_words: typing.List[str] = None,
        cutoff_pattern: str = None,
    ) -> typing.Optional[VectorFst]:
        if interjection_words is None:
            interjection_words = []
        default_interjection_cost = 3.0
        cutoff_interjection_cost = default_interjection_cost
        cutoff_symbol = -1
        if cutoff_pattern is not None and self.word_table.member(cutoff_pattern):
            cutoff_symbol = self.to_int(cutoff_pattern)
        interjection_costs = {}
        if interjection_words:
            for iw in interjection_words:
                if not self.word_table.member(iw):
                    continue
                if isinstance(interjection_words, dict):
                    interjection_cost = interjection_words[iw] * default_interjection_cost
                else:
                    interjection_cost = default_interjection_cost
                if isinstance(iw, str):
                    iw = self.to_int(iw)
                if iw == cutoff_symbol:
                    cutoff_interjection_cost = interjection_cost
                    continue
                interjection_costs[iw] = interjection_cost
        g = pynini.Fst()
        start_state = g.add_state()
        g.set_start(start_state)
        if isinstance(transcript, str):
            transcript = transcript.split()
        for word_symbol in transcript:
            if not isinstance(word_symbol, int):
                word_symbol = self.to_int(word_symbol)
            interjection_state = g.add_state()
            for iw_symbol, interjection_cost in interjection_costs.items():
                g.add_arc(
                    start_state,
                    pywrapfst.Arc(
                        iw_symbol,
                        iw_symbol,
                        pywrapfst.Weight(g.weight_type(), interjection_cost),
                        interjection_state,
                    ),
                )
            if cutoff_pattern is not None:
                cutoff_word = f"{cutoff_pattern[:-1]}-{self.word_table.find(word_symbol)}{cutoff_pattern[-1]}"
                if self.word_table.member(cutoff_word):
                    iw_symbol = self.to_int(cutoff_word)
                    g.add_arc(
                        start_state,
                        pywrapfst.Arc(
                            iw_symbol,
                            iw_symbol,
                            pywrapfst.Weight(g.weight_type(), cutoff_interjection_cost),
                            interjection_state,
                        ),
                    )
            g.add_arc(
                start_state,
                pywrapfst.Arc(
                    word_symbol,
                    word_symbol,
                    pywrapfst.Weight(g.weight_type(), default_interjection_cost),
                    interjection_state,
                ),
            )
            g.add_arc(
                start_state,
                pywrapfst.Arc(
                    self.word_table.find("<eps>"),
                    self.word_table.find("<eps>"),
                    pywrapfst.Weight(g.weight_type(), 1.0),
                    interjection_state,
                ),
            )
            final_state = g.add_state()
            g.add_arc(
                interjection_state,
                pywrapfst.Arc(
                    word_symbol,
                    word_symbol,
                    pywrapfst.Weight.one(g.weight_type()),
                    final_state,
                ),
            )
            start_state = final_state
        final_state = g.add_state()
        for iw_symbol, interjection_cost in interjection_costs.items():
            g.add_arc(
                start_state,
                pywrapfst.Arc(
                    iw_symbol,
                    iw_symbol,
                    pywrapfst.Weight(g.weight_type(), interjection_cost),
                    final_state,
                ),
            )
        g.add_arc(
            start_state,
            pywrapfst.Arc(
                self.word_table.find("<eps>"),
                self.word_table.find("<eps>"),
                pywrapfst.Weight.one(g.weight_type()),
                final_state,
            ),
        )
        g.set_final(final_state, pywrapfst.Weight.one(g.weight_type()))
        g = VectorFst.from_pynini(g)
        return g

    def compile_fst(
        self,
        transcript: str,
        interjection_words: typing.List[str] = None,
        cutoff_pattern: str = None,
    ) -> typing.Optional[VectorFst]:
        """
        Compile a transcript to a training graph

        Parameters
        ----------
        transcript: str
            Orthographic transcript to compile
        interjection_words: list[str], optional
            List of words to add as interjections to the transcript
        cutoff_pattern: str, optional
            Cutoff symbol to use for inserting cutoffs before words

        Returns
        -------
        :class:`_kalpy.fstext.VectorFst`
            Training graph of transcript
        """
        if self.use_g2p:
            g_fst = pynini.accep(transcript, token_type=self.word_table)
            lg_fst = pynini.compose(g_fst, self._fst, compose_filter="alt_sequence")
            lg_fst = lg_fst.project("output").rmepsilon()
            weight_type = lg_fst.weight_type()
            weight_threshold = pywrapfst.Weight(weight_type, 2.0)
            state_threshold = 256 + 2 * lg_fst.num_states()
            lg_fst = pynini.determinize(lg_fst, nstate=state_threshold, weight=weight_threshold)
            lg_fst = VectorFst.from_pynini(lg_fst)
            disambig_syms_in = (
                []
                if not self.lexicon_compiler.disambiguation
                else self.lexicon_compiler.disambiguation_symbols
            )
            lg_fst = fst_determinize_star(lg_fst, use_log=True)
            fst_minimize_encoded(lg_fst)
            fst_push_special(lg_fst)
            clg_fst, disambig_out, ilabels = fst_compose_context(
                lg_fst,
                disambig_syms_in,
                self.tree.ContextWidth(),
                self.tree.CentralPosition(),
            )
            fst_arc_sort(clg_fst, sort_type="ilabel")
            h, disambig = make_h_transducer(self.tree, self.transition_model, ilabels)
            fst = fst_table_compose(h, clg_fst)
            if fst.Start() == pywrapfst.NO_STATE_ID:
                logger.debug(f"Falling back to pynini compose for '{transcript}")
                h = kaldi_to_pynini(h)
                clg_fst = kaldi_to_pynini(clg_fst)
                fst = pynini_to_kaldi(pynini.compose(h, clg_fst))
            fst_determinize_star(fst, use_log=True)
            fst_rm_symbols(fst, disambig)
            fst_rm_eps_local(fst)
            fst_minimize_encoded(fst)
            fst_add_self_loops(
                fst, self.transition_model, disambig_syms_in, self.options.self_loop_scale
            )
        elif interjection_words:
            g = self.generate_utterance_graph(transcript, interjection_words, cutoff_pattern)
            # fst = VectorFst()
            # self.compiler.CompileGraph(g, fst)
            # lg_fst = pynini.compose(self._fst, g, compose_filter="alt_sequence")
            # lg_fst = VectorFst.from_pynini(lg_fst)
            lg_fst = fst_table_compose(self._kaldi_fst, g)

            disambig_syms_in = (
                []
                if not self.lexicon_compiler.disambiguation
                else self.lexicon_compiler.disambiguation_symbols
            )
            lg_fst = fst_determinize_star(lg_fst, use_log=True)
            fst_minimize_encoded(lg_fst)
            fst_push_special(lg_fst)
            clg_fst, disambig_out, ilabels = fst_compose_context(
                lg_fst,
                disambig_syms_in,
                self.tree.ContextWidth(),
                self.tree.CentralPosition(),
            )
            fst_arc_sort(clg_fst, sort_type="ilabel")
            h, disambig = make_h_transducer(self.tree, self.transition_model, ilabels)
            fst = fst_table_compose(h, clg_fst)
            if fst.Start() == pywrapfst.NO_STATE_ID:
                logger.debug(f"Falling back to pynini compose for '{transcript}")
                h = kaldi_to_pynini(h)
                clg_fst = kaldi_to_pynini(clg_fst)
                fst = pynini_to_kaldi(pynini.compose(h, clg_fst))
            fst_determinize_star(fst, use_log=True)
            fst_rm_symbols(fst, disambig)
            fst_rm_eps_local(fst)
            fst_minimize_encoded(fst)
            fst_add_self_loops(
                fst, self.transition_model, disambig_syms_in, self.options.self_loop_scale
            )
        else:
            transcript_symbols = [self.to_int(x) for x in transcript.split()]
            fst = self.compiler.CompileGraphFromText(transcript_symbols)
        if fst.Start() == pywrapfst.NO_STATE_ID:
            logger.warning(f"Could not construct FST for '{transcript}")
            return None
        return fst
