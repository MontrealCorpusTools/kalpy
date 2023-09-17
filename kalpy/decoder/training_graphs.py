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
    lexicon: typing.Union[pathlib.Path, str, :class:`~kalpy.fstext.lexicon.LexiconCompiler`, VectorFst]
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
        lexicon: typing.Union[pathlib.Path, str, LexiconCompiler, VectorFst],
        words_symbols: typing.Union[pathlib.Path, str, pywrapfst.SymbolTable],
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
        self.lexicon_compiler = None
        self._fst = None
        if isinstance(lexicon, LexiconCompiler):
            self.lexicon_compiler = lexicon
            if self.use_g2p:
                self._fst = self.lexicon_compiler.fst
            else:
                self._fst = self.lexicon_compiler.fst
            disambiguation_symbols = self.lexicon_compiler.disambiguation_symbols
        elif isinstance(lexicon, VectorFst):
            self._fst = lexicon
        else:
            self.lexicon_path = str(lexicon)
        if isinstance(words_symbols, pywrapfst.SymbolTable):
            self.word_table = words_symbols
        else:
            self.word_table = pywrapfst.SymbolTable.read_text(words_symbols)
        self.oov_word = oov_word
        if disambiguation_symbols is None:
            disambiguation_symbols = []
        self.disambiguation_symbols = disambiguation_symbols

    def __del__(self):
        del self._compiler
        del self._fst

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
    def fst(self):
        if self._fst is None:
            return pynini.Fst.read(self.lexicon_path)

    @property
    def compiler(self):
        if self._compiler is None:
            if self._fst is None:
                if self.lexicon_compiler is None:
                    self._fst = pynini.Fst.read(str(self.lexicon_path))
                else:
                    self._fst = self.lexicon_compiler.fst
            self._compiler = _TrainingGraphCompiler(
                self.transition_model,
                self.tree,
                VectorFst.from_pynini(self._fst),
                self.disambiguation_symbols,
                self.options,
            )
        return self._compiler

    def export_graphs(
        self,
        file_name: typing.Union[pathlib.Path, str],
        transcripts: typing.Iterable[typing.Tuple[str, str]],
        write_scp: bool = False,
        callback: typing.Callable = None,
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
        """
        write_specifier = generate_write_specifier(file_name, write_scp)
        writer = VectorFstWriter(write_specifier)
        keys = []
        transcript_batch = []
        num_done = 0
        num_error = 0
        for key, transcript in transcripts:
            keys.append(key)
            if self.use_g2p:
                transcript_batch.append(transcript)
            else:
                transcript_batch.append([self.to_int(x) for x in transcript.split()])
            if len(keys) >= self.batch_size:
                if self.use_g2p:
                    fsts = []
                    for t in transcript_batch:
                        fsts.append(self.compile_fst(t))
                else:
                    fsts = self.compiler.CompileGraphsFromText(transcript_batch)
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
                num_done += batch_done
                num_error += batch_error
                if callback:
                    callback(batch_done)
                keys = []
                transcript_batch = []
        if keys:
            if self.use_g2p:
                fsts = []
                for t in transcript_batch:
                    fsts.append(self.compile_fst(t))
            else:
                fsts = self.compiler.CompileGraphsFromText(transcript_batch)
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
            num_done += batch_done
            num_error += batch_error
            if callback:
                callback(batch_done)
        writer.Close()
        logger.info(f"Done {num_done} utterances, errors on {num_error}.")

    def compile_fst(self, transcript: str) -> typing.Optional[VectorFst]:
        """
        Compile a transcript to a training graph

        Parameters
        ----------
        transcript: str
            Orthographic transcript to compile

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

            lg_fst = fst_determinize_star(lg_fst, use_log=True)
            fst_minimize_encoded(lg_fst)
            fst_push_special(lg_fst)
            clg_fst, disambig_out, ilabels = fst_compose_context(
                lg_fst,
                self.disambiguation_symbols,
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
            fst_add_self_loops(fst, self.transition_model, [], self.options.self_loop_scale)
        else:
            transcript_symbols = [self.to_int(x) for x in transcript.split()]
            fst = self.compiler.CompileGraphFromText(transcript_symbols)
        if fst.Start() == pywrapfst.NO_STATE_ID:
            logger.warning(f"Could not construct FST for '{transcript}")
            return None
        return fst
