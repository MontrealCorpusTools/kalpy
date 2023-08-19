"""Classes for generating training graphs"""
import logging
import os
import pathlib
import re
import typing

import pywrapfst

from _kalpy.decoder import TrainingGraphCompilerOptions
from _kalpy.fstext import (
    ConstFst,
    VectorFst,
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
from _kalpy.lm import ArpaParseOptions, BuildConstArpaLm, ConstArpaLm, arpa_to_fst
from _kalpy.tree import ContextDependency
from _kalpy.util import ReadKaldiObject
from kalpy.fstext.lexicon import LexiconCompiler
from kalpy.fstext.utils import kaldi_to_pynini, pynini_to_kaldi, pynini_to_kaldi_const

logger = logging.getLogger("kalpy.decode_graph")
logger.write = lambda msg: logger.info(msg) if msg != "\n" else None
logger.flush = lambda: None


class DecodeGraphCompiler:
    """
    Parameters
    ----------
    acoustic_model_path: str
        Path to model file
    tree_path: str
        Path to tree file
    lexicon_compiler: :class:`~kalpy.fstext.lexicon.LexiconCompiler`
        Lexicon compiler to use in generating training graphs
    transition_scale: float
        Scale on transitions, typically set to 0 as it will be defined during alignment
    self_loop_scale: float
        Scale on transitions, typically set to 0 as it will be defined during alignment

    Attributes
    ----------
    transition_model: :class:`~_kalpy.hmm.TransitionModel`
        Transition model to use in compiling the graphs
    tree: :class:`~_kalpy.hmm.TransitionModel`
        Context dependency tree to use in compiling the graphs
    """

    def __init__(
        self,
        acoustic_model_path: typing.Union[str, pathlib.Path],
        tree_path: typing.Union[str, pathlib.Path],
        lexicon_compiler: LexiconCompiler,
        arpa_path: typing.Union[str, pathlib.Path] = None,
        transition_scale: float = 1.0,
        self_loop_scale: float = 0.1,
    ):
        self.acoustic_model_path = acoustic_model_path
        self.transition_model = TransitionModel()
        ReadKaldiObject(str(acoustic_model_path), self.transition_model)
        self.tree = ContextDependency()
        ReadKaldiObject(str(tree_path), self.tree)
        self.lexicon_compiler = lexicon_compiler
        self.context_width = self.tree.ContextWidth()
        self.central_pos = self.tree.CentralPosition()
        self.options = TrainingGraphCompilerOptions(transition_scale, self_loop_scale)
        self.hclg_fst = None
        self.g_fst = None
        self.lg_fst = None
        self.g_carpa_path = None
        if arpa_path is not None:
            self.compile_hclg_fst(arpa_path)

    def export_hclg(
        self,
        arpa_path: typing.Union[pathlib.Path, str],
        file_name: typing.Union[pathlib.Path, str],
    ) -> None:
        """
        Export HCLG.fst

        Parameters
        ----------
        file_name: :class:`~pathlib.Path` or str
            Path to save HCLG.fst
        arpa_path: :class:`~pathlib.Path` or str
            Path to ARPA format language model
        """
        file_name = str(file_name)
        if self.hclg_fst is None:
            self.compile_hclg_fst(str(arpa_path))
        self.hclg_fst.Write(file_name)

    def export_g(
        self,
        file_name: typing.Union[pathlib.Path, str],
    ) -> None:
        """
        Export g.fst

        Parameters
        ----------
        file_name: :class:`~pathlib.Path` or str
            Path to save g.fst
        """
        if self.g_fst is None:
            return
        file_name = str(file_name)
        self.g_fst.Write(file_name)

    def load_from_file(
        self,
        hclg_fst_path: typing.Union[pathlib.Path, str],
    ) -> None:
        """
        Read HCLG.fst from file

        Parameters
        ----------
        hclg_fst_path: :class:`~pathlib.Path` or str
            Path to read HCLG.fst
        """
        hclg_fst_path = str(hclg_fst_path)
        self.hclg_fst = ConstFst.Read(hclg_fst_path)

    def load_g_from_file(
        self,
        g_fst_path: typing.Union[pathlib.Path, str],
    ) -> None:
        """
        Read g.fst from file

        Parameters
        ----------
        g_fst_path: :class:`~pathlib.Path` or str
            Path to read HCLG.fst
        """
        g_fst_path = str(g_fst_path)
        self.g_fst = VectorFst.Read(g_fst_path)

    def compile_g_fst(self, arpa_path: typing.Union[str, pathlib.Path]) -> VectorFst:
        """
        Compile G.fst for a language model

        Parameters
        ----------
        arpa_path: str
            Path to ARPA format language model

        Returns
        -------
        :class:`_kalpy.fstext.VectorFst`
            G.fst
        """
        if self.g_fst is None:
            self.g_fst = arpa_to_fst(str(arpa_path), self.lexicon_compiler.word_table)

        return self.g_fst

    @property
    def g_carpa(self) -> ConstArpaLm:
        g_carpa = ConstArpaLm()
        ReadKaldiObject(self.g_carpa_path, g_carpa)
        return g_carpa

    def compile_g_carpa(
        self,
        arpa_path: typing.Union[str, pathlib.Path],
        compiled_path: typing.Union[str, pathlib.Path],
    ) -> None:
        """
        Compile G.carpa for a language model

        Parameters
        ----------
        arpa_path: str
            Path to ARPA format language model
        compiled_path: str
            Path to compiled G.carpa
        """
        compiled_path = str(compiled_path)
        temp_carpa_path = compiled_path + ".temp"
        with open(arpa_path, "r", encoding="utf8") as f, open(
            temp_carpa_path, "w", encoding="utf8"
        ) as outf:
            current_order = -1
            num_oov_lines = 0
            for line in f:
                line = line.strip()
                col = line.split()
                if current_order == -1 and not re.match(r"^\\data\\$", line):
                    continue
                if re.match(r"^\\data\\$", line):
                    current_order = 0
                    outf.write(line + "\n")
                elif re.match(r"^\\[0-9]*-grams:$", line):
                    current_order = int(re.sub(r"\\([0-9]*)-grams:$", r"\1", line))
                    outf.write(line + "\n")
                elif re.match(r"^\\end\\$", line):
                    outf.write(line + "\n")
                elif not line:
                    if current_order >= 1:
                        outf.write("\n")
                else:
                    if current_order == 0:
                        outf.write(line + "\n")
                    else:
                        if len(col) > 2 + current_order or len(col) < 1 + current_order:
                            raise Exception(f'Bad line in arpa lm "{line}"')
                        prob = col.pop(0)
                        is_oov = False
                        for i in range(current_order):
                            if not self.lexicon_compiler.word_table.member(col[i]):
                                is_oov = True
                                num_oov_lines += 1
                                break
                            col[i] = str(self.lexicon_compiler.word_table.find(col[i]))
                        if not is_oov:
                            rest_of_line = " ".join(col)
                            outf.write(f"{prob}\t{rest_of_line}\n")
        options = ArpaParseOptions()
        options.bos_symbol = self.lexicon_compiler.word_table.find("<s>")
        options.eos_symbol = self.lexicon_compiler.word_table.find("</s>")
        options.unk_symbol = self.lexicon_compiler.word_table.find(self.lexicon_compiler.oov_word)
        self.g_carpa_path = str(compiled_path)
        success = BuildConstArpaLm(options, str(temp_carpa_path), self.g_carpa_path)
        if not success:
            logger.error(f"Error compiling G.carpa from {arpa_path}")
        else:
            os.remove(temp_carpa_path)

    def compile_lg_fst(self, arpa_path: typing.Union[str, pathlib.Path]) -> VectorFst:
        """
        Compile LG.fst for a language model

        Parameters
        ----------
        arpa_path: str
            Path to ARPA format language model

        Returns
        -------
        :class:`_kalpy.fstext.VectorFst`
            LG.fst
        """
        g_fst = self.compile_g_fst(arpa_path)
        l_fst = pynini_to_kaldi(self.lexicon_compiler.fst)
        lg_fst = fst_table_compose(l_fst, g_fst)
        fst_determinize_star(lg_fst, use_log=True)
        fst_minimize_encoded(lg_fst)
        fst_push_special(lg_fst)
        self.lg_fst = lg_fst
        return lg_fst

    def compile_clg_fst(
        self, arpa_path: typing.Union[str, pathlib.Path]
    ) -> typing.Tuple[VectorFst, typing.List[typing.List[int]]]:
        """
        Compile CLG.fst for a language model

        Parameters
        ----------
        arpa_path: str
            Path to ARPA format language model

        Returns
        -------
        :class:`_kalpy.fstext.VectorFst`
            CLG.fst
        list[list[int]]
            Ilabels in CLG FST
        """
        if self.lg_fst is None:
            self.lg_fst = self.compile_lg_fst(arpa_path)
        clg_fst, disambig_out, ilabels = fst_compose_context(
            self.lg_fst,
            self.lexicon_compiler.disambiguation_symbols,
            self.context_width,
            self.central_pos,
        )

        fst_arc_sort(clg_fst, sort_type="ilabel")
        return clg_fst, ilabels

    def compile_hclg_fst(self, arpa_path: typing.Union[str, pathlib.Path]):
        """
        Compile HCLG.fst for a language model

        Parameters
        ----------
        arpa_path: str
            Path to ARPA format language model

        Returns
        -------
        :class:`_kalpy.fstext.ConstFst`
            HCLG.fst
        """
        clg_fst, ilabels = self.compile_clg_fst(arpa_path)

        h, disambig = make_h_transducer(self.tree, self.transition_model, ilabels)
        hclga_fst = fst_table_compose(h, clg_fst)
        fst_determinize_star(hclga_fst, use_log=True)
        fst_rm_symbols(hclga_fst, disambig)
        fst_rm_eps_local(hclga_fst)
        fst_minimize_encoded(hclga_fst)
        fst_add_self_loops(hclga_fst, self.transition_model, [], self.options.self_loop_scale)
        hclg = pywrapfst.convert(kaldi_to_pynini(hclga_fst), "const")
        self.hclg_fst = pynini_to_kaldi_const(hclg)
