"""Classes for generating training graphs"""
import pathlib
import typing

from _kalpy.decoder import TrainingGraphCompiler as _TrainingGraphCompiler
from _kalpy.decoder import TrainingGraphCompilerOptions
from _kalpy.fstext import VectorFst, VectorFstWriter
from _kalpy.hmm import TransitionModel
from _kalpy.tree import ContextDependency
from _kalpy.util import ReadKaldiObject
from kalpy.fstext.lexicon import LexiconCompiler
from kalpy.fstext.utils import pynini_to_kaldi


class TrainingGraphCompiler:
    """
    Parameters
    ----------
    transition_model_path: str
    tree_path: str
    lexicon_compiler: :class:`~kalpy.fstext.lexicon.LexiconCompiler`
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
        transition_model_path: str,
        tree_path: str,
        lexicon_compiler: LexiconCompiler,
        transition_scale: float = 0.0,
        self_loop_scale: float = 0.0,
        batch_size: int = 250,
    ):
        self.transition_model = TransitionModel()
        ReadKaldiObject(str(transition_model_path), self.transition_model)
        self.tree = ContextDependency()
        ReadKaldiObject(str(tree_path), self.tree)
        self.lexicon_compiler = lexicon_compiler
        self.batch_size = batch_size
        kaldi_fst = pynini_to_kaldi(lexicon_compiler.fst)
        self.options = TrainingGraphCompilerOptions(transition_scale, self_loop_scale)
        self.compiler = _TrainingGraphCompiler(
            self.transition_model,
            self.tree,
            kaldi_fst,
            lexicon_compiler.disambiguation_symbols,
            self.options,
        )

    def export_graphs(
        self,
        file_name: typing.Union[pathlib.Path, str],
        transcripts: typing.Dict[str, str],
        write_scp: bool = False,
    ) -> None:
        """
        Export training graphs to a kaldi archive file (i.e., fsts.ark)

        Parameters
        ----------
        file_name: :class:`~pathlib.Path` or str
            Archive file path to export to
        transcripts: dict[str, str]
            Dictionary of utterance IDs to transcripts
        write_scp: bool
            Flag for whether an SCP file should be generated as well
        """
        file_name = str(file_name)
        if not file_name.endswith(".ark"):
            file_name += ".ark"
        if write_scp:
            write_specifier = f"ark,scp:{file_name},{file_name.replace('.ark', '.scp')}"
        else:
            write_specifier = f"ark:{file_name}"
        writer = VectorFstWriter(write_specifier)
        for key, transcript in transcripts.items():
            fst = self.compile_fst(transcript)
            writer.Write(str(key), fst)
        writer.Close()

    def compile_fst(self, transcript: str) -> VectorFst:
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
        transcript = [self.lexicon_compiler.to_int(x) for x in transcript.split()]
        fst = self.compiler.CompileGraphFromText(transcript)
        return fst
