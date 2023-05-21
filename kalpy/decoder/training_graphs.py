import typing

from _kalpy.decoder import TrainingGraphCompiler as _TrainingGraphCompiler
from _kalpy.decoder import TrainingGraphCompilerOptions
from _kalpy.fstext import VectorFstWriter
from _kalpy.hmm import TransitionModel
from _kalpy.tree import ContextDependency
from kalpy.fstext.lexicon import LexiconCompiler
from kalpy.fstext.utils import pynini_to_kaldi


class TrainingGraphCompiler:
    def __init__(
        self,
        transition_model: TransitionModel,
        tree: ContextDependency,
        lexicon_compiler: LexiconCompiler,
        transition_scale: float = 0.0,
        self_loop_scale: float = 0.0,
        batch_size: int = 250,
        reorder: bool = True,
    ):
        self.lexicon = lexicon_compiler
        self.transition_model = transition_model
        self.tree = tree
        self.batch_size = batch_size
        kaldi_fst = pynini_to_kaldi(lexicon_compiler.compile_lexicon())
        self.options = TrainingGraphCompilerOptions(transition_scale, self_loop_scale, reorder)
        self.compiler = _TrainingGraphCompiler(
            transition_model,
            tree,
            kaldi_fst,
            lexicon_compiler.disambiguation_symbols,
            self.options,
        )

    def export_graphs(self, file_name: str, transcripts: typing.Dict[str, str]):

        writer = VectorFstWriter(f"ark:{file_name}")
        for key, transcript in transcripts.items():
            fst = self.compile_fst(transcript)
            writer.Write(key, fst)
        writer.Close()

    def compile_fst(self, transcript):
        transcript = [self.lexicon.word_table.find(x) for x in transcript.split()]
        fst = self.compiler.CompileGraphFromText(transcript)
        return fst
