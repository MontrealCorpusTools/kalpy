import pywrapfst

from _kalpy.hmm import TransitionModel
from _kalpy.tree import ContextDependency
from _kalpy.util import ReadKaldiObject
from kalpy.decoder.training_graphs import TrainingGraphCompiler
from kalpy.fstext.lexicon import LexiconCompiler
from kalpy.fstext.utils import kaldi_to_pynini


def test_transition_model(tree_path, transition_model_path, dictionary_path, temp_dir):
    tm = TransitionModel()
    ReadKaldiObject(str(transition_model_path), tm)
    tree = ContextDependency()
    ReadKaldiObject(str(tree_path), tree)
    lc = LexiconCompiler(position_dependent_phones=False)
    lc.load_pronunciations(dictionary_path)

    gc = TrainingGraphCompiler(tm, tree, lc)
    graph = kaldi_to_pynini(gc.compile_fst("this is the acoustic corpus"))
    print(graph)
    assert graph.num_states() > 0
    assert graph.start() != pywrapfst.NO_STATE_ID

    output_file_name = temp_dir.joinpath("fsts.ark")
    gc.export_graphs(output_file_name, {"1": "this is the acoustic corpus"})
    assert output_file_name.exists()
