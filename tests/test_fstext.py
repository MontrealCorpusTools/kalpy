import pynini
import pywrapfst

from _kalpy.fstext import VectorFst


def test_fst_casting():
    fst = pynini.Fst()
    start = fst.add_state()
    end = fst.add_state()
    fst.set_start(start)
    fst.set_final(end, 0)
    fst.add_arc(start, pywrapfst.Arc(0, 0, 0, end))
    kaldi_fst = VectorFst.from_pynini(fst)
    assert kaldi_fst.Start() == start
    assert kaldi_fst.NumStates() == 2
    assert kaldi_fst.NumArcs(start) == 1


def test_fst_read(mono_temp_dir):
    fst_path = mono_temp_dir.joinpath("lexicon.fst")
    kaldi_fst = VectorFst.Read(str(fst_path))
    assert kaldi_fst.Start() == 0
    assert kaldi_fst.NumStates() > 0
