from _kalpy.hmm import TransitionModel
from _kalpy.util import ReadKaldiObject
from kalpy.utils import read_kaldi_object


def test_transition_model(mono_model_path):
    tm = TransitionModel()
    ReadKaldiObject(str(mono_model_path), tm)
    assert len(tm.GetPhones()) > 0


def test_read_kaldi_hmm(mono_model_path):
    tm = read_kaldi_object(TransitionModel, mono_model_path)
    assert len(tm.GetPhones()) > 0
