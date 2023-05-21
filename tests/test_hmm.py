from _kalpy.hmm import TransitionModel
from _kalpy.util import ReadKaldiObject


def test_transition_model(transition_model_path):
    tm = TransitionModel()
    ReadKaldiObject(str(transition_model_path), tm)
    assert len(tm.GetPhones()) > 0
