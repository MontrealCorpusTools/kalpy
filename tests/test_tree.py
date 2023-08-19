from _kalpy.tree import ContextDependency
from _kalpy.util import ReadKaldiObject


def test_transition_model(mono_tree_path):
    tree = ContextDependency()
    ReadKaldiObject(str(mono_tree_path), tree)
    assert tree.NumPdfs() > 0
