from _kalpy.tree import ContextDependency
from _kalpy.util import ReadKaldiObject


def test_transition_model(tree_path):
    tree = ContextDependency()
    ReadKaldiObject(str(tree_path), tree)
    assert tree.NumPdfs() > 0
