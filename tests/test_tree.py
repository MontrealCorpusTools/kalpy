import collections
import itertools

from _kalpy.tree import ContextDependency
from _kalpy.util import ReadKaldiObject

from kalpy.gmm.utils import read_transition_model


def test_transition_model(mono_tree_path):
    tree = ContextDependency()
    ReadKaldiObject(str(mono_tree_path), tree)
    assert tree.NumPdfs() > 0


def test_read_from_path(sat_tree_path):
    tree = ContextDependency()
    ReadKaldiObject(str(sat_tree_path), tree)
    loaded = ContextDependency(str(sat_tree_path))
    assert loaded.NumPdfs() == tree.NumPdfs()
    assert loaded.ContextWidth() == tree.ContextWidth()


def pdfs_by_phone_and_pdf_class(model_path):
    """The pdfs the transition model uses for each (phone, pdf class)"""
    transition_model = read_transition_model(model_path)
    pdfs = collections.defaultdict(set)
    for transition_id in range(1, transition_model.NumTransitionIds() + 1):
        key = (
            transition_model.TransitionIdToPhone(transition_id),
            transition_model.TransitionIdToPdfClass(transition_id),
        )
        pdfs[key].add(transition_model.TransitionIdToPdf(transition_id))
    return pdfs


def test_compute_monophone(mono_tree_path, mono_model_path):
    tree = ContextDependency()
    ReadKaldiObject(str(mono_tree_path), tree)
    expected = pdfs_by_phone_and_pdf_class(mono_model_path)
    for (phone, pdf_class), pdfs in expected.items():
        succeeded, pdf_id = tree.Compute([phone], pdf_class)
        assert succeeded
        assert {pdf_id} == pdfs


def test_compute_triphone(sat_tree_path, sat_model_path):
    tree = ContextDependency()
    ReadKaldiObject(str(sat_tree_path), tree)
    assert tree.ContextWidth() == 3
    expected = pdfs_by_phone_and_pdf_class(sat_model_path)
    phones = sorted({phone for phone, _ in expected})[:6]
    for left, phone, right in itertools.product(phones, repeat=3):
        for pdf_class in sorted(c for p, c in expected if p == phone):
            succeeded, pdf_id = tree.Compute([left, phone, right], pdf_class)
            assert succeeded
            assert pdf_id in expected[(phone, pdf_class)]
