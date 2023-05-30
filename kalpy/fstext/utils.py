"""Utility functions for working with fstext"""
import pynini

from _kalpy.fstext import VectorFst


def kaldi_to_pynini(fst: VectorFst) -> pynini.Fst:
    """
    Converts an FST from Kaldi to pynini

    Parameters
    ----------
    fst: :class:`_kalpy.fstext.VectorFst`
        FST from Kaldi

    Returns
    -------
    :class:`~pynini.Fst`
        FST for use in pynini
    """
    return pynini.Fst.read_from_string(fst.write_to_string())


def pynini_to_kaldi(fst: pynini.Fst) -> VectorFst:
    """
    Converts an FST from pynini to Kaldi

    Parameters
    ----------
    fst: :class:`~pynini.Fst`
        FST from pynini

    Returns
    -------
    :class:`_kalpy.fstext.VectorFst`
        VectorFst for use in Kaldi
    """
    return VectorFst.from_string(fst.write_to_string())
