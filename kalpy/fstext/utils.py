"""Utility functions for working with fstext"""
import typing

import pynini

from _kalpy.fstext import ConstFst, VectorFst


def kaldi_to_pynini(fst: typing.Union[VectorFst, ConstFst]) -> pynini.Fst:
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


def pynini_to_kaldi_const(fst: pynini.Fst) -> ConstFst:
    """
    Converts an FST from pynini to Kaldi ConstFst

    Parameters
    ----------
    fst: :class:`~pynini.Fst`
        FST from pynini

    Returns
    -------
    :class:`_kalpy.fstext.ConstFst`
        ConstFst for use in Kaldi
    """
    return ConstFst.from_string(fst.write_to_string())
