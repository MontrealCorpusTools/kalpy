import pynini

from _kalpy.fstext import VectorFst


def kaldi_to_pynini(fst: VectorFst) -> pynini.Fst:
    return pynini.Fst.read_from_string(fst.write_to_string())


def pynini_to_kaldi(fst: pynini.Fst) -> VectorFst:
    return VectorFst.from_string(fst.write_to_string())
