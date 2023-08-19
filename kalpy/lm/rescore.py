"""Classes for GMM alignment"""
from __future__ import annotations

import logging
import pathlib
import typing

from _kalpy.fstext import VectorFst
from _kalpy.lat import (
    CompactLattice,
    CompactLatticeWriter,
    ComposeLatticePrunedOptions,
    lm_rescore,
    lm_rescore_carpa,
)
from _kalpy.lm import ConstArpaLm
from kalpy.gmm.data import LatticeArchive
from kalpy.utils import generate_write_specifier

logger = logging.getLogger("kalpy.lm")
logger.setLevel(logging.DEBUG)
logger.write = lambda msg: logger.info(msg) if msg != "\n" else None
logger.flush = lambda: None


class LmRescorer:
    def __init__(
        self,
        g_fst: VectorFst,
        acoustic_scale: float = 0.1,
        lm_scale: float = 1.0,
        lattice_compose_beam: float = 6.0,
        max_arcs: int = 100000,
        initial_num_arcs: int = 100,
        growth_ratio: float = 1.5,
    ):
        self.g_fst = g_fst
        self.acoustic_scale = acoustic_scale
        self.lm_scale = lm_scale
        self.num_done = 0
        self.num_error = 0
        self.options = ComposeLatticePrunedOptions()
        self.options.lattice_compose_beam = lattice_compose_beam
        self.options.max_arcs = max_arcs
        self.options.initial_num_arcs = initial_num_arcs
        self.options.growth_ratio = growth_ratio

    def rescore_utterance(
        self,
        lattice: CompactLattice,
        add_lm: typing.Union[VectorFst, ConstArpaLm],
        utterance_id: str = None,
    ):
        try:
            if isinstance(add_lm, ConstArpaLm):
                new_lattice = lm_rescore_carpa(
                    lattice,
                    self.g_fst,
                    add_lm,
                    self.options,
                    self.lm_scale,
                    self.acoustic_scale,
                )
            else:
                new_lattice = lm_rescore(
                    lattice,
                    self.g_fst,
                    add_lm,
                    self.options,
                    self.lm_scale,
                    self.acoustic_scale,
                )
            self.num_done += 1
        except Exception:
            self.num_error += 1
            logger.warning(f"Error in rescoring {utterance_id}")
            raise
        return new_lattice

    def rescore_utterances(
        self, lattice_archive: LatticeArchive, add_lm: typing.Union[VectorFst, ConstArpaLm]
    ) -> typing.Generator[CompactLattice]:
        for (utterance_id, lattice) in lattice_archive:
            yield utterance_id, self.rescore_utterance(lattice, add_lm, utterance_id)

    def export_lattices(
        self,
        file_name: typing.Union[str, pathlib.Path],
        lattice_archive: LatticeArchive,
        add_lm: typing.Union[VectorFst, ConstArpaLm],
        write_scp: bool = False,
        callback: typing.Callable = None,
    ):
        write_specifier = generate_write_specifier(file_name, write_scp)
        writer = CompactLatticeWriter(write_specifier)
        for utterance_id, lattice in self.rescore_utterances(lattice_archive, add_lm):
            if callback:
                callback(utterance_id)
            writer.Write(str(utterance_id), lattice)
        writer.Close()
