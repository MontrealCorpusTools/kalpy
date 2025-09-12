from __future__ import annotations

import os.path
import pathlib
import typing

from _kalpy.gmm import AmDiagGmm
from _kalpy.hmm import HmmTopology, TransitionModel
from _kalpy.tree import ContextDependency
from _kalpy.util import Input, Output, ReadKaldiObject
from kalpy.exceptions import ReadError


def read_transition_model(model_path: typing.Union[str, pathlib.Path]) -> TransitionModel:
    if not os.path.exists(model_path):
        raise ReadError(f"The specified model path {model_path} does not exist.")
    ki = Input()
    ki.Open(str(model_path), True)
    transition_model = TransitionModel()
    transition_model.Read(ki.Stream(), True)
    ki.Close()
    return transition_model


def read_gmm_model(
    model_path: typing.Union[str, pathlib.Path]
) -> typing.Tuple[TransitionModel, AmDiagGmm]:
    if not os.path.exists(model_path):
        raise ReadError(f"The specified model path {model_path} does not exist.")
    ki = Input()
    ki.Open(str(model_path), True)
    transition_model = TransitionModel()
    transition_model.Read(ki.Stream(), True)
    acoustic_model = AmDiagGmm()
    acoustic_model.Read(ki.Stream(), True)
    ki.Close()
    return transition_model, acoustic_model


def read_topology(topo_path: typing.Union[str, pathlib.Path]) -> HmmTopology:
    if not os.path.exists(topo_path):
        raise ReadError(f"The specified topo path {topo_path} does not exist.")
    ki = Input()
    ki.Open(str(topo_path), False)
    topo = HmmTopology()
    topo.Read(ki.Stream(), False)
    ki.Close()
    return topo


def write_gmm_model(
    model_path: typing.Union[str, pathlib.Path],
    transition_model: TransitionModel,
    acoustic_model: AmDiagGmm,
    binary: bool = True,
) -> None:
    ko = Output(str(model_path), binary)
    transition_model.Write(ko.Stream(), binary)
    acoustic_model.Write(ko.Stream(), binary)
    ko.Close()


def write_tree(
    tree_path: typing.Union[str, pathlib.Path], tree: ContextDependency, binary: bool = True
) -> None:
    ko = Output(str(tree_path), binary)
    tree.Write(ko.Stream(), binary)
    ko.Close()


def read_tree(tree_path: typing.Union[str, pathlib.Path]) -> ContextDependency:
    if not os.path.exists(tree_path):
        raise ReadError(f"The specified tree path {tree_path} does not exist.")
    tree = ContextDependency()
    ReadKaldiObject(str(tree_path), tree)
    return tree
