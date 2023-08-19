import pathlib
import typing

from _kalpy.gmm import AmDiagGmm
from _kalpy.hmm import HmmTopology, TransitionModel
from _kalpy.tree import ContextDependency
from _kalpy.util import Input, Output, ReadKaldiObject


def read_transition_model(model_path: typing.Union[str, pathlib.Path]) -> TransitionModel:
    ki = Input()
    ki.Open(str(model_path), True)
    transition_model = TransitionModel()
    transition_model.Read(ki.Stream(), True)
    ki.Close()
    return transition_model


def read_gmm_model(
    model_path: typing.Union[str, pathlib.Path]
) -> typing.Tuple[TransitionModel, AmDiagGmm]:
    ki = Input()
    ki.Open(str(model_path), True)
    transition_model = TransitionModel()
    transition_model.Read(ki.Stream(), True)
    acoustic_model = AmDiagGmm()
    acoustic_model.Read(ki.Stream(), True)
    ki.Close()
    return transition_model, acoustic_model


def read_topology(topo_path: typing.Union[str, pathlib.Path]) -> HmmTopology:
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
    tree = ContextDependency()
    ReadKaldiObject(str(tree_path), tree)
    return tree
