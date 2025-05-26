from __future__ import annotations

import pathlib
import typing

from _kalpy.gmm import AmDiagGmm
from _kalpy.hmm import HmmTopology, TransitionModel
from _kalpy.tree import ContextDependency
from _kalpy.util import Input, Output, ReadKaldiObject, align_intervals
from kalpy.gmm.data import CtmInterval


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


def fix_many_to_one_alignments(alignments, custom_mapping):
    test_keys = set(x for x in custom_mapping.keys() if " " in x)
    ref_keys = set()
    for val in custom_mapping.values():
        ref_keys.update(x for x in val if " " in x)
    new_ref = []
    new_test = []
    for a in alignments:
        for i, sa in enumerate(a.seqA):
            sb = a.seqB[i]
            if i != 0:
                prev_sa = a.seqA[i - 1]
                prev_sb = a.seqB[i - 1]
                ref_key = " ".join(x.label for x in [prev_sa, sa] if x != "-")
                test_key = " ".join(x.label for x in [prev_sb, sb] if x != "-")
                if (
                    ref_key in ref_keys
                    and test_key in custom_mapping
                    and ref_key in custom_mapping[test_key]
                ):
                    new_ref[-1].label = ref_key
                    new_ref[-1].end = sa.end
                    if sb != "-":
                        new_test.append(sb)
                    continue
                if (
                    test_key in test_keys
                    and test_key in custom_mapping
                    and ref_key in custom_mapping[test_key]
                ):
                    new_test[-1].label = test_key
                    new_test[-1].end = sb.end
                    if sa != "-":
                        new_ref.append(sa)
                    continue
            if sa != "-":
                new_ref.append(sa)
            if sb != "-":
                new_test.append(sb)
        return new_ref, new_test


def align_phones(
    ref: typing.List[CtmInterval],
    test: typing.List[CtmInterval],
    silence_phone: str,
    ignored_phones: typing.Set[str] = None,
    custom_mapping: typing.Optional[typing.Dict[str, typing.Collection[str]]] = None,
    debug: bool = False,
) -> typing.Tuple[float, float, typing.Dict[typing.Tuple[str, str], int]]:
    """
    Align phones based on how much they overlap and their phone label, with the ability to specify a custom mapping for
    different phone labels to be scored as if they're the same phone

    Parameters
    ----------
    ref: list[:class:`~montreal_forced_aligner.data.CtmInterval`]
        List of CTM intervals as reference
    test: list[:class:`~montreal_forced_aligner.data.CtmInterval`]
        List of CTM intervals to compare to reference
    silence_phone: str
        Silence phone (these are ignored in the final calculation)
    ignored_phones: set[str], optional
        Phones that should be ignored in score calculations (silence phone is automatically added)
    custom_mapping: dict[str, str], optional
        Mapping of phones to treat as matches even if they have different symbols
    debug: bool, optional
        Flag for logging extra information about alignments

    Returns
    -------
    float
        Score based on the average amount of overlap in phone intervals
    float
        Phone error rate
    dict[tuple[str, str], int]
        Dictionary of error pairs with their counts
    """

    if ignored_phones is None:
        ignored_phones = set()
    if not isinstance(ignored_phones, set):
        ignored_phones = set(ignored_phones)
    ref = [x.to_kalpy_interval() for x in ref]
    test = [x.to_kalpy_interval() for x in test]
    if custom_mapping is None:
        custom_mapping = {}
    alignment = align_intervals(ref, test, silence_phone, custom_mapping)
    if custom_mapping:
        ref, test = fix_many_to_one_alignments(alignment, custom_mapping)

    overlap_count = 0
    overlap_sum = 0
    num_insertions = 0
    num_deletions = 0
    num_substitutions = 0
    errors = collections.Counter()
    ignored_phones.add(silence_phone)
    for a in alignments:
        for i, sa in enumerate(a.seqA):
            sb = a.seqB[i]
            if sa == "-":
                if sb.label not in ignored_phones:
                    errors[(sa, sb.label)] += 1
                    num_insertions += 1
                else:
                    continue
            elif sb == "-":
                if sa.label not in ignored_phones:
                    errors[(sa.label, sb)] += 1
                    num_deletions += 1
                else:
                    continue
            else:
                if sa.label in ignored_phones:
                    continue
                overlap_sum += (abs(sa.begin - sb.begin) + abs(sa.end - sb.end)) / 2
                overlap_count += 1
                if compare_labels(sa.label, sb.label, silence_phone, mapping=custom_mapping) > 0:
                    num_substitutions += 1
                    errors[(sa.label, sb.label)] += 1
    if overlap_count:
        score = overlap_sum / overlap_count
    else:
        score = None
    phone_error_rate = (num_insertions + num_deletions + (2 * num_substitutions)) / len(ref)
    if debug:
        import logging

        logger = logging.getLogger("mfa")
        logger.debug(
            f"{pairwise2.format_alignment(*alignments[0])}\nScore: {score}\nPER: {phone_error_rate}\nErrors: {errors}"
        )
    return score, phone_error_rate, errors
