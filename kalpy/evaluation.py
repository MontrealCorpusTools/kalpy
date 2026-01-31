"""Module for evaluating alignments"""
from __future__ import annotations

import collections
import typing

from _kalpy.util import IntervalAlignment, align_intervals
from kalpy.gmm.data import CtmInterval, WordCtmInterval

if typing.TYPE_CHECKING:
    from kalpy.fstext.lexicon import LexiconCompiler


def compare_labels(
    reference_label: str,
    test_label: str,
    silence_phones: typing.Set[str],
    mapping: typing.Optional[typing.Dict[str, typing.Collection[str]]] = None,
) -> int:
    """
    Score two labels based on whether they match or count as the same phone based on the mapping

    Parameters
    ----------
    reference_label: str
        Reference label
    test_label: str
        Label to compare to reference label
    silence_phones: set[str]
        Label corresponding to optionally inserted silence
    mapping: Optional[dict[str, set[str]]]
        Mapping to form equivalence between different label sets

    Returns
    -------
    int
        0 if labels match or if they're in the mapping 2 otherwise. If one of the labels is the silence_phone, returns 10
    """
    if reference_label == test_label:
        return 0
    if mapping is not None and test_label in mapping:
        if isinstance(mapping[test_label], str):
            if mapping[test_label] == reference_label:
                return 0
        elif reference_label in mapping[test_label]:
            return 0
    ref = reference_label.lower()
    test = test_label.lower()
    if ref == test:
        return 0
    if reference_label in silence_phones or test_label in silence_phones:
        return 10
    return 2


def overlap_scoring(
    first_element: CtmInterval,
    second_element: CtmInterval,
    silence_phones: typing.Set[str],
    mapping: typing.Optional[typing.Dict[str, str]] = None,
) -> float:
    r"""
    Method to calculate overlap scoring

    .. math::

       Score = -(\lvert begin_{1} - begin_{2} \rvert + \lvert end_{1} - end_{2} \rvert + \begin{cases}
                0, & if label_{1} = label_{2} \\
                2, & otherwise
                \end{cases})

    See Also
    --------
    `Blog post <https://memcauliffe.com/update-on-montreal-forced-aligner-performance.html>`_
        For a detailed example that using this metric

    Parameters
    ----------
    first_element: :class:`~montreal_forced_aligner.data.CtmInterval`
        First CTM interval to compare
    second_element: :class:`~montreal_forced_aligner.data.CtmInterval`
        Second CTM interval
    silence_phones: set[str]
        Labels corresponding to optionally inserted silence
    mapping: Optional[dict[str, str]]
        Optional mapping of phones to treat as matches even if they have different symbols

    Returns
    -------
    float
        Score calculated as the negative sum of the absolute different in begin timestamps, absolute difference in end
        timestamps and the label score
    """
    begin_diff = abs(first_element.begin - second_element.begin)
    end_diff = abs(first_element.end - second_element.end)
    label_diff = compare_labels(first_element.label, second_element.label, silence_phones, mapping)
    return -1 * (begin_diff + end_diff + label_diff)


def fix_many_to_one_alignments(
    alignment: IntervalAlignment, custom_mapping: typing.Optional[typing.Dict[str, str]]
):
    test_keys = set(x for x in custom_mapping.keys() if " " in x)
    ref_keys = set()
    for val in custom_mapping.values():
        ref_keys.update(x for x in val if " " in x)
    new_ref = []
    new_test = []
    for i, (sa, sb) in enumerate(alignment.alignment):
        if i != 0:
            prev_sa, prev_sb = alignment.alignment[i - 1]
            ref_key = " ".join(x.label for x in [prev_sa, sa] if x.label != "-")
            test_key = " ".join(x.label for x in [prev_sb, sb] if x.label != "-")
            if (
                ref_key in ref_keys
                and test_key in custom_mapping
                and ref_key in custom_mapping[test_key]
            ):
                new_ref[-1].label = ref_key
                new_ref[-1].end = sa.end
                if sb.label != "-":
                    new_test.append(sb)
                continue
            if (
                test_key in test_keys
                and test_key in custom_mapping
                and ref_key in custom_mapping[test_key]
            ):
                new_test[-1].label = test_key
                new_test[-1].end = sb.end
                if sa.label != "-":
                    new_ref.append(sa)
                continue
        if sa.label != "-":
            new_ref.append(sa)
        if sb.label != "-":
            new_test.append(sb)
    return new_ref, new_test


def align_phones(
    ref: typing.List[CtmInterval],
    test: typing.List[CtmInterval],
    silence_phones: typing.Union[str, typing.Set[str]],
    ignored_phones: typing.Set[str] = None,
    custom_mapping: typing.Optional[typing.Dict[str, typing.Set[str]]] = None,
    debug: bool = False,
) -> typing.Tuple[
    float,
    float,
    typing.Dict[typing.Tuple[str, str], int],
    float,
    typing.List[typing.Dict[str, typing.Any]],
]:
    """
    Align phones based on how much they overlap and their phone label, with the ability to specify a custom mapping for
    different phone labels to be scored as if they're the same phone

    Parameters
    ----------
    ref: list[:class:`~montreal_forced_aligner.data.CtmInterval`]
        List of CTM intervals as reference
    test: list[:class:`~montreal_forced_aligner.data.CtmInterval`]
        List of CTM intervals to compare to reference
    silence_phones: str or set[str]
        Silence phones (these are ignored in the final calculation)
    ignored_phones: set[str], optional
        Phones that should be ignored in score calculations (silence phone is automatically added)
    custom_mapping: dict[str, set[str]], optional
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
    float
        Edit distance score of alignment
    list[dict[str, any]]
        Boundary errors
    """
    if isinstance(silence_phones, str):
        silence_phones = {silence_phones}
    if ignored_phones is None:
        ignored_phones = set()
    if not isinstance(ignored_phones, set):
        ignored_phones = set(ignored_phones)
    if custom_mapping is None:
        custom_mapping = {}
    else:
        for k, v in custom_mapping.items():
            if k in silence_phones:
                if isinstance(v, str):
                    silence_phones.add(v)
                else:
                    silence_phones.update(v)
            elif v & silence_phones:
                silence_phones.add(k)
    ignored_phones.update(silence_phones)
    try:
        alignment = align_intervals(ref, test, silence_phones, custom_mapping)
    except Exception:
        print(ref)
        print(test)
        raise
    if custom_mapping is not None:
        ref, test = fix_many_to_one_alignments(alignment, custom_mapping)
        alignment = align_intervals(ref, test, silence_phones, custom_mapping)
    overlap_count = 0
    overlap_sum = 0
    num_insertions = 0
    num_deletions = 0
    num_substitutions = 0
    errors = collections.Counter()
    boundary_errors = []
    for i, (sa, sb) in enumerate(alignment.alignment):
        if sa.label == "-":
            if sb.label not in ignored_phones:
                errors[(sa.label, sb.label)] += 1
                num_insertions += 1
            else:
                continue
        elif sb.label == "-":
            if sa.label not in ignored_phones:
                errors[(sa.label, sb.label)] += 1
                num_deletions += 1
            else:
                continue
        else:
            if i != 0:
                prev_sa, prev_sb = alignment.alignment[i - 1]
                if prev_sa.label != "-" and prev_sb.label != "-":
                    boundary_errors.append(
                        {
                            "following_reference_phone": sa.label,
                            "following_test_phone": sb.label,
                            "previous_reference_phone": prev_sa.label,
                            "previous_test_phone": prev_sb.label,
                            "boundary_error": round(sa.begin - sb.begin, 3),
                            "reference_boundary": round(sa.begin, 3),
                            "test_boundary": round(sb.begin, 3),
                        }
                    )
            if sa.label in ignored_phones:
                continue
            overlap_sum += (abs(sa.begin - sb.begin) + abs(sa.end - sb.end)) / 2
            overlap_count += 1
            if compare_labels(sa.label, sb.label, silence_phones, mapping=custom_mapping) > 0:
                num_substitutions += 1
                errors[(sa.label, sb.label)] += 1
    if overlap_count:
        score = round(overlap_sum / overlap_count, 3)
    else:
        score = None
    phone_error_rate = (num_insertions + num_deletions + (2 * num_substitutions)) / len(ref)
    if debug:
        import logging

        logger = logging.getLogger("mfa")
        if errors:
            logger.debug(
                f"PER: {phone_error_rate}\nErrors: {errors}\n{format_alignment(alignment)}"
            )
        else:
            logger.debug(f"PER: {phone_error_rate}\n{format_alignment(alignment)}")
    return score, phone_error_rate, errors, alignment.score, boundary_errors


def fix_unk_words(
    ref: typing.List[str],
    test: typing.List[CtmInterval],
    lexicon_compiler: LexiconCompiler,
) -> typing.List[WordCtmInterval]:
    """
    Takes in word-level alignments and looks up original label of unknown words

    Parameters
    ----------
    ref: list[str]
        Reference text
    test: list[:class:`~kalpy.gmm.data.WordCtmInterval`]
        Aligned word intervals with unknown tokens
    lexicon_compiler: LexiconCompiler
        Lexicon compiler to use for evaluating the identity of OOV items

    Returns
    -------
    list[:class:`~kalpy.gmm.data.WordCtmInterval`]
        Aligned words with unknown word tokens replaced with their original label
    """

    mapping = {}
    ref_intervals = []
    for w in ref:
        ref_intervals.append(WordCtmInterval(w, lexicon_compiler.to_int(w), []))
        if lexicon_compiler.to_int(w) == lexicon_compiler.to_int(lexicon_compiler.oov_word):
            if w not in mapping:
                mapping[w] = {lexicon_compiler.oov_word}

    alignment = align_intervals(ref_intervals, test, {lexicon_compiler.silence_word}, {})
    output_ctm = []
    for sa, sb in alignment.alignment:
        if sa.label == "-":
            output_ctm.append(sb)
        elif sb.label == "-":
            continue
        else:
            if sa.label != sb.label and sb.label == lexicon_compiler.oov_word:
                sb.label = sa.label
            output_ctm.append(sb)
    return output_ctm


def align_words(
    ref: typing.Union[typing.List[str], typing.List[CtmInterval]],
    test: typing.List[CtmInterval],
    silence_words: typing.Union[str, typing.Set[str]],
    ignored_words: typing.Set[str] = None,
    debug: bool = False,
) -> typing.Tuple[float, float, float]:
    """
    Align words based on how much their time points overlap and their label

    Parameters
    ----------
    ref: list[:class:`~montreal_forced_aligner.data.CtmInterval`]
        List of CTM intervals as reference
    test: list[:class:`~montreal_forced_aligner.data.CtmInterval`]
        List of CTM intervals to compare to reference
    silence_words: str or set[str]
        Silence word (these are ignored in the final calculation)
    ignored_words: set[str], optional
        Words that should be ignored in score calculations (silence phone is automatically added)
    debug: bool, optional
        Flag for logging extra information about alignments

    Returns
    -------
    float
        Extra duration of new words
    float
        Word error rate
    float
        Aligned duration of found words
    """

    if isinstance(silence_words, str):
        silence_words = {silence_words}
    if ignored_words is None:
        ignored_words = set()
    if not isinstance(ignored_words, set):
        ignored_words = set(ignored_words)
    ignored_words.update(silence_words)
    ref_intervals = []
    for w in ref:
        if not isinstance(w, CtmInterval):
            ref_intervals.append(CtmInterval(0.0, 0.0, w))
        else:
            ref_intervals.append(w)
    alignment = align_intervals(ref_intervals, test, silence_words, {})

    num_insertions = 0
    num_deletions = 0
    num_substitutions = 0

    extra_duration = 0
    aligned_duration = 0
    for sa, sb in alignment.alignment:
        if sa.label == "-":
            if sb.label not in ignored_words:
                num_insertions += 1
                extra_duration += sb.end - sb.begin
            else:
                continue
        elif sb.label == "-":
            if sa not in ignored_words:
                num_deletions += 1
            else:
                continue
        else:
            if sa.label in ignored_words:
                continue
            if sa.label != sb.label:
                num_substitutions += 1
            else:
                aligned_duration += sb.end - sb.begin
    word_error_rate = (num_insertions + num_deletions + (2 * num_substitutions)) / len(ref)
    if debug:
        import logging

        logger = logging.getLogger("mfa")
        logger.debug(
            f"{format_alignment(alignment)}\nExtra word duration: {extra_duration}\nWER: {word_error_rate}"
        )
    return extra_duration, word_error_rate, aligned_duration


def format_alignment(alignment: IntervalAlignment) -> str:
    """Format the alignment prettily into a string.

    Adapted largely from
    `Bio python's pairwise2 format_alignment <https://github.com/biopython/biopython/blob/master/Bio/pairwise2.py#L1348>`_

    Parameters
    ----------
    alignment: IntervalAlignment
        Alignment to format into human-readable form

    Returns
    -------
    str
        Formatted alignment string
    """
    begin = 0
    end = len(alignment)
    align1 = alignment.reference
    align2 = alignment.test
    full_sequences = False
    align_begin = begin
    align_end = end
    start1 = start2 = ""
    start_m = begin  # Begin of match line (how many spaces to include)
    # For local alignments:
    if not full_sequences and (begin != 0 or end != len(align1)):
        # Calculate the actual start positions in the un-aligned sequences
        # This will only work if the gap symbol is '-' or ['-']!
        start1 = str(len(align1[:begin]) - align1[:begin].count("-") + 1) + " "
        start2 = str(len(align2[:begin]) - align2[:begin].count("-") + 1) + " "
        start_m = max(len(start1), len(start2))
    elif full_sequences:
        start_m = 0
        begin = 0
        end = len(align1)

    s1_line = ["{:>{width}}".format(start1, width=start_m)]  # seq1 line
    m_line = [" " * start_m]  # match line
    s2_line = ["{:>{width}}".format(start2, width=start_m)]  # seq2 line

    for n, (r, t) in enumerate(alignment.alignment[begin:end]):
        # Since list elements can be of different length, we center them,
        # using the maximum length of the two compared elements as width
        a = r.label + " "
        b = t.label + " "
        m_len = max(len(a), len(b))
        s1_line.append("{:^{width}}".format(a, width=m_len))
        s2_line.append("{:^{width}}".format(b, width=m_len))
        if full_sequences and (n < align_begin or n >= align_end):
            m_line.append("{:^{width}}".format(" ", width=m_len))  # space
            continue
        if a == b:
            m_line.append("{:^{width}}".format("|", width=m_len))  # match
        elif a.strip() == "-" or b.strip() == "-":
            m_line.append("{:^{width}}".format(" ", width=m_len))  # gap
        else:
            m_line.append("{:^{width}}".format(".", width=m_len))  # mismatch

    s2_line.append(f"\n  Score={alignment.score:g}\n")
    return "\n".join(["".join(s1_line), "".join(m_line), "".join(s2_line)])
