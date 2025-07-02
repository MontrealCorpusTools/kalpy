import pickle

from _kalpy.util import CtmInterval, Interval, IntervalAlignment, WordCtmInterval, align_intervals
from kalpy.evaluation import format_alignment


def test_intervals():
    i = Interval(0.0, 1.0, "hello")
    assert i.begin == 0.0
    assert i.end == 1.0
    assert i.label == "hello"
    mapping = {}
    silence_phone = "sil"
    assert i.compare_labels(silence_phone, silence_phone, mapping) == 10
    assert i.compare_labels("hello", silence_phone, mapping) == 0.0
    assert i.compare_labels("other_hello", silence_phone, mapping) == 2.0

    mapping = {"other_hello": {"hello"}}
    assert i.compare_labels("other_hello", silence_phone, mapping) == 0.0

    i_data = pickle.dumps(i)
    i_new = pickle.loads(i_data)
    assert i_new.begin == i.begin
    assert i_new.end == i.end
    assert i_new.label == i.label


def test_ctm_intervals():
    i = CtmInterval(0.0, 1.0, "hello", 0)
    assert i.begin == 0.0
    assert i.end == 1.0
    assert i.label == "hello"
    assert i.symbol == 0
    mapping = {}
    silence_phone = "sil"
    assert i.compare_labels(silence_phone, silence_phone, mapping) == 10
    assert i.compare_labels("hello", silence_phone, mapping) == 0.0
    assert i.compare_labels("other_hello", silence_phone, mapping) == 2.0

    mapping = {"other_hello": {"hello"}}
    assert i.compare_labels("other_hello", silence_phone, mapping) == 0.0

    i_data = pickle.dumps(i)
    i_new = pickle.loads(i_data)
    assert i_new.begin == i.begin
    assert i_new.end == i.end
    assert i_new.label == i.label


def test_word_ctm_intervals():
    i = CtmInterval(0.0, 1.0, "hh", 0)
    w = WordCtmInterval("hello", 0, [i])
    assert w.begin == 0.0
    assert w.end == 1.0
    assert w.label == "hello"
    assert w.symbol == 0
    assert w.phones[0].begin == 0.0
    assert w.phones[0].end == 1.0
    assert w.phones[0].label == "hh"
    assert w.phones[0].symbol == 0
    mapping = {}
    silence_phone = "sil"
    assert w.compare_labels(silence_phone, silence_phone, mapping) == 10
    assert w.compare_labels("hello", silence_phone, mapping) == 0.0
    assert w.compare_labels("other_hello", silence_phone, mapping) == 2.0

    mapping = {"other_hello": {"hello"}}
    assert w.compare_labels("other_hello", silence_phone, mapping) == 0.0

    w_data = pickle.dumps(w)
    w_new = pickle.loads(w_data)
    assert w_new.begin == w.begin
    assert w_new.end == w.end
    assert w_new.label == w.label
    assert len(w_new.phones) == 1
    assert w_new.phones[0].begin == w.phones[0].begin
    assert w_new.phones[0].end == w.phones[0].end
    assert w_new.phones[0].label == w.phones[0].label
    assert w_new.phones[0].symbol == w.phones[0].symbol


def test_align_intervals(reference_hello_intervals, test_hi_intervals):
    mapping = {}
    silence_phone = "sil"
    alignment = align_intervals(
        reference_hello_intervals, test_hi_intervals, silence_phone, mapping
    )
    assert len(alignment) == 4
    for r, t in alignment:
        print(r, t)
    assert alignment[0][0].label == alignment[0][1].label
    assert alignment[1][0].label != alignment[1][1].label
    assert alignment[2][1].label == alignment[3][1].label == "-"
    assert alignment[2][1].begin == alignment[3][1].begin == -1.0


def test_format_alignment(reference_hello_intervals, test_hi_intervals):
    mapping = {}
    silence_phone = "sil"
    alignment = align_intervals(
        reference_hello_intervals, test_hi_intervals, silence_phone, mapping
    )
    assert len(alignment) == 4
    for r, t in alignment:
        print(r, t)
    assert alignment[0][0].label == alignment[0][1].label
    assert alignment[1][0].label != alignment[1][1].label
    assert alignment[2][1].label == alignment[3][1].label == "-"
    assert alignment[2][1].begin == alignment[3][1].begin == -1.0
    formatted = format_alignment(alignment)
    print(formatted)
    assert "hh eh l ow" in formatted
    assert "   |  .      " in formatted
    assert "hh ay - -" in formatted
    assert "Score=4" in formatted

    mapping = {"ay": {"l"}}
    silence_phone = "sil"
    alignment = align_intervals(
        reference_hello_intervals, test_hi_intervals, silence_phone, mapping
    )
    assert len(alignment) == 4
    for r, t in alignment:
        print(r, t)
    assert alignment[0][0].label == alignment[0][1].label
    assert alignment[1][0].label == "eh"
    assert alignment[1][1].label == "-"
    assert alignment[2][0].label == "l"
    assert alignment[2][1].label == "ay"
    assert alignment[1][1].label == alignment[3][1].label == "-"
    assert alignment[1][1].begin == alignment[3][1].begin == -1.0
    formatted = format_alignment(alignment)
    print(formatted)
    assert "hh eh l  ow" in formatted
    assert "|     . " in formatted
    assert "hh -  ay -" in formatted
    assert "Score=4" in formatted
