from kalpy.aligner import KalpyAligner
from kalpy.data import Segment
from kalpy.models import AcousticModel
from kalpy.utterance import Utterance


def test_aligner_align(
    english_mfa_am_dir,
    mfa_dictionary_path,
    mfa_wav_path,
):
    am = AcousticModel(english_mfa_am_dir)
    lc = am.lexicon_compiler
    lc.load_pronunciations(mfa_dictionary_path)
    aligner = KalpyAligner(am, lc)
    seg = Segment(mfa_wav_path)
    utterance = Utterance(seg, "montreal forced aligner")
    ctm = aligner.align_utterance(utterance)
    assert len(ctm.word_intervals) == 3


def test_aligner_fine_tune(
    english_mfa_am_dir,
    mfa_dictionary_path,
    mfa_wav_path,
):
    am = AcousticModel(english_mfa_am_dir)
    lc = am.lexicon_compiler
    lc.load_pronunciations(mfa_dictionary_path)
    aligner = KalpyAligner(am, lc)
    seg = Segment(mfa_wav_path)
    utterance = Utterance(seg, "montreal forced aligner")
    ctm = aligner.align_utterance(utterance)
    assert len(ctm.word_intervals) == 3
    fine_tuned_ctm = aligner.fine_tune_alignments(utterance)
    assert len(fine_tuned_ctm.word_intervals) == 3
    boundaries = ctm.phone_boundaries
    fine_tuned_boundaries = fine_tuned_ctm.phone_boundaries
    assert len(boundaries) == len(fine_tuned_boundaries)
    for i, b in enumerate(boundaries):
        assert abs(b - fine_tuned_boundaries[i]) < 0.01
