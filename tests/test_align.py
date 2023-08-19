import pytest

from kalpy.data import KaldiMapping
from kalpy.decoder.data import FstArchive
from kalpy.feat.data import FeatureArchive
from kalpy.fstext.lexicon import LexiconCompiler
from kalpy.gmm.align import GmmAligner
from kalpy.gmm.data import AlignmentArchive


@pytest.mark.order(3)
def test_align(mono_tree_path, mono_model_path, dictionary_path, mono_temp_dir):
    lc = LexiconCompiler(position_dependent_phones=False)
    lc.load_pronunciations(dictionary_path)

    cmvn_file_name = mono_temp_dir.joinpath("cmvn.ark")
    training_graph_archive = FstArchive(mono_temp_dir.joinpath("fsts.ark"))
    utt2spk = KaldiMapping()
    utt2spk["1"] = "1"
    feature_archive = FeatureArchive(
        mono_temp_dir.joinpath("mfccs.ark"),
        utt2spk=utt2spk,
        cmvn_file_name=cmvn_file_name,
        deltas=True,
    )
    aligner = GmmAligner(mono_model_path, beam=1000, retry_beam=4000)
    for alignment in aligner.align_utterances(training_graph_archive, feature_archive):
        assert alignment.utterance_id == "1"
        assert len(alignment.alignment) == 2672
        assert alignment.per_frame_likelihoods.numpy().shape[0] == 2672
        ctm = alignment.generate_ctm(aligner.transition_model, lc.phone_table)
        assert len(ctm) == 242


@pytest.mark.order(3)
def test_align_sat_first_pass(
    sat_tree_path,
    sat_align_model_path,
    sat_lda_mat_path,
    sat_dictionary_path,
    sat_temp_dir,
    sat_phones,
):
    lc = LexiconCompiler(position_dependent_phones=False, phones=sat_phones)
    lc.load_pronunciations(sat_dictionary_path)

    cmvn_file_name = sat_temp_dir.joinpath("cmvn.ark")
    alignments_file_name = sat_temp_dir.joinpath("ali.ark")
    word_file_name = sat_temp_dir.joinpath("words.ark")
    training_graph_archive = FstArchive(sat_temp_dir.joinpath("fsts.ark"))
    utt2spk = KaldiMapping()
    textgrid_name = sat_temp_dir.joinpath("first_pass.TextGrid")
    utt2spk["1"] = "1"
    feature_archive = FeatureArchive(
        sat_temp_dir.joinpath("mfccs.ark"),
        utt2spk=utt2spk,
        cmvn_file_name=cmvn_file_name,
        lda_mat_file_name=sat_lda_mat_path,
        splices=True,
    )
    aligner = GmmAligner(sat_align_model_path, beam=10, retry_beam=40)
    aligner.boost_silence(20.0, lc.silence_symbols)
    aligner.export_alignments(
        alignments_file_name,
        training_graph_archive,
        feature_archive,
        word_file_name=word_file_name,
    )
    assert alignments_file_name.exists()
    alignment_archive = AlignmentArchive(alignments_file_name, words_file_name=word_file_name)
    alignment = alignment_archive["1"]
    assert len(alignment.alignment) == 2670
    intervals = alignment.generate_ctm(aligner.transition_model, lc.phone_table)
    text = " ".join(lc.word_table.find(x) for x in alignment.words)
    ctm = lc.phones_to_pronunciations(text, alignment.words, intervals)
    ctm.export_textgrid(textgrid_name, file_duration=26.72)


@pytest.mark.order(5)
def test_align_sat_second_pass(
    sat_tree_path, sat_model_path, sat_lda_mat_path, sat_dictionary_path, sat_temp_dir, sat_phones
):
    lc = LexiconCompiler(position_dependent_phones=False, phones=sat_phones)
    lc.load_pronunciations(sat_dictionary_path)
    cmvn_file_name = sat_temp_dir.joinpath("cmvn.ark")
    trans_file_name = sat_temp_dir.joinpath("trans.ark")
    word_file_name = sat_temp_dir.joinpath("words.ark")
    textgrid_name = sat_temp_dir.joinpath("second_pass.TextGrid")
    alignments_file_name = sat_temp_dir.joinpath("ali_second_pass.ark")
    training_graph_archive = FstArchive(sat_temp_dir.joinpath("fsts.ark"))
    utt2spk = KaldiMapping()
    utt2spk["1"] = "1"
    feature_archive = FeatureArchive(
        sat_temp_dir.joinpath("mfccs.ark"),
        utt2spk=utt2spk,
        cmvn_file_name=cmvn_file_name,
        lda_mat_file_name=sat_lda_mat_path,
        transform_file_name=trans_file_name,
        splices=True,
    )
    aligner = GmmAligner(sat_model_path, beam=10, retry_beam=40)
    aligner.boost_silence(20.0, lc.silence_symbols)
    aligner.export_alignments(
        alignments_file_name,
        training_graph_archive,
        feature_archive,
        word_file_name=word_file_name,
    )
    assert alignments_file_name.exists()
    alignment_archive = AlignmentArchive(alignments_file_name, words_file_name=word_file_name)
    alignment = alignment_archive["1"]
    assert len(alignment.alignment) == 2670
    intervals = alignment.generate_ctm(aligner.transition_model, lc.phone_table)
    text = " ".join(lc.word_table.find(x) for x in alignment.words)
    ctm = lc.phones_to_pronunciations(text, alignment.words, intervals)
    ctm.export_textgrid(textgrid_name, file_duration=26.72)
