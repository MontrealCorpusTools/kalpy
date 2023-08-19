import pytest

from kalpy.data import KaldiMapping
from kalpy.decoder.decode_graph import DecodeGraphCompiler
from kalpy.feat.data import FeatureArchive
from kalpy.fstext.lexicon import LexiconCompiler
from kalpy.gmm.data import AlignmentArchive, LatticeArchive
from kalpy.gmm.decode import GmmDecoder
from kalpy.lm.rescore import LmRescorer


@pytest.mark.order(3)
def test_decode(mono_tree_path, mono_model_path, dictionary_path, mono_temp_dir, lm_path):
    lc = LexiconCompiler(position_dependent_phones=False, disambiguation=True)
    lc.load_pronunciations(dictionary_path)

    gc = DecodeGraphCompiler(mono_model_path, mono_tree_path, lc, arpa_path=lm_path)
    cmvn_file_name = mono_temp_dir.joinpath("cmvn.ark")
    utt2spk = KaldiMapping()
    utt2spk["1"] = "1"
    feature_archive = FeatureArchive(
        mono_temp_dir.joinpath("mfccs.ark"),
        utt2spk=utt2spk,
        cmvn_file_name=cmvn_file_name,
        deltas=True,
    )
    aligner = GmmDecoder(mono_model_path, gc.hclg_fst, beam=1000)
    for alignment in aligner.decode_utterances(feature_archive):
        assert alignment.utterance_id == "1"
        assert len(alignment.alignment) == 2672
        assert alignment.per_frame_likelihoods.numpy().shape[0] == 2672
        ctm = alignment.generate_ctm(aligner.transition_model, lc.phone_table)
        assert len(ctm) > 0


@pytest.mark.order(3)
def test_decode_sat_first_pass(
    sat_tree_path,
    sat_align_model_path,
    sat_lda_mat_path,
    sat_dictionary_path,
    sat_temp_dir,
    sat_phones,
    lm_path,
):
    lc = LexiconCompiler(position_dependent_phones=False, phones=sat_phones, disambiguation=True)
    lc.load_pronunciations(sat_dictionary_path)
    hclg_path = sat_temp_dir.joinpath("hclg.fst")

    gc = DecodeGraphCompiler(sat_align_model_path, sat_tree_path, lc)
    gc.load_from_file(hclg_path)
    cmvn_file_name = sat_temp_dir.joinpath("cmvn.ark")
    lattice_file_name = sat_temp_dir.joinpath("lat.ark")
    alignment_file_name = sat_temp_dir.joinpath("ali_decode.ark")
    word_file_name = sat_temp_dir.joinpath("words.ark")
    utt2spk = KaldiMapping()
    textgrid_name = sat_temp_dir.joinpath("first_pass_decode.TextGrid")
    utt2spk["1"] = "1"
    feature_archive = FeatureArchive(
        sat_temp_dir.joinpath("mfccs.ark"),
        utt2spk=utt2spk,
        cmvn_file_name=cmvn_file_name,
        lda_mat_file_name=sat_lda_mat_path,
        splices=True,
    )
    decoder = GmmDecoder(sat_align_model_path, gc.hclg_fst)
    decoder.boost_silence(20.0, lc.silence_symbols)
    decoder.export_lattices(
        lattice_file_name,
        feature_archive,
        word_file_name=word_file_name,
        alignment_file_name=alignment_file_name,
    )
    assert lattice_file_name.exists()
    assert alignment_file_name.exists()
    alignment_archive = AlignmentArchive(alignment_file_name, words_file_name=word_file_name)
    alignment = alignment_archive["1"]
    assert len(alignment.alignment) == 2670
    intervals = alignment.generate_ctm(decoder.transition_model, lc.phone_table)
    text = " ".join(lc.word_table.find(x) for x in alignment.words)
    ctm = lc.phones_to_pronunciations(text, alignment.words, intervals)
    ctm.export_textgrid(textgrid_name, file_duration=26.72)


@pytest.mark.order(5)
def test_decode_sat_second_pass(
    sat_tree_path,
    sat_model_path,
    sat_lda_mat_path,
    sat_dictionary_path,
    sat_temp_dir,
    sat_phones,
    lm_path,
):
    lc = LexiconCompiler(position_dependent_phones=False, phones=sat_phones, disambiguation=True)
    lc.load_pronunciations(sat_dictionary_path)

    hclg_path = sat_temp_dir.joinpath("hclg.fst")
    gc = DecodeGraphCompiler(sat_model_path, sat_tree_path, lc)
    gc.load_from_file(hclg_path)
    cmvn_file_name = sat_temp_dir.joinpath("cmvn.ark")
    trans_file_name = sat_temp_dir.joinpath("trans_decode.ark")
    word_file_name = sat_temp_dir.joinpath("words.ark")
    textgrid_name = sat_temp_dir.joinpath("second_pass_decode.TextGrid")
    lattice_file_name = sat_temp_dir.joinpath("lat_second_pass.ark")
    alignment_file_name = sat_temp_dir.joinpath("ali_decode_second_pass.ark")
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
    decoder = GmmDecoder(sat_model_path, gc.hclg_fst)
    decoder.boost_silence(20.0, lc.silence_symbols)
    decoder.export_lattices(
        lattice_file_name,
        feature_archive,
        word_file_name=word_file_name,
        alignment_file_name=alignment_file_name,
    )
    assert lattice_file_name.exists()
    assert alignment_file_name.exists()
    alignment_archive = AlignmentArchive(alignment_file_name, words_file_name=word_file_name)
    alignment = alignment_archive["1"]
    assert len(alignment.alignment) == 2670
    intervals = alignment.generate_ctm(decoder.transition_model, lc.phone_table)
    text = " ".join(lc.word_table.find(x) for x in alignment.words)
    ctm = lc.phones_to_pronunciations(text, alignment.words, intervals)
    ctm.export_textgrid(textgrid_name, file_duration=26.72)


@pytest.mark.order(6)
def test_decode_sat_lm_rescore(
    sat_tree_path,
    sat_model_path,
    sat_lda_mat_path,
    sat_dictionary_path,
    sat_temp_dir,
    sat_phones,
    lm_path,
):
    lc = LexiconCompiler(position_dependent_phones=False, phones=sat_phones, disambiguation=True)
    lc.load_pronunciations(sat_dictionary_path)

    hclg_path = sat_temp_dir.joinpath("hclg.fst")
    g_path = sat_temp_dir.joinpath("g.fst")
    gc = DecodeGraphCompiler(sat_model_path, sat_tree_path, lc)
    gc.load_from_file(hclg_path)
    gc.load_g_from_file(g_path)
    gc.compile_g_carpa(lm_path, sat_temp_dir.joinpath("g.carpa"))
    lattice_file_name = sat_temp_dir.joinpath("lat_second_pass.ark")
    lattice_output_file_name = sat_temp_dir.joinpath("lat_second_pass_rescore.ark")
    utt2spk = KaldiMapping()
    utt2spk["1"] = "1"
    decoder = LmRescorer(gc.g_fst)
    lattice_archive = LatticeArchive(lattice_file_name)
    decoder.export_lattices(lattice_output_file_name, lattice_archive, gc.g_carpa)
    assert lattice_output_file_name.exists()
