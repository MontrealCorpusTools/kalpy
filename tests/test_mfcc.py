import os

import pytest

from kalpy.data import KaldiMapping, Segment
from kalpy.decoder.training_graphs import TrainingGraphCompiler
from kalpy.feat.cmvn import CmvnComputer
from kalpy.feat.data import FeatureArchive
from kalpy.feat.fmllr import FmllrComputer
from kalpy.feat.mfcc import MfccComputer
from kalpy.fstext.lexicon import LexiconCompiler
from kalpy.gmm.align import GmmAligner
from kalpy.gmm.data import AlignmentArchive, LatticeArchive


@pytest.mark.order(1)
def test_generate_mfcc(wav_path):
    feature_generator = MfccComputer(snip_edges=False)
    mfccs = feature_generator.compute_mfccs(Segment(wav_path))
    assert mfccs.shape[0] == 2672
    assert mfccs.shape[1] == 13
    mfccs = feature_generator.compute_mfccs(Segment(wav_path, 1, 2, 0))
    assert mfccs.shape[0] == 100
    assert mfccs.shape[1] == 13


@pytest.mark.order(1)
def test_export_mfcc(wav_path, mono_temp_dir):
    output_file_name = mono_temp_dir.joinpath("mfccs.ark")
    feature_generator = MfccComputer(snip_edges=False)
    segments = {"1": Segment(wav_path)}
    feature_generator.export_feats(output_file_name, segments.items())
    assert output_file_name.exists()

    archive = FeatureArchive(output_file_name)
    for utt, mfccs in archive:
        mfccs = mfccs.numpy()
        assert utt == "1"
        assert mfccs.shape[0] == 2672
        assert mfccs.shape[1] == 13
    mfccs = archive["1"].numpy()
    print(mfccs)
    print(mfccs.shape)
    assert mfccs.shape[0] == 2672
    assert mfccs.shape[1] == 13
    archive.close()
    os.remove(output_file_name)
    feature_generator.export_feats(
        output_file_name, segments.items(), write_scp=True, compress=True
    )
    assert output_file_name.with_suffix(".scp").exists()

    archive = FeatureArchive(output_file_name.with_suffix(".scp"))
    for utt, mfccs in archive:
        mfccs = mfccs.numpy()
        assert utt == "1"
        assert mfccs.shape[0] == 2672
        assert mfccs.shape[1] == 13
    mfccs = archive["1"].numpy()
    assert mfccs.shape[0] == 2672
    assert mfccs.shape[1] == 13


@pytest.mark.order(2)
def test_cmvn(mono_temp_dir):
    feature_file_name = mono_temp_dir.joinpath("mfccs.ark")
    output_file_name = mono_temp_dir.joinpath("cmvn.ark")
    feature_generator = CmvnComputer(online=False)
    spk2utt = KaldiMapping()
    spk2utt["1"] = ["1"]
    feature_archive = FeatureArchive(feature_file_name)
    feature_generator.export_cmvn(output_file_name, feature_archive, spk2utt)
    assert output_file_name.exists()


@pytest.mark.order(1)
def test_export_mfcc_sat(wav_path, sat_temp_dir):
    output_file_name = sat_temp_dir.joinpath("mfccs.ark")
    feature_generator = MfccComputer(snip_edges=True)
    segments = {"1": Segment(wav_path)}
    feature_generator.export_feats(output_file_name, segments.items())
    assert output_file_name.exists()

    archive = FeatureArchive(output_file_name)
    for utt, mfccs in archive:
        mfccs = mfccs.numpy()
        assert utt == "1"
        assert mfccs.shape[0] == 2670
        assert mfccs.shape[1] == 13
    mfccs = archive["1"].numpy()
    print(mfccs)
    print(mfccs.shape)
    assert mfccs.shape[0] == 2670
    assert mfccs.shape[1] == 13

    archive.close()
    os.remove(output_file_name)
    feature_generator.export_feats(
        output_file_name, segments.items(), write_scp=True, compress=True
    )
    assert output_file_name.with_suffix(".scp").exists()

    archive = FeatureArchive(output_file_name.with_suffix(".scp"))
    for utt, mfccs in archive:
        mfccs = mfccs.numpy()
        assert utt == "1"
        assert mfccs.shape[0] == 2670
        assert mfccs.shape[1] == 13
    mfccs = archive["1"].numpy()
    assert mfccs.shape[0] == 2670
    assert mfccs.shape[1] == 13


@pytest.mark.order(2)
def test_cmvn_sat(sat_temp_dir):
    feature_file_name = sat_temp_dir.joinpath("mfccs.ark")
    output_file_name = sat_temp_dir.joinpath("cmvn.ark")
    feature_generator = CmvnComputer(online=False)
    spk2utt = KaldiMapping()
    spk2utt["1"] = ["1"]
    feature_archive = FeatureArchive(feature_file_name)
    feature_generator.export_cmvn(output_file_name, feature_archive, spk2utt)
    assert output_file_name.exists()


@pytest.mark.order(4)
def test_fmllr_sat(
    sat_tree_path, sat_model_path, sat_lda_mat_path, sat_dictionary_path, sat_temp_dir, sat_phones
):
    lc = LexiconCompiler(position_dependent_phones=False, phones=sat_phones)
    lc.load_pronunciations(sat_dictionary_path)
    utt2spk = KaldiMapping()
    utt2spk["1"] = "1"
    cmvn_file_name = sat_temp_dir.joinpath("cmvn.ark")
    feature_archive = FeatureArchive(
        sat_temp_dir.joinpath("mfccs.ark"),
        utt2spk=utt2spk,
        cmvn_file_name=cmvn_file_name,
        lda_mat_file_name=sat_lda_mat_path,
        splices=True,
    )
    alignment_archive = AlignmentArchive(sat_temp_dir.joinpath("ali.ark"))
    output_file_name = sat_temp_dir.joinpath("trans.ark")
    spk2utt = KaldiMapping()
    spk2utt["1"] = ["1"]
    fmllr_computer = FmllrComputer(
        sat_model_path, silence_phones=lc.silence_symbols, spk2utt=spk2utt
    )
    fmllr_computer.export_transforms(output_file_name, feature_archive, alignment_archive)
    assert output_file_name.exists()


@pytest.mark.order(4)
def test_fmllr_decode_sat(
    sat_tree_path, sat_model_path, sat_lda_mat_path, sat_dictionary_path, sat_temp_dir, sat_phones
):
    lc = LexiconCompiler(position_dependent_phones=False, phones=sat_phones)
    lc.load_pronunciations(sat_dictionary_path)
    utt2spk = KaldiMapping()
    utt2spk["1"] = "1"
    cmvn_file_name = sat_temp_dir.joinpath("cmvn.ark")
    feature_archive = FeatureArchive(
        sat_temp_dir.joinpath("mfccs.ark"),
        utt2spk=utt2spk,
        cmvn_file_name=cmvn_file_name,
        lda_mat_file_name=sat_lda_mat_path,
        splices=True,
    )
    alignment_archive = LatticeArchive(sat_temp_dir.joinpath("lat.ark"), determinized=False)
    output_file_name = sat_temp_dir.joinpath("trans_decode.ark")
    spk2utt = KaldiMapping()
    spk2utt["1"] = ["1"]
    fmllr_computer = FmllrComputer(
        sat_model_path, silence_phones=lc.silence_symbols, spk2utt=spk2utt
    )
    fmllr_computer.export_transforms(output_file_name, feature_archive, alignment_archive)
    assert output_file_name.exists()
