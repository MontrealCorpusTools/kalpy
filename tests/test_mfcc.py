import os

import numpy as np
import pytest

from _kalpy import transform
from _kalpy.feat import RandomAccessWaveReader
from _kalpy.matrix import CompressedMatrix
from _kalpy.util import (
    BaseFloatMatrixWriter,
    CompressedMatrixWriter,
    RandomAccessBaseDoubleMatrixReader,
    RandomAccessBaseFloatMatrixReader,
)
from kalpy.data import KaldiMapping, MatrixArchive, Segment
from kalpy.decoder.training_graphs import TrainingGraphCompiler
from kalpy.feat.cmvn import CmvnComputer
from kalpy.feat.data import FeatureArchive
from kalpy.feat.fmllr import FmllrComputer
from kalpy.feat.mfcc import MfccComputer
from kalpy.fstext.lexicon import LexiconCompiler
from kalpy.gmm.align import GmmAligner
from kalpy.gmm.data import AlignmentArchive, LatticeArchive
from kalpy.utils import generate_read_specifier, generate_write_specifier


@pytest.mark.order(1)
def test_wave(wav_path, reference_dir):
    ref_wav_scp = reference_dir.joinpath("wav.scp")
    wav_rspecifier = generate_read_specifier(ref_wav_scp)
    wave_reader = RandomAccessWaveReader(wav_rspecifier)
    kaldi_wave = wave_reader.Value("1-1").Data().numpy()[0, :]
    segment = Segment(wav_path)
    kalpy_wave = segment.kaldi_wave

    np.testing.assert_allclose(kaldi_wave, kalpy_wave)


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
    segments = {"1-1": Segment(wav_path)}
    feature_generator.export_feats(output_file_name, segments.items())
    assert output_file_name.exists()

    archive = FeatureArchive(output_file_name)
    for utt, mfccs in archive:
        mfccs = mfccs.numpy()
        assert utt == "1-1"
        assert mfccs.shape[0] == 2672
        assert mfccs.shape[1] == 13
    mfccs = archive["1-1"].numpy()
    print(mfccs)
    print(mfccs.shape)
    assert mfccs.shape[0] == 2672
    assert mfccs.shape[1] == 13
    archive.close()
    os.remove(output_file_name)
    feature_generator.export_feats(
        output_file_name, segments.items(), write_scp=True, compress=False
    )
    assert output_file_name.with_suffix(".scp").exists()

    archive = FeatureArchive(output_file_name.with_suffix(".scp"))
    for utt, mfccs in archive:
        mfccs = mfccs.numpy()
        assert utt == "1-1"
        assert mfccs.shape[0] == 2672
        assert mfccs.shape[1] == 13
    mfccs = archive["1-1"].numpy()
    assert mfccs.shape[0] == 2672
    assert mfccs.shape[1] == 13


@pytest.mark.order(2)
def test_cmvn(mono_temp_dir):
    feature_file_name = mono_temp_dir.joinpath("mfccs.ark")
    output_file_name = mono_temp_dir.joinpath("cmvn.ark")
    feature_generator = CmvnComputer(online=False)
    spk2utt = KaldiMapping()
    spk2utt["1"] = ["1-1"]
    feature_archive = FeatureArchive(feature_file_name)
    feature_generator.export_cmvn(output_file_name, feature_archive, spk2utt)
    assert output_file_name.exists()


@pytest.mark.order(1)
def test_export_mfcc_sat(wav_path, sat_temp_dir, reference_dir, reference_mfcc_path):
    output_file_name = sat_temp_dir.joinpath("mfccs.ark")
    feature_generator = MfccComputer(snip_edges=False, dither=0, use_energy=False)
    segments = {"1-1": Segment(wav_path, begin=0.0, end=26.72325)}
    feature_generator.export_feats(output_file_name, segments.items())
    assert output_file_name.exists()

    archive = FeatureArchive(output_file_name)
    for utt, mfccs in archive:
        mfccs = mfccs.numpy()
        assert utt == "1-1"
        assert mfccs.shape[0] == 2672
        assert mfccs.shape[1] == 13
    mfccs = archive["1-1"].numpy()
    print(mfccs)
    print(mfccs.shape)
    assert mfccs.shape[0] == 2672
    assert mfccs.shape[1] == 13

    archive.close()
    os.remove(output_file_name)
    feature_generator.export_feats(
        output_file_name, segments.items(), write_scp=True, compress=False
    )
    assert output_file_name.with_suffix(".scp").exists()

    archive = FeatureArchive(output_file_name.with_suffix(".scp"))
    ref_archive = FeatureArchive(reference_mfcc_path)
    for utt, mfccs in archive:
        mfccs = mfccs.numpy()
        assert utt == "1-1"
        assert mfccs.shape[0] == 2672
        assert mfccs.shape[1] == 13
        ref_mfccs = ref_archive["1-1"]
        np.testing.assert_allclose(mfccs, ref_mfccs.numpy())
    mfccs = archive["1-1"].numpy()
    assert mfccs.shape[0] == 2672
    assert mfccs.shape[1] == 13


@pytest.mark.order(2)
def test_cmvn_sat(
    sat_temp_dir,
    reference_dir,
    reference_mfcc_path,
    reference_cmvn_path,
    reference_final_features_path,
):
    feature_file_name = sat_temp_dir.joinpath("mfccs.ark")
    output_file_name = sat_temp_dir.joinpath("cmvn.ark")
    feature_generator = CmvnComputer(online=False)
    spk2utt = KaldiMapping()
    spk2utt["1"] = ["1-1"]
    utt2spk = KaldiMapping()
    utt2spk["1-1"] = "1"
    feature_archive = FeatureArchive(feature_file_name)
    feature_generator.export_cmvn(output_file_name, feature_archive, spk2utt, write_scp=True)
    assert output_file_name.exists()
    cmvn_read_specifier = generate_read_specifier(output_file_name)
    cmvn_reader = RandomAccessBaseDoubleMatrixReader(cmvn_read_specifier)
    ref_cmvn_read_specifier = generate_read_specifier(reference_cmvn_path)
    ref_cmvn_reader = RandomAccessBaseDoubleMatrixReader(ref_cmvn_read_specifier)
    cmvn = cmvn_reader.Value("1")
    ref_cmvn = ref_cmvn_reader.Value("1")
    np.testing.assert_allclose(cmvn.numpy(), ref_cmvn.numpy())
    final_archive = FeatureArchive(
        feature_file_name,
        utt2spk=utt2spk,
        cmvn_file_name=output_file_name,
    )
    feat_archive = FeatureArchive(
        feature_file_name,
    )
    ref_final_features_archive = FeatureArchive(reference_final_features_path)
    feats = feat_archive["1-1"]
    cmvned_feats = transform.apply_cmvn(feats, cmvn)
    ref_cmvned_feats = ref_final_features_archive["1-1"]
    np.testing.assert_allclose(cmvned_feats.numpy(), ref_cmvned_feats.numpy())
    temp_ark_path = sat_temp_dir.joinpath("final_features.ark")
    temp_scp_path = sat_temp_dir.joinpath("final_features.scp")
    write_specifier = generate_write_specifier(temp_ark_path, write_scp=True)
    feature_writer = BaseFloatMatrixWriter(write_specifier)
    np.testing.assert_allclose(
        final_archive["1-1"].numpy(), ref_final_features_archive["1-1"].numpy()
    )
    for utt_id, mfccs in final_archive:
        feature_writer.Write(utt_id, mfccs)
    feature_writer.Close()
    feature_archive = FeatureArchive(temp_scp_path)
    np.testing.assert_allclose(
        feature_archive["1-1"].numpy(), ref_final_features_archive["1-1"].numpy()
    )
    cmvn_reader.Close()
    ref_cmvn_reader.Close()


@pytest.mark.order(4)
def test_fmllr_sat(
    sat_tree_path,
    sat_model_path,
    sat_align_model_path,
    sat_lda_mat_path,
    sat_dictionary_path,
    sat_temp_dir,
    sat_phones,
    reference_trans_path,
):
    lc = LexiconCompiler(position_dependent_phones=False, phones=sat_phones)
    lc.load_pronunciations(sat_dictionary_path)
    utt2spk = KaldiMapping()
    utt2spk["1-1"] = "1"
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
    spk2utt["1"] = ["1-1"]
    fmllr_computer = FmllrComputer(
        sat_align_model_path, sat_model_path, silence_phones=lc.silence_symbols, spk2utt=spk2utt
    )
    fmllr_computer.export_transforms(output_file_name, feature_archive, alignment_archive)
    assert output_file_name.exists()
    feature_archive.close()
    alignment_archive.close()
    trans_read_specifier = generate_read_specifier(output_file_name)
    trans_reader = RandomAccessBaseFloatMatrixReader(trans_read_specifier)
    ref_trans_read_specifier = generate_read_specifier(reference_trans_path)
    ref_trans_reader = RandomAccessBaseFloatMatrixReader(ref_trans_read_specifier)
    trans = trans_reader.Value("1")
    ref_trans = ref_trans_reader.Value("1")
    np.testing.assert_allclose(trans.numpy(), ref_trans.numpy())


@pytest.mark.order(4)
def test_fmllr_sat_no_two_model(
    sat_tree_path,
    sat_align_model_path,
    sat_lda_mat_path,
    sat_dictionary_path,
    sat_temp_dir,
    sat_phones,
    reference_trans_path,
):
    lc = LexiconCompiler(position_dependent_phones=False, phones=sat_phones)
    lc.load_pronunciations(sat_dictionary_path)
    utt2spk = KaldiMapping()
    utt2spk["1-1"] = "1"
    cmvn_file_name = sat_temp_dir.joinpath("cmvn.ark")
    feature_archive = FeatureArchive(
        sat_temp_dir.joinpath("mfccs.ark"),
        utt2spk=utt2spk,
        cmvn_file_name=cmvn_file_name,
        lda_mat_file_name=sat_lda_mat_path,
        splices=True,
    )
    alignment_archive = AlignmentArchive(sat_temp_dir.joinpath("ali.ark"))
    output_file_name = sat_temp_dir.joinpath("trans_no_two_model.ark")
    spk2utt = KaldiMapping()
    spk2utt["1"] = ["1-1"]
    fmllr_computer = FmllrComputer(
        sat_align_model_path,
        sat_align_model_path,
        silence_phones=lc.silence_symbols,
        spk2utt=spk2utt,
    )
    fmllr_computer.export_transforms(output_file_name, feature_archive, alignment_archive)
    assert output_file_name.exists()
    feature_archive.close()
    alignment_archive.close()


@pytest.mark.order(4)
def test_fmllr_decode_sat(
    sat_tree_path,
    sat_model_path,
    sat_align_model_path,
    sat_lda_mat_path,
    sat_dictionary_path,
    sat_temp_dir,
    sat_phones,
):
    lc = LexiconCompiler(position_dependent_phones=False, phones=sat_phones)
    lc.load_pronunciations(sat_dictionary_path)
    utt2spk = KaldiMapping()
    utt2spk["1-1"] = "1"
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
    spk2utt["1"] = ["1-1"]
    fmllr_computer = FmllrComputer(
        sat_align_model_path, sat_model_path, silence_phones=lc.silence_symbols, spk2utt=spk2utt
    )
    fmllr_computer.export_transforms(output_file_name, feature_archive, alignment_archive)
    assert output_file_name.exists()


@pytest.mark.order(6)
def test_fmllr_compose(
    sat_tree_path,
    sat_model_path,
    sat_lda_mat_path,
    sat_dictionary_path,
    sat_temp_dir,
    sat_phones,
    reference_trans_path,
    reference_trans_compose_path,
):
    lc = LexiconCompiler(position_dependent_phones=False, phones=sat_phones)
    lc.load_pronunciations(sat_dictionary_path)
    utt2spk = KaldiMapping()
    utt2spk["1-1"] = "1"
    cmvn_file_name = sat_temp_dir.joinpath("cmvn.ark")
    feature_archive = FeatureArchive(
        sat_temp_dir.joinpath("mfccs.ark"),
        utt2spk=utt2spk,
        cmvn_file_name=cmvn_file_name,
        lda_mat_file_name=sat_lda_mat_path,
        transform_file_name=reference_trans_path,
        splices=True,
    )
    previous_transform_archive = MatrixArchive(reference_trans_path)
    alignment_archive = AlignmentArchive(sat_temp_dir.joinpath("ali_second_pass.ark"))
    output_file_name = sat_temp_dir.joinpath("trans_second_pass.ark")
    spk2utt = KaldiMapping()
    spk2utt["1"] = ["1-1"]
    fmllr_computer = FmllrComputer(
        sat_model_path, sat_model_path, silence_phones=lc.silence_symbols, spk2utt=spk2utt
    )
    fmllr_computer.export_transforms(
        output_file_name,
        feature_archive,
        alignment_archive,
        previous_transform_archive=previous_transform_archive,
        write_scp=True,
    )
    assert output_file_name.exists()
    feature_archive.close()
    alignment_archive.close()
    trans_read_specifier = generate_read_specifier(output_file_name)
    trans_reader = RandomAccessBaseFloatMatrixReader(trans_read_specifier)
    ref_trans_composed_read_specifier = generate_read_specifier(reference_trans_compose_path)
    ref_trans_composed_reader = RandomAccessBaseFloatMatrixReader(
        ref_trans_composed_read_specifier
    )
    ref_trans_read_specifier = generate_read_specifier(reference_trans_path)
    ref_trans_reader = RandomAccessBaseFloatMatrixReader(ref_trans_read_specifier)
    trans = trans_reader.Value("1")
    ref_trans = ref_trans_reader.Value("1")
    ref_trans_composed = ref_trans_composed_reader.Value("1")
    np.testing.assert_allclose(trans.numpy(), ref_trans_composed.numpy())
    np.testing.assert_raises(
        AssertionError, np.testing.assert_allclose, trans.numpy(), ref_trans.numpy()
    )
