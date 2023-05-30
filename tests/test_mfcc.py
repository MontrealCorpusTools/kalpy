import os

from kalpy.feat.mfcc import FeatureArchive, MfccComputer, Segment


def test_generate_mfcc(wav_path):
    feature_generator = MfccComputer(snip_edges=False)
    mfccs = feature_generator.compute_mfccs(Segment(wav_path))
    assert mfccs.shape[0] == 2672
    assert mfccs.shape[1] == 13
    mfccs = feature_generator.compute_mfccs(Segment(wav_path, 1, 2, 0))
    assert mfccs.shape[0] == 100
    assert mfccs.shape[1] == 13


def test_export_mfcc(wav_path, temp_dir):
    output_file_name = temp_dir.joinpath("mfccs.ark")
    feature_generator = MfccComputer(snip_edges=False)
    segments = {"1": Segment(wav_path, 1, 2)}
    feature_generator.export_feats(output_file_name, segments)
    assert output_file_name.exists()

    archive = FeatureArchive(output_file_name)
    for utt, mfccs in archive:
        assert utt == "1"
        assert mfccs.shape[0] == 100
        assert mfccs.shape[1] == 13
    mfccs = archive["1"]
    assert mfccs.shape[0] == 100
    assert mfccs.shape[1] == 13

    os.remove(output_file_name)
    feature_generator.export_feats(output_file_name, segments, write_scp=True, compress=True)
    assert output_file_name.with_suffix(".scp").exists()

    archive = FeatureArchive(output_file_name.with_suffix(".scp"))
    for utt, mfccs in archive:
        assert utt == "1"
        assert mfccs.shape[0] == 100
        assert mfccs.shape[1] == 13
    mfccs = archive["1"]
    assert mfccs.shape[0] == 100
    assert mfccs.shape[1] == 13
