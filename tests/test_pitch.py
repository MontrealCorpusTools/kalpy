import os

from kalpy.data import Segment
from kalpy.feat.data import FeatureArchive
from kalpy.feat.pitch import PitchComputer


def test_generate_pitch(wav_path):
    feature_generator = PitchComputer(snip_edges=False)
    pitch = feature_generator.compute_pitch(Segment(wav_path))
    assert pitch.shape[0] == 2672
    assert pitch.shape[1] == 3
    pitch = feature_generator.compute_pitch(Segment(wav_path, 1, 2, 0))
    assert pitch.shape[0] == 100
    assert pitch.shape[1] == 3


def test_export_pitch(wav_path, temp_dir):
    output_file_name = temp_dir.joinpath("mfccs.ark")
    feature_generator = PitchComputer(snip_edges=False)
    segments = {"1": Segment(wav_path, 1, 2)}
    feature_generator.export_feats(output_file_name, segments.items())
    assert output_file_name.exists()

    archive = FeatureArchive(output_file_name)
    for utt, pitch in archive:
        pitch = pitch.numpy()
        assert utt == "1"
        assert pitch.shape[0] == 100
        assert pitch.shape[1] == 3
    pitch = archive["1"].numpy()
    assert pitch.shape[0] == 100
    assert pitch.shape[1] == 3

    archive.close()
    os.remove(output_file_name)
    feature_generator.export_feats(
        output_file_name, segments.items(), write_scp=True, compress=True
    )
    assert output_file_name.with_suffix(".scp").exists()

    archive = FeatureArchive(output_file_name.with_suffix(".scp"))
    for utt, pitch in archive:
        pitch = pitch.numpy()
        assert utt == "1"
        assert pitch.shape[0] == 100
        assert pitch.shape[1] == 3
    pitch = archive["1"].numpy()
    assert pitch.shape[0] == 100
    assert pitch.shape[1] == 3
