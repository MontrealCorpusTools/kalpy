from kalpy.feat.mfcc import MfccComputer


def test_generate_mfcc(wav_path):
    feature_generator = MfccComputer(snip_edges=False)
    mfccs = feature_generator.compute_mfccs(wav_path)
    assert mfccs.shape[0] == 2672
    assert mfccs.shape[1] == 13
    mfccs = feature_generator.compute_mfccs(wav_path, 1, 2)
    assert mfccs.shape[0] == 100
    assert mfccs.shape[1] == 13
