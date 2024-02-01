import os
import pathlib
import subprocess
from io import BytesIO

import librosa
import pytest
import soundfile


@pytest.fixture(scope="session")
def test_dir():
    base = pathlib.Path(__file__).parent
    return base.joinpath("data")


@pytest.fixture(scope="session")
def wav_dir(test_dir):
    return test_dir.joinpath("wav")


@pytest.fixture(scope="session")
def am_dir(test_dir):
    return test_dir.joinpath("acoustic_models")


@pytest.fixture(scope="session")
def mono_am_dir(am_dir):
    return am_dir.joinpath("mono")


@pytest.fixture(scope="session")
def sat_am_dir(am_dir):
    return am_dir.joinpath("sat")


@pytest.fixture(scope="session")
def temp_dir(test_dir):
    p = test_dir.joinpath("temp")
    p.mkdir(exist_ok=True)
    return p


@pytest.fixture(scope="session")
def reference_dir(test_dir):
    p = test_dir.joinpath("kaldi")
    return p


@pytest.fixture(scope="session")
def reference_mfcc_path(wav_path, reference_dir):
    ark_path = reference_dir.joinpath("mfccs.ark")
    scp_path = reference_dir.joinpath("mfccs.scp")

    mfcc_proc = subprocess.Popen(
        [
            "compute-mfcc-feats",
            "--use-energy=false",
            "--dither=0",
            "--energy-floor=0",
            "--num-ceps=13",
            "--num-mel-bins=23",
            "--cepstral-lifter=22",
            "--preemphasis-coefficient=0.97",
            "--frame-shift=10",
            "--frame-length=25",
            "--low-freq=20",
            "--high-freq=7800",
            "--sample-frequency=16000",
            "--allow-downsample=true",
            "--allow-upsample=true",
            "--snip-edges=false",
            "ark,s,cs:-",
            f"ark,scp:{ark_path},{scp_path}",
        ],
        stdin=subprocess.PIPE,
        env=os.environ,
    )

    wave, _ = librosa.load(
        wav_path,
        sr=16000,
        offset=0.0,
        duration=26.72325,
        mono=False,
    )
    bio = BytesIO()
    soundfile.write(bio, wave, samplerate=16000, format="WAV")
    mfcc_proc.stdin.write(f"1-1\t".encode("utf8"))
    mfcc_proc.stdin.write(bio.getvalue())
    mfcc_proc.stdin.flush()
    mfcc_proc.stdin.close()
    mfcc_proc.communicate()
    return scp_path


@pytest.fixture(scope="session")
def reference_cmvn_path(wav_path, reference_dir, reference_mfcc_path):
    ark_path = reference_dir.joinpath("cmvn.ark")
    scp_path = reference_dir.joinpath("cmvn.scp")
    spk2utt = reference_dir.joinpath("spk2utt.scp")
    with open(spk2utt, "w", encoding="utf8") as f:
        f.write("1 1-1\n")
    subprocess.call(
        [
            "compute-cmvn-stats",
            f"--spk2utt=ark:{spk2utt}",
            f"scp:{reference_mfcc_path}",
            f"ark,scp:{ark_path},{scp_path}",
        ],
        env=os.environ,
    )
    return scp_path


@pytest.fixture(scope="session")
def reference_final_features_path(
    wav_path, reference_dir, reference_mfcc_path, reference_cmvn_path
):
    ark_path = reference_dir.joinpath("final_features.ark")
    scp_path = reference_dir.joinpath("final_features.scp")
    utt2spk = reference_dir.joinpath("utt2spk.scp")
    with open(utt2spk, "w", encoding="utf8") as f:
        f.write("1-1 1\n")
    subprocess.call(
        [
            "apply-cmvn",
            f"--utt2spk=ark:{utt2spk}",
            f"scp:{reference_cmvn_path}",
            f"scp:{reference_mfcc_path}",
            f"ark,scp:{ark_path},{scp_path}",
        ],
        env=os.environ,
    )
    return scp_path


@pytest.fixture(scope="session")
def reference_si_feature_string(reference_final_features_path, sat_lda_mat_path):
    return (
        f'ark,s,cs:splice-feats --left-context=3 --right-context=3 scp,s,cs:"{reference_final_features_path}" ark:- '
        f'| transform-feats "{sat_lda_mat_path}" ark:- ark:- |'
    )


@pytest.fixture(scope="session")
def reference_sat_feature_string(
    reference_dir, reference_final_features_path, sat_lda_mat_path, reference_trans_path
):
    utt2spk = reference_dir.joinpath("utt2spk.scp")
    return (
        f'ark,s,cs:splice-feats --left-context=3 --right-context=3 scp,s,cs:"{reference_final_features_path}" ark:- '
        f'| transform-feats "{sat_lda_mat_path}" ark:- ark:- | transform-feats --utt2spk=ark:"{utt2spk}" scp:"{reference_trans_path}" ark:- ark:- |'
    )


@pytest.fixture(scope="session")
def reference_first_pass_ali_path(
    wav_path,
    reference_dir,
    sat_align_model_path,
    sat_temp_dir,
    align_options,
    reference_si_feature_string,
):
    ali_path = reference_dir.joinpath("ali_first_pass.ark")
    fst_path = sat_temp_dir.joinpath("fsts.ark")
    subprocess.call(
        [
            "gmm-align-compiled",
            f"--transition-scale={align_options['transition_scale']}",
            f"--acoustic-scale={align_options['acoustic_scale']}",
            f"--self-loop-scale={align_options['self_loop_scale']}",
            f"--beam={align_options['beam']}",
            f"--retry-beam={align_options['retry_beam']}",
            "--careful=false",
            sat_align_model_path,
            f"ark,s,cs:{fst_path}",
            reference_si_feature_string,
            f"ark:{ali_path}",
        ],
        env=os.environ,
    )
    return ali_path


@pytest.fixture(scope="session")
def reference_second_pass_ali_path(
    wav_path,
    reference_dir,
    sat_model_path,
    sat_temp_dir,
    align_options,
    reference_sat_feature_string,
):
    ali_path = reference_dir.joinpath("ali_second_pass.ark")
    fst_path = sat_temp_dir.joinpath("fsts.ark")
    subprocess.call(
        [
            "gmm-align-compiled",
            f"--transition-scale={align_options['transition_scale']}",
            f"--acoustic-scale={align_options['acoustic_scale']}",
            f"--self-loop-scale={align_options['self_loop_scale']}",
            f"--beam={align_options['beam']}",
            f"--retry-beam={align_options['retry_beam']}",
            "--careful=false",
            sat_model_path,
            f"ark,s,cs:{fst_path}",
            reference_sat_feature_string,
            f"ark:{ali_path}",
        ],
        env=os.environ,
    )
    return ali_path


@pytest.fixture(scope="session")
def align_options():
    return {
        "transition_scale": 1.0,
        "acoustic_scale": 0.1,
        "self_loop_scale": 0.1,
        "beam": 10,
        "retry_beam": 40,
    }


@pytest.fixture(scope="session")
def fmllr_options():
    return {"fmllr_update_type": "full", "silence_weight": 0.0, "silence_csl": "1:2"}


@pytest.fixture(scope="session")
def reference_trans_path(
    wav_path,
    reference_dir,
    reference_final_features_path,
    reference_si_feature_string,
    fmllr_options,
    sat_align_model_path,
    sat_model_path,
):
    ark_path = reference_dir.joinpath("trans.ark")
    scp_path = reference_dir.joinpath("trans.scp")
    ali_path = reference_dir.joinpath("ali_first_pass.ark")
    spk2utt = reference_dir.joinpath("spk2utt.scp")

    post_proc = subprocess.Popen(
        ["ali-to-post", f"ark,s,cs:{ali_path}", "ark:-"],
        stdout=subprocess.PIPE,
        env=os.environ,
    )

    weight_proc = subprocess.Popen(
        [
            "weight-silence-post",
            "0.0",
            fmllr_options["silence_csl"],
            sat_align_model_path,
            "ark,s,cs:-",
            "ark:-",
        ],
        stdin=post_proc.stdout,
        stdout=subprocess.PIPE,
        env=os.environ,
    )
    post_gpost_proc = subprocess.Popen(
        [
            "gmm-post-to-gpost",
            sat_align_model_path,
            reference_si_feature_string,
            "ark,s,cs:-",
            "ark:-",
        ],
        stdin=weight_proc.stdout,
        stdout=subprocess.PIPE,
        env=os.environ,
    )
    est_proc = subprocess.Popen(
        [
            "gmm-est-fmllr-gpost",
            f"--fmllr-update-type={fmllr_options['fmllr_update_type']}",
            f"--spk2utt=ark:{spk2utt}",
            sat_model_path,
            reference_si_feature_string,
            "ark,s,cs:-",
            f"ark,scp:{ark_path},{scp_path}",
        ],
        encoding="utf8",
        stdin=post_gpost_proc.stdout,
        env=os.environ,
    )
    est_proc.communicate()
    return scp_path


@pytest.fixture(scope="session")
def reference_trans_compose_path(
    wav_path,
    reference_dir,
    reference_final_features_path,
    reference_sat_feature_string,
    fmllr_options,
    sat_model_path,
    reference_trans_path,
    reference_second_pass_ali_path,
):
    temp_ark_path = reference_dir.joinpath("trans_second.ark")
    temp_scp_path = reference_dir.joinpath("trans_second.scp")
    ark_path = reference_dir.joinpath("trans_composed.ark")
    scp_path = reference_dir.joinpath("trans_composed.scp")
    spk2utt = reference_dir.joinpath("spk2utt.scp")

    post_proc = subprocess.Popen(
        ["ali-to-post", f"ark,s,cs:{reference_second_pass_ali_path}", "ark:-"],
        stdout=subprocess.PIPE,
        env=os.environ,
    )

    weight_proc = subprocess.Popen(
        [
            "weight-silence-post",
            "0.0",
            fmllr_options["silence_csl"],
            sat_model_path,
            "ark,s,cs:-",
            "ark:-",
        ],
        stdin=post_proc.stdout,
        stdout=subprocess.PIPE,
        env=os.environ,
    )
    est_proc = subprocess.Popen(
        [
            "gmm-est-fmllr",
            f"--fmllr-update-type={fmllr_options['fmllr_update_type']}",
            f"--spk2utt=ark:{spk2utt}",
            sat_model_path,
            reference_sat_feature_string,
            "ark,s,cs:-",
            f"ark,scp:{temp_ark_path},{temp_scp_path}",
        ],
        encoding="utf8",
        stdin=weight_proc.stdout,
        env=os.environ,
    )
    est_proc.communicate()
    compose_proc = subprocess.Popen(
        [
            "compose-transforms",
            "--b-is-affine=true",
            f"scp:{temp_scp_path}",
            f"scp:{reference_trans_path}",
            f"ark,scp:{ark_path},{scp_path}",
        ],
        env=os.environ,
    )
    compose_proc.communicate()
    return scp_path


@pytest.fixture(scope="session")
def mono_temp_dir(temp_dir):
    p = temp_dir.joinpath("mono")
    p.mkdir(exist_ok=True)
    return p


@pytest.fixture(scope="session")
def sat_temp_dir(temp_dir):
    p = temp_dir.joinpath("sat")
    p.mkdir(exist_ok=True)
    return p


@pytest.fixture(scope="session")
def dictionaries_dir(test_dir):
    return test_dir.joinpath("dictionaries")


@pytest.fixture(scope="session")
def lm_dir(test_dir):
    return test_dir.joinpath("language_models")


@pytest.fixture(scope="session")
def acoustic_corpus_text():
    return "this is the acoustic corpus i'm talking pretty fast here there's nothing going else going on we're just yknow there's some speech errors but who cares um this is me talking really slow and slightly lower in intensity uh we're just saying some words and here's some more words words words words um and that should be all thanks"


@pytest.fixture(scope="session")
def sat_phones():
    return [
        "a",
        "aj",
        "aw",
        "aː",
        "b",
        "bʲ",
        "c",
        "cʰ",
        "cʷ",
        "d",
        "dʒ",
        "dʲ",
        "d̪",
        "e",
        "ej",
        "eː",
        "f",
        "fʲ",
        "fʷ",
        "h",
        "i",
        "iː",
        "j",
        "k",
        "kp",
        "kʰ",
        "kʷ",
        "l",
        "m",
        "mʲ",
        "m̩",
        "n",
        "n̩",
        "o",
        "ow",
        "oː",
        "p",
        "pʰ",
        "pʲ",
        "pʷ",
        "s",
        "t",
        "tʃ",
        "tʰ",
        "tʲ",
        "tʷ",
        "t̪",
        "u",
        "uː",
        "v",
        "vʲ",
        "vʷ",
        "w",
        "z",
        "æ",
        "ç",
        "ð",
        "ŋ",
        "ɐ",
        "ɑ",
        "ɑː",
        "ɒ",
        "ɒː",
        "ɔ",
        "ɔj",
        "ɖ",
        "ə",
        "əw",
        "ɚ",
        "ɛ",
        "ɛː",
        "ɜ",
        "ɜː",
        "ɝ",
        "ɟ",
        "ɟʷ",
        "ɡ",
        "ɡb",
        "ɡʷ",
        "ɪ",
        "ɫ",
        "ɫ̩",
        "ɱ",
        "ɲ",
        "ɹ",
        "ɾ",
        "ɾʲ",
        "ɾ̃",
        "ʃ",
        "ʈ",
        "ʈʲ",
        "ʈʷ",
        "ʉ",
        "ʉː",
        "ʊ",
        "ʋ",
        "ʎ",
        "ʒ",
        "ʔ",
        "θ",
    ]


@pytest.fixture(scope="session")
def wav_path(wav_dir):
    return wav_dir.joinpath("acoustic_corpus.wav")


@pytest.fixture(scope="session")
def lm_path(lm_dir):
    return lm_dir.joinpath("test_lm.arpa")


@pytest.fixture(scope="session")
def mono_model_path(mono_am_dir):
    return mono_am_dir.joinpath("final.mdl")


@pytest.fixture(scope="session")
def mono_tree_path(mono_am_dir):
    return mono_am_dir.joinpath("tree")


@pytest.fixture(scope="session")
def sat_align_model_path(sat_am_dir):
    return sat_am_dir.joinpath("final.alimdl")


@pytest.fixture(scope="session")
def sat_model_path(sat_am_dir):
    return sat_am_dir.joinpath("final.mdl")


@pytest.fixture(scope="session")
def sat_tree_path(sat_am_dir):
    return sat_am_dir.joinpath("tree")


@pytest.fixture(scope="session")
def sat_lda_mat_path(sat_am_dir):
    return sat_am_dir.joinpath("lda.mat")


@pytest.fixture(scope="session")
def dictionary_path(dictionaries_dir):
    return dictionaries_dir.joinpath("test_basic.txt")


@pytest.fixture(scope="session")
def sat_dictionary_path(dictionaries_dir):
    return dictionaries_dir.joinpath("test_sat.txt")
