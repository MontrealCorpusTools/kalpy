import pathlib

import pytest


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
