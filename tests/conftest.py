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
def temp_dir(test_dir):
    p = test_dir.joinpath("temp")
    p.mkdir(exist_ok=True)
    return p


@pytest.fixture(scope="session")
def dictionaries_dir(test_dir):
    return test_dir.joinpath("dictionaries")


@pytest.fixture(scope="session")
def wav_path(wav_dir):
    return wav_dir.joinpath("acoustic_corpus.wav")


@pytest.fixture(scope="session")
def transition_model_path(am_dir):
    return am_dir.joinpath("final.mdl")


@pytest.fixture(scope="session")
def tree_path(am_dir):
    return am_dir.joinpath("tree")


@pytest.fixture(scope="session")
def dictionary_path(dictionaries_dir):
    return dictionaries_dir.joinpath("test_basic.txt")
