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
def wav_path(wav_dir):
    return wav_dir.joinpath("acoustic_corpus.wav")
