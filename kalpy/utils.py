import logging
import pathlib
import typing
from contextlib import contextmanager

from _kalpy.util import Input, Output


def generate_read_specifier(file_name: typing.Union[str, pathlib.Path], sorted=True) -> str:
    file_name = str(file_name)
    if sorted:
        read_identifier = "ark,s,cs"
    else:
        read_identifier = "ark"
    if file_name.endswith(".scp"):
        if sorted:
            read_identifier = "scp,s,cs"
        else:
            read_identifier = "scp"
    return f"{read_identifier}:{file_name}"


def generate_write_specifier(
    file_name: typing.Union[str, pathlib.Path], write_scp: bool = False
) -> str:
    file_name = str(file_name)
    if not file_name.endswith(".ark"):
        file_name += ".ark"
    if write_scp:
        return f"ark,scp:{file_name},{file_name.replace('.ark', '.scp')}"
    return f"ark:{file_name}"


def write_kaldi_object(obj, path, binary=True):
    ko = Output(str(path), binary)
    obj.Write(ko.Stream(), binary)
    ko.Close()


def read_kaldi_object(obj_type, path, binary=True):
    ki = Input()
    ki.Open(str(path), True)
    obj = obj_type()
    obj.Read(ki.Stream(), binary)
    ki.Close()
    return obj


@contextmanager
def kalpy_logger(log_name: str, log_path: typing.Union[pathlib.Path, str]) -> logging.Logger:
    kalpy_logging = logging.getLogger(log_name)
    file_handler = logging.FileHandler(log_path, encoding="utf8")
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    try:
        kalpy_logging.addHandler(file_handler)
        yield kalpy_logging
    finally:
        file_handler.close()
        kalpy_logging.removeHandler(file_handler)


def get_kalpy_version() -> str:
    """
    Get the current Kalpy version

    Returns
    -------
    str
        Kalpy version
    """
    try:
        from ._version import version as __version__  # noqa
    except ImportError:
        __version__ = "0.2.0"
    return __version__
