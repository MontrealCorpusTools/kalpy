from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    import pynini

    from kalpy.gmm.data import CtmInterval, WordCtmInterval


class KalpyError(Exception):
    """
    Base exception class
    """

    def __init__(self, base_error_message: str, *args, **kwargs):
        self.message_lines: typing.List[str] = [base_error_message]

    @property
    def message(self) -> str:
        """Formatted exception message"""
        return "\n".join(self.message_lines)

    def __str__(self) -> str:
        """Output the error"""
        message = type(self).__name__ + ":"
        message += "\n\n" + self.message
        return message


class CtmError(KalpyError):
    """
    Class for errors in creating CTM intervals

    Parameters
    ----------
    ctm: :class:`~kalpy.gmm.data.CtmInterval`
        CTM interval that was not parsed correctly
    """

    def __init__(self, ctm: typing.Union[CtmInterval, WordCtmInterval]):
        KalpyError.__init__(self, f"Error was encountered in processing CTM interval: {ctm}")


class LexiconError(KalpyError):
    pass


class PhonesToPronunciationsError(KalpyError):
    """
    Class for errors in creating pronunciations from phones

    Parameters
    ----------
    ctm: :class:`~kalpy.gmm.data.CtmInterval`
        CTM interval that was not parsed correctly
    """

    def __init__(self, text: str, phones: str, phone_fst: pynini.Fst, phone_to_word: pynini.Fst):
        KalpyError.__init__(self, f"Error was encountered in creating pronunciations for: {text}")
        self.message_lines.append(f"Phones: {phones}")
        self.message_lines.append("Phone FST:")
        self.message_lines.append(str(phone_fst))
        self.message_lines.append("Phone to word FST:")
        self.message_lines.append(str(phone_to_word))
