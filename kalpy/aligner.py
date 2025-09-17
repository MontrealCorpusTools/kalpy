"""Classes for higher level alignment"""
from __future__ import annotations

import typing

from _kalpy.gmm import gmm_interpolate_boundary_fast
from _kalpy.hmm import SplitToPhones
from _kalpy.matrix import DoubleMatrix, FloatMatrix, FloatSubMatrix
from kalpy.data import Segment
from kalpy.decoder.training_graphs import TrainingGraphCompiler
from kalpy.exceptions import AlignerError
from kalpy.gmm.align import GmmAligner
from kalpy.gmm.data import CtmInterval
from kalpy.models import AcousticModel

if typing.TYPE_CHECKING:
    from kalpy.fstext.lexicon import LexiconCompiler
    from kalpy.gmm.data import Alignment, HierarchicalCtm
    from kalpy.utterance import Utterance


class KalpyAligner:
    """
    Aligner class for generating alignments from a :class:`~kalpy.models.AcousticModel` and
    :class:`~kalpy.fstext.lexicon.LexiconCompiler`.

    Parameters
    ----------
    acoustic_model: :class:`~kalpy.models.AcousticModel`
        Acoustic model
    lexicon_compiler: :class:`~kalpy.fstext.lexicon.LexiconCompiler` or dict[Any, :class:`~kalpy.fstext.lexicon.LexiconCompiler`]
        Single lexicon compiler to use for all utterances or a dictionary of different lexicon compilers to use
        depending on the utterance
    beam: int
        Size of the beam to use in decoding, defaults to 10
    retry_beam : int
        Size of the beam to use in decoding if it fails with the initial beam width, defaults to 40
    transition_scale : float
        Transition scale, defaults to 1.0
    acoustic_scale : float
        Acoustic scale, defaults to 0.1
    self_loop_scale : float
        Self-loop scale, defaults to 0.1
    boost_silence : float
        Factor to boost silence probabilities, 1.0 is no boost or reduction
    careful : bool
        Flag for extra error checking on reaching final state, defaults to False
    """

    def __init__(
        self,
        acoustic_model: AcousticModel,
        lexicon_compiler: typing.Union[LexiconCompiler, typing.Dict[typing.Any, LexiconCompiler]],
        beam: int = 10,
        retry_beam: int = 40,
        transition_scale: float = 1.0,
        acoustic_scale: float = 0.1,
        self_loop_scale: float = 0.1,
        boost_silence: float = 1.0,
        careful: bool = False,
    ):
        self.acoustic_model = acoustic_model
        if isinstance(lexicon_compiler, dict) and len(lexicon_compiler) == 1:
            lexicon_compiler = next(iter(lexicon_compiler.values()))
        self.lexicon_compiler = lexicon_compiler
        self.has_multiple_lexicons = isinstance(self.lexicon_compiler, dict)

        if self.has_multiple_lexicons:
            self.graph_compiler = {}
            for k, v in self.lexicon_compiler.items():
                self.graph_compiler[k] = TrainingGraphCompiler(
                    self.acoustic_model.model_path,
                    self.acoustic_model.tree_path,
                    v,
                )
        else:
            self.graph_compiler = TrainingGraphCompiler(
                self.acoustic_model.model_path,
                self.acoustic_model.tree_path,
                self.lexicon_compiler,
            )
        self.beam = beam
        self.retry_beam = retry_beam
        self.transition_scale = transition_scale
        self.acoustic_scale = acoustic_scale
        self.self_loop_scale = self_loop_scale
        self.boost_silence = boost_silence
        self.careful = careful
        self.aligner = GmmAligner(
            self.acoustic_model.model_path,
            beam=beam,
            retry_beam=retry_beam,
            transition_scale=transition_scale,
            acoustic_scale=acoustic_scale,
            self_loop_scale=self_loop_scale,
            careful=careful,
        )
        self.ali_aligner = None
        if self.acoustic_model.alignment_model_path != self.acoustic_model.model_path:
            self.ali_aligner = GmmAligner(
                self.acoustic_model.alignment_model_path,
                beam=beam,
                retry_beam=retry_beam,
                transition_scale=transition_scale,
                acoustic_scale=acoustic_scale,
                self_loop_scale=self_loop_scale,
                careful=careful,
            )
        if self.boost_silence != 1.0:
            if self.has_multiple_lexicons:
                silence_symbols = next(iter(self.lexicon_compiler.values())).silence_symbols
            else:
                silence_symbols = self.lexicon_compiler.silence_symbols
            if self.ali_aligner is not None:
                self.ali_aligner.boost_silence(boost_silence, silence_symbols)
            self.aligner.boost_silence(boost_silence, silence_symbols)

    def _align_utterance(
        self,
        utterance: Utterance,
        cmvn: DoubleMatrix = None,
        fmllr_trans: FloatMatrix = None,
        dictionary_id: typing.Any = None,
    ) -> Alignment:
        """
        Internal function for generating an :class:`~kalpy.gmm.data.Alignment` from an :class:`~kalpy.utterance.Utterance`

        Parameters
        ----------
        utterance: :class:`~kalpy.utterance.Utterance`
            Utterance to align
        cmvn: :class:`~_kalpy.matrix.DoubleMatrix`, optional
            CMVN transformation to use, if not provided, a CMVN transform will be applied based on the utterance features
        fmllr_trans: :class:`~_kalpy.matrix.FloatMatrix`, optional
            Feature transformation matrix for a speaker
        dictionary_id: int or str, optional
            Identifier for which lexicon compiler to use, if not specified, defaults to using the first lexicon compiler

        Raises
        ------
        :class:`~kalpy.exceptions.AlignerError`
            Raised when no alignment is generated

        Returns
        -------
        :class:`~kalpy.gmm.data.Alignment`
            Alignment object with list of transition IDs and other information for the utterance
        """
        feats = utterance.generate_features(
            self.acoustic_model, fmllr_trans=fmllr_trans, cmvn=cmvn
        )
        graph_compiler = self.graph_compiler
        if self.has_multiple_lexicons:
            if dictionary_id is None:
                dictionary_id = next(iter(self.lexicon_compiler.keys()))
            graph_compiler = self.graph_compiler[dictionary_id]

        fst = graph_compiler.compile_fst(utterance.transcript)
        aligner = (
            self.ali_aligner
            if fmllr_trans is None and self.ali_aligner is not None
            else self.aligner
        )
        alignment = aligner.align_utterance(fst, feats)
        if alignment is None:
            raise AlignerError(
                f"Could not align the file with the current beam size ({aligner.beam}, "
                "please try increasing the beam size via `--beam X`"
            )
        return alignment

    def align_utterance(
        self,
        utterance: Utterance,
        cmvn: DoubleMatrix = None,
        fmllr_trans: FloatMatrix = None,
        dictionary_id: typing.Any = None,
    ) -> HierarchicalCtm:
        """
        Function for generating an :class:`~kalpy.gmm.data.Alignment` from an :class:`~kalpy.utterance.Utterance`

        Parameters
        ----------
        utterance: :class:`~kalpy.utterance.Utterance`
            Utterance to align
        cmvn: :class:`~_kalpy.matrix.DoubleMatrix`, optional
            CMVN transformation to use, if not provided, a CMVN transform will be applied based on the utterance features
        fmllr_trans: :class:`~_kalpy.matrix.FloatMatrix`, optional
            Feature transformation matrix for a speaker
        dictionary_id: int or str, optional
            Identifier for which lexicon compiler to use, if not specified, defaults to using the first lexicon compiler

        Raises
        ------
        :class:`~kalpy.exceptions.AlignerError`
            Raised when no alignment is generated

        Returns
        -------
        :class:`~kalpy.gmm.data.HierarchicalCtm`
            Hierarchical CTM object with word and phone intervals for the utterance
        """
        lexicon_compiler = self._get_lexicon_compiler(dictionary_id)
        aligner = (
            self.ali_aligner
            if fmllr_trans is None and self.ali_aligner is not None
            else self.aligner
        )
        alignment = self._align_utterance(utterance, cmvn, fmllr_trans, dictionary_id)
        phone_intervals = alignment.generate_ctm(
            aligner.transition_model,
            lexicon_compiler.phone_table,
            self.acoustic_model.mfcc_computer.frame_shift,
        )
        ctm = lexicon_compiler.phones_to_pronunciations(
            alignment.words, phone_intervals, transcription=False, text=utterance.transcript
        )
        ctm.likelihood = alignment.likelihood
        ctm.update_utterance_boundaries(utterance.segment.begin, utterance.segment.end)
        return ctm

    def _get_lexicon_compiler(self, dictionary_id: typing.Any = None) -> LexiconCompiler:
        """
        Internal function for looking up the lexicon compiler for the specified key

        Parameters
        ----------
        dictionary_id: int or str, optional
            Key to look up lexicon compiler

        Returns
        -------
        :class:`~kalpy.fstext.lexicon.LexiconCompiler`
            Lexicon compiler
        """
        lexicon_compiler = self.lexicon_compiler
        if self.has_multiple_lexicons:
            if dictionary_id is None:
                dictionary_id = next(iter(self.lexicon_compiler.keys()))
            lexicon_compiler = self.lexicon_compiler[dictionary_id]
        return lexicon_compiler

    def fine_tune_alignments(
        self,
        utterance: Utterance,
        alignment: Alignment = None,
        boundary_tolerance: typing.Optional[float] = None,
        cmvn: DoubleMatrix = None,
        fmllr_trans: FloatMatrix = None,
        dictionary_id: typing.Any = None,
    ) -> HierarchicalCtm:
        """
        Function for finetuning an :class:`~kalpy.gmm.data.Alignment` for an :class:`~kalpy.utterance.Utterance`

        Parameters
        ----------
        utterance: :class:`~kalpy.utterance.Utterance`
            Utterance to align
        alignment: :class:`~kalpy.gmm.data.Alignment`, optional
            Alignment to finetune, if not specified, alignment will be generated
        boundary_tolerance: float, optional
            The range around a boundary that it is allowed to be moved in seconds,
            if not specified, it will default to two frames from the acoustic model features, one frame on either side
        cmvn: :class:`~_kalpy.matrix.DoubleMatrix`, optional
            CMVN transformation to use, if not provided, a CMVN transform will be applied based on the utterance features
        fmllr_trans: :class:`~_kalpy.matrix.FloatMatrix`, optional
            Feature transformation matrix for a speaker
        dictionary_id: int or str, optional
            Identifier for which lexicon compiler to use, if not specified, defaults to using the first lexicon compiler

        Raises
        ------
        :class:`~kalpy.exceptions.AlignerError`
            Raised when no alignment is generated

        Returns
        -------
        :class:`~kalpy.gmm.data.HierarchicalCtm`
            Hierarchical CTM object with word and phone intervals for the utterance
        """
        lexicon_compiler = self._get_lexicon_compiler(dictionary_id)
        aligner = (
            self.ali_aligner
            if fmllr_trans is None and self.ali_aligner is not None
            else self.aligner
        )
        if alignment is None:
            alignment = self._align_utterance(
                utterance, cmvn=cmvn, fmllr_trans=fmllr_trans, dictionary_id=dictionary_id
            )
        split = SplitToPhones(aligner.transition_model, alignment.alignment)
        phone_intervals = []
        phone_start = 0.0
        original_start = 0.0
        feature_padding = 0.04
        if boundary_tolerance is None:
            boundary_tolerance = self.acoustic_model.frame_shift_seconds * 2
        for i, s in enumerate(split):
            phone_id = aligner.transition_model.TransitionIdToPhone(s[0])
            duration = len(s) * self.acoustic_model.frame_shift_seconds
            original_end = original_start + duration
            boundary = original_end
            label = lexicon_compiler.phone_table.find(phone_id)
            if i != len(split) - 1:
                feature_segment_begin = max(
                    round(boundary - feature_padding, 4),
                    0,
                )
                feature_segment_end = min(
                    round(boundary + feature_padding, 4),
                    utterance.segment.end,
                )
                following_phone_duration = (
                    len(split[i + 1]) * self.acoustic_model.frame_shift_seconds
                )
                previous_phone_offset_window = min(boundary_tolerance, duration / 2) / 2
                following_phone_offset_window = (
                    min(boundary_tolerance, following_phone_duration / 2) / 2
                )
                begin_offset = round(
                    max(boundary - previous_phone_offset_window - feature_segment_begin, 0.0), 4
                )
                end_offset = round(
                    min(
                        boundary + following_phone_offset_window - feature_segment_begin,
                        feature_segment_end - feature_segment_begin,
                    ),
                    4,
                )
                seg = Segment(
                    utterance.segment.file_path, feature_segment_begin, feature_segment_end
                )
                feats = self.acoustic_model.generate_features_for_fine_tune(
                    seg, cmvn=cmvn, fmllr_trans=fmllr_trans
                )
                begin_index = int(round(begin_offset * 1000))
                end_index = int(round(end_offset * 1000))
                sub_matrix = FloatSubMatrix(
                    feats, begin_index, end_index - begin_index, 0, feats.NumCols()
                )
                feats = FloatMatrix(sub_matrix)
                previous_transition_id = s[-1]
                following_transition_id = split[i + 1][0]
                new_boundary_index = gmm_interpolate_boundary_fast(
                    aligner.acoustic_model,
                    self.acoustic_model.transition_model,
                    feats,
                    previous_transition_id,
                    following_transition_id,
                )
                boundary = round(
                    feature_segment_begin + begin_offset + (new_boundary_index * 0.001), 3
                )
            confidence = 0.0
            phone_intervals.append(
                CtmInterval(
                    round(phone_start, 3),
                    round(boundary, 3),
                    label,
                    phone_id,
                    confidence,
                )
            )
            phone_start = boundary
            original_start = original_end
        ctm = lexicon_compiler.phones_to_pronunciations(
            alignment.words, phone_intervals, transcription=False, text=utterance.transcript
        )
        ctm.likelihood = alignment.likelihood
        ctm.update_utterance_boundaries(utterance.segment.begin, utterance.segment.end)
        return ctm
