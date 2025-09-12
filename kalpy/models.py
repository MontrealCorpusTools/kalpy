"""
Model classes
=============

"""
from __future__ import annotations

import json
import logging
import os
import typing
from pathlib import Path

import pywrapfst
import yaml

from _kalpy.feat import (
    DeltaFeaturesOptions,
    SlidingWindowCmnOptions,
    compute_deltas,
    paste_feats,
    sliding_window_cmn,
    splice_frames,
)
from _kalpy.matrix import DoubleMatrix, FloatMatrix
from _kalpy.transform import ApplyCmvn, apply_transform
from kalpy.data import Segment
from kalpy.exceptions import AcousticModelError
from kalpy.feat.cmvn import CmvnComputer
from kalpy.feat.mfcc import MfccComputer
from kalpy.feat.pitch import PitchComputer
from kalpy.fstext.lexicon import LexiconCompiler
from kalpy.gmm.utils import read_gmm_model
from kalpy.utils import read_kaldi_object

if typing.TYPE_CHECKING:
    from _kalpy.gmm import AmDiagGmm
    from _kalpy.hmm import TransitionModel

logger = logging.getLogger("mfa")


class AcousticModel:
    """
    Class for storing acoustic models in MFA, exported as zip files containing the necessary Kaldi files
    to be reused

    """

    def __init__(
        self,
        directory: typing.Union[str, Path],
        validate: bool = True,
    ):
        self.directory = Path(directory)
        if validate:
            self.validate_directory()
        self._am = None
        self._tm = None
        self._meta = None
        self._phone_pdf_counts = None

    def validate_directory(self):
        required_files = [
            "final.mdl",
            "tree",
        ]
        for f in required_files:
            if not self.directory.joinpath(f).exists():
                raise AcousticModelError(f"Could not find {f} in {self.directory}.")

    def generate_features(
        self,
        segment: Segment,
        cmvn: typing.Optional[DoubleMatrix] = None,
        fmllr_trans: FloatMatrix = None,
        uses_speaker_adaptation: bool = True,
        uses_splices: bool = False,
        uses_deltas: bool = True,
        splice_context: int = 3,
    ):
        if self.lda_mat is not None:
            uses_splices = True
            uses_deltas = False
        feats = self.mfcc_computer.compute_mfccs_for_export(segment, compress=False)
        if cmvn is None:
            cmvn_computer = CmvnComputer()
            cmvn = cmvn_computer.compute_cmvn_from_features([feats])
        ApplyCmvn(cmvn, False, feats)

        if self.pitch_computer is not None:
            pitch = self.pitch_computer.compute_pitch_for_export(segment, compress=False)
            feats = paste_feats([feats, pitch], 1)
        if uses_splices:
            feats = splice_frames(feats, splice_context, splice_context)
            if self.lda_mat is not None:
                feats = apply_transform(feats, self.lda_mat)
        elif uses_deltas:
            delta_options = DeltaFeaturesOptions()
            feats = compute_deltas(delta_options, feats)
        if uses_speaker_adaptation and fmllr_trans is not None:
            feats = apply_transform(feats, fmllr_trans)
        return feats

    def generate_features_for_fine_tune(
        self,
        segment: Segment,
        cmvn: typing.Optional[DoubleMatrix] = None,
        fmllr_trans: FloatMatrix = None,
        uses_speaker_adaptation: bool = True,
        uses_splices: bool = False,
        uses_deltas: bool = True,
        splice_context: int = 3,
    ):

        mfcc_options = self.mfcc_options
        mfcc_options["frame_shift"] /= 10
        mfcc_computer = MfccComputer(**mfcc_options)
        if self.lda_mat is not None:
            uses_splices = True
            uses_deltas = False
        feats = mfcc_computer.compute_mfccs_for_export(segment, compress=False)
        if cmvn is None:
            cmvn_computer = CmvnComputer()
            cmvn = cmvn_computer.compute_cmvn_from_features([feats])
        ApplyCmvn(cmvn, False, feats)

        if self.pitch_computer is not None:
            pitch_options = self.pitch_options
            pitch_options["frame_shift"] /= 10
            pitch_computer = PitchComputer(**pitch_options)
            pitch = pitch_computer.compute_pitch_for_export(segment, compress=False)
            feats = paste_feats([feats, pitch], 1)
        if uses_splices:
            feats = splice_frames(feats, splice_context, splice_context)
            if self.lda_mat is not None:
                feats = apply_transform(feats, self.lda_mat)
        elif uses_deltas:
            delta_options = DeltaFeaturesOptions()
            feats = compute_deltas(delta_options, feats)
        if uses_speaker_adaptation and fmllr_trans is not None:
            feats = apply_transform(feats, fmllr_trans)
        return feats

    @property
    def version(self):
        return self.meta["version"]

    @property
    def uses_cmvn(self):
        return self.meta["features"]["uses_cmvn"]

    @property
    def parameters(self) -> typing.Dict[str, typing.Any]:
        """Parameters to pass to top-level workers"""
        params = {**self.meta["features"]}
        params["non_silence_phones"] = {x for x in self.meta["phones"]}
        params["oov_phone"] = self.meta["oov_phone"]
        params["language"] = self.meta["language"]
        params["optional_silence_phone"] = self.meta["optional_silence_phone"]
        params["phone_set_type"] = self.meta["phone_set_type"]
        params["silence_probability"] = self.meta.get("silence_probability", 0.5)
        params["initial_silence_probability"] = self.meta.get("initial_silence_probability", 0.5)
        params["final_non_silence_correction"] = self.meta.get(
            "final_non_silence_correction", None
        )
        params["final_silence_correction"] = self.meta.get("final_silence_correction", None)
        if "other_noise_phone" in self.meta:
            params["other_noise_phone"] = self.meta["other_noise_phone"]
        if (
            "dictionaries" in self.meta
            and "position_dependent_phones" in self.meta["dictionaries"]
        ):
            params["position_dependent_phones"] = self.meta["dictionaries"][
                "position_dependent_phones"
            ]
        else:
            params["position_dependent_phones"] = self.meta.get("position_dependent_phones", True)
        return params

    @property
    def tree_path(self) -> Path:
        """Current acoustic model path"""
        return self.directory.joinpath("tree")

    @property
    def lda_mat_path(self) -> Path:
        """Current acoustic model path"""
        return self.directory.joinpath("lda.mat")

    @property
    def model_path(self) -> Path:
        """Current acoustic model path"""
        return self.directory.joinpath("final.mdl")

    @property
    def phone_symbol_path(self) -> Path:
        """Path to phone symbol table"""
        return self.directory.joinpath("phones.txt")

    @property
    def phone_pdf_counts_path(self) -> Path:
        """Path to phone symbol table"""
        return self.directory.joinpath("phone_pdf.counts")

    @property
    def phone_pdf_counts(self):
        if not self.phone_pdf_counts_path.exists():
            return {}
        if self._phone_pdf_counts is None:
            with open(self.phone_pdf_counts_path, "r", encoding="utf8") as f:
                data = json.load(f)
            self._phone_pdf_counts = {}
            for phone, pdf_counts in data.items():
                self._phone_pdf_counts[phone] = {}
                for pdf, count in pdf_counts.items():
                    self._phone_pdf_counts[phone][int(pdf)] = count

            for phone, pdf_counts in self._phone_pdf_counts.items():
                phone_total = sum(pdf_counts.values())
                for pdf, count in pdf_counts.items():
                    self._phone_pdf_counts[phone][int(pdf)] = count / phone_total
        return self._phone_pdf_counts

    @property
    def alignment_model_path(self) -> Path:
        """Alignment model path"""
        path = self.model_path.with_suffix(".alimdl")
        if os.path.exists(path):
            return path
        return self.model_path

    @property
    def acoustic_model(self) -> AmDiagGmm:
        if not self.alignment_model_path.exists():
            raise AcousticModelError(f"Could not find {self.alignment_model_path}")
        if self._am is None:
            self._tm, self._am = read_gmm_model(self.alignment_model_path)
        return self._am

    @property
    def transition_model(self) -> TransitionModel:
        if not self.alignment_model_path.exists():
            raise AcousticModelError(f"Could not find {self.alignment_model_path}")
        if self._tm is None:
            self._tm, self._am = read_gmm_model(self.alignment_model_path)
        return self._tm

    @property
    def lexicon_compiler(self):
        lc = LexiconCompiler(
            silence_probability=self.meta.get("silence_probability", 0.5),
            initial_silence_probability=self.meta.get("initial_silence_probability", 0.5),
            final_silence_correction=self.meta.get("final_silence_correction", None),
            final_non_silence_correction=self.meta.get("final_non_silence_correction", None),
            silence_phone=self.meta.get("optional_silence_phone", "sil"),
            oov_phone=self.meta.get("oov_phone", "sil"),
            position_dependent_phones=self.meta.get("position_dependent_phones", False),
            phones={x for x in self.meta["phones"]},
        )
        if self.meta.get("phone_mapping", None):
            lc.phone_table = pywrapfst.SymbolTable()
            for k, v in self.meta["phone_mapping"].items():
                lc.phone_table.add_symbol(k, v)
        elif self.phone_symbol_path.exists():
            lc.phone_table = pywrapfst.SymbolTable.read_text(self.phone_symbol_path)
        return lc

    @property
    def mfcc_computer(self) -> MfccComputer:
        return MfccComputer(**self.mfcc_options)

    @property
    def pitch_computer(self) -> typing.Optional[PitchComputer]:
        if self.meta["features"].get("use_pitch", False):
            return PitchComputer(**self.pitch_options)
        return

    @property
    def lda_mat(self) -> FloatMatrix:
        lda_mat_path = self.directory.joinpath("lda.mat")
        lda_mat = None
        if lda_mat_path.exists():
            lda_mat = read_kaldi_object(FloatMatrix, lda_mat_path)
        return lda_mat

    @property
    def mfcc_options(self) -> typing.Dict[str, typing.Any]:
        """Parameters to use in computing MFCC features."""
        return {
            "sample_frequency": self.meta["features"].get("sample_frequency", 16000),
            "frame_shift": self.meta["features"].get("frame_shift", 10),
            "frame_length": self.meta["features"].get("frame_length", 25),
            "dither": self.meta["features"].get("dither", 0.0001),
            "preemphasis_coefficient": self.meta["features"].get("preemphasis_coefficient", 0.97),
            "snip_edges": self.meta["features"].get("snip_edges", True),
            "num_mel_bins": self.meta["features"].get("num_mel_bins", 23),
            "low_frequency": self.meta["features"].get("low_frequency", 20),
            "high_frequency": self.meta["features"].get("high_frequency", 7800),
            "num_coefficients": self.meta["features"].get("num_coefficients", 13),
            "use_energy": self.meta["features"].get("use_energy", False),
            "energy_floor": self.meta["features"].get("energy_floor", 1.0),
            "raw_energy": self.meta["features"].get("raw_energy", True),
            "cepstral_lifter": self.meta["features"].get("cepstral_lifter", 22),
        }

    @property
    def frame_shift_seconds(self):
        return round(self.meta["features"].get("frame_shift", 10) / 1000, 4)

    @property
    def pitch_options(self) -> typing.Dict[str, typing.Any]:
        """Parameters to use in computing pitch features."""
        use_pitch = self.meta["features"].get("use_pitch", False)
        use_voicing = self.meta["features"].get("use_voicing", False)
        use_delta_pitch = self.meta["features"].get("use_delta_pitch", False)
        normalize = self.meta["features"].get("normalize_pitch", True)
        options = {
            "frame_shift": self.meta["features"].get("frame_shift", 10),
            "frame_length": self.meta["features"].get("frame_length", 25),
            "min_f0": self.meta["features"].get("min_f0", 50),
            "max_f0": self.meta["features"].get("max_f0", 800),
            "sample_frequency": self.meta["features"].get("sample_frequency", 16000),
            "penalty_factor": self.meta["features"].get("penalty_factor", 0.1),
            "delta_pitch": self.meta["features"].get("delta_pitch", 0.005),
            "snip_edges": self.meta["features"].get("snip_edges", True),
            "add_normalized_log_pitch": False,
            "add_delta_pitch": False,
            "add_pov_feature": False,
        }
        if use_pitch:
            options["add_normalized_log_pitch"] = normalize
            options["add_raw_log_pitch"] = not normalize
        options["add_delta_pitch"] = use_delta_pitch
        options["add_pov_feature"] = use_voicing
        return options

    @property
    def lda_options(self) -> typing.Dict[str, typing.Any]:
        """Parameters to use in computing MFCC features."""
        return {
            "splice_left_context": self.meta["features"].get("splice_left_context", 3),
            "splice_right_context": self.meta["features"].get("splice_right_context", 3),
        }

    def _load_meta_data(self):
        default_features = {
            "feature_type": "mfcc",
            "use_energy": False,
            "frame_shift": 10,
            "snip_edges": True,
            "low_frequency": 20,
            "high_frequency": 7800,
            "sample_frequency": 16000,
            "allow_downsample": True,
            "allow_upsample": True,
            "use_pitch": False,
            "use_voicing": False,
            "uses_cmvn": True,
            "uses_deltas": True,
            "uses_splices": False,
            "uses_voiced": False,
            "uses_speaker_adaptation": False,
            "silence_weight": 0.0,
            "fmllr_update_type": "full",
            "splice_left_context": 3,
            "splice_right_context": 3,
        }
        if not self._meta:
            meta_path = self.directory.joinpath("meta.json")
            file_format = "json"
            if not os.path.exists(meta_path):
                meta_path = self.directory.joinpath("meta.yaml")
                file_format = "yaml"
            if not os.path.exists(meta_path):
                self._meta = {
                    "version": "0.9.0",
                    "architecture": "gmm-hmm",
                    "features": default_features,
                }
            else:
                with open(meta_path, "r", encoding="utf8") as f:
                    if file_format == "yaml":
                        self._meta = yaml.load(f, Loader=yaml.Loader)
                    else:
                        self._meta = json.load(f)
                if self._meta["features"] == "mfcc+deltas":
                    self._meta["features"] = default_features
                    if "pitch" in self._meta["features"]:
                        self._meta["features"]["use_pitch"] = self._meta["features"].pop("pitch")
                if (
                    self._meta["features"].get("use_pitch", False)
                    and self._meta["version"] < "2.0.6"
                ):
                    self._meta["features"]["use_delta_pitch"] = True
            if "phone_type" not in self._meta:
                self._meta["phone_type"] = "triphone"
            if "optional_silence_phone" not in self._meta:
                self._meta["optional_silence_phone"] = "sil"
            if "oov_phone" not in self._meta:
                self._meta["oov_phone"] = "spn"
            if file_format == "yaml":
                self._meta["other_noise_phone"] = "sp"
            if "phone_set_type" not in self._meta:
                self._meta["phone_set_type"] = "UNKNOWN"
            if "language" not in self._meta or self._meta["version"] < "3.0":
                self._meta["language"] = "unknown"
            self._meta["phones"] = set(self._meta.get("phones", []))
            if (
                "uses_speaker_adaptation" not in self._meta["features"]
                or not self._meta["features"]["uses_speaker_adaptation"]
            ):
                self._meta["features"]["uses_speaker_adaptation"] = os.path.exists(
                    self.directory.joinpath("final.alimdl")
                )
            if self._meta["version"] in {"0.9.0", "1.0.0"}:
                self._meta["features"]["uses_speaker_adaptation"] = True
            if (
                "uses_splices" not in self._meta["features"]
                or not self._meta["features"]["uses_splices"]
            ):
                self._meta["features"]["uses_splices"] = os.path.exists(
                    self.directory.joinpath("lda.mat")
                )
                if self._meta["features"]["uses_splices"]:
                    self._meta["features"]["uses_deltas"] = False
            if (
                self._meta["features"].get("use_pitch", False)
                and "use_voicing" not in self._meta["features"]
            ):
                self._meta["features"]["use_voicing"] = True
            if (
                "dictionaries" in self._meta
                and "position_dependent_phones" not in self._meta["dictionaries"]
            ):
                if self._meta["version"] < "2.0":
                    default_value = True
                else:
                    default_value = False
                self._meta["dictionaries"]["position_dependent_phones"] = self._meta.get(
                    "position_dependent_phones", default_value
                )
        self.parse_old_features()

    @property
    def meta(self) -> typing.Dict[str, typing.Any]:
        """
        Metadata information for the acoustic model
        """
        if not self._meta:
            self._load_meta_data()
        return self._meta

    def parse_old_features(self) -> None:
        """
        Parse MFA model's features and ensure that they are up-to-date with current functionality
        """
        if "features" not in self._meta:
            return
        feature_key_remapping = {
            "type": "feature_type",
            "deltas": "uses_deltas",
            "fmllr": "uses_speaker_adaptation",
        }

        for key, new_key in feature_key_remapping.items():
            if key in self._meta["features"]:
                self._meta["features"][new_key] = self._meta["features"][key]
                del self._meta["features"][key]
        if "uses_splices" not in self._meta["features"]:  # Backwards compatibility
            self._meta["features"]["uses_splices"] = os.path.exists(
                self.directory.joinpath("lda.mat")
            )
        if "uses_speaker_adaptation" not in self._meta["features"]:
            self._meta["features"]["uses_speaker_adaptation"] = os.path.exists(
                self.directory.joinpath("final.alimdl")
            )
