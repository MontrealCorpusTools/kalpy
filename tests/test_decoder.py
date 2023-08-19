import logging
import os
import sys

import pytest
import pywrapfst

from kalpy.decoder.decode_graph import DecodeGraphCompiler
from kalpy.decoder.training_graphs import TrainingGraphCompiler
from kalpy.fstext.lexicon import LexiconCompiler
from kalpy.fstext.utils import kaldi_to_pynini

logger = logging.getLogger("kalpy.graphs")
handler = logging.StreamHandler(sys.stdout)
logger.addHandler(handler)


@pytest.mark.order(1)
def test_training_graphs(
    mono_tree_path, mono_model_path, dictionary_path, mono_temp_dir, acoustic_corpus_text
):
    lc = LexiconCompiler(position_dependent_phones=False)
    lc.load_pronunciations(dictionary_path)
    lc.fst.write(str(mono_temp_dir.joinpath("lexicon.fst")))
    gc = TrainingGraphCompiler(mono_model_path, mono_tree_path, lc, lc.word_table)
    graph = kaldi_to_pynini(gc.compile_fst(acoustic_corpus_text))
    assert graph.num_states() > 0
    assert graph.start() != pywrapfst.NO_STATE_ID
    output_file_name = mono_temp_dir.joinpath("fsts.ark")
    gc.export_graphs(output_file_name, [("1", acoustic_corpus_text)])
    assert output_file_name.exists()
    os.remove(output_file_name)
    gc.export_graphs(output_file_name, [("1", acoustic_corpus_text)], write_scp=True)
    assert output_file_name.exists()
    assert output_file_name.with_suffix(".scp").exists()


@pytest.mark.order(1)
def test_training_graphs_sat(
    sat_tree_path,
    sat_model_path,
    sat_dictionary_path,
    sat_temp_dir,
    acoustic_corpus_text,
    sat_phones,
):
    lc = LexiconCompiler(position_dependent_phones=False, phones=sat_phones)
    lc.load_pronunciations(sat_dictionary_path)
    lc.fst.write(str(sat_temp_dir.joinpath("L_debug.fst")))
    lc.word_table.write_text(str(sat_temp_dir.joinpath("words.txt")))
    lc.phone_table.write_text(str(sat_temp_dir.joinpath("phones.txt")))
    gc = TrainingGraphCompiler(sat_model_path, sat_tree_path, lc, lc.word_table)
    graph = kaldi_to_pynini(gc.compile_fst(acoustic_corpus_text))
    assert graph.num_states() > 0
    assert graph.start() != pywrapfst.NO_STATE_ID
    graph.write(str(sat_temp_dir.joinpath("LG_debug.fst")))

    output_file_name = sat_temp_dir.joinpath("fsts.ark")
    gc.export_graphs(output_file_name, [("1", acoustic_corpus_text)])
    assert output_file_name.exists()
    os.remove(output_file_name)
    gc.export_graphs(output_file_name, [("1", acoustic_corpus_text)], write_scp=True)
    assert output_file_name.exists()
    assert output_file_name.with_suffix(".scp").exists()


@pytest.mark.order(1)
def test_decoding_model_sat(
    sat_tree_path,
    sat_model_path,
    sat_dictionary_path,
    sat_temp_dir,
    acoustic_corpus_text,
    sat_phones,
    lm_path,
):
    lc = LexiconCompiler(position_dependent_phones=False, phones=sat_phones, disambiguation=True)
    lc.load_pronunciations(sat_dictionary_path)
    lc.fst.write(str(sat_temp_dir.joinpath("L_debug.fst")))
    lc.word_table.write_text(str(sat_temp_dir.joinpath("words.txt")))
    lc.phone_table.write_text(str(sat_temp_dir.joinpath("phones.txt")))
    gc = DecodeGraphCompiler(sat_model_path, sat_tree_path, lc)
    gc.compile_hclg_fst(lm_path)
    hclg = gc.hclg_fst
    assert hclg.Type() == "const"
    assert hclg.NumStates() > 0
    assert hclg.Start() != pywrapfst.NO_STATE_ID

    output_file_name = sat_temp_dir.joinpath("hclg.fst")
    gc.export_hclg(lm_path, output_file_name)
    gc.export_g(sat_temp_dir.joinpath("g.fst"))
    assert output_file_name.exists()
    gc.load_from_file(output_file_name)
    assert gc.hclg_fst.Type() == "const"
    assert gc.hclg_fst.Start() != pywrapfst.NO_STATE_ID
    assert gc.hclg_fst.NumStates() == hclg.NumStates()
