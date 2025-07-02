
.. _changelog:

Changelog
=========

0.7.0
-----

- Added support for aligning from manual alignments to use in training/adapting
- Implemented Levenshtein alignment for interval overlap scoring
- Added a workaround for a bug in mismatched MFCC and pitch features differing in frames under certain conditions in Kaldi

0.6.6
-----

- Updated code in parsing phones to word pronunciations to better support transcription capabilities
- Change build system to ignore CUDA dependencies in an effort to simplify conda-forge build system

0.6.5
-----

- Changed how the :code:`silence_probability` parameter of :code:`LexiconCompiler` works with pronunciations that have silence probabilities, so that setting it to 0.0 will ensure that no optional silences are included
- Changed :code:`TrainingGraphCompiler` signature to require a :code:`LexiconCompiler` rather than an FST/path and a word table
- Added the functionality for adding interjection words in between each word in an alignment

0.6.0
-----

- Fixed a bug in feature archives where fMLLR transforms were being ignored

0.5.1
-----

- Added better error handling for creating archives for files that don't exist
- Added more input types for computing MFCCs

0.5.0
-----

- First official release
- Expanded functionality across all modules

0.1.0
------

Initial release
