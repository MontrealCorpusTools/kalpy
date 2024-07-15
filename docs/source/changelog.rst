
.. _changelog:

Changelog
=========

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
