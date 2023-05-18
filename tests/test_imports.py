def test_import():
    import _kalpy

    print(_kalpy.__version__)
    import _kalpy.chain
    import _kalpy.cudamatrix
    import _kalpy.decoder
    import _kalpy.feat
    import _kalpy.fstext
    import _kalpy.gmm
    import _kalpy.hmm
    import _kalpy.ivector
    import _kalpy.kws
    import _kalpy.lat
    import _kalpy.lm
    import _kalpy.matrix
    import _kalpy.tree
    import _kalpy.util
