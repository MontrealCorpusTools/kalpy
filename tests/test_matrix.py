import numpy as np

from _kalpy.matrix import FloatMatrix


def test_casting():
    t = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="float32")
    mat = FloatMatrix()
    mat.from_numpy(t)
    print(mat.Stride())
    assert mat.NumRows() == 2
    assert mat.NumCols() == 3
    assert np.all(mat.numpy() == t)
