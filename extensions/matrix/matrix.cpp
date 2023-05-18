#include "matrix/pybind_matrix.h"

#include "matrix/compressed-matrix.h"
#include "matrix/kaldi-matrix.h"
#include "matrix/matrix-common.h"
#include "matrix/kaldi-vector.h"
#include "matrix/matrix-common.h"
#include "matrix/sparse-matrix.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

using namespace kaldi;

void pybind_compressed_matrix(py::module& m) {
  py::enum_<CompressionMethod>(
      m, "CompressionMethod", py::arithmetic(),
      "The enum CompressionMethod is used when creating a CompressedMatrix (a "
      "lossily compressed matrix) from a regular Matrix. It dictates how we "
      "choose the compressed format and how we choose the ranges of floats "
      "that are represented by particular integers.")
      .value(
          "kAutomaticMethod", kAutomaticMethod,
          "This is the default when you don't specify the compression method. "
          " It is a shorthand for using kSpeechFeature if the num-rows is "
          "more than 8, and kTwoByteAuto otherwise.")
      .value(
          "kSpeechFeature", kSpeechFeature,
          "This is the most complicated of the compression methods, and was "
          "designed for speech features which have a roughly Gaussian "
          "distribution with different ranges for each dimension.  Each "
          "element is stored in one byte, but there is an 8-byte header per "
          "column; the spacing of the integer values is not uniform but is in "
          "3 ranges.")
      .value("kTwoByteAuto", kTwoByteAuto,
             "Each element is stored in two bytes as a uint16, with the "
             "representable range of values chosen automatically with the "
             "minimum and maximum elements of the matrix as its edges.")
      .value("kTwoByteSignedInteger", kTwoByteSignedInteger,
             "Each element is stored in two bytes as a uint16, with the "
             "representable range of value chosen to coincide with what you'd "
             "get if you stored signed integers, i.e. [-32768.0, 32767.0].  "
             "Suitable for waveform data that was previously stored as 16-bit "
             "PCM.")
      .value("kOneByteAuto", kOneByteAuto,
             "Each element is stored in one byte as a uint8, with the "
             "representable range of values chosen automatically with the "
             "minimum and maximum elements of the matrix as its edges.")
      .value("kOneByteUnsignedInteger", kOneByteUnsignedInteger,
             "Each element is stored in one byte as a uint8, with the "
             "representable range of values equal to [0.0, 255.0].")
      .value("kOneByteZeroOne", kOneByteZeroOne,
             "Each element is stored in one byte as a uint8, with the "
             "representable range of values equal to [0.0, 1.0].  Suitable for "
             "image data that has previously been compressed as int8.")
      .export_values();
  {
    using PyClass = CompressedMatrix;
    py::class_<PyClass>(m, "CompressedMatrix")
        .def(py::init<>())
        .def(py::init<const MatrixBase<float>&, CompressionMethod>(),
             py::arg("mat"), py::arg("method") = kAutomaticMethod)
        .def("NumRows", &PyClass::NumRows,
             "Returns number of rows (or zero for emtpy matrix).")
        .def("NumCols", &PyClass::NumCols,
             "Returns number of columns (or zero for emtpy matrix).");
  }
}



void pybind_kaldi_matrix(py::module& m) {
  py::class_<MatrixBase<float>,
             std::unique_ptr<MatrixBase<float>, py::nodelete>>(
      m, "FloatMatrixBase",
      "Base class which provides matrix operations not involving resizing\n"
      "or allocation.   Classes Matrix and SubMatrix inherit from it and take "
      "care of allocation and resizing.")
      .def("NumRows", &MatrixBase<float>::NumRows, "Return number of rows")
      .def("NumCols", &MatrixBase<float>::NumCols, "Return number of columns")
      .def("Stride", &MatrixBase<float>::Stride, "Return stride")
      .def("__repr__",
           [](const MatrixBase<float>& b) -> std::string {
             std::ostringstream str;
             b.Write(str, false);
             return str.str();
           })
      .def("__getitem__",
           [](const MatrixBase<float>& m, std::pair<ssize_t, ssize_t> i) {
             return m(i.first, i.second);
           })
      .def("__setitem__",
           [](MatrixBase<float>& m, std::pair<ssize_t, ssize_t> i, float v) {
             m(i.first, i.second) = v;
           })
      .def("numpy", [](py::object obj) {
        auto* m = obj.cast<MatrixBase<float>*>();
        return py::array_t<float>(
            {m->NumRows(), m->NumCols()},                  // shape
            {sizeof(float) * m->Stride(), sizeof(float)},  // stride in bytes
            m->Data(),                                     // ptr
            obj);  // it will increase the reference count of **this** matrix
      });

  py::class_<Matrix<float>, MatrixBase<float>>(m, "FloatMatrix",
                                               pybind11::buffer_protocol())
      .def_buffer([](const Matrix<float>& m) -> pybind11::buffer_info {
        return pybind11::buffer_info(
            (void*)m.Data(),  // pointer to buffer
            sizeof(float),    // size of one scalar
            pybind11::format_descriptor<float>::format(),
            2,                           // num-axes
            {m.NumRows(), m.NumCols()},  // buffer dimensions
            {sizeof(float) * m.Stride(),
             sizeof(float)});  // stride for each index (in chars)
      })
      .def(py::init<>())
      .def(py::init<const MatrixIndexT, const MatrixIndexT, MatrixResizeType,
                    MatrixStrideType>(),
           py::arg("row"), py::arg("col"), py::arg("resize_type") = kSetZero,
           py::arg("stride_type") = kDefaultStride)
      .def(py::init<const MatrixBase<float>&, MatrixTransposeType>(),
           py::arg("M"), py::arg("trans") = kNoTrans)
      .def("Read", &Matrix<float>::Read, "allows resizing", py::arg("is"),
           py::arg("binary"), py::arg("add") = false)
      .def("Write", &Matrix<float>::Write, py::arg("out"), py::arg("binary"));

  py::class_<SubMatrix<float>, MatrixBase<float>>(m, "FloatSubMatrix")
      .def(py::init([](py::buffer b) {
        py::buffer_info info = b.request();
        if (info.format != py::format_descriptor<float>::format()) {
          KALDI_ERR << "Expected format: "
                    << py::format_descriptor<float>::format() << "\n"
                    << "Current format: " << info.format;
        }
        if (info.ndim != 2) {
          KALDI_ERR << "Expected dim: 2\n"
                    << "Current dim: " << info.ndim;
        }

        // numpy is row major by default, so we use strides[0]
        return new SubMatrix<float>(reinterpret_cast<float*>(info.ptr),
                                    info.shape[0], info.shape[1],
                                    info.strides[0] / sizeof(float));
      }));

  py::class_<Matrix<double>, std::unique_ptr<Matrix<double>, py::nodelete>>(
      m, "DoubleMatrix",
      "This bind is only for internal use, e.g. by OnlineCmvnState.")
      .def(py::init<const Matrix<float>&>(), py::arg("src"));
}


void pybind_kaldi_vector(py::module& m) {
  py::class_<VectorBase<float>,
             std::unique_ptr<VectorBase<float>, py::nodelete>>(
      m, "FloatVectorBase",
      "Provides a vector abstraction class.\n"
      "This class provides a way to work with vectors in kaldi.\n"
      "It encapsulates basic operations and memory optimizations.")
      .def("Dim", &VectorBase<float>::Dim,
           "Returns the dimension of the vector.")
      .def("__repr__",
           [](const VectorBase<float>& v) -> std::string {
             std::ostringstream str;
             v.Write(str, false);
             return str.str();
           })
      .def("__getitem__",
           [](const VectorBase<float>& v, int i) { return v(i); })
      .def("__setitem__",
           [](VectorBase<float>& v, int i, float val) { v(i) = val; })
      .def("from_numpy", [](VectorBase<float>& v, py::array_t<float> x){
        auto r = x.unchecked<1>();
        for (py::size_t i = 0; i < r.shape(0); i++)
          v(i) = r(i);
      })
      .def("numpy", [](py::object obj) {
        auto* v = obj.cast<VectorBase<float>*>();
        return py::array_t<float>(
            {v->Dim()},       // shape
            {sizeof(float)},  // stride in bytes
            v->Data(),        // ptr
            obj);  // it will increase the reference count of **this** vector
      });

  py::class_<Vector<float>, VectorBase<float>>(m, "FloatVector",
                                               py::buffer_protocol())
      .def_buffer([](const Vector<float>& v) -> py::buffer_info {
        return py::buffer_info((void*)v.Data(), sizeof(float),
                               py::format_descriptor<float>::format(),
                               1,  // num-axes
                               {v.Dim()},
                               {sizeof(float)});  // strides (in chars)
      })
      .def(py::init<>())
      .def(py::init<const MatrixIndexT, MatrixResizeType>(), py::arg("size"),
           py::arg("resize_type") = kSetZero)
      .def(py::init<const VectorBase<float>&>(), py::arg("v"),
           "Copy-constructor from base-class, needed to copy from SubVector.")
      .def("Read", &Vector<float>::Read,
           "Reads from C++ stream (option to add to existing contents).Throws "
           "exception on failure",
           py::arg("in"), py::arg("binary"), py::arg("add") = false);

  py::class_<SubVector<float>, VectorBase<float>>(m, "FloatSubVector")
      .def(py::init([](py::buffer b) {
        py::buffer_info info = b.request();
        if (info.format != py::format_descriptor<float>::format()) {
          KALDI_ERR << "Expected format: "
                    << py::format_descriptor<float>::format() << "\n"
                    << "Current format: " << info.format;
        }
        if (info.ndim != 1) {
          KALDI_ERR << "Expected dim: 1\n"
                    << "Current dim: " << info.ndim;
        }
        return new SubVector<float>(reinterpret_cast<float*>(info.ptr),
                                    info.shape[0]);
      }));

}


void pybind_matrix_common(py::module& m) {
  py::enum_<MatrixResizeType>(m, "MatrixResizeType", py::arithmetic(),
                              "Matrix initialization policies")
      .value("kSetZero", kSetZero, "Set to zero")
      .value("kUndefined", kUndefined, "Leave undefined")
      .value("kCopyData", kCopyData, "Copy any previously existing data")
      .export_values();

  py::enum_<MatrixStrideType>(m, "MatrixStrideType", py::arithmetic(),
                              "Matrix stride policies")
      .value("kDefaultStride", kDefaultStride,
             "Set to a multiple of 16 in bytes")
      .value("kStrideEqualNumCols", kStrideEqualNumCols,
             "Set to the number of columns")
      .export_values();

  py::enum_<MatrixTransposeType>(
      m, "MatrixTransposeType", py::arithmetic(),
      "this enums equal to CblasTrans and CblasNoTrans constants from CBLAS "
      "library we are writing them as literals because we don't want to "
      "include here matrix/kaldi-blas.h, which puts many symbols into global "
      "scope (like 'real') via the header f2c.h")
      .value("kTrans", kTrans, "CblasTrans == 112")
      .value("kNoTrans", kNoTrans, "CblasNoTrans == 111")
      .export_values();
}



void pybind_sparse_matrix(py::module& m) {
  {
    using PyClass = SparseMatrix<BaseFloat>;
    py::class_<PyClass>(m, "SparseMatrix",
                        "This class is defined for sparse matrix type.")
        .def(py::init<>())
        .def(py::init<const MatrixBase<BaseFloat>&>(), py::arg("mat"))
        .def(py::init<int32, int32>(), py::arg("num_rows"), py::arg("num_cols"))
        .def("NumRows", &PyClass::NumRows, "Return number of rows")
        .def("NumCols", &PyClass::NumCols, "Return number of columns")
        .def("NumElements", &PyClass::NumElements, "Return number of elements");
  }
  {
    using PyClass = GeneralMatrix;
    py::class_<PyClass>(
        m, "GeneralMatrix",
        "This class is a wrapper that enables you to store a matrix in one of "
        "three forms: either as a Matrix<BaseFloat>, or a CompressedMatrix, or "
        "a SparseMatrix<BaseFloat>.  It handles the I/O for you, i.e. you read "
        "and write a single object type.  It is useful for neural-net training "
        "targets which might be sparse or not, and might be compressed or not.")
        .def(py::init<>())
        .def(py::init<const MatrixBase<BaseFloat>&>(), py::arg("mat"))
        .def("GetMatrix", &PyClass::GetMatrix,
             "Outputs the contents as a matrix returns kFullMatrix, this will "
             "work regardless of Type().",
             py::arg("mat"));
  }
}


void init_matrix(py::module &_m) {
  py::module m = _m.def_submodule("matrix", "matrix pybind for Kaldi");
  pybind_matrix_common(m);
  pybind_kaldi_vector(m);
  pybind_sparse_matrix(m);
  pybind_kaldi_matrix(m);
  pybind_compressed_matrix(m);
}
