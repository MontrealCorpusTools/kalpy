#include "matrix/pybind_matrix.h"

#include "matrix/compressed-matrix.h"
#include "matrix/kaldi-matrix.h"
#include "matrix/matrix-common.h"
#include "matrix/kaldi-vector.h"
#include "matrix/matrix-common.h"
#include "matrix/sparse-matrix.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "util/pybind_util.h"

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
        .def(py::init<const MatrixBase<double>&, CompressionMethod>(),
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
      .def("Row",
            (const SubVector<float> (MatrixBase<float>::*)(MatrixIndexT) const) &MatrixBase<float>::Row,
            "Return specific row of matrix.",
             py::arg("i"))
      .def("Row",
            (SubVector<float> (MatrixBase<float>::*)(MatrixIndexT)) &MatrixBase<float>::Row,
            "Return specific row of matrix.",
             py::arg("i"))
      .def("Range", &MatrixBase<float>::Range, "Return a sub-part of matrix.",
             py::arg("row_offset"),
             py::arg("num_rows"),
             py::arg("col_offset"),
             py::arg("num_cols"))
      .def("RowRange", &MatrixBase<float>::RowRange,
             py::arg("row_offset"),
             py::arg("num_rows"))
      .def("ColRange", &MatrixBase<float>::ColRange,
             py::arg("col_offset"),
             py::arg("num_cols"))
      .def("NumRows", &MatrixBase<float>::NumRows, "Return number of rows")
      .def("NumCols", &MatrixBase<float>::NumCols, "Return number of columns")
      .def("Stride", &MatrixBase<float>::Stride, "Return stride")
      .def("LogDet", &MatrixBase<float>::LogDet, "Returns logdet of matrix.",
             py::arg("det_sign") = NULL)
      .def("SetUnit",
            &MatrixBase<float>::SetUnit,
            "Sets to zero, except ones along diagonal [for non-square matrices too]")
      .def("SetZero",
            &MatrixBase<float>::SetZero,
            "Sets matrix to zero.")
      .def("SetRandn",
            &MatrixBase<float>::SetRandn,
            "Sets to random values of a normal distribution")
      .def("SetRandUniform",
            &MatrixBase<float>::SetRandUniform,
            "Sets to numbers uniformly distributed on (0, 1)")
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
      })
     .def_static("read_from_file", [](std::string file_path) {

               static Matrix<float> mat;
               ReadKaldiObject(file_path, &mat);
               return &mat;
          }, py::return_value_policy::reference);

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
           py::arg("binary"), py::arg("add") = false,
      py::call_guard<py::gil_scoped_release>())
      .def("Write", &Matrix<float>::Write, py::arg("out"), py::arg("binary"),
      py::call_guard<py::gil_scoped_release>())
      .def("Resize", &Matrix<float>::Resize,
           "Set vector to a specified size (can be zero). "
          "The value of the new data depends on resize_type: "
          "  -if kSetZero, the new data will be zero "
          "  -if kUndefined, the new data will be undefined "
          "  -if kCopyData, the new data will be the same as the old data in any "
          "     shared positions, and zero elsewhere. "
          "This function takes time proportional to the number of data elements.",
           py::arg("r"),
           py::arg("c"), py::arg("resize_type") = kSetZero, py::arg("stride_type") = kDefaultStride)
      .def("from_numpy",[](
        Matrix<float> &M,
        py::array_t<float> array
      ){
        py::buffer_info info = array.request();
        int32 ndim = array.ndim();
        KALDI_ASSERT(ndim == 2);
        auto r = array.unchecked<2>(); // x must have ndim = 2; can be non-writeable
        int32 num_rows = array.shape(0);
        int32 num_cols = array.shape(1);
        M.Resize(num_rows, num_cols);

        float* this_data;
      for (MatrixIndexT i = 0; i < num_rows; i++)
      {
        this_data = M.Row(i).Data();
        for (MatrixIndexT j = 0; j < num_cols; j++)
          this_data[j] = r(i, j);
      }
      })
      .def(py::pickle(
        [](const Matrix<float> &p) { // __getstate__
            /* Return a tuple that fully encodes the state of the object */
             std::ostringstream os;
             bool binary = true;
             p.Write(os, binary);
            return py::make_tuple(
                    py::bytes(os.str()));
        },
        [](py::tuple t) { // __setstate__
            if (t.size() != 1)
                throw std::runtime_error("Invalid state!");

            /* Create a new C++ instance */
            Matrix<float> *p = new Matrix<float>();

            /* Assign any additional state */
            std::istringstream str(t[0].cast<std::string>());
               p->Read(str, true, false);

            return p;
        }
    ));

  py::class_<SubMatrix<float>, MatrixBase<float>>(m, "FloatSubMatrix")
      .def(py::init<const MatrixBase<float>& ,
            const MatrixIndexT ,
            const MatrixIndexT ,
            const MatrixIndexT ,
            const MatrixIndexT >(),
           py::arg("T"),
           py::arg("ro"),
           py::arg("r"),
           py::arg("co"),
           py::arg("c"))
      .def(py::init<float *,
            MatrixIndexT ,
            MatrixIndexT ,
            MatrixIndexT >(),
           py::arg("data"),
           py::arg("num_rows"),
           py::arg("num_cols"),
           py::arg("stride"))
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

  py::class_<MatrixBase<double>,
             std::unique_ptr<MatrixBase<double>, py::nodelete>>(
      m, "DoubleMatrixBase",
      "Base class which provides matrix operations not involving resizing\n"
      "or allocation.   Classes Matrix and SubMatrix inherit from it and take "
      "care of allocation and resizing.")
      .def("Row",
            (const SubVector<double> (MatrixBase<double>::*)(MatrixIndexT) const) &MatrixBase<double>::Row,
            "Return specific row of matrix.",
             py::arg("i"))
      .def("Row",
            (SubVector<double> (MatrixBase<double>::*)(MatrixIndexT)) &MatrixBase<double>::Row,
            "Return specific row of matrix.",
             py::arg("i"))
      .def("Range", &MatrixBase<double>::Range, "Return a sub-part of matrix.",
             py::arg("row_offset"),
             py::arg("num_rows"),
             py::arg("col_offset"),
             py::arg("num_cols"))
      .def("LogDet", &MatrixBase<double>::LogDet, "Returns logdet of matrix.",
             py::arg("det_sign") = NULL)
      .def("RowRange", &MatrixBase<double>::RowRange,
             py::arg("row_offset"),
             py::arg("num_rows"))
      .def("ColRange", &MatrixBase<double>::ColRange,
             py::arg("col_offset"),
             py::arg("num_cols"))
      .def("NumRows", &MatrixBase<double>::NumRows, "Return number of rows")
      .def("SetUnit",
            &MatrixBase<double>::SetUnit,
            "Sets to zero, except ones along diagonal [for non-square matrices too]")
      .def("SetZero",
            &MatrixBase<double>::SetZero,
            "Sets matrix to zero.")
      .def("SetRandn",
            &MatrixBase<double>::SetRandn,
            "Sets to random values of a normal distribution")
      .def("SetRandUniform",
            &MatrixBase<double>::SetRandUniform,
            "Sets to numbers uniformly distributed on (0, 1)")
      .def("NumCols", &MatrixBase<double>::NumCols, "Return number of columns")
      .def("Stride", &MatrixBase<double>::Stride, "Return stride")
      .def("__repr__",
           [](const MatrixBase<double>& b) -> std::string {
             std::ostringstream str;
             b.Write(str, false);
             return str.str();
           })
      .def("__getitem__",
           [](const MatrixBase<double>& m, std::pair<ssize_t, ssize_t> i) {
             return m(i.first, i.second);
           })
      .def("__setitem__",
           [](MatrixBase<double>& m, std::pair<ssize_t, ssize_t> i, double v) {
             m(i.first, i.second) = v;
           })
      .def("numpy", [](py::object obj) {
        auto* m = obj.cast<MatrixBase<double>*>();
        return py::array_t<double>(
            {m->NumRows(), m->NumCols()},                  // shape
            {sizeof(double) * m->Stride(), sizeof(double)},  // stride in bytes
            m->Data(),                                     // ptr
            obj);  // it will increase the reference count of **this** matrix
      })
     .def_static("read_from_file", [](std::string file_path) {

               static Matrix<double> mat;
               ReadKaldiObject(file_path, &mat);
               return &mat;
          }, py::return_value_policy::reference);

  py::class_<Matrix<double>, MatrixBase<double>>(m, "DoubleMatrix",
                                               pybind11::buffer_protocol())
      .def_buffer([](const Matrix<double>& m) -> pybind11::buffer_info {
        return pybind11::buffer_info(
            (void*)m.Data(),  // pointer to buffer
            sizeof(double),    // size of one scalar
            pybind11::format_descriptor<double>::format(),
            2,                           // num-axes
            {m.NumRows(), m.NumCols()},  // buffer dimensions
            {sizeof(double) * m.Stride(),
             sizeof(double)});  // stride for each index (in chars)
      })
      .def(py::init<>())
      .def(py::init<const MatrixIndexT, const MatrixIndexT, MatrixResizeType,
                    MatrixStrideType>(),
           py::arg("row"), py::arg("col"), py::arg("resize_type") = kSetZero,
           py::arg("stride_type") = kDefaultStride)
      .def(py::init<const MatrixBase<double>&, MatrixTransposeType>(),
           py::arg("M"), py::arg("trans") = kNoTrans)
      .def("Read", &Matrix<double>::Read, "allows resizing", py::arg("is"),
           py::arg("binary"), py::arg("add") = false,
      py::call_guard<py::gil_scoped_release>())
      .def("Write", &Matrix<double>::Write, py::arg("out"), py::arg("binary"),
      py::call_guard<py::gil_scoped_release>())
      .def("Resize", &Matrix<double>::Resize,
           "Set vector to a specified size (can be zero). "
          "The value of the new data depends on resize_type: "
          "  -if kSetZero, the new data will be zero "
          "  -if kUndefined, the new data will be undefined "
          "  -if kCopyData, the new data will be the same as the old data in any "
          "     shared positions, and zero elsewhere. "
          "This function takes time proportional to the number of data elements.",
           py::arg("r"),
           py::arg("c"), py::arg("resize_type") = kSetZero, py::arg("stride_type") = kDefaultStride)
      .def("from_numpy",[](
        Matrix<double> &M,
        py::array_t<double> array
      ){
        py::buffer_info info = array.request();
        int32 ndim = array.ndim();
        KALDI_ASSERT(ndim == 2);
        auto r = array.unchecked<2>(); // x must have ndim = 2; can be non-writeable
        int32 num_rows = array.shape(0);
        int32 num_cols = array.shape(1);
        M.Resize(num_rows, num_cols);

        double* this_data;
      for (MatrixIndexT i = 0; i < num_rows; i++)
      {
        this_data = M.Row(i).Data();
        for (MatrixIndexT j = 0; j < num_cols; j++)
          this_data[j] = r(i, j);
      }
      })
      .def(py::pickle(
        [](const Matrix<double> &p) { // __getstate__
            /* Return a tuple that fully encodes the state of the object */
             std::ostringstream os;
             bool binary = true;
             p.Write(os, binary);
            return py::make_tuple(
                    py::bytes(os.str()));
        },
        [](py::tuple t) { // __setstate__
            if (t.size() != 1)
                throw std::runtime_error("Invalid state!");

            /* Create a new C++ instance */
            Matrix<double> *p = new Matrix<double>();

            /* Assign any additional state */
            std::istringstream str(t[0].cast<std::string>());
               p->Read(str, true, false);

            return p;
        }
    ));

  py::class_<SubMatrix<double>, MatrixBase<double>>(m, "DoubleSubMatrix")
      .def(py::init([](py::buffer b) {
        py::buffer_info info = b.request();
        if (info.format != py::format_descriptor<double>::format()) {
          KALDI_ERR << "Expected format: "
                    << py::format_descriptor<double>::format() << "\n"
                    << "Current format: " << info.format;
        }
        if (info.ndim != 2) {
          KALDI_ERR << "Expected dim: 2\n"
                    << "Current dim: " << info.ndim;
        }

        // numpy is row major by default, so we use strides[0]
        return new SubMatrix<double>(reinterpret_cast<double*>(info.ptr),
                                    info.shape[0], info.shape[1],
                                    info.strides[0] / sizeof(double));
      }));

  /*py::class_<Matrix<double>, std::unique_ptr<Matrix<double>, py::nodelete>>(
      m, "DoubleMatrix",
      "This bind is only for internal use, e.g. by OnlineCmvnState.")
      .def(py::init<const Matrix<float>&>(), py::arg("src"));
      */
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
      .def("SetZero", &VectorBase<float>::SetZero,
           "Set vector to all zeros.")
      .def("IsZero", &VectorBase<float>::IsZero,
           "Returns true if matrix is all zeros.",
        py::arg("cutoff") = 1.0e-06)
      .def("Set", &VectorBase<float>::Set,
           "Set all members of a vector to a specified value.",
        py::arg("f"))
      .def("SetRandn", &VectorBase<float>::SetRandn,
           "Set vector to random normally-distributed noise.")
      .def("SetRandUniform", &VectorBase<float>::SetRandUniform,
           "Sets to numbers uniformly distributed on (0,1)")
      .def("RandCategorical", &VectorBase<float>::RandCategorical,
           "This function returns a random index into this vector, "
          "chosen with probability proportional to the corresponding "
          "element.  Requires that this->Min() >= 0 and this->Sum() > 0.")
      .def("Floor", &VectorBase<float>::Floor,
           "Applies floor to all elements. Returns number of elements "
        "floored in floored_count if it is non-null.",
        py::arg("v"),
        py::arg("floor_val"),
        py::arg("floored_count") = nullptr)
      .def("Ceiling", &VectorBase<float>::Ceiling,
           "Applies ceiling to all elements. Returns number of elements "
  "changed in ceiled_count if it is non-null.",
        py::arg("v"),
        py::arg("ceil_val"),
        py::arg("ceiled_count") = nullptr)
      .def("Pow", &VectorBase<float>::Pow,
        py::arg("v"),
        py::arg("power"))
      .def("ApplyLog", &VectorBase<float>::ApplyLog,
        "Apply natural log to all elements.  Throw if any element of "
        "the vector is negative (but doesn't complain about zero; the "
        "log will be -infinity")
      .def("ApplyLogAndCopy", &VectorBase<float>::ApplyLogAndCopy,
        "Apply natural log to another vector and put result in *this.",
        py::arg("v"))
      .def("ApplyExp", &VectorBase<float>::ApplyExp,
        "Apply exponential to each value in vector.")
      .def("ApplyAbs", &VectorBase<float>::ApplyAbs,
        "Take absolute value of each of the elements")
      .def("ApplyFloor",
        py::overload_cast<float, MatrixIndexT *>(&VectorBase<float>::ApplyFloor),
        "Applies floor to all elements. Returns number of elements "
  "floored in floored_count if it is non-null.",
        py::arg("floor_val"),
        py::arg("floored_count") = nullptr)
      .def("ApplyCeiling", &VectorBase<float>::ApplyCeiling,
        "Applies ceiling to all elements. Returns number of elements "
        "changed in ceiled_count if it is non-null.",
        py::arg("ceil_val"),
        py::arg("ceiled_count") = nullptr)
      .def("ApplyFloor",
        py::overload_cast<const VectorBase<float> &>(&VectorBase<float>::ApplyFloor),
        "Applies floor to all elements. Returns number of elements floored.",
        py::arg("floor_vec"))
      .def("ApplySoftMax",
        &VectorBase<float>::ApplySoftMax,
        "Apply soft-max to vector and return normalizer (log sum of exponentials). "
        "This is the same as: \\f$ x(i) = exp(x(i)) / \\sum_i exp(x(i)) \\f$")
      .def("ApplyLogSoftMax",
        &VectorBase<float>::ApplyLogSoftMax,
        "Applies log soft-max to vector and returns normalizer (log sum of "
        "exponentials). "
        "This is the same as: \\f$ x(i) = x(i) - log(\\sum_i exp(x(i))) \\f$")
      .def("Tanh",
        &VectorBase<float>::Tanh,
        "Sets each element of *this to the tanh of the corresponding element of \"src\".",
        py::arg("src"))
      .def("Sigmoid",
        &VectorBase<float>::Sigmoid,
        "Sets each element of *this to the sigmoid function of the corresponding "
      "element of \"src\".",
        py::arg("src"))
      .def("ApplyPow",
        &VectorBase<float>::ApplyPow,
        "Take all  elements of vector to a power.",
        py::arg("power"))
      .def("ApplyPowAbs",
        &VectorBase<float>::ApplyPowAbs,
        "Take the absolute value of all elements of a vector to a power. "
        "Include the sign of the input element if include_sign == true. "
        "If power is negative and the input value is zero, the output is set zero.",
        py::arg("power"),
        py::arg("include_sign") = false)
      .def("Norm",
        &VectorBase<float>::Norm,
        "Compute the p-th norm of the vector.",
        py::arg("p"),
      py::call_guard<py::gil_scoped_release>())
      .def("ApproxEqual",
        &VectorBase<float>::ApproxEqual,
        "Returns true if ((*this)-other).Norm(2.0) <= tol * (*this).Norm(2.0).",
        py::arg("other"),
        py::arg("tol") = 0.01)
      .def("AddVec",
        py::overload_cast<const float, const VectorBase<float> &>(&VectorBase<float>::AddVec<float>),
        "Add vector : *this = *this + alpha * rv (with casting between floats and doubles)",
        py::arg("alpha"),
        py::arg("v"),
      py::call_guard<py::gil_scoped_release>())
      .def("CopyFromVec",
        [](
          VectorBase<float> &v,
          const VectorBase<float> &other
        ){
          v.CopyFromVec(other);
        },
        "Copy data from another vector (must match own size).",
        py::arg("other"),
      py::call_guard<py::gil_scoped_release>())
      /*.def("AddVec2",
        py::overload_cast<const float, const VectorBase<float> &>(&VectorBase<float>::AddVec2<float>),
        "Add vector : *this = *this + alpha * rv^2  [element-wise squaring].",
        py::arg("alpha"),
        py::arg("v"))*/
      .def("AddMatVec",
        &VectorBase<float>::AddMatVec,
        "Add matrix times vector : this <-- beta*this + alpha*M*v. Calls BLAS GEMV.",
        py::arg("alpha"),
        py::arg("M"),
        py::arg("trans"),
        py::arg("v"),
        py::arg("beta"),
      py::call_guard<py::gil_scoped_release>())
      .def("AddMatSvec",
        &VectorBase<float>::AddMatSvec,
        "This is as AddMatVec, except optimized for where v contains a lot of zeros.",
        py::arg("alpha"),
        py::arg("M"),
        py::arg("trans"),
        py::arg("v"),
        py::arg("beta"),
      py::call_guard<py::gil_scoped_release>())
      /*.def("AddSpVec",
        &VectorBase<float>::AddSpVec,
        "Add symmetric positive definite matrix times vector: "
        "this <-- beta*this + alpha*M*v.   Calls BLAS SPMV.",
        py::arg("alpha"),
        py::arg("M"),
        py::arg("v"),
        py::arg("beta"))
      .def("AddTpVec",
        &VectorBase<float>::AddTpVec,
        "Add triangular matrix times vector: this <-- beta*this + alpha*M*v. "
        "Works even if rv == *this.",
        py::arg("alpha"),
        py::arg("M"),
        py::arg("trans"),
        py::arg("v"),
        py::arg("beta"))*/
      .def("ReplaceValue",
        &VectorBase<float>::ReplaceValue,
        "Set each element to y = (x == orig ? changed : x).",
        py::arg("orig"),
        py::arg("changed"),
      py::call_guard<py::gil_scoped_release>())
      .def("MulElements",
        (void (VectorBase<float>::*)(const VectorBase<float> &))&VectorBase<float>::MulElements,
        "Multiply element-by-element by another vector.",
        py::arg("v"),
      py::call_guard<py::gil_scoped_release>())
      .def("DivElements",
        (void (VectorBase<float>::*)(const VectorBase<float> &))&VectorBase<float>::DivElements,
        "Divide element-by-element by another vector.",
        py::arg("v"),
      py::call_guard<py::gil_scoped_release>())
      .def("Add",
        &VectorBase<float>::Add,
        "Add a constant to each element of a vector.",
        py::arg("c"),
      py::call_guard<py::gil_scoped_release>())
      .def("AddVecVec",
        &VectorBase<float>::AddVecVec,
        "Add element-by-element product of vectors: this <-- alpha * v .* r + beta*this .",
        py::arg("alpha"),
        py::arg("v"),
        py::arg("r"),
        py::arg("beta"),
      py::call_guard<py::gil_scoped_release>())
      .def("AddVecDivVec",
        &VectorBase<float>::AddVecDivVec,
        "Add element-by-element quotient of two vectors. this <---- alpha*v/r + beta*this",
        py::arg("alpha"),
        py::arg("v"),
        py::arg("r"),
        py::arg("beta"),
      py::call_guard<py::gil_scoped_release>())
      .def("Scale",
        &VectorBase<float>::Scale,
        "Multiplies all elements by this constant.",
        py::arg("alpha"),
      py::call_guard<py::gil_scoped_release>())
      /*.def("MulTp",
        &VectorBase<float>::MulTp,
        "Multiplies this vector by lower-triangular matrix:  *this <-- *this *M",
        py::arg("M"),
        py::arg("trans"))
      .def("Solve",
        &VectorBase<float>::Solve,
        "If trans == kNoTrans, solves M x = b, where b is the value of *this at input "
        "and x is the value of *this at output. "
        "If trans == kTrans, solves M' x = b. "
        "Does not test for M being singular or near-singular, so test it before "
        "calling this routine.",
        py::arg("M"),
        py::arg("trans"))*/
      .def("Max",
        (float (VectorBase<float>::*)()const)&VectorBase<float>::Max,
        "Returns the maximum value of any element, or -infinity for the empty vector.",
      py::call_guard<py::gil_scoped_release>())
      .def("Max",
        (float (VectorBase<float>::*)(MatrixIndexT *)const)&VectorBase<float>::Max,
        "Returns the maximum value of any element, and the associated index. Error if vector is empty.",
        py::arg("index"),
      py::call_guard<py::gil_scoped_release>())
      .def("Min",
        (float (VectorBase<float>::*)()const)&VectorBase<float>::Min,
        "Returns the minimum value of any element, or +infinity for the empty vector.",
      py::call_guard<py::gil_scoped_release>())
      .def("Min",
        (float (VectorBase<float>::*)(MatrixIndexT *)const)&VectorBase<float>::Min,
        "Returns the minimum value of any element, and the associated index. Error if vector is empty.",
        py::arg("index"),
      py::call_guard<py::gil_scoped_release>())
      .def("Sum",
        &VectorBase<float>::Sum,
        "Returns sum of the elements",
      py::call_guard<py::gil_scoped_release>())
      .def("SumLog",
        &VectorBase<float>::SumLog,
        "Returns sum of the logs of the elements.  More efficient than "
  "just taking log of each.  Will return NaN if any elements are "
  "negative.",
      py::call_guard<py::gil_scoped_release>())
      .def("AddRowSumMat",
        &VectorBase<float>::AddRowSumMat,
        "Does *this = alpha * (sum of rows of M) + beta * *this.",
        py::arg("alpha"),
        py::arg("M"),
        py::arg("beta") = 1.0,
      py::call_guard<py::gil_scoped_release>())
      .def("AddColSumMat",
        &VectorBase<float>::AddColSumMat,
        "Does *this = alpha * (sum of columns of M) + beta * *this.",
        py::arg("alpha"),
        py::arg("M"),
        py::arg("beta") = 1.0,
      py::call_guard<py::gil_scoped_release>())
      .def("AddDiagMat2",
        &VectorBase<float>::AddDiagMat2,
        "Add the diagonal of a matrix times itself: "
  "*this = diag(M M^T) +  beta * *this (if trans == kNoTrans), or "
  "*this = diag(M^T M) +  beta * *this (if trans == kTrans).",
        py::arg("alpha"),
        py::arg("M"),
        py::arg("trans") = kNoTrans,
        py::arg("beta") = 1.0,
      py::call_guard<py::gil_scoped_release>())
      .def("AddDiagMatMat",
        &VectorBase<float>::AddDiagMatMat,
        "Add the diagonal of a matrix product: *this = diag(M N), assuming the "
  "\"trans\" arguments are both kNoTrans; for transpose arguments, it behaves "
  "as you would expect.",
        py::arg("alpha"),
        py::arg("M"),
        py::arg("trans"),
        py::arg("N"),
        py::arg("transN"),
        py::arg("beta") = 1.0,
      py::call_guard<py::gil_scoped_release>())
      .def("LogSumExp",
        &VectorBase<float>::LogSumExp,
        "Returns log(sum(exp())) without exp overflow "
        "If prune > 0.0, ignores terms less than the max - prune. "
        "[Note: in future, if prune = 0.0, it will take the max. "
        "For now, use -1 if you don't want it to prune.]",
        py::arg("prune") = -1.0,
      py::call_guard<py::gil_scoped_release>())
      .def("Read",
        &VectorBase<float>::Read,
        "Reads from C++ stream (option to add to existing contents). "
  "Throws exception on failure",
        py::arg("in"),
        py::arg("binary"),
        py::arg("add") = false,
      py::call_guard<py::gil_scoped_release>())
      .def("Write",
        &VectorBase<float>::Write,
        "Writes to C++ stream (option to write in binary).",
        py::arg("Out"),
        py::arg("binary"),
      py::call_guard<py::gil_scoped_release>())
      .def("InvertElements",
        &VectorBase<float>::InvertElements,
        "Invert all elements.",
      py::call_guard<py::gil_scoped_release>())
      .def("SizeInBytes", &VectorBase<float>::SizeInBytes,
           "Returns the size in memory of the vector, in bytes.")
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
           py::arg("in"), py::arg("binary"), py::arg("add") = false,
      py::call_guard<py::gil_scoped_release>())
      .def("Resize", &Vector<float>::Resize,
           "Set vector to a specified size (can be zero). "
          "The value of the new data depends on resize_type: "
          "  -if kSetZero, the new data will be zero "
          "  -if kUndefined, the new data will be undefined "
          "  -if kCopyData, the new data will be the same as the old data in any "
          "     shared positions, and zero elsewhere. "
          "This function takes time proportional to the number of data elements.",
           py::arg("length"), py::arg("resize_type") = kSetZero)
      .def("from_numpy", [](Vector<float>& v, py::array_t<float> x){
        auto r = x.unchecked<1>();
        v.Resize(r.shape(0));
        for (py::size_t i = 0; i < r.shape(0); i++)
          v(i) = r(i);
      })
      .def(py::pickle(
        [](const Vector<float> &p) { // __getstate__
            /* Return a tuple that fully encodes the state of the object */
             std::ostringstream os;
             bool binary = true;
             p.Write(os, binary);
            return py::make_tuple(
                    py::bytes(os.str()));
        },
        [](py::tuple t) { // __setstate__
            if (t.size() != 1)
                throw std::runtime_error("Invalid state!");

            /* Create a new C++ instance */
            Vector<float> *p = new Vector<float>();

            /* Assign any additional state */
            std::istringstream str(t[0].cast<std::string>());
               p->Read(str, true, false);

            return p;
        }
    ));

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

  py::class_<VectorBase<double>,
             std::unique_ptr<VectorBase<double>, py::nodelete>>(
      m, "DoubleVectorBase",
      "Provides a vector abstraction class.\n"
      "This class provides a way to work with vectors in kaldi.\n"
      "It encapsulates basic operations and memory optimizations.")
      .def("Dim", &VectorBase<double>::Dim,
           "Returns the dimension of the vector.")
      .def("SetZero", &VectorBase<double>::SetZero,
           "Set vector to all zeros.",
      py::call_guard<py::gil_scoped_release>())
      .def("IsZero", &VectorBase<double>::IsZero,
           "Returns true if matrix is all zeros.",
        py::arg("cutoff") = 1.0e-06,
      py::call_guard<py::gil_scoped_release>())
      .def("Set", &VectorBase<double>::Set,
           "Set all members of a vector to a specified value.",
        py::arg("f"),
      py::call_guard<py::gil_scoped_release>())
      .def("SetRandn", &VectorBase<double>::SetRandn,
           "Set vector to random normally-distributed noise.",
      py::call_guard<py::gil_scoped_release>())
      .def("SetRandUniform", &VectorBase<double>::SetRandUniform,
           "Sets to numbers uniformly distributed on (0,1)",
      py::call_guard<py::gil_scoped_release>())
      .def("RandCategorical", &VectorBase<double>::RandCategorical,
           "This function returns a random index into this vector, "
          "chosen with probability proportional to the corresponding "
          "element.  Requires that this->Min() >= 0 and this->Sum() > 0.")
      .def("Floor", &VectorBase<double>::Floor,
           "Applies floor to all elements. Returns number of elements "
        "floored in floored_count if it is non-null.",
        py::arg("v"),
        py::arg("floor_val"),
        py::arg("floored_count") = nullptr,
      py::call_guard<py::gil_scoped_release>())
      .def("Ceiling", &VectorBase<double>::Ceiling,
           "Applies ceiling to all elements. Returns number of elements "
  "changed in ceiled_count if it is non-null.",
        py::arg("v"),
        py::arg("ceil_val"),
        py::arg("ceiled_count") = nullptr,
      py::call_guard<py::gil_scoped_release>())
      .def("Pow", &VectorBase<double>::Pow,
        py::arg("v"),
        py::arg("power"),
      py::call_guard<py::gil_scoped_release>())
      .def("ApplyLog", &VectorBase<double>::ApplyLog,
        "Apply natural log to all elements.  Throw if any element of "
        "the vector is negative (but doesn't complain about zero; the "
        "log will be -infinity",
      py::call_guard<py::gil_scoped_release>())
      .def("ApplyLogAndCopy", &VectorBase<double>::ApplyLogAndCopy,
        "Apply natural log to another vector and put result in *this.",
        py::arg("v"),
      py::call_guard<py::gil_scoped_release>())
      .def("ApplyExp", &VectorBase<double>::ApplyExp,
        "Apply exponential to each value in vector.",
      py::call_guard<py::gil_scoped_release>())
      .def("ApplyAbs", &VectorBase<double>::ApplyAbs,
        "Take absolute value of each of the elements",
      py::call_guard<py::gil_scoped_release>())
      .def("ApplyFloor",
        py::overload_cast<double, MatrixIndexT *>(&VectorBase<double>::ApplyFloor),
        "Applies floor to all elements. Returns number of elements "
  "floored in floored_count if it is non-null.",
        py::arg("floor_val"),
        py::arg("floored_count") = nullptr,
      py::call_guard<py::gil_scoped_release>())
      .def("ApplyCeiling", &VectorBase<double>::ApplyCeiling,
        "Applies ceiling to all elements. Returns number of elements "
        "changed in ceiled_count if it is non-null.",
        py::arg("ceil_val"),
        py::arg("ceiled_count") = nullptr,
      py::call_guard<py::gil_scoped_release>())
      .def("ApplyFloor",
        py::overload_cast<const VectorBase<double> &>(&VectorBase<double>::ApplyFloor),
        "Applies floor to all elements. Returns number of elements floored.",
        py::arg("floor_vec"),
      py::call_guard<py::gil_scoped_release>())
      .def("ApplySoftMax",
        &VectorBase<double>::ApplySoftMax,
        "Apply soft-max to vector and return normalizer (log sum of exponentials). "
        "This is the same as: \\f$ x(i) = exp(x(i)) / \\sum_i exp(x(i)) \\f$",
      py::call_guard<py::gil_scoped_release>())
      .def("ApplyLogSoftMax",
        &VectorBase<double>::ApplyLogSoftMax,
        "Applies log soft-max to vector and returns normalizer (log sum of "
        "exponentials). "
        "This is the same as: \\f$ x(i) = x(i) - log(\\sum_i exp(x(i))) \\f$",
      py::call_guard<py::gil_scoped_release>())
      .def("Tanh",
        &VectorBase<double>::Tanh,
        "Sets each element of *this to the tanh of the corresponding element of \"src\".",
        py::arg("src"),
      py::call_guard<py::gil_scoped_release>())
      .def("Sigmoid",
        &VectorBase<double>::Sigmoid,
        "Sets each element of *this to the sigmoid function of the corresponding "
      "element of \"src\".",
        py::arg("src"),
      py::call_guard<py::gil_scoped_release>())
      .def("ApplyPow",
        &VectorBase<double>::ApplyPow,
        "Take all  elements of vector to a power.",
        py::arg("power"),
      py::call_guard<py::gil_scoped_release>())
      .def("ApplyPowAbs",
        &VectorBase<double>::ApplyPowAbs,
        "Take the absolute value of all elements of a vector to a power. "
        "Include the sign of the input element if include_sign == true. "
        "If power is negative and the input value is zero, the output is set zero.",
        py::arg("power"),
        py::arg("include_sign") = false,
      py::call_guard<py::gil_scoped_release>())
      .def("Norm",
        &VectorBase<double>::Norm,
        "Compute the p-th norm of the vector.",
        py::arg("p"),
      py::call_guard<py::gil_scoped_release>())
      .def("ApproxEqual",
        &VectorBase<double>::ApproxEqual,
        "Returns true if ((*this)-other).Norm(2.0) <= tol * (*this).Norm(2.0).",
        py::arg("other"),
        py::arg("tol") = 0.01)
      .def("AddVec",
        py::overload_cast<const double, const VectorBase<double> &>(&VectorBase<double>::AddVec<double>),
        "Add vector : *this = *this + alpha * rv (with casting between floats and doubles)",
        py::arg("alpha"),
        py::arg("v"),
      py::call_guard<py::gil_scoped_release>())
      .def("CopyFromVec",
        [](
          VectorBase<double> &v,
          const VectorBase<double> &other
        ){
          v.CopyFromVec(other);
        },
        "Copy data from another vector (must match own size).",
        py::arg("other"),
      py::call_guard<py::gil_scoped_release>())
      /*.def("AddVec2",
        py::overload_cast<const double, const VectorBase<double> &>(&VectorBase<double>::AddVec2<double>),
        "Add vector : *this = *this + alpha * rv^2  [element-wise squaring].",
        py::arg("alpha"),
        py::arg("v"))*/
      .def("AddMatVec",
        &VectorBase<double>::AddMatVec,
        "Add matrix times vector : this <-- beta*this + alpha*M*v. Calls BLAS GEMV.",
        py::arg("alpha"),
        py::arg("M"),
        py::arg("trans"),
        py::arg("v"),
        py::arg("beta"),
      py::call_guard<py::gil_scoped_release>())
      .def("AddMatSvec",
        &VectorBase<double>::AddMatSvec,
        "This is as AddMatVec, except optimized for where v contains a lot of zeros.",
        py::arg("alpha"),
        py::arg("M"),
        py::arg("trans"),
        py::arg("v"),
        py::arg("beta"),
      py::call_guard<py::gil_scoped_release>())
      /*.def("AddSpVec",
        &VectorBase<double>::AddSpVec,
        "Add symmetric positive definite matrix times vector: "
        "this <-- beta*this + alpha*M*v.   Calls BLAS SPMV.",
        py::arg("alpha"),
        py::arg("M"),
        py::arg("v"),
        py::arg("beta"))
      .def("AddTpVec",
        &VectorBase<double>::AddTpVec,
        "Add triangular matrix times vector: this <-- beta*this + alpha*M*v. "
        "Works even if rv == *this.",
        py::arg("alpha"),
        py::arg("M"),
        py::arg("trans"),
        py::arg("v"),
        py::arg("beta"))*/
      .def("ReplaceValue",
        &VectorBase<double>::ReplaceValue,
        "Set each element to y = (x == orig ? changed : x).",
        py::arg("orig"),
        py::arg("changed"),
      py::call_guard<py::gil_scoped_release>())
      .def("MulElements",
        (void (VectorBase<double>::*)(const VectorBase<double> &))&VectorBase<double>::MulElements,
        "Multiply element-by-element by another vector.",
        py::arg("v"),
      py::call_guard<py::gil_scoped_release>())
      .def("DivElements",
        (void (VectorBase<double>::*)(const VectorBase<double> &))&VectorBase<double>::DivElements,
        "Divide element-by-element by another vector.",
        py::arg("v"),
      py::call_guard<py::gil_scoped_release>())
      .def("Add",
        &VectorBase<double>::Add,
        "Add a constant to each element of a vector.",
        py::arg("c"),
      py::call_guard<py::gil_scoped_release>())
      .def("AddVecVec",
        &VectorBase<double>::AddVecVec,
        "Add element-by-element product of vectors: this <-- alpha * v .* r + beta*this .",
        py::arg("alpha"),
        py::arg("v"),
        py::arg("r"),
        py::arg("beta"),
      py::call_guard<py::gil_scoped_release>())
      .def("AddVecDivVec",
        &VectorBase<double>::AddVecDivVec,
        "Add element-by-element quotient of two vectors. this <---- alpha*v/r + beta*this",
        py::arg("alpha"),
        py::arg("v"),
        py::arg("r"),
        py::arg("beta"),
      py::call_guard<py::gil_scoped_release>())
      .def("Scale",
        &VectorBase<double>::Scale,
        "Multiplies all elements by this constant.",
        py::arg("alpha"),
      py::call_guard<py::gil_scoped_release>())
      /*.def("MulTp",
        &VectorBase<double>::MulTp,
        "Multiplies this vector by lower-triangular matrix:  *this <-- *this *M",
        py::arg("M"),
        py::arg("trans"))
      .def("Solve",
        &VectorBase<double>::Solve,
        "If trans == kNoTrans, solves M x = b, where b is the value of *this at input "
        "and x is the value of *this at output. "
        "If trans == kTrans, solves M' x = b. "
        "Does not test for M being singular or near-singular, so test it before "
        "calling this routine.",
        py::arg("M"),
        py::arg("trans"))*/
      .def("Max",
        (double (VectorBase<double>::*)()const)&VectorBase<double>::Max,
        "Returns the maximum value of any element, or -infinity for the empty vector.",
      py::call_guard<py::gil_scoped_release>())
      .def("Max",
        (double (VectorBase<double>::*)(MatrixIndexT *)const)&VectorBase<double>::Max,
        "Returns the maximum value of any element, and the associated index. Error if vector is empty.",
        py::arg("index"),
      py::call_guard<py::gil_scoped_release>())
      .def("Min",
        (double (VectorBase<double>::*)()const)&VectorBase<double>::Min,
        "Returns the minimum value of any element, or +infinity for the empty vector.",
      py::call_guard<py::gil_scoped_release>())
      .def("Min",
        (double (VectorBase<double>::*)(MatrixIndexT *)const)&VectorBase<double>::Min,
        "Returns the minimum value of any element, and the associated index. Error if vector is empty.",
        py::arg("index"),
      py::call_guard<py::gil_scoped_release>())
      .def("Sum",
        &VectorBase<double>::Sum,
        "Returns sum of the elements",
      py::call_guard<py::gil_scoped_release>())
      .def("SumLog",
        &VectorBase<double>::SumLog,
        "Returns sum of the logs of the elements.  More efficient than "
  "just taking log of each.  Will return NaN if any elements are "
  "negative.",
      py::call_guard<py::gil_scoped_release>())
      .def("AddRowSumMat",
        &VectorBase<double>::AddRowSumMat,
        "Does *this = alpha * (sum of rows of M) + beta * *this.",
        py::arg("alpha"),
        py::arg("M"),
        py::arg("beta") = 1.0,
      py::call_guard<py::gil_scoped_release>())
      .def("AddColSumMat",
        &VectorBase<double>::AddColSumMat,
        "Does *this = alpha * (sum of columns of M) + beta * *this.",
        py::arg("alpha"),
        py::arg("M"),
        py::arg("beta") = 1.0,
      py::call_guard<py::gil_scoped_release>())
      .def("AddDiagMat2",
        &VectorBase<double>::AddDiagMat2,
        "Add the diagonal of a matrix times itself: "
  "*this = diag(M M^T) +  beta * *this (if trans == kNoTrans), or "
  "*this = diag(M^T M) +  beta * *this (if trans == kTrans).",
        py::arg("alpha"),
        py::arg("M"),
        py::arg("trans") = kNoTrans,
        py::arg("beta") = 1.0,
      py::call_guard<py::gil_scoped_release>())
      .def("AddDiagMatMat",
        &VectorBase<double>::AddDiagMatMat,
        "Add the diagonal of a matrix product: *this = diag(M N), assuming the "
  "\"trans\" arguments are both kNoTrans; for transpose arguments, it behaves "
  "as you would expect.",
        py::arg("alpha"),
        py::arg("M"),
        py::arg("trans"),
        py::arg("N"),
        py::arg("transN"),
        py::arg("beta") = 1.0,
      py::call_guard<py::gil_scoped_release>())
      .def("LogSumExp",
        &VectorBase<double>::LogSumExp,
        "Returns log(sum(exp())) without exp overflow "
        "If prune > 0.0, ignores terms less than the max - prune. "
        "[Note: in future, if prune = 0.0, it will take the max. "
        "For now, use -1 if you don't want it to prune.]",
        py::arg("prune") = -1.0,
      py::call_guard<py::gil_scoped_release>())
      .def("Read",
        &VectorBase<double>::Read,
        "Reads from C++ stream (option to add to existing contents). "
  "Throws exception on failure",
        py::arg("in"),
        py::arg("binary"),
        py::arg("add") = false,
      py::call_guard<py::gil_scoped_release>())
      .def("Write",
        &VectorBase<double>::Write,
        "Writes to C++ stream (option to write in binary).",
        py::arg("Out"),
        py::arg("binary"),
      py::call_guard<py::gil_scoped_release>())
      .def("InvertElements",
        &VectorBase<double>::InvertElements,
        "Invert all elements.",
      py::call_guard<py::gil_scoped_release>())
      .def("SizeInBytes", &VectorBase<double>::SizeInBytes,
           "Returns the size in memory of the vector, in bytes.")
      .def("__repr__",
           [](const VectorBase<double>& v) -> std::string {
             std::ostringstream str;
             v.Write(str, false);
             return str.str();
           })
      .def("__getitem__",
           [](const VectorBase<double>& v, int i) { return v(i); })
      .def("__setitem__",
           [](VectorBase<double>& v, int i, double val) { v(i) = val; })
      .def("numpy", [](py::object obj) {
        auto* v = obj.cast<VectorBase<double>*>();
        return py::array_t<double>(
            {v->Dim()},       // shape
            {sizeof(double)},  // stride in bytes
            v->Data(),        // ptr
            obj);  // it will increase the reference count of **this** vector
      });

  py::class_<Vector<double>, VectorBase<double>>(m, "DoubleVector",
                                               py::buffer_protocol())
      .def_buffer([](const Vector<double>& v) -> py::buffer_info {
        return py::buffer_info((void*)v.Data(), sizeof(double),
                               py::format_descriptor<double>::format(),
                               1,  // num-axes
                               {v.Dim()},
                               {sizeof(double)});  // strides (in chars)
      })
      .def(py::init<>())
      .def(py::init<const MatrixIndexT, MatrixResizeType>(), py::arg("size"),
           py::arg("resize_type") = kSetZero)
      .def(py::init<const VectorBase<double>&>(), py::arg("v"),
           "Copy-constructor from base-class, needed to copy from SubVector.")
      .def("Read", &Vector<double>::Read,
           "Reads from C++ stream (option to add to existing contents).Throws "
           "exception on failure",
           py::arg("in"), py::arg("binary"), py::arg("add") = false,
      py::call_guard<py::gil_scoped_release>())
      .def("Resize", &Vector<double>::Resize,
           "Set vector to a specified size (can be zero). "
          "The value of the new data depends on resize_type: "
          "  -if kSetZero, the new data will be zero "
          "  -if kUndefined, the new data will be undefined "
          "  -if kCopyData, the new data will be the same as the old data in any "
          "     shared positions, and zero elsewhere. "
          "This function takes time proportional to the number of data elements.",
           py::arg("length"), py::arg("resize_type") = kSetZero)
      .def("from_numpy", [](Vector<double>& v, py::array_t<double> x){
        auto r = x.unchecked<1>();
        v.Resize(r.shape(0));
        for (py::size_t i = 0; i < r.shape(0); i++)
          v(i) = r(i);
      })
      .def("to_bytes",
      [](const Vector<double> &p) {
             std::ostringstream os;
             bool binary = false;
             p.Write(os, binary);
            return py::bytes(os.str());
        })
      .def("from_bytes",
      [](Vector<double> &p, const std::string &in, bool add = false) {
             std::istringstream str(in);
               p.Read(str, false, add);
        },
              py::arg("in"),
              py::arg("add") = false)
      .def(py::pickle(
        [](const Vector<double> &p) { // __getstate__
            /* Return a tuple that fully encodes the state of the object */
             std::ostringstream os;
             bool binary = true;
             p.Write(os, binary);
            return py::make_tuple(
                    py::bytes(os.str()));
        },
        [](py::tuple t) { // __setstate__
            if (t.size() != 1)
                throw std::runtime_error("Invalid state!");

            /* Create a new C++ instance */
            Vector<double> *p = new Vector<double>();

            /* Assign any additional state */
            std::istringstream str(t[0].cast<std::string>());
               p->Read(str, true, false);

            return p;
        }
    ));

  py::class_<SubVector<double>, VectorBase<double>>(m, "DoubleSubVector")
      .def(py::init([](py::buffer b) {
        py::buffer_info info = b.request();
        if (info.format != py::format_descriptor<double>::format()) {
          KALDI_ERR << "Expected format: "
                    << py::format_descriptor<double>::format() << "\n"
                    << "Current format: " << info.format;
        }
        if (info.ndim != 1) {
          KALDI_ERR << "Expected dim: 1\n"
                    << "Current dim: " << info.ndim;
        }
        return new SubVector<double>(reinterpret_cast<double*>(info.ptr),
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
