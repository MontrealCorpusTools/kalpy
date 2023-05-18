
#include "cudamatrix/pybind_cudamatrix.h"

using namespace kaldi;

void init_cudamatrix(py::module &_m) {
  py::module m = _m.def_submodule("cudamatrix", "cudamatrix pybind for Kaldi");
    m.def("SelectGpuDevice",
        [](int device_id) {
#if HAVE_CUDA == 1
          CuDevice::Instantiate().SelectGpuDevice(device_id);
#else
          KALDI_LOG << "Kaldi is NOT compiled with GPU! Ignore it.";
#endif
        },
        py::arg("device_id"));

  m.def("SelectGpuId",
        [](const std::string& use_gpu) {
#if HAVE_CUDA == 1
          CuDevice::Instantiate().SelectGpuId(use_gpu);
#else
          KALDI_LOG << "Kaldi is NOT compiled with GPU! Ignore it.";
#endif
        },
        py::arg("use_gpu"));

  m.def("CuDeviceAllowMultithreading", []() {
#if HAVE_CUDA == 1
    CuDevice::Instantiate().AllowMultithreading();
#else
    KALDI_LOG << "Kaldi is NOT compiled with GPU! Ignore it.";
#endif
  });

  m.def("CudaCompiled",
        []() -> bool {
#if HAVE_CUDA == 1
          return true;
#else
          return false;
#endif
        },
        "true if kaldi is compiled with GPU support; false otherwise");

  {
    using PyClass = CuMatrixBase<float>;
    py::class_<PyClass, std::unique_ptr<PyClass, py::nodelete>>(
        m, "FloatCuMatrixBase", "Matrix for CUDA computing")
        .def("NumRows", &PyClass::NumRows, "Return number of rows")
        .def("NumCols", &PyClass::NumCols, "Return number of columns")
        .def("Stride", &PyClass::Stride, "Return stride")
        .def("ApplyExp", &PyClass::ApplyExp)
        .def("SetZero", &PyClass::SetZero)
        .def("Set", &PyClass::Set, py::arg("value"))
        .def("Add", &PyClass::Add, py::arg("value"))
        .def("Scale", &PyClass::Scale, py::arg("value"))
        .def("__getitem__",
             [](const PyClass& m, std::pair<ssize_t, ssize_t> i) {
               return m(i.first, i.second);
             });
  }

  {
    using PyClass = CuMatrix<float>;
    py::class_<PyClass, CuMatrixBase<float>>(m, "FloatCuMatrix")
        .def(py::init<>())
        .def(py::init<MatrixIndexT, MatrixIndexT, MatrixResizeType,
                      MatrixStrideType>(),
             py::arg("rows"), py::arg("cols"),
             py::arg("resize_type") = MatrixResizeType::kSetZero,
             py::arg("MatrixStrideType") = MatrixStrideType::kDefaultStride)
        .def(py::init<const MatrixBase<float>&, MatrixTransposeType>(),
             py::arg("other"), py::arg("trans") = kNoTrans);
  }
  {
    using PyClass = CuSubMatrix<float>;
    py::class_<PyClass, CuMatrixBase<float>>(m, "FloatCuSubMatrix");
  }

  {
    using PyClass = CuVectorBase<float>;
    py::class_<PyClass, std::unique_ptr<PyClass, py::nodelete>>(
        m, "FloatCuVectorBase", "Vector for CUDA computing")
        .def("Dim", &PyClass::Dim, "Dimensions")
        .def("SetZero", &PyClass::SetZero)
        .def("Set", &PyClass::Set, py::arg("value"))
        .def("Add", &PyClass::Add, py::arg("value"))
        .def("Scale", &PyClass::Scale, py::arg("value"))
        .def("__getitem__", [](const PyClass& v, int i) { return v(i); });
  }
  {
    using PyClass = CuVector<float>;
    py::class_<PyClass, CuVectorBase<float>>(m, "FloatCuVector")
        .def(py::init<>())
        .def(py::init<MatrixIndexT, MatrixResizeType>(), py::arg("dim"),
             py::arg("MatrixResizeType") = kSetZero)
        .def(py::init<const VectorBase<float>&>(), py::arg("v"));
  }
  {
    using PyClass = CuSubVector<float>;
    py::class_<PyClass, CuVectorBase<float>>(m, "FloatCuSubVector");
  }
}
