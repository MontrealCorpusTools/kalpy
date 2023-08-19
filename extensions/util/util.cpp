
#include "util/pybind_util.h"
#include <pybind11/iostream.h>
#include "util/kaldi-table.h"
#include "util/kaldi-io.h"
#include "util/kaldi-holder-inl.h"
#include "util/parse-options.h"
#include "util/const-integer-set.h"
#include "itf/options-itf.h"

#include "hmm/transition-model.h"
#include "tree/context-dep.h"
#include "matrix/kaldi-matrix.h"
#include "lm/const-arpa-lm.h"

#include "base/kaldi-error.h"

using namespace kaldi;


namespace {

void ignore_logs(const LogMessageEnvelope &envelope,
                           const char *message){
                           }
LogHandler kalpy_log_handler = &ignore_logs;
LogHandler old_logger = SetLogHandler(*kalpy_log_handler);

template <typename Type>
struct ArgName;

#define DEFINE_ARG_NAME(type, name)             \
  template <>                                   \
  struct ArgName<type> {                        \
    static constexpr const char* value = #name; \
  }

DEFINE_ARG_NAME(bool, BoolArg);
DEFINE_ARG_NAME(int32, Int32Arg);
DEFINE_ARG_NAME(uint32, UInt32Arg);
DEFINE_ARG_NAME(float, FloatArg);
DEFINE_ARG_NAME(double, DoubleArg);
DEFINE_ARG_NAME(std::string, StringArg);

#undef DEFINE_ARG_NAME

template <typename Type>
struct Arg {
  Type value{};
  Arg() = default;
  Arg(const Type& v) : value(v) {}
};

template <typename Type, typename Opt>
void pybind_arg(py::module& m, Opt& opt) {
  using PyClass = Arg<Type>;
  py::class_<PyClass>(m, ArgName<Type>::value)
      .def(py::init<>())
      .def(py::init<const Type&>(), py::arg("v"))
      .def_readwrite("value", &PyClass::value)
      .def("__str__", [](const PyClass& arg) {
        std::ostringstream os;
        os << arg.value;
        return os.str();
      });

  opt.def("Register",
          [](typename Opt::type* o, const std::string& name, PyClass* arg,
             const std::string& doc) { o->Register(name, &arg->value, doc); },
          py::arg("name"), py::arg("arg"), py::arg("doc"));
}

}  // namespace

void init_util(py::module &_m) {
  py::module m = _m.def_submodule("util", "util pybind for Kaldi");
  pybind_basic_vector_holder<int32>(m, "IntVectorHolder");
  py::class_<std::istream>(m, "istream");
  {
    using PyClass = Input;
    py::class_<PyClass>(m, "Input")
        .def(py::init<>())
        .def(py::init<const std::string &, bool *>())
        .def("Open",
             [](PyClass* ki, const std::string& rxfilename,
                bool read_header = false) -> std::vector<bool> {
               std::vector<bool> result(1, false);
               if (read_header) {
                 result.resize(2, false);
                 bool tmp;
                 result[0] = ki->Open(rxfilename, &tmp);
                 result[1] = tmp;
               } else {
                 result[0] = ki->Open(rxfilename);
               }
               return result;
             },
             "Open the stream for reading. "
             "Return a vector containing one bool or two depending on "
             "whether `read_header` is false or true."
             "\n",
             "(1) If `read_header` is true, it returns [opened, binary], where "
             "`opened` is true if the stream was opened successfully, false "
             "otherwise;\n"
             "`binary` is true if the stream was opened **and** in binary "
             "format\n"
             "\n"
             "(2) If `read_header` is false, it returns [opened], where "
             "`opened` is true if the stream was opened successfully, false "
             "otherwise",
             py::arg("rxfilename"), py::arg("read_header") = false)
        // the constructor and `Open` method both require a `bool*` argument
        // but pybind11 does not support passing a pointer to a primitive
        // type, only pointer to customized type is allowed.
        //
        // For more information, please refer to
        // https://github.com/pybind/pybind11/pull/1760/commits/1d8caa5fbd0903cece06ae646447fff9b4aa33c0
        // https://github.com/pybind/pybind11/pull/1760
        //
        // Was it a `bool*`, would it always be non-NULL in C++!
        //
        // Therefore, we wrap the `Open` method and do NOT wrap the
        // `constructor` with `bool*` arguments
        .def("IsOpen", &PyClass::IsOpen,
             "Return true if currently open for reading and Stream() will "
             "succeed.  Does not guarantee that the stream is good.")
        .def("Close", &PyClass::Close,
             "It is never necessary or helpful to call Close, except if you "
             "are concerned about to many filehandles being open. Close does "
             "not throw. It returns the exit code as int32 in the case of a "
             "pipe [kPipeInput], and always zero otherwise.")
        .def("Stream", &PyClass::Stream,
             "Returns the underlying stream. Throws if !IsOpen()",
             py::return_value_policy::reference);
  }

  py::class_<std::ostream>(m, "ostream");
  {
    using PyClass = Output;
    py::class_<PyClass>(m, "Output")
        .def(py::init<>())
        .def(py::init<const std::string&, bool, bool>(),
             "The normal constructor, provided for convenience. Equivalent to "
             "calling with default constructor then Open() with these "
             "arguments.",
             py::arg("filename"), py::arg("binary"),
             py::arg("write_header") = true)
        .def("Open", &PyClass::Open,
             "This opens the stream, with the given mode (binary or text).  It "
             "returns true on success and false on failure.  However, it will "
             "throw if something was already open and could not be closed (to "
             "avoid this, call Close() first.  if write_header == true and "
             "binary == true, it writes the Kaldi binary-mode header ('\0' "
             "then 'B').  You may call Open even if it is already open; it "
             "will close the existing stream and reopen (however if closing "
             "the old stream failed it will throw).",
             py::arg("wxfilename"), py::arg("binary"), py::arg("write_header"))
        .def("IsOpen", &PyClass::IsOpen,
             "return true if we have an open "
             "stream.  Does not imply stream is "
             "good for writing.")
        .def("Stream", &PyClass::Stream,
             "will throw if not open; else returns stream.",
             py::return_value_policy::reference)
        .def("Close", &PyClass::Close,
             "Close closes the stream. Calling Close is never necessary unless "
             "you want to avoid exceptions being thrown.  There are times when "
             "calling Close will hurt efficiency (basically, when using "
             "offsets into files, and using the same Input object), but most "
             "of the time the user won't be doing this directly, it will be "
             "done in kaldi-table.{h, cc}, so you don't have to worry about "
             "it.");
  }
  {
    using PyClass = ConstIntegerSet<int32>;
    py::class_<PyClass>(m, "ConstIntegerSet")
        .def(py::init<>())
        .def(py::init<const std::vector<int32> &>(),
             py::arg("input"))
        .def(py::init<const std::set<int32> &>(),
             py::arg("input"))
        .def(py::init<const ConstIntegerSet<int32> &>(),
             py::arg("other"))
        .def("count", &PyClass::count,
             py::arg("i"))
        .def("begin", &PyClass::begin)
        .def("end", &PyClass::end)
        .def("size", &PyClass::size)
        .def("empty", &PyClass::empty)
        .def("Write", &PyClass::Write,
             py::arg("os"),
             py::arg("binary"),
      py::call_guard<py::gil_scoped_release>())
        .def("Read", &PyClass::Read,
             py::arg("is"),
             py::arg("binary"),
      py::call_guard<py::gil_scoped_release>());
  }


  pybind_sequential_table_reader<KaldiObjectHolder<Matrix<float>>>(
      m, "SequentialBaseFloatMatrixReader");

  pybind_sequential_table_reader<KaldiObjectHolder<Matrix<double>>>(
      m, "SequentialBaseDoubleMatrixReader");

  pybind_random_access_table_reader<KaldiObjectHolder<Matrix<float>>>(
      m, "RandomAccessBaseFloatMatrixReader");

  pybind_random_access_table_reader<KaldiObjectHolder<Matrix<double>>>(
      m, "RandomAccessBaseDoubleMatrixReader");

  pybind_table_writer<KaldiObjectHolder<Matrix<float>>>(
      m, "BaseFloatMatrixWriter");

  pybind_table_writer<KaldiObjectHolder<Matrix<double>>>(
      m, "BaseDoubleMatrixWriter");

  pybind_sequential_table_reader<KaldiObjectHolder<Vector<float>>>(
      m, "SequentialBaseFloatVectorReader");

  pybind_sequential_table_reader<KaldiObjectHolder<Vector<double>>>(
      m, "SequentialBaseDoubleVectorReader");

  pybind_random_access_table_reader<KaldiObjectHolder<Vector<float>>>(
      m, "RandomAccessBaseFloatVectorReader");

  pybind_random_access_table_reader<KaldiObjectHolder<Vector<double>>>(
      m, "RandomAccessBaseDoubleVectorReader");

  pybind_table_writer<KaldiObjectHolder<Vector<float>>>(
      m, "BaseFloatVectorWriter");

  pybind_table_writer<KaldiObjectHolder<Vector<double>>>(
      m, "BaseDoubleVectorWriter");

  pybind_table_writer<KaldiObjectHolder<CompressedMatrix>>(
      m, "CompressedMatrixWriter");

  pybind_sequential_table_reader<BasicVectorHolder<int32>>(
      m, "SequentialInt32VectorReader");

  pybind_random_access_table_reader<BasicVectorHolder<int32>>(
      m, "RandomAccessInt32VectorReader");

  pybind_table_writer<BasicVectorHolder<int32>>(m, "Int32VectorWriter");

  pybind_sequential_table_reader<BasicVectorVectorHolder<int32>>(
      m, "SequentialInt32VectorVectorReader");

  pybind_random_access_table_reader<BasicVectorVectorHolder<int32>>(
      m, "RandomAccessInt32VectorVectorReader");

  pybind_table_writer<BasicVectorVectorHolder<int32>>(m, "Int32VectorVectorWriter");

  pybind_sequential_table_reader<BasicHolder<int32>>(
      m, "SequentialInt32Reader");

  pybind_random_access_table_reader<BasicHolder<int32>>(
      m, "RandomAccessInt32Reader");

  pybind_table_writer<BasicHolder<int32>>(m, "Int32Writer");

    pybind_read_kaldi_object<TransitionModel>(m);
    pybind_read_kaldi_object<ContextDependency>(m);
    pybind_read_kaldi_object<MatrixBase<float>>(m);
    pybind_read_kaldi_object<MatrixBase<double>>(m);
    pybind_read_kaldi_object<ConstArpaLm>(m);

}
