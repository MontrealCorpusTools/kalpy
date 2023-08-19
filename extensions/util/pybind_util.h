
#ifndef KALPY_PYBIND_UTIL_H_
#define KALPY_PYBIND_UTIL_H_

#include "pybind/kaldi_pybind.h"

#include "util/kaldi-table.h"
#include "util/kaldi-io.h"
#include "util/kaldi-holder-inl.h"
#include "util/parse-options.h"

using namespace kaldi;

template <class Holder>
void pybind_sequential_table_reader(py::module& m,
                                    const std::string& class_name,
                                    const std::string& class_help_doc = "") {
  using PyClass = SequentialTableReader<Holder>;
  py::class_<PyClass>(m, class_name.c_str(), class_help_doc.c_str())
      .def(py::init<>())
      .def(py::init<const std::string&>(),
           "This constructor equivalent to default constructor + 'open', but "
           "throws on error.",
           py::arg("rspecifier"))
      .def("Open", &PyClass::Open,
           "Opens the table.  Returns exit status; but does throw if "
           "previously open stream was in error state.  You can call Close to "
           "prevent this; anyway, calling Open more than once is not usually "
           "needed.",
           py::arg("rspecifier"))
      .def("Done", &PyClass::Done,
           "Returns true if we're done.  It will also return true if there's "
           "some kind of error and we can't read any more; in this case, you "
           "can detect the error by calling Close and checking the return "
           "status; otherwise the destructor will throw.")
      .def("Key", &PyClass::Key,
           "Only valid to call Key() if Done() returned false.",
      py::call_guard<py::gil_scoped_release>())
      .def("FreeCurrent", &PyClass::FreeCurrent,
           "FreeCurrent() is provided as an optimization to save memory, for "
           "large objects.  It instructs the class to deallocate the current "
           "value. The reference Value() will be invalidated by this.")
      .def("Value", &PyClass::Value,
           "Return reference to the current value.  It's only valid to call "
           "this if Done() returned false.  The reference is valid till next "
           "call to this object.  It will throw if you are reading an scp "
           "file, did not specify the 'permissive' (p) option and the file "
           "cannot be read.  [The permissive option makes it behave as if that "
           "key does not even exist, if the corresponding file cannot be "
           "read.]  You probably wouldn't want to catch this exception; the "
           "user can just specify the p option in the rspecifier. We make this "
           "non-const to enable things like shallow swap on the held object in "
           "situations where this would avoid making a redundant copy.",
           py::return_value_policy::reference,
      py::call_guard<py::gil_scoped_release>())
      .def("Next", &PyClass::Next,
           "Next goes to the next key.  It will not throw; any error will "
           "result in Done() returning true, and then the destructor will "
           "throw unless you call Close().",
      py::call_guard<py::gil_scoped_release>())
      .def("IsOpen", &PyClass::IsOpen,
           "Returns true if table is open for reading (does not imply stream "
           "is in good state).")
      .def("Close", &PyClass::Close,
           "Close() will return false (failure) if Done() became true because "
           "of an error/ condition rather than because we are really done "
           "[e.g. because of an error or early termination in the archive]. If "
           "there is an error and you don't call Close(), the destructor will "
           "fail. Close()");
}

template <class Holder>
void pybind_random_access_table_reader(py::module& m,
                                       const std::string& class_name,
                                       const std::string& class_help_doc = "") {
  using PyClass = RandomAccessTableReader<Holder>;
  py::class_<PyClass>(m, class_name.c_str(), class_help_doc.c_str())
      .def(py::init<>())
      .def(py::init<const std::string&>(),
           "This constructor equivalent to default constructor + 'open', but "
           "throws on error.",
           py::arg("rspecifier"))
      .def("Open", &PyClass::Open, "Opens the table.", py::arg("rspecifier"))
      .def("IsOpen", &PyClass::IsOpen, "Returns true if table is open")
      .def("Close", &PyClass::Close,
           "Close() will close the table [throws if it was not open], and "
           "returns true on success (false if we were reading an archive and "
           "we discovered an error in the archive).")
      .def("HasKey", &PyClass::HasKey,
           "Says if it has this key. If you are using the 'permissive' (p) "
           "read option, it will return false for keys whose corresponding "
           "entry in the scp file cannot be read.",
           py::arg("key"),
      py::call_guard<py::gil_scoped_release>())
      .def("Value", &PyClass::Value,
           "Value() may throw if you are reading an scp file, you do not have "
           "the ' permissive'  (p) option, and an entry in the scp file cannot "
           "be read. Typically you won't want to catch this error.",
           py::return_value_policy::reference,
      py::call_guard<py::gil_scoped_release>());
}

template <class Holder>
void pybind_table_writer(py::module& m, const std::string& class_name,
                         const std::string& class_help_doc = "") {
  using PyClass = TableWriter<Holder>;
  py::class_<PyClass>(m, class_name.c_str(), class_help_doc.c_str())
      .def(py::init<>())
      .def(py::init<const std::string&>(),
           "This constructor equivalent to default constructor + 'open', but "
           "throws on error.",
           py::arg("wspecifier"))
      .def("Open", &PyClass::Open,
           "Opens the table.  See docs for wspecifier above. If it returns "
           "true, it is open.",
           py::arg("wspecifier"))
      .def("IsOpen", &PyClass::IsOpen, "Returns true if open for writing.")
      .def("Write", &PyClass::Write,
           "Write the object. Throws KaldiFatalError on error via the "
           "KALDI_ERR macro.",
           py::arg("key"), py::arg("value"),
      py::call_guard<py::gil_scoped_release>())
      .def("Flush", &PyClass::Flush,
           "Flush will flush any archive; it does not return error status or "
           "throw, any errors will be reported on the next Write or Close. "
           "Useful if we may be writing to a command in a pipe and want to "
           "ensure good CPU utilization.")
      .def("Close", &PyClass::Close,
           "Close() is not necessary to call, as the destructor closes it; "
           "it's mainly useful if you want to handle error states because the "
           "destructor will throw on error if you do not call Close().");
}

template <class C>
void pybind_read_kaldi_object(py::module& m) {
  m.def("ReadKaldiObject",
          py::overload_cast<const std::string &,
                                        C *>(&ReadKaldiObject<C>),
           py::arg("filename"),
           py::arg("c"),
      py::call_guard<py::gil_scoped_release>()
     );
}

template <class BasicType>
void pybind_basic_vector_holder(py::module& m, const std::string& class_name,
                                const std::string& class_help_doc = "") {
  using PyClass = BasicVectorHolder<BasicType>;
  py::class_<PyClass>(m, class_name.c_str(), class_help_doc.c_str())
      .def(py::init<>())
      .def("Clear", &PyClass::Clear)
      .def("Read", &PyClass::Read, py::arg("is"),
      py::call_guard<py::gil_scoped_release>())
      .def_static("IsReadInBinary", &PyClass::IsReadInBinary)
      .def("Value", &PyClass::Value, py::return_value_policy::reference,
      py::call_guard<py::gil_scoped_release>());
  // TODO(fangjun): wrap other methods when needed
}

void init_util(py::module &);
#endif  // KALPY_PYBIND_UTIL_H_
