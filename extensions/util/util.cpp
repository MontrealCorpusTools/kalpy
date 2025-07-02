
#include "util/pybind_util.h"
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

struct Interval {
    Interval(const float &begin, const float &end, const std::string &label) : begin(begin), end(end), label(label) { }
    Interval() : begin(-1.0), end(-1.0), label("-") { }


    virtual float compare_labels(const std::string &other_label, const std::string &silence_phone, const std::map<std::string, std::set<std::string>> &mapping) const{
        if (label == other_label){
            return 0.0;
        }
        else if (label == silence_phone || other_label == silence_phone){
            return 10.0;
        }
        std::map<std::string, std::set<std::string>>::const_iterator pos = mapping.find(other_label);
        if (pos != mapping.end() && pos->second.find(label) != pos->second.end()){
            return 0.0;
        }
        return 2.0;
    }

    virtual float calculate_score(const Interval &other_interval, const std::string &silence_phone, const std::map<std::string, std::set<std::string>> &mapping) const{
        float score = compare_labels(other_interval.label, silence_phone, mapping);
        if (begin >= 0.0 && end > 0.0){
          score += std::abs(begin - other_interval.begin);
          score += std::abs(end - other_interval.end);
        }
        return score;
    }

    std::string label;
    float begin;
    float end;
};

struct PyInterval : Interval {
    using Interval::Interval; // Inherit constructors

    float compare_labels(const std::string &other_label, const std::string &silence_phone, const std::map<std::string, std::set<std::string>> &mapping) const override {
      PYBIND11_OVERRIDE_PURE(float, Interval, compare_labels, other_label, silence_phone, mapping);
    }

    float calculate_score(const Interval &other_interval, const std::string &silence_phone, const std::map<std::string, std::set<std::string>> &mapping) const override {
      PYBIND11_OVERRIDE_PURE(float, Interval, calculate_score, other_interval, silence_phone, mapping);
    }
};

struct CtmInterval {
    CtmInterval(
      const float &begin,
      const float &end,
      const std::string &label,
      const int32 &symbol,
      const float &confidence=0.0) : begin(begin), end(end), label(label), symbol(symbol), confidence(confidence) { }
    CtmInterval(
      const float &begin,
      const float &end,
      const std::string &label) : begin(begin), end(end), label(label), symbol(-1), confidence(0.0) { }
    CtmInterval() : begin(-1.0), end(-1.0), label("-"), symbol(-1), confidence(0.0) { }


    virtual float compare_labels(const std::string &other_label, const std::string &silence_phone, const std::map<std::string, std::set<std::string>> &mapping) const{
        if (label == other_label){
            return 0.0;
        }
        else if (label == silence_phone || other_label == silence_phone){
            return 10.0;
        }
        std::map<std::string, std::set<std::string>>::const_iterator pos = mapping.find(other_label);
        if (pos != mapping.end() && pos->second.find(label) != pos->second.end()){
            return 0.0;
        }
        return 2.0;
    }

    virtual float calculate_score(const CtmInterval &other_interval, const std::string &silence_phone, const std::map<std::string, std::set<std::string>> &mapping) const{
        float score = compare_labels(other_interval.label, silence_phone, mapping);
        if (begin >= 0.0 && end > 0.0){
          score += std::abs(begin - other_interval.begin);
          score += std::abs(end - other_interval.end);
        }
        return score;
    }

    std::string label;
    float begin;
    float end;
    int32 symbol;
    float confidence;
};

struct PyCtmInterval : CtmInterval {
    using CtmInterval::CtmInterval; // Inherit constructors

    float compare_labels(const std::string &other_label, const std::string &silence_phone, const std::map<std::string, std::set<std::string>> &mapping) const override {
      PYBIND11_OVERRIDE_PURE(float, CtmInterval, compare_labels, other_label, silence_phone, mapping);
    }

    float calculate_score(const CtmInterval &other_interval, const std::string &silence_phone, const std::map<std::string, std::set<std::string>> &mapping) const override {
      PYBIND11_OVERRIDE_PURE(float, CtmInterval, calculate_score, other_interval, silence_phone, mapping);
    }
};

struct WordCtmInterval {
    WordCtmInterval(
      const std::string &label,
      const int32 &symbol,
      const std::vector<CtmInterval> &phones) : label(label), symbol(symbol), phones(phones) { }
    WordCtmInterval() :  label("-"), symbol(-1), phones() { }

    virtual float compare_labels(const std::string &other_label, const std::string &silence_phone, const std::map<std::string, std::set<std::string>> &mapping) const{
        if (label == other_label){
            return 0.0;
        }
        else if (label == silence_phone || other_label == silence_phone){
            return 10.0;
        }
        std::map<std::string, std::set<std::string>>::const_iterator pos = mapping.find(other_label);
        if (pos != mapping.end() && pos->second.find(label) != pos->second.end()){
            return 0.0;
        }
        return 2.0;
    }

    virtual float calculate_score(const WordCtmInterval &other_interval, const std::string &silence_phone, const std::map<std::string, std::set<std::string>> &mapping) const{
        float score = compare_labels(other_interval.label, silence_phone, mapping);
        float begin = getBegin();
        float end = getEnd();
        if (begin >= 0.0 && end > 0.0){
          score += std::abs(begin - other_interval.getBegin());
          score += std::abs(end - other_interval.getEnd());
        }
        return score;
    }

    const float getBegin() const {
      if (!phones.empty()){
        return phones.front().begin;
      }
      return 0.0;
    }

    const float getEnd() const {
      if (!phones.empty()){
        return phones.back().end;
      }
      return 0.0;
    }

    std::string getPronunciation() const {
      if (phones.empty()){
        return "";
      }
      std::string delimiter = " ";
      std::string result = std::accumulate(std::next(phones.begin()), phones.end(),
        phones[0].label,
        [&delimiter](std::string& a, CtmInterval b) {
          return a + delimiter+ b.label;
        });
      return result;
    }

    std::string label;
    int32 symbol;
    std::vector<CtmInterval> phones;
};

struct PyWordCtmInterval : WordCtmInterval {
    using WordCtmInterval::WordCtmInterval; // Inherit constructors

    float compare_labels(const std::string &other_label, const std::string &silence_phone, const std::map<std::string, std::set<std::string>> &mapping) const override {
      PYBIND11_OVERRIDE_PURE(float, WordCtmInterval, compare_labels, other_label, silence_phone, mapping);
    }

    float calculate_score(const WordCtmInterval &other_interval, const std::string &silence_phone, const std::map<std::string, std::set<std::string>> &mapping) const override {
      PYBIND11_OVERRIDE_PURE(float, WordCtmInterval, calculate_score, other_interval, silence_phone, mapping);
    }
};

template<typename IntervalType>
struct IntervalAlignment {
    IntervalAlignment(const std::vector<std::pair<IntervalType, IntervalType>> &alignment, const float &score) : alignment(alignment), score(score) { }

    void setReference(const std::vector<IntervalType> &refrence)  {
      KALDI_ASSERT(refrence.size() == alignment.size());
      for (size_t i = 1; i < alignment.size(); i++) {
        alignment[i] = std::make_pair(refrence[i], alignment[i].second);
      }
    }

    std::vector<IntervalType> getReference() {
      std::vector<IntervalType> reference;
      for (size_t i = 1; i < alignment.size(); i++) {
        reference.push_back(alignment[i].first);
      }
      return reference;
    }

    void setTest(const std::vector<IntervalType> &test) {
      KALDI_ASSERT(test.size() == alignment.size());
      for (size_t i = 1; i < alignment.size(); i++) {
        alignment[i] = std::make_pair(alignment[i].first, test[i]);
      }
    }

    std::vector<IntervalType> getTest() {
      std::vector<IntervalType> test;
      for (size_t i = 1; i < alignment.size(); i++) {
        test.push_back(alignment[i].second);
      }
      return test;
    }

    const size_t getLength() const { return alignment.size(); }

    std::vector<std::pair<IntervalType, IntervalType>> alignment;
    float score;
};

template<typename IntervalType>
IntervalAlignment<IntervalType> align_intervals(
  const std::vector<IntervalType> &reference_intervals,
  const std::vector<IntervalType> &hypothesis_intervals,
  const std::string &silence_phone,
  const std::map<std::string, std::set<std::string>> &mapping) {

            py::gil_scoped_release release;
            std::vector<std::pair<IntervalType, IntervalType> > output;
            IntervalType eps_interval = IntervalType();
            // This is very memory-inefficiently implemented using a vector of vectors.
            size_t M = reference_intervals.size(), N = hypothesis_intervals.size();
            size_t m, n;
            std::vector<std::vector<float> > e(M+1);
            for (m = 0; m <=M; m++) e[m].resize(N+1);
            for (n = 0; n <= N; n++)
              e[0][n]  = n;
            for (m = 1; m <= M; m++) {
              e[m][0] = e[m-1][0] + 1;
              for (n = 1; n <= N; n++) {
                float sub_or_ok = e[m-1][n-1] + reference_intervals[m-1].calculate_score(hypothesis_intervals[n-1], silence_phone, mapping);
                float del = e[m-1][n] + 1.0;  // assumes a == ref, b == hyp.
                float ins = e[m][n-1] + 1.0;
                e[m][n] = std::min(sub_or_ok, std::min(del, ins));
              }
            }
            // get time-reversed output first: trace back.
            m = M;
            n = N;
            while (m != 0 || n != 0) {
              size_t last_m, last_n;
              if (m == 0) {
                last_m = m;
                last_n = n-1;
              } else if (n == 0) {
                last_m = m-1;
                last_n = n;
              } else {
                float sub_or_ok = e[m-1][n-1] + reference_intervals[m-1].calculate_score(hypothesis_intervals[n-1], silence_phone, mapping);
                float del = e[m-1][n] + 1.0;  // assumes a == ref, b == hyp.
                float ins = e[m][n-1] + 1.0;
                // choose sub_or_ok if all else equal.
                if (sub_or_ok <= std::min(del, ins)) {
                  last_m = m-1;
                  last_n = n-1;
                } else {
                  if (del <= ins) {  // choose del over ins if equal.
                    last_m = m-1;
                    last_n = n;
                  } else {
                    last_m = m;
                    last_n = n-1;
                  }
                }
              }
              IntervalType a_sym, b_sym;
              a_sym = (last_m == m ? eps_interval : reference_intervals[last_m]);
              b_sym = (last_n == n ? eps_interval : hypothesis_intervals[last_n]);
              output.push_back(std::make_pair(a_sym, b_sym));
              m = last_m;
              n = last_n;
            }
            size_t sz = output.size();
            for (size_t i = 0; i < sz/2; i++)
              std::swap( output[i], output[sz-1-i]);
            py::gil_scoped_acquire gil_acquire;
            return IntervalAlignment<IntervalType>(output, e[M][N]);

                         }

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


    PYBIND11_CONSTINIT static py::gil_safe_call_once_and_store<py::object> exc_storage;

    exc_storage.call_once_and_store_result(
        [&]() { return py::exception<KaldiFatalError>(m, "KaldiFatalError", PyExc_RuntimeError); });

    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) std::rethrow_exception(p);
        } catch (const KaldiFatalError &e) {
            py::set_error(exc_storage.get_stored(), e.KaldiMessage());
        }
    });

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

    py::class_<Interval, PyInterval>(m, "Interval")
        .def(py::init<const float &, const float &, const std::string &>(),
        py::arg("begin"),
        py::arg("end"),
        py::arg("label"))
        .def_readwrite("begin", &Interval::begin)
        .def_readwrite("end", &Interval::end)
        .def_readwrite("label", &Interval::label)
        .def("compare_labels", &Interval::compare_labels,
          "Score two labels based on whether they match or count as the same phone based on the mapping",
            py::arg("other_label"),
            py::arg("silence_phone"),
            py::arg("mapping"))
        .def("calculate_score", &Interval::calculate_score,
          "Method to calculate overlap scoring",
            py::arg("other_interval"),
            py::arg("silence_phone"),
            py::arg("mapping"))
        .def("__add__",
              [](Interval &a, const float other) {
                a.begin += other;
                a.end += other;
            },
            py::arg("other"))
        .def("__add__",
              [](Interval &a, const std::string &other) {
                a.label += other;
            },
            py::arg("other"))
        .def("__lt__",
              [](const Interval &a, const Interval &other) {
                return a.begin < other.begin;
            },
            py::arg("other"))
        .def("__lte__",
              [](const Interval &a, const Interval &other) {
                return a.begin <= other.begin;
            },
            py::arg("other"))
        .def("__gt__",
              [](const Interval &a, const Interval &other) {
                return a.begin > other.begin;
            },
            py::arg("other"))
        .def("__gte__",
              [](const Interval &a, const Interval &other) {
                return a.begin >= other.begin;
            },
            py::arg("other"))
        .def("__repr__",
              [](const Interval &a) {
                return "<Interval of labeled '" + a.label + "' from " + std::to_string(a.begin) + " to " + std::to_string(a.end) +">";
            })
        .def(py::pickle(
            [](const Interval &p) { // __getstate__
                return py::make_tuple(p.begin, p.end, p.label);
            },
            [](py::tuple t) { // __setstate__
                if (t.size() == 3){
                  Interval p(t[0].cast<float>(), t[1].cast<float>(), t[2].cast<std::string>());
                return p;
                }

                    throw pybind11::type_error("Invalid state!");


            }
        ));

    py::class_<CtmInterval, PyCtmInterval>(m, "CtmInterval")
        .def(py::init<
          const float &,
          const float &,
          const std::string &,
          const int32 &,
          const float &>(),
        py::arg("begin"),
        py::arg("end"),
        py::arg("label"),
        py::arg("symbol"),
        py::arg("confidence")=0.0)
        .def(py::init<
          const float &,
          const float &,
          const std::string &>(),
        py::arg("begin"),
        py::arg("end"),
        py::arg("label"))
        .def_readwrite("begin", &CtmInterval::begin)
        .def_readwrite("end", &CtmInterval::end)
        .def_readwrite("label", &CtmInterval::label)
        .def_readwrite("symbol", &CtmInterval::symbol)
        .def_readwrite("confidence", &CtmInterval::confidence)
        .def("compare_labels", &CtmInterval::compare_labels,
          "Score two labels based on whether they match or count as the same phone based on the mapping",
            py::arg("other_label"),
            py::arg("silence_phone"),
            py::arg("mapping"))
        .def("calculate_score", &CtmInterval::calculate_score,
          "Method to calculate overlap scoring",
            py::arg("other_interval"),
            py::arg("silence_phone"),
            py::arg("mapping"))
        .def("__add__",
              [](CtmInterval &a, const float other) {
                a.begin += other;
                a.end += other;
            },
            py::arg("other"))
        .def("__add__",
              [](CtmInterval &a, const std::string &other) {
                a.label += other;
            },
            py::arg("other"))
        .def("__lt__",
              [](const CtmInterval &a, const Interval &other) {
                return a.begin < other.begin;
            },
            py::arg("other"))
        .def("__lte__",
              [](const CtmInterval &a, const Interval &other) {
                return a.begin <= other.begin;
            },
            py::arg("other"))
        .def("__gt__",
              [](const CtmInterval &a, const Interval &other) {
                return a.begin > other.begin;
            },
            py::arg("other"))
        .def("__gte__",
              [](const CtmInterval &a, const Interval &other) {
                return a.begin >= other.begin;
            },
            py::arg("other"))
        .def("__eq__",
              [](const CtmInterval &a, const CtmInterval &other) {
                return a.begin == other.begin && a.end == other.end && a.label == other.label;
            },
            py::arg("other"))
        .def("__repr__",
              [](const CtmInterval &a) {
                return "<CtmInterval of labeled '" + a.label + "' from " + std::to_string(a.begin) + " to " + std::to_string(a.end) +">";
            })
        .def(py::pickle(
            [](const CtmInterval &p) { // __getstate__
                return py::make_tuple(p.begin, p.end, p.label, p.symbol, p.confidence);
            },
            [](py::tuple t) { // __setstate__
                if (t.size() != 5)
                    throw pybind11::type_error("Invalid state!");

                CtmInterval p(t[0].cast<float>(), t[1].cast<float>(), t[2].cast<std::string>(), t[3].cast<int32>(), t[4].cast<float>());

                return p;
            }
        ));

    py::class_<WordCtmInterval, PyWordCtmInterval>(m, "WordCtmInterval")
        .def(py::init<
          const std::string &,
          const int32 &,
          const std::vector<CtmInterval> &>(),
        py::arg("label"),
        py::arg("symbol"),
        py::arg("phones"))
        .def_readwrite("label", &WordCtmInterval::label)
        .def_readwrite("symbol", &WordCtmInterval::symbol)
        .def_readwrite("phones", &WordCtmInterval::phones)
        .def_property_readonly("begin", &WordCtmInterval::getBegin)
        .def_property_readonly("end", &WordCtmInterval::getEnd)
        .def_property_readonly("pronunciation", &WordCtmInterval::getPronunciation)
        .def("compare_labels", &WordCtmInterval::compare_labels,
          "Score two labels based on whether they match or count as the same phone based on the mapping",
            py::arg("other_label"),
            py::arg("silence_phone"),
            py::arg("mapping"))
        .def("calculate_score", &WordCtmInterval::calculate_score,
          "Method to calculate overlap scoring",
            py::arg("other_interval"),
            py::arg("silence_phone"),
            py::arg("mapping"))
        .def("__lt__",
              [](const WordCtmInterval &a, const WordCtmInterval &other) {
                return a.getBegin() < other.getBegin();
            },
            py::arg("other"))
        .def("__lte__",
              [](const WordCtmInterval &a, const WordCtmInterval &other) {
                return a.getBegin() <= other.getBegin();
            },
            py::arg("other"))
        .def("__gt__",
              [](const WordCtmInterval &a, const WordCtmInterval &other) {
                return a.getBegin() > other.getBegin();
            },
            py::arg("other"))
        .def("__gte__",
              [](const WordCtmInterval &a, const WordCtmInterval &other) {
                return a.getBegin() >= other.getBegin();
            },
            py::arg("other"))
        .def("__repr__",
              [](const WordCtmInterval &a) {
                return "<WordCtmInterval of labeled '" + a.label + "' from " + std::to_string(a.getBegin()) + " to " + std::to_string(a.getEnd()) +">";
            })
        .def(py::pickle(
            [](const WordCtmInterval &p) { // __getstate__
                return py::make_tuple(p.label, p.symbol, p.phones);
            },
            [](py::tuple t) { // __setstate__
                if (t.size() != 3)
                    throw pybind11::type_error("Invalid state!");

                WordCtmInterval p(t[0].cast<std::string>(), t[1].cast<int32>(), t[2].cast<std::vector<CtmInterval>>());

                return p;
            }
        ));

    py::class_<IntervalAlignment<Interval>>(m, "IntervalAlignment")
        .def(py::init<const std::vector<std::pair<Interval, Interval>> &, const float &>(),
        py::arg("alignment"),
        py::arg("score"))
        .def_readwrite("alignment", &IntervalAlignment<Interval>::alignment)
        .def_readwrite("score", &IntervalAlignment<Interval>::score)
        .def_property("reference", &IntervalAlignment<Interval>::getReference, &IntervalAlignment<Interval>::setReference)
        .def_property("test", &IntervalAlignment<Interval>::getTest, &IntervalAlignment<Interval>::setTest)
        .def("__getitem__",
              [](const IntervalAlignment<Interval> &a, const int32 index) {
                return a.alignment[index];
            },
            py::arg("index"))
        .def("__len__",
              [](const IntervalAlignment<Interval> &a) {
                return a.getLength();
            })
        .def("__repr__",
              [](const IntervalAlignment<Interval> &a) {
                return "<IntervalAlignment of length=" + std::to_string(a.getLength()) + " and score=" + std::to_string(a.score) +">";
            })
        .def("__iter__", [](const IntervalAlignment<Interval> &a) { return py::make_iterator(a.alignment.begin(), a.alignment.end()); },
                         py::keep_alive<0, 1>() /* Essential: keep object alive while iterator exists */);

    py::class_<IntervalAlignment<CtmInterval>>(m, "CtmIntervalAlignment")
        .def(py::init<const std::vector<std::pair<CtmInterval, CtmInterval>> &, const float &>(),
        py::arg("alignment"),
        py::arg("score"))
        .def_readwrite("alignment", &IntervalAlignment<CtmInterval>::alignment)
        .def_readwrite("score", &IntervalAlignment<CtmInterval>::score)
        .def_property("reference", &IntervalAlignment<CtmInterval>::getReference, &IntervalAlignment<CtmInterval>::setReference)
        .def_property("test", &IntervalAlignment<CtmInterval>::getTest, &IntervalAlignment<CtmInterval>::setTest)
        .def("__getitem__",
              [](const IntervalAlignment<CtmInterval> &a, const int32 index) {
                return a.alignment[index];
            },
            py::arg("index"))
        .def("__len__",
              [](const IntervalAlignment<CtmInterval> &a) {
                return a.getLength();
            })
        .def("__repr__",
              [](const IntervalAlignment<CtmInterval> &a) {
                return "<IntervalAlignment of length=" + std::to_string(a.getLength()) + " and score=" + std::to_string(a.score) +">";
            })
        .def("__iter__", [](const IntervalAlignment<CtmInterval> &a) { return py::make_iterator(a.alignment.begin(), a.alignment.end()); },
                         py::keep_alive<0, 1>() /* Essential: keep object alive while iterator exists */);

    py::class_<IntervalAlignment<WordCtmInterval>>(m, "WordCtmIntervalAlignment")
        .def(py::init<const std::vector<std::pair<WordCtmInterval, WordCtmInterval>> &, const float &>(),
        py::arg("alignment"),
        py::arg("score"))
        .def_readwrite("alignment", &IntervalAlignment<WordCtmInterval>::alignment)
        .def_readwrite("score", &IntervalAlignment<WordCtmInterval>::score)
        .def_property("reference", &IntervalAlignment<WordCtmInterval>::getReference, &IntervalAlignment<WordCtmInterval>::setReference)
        .def_property("test", &IntervalAlignment<WordCtmInterval>::getTest, &IntervalAlignment<WordCtmInterval>::setTest)
        .def("__getitem__",
              [](const IntervalAlignment<WordCtmInterval> &a, const int32 index) {
                return a.alignment[index];
            },
            py::arg("index"))
        .def("__len__",
              [](const IntervalAlignment<WordCtmInterval> &a) {
                return a.getLength();
            })
        .def("__repr__",
              [](const IntervalAlignment<WordCtmInterval> &a) {
                return "<IntervalAlignment of length=" + std::to_string(a.getLength()) + " and score=" + std::to_string(a.score) +">";
            })
        .def("__iter__", [](const IntervalAlignment<WordCtmInterval> &a) { return py::make_iterator(a.alignment.begin(), a.alignment.end()); },
                         py::keep_alive<0, 1>() /* Essential: keep object alive while iterator exists */);

    m.def("align_intervals",
          &align_intervals<Interval>,
          "Align intervals based on how much they overlap and their label, with the ability to specify a custom mapping for "
    "different labels to be scored as if they're the same",
          py::arg("reference_intervals"),
          py::arg("hypothesis_intervals"),
          py::arg("silence_phone"),
          py::arg("mapping"),
          py::return_value_policy::take_ownership
          );

    m.def("align_intervals",
          &align_intervals<CtmInterval>,
          "Align intervals based on how much they overlap and their label, with the ability to specify a custom mapping for "
    "different labels to be scored as if they're the same",
          py::arg("reference_intervals"),
          py::arg("hypothesis_intervals"),
          py::arg("silence_phone"),
          py::arg("mapping"),
          py::return_value_policy::take_ownership
          );

    m.def("align_intervals",
          &align_intervals<WordCtmInterval>,
          "Align intervals based on how much they overlap and their label, with the ability to specify a custom mapping for "
    "different labels to be scored as if they're the same",
          py::arg("reference_intervals"),
          py::arg("hypothesis_intervals"),
          py::arg("silence_phone"),
          py::arg("mapping"),
          py::return_value_policy::take_ownership
          );

}
