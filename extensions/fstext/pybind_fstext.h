
#ifndef KALPY_PYBIND_FSTEXT_H_
#define KALPY_PYBIND_FSTEXT_H_

#include "pybind/kaldi_pybind.h"
#include "util/pybind_util.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"

#include "fst/script/fst-class.h"
#include "fst/script/info-impl.h"
#include "fst/script/print-impl.h"
#include "fst/vector-fst.h"
#include "fst/fst.h"
#include "fst/fstlib.h"
#include "fst/fst-decl.h"
#include "fstext/fstext-utils.h"
#include "fstext/kaldi-fst-io.h"
#include "fstext/lattice-utils.h"
#include "fstext/lattice-utils-inl.h"

using namespace fst;

template <typename A>
class PyFst : public Fst<A> {
public:
    //Inherit the constructors
    using Fst<A>::Fst;

    //Trampoline (need one for each virtual function)
    typename Fst<A>::StateId Start() const override {
        PYBIND11_OVERRIDE_PURE(
            int, //Return type (ret_type)
            Fst<A>,      //Parent class (cname)
            Start          //Name of function in C++ (must match Python name) (fn)
                  //Argument(s) (...)
        );
    }

    typename Fst<A>::Weight Final(typename Fst<A>::StateId s) const override {
        PYBIND11_OVERRIDE_PURE(
            typename Fst<A>::Weight, //Return type (ret_type)
            Fst<A>,      //Parent class (cname)
            Final,          //Name of function in C++ (must match Python name) (fn)
            s      //Argument(s) (...)
        );
    }

    size_t NumArcs(typename Fst<A>::StateId s) const override {
        PYBIND11_OVERRIDE_PURE(
            size_t, //Return type (ret_type)
            Fst<A>,      //Parent class (cname)
            NumArcs,        //Name of function in C++ (must match Python name) (fn)
            s      //Argument(s) (...)
        );
    }

    size_t NumInputEpsilons(typename Fst<A>::StateId s) const override {
        PYBIND11_OVERRIDE_PURE(
            size_t, //Return type (ret_type)
            Fst<A>,      //Parent class (cname)
            NumInputEpsilons,          //Name of function in C++ (must match Python name) (fn)
            s      //Argument(s) (...)
        );
    }

    size_t NumOutputEpsilons(typename Fst<A>::StateId s) const override {
        PYBIND11_OVERRIDE_PURE(
            size_t, //Return type (ret_type)
            Fst<A>,      //Parent class (cname)
            NumOutputEpsilons,          //Name of function in C++ (must match Python name) (fn)
            s      //Argument(s) (...)
        );
    }

    uint64_t Properties(uint64_t mask, bool test) const override {
        PYBIND11_OVERRIDE_PURE(
            uint64_t, //Return type (ret_type)
            Fst<A>,      //Parent class (cname)
            Properties,          //Name of function in C++ (must match Python name) (fn)
            mask, test      //Argument(s) (...)
        );
    }

    const std::string &Type() const override {
        PYBIND11_OVERRIDE_PURE(
            const std::string &, //Return type (ret_type)
            Fst<A>,      //Parent class (cname)
            Type          //Name of function in C++ (must match Python name) (fn)
                  //Argument(s) (...)
        );
    }

    Fst<A> *Copy(bool safe = false) const override {
        PYBIND11_OVERRIDE_PURE(
            Fst<A> *, //Return type (ret_type)
            Fst<A>,      //Parent class (cname)
            Copy,          //Name of function in C++ (must match Python name) (fn)
            safe      //Argument(s) (...)
        );
    }

    const SymbolTable *InputSymbols() const override {
        PYBIND11_OVERRIDE_PURE(
            const SymbolTable *, //Return type (ret_type)
            Fst<A>,      //Parent class (cname)
            InputSymbols          //Name of function in C++ (must match Python name) (fn)
                  //Argument(s) (...)
        );
    }

    const SymbolTable *OutputSymbols() const override {
        PYBIND11_OVERRIDE_PURE(
            const SymbolTable *, //Return type (ret_type)
            Fst<A>,      //Parent class (cname)
            OutputSymbols          //Name of function in C++ (must match Python name) (fn)
                  //Argument(s) (...)
        );
    }

    void InitStateIterator(StateIteratorData<A> *data) const override {
        PYBIND11_OVERRIDE_PURE(
            void, //Return type (ret_type)
            Fst<A>,      //Parent class (cname)
            InitStateIterator,          //Name of function in C++ (must match Python name) (fn)
            data      //Argument(s) (...)
        );
    }

    void InitArcIterator(typename Fst<A>::StateId s, ArcIteratorData<A> *data) const override {
        PYBIND11_OVERRIDE_PURE(
            void, //Return type (ret_type)
            Fst<A>,      //Parent class (cname)
            InitArcIterator,          //Name of function in C++ (must match Python name) (fn)
            s, data      //Argument(s) (...)
        );
    }

    MatcherBase<A> *InitMatcher(MatchType match_type) const override {
        PYBIND11_OVERRIDE_PURE(
            MatcherBase<A> *, //Return type (ret_type)
            Fst<A>,      //Parent class (cname)
            InitMatcher,          //Name of function in C++ (must match Python name) (fn)
            match_type      //Argument(s) (...)
        );
    }
};

template <typename A>
class PyExpandedFst : public ExpandedFst<A> {
public:
    //Inherit the constructors
    using ExpandedFst<A>::ExpandedFst;

    //Trampoline (need one for each virtual function)
    typename ExpandedFst<A>::StateId Start() const override {
        PYBIND11_OVERRIDE_PURE(
            int, //Return type (ret_type)
            ExpandedFst<A>,      //Parent class (cname)
            Start          //Name of function in C++ (must match Python name) (fn)
                  //Argument(s) (...)
        );
    }

    typename A::Weight Final(typename Fst<A>::StateId s) const override {
        PYBIND11_OVERRIDE_PURE(
            typename A::Weight, //Return type (ret_type)
            ExpandedFst<A>,      //Parent class (cname)
            Final,          //Name of function in C++ (must match Python name) (fn)
            s      //Argument(s) (...)
        );
    }

    size_t NumArcs(typename ExpandedFst<A>::StateId s) const override {
        PYBIND11_OVERRIDE_PURE(
            size_t, //Return type (ret_type)
            ExpandedFst<A>,      //Parent class (cname)
            NumArcs,        //Name of function in C++ (must match Python name) (fn)
            s      //Argument(s) (...)
        );
    }

    typename ExpandedFst<A>::StateId NumStates() const override {
        PYBIND11_OVERRIDE_PURE(
            typename ExpandedFst<A>::StateId, //Return type (ret_type)
            ExpandedFst<A>,      //Parent class (cname)
            NumStates        //Name of function in C++ (must match Python name) (fn)
                  //Argument(s) (...)
        );
    }

    size_t NumInputEpsilons(typename ExpandedFst<A>::StateId s) const override {
        PYBIND11_OVERRIDE_PURE(
            size_t, //Return type (ret_type)
            ExpandedFst<A>,      //Parent class (cname)
            NumInputEpsilons,          //Name of function in C++ (must match Python name) (fn)
            s      //Argument(s) (...)
        );
    }

    size_t NumOutputEpsilons(typename ExpandedFst<A>::StateId s) const override {
        PYBIND11_OVERRIDE_PURE(
            size_t, //Return type (ret_type)
            ExpandedFst<A>,      //Parent class (cname)
            NumOutputEpsilons,          //Name of function in C++ (must match Python name) (fn)
            s      //Argument(s) (...)
        );
    }

    uint64_t Properties(uint64_t mask, bool test) const override {
        PYBIND11_OVERRIDE_PURE(
            uint64_t, //Return type (ret_type)
            ExpandedFst<A>,      //Parent class (cname)
            Properties,          //Name of function in C++ (must match Python name) (fn)
            mask, test      //Argument(s) (...)
        );
    }

    const std::string &Type() const override {
        PYBIND11_OVERRIDE_PURE(
            const std::string &, //Return type (ret_type)
            ExpandedFst<A>,      //Parent class (cname)
            Type          //Name of function in C++ (must match Python name) (fn)
                  //Argument(s) (...)
        );
    }

    ExpandedFst<A> *Copy(bool safe = false) const override {
        PYBIND11_OVERRIDE_PURE(
            ExpandedFst<A> *, //Return type (ret_type)
            ExpandedFst<A>,      //Parent class (cname)
            Copy,          //Name of function in C++ (must match Python name) (fn)
            safe      //Argument(s) (...)
        );
    }

    const SymbolTable *InputSymbols() const override {
        PYBIND11_OVERRIDE_PURE(
            const SymbolTable *, //Return type (ret_type)
            ExpandedFst<A>,      //Parent class (cname)
            InputSymbols          //Name of function in C++ (must match Python name) (fn)
                  //Argument(s) (...)
        );
    }

    const SymbolTable *OutputSymbols() const override {
        PYBIND11_OVERRIDE_PURE(
            const SymbolTable *, //Return type (ret_type)
            ExpandedFst<A>,      //Parent class (cname)
            OutputSymbols          //Name of function in C++ (must match Python name) (fn)
                  //Argument(s) (...)
        );
    }

    void InitStateIterator(StateIteratorData<A> *data) const override {
        PYBIND11_OVERRIDE_PURE(
            void, //Return type (ret_type)
            ExpandedFst<A>,      //Parent class (cname)
            InitStateIterator,          //Name of function in C++ (must match Python name) (fn)
            data      //Argument(s) (...)
        );
    }

    void InitArcIterator(typename Fst<A>::StateId s, ArcIteratorData<A> *data) const override {
        PYBIND11_OVERRIDE_PURE(
            void, //Return type (ret_type)
            ExpandedFst<A>,      //Parent class (cname)
            InitArcIterator,          //Name of function in C++ (must match Python name) (fn)
            s, data      //Argument(s) (...)
        );
    }

    MatcherBase<A> *InitMatcher(MatchType match_type) const override {
        PYBIND11_OVERRIDE_PURE(
            MatcherBase<A> *, //Return type (ret_type)
            ExpandedFst<A>,      //Parent class (cname)
            InitMatcher,          //Name of function in C++ (must match Python name) (fn)
            match_type      //Argument(s) (...)
        );
    }
};

template <typename A>
class PyMutableFst : public MutableFst<A> {
public:
    //Inherit the constructors
    using MutableFst<A>::MutableFst;

    //Trampoline (need one for each virtual function)
    typename MutableFst<A>::StateId Start() const override {
        PYBIND11_OVERRIDE_PURE(
            int, //Return type (ret_type)
            MutableFst<A>,      //Parent class (cname)
            Start          //Name of function in C++ (must match Python name) (fn)
                  //Argument(s) (...)
        );
    }

    typename MutableFst<A>::Weight Final(typename MutableFst<A>::StateId s) const override {
        PYBIND11_OVERRIDE_PURE(
            typename MutableFst<A>::Weight, //Return type (ret_type)
            MutableFst<A>,      //Parent class (cname)
            Final,          //Name of function in C++ (must match Python name) (fn)
            s      //Argument(s) (...)
        );
    }

    size_t NumArcs(typename MutableFst<A>::StateId s) const override {
        PYBIND11_OVERRIDE_PURE(
            size_t, //Return type (ret_type)
            MutableFst<A>,      //Parent class (cname)
            NumArcs,        //Name of function in C++ (must match Python name) (fn)
            s      //Argument(s) (...)
        );
    }

    size_t NumInputEpsilons(typename MutableFst<A>::StateId s) const override {
        PYBIND11_OVERRIDE_PURE(
            size_t, //Return type (ret_type)
            MutableFst<A>,      //Parent class (cname)
            NumInputEpsilons,          //Name of function in C++ (must match Python name) (fn)
            s      //Argument(s) (...)
        );
    }

    size_t NumOutputEpsilons(typename MutableFst<A>::StateId s) const override {
        PYBIND11_OVERRIDE_PURE(
            size_t, //Return type (ret_type)
            MutableFst<A>,      //Parent class (cname)
            NumOutputEpsilons,          //Name of function in C++ (must match Python name) (fn)
            s      //Argument(s) (...)
        );
    }

    uint64_t Properties(uint64_t mask, bool test) const override {
        PYBIND11_OVERRIDE_PURE(
            uint64_t, //Return type (ret_type)
            MutableFst<A>,      //Parent class (cname)
            Properties,          //Name of function in C++ (must match Python name) (fn)
            mask, test      //Argument(s) (...)
        );
    }

    const std::string &Type() const override {
        PYBIND11_OVERRIDE_PURE(
            const std::string &, //Return type (ret_type)
            MutableFst<A>,      //Parent class (cname)
            Type          //Name of function in C++ (must match Python name) (fn)
                  //Argument(s) (...)
        );
    }

    MutableFst<A> *Copy(bool safe = false) const override {
        PYBIND11_OVERRIDE_PURE(
            MutableFst<A> *, //Return type (ret_type)
            MutableFst<A>,      //Parent class (cname)
            Copy,          //Name of function in C++ (must match Python name) (fn)
            safe      //Argument(s) (...)
        );
    }

    const SymbolTable *InputSymbols() const override {
        PYBIND11_OVERRIDE_PURE(
            const SymbolTable *, //Return type (ret_type)
            MutableFst<A>,      //Parent class (cname)
            InputSymbols          //Name of function in C++ (must match Python name) (fn)
                  //Argument(s) (...)
        );
    }

    const SymbolTable *OutputSymbols() const override {
        PYBIND11_OVERRIDE_PURE(
            const SymbolTable *, //Return type (ret_type)
            MutableFst<A>,      //Parent class (cname)
            OutputSymbols          //Name of function in C++ (must match Python name) (fn)
                  //Argument(s) (...)
        );
    }

    void InitStateIterator(StateIteratorData<A> *data) const override {
        PYBIND11_OVERRIDE_PURE(
            void, //Return type (ret_type)
            MutableFst<A>,      //Parent class (cname)
            InitStateIterator,          //Name of function in C++ (must match Python name) (fn)
            data      //Argument(s) (...)
        );
    }

    void InitArcIterator(typename MutableFst<A>::StateId s, ArcIteratorData<A> *data) const override {
        PYBIND11_OVERRIDE_PURE(
            void, //Return type (ret_type)
            MutableFst<A>,      //Parent class (cname)
            InitArcIterator,          //Name of function in C++ (must match Python name) (fn)
            s, data      //Argument(s) (...)
        );
    }

    MatcherBase<A> *InitMatcher(MatchType match_type) const override {
        PYBIND11_OVERRIDE_PURE(
            MatcherBase<A> *, //Return type (ret_type)
            MutableFst<A>,      //Parent class (cname)
            InitMatcher,          //Name of function in C++ (must match Python name) (fn)
            match_type      //Argument(s) (...)
        );
    }

    typename MutableFst<A>::StateId NumStates() const override {
        PYBIND11_OVERRIDE_PURE(
            typename MutableFst<A>::StateId, //Return type (ret_type)
            MutableFst<A>,      //Parent class (cname)
            NumStates        //Name of function in C++ (must match Python name) (fn)
                  //Argument(s) (...)
        );
    }

    MutableFst<A> & operator=(const Fst<A> &fst) override {
        PYBIND11_OVERRIDE_PURE(
            MutableFst<A> &, //Return type (ret_type)
            MutableFst<A>,      //Parent class (cname)
            operator=,          //Name of function in C++ (must match Python name) (fn)
            fst      //Argument(s) (...)
        );
    }

    void SetStart(typename MutableFst<A>::StateId s) override {
        PYBIND11_OVERRIDE_PURE(
            void, //Return type (ret_type)
            MutableFst<A>,      //Parent class (cname)
            SetStart,          //Name of function in C++ (must match Python name) (fn)
            s      //Argument(s) (...)
        );
    }

    void SetFinal(typename MutableFst<A>::StateId s, typename MutableFst<A>::Weight weight = MutableFst<A>::Weight::One()) override {
        PYBIND11_OVERRIDE_PURE(
            void, //Return type (ret_type)
            MutableFst<A>,      //Parent class (cname)
            SetFinal,          //Name of function in C++ (must match Python name) (fn)
            s, weight      //Argument(s) (...)
        );
    }

    void SetProperties(uint64_t props, uint64_t mask) override {
        PYBIND11_OVERRIDE_PURE(
            void, //Return type (ret_type)
            MutableFst<A>,      //Parent class (cname)
            SetProperties,          //Name of function in C++ (must match Python name) (fn)
            props, mask      //Argument(s) (...)
        );
    }

    typename MutableFst<A>::StateId AddState() override {
        PYBIND11_OVERRIDE_PURE(
            typename MutableFst<A>::StateId, //Return type (ret_type)
            MutableFst<A>,      //Parent class (cname)
            SetProperties          //Name of function in C++ (must match Python name) (fn)
                 //Argument(s) (...)
        );
    }

    void AddStates(size_t n) override {
        PYBIND11_OVERRIDE_PURE(
            void, //Return type (ret_type)
            MutableFst<A>,      //Parent class (cname)
            AddStates,          //Name of function in C++ (must match Python name) (fn)
            n     //Argument(s) (...)
        );
    }

    void AddArc(typename MutableFst<A>::StateId s, const A & arc) override {
        PYBIND11_OVERRIDE_PURE(
            void, //Return type (ret_type)
            MutableFst<A>,      //Parent class (cname)
            AddArc,          //Name of function in C++ (must match Python name) (fn)
            s,arc     //Argument(s) (...)
        );
    }

    void DeleteStates(const std::vector<typename MutableFst<A>::StateId> & s) override {
        PYBIND11_OVERRIDE_PURE(
            void, //Return type (ret_type)
            MutableFst<A>,      //Parent class (cname)
            DeleteStates,          //Name of function in C++ (must match Python name) (fn)
            s     //Argument(s) (...)
        );
    }

    void DeleteStates() override {
        PYBIND11_OVERRIDE_PURE(
            void, //Return type (ret_type)
            MutableFst<A>,      //Parent class (cname)
            DeleteStates          //Name of function in C++ (must match Python name) (fn)
                 //Argument(s) (...)
        );
    }

    void DeleteArcs(typename MutableFst<A>::StateId s, size_t n) override {
        PYBIND11_OVERRIDE_PURE(
            void, //Return type (ret_type)
            MutableFst<A>,      //Parent class (cname)
            DeleteArcs,          //Name of function in C++ (must match Python name) (fn)
            s, n     //Argument(s) (...)
        );
    }

    void DeleteArcs(typename MutableFst<A>::StateId s) override {
        PYBIND11_OVERRIDE_PURE(
            void, //Return type (ret_type)
            MutableFst<A>,      //Parent class (cname)
            DeleteArcs,          //Name of function in C++ (must match Python name) (fn)
            s     //Argument(s) (...)
        );
    }

    void ReserveStates(size_t n) override {
        PYBIND11_OVERRIDE_PURE(
            void, //Return type (ret_type)
            MutableFst<A>,      //Parent class (cname)
            ReserveStates,          //Name of function in C++ (must match Python name) (fn)
            n     //Argument(s) (...)
        );
    }

    void ReserveArcs(typename MutableFst<A>::StateId s, size_t n) override {
        PYBIND11_OVERRIDE_PURE(
            void, //Return type (ret_type)
            MutableFst<A>,      //Parent class (cname)
            ReserveArcs,          //Name of function in C++ (must match Python name) (fn)
            s, n     //Argument(s) (...)
        );
    }

    SymbolTable * MutableInputSymbols() override {
        PYBIND11_OVERRIDE_PURE(
            SymbolTable *, //Return type (ret_type)
            MutableFst<A>,      //Parent class (cname)
            MutableInputSymbols          //Name of function in C++ (must match Python name) (fn)
                 //Argument(s) (...)
        );
    }

    SymbolTable * MutableOutputSymbols() override {
        PYBIND11_OVERRIDE_PURE(
            SymbolTable *, //Return type (ret_type)
            MutableFst<A>,      //Parent class (cname)
            MutableOutputSymbols          //Name of function in C++ (must match Python name) (fn)
                 //Argument(s) (...)
        );
    }

    void SetInputSymbols(const SymbolTable *isyms) override {
        PYBIND11_OVERRIDE_PURE(
            void, //Return type (ret_type)
            MutableFst<A>,      //Parent class (cname)
            SetInputSymbols,          //Name of function in C++ (must match Python name) (fn)
            isyms     //Argument(s) (...)
        );
    }

    void SetOutputSymbols(const SymbolTable *osyms) override {
        PYBIND11_OVERRIDE_PURE(
            void, //Return type (ret_type)
            MutableFst<A>,      //Parent class (cname)
            SetOutputSymbols,          //Name of function in C++ (must match Python name) (fn)
            osyms     //Argument(s) (...)
        );
    }

    void InitMutableArcIterator(typename MutableFst<A>::StateId s,
                                      MutableArcIteratorData<A> *data) override {
        PYBIND11_OVERRIDE_PURE(
            void, //Return type (ret_type)
            MutableFst<A>,      //Parent class (cname)
            InitMutableArcIterator,          //Name of function in C++ (must match Python name) (fn)
            s, data     //Argument(s) (...)
        );
    }
};

template<class Arc, class I>
void RemoveArcsWithSomeInputSymbols(const std::vector<I> &symbols_in,
                                    VectorFst<Arc> *fst) {
  typedef typename Arc::StateId StateId;

  kaldi::ConstIntegerSet<I> symbol_set(symbols_in);

  StateId num_states = fst->NumStates();
  StateId dead_state = fst->AddState();
  for (StateId s = 0; s < num_states; s++) {
    for (MutableArcIterator<VectorFst<Arc> > iter(fst, s);
         !iter.Done(); iter.Next()) {
      if (symbol_set.count(iter.Value().ilabel) != 0) {
        Arc arc = iter.Value();
        arc.nextstate = dead_state;
        iter.SetValue(arc);
      }
    }
  }
  // Connect() will actually remove the arcs, and the dead state.
  Connect(fst);
  if (fst->NumStates() == 0)
    KALDI_WARN << "After Connect(), fst was empty.";
}

template<class Arc, class I>
void PenalizeArcsWithSomeInputSymbols(const std::vector<I> &symbols_in,
                                      float penalty,
                                      VectorFst<Arc> *fst) {
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Label Label;
  typedef typename Arc::Weight Weight;

  Weight penalty_weight(penalty);

  kaldi::ConstIntegerSet<I> symbol_set(symbols_in);

  StateId num_states = fst->NumStates();
  for (StateId s = 0; s < num_states; s++) {
    for (MutableArcIterator<VectorFst<Arc> > iter(fst, s);
         !iter.Done(); iter.Next()) {
      if (symbol_set.count(iter.Value().ilabel) != 0) {
        Arc arc = iter.Value();
        arc.weight = Times(arc.weight, penalty_weight);
        iter.SetValue(arc);
      }
    }
  }
}


template <typename W>
void pybind_arc_impl(py::module& m, const std::string& class_name,
                     const std::string& class_help_doc = "") {
  using PyClass = fst::ArcTpl<W>;
  using Weight = typename PyClass::Weight;
  using Label = typename PyClass::Label;
  using StateId = typename PyClass::StateId;

  py::class_<PyClass>(m, class_name.c_str(), class_help_doc.c_str())
      .def(py::init<>())
      .def(py::init<Label, Label, Weight, StateId>(), py::arg("ilabel"),
           py::arg("olabel"), py::arg("weight"), py::arg("nextstate"))
      .def(py::init<const PyClass&>(), py::arg("weight"))
      .def_readwrite("ilabel", &PyClass::ilabel)
      .def_readwrite("olabel", &PyClass::olabel)
      .def_readwrite("weight", &PyClass::weight)
      .def_readwrite("nextstate", &PyClass::nextstate)
      .def("__str__",
           [](const PyClass& arc) {
             std::ostringstream os;
             os << "(ilabel: " << arc.ilabel << ", "
                << "olabel: " << arc.olabel << ", "
                << "weight: " << arc.weight << ", "
                << "nextstate: " << arc.nextstate << ")";
             return os.str();
           })
      .def_static("Type", &PyClass::Type, py::return_value_policy::reference);
}

template <typename A>
void pybind_state_iterator_base_impl(py::module& m,
                                     const std::string& class_name,
                                     const std::string& class_help_doc = "") {
  using PyClass = fst::StateIteratorBase<A>;
  py::class_<PyClass>(m, class_name.c_str(), class_help_doc.c_str())
      .def("Done", &PyClass::Done, "End of iterator?")
      .def("Value", &PyClass::Value, "Returns current state (when !Done()).")
      .def("Next", &PyClass::Next, "Advances to next state (when !Done()).")
      .def("Reset", &PyClass::Reset, "Resets to initial condition.");
}

template <typename A>
void pybind_state_iterator_data_impl(py::module& m,
                                     const std::string& class_name,
                                     const std::string& class_help_doc = "") {
  using PyClass = fst::StateIteratorData<A>;
  py::class_<PyClass, std::unique_ptr<PyClass, py::nodelete>>(
      m, class_name.c_str(), class_help_doc.c_str())
      .def(py::init<>())
      .def_readwrite("base", &PyClass::base,
                     "Specialized iterator if non-zero.")
      .def_readwrite("nstates", &PyClass::nstates,
                     "Otherwise, the total number of states.");
}

template <typename A>
void pybind_arc_iterator_base_impl(py::module& m, const std::string& class_name,
                                   const std::string& class_help_doc = "") {
  using PyClass = fst::ArcIteratorBase<A>;
  py::class_<PyClass>(m, class_name.c_str(), class_help_doc.c_str())
      .def("Done", &PyClass::Done, "End of iterator?")
      .def("Value", &PyClass::Value, "Returns current arc (when !Done()).",
           py::return_value_policy::reference)
      .def("Next", &PyClass::Next, "Advances to next arc (when !Done()).")
      .def("Position", &PyClass::Position, "Returns current position.")
      .def("Reset", &PyClass::Reset, "Resets to initial condition.")
      .def("Seek", &PyClass::Seek, "Advances to arbitrary arc by position.")
      .def("Flags", &PyClass::Flags, "Returns current behavorial flags.")
      .def("SetFlags", &PyClass::SetFlags, "Sets behavorial flags.");
}

template <typename A>
void pybind_arc_iterator_data_impl(py::module& m, const std::string& class_name,
                                   const std::string& class_help_doc = "") {
  using PyClass = fst::ArcIteratorData<A>;
  py::class_<PyClass, std::unique_ptr<PyClass, py::nodelete>>(
      m, class_name.c_str(), class_help_doc.c_str())
      .def(py::init<>())
      .def_readwrite("base", &PyClass::base,
                     "Specialized iterator if non-zero.")
      .def_readwrite("arcs", &PyClass::arcs, "Otherwise arcs pointer")
      .def_readwrite("narcs", &PyClass::narcs, "arc count")
      .def_readwrite("ref_count", &PyClass::ref_count,
                     "reference count if non-zero.");
}

template <typename FST>
void pybind_state_iterator_impl(py::module& m, const std::string& class_name,
                                const std::string& class_help_doc = "") {
  using PyClass = fst::StateIterator<FST>;
  py::class_<PyClass>(m, class_name.c_str(), class_help_doc.c_str())
      .def(py::init<const FST&>(), py::arg("fst"))
      .def("Done", &PyClass::Done)
      .def("Value", &PyClass::Value)
      .def("Next", &PyClass::Next)
      .def("Reset", &PyClass::Reset);
}

template <typename FST>
void pybind_arc_iterator_impl(py::module& m, const std::string& class_name,
                              const std::string& class_help_doc = "") {
  using PyClass = fst::ArcIterator<FST>;
  using StateId = typename FST::StateId;
  py::class_<PyClass>(m, class_name.c_str(), class_help_doc.c_str())
      .def(py::init<const FST&, StateId>(), py::arg("fst"), py::arg("s"))
      .def("Done", &PyClass::Done)
      .def("Value", &PyClass::Value, py::return_value_policy::reference)
      .def("Next", &PyClass::Next)
      .def("Reset", &PyClass::Reset)
      .def("Seek", &PyClass::Seek, py::arg("a"))
      .def("Position", &PyClass::Position)
      .def("Flags", &PyClass::Flags)
      .def("SetFlags", &PyClass::SetFlags);
}

template <typename FST>
void pybind_mutable_arc_iterator_impl(py::module& m,
                                      const std::string& class_name,
                                      const std::string& class_help_doc = "") {
  using PyClass = fst::MutableArcIterator<FST>;
  using StateId = typename PyClass::StateId;

  py::class_<PyClass>(m, class_name.c_str(), class_help_doc.c_str())
      .def(py::init<FST*, StateId>(), py::arg("fst"), py::arg("s"))
      .def("Done", &PyClass::Done)
      .def("Value", &PyClass::Value, py::return_value_policy::reference)
      .def("SetValue", &PyClass::SetValue, py::arg("arc"))
      .def("Next", &PyClass::Next)
      .def("Reset", &PyClass::Reset)
      .def("Seek", &PyClass::Seek, py::arg("a"))
      .def("Position", &PyClass::Position)
      .def("Flags", &PyClass::Flags)
      .def("SetFlags", &PyClass::SetFlags);
}

template <typename A>
void pybind_fst_impl(py::module& m, const std::string& class_name,
                            const std::string& class_help_doc = "") {
  using PyClass = fst::Fst<A>;
  using Arc = typename PyClass::Arc;

  py::class_<PyClass, PyFst<A>>(m, class_name.c_str(), class_help_doc.c_str())
      .def(py::init<>());
}

template <typename A>
void pybind_expanded_fst_impl(py::module& m, const std::string& class_name,
                            const std::string& class_help_doc = "") {
  using PyClass = fst::ExpandedFst<A>;
  using Arc = typename PyClass::Arc;

  py::class_<PyClass, fst::Fst<A>, PyExpandedFst<A>>(m, class_name.c_str(), class_help_doc.c_str())
      .def(py::init<>());
}

template <typename A>
void pybind_mutable_fst_impl(py::module& m, const std::string& class_name,
                            const std::string& class_help_doc = "") {
  using PyClass = fst::MutableFst<A>;
  using Arc = typename PyClass::Arc;

  py::class_<PyClass, fst::ExpandedFst<A>, PyMutableFst<A>>(m, class_name.c_str(), class_help_doc.c_str())
      .def(py::init<>());
}

template <typename A>
void pybind_vector_fst_impl(py::module& m, const std::string& class_name,
                            const std::string& class_help_doc = "") {
  using PyClass = fst::VectorFst<A>;
  using Arc = typename PyClass::Arc;
  using Weight = typename PyClass::Arc::Weight;
  using StateId = typename PyClass::StateId;
  using State = typename PyClass::State;

    auto c = py::class_<PyClass, fst::MutableFst<A>>(m, class_name.c_str(), class_help_doc.c_str());
    c
      .def(py::init<>())
      .def(py::init<const fst::Fst<Arc>&>(), py::arg("fst"))
      .def(py::init<const PyClass&, bool>(), py::arg("fst"),
           py::arg("safe") = false)
      .def("Start", &PyClass::Start)
      .def("Final", &PyClass::Final, py::arg("s"))
      .def("SetStart", &PyClass::SetStart, py::arg("s"))
      .def("SetFinal", &PyClass::SetFinal, py::arg("s"), py::arg("weight"))
      .def("SetProperties", &PyClass::SetProperties, py::arg("props"),
           py::arg("mask"))
      .def("AddState", (StateId (PyClass::*)()) & PyClass::AddState)
      .def("AddArc", (void (PyClass::*)(StateId, const Arc&))(&PyClass::AddArc), py::arg("s"), py::arg("arc"))
      .def("DeleteStates", (void (PyClass::*)(const std::vector<StateId>&)) &
                               PyClass::DeleteStates,
           py::arg("dstates"))
      .def("DeleteStates", (void (PyClass::*)()) & PyClass::DeleteStates,
           "Delete all states")
      .def("DeleteArcs",
           (void (PyClass::*)(StateId, size_t)) & PyClass::DeleteArcs,
           py::arg("state"), py::arg("n"))
      .def("DeleteArcs", (void (PyClass::*)(StateId)) & PyClass::DeleteArcs,
           py::arg("s"))
      .def("ReserveStates", &PyClass::ReserveStates, py::arg("s"))
      .def("ReserveArcs", &PyClass::ReserveArcs, py::arg("s"), py::arg("n"))
      .def("InputSymbols", &PyClass::InputSymbols,
           "Returns input label symbol table; return nullptr if not "
           "specified.",
           py::return_value_policy::reference)
      .def("OutputSymbols", &PyClass::OutputSymbols,
           "Returns output label symbol table; return nullptr if not "
           "specified.",
           py::return_value_policy::reference)
      .def("MutableInputSymbols", &PyClass::MutableInputSymbols,
           "Returns input label symbol table; return nullptr if not "
           "specified.",
           py::return_value_policy::reference)
      .def("MutableOutputSymbols", &PyClass::MutableOutputSymbols,
           "Returns output label symbol table; return nullptr if not "
           "specified.",
           py::return_value_policy::reference)
      .def("SetInputSymbols", &PyClass::SetInputSymbols, py::arg("isyms"))
      .def("SetOutputSymbols", &PyClass::SetOutputSymbols, py::arg("osyms"))
      .def("NumStates", &PyClass::NumStates)
      .def("NumArcs", &PyClass::NumArcs, py::arg("s"))
      .def("NumInputEpsilons", &PyClass::NumInputEpsilons, py::arg("s"))
      .def("NumOutputEpsilons", &PyClass::NumOutputEpsilons, py::arg("s"))
      .def("Properties", &PyClass::Properties, py::arg("mask"), py::arg("test"))
      .def("Type", &PyClass::Type, "FST typename",
           py::return_value_policy::reference)
      .def("Copy", &PyClass::Copy,
           "Get a copy of this VectorFst. See Fst<>::Copy() for further "
           "doc.",
           py::arg("safe") = false, py::return_value_policy::take_ownership)
      .def_static("Read",
                  // clang-format off
            overload_cast_<std::istream&, const fst::FstReadOptions&>()(&PyClass::Read),
                  // clang-format on
                  "Reads a VectorFst from an input stream, returning nullptr "
                  "on error.",
                  py::arg("strm"), py::arg("opts"),
                  py::return_value_policy::take_ownership,
      py::call_guard<py::gil_scoped_release>())
      .def_static("Read", overload_cast_<const std::string&>()(&PyClass::Read),
                  "Read a VectorFst from a file, returning nullptr on error; "
                  "empty "
                  "filename reads from standard input.",
                  py::arg("filename"), py::return_value_policy::reference,
      py::call_guard<py::gil_scoped_release>())
      .def("Write",
           // clang-format off
            (bool (PyClass::*)(std::ostream&, const fst::FstWriteOptions&)const)&PyClass::Write,
           // clang-format on
           "Writes an FST to an output stream; returns false on error.",
           py::arg("strm"), py::arg("opts"),
      py::call_guard<py::gil_scoped_release>())
      .def("Write",
           (bool (PyClass::*)(const std::string&) const) & PyClass::Write,
           "Writes an FST to a file; returns false on error; an empty\n"
           "filename results in writing to standard output.",
           py::arg("filename"),
      py::call_guard<py::gil_scoped_release>())
      .def_static("WriteFst", &PyClass::template WriteFst<PyClass>,
                  py::arg("fst"), py::arg("strm"), py::arg("opts"))
      .def("InitStateIterator", &PyClass::InitStateIterator,
           "For generic state iterator construction (not normally called "
           "directly by users). Does not copy the FST.",
           py::arg("data"))
      .def("InitArcIterator", &PyClass::InitArcIterator,
           "For generic arc iterator construction (not normally called "
           "directly by users). Does not copy the FST.",
           py::arg("s"), py::arg("data"))
      .def("Connect", [](PyClass* f){
             fst::Connect<Arc>(f);
      })
       /*.def("to_pynini", [](PyClass& f){

        auto pywrapfst_mod = py::module_::import("pywrapfst");

        VectorFstStruct py_fst;
        fst::script::MutableFstClass vf(f);
        py_fst.__pyx_base._mfst = std::shared_ptr<fst::script::MutableFstClass>(&vf);
        return py_fst;
      }, py::return_value_policy::take_ownership)*/
      .def_static("from_pynini", [](py::object fst) {
        auto pywrapfst_mod = py::module_::import("pywrapfst");
        auto ptr = reinterpret_cast<VectorFstStruct*>(fst.ptr());
        auto mf = ptr->__pyx_base._mfst->GetMutableFst<A>();
            PyClass *vf = new PyClass(*mf);
            return vf;
      },
           py::return_value_policy::reference)
      .def("write_to_string", [](const PyClass& f){
             std::ostringstream os;
             fst::FstWriteOptions opts;
             opts.stream_write = true;
             f.Write(os, opts);
            return py::bytes(os.str());
      })
      .def_static("from_string", [](const std::string &bytes){
            std::istringstream str(bytes);

            fst::FstHeader hdr;
            if (!hdr.Read(str, "<unspecified>"))
            KALDI_ERR << "Reading FST: error reading FST header";
            fst::FstReadOptions ropts("<unspecified>", &hdr);
            PyClass *f = PyClass::Read(str, ropts);
            return f;
      },
            py::arg("bytes"),
           py::return_value_policy::reference)
      .def(py::pickle(
        [](const PyClass &f) { // __getstate__
            /* Return a tuple that fully encodes the state of the object */
             std::ostringstream os;
             fst::FstWriteOptions opts;
             opts.stream_write = true;
             f.Write(os, opts);
            return py::make_tuple(
                    py::bytes(os.str()));
        },
        [](py::tuple t) { // __setstate__
            if (t.size() != 1)
                throw std::runtime_error("Invalid state!");

            std::istringstream str(t[0].cast<std::string>());

            fst::FstHeader hdr;
            if (!hdr.Read(str, "<unspecified>"))
            KALDI_ERR << "Reading FST: error reading FST header";
            fst::FstReadOptions ropts("<unspecified>", &hdr);
            PyClass *f = PyClass::Read(str, ropts);

            return f;
        }
    ));
}


template <typename A>
void pybind_lattice_fst_impl(py::module& m, const std::string& class_name,
                            const std::string& class_help_doc = "") {
  using PyClass = fst::VectorFst<A>;
  using Arc = typename PyClass::Arc;
  using Weight = typename PyClass::Weight;
  using StateId = typename PyClass::StateId;
  using State = typename PyClass::State;

    auto c = py::class_<PyClass, fst::MutableFst<A>>(m, class_name.c_str(), class_help_doc.c_str());
    c
      .def(py::init<>())
      .def(py::init<const fst::Fst<Arc>&>(), py::arg("fst"))
      .def(py::init<const PyClass&, bool>(), py::arg("fst"),
           py::arg("safe") = false)
      .def("Start", &PyClass::Start)
      .def("Final", &PyClass::Final, py::arg("s"))
      .def("SetStart", &PyClass::SetStart, py::arg("s"))
      .def("SetFinal", &PyClass::SetFinal, py::arg("s"), py::arg("weight"))
      .def("SetProperties", &PyClass::SetProperties, py::arg("props"),
           py::arg("mask"))
      .def("AddState", (StateId (PyClass::*)()) & PyClass::AddState)
      .def("AddArc", (void (PyClass::*)(StateId, const Arc&))(&PyClass::AddArc), py::arg("s"), py::arg("arc"))
      .def("DeleteStates", (void (PyClass::*)(const std::vector<StateId>&)) &
                               PyClass::DeleteStates,
           py::arg("dstates"))
      .def("DeleteStates", (void (PyClass::*)()) & PyClass::DeleteStates,
           "Delete all states")
      .def("DeleteArcs",
           (void (PyClass::*)(StateId, size_t)) & PyClass::DeleteArcs,
           py::arg("state"), py::arg("n"))
      .def("DeleteArcs", (void (PyClass::*)(StateId)) & PyClass::DeleteArcs,
           py::arg("s"))
      .def("ReserveStates", &PyClass::ReserveStates, py::arg("s"))
      .def("ReserveArcs", &PyClass::ReserveArcs, py::arg("s"), py::arg("n"))
      .def("InputSymbols", &PyClass::InputSymbols,
           "Returns input label symbol table; return nullptr if not "
           "specified.",
           py::return_value_policy::reference)
      .def("OutputSymbols", &PyClass::OutputSymbols,
           "Returns output label symbol table; return nullptr if not "
           "specified.",
           py::return_value_policy::reference)
      .def("MutableInputSymbols", &PyClass::MutableInputSymbols,
           "Returns input label symbol table; return nullptr if not "
           "specified.",
           py::return_value_policy::reference)
      .def("MutableOutputSymbols", &PyClass::MutableOutputSymbols,
           "Returns output label symbol table; return nullptr if not "
           "specified.",
           py::return_value_policy::reference)
      .def("SetInputSymbols", &PyClass::SetInputSymbols, py::arg("isyms"))
      .def("SetOutputSymbols", &PyClass::SetOutputSymbols, py::arg("osyms"))
      .def("NumStates", &PyClass::NumStates)
      .def("NumArcs", &PyClass::NumArcs, py::arg("s"))
      .def("NumInputEpsilons", &PyClass::NumInputEpsilons, py::arg("s"))
      .def("NumOutputEpsilons", &PyClass::NumOutputEpsilons, py::arg("s"))
      .def("Properties", &PyClass::Properties, py::arg("mask"), py::arg("test"))
      .def("Type", &PyClass::Type, "FST typename",
           py::return_value_policy::reference)
      .def("Copy", &PyClass::Copy,
           "Get a copy of this VectorFst. See Fst<>::Copy() for further "
           "doc.",
           py::arg("safe") = false, py::return_value_policy::take_ownership)
      .def_static("Read",
                  // clang-format off
            overload_cast_<std::istream&, const fst::FstReadOptions&>()(&PyClass::Read),
                  // clang-format on
                  "Reads a VectorFst from an input stream, returning nullptr "
                  "on error.",
                  py::arg("strm"), py::arg("opts"),
                  py::return_value_policy::take_ownership)
      .def_static("Read", overload_cast_<const std::string&>()(&PyClass::Read),
                  "Read a VectorFst from a file, returning nullptr on error; "
                  "empty "
                  "filename reads from standard input.",
                  py::arg("filename"), py::return_value_policy::take_ownership)
      .def("Write",
           // clang-format off
            (bool (PyClass::*)(std::ostream&, const fst::FstWriteOptions&)const)&PyClass::Write,
           // clang-format on
           "Writes an FST to an output stream; returns false on error.",
           py::arg("strm"), py::arg("opts"))
      .def("Write",
           (bool (PyClass::*)(const std::string&) const) & PyClass::Write,
           "Writes an FST to a file; returns false on error; an empty\n"
           "filename results in writing to standard output.",
           py::arg("filename"))
      .def_static("WriteFst", &PyClass::template WriteFst<PyClass>,
                  py::arg("fst"), py::arg("strm"), py::arg("opts"))
      .def("InitStateIterator", &PyClass::InitStateIterator,
           "For generic state iterator construction (not normally called "
           "directly by users). Does not copy the FST.",
           py::arg("data"))
      .def("InitArcIterator", &PyClass::InitArcIterator,
           "For generic arc iterator construction (not normally called "
           "directly by users). Does not copy the FST.",
           py::arg("s"), py::arg("data"))
      .def("Connect", [](PyClass* f){
          py::gil_scoped_release release;
             fst::Connect<Arc>(f);
      })
      .def("ScaleLattice", [](PyClass* f, BaseFloat acoustic_scale = 1.0,
            BaseFloat lm_scale = 1.0){
          py::gil_scoped_release release;

            if (acoustic_scale != 1.0 || lm_scale != 1.0)
                fst::ScaleLattice(fst::LatticeScale(lm_scale, acoustic_scale), f);
      },
                  py::arg("acoustic_scale") = 1.0,
                  py::arg("lm_scale") = 1.0)
      .def("TopSort", [](PyClass* f){
          py::gil_scoped_release release;
            kaldi::uint64 props = f->Properties(fst::kFstProperties, false);
            if (!(props & fst::kTopSorted)) {
                if (!fst::TopSort(f))
                KALDI_ERR << "Cycles detected in lattice.";
            }
      })
      .def("write_to_string", [](const PyClass& f){
             std::ostringstream os;
             fst::FstWriteOptions opts;
             opts.stream_write = true;
             f.Write(os, opts);
            return py::bytes(os.str());
      })
      .def_static("from_string", [](const std::string &bytes){
            std::istringstream str(bytes);

            fst::FstHeader hdr;
            if (!hdr.Read(str, "<unspecified>"))
            KALDI_ERR << "Reading FST: error reading FST header";
            fst::FstReadOptions ropts("<unspecified>", &hdr);
            PyClass *f = PyClass::Read(str, ropts);
            return f;
      },
            py::arg("bytes"),
           py::return_value_policy::reference);
}

template <typename A>
void pybind_const_fst_impl(py::module& m, const std::string& class_name,
                            const std::string& class_help_doc = "") {
  using PyClass = fst::ConstFst<A>;
  using Arc = typename PyClass::Arc;
  using StateId = typename PyClass::StateId;

  py::class_<PyClass, fst::ExpandedFst<A>>(m, class_name.c_str(), class_help_doc.c_str())
      .def(py::init<>())
      .def(py::init<const fst::Fst<Arc>&>(), py::arg("fst"))
      .def(py::init<const PyClass&, bool>(), py::arg("fst"),
           py::arg("safe") = false)
      .def("Start", &PyClass::Start)
      .def("Final", &PyClass::Final, py::arg("s"))
      .def("InputSymbols", &PyClass::InputSymbols,
           "Returns input label symbol table; return nullptr if not "
           "specified.",
           py::return_value_policy::reference)
      .def("OutputSymbols", &PyClass::OutputSymbols,
           "Returns output label symbol table; return nullptr if not "
           "specified.",
           py::return_value_policy::reference)
      .def("NumStates", &PyClass::NumStates)
      .def("NumArcs", &PyClass::NumArcs, py::arg("s"))
      .def("NumInputEpsilons", &PyClass::NumInputEpsilons, py::arg("s"))
      .def("NumOutputEpsilons", &PyClass::NumOutputEpsilons, py::arg("s"))
      .def("Properties", &PyClass::Properties, py::arg("mask"), py::arg("test"))
      .def("Type", &PyClass::Type, "FST typename",
           py::return_value_policy::reference)
      .def("Copy", &PyClass::Copy,
           "Get a copy of this VectorFst. See Fst<>::Copy() for further "
           "doc.",
           py::arg("safe") = false, py::return_value_policy::take_ownership)
      .def_static("Read",
                  // clang-format off
            overload_cast_<std::istream&, const fst::FstReadOptions&>()(&PyClass::Read),
                  // clang-format on
                  "Reads a VectorFst from an input stream, returning nullptr "
                  "on error.",
                  py::arg("strm"), py::arg("opts"),
                  py::return_value_policy::take_ownership)
      .def_static("Read", overload_cast_<const std::string&>()(&PyClass::Read),
                  "Read a VectorFst from a file, returning nullptr on error; "
                  "empty "
                  "filename reads from standard input.",
                  py::arg("filename"), py::return_value_policy::take_ownership)
      .def("Write",
           // clang-format off
            (bool (PyClass::*)(std::ostream&, const fst::FstWriteOptions&)const)&PyClass::Write,
           // clang-format on
           "Writes an FST to an output stream; returns false on error.",
           py::arg("strm"), py::arg("opts"))
      .def("Write",
           (bool (PyClass::*)(const std::string&) const) & PyClass::Write,
           "Writes an FST to a file; returns false on error; an empty\n"
           "filename results in writing to standard output.",
           py::arg("filename"))
      .def_static("WriteFst", &PyClass::template WriteFst<PyClass>,
                  py::arg("fst"), py::arg("strm"), py::arg("opts"))
      .def("InitStateIterator", &PyClass::InitStateIterator,
           "For generic state iterator construction (not normally called "
           "directly by users). Does not copy the FST.",
           py::arg("data"))
      .def("InitArcIterator", &PyClass::InitArcIterator,
           "For generic arc iterator construction (not normally called "
           "directly by users). Does not copy the FST.",
           py::arg("s"), py::arg("data"))
      .def("write_to_string", [](const PyClass& f){
             std::ostringstream os;
             fst::FstWriteOptions opts;
             opts.stream_write = true;
             f.Write(os, opts);
            return py::bytes(os.str());
      })
      .def_static("from_string", [](const std::string &bytes){
            std::istringstream str(bytes);

            fst::FstHeader hdr;
            if (!hdr.Read(str, "<unspecified>"))
            KALDI_ERR << "Reading FST: error reading FST header";
            fst::FstReadOptions ropts("<unspecified>", &hdr);
            PyClass *f = PyClass::Read(str, ropts);
            return f;
      },
            py::arg("bytes"),
           py::return_value_policy::reference);
}

void pybind_fst_symbol_table(py::module& m);

void pybind_fstext_context_fst(py::module &);
void pybind_fstext_deterministic_fst(py::module &);
void pybind_fstext_deterministic_lattice(py::module &);
void pybind_fstext_determinize_star(py::module &);
void pybind_fstext_epsilon_property(py::module &);
void pybind_fstext_factor(py::module &);
void pybind_fstext_fstext_utils(py::module &);
void pybind_fstext_grammar_context_fst(py::module &);
void pybind_fstext_lattice_utils(py::module &);
void pybind_fstext_pre_determinize(py::module &);
void pybind_fstext_prune_special(py::module &);
void pybind_fstext_rand_fst(py::module &);
void pybind_fstext_remove_eps_local(py::module &);
void pybind_kaldi_fst_io(py::module &);
void init_fstext(py::module &);
#endif  // KALPY_PYBIND_FSTEXT_H_
