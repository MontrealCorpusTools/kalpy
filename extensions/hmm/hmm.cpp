#include "hmm/pybind_hmm.h"
#include "hmm/hmm-topology.h"
#include "hmm/transition-model.h"

using namespace kaldi;

void init_hmm(py::module &_m) {
  py::module m = _m.def_submodule("hmm", "hmm pybind for Kaldi");
    {
  using PyClass = HmmTopology;
  auto hmm = py::class_<PyClass>(
      m, "HmmTopology",
      "A class for storing topology information for phones. See `hmm` for "
      "context. This object is sometimes accessed in a file by itself, but "
      "more often as a class member of the Transition class (this is for "
      "convenience to reduce the number of files programs have to access).");

  using State = HmmTopology::HmmState;
  py::class_<State>(
      hmm, "HmmState",
      "A structure defined inside HmmTopology to represent a HMM state.")
      .def(py::init<>())
      .def(py::init<int>(), py::arg("pdf_class"))
      .def(py::init<int, int>(), py::arg("forward_pdf_class"),
           py::arg("self_loop_pdf_class"))
      .def_readwrite("forward_pdf_class", &State::forward_pdf_class)
      .def_readwrite("self_loop_pdf_class", &State::self_loop_pdf_class)
      .def_readwrite("transitions", &State::transitions)
      .def("__eq__", [](const State& s1, const State& s2) { return s1 == s2; })
      .def("__str__", [](const State& s) {
        std::ostringstream os;
        os << "forward_pdf_class: " << s.forward_pdf_class << "\n";
        os << "self_loop_pdf_class: " << s.self_loop_pdf_class << "\n";
        return os.str();
      });

  hmm.def(py::init<>())
      .def("Read", &PyClass::Read, py::arg("is"), py::arg("binary"))
      .def("Write", &PyClass::Write, py::arg("os"), py::arg("binary"))
      .def("Check", &PyClass::Check,
           "Checks that the object is valid, and throw exception otherwise.")
      .def("IsHmm", &PyClass::IsHmm,
           "Returns true if this HmmTopology is really 'hmm-like', i.e. the "
           "pdf-class on the self-loops and forward transitions of all states "
           "are identical. [note: in HMMs, the densities are associated with "
           "the states.] We have extended this to support 'non-hmm-like' "
           "topologies (where those pdf-classes are different), in order to "
           "make for more compact decoding graphs in our so-called 'chain "
           "models' (AKA lattice-free MMI), where we use 1-state topologies "
           "that have different pdf-classes for the self-loop and the forward "
           "transition. Note that we always use the 'reorder=true' option so "
           "the 'forward transition' actually comes before the self-loop.")
      .def("TopologyForPhone", &PyClass::TopologyForPhone,
           "Returns the topology entry (i.e. vector of HmmState) for this "
           "phone; will throw exception if phone not covered by the topology.",
           py::arg("phone"), py::return_value_policy::reference)
      .def("NumPdfClasses", &PyClass::NumPdfClasses,
           "Returns the number of 'pdf-classes' for this phone; throws "
           "exception if phone not covered by this topology.",
           py::arg("phone"))
      .def("GetPhones", &PyClass::GetPhones,
           "Returns a reference to a sorted, unique list of phones covered by "
           "the topology (these phones will be positive integers, and usually "
           "contiguous and starting from one but the toolkit doesn't assume "
           "they are contiguous).",
           py::return_value_policy::reference)
      .def("GetPhoneToNumPdfClasses",
           [](const PyClass& topo) -> std::vector<int> {
             std::vector<int> phone2num_pdf_classes;
             topo.GetPhoneToNumPdfClasses(&phone2num_pdf_classes);
             return phone2num_pdf_classes;
           },
           "Outputs a vector of int32, indexed by phone, that gives the number "
           "of \ref pdf_class pdf-classes for the phones; this is used by "
           "tree-building code such as BuildTree().")
      .def("MinLength", &PyClass::MinLength,
           "Returns the minimum number of frames it takes to traverse this "
           "model for this phone: e.g. 3 for the normal HMM topology.",
           py::arg("phone"))
      .def("__eq__",
           [](const PyClass& t1, const PyClass& t2) { return t1 == t2; })
      .def("__str__", [](const PyClass& topo) {
        std::ostringstream os;
        bool binary = false;
        topo.Write(os, binary);
        return os.str();
      });
}
{
  using PyClass = TransitionModel;
  py::class_<PyClass, TransitionInformation>(m, "TransitionModel")
      .def(py::init<>())
      .def(py::init<const ContextDependencyInterface&, const HmmTopology&>(),
           "Initialize the object [e.g. at the start of training]. The class "
           "keeps a copy of the HmmTopology object, but not the "
           "ContextDependency object.",
           py::arg("ctx_dep"), py::arg("hmm_topo"))
     .def_static("read_from_file", [](std::string file_path) {

               TransitionModel trans_model;
               ReadKaldiObject(file_path, &trans_model);
               return &trans_model;
          }, py::return_value_policy::reference)
      .def("Read", &PyClass::Read, py::arg("is"), py::arg("binary"))
      .def("Write", &PyClass::Write, py::arg("os"), py::arg("binary"))
      .def("GetTopo", &PyClass::GetTopo, py::return_value_policy::reference)
      .def("TupleToTransitionState", &PyClass::TupleToTransitionState,
           py::arg("phone"), py::arg("hmm_state"), py::arg("pdf"),
           py::arg("self_loop_pdf"))
      .def("PairToTransitionId", &PyClass::PairToTransitionId,
           py::arg("trans_state"), py::arg("trans_index"))
      .def("TransitionIdToTransitionState",
           &PyClass::TransitionIdToTransitionState, py::arg("trans_id"))
      .def("TransitionIdToTransitionIndex",
           &PyClass::TransitionIdToTransitionIndex, py::arg("trans_id"))
      .def("TransitionStateToPhone", &PyClass::TransitionStateToPhone,
           py::arg("trans_state"))
      .def("TransitionStateToHmmState", &PyClass::TransitionStateToHmmState,
           py::arg("trans_state"))
      .def("TransitionStateToForwardPdfClass",
           &PyClass::TransitionStateToForwardPdfClass, py::arg("trans_state"))
      .def("TransitionStateToSelfLoopPdfClass",
           &PyClass::TransitionStateToSelfLoopPdfClass, py::arg("trans_state"))
      .def("TransitionStateToForwardPdf", &PyClass::TransitionStateToForwardPdf,
           py::arg("trans_state"))
      .def("TransitionStateToSelfLoopPdf",
           &PyClass::TransitionStateToSelfLoopPdf, py::arg("trans_state"))
      .def("SelfLoopOf", &PyClass::SelfLoopOf,
           "returns the self-loop transition-id, or zero if this state "
           "doesn't have a self-loop.",
           py::arg("trans_state"))
      .def("TransitionIdToPdf", &PyClass::TransitionIdToPdf,
           py::arg("trans_id"))
      .def("TransitionIdToPdfFast", &PyClass::TransitionIdToPdfFast,
           "TransitionIdToPdfFast is as TransitionIdToPdf but skips an "
           "assertion (unless we're in paranoid mode).",
           py::arg("trans_id"))
      .def("TransitionIdToPhone", &PyClass::TransitionIdToPhone,
           py::arg("trans_id"))
      .def("TransitionIdToPdfClass", &PyClass::TransitionIdToPdfClass,
           py::arg("trans_id"))
      .def("TransitionIdToHmmState", &PyClass::TransitionIdToHmmState,
           py::arg("trans_id"))
      .def("IsFinal", &PyClass::IsFinal,
           "returns true if this trans_id goes to the final state (which is "
           "bound to be nonemitting).",
           py::arg("trans_id"))
      .def("IsSelfLoop", &PyClass::IsSelfLoop,
           "return true if this trans_id corresponds to a self-loop.",
           py::arg("trans_id"))
      .def("NumTransitionIds", &PyClass::NumTransitionIds,
           "Returns the total number of transition-ids (note, these are "
           "one-based).")
      .def("NumTransitionIndices", &PyClass::NumTransitionIndices,
           "Returns the number of transition-indices for a particular "
           "transition-state. Note: 'Indices' is the plural of 'index'. "
           "Index is not the same as 'id', here. A transition-index is a "
           "zero-based offset into the transitions out of a particular "
           "transition state.",
           py::arg("trans_state"))
      .def("NumTransitionStates", &PyClass::NumTransitionStates,
           "Returns the total number of transition-states (note, these are "
           "one-based).")
      .def("NumPdfs", &PyClass::NumPdfs,
           "NumPdfs() actually returns the highest-numbered pdf we ever saw, "
           "plus one. In normal cases this should equal the number of pdfs "
           "in the system, but if you initialized this object with fewer "
           "than all the phones, and it happens that an unseen phone has the "
           "highest-numbered pdf, this might be different.")
      .def("NumPhones", &PyClass::NumPhones,
           "This loops over the tuples and finds the highest phone index "
           "present. If the FST symbol table for the phones is created in "
           "the expected way, i.e.: starting from 1 (<eps> is 0) and "
           "numbered contiguously till the last phone, this will be the "
           "total number of phones.")
      .def("GetPhones", &PyClass::GetPhones,
           "Returns a sorted, unique list of phones.",
           py::return_value_policy::reference)
      .def("GetTransitionProb", &PyClass::GetTransitionProb,
           py::arg("trans_id"))
      .def("GetTransitionLogProb", &PyClass::GetTransitionLogProb,
           py::arg("trans_id"))
      .def("GetTransitionLogProbIgnoringSelfLoops",
           &PyClass::GetTransitionLogProbIgnoringSelfLoops,
           "Returns the log-probability of a particular non-self-loop "
           "transition after subtracting the probability mass of the "
           "self-loop and renormalizing; will crash if called on a "
           "self-loop.  Specifically: for non-self-loops it returns the log "
           "of (that prob divided by (1 minus "
           "self-loop-prob-for-that-state)).",
           py::arg("trans_id"))
      .def("GetNonSelfLoopLogProb", &PyClass::GetNonSelfLoopLogProb,
           "Returns the log-prob of the non-self-loop probability mass for "
           "this transition state. (you can get the self-loop prob, if a "
           "self-loop exists, by calling "
           "GetTransitionLogProb(SelfLoopOf(trans_state)).",
           py::arg("trans_id"))
      .def("Print", &PyClass::Print, py::arg("os"), py::arg("phone_names"),
           py::arg("occs") = nullptr)
      .def("Compatible", &PyClass::Compatible,
           "returns true if all the integer class members are identical (but "
           "does not compare the transition probabilities.")
      .def("Print", &PyClass::Print,
           "Print will print the transition model in a human-readable way, "
           "for purposes of human inspection.  The 'occs' are optional (they "
           "are indexed by pdf-id).",
           py::arg("os"), py::arg("phone_names"), py::arg("occs") = nullptr)
      .def("__str__",
           [](const PyClass& mdl) {
             std::ostringstream os;
             bool binary = false;
             mdl.Write(os, binary);
             return os.str();
           })
      // TODO(fangjun): the following methods are not wrapped yet:
      // MleUpdate, MapUpdate, InitStats, Accumulate
      // Wrap them when needed
      ;
  m.def("GetPdfsForPhones",
        [](const TransitionModel& trans_model, const std::vector<int32>& phones)
            -> std::pair<bool, std::vector<int>> {
              std::vector<int> pdfs;
              bool is_succeeded = GetPdfsForPhones(trans_model, phones, &pdfs);
              return std::make_pair(is_succeeded, pdfs);
            },
        "Return a pair of [is_succeeded, pdfs]"
        "\n"
        "Works out which pdfs might correspond to the given phones. Will "
        "return true if these pdfs correspond *just* to these phones, false if "
        "these pdfs are also used by other phones."
        "\n"
        "trans_model [in] Transition-model used to work out this information"
        "\n"
        "phones [in] A sorted, uniq vector that represents a set of phones"
        "\n"
        "pdfs [out] Will be set to a sorted, uniq list of pdf-ids that "
        "correspond to one of this set of phones."
        "\n"
        "is_succeeded is true if all of the pdfs output to 'pdfs' correspond "
        "to phones from just this set (false if they may be shared with phones "
        "outside this set).",
        py::arg("trans_model"), py::arg("phones"));

  m.def(
      "GetPhonesForPdfs",
      [](const TransitionModel& trans_model,
         const std::vector<int32>& pdfs) -> std::pair<bool, std::vector<int>> {
        std::vector<int> phones;
        bool is_succeeded = GetPhonesForPdfs(trans_model, pdfs, &phones);
        return std::make_pair(is_succeeded, phones);
      },
      "Return a pair of [is_succeeded, phones]",
      "\n"
      "Works out which phones might correspond to the given pdfs. Similar "
      "to GetPdfsForPhones(, ,)",
      py::arg("trans_model"), py::arg("pdfs"));
}

}
