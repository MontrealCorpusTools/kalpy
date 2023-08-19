#include "itf/pybind_itf.h"
#include "itf/clusterable-itf.h"
#include "itf/context-dep-itf.h"
#include "itf/decodable-itf.h"
#include "itf/online-feature-itf.h"
#include "itf/optimizable-itf.h"
#include "itf/options-itf.h"
#include "itf/transition-information.h"


using namespace kaldi;

void pybind_context_dep_itf(py::module& m) {
  {
    using PyClass = ContextDependencyInterface;
    py::class_<PyClass>(m, "ContextDependencyInterface",
                        "context-dep-itf.h provides a link between the "
                        "tree-building code in ../tree/, and the FST code in "
                        "../fstext/ (particularly, ../fstext/context-dep.h).  "
                        "It is an abstract interface that describes an object "
                        "that can map from a phone-in-context to a sequence of "
                        "integer leaf-ids.")
        .def("ContextWidth", &PyClass::ContextWidth,
             "ContextWidth() returns the value N (e.g. 3 for triphone models) "
             "that says how many phones are considered for computing context.")
        .def("CentralPosition", &PyClass::CentralPosition,
             "Central position P of the phone context, in 0-based numbering, "
             "e.g. P = 1 for typical triphone system.  We have to see if we "
             "can do without this function.")
        .def("Compute",
             [](const PyClass& ctx, const std::vector<int32>& phoneseq,
                int32 pdf_class) -> std::vector<int> {
               std::vector<int> res(2, 0);
               res[0] = ctx.Compute(phoneseq, pdf_class, &res[1]);
               return res;
             },
             "Return a pair [is_succeeded, pdf_id], where is_succeeded is 0 "
             "if expansion somehow failed."
             "\n"
             "The 'new' Compute interface.  For typical topologies, pdf_class "
             "would be 0, 1, 2."
             "\n"
             "'Compute' is the main function of this interface, that takes a "
             "sequence of N phones (and it must be N phones), possibly "
             "including epsilons (symbol id zero) but only at positions other "
             "than P [these represent unknown phone context due to end or "
             "begin of sequence].  We do not insist that Compute must always "
             "output (into stateseq) a nonempty sequence of states, but we "
             "anticipate that stateseq will always be nonempty at output "
             "in typical use cases.  'Compute' returns false if expansion "
             "somehow failed.  Normally the calling code should raise an "
             "exception if this happens.  We can define a different interface "
             "later in order to handle other kinds of information-- the "
             "underlying data-structures from event-map.h are very flexible.",
             py::arg("phoneseq"), py::arg("pdf_class"))
        .def("GetPdfInfo",
             [](const PyClass* ctx,
                const std::vector<int32>& phones,          // list of phones
                const std::vector<int32>& num_pdf_classes  // indexed by phone,
                ) {
               std::vector<std::vector<std::pair<int, int>>> pdf_info;
               ctx->GetPdfInfo(phones, num_pdf_classes, &pdf_info);
               return pdf_info;
             },
             "GetPdfInfo returns a vector indexed by pdf-id, saying for each "
             "pdf which pairs of (phone, pdf-class) it can correspond to.  "
             "(Usually just one). c.f. hmm/hmm-topology.h for meaning of "
             "pdf-class. This is the old, simpler interface of GetPdfInfo(), "
             "and that this one can only be called if the HmmTopology object's "
             "IsHmm() function call returns true.",
             py::arg("phones"), py::arg("num_pdf_classes"))
        .def("GetPdfInfo",
             [](const PyClass* ctx, const std::vector<int32>& phones,
                const std::vector<std::vector<std::pair<int32, int32>>>&
                    pdf_class_pairs) {
               std::vector<std::vector<std::vector<std::pair<int, int>>>>
                   pdf_info;
               ctx->GetPdfInfo(phones, pdf_class_pairs, &pdf_info);
               return pdf_info;
             },
             "This function outputs information about what possible pdf-ids "
             "can be generated for HMM-states; it covers the general case "
             "where the self-loop pdf-class may be different from the "
             "forward-transition pdf-class, so we are asking not about the set "
             "of possible pdf-ids for a given (phone, pdf-class), but the set "
             "of possible ordered pairs (forward-transition-pdf, "
             "self-loop-pdf) for a given (phone, forward-transition-pdf-class, "
             "self-loop-pdf-class). Note: 'phones' is a list of integer ids of "
             "phones, and 'pdf-class-pairs', indexed by phone, is a list of "
             "pairs (forward-transition-pdf-class, self-loop-pdf-class) that "
             "we can have for that phone. The output 'pdf_info' is indexed "
             "first by phone and then by the same index that indexes each "
             "element of 'pdf_class_pairs', and tells us for each pair in "
             "'pdf_class_pairs', what is the list of possible "
             "(forward-transition-pdf-id, self-loop-pdf-id) that we can have. "
             "This is less efficient than the other version of GetPdfInfo().",
             py::arg("phones"), py::arg("pdf_class_pairs"))
        .def("NumPdfs", &PyClass::NumPdfs,
             "NumPdfs() returns the number of acoustic pdfs (they are numbered "
             "0.. NumPdfs()-1).")
        .def("Copy", &PyClass::Copy,
             "Returns pointer to new object which is copy of current one.",
             py::return_value_policy::take_ownership);
  }
}


void pybind_decodable_itf(py::module& m) {
  using PyClass = DecodableInterface;
  py::class_<PyClass>(m, "DecodableInterface",
                      R"doc(
    DecodableInterface provides a link between the (acoustic-modeling and
    feature-processing) code and the decoder.  The idea is to make this
    interface as small as possible, and to make it as agnostic as possible about
    the form of the acoustic model (e.g. don't assume the probabilities are a
    function of just a vector of floats), and about the decoder (e.g. don't
    assume it accesses frames in strict left-to-right order).  For normal
    models, without on-line operation, the "decodable" sub-class will just be a
    wrapper around a matrix of features and an acoustic model, and it will
    answer the question 'what is the acoustic likelihood for this index and this
    frame?'.

    For online decoding, where the features are coming in in real time, it is
    important to understand the IsLastFrame() and NumFramesReady() functions.
    There are two ways these are used: the old online-decoding code, in ../online/,
    and the new online-decoding code, in ../online2/.  In the old online-decoding
    code, the decoder would do:
    \code{.cc}
    for (int frame = 0; !decodable.IsLastFrame(frame); frame++) {
      // Process this frame
    }
    \endcode
   and the call to IsLastFrame would block if the features had not arrived yet.
   The decodable object would have to know when to terminate the decoding.  This
   online-decoding mode is still supported, it is what happens when you call, for
   example, LatticeFasterDecoder::Decode().

   We realized that this "blocking" mode of decoding is not very convenient
   because it forces the program to be multi-threaded and makes it complex to
   control endpointing.  In the "new" decoding code, you don't call (for example)
   LatticeFasterDecoder::Decode(), you call LatticeFasterDecoder::InitDecoding(),
   and then each time you get more features, you provide them to the decodable
   object, and you call LatticeFasterDecoder::AdvanceDecoding(), which does
   something like this:
   \code{.cc}
   while (num_frames_decoded_ < decodable.NumFramesReady()) {
     // Decode one more frame [increments num_frames_decoded_]
   }
   \endcode
   So the decodable object never has IsLastFrame() called.  For decoding where
   you are starting with a matrix of features, the NumFramesReady() function will
   always just return the number of frames in the file, and IsLastFrame() will
   return true for the last frame.

   For truly online decoding, the "old" online decodable objects in ../online/
   have a "blocking" IsLastFrame() and will crash if you call NumFramesReady().
   The "new" online decodable objects in ../online2/ return the number of frames
   currently accessible if you call NumFramesReady().  You will likely not need
   to call IsLastFrame(), but we implement it to only return true for the last
   frame of the file once we've decided to terminate decoding.)doc")
      .def("LogLikelihood", &PyClass::LogLikelihood,
           "Returns the log likelihood, which will be negated in the decoder. "
           "The 'frame' starts from zero.  You should verify that "
           "NumFramesReady() > frame before calling this.",
           py::arg("frame"), py::arg("index"))
      .def("IsLastFrame", &PyClass::IsLastFrame,
           R"doc(
  Returns true if this is the last frame.  Frames are zero-based, so the
  first frame is zero.  IsLastFrame(-1) will return false, unless the file
  is empty (which is a case that I'm not sure all the code will handle, so
  be careful).  Caution: the behavior of this function in an online setting
  is being changed somewhat.  In future it may return false in cases where
  we haven't yet decided to terminate decoding, but later true if we decide
  to terminate decoding.  The plan in future is to rely more on
  NumFramesReady(), and in future, IsLastFrame() would always return false
  in an online-decoding setting, and would only return true in a
  decoding-from-matrix setting where we want to allow the last delta or LDA
  features to be flushed out for compatibility with the baseline setup.
          )doc",
           py::arg("frame"))
      .def("NumFramesReady", &PyClass::NumFramesReady,
           R"doc(
  The call NumFramesReady() will return the number of frames currently available
  for this decodable object.  This is for use in setups where you don't want the
  decoder to block while waiting for input.  This is newly added as of Jan 2014,
  and I hope, going forward, to rely on this mechanism more than IsLastFrame to
  know when to stop decoding.
          )doc")
      .def("NumIndices", &PyClass::NumIndices,
           R"doc(
  Returns the number of states in the acoustic model
  (they will be indexed one-based, i.e. from 1 to NumIndices();
  this is for compatibility with OpenFst).
          )doc");
}


void pybind_clusterable_itf(py::module& m) {
  using PyClass = Clusterable;
  py::class_<PyClass>(m, "Clusterable")
      .def("Copy", &PyClass::Copy,
           "Return a copy of this object.")
      .def("Objf", &PyClass::Objf,
           "Return the objective function associated with the stats")
      .def("Normalizer", &PyClass::Normalizer,
           "Return the normalizer (typically, count) associated with the stats")
      .def("SetZero", &PyClass::SetZero,
           "Set stats to empty.")
      .def("Add", &PyClass::Add,
           "Add other stats.", py::arg("other"),
      py::call_guard<py::gil_scoped_release>())
      .def("Sub", &PyClass::Sub,
           "Subtract other stats", py::arg("other"))
      .def("Scale", &PyClass::Scale,
           "Scale the stats by a positive number f [not mandatory to supply this].", py::arg("f"))
      .def("Type", &PyClass::Type,
           "Return a string that describes the inherited type.")
      .def("Write", &PyClass::Write,
           "Write data to stream.", py::arg("os"), py::arg("binary"))
      .def("ReadNew", &PyClass::ReadNew,
          "Read data from a stream and return the corresponding object (const "
          "function; it's a class member because we need access to the vtable "
          "so generic code can read derived types).", py::arg("os"), py::arg("binary"))
      .def("ObjfPlus", &PyClass::ObjfPlus,
           "Return the objective function of the combined object this + other.", py::arg("other"))
      .def("ObjfMinus", &PyClass::ObjfMinus,
           "Return the objective function of the subtracted object this - other.", py::arg("other"))
      .def("Distance", &PyClass::Distance,
           "Return the objective function decrease from merging the two "
           "clusters, negated to be a positive number (or zero).", py::arg("other"));
}

void pybind_transition_information_itf(py::module& m) {
     using PyClass = TransitionInformation;
  py::class_<PyClass>(m, "TransitionInformation")
      .def("TransitionIdsEquivalent", &PyClass::TransitionIdsEquivalent,
           "Returns true if trans_id1 and trans_id2 can correspond to the "
          "same phone when trans_id1 immediately precedes trans_id2 (i.e., "
          "trans_id1 occurss at timestep t, and trans_id2 ocurs at timestep "
          "2) (possibly with epsilons between trans_id1 and trans_id2) OR "
          "trans_id1 ocurs before trans_id2, with some number of "
          "trans_id_{k} values, all of which fulfill "
          "TransitionIdsEquivalent(trans_id1, trans_id_{k}) "
          "\n"
          "If trans_id1 == trans_id2, it must be the case that "
          "TransitionIdsEquivalent(trans_id1, trans_id2) == true", py::arg("trans_id1"), py::arg("trans_id2"))
      .def("TransitionIdIsStartOfPhone", &PyClass::TransitionIdIsStartOfPhone,
           "Returns true if this trans_id corresponds to the start of a phone.", py::arg("trans_id"))
      .def("TransitionIdToPhone", &PyClass::TransitionIdToPhone,
           "Phone is a historical term, and should be understood in a wider "
          "sense that also includes graphemes, word pieces, etc.: any "
          "minimal entity in your problem domain which is represented by a "
          "sequence of transitions with a PDF assigned to each of them by "
          "the model. In this sense, Token is a better word. Since "
          "TransitionInformation was added to subsume TransitionModel, we "
          "did not want to change the call site of every "
          "TransitionModel::TransitionIdToPhone to "
          "TransitionInformation::TransitionIdToToken.", py::arg("trans_id"))
      .def("IsFinal", &PyClass::IsFinal,
           "Returns true if the destination of any edge with this trans_id "
          "as its ilabel is a final state (or if a final state is "
          "epsilon-reachable from its destination state).", py::arg("trans_id"))
      .def("IsSelfLoop", &PyClass::IsSelfLoop,
           "Returns true if *all* of the FST edge labeled by this trans_id "
          "have the same start and end states.", py::arg("trans_id"))
      .def("TransitionIdToPdf", &PyClass::TransitionIdToPdf, py::arg("trans_id"))
      .def("TransitionIdToPdfArray", &PyClass::TransitionIdToPdfArray)
      .def("NumTransitionIds", &PyClass::NumTransitionIds)
      .def("NumPdfs", &PyClass::NumPdfs);
}

void pybind_options_itf(py::module& m) {
  using PyClass = OptionsItf;
  py::class_<PyClass>(m, "OptionsItf");
}

void init_itf(py::module &_m) {
  py::module m = _m.def_submodule("itf", "itf pybind for Kaldi");
  pybind_context_dep_itf(m);
  pybind_decodable_itf(m);
  pybind_options_itf(m);
  pybind_clusterable_itf(m);
  pybind_transition_information_itf(m);

}
