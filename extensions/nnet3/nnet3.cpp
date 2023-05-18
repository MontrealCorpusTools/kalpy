
#include "nnet3/pybind_nnet3.h"

#include "nnet3/nnet-common.h"
#include "nnet3/nnet-chain-example.h"
#include "nnet3/nnet-nnet.h"
#include "nnet3/nnet-simple-component.h"
#include "nnet3/nnet-normalize-component.h"
#include "nnet3/nnet-example.h"
#include "util/pybind_util.h"
#include "nnet3/nnet-convolutional-component.h"
#include "nnet3/nnet-component-itf.h"
#include "nnet3/nnet-common.h"

using namespace kaldi;
using namespace kaldi::nnet3;
using namespace kaldi::chain;

void pybind_nnet_chain_example(py::module& m) {
  {
    using PyClass = NnetChainSupervision;
    py::class_<PyClass>(
        m, "NnetChainSupervision",
        "For regular setups we use struct 'NnetIo' as the output.  For the "
        "'chain' models, the output supervision is a little more complex as it "
        "involves a lattice and we need to do forward-backward, so we use a "
        "separate struct for it.  The 'output' name means that it pertains to "
        "the output of the network, as opposed to the features which pertain "
        "to the input of the network.  It actually stores the lattice-like "
        "supervision information at the output of the network (which imposes "
        "constraints on which frames each phone can be active on")
        .def(py::init<>())
        .def_readwrite("name", &PyClass::name,
                       "the name of the output in the neural net; in simple "
                       "setups it will just be 'output'.")
        .def_readwrite(
            "indexes", &PyClass::indexes,
            "The indexes that the output corresponds to.  The size of this "
            "vector will be equal to supervision.num_sequences * "
            "supervision.frames_per_sequence. Be careful about the order of "
            "these indexes-- it is a little confusing. The indexes in the "
            "'index' vector are ordered as: (frame 0 of each sequence); (frame "
            "1 of each sequence); and so on.  But in the 'supervision' object, "
            "the FST contains (sequence 0; sequence 1; ...).  So reordering is "
            "needed when doing the numerator computation. We order 'indexes' "
            "in this way for efficiency in the denominator computation (it "
            "helps memory locality), as well as to avoid the need for the nnet "
            "to reorder things internally to match the requested output (for "
            "layers inside the neural net, the ordering is (frame 0; frame 1 "
            "...) as this corresponds to the order you get when you sort a "
            "vector of Index).")
        .def_readwrite("supervision", &PyClass::supervision,
                       "The supervision object, containing the FST.")
        .def_readwrite(
            "deriv_weights", &PyClass::deriv_weights,
            "This is a vector of per-frame weights, required to be between 0 "
            "and 1, that is applied to the derivative during training (but not "
            "during model combination, where the derivatives need to agree "
            "with the computed objf values for the optimization code to work). "
            " The reason for this is to more exactly handle edge effects and "
            "to ensure that no frames are 'double-counted'.  The order of this "
            "vector corresponds to the order of the 'indexes' (i.e. all the "
            "first frames, then all the second frames, etc.) If this vector is "
            "empty it means we're not applying per-frame weights, so it's "
            "equivalent to a vector of all ones.  This vector is written to "
            "disk compactly as unsigned char.")
        .def("CheckDim", &PyClass::CheckDim)
        .def("__str__",
             [](const PyClass& sup) {
               std::ostringstream os;
               os << "name: " << sup.name << "\n";
               return os.str();
             })
        // TODO(fangjun): other methods can be wrapped when needed
        ;
  }
  {
    using PyClass = NnetChainExample;
    py::class_<PyClass>(m, "NnetChainExample")
        .def(py::init<>())
        .def_readwrite("inputs", &PyClass::inputs)
        .def_readwrite("outputs", &PyClass::outputs)
        .def("Compress", &PyClass::Compress,
             "Compresses the input features (if not compressed)")
        .def("__eq__",
             [](const PyClass& a, const PyClass& b) { return a == b; })
        .def("Read", &PyClass::Read, py::arg("is"), py::arg("binary"));

    // (fangjun): we follow the PyKaldi style to prepend a underline before the
    // registered classes and the user in general should not use them directly;
    // instead, they should use the corresponding python classes that are more
    // easier to use.
    pybind_sequential_table_reader<KaldiObjectHolder<PyClass>>(
        m, "_SequentialNnetChainExampleReader");

    pybind_random_access_table_reader<KaldiObjectHolder<PyClass>>(
        m, "_RandomAccessNnetChainExampleReader");

    pybind_table_writer<KaldiObjectHolder<PyClass>>(m,
                                                    "_NnetChainExampleWriter");
  }
}


void pybind_nnet_common(py::module& m) {
  {
    // Index is need by NnetChainSupervision in nnet_chain_example_pybind.cc
    using PyClass = Index;
    py::class_<PyClass>(
        m, "Index",
        "struct Index is intended to represent the various indexes by which we "
        "number the rows of the matrices that the Components process: mainly "
        "'n', the index of the member of the minibatch, 't', used for the "
        "frame index in speech recognition, and 'x', which is a catch-all "
        "extra index which we might use in convolutional setups or for other "
        "reasons.  It is possible to extend this by adding new indexes if "
        "needed.")
        .def(py::init<>())
        .def(py::init<int, int, int>(), py::arg("n"), py::arg("t"),
             py::arg("x") = 0)
        .def_readwrite("n", &PyClass::n, "member-index of minibatch, or zero.")
        .def_readwrite("t", &PyClass::t, "time-frame.")
        .def_readwrite("x", &PyClass::x,
                       "this may come in useful in convolutional approaches. "
                       "it is possible to add extra index here, if needed.")
        .def("__eq__",
             [](const PyClass& a, const PyClass& b) { return a == b; })
        .def("__ne__",
             [](const PyClass& a, const PyClass& b) { return a != b; })
        .def("__lt__", [](const PyClass& a, const PyClass& b) { return a < b; })
        .def(py::self + py::self)
        .def(py::self += py::self)
        // TODO(fangjun): other methods can be wrapped when needed
        ;
  }
}

void pybind_nnet_component_itf(py::module& m) {
  using PyClass = Component;
  py::class_<PyClass>(m, "Component",
                   "Abstract base-class for neural-net components.")
      .def("Type", &PyClass::Type,
           "Returns a string such as \"SigmoidComponent\", describing the "
           "type of the object.")
      .def("Info", &PyClass::Info,
           "Returns some text-form information about this component, for "
           "diagnostics. Starts with the type of the component.  E.g. "
           "\"SigmoidComponent dim=900\", although most components will have "
           "much more info.")
      .def_static("NewComponentOfType", &PyClass::NewComponentOfType,
                  py::return_value_policy::take_ownership);
}


void pybind_nnet_convolutional_component(py::module& m) {
  using TC = kaldi::nnet3::TdnnComponent;
  py::class_<TC, Component>(m, "TdnnComponent")
      .def("LinearParams", &TC::LinearParams,
           py::return_value_policy::reference)
      .def("BiasParams", &TC::BiasParams,
           py::return_value_policy::reference);
}


void pybind_nnet_example(py::module& m) {
  {
    using PyClass = NnetIo;
    py::class_<PyClass>(m, "NnetIo")
        .def(py::init<>())
        .def_readwrite("name", &PyClass::name,
                       "the name of the input in the neural net; in simple "
                       "setups it will just be 'input'.")
        .def_readwrite(
            "features", &PyClass::features,
            "The features or labels.  GeneralMatrix may contain either "
            "a CompressedMatrix, a Matrix, or SparseMatrix (a "
            "SparseMatrix would be the natural format for posteriors).");
    // TODO(fangjun): other constructors, fields and methods can be wrapped when
  }
  {
    using PyClass = NnetExample;
    py::class_<PyClass>(m, "NnetExample",
    "NnetExample is the input data and corresponding label (or labels) for one or "
    "more frames of input, used for standard cross-entropy training of neural "
    "nets (and possibly for other objective functions). ")
        .def(py::init<>())
        .def_readwrite("io", &PyClass::io,
        "\"io\" contains the input and output.  In principle there can be multiple "
        "types of both input and output, with different names.  The order is "
        "irrelevant.")
        .def("Compress", &PyClass::Compress,
             "Compresses any (input) features that are not sparse.")
        .def("Read", &PyClass::Read, py::arg("is"), py::arg("binary"));

    pybind_sequential_table_reader<KaldiObjectHolder<PyClass>>(
      m, "_SequentialNnetExampleReader");

    pybind_random_access_table_reader<KaldiObjectHolder<PyClass>>(
      m, "_RandomAccessNnetExampleReader");
  }
}

void pybind_nnet_normalize_component(py::module& m) {
  using PyClass = kaldi::nnet3::BatchNormComponent;
  py::class_<PyClass, Component>(m, "BatchNormComponent")
      .def("SetTestMode", &PyClass::SetTestMode, py::arg("test_mode"))
      .def("Offset", &PyClass::Offset, py::return_value_policy::reference)
      .def("Scale", overload_cast_<>()(&PyClass::Scale, py::const_),
           py::return_value_policy::reference);
}

void pybind_nnet_simple_component(py::module& m) {
  using FAC = FixedAffineComponent;
  py::class_<FAC, Component>(m, "FixedAffineComponent")
      .def("LinearParams", &FAC::LinearParams,
           py::return_value_policy::reference)
      .def("BiasParams", &FAC::BiasParams, py::return_value_policy::reference);

  using LC = LinearComponent;
  py::class_<LC, Component>(m, "LinearComponent");

  using AC = AffineComponent;
  py::class_<AC, Component>(m, "AffineComponent")
      .def("LinearParams", overload_cast_<>()(&AC::LinearParams, py::const_),
           py::return_value_policy::reference)
      .def("BiasParams", overload_cast_<>()(&AC::BiasParams, py::const_),
           py::return_value_policy::reference);

  using NGAC = NaturalGradientAffineComponent;
  py::class_<NGAC, AC>(m, "NaturalGradientAffineComponent");
}

void pybind_nnet_nnet(py::module& m) {
  using PyClass = kaldi::nnet3::Nnet;
  auto nnet = py::class_<PyClass>(
      m, "Nnet",
      "This function can be used either to initialize a new Nnet from a "
      "config file, or to add to an existing Nnet, possibly replacing "
      "certain parts of it.  It will die with error if something went wrong. "
      "Also see the function ReadEditConfig() in nnet-utils.h (it's made a "
      "non-member because it doesn't need special access).");
  nnet.def(py::init<>())
      .def("Read", &PyClass::Read, py::arg("is"), py::arg("binary"))
      .def("GetComponentNames", &PyClass::GetComponentNames,
           "returns vector of component names (needed by some parsing code, "
           "for instance).",
           py::return_value_policy::reference)
      .def("GetComponentName", &PyClass::GetComponentName,
           py::arg("component_index"))
      .def("Info", &PyClass::Info,
           "returns some human-readable information about the network, "
           "mostly for debugging purposes. Also see function NnetInfo() in "
           "nnet-utils.h, which prints out more extensive infoformation.")
      .def("NumComponents", &PyClass::NumComponents)
      .def("NumNodes", &PyClass::NumNodes)
      .def("GetComponent", (Component * (PyClass::*)(int32)) & PyClass::GetComponent,
           py::arg("c"), py::return_value_policy::reference);
}


void init_nnet3(py::module &_m) {
  py::module m = _m.def_submodule("nnet3", "nnet3 pybind for Kaldi");
  pybind_nnet_common(m);
  pybind_nnet_component_itf(m);
  pybind_nnet_convolutional_component(m);
  pybind_nnet_example(m);
  pybind_nnet_chain_example(m);
  pybind_nnet_nnet(m);
  pybind_nnet_normalize_component(m);
  pybind_nnet_simple_component(m);
}
