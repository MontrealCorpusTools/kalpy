
#include "nnet2/pybind_nnet2.h"

#include "nnet2/am-nnet.h"
#include "nnet2/combine-nnet-a.h"
#include "nnet2/combine-nnet-fast.h"
#include "nnet2/combine-nnet.h"
#include "nnet2/decodable-am-nnet.h"
#include "nnet2/get-feature-transform.h"
#include "nnet2/mixup-nnet.h"
#include "nnet2/nnet-component.h"
#include "nnet2/nnet-compute-discriminative-parallel.h"
#include "nnet2/nnet-compute-discriminative.h"
#include "nnet2/nnet-compute-online.h"
#include "nnet2/nnet-compute.h"
#include "nnet2/nnet-example-functions.h"
#include "nnet2/nnet-example.h"
#include "nnet2/nnet-fix.h"
#include "nnet2/nnet-functions.h"
#include "nnet2/nnet-limit-rank.h"
#include "nnet2/nnet-nnet.h"
#include "nnet2/nnet-precondition-online.h"
#include "nnet2/nnet-precondition.h"
#include "nnet2/nnet-stats.h"
#include "nnet2/nnet-update-parallel.h"
#include "nnet2/nnet-update.h"
#include "nnet2/online-nnet2-decodable.h"
#include "nnet2/rescale-nnet.h"
#include "nnet2/shrink-nnet.h"
#include "nnet2/train-nnet-ensemble.h"
#include "nnet2/train-nnet.h"
#include "nnet2/widen-nnet.h"
#include "util/pybind_util.h"

using namespace kaldi;
using namespace kaldi::nnet2;



class PyNnet2Component : public Component {
public:
    //Inherit the constructors
    using Component::Component;

    //Trampoline (need one for each virtual function)
    std::string Type() const override {
        PYBIND11_OVERRIDE_PURE(
            std::string, //Return type (ret_type)
            Component,      //Parent class (cname)
            Type          //Name of function in C++ (must match Python name) (fn)
                  //Argument(s) (...)
        );
    }

    void InitFromString(std::string args) override {
        PYBIND11_OVERRIDE_PURE(
            void, //Return type (ret_type)
            Component,      //Parent class (cname)
            InitFromString,          //Name of function in C++ (must match Python name) (fn)
            args      //Argument(s) (...)
        );
    }

    int32 InputDim() const override {
        PYBIND11_OVERRIDE_PURE(
            int32, //Return type (ret_type)
            Component,      //Parent class (cname)
            InputDim          //Name of function in C++ (must match Python name) (fn)
                  //Argument(s) (...)
        );
    }

    int32 OutputDim() const override {
        PYBIND11_OVERRIDE_PURE(
            int32, //Return type (ret_type)
            Component,      //Parent class (cname)
            OutputDim          //Name of function in C++ (must match Python name) (fn)
                  //Argument(s) (...)
        );
    }

    void Propagate(const ChunkInfo &in_info,
                         const ChunkInfo &out_info,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const override {
        PYBIND11_OVERRIDE_PURE(
            void, //Return type (ret_type)
            Component,      //Parent class (cname)
            Propagate          //Name of function in C++ (must match Python name) (fn)
             in_info, out_info, in, out    //Argument(s) (...)
        );
    }

    void Backprop(const ChunkInfo &in_info,
                        const ChunkInfo &out_info,
                        const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &out_value,
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        Component *to_update, // may be identical to "this".
                        CuMatrix<BaseFloat> *in_deriv) const override {
        PYBIND11_OVERRIDE_PURE(
            void, //Return type (ret_type)
            Component,      //Parent class (cname)
            Backprop          //Name of function in C++ (must match Python name) (fn)
             in_info, out_info, in_value,
             out_value, out_deriv, to_update, in_deriv   //Argument(s) (...)
        );
    }
    void Read(std::istream &is, bool binary) override {
        PYBIND11_OVERRIDE_PURE(
            void, //Return type (ret_type)
            Component,      //Parent class (cname)
            Read,          //Name of function in C++ (must match Python name) (fn)
             is, binary     //Argument(s) (...)
        );
    }
    void Write(std::ostream &os, bool binary) const override {
        PYBIND11_OVERRIDE_PURE(
            void, //Return type (ret_type)
            Component,      //Parent class (cname)
            Write,          //Name of function in C++ (must match Python name) (fn)
             os, binary     //Argument(s) (...)
        );
    }
    Component* Copy() const override {
        PYBIND11_OVERRIDE_PURE(
            Component*, //Return type (ret_type)
            Component,      //Parent class (cname)
            Copy          //Name of function in C++ (must match Python name) (fn)
                  //Argument(s) (...)
        );
    }
};

class PyNnet2NonlinearComponent : public NonlinearComponent {
public:
    //Inherit the constructors
    using NonlinearComponent::NonlinearComponent;

    //Trampoline (need one for each virtual function)
    std::string Type() const override {
        PYBIND11_OVERRIDE_PURE(
            std::string, //Return type (ret_type)
            Component,      //Parent class (cname)
            Type          //Name of function in C++ (must match Python name) (fn)
                  //Argument(s) (...)
        );
    }

    void InitFromString(std::string args) override {
        PYBIND11_OVERRIDE_PURE(
            void, //Return type (ret_type)
            Component,      //Parent class (cname)
            InitFromString,          //Name of function in C++ (must match Python name) (fn)
            args      //Argument(s) (...)
        );
    }

    int32 InputDim() const override {
        PYBIND11_OVERRIDE_PURE(
            int32, //Return type (ret_type)
            Component,      //Parent class (cname)
            InputDim          //Name of function in C++ (must match Python name) (fn)
                  //Argument(s) (...)
        );
    }

    int32 OutputDim() const override {
        PYBIND11_OVERRIDE_PURE(
            int32, //Return type (ret_type)
            Component,      //Parent class (cname)
            OutputDim          //Name of function in C++ (must match Python name) (fn)
                  //Argument(s) (...)
        );
    }

    void Propagate(const ChunkInfo &in_info,
                         const ChunkInfo &out_info,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const override {
        PYBIND11_OVERRIDE_PURE(
            void, //Return type (ret_type)
            Component,      //Parent class (cname)
            Propagate          //Name of function in C++ (must match Python name) (fn)
             in_info, out_info, in, out    //Argument(s) (...)
        );
    }

    void Backprop(const ChunkInfo &in_info,
                        const ChunkInfo &out_info,
                        const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &out_value,
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        Component *to_update, // may be identical to "this".
                        CuMatrix<BaseFloat> *in_deriv) const override {
        PYBIND11_OVERRIDE_PURE(
            void, //Return type (ret_type)
            Component,      //Parent class (cname)
            Backprop          //Name of function in C++ (must match Python name) (fn)
             in_info, out_info, in_value,
             out_value, out_deriv, to_update, in_deriv   //Argument(s) (...)
        );
    }
    void Read(std::istream &is, bool binary) override {
        PYBIND11_OVERRIDE_PURE(
            void, //Return type (ret_type)
            Component,      //Parent class (cname)
            Read,          //Name of function in C++ (must match Python name) (fn)
             is, binary     //Argument(s) (...)
        );
    }
    void Write(std::ostream &os, bool binary) const override {
        PYBIND11_OVERRIDE_PURE(
            void, //Return type (ret_type)
            Component,      //Parent class (cname)
            Write,          //Name of function in C++ (must match Python name) (fn)
             os, binary     //Argument(s) (...)
        );
    }
    Component* Copy() const override {
        PYBIND11_OVERRIDE_PURE(
            Component*, //Return type (ret_type)
            Component,      //Parent class (cname)
            Copy          //Name of function in C++ (must match Python name) (fn)
                  //Argument(s) (...)
        );
    }
};


class PyNnet2UpdatableComponent : public UpdatableComponent {
public:
    //Inherit the constructors
    using UpdatableComponent::UpdatableComponent;

    //Trampoline (need one for each virtual function)
    std::string Type() const override {
        PYBIND11_OVERRIDE_PURE(
            std::string, //Return type (ret_type)
            Component,      //Parent class (cname)
            Type          //Name of function in C++ (must match Python name) (fn)
                  //Argument(s) (...)
        );
    }

    void InitFromString(std::string args) override {
        PYBIND11_OVERRIDE_PURE(
            void, //Return type (ret_type)
            Component,      //Parent class (cname)
            InitFromString,          //Name of function in C++ (must match Python name) (fn)
            args      //Argument(s) (...)
        );
    }

    int32 InputDim() const override {
        PYBIND11_OVERRIDE_PURE(
            int32, //Return type (ret_type)
            Component,      //Parent class (cname)
            InputDim          //Name of function in C++ (must match Python name) (fn)
                  //Argument(s) (...)
        );
    }

    int32 OutputDim() const override {
        PYBIND11_OVERRIDE_PURE(
            int32, //Return type (ret_type)
            Component,      //Parent class (cname)
            OutputDim          //Name of function in C++ (must match Python name) (fn)
                  //Argument(s) (...)
        );
    }

    void Propagate(const ChunkInfo &in_info,
                         const ChunkInfo &out_info,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const override {
        PYBIND11_OVERRIDE_PURE(
            void, //Return type (ret_type)
            Component,      //Parent class (cname)
            Propagate          //Name of function in C++ (must match Python name) (fn)
             in_info, out_info, in, out    //Argument(s) (...)
        );
    }

    void Backprop(const ChunkInfo &in_info,
                        const ChunkInfo &out_info,
                        const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &out_value,
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        Component *to_update, // may be identical to "this".
                        CuMatrix<BaseFloat> *in_deriv) const override {
        PYBIND11_OVERRIDE_PURE(
            void, //Return type (ret_type)
            Component,      //Parent class (cname)
            Backprop          //Name of function in C++ (must match Python name) (fn)
             in_info, out_info, in_value,
             out_value, out_deriv, to_update, in_deriv   //Argument(s) (...)
        );
    }
    void Read(std::istream &is, bool binary) override {
        PYBIND11_OVERRIDE_PURE(
            void, //Return type (ret_type)
            Component,      //Parent class (cname)
            Read,          //Name of function in C++ (must match Python name) (fn)
             is, binary     //Argument(s) (...)
        );
    }
    void Write(std::ostream &os, bool binary) const override {
        PYBIND11_OVERRIDE_PURE(
            void, //Return type (ret_type)
            Component,      //Parent class (cname)
            Write,          //Name of function in C++ (must match Python name) (fn)
             os, binary     //Argument(s) (...)
        );
    }
    Component* Copy() const override {
        PYBIND11_OVERRIDE_PURE(
            Component*, //Return type (ret_type)
            Component,      //Parent class (cname)
            Copy          //Name of function in C++ (must match Python name) (fn)
                  //Argument(s) (...)
        );
    }
    void SetZero(bool treat_as_gradient) override {
        PYBIND11_OVERRIDE_PURE(
            void, //Return type (ret_type)
            UpdatableComponent,      //Parent class (cname)
            SetZero,          //Name of function in C++ (must match Python name) (fn)
             treat_as_gradient     //Argument(s) (...)
        );
    }
    BaseFloat DotProduct(const UpdatableComponent &other) const override {
        PYBIND11_OVERRIDE_PURE(
            BaseFloat, //Return type (ret_type)
            UpdatableComponent,      //Parent class (cname)
            DotProduct,          //Name of function in C++ (must match Python name) (fn)
             other     //Argument(s) (...)
        );
    }
    void PerturbParams(BaseFloat stddev) override {
        PYBIND11_OVERRIDE_PURE(
            void, //Return type (ret_type)
            UpdatableComponent,      //Parent class (cname)
            PerturbParams,          //Name of function in C++ (must match Python name) (fn)
             stddev     //Argument(s) (...)
        );
    }
    void Scale(BaseFloat scale) override {
        PYBIND11_OVERRIDE_PURE(
            void, //Return type (ret_type)
            UpdatableComponent,      //Parent class (cname)
            Scale,          //Name of function in C++ (must match Python name) (fn)
             scale     //Argument(s) (...)
        );
    }
    void Add(BaseFloat alpha, const UpdatableComponent &other) override {
        PYBIND11_OVERRIDE_PURE(
            void, //Return type (ret_type)
            UpdatableComponent,      //Parent class (cname)
            Add,          //Name of function in C++ (must match Python name) (fn)
             alpha, other    //Argument(s) (...)
        );
    }
};

void pybind_nnet2_am_nnet(py::module& m) {

  {
    using PyClass = AmNnet;

    auto am_nnet = py::class_<PyClass>(
        m, "AmNnet",
        "The class AmNnet (AM stands for \"acoustic model\") has the job of taking the "
          "\"Nnet\" class, which is a quite general neural network, and giving it an "
          "interface that's suitable for acoustic modeling; it deals with storing, and "
          "dividing by, the prior of each context-dependent state.");
    am_nnet.def(py::init<>())
      .def(py::init<const AmNnet &>(),
        py::arg("other"))
      .def("Init",
        py::overload_cast<std::istream &>(&PyClass::Init),
        "Initialize the neural network based acoustic model from a config file. "
          "At this point the priors won't be initialized; you'd have to do "
          "SetPriors for that.",
        py::arg("config_is"))
      .def("Init",
        py::overload_cast<const Nnet &>(&PyClass::Init),
        "Initialize from a neural network that's already been set up. "
          "Again, the priors will be empty at this point.",
        py::arg("nnet"))
      .def("NumPdfs",
        &PyClass::NumPdfs)
      .def("Read",
        &PyClass::Read,
        py::arg("is"),
        py::arg("binary"))
      .def("Write",
        &PyClass::Write,
        py::arg("os"),
        py::arg("binary"))
      .def("GetNnet",
        py::overload_cast<>(&PyClass::GetNnet))
      .def("SetPriors",
        &PyClass::SetPriors,
        py::arg("priors"))
      .def("Priors",
        &PyClass::Priors)
      .def("Info",
        &PyClass::Info)
      .def("ResizeOutputLayer",
        &PyClass::ResizeOutputLayer,
        "This function is used when doing transfer learning to a new system. "
          "It will set the priors to be all the same.",
        py::arg("new_num_pdfs"));
  }
}

void pybind_nnet2_combine_nnet_a(py::module& m) {

  {
    using PyClass = NnetCombineAconfig;

    auto nnet_combine_a_config = py::class_<PyClass>(
        m, "NnetCombineAconfig");
    nnet_combine_a_config.def(py::init<>())
      .def_readwrite("num_bfgs_iters", &PyClass::num_bfgs_iters)
      .def_readwrite("initial_step", &PyClass::initial_step)
      .def_readwrite("valid_impr_thresh", &PyClass::valid_impr_thresh)
      .def_readwrite("overshoot", &PyClass::overshoot)
      .def_readwrite("min_learning_rate_factor", &PyClass::min_learning_rate_factor)
      .def_readwrite("max_learning_rate_factor", &PyClass::max_learning_rate_factor)
      .def_readwrite("min_learning_rate", &PyClass::min_learning_rate);
  }
  m.def("CombineNnetsA",
        &CombineNnetsA,
        py::arg("combine_config"),
        py::arg("validation_set"),
        py::arg("nnets_in"),
        py::arg("nnet_out"));
}

void pybind_nnet2_combine_nnet_fast(py::module& m) {

  {
    using PyClass = NnetCombineFastConfig;

    auto nnet_combine_fast_config = py::class_<PyClass>(
        m, "NnetCombineFastConfig");
    nnet_combine_fast_config.def(py::init<>())
      .def_readwrite("initial_model", &PyClass::initial_model)
      .def_readwrite("num_lbfgs_iters", &PyClass::num_lbfgs_iters)
      .def_readwrite("num_threads", &PyClass::num_threads)
      .def_readwrite("initial_impr", &PyClass::initial_impr)
      .def_readwrite("fisher_floor", &PyClass::fisher_floor)
      .def_readwrite("alpha", &PyClass::alpha)
      .def_readwrite("fisher_minibatch_size", &PyClass::fisher_minibatch_size)
      .def_readwrite("minibatch_size", &PyClass::minibatch_size)
      .def_readwrite("max_lbfgs_dim", &PyClass::max_lbfgs_dim)
      .def_readwrite("regularizer", &PyClass::regularizer);
  }
  m.def("CombineNnetsFast",
        &CombineNnetsFast,
        py::arg("combine_config"),
        py::arg("validation_set"),
        py::arg("nnets_in"),
        py::arg("nnet_out"));
}

void pybind_nnet2_combine_nnet(py::module& m) {

  {
    using PyClass = NnetCombineConfig;

    auto nnet_combine_config = py::class_<PyClass>(
        m, "NnetCombineConfig");
    nnet_combine_config.def(py::init<>())
      .def_readwrite("initial_model", &PyClass::initial_model)
      .def_readwrite("num_bfgs_iters", &PyClass::num_bfgs_iters)
      .def_readwrite("initial_impr", &PyClass::initial_impr)
      .def_readwrite("test_gradient", &PyClass::test_gradient);
  }
  m.def("CombineNnets",
        &CombineNnets,
        py::arg("combine_config"),
        py::arg("validation_set"),
        py::arg("nnets_in"),
        py::arg("nnet_out"));
}

void pybind_nnet2_decodable_am_net(py::module& m) {

  {
    using PyClass = DecodableAmNnet;

    auto decodable_am_net = py::class_<PyClass, DecodableInterface>(
        m, "DecodableAmNnet");
    decodable_am_net.def(py::init<const TransitionModel &,
                  const AmNnet &,
                  const CuMatrixBase<BaseFloat> &,
                  bool ,
                  BaseFloat >(),
        py::arg("trans_model"),
        py::arg("am_nnet"),
        py::arg("feats"),
        py::arg("pad_input") = true,
        py::arg("prob_scale") = 1.0)
        .def("LogLikelihood", &PyClass::LogLikelihood,
               "Note, frames are numbered from zero.  But transition_id is numbered "
               "from one (this routine is called by FSTs).",
               py::arg("frame"),
               py::arg("transition_id"))
        .def("NumFramesReady", &PyClass::NumFramesReady)
        .def("NumIndices", &PyClass::NumIndices)
        .def("IsLastFrame", &PyClass::IsLastFrame,
               py::arg("frame"));
  }
  {
    using PyClass = DecodableAmNnetParallel;

    auto decodable_am_net_parallel = py::class_<PyClass, DecodableInterface>(
        m, "DecodableAmNnetParallel");
    decodable_am_net_parallel.def(py::init<const TransitionModel &,
                  const AmNnet &,
                  const CuMatrix<BaseFloat> *,
                  bool ,
                  BaseFloat >(),
        py::arg("trans_model"),
        py::arg("am_nnet"),
        py::arg("feats"),
        py::arg("pad_input") = true,
        py::arg("prob_scale") = 1.0)
        .def("Compute", &PyClass::Compute)
        .def("LogLikelihood", &PyClass::LogLikelihood,
               "Note, frames are numbered from zero.  But transition_id is numbered "
               "from one (this routine is called by FSTs).",
               py::arg("frame"),
               py::arg("transition_id"))
        .def("NumFramesReady", &PyClass::NumFramesReady)
        .def("NumIndices", &PyClass::NumIndices)
        .def("IsLastFrame", &PyClass::IsLastFrame,
               py::arg("frame"));
  }
}

void pybind_nnet2_get_feature_transform(py::module& m) {

    {

      using PyClass = FeatureTransformEstimateOptions;
      auto feature_transform_estimate_options = py::class_<PyClass>(
          m, "FeatureTransformEstimateOptions");
      feature_transform_estimate_options.def(py::init<>())
      .def_readwrite("remove_offset", &PyClass::remove_offset)
      .def_readwrite("dim", &PyClass::dim)
      .def_readwrite("within_class_factor", &PyClass::within_class_factor)
      .def_readwrite("max_singular_value", &PyClass::max_singular_value);
    }
    {

      using PyClass = FeatureTransformEstimate;
      auto feature_transform_estimate = py::class_<PyClass, LdaEstimate>(
          m, "FeatureTransformEstimate");
      feature_transform_estimate.def(py::init<>())
        .def("Estimate", &PyClass::Estimate,
               "Estimates the LDA transform matrix m.  If Mfull != NULL, it also outputs "
               "the full matrix (without dimensionality reduction), which is useful for "
               "some purposes.  If opts.remove_offset == true, it will output both matrices "
               "with an extra column which corresponds to mean-offset removal (the matrix "
               "should be multiplied by the feature with a 1 appended to give the correct "
               "result, as with other Kaldi transforms.) "
               "\"within_cholesky\" is a pointer to an SpMatrix that, if non-NULL, will "
               "be set to the Cholesky factor of the within-class covariance matrix. "
               "This is used for perturbing features.",
               py::arg("opts"),
               py::arg("M"),
               py::arg("within_cholesky"));
    }
    {

      using PyClass = FeatureTransformEstimateMulti;
      auto feature_transform_estimate_multi = py::class_<PyClass, FeatureTransformEstimate>(
          m, "FeatureTransformEstimateMulti");
      feature_transform_estimate_multi.def(py::init<>())
        .def("Estimate", &PyClass::Estimate,
               "This is as FeatureTransformEstimate, but for use in "
               "nnet-get-feature-transform-multi.cc, see the usage message "
               "of that program for a description of what it does.",
               py::arg("opts"),
               py::arg("indexes"),
               py::arg("M"));
    }
}


void pybind_nnet2_mixup_nnet(py::module& m) {

    {

      using PyClass = NnetMixupConfig;
      auto nnet_mixup_config = py::class_<PyClass>(
          m, "NnetMixupConfig");
      nnet_mixup_config.def(py::init<>())
      .def_readwrite("power", &PyClass::power)
      .def_readwrite("min_count", &PyClass::min_count)
      .def_readwrite("num_mixtures", &PyClass::num_mixtures)
      .def_readwrite("perturb_stddev", &PyClass::perturb_stddev);
    }
  m.def("MixupNnet",
        &MixupNnet,
        "This function does something similar to Gaussian mixture splitting for "
          "GMMs, except applied to the output layer of the neural network. "
          "We create additional outputs, which will be summed over using a "
          "SumGroupComponent.",
        py::arg("mixup_config"),
        py::arg("nnet"));
}

void pybind_nnet2_nnet_component(py::module& m) {

    {

      using PyClass = ChunkInfo;
      auto chunk_info = py::class_<PyClass>(
          m, "ChunkInfo");
      chunk_info
          .def(py::init<>())
          .def(py::init<int32 , int32 ,
            int32 , int32  >(),
          py::arg("feat_dim"),
          py::arg("num_chunks"),
          py::arg("first_offset"),
          py::arg("last_offset"))
          .def(py::init<int32 , int32 ,
            const std::vector<int32> >(),
          py::arg("feat_dim"),
          py::arg("num_chunks"),
          py::arg("offsets"))
      .def("GetIndex",
        &PyClass::GetIndex,
          py::arg("offset"))
      .def("GetOffset",
        &PyClass::GetOffset,
          py::arg("index"))
      .def("MakeOffsetsContiguous",
        &PyClass::MakeOffsetsContiguous)
      .def("ChunkSize",
        &PyClass::ChunkSize)
      .def("NumChunks",
        &PyClass::NumChunks)
      .def("NumRows",
        &PyClass::NumRows)
      .def("NumCols",
        &PyClass::NumCols)
      .def("CheckSize",
        &PyClass::CheckSize,
          py::arg("mat"))
      .def("Check",
        &PyClass::Check);

    }
    {

      using PyClass = Component;
      auto component = py::class_<PyClass, PyNnet2Component>(
          m, "Component");
      component
          .def(py::init<>())
      .def("Type",
        &PyClass::Type)
      .def("Index",
        &PyClass::Index)
      .def("SetIndex",
        &PyClass::SetIndex,
          py::arg("index"))
      .def("InitFromString",
        &PyClass::InitFromString,
          py::arg("args"))
      .def("InputDim",
        &PyClass::InputDim)
      .def("OutputDim",
        &PyClass::OutputDim)
      .def("Context",
        &PyClass::Context)
      .def("Propagate",
        static_cast< void (PyClass::*)(const ChunkInfo &,
                 const ChunkInfo &,
                 const CuMatrixBase<BaseFloat> &,
                 CuMatrix<BaseFloat> *) const>(&PyClass::Propagate),
          py::arg("in_info"),
          py::arg("out_info"),
          py::arg("in"),
          py::arg("out"))
      .def("Backprop",
        &PyClass::Backprop,
          py::arg("in_info"),
          py::arg("out_info"),
          py::arg("in_value"),
          py::arg("out_value"),
          py::arg("out_deriv"),
          py::arg("to_update"),
          py::arg("in_deriv"))
      .def("BackpropNeedsInput",
        &PyClass::BackpropNeedsInput)
      .def("BackpropNeedsOutput",
        &PyClass::BackpropNeedsOutput)
      .def_static("ReadNew",
        &PyClass::ReadNew,
          py::arg("is"),
          py::arg("binary"))
      .def("Copy",
        &PyClass::Copy)
      .def_static("NewFromString",
        &PyClass::NewFromString,
          py::arg("initializer_line"))
      .def_static("NewComponentOfType",
        &PyClass::NewComponentOfType,
          py::arg("type"))
      .def("Read",
        &PyClass::Read,
        py::arg("is"),
        py::arg("binary"))
      .def("Write",
        &PyClass::Write,
        py::arg("os"),
        py::arg("binary"))
      .def("Info",
        &PyClass::Info);

    }
    {

      using PyClass = UpdatableComponent;
      auto updatable_component = py::class_<PyClass, PyNnet2UpdatableComponent>(
          m, "UpdatableComponent");
      updatable_component
          .def(py::init<>())
          .def(py::init<BaseFloat>(),
          py::arg("learning_rate"))
          //.def(py::init<const UpdatableComponent & >(),
          //py::arg("other"))
      .def("Init",
        &PyClass::Init,
          py::arg("learning_rate"))
      .def("SetZero",
        &PyClass::SetZero,
          py::arg("treat_as_gradient"))
      .def("DotProduct",
        &PyClass::DotProduct,
          py::arg("other"))
      .def("PerturbParams",
        &PyClass::PerturbParams,
          py::arg("stddev"))
      .def("Scale",
        &PyClass::Scale,
          py::arg("scale"))
      .def("Add",
        &PyClass::Add,
          py::arg("alpha"),
          py::arg("other"))
      .def("SetLearningRate",
        &PyClass::SetLearningRate,
          py::arg("lrate"))
      .def("LearningRate",
        &PyClass::LearningRate)
      .def("Info",
        &PyClass::Info)
      .def("GetParameterDim",
        &PyClass::GetParameterDim)
      .def("Vectorize",
        &PyClass::Vectorize,
          py::arg("params"))
      .def("UnVectorize",
        &PyClass::UnVectorize,
          py::arg("params"));

    }
    {

      using PyClass = NonlinearComponent;
      auto nonlinear_component = py::class_<PyClass, PyNnet2NonlinearComponent>(
          m, "NonlinearComponent");
      nonlinear_component
          .def(py::init<>())
          .def(py::init<int32 >(),
               py::arg("dim"))
          //.def(py::init<const NonlinearComponent &>(),
          //     py::arg("other"))
      .def("Init",
        &PyClass::Init,
          py::arg("dim"))
      .def("InputDim",
        &PyClass::InputDim)
      .def("OutputDim",
        &PyClass::OutputDim)
      .def("InitFromString",
        &PyClass::InitFromString,
          py::arg("args"))
      .def("Read",
        &PyClass::Read,
        py::arg("is"),
        py::arg("binary"))
      .def("Write",
        &PyClass::Write,
        py::arg("os"),
        py::arg("binary"))
      .def("Scale",
        &PyClass::Scale,
          py::arg("scale"))
      .def("Add",
        &PyClass::Add,
          py::arg("alpha"),
          py::arg("other"))
      .def("ValueSum",
        &PyClass::ValueSum)
      .def("DerivSum",
        &PyClass::DerivSum)
      .def("Count",
        &PyClass::Count)
      .def("SetDim",
        &PyClass::SetDim,
          py::arg("dim"));

    }
    {

      using PyClass = MaxoutComponent;
      auto maxout_component = py::class_<PyClass, Component>(
          m, "MaxoutComponent");
      maxout_component
          .def(py::init<>())
          .def(py::init<int32, int32 >(),
          py::arg("input_dim"),
          py::arg("output_dim"))
      .def("Init",
        &PyClass::Init,
          py::arg("input_dim"),
          py::arg("output_dim"))
      .def("Type",
        &PyClass::Type)
      .def("InitFromString",
        &PyClass::InitFromString,
          py::arg("args"))
      .def("InputDim",
        &PyClass::InputDim)
      .def("OutputDim",
        &PyClass::OutputDim)
      .def("Propagate",
        static_cast< void (PyClass::*)(const ChunkInfo &,
                 const ChunkInfo &,
                 const CuMatrixBase<BaseFloat> &,
                 CuMatrix<BaseFloat> *) const>(&PyClass::Propagate),
          py::arg("in_info"),
          py::arg("out_info"),
          py::arg("in"),
          py::arg("out"))
      .def("Backprop",
        &PyClass::Backprop,
          py::arg("in_info"),
          py::arg("out_info"),
          py::arg("in_value"),
          py::arg("out_value"),
          py::arg("out_deriv"),
          py::arg("to_update"),
          py::arg("in_deriv"))
      .def("BackpropNeedsInput",
        &PyClass::BackpropNeedsInput)
      .def("BackpropNeedsOutput",
        &PyClass::BackpropNeedsOutput)
      .def("Copy",
        &PyClass::Copy)
      .def("Read",
        &PyClass::Read,
        py::arg("is"),
        py::arg("binary"))
      .def("Write",
        &PyClass::Write,
        py::arg("os"),
        py::arg("binary"))
      .def("Info",
        &PyClass::Info);

    }
    {

      using PyClass = MaxpoolingComponent;
      auto maxpooling_component = py::class_<PyClass, Component>(
          m, "MaxpoolingComponent");
      maxpooling_component
          .def(py::init<>())
          .def(py::init<int32, int32, int32, int32 >(),
          py::arg("input_dim"),
          py::arg("output_dim"),
          py::arg("pool_size"),
          py::arg("pool_stride"))
      .def("Init",
        &PyClass::Init,
          py::arg("input_dim"),
          py::arg("output_dim"),
          py::arg("pool_size"),
          py::arg("pool_stride"))
      .def("Type",
        &PyClass::Type)
      .def("InitFromString",
        &PyClass::InitFromString,
          py::arg("args"))
      .def("InputDim",
        &PyClass::InputDim)
      .def("OutputDim",
        &PyClass::OutputDim)
      .def("Propagate",
        static_cast< void (PyClass::*)(const ChunkInfo &,
                 const ChunkInfo &,
                 const CuMatrixBase<BaseFloat> &,
                 CuMatrix<BaseFloat> *) const>(&PyClass::Propagate),
          py::arg("in_info"),
          py::arg("out_info"),
          py::arg("in"),
          py::arg("out"))
      .def("Backprop",
        &PyClass::Backprop,
          py::arg("in_info"),
          py::arg("out_info"),
          py::arg("in_value"),
          py::arg("out_value"),
          py::arg("out_deriv"),
          py::arg("to_update"),
          py::arg("in_deriv"))
      .def("BackpropNeedsInput",
        &PyClass::BackpropNeedsInput)
      .def("BackpropNeedsOutput",
        &PyClass::BackpropNeedsOutput)
      .def("Copy",
        &PyClass::Copy)
      .def("Read",
        &PyClass::Read,
        py::arg("is"),
        py::arg("binary"))
      .def("Write",
        &PyClass::Write,
        py::arg("os"),
        py::arg("binary"))
      .def("Info",
        &PyClass::Info);

    }
    {

      using PyClass = PnormComponent;
      auto pnorm_component = py::class_<PyClass, Component>(
          m, "PnormComponent");
      pnorm_component
          .def(py::init<>())
          .def(py::init<int32, int32, BaseFloat >(),
          py::arg("input_dim"),
          py::arg("output_dim"),
          py::arg("p"))
      .def("Init",
        &PyClass::Init,
          py::arg("input_dim"),
          py::arg("output_dim"),
          py::arg("p"))
      .def("Type",
        &PyClass::Type)
      .def("InitFromString",
        &PyClass::InitFromString,
          py::arg("args"))
      .def("InputDim",
        &PyClass::InputDim)
      .def("OutputDim",
        &PyClass::OutputDim)
      .def("Propagate",
        static_cast< void (PyClass::*)(const ChunkInfo &,
                 const ChunkInfo &,
                 const CuMatrixBase<BaseFloat> &,
                 CuMatrix<BaseFloat> *) const>(&PyClass::Propagate),
          py::arg("in_info"),
          py::arg("out_info"),
          py::arg("in"),
          py::arg("out"))
      .def("Backprop",
        &PyClass::Backprop,
          py::arg("in_info"),
          py::arg("out_info"),
          py::arg("in_value"),
          py::arg("out_value"),
          py::arg("out_deriv"),
          py::arg("to_update"),
          py::arg("in_deriv"))
      .def("BackpropNeedsInput",
        &PyClass::BackpropNeedsInput)
      .def("BackpropNeedsOutput",
        &PyClass::BackpropNeedsOutput)
      .def("Copy",
        &PyClass::Copy)
      .def("Read",
        &PyClass::Read,
        py::arg("is"),
        py::arg("binary"))
      .def("Write",
        &PyClass::Write,
        py::arg("os"),
        py::arg("binary"))
      .def("Info",
        &PyClass::Info);

    }
    {

      using PyClass = NormalizeComponent;
      auto normalize_component = py::class_<PyClass, NonlinearComponent>(
          m, "NormalizeComponent");
      normalize_component
          .def(py::init<>())
          .def(py::init<int32 >(),
               py::arg("dim"))
          //.def(py::init<const NonlinearComponent &>(),
          //     py::arg("other"))
      .def("Type",
        &PyClass::Type)
      .def("Copy",
        &PyClass::Copy)
      .def("BackpropNeedsInput",
        &PyClass::BackpropNeedsInput)
      .def("BackpropNeedsOutput",
        &PyClass::BackpropNeedsOutput)
      .def("Propagate",
        static_cast< void (PyClass::*)(const ChunkInfo &,
                 const ChunkInfo &,
                 const CuMatrixBase<BaseFloat> &,
                 CuMatrix<BaseFloat> *) const>(&PyClass::Propagate),
          py::arg("in_info"),
          py::arg("out_info"),
          py::arg("in"),
          py::arg("out"))
      .def("Backprop",
        &PyClass::Backprop,
          py::arg("in_info"),
          py::arg("out_info"),
          py::arg("in_value"),
          py::arg("out_value"),
          py::arg("out_deriv"),
          py::arg("to_update"),
          py::arg("in_deriv"));

    }
    {

      using PyClass = SigmoidComponent;
      auto sigmoid_component = py::class_<PyClass, NonlinearComponent>(
          m, "SigmoidComponent");
      sigmoid_component
          .def(py::init<>())
          .def(py::init<int32 >(),
               py::arg("dim"))
          //.def(py::init<const NonlinearComponent &>(),
          //     py::arg("other"))
      .def("Type",
        &PyClass::Type)
      .def("Copy",
        &PyClass::Copy)
      .def("BackpropNeedsInput",
        &PyClass::BackpropNeedsInput)
      .def("BackpropNeedsOutput",
        &PyClass::BackpropNeedsOutput)
      .def("Propagate",
        static_cast< void (PyClass::*)(const ChunkInfo &,
                 const ChunkInfo &,
                 const CuMatrixBase<BaseFloat> &,
                 CuMatrix<BaseFloat> *) const>(&PyClass::Propagate),
          py::arg("in_info"),
          py::arg("out_info"),
          py::arg("in"),
          py::arg("out"))
      .def("Backprop",
        &PyClass::Backprop,
          py::arg("in_info"),
          py::arg("out_info"),
          py::arg("in_value"),
          py::arg("out_value"),
          py::arg("out_deriv"),
          py::arg("to_update"),
          py::arg("in_deriv"));

    }
    {

      using PyClass = TanhComponent;
      auto tanh_component = py::class_<PyClass, NonlinearComponent>(
          m, "TanhComponent");
      tanh_component
          .def(py::init<>())
          .def(py::init<int32 >(),
               py::arg("dim"))
          //.def(py::init<const NonlinearComponent &>(),
          //     py::arg("other"))
      .def("Type",
        &PyClass::Type)
      .def("Copy",
        &PyClass::Copy)
      .def("BackpropNeedsInput",
        &PyClass::BackpropNeedsInput)
      .def("BackpropNeedsOutput",
        &PyClass::BackpropNeedsOutput)
      .def("Propagate",
        static_cast< void (PyClass::*)(const ChunkInfo &,
                 const ChunkInfo &,
                 const CuMatrixBase<BaseFloat> &,
                 CuMatrix<BaseFloat> *) const>(&PyClass::Propagate),
          py::arg("in_info"),
          py::arg("out_info"),
          py::arg("in"),
          py::arg("out"))
      .def("Backprop",
        &PyClass::Backprop,
          py::arg("in_info"),
          py::arg("out_info"),
          py::arg("in_value"),
          py::arg("out_value"),
          py::arg("out_deriv"),
          py::arg("to_update"),
          py::arg("in_deriv"));

    }
    {

      using PyClass = PowerComponent;
      auto power_component = py::class_<PyClass, NonlinearComponent>(
          m, "PowerComponent");
      power_component
          .def(py::init<>())
          .def(py::init<int32 >(),
               py::arg("dim"))
          //.def(py::init<const NonlinearComponent &>(),
          //     py::arg("other"))
      .def("Type",
        &PyClass::Type)
      .def("Copy",
        &PyClass::Copy)
      .def("BackpropNeedsInput",
        &PyClass::BackpropNeedsInput)
      .def("BackpropNeedsOutput",
        &PyClass::BackpropNeedsOutput)
      .def("Propagate",
        static_cast< void (PyClass::*)(const ChunkInfo &,
                 const ChunkInfo &,
                 const CuMatrixBase<BaseFloat> &,
                 CuMatrix<BaseFloat> *) const>(&PyClass::Propagate),
          py::arg("in_info"),
          py::arg("out_info"),
          py::arg("in"),
          py::arg("out"))
      .def("Backprop",
        &PyClass::Backprop,
          py::arg("in_info"),
          py::arg("out_info"),
          py::arg("in_value"),
          py::arg("out_value"),
          py::arg("out_deriv"),
          py::arg("to_update"),
          py::arg("in_deriv"));

    }
    {

      using PyClass = RectifiedLinearComponent;
      auto rectified_linear_component = py::class_<PyClass, NonlinearComponent>(
          m, "RectifiedLinearComponent");
      rectified_linear_component
          .def(py::init<>())
          .def(py::init<int32 >(),
               py::arg("dim"))
          //.def(py::init<const NonlinearComponent &>(),
          //     py::arg("other"))
      .def("Type",
        &PyClass::Type)
      .def("Copy",
        &PyClass::Copy)
      .def("BackpropNeedsInput",
        &PyClass::BackpropNeedsInput)
      .def("BackpropNeedsOutput",
        &PyClass::BackpropNeedsOutput)
      .def("Propagate",
        static_cast< void (PyClass::*)(const ChunkInfo &,
                 const ChunkInfo &,
                 const CuMatrixBase<BaseFloat> &,
                 CuMatrix<BaseFloat> *) const>(&PyClass::Propagate),
          py::arg("in_info"),
          py::arg("out_info"),
          py::arg("in"),
          py::arg("out"))
      .def("Backprop",
        &PyClass::Backprop,
          py::arg("in_info"),
          py::arg("out_info"),
          py::arg("in_value"),
          py::arg("out_value"),
          py::arg("out_deriv"),
          py::arg("to_update"),
          py::arg("in_deriv"));

    }
    {

      using PyClass = SoftHingeComponent;
      auto soft_hinge_component = py::class_<PyClass, NonlinearComponent>(
          m, "SoftHingeComponent");
      soft_hinge_component
          .def(py::init<>())
          .def(py::init<int32 >(),
               py::arg("dim"))
          //.def(py::init<const NonlinearComponent &>(),
          //     py::arg("other"))
      .def("Type",
        &PyClass::Type)
      .def("Copy",
        &PyClass::Copy)
      .def("BackpropNeedsInput",
        &PyClass::BackpropNeedsInput)
      .def("BackpropNeedsOutput",
        &PyClass::BackpropNeedsOutput)
      .def("Propagate",
        static_cast< void (PyClass::*)(const ChunkInfo &,
                 const ChunkInfo &,
                 const CuMatrixBase<BaseFloat> &,
                 CuMatrix<BaseFloat> *) const>(&PyClass::Propagate),
          py::arg("in_info"),
          py::arg("out_info"),
          py::arg("in"),
          py::arg("out"))
      .def("Backprop",
        &PyClass::Backprop,
          py::arg("in_info"),
          py::arg("out_info"),
          py::arg("in_value"),
          py::arg("out_value"),
          py::arg("out_deriv"),
          py::arg("to_update"),
          py::arg("in_deriv"));

    }
    {

      using PyClass = ScaleComponent;
      auto scale_component = py::class_<PyClass, Component>(
          m, "ScaleComponent");
      scale_component
          .def(py::init<>())
          .def(py::init<int32, BaseFloat >(),
               py::arg("dim"),
               py::arg("scale"))
          //.def(py::init<const NonlinearComponent &>(),
          //     py::arg("other"))
      .def("Type",
        &PyClass::Type)
      .def("Copy",
        &PyClass::Copy)
      .def("BackpropNeedsInput",
        &PyClass::BackpropNeedsInput)
      .def("BackpropNeedsOutput",
        &PyClass::BackpropNeedsOutput)
      .def("Propagate",
        static_cast< void (PyClass::*)(const ChunkInfo &,
                 const ChunkInfo &,
                 const CuMatrixBase<BaseFloat> &,
                 CuMatrix<BaseFloat> *) const>(&PyClass::Propagate),
          py::arg("in_info"),
          py::arg("out_info"),
          py::arg("in"),
          py::arg("out"))
      .def("Backprop",
        &PyClass::Backprop,
          py::arg("in_info"),
          py::arg("out_info"),
          py::arg("in_value"),
          py::arg("out_value"),
          py::arg("out_deriv"),
          py::arg("to_update"),
          py::arg("in_deriv"))
      .def("InputDim",
        &PyClass::InputDim)
      .def("OutputDim",
        &PyClass::OutputDim)
      .def("Read",
        &PyClass::Read,
        py::arg("is"),
        py::arg("binary"))
      .def("Write",
        &PyClass::Write,
        py::arg("os"),
        py::arg("binary"))
      .def("Init",
        &PyClass::Init,
        py::arg("dim"),
        py::arg("scale"))
      .def("InitFromString",
        &PyClass::InitFromString,
        py::arg("args"))
      .def("Info",
        &PyClass::Info);

    }
    {

      using PyClass = SoftmaxComponent;
      auto softmax_component = py::class_<PyClass, NonlinearComponent>(
          m, "SoftmaxComponent");
      softmax_component
          .def(py::init<>())
          .def(py::init<int32 >(),
               py::arg("dim"))
          //.def(py::init<const NonlinearComponent &>(),
          //     py::arg("other"))
      .def("Type",
        &PyClass::Type)
      .def("Copy",
        &PyClass::Copy)
      .def("BackpropNeedsInput",
        &PyClass::BackpropNeedsInput)
      .def("BackpropNeedsOutput",
        &PyClass::BackpropNeedsOutput)
      .def("Propagate",
        static_cast< void (PyClass::*)(const ChunkInfo &,
                 const ChunkInfo &,
                 const CuMatrixBase<BaseFloat> &,
                 CuMatrix<BaseFloat> *) const>(&PyClass::Propagate),
          py::arg("in_info"),
          py::arg("out_info"),
          py::arg("in"),
          py::arg("out"))
      .def("Backprop",
        &PyClass::Backprop,
          py::arg("in_info"),
          py::arg("out_info"),
          py::arg("in_value"),
          py::arg("out_value"),
          py::arg("out_deriv"),
          py::arg("to_update"),
          py::arg("in_deriv"))
      .def("MixUp",
        &PyClass::MixUp,
          py::arg("num_mixtures"),
          py::arg("power"),
          py::arg("min_count"),
          py::arg("perturb_stddev"),
          py::arg("ac"),
          py::arg("sc"));

    }
    {

      using PyClass = LogSoftmaxComponent;
      auto log_softmax_component = py::class_<PyClass, NonlinearComponent>(
          m, "LogSoftmaxComponent");
      log_softmax_component
          .def(py::init<>())
          .def(py::init<int32 >(),
               py::arg("dim"))
          //.def(py::init<const NonlinearComponent &>(),
          //     py::arg("other"))
      .def("Type",
        &PyClass::Type)
      .def("Copy",
        &PyClass::Copy)
      .def("BackpropNeedsInput",
        &PyClass::BackpropNeedsInput)
      .def("BackpropNeedsOutput",
        &PyClass::BackpropNeedsOutput)
      .def("Propagate",
        static_cast< void (PyClass::*)(const ChunkInfo &,
                 const ChunkInfo &,
                 const CuMatrixBase<BaseFloat> &,
                 CuMatrix<BaseFloat> *) const>(&PyClass::Propagate),
          py::arg("in_info"),
          py::arg("out_info"),
          py::arg("in"),
          py::arg("out"))
      .def("Backprop",
        &PyClass::Backprop,
          py::arg("in_info"),
          py::arg("out_info"),
          py::arg("in_value"),
          py::arg("out_value"),
          py::arg("out_deriv"),
          py::arg("to_update"),
          py::arg("in_deriv"));

    }
    {

      using PyClass = AffineComponent;
      auto affine_component = py::class_<PyClass, UpdatableComponent>(
          m, "AffineComponent");
      affine_component
          .def(py::init<>())
          .def(py::init<const CuMatrixBase<BaseFloat> &,
                  const CuVectorBase<BaseFloat> &,
                  BaseFloat >(),
               py::arg("linear_params"),
               py::arg("bias_params"),
               py::arg("learning_rate"))
          //.def(py::init<const NonlinearComponent &>(),
          //     py::arg("other"))
      .def("InputDim",
        &PyClass::InputDim)
      .def("OutputDim",
        &PyClass::OutputDim)
      .def("Init",
        py::overload_cast<BaseFloat ,
            int32 , int32 ,
            BaseFloat , BaseFloat >(&PyClass::Init),
               py::arg("learning_rate"),
               py::arg("input_dim"),
               py::arg("output_dim"),
               py::arg("param_stddev"),
               py::arg("bias_stddev"))
      .def("Init",
        py::overload_cast<BaseFloat ,
            std::string >(&PyClass::Init),
               py::arg("learning_rate"),
               py::arg("matrix_filename"))
      .def("Resize",
        &PyClass::Resize,
               py::arg("input_dim"),
               py::arg("output_dim"))
      .def("CollapseWithNext",
       static_cast< Component *(PyClass::*)(const AffineComponent &) const>(&PyClass::CollapseWithNext),
               py::arg("next"))
      .def("CollapseWithNext",
        static_cast< Component *(PyClass::*)(const FixedAffineComponent &) const>(&PyClass::CollapseWithNext),
               py::arg("next"))
      .def("CollapseWithNext",
        static_cast< Component *(PyClass::*)(const FixedScaleComponent &) const>(&PyClass::CollapseWithNext),
               py::arg("next"))
      .def("CollapseWithPrevious",
        &PyClass::CollapseWithPrevious,
               py::arg("prev"))
      .def("Info",
        &PyClass::Info)
      .def("InitFromString",
        &PyClass::InitFromString,
        py::arg("args"))
      .def("Type",
        &PyClass::Type)
      .def("Copy",
        &PyClass::Copy)
      .def("BackpropNeedsInput",
        &PyClass::BackpropNeedsInput)
      .def("BackpropNeedsOutput",
        &PyClass::BackpropNeedsOutput)
      .def("Propagate",
        static_cast< void (PyClass::*)(const ChunkInfo &,
                 const ChunkInfo &,
                 const CuMatrixBase<BaseFloat> &,
                 CuMatrix<BaseFloat> *) const>(&PyClass::Propagate),
          py::arg("in_info"),
          py::arg("out_info"),
          py::arg("in"),
          py::arg("out"))
      .def("Scale",
        &PyClass::Scale,
          py::arg("scale"))
      .def("Add",
        &PyClass::Add,
          py::arg("alpha"),
          py::arg("other"))
      .def("Backprop",
        &PyClass::Backprop,
          py::arg("in_info"),
          py::arg("out_info"),
          py::arg("in_value"),
          py::arg("out_value"),
          py::arg("out_deriv"),
          py::arg("to_update"),
          py::arg("in_deriv"))
      .def("SetZero",
        &PyClass::SetZero,
          py::arg("treat_as_gradient"))
      .def("Read",
        &PyClass::Read,
        py::arg("is"),
        py::arg("binary"))
      .def("Write",
        &PyClass::Write,
        py::arg("os"),
        py::arg("binary"))
      .def("DotProduct",
        &PyClass::DotProduct,
          py::arg("other"))
      .def("PerturbParams",
        &PyClass::PerturbParams,
          py::arg("stddev"))
      .def("BiasParams",
        &PyClass::BiasParams)
      .def("LinearParams",
        &PyClass::LinearParams)
      .def("GetParameterDim",
        &PyClass::GetParameterDim)
      .def("Vectorize",
        &PyClass::Vectorize,
          py::arg("params"))
      .def("UnVectorize",
        &PyClass::UnVectorize,
          py::arg("params"))
      //.def("LimitRank",
      //  static_cast< void (PyClass::*)(int32,
      //                   AffineComponent **, AffineComponent **) const>(&PyClass::LimitRank),
      //    py::arg("dimension"),
      //    py::arg("a"),
      //    py::arg("b"))
      .def("Widen",
        &PyClass::Widen,
          py::arg("new_dimension"),
          py::arg("param_stddev"),
          py::arg("bias_stddev"),
          py::arg("c2"),
          py::arg("c3"));

    }
    {

      using PyClass = AffineComponentPreconditioned;
      auto affine_component_preconditioned = py::class_<PyClass, AffineComponent>(
          m, "AffineComponentPreconditioned");
      affine_component_preconditioned
          .def(py::init<>())
      .def("Type",
        &PyClass::Type)
      .def("Read",
        &PyClass::Read,
        py::arg("is"),
        py::arg("binary"))
      .def("Write",
        &PyClass::Write,
        py::arg("os"),
        py::arg("binary"))
      .def("Init",
        py::overload_cast<BaseFloat ,
            int32 , int32 ,
            BaseFloat , BaseFloat ,
            BaseFloat , BaseFloat >(&PyClass::Init),
               py::arg("learning_rate"),
               py::arg("input_dim"),
               py::arg("output_dim"),
               py::arg("param_stddev"),
               py::arg("bias_stddev"),
               py::arg("alpha"),
               py::arg("max_change"))
      .def("Init",
        py::overload_cast<BaseFloat , BaseFloat ,
            BaseFloat , std::string >(&PyClass::Init),
               py::arg("learning_rate"),
               py::arg("alpha"),
               py::arg("max_change"),
               py::arg("matrix_filename"))
      .def("InitFromString",
        &PyClass::InitFromString,
        py::arg("args"))
      .def("Info",
        &PyClass::Info)
      .def("Copy",
        &PyClass::Copy)
      .def("SetMaxChange",
        &PyClass::SetMaxChange,
        py::arg("max_change"));

    }
    {

      using PyClass = AffineComponentPreconditionedOnline;
      auto affine_component_preconditioned_online = py::class_<PyClass, AffineComponent>(
          m, "AffineComponentPreconditionedOnline");
      affine_component_preconditioned_online
          .def(py::init<>())
          .def(py::init<const AffineComponent &,
                                      int32 , int32 ,
                                      int32 ,
                                      BaseFloat , BaseFloat >(),
               py::arg("orig"),
               py::arg("rank_in"),
               py::arg("rank_out"),
               py::arg("update_period"),
               py::arg("eta"),
               py::arg("alpha"))
      .def("Type",
        &PyClass::Type)
      .def("Read",
        &PyClass::Read,
        py::arg("is"),
        py::arg("binary"))
      .def("Write",
        &PyClass::Write,
        py::arg("os"),
        py::arg("binary"))
      .def("Init",
        py::overload_cast<BaseFloat ,
            int32 , int32 ,
            BaseFloat , BaseFloat ,
            int32 , int32 , int32 ,
            BaseFloat , BaseFloat ,
            BaseFloat >(&PyClass::Init),
               py::arg("learning_rate"),
               py::arg("input_dim"),
               py::arg("output_dim"),
               py::arg("param_stddev"),
               py::arg("bias_stddev"),
               py::arg("rank_in"),
               py::arg("rank_out"),
               py::arg("update_period"),
               py::arg("num_samples_history"),
               py::arg("alpha"),
               py::arg("max_change_per_sample"))
      .def("Init",
        py::overload_cast<BaseFloat , int32 ,
            int32 , int32 ,
            BaseFloat ,
            BaseFloat , BaseFloat ,
            std::string >(&PyClass::Init),
               py::arg("learning_rate"),
               py::arg("rank_in"),
               py::arg("rank_out"),
               py::arg("update_period"),
               py::arg("num_samples_history"),
               py::arg("alpha"),
               py::arg("max_change_per_sample"),
               py::arg("matrix_filename"))
      .def("Resize",
        &PyClass::Resize,
        py::arg("input_dim"),
        py::arg("output_dim"))
      .def("InitFromString",
        &PyClass::InitFromString,
        py::arg("args"))
      .def("Info",
        &PyClass::Info)
      .def("Copy",
        &PyClass::Copy);

    }

    {

      using PyClass = RandomComponent;
      auto random_component = py::class_<PyClass, Component>(
          m, "RandomComponent");
      random_component
      .def("ResetGenerator",
        &PyClass::ResetGenerator);

    }
    {

      using PyClass = SpliceComponent;
      auto splice_component = py::class_<PyClass, Component>(
          m, "SpliceComponent");
      splice_component
          .def(py::init<>())
      .def("Init",
        &PyClass::Init,
        py::arg("input_dim"),
        py::arg("context"),
        py::arg("const_component_dim") = 0)
      .def("Type",
        &PyClass::Type)
      .def("Copy",
        &PyClass::Copy)
      .def("Context",
        &PyClass::Context)
      .def("BackpropNeedsInput",
        &PyClass::BackpropNeedsInput)
      .def("BackpropNeedsOutput",
        &PyClass::BackpropNeedsOutput)
      .def("Propagate",
        static_cast< void (PyClass::*)(const ChunkInfo &,
                 const ChunkInfo &,
                 const CuMatrixBase<BaseFloat> &,
                 CuMatrix<BaseFloat> *) const>(&PyClass::Propagate),
          py::arg("in_info"),
          py::arg("out_info"),
          py::arg("in"),
          py::arg("out"))
      .def("Backprop",
        &PyClass::Backprop,
          py::arg("in_info"),
          py::arg("out_info"),
          py::arg("in_value"),
          py::arg("out_value"),
          py::arg("out_deriv"),
          py::arg("to_update"),
          py::arg("in_deriv"))
      .def("InputDim",
        &PyClass::InputDim)
      .def("OutputDim",
        &PyClass::OutputDim)
      .def("Read",
        &PyClass::Read,
        py::arg("is"),
        py::arg("binary"))
      .def("Write",
        &PyClass::Write,
        py::arg("os"),
        py::arg("binary"))
      .def("InitFromString",
        &PyClass::InitFromString,
        py::arg("args"))
      .def("Info",
        &PyClass::Info);

    }
    {

      using PyClass = SpliceMaxComponent;
      auto splice_max_component = py::class_<PyClass, Component>(
          m, "SpliceMaxComponent");
      splice_max_component
          .def(py::init<>())
      .def("Init",
        &PyClass::Init,
        py::arg("dim"),
        py::arg("context"))
      .def("Type",
        &PyClass::Type)
      .def("Copy",
        &PyClass::Copy)
      .def("Context",
        &PyClass::Context)
      .def("BackpropNeedsInput",
        &PyClass::BackpropNeedsInput)
      .def("BackpropNeedsOutput",
        &PyClass::BackpropNeedsOutput)
      .def("Propagate",
        static_cast< void (PyClass::*)(const ChunkInfo &,
                 const ChunkInfo &,
                 const CuMatrixBase<BaseFloat> &,
                 CuMatrix<BaseFloat> *) const>(&PyClass::Propagate),
          py::arg("in_info"),
          py::arg("out_info"),
          py::arg("in"),
          py::arg("out"))
      .def("Backprop",
        &PyClass::Backprop,
          py::arg("in_info"),
          py::arg("out_info"),
          py::arg("in_value"),
          py::arg("out_value"),
          py::arg("out_deriv"),
          py::arg("to_update"),
          py::arg("in_deriv"))
      .def("InputDim",
        &PyClass::InputDim)
      .def("OutputDim",
        &PyClass::OutputDim)
      .def("Read",
        &PyClass::Read,
        py::arg("is"),
        py::arg("binary"))
      .def("Write",
        &PyClass::Write,
        py::arg("os"),
        py::arg("binary"))
      .def("InitFromString",
        &PyClass::InitFromString,
        py::arg("args"))
      .def("Info",
        &PyClass::Info);

    }
    {

      using PyClass = BlockAffineComponent;
      auto block_affine_component = py::class_<PyClass, UpdatableComponent>(
          m, "BlockAffineComponent");
      block_affine_component
          .def(py::init<>())
      .def("InputDim",
        &PyClass::InputDim)
      .def("OutputDim",
        &PyClass::OutputDim)
      .def("GetParameterDim",
        &PyClass::GetParameterDim)
      .def("Vectorize",
        &PyClass::Vectorize,
          py::arg("params"))
      .def("UnVectorize",
        &PyClass::UnVectorize,
          py::arg("params"))
      .def("Init",
          &PyClass::Init,
               py::arg("learning_rate"),
               py::arg("input_dim"),
               py::arg("output_dim"),
               py::arg("param_stddev"),
               py::arg("bias_stddev"),
               py::arg("num_blocks"))
      .def("SetZero",
        &PyClass::SetZero,
          py::arg("treat_as_gradient"))
      .def("Read",
        &PyClass::Read,
        py::arg("is"),
        py::arg("binary"))
      .def("Write",
        &PyClass::Write,
        py::arg("os"),
        py::arg("binary"))
      .def("DotProduct",
        &PyClass::DotProduct,
          py::arg("other"))
      .def("Copy",
        &PyClass::Copy)
      .def("PerturbParams",
        &PyClass::PerturbParams,
          py::arg("stddev"))
      .def("Scale",
        &PyClass::Scale,
          py::arg("scale"))
      .def("Add",
        &PyClass::Add,
          py::arg("alpha"),
          py::arg("other"));

    }
    {

      using PyClass = BlockAffineComponentPreconditioned;
      auto block_affine_component_preconditioned = py::class_<PyClass, BlockAffineComponent>(
          m, "BlockAffineComponentPreconditioned");
      block_affine_component_preconditioned
          .def(py::init<>())
      .def("Init",
        &PyClass::Init,
               py::arg("learning_rate"),
               py::arg("input_dim"),
               py::arg("output_dim"),
               py::arg("param_stddev"),
               py::arg("bias_stddev"),
               py::arg("num_blocks"),
               py::arg("alpha"))
      .def("InitFromString",
        &PyClass::InitFromString,
        py::arg("args"))
      .def("Type",
        &PyClass::Type)
      .def("SetZero",
        &PyClass::SetZero,
          py::arg("treat_as_gradient"))
      .def("Read",
        &PyClass::Read,
        py::arg("is"),
        py::arg("binary"))
      .def("Write",
        &PyClass::Write,
        py::arg("os"),
        py::arg("binary"))
      .def("Copy",
        &PyClass::Copy);

    }
    {

      using PyClass = SumGroupComponent;
      auto sum_group_component = py::class_<PyClass, Component>(
          m, "SumGroupComponent");
      sum_group_component
          .def(py::init<>())
      .def("InputDim",
        &PyClass::InputDim)
      .def("OutputDim",
        &PyClass::OutputDim)
      .def("Init",
        &PyClass::Init,
        py::arg("sizes"))
      .def("InitFromString",
        &PyClass::InitFromString,
        py::arg("args"))
      .def("Type",
        &PyClass::Type)
      .def("BackpropNeedsInput",
        &PyClass::BackpropNeedsInput)
      .def("BackpropNeedsOutput",
        &PyClass::BackpropNeedsOutput)
      .def("Propagate",
        static_cast< void (PyClass::*)(const ChunkInfo &,
                 const ChunkInfo &,
                 const CuMatrixBase<BaseFloat> &,
                 CuMatrix<BaseFloat> *) const>(&PyClass::Propagate),
          py::arg("in_info"),
          py::arg("out_info"),
          py::arg("in"),
          py::arg("out"))
      .def("Backprop",
        &PyClass::Backprop,
          py::arg("in_info"),
          py::arg("out_info"),
          py::arg("in_value"),
          py::arg("out_value"),
          py::arg("out_deriv"),
          py::arg("to_update"),
          py::arg("in_deriv"))
      .def("Copy",
        &PyClass::Copy)
      .def("Read",
        &PyClass::Read,
        py::arg("is"),
        py::arg("binary"))
      .def("Write",
        &PyClass::Write,
        py::arg("os"),
        py::arg("binary"));

    }
    {

      using PyClass = PermuteComponent;
      auto permute_component = py::class_<PyClass, Component>(
          m, "PermuteComponent");
      permute_component
          .def(py::init<>())
      .def("Init",
        py::overload_cast<int32>(&PyClass::Init),
        py::arg("dim"))
      .def("Init",
        py::overload_cast<const std::vector<int32> &>(&PyClass::Init),
        py::arg("reorder"))
      .def("InputDim",
        &PyClass::InputDim)
      .def("OutputDim",
        &PyClass::OutputDim)
      .def("Copy",
        &PyClass::Copy)
      .def("InitFromString",
        &PyClass::InitFromString,
        py::arg("args"))
      .def("Read",
        &PyClass::Read,
        py::arg("is"),
        py::arg("binary"))
      .def("Write",
        &PyClass::Write,
        py::arg("os"),
        py::arg("binary"))
      .def("Type",
        &PyClass::Type)
      .def("BackpropNeedsInput",
        &PyClass::BackpropNeedsInput)
      .def("BackpropNeedsOutput",
        &PyClass::BackpropNeedsOutput)
      .def("Propagate",
        static_cast< void (PyClass::*)(const ChunkInfo &,
                 const ChunkInfo &,
                 const CuMatrixBase<BaseFloat> &,
                 CuMatrix<BaseFloat> *) const>(&PyClass::Propagate),
          py::arg("in_info"),
          py::arg("out_info"),
          py::arg("in"),
          py::arg("out"))
      .def("Backprop",
        &PyClass::Backprop,
          py::arg("in_info"),
          py::arg("out_info"),
          py::arg("in_value"),
          py::arg("out_value"),
          py::arg("out_deriv"),
          py::arg("to_update"),
          py::arg("in_deriv"));

    }
    {

      using PyClass = DctComponent;
      auto dct_component = py::class_<PyClass, Component>(
          m, "DctComponent");
      dct_component
          .def(py::init<>())
      .def("Type",
        &PyClass::Type)
      .def("Info",
        &PyClass::Info)
      .def("Init",
        &PyClass::Init,
        py::arg("dim"),
        py::arg("dct_dim"),
        py::arg("reorder"),
        py::arg("keep_dct_dim") = 0)
      .def("InitFromString",
        &PyClass::InitFromString,
        py::arg("args"))
      .def("InputDim",
        &PyClass::InputDim)
      .def("OutputDim",
        &PyClass::OutputDim)
      .def("Copy",
        &PyClass::Copy)
      .def("Read",
        &PyClass::Read,
        py::arg("is"),
        py::arg("binary"))
      .def("Write",
        &PyClass::Write,
        py::arg("os"),
        py::arg("binary"))
      .def("BackpropNeedsInput",
        &PyClass::BackpropNeedsInput)
      .def("BackpropNeedsOutput",
        &PyClass::BackpropNeedsOutput)
      .def("Propagate",
        static_cast< void (PyClass::*)(const ChunkInfo &,
                 const ChunkInfo &,
                 const CuMatrixBase<BaseFloat> &,
                 CuMatrix<BaseFloat> *) const>(&PyClass::Propagate),
          py::arg("in_info"),
          py::arg("out_info"),
          py::arg("in"),
          py::arg("out"))
      .def("Backprop",
        &PyClass::Backprop,
          py::arg("in_info"),
          py::arg("out_info"),
          py::arg("in_value"),
          py::arg("out_value"),
          py::arg("out_deriv"),
          py::arg("to_update"),
          py::arg("in_deriv"));

    }
    {

      using PyClass = FixedLinearComponent;
      auto fixed_linear_component = py::class_<PyClass, Component>(
          m, "FixedLinearComponent");
      fixed_linear_component
          .def(py::init<>())
      .def("Type",
        &PyClass::Type)
      .def("Info",
        &PyClass::Info)
      .def("Init",
        &PyClass::Init,
        py::arg("matrix"))
      .def("InitFromString",
        &PyClass::InitFromString,
        py::arg("args"))
      .def("InputDim",
        &PyClass::InputDim)
      .def("OutputDim",
        &PyClass::OutputDim)
      .def("Copy",
        &PyClass::Copy)
      .def("Read",
        &PyClass::Read,
        py::arg("is"),
        py::arg("binary"))
      .def("Write",
        &PyClass::Write,
        py::arg("os"),
        py::arg("binary"))
      .def("BackpropNeedsInput",
        &PyClass::BackpropNeedsInput)
      .def("BackpropNeedsOutput",
        &PyClass::BackpropNeedsOutput)
      .def("Propagate",
        static_cast< void (PyClass::*)(const ChunkInfo &,
                 const ChunkInfo &,
                 const CuMatrixBase<BaseFloat> &,
                 CuMatrix<BaseFloat> *) const>(&PyClass::Propagate),
          py::arg("in_info"),
          py::arg("out_info"),
          py::arg("in"),
          py::arg("out"))
      .def("Backprop",
        &PyClass::Backprop,
          py::arg("in_info"),
          py::arg("out_info"),
          py::arg("in_value"),
          py::arg("out_value"),
          py::arg("out_deriv"),
          py::arg("to_update"),
          py::arg("in_deriv"));

    }
    {

      using PyClass = FixedAffineComponent;
      auto fixed_affine_component = py::class_<PyClass, Component>(
          m, "FixedAffineComponent");
      fixed_affine_component
          .def(py::init<>())
      .def("Type",
        &PyClass::Type)
      .def("Info",
        &PyClass::Info)
      .def("Init",
        &PyClass::Init,
        py::arg("matrix"))
      .def("InitFromString",
        &PyClass::InitFromString,
        py::arg("args"))
      .def("InputDim",
        &PyClass::InputDim)
      .def("OutputDim",
        &PyClass::OutputDim)
      .def("Copy",
        &PyClass::Copy)
      .def("Read",
        &PyClass::Read,
        py::arg("is"),
        py::arg("binary"))
      .def("Write",
        &PyClass::Write,
        py::arg("os"),
        py::arg("binary"))
      .def("BackpropNeedsInput",
        &PyClass::BackpropNeedsInput)
      .def("BackpropNeedsOutput",
        &PyClass::BackpropNeedsOutput)
      .def("Propagate",
        static_cast< void (PyClass::*)(const ChunkInfo &,
                 const ChunkInfo &,
                 const CuMatrixBase<BaseFloat> &,
                 CuMatrix<BaseFloat> *) const>(&PyClass::Propagate),
          py::arg("in_info"),
          py::arg("out_info"),
          py::arg("in"),
          py::arg("out"))
      .def("Backprop",
        &PyClass::Backprop,
          py::arg("in_info"),
          py::arg("out_info"),
          py::arg("in_value"),
          py::arg("out_value"),
          py::arg("out_deriv"),
          py::arg("to_update"),
          py::arg("in_deriv"))
      .def("LinearParams",
        &PyClass::LinearParams);

    }
    {

      using PyClass = FixedScaleComponent;
      auto fixed_scale_component = py::class_<PyClass, Component>(
          m, "FixedScaleComponent");
      fixed_scale_component
          .def(py::init<>())
      .def("Type",
        &PyClass::Type)
      .def("Info",
        &PyClass::Info)
      .def("Init",
        &PyClass::Init,
        py::arg("matrix"))
      .def("InitFromString",
        &PyClass::InitFromString,
        py::arg("args"))
      .def("InputDim",
        &PyClass::InputDim)
      .def("OutputDim",
        &PyClass::OutputDim)
      .def("Copy",
        &PyClass::Copy)
      .def("Read",
        &PyClass::Read,
        py::arg("is"),
        py::arg("binary"))
      .def("Write",
        &PyClass::Write,
        py::arg("os"),
        py::arg("binary"))
      .def("BackpropNeedsInput",
        &PyClass::BackpropNeedsInput)
      .def("BackpropNeedsOutput",
        &PyClass::BackpropNeedsOutput)
      .def("Propagate",
        static_cast< void (PyClass::*)(const ChunkInfo &,
                 const ChunkInfo &,
                 const CuMatrixBase<BaseFloat> &,
                 CuMatrix<BaseFloat> *) const>(&PyClass::Propagate),
          py::arg("in_info"),
          py::arg("out_info"),
          py::arg("in"),
          py::arg("out"))
      .def("Backprop",
        &PyClass::Backprop,
          py::arg("in_info"),
          py::arg("out_info"),
          py::arg("in_value"),
          py::arg("out_value"),
          py::arg("out_deriv"),
          py::arg("to_update"),
          py::arg("in_deriv"));

    }
    {

      using PyClass = FixedBiasComponent;
      auto fixed_bias_component = py::class_<PyClass, Component>(
          m, "FixedBiasComponent");
      fixed_bias_component
          .def(py::init<>())
      .def("Type",
        &PyClass::Type)
      .def("Info",
        &PyClass::Info)
      .def("Init",
        &PyClass::Init,
        py::arg("matrix"))
      .def("InitFromString",
        &PyClass::InitFromString,
        py::arg("args"))
      .def("InputDim",
        &PyClass::InputDim)
      .def("OutputDim",
        &PyClass::OutputDim)
      .def("Copy",
        &PyClass::Copy)
      .def("Read",
        &PyClass::Read,
        py::arg("is"),
        py::arg("binary"))
      .def("Write",
        &PyClass::Write,
        py::arg("os"),
        py::arg("binary"))
      .def("BackpropNeedsInput",
        &PyClass::BackpropNeedsInput)
      .def("BackpropNeedsOutput",
        &PyClass::BackpropNeedsOutput)
      .def("Propagate",
        static_cast< void (PyClass::*)(const ChunkInfo &,
                 const ChunkInfo &,
                 const CuMatrixBase<BaseFloat> &,
                 CuMatrix<BaseFloat> *) const>(&PyClass::Propagate),
          py::arg("in_info"),
          py::arg("out_info"),
          py::arg("in"),
          py::arg("out"))
      .def("Backprop",
        &PyClass::Backprop,
          py::arg("in_info"),
          py::arg("out_info"),
          py::arg("in_value"),
          py::arg("out_value"),
          py::arg("out_deriv"),
          py::arg("to_update"),
          py::arg("in_deriv"));

    }
    {

      using PyClass = DropoutComponent;
      auto dropout_component = py::class_<PyClass, RandomComponent>(
          m, "DropoutComponent");
      dropout_component
          .def(py::init<>())
          .def(py::init<int32, BaseFloat , BaseFloat>(),
        py::arg("dim"),
        py::arg("dp") = 0.5,
        py::arg("sc")  = 0.0)
      .def("Type",
        &PyClass::Type)
      .def("Info",
        &PyClass::Info)
      .def("Init",
        &PyClass::Init,
        py::arg("dim"),
        py::arg("dropout_proportion") = 0.5,
        py::arg("dropout_scale") = 0.0)
      .def("InitFromString",
        &PyClass::InitFromString,
        py::arg("args"))
      .def("SetDropoutScale",
        &PyClass::SetDropoutScale,
        py::arg("scale"))
      .def("InputDim",
        &PyClass::InputDim)
      .def("OutputDim",
        &PyClass::OutputDim)
      .def("Copy",
        &PyClass::Copy)
      .def("Read",
        &PyClass::Read,
        py::arg("is"),
        py::arg("binary"))
      .def("Write",
        &PyClass::Write,
        py::arg("os"),
        py::arg("binary"))
      .def("BackpropNeedsInput",
        &PyClass::BackpropNeedsInput)
      .def("BackpropNeedsOutput",
        &PyClass::BackpropNeedsOutput)
      .def("Propagate",
        static_cast< void (PyClass::*)(const ChunkInfo &,
                 const ChunkInfo &,
                 const CuMatrixBase<BaseFloat> &,
                 CuMatrix<BaseFloat> *) const>(&PyClass::Propagate),
          py::arg("in_info"),
          py::arg("out_info"),
          py::arg("in"),
          py::arg("out"))
      .def("Backprop",
        &PyClass::Backprop,
          py::arg("in_info"),
          py::arg("out_info"),
          py::arg("in_value"),
          py::arg("out_value"),
          py::arg("out_deriv"),
          py::arg("to_update"),
          py::arg("in_deriv"));

    }
    {

      using PyClass = AdditiveNoiseComponent;
      auto additive_noise_component = py::class_<PyClass, RandomComponent>(
          m, "AdditiveNoiseComponent");
      additive_noise_component
          .def(py::init<>())
          .def(py::init<int32, BaseFloat>(),
        py::arg("dim"),
        py::arg("stddev"))
      .def("Type",
        &PyClass::Type)
      .def("Info",
        &PyClass::Info)
      .def("Init",
        &PyClass::Init,
        py::arg("dim"),
        py::arg("stddev"))
      .def("InitFromString",
        &PyClass::InitFromString,
        py::arg("args"))
      .def("InputDim",
        &PyClass::InputDim)
      .def("OutputDim",
        &PyClass::OutputDim)
      .def("Copy",
        &PyClass::Copy)
      .def("Read",
        &PyClass::Read,
        py::arg("is"),
        py::arg("binary"))
      .def("Write",
        &PyClass::Write,
        py::arg("os"),
        py::arg("binary"))
      .def("BackpropNeedsInput",
        &PyClass::BackpropNeedsInput)
      .def("BackpropNeedsOutput",
        &PyClass::BackpropNeedsOutput)
      .def("Propagate",
        static_cast< void (PyClass::*)(const ChunkInfo &,
                 const ChunkInfo &,
                 const CuMatrixBase<BaseFloat> &,
                 CuMatrix<BaseFloat> *) const>(&PyClass::Propagate),
          py::arg("in_info"),
          py::arg("out_info"),
          py::arg("in"),
          py::arg("out"))
      .def("Backprop",
        &PyClass::Backprop,
          py::arg("in_info"),
          py::arg("out_info"),
          py::arg("in_value"),
          py::arg("out_value"),
          py::arg("out_deriv"),
          py::arg("to_update"),
          py::arg("in_deriv"));

    }
    {

      using PyClass = Convolutional1dComponent;
      auto convolutional_1d_component = py::class_<PyClass, UpdatableComponent>(
          m, "Convolutional1dComponent");
      convolutional_1d_component
          .def(py::init<>())
          .def(py::init<const CuMatrixBase<BaseFloat> &,
                           const CuVectorBase<BaseFloat> &,
                           BaseFloat >(),
          py::arg("filter_params"),
          py::arg("bias_params"),
          py::arg("learning_rate"))
      .def("InputDim",
        &PyClass::InputDim)
      .def("OutputDim",
        &PyClass::OutputDim)
      .def("GetParameterDim",
        &PyClass::GetParameterDim)
      .def("LinearParams",
        &PyClass::LinearParams)
      .def("BiasParams",
        &PyClass::BiasParams)
      .def("Init",
          py::overload_cast<BaseFloat , int32 , int32 ,
            int32 , int32, int32 ,
            BaseFloat , BaseFloat , bool >(&PyClass::Init),
               py::arg("learning_rate"),
               py::arg("input_dim"),
               py::arg("output_dim"),
               py::arg("patch_dim"),
               py::arg("patch_step"),
               py::arg("patch_stride"),
               py::arg("param_stddev"),
               py::arg("bias_stddev"),
               py::arg("appended_conv"))
      .def("Init",
          py::overload_cast<BaseFloat ,
            int32 , int32 , int32 ,
            std::string , bool>(&PyClass::Init),
               py::arg("learning_rate"),
               py::arg("patch_dim"),
               py::arg("patch_step"),
               py::arg("patch_stride"),
               py::arg("matrix_filename"),
               py::arg("appended_conv"))
      .def("Resize",
        &PyClass::Resize,
          py::arg("input_dim"),
          py::arg("output_dim"))
      .def("Update",
        &PyClass::Update,
          py::arg("in_value"),
          py::arg("out_deriv"))
      .def("SetZero",
        &PyClass::SetZero,
          py::arg("treat_as_gradient"))
      .def("Read",
        &PyClass::Read,
        py::arg("is"),
        py::arg("binary"))
      .def("Write",
        &PyClass::Write,
        py::arg("os"),
        py::arg("binary"))
      .def("DotProduct",
        &PyClass::DotProduct,
          py::arg("other"))
      .def("Copy",
        &PyClass::Copy)
      .def("PerturbParams",
        &PyClass::PerturbParams,
          py::arg("stddev"))
      .def("Scale",
        &PyClass::Scale,
          py::arg("scale"))
      .def("Add",
        &PyClass::Add,
          py::arg("alpha"),
          py::arg("other"));

    }
  m.def("ParseFromString",
       py::overload_cast<const std::string &, std::string *,
                     int32 *>( &ParseFromString),
        "Functions used in Init routines.  Suppose name==\"foo\", if \"string\" has a "
          "field like foo=12, this function will set \"param\" to 12 and remove that "
          "element from \"string\".  It returns true if the parameter was read.",
        py::arg("name"),
        py::arg("string"),
        py::arg("param"));
  m.def("ParseFromString",
       py::overload_cast<const std::string &, std::string *,
                     BaseFloat *>( &ParseFromString),
        "Functions used in Init routines.  Suppose name==\"foo\", if \"string\" has a "
          "field like foo=12, this function will set \"param\" to 12 and remove that "
          "element from \"string\".  It returns true if the parameter was read.",
        py::arg("name"),
        py::arg("string"),
        py::arg("param"));
  m.def("ParseFromString",
       py::overload_cast<const std::string &, std::string *,
                     std::vector<int32> *>( &ParseFromString),
        "Functions used in Init routines.  Suppose name==\"foo\", if \"string\" has a "
          "field like foo=12, this function will set \"param\" to 12 and remove that "
          "element from \"string\".  It returns true if the parameter was read.",
        py::arg("name"),
        py::arg("string"),
        py::arg("param"));
  m.def("ParseFromString",
       py::overload_cast<const std::string &, std::string *,
                     bool *>( &ParseFromString),
        "Functions used in Init routines.  Suppose name==\"foo\", if \"string\" has a "
          "field like foo=12, this function will set \"param\" to 12 and remove that "
          "element from \"string\".  It returns true if the parameter was read.",
        py::arg("name"),
        py::arg("string"),
        py::arg("param"));
}

void pybind_nnet2_nnet_compute_discriminative_parallel(py::module& m) {

  m.def("NnetDiscriminativeUpdateParallel",
        &NnetDiscriminativeUpdateParallel,
        py::arg("am_nnet"),
        py::arg("tmodel"),
        py::arg("opts"),
        py::arg("num_threads"),
        py::arg("example_reader"),
        py::arg("nnet_to_update"),
        py::arg("stats"));
}

void pybind_nnet2_nnet_compute_discriminative(py::module& m) {

  py::class_<NnetDiscriminativeUpdateOptions>(m, "NnetDiscriminativeUpdateOptions")
      .def(py::init<>())
      .def_readwrite("criterion", &NnetDiscriminativeUpdateOptions::criterion)
      .def_readwrite("acoustic_scale", &NnetDiscriminativeUpdateOptions::acoustic_scale)
      .def_readwrite("drop_frames", &NnetDiscriminativeUpdateOptions::drop_frames)
      .def_readwrite("one_silence_class", &NnetDiscriminativeUpdateOptions::one_silence_class)
      .def_readwrite("boost", &NnetDiscriminativeUpdateOptions::boost)
      .def_readwrite("silence_phones_str", &NnetDiscriminativeUpdateOptions::silence_phones_str);

  py::class_<NnetDiscriminativeStats>(m, "NnetDiscriminativeStats")
      .def(py::init<>())
      .def_readwrite("tot_t", &NnetDiscriminativeStats::tot_t)
      .def_readwrite("tot_t_weighted", &NnetDiscriminativeStats::tot_t_weighted)
      .def_readwrite("tot_num_count", &NnetDiscriminativeStats::tot_num_count)
      .def_readwrite("tot_num_objf", &NnetDiscriminativeStats::tot_num_objf)
      .def_readwrite("tot_den_objf", &NnetDiscriminativeStats::tot_den_objf)
      .def("Print",
          &NnetDiscriminativeStats::Print,
        py::arg("criterion"))
      .def("Add",
          &NnetDiscriminativeStats::Add,
        py::arg("other"));

  m.def("NnetDiscriminativeUpdate",
        &NnetDiscriminativeUpdate,
        "Does the neural net computation, lattice forward-backward, and backprop, "
          "for either the MMI, MPFE or SMBR objective functions. "
          "If nnet_to_update == &(am_nnet.GetNnet()), then this does stochastic "
          "gradient descent, otherwise (assuming you have called SetZero(true) "
          "on *nnet_to_update) it will compute the gradient on this data. "
          "If nnet_to_update_ == NULL, no backpropagation is done. "

          "Note: we ignore any existing acoustic score in the lattice of \"eg\". "

          "For display purposes you should normalize the sum of this return value by "
          "dividing by the sum over the examples, of the number of frames "
          "(num_ali.size()) times the weight. "

          "Something you need to be careful with is that the occupation counts and the "
          "derivative are, following tradition, missing a factor equal to the acoustic "
          "scale.  So you need to multiply them by that scale if you plan to do "
          "something like L-BFGS in which you look at both the derivatives and function "
          "values.",
        py::arg("am_nnet"),
        py::arg("tmodel"),
        py::arg("opts"),
        py::arg("eg"),
        py::arg("nnet_to_update"),
        py::arg("stats"));
}

void pybind_nnet2_nnet_compute_online(py::module& m) {

  {
    using PyClass = NnetOnlineComputer;

    auto nnet_online_computer = py::class_<PyClass>(
        m, "NnetOnlineComputer");
    nnet_online_computer
      .def(py::init<const Nnet &,
                     bool >(),
        py::arg("nnet"),
        py::arg("pad_input"))
      .def("Compute",
        &PyClass::Compute,
        "This function works as follows: given a chunk of input (interpreted "
          "as following in time any previously supplied data), do the computation "
          "and produce all the frames of output we can.  In the middle of the "
          "file, the dimensions of input and output will be the same, but at "
          "the beginning of the file, output will have fewer frames than input "
          "due to required context. "
          "It is the responsibility of the user to keep track of frame indices, if "
          "required.  This class won't output any frame twice.",
        py::arg("input"),
        py::arg("output"))
      .def("Flush",
        &PyClass::Flush,
        "This flushes out the last frames of output; you call this when all "
          "input has finished.  It's invalid to call Compute or Flush after "
          "calling Flush.  It's valid to call Flush if no frames have been "
          "input or if no frames have been output; this produces empty output.",
        py::arg("output"));
  }
}

void pybind_nnet2_nnet_compute(py::module& m) {


  m.def("NnetComputation",
        &NnetComputation,
        "Does the basic neural net computation, on a sequence of data (e.g. "
          "an utterance).  If pad_input==true we'll pad the input with enough "
          "frames of context, and the output will be a matrix of #frames by "
          "the output-dim of the network, typically representing state-level "
          "posteriors.   If pad_input==false we won't do this and the "
          "output will have a lower #frames than the input; we lose "
          "nnet.LeftContext() at the left and nnet.RightContext() at the "
          "output.",
        py::arg("nnet"),
        py::arg("input"),
        py::arg("pad_input"),
        py::arg("output"));

  m.def("NnetComputationChunked",
        &NnetComputationChunked,
        "Does the basic neural net computation, on a sequence of data (e.g. "
          "an utterance).  This variant of NnetComputation chunks the input "
          "according to chunk_size and does the posterior computation chunk  "
          "by chunk.  This allows the computation to be performed on the GPU "
          "when the input matrix is very large.  Input is padded with enough "
          "frames of context so that the output will be a matrix with  "
          "input.NumRows().",
        py::arg("nnet"),
        py::arg("input"),
        py::arg("chunk_size"),
        py::arg("output"));

  /*m.def("NnetGradientComputation",
        &NnetGradientComputation,
        "Does the neural net computation and backprop, given input and labels. "
          "Note: if pad_input==true the number of rows of input should be the "
          "same as the number of labels, and if false, you should omit "
          "nnet.LeftContext() labels on the left and nnet.RightContext() on "
          "the right.  If nnet_to_update == &nnet, then this does stochastic "
          "gradient descent, otherwise (assuming you have called SetZero(true) "
          "on *nnet_to_update) it will compute the gradient on this data. "
          "Returns the total objective function summed over the frames, times "
          "the utterance weight.",
        py::arg("nnet"),
        py::arg("input"),
        py::arg("pad_input"),
        py::arg("utterance_weight"),
        py::arg("labels"),
        py::arg("nnet_to_update"));*/
}

void pybind_nnet2_nnet_example_functions(py::module& m) {

  py::class_<SplitDiscriminativeExampleConfig>(m, "SplitDiscriminativeExampleConfig")
      .def(py::init<>())
      .def_readwrite("max_length", &SplitDiscriminativeExampleConfig::max_length)
      .def_readwrite("criterion", &SplitDiscriminativeExampleConfig::criterion)
      .def_readwrite("collapse_transition_ids", &SplitDiscriminativeExampleConfig::collapse_transition_ids)
      .def_readwrite("determinize", &SplitDiscriminativeExampleConfig::determinize)
      .def_readwrite("minimize", &SplitDiscriminativeExampleConfig::minimize)
      .def_readwrite("test", &SplitDiscriminativeExampleConfig::test)
      .def_readwrite("drop_frames", &SplitDiscriminativeExampleConfig::drop_frames)
      .def_readwrite("split", &SplitDiscriminativeExampleConfig::split)
      .def_readwrite("excise", &SplitDiscriminativeExampleConfig::excise);

  py::class_<SplitExampleStats>(m, "SplitExampleStats")
      .def(py::init<>())
      .def_readwrite("num_lattices", &SplitExampleStats::num_lattices)
      .def_readwrite("longest_lattice", &SplitExampleStats::longest_lattice)
      .def_readwrite("num_segments", &SplitExampleStats::num_segments)
      .def_readwrite("num_kept_segments", &SplitExampleStats::num_kept_segments)
      .def_readwrite("num_frames_orig", &SplitExampleStats::num_frames_orig)
      .def_readwrite("num_frames_must_keep", &SplitExampleStats::num_frames_must_keep)
      .def_readwrite("num_frames_kept_after_split", &SplitExampleStats::num_frames_kept_after_split)
      .def_readwrite("longest_segment_after_split", &SplitExampleStats::longest_segment_after_split)
      .def_readwrite("num_frames_kept_after_excise", &SplitExampleStats::num_frames_kept_after_excise)
      .def_readwrite("longest_segment_after_excise", &SplitExampleStats::longest_segment_after_excise)
      .def("Print",
          &SplitExampleStats::Print);
  m.def("LatticeToDiscriminativeExample",
        &LatticeToDiscriminativeExample,
        "Converts lattice to discriminative training example.  returns true on "
          "success, false on failure such as mismatched input (will also warn in this "
          "case).",
        py::arg("alignment"),
        py::arg("feats"),
        py::arg("clat"),
        py::arg("weight"),
        py::arg("left_context"),
        py::arg("right_context"),
        py::arg("eg"));
  m.def("SplitDiscriminativeExample",
        &SplitDiscriminativeExample,
        "Split a \"discriminative example\" into multiple pieces, "
          "splitting where the lattice has \"pinch points\".",
        py::arg("config"),
        py::arg("tmodel"),
        py::arg("eg"),
        py::arg("egs_out"),
        py::arg("stats_out"));
  m.def("ExciseDiscriminativeExample",
        &ExciseDiscriminativeExample,
        "Remove unnecessary frames from discriminative training "
          "example.  The output egs_out will be of size zero or one "
          "(usually one) after being called.",
        py::arg("config"),
        py::arg("tmodel"),
        py::arg("eg"),
        py::arg("egs_out"),
        py::arg("stats_out"));
  m.def("AppendDiscriminativeExamples",
        &AppendDiscriminativeExamples,
        "Appends the given vector of examples (which must be non-empty) into "
          "a single output example (called by CombineExamples, which might be "
          "a more convenient interface). "
          "\n"
          "When combining examples it directly appends the features, and then adds a "
          "\"fake\" segment to the lattice and alignment in between, padding with "
          "transition-ids that are all ones.  This is necessary in case the network "
          "needs acoustic context, and only because of a kind of limitation in the nnet "
          "training code that doesn't support varying 'chunk' sizes within a minibatch. "
          "\n"
          "Will fail if all the input examples don't have the same weight (this will "
          "normally be 1.0 anyway), or if the feature dimension (i.e. basic feature "
          "dimension plus spk_info dimension) differs between the examples.",
        py::arg("input"),
        py::arg("output"));
  m.def("CombineDiscriminativeExamples",
        &CombineDiscriminativeExamples,
        "This function is used to combine multiple discriminative-training "
          "examples (each corresponding to a segment of a lattice), into one. "

          "It combines examples into groups such that each group will have a "
          "total length (number of rows of the feature matrix) less than or "
          "equal to max_length.  However, if individual examples are longer "
          "than max_length they will still be processed; they will be given "
          "their own group. "

          "See also the documentation for AppendDiscriminativeExamples() which "
          "gives more details on how we append the examples. "

          "Will fail if all the input examples don't have the same weight (this will "
          "normally be 1.0 anyway). "

          "If the spk_info variables are non-empty, it will move them into the features "
          "of the output, so the spk_info of the output will be empty but the "
          "appropriate speaker vectors will be appended to each row of the features.",
        py::arg("max_length"),
        py::arg("input"),
        py::arg("output"));
  m.def("SolvePackingProblem",
        &SolvePackingProblem,
        "This function solves the \"packing problem\" using the \"first fit\" algorithm. "
   "It groups together the indices 0 through sizes.size() - 1, such that the sum "
   "of cost within each group does not exceed max_lcost.  [However, if there "
   "are single examples that exceed max_cost, it puts them in their own bin]. "
   "The algorithm is not particularly efficient-- it's more n^2 than n log(n) "
   "which it should be.",
        py::arg("max_cost"),
        py::arg("costs"),
        py::arg("groups"));
  m.def("ExampleToPdfPost",
        &ExampleToPdfPost,
        "Given a discriminative training example, this function works out posteriors "
          "at the pdf level (note: these are \"discriminative-training posteriors\" that "
          "may be positive or negative.  The denominator lattice \"den_lat\" in the "
          "example \"eg\" should already have had acoustic-rescoring done so that its "
          "acoustic probs are up to date, and any acoustic scaling should already have "
          "been applied. "
          "\n"
          "\"criterion\" may be \"mmi\" or \"mpfe\" or \"smbr\".  If criterion "
          "is \"mmi\", \"drop_frames\" means we don't include derivatives for frames "
          "where the numerator pdf is not in the denominator lattice. "
          "\n"
          "if \"one_silence_class\" is true you can get a newer behavior for MPE/SMBR "
          "which will tend to reduce insertions. "
          "\n"
          "\"silence_phones\" is a list of silence phones (this is only relevant for mpfe "
          "or smbr, if we want to treat silence specially).",
        py::arg("tmodel"),
        py::arg("silence_phones"),
        py::arg("criterion"),
        py::arg("drop_frames"),
        py::arg("one_silence_class"),
        py::arg("eg"),
        py::arg("post"));
  m.def("UpdateHash",
        &UpdateHash,
        "This function is used in code that tests the functionality that we provide "
          "here, about splitting and excising nnet examples.  It adds to a \"hash "
          "function\" that is a function of a set of examples; the hash function is of "
          "dimension (number of pdf-ids x features dimension).  The hash function "
          "consists of the (denominator - numerator) posteriors over pdf-ids, times the "
          "average over the context-window (left-context on the left, right-context on "
          "the right), of the features.  This is useful because the various "
          "manipulations we do are supposed to preserve this, and if there is a bug "
          "it will most likely cause the hash function to change. "
          "\n"
          "This function will resize the matrix if it is empty. "
          "\n"
          "Any acoustic scaling of the lattice should be done before you call this "
          "function. "
          "\n"
          "'criterion' should be 'mmi', 'mpfe', or 'smbr'. "
          "\n"
          "You should set drop_frames to true if you are doing MMI with drop-frames "
          "== true.  Then it will not compute the hash for frames where the numerator "
          "pdf-id is not in the denominator lattice. "
          "\n"
          "You can set one_silence_class to true for a newer optional behavior that will "
          "reduce insertions in the trained model (or false for the traditional "
          "behavior). "
          "\n"
          "The function will also accumulate the total numerator and denominator weights "
          "used as num_weight and den_weight, for an additional diagnostic, and the total "
          "number of frames, as tot_t.",
        py::arg("tmodel"),
        py::arg("eg"),
        py::arg("criterion"),
        py::arg("drop_frames"),
        py::arg("one_silence_class"),
        py::arg("hash"),
        py::arg("num_weight"),
        py::arg("den_weight"),
        py::arg("tot_t"));
}

void pybind_nnet2_nnet_example(py::module& m) {

  {
    using PyClass = NnetExample;

    auto nnet_example = py::class_<PyClass>(
        m, "NnetExample");
    nnet_example
     .def(py::init<>())
      .def_readwrite("labels", &PyClass::labels)
      .def_readwrite("input_frames", &PyClass::input_frames)
      .def_readwrite("left_context", &PyClass::left_context)
      .def_readwrite("spk_info", &PyClass::spk_info)
      .def(py::init<const NnetExample &,
              int32 ,
              int32 ,
              int32 ,
              int32 >(),
        py::arg("input"),
        py::arg("start_frame"),
        py::arg("num_frames"),
        py::arg("left_context"),
        py::arg("right_context"))
      .def("Read",
        &PyClass::Read,
        py::arg("is"),
        py::arg("binary"))
      .def("Write",
        &PyClass::Write,
        py::arg("os"),
        py::arg("binary"))
      .def("SetLabelSingle",
        &PyClass::SetLabelSingle,
        py::arg("frame"),
        py::arg("pdf_id"),
        py::arg("weight") = 1.0)
      .def("GetLabelSingle",
        &PyClass::GetLabelSingle,
        py::arg("frame"),
        py::arg("weight") = NULL);
  }
  pybind_table_writer<KaldiObjectHolder<NnetExample >>(m, "_NnetExampleWriter");
  pybind_sequential_table_reader<KaldiObjectHolder<NnetExample >>(
      m, "_SequentialNnetExampleReader");

  pybind_random_access_table_reader<KaldiObjectHolder<NnetExample >>(
      m, "_RandomAccessNnetExampleReader");
}

void pybind_nnet2_nnet_fix(py::module& m) {


  py::class_<NnetFixConfig>(m, "NnetFixConfig")
      .def(py::init<>())
      .def_readwrite("min_average_deriv", &NnetFixConfig::min_average_deriv)
      .def_readwrite("max_average_deriv", &NnetFixConfig::max_average_deriv)
      .def_readwrite("parameter_factor", &NnetFixConfig::parameter_factor)
      .def_readwrite("relu_bias_change", &NnetFixConfig::relu_bias_change);
  m.def("FixNnet",
        &FixNnet,
        py::arg("config"),
        py::arg("nnet"));
}

void pybind_nnet2_nnet_functions(py::module& m) {
  m.def("IndexOfSoftmaxLayer",
        &IndexOfSoftmaxLayer,
        py::arg("nnet"));
  m.def("InsertComponents",
        &InsertComponents,
        py::arg("src_nnet"),
        py::arg("c"),
        py::arg("dest_nnet"));
  m.def("ReplaceLastComponents",
        &ReplaceLastComponents,
        py::arg("src_nnet"),
        py::arg("num_to_remove"),
        py::arg("dest_nnet"));

}

void pybind_nnet2_nnet_limit_rank(py::module& m) {


  py::class_<NnetLimitRankOpts>(m, "NnetLimitRankOpts")
      .def(py::init<>())
      .def_readwrite("num_threads", &NnetLimitRankOpts::num_threads)
      .def_readwrite("parameter_proportion", &NnetLimitRankOpts::parameter_proportion);
  m.def("LimitRankParallel",
        &LimitRankParallel,
        "This function limits the rank of each affine transform in the "
          "neural net, by zeroing out the smallest singular values.  The number of "
          "singular values to zero out is determined on a layer by layer basis, using "
          "\"parameter_proportion\" to set the proportion of parameters to remove.",
        py::arg("opts"),
        py::arg("nnet"));
}

void pybind_nnet2_nnet_nnet(py::module& m) {

  {
    using PyClass = Nnet;

    auto nnet2_nnet = py::class_<PyClass>(
        m, "Nnet");
    nnet2_nnet.def(py::init<>())
      .def(py::init<const Nnet &>(),
        py::arg("other"))
      .def(py::init<const Nnet &, const Nnet &>(),
        py::arg("nnet1"),
        py::arg("nnet2"))
      .def("NumComponents",
        &PyClass::NumComponents,
        "Returns number of components-- think of this as similar to # of layers, but "
          "e.g. the nonlinearity and the linear part count as separate components, "
          "so the number of components will be more than the number of layers.")
      .def("GetComponent", static_cast< const Component & (PyClass::*)(int32) const>(&PyClass::GetComponent),
        py::arg("c"))
      .def("GetComponent", static_cast< Component & (PyClass::*)(int32)>(&PyClass::GetComponent),
        py::arg("c"))
      .def("SetComponent", &PyClass::SetComponent,
        py::arg("c"),
        py::arg("component"))
      .def("LeftContext", &PyClass::LeftContext)
      .def("RightContext", &PyClass::RightContext)
      .def("OutputDim", &PyClass::OutputDim)
      .def("InputDim", &PyClass::InputDim)
      .def("ComputeChunkInfo", &PyClass::ComputeChunkInfo,
        py::arg("input_chunk_size"),
        py::arg("num_chunks"),
        py::arg("chunk_info_out"))
      .def("ZeroStats", &PyClass::ZeroStats)
      .def("CopyStatsFrom", &PyClass::CopyStatsFrom,
        py::arg("nnet"))
      .def("FirstUpdatableComponent", &PyClass::FirstUpdatableComponent)
      .def("LastUpdatableComponent", &PyClass::LastUpdatableComponent)
      .def("NumUpdatableComponents", &PyClass::NumUpdatableComponents)
      .def("ScaleComponents", &PyClass::ScaleComponents,
        py::arg("scales"))
      .def("RemoveDropout", &PyClass::RemoveDropout)
      .def("SetDropoutScale", &PyClass::SetDropoutScale,
        py::arg("scale"))
      .def("RemovePreconditioning", &PyClass::RemovePreconditioning)
      .def("SwitchToOnlinePreconditioning", &PyClass::SwitchToOnlinePreconditioning,
        py::arg("rank_in"),
        py::arg("rank_out"),
        py::arg("update_period"),
        py::arg("num_samples_history"),
        py::arg("alpha"))
      .def("AddNnet",
          py::overload_cast<const VectorBase<BaseFloat> & ,
               const Nnet &>(&PyClass::AddNnet),
        py::arg("scales"),
        py::arg("other"))
      .def("Scale", &PyClass::Scale,
        py::arg("scale"))
      .def("AddNnet",
          py::overload_cast<BaseFloat ,
               const Nnet &>(&PyClass::AddNnet),
        py::arg("alpha"),
        py::arg("other"))
      .def("LimitRankOfLastLayer", &PyClass::LimitRankOfLastLayer,
        py::arg("dimension"))
      .def("AddNnet",
          py::overload_cast<BaseFloat ,
               Nnet *, BaseFloat>(&PyClass::AddNnet),
        py::arg("alpha"),
        py::arg("other"),
        py::arg("beta"))
      .def("Resize", &PyClass::Resize,
        py::arg("num_components"))
      .def("Collapse", &PyClass::Collapse,
        py::arg("match_updatableness"))
      .def("SetIndexes", &PyClass::SetIndexes)
      .def("Init", py::overload_cast<std::istream &>(&PyClass::Init),
        py::arg("is"))
      .def("Init", py::overload_cast<std::vector<Component*> *>(&PyClass::Init),
        py::arg("components"))
      .def("Append", &PyClass::Append,
        py::arg("new_component"))
      .def("Info", &PyClass::Info)
      .def("Destroy", &PyClass::Destroy)
      .def("Write",
        &PyClass::Write,
        py::arg("os"),
        py::arg("binary"))
      .def("Read",
        &PyClass::Read,
        py::arg("is"),
        py::arg("binary"))
      .def("SetZero",
        &PyClass::SetZero,
        py::arg("treat_as_gradient"))
      .def("ResizeOutputLayer",
        &PyClass::ResizeOutputLayer,
        py::arg("new_num_pdfs"))
      .def("ScaleLearningRates",
        py::overload_cast<BaseFloat>(&PyClass::ScaleLearningRates),
        py::arg("factor"))
      .def("ScaleLearningRates",
        py::overload_cast<std::map<std::string, BaseFloat>>(&PyClass::ScaleLearningRates),
        py::arg("scale_factors"))
      .def("SetLearningRates",
        py::overload_cast<BaseFloat>(&PyClass::SetLearningRates),
        py::arg("learning_rates"))
      .def("SetLearningRates",
        py::overload_cast<const VectorBase<BaseFloat> &>(&PyClass::SetLearningRates),
        py::arg("learning_rates"))
      .def("GetLearningRates",
        &PyClass::GetLearningRates,
        py::arg("learning_rates"))
      .def("ComponentDotProducts",
        &PyClass::ComponentDotProducts,
        py::arg("other"),
        py::arg("dot_prod"))
      .def("Check",
        &PyClass::Check)
      .def("ResetGenerators",
        &PyClass::ResetGenerators)
      .def("GetParameterDim",
        &PyClass::GetParameterDim)
      .def("Vectorize",
        &PyClass::Vectorize,
        py::arg("params"))
      .def("UnVectorize",
        &PyClass::UnVectorize,
        py::arg("params"));
  }
}

void pybind_nnet2_nnet_precondition_online(py::module& m) {

  {
    using PyClass = OnlinePreconditioner;

    auto online_preconditioner = py::class_<PyClass>(
        m, "OnlinePreconditioner");
    online_preconditioner.def(py::init<>())
      .def(py::init<const OnlinePreconditioner &>(),
        py::arg("other"))
      .def("SetRank",
        &PyClass::SetRank,
        py::arg("rank"))
      .def("SetUpdatePeriod",
        &PyClass::SetUpdatePeriod,
        py::arg("update_period"))
      .def("SetNumSamplesHistory",
        &PyClass::SetNumSamplesHistory,
        py::arg("num_samples_history"))
      .def("SetAlpha",
        &PyClass::SetAlpha,
        py::arg("alpha"))
      .def("TurnOnDebug",
        &PyClass::TurnOnDebug)
      .def("GetNumSamplesHistory",
        &PyClass::GetNumSamplesHistory)
      .def("GetAlpha",
        &PyClass::GetAlpha)
      .def("GetRank",
        &PyClass::GetRank)
      .def("GetUpdatePeriod",
        &PyClass::GetUpdatePeriod)
      .def("PreconditionDirections",
        &PyClass::PreconditionDirections,
        py::arg("R"),
        py::arg("row_prod"),
        py::arg("scale"));
  }
}

void pybind_nnet2_nnet_precondition(py::module& m) {

  m.def("PreconditionDirections",
        &PreconditionDirections,
        "The function PreconditionDirections views the input R as "
          "a set of directions or gradients, each row r_i being one of the "
          "directions.  For each i it constructs a preconditioning matrix "
          "G_i formed from the *other* i's, using the formula: "
          "\n"
          "G_i = (\\lambda I + (1/(N-1)) \\sum_{j \neq i} r_j r_j^T)^{-1}, "
          "\n"
          "where N is the number of rows in R.  This can be seen as a kind "
          "of estimated Fisher matrix that has been smoothed with the "
          "identity to make it invertible.  We recommend that you set "
          "\\lambda using: "
          "\\lambda = \alpha/(N D) trace(R^T, R) "
          "for some small \alpha such as \\alpha = 0.1.  However, we leave "
          "this to the caller because there are reasons relating to "
          "unbiased-ness of the resulting stochastic gradient descent, why you "
          "might want to set \\lambda using \"other\" data, e.g. a previous "
          "minibatch. "
          "\n"
          "The output of this function is a matrix P, each row p_i of "
          "which is related to r_i by: "
          "p_i = G_i r_i "
          "Here, p_i is preconditioned by an estimated Fisher matrix "
          "in such a way that it's suitable to be used as an update direction.",
        py::arg("R"),
        py::arg("lambda"),
        py::arg("P"));
  m.def("PreconditionDirectionsAlphaRescaled",
        &PreconditionDirectionsAlphaRescaled,
        "This wrapper for PreconditionDirections computes lambda "
         "using \\lambda = \\alpha/(N D) trace(R^T, R), and calls "
          "PreconditionDirections.  It then rescales *P so that "
          "its 2-norm is the same as that of R. ",
        py::arg("R"),
        py::arg("lambda"),
        py::arg("P"));
}

void pybind_nnet2_nnet_stats(py::module& m) {


  py::class_<NnetStatsConfig>(m, "NnetStatsConfig")
      .def(py::init<>())
      .def_readwrite("bucket_width", &NnetStatsConfig::bucket_width);

  {
    using PyClass = NnetStats;

    auto nnet_stats = py::class_<PyClass>(
        m, "NnetStats");
    nnet_stats
      .def(py::init<int32 , BaseFloat>(),
        py::arg("affine_component_index"),
        py::arg("bucket_width"))
      .def("AddStats",
        &PyClass::AddStats,
        py::arg("avg_deriv"),
        py::arg("avg_value"))
      .def("AddStatsFromNnet",
        &PyClass::AddStatsFromNnet,
        py::arg("nnet"))
      .def("PrintStats",
        &PyClass::PrintStats,
        py::arg("os"));
  }
  m.def("GetNnetStats",
        &GetNnetStats,
        py::arg("config"),
        py::arg("nnet"),
        py::arg("stats"));
}

void pybind_nnet2_nnet_update_parallel(py::module& m) {

  m.def("DoBackpropParallel",
        py::overload_cast<const Nnet &,
                          int32,
                          SequentialNnetExampleReader *,
                          double *,
                          Nnet *>(&DoBackpropParallel),
        "This function is similar to \"DoBackprop\" in nnet-update.h "
          "This function computes the objective function and either updates the model "
          "or computes parameter gradients.  It returns the cross-entropy objective "
          "function summed over all samples, weighted, and the total weight of "
          "the samples (typically the same as the #frames) into total_weight. "
          "It is mostly a wrapper for "
          "a class NnetUpdater that's defined in nnet-update.cc, but we "
          "don't want to expose that complexity at this level. "
          "Note: this function  "
          "If &nnet == nnet_to_update, it assumes we're doing SGD and does "
          "something like Hogwild; otherwise it assumes we're computing a "
          "gradient and it sums up the gradients. "
          "The return value is the total log-prob summed over the #frames. It also "
          "outputs the #frames into \"num_frames\".",
        py::arg("nnet"),
        py::arg("minibatch_size"),
        py::arg("example_reader"),
        py::arg("tot_weight"),
        py::arg("nnet_to_update"));
  m.def("DoBackpropParallel",
        py::overload_cast<const Nnet &,
                          int32 ,
                          int32 ,
                          const std::vector<NnetExample> &,
                          double *,
                          Nnet *>(&DoBackpropParallel),
        "This version of DoBackpropParallel takes a vector of examples, and will "
          "typically be used to compute the exact gradient. ",
        py::arg("nnet"),
        py::arg("minibatch_size"),
        py::arg("num_threads"),
        py::arg("examples"),
        py::arg("num_frames"),
        py::arg("nnet_to_update"));
  m.def("ComputeNnetObjfParallel",
        &ComputeNnetObjfParallel,
        "This is basically to clarify the fact that DoBackpropParallel will "
          "also work with nnet_to_update == NULL, and will compute the objf. "
          "Both versions of the function will support it, but this "
          "version (that takes a vector) is currently the only one we need "
          "to do this with.",
        py::arg("nnet"),
        py::arg("minibatch_size"),
        py::arg("num_threads"),
        py::arg("examples"),
        py::arg("num_frames"));
}

void pybind_nnet2_nnet_update(py::module& m) {

  {
    using PyClass = NnetUpdater;

    auto nnet_updater = py::class_<PyClass>(
        m, "NnetUpdater");
    nnet_updater
      .def(py::init<const Nnet &,
              Nnet *>(),
        py::arg("nnet"),
        py::arg("nnet_to_update"))
      .def("ComputeForMinibatch",
        py::overload_cast<const std::vector<NnetExample> &,
                             double *>(&PyClass::ComputeForMinibatch),
        py::arg("data"),
        py::arg("tot_accuracy"))
      .def("ComputeForMinibatch",
        py::overload_cast<const std::vector<NnetExample> &,
                             Matrix<BaseFloat> *,
                             double *>(&PyClass::ComputeForMinibatch),
        py::arg("data"),
        py::arg("formatted_data"),
        py::arg("tot_accuracy"))
      .def("GetOutput",
        &PyClass::GetOutput,
        py::arg("output"));
  }
  m.def("FormatNnetInput",
        &FormatNnetInput,
        "Takes the input to the nnet for a minibatch of examples, and formats as a "
          "single matrix.  data.size() must be > 0.  Note: you will probably want to "
          "copy this to CuMatrix after you call this function. "
          "The num-rows of the output will, at exit, equal  "
          "(1 + nnet.LeftContext() + nnet.RightContext()) * data.size(). "
          "The nnet is only needed so we can call LeftContext(), RightContext() "
          "and InputDim() on it.",
        py::arg("nnet"),
        py::arg("data"),
        py::arg("mat"));
  m.def("DoBackprop",
        py::overload_cast<const Nnet &,
                  const std::vector<NnetExample> &,
                  Nnet *,
                  double *>(&DoBackprop),
        "This function computes the objective function and either updates the model "
          "or adds to parameter gradients.  Returns the cross-entropy objective "
          "function summed over all samples (normalize this by dividing by "
          "TotalNnetTrainingWeight(examples)).  It is mostly a wrapper for "
          "a class NnetUpdater that's defined in nnet-update.cc, but we "
          "don't want to expose that complexity at this level. "
          "All these examples will be treated as one minibatch. "
          "If tot_accuracy != NULL, it outputs to that pointer the total (weighted) "
          "accuracy.",
        py::arg("nnet"),
        py::arg("examples"),
        py::arg("nnet_to_update"),
        py::arg("tot_accuracy") = NULL);
  m.def("DoBackprop",
        py::overload_cast<const Nnet &,
                  const std::vector<NnetExample> &,
                  Matrix<BaseFloat> *,
                  Nnet *,
                  double *>(&DoBackprop),
        "This version of DoBackprop allows you to separately call "
          "FormatNnetInput and provide the result to DoBackprop; this "
          "can be useful when using GPUs because the call to FormatNnetInput "
          "can be in a separate thread from the one that uses the GPU. "
          "\"examples_formatted\" is really an input, but it's a pointer "
          "because internally we call Swap() on it, so we destroy "
          "its contents.",
        py::arg("nnet"),
        py::arg("examples"),
        py::arg("examples_formatted"),
        py::arg("nnet_to_update"),
        py::arg("tot_accuracy") = NULL);
  m.def("TotalNnetTrainingWeight",
        &TotalNnetTrainingWeight,
        "Returns the total weight summed over all the examples... just a simple utility function.",
        py::arg("egs"));
  m.def("ComputeNnetObjf",
        py::overload_cast<const Nnet &,
                       const std::vector<NnetExample> &,
                       double *>(&ComputeNnetObjf),
        "Computes objective function over a minibatch.  Returns the *total* weighted "
          "objective function over the minibatch. "
          "If tot_accuracy != NULL, it outputs to that pointer the total (weighted) "
          "accuracy.",
        py::arg("nnet"),
        py::arg("examples"),
        py::arg("tot_accuracy") = NULL);
  m.def("ComputeNnetObjf",
        py::overload_cast<const Nnet &,
                       const std::vector<NnetExample> &,
                       int32,
                       double *>(&ComputeNnetObjf),
        "This version of ComputeNnetObjf breaks up the examples into "
          "multiple minibatches to do the computation. "
          "Returns the *total* (weighted) objective function. "
          "If tot_accuracy != NULL, it outputs to that pointer the total (weighted) "
          "accuracy.",
        py::arg("nnet"),
        py::arg("examples"),
        py::arg("minibatch_size"),
        py::arg("tot_accuracy") = NULL);
  m.def("ComputeNnetGradient",
        &ComputeNnetGradient,
        "ComputeNnetGradient is mostly used to compute gradients on validation sets; "
          "it divides the example into batches and calls DoBackprop() on each. "
          "It returns the *average* objective function per frame.",
        py::arg("nnet"),
        py::arg("examples"),
        py::arg("batch_size"),
        py::arg("gradient"));
}

void pybind_nnet2_online_nnet2_decodable(py::module& m) {

  py::class_<DecodableNnet2OnlineOptions>(m, "DecodableNnet2OnlineOptions")
      .def(py::init<>())
      .def_readwrite("acoustic_scale", &DecodableNnet2OnlineOptions::acoustic_scale)
      .def_readwrite("pad_input", &DecodableNnet2OnlineOptions::pad_input)
      .def_readwrite("max_nnet_batch_size", &DecodableNnet2OnlineOptions::max_nnet_batch_size);


  {
    using PyClass = DecodableNnet2Online;

    auto decodable_nnet2_online = py::class_<PyClass, DecodableInterface>(
        m, "DecodableNnet2Online");
    decodable_nnet2_online
      .def(py::init<const AmNnet &,
                       const TransitionModel &,
                       const DecodableNnet2OnlineOptions &,
                       OnlineFeatureInterface *>(),
        py::arg("nnet"),
        py::arg("trans_model"),
        py::arg("opts"),
        py::arg("input_feats"))
      .def("LogLikelihood",
        &PyClass::LogLikelihood,
        py::arg("frame"),
        py::arg("index"))
      .def("IsLastFrame",
        &PyClass::IsLastFrame,
        py::arg("frame"))
      .def("NumFramesReady",
        &PyClass::NumFramesReady)
      .def("NumIndices",
        &PyClass::NumIndices);
  }
}

void pybind_nnet2_rescale_nnet(py::module& m) {

  py::class_<NnetRescaleConfig>(m, "NnetRescaleConfig")
      .def(py::init<>())
      .def_readwrite("target_avg_deriv", &NnetRescaleConfig::target_avg_deriv)
      .def_readwrite("target_first_layer_avg_deriv", &NnetRescaleConfig::target_first_layer_avg_deriv)
      .def_readwrite("target_last_layer_avg_deriv", &NnetRescaleConfig::target_last_layer_avg_deriv)
      .def_readwrite("num_iters", &NnetRescaleConfig::num_iters)
      .def_readwrite("delta", &NnetRescaleConfig::delta)
      .def_readwrite("max_change", &NnetRescaleConfig::max_change)
      .def_readwrite("min_change", &NnetRescaleConfig::min_change);
  m.def("RescaleNnet",
        &RescaleNnet,
        py::arg("rescale_config"),
        py::arg("examples"),
        py::arg("nnet"));

}

void pybind_nnet2_shrink_nnet(py::module& m) {

  py::class_<NnetShrinkConfig>(m, "NnetShrinkConfig")
      .def(py::init<>())
      .def_readwrite("num_bfgs_iters", &NnetShrinkConfig::num_bfgs_iters)
      .def_readwrite("initial_step", &NnetShrinkConfig::initial_step);
  m.def("ShrinkNnet",
        &ShrinkNnet,
        py::arg("shrink_config"),
        py::arg("validation_set"),
        py::arg("nnet"));

}

void pybind_nnet2_train_nnet_ensemble(py::module& m) {


  py::class_<NnetEnsembleTrainerConfig>(m, "NnetEnsembleTrainerConfig")
      .def(py::init<>())
      .def_readwrite("minibatch_size", &NnetEnsembleTrainerConfig::minibatch_size)
      .def_readwrite("minibatches_per_phase", &NnetEnsembleTrainerConfig::minibatches_per_phase)
      .def_readwrite("beta", &NnetEnsembleTrainerConfig::beta);
  {
    using PyClass = NnetEnsembleTrainer;

    auto nnet_ensemble_trainer = py::class_<PyClass>(
        m, "NnetEnsembleTrainer");
    nnet_ensemble_trainer
      .def(py::init<const NnetEnsembleTrainerConfig &,
                      std::vector<Nnet*> >(),
        py::arg("config"),
        py::arg("nnet_ensemble"))
      .def("TrainOnExample",
        &PyClass::TrainOnExample,
        py::arg("value"));
  }
}

void pybind_nnet2_train_nnet(py::module& m) {

  py::class_<NnetSimpleTrainerConfig>(m, "NnetSimpleTrainerConfig")
      .def(py::init<>())
      .def_readwrite("minibatch_size", &NnetSimpleTrainerConfig::minibatch_size)
      .def_readwrite("minibatches_per_phase", &NnetSimpleTrainerConfig::minibatches_per_phase);

  m.def("TrainNnetSimple",
        &TrainNnetSimple,
        "Train on all the examples it can read from the reader.  This does training "
          "in a single thread, but it uses a separate thread to read in the examples "
          "and format the input data on the CPU; this saves us time when using GPUs. "
          "Returns the number of examples processed. "
          "Outputs to tot_weight and tot_logprob_per_frame, if non-NULL, the total "
          "weight of the examples (typically equal to the number of examples) and the "
          "total logprob objective function.",
        py::arg("config"),
        py::arg("nnet"),
        py::arg("reader"),
        py::arg("tot_weight") = NULL,
        py::arg("tot_logprob") = NULL);
}

void pybind_nnet2_widen_nnet(py::module& m) {

  py::class_<NnetWidenConfig>(m, "NnetWidenConfig")
      .def(py::init<>())
      .def_readwrite("hidden_layer_dim", &NnetWidenConfig::hidden_layer_dim)
      .def_readwrite("param_stddev_factor", &NnetWidenConfig::param_stddev_factor)
      .def_readwrite("bias_stddev", &NnetWidenConfig::bias_stddev);

  m.def("WidenNnet",
        &WidenNnet,
        " This function widens a neural network by increasing the hidden-layer "
          "dimensions to the target.",
        py::arg("widen_config"),
        py::arg("nnet"));
}



void init_nnet2(py::module &_m) {
     py::module m = _m.def_submodule("nnet2", "nnet2 pybind for Kaldi");
     pybind_nnet2_nnet_component(m);
     pybind_nnet2_nnet_nnet(m);
     pybind_nnet2_am_nnet(m);
     pybind_nnet2_combine_nnet_a(m);
     pybind_nnet2_combine_nnet_fast(m);
     pybind_nnet2_combine_nnet(m);
     pybind_nnet2_decodable_am_net(m);
     pybind_nnet2_get_feature_transform(m);
     pybind_nnet2_mixup_nnet(m);
     pybind_nnet2_nnet_compute_discriminative_parallel(m);
     pybind_nnet2_nnet_compute_discriminative(m);
     pybind_nnet2_nnet_compute_online(m);
     pybind_nnet2_nnet_compute(m);
     pybind_nnet2_nnet_example_functions(m);
     pybind_nnet2_nnet_example(m);
     pybind_nnet2_nnet_fix(m);
     pybind_nnet2_nnet_functions(m);
     pybind_nnet2_nnet_limit_rank(m);
     pybind_nnet2_nnet_precondition_online(m);
     pybind_nnet2_nnet_precondition(m);
     pybind_nnet2_nnet_stats(m);
     pybind_nnet2_nnet_update_parallel(m);
     pybind_nnet2_nnet_update(m);
     pybind_nnet2_online_nnet2_decodable(m);
     pybind_nnet2_rescale_nnet(m);
     pybind_nnet2_shrink_nnet(m);
     pybind_nnet2_train_nnet_ensemble(m);
     pybind_nnet2_train_nnet(m);
     pybind_nnet2_widen_nnet(m);
}
