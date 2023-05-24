
#include "nnet/pybind_nnet.h"

#include "base/kaldi-types.h"
#include "nnet/nnet-activation.h"
#include "nnet/nnet-affine-transform.h"
#include "nnet/nnet-average-pooling-component.h"
#include "nnet/nnet-blstm-projected.h"
#include "nnet/nnet-component.h"
#include "nnet/nnet-convolutional-component.h"
#include "nnet/nnet-frame-pooling-component.h"
#include "nnet/nnet-kl-hmm.h"
#include "nnet/nnet-linear-transform.h"
#include "nnet/nnet-loss.h"
#include "nnet/nnet-lstm-projected.h"
#include "nnet/nnet-matrix-buffer.h"
#include "nnet/nnet-max-pooling-component.h"
//#include "nnet/nnet-multibasis-component.h"
#include "nnet/nnet-nnet.h"
#include "nnet/nnet-parallel-component.h"
#include "nnet/nnet-parametric-relu.h"
#include "nnet/nnet-pdf-prior.h"
#include "nnet/nnet-randomizer.h"
#include "nnet/nnet-rbm.h"
#include "nnet/nnet-recurrent.h"
#include "nnet/nnet-sentence-averaging-component.h"
#include "nnet/nnet-trnopts.h"
#include "nnet/nnet-utils.h"
#include "nnet/nnet-various.h"

using namespace kaldi;
using namespace kaldi::nnet1;


class PyComponent : public Component {
public:
    //Inherit the constructors
    using Component::Component;

    //Trampoline (need one for each virtual function)
    Component* Copy() const override {
        PYBIND11_OVERRIDE_PURE(
            Component*, //Return type (ret_type)
            Component,      //Parent class (cname)
            Copy          //Name of function in C++ (must match Python name) (fn)
                  //Argument(s) (...)
        );
    }

    ComponentType GetType() const override {
        PYBIND11_OVERRIDE_PURE(
            ComponentType, //Return type (ret_type)
            Component,      //Parent class (cname)
            GetType          //Name of function in C++ (must match Python name) (fn)
                  //Argument(s) (...)
        );
    }

    std::string Info() const override {
        PYBIND11_OVERRIDE_PURE(
            std::string, //Return type (ret_type)
            Component,      //Parent class (cname)
            Info          //Name of function in C++ (must match Python name) (fn)
                  //Argument(s) (...)
        );
    }

    std::string InfoGradient() const override {
        PYBIND11_OVERRIDE_PURE(
            std::string, //Return type (ret_type)
            Component,      //Parent class (cname)
            InfoGradient          //Name of function in C++ (must match Python name) (fn)
                  //Argument(s) (...)
        );
    }
protected:
    void PropagateFnc(const CuMatrixBase<BaseFloat> &in,
                            CuMatrixBase<BaseFloat> *out) override {
        PYBIND11_OVERRIDE_PURE(
            void, //Return type (ret_type)
            Component,      //Parent class (cname)
            PropagateFnc,          //Name of function in C++ (must match Python name) (fn)
            in, out      //Argument(s) (...)
        );
    }

    void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in,
                                const CuMatrixBase<BaseFloat> &out,
                                const CuMatrixBase<BaseFloat> &out_diff,
                                CuMatrixBase<BaseFloat> *in_diff) override {
        PYBIND11_OVERRIDE_PURE(
            void, //Return type (ret_type)
            Component,      //Parent class (cname)
            BackpropagateFnc,          //Name of function in C++ (must match Python name) (fn)
            in, out, out_diff, in_diff       //Argument(s) (...)
        );
    }
};
class PyRbmBase : public RbmBase {
public:
    //Inherit the constructors
    using RbmBase::RbmBase;

    //Trampoline (need one for each virtual function)
    Component* Copy() const override {
        PYBIND11_OVERRIDE_PURE(
            Component*, //Return type (ret_type)
            Component,      //Parent class (cname)
            Copy          //Name of function in C++ (must match Python name) (fn)
                  //Argument(s) (...)
        );
    }

    ComponentType GetType() const override {
        PYBIND11_OVERRIDE_PURE(
            ComponentType, //Return type (ret_type)
            Component,      //Parent class (cname)
            GetType          //Name of function in C++ (must match Python name) (fn)
                  //Argument(s) (...)
        );
    }

    std::string Info() const override {
        PYBIND11_OVERRIDE_PURE(
            std::string, //Return type (ret_type)
            Component,      //Parent class (cname)
            Info          //Name of function in C++ (must match Python name) (fn)
                  //Argument(s) (...)
        );
    }

    std::string InfoGradient() const override {
        PYBIND11_OVERRIDE_PURE(
            std::string, //Return type (ret_type)
            Component,      //Parent class (cname)
            InfoGradient          //Name of function in C++ (must match Python name) (fn)
                  //Argument(s) (...)
        );
    }

    void Reconstruct(
    const CuMatrixBase<BaseFloat> &hid_state,
    CuMatrix<BaseFloat> *vis_probs
  ) override {
        PYBIND11_OVERRIDE_PURE(
            void, //Return type (ret_type)
            RbmBase,      //Parent class (cname)
            Reconstruct,          //Name of function in C++ (must match Python name) (fn)
            hid_state, vis_probs      //Argument(s) (...)
        );
    }

    void RbmUpdate(
    const CuMatrixBase<BaseFloat> &pos_vis,
    const CuMatrixBase<BaseFloat> &pos_hid,
    const CuMatrixBase<BaseFloat> &neg_vis,
    const CuMatrixBase<BaseFloat> &neg_hid
  ) override {
        PYBIND11_OVERRIDE_PURE(
            void, //Return type (ret_type)
            RbmBase,      //Parent class (cname)
            RbmUpdate,          //Name of function in C++ (must match Python name) (fn)
            pos_vis, pos_hid, neg_vis, neg_hid      //Argument(s) (...)
        );
    }

    RbmNodeType VisType() const override {
        PYBIND11_OVERRIDE_PURE(
            RbmNodeType, //Return type (ret_type)
            RbmBase,      //Parent class (cname)
            VisType          //Name of function in C++ (must match Python name) (fn)
                  //Argument(s) (...)
        );
    }

    RbmNodeType HidType() const override {
        PYBIND11_OVERRIDE_PURE(
            RbmNodeType, //Return type (ret_type)
            RbmBase,      //Parent class (cname)
            HidType          //Name of function in C++ (must match Python name) (fn)
                  //Argument(s) (...)
        );
    }

    void WriteAsNnet(std::ostream& os, bool binary) const override {
        PYBIND11_OVERRIDE_PURE(
            void, //Return type (ret_type)
            RbmBase,      //Parent class (cname)
            WriteAsNnet          //Name of function in C++ (must match Python name) (fn)
            os, binary      //Argument(s) (...)
        );
    }
protected:
    void PropagateFnc(const CuMatrixBase<BaseFloat> &in,
                            CuMatrixBase<BaseFloat> *out) override {
        PYBIND11_OVERRIDE_PURE(
            void, //Return type (ret_type)
            Component,      //Parent class (cname)
            PropagateFnc,          //Name of function in C++ (must match Python name) (fn)
            in, out      //Argument(s) (...)
        );
    }

    void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in,
                                const CuMatrixBase<BaseFloat> &out,
                                const CuMatrixBase<BaseFloat> &out_diff,
                                CuMatrixBase<BaseFloat> *in_diff) override {
        PYBIND11_OVERRIDE_PURE(
            void, //Return type (ret_type)
            Component,      //Parent class (cname)
            BackpropagateFnc,          //Name of function in C++ (must match Python name) (fn)
            in, out, out_diff, in_diff       //Argument(s) (...)
        );
    }
};


class PyLossItf : public LossItf {
public:
    //Inherit the constructors
    using LossItf::LossItf;

    //Trampoline (need one for each virtual function)
    void Eval(const VectorBase<BaseFloat> &frame_weights,
            const CuMatrixBase<BaseFloat> &net_out,
            const CuMatrixBase<BaseFloat> &target,
            CuMatrix<BaseFloat> *diff) override {
        PYBIND11_OVERRIDE_PURE(
            void, //Return type (ret_type)
            LossItf,      //Parent class (cname)
            Eval,          //Name of function in C++ (must match Python name) (fn)
            frame_weights, net_out, target, diff      //Argument(s) (...)
        );
    }
    void Eval(const VectorBase<BaseFloat> &frame_weights,
            const CuMatrixBase<BaseFloat> &net_out,
            const Posterior &target,
            CuMatrix<BaseFloat> *diff) override {
        PYBIND11_OVERRIDE_PURE(
            void, //Return type (ret_type)
            LossItf,      //Parent class (cname)
            Eval,          //Name of function in C++ (must match Python name) (fn)
            frame_weights, net_out, target, diff      //Argument(s) (...)
        );
    }
    std::string Report() override {
        PYBIND11_OVERRIDE_PURE(
            std::string, //Return type (ret_type)
            LossItf,      //Parent class (cname)
            Report          //Name of function in C++ (must match Python name) (fn)
                  //Argument(s) (...)
        );
    }
    BaseFloat AvgLoss() override {
        PYBIND11_OVERRIDE_PURE(
            BaseFloat, //Return type (ret_type)
            LossItf,      //Parent class (cname)
            AvgLoss          //Name of function in C++ (must match Python name) (fn)
                  //Argument(s) (...)
        );
    }
};


class PyUpdatableComponent : public UpdatableComponent {
public:
    //Inherit the constructors
    using UpdatableComponent::UpdatableComponent;

    //Trampoline (need one for each virtual function)
    Component* Copy() const override {
        PYBIND11_OVERRIDE_PURE(
            Component*, //Return type (ret_type)
            Component,      //Parent class (cname)
            Copy          //Name of function in C++ (must match Python name) (fn)
                  //Argument(s) (...)
        );
    }

    ComponentType GetType() const override {
        PYBIND11_OVERRIDE_PURE(
            ComponentType, //Return type (ret_type)
            Component,      //Parent class (cname)
            GetType          //Name of function in C++ (must match Python name) (fn)
                  //Argument(s) (...)
        );
    }

    std::string Info() const override {
        PYBIND11_OVERRIDE_PURE(
            std::string, //Return type (ret_type)
            Component,      //Parent class (cname)
            Info          //Name of function in C++ (must match Python name) (fn)
                  //Argument(s) (...)
        );
    }

    std::string InfoGradient() const override {
        PYBIND11_OVERRIDE_PURE(
            std::string, //Return type (ret_type)
            Component,      //Parent class (cname)
            InfoGradient          //Name of function in C++ (must match Python name) (fn)
                  //Argument(s) (...)
        );
    }
    int32 NumParams() const override {
        PYBIND11_OVERRIDE_PURE(
            int32, //Return type (ret_type)
            UpdatableComponent,      //Parent class (cname)
            NumParams          //Name of function in C++ (must match Python name) (fn)
                  //Argument(s) (...)
        );
    }
    void GetGradient(VectorBase<BaseFloat> *gradient) const override {
        PYBIND11_OVERRIDE_PURE(
            void, //Return type (ret_type)
            UpdatableComponent,      //Parent class (cname)
            GetGradient,          //Name of function in C++ (must match Python name) (fn)
            gradient      //Argument(s) (...)
        );
    }
    void GetParams(VectorBase<BaseFloat> *params) const override {
        PYBIND11_OVERRIDE_PURE(
            void, //Return type (ret_type)
            UpdatableComponent,      //Parent class (cname)
            GetParams,          //Name of function in C++ (must match Python name) (fn)
            params      //Argument(s) (...)
        );
    }
    void SetParams(const VectorBase<BaseFloat> &params) override {
        PYBIND11_OVERRIDE_PURE(
            void, //Return type (ret_type)
            UpdatableComponent,      //Parent class (cname)
            SetParams,          //Name of function in C++ (must match Python name) (fn)
            params      //Argument(s) (...)
        );
    }
    void Update(const CuMatrixBase<BaseFloat> &input,
                      const CuMatrixBase<BaseFloat> &diff) override {
        PYBIND11_OVERRIDE_PURE(
            void, //Return type (ret_type)
            UpdatableComponent,      //Parent class (cname)
            Update,          //Name of function in C++ (must match Python name) (fn)
            input, diff      //Argument(s) (...)
        );
    }
    void InitData(std::istream &is) override {
        PYBIND11_OVERRIDE_PURE(
            void, //Return type (ret_type)
            UpdatableComponent,      //Parent class (cname)
            InitData,          //Name of function in C++ (must match Python name) (fn)
            is      //Argument(s) (...)
        );
    }
protected:
    void PropagateFnc(const CuMatrixBase<BaseFloat> &in,
                            CuMatrixBase<BaseFloat> *out) override {
        PYBIND11_OVERRIDE_PURE(
            void, //Return type (ret_type)
            Component,      //Parent class (cname)
            PropagateFnc,          //Name of function in C++ (must match Python name) (fn)
            in, out      //Argument(s) (...)
        );
    }

    void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in,
                                const CuMatrixBase<BaseFloat> &out,
                                const CuMatrixBase<BaseFloat> &out_diff,
                                CuMatrixBase<BaseFloat> *in_diff) override {
        PYBIND11_OVERRIDE_PURE(
            void, //Return type (ret_type)
            Component,      //Parent class (cname)
            BackpropagateFnc,          //Name of function in C++ (must match Python name) (fn)
            in, out, out_diff, in_diff       //Argument(s) (...)
        );
    }
};


class PyMultistreamComponent : public MultistreamComponent {
public:
    //Inherit the constructors
    using MultistreamComponent::MultistreamComponent;

    //Trampoline (need one for each virtual function)
    Component* Copy() const override {
        PYBIND11_OVERRIDE_PURE(
            Component*, //Return type (ret_type)
            Component,      //Parent class (cname)
            Copy          //Name of function in C++ (must match Python name) (fn)
                  //Argument(s) (...)
        );
    }

    ComponentType GetType() const override {
        PYBIND11_OVERRIDE_PURE(
            ComponentType, //Return type (ret_type)
            Component,      //Parent class (cname)
            GetType          //Name of function in C++ (must match Python name) (fn)
                  //Argument(s) (...)
        );
    }

    std::string Info() const override {
        PYBIND11_OVERRIDE_PURE(
            std::string, //Return type (ret_type)
            Component,      //Parent class (cname)
            Info          //Name of function in C++ (must match Python name) (fn)
                  //Argument(s) (...)
        );
    }

    std::string InfoGradient() const override {
        PYBIND11_OVERRIDE_PURE(
            std::string, //Return type (ret_type)
            Component,      //Parent class (cname)
            InfoGradient          //Name of function in C++ (must match Python name) (fn)
                  //Argument(s) (...)
        );
    }
    int32 NumParams() const override {
        PYBIND11_OVERRIDE_PURE(
            int32, //Return type (ret_type)
            UpdatableComponent,      //Parent class (cname)
            NumParams          //Name of function in C++ (must match Python name) (fn)
                  //Argument(s) (...)
        );
    }
    void GetGradient(VectorBase<BaseFloat> *gradient) const override {
        PYBIND11_OVERRIDE_PURE(
            void, //Return type (ret_type)
            UpdatableComponent,      //Parent class (cname)
            GetGradient,          //Name of function in C++ (must match Python name) (fn)
            gradient      //Argument(s) (...)
        );
    }
    void GetParams(VectorBase<BaseFloat> *params) const override {
        PYBIND11_OVERRIDE_PURE(
            void, //Return type (ret_type)
            UpdatableComponent,      //Parent class (cname)
            GetParams,          //Name of function in C++ (must match Python name) (fn)
            params      //Argument(s) (...)
        );
    }
    void SetParams(const VectorBase<BaseFloat> &params) override {
        PYBIND11_OVERRIDE_PURE(
            void, //Return type (ret_type)
            UpdatableComponent,      //Parent class (cname)
            SetParams,          //Name of function in C++ (must match Python name) (fn)
            params      //Argument(s) (...)
        );
    }
    void Update(const CuMatrixBase<BaseFloat> &input,
                      const CuMatrixBase<BaseFloat> &diff) override {
        PYBIND11_OVERRIDE_PURE(
            void, //Return type (ret_type)
            UpdatableComponent,      //Parent class (cname)
            Update,          //Name of function in C++ (must match Python name) (fn)
            input, diff      //Argument(s) (...)
        );
    }
    void InitData(std::istream &is) override {
        PYBIND11_OVERRIDE_PURE(
            void, //Return type (ret_type)
            UpdatableComponent,      //Parent class (cname)
            InitData,          //Name of function in C++ (must match Python name) (fn)
            is      //Argument(s) (...)
        );
    }
protected:
    void PropagateFnc(const CuMatrixBase<BaseFloat> &in,
                            CuMatrixBase<BaseFloat> *out) override {
        PYBIND11_OVERRIDE_PURE(
            void, //Return type (ret_type)
            Component,      //Parent class (cname)
            PropagateFnc,          //Name of function in C++ (must match Python name) (fn)
            in, out      //Argument(s) (...)
        );
    }

    void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in,
                                const CuMatrixBase<BaseFloat> &out,
                                const CuMatrixBase<BaseFloat> &out_diff,
                                CuMatrixBase<BaseFloat> *in_diff) override {
        PYBIND11_OVERRIDE_PURE(
            void, //Return type (ret_type)
            Component,      //Parent class (cname)
            BackpropagateFnc,          //Name of function in C++ (must match Python name) (fn)
            in, out, out_diff, in_diff       //Argument(s) (...)
        );
    }
};


void pybind_nnet_nnet_activation(py::module& m) {

  {
    using PyClass = Softmax;

    auto softmax = py::class_<Softmax, Component>(
        m, "Softmax");
    softmax.def(py::init<int32 , int32 >(),
        py::arg("dim_in"),
        py::arg("dim_out"))
      .def("Copy",
        &PyClass::Copy)
      .def("GetType",
        &PyClass::GetType)
      .def("PropagateFnc",
        &PyClass::PropagateFnc,
        py::arg("in"),
        py::arg("out"))
      .def("BackpropagateFnc",
        &PyClass::BackpropagateFnc,
        py::arg("in"),
        py::arg("out"),
        py::arg("out_diff"),
        py::arg("in_diff"));

  }
  {
    using PyClass = HiddenSoftmax;

    auto hidden_softmax = py::class_<HiddenSoftmax, Component>(
        m, "HiddenSoftmax");
    hidden_softmax.def(py::init<int32 , int32 >(),
        py::arg("dim_in"),
        py::arg("dim_out"))
      .def("Copy",
        &PyClass::Copy)
      .def("GetType",
        &PyClass::GetType)
      .def("PropagateFnc",
        &PyClass::PropagateFnc,
        py::arg("in"),
        py::arg("out"))
      .def("BackpropagateFnc",
        &PyClass::BackpropagateFnc,
        py::arg("in"),
        py::arg("out"),
        py::arg("out_diff"),
        py::arg("in_diff"));

  }
  {
    using PyClass = BlockSoftmax;

    auto block_softmax = py::class_<BlockSoftmax, Component>(
        m, "BlockSoftmax");
    block_softmax.def(py::init<int32 , int32 >(),
        py::arg("dim_in"),
        py::arg("dim_out"))
      .def("Copy",
        &PyClass::Copy)
      .def("GetType",
        &PyClass::GetType)
      .def("InitData",
        &PyClass::InitData,
        py::arg("is"))
      .def("ReadData",
        &PyClass::ReadData,
        py::arg("is"),
        py::arg("binary"))
      .def("WriteData",
        &PyClass::WriteData,
        py::arg("os"),
        py::arg("binary"))
      .def("PropagateFnc",
        &PyClass::PropagateFnc,
        py::arg("in"),
        py::arg("out"))
      .def("BackpropagateFnc",
        &PyClass::BackpropagateFnc,
        py::arg("in"),
        py::arg("out"),
        py::arg("out_diff"),
        py::arg("in_diff"))
      .def("Info",
        &PyClass::Info)
      .def_readwrite("block_dims", &PyClass::block_dims)
      .def_readwrite("block_offset", &PyClass::block_offset);

  }
  {
    using PyClass = Sigmoid;

    auto sigmoid = py::class_<Sigmoid, Component>(
        m, "Sigmoid");
    sigmoid.def(py::init<int32 , int32 >(),
        py::arg("dim_in"),
        py::arg("dim_out"))
      .def("Copy",
        &PyClass::Copy)
      .def("GetType",
        &PyClass::GetType)
      .def("PropagateFnc",
        &PyClass::PropagateFnc,
        py::arg("in"),
        py::arg("out"))
      .def("BackpropagateFnc",
        &PyClass::BackpropagateFnc,
        py::arg("in"),
        py::arg("out"),
        py::arg("out_diff"),
        py::arg("in_diff"));

  }
  {
    using PyClass = Tanh;

    auto tanh = py::class_<Tanh, Component>(
        m, "Tanh");
    tanh.def(py::init<int32 , int32 >(),
        py::arg("dim_in"),
        py::arg("dim_out"))
      .def("Copy",
        &PyClass::Copy)
      .def("GetType",
        &PyClass::GetType)
      .def("PropagateFnc",
        &PyClass::PropagateFnc,
        py::arg("in"),
        py::arg("out"))
      .def("BackpropagateFnc",
        &PyClass::BackpropagateFnc,
        py::arg("in"),
        py::arg("out"),
        py::arg("out_diff"),
        py::arg("in_diff"));

  }
  {
    using PyClass = Dropout;

    auto dropout = py::class_<Dropout, Component>(
        m, "Dropout");
    dropout.def(py::init<int32 , int32 >(),
        py::arg("dim_in"),
        py::arg("dim_out"))
      .def("Copy",
        &PyClass::Copy)
      .def("GetType",
        &PyClass::GetType)
      .def("InitData",
        &PyClass::InitData,
        py::arg("is"))
      .def("ReadData",
        &PyClass::ReadData,
        py::arg("is"),
        py::arg("binary"))
      .def("WriteData",
        &PyClass::WriteData,
        py::arg("os"),
        py::arg("binary"))
      .def("PropagateFnc",
        &PyClass::PropagateFnc,
        py::arg("in"),
        py::arg("out"))
      .def("BackpropagateFnc",
        &PyClass::BackpropagateFnc,
        py::arg("in"),
        py::arg("out"),
        py::arg("out_diff"),
        py::arg("in_diff"))
      .def("GetDropoutRate",
        &PyClass::GetDropoutRate)
      .def("SetDropoutRate",
        &PyClass::SetDropoutRate,
        py::arg("dr"));

  }
}

void pybind_nnet_nnet_affine_transform(py::module& m) {

  {
    using PyClass = AffineTransform;

    auto affine_transform = py::class_<AffineTransform, UpdatableComponent>(
        m, "AffineTransform");
    affine_transform.def(py::init<int32 , int32 >(),
        py::arg("dim_in"),
        py::arg("dim_out"))
      .def("Copy",
        &PyClass::Copy)
      .def("GetType",
        &PyClass::GetType)
      .def("InitData",
        &PyClass::InitData,
        py::arg("is"))
      .def("ReadData",
        &PyClass::ReadData,
        py::arg("is"),
        py::arg("binary"))
      .def("WriteData",
        &PyClass::WriteData,
        py::arg("os"),
        py::arg("binary"))
      .def("NumParams",
        &PyClass::NumParams,
        "Number of trainable parameters")
      .def("GetGradient",
        &PyClass::GetGradient,
        "Get gradient reshaped as a vector",
        py::arg("gradient"))
      .def("GetParams",
        &PyClass::GetParams,
        "Get the trainable parameters reshaped as a vector",
        py::arg("params"))
      .def("SetParams",
        &PyClass::SetParams,
        "Set the trainable parameters from, reshaped as a vector",
        py::arg("params"))
      .def("Info",
        &PyClass::Info,
        "Print some additional info (after <ComponentName> and the dims)")
      .def("InfoGradient",
        &PyClass::InfoGradient,
        "Print some additional info about gradient (after <...> and dims)")
      .def("Update",
        &PyClass::Update,
        "Compute gradient and update parameters",
        py::arg("input"),
        py::arg("diff"))
      .def("GetBias",
        &PyClass::GetBias)
      .def("SetBias",
        &PyClass::SetBias,
        py::arg("bias"))
      .def("GetLinearity",
        &PyClass::GetLinearity)
      .def("SetLinearity",
        &PyClass::SetLinearity,
        py::arg("linearity"));
  }
}

void pybind_nnet_nnet_average_pooling_component(py::module& m) {

  {
    using PyClass = AveragePoolingComponent;

    auto average_pooling_component = py::class_<AveragePoolingComponent, Component>(
        m, "AveragePoolingComponent");
    average_pooling_component.def(py::init<int32 , int32 >(),
        py::arg("dim_in"),
        py::arg("dim_out"))
      .def("Copy",
        &PyClass::Copy)
      .def("GetType",
        &PyClass::GetType)
      .def("InitData",
        &PyClass::InitData,
        py::arg("is"))
      .def("ReadData",
        &PyClass::ReadData,
        py::arg("is"),
        py::arg("binary"))
      .def("WriteData",
        &PyClass::WriteData,
        py::arg("os"),
        py::arg("binary"))
      .def("PropagateFnc",
        &PyClass::PropagateFnc,
        py::arg("in"),
        py::arg("out"))
      .def("BackpropagateFnc",
        &PyClass::BackpropagateFnc,
        py::arg("in"),
        py::arg("out"),
        py::arg("out_diff"),
        py::arg("in_diff"));

  }
}

void pybind_nnet_nnet_blstm_projected(py::module& m) {

  {
    using PyClass = BlstmProjected;

    auto blstm_projected = py::class_<BlstmProjected, MultistreamComponent>(
        m, "BlstmProjected");
    blstm_projected.def(py::init<int32 , int32 >(),
        py::arg("input_dim"),
        py::arg("output_dim"))
      .def("Copy",
        &PyClass::Copy)
      .def("GetType",
        &PyClass::GetType)
      .def("InitData",
        &PyClass::InitData,
        py::arg("is"))
      .def("ReadData",
        &PyClass::ReadData,
        py::arg("is"),
        py::arg("binary"))
      .def("WriteData",
        &PyClass::WriteData,
        py::arg("os"),
        py::arg("binary"))
      .def("NumParams",
        &PyClass::NumParams,
        "Number of trainable parameters")
      .def("GetGradient",
        &PyClass::GetGradient,
        "Get gradient reshaped as a vector",
        py::arg("gradient"))
      .def("GetParams",
        &PyClass::GetParams,
        "Get the trainable parameters reshaped as a vector",
        py::arg("params"))
      .def("SetParams",
        &PyClass::SetParams,
        "Set the trainable parameters from, reshaped as a vector",
        py::arg("params"))
      .def("Info",
        &PyClass::Info,
        "Print some additional info (after <ComponentName> and the dims)")
      .def("InfoGradient",
        &PyClass::InfoGradient,
        "Print some additional info about gradient (after <...> and dims)")
      .def("ResetStreams",
        &PyClass::ResetStreams,
        py::arg("stream_reset_flag"))
      .def("PropagateFnc",
        &PyClass::PropagateFnc,
        py::arg("in"),
        py::arg("out"))
      .def("BackpropagateFnc",
        &PyClass::BackpropagateFnc,
        py::arg("in"),
        py::arg("out"),
        py::arg("out_diff"),
        py::arg("in_diff"))
      .def("Update",
        &PyClass::Update,
        "Compute gradient and update parameters",
        py::arg("input"),
        py::arg("diff"));
  }
}

void pybind_nnet_nnet_component(py::module& m) {

  {
    using PyClass = Component;

    auto component = py::class_<PyClass, PyComponent>(
        m, "Component");
    component.def(py::init<int32 , int32 >(),
        py::arg("dim_in"),
        py::arg("dim_out"))
      .def_static("TypeToMarker",
        &PyClass::TypeToMarker,
        "Converts component type to marker",
        py::arg("t"))
      .def_static("MarkerToType",
        &PyClass::MarkerToType,
        "Converts marker to component type (case insensitive)",
        py::arg("s"))
      .def("Copy",
        &PyClass::Copy)
      .def("GetType",
        &PyClass::GetType,
        "Get Type Identification of the component")
      .def("IsUpdatable",
        &PyClass::IsUpdatable,
        "Check if component has 'Updatable' interface (trainable components)")
      .def("IsMultistream",
        &PyClass::IsMultistream,
        "Check if component has 'Recurrent' interface (trainable and recurrent)")
      .def("InputDim",
        &PyClass::InputDim,
        "Get the dimension of the input")
      .def("OutputDim",
        &PyClass::OutputDim,
        "Get the dimension of the output")
      .def_static("Init",
        &PyClass::Init,
        py::arg("conf_line"))
      .def_static("Read",
        &PyClass::Read,
        py::arg("is"),
        py::arg("binary"))
      .def("Write",
        &PyClass::Write,
        py::arg("os"),
        py::arg("binary"))
      .def("Info",
        &PyClass::Info,
        "Print some additional info (after <ComponentName> and the dims)")
      .def("InfoGradient",
        &PyClass::InfoGradient,
        "Print some additional info about gradient (after <...> and dims)");

  py::enum_<Component::ComponentType>(component, "ComponentType")
    .value("kUnknown", Component::ComponentType::kUnknown)
    .value("kUpdatableComponent", Component::ComponentType::kUpdatableComponent)
    .value("kAffineTransform", Component::ComponentType::kAffineTransform)
    .value("kLinearTransform", Component::ComponentType::kLinearTransform)
    .value("kConvolutionalComponent", Component::ComponentType::kConvolutionalComponent)
    .value("kLstmProjected", Component::ComponentType::kLstmProjected)
    .value("kBlstmProjected", Component::ComponentType::kBlstmProjected)
    .value("kRecurrentComponent", Component::ComponentType::kRecurrentComponent)
    .value("kActivationFunction", Component::ComponentType::kActivationFunction)
    .value("kSoftmax", Component::ComponentType::kSoftmax)
    .value("kHiddenSoftmax", Component::ComponentType::kHiddenSoftmax)
    .value("kBlockSoftmax", Component::ComponentType::kBlockSoftmax)
    .value("kSigmoid", Component::ComponentType::kSigmoid)
    .value("kTanh", Component::ComponentType::kTanh)
    .value("kParametricRelu", Component::ComponentType::kParametricRelu)
    .value("kDropout", Component::ComponentType::kDropout)
    .value("kLengthNormComponent", Component::ComponentType::kLengthNormComponent)
    .value("kTranform", Component::ComponentType::kTranform)
    .value("kRbm", Component::ComponentType::kRbm)
    .value("kSplice", Component::ComponentType::kSplice)
    .value("kCopy", Component::ComponentType::kCopy)
    .value("kTranspose", Component::ComponentType::kTranspose)
    .value("kBlockLinearity", Component::ComponentType::kBlockLinearity)
    .value("kAddShift", Component::ComponentType::kAddShift)
    .value("kRescale", Component::ComponentType::kRescale)
    .value("kKlHmm", Component::ComponentType::kKlHmm)
    .value("kSentenceAveragingComponent", Component::ComponentType::kSentenceAveragingComponent)
    .value("kSimpleSentenceAveragingComponent", Component::ComponentType::kSimpleSentenceAveragingComponent)
    .value("kAveragePoolingComponent", Component::ComponentType::kAveragePoolingComponent)
    .value("kMaxPoolingComponent", Component::ComponentType::kMaxPoolingComponent)
    .value("kFramePoolingComponent", Component::ComponentType::kFramePoolingComponent)
    .value("kParallelComponent", Component::ComponentType::kParallelComponent)
    .value("kMultiBasisComponent", Component::ComponentType::kMultiBasisComponent)
    .export_values();
    //py::class_<PyClass::key_value>(component, "key_value")
    //.def_readonly("key", &PyClass::key_value::key)
    //.def_readonly("value", &PyClass::key_value::value);
 }
  {
    using PyClass = UpdatableComponent;

    auto updatable_component = py::class_<UpdatableComponent, PyUpdatableComponent>(
        m, "UpdatableComponent");
    updatable_component.def(py::init<int32 , int32 >(),
        py::arg("dim_in"),
        py::arg("dim_out"))
      .def("IsUpdatable",
        &PyClass::IsUpdatable,
        "Check if contains trainable parameters")
      .def("NumParams",
        &PyClass::NumParams,
        "Number of trainable parameters")
      .def("GetGradient",
        &PyClass::GetGradient,
        "Get gradient reshaped as a vector",
        py::arg("gradient"))
      .def("GetParams",
        &PyClass::GetParams,
        "Get the trainable parameters reshaped as a vector",
        py::arg("params"))
      .def("SetParams",
        &PyClass::SetParams,
        "Set the trainable parameters from, reshaped as a vector",
        py::arg("params"))
      .def("Update",
        &PyClass::Update,
        "Compute gradient and update parameters",
        py::arg("input"),
        py::arg("diff"))
      .def("SetTrainOptions",
        &PyClass::SetTrainOptions,
        "Set the training options to the component",
        py::arg("opts"))
      .def("GetTrainOptions",
        &PyClass::GetTrainOptions,
        "Get the training options from the component")
      .def("SetLearnRateCoef",
        &PyClass::SetLearnRateCoef,
        "Set the learn-rate coefficient",
        py::arg("val"))
      .def("SetBiasLearnRateCoef",
        &PyClass::SetBiasLearnRateCoef,
        "Set the learn-rate coefficient for bias",
        py::arg("val"))
      .def("InitData",
        &PyClass::InitData,
        "Initialize the content of the component by the 'line' from the prototype",
        py::arg("is"));
  }
  {
    using PyClass = MultistreamComponent;

    auto multistream_component = py::class_<MultistreamComponent, PyMultistreamComponent>(
        m, "MultistreamComponent");
    multistream_component.def(py::init<int32 , int32 >(),
        py::arg("dim_in"),
        py::arg("dim_out"))
      .def("IsMultistream",
        &PyClass::IsMultistream)
      .def("SetSeqLengths",
        &PyClass::SetSeqLengths,
        py::arg("sequence_lengths"))
      .def("NumStreams",
        &PyClass::NumStreams)
      .def("ResetStreams",
        &PyClass::ResetStreams);
  }
}

void pybind_nnet_nnet_convolutional_component(py::module& m) {

  {
    using PyClass = ConvolutionalComponent;

    auto convolutional_component = py::class_<ConvolutionalComponent, UpdatableComponent>(
        m, "ConvolutionalComponent");
    convolutional_component.def(py::init<int32 , int32 >(),
        py::arg("dim_in"),
        py::arg("dim_out"))
      .def("Copy",
        &PyClass::Copy)
      .def("GetType",
        &PyClass::GetType)
      .def("InitData",
        &PyClass::InitData,
        py::arg("is"))
      .def("ReadData",
        &PyClass::ReadData,
        py::arg("is"),
        py::arg("binary"))
      .def("WriteData",
        &PyClass::WriteData,
        py::arg("os"),
        py::arg("binary"))
      .def("NumParams",
        &PyClass::NumParams,
        "Number of trainable parameters")
      .def("GetGradient",
        &PyClass::GetGradient,
        "Get gradient reshaped as a vector",
        py::arg("gradient"))
      .def("GetParams",
        &PyClass::GetParams,
        "Get the trainable parameters reshaped as a vector",
        py::arg("params"))
      .def("SetParams",
        &PyClass::SetParams,
        "Set the trainable parameters from, reshaped as a vector",
        py::arg("params"))
      .def("Info",
        &PyClass::Info,
        "Print some additional info (after <ComponentName> and the dims)")
      .def("InfoGradient",
        &PyClass::InfoGradient,
        "Print some additional info about gradient (after <...> and dims)")
      .def("ReverseIndexes",
        &PyClass::ReverseIndexes,
        "This function does an operation similar to reversing a map, "
        "except it handles maps that are not one-to-one by outputting "
        "the reversed map as a vector of lists. "
        "@param[in] forward_indexes is a vector of int32, each of whose "
        "            elements is between 0 and input_dim - 1. "
        "@param[in] input_dim. See definitions of forward_indexes and "
        "            backward_indexes. "
        "@param[out] backward_indexes is a vector of dimension input_dim "
        "            of lists, The list at (backward_indexes[i]) is a list "
        "            of all indexes j such that forward_indexes[j] = i.",
        py::arg("forward_indexes"),
        py::arg("backward_indexes"))
      .def("RearrangeIndexes",
        &PyClass::RearrangeIndexes,
        "This function transforms a vector of lists into a list of vectors, "
        "padded with -1. "
        "@param[in] The input vector of lists. Let in.size() be D, and let "
        "            the longest list length (i.e. the max of in[i].size()) be L. "
        "@param[out] The output list of vectors. The length of the list will "
        "            be L, each vector-dimension will be D (i.e. out[i].size() == D), "
        "            and if in[i] == j, then for some k we will have that "
        "            out[k][j] = i. The output vectors are padded with -1 "
        "            where necessary if not all the input lists have the same side.",
        py::arg("in"),
        py::arg("out"))
      .def("Update",
        &PyClass::Update,
        "Compute gradient and update parameters",
        py::arg("input"),
        py::arg("diff"));
  }
}

void pybind_nnet_nnet_frame_pooling_component(py::module& m) {

  {
    using PyClass = FramePoolingComponent;

    auto frame_pooling_component = py::class_<FramePoolingComponent, UpdatableComponent>(
        m, "FramePoolingComponent");
    frame_pooling_component.def(py::init<int32 , int32 >(),
        py::arg("dim_in"),
        py::arg("dim_out"))
      .def("Copy",
        &PyClass::Copy)
      .def("GetType",
        &PyClass::GetType)
      .def("InitData",
        &PyClass::InitData,
        py::arg("is"))
      .def("ReadData",
        &PyClass::ReadData,
        py::arg("is"),
        py::arg("binary"))
      .def("WriteData",
        &PyClass::WriteData,
        py::arg("os"),
        py::arg("binary"))
      .def("NumParams",
        &PyClass::NumParams,
        "Number of trainable parameters")
      .def("GetGradient",
        &PyClass::GetGradient,
        "Get gradient reshaped as a vector",
        py::arg("gradient"))
      .def("GetParams",
        &PyClass::GetParams,
        "Get the trainable parameters reshaped as a vector",
        py::arg("params"))
      .def("SetParams",
        &PyClass::SetParams,
        "Set the trainable parameters from, reshaped as a vector",
        py::arg("params"))
      .def("Info",
        &PyClass::Info,
        "Print some additional info (after <ComponentName> and the dims)")
      .def("InfoGradient",
        &PyClass::InfoGradient,
        "Print some additional info about gradient (after <...> and dims)")
      .def("Update",
        &PyClass::Update,
        "Compute gradient and update parameters",
        py::arg("input"),
        py::arg("diff"));
  }
}

void pybind_nnet_nnet_kl_hmm(py::module& m) {

  {
    using PyClass = KlHmm;

    auto kl_hmm_component = py::class_<KlHmm, Component>(
        m, "KlHmm");
    kl_hmm_component.def(py::init<int32 , int32 >(),
        py::arg("dim_in"),
        py::arg("dim_out"))
      .def("Copy",
        &PyClass::Copy)
      .def("GetType",
        &PyClass::GetType)
      .def("PropagateFnc",
        &PyClass::PropagateFnc,
        py::arg("in"),
        py::arg("out"))
      .def("BackpropagateFnc",
        &PyClass::BackpropagateFnc,
        py::arg("in"),
        py::arg("out"),
        py::arg("out_diff"),
        py::arg("in_diff"))
      .def("WriteData",
        &PyClass::WriteData,
        "Writes the component content",
        py::arg("os"),
        py::arg("binary"))
      .def("SetStats",
        &PyClass::SetStats,
        "Set the statistics matrix",
        py::arg("mat"))
      .def("Accumulate",
        &PyClass::Accumulate,
        "Accumulate the statistics for KL-HMM paramter estimation",
        py::arg("posteriors"),
        py::arg("alignment"));
  }
}

void pybind_nnet_nnet_linear_transform(py::module& m) {

  {
    using PyClass = LinearTransform;

    auto linear_transform = py::class_<LinearTransform, UpdatableComponent>(
        m, "LinearTransform");
    linear_transform.def(py::init<int32 , int32 >(),
        py::arg("dim_in"),
        py::arg("dim_out"))
      .def("Copy",
        &PyClass::Copy)
      .def("GetType",
        &PyClass::GetType)
      .def("InitData",
        &PyClass::InitData,
        py::arg("is"))
      .def("ReadData",
        &PyClass::ReadData,
        py::arg("is"),
        py::arg("binary"))
      .def("WriteData",
        &PyClass::WriteData,
        py::arg("os"),
        py::arg("binary"))
      .def("NumParams",
        &PyClass::NumParams,
        "Number of trainable parameters")
      .def("GetGradient",
        &PyClass::GetGradient,
        "Get gradient reshaped as a vector",
        py::arg("gradient"))
      .def("GetParams",
        &PyClass::GetParams,
        "Get the trainable parameters reshaped as a vector",
        py::arg("params"))
      .def("SetParams",
        &PyClass::SetParams,
        "Set the trainable parameters from, reshaped as a vector",
        py::arg("params"))
      .def("SetLinearity",
        py::overload_cast<const MatrixBase<BaseFloat>&>(&PyClass::SetLinearity),
        py::arg("l"))
      .def("SetLinearity",
        py::overload_cast<const CuMatrixBase<BaseFloat>&>(&PyClass::SetLinearity),
        py::arg("linearity"))
      .def("Info",
        &PyClass::Info,
        "Print some additional info (after <ComponentName> and the dims)")
      .def("InfoGradient",
        &PyClass::InfoGradient,
        "Print some additional info about gradient (after <...> and dims)")
      .def("PropagateFnc",
        &PyClass::PropagateFnc,
        py::arg("in"),
        py::arg("out"))
      .def("BackpropagateFnc",
        &PyClass::BackpropagateFnc,
        py::arg("in"),
        py::arg("out"),
        py::arg("out_diff"),
        py::arg("in_diff"))
      .def("GetLinearity",
        &PyClass::GetLinearity)
      .def("GetLinearityCorr",
        &PyClass::GetLinearityCorr)
      .def("Update",
        &PyClass::Update,
        "Compute gradient and update parameters",
        py::arg("input"),
        py::arg("diff"))
      ;
  }
}

void pybind_nnet_nnet_loss(py::module& m) {

  {
    using PyClass = LossOptions;

    auto loss_options = py::class_<PyClass>(
        m, "LossOptions");
    loss_options.def(py::init<>())
      .def_readwrite("loss_report_frames", &PyClass::loss_report_frames);
  }
  {
    using PyClass = LossItf;

    auto loss_itf = py::class_<PyClass, PyLossItf>(
        m, "LossItf");
    loss_itf.def(py::init<LossOptions& >(),
        py::arg("opts"))
      .def("Eval",
        py::overload_cast<const VectorBase<BaseFloat> &,
            const CuMatrixBase<BaseFloat> &,
            const CuMatrixBase<BaseFloat> &,
            CuMatrix<BaseFloat> *>(&PyClass::Eval),
        "Evaluate cross entropy using target-matrix (supports soft labels)",
        py::arg("frame_weights"),
        py::arg("net_out"),
        py::arg("target"),
        py::arg("diff"))
      .def("Eval",
        py::overload_cast<const VectorBase<BaseFloat> &,
            const CuMatrixBase<BaseFloat> &,
            const Posterior &,
            CuMatrix<BaseFloat> *>(&PyClass::Eval),
        "Evaluate cross entropy using target-matrix (supports soft labels)",
        py::arg("frame_weights"),
        py::arg("net_out"),
        py::arg("target"),
        py::arg("diff"))
      .def("Report",
        &PyClass::Report,
        "Generate string with error report")
      .def("AvgLoss",
        &PyClass::AvgLoss,
        "Get loss value (frame average)");
  }
  {
    using PyClass = Xent;

    auto xent = py::class_<Xent, LossItf>(
        m, "Xent");
    xent.def(py::init<LossOptions& >(),
        py::arg("opts"))
      .def("Eval",
        py::overload_cast<const VectorBase<BaseFloat> &,
            const CuMatrixBase<BaseFloat> &,
            const CuMatrixBase<BaseFloat> &,
            CuMatrix<BaseFloat> *>(&PyClass::Eval),
        "Evaluate cross entropy using target-matrix (supports soft labels)",
        py::arg("frame_weights"),
        py::arg("net_out"),
        py::arg("target"),
        py::arg("diff"))
      .def("Eval",
        py::overload_cast<const VectorBase<BaseFloat> &,
            const CuMatrixBase<BaseFloat> &,
            const Posterior &,
            CuMatrix<BaseFloat> *>(&PyClass::Eval),
        "Evaluate cross entropy using target-matrix (supports soft labels)",
        py::arg("frame_weights"),
        py::arg("net_out"),
        py::arg("target"),
        py::arg("diff"))
      .def("Report",
        &PyClass::Report,
        "Generate string with error report")
      .def("ReportPerClass",
        &PyClass::ReportPerClass,
        "Generate string with per-class error report")
      .def("AvgLoss",
        &PyClass::AvgLoss,
        "Get loss value (frame average)");
  }
  {
    using PyClass = Mse;

    auto mse = py::class_<Mse, LossItf>(
        m, "Mse");
    mse.def(py::init<LossOptions& >(),
        py::arg("opts"))
      .def("Eval",
        py::overload_cast<const VectorBase<BaseFloat> &,
            const CuMatrixBase<BaseFloat> &,
            const CuMatrixBase<BaseFloat> &,
            CuMatrix<BaseFloat> *>(&PyClass::Eval),
        "Evaluate cross entropy using target-matrix (supports soft labels)",
        py::arg("frame_weights"),
        py::arg("net_out"),
        py::arg("target"),
        py::arg("diff"))
      .def("Eval",
        py::overload_cast<const VectorBase<BaseFloat> &,
            const CuMatrixBase<BaseFloat> &,
            const Posterior &,
            CuMatrix<BaseFloat> *>(&PyClass::Eval),
        "Evaluate cross entropy using target-matrix (supports soft labels)",
        py::arg("frame_weights"),
        py::arg("net_out"),
        py::arg("target"),
        py::arg("diff"))
      .def("Report",
        &PyClass::Report,
        "Generate string with error report")
      .def("AvgLoss",
        &PyClass::AvgLoss,
        "Get loss value (frame average)");
  }
  {
    using PyClass = MultiTaskLoss;

    auto multi_task_loss = py::class_<MultiTaskLoss, LossItf>(
        m, "MultiTaskLoss");
    multi_task_loss.def(py::init<LossOptions& >(),
        py::arg("opts"))
      .def("Eval",
        py::overload_cast<const VectorBase<BaseFloat> &,
            const CuMatrixBase<BaseFloat> &,
            const CuMatrixBase<BaseFloat> &,
            CuMatrix<BaseFloat> *>(&PyClass::Eval),
        "Evaluate cross entropy using target-matrix (supports soft labels)",
        py::arg("frame_weights"),
        py::arg("net_out"),
        py::arg("target"),
        py::arg("diff"))
      .def("Eval",
        py::overload_cast<const VectorBase<BaseFloat> &,
            const CuMatrixBase<BaseFloat> &,
            const Posterior &,
            CuMatrix<BaseFloat> *>(&PyClass::Eval),
        "Evaluate cross entropy using target-matrix (supports soft labels)",
        py::arg("frame_weights"),
        py::arg("net_out"),
        py::arg("target"),
        py::arg("diff"))
      .def("Report",
        &PyClass::Report,
        "Generate string with error report")
      .def("AvgLoss",
        &PyClass::AvgLoss,
        "Get loss value (frame average)")
      .def("InitFromString",
        &PyClass::InitFromString,
        "Initialize from string, the format for string 's' is : "
        "'multitask,<type1>,<dim1>,<weight1>,...,<typeN>,<dimN>,<weightN>' "
        "\n"
        "Practically it can look like this : "
        "'multitask,xent,2456,1.0,mse,440,0.001'",
        py::arg("s"));
  }
}

void pybind_nnet_nnet_lstm_projected(py::module& m) {

  {
    using PyClass = LstmProjected;

    auto lstm_projected = py::class_<LstmProjected, MultistreamComponent>(
        m, "LstmProjected");
    lstm_projected.def(py::init<int32 , int32 >(),
        py::arg("input_dim"),
        py::arg("output_dim"))
      .def("Copy",
        &PyClass::Copy)
      .def("GetType",
        &PyClass::GetType)
      .def("InitData",
        &PyClass::InitData,
        py::arg("is"))
      .def("ReadData",
        &PyClass::ReadData,
        py::arg("is"),
        py::arg("binary"))
      .def("WriteData",
        &PyClass::WriteData,
        py::arg("os"),
        py::arg("binary"))
      .def("NumParams",
        &PyClass::NumParams,
        "Number of trainable parameters")
      .def("GetGradient",
        &PyClass::GetGradient,
        "Get gradient reshaped as a vector",
        py::arg("gradient"))
      .def("GetParams",
        &PyClass::GetParams,
        "Get the trainable parameters reshaped as a vector",
        py::arg("params"))
      .def("SetParams",
        &PyClass::SetParams,
        "Set the trainable parameters from, reshaped as a vector",
        py::arg("params"))
      .def("Info",
        &PyClass::Info,
        "Print some additional info (after <ComponentName> and the dims)")
      .def("InfoGradient",
        &PyClass::InfoGradient,
        "Print some additional info about gradient (after <...> and dims)")
      .def("ResetStreams",
        &PyClass::ResetStreams,
        py::arg("stream_reset_flag"))
      .def("PropagateFnc",
        &PyClass::PropagateFnc,
        py::arg("in"),
        py::arg("out"))
      .def("BackpropagateFnc",
        &PyClass::BackpropagateFnc,
        py::arg("in"),
        py::arg("out"),
        py::arg("out_diff"),
        py::arg("in_diff"))
      .def("Update",
        &PyClass::Update,
        "Compute gradient and update parameters",
        py::arg("input"),
        py::arg("diff"));
  }
}

void pybind_nnet_nnet_matrix_buffer(py::module& m) {

  {
    using PyClass = MatrixBufferOptions;

    auto matrix_buffer_options = py::class_<PyClass>(
        m, "MatrixBufferOptions");
    matrix_buffer_options.def(py::init<>())
      .def_readwrite("matrix_buffer_size", &PyClass::matrix_buffer_size);
  }
  {
    using PyClass = MatrixBuffer;

    auto matrix_buffer = py::class_<PyClass>(
        m, "MatrixBuffer");
    matrix_buffer.def(py::init<>())
      .def("Init",
        &PyClass::Init,
        py::arg("reader"),
        py::arg("opts") = MatrixBufferOptions())
      .def("Done",
        &PyClass::Done)
      .def("Next",
        &PyClass::Next)
      .def("ResetLength",
        &PyClass::ResetLength)
      .def("Key",
        &PyClass::Key)
      .def("Value",
        &PyClass::Value)
      .def("SizeInBytes",
        &PyClass::SizeInBytes)
      .def("SizeInMegaBytes",
        &PyClass::SizeInMegaBytes)
      .def("NumPairs",
        &PyClass::NumPairs);
  }
}

void pybind_nnet_nnet_max_pooling_component(py::module& m) {

  {
    using PyClass = MaxPoolingComponent;

    auto max_pooling_component = py::class_<MaxPoolingComponent, Component>(
        m, "MaxPoolingComponent");
    max_pooling_component.def(py::init<int32 , int32 >(),
        py::arg("input_dim"),
        py::arg("output_dim"))
      .def("Copy",
        &PyClass::Copy)
      .def("GetType",
        &PyClass::GetType)
      .def("InitData",
        &PyClass::InitData,
        py::arg("is"))
      .def("ReadData",
        &PyClass::ReadData,
        py::arg("is"),
        py::arg("binary"))
      .def("WriteData",
        &PyClass::WriteData,
        py::arg("os"),
        py::arg("binary"))
      .def("PropagateFnc",
        &PyClass::PropagateFnc,
        py::arg("in"),
        py::arg("out"))
      .def("BackpropagateFnc",
        &PyClass::BackpropagateFnc,
        py::arg("in"),
        py::arg("out"),
        py::arg("out_diff"),
        py::arg("in_diff"));
  }
}

void pybind_nnet_nnet_multibasis_component(py::module& m) {

  {
      /*
    using PyClass = MultiBasisComponent;
    auto multi_basis_component = py::class_<MultiBasisComponent, UpdatableComponent>(
        m, "MultiBasisComponent");
    multi_basis_component.def(py::init<int32 , int32 >(),
        py::arg("input_dim"),
        py::arg("output_dim"))
      .def("Copy",
        &PyClass::Copy)
      .def("GetType",
        &PyClass::GetType)
      .def("InitData",
        &PyClass::InitData,
        py::arg("is"))
      .def("ReadData",
        &PyClass::ReadData,
        py::arg("is"),
        py::arg("binary"))
      .def("WriteData",
        &PyClass::WriteData,
        py::arg("os"),
        py::arg("binary"))
      .def("NumParams",
        &PyClass::NumParams)
      .def("GetParams",
        &PyClass::GetParams,
        py::arg("params"))
      .def("SetParams",
        &PyClass::SetParams,
        py::arg("params"))
      .def("Info",
        &PyClass::Info)
      .def("InfoGradient",
        &PyClass::InfoGradient)
      .def("InfoPropagate",
        &PyClass::InfoPropagate)
      .def("InfoBackPropagate",
        &PyClass::InfoBackPropagate)
      .def("PropagateFnc",
        &PyClass::PropagateFnc,
        py::arg("in"),
        py::arg("out"))
      .def("BackpropagateFnc",
        &PyClass::BackpropagateFnc,
        py::arg("in"),
        py::arg("out"),
        py::arg("out_diff"),
        py::arg("in_diff"))
      .def("SetTrainOptions",
        &PyClass::SetTrainOptions,
        py::arg("opts"))
      .def("SetLearnRateCoef",
        &PyClass::SetLearnRateCoef,
        py::arg("val"))
      .def("SetBiasLearnRateCoef",
        &PyClass::SetBiasLearnRateCoef,
        py::arg("val"));*/
  }
}

void pybind_nnet_nnet_nnet(py::module& m) {

  {
    using PyClass = Nnet;

    auto nnet = py::class_<PyClass>(
        m, "Nnet");
    nnet.def(py::init<>())
      .def(py::init<const Nnet&>(),
        py::arg("other"))
      .def("Propagate",
        &PyClass::Propagate,
        "Perform forward pass through the network",
        py::arg("in"),
        py::arg("out"))
      .def("Backpropagate",
        &PyClass::Backpropagate,
        "Perform backward pass through the network",
        py::arg("out_diff"),
        py::arg("in_diff"))
      .def("Feedforward",
        &PyClass::Feedforward,
        "Perform forward pass through the network (with 2 swapping buffers)",
        py::arg("in"),
        py::arg("out"))
      .def("InputDim",
        &PyClass::InputDim,
        "Dimensionality on network input (input feature dim.)")
      .def("OutputDim",
        &PyClass::OutputDim,
        "Dimensionality of network outputs (posteriors | bn-features | etc.)")
      .def("NumComponents",
        &PyClass::NumComponents,
        "Returns the number of 'Components' which form the NN. "
        "Typically a NN layer is composed of 2 components: "
        "the <AffineTransform> with trainable parameters "
        "and a non-linearity like <Sigmoid> or <Softmax>. "
        "Usually there are 2x more Components than the NN layers.")
      .def("ReplaceComponent",
        &PyClass::ReplaceComponent,
        py::arg("c"),
        py::arg("comp"))
      //.def("SwapComponent",
      //  static_cast< void (PyClass::*)(int32, Component**)>(&PyClass::SwapComponent),
      //  py::arg("c"),
      //  py::arg("comp"))
      .def("AppendComponent",
        &PyClass::AppendComponent,
        py::arg("comp"))
      .def("AppendComponentPointer",
        &PyClass::AppendComponentPointer,
        py::arg("dynamically_allocated_comp"))
      .def("AppendNnet",
        &PyClass::AppendNnet,
        py::arg("nnet_to_append"))
      .def("RemoveComponent",
        &PyClass::RemoveComponent,
        py::arg("c"))
      .def("RemoveLastComponent",
        &PyClass::RemoveLastComponent)
      .def("PropagateBuffer",
        &PyClass::PropagateBuffer)
      .def("BackpropagateBuffer",
        &PyClass::BackpropagateBuffer)
      .def("NumParams",
        &PyClass::NumParams)
      .def("GetGradient",
        &PyClass::GetGradient,
        py::arg("gradient"))
      .def("GetParams",
        &PyClass::GetParams,
        py::arg("params"))
      .def("SetParams",
        &PyClass::SetParams,
        py::arg("params"))
      .def("SetDropoutRate",
        &PyClass::SetDropoutRate,
        py::arg("r"))
      .def("ResetStreams",
        &PyClass::ResetStreams,
        py::arg("stream_reset_flag"))
      .def("SetSeqLengths",
        &PyClass::SetSeqLengths,
        py::arg("sequence_lengths"))
      .def("Init",
        &PyClass::Init,
        py::arg("proto_file"))
      .def("Read",
        py::overload_cast<const std::string &>(&PyClass::Read),
        py::arg("rxfilename"))
      .def("Read",
        py::overload_cast<std::istream &, bool>(&PyClass::Read),
        py::arg("in"),
        py::arg("binary"))
      .def("Write",
        static_cast< void (PyClass::*)(const std::string &, bool) const>(&PyClass::Write),
        py::arg("wxfilename"),
        py::arg("binary"))
      .def("Write",
        static_cast< void (PyClass::*)(std::ostream &, bool) const>(&PyClass::Write),
        py::arg("out"),
        py::arg("binary"))
      .def("Info",
        &PyClass::Info)
      .def("InfoGradient",
        &PyClass::InfoGradient,
        py::arg("header") = true)
      .def("InfoPropagate",
        &PyClass::InfoPropagate,
        py::arg("header") = true)
      .def("InfoBackPropagate",
        &PyClass::InfoBackPropagate,
        py::arg("header") = true)
      .def("Check",
        &PyClass::Check)
      .def("Destroy",
        &PyClass::Destroy)
      .def("SetTrainOptions",
        &PyClass::SetTrainOptions,
        py::arg("opts"))
      .def("GetTrainOptions",
        &PyClass::GetTrainOptions);
  }
}

void pybind_nnet_nnet_parallel_component(py::module& m) {

  {
    using PyClass = ParallelComponent;

    auto parallel_component = py::class_<ParallelComponent, MultistreamComponent>(
        m, "ParallelComponent");
    parallel_component.def(py::init<int32 , int32 >(),
        py::arg("input_dim"),
        py::arg("output_dim"))
      .def("Copy",
        &PyClass::Copy)
      .def("GetType",
        &PyClass::GetType)
      .def("GetNestedNnet",
        py::overload_cast<int32>(&PyClass::GetNestedNnet),
        py::arg("id"))
      .def("InitData",
        &PyClass::InitData,
        py::arg("is"))
      .def("ReadData",
        &PyClass::ReadData,
        py::arg("is"),
        py::arg("binary"))
      .def("WriteData",
        &PyClass::WriteData,
        py::arg("os"),
        py::arg("binary"))
      .def("NumParams",
        &PyClass::NumParams)
      .def("GetGradient",
        &PyClass::GetGradient,
        py::arg("gradient"))
      .def("GetParams",
        &PyClass::GetParams,
        py::arg("params"))
      .def("SetParams",
        &PyClass::SetParams,
        py::arg("params"))
      .def("Info",
        &PyClass::Info)
      .def("InfoGradient",
        &PyClass::InfoGradient)
      .def("InfoPropagate",
        &PyClass::InfoPropagate)
      .def("InfoBackPropagate",
        &PyClass::InfoBackPropagate)
      .def("PropagateFnc",
        &PyClass::PropagateFnc,
        py::arg("in"),
        py::arg("out"))
      .def("BackpropagateFnc",
        &PyClass::BackpropagateFnc,
        py::arg("in"),
        py::arg("out"),
        py::arg("out_diff"),
        py::arg("in_diff"))
      .def("SetTrainOptions",
        &PyClass::SetTrainOptions,
        py::arg("opts"))
      .def("SetLearnRateCoef",
        &PyClass::SetLearnRateCoef,
        py::arg("val"))
      .def("SetBiasLearnRateCoef",
        &PyClass::SetBiasLearnRateCoef,
        py::arg("val"))
      .def("SetSeqLengths",
        &PyClass::SetSeqLengths,
        py::arg("sequence_lengths"));
  }
}

void pybind_nnet_nnet_parametric_relu(py::module& m) {

  {
    using PyClass = ParametricRelu;

    auto parametric_relu = py::class_<ParametricRelu, UpdatableComponent>(
        m, "ParametricRelu");
    parametric_relu.def(py::init<int32 , int32 >(),
        py::arg("input_dim"),
        py::arg("output_dim"))
      .def("Copy",
        &PyClass::Copy)
      .def("GetType",
        &PyClass::GetType)
      .def("InitData",
        &PyClass::InitData,
        py::arg("is"))
      .def("ReadData",
        &PyClass::ReadData,
        py::arg("is"),
        py::arg("binary"))
      .def("WriteData",
        &PyClass::WriteData,
        py::arg("os"),
        py::arg("binary"))
      .def("NumParams",
        &PyClass::NumParams)
      .def("GetGradient",
        &PyClass::GetGradient,
        py::arg("gradient"))
      .def("GetParams",
        &PyClass::GetParams,
        py::arg("params"))
      .def("SetParams",
        &PyClass::SetParams,
        py::arg("params"))
      .def("Info",
        &PyClass::Info)
      .def("InfoGradient",
        &PyClass::InfoGradient)
      .def("PropagateFnc",
        &PyClass::PropagateFnc,
        py::arg("in"),
        py::arg("out"))
      .def("BackpropagateFnc",
        &PyClass::BackpropagateFnc,
        py::arg("in"),
        py::arg("out"),
        py::arg("out_diff"),
        py::arg("in_diff"))
      .def("Update",
        &PyClass::Update,
        "Compute gradient and update parameters",
        py::arg("input"),
        py::arg("diff"));
  }
}

void pybind_nnet_nnet_pdf_prior(py::module& m) {

  {
    using PyClass = PdfPriorOptions;

    auto pdf_prior_options = py::class_<PyClass>(
        m, "PdfPriorOptions");
    pdf_prior_options.def(py::init<>())
      .def_readwrite("class_frame_counts", &PyClass::class_frame_counts)
      .def_readwrite("prior_scale", &PyClass::prior_scale)
      .def_readwrite("prior_floor", &PyClass::prior_floor);
  }
  {
    using PyClass = PdfPrior;

    auto pdf_prior = py::class_<PyClass>(
        m, "PdfPrior");
    pdf_prior.def(py::init<const PdfPriorOptions &>(),
        py::arg("opts"))
      .def("SubtractOnLogpost",
        &PyClass::SubtractOnLogpost,
        "Subtract pdf priors from log-posteriors to get pseudo log-likelihoods",
        py::arg("llk"));
  }
}

void pybind_nnet_nnet_randomizer(py::module& m) {

  {
    using PyClass = NnetDataRandomizerOptions;

    auto nnet_data_randomizer_options = py::class_<PyClass>(
        m, "NnetDataRandomizerOptions");
    nnet_data_randomizer_options.def(py::init<>())
      .def_readwrite("randomizer_size", &PyClass::randomizer_size)
      .def_readwrite("randomizer_seed", &PyClass::randomizer_seed)
      .def_readwrite("minibatch_size", &PyClass::minibatch_size);
  }
  {
    using PyClass = RandomizerMask;

    auto randomizer_mask = py::class_<PyClass>(
        m, "RandomizerMask");
    randomizer_mask.def(py::init<const NnetDataRandomizerOptions &>(),
        py::arg("conf"))
      .def("Init",
        &PyClass::Init,
        py::arg("conf"))
      .def("Generate",
        &PyClass::Generate,
        py::arg("mask_size"));
  }
  {
    using PyClass = MatrixRandomizer;

    auto matrix_randomizer = py::class_<PyClass>(
        m, "MatrixRandomizer");
    matrix_randomizer.def(py::init<>())
      .def(py::init<const NnetDataRandomizerOptions &>(),
        py::arg("conf"))
      .def("Init",
        &PyClass::Init,
        py::arg("conf"))
      .def("AddData",
        &PyClass::AddData,
        py::arg("m"))
      .def("IsFull",
        &PyClass::IsFull)
      .def("NumFrames",
        &PyClass::NumFrames)
      .def("Randomize",
        &PyClass::Randomize,
        py::arg("mask"))
      .def("Done",
        &PyClass::Done)
      .def("Next",
        &PyClass::Next)
      .def("Value",
        &PyClass::Value);
  }
  {
    using PyClass = VectorRandomizer;

    auto vector_randomizer = py::class_<PyClass>(
        m, "VectorRandomizer");
    vector_randomizer.def(py::init<>())
      .def(py::init<const NnetDataRandomizerOptions &>(),
        py::arg("conf"))
      .def("Init",
        &PyClass::Init,
        py::arg("conf"))
      .def("AddData",
        &PyClass::AddData,
        py::arg("m"))
      .def("IsFull",
        &PyClass::IsFull)
      .def("NumFrames",
        &PyClass::NumFrames)
      .def("Randomize",
        &PyClass::Randomize,
        py::arg("mask"))
      .def("Done",
        &PyClass::Done)
      .def("Next",
        &PyClass::Next)
      .def("Value",
        &PyClass::Value);
  }
  {
    using PyClass = Int32VectorRandomizer;

    auto int32_vector_randomizer = py::class_<PyClass>(
        m, "Int32VectorRandomizer");
    int32_vector_randomizer.def(py::init<>())
      .def(py::init<const NnetDataRandomizerOptions &>(),
        py::arg("conf"))
      .def("Init",
        &PyClass::Init,
        py::arg("conf"))
      .def("AddData",
        &PyClass::AddData,
        py::arg("m"))
      .def("IsFull",
        &PyClass::IsFull)
      .def("NumFrames",
        &PyClass::NumFrames)
      .def("Randomize",
        &PyClass::Randomize,
        py::arg("mask"))
      .def("Done",
        &PyClass::Done)
      .def("Next",
        &PyClass::Next)
      .def("Value",
        &PyClass::Value);
  }
  {
    using PyClass = PosteriorRandomizer;

    auto posterior_randomizer = py::class_<PyClass>(
        m, "PosteriorRandomizer");
    posterior_randomizer.def(py::init<>())
      .def(py::init<const NnetDataRandomizerOptions &>(),
        py::arg("conf"))
      .def("Init",
        &PyClass::Init,
        py::arg("conf"))
      .def("AddData",
        &PyClass::AddData,
        py::arg("m"))
      .def("IsFull",
        &PyClass::IsFull)
      .def("NumFrames",
        &PyClass::NumFrames)
      .def("Randomize",
        &PyClass::Randomize,
        py::arg("mask"))
      .def("Done",
        &PyClass::Done)
      .def("Next",
        &PyClass::Next)
      .def("Value",
        &PyClass::Value);
  }
}

void pybind_nnet_nnet_rbm(py::module& m) {

  {
    using PyClass = RbmBase;

    auto rbm_base = py::class_<RbmBase, PyRbmBase>(
        m, "RbmBase");
    rbm_base.def(py::init<int32 , int32>(),
        py::arg("dim_in"),
        py::arg("dim_out"))
      .def("Reconstruct",
        &PyClass::Reconstruct,
        py::arg("hid_state"),
        py::arg("vis_probs"))
      .def("RbmUpdate",
        &PyClass::RbmUpdate,
        py::arg("pos_vis"),
        py::arg("pos_hid"),
        py::arg("neg_vis"),
        py::arg("neg_hid"))
      .def("VisType",
        &PyClass::VisType)
      .def("HidType",
        &PyClass::HidType)
      .def("WriteAsNnet",
        &PyClass::WriteAsNnet,
        py::arg("os"),
        py::arg("binary"))
      .def("SetRbmTrainOptions",
        &PyClass::SetRbmTrainOptions,
        py::arg("opts"))
      .def("GetRbmTrainOptions",
        &PyClass::GetRbmTrainOptions);

    py::enum_<RbmBase::RbmNodeType>(rbm_base, "RbmNodeType")
      .value("Bernoulli", RbmBase::RbmNodeType::Bernoulli)
      .value("Gaussian", RbmBase::RbmNodeType::Gaussian)
      .export_values();
  }
  {
    using PyClass = Rbm;

    auto rbm = py::class_<Rbm, RbmBase>(
        m, "Rbm");
    rbm.def(py::init<int32 , int32>(),
        py::arg("dim_in"),
        py::arg("dim_out"))
      .def("Copy",
        &PyClass::Copy)
      .def("GetType",
        &PyClass::GetType)
      .def("InitData",
        &PyClass::InitData,
        py::arg("is"))
      .def("ReadData",
        &PyClass::ReadData,
        py::arg("is"),
        py::arg("binary"))
      .def("WriteData",
        &PyClass::WriteData,
        py::arg("os"),
        py::arg("binary"))
      .def("Reconstruct",
        &PyClass::Reconstruct,
        py::arg("hid_state"),
        py::arg("vis_probs"))
      .def("RbmUpdate",
        &PyClass::RbmUpdate,
        py::arg("pos_vis"),
        py::arg("pos_hid"),
        py::arg("neg_vis"),
        py::arg("neg_hid"))
      .def("VisType",
        &PyClass::VisType)
      .def("HidType",
        &PyClass::HidType)
      .def("WriteAsNnet",
        &PyClass::WriteAsNnet,
        py::arg("os"),
        py::arg("binary"));

  }
}

void pybind_nnet_nnet_recurrent(py::module& m) {

  {
    using PyClass = RecurrentComponent;

    auto recurrent_component = py::class_<RecurrentComponent, MultistreamComponent>(
        m, "RecurrentComponent");
    recurrent_component.def(py::init<int32 , int32 >(),
        py::arg("dim_in"),
        py::arg("dim_out"))
      .def("Copy",
        &PyClass::Copy)
      .def("GetType",
        &PyClass::GetType)
      .def("InitData",
        &PyClass::InitData,
        py::arg("is"))
      .def("ReadData",
        &PyClass::ReadData,
        py::arg("is"),
        py::arg("binary"))
      .def("WriteData",
        &PyClass::WriteData,
        py::arg("os"),
        py::arg("binary"))
      .def("NumParams",
        &PyClass::NumParams,
        "Number of trainable parameters")
      .def("GetGradient",
        &PyClass::GetGradient,
        "Get gradient reshaped as a vector",
        py::arg("gradient"))
      .def("GetParams",
        &PyClass::GetParams,
        "Get the trainable parameters reshaped as a vector",
        py::arg("params"))
      .def("SetParams",
        &PyClass::SetParams,
        "Set the trainable parameters from, reshaped as a vector",
        py::arg("params"))
      .def("Info",
        &PyClass::Info,
        "Print some additional info (after <ComponentName> and the dims)")
      .def("InfoGradient",
        &PyClass::InfoGradient,
        "Print some additional info about gradient (after <...> and dims)")
      .def("PropagateFnc",
        &PyClass::PropagateFnc,
        py::arg("in"),
        py::arg("out"))
      .def("BackpropagateFnc",
        &PyClass::BackpropagateFnc,
        py::arg("in"),
        py::arg("out"),
        py::arg("out_diff"),
        py::arg("in_diff"))
      .def("Update",
        &PyClass::Update,
        "Compute gradient and update parameters",
        py::arg("input"),
        py::arg("diff"));
  }
}

void pybind_nnet_nnet_sentence_averaging_component(py::module& m) {

  {
    using PyClass = SimpleSentenceAveragingComponent;

    auto simple_sentence_averaging_component = py::class_<SimpleSentenceAveragingComponent, Component>(
        m, "SimpleSentenceAveragingComponent");
    simple_sentence_averaging_component.def(py::init<int32 , int32 >(),
        py::arg("dim_in"),
        py::arg("dim_out"))
      .def("Copy",
        &PyClass::Copy)
      .def("GetType",
        &PyClass::GetType)
      .def("PropagateFnc",
        &PyClass::PropagateFnc,
        py::arg("in"),
        py::arg("out"))
      .def("BackpropagateFnc",
        &PyClass::BackpropagateFnc,
        py::arg("in"),
        py::arg("out"),
        py::arg("out_diff"),
        py::arg("in_diff"));

  }
}

void pybind_nnet_nnet_trnopts(py::module& m) {

    {

      using PyClass = NnetTrainOptions;
      auto nnet_train_options = py::class_<PyClass>(
          m, "NnetTrainOptions");
      nnet_train_options.def(py::init<>())
        .def_readwrite("learn_rate", &PyClass::learn_rate)
        .def_readwrite("momentum", &PyClass::momentum)
        .def_readwrite("l2_penalty", &PyClass::l2_penalty)
        .def_readwrite("l1_penalty", &PyClass::l1_penalty);
    }
    {

      using PyClass = RbmTrainOptions;
      auto rbm_train_options = py::class_<PyClass>(
          m, "RbmTrainOptions");
      rbm_train_options.def(py::init<>())
        .def_readwrite("learn_rate", &PyClass::learn_rate)
        .def_readwrite("momentum", &PyClass::momentum)
        .def_readwrite("momentum_max", &PyClass::momentum_max)
        .def_readwrite("momentum_steps", &PyClass::momentum_steps)
        .def_readwrite("momentum_step_period", &PyClass::momentum_step_period)
        .def_readwrite("l2_penalty", &PyClass::l2_penalty);
    }
}

void pybind_nnet_nnet_various(py::module& m) {

  {
    using PyClass = Splice;

    auto splice = py::class_<Splice, Component>(
        m, "Splice");
    splice.def(py::init<int32 , int32 >(),
        py::arg("dim_in"),
        py::arg("dim_out"))
      .def("Copy",
        &PyClass::Copy)
      .def("GetType",
        &PyClass::GetType)
      .def("PropagateFnc",
        &PyClass::PropagateFnc,
        py::arg("in"),
        py::arg("out"))
      .def("BackpropagateFnc",
        &PyClass::BackpropagateFnc,
        py::arg("in"),
        py::arg("out"),
        py::arg("out_diff"),
        py::arg("in_diff"));

  }
  {
    using PyClass = CopyComponent;

    auto copy_component = py::class_<CopyComponent, Component>(
        m, "CopyComponent");
    copy_component.def(py::init<int32 , int32 >(),
        py::arg("dim_in"),
        py::arg("dim_out"))
      .def("Copy",
        &PyClass::Copy)
      .def("GetType",
        &PyClass::GetType)
      .def("PropagateFnc",
        &PyClass::PropagateFnc,
        py::arg("in"),
        py::arg("out"))
      .def("BackpropagateFnc",
        &PyClass::BackpropagateFnc,
        py::arg("in"),
        py::arg("out"),
        py::arg("out_diff"),
        py::arg("in_diff"));

  }
  {
    using PyClass = LengthNormComponent;

    auto length_norm_component = py::class_<LengthNormComponent, Component>(
        m, "LengthNormComponent");
    length_norm_component.def(py::init<int32 , int32 >(),
        py::arg("dim_in"),
        py::arg("dim_out"))
      .def("Copy",
        &PyClass::Copy)
      .def("GetType",
        &PyClass::GetType)
      .def("PropagateFnc",
        &PyClass::PropagateFnc,
        py::arg("in"),
        py::arg("out"))
      .def("BackpropagateFnc",
        &PyClass::BackpropagateFnc,
        py::arg("in"),
        py::arg("out"),
        py::arg("out_diff"),
        py::arg("in_diff"));

  }

  {
    using PyClass = AddShift;

    auto add_shift_component = py::class_<AddShift, UpdatableComponent>(
        m, "AddShift");
    add_shift_component.def(py::init<int32 , int32 >(),
        py::arg("dim_in"),
        py::arg("dim_out"))
      .def("Copy",
        &PyClass::Copy)
      .def("GetType",
        &PyClass::GetType)
      .def("InitData",
        &PyClass::InitData,
        py::arg("is"))
      .def("ReadData",
        &PyClass::ReadData,
        py::arg("is"),
        py::arg("binary"))
      .def("WriteData",
        &PyClass::WriteData,
        py::arg("os"),
        py::arg("binary"))
      .def("NumParams",
        &PyClass::NumParams,
        "Number of trainable parameters")
      .def("GetGradient",
        &PyClass::GetGradient,
        "Get gradient reshaped as a vector",
        py::arg("gradient"))
      .def("GetParams",
        &PyClass::GetParams,
        "Get the trainable parameters reshaped as a vector",
        py::arg("params"))
      .def("SetParams",
        &PyClass::SetParams,
        "Set the trainable parameters from, reshaped as a vector",
        py::arg("params"))
      .def("Info",
        &PyClass::Info,
        "Print some additional info (after <ComponentName> and the dims)")
      .def("InfoGradient",
        &PyClass::InfoGradient,
        "Print some additional info about gradient (after <...> and dims)")
      .def("Update",
        &PyClass::Update,
        "Compute gradient and update parameters",
        py::arg("input"),
        py::arg("diff"))
      .def("SetLearnRateCoef",
        &PyClass::SetLearnRateCoef,
        py::arg("c"));
  }
  {
    using PyClass = Rescale;

    auto rescale_component = py::class_<Rescale, UpdatableComponent>(
        m, "Rescale");
    rescale_component.def(py::init<int32 , int32 >(),
        py::arg("dim_in"),
        py::arg("dim_out"))
      .def("Copy",
        &PyClass::Copy)
      .def("GetType",
        &PyClass::GetType)
      .def("InitData",
        &PyClass::InitData,
        py::arg("is"))
      .def("ReadData",
        &PyClass::ReadData,
        py::arg("is"),
        py::arg("binary"))
      .def("WriteData",
        &PyClass::WriteData,
        py::arg("os"),
        py::arg("binary"))
      .def("NumParams",
        &PyClass::NumParams,
        "Number of trainable parameters")
      .def("GetGradient",
        &PyClass::GetGradient,
        "Get gradient reshaped as a vector",
        py::arg("gradient"))
      .def("GetParams",
        &PyClass::GetParams,
        "Get the trainable parameters reshaped as a vector",
        py::arg("params"))
      .def("SetParams",
        &PyClass::SetParams,
        "Set the trainable parameters from, reshaped as a vector",
        py::arg("params"))
      .def("Info",
        &PyClass::Info,
        "Print some additional info (after <ComponentName> and the dims)")
      .def("InfoGradient",
        &PyClass::InfoGradient,
        "Print some additional info about gradient (after <...> and dims)")
      .def("Update",
        &PyClass::Update,
        "Compute gradient and update parameters",
        py::arg("input"),
        py::arg("diff"))
      .def("SetLearnRateCoef",
        &PyClass::SetLearnRateCoef,
        py::arg("c"));
  }
}


void init_nnet(py::module &_m) {
  py::module m = _m.def_submodule("nnet", "nnet pybind for Kaldi");
  pybind_nnet_nnet_component(m);
  pybind_nnet_nnet_nnet(m);
  pybind_nnet_nnet_activation(m);
  pybind_nnet_nnet_affine_transform(m);
  pybind_nnet_nnet_average_pooling_component(m);
  pybind_nnet_nnet_blstm_projected(m);
  pybind_nnet_nnet_convolutional_component(m);
  pybind_nnet_nnet_frame_pooling_component(m);
  pybind_nnet_nnet_kl_hmm(m);
  pybind_nnet_nnet_linear_transform(m);
  pybind_nnet_nnet_loss(m);
  pybind_nnet_nnet_lstm_projected(m);
  pybind_nnet_nnet_matrix_buffer(m);
  pybind_nnet_nnet_max_pooling_component(m);
  pybind_nnet_nnet_multibasis_component(m);
  pybind_nnet_nnet_parallel_component(m);
  pybind_nnet_nnet_parametric_relu(m);
  pybind_nnet_nnet_pdf_prior(m);
  pybind_nnet_nnet_randomizer(m);
  pybind_nnet_nnet_rbm(m);
  pybind_nnet_nnet_recurrent(m);
  pybind_nnet_nnet_sentence_averaging_component(m);
  pybind_nnet_nnet_trnopts(m);
  pybind_nnet_nnet_various(m);
}
