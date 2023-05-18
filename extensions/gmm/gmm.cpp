
#include "gmm/pybind_gmm.h"
#include "gmm/am-diag-gmm.h"
#include "gmm/decodable-am-diag-gmm.h"
#include "gmm/diag-gmm-normal.h"
#include "gmm/diag-gmm.h"
#include "gmm/ebw-diag-gmm.h"
#include "gmm/full-gmm-normal.h"
#include "gmm/full-gmm.h"
#include "gmm/indirect-diff-diag-gmm.h"
#include "gmm/mle-am-diag-gmm.h"
#include "gmm/mle-diag-gmm.h"
#include "gmm/mle-full-gmm.h"
#include "gmm/model-common.h"

using namespace kaldi;

void pybind_am_diag_gmm(py::module& m) {
  {
    using PyClass = AmDiagGmm;

    auto am_diag_gmm = py::class_<PyClass>(
        m, "AmDiagGmm");
    am_diag_gmm.def(py::init<>())
      .def("Init",
        &PyClass::Init,
        "Initializes with a single \"prototype\" GMM.",
        py::arg("proto"), py::arg("num_pdfs"))
      .def("AddPdf",
        &PyClass::AddPdf,
        "Adds a GMM to the model, and increments the total number of PDFs.",
        py::arg("gmm"))
      .def("CopyFromAmDiagGmm",
        &PyClass::CopyFromAmDiagGmm,
        "Copies the parameters from another model. Allocates necessary memory.",
        py::arg("other"))
      .def("SplitPdf",
        &PyClass::SplitPdf,
        py::arg("idx"),
        py::arg("target_components"),
        py::arg("perturb_factor"))
      .def("SplitByCount",
        &PyClass::SplitByCount,
        "In SplitByCount we use the \"target_components\" and \"power\" "
        "to work out targets for each state (according to power-of-occupancy rule), "
        "and any state less than its target gets mixed up.  If some states "
        "were over their target, this may take the #Gauss over the target. "
        "we enforce a min-count on Gaussians while splitting (don't split "
        "if it would take it below min-count).",
        py::arg("state_occs"),
        py::arg("target_components"),
        py::arg("perturb_factor"),
        py::arg("power"),
        py::arg("min_count"))
      .def("MergeByCount",
        &PyClass::MergeByCount,
        "In MergeByCount we use the \"target_components\" and \"power\" "
        "to work out targets for each state (according to power-of-occupancy rule), "
        "and any state over its target gets mixed down.  If some states "
        "were under their target, this may take the #Gauss below the target.",
        py::arg("state_occs"),
        py::arg("target_components"),
        py::arg("power"),
        py::arg("min_count"))
      .def("ComputeGconsts",
        &PyClass::ComputeGconsts,
        "Sets the gconsts for all the PDFs. Returns the total number of Gaussians "
        "over all PDFs that are \"invalid\" e.g. due to zero weights or variances.")
      .def("LogLikelihood",
        &PyClass::LogLikelihood,
        py::arg("pdf_index"),
        py::arg("data"))
      .def("Read", &PyClass::Read, py::arg("in_stream"), py::arg("binary"))
      .def("Write", &PyClass::Write, py::arg("out_stream"), py::arg("binary"))
      .def("Dim", &PyClass::Dim)
      .def("NumPdfs", &PyClass::NumPdfs)
      .def("NumGauss", &PyClass::NumGauss)
      .def("NumGaussInPdf", &PyClass::NumGaussInPdf, py::arg("pdf_index"))
      .def("GetPdf", &PyClass::NumGaussInPdf, py::arg("pdf_index"))
      .def("GetGaussianMean", &PyClass::GetGaussianMean,
          py::arg("pdf_index"), py::arg("gauss"), py::arg("out"))
      .def("GetGaussianVariance", &PyClass::GetGaussianVariance,
          py::arg("pdf_index"), py::arg("gauss"), py::arg("out"))
      .def("SetGaussianMean", &PyClass::SetGaussianMean,
          py::arg("pdf_index"), py::arg("gauss_index"), py::arg("in"));
  }
  {
    using PyClass = UbmClusteringOptions;

    auto ubm_clustering_options = py::class_<PyClass>(
        m, "UbmClusteringOptions");
    ubm_clustering_options.def(py::init<>())
      .def(py::init<int32, BaseFloat, int32,
                       BaseFloat, int32>(),
                       py::arg("ncomp"),
                       py::arg("red"),
                       py::arg("interm_gauss"),
                       py::arg("vfloor"),
                       py::arg("max_am_gauss"))
      .def_readwrite("ubm_num_gauss", &PyClass::ubm_num_gauss)
      .def_readwrite("reduce_state_factor", &PyClass::reduce_state_factor)
      .def_readwrite("intermediate_num_gauss", &PyClass::intermediate_num_gauss)
      .def_readwrite("cluster_varfloor", &PyClass::cluster_varfloor)
      .def_readwrite("max_am_gauss", &PyClass::max_am_gauss)
      .def("Check",
        &PyClass::Check);
  }
  m.def("ClusterGaussiansToUbm",
        &ClusterGaussiansToUbm,
        "Clusters the Gaussians in an acoustic model to a single GMM with specified "
        "number of components. First the each state is mixed-down to a single "
        "Gaussian, then the states are clustered by clustering these Gaussians in a "
        "bottom-up fashion. Number of clusters is determined by reduce_state_factor. "
        "The Gaussians for each cluster of states are then merged based on the least "
        "likelihood reduction till there are intermediate_numcomp Gaussians, which "
        "are then merged into ubm_num_gauss Gaussians. "
        "This is the UBM initialization algorithm described in section 2.1 of Povey, "
        "et al., \"The subspace Gaussian mixture model - A structured model for speech "
        "recognition\", In Computer Speech and Language, April 2011.",
        py::arg("am"),
        py::arg("state_occs"),
        py::arg("opts"),
        py::arg("ubm_out"));
}

void pybind_decodable_am_diag_gmm(py::module& m) {

  {
    using PyClass = DecodableAmDiagGmmUnmapped;

    auto decodable_am_diag_gmm_unmapped = py::class_<DecodableAmDiagGmmUnmapped, DecodableInterface>(
        m, "DecodableAmDiagGmmUnmapped",
        "DecodableAmDiagGmmUnmapped is a decodable object that "
        "takes indices that correspond to pdf-id's plus one. "
        "This may be used in future in a decoder that doesn't need "
        "to output alignments, if we create FSTs that have the pdf-ids "
        "plus one as the input labels (we couldn't use the pdf-ids "
        "themselves because they start from zero, and the graph might "
        "have epsilon transitions).");
    decodable_am_diag_gmm_unmapped.def(py::init<const AmDiagGmm &,
                             const Matrix<BaseFloat> &,
                             BaseFloat>(),
                             "If you set log_sum_exp_prune to a value greater than 0 it will prune "
                              "in the LogSumExp operation (larger = more exact); I suggest 5. "
                              "This is advisable if it's spending a long time doing exp  "
                              "operations.",
          py::arg("am"),
          py::arg("feats"),
          py::arg("log_sum_exp_prune"))
      .def("LogLikelihood",
        &PyClass::LogLikelihood,
        "Note, frames are numbered from zero.  But state_index is numbered "
        "from one (this routine is called by FSTs).",
          py::arg("frame"),
          py::arg("state_index"))
      .def("NumFramesReady",
        &PyClass::NumFramesReady)
      .def("NumIndices",
        &PyClass::NumIndices,
        "Indices are one-based!  This is for compatibility with OpenFst.")
      .def("IsLastFrame",
        &PyClass::IsLastFrame,
          py::arg("frame"));
  }
  {
    using PyClass = DecodableAmDiagGmm;

    auto decodable_am_diag_gmm = py::class_<DecodableAmDiagGmm, DecodableAmDiagGmmUnmapped>(
        m, "DecodableAmDiagGmm");
    decodable_am_diag_gmm.def(py::init<const AmDiagGmm &,
                            const TransitionModel &,
                             const Matrix<BaseFloat> &,
                             BaseFloat>(),
          py::arg("am"),
          py::arg("tm"),
          py::arg("feats"),
          py::arg("log_sum_exp_prune"))
      .def("LogLikelihood",
        &PyClass::LogLikelihood,
        "Note, frames are numbered from zero.",
          py::arg("frame"),
          py::arg("tid"))
      .def("NumIndices",
        &PyClass::NumIndices,
        "Indices are one-based!  This is for compatibility with OpenFst.")
      .def("TransModel",
        &PyClass::TransModel);
  }
  {
    using PyClass = DecodableAmDiagGmmScaled;

    auto decodable_am_diag_gmm_scaled = py::class_<DecodableAmDiagGmmScaled, DecodableAmDiagGmmUnmapped>(
        m, "DecodableAmDiagGmmScaled");
    decodable_am_diag_gmm_scaled.def(py::init<const AmDiagGmm &,
                            const TransitionModel &,
                             const Matrix<BaseFloat> &,
                           BaseFloat,
                             BaseFloat>(),
          py::arg("am"),
          py::arg("tm"),
          py::arg("feats"),
          py::arg("scale"),
          py::arg("log_sum_exp_prune"))
      .def(py::init<const AmDiagGmm &,
                    const TransitionModel &,
                    BaseFloat,
                    BaseFloat,
                    Matrix<BaseFloat> *>(),
          py::arg("am"),
          py::arg("tm"),
          py::arg("scale"),
          py::arg("log_sum_exp_prune"),
          py::arg("feats"))
      .def("LogLikelihood",
        &PyClass::LogLikelihood,
        "Note, frames are numbered from zero but transition-ids from one.",
          py::arg("frame"),
          py::arg("tid"))
      .def("NumIndices",
        &PyClass::NumIndices,
        "Indices are one-based!  This is for compatibility with OpenFst.")
      .def("TransModel",
        &PyClass::TransModel);
  }
}

void pybind_diag_gmm_normal(py::module& m) {
  {
    using PyClass = DiagGmmNormal;

    auto diag_gmm_normal = py::class_<PyClass>(
        m, "DiagGmmNormal");
    diag_gmm_normal.def(py::init<>())
      .def(py::init<const DiagGmm &>(),
          py::arg("gmm"))
      .def("Resize",
        &PyClass::Resize,
        "Resizes arrays to this dim. Does not initialize data.",
          py::arg("nMix"),
          py::arg("dim"))
      .def("CopyFromDiagGmm",
        &PyClass::CopyFromDiagGmm,
        "Copies from given DiagGmm",
          py::arg("diaggmm"))
      .def("CopyToDiagGmm",
        &PyClass::CopyToDiagGmm,
        "Copies to DiagGmm the requested parameters",
          py::arg("diaggmm"),
          py::arg("flags"))
      .def("NumGauss",
        &PyClass::NumGauss)
      .def("Dim",
        &PyClass::Dim);
  }

}

void pybind_diag_gmm(py::module& m) {
  {
    using PyClass = DiagGmm;

    auto diag_gmm = py::class_<PyClass>(
        m, "DiagGmm");
    diag_gmm.def(py::init<>())
      .def(py::init<const DiagGmm &>(),
          py::arg("gmm"))
      .def(py::init<const GaussClusterable &, BaseFloat>(),
      "Initializer from GaussClusterable initializes the DiagGmm as "
      "a single Gaussian from tree stats.",
          py::arg("gc"),
          py::arg("var_floor"))
      .def(py::init<int32, int32>(),
          py::arg("nMix"),
          py::arg("dim"))
      .def(py::init<const std::vector<std::pair<BaseFloat, const DiagGmm*> > &>(),
          "Constructor that allows us to merge GMMs with weights.  Weights must sum "
          "to one, or this GMM will not be properly normalized (we don't check this). "
          "Weights must be positive (we check this).",
          py::arg("gmms"))
      .def("CopyFromNormal",
        &PyClass::CopyFromNormal,
        "Copies from DiagGmmNormal; does not resize.",
          py::arg("diag_gmm_normal"))
      .def("Resize",
        &PyClass::Resize,
        "Resizes arrays to this dim. Does not initialize data.",
          py::arg("nMix"),
          py::arg("dim"))
      .def("CopyFromDiagGmm",
        &PyClass::CopyFromDiagGmm,
        "Copies from given DiagGmm",
          py::arg("diaggmm"))
      .def("CopyFromFullGmm",
        &PyClass::CopyFromFullGmm,
        "Copies from given FullGmm",
          py::arg("fullgmm"))
      .def("NumGauss",
        &PyClass::NumGauss,
        "Returns the number of mixture components in the GMM")
      .def("Dim",
        &PyClass::Dim,
        "Returns the dimensionality of the Gaussian mean vectors")
      .def("LogLikelihood",
        &PyClass::LogLikelihood,
        "Returns the log-likelihood of a data point (vector) given the GMM",
          py::arg("data"))
      .def("LogLikelihoods",
        py::overload_cast<const VectorBase<BaseFloat> &,
                      Vector<BaseFloat> *>(&PyClass::LogLikelihoods, py::const_),
        "Outputs the per-component log-likelihoods",
          py::arg("data"),
          py::arg("loglikes"))
      .def("LogLikelihoods",
        py::overload_cast<const MatrixBase<BaseFloat> &,
                      Matrix<BaseFloat> *>(&PyClass::LogLikelihoods, py::const_),
        "Outputs the per-component log-likelihoods",
          py::arg("data"),
          py::arg("loglikes"))
      .def("LogLikelihoodsPreselect",
        &PyClass::LogLikelihoodsPreselect,
        "Outputs the per-component log-likelihoods of a subset of mixture "
        "components.  Note: at output, loglikes->Dim() will equal indices.size(). "
        "loglikes[i] will correspond to the log-likelihood of the Gaussian "
        "indexed indices[i], including the mixture weight.",
          py::arg("data"),
          py::arg("indices"),
          py::arg("loglikes"))
      .def("GaussianSelection",
        py::overload_cast<const VectorBase<BaseFloat> &,
                              int32,
                              std::vector<int32> *>(&PyClass::GaussianSelection, py::const_),
        "Get gaussian selection information for one frame.  Returns log-like "
        "this frame.  Output is the best \"num_gselect\" indices, sorted from best to "
        "worst likelihood.  If \"num_gselect\" > NumGauss(), sets it to NumGauss().",
          py::arg("data"),
          py::arg("num_gselect"),
          py::arg("output"))
      .def("GaussianSelection",
        py::overload_cast<const MatrixBase<BaseFloat> &,
                              int32,
                              std::vector<std::vector<int32> > *>(&PyClass::GaussianSelection, py::const_),
        "Get gaussian selection information for one frame.  Returns log-like "
        "this frame.  Output is the best \"num_gselect\" indices, sorted from best to "
        "worst likelihood.  If \"num_gselect\" > NumGauss(), sets it to NumGauss().",
          py::arg("data"),
          py::arg("num_gselect"),
          py::arg("output"))
      .def("GaussianSelectionPreselect",
        &PyClass::GaussianSelectionPreselect,
        "Get gaussian selection information for one frame.  Returns log-like for "
        "this frame.  Output is the best \"num_gselect\" indices that were "
        "preselected, sorted from best to worst likelihood.  If \"num_gselect\" > "
        "NumGauss(), sets it to NumGauss().",
          py::arg("data"),
          py::arg("preselect"),
          py::arg("num_gselect"),
          py::arg("output"))
      .def("ComponentPosteriors",
        &PyClass::ComponentPosteriors,
        "Computes the posterior probabilities of all Gaussian components given "
        "a data point. Returns the log-likehood of the data given the GMM.",
          py::arg("data"),
          py::arg("posteriors"))
      .def("ComponentLogLikelihood",
        &PyClass::ComponentLogLikelihood,
        "Computes the log-likelihood of a data point given a single Gaussian "
        "component. NOTE: Currently we make no guarantees about what happens if "
        "one of the variances is zero.",
          py::arg("data"),
          py::arg("comp_id"))
      .def("ComputeGconsts",
        &PyClass::ComputeGconsts,
        "Sets the gconsts.  Returns the number that are \"invalid\" e.g. because of "
        "zero weights or variances.")
      .def("Generate",
        &PyClass::Generate,
        "Generates a random data-point from this distribution.",
          py::arg("output"))
      .def("Split",
        &PyClass::Split,
        "Split the components and remember the order in which the components were split",
          py::arg("target_components"),
          py::arg("perturb_factor"),
          py::arg("history"))
      .def("Perturb",
        &PyClass::Perturb,
        "Perturbs the component means with a random vector multiplied by the "
        "pertrub factor.",
          py::arg("perturb_factor"))
      .def("Merge",
        &PyClass::Merge,
        "Merge the components and remember the order in which the components were "
        "merged (flat list of pairs)",
          py::arg("target_components"),
          py::arg("history") = NULL)
      .def("MergeKmeans",
        &PyClass::MergeKmeans,
        "Merge the components to a specified target #components: this "
        "version uses a different approach based on K-means.",
          py::arg("target_components"),
          py::arg("cfg") = ClusterKMeansOptions())
      .def("Write",
        &PyClass::Write,
          py::arg("os"),
          py::arg("binary"))
      .def("Read",
        &PyClass::Read,
          py::arg("is"),
          py::arg("binary"))
      .def("Interpolate",
        py::overload_cast<BaseFloat, const DiagGmm &,
                   GmmFlagsType>(&PyClass::Interpolate),
        "this = rho x source + (1-rho) x this",
          py::arg("rho"),
          py::arg("source"),
          py::arg("flags") = kGmmAll)
      .def("Interpolate",
        py::overload_cast<BaseFloat, const FullGmm &,
                   GmmFlagsType>(&PyClass::Interpolate),
        "this = rho x source + (1-rho) x this",
          py::arg("rho"),
          py::arg("source"),
          py::arg("flags") = kGmmAll)
      .def("gconsts",
        &PyClass::gconsts)
      .def("weights",
        &PyClass::weights)
      .def("means_invvars",
        &PyClass::means_invvars)
      .def("inv_vars",
        &PyClass::inv_vars)
      .def("valid_gconsts",
        &PyClass::valid_gconsts)
      .def("RemoveComponent",
        &PyClass::RemoveComponent,
        "Removes single component from model",
          py::arg("gauss"),
          py::arg("renorm_weights"))
      .def("RemoveComponents",
        &PyClass::RemoveComponents,
        "Removes multiple components from model; \"gauss\" must not have dups.",
          py::arg("gauss"),
          py::arg("renorm_weights"))
      .def("SetWeights",
        &PyClass::SetWeights<float>,
          py::arg("w"))
      .def("SetMeans",
        &PyClass::SetMeans<float>,
        "Use SetMeans to update only the Gaussian means (and not variances)",
          py::arg("m"))
      .def("SetInvVarsAndMeans",
        &PyClass::SetInvVarsAndMeans<float>,
        "Use SetInvVarsAndMeans if updating both means and (inverse) variances",
          py::arg("invvars"),
          py::arg("means"))
      .def("SetInvVars",
        &PyClass::SetInvVars<float>,
        "Set the (inverse) variances and recompute means_invvars_",
          py::arg("v"))
      .def("GetVars",
        &PyClass::GetVars<float>,
        "Accessor for covariances.",
          py::arg("v"))
      .def("GetMeans",
        &PyClass::GetMeans<float>,
        "Accessor for means.",
          py::arg("m"))
      .def("SetComponentMean",
        &PyClass::SetComponentMean<float>,
        "Set mean for a single component - internally multiplies with inv(var)",
          py::arg("gauss"),
          py::arg("in"))
      .def("SetComponentInvVar",
        &PyClass::SetComponentInvVar<float>,
        "Set inv-var for single component (recommend to do this before "
        "setting the mean, if doing both, for numerical reasons).",
          py::arg("gauss"),
          py::arg("in"))
      .def("SetComponentWeight",
        &PyClass::SetComponentWeight,
        "Set weight for single component.",
          py::arg("gauss"),
          py::arg("weight"))
      .def("GetComponentMean",
        &PyClass::GetComponentMean<float>,
        "Accessor for single component mean",
          py::arg("gauss"),
          py::arg("out"))
      .def("GetComponentVariance",
        &PyClass::GetComponentVariance<float>,
        "Accessor for single component variance",
          py::arg("gauss"),
          py::arg("out"));
  }

}

void pybind_ebw_diag_gmm(py::module& m) {

  {
    using PyClass = EbwOptions;

    auto ebw_options = py::class_<PyClass>(
        m, "EbwOptions");
    ebw_options.def(py::init<>())
      .def_readwrite("E", &PyClass::E)
      .def_readwrite("tau", &PyClass::tau);
  }
  {
    using PyClass = EbwWeightOptions;

    auto ebw_weight_options = py::class_<PyClass>(
        m, "EbwWeightOptions");
    ebw_weight_options.def(py::init<>())
      .def_readwrite("min_num_count_weight_update",
                &PyClass::min_num_count_weight_update,
                "minimum numerator count at state level, before we update.")
      .def_readwrite("min_gaussian_weight",
                &PyClass::min_gaussian_weight)
      .def_readwrite("tau",
                &PyClass::tau,
                "tau value for smoothing stats in weight update.  Should probably "
                "be 10.0 or so, leaving it at 0 for back-compatibility.");
  }
  m.def("UpdateEbwDiagGmm",
        &UpdateEbwDiagGmm,
        "Update Gaussian parameters only (no weights) "
        "The pointer parameters auxf_change_out etc. are incremented, not set.",
        py::arg("num_stats"),
        py::arg("den_stats"),
        py::arg("flags"),
        py::arg("opts"),
        py::arg("gmm"),
        py::arg("auxf_change_out"),
        py::arg("count_out"),
        py::arg("num_floored_out"));
  m.def("UpdateEbwAmDiagGmm",
        &UpdateEbwAmDiagGmm,
        py::arg("num_stats"),
        py::arg("den_stats"),
        py::arg("flags"),
        py::arg("opts"),
        py::arg("am_gmm"),
        py::arg("auxf_change_out"),
        py::arg("count_out"),
        py::arg("num_floored_out"));
  m.def("UpdateEbwWeightsDiagGmm",
        &UpdateEbwWeightsDiagGmm,
        "Updates the weights using the EBW-like method described in Dan Povey's thesis "
        "(this method has no tunable parameters). "
        "The pointer parameters auxf_change_out etc. are incremented, not set.",
        py::arg("num_stats"),
        py::arg("den_stats"),
        py::arg("opts"),
        py::arg("gmm"),
        py::arg("auxf_change_out"),
        py::arg("count_out"));
  m.def("UpdateEbwWeightsAmDiagGmm",
        &UpdateEbwWeightsAmDiagGmm,
        py::arg("num_stats"),
        py::arg("den_stats"),
        py::arg("opts"),
        py::arg("am_gmm"),
        py::arg("auxf_change_out"),
        py::arg("count_out"));
  m.def("IsmoothStatsDiagGmm",
        &IsmoothStatsDiagGmm,
        "I-Smooth the stats.  src_stats and dst_stats do not have to be different.",
        py::arg("src_stats"),
        py::arg("tau"),
        py::arg("dst_stats"));
  m.def("DiagGmmToStats",
        &DiagGmmToStats,
        "Creates stats from the GMM.  Resizes them as needed.",
        py::arg("gmm"),
        py::arg("flags"),
        py::arg("state_occ"),
        py::arg("dst_stats"));
  m.def("IsmoothStatsAmDiagGmm",
        &IsmoothStatsAmDiagGmm,
        "Smooth \"dst_stats\" with \"src_stats\".  They don't have to be different.",
        py::arg("src_stats"),
        py::arg("tau"),
        py::arg("dst_stats"));
  m.def("IsmoothStatsAmDiagGmmFromModel",
        &IsmoothStatsAmDiagGmmFromModel,
        "This version of the I-smoothing function takes a model as input.",
        py::arg("src_model"),
        py::arg("tau"),
        py::arg("dst_stats"));

}

void pybind_full_gmm_normal(py::module& m) {

  {
    using PyClass = FullGmmNormal;

    auto full_gmm_normal = py::class_<PyClass>(
        m, "FullGmmNormal");
    full_gmm_normal.def(py::init<>())
      .def(py::init<const FullGmm &>(),
        py::arg("gmm"))
      .def("Resize",
        &PyClass::Resize,
        "Resizes arrays to this dim. Does not initialize data.",
        py::arg("nMix"),
        py::arg("dim"))
      .def("CopyFromFullGmm",
        &PyClass::CopyFromFullGmm,
        "Copies from given FullGmm",
        py::arg("fullgmm"))
      .def("CopyToFullGmm",
        &PyClass::CopyToFullGmm,
        "Copies to FullGmm",
        py::arg("fullgmm"),
        py::arg("flags") = kGmmAll)
      .def("Rand",
        &PyClass::Rand,
        "Generates random features from the model.",
        py::arg("feats"));
  }
}

void pybind_full_gmm(py::module& m) {

  {
    using PyClass = FullGmm;

    auto full_gmm = py::class_<PyClass>(
        m, "FullGmm");
    full_gmm.def(py::init<>())
      .def(py::init<const FullGmm &>(),
        py::arg("gmm"))
      .def(py::init<int32, int32>(),
        py::arg("nMix"),
        py::arg("dim"))
      .def("Resize",
        &PyClass::Resize,
        "Resizes arrays to this dim. Does not initialize data.",
        py::arg("nMix"),
        py::arg("dim"))
      .def("NumGauss",
        &PyClass::NumGauss,
        "Returns the number of mixture components in the GMM")
      .def("Dim",
        &PyClass::Dim,
        "Returns the dimensionality of the Gaussian mean vectors")
      .def("CopyFromFullGmm",
        &PyClass::CopyFromFullGmm,
        "Copies from given FullGmm",
        py::arg("fullgmm"))
      .def("CopyFromDiagGmm",
        &PyClass::CopyFromDiagGmm,
        "Copies from given DiagGmm",
        py::arg("diaggmm"))
      .def("LogLikelihood",
        &PyClass::LogLikelihood,
        "Returns the log-likelihood of a data point (vector) given the GMM",
          py::arg("data"))
      .def("LogLikelihoods",
        &PyClass::LogLikelihoods,
        "Outputs the per-component contributions to the log-likelihood",
          py::arg("data"),
          py::arg("loglikes"))
      .def("LogLikelihoodsPreselect",
        &PyClass::LogLikelihoodsPreselect,
        "Outputs the per-component log-likelihoods of a subset of mixture "
        "components. Note: indices.size() will equal loglikes->Dim() at output. "
        "loglikes[i] will correspond to the log-likelihood of the Gaussian "
        "indexed indices[i].",
          py::arg("data"),
          py::arg("indices"),
          py::arg("loglikes"))
      .def("GaussianSelection",
        &PyClass::GaussianSelection,
        "Get gaussian selection information for one frame.  Returns log-like for "
        "this frame.  Output is the best \"num_gselect\" indices, sorted from best to "
        "worst likelihood.  If \"num_gselect\" > NumGauss(), sets it to NumGauss().",
          py::arg("data"),
          py::arg("num_gselect"),
          py::arg("output"))
      .def("GaussianSelectionPreselect",
        &PyClass::GaussianSelectionPreselect,
        "Get gaussian selection information for one frame.  Returns log-like for "
        "this frame.  Output is the best \"num_gselect\" indices that were "
        "preselected, sorted from best to worst likelihood.  If \"num_gselect\" > "
        "NumGauss(), sets it to NumGauss().",
          py::arg("data"),
          py::arg("preselect"),
          py::arg("num_gselect"),
          py::arg("output"))
      .def("ComponentPosteriors",
        &PyClass::ComponentPosteriors,
        "Computes the posterior probabilities of all Gaussian components given "
        "a data point. Returns the log-likehood of the data given the GMM.",
          py::arg("data"),
          py::arg("posterior"))
      .def("ComponentLogLikelihood",
        &PyClass::ComponentLogLikelihood,
        "Computes the contribution log-likelihood of a data point from a single "
        "Gaussian component. NOTE: Currently we make no guarantees about what "
        "happens if one of the variances is zero.",
          py::arg("data"),
          py::arg("comp_id"))
      .def("ComputeGconsts",
        &PyClass::ComputeGconsts,
        "Sets the gconsts.  Returns the number that are \"invalid\" e.g. because of "
        "zero weights or variances.")
      .def("Split",
        &PyClass::Split,
        "Merge the components and remember the order in which the components were "
        "merged (flat list of pairs)",
          py::arg("target_components"),
          py::arg("perturb_factor"),
          py::arg("history") = NULL)
      .def("Perturb",
        &PyClass::Perturb,
        "Perturbs the component means with a random vector multiplied by the "
        "pertrub factor.",
          py::arg("perturb_factor"))
      .def("Merge",
        &PyClass::Merge,
        "Merge the components and remember the order in which the components were "
        "merged (flat list of pairs)",
          py::arg("target_components"),
          py::arg("history") = NULL)
      .def("MergePreselect",
        &PyClass::MergePreselect,
        "Merge the components and remember the order in which the components were "
        "merged (flat list of pairs); this version only considers merging "
        "pairs in \"preselect_pairs\" (or their descendants after merging). "
        "This is for efficiency, for large models.  Returns the delta likelihood.",
          py::arg("target_components"),
          py::arg("preselect_pairs"))
      .def("Write",
        &PyClass::Write,
          py::arg("os"),
          py::arg("binary"))
      .def("Read",
        &PyClass::Read,
          py::arg("is"),
          py::arg("binary"))
      .def("Interpolate",
        &PyClass::Interpolate,
        "this = rho x source + (1-rho) x this",
          py::arg("rho"),
          py::arg("source"),
          py::arg("flags") = kGmmAll)
      .def("gconsts",
        &PyClass::gconsts)
      .def("weights",
        &PyClass::weights)
      //.def("means_invcovars",
      //  &PyClass::means_invcovars)
      //.def("inv_covars",
      //  &PyClass::inv_covars)
      .def("SetWeights",
        &PyClass::SetWeights<float>,
          py::arg("w"))
      .def("SetWeights",
        &PyClass::SetWeights<double>,
          py::arg("w"))
      .def("SetMeans",
        &PyClass::SetMeans<float>,
        "Use SetMeans to update only the Gaussian means (and not variances)",
          py::arg("m"))
      .def("SetMeans",
        &PyClass::SetMeans<double>,
        "Use SetMeans to update only the Gaussian means (and not variances)",
          py::arg("m"))
      .def("SetInvCovarsAndMeans",
        &PyClass::SetInvCovarsAndMeans<float>,
        "Use SetInvCovarsAndMeans if updating both means and (inverse) variances",
          py::arg("invcovars"),
          py::arg("means"))
      .def("SetInvCovarsAndMeans",
        &PyClass::SetInvCovarsAndMeans<double>,
        "Use SetInvCovarsAndMeans if updating both means and (inverse) variances",
          py::arg("invcovars"),
          py::arg("means"))
      .def("SetInvCovarsAndMeansInvCovars",
        &PyClass::SetInvCovarsAndMeansInvCovars<float>,
        "Use this if setting both, in the class's native format.",
          py::arg("invcovars"),
          py::arg("means_invcovars"))
      .def("SetInvCovarsAndMeansInvCovars",
        &PyClass::SetInvCovarsAndMeansInvCovars<double>,
        "Use this if setting both, in the class's native format.",
          py::arg("invcovars"),
          py::arg("means_invcovars"))
      .def("SetInvCovars",
        &PyClass::SetInvCovars<float>,
        "Set the (inverse) covariances and recompute means_invcovars_",
          py::arg("v"))
      .def("SetInvCovars",
        &PyClass::SetInvCovars<double>,
        "Set the (inverse) covariances and recompute means_invcovars_",
          py::arg("v"))
      .def("GetCovars",
        &PyClass::GetCovars<float>,
        "Accessor for covariances.",
          py::arg("v"))
      .def("GetCovars",
        &PyClass::GetCovars<double>,
        "Accessor for covariances.",
          py::arg("v"))
      .def("GetMeans",
        &PyClass::GetMeans<float>,
        "Accessor for means.",
          py::arg("m"))
      .def("GetMeans",
        &PyClass::GetMeans<double>,
        "Accessor for means.",
          py::arg("m"))
      .def("GetCovarsAndMeans",
        &PyClass::GetCovarsAndMeans<float>,
        "Accessor for covariances and means",
          py::arg("covars"),
          py::arg("m"))
      .def("GetCovarsAndMeans",
        &PyClass::GetCovarsAndMeans<double>,
        "Accessor for covariances and means",
          py::arg("covars"),
          py::arg("m"))
      .def("RemoveComponent",
        &PyClass::RemoveComponent,
        "Removes single component from model",
          py::arg("gauss"),
          py::arg("renorm_weights"))
      .def("RemoveComponents",
        &PyClass::RemoveComponents,
        "Removes multiple components from model; \"gauss\" must not have dups.",
          py::arg("gauss"),
          py::arg("renorm_weights"))
      .def("GetComponentMean",
        &PyClass::GetComponentMean<float>,
        "Accessor for single component mean",
          py::arg("gauss"),
          py::arg("out"))
      .def("GetComponentMean",
        &PyClass::GetComponentMean<double>,
        "Accessor for single component mean",
          py::arg("gauss"),
          py::arg("out"));
  }

}

void pybind_indirect_diff_diag_gmm(py::module& m) {

  m.def("GetStatsDerivative",
        &GetStatsDerivative,
        "This gets the derivative of the (MMI or MPE) objective function w.r.t. the "
        "statistics for ML update, assuming we're doing an ML update-- as described in "
        "the original fMPE paper.  This is used in fMPE/fMMI, for the \"indirect "
        "differential\".  This derivative is represented as class AccumDiagGmm, as "
        "derivatives w.r.t. the x and x^2 stats directly (not w.r.t. the mean and "
        "variance). "
        "\n"
        "If the parameter \"rescaling\" is true, this function will assume that instead "
        "of the ML update, you will do a \"rescaling\" update as in the function "
        "DoRescalingUpdate(). "
        "\n"
        "CAUTION: for fMPE (as opposed to fMMI), to get the right answer, you would have "
        "to pre-scale the num and den accs by the acoustic scale (e.g. 0.1).",
        py::arg("gmm"),
        py::arg("num_accs"),
        py::arg("den_accs"),
        py::arg("ml_accs"),
        py::arg("min_variance"),
        py::arg("min_gaussian_occupancy"),
        py::arg("out_accs"));
  m.def("DoRescalingUpdate",
        &DoRescalingUpdate,
        "This function \"DoRescalingUpdate\" updates the GMMs in a special way-- it "
        "first works out how the Gaussians differ from the old stats (in terms of an "
        "offset on the mean, a scale on the variance, and a factor on the weights), "
        "and it updates the model so that it will differ in the same way from the new "
        "stats. "
        "\n"
        "The idea here is that the original model may have been discriminatively "
        "trained, but we may have changed the features or the domain or something "
        "of that nature, and we want to update the model but preserve the discriminative "
        "training (viewed as an offset).",
        py::arg("old_ml_accs"),
        py::arg("new_ml_accs"),
        py::arg("min_variance"),
        py::arg("min_gaussian_occupancy"),
        py::arg("gmm"));

}

void pybind_mle_am_diag_gmm(py::module& m) {

  {
    using PyClass = AccumAmDiagGmm;

    auto accum_am_diag_gmm = py::class_<PyClass>(
        m, "AccumAmDiagGmm");
    accum_am_diag_gmm.def(py::init<>())
      .def("Read",
        &PyClass::Read,
        py::arg("in_stream"),
        py::arg("binary"),
        py::arg("add") = false)
      .def("Write",
        &PyClass::Write,
        py::arg("out_stream"),
        py::arg("binary"))
      .def("Init",
        py::overload_cast<const AmDiagGmm &, GmmFlagsType>(&PyClass::Init),
        "Initializes accumulators for each GMM based on the number of components "
        "and dimension.",
        py::arg("model"),
        py::arg("flags"))
      .def("Init",
        py::overload_cast<const AmDiagGmm &, int32, GmmFlagsType>(&PyClass::Init),
        "Initialization using different dimension than model.",
        py::arg("model"),
        py::arg("dim"),
        py::arg("flags"))
      .def("SetZero",
        &PyClass::SetZero,
        py::arg("flags"))
      .def("AccumulateForGmm",
        &PyClass::AccumulateForGmm,
        "Accumulate stats for a single GMM in the model; returns log likelihood. "
        "This does not work with multiple feature transforms.",
        py::arg("model"),
        py::arg("data"),
        py::arg("gmm_index"),
        py::arg("weight"))
      .def("AccumulateForGmmTwofeats",
        &PyClass::AccumulateForGmmTwofeats,
        "Accumulate stats for a single GMM in the model; uses data1 for "
        "getting posteriors and data2 for stats. Returns log likelihood.",
        py::arg("model"),
        py::arg("data1"),
        py::arg("data2"),
        py::arg("gmm_index"),
        py::arg("weight"))
      .def("AccumulateFromPosteriors",
        &PyClass::AccumulateFromPosteriors,
        "Accumulates stats for a single GMM in the model using pre-computed "
        "Gaussian posteriors.",
        py::arg("model"),
        py::arg("data"),
        py::arg("gmm_index"),
        py::arg("posteriors"))
      .def("AccumulateForGaussian",
        &PyClass::AccumulateForGaussian,
        "Accumulate stats for a single Gaussian component in the model.",
        py::arg("am"),
        py::arg("data"),
        py::arg("gmm_index"),
        py::arg("gauss_index"),
        py::arg("weight"))
      .def("NumAccs",
        py::overload_cast<>(&PyClass::NumAccs))
      .def("TotStatsCount",
        &PyClass::TotStatsCount)
      .def("TotCount",
        &PyClass::TotCount)
      .def("TotLogLike",
        &PyClass::TotLogLike)
      .def("GetAcc",
        py::overload_cast<int32>(&PyClass::GetAcc),
        py::arg("index"))
      .def("Add",
        &PyClass::Add,
        py::arg("scale"),
        py::arg("other"))
      .def("Scale",
        &PyClass::Scale,
        py::arg("scale"))
      .def("Dim",
        &PyClass::Dim);
  }
  m.def("MleAmDiagGmmUpdate",
        &MleAmDiagGmmUpdate,
        "for computing the maximum-likelihood estimates of the parameters of "
        "an acoustic model that uses diagonal Gaussian mixture models as emission densities.",
        py::arg("config"),
        py::arg("amdiaggmm_acc"),
        py::arg("flags"),
        py::arg("am_gmm"),
        py::arg("obj_change_out"),
        py::arg("count_out"));
  m.def("MapAmDiagGmmUpdate",
        &MapAmDiagGmmUpdate,
        "Maximum A Posteriori update.",
        py::arg("config"),
        py::arg("diag_gmm_acc"),
        py::arg("flags"),
        py::arg("gmm"),
        py::arg("obj_change_out"),
        py::arg("count_out"));
}

void pybind_mle_diag_gmm(py::module& m) {

  {
    using PyClass = MleDiagGmmOptions;

    auto mle_diag_gmm_options = py::class_<PyClass>(
        m, "MleDiagGmmOptions");
    mle_diag_gmm_options.def(py::init<>())
      .def_readwrite("variance_floor_vector", &PyClass::variance_floor_vector,
                      "Variance floor for each dimension [empty if not supplied]. "
              "It is in double since the variance is computed in double precision.")
      .def_readwrite("min_gaussian_weight", &PyClass::min_gaussian_weight,
                      "Minimum weight below which a Gaussian is not updated (and is "
                    "removed, if remove_low_count_gaussians == true);")
      .def_readwrite("min_gaussian_occupancy", &PyClass::min_gaussian_occupancy,
                      "Minimum count below which a Gaussian is not updated (and is "
                      "removed, if remove_low_count_gaussians == true).")
      .def_readwrite("min_variance", &PyClass::min_variance,
                      "Minimum allowed variance in any dimension (if no variance floor) "
                    "It is in double since the variance is computed in double precision.")
      .def_readwrite("remove_low_count_gaussians", &PyClass::remove_low_count_gaussians);
  }
  {
    using PyClass = MapDiagGmmOptions;

    auto map_diag_gmm_options = py::class_<PyClass>(
        m, "MapDiagGmmOptions");
    map_diag_gmm_options.def(py::init<>())
      .def_readwrite("mean_tau", &PyClass::mean_tau,
                      "Tau value for the means.")
      .def_readwrite("variance_tau", &PyClass::variance_tau,
                      "Tau value for the variances.  (Note: "
              "whether or not the variances are updated at all will "
              "be controlled by flags.)")
      .def_readwrite("weight_tau", &PyClass::weight_tau,
                      "Tau value for the weights-- this tau value is applied "
                      "per state, not per Gaussian.");
  }
  {
    using PyClass = AccumDiagGmm;

    auto accum_diag_gmm = py::class_<PyClass>(
        m, "AccumDiagGmm");
    accum_diag_gmm.def(py::init<>())
      .def(py::init<const DiagGmm &, GmmFlagsType>(),
        py::arg("gmm"),
        py::arg("flags"))
      .def(py::init<const AccumDiagGmm &>(),
        py::arg("other"))
      .def("Read", &PyClass::Read,
        py::arg("is"),
        py::arg("binary"),
        py::arg("add"))
      .def("Write", &PyClass::Write,
        py::arg("os"),
        py::arg("binary"))
      .def("Resize",
        py::overload_cast<int32, int32, GmmFlagsType>(&PyClass::Resize),
        "Allocates memory for accumulators",
        py::arg("num_gauss"),
        py::arg("dim"),
        py::arg("flags"))
      .def("Resize",
        py::overload_cast<const DiagGmm &, GmmFlagsType>(&PyClass::Resize),
        "Calls ResizeAccumulators with arguments based on gmm",
        py::arg("gmm"),
        py::arg("flags"))
      .def("NumGauss",
        &PyClass::NumGauss,
        "Returns the number of mixture components")
      .def("Dim",
        &PyClass::Dim,
        "Returns the dimensionality of the feature vectors")
      .def("SetZero",
        &PyClass::SetZero,
        py::arg("flags"))
      .def("Scale",
        &PyClass::Scale,
        py::arg("f"),
        py::arg("flags"))
      .def("AccumulateForComponent",
        &PyClass::AccumulateForComponent,
        "Accumulate for a single component, given the posterior",
        py::arg("data"),
        py::arg("comp_index"),
        py::arg("weight"))
      .def("AccumulateFromPosteriors",
        &PyClass::AccumulateFromPosteriors,
        "Accumulate for all components, given the posteriors.",
        py::arg("data"),
        py::arg("gauss_posteriors"))
      .def("AccumulateFromDiag",
        &PyClass::AccumulateFromDiag,
        "Accumulate for all components given a diagonal-covariance GMM. "
        "Computes posteriors and returns log-likelihood",
        py::arg("gmm"),
        py::arg("data"),
        py::arg("frame_posterior"))
      .def("AccumulateFromDiagMultiThreaded",
        &PyClass::AccumulateFromDiagMultiThreaded,
        "This does the same job as AccumulateFromDiag, but using "
        "multiple threads.  Returns sum of (log-likelihood times "
        "frame weight) over all frames.",
        py::arg("gmm"),
        py::arg("data"),
        py::arg("frame_weights"),
        py::arg("num_threads"))
      .def("AddStatsForComponent",
        &PyClass::AddStatsForComponent,
        "Increment the stats for this component by the specified amount "
        "(not all parts may be taken, depending on flags). "
        "Note: x_stats and x2_stats are assumed to already be multiplied by \"occ\"",
        py::arg("comp_id"),
        py::arg("occ"),
        py::arg("x_stats"),
        py::arg("x2_stats"))
      .def("Add",
        &PyClass::Add,
        "Increment with stats from this other accumulator (times scale)",
        py::arg("scale"),
        py::arg("acc"))
      .def("SmoothStats",
        &PyClass::SmoothStats,
        "Smooths the accumulated counts by adding 'tau' extra frames. An example "
        "use for this is I-smoothing for MMIE.   Calls SmoothWithAccum.",
        py::arg("tau"))
      .def("SmoothWithAccum",
        &PyClass::SmoothWithAccum,
        "Smooths the accumulated counts using some other accumulator. Performs a "
        "weighted sum of the current accumulator with the given one. An example use "
        "for this is I-smoothing for MMI and MPE. Both accumulators must have the "
        "same dimension and number of components.",
        py::arg("tau"),
        py::arg("src_acc"))
      .def("SmoothWithModel",
        &PyClass::SmoothWithModel,
        "Smooths the accumulated counts using the parameters of a given model. "
        "An example use of this is MAP-adaptation. The model must have the "
        "same dimension and number of components as the current accumulator.",
        py::arg("tau"),
        py::arg("src_gmm"))
      .def("Flags",
        &PyClass::Flags)
      .def("occupancy",
        &PyClass::occupancy)
      .def("mean_accumulator",
        &PyClass::mean_accumulator)
      .def("variance_accumulator",
        &PyClass::variance_accumulator)
      .def("AssertEqual",
        &PyClass::AssertEqual,
        "used in testing.");
  }
  m.def("AugmentGmmFlags",
        &AugmentGmmFlags,
        "Returns \"augmented\" version of flags: e.g. if just updating means, need "
        "weights too.",
        py::arg("f"));
  m.def("MleDiagGmmUpdate",
        &MleDiagGmmUpdate,
        "for computing the maximum-likelihood estimates of the parameters of "
        "a Gaussian mixture model. "
        "Update using the DiagGmm: exponential form.  Sets, does not increment, "
        "objf_change_out, floored_elements_out and floored_gauss_out.",
        py::arg("config"),
        py::arg("diag_gmm_acc"),
        py::arg("flags"),
        py::arg("gmm"),
        py::arg("obj_change_out"),
        py::arg("count_out"),
        py::arg("floored_elements_out") = NULL,
        py::arg("floored_gauss_out") = NULL,
        py::arg("removed_gauss_out") = NULL);
  m.def("MapDiagGmmUpdate",
        &MapDiagGmmUpdate,
        "Maximum A Posteriori estimation of the model.",
        py::arg("config"),
        py::arg("diag_gmm_acc"),
        py::arg("flags"),
        py::arg("gmm"),
        py::arg("obj_change_out"),
        py::arg("count_out"));
  m.def("MlObjective",
        py::overload_cast<const DiagGmm &,
                      const AccumDiagGmm &>(&MlObjective),
        "Calc using the DiagGMM exponential form",
        py::arg("gmm"),
        py::arg("diaggmm_acc"));
}

void pybind_mle_full_gmm(py::module& m) {

  {
    using PyClass = MleFullGmmOptions;

    auto mle_full_gmm_options = py::class_<PyClass>(
        m, "MleFullGmmOptions");
    mle_full_gmm_options.def(py::init<>())
      .def_readwrite("min_gaussian_weight", &PyClass::min_gaussian_weight,
                      "Minimum weight below which a Gaussian is removed")
      .def_readwrite("min_gaussian_occupancy", &PyClass::min_gaussian_occupancy,
                      "Minimum occupancy count below which a Gaussian is removed")
      .def_readwrite("variance_floor", &PyClass::variance_floor,
                      "Floor on eigenvalues of covariance matrices")
      .def_readwrite("max_condition", &PyClass::max_condition,
                      "Maximum condition number of covariance matrices (apply "
                      "floor to eigenvalues if they pass this).")
      .def_readwrite("remove_low_count_gaussians", &PyClass::remove_low_count_gaussians);
  }
  {
    using PyClass = AccumFullGmm;

    auto accum_full_gmm = py::class_<PyClass>(
        m, "AccumFullGmm");
    accum_full_gmm.def(py::init<>())
      .def(py::init<int32, int32, GmmFlagsType>(),
        py::arg("num_comp"),
        py::arg("dim"),
        py::arg("flags"))
      .def(py::init<const FullGmm &, GmmFlagsType>(),
        py::arg("gmm"),
        py::arg("flags"))
      .def(py::init<const AccumFullGmm &>(),
        py::arg("other"))
      .def("Read", &PyClass::Read,
        py::arg("is"),
        py::arg("binary"),
        py::arg("add"))
      .def("Write", &PyClass::Write,
        py::arg("os"),
        py::arg("binary"))
      .def("Resize",
        py::overload_cast<int32, int32, GmmFlagsType>(&PyClass::Resize),
        "Allocates memory for accumulators",
        py::arg("num_components"),
        py::arg("dim"),
        py::arg("flags"))
      .def("Resize",
        py::overload_cast<const FullGmm &, GmmFlagsType>(&PyClass::Resize),
        "Calls ResizeAccumulators with arguments based on gmm",
        py::arg("gmm"),
        py::arg("flags"))
      .def("ResizeVarAccumulator",
        &PyClass::ResizeVarAccumulator,
        py::arg("num_comp"),
        py::arg("dim"))
      .def("NumGauss",
        &PyClass::NumGauss,
        "Returns the number of mixture components")
      .def("Dim",
        &PyClass::Dim,
        "Returns the dimensionality of the feature vectors")
      .def("SetZero",
        &PyClass::SetZero,
        py::arg("flags"))
      .def("Scale",
        &PyClass::Scale,
        py::arg("f"),
        py::arg("flags"))
      .def("AccumulateForComponent",
        &PyClass::AccumulateForComponent,
        "Accumulate for a single component, given the posterior",
        py::arg("data"),
        py::arg("comp_index"),
        py::arg("weight"))
      .def("AccumulateFromPosteriors",
        &PyClass::AccumulateFromPosteriors,
        "Accumulate for all components, given the posteriors.",
        py::arg("data"),
        py::arg("gauss_posteriors"))
      .def("AccumulateFromFull",
        &PyClass::AccumulateFromFull,
        "Accumulate for all components given a full-covariance GMM. "
        "Computes posteriors and returns log-likelihood",
        py::arg("gmm"),
        py::arg("data"),
        py::arg("frame_posterior"))
      .def("AccumulateFromDiag",
        &PyClass::AccumulateFromDiag,
        "Accumulate for all components given a diagonal-covariance GMM. "
        "Computes posteriors and returns log-likelihood",
        py::arg("gmm"),
        py::arg("data"),
        py::arg("frame_posterior"))
      .def("Flags",
        &PyClass::Flags)
      .def("occupancy",
        &PyClass::occupancy)
      .def("mean_accumulator",
        &PyClass::mean_accumulator)
      .def("covariance_accumulator",
        &PyClass::covariance_accumulator);
  }
  m.def("MleFullGmmUpdate",
        &MleFullGmmUpdate,
        "for computing the maximum-likelihood estimates of the parameters of a "
        "Gaussian mixture model.  Update using the FullGmm exponential form",
        py::arg("config"),
        py::arg("fullgmm_acc"),
        py::arg("flags"),
        py::arg("gmm"),
        py::arg("obj_change_out"),
        py::arg("count_out"));
  m.def("MlObjective",
        py::overload_cast<const FullGmm &,
                      const AccumFullGmm &>(&MlObjective),
        "Calc using the DiagGMM exponential form",
        py::arg("gmm"),
        py::arg("fullgmm_acc"));
}

void pybind_model_common(py::module& m) {

  py::enum_<GmmUpdateFlags>(m, "GmmUpdateFlags")
    .value("kGmmMeans", GmmUpdateFlags::kGmmMeans)
    .value("kGmmVariances", GmmUpdateFlags::kGmmVariances)
    .value("kGmmWeights", GmmUpdateFlags::kGmmWeights)
    .value("kGmmTransitions", GmmUpdateFlags::kGmmTransitions)
    .value("kGmmAll", GmmUpdateFlags::kGmmAll)
    .export_values();
  m.def("AugmentGmmFlags",
        &AugmentGmmFlags,
        "Make sure that the flags make sense, i.e. if there is variance "
        "accumulation that there is also mean accumulation",
        py::arg("flags"));
  py::enum_<SgmmUpdateFlags>(m, "SgmmUpdateFlags")
    .value("kSgmmPhoneVectors", SgmmUpdateFlags::kSgmmPhoneVectors)
    .value("kSgmmPhoneProjections", SgmmUpdateFlags::kSgmmPhoneProjections)
    .value("kSgmmPhoneWeightProjections", SgmmUpdateFlags::kSgmmPhoneWeightProjections)
    .value("kSgmmCovarianceMatrix", SgmmUpdateFlags::kSgmmCovarianceMatrix)
    .value("kSgmmSubstateWeights", SgmmUpdateFlags::kSgmmSubstateWeights)
    .value("kSgmmSpeakerProjections", SgmmUpdateFlags::kSgmmSpeakerProjections)
    .value("kSgmmTransitions", SgmmUpdateFlags::kSgmmTransitions)
    .value("kSgmmSpeakerWeightProjections", SgmmUpdateFlags::kSgmmSpeakerWeightProjections)
    .value("kSgmmAll", SgmmUpdateFlags::kSgmmAll)
    .export_values();
  py::enum_<SgmmWriteFlags>(m, "SgmmWriteFlags")
    .value("kSgmmGlobalParams", SgmmWriteFlags::kSgmmGlobalParams)
    .value("kSgmmStateParams", SgmmWriteFlags::kSgmmStateParams)
    .value("kSgmmNormalizers", SgmmWriteFlags::kSgmmNormalizers)
    .value("kSgmmBackgroundGmms", SgmmWriteFlags::kSgmmBackgroundGmms)
    .value("kSgmmWriteAll", SgmmWriteFlags::kSgmmWriteAll)
    .export_values();
  m.def("GetSplitTargets",
        &GetSplitTargets,
        "Get Gaussian-mixture or substate-mixture splitting targets, "
        "according to a power rule (e.g. typically power = 0.2). "
        "Returns targets for number of mixture components (Gaussians, "
        "or sub-states), allocating the Gaussians or whatever according "
        "to a power of occupancy in order to acheive the total supplied "
        "\"target\".  During splitting we ensure that "
        "each Gaussian [or sub-state] would get a count of at least "
        "\"min-count\", assuming counts were evenly distributed between "
        "Gaussians in a state. "
        "The vector \"targets\" will be resized to the appropriate dimension; "
        "its value at input is ignored.",
        py::arg("state_occs"),
        py::arg("target_components"),
        py::arg("power"),
        py::arg("min_count"),
        py::arg("targets"));
}

void init_gmm(py::module &_m) {
  py::module m = _m.def_submodule("gmm", "gmm pybind for Kaldi");

  pybind_model_common(m);
  pybind_am_diag_gmm(m);
  pybind_decodable_am_diag_gmm(m);
  pybind_diag_gmm_normal(m);
  pybind_diag_gmm(m);
  pybind_ebw_diag_gmm(m);
  pybind_full_gmm_normal(m);
  pybind_full_gmm(m);
  pybind_indirect_diff_diag_gmm(m);
  pybind_mle_am_diag_gmm(m);
  pybind_mle_diag_gmm(m);
  pybind_mle_full_gmm(m);
}
