
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
#include "hmm/transition-model.h"
#include "hmm/posterior.h"
#include "hmm/hmm-utils.h"
#include "lat/kaldi-lattice.h"
#include "lat/lattice-functions.h"
#include "tree/context-dep.h"
#include "tree/build-tree-questions.h"
#include "tree/build-tree-utils.h"
#include "tree/context-dep.h"
#include "tree/clusterable-classes.h"
#include "fstext/fstext-lib.h"
#include "decoder/training-graph-compiler.h"
#include "decoder/decoder-wrappers.h"
#include "decoder/faster-decoder.h"

using namespace kaldi;
using namespace fst;

/// Get state occupation counts.
void GetOccs(const BuildTreeStatsType &stats,
             const EventMap &to_pdf_map,
             Vector<BaseFloat> *occs) {

    // Get stats split by tree-leaf ( == pdf):
  std::vector<BuildTreeStatsType> split_stats;
  SplitStatsByMap(stats, to_pdf_map, &split_stats);
  if (split_stats.size() != to_pdf_map.MaxResult()+1) {
    KALDI_ASSERT(split_stats.size() < to_pdf_map.MaxResult()+1);
    split_stats.resize(to_pdf_map.MaxResult()+1);
  }
  occs->Resize(split_stats.size());
  for (int32 pdf = 0; pdf < occs->Dim(); pdf++)
    (*occs)(pdf) = SumNormalizer(split_stats[pdf]);
}
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
      .def("transform_means",
      [](
        PyClass &am_gmm,
        const Matrix<BaseFloat> &mat
      ){
          py::gil_scoped_release gil_release;

        int32 dim = am_gmm.Dim();
        if (mat.NumRows() != dim)
          KALDI_ERR << "Transform matrix has " << mat.NumRows() << " rows but "
              "model has dimension " << am_gmm.Dim();
        if (mat.NumCols() != dim
          && mat.NumCols()  != dim+1)
          KALDI_ERR << "Transform matrix has " << mat.NumCols() << " columns but "
              "model has dimension " << am_gmm.Dim() << " (neither a linear nor an "
              "affine transform";

        for (int32 i = 0; i < am_gmm.NumPdfs(); i++) {
          DiagGmm &gmm = am_gmm.GetPdf(i);

          Matrix<BaseFloat> means;
          gmm.GetMeans(&means);
          Matrix<BaseFloat> new_means(means.NumRows(), means.NumCols());
          if (mat.NumCols() == dim) {  // linear case
            // Right-multiply means by mat^T (equivalent to left-multiplying each
            // row by mat).
            new_means.AddMatMat(1.0, means, kNoTrans, mat, kTrans, 0.0);
          } else { // affine case
            Matrix<BaseFloat> means_ext(means.NumRows(), means.NumCols()+1);
            means_ext.Set(1.0);  // set all elems to 1.0
            SubMatrix<BaseFloat> means_part(means_ext, 0, means.NumRows(),
                                            0, means.NumCols());
            means_part.CopyFromMat(means);  // copy old part...
            new_means.AddMatMat(1.0, means_ext, kNoTrans, mat, kTrans, 0.0);
          }
          gmm.SetMeans(new_means);
          gmm.ComputeGconsts();
        }
      },
        py::arg("mat"))
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
      .def("Read", &PyClass::Read, py::arg("in_stream"), py::arg("binary"),
      py::call_guard<py::gil_scoped_release>())
      .def("Write", &PyClass::Write, py::arg("out_stream"), py::arg("binary"),
      py::call_guard<py::gil_scoped_release>())
      .def("Dim", &PyClass::Dim)
      .def("NumPdfs", &PyClass::NumPdfs)
      .def("NumGauss", &PyClass::NumGauss)
      .def("NumGaussInPdf", &PyClass::NumGaussInPdf, py::arg("pdf_index"))
      .def("GetPdf",
          (const DiagGmm& (PyClass::*)(int32 pdf_index)const)&PyClass::GetPdf,
          py::arg("pdf_index"))
      .def("GetPdf",
          (DiagGmm& (PyClass::*)(int32 pdf_index))&PyClass::GetPdf,
          py::arg("pdf_index"))
      .def("GetGaussianMean", &PyClass::GetGaussianMean,
          py::arg("pdf_index"), py::arg("gauss"), py::arg("out"))
      .def("GetGaussianVariance", &PyClass::GetGaussianVariance,
          py::arg("pdf_index"), py::arg("gauss"), py::arg("out"))
      .def("SetGaussianMean", &PyClass::SetGaussianMean,
          py::arg("pdf_index"), py::arg("gauss_index"), py::arg("in"))
      .def("boost_silence",
          [](PyClass& am_gmm, const TransitionModel &trans_model,
           std::vector<int32> silence_phones, BaseFloat boost){
          py::gil_scoped_release gil_release;

          std::vector<int32> pdfs;
          bool ans = GetPdfsForPhones(trans_model, silence_phones, &pdfs);
          for (size_t i = 0; i < pdfs.size(); i++) {
            int32 pdf = pdfs[i];
            DiagGmm &gmm = am_gmm.GetPdf(pdf);
            Vector<BaseFloat> weights(gmm.weights());
            weights.Scale(boost);
            gmm.SetWeights(weights);
            gmm.ComputeGconsts();
          }
          },
          py::arg("trans_model"),
          py::arg("silence_phones"),
          py::arg("boost"))
      .def("mle_update",
          [](PyClass& am_gmm, const AccumAmDiagGmm &gmm_accs,
          int32 mixup = 0,
          int32 mixdown = 0,
          BaseFloat perturb_factor = 0.01,
          BaseFloat power = 0.2,
          BaseFloat min_count = 20.0,
          BaseFloat min_gaussian_weight = 1.0e-05,
          BaseFloat min_gaussian_occupancy = 10.0,
          BaseFloat min_variance= 0.001,
          BaseFloat remove_low_count_gaussians = true,
          std::string update_flags_str = "mvwt"
          ){
        kaldi::GmmFlagsType update_flags =
            StringToGmmFlags(update_flags_str);

        MleDiagGmmOptions gmm_opts;
        gmm_opts.min_gaussian_weight = min_gaussian_weight;
        gmm_opts.min_gaussian_occupancy = min_gaussian_occupancy;
        gmm_opts.min_variance = min_variance;
        gmm_opts.remove_low_count_gaussians = remove_low_count_gaussians;
      BaseFloat objf_impr, count;
      MleAmDiagGmmUpdate(gmm_opts, gmm_accs, update_flags, &am_gmm,
                         &objf_impr, &count);

        if (mixup != 0 || mixdown != 0) {
          // get pdf occupation counts
          Vector<BaseFloat> pdf_occs;
          pdf_occs.Resize(gmm_accs.NumAccs());
          for (int i = 0; i < gmm_accs.NumAccs(); i++)
            pdf_occs(i) = gmm_accs.GetAcc(i).occupancy().Sum();

          if (mixdown != 0)
            am_gmm.MergeByCount(pdf_occs, mixdown, power, min_count);

          if (mixup != 0)
            am_gmm.SplitByCount(pdf_occs, mixup, perturb_factor,
                                power, min_count);
        }
        return py::make_tuple(objf_impr, count);
          },
          py::arg("gmm_accs"),
          py::arg("mixup") = 0,
          py::arg("mixdown") = 0,
          py::arg("perturb_factor") = 0.01,
          py::arg("power") = 0.2,
          py::arg("min_count") = 20.0,
          py::arg("min_gaussian_weight") = 1.0e-05,
          py::arg("min_gaussian_occupancy") = 10.0,
          py::arg("min_variance")= 0.001,
          py::arg("remove_low_count_gaussians") = true,
          py::arg("update_flags_str") = "mvwt")
      .def(py::pickle(
        [](const PyClass &p) { // __getstate__
            /* Return a tuple that fully encodes the state of the object */
             std::ostringstream os;
             bool binary = true;
             p.Write(os, binary);
            return py::make_tuple(
                    py::bytes(os.str()));
        },
        [](py::tuple t) { // __setstate__
            if (t.size() != 1)
                throw std::runtime_error("Invalid state!");

            /* Create a new C++ instance */
            PyClass *p = new PyClass();

            /* Assign any additional state */
            std::istringstream str(t[0].cast<std::string>());
               p->Read(str, true);

            return p;
        }
    ));
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
        &PyClass::NumFramesReady,
          py::return_value_policy::reference)
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
          py::arg("log_sum_exp_prune") = -1.0)
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
      .def("mle_update",
          [](PyClass& gmm, const AccumDiagGmm &gmm_accs,
          int32 mixup = 0,
          BaseFloat perturb_factor = 0.01,
          BaseFloat min_gaussian_weight = 1.0e-05,
          BaseFloat min_gaussian_occupancy = 10.0,
          BaseFloat min_variance= 0.001,
          BaseFloat remove_low_count_gaussians = true,
          std::string update_flags_str = "mvw"
          ){
          py::gil_scoped_release gil_release;

        MleDiagGmmOptions gmm_opts;
        gmm_opts.min_gaussian_weight = min_gaussian_weight;
        gmm_opts.min_gaussian_occupancy = min_gaussian_occupancy;
        gmm_opts.min_variance = min_variance;
        gmm_opts.remove_low_count_gaussians = remove_low_count_gaussians;
        {  // Update GMMs.
          BaseFloat objf_impr, count;
          MleDiagGmmUpdate(gmm_opts, gmm_accs,
                          StringToGmmFlags(update_flags_str),
                          &gmm, &objf_impr, &count);
        }

        if (mixup != 0)
          gmm.Split(mixup, perturb_factor);
          },
          py::arg("gmm_accs"),
          py::arg("mixup") = 0,
          py::arg("perturb_factor") = 0.01,
          py::arg("min_gaussian_weight") = 1.0e-05,
          py::arg("min_gaussian_occupancy") = 10.0,
          py::arg("min_variance")= 0.001,
          py::arg("remove_low_count_gaussians") = true,
          py::arg("update_flags_str") = "mvw")
      .def("generate_post",
        [](
          PyClass&gmm,
          const MatrixBase<BaseFloat> &feats,
          int32 num_post = 50,
          BaseFloat min_post = 0.0
        ){
          py::gil_scoped_release release;
      int32 T = feats.NumRows();
      Matrix<BaseFloat> loglikes;
      gmm.LogLikelihoods(feats, &loglikes);

      Posterior post(T);
      double log_like_this_file = 0.0;
      for (int32 t = 0; t < T; t++) {
        log_like_this_file +=
            VectorToPosteriorEntry(loglikes.Row(t), num_post,
                                   min_post, &(post[t]));
      }
            py::gil_scoped_acquire acquire;
        return py::make_tuple(post, log_like_this_file);
        },
          py::arg("feats"),
          py::arg("num_post") = 50,
          py::arg("min_post") = 0.0)
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
      .def("gaussian_selection",
        [](
          PyClass &gmm,
          const MatrixBase<BaseFloat> &data,
                                     int32 num_gselect
        ){

          py::gil_scoped_release release;
      std::vector<std::vector<int32> > gselect(data.NumRows());
        double tot_like_this_file = gmm.GaussianSelection(data, num_gselect, &gselect);
            py::gil_scoped_acquire acquire;
            return py::make_tuple(gselect, tot_like_this_file);
        },
        "Get gaussian selection information for one frame.  Returns log-like "
        "this frame.  Output is the best \"num_gselect\" indices, sorted from best to "
        "worst likelihood.  If \"num_gselect\" > NumGauss(), sets it to NumGauss().",
          py::arg("data"),
          py::arg("num_gselect"))
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
      .def("MergeKmeans",
        &PyClass::MergeKmeans,
        "Merge the components to a specified target #components: this "
        "version uses a different approach based on K-means.",
          py::arg("target_components"),
          py::arg("cfg") = ClusterKMeansOptions())
      .def("Write",
        &PyClass::Write,
          py::arg("os"),
          py::arg("binary"),
      py::call_guard<py::gil_scoped_release>())
      .def("Read",
        &PyClass::Read,
          py::arg("is"),
          py::arg("binary"),
      py::call_guard<py::gil_scoped_release>())
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
          py::arg("out"))
      .def("init_from_random_frames",
        [](
          PyClass* gmm,
          const Matrix<BaseFloat> &feats
        ){
          int32 num_gauss = gmm->NumGauss(), num_frames = feats.NumRows(),
              dim = feats.NumCols();
          KALDI_ASSERT(num_frames >= 10 * num_gauss && "Too few frames to train on");
          Vector<double> mean(dim), var(dim);
          for (int32 i = 0; i < num_frames; i++) {
            mean.AddVec(1.0 / num_frames, feats.Row(i));
            var.AddVec2(1.0 / num_frames, feats.Row(i));
          }
          var.AddVec2(-1.0, mean);
          if (var.Max() <= 0.0)
            KALDI_ERR << "Features do not have positive variance " << var;

          DiagGmmNormal gmm_normal(*gmm);

          std::set<int32> used_frames;
          for (int32 g = 0; g < num_gauss; g++) {
            int32 random_frame = RandInt(0, num_frames - 1);
            while (used_frames.count(random_frame) != 0)
              random_frame = RandInt(0, num_frames - 1);
            used_frames.insert(random_frame);
            gmm_normal.weights_(g) = 1.0 / num_gauss;
            gmm_normal.means_.Row(g).CopyFromVec(feats.Row(random_frame));
            gmm_normal.vars_.Row(g).CopyFromVec(var);
          }
          gmm->CopyFromNormal(gmm_normal);
          gmm->ComputeGconsts();
        },
        "We initialize the GMM parameters by setting the variance to the global "
"variance of the features, and the means to distinct randomly chosen frames.",
          py::arg("feats"))
      .def("train_one_iter",
        [](
          PyClass* gmm,
          const Matrix<BaseFloat> &feats,
          const MleDiagGmmOptions &gmm_opts,
          int32 iter,
          int32 num_threads
        ){
        AccumDiagGmm gmm_acc(*gmm, kGmmAll);

        Vector<BaseFloat> frame_weights(feats.NumRows(), kUndefined);
        frame_weights.Set(1.0);

        double tot_like;
        tot_like = gmm_acc.AccumulateFromDiagMultiThreaded(*gmm, feats, frame_weights,
                                                          num_threads);

        KALDI_LOG << "Likelihood per frame on iteration " << iter
                  << " was " << (tot_like / feats.NumRows()) << " over "
                  << feats.NumRows() << " frames.";

        BaseFloat objf_change, count;
        MleDiagGmmUpdate(gmm_opts, gmm_acc, kGmmAll, gmm, &objf_change, &count);

        KALDI_LOG << "Objective-function change on iteration " << iter << " was "
                  << (objf_change / count) << " over " << count << " frames.";
        },
        "We initialize the GMM parameters by setting the variance to the global "
"variance of the features, and the means to distinct randomly chosen frames.",
          py::arg("feats"),
          py::arg("gmm_opts"),
          py::arg("iter"),
          py::arg("num_threads"));
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
          py::arg("binary"),
      py::call_guard<py::gil_scoped_release>())
      .def("Read",
        &PyClass::Read,
          py::arg("is"),
          py::arg("binary"),
      py::call_guard<py::gil_scoped_release>())
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
        py::arg("add") = false,
      py::call_guard<py::gil_scoped_release>())
      .def("Write",
        &PyClass::Write,
        py::arg("out_stream"),
        py::arg("binary"),
      py::call_guard<py::gil_scoped_release>())
      .def("Init",
        py::overload_cast<const AmDiagGmm &, GmmFlagsType>(&PyClass::Init),
        "Initializes accumulators for each GMM based on the number of components "
        "and dimension.",
        py::arg("model"),
        py::arg("flags"))
      .def("init",
        [](
          PyClass &gmm_accs,
          const AmDiagGmm &am_gmm
        ){
        gmm_accs.Init(am_gmm, kGmmAll);

        },
        "Initializes accumulators for each GMM based on the number of components "
        "and dimension.",
        py::arg("am_gmm"))
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
        py::arg("weight"),
      py::call_guard<py::gil_scoped_release>())
      .def("AccumulateForGmmTwofeats",
        &PyClass::AccumulateForGmmTwofeats,
        "Accumulate stats for a single GMM in the model; uses data1 for "
        "getting posteriors and data2 for stats. Returns log likelihood.",
        py::arg("model"),
        py::arg("data1"),
        py::arg("data2"),
        py::arg("gmm_index"),
        py::arg("weight"),
      py::call_guard<py::gil_scoped_release>())
      .def("AccumulateFromPosteriors",
        &PyClass::AccumulateFromPosteriors,
        "Accumulates stats for a single GMM in the model using pre-computed "
        "Gaussian posteriors.",
        py::arg("model"),
        py::arg("data"),
        py::arg("gmm_index"),
        py::arg("posteriors"),
      py::call_guard<py::gil_scoped_release>())
      .def("AccumulateForGaussian",
        &PyClass::AccumulateForGaussian,
        "Accumulate stats for a single Gaussian component in the model.",
        py::arg("am"),
        py::arg("data"),
        py::arg("gmm_index"),
        py::arg("gauss_index"),
        py::arg("weight"),
      py::call_guard<py::gil_scoped_release>())
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
        py::arg("other"),
      py::call_guard<py::gil_scoped_release>())
      .def("Scale",
        &PyClass::Scale,
        py::arg("scale"),
      py::call_guard<py::gil_scoped_release>())
      .def("Dim",
        &PyClass::Dim)
        .def("acc_stats",
                  [](PyClass &gmm_accs,
                  const AmDiagGmm &am_gmm,
                  const TransitionModel &trans_model,
                  const std::vector<int32> &alignment,
                                  const Matrix<BaseFloat> &mat){

          py::gil_scoped_release gil_release;
                BaseFloat tot_like_this_file = 0.0;
                for (size_t i = 0; i < alignment.size(); i++) {
                  int32 tid = alignment[i],  // transition identifier.
                  pdf_id = trans_model.TransitionIdToPdf(tid);
                  tot_like_this_file += gmm_accs.AccumulateForGmm(am_gmm, mat.Row(i),
                                                              pdf_id, 1.0);
                }
                return tot_like_this_file;
                  },
              py::arg("am_gmm"),
              py::arg("trans_model"),
              py::arg("alignment"),
              py::arg("mat")
              )
        .def("acc_twofeats",
                  [](PyClass &gmm_accs,
                  const AmDiagGmm &am_gmm,
                                  const Posterior &pdf_posterior,
                                  const Matrix<BaseFloat> &mat1,
                                  const Matrix<BaseFloat> &mat2){

          py::gil_scoped_release gil_release;
        BaseFloat tot_like_this_file = 0.0,
            tot_weight_this_file = 0.0;
              for (size_t i = 0; i < pdf_posterior.size(); i++) {
                // Accumulates for GMM.
                for (size_t j = 0; j <pdf_posterior[i].size(); j++) {
                  int32 pdf_id = pdf_posterior[i][j].first;
                  BaseFloat weight = pdf_posterior[i][j].second;
                  tot_like_this_file += weight *
                      gmm_accs.AccumulateForGmmTwofeats(am_gmm,
                                                        mat1.Row(i),
                                                        mat2.Row(i),
                                                        pdf_id,
                                                        weight);
                  tot_weight_this_file += weight;
                }
                }
                return tot_like_this_file;
                  },
              py::arg("am_gmm"),
              py::arg("pdf_posterior"),
              py::arg("mat1"),
              py::arg("mat2")
              )
      .def(py::pickle(
        [](const PyClass &p) { // __getstate__
            /* Return a tuple that fully encodes the state of the object */
             std::ostringstream os;
             bool binary = true;
             p.Write(os, binary);
            return py::make_tuple(
                    py::bytes(os.str()));
        },
        [](py::tuple t) { // __setstate__
            if (t.size() != 1)
                throw std::runtime_error("Invalid state!");

            /* Create a new C++ instance */
            PyClass *p = new PyClass();

            /* Assign any additional state */
            std::istringstream str(t[0].cast<std::string>());
               p->Read(str, true, false);

            return p;
        }
    ));
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
        py::arg("count_out"),
      py::call_guard<py::gil_scoped_release>());
  m.def("MapAmDiagGmmUpdate",
        &MapAmDiagGmmUpdate,
        "Maximum A Posteriori update.",
        py::arg("config"),
        py::arg("diag_gmm_acc"),
        py::arg("flags"),
        py::arg("gmm"),
        py::arg("obj_change_out"),
        py::arg("count_out"),
      py::call_guard<py::gil_scoped_release>());
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
        py::arg("add") = false,
      py::call_guard<py::gil_scoped_release>())
      .def("Write", &PyClass::Write,
        py::arg("os"),
        py::arg("binary"),
      py::call_guard<py::gil_scoped_release>())
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
        py::arg("acc"),
      py::call_guard<py::gil_scoped_release>())
      .def("SmoothStats",
        &PyClass::SmoothStats,
        "Smooths the accumulated counts by adding 'tau' extra frames. An example "
        "use for this is I-smoothing for MMIE.   Calls SmoothWithAccum.",
        py::arg("tau"),
      py::call_guard<py::gil_scoped_release>())
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
        "used in testing.")
      .def("accumulate_from_gselect",
        [](
            PyClass &gmm_accs,
            const DiagGmm &gmm,
            const std::vector<std::vector<int32> > &gselect,
            const Matrix<BaseFloat> &mat
        ){
          py::gil_scoped_release gil_release;

          int32 file_frames = mat.NumRows();
          BaseFloat file_like = 0.0;
          for (int32 i = 0; i < file_frames; i++) {
            SubVector<BaseFloat> data(mat, i);
            const std::vector<int32> &this_gselect = gselect[i];
            int32 gselect_size = this_gselect.size();
            Vector<BaseFloat> loglikes;
            gmm.LogLikelihoodsPreselect(data, this_gselect, &loglikes);
            file_like += loglikes.ApplySoftMax();
            for (int32 j = 0; j < loglikes.Dim(); j++)
              gmm_accs.AccumulateForComponent(data, this_gselect[j], loglikes(j));
          }
          return file_like;
        },
        py::arg("gmm"),
        py::arg("gselect"),
        py::arg("features"))
      .def(py::pickle(
        [](const PyClass &p) { // __getstate__
            /* Return a tuple that fully encodes the state of the object */
             std::ostringstream os;
             bool binary = true;
             p.Write(os, binary);
            return py::make_tuple(
                    py::bytes(os.str()));
        },
        [](py::tuple t) { // __setstate__
            if (t.size() != 1)
                throw std::runtime_error("Invalid state!");

            /* Create a new C++ instance */
            PyClass *p = new PyClass();

            /* Assign any additional state */
            std::istringstream str(t[0].cast<std::string>());
               p->Read(str, true, false);

            return p;
        }
    ));
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
        py::arg("add"),
      py::call_guard<py::gil_scoped_release>())
      .def("Write", &PyClass::Write,
        py::arg("os"),
        py::arg("binary"),
      py::call_guard<py::gil_scoped_release>())
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
  m.def("StringToGmmFlags",
        &StringToGmmFlags,
        py::arg("string"));
  m.def("GmmFlagsToString",
        &GmmFlagsToString,
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

  m.def("gmm_post_to_gpost",
          [](const TransitionModel &trans_model,
                    const AmDiagGmm &am_gmm,
                            const Posterior &posterior,
                            const Matrix<BaseFloat> &mat,
                            BaseFloat rand_prune = 0.0){

          py::gil_scoped_release release;
              GaussPost gpost(posterior.size());
              BaseFloat tot_like_this_file = 0.0, tot_weight = 0.0;
              Posterior pdf_posterior;
              ConvertPosteriorToPdfs(trans_model, posterior, &pdf_posterior);
              for (size_t i = 0; i < posterior.size(); i++) {
                  gpost[i].reserve(pdf_posterior[i].size());
                  for (size_t j = 0; j < pdf_posterior[i].size(); j++) {
                  int32 pdf_id = pdf_posterior[i][j].first;
                  BaseFloat weight = pdf_posterior[i][j].second;
                  const DiagGmm &gmm = am_gmm.GetPdf(pdf_id);
                  Vector<BaseFloat> this_post_vec;
                  BaseFloat like =
                        gmm.ComponentPosteriors(mat.Row(i), &this_post_vec);
                  this_post_vec.Scale(weight);
                  if (rand_prune > 0.0)
                  for (int32 k = 0; k < this_post_vec.Dim(); k++)
                        this_post_vec(k) = RandPrune(this_post_vec(k),
                                                      rand_prune);
                  if (!this_post_vec.IsZero())
                  gpost[i].push_back(std::make_pair(pdf_id, this_post_vec));
                  tot_like_this_file += like * weight;
                  tot_weight += weight;
                  }
              }
            py::gil_scoped_acquire acquire;
              return py::make_tuple(gpost, tot_like_this_file, tot_weight);
          },
        py::arg("trans_model"),
        py::arg("am_gmm"),
        py::arg("posterior"),
        py::arg("mat"),
        py::arg("rand_prune") = 0.0
        );

  m.def("gmm_rescore_lattice",
          [](CompactLattice &clat,
          const Matrix<BaseFloat> &feats,
          const AmDiagGmm &am_gmm,
          const TransitionModel &trans_model){

          py::gil_scoped_release release;
    kaldi::BaseFloat old_acoustic_scale = 0.0;
      fst::ScaleLattice(fst::AcousticLatticeScale(old_acoustic_scale), &clat);

      DecodableAmDiagGmm gmm_decodable(am_gmm, trans_model, feats);
      bool ans = kaldi::RescoreCompactLattice(&gmm_decodable, &clat);
            py::gil_scoped_acquire acquire;
        return py::make_tuple(ans, clat);
          },
        py::arg("clat"),
        py::arg("feats"),
        py::arg("am_gmm"),
        py::arg("trans_model")
        );

  m.def("gmm_align_equal",
          [](
      const VectorFst<StdArc> &decode_fst,
      const Matrix<BaseFloat> &features){
          py::gil_scoped_release release;
        VectorFst<StdArc> path;
        int32 rand_seed = 1234;
        std::vector<int32> aligned_seq, words;

        if (EqualAlign(decode_fst, features.NumRows(), rand_seed, &path) ) {
          StdArc::Weight w;
          GetLinearSymbolSequence(path, &aligned_seq, &words, &w);
          KALDI_ASSERT(aligned_seq.size() == features.NumRows());
        }
            py::gil_scoped_acquire acquire;
          return py::make_tuple(aligned_seq, words);
          },
        py::arg("decode_fst"),
        py::arg("features")
        );

  m.def("gmm_align_compiled",
          [](
      const TransitionModel  &trans_model,
      const AmDiagGmm  &am_gmm,
      VectorFst<StdArc> *decode_fst,
      const Matrix<BaseFloat> &features,

    BaseFloat acoustic_scale = 1.0,
    BaseFloat transition_scale = 1.0,
    BaseFloat self_loop_scale = 1.0,
  BaseFloat beam=10.0,
  BaseFloat retry_beam=40.0,
  bool careful=false

      ){
          py::gil_scoped_release gil_release;

        {  // Add transition-probs to the FST.
          std::vector<int32> disambig_syms;  // empty.
          AddTransitionProbs(trans_model, disambig_syms,
                             transition_scale, self_loop_scale,
                             decode_fst);
        }

        std::vector<int32> alignment;
        std::vector<int32> words;
        LatticeWeight weight;
        BaseFloat like = 0.0;
        Vector<BaseFloat> per_frame_loglikes;
        DecodableAmDiagGmmScaled decodable(am_gmm, trans_model, features,
                                               acoustic_scale);

        if (careful)
          ModifyGraphForCarefulAlignment(decode_fst);

        FasterDecoderOptions decode_opts;
        decode_opts.beam = beam;

        FasterDecoder decoder(*decode_fst, decode_opts);
        decoder.Decode(&decodable);

        bool ans = decoder.ReachedFinal();  // consider only final states.
        bool retried = false;
        if (!ans && retry_beam != 0.0) {
          decode_opts.beam = retry_beam;
          decoder.SetOptions(decode_opts);
          decoder.Decode(&decodable);
          ans = decoder.ReachedFinal();
          retried = true;
        }

        if (!ans) {  // Still did not reach final state.
          py::gil_scoped_acquire gil_acquire;
        return py::make_tuple(alignment, words, like, per_frame_loglikes, ans, retried);
        }

        fst::VectorFst<LatticeArc> decoded;  // linear FST.
        decoder.GetBestPath(&decoded);
        if (decoded.NumStates() == 0) {
          py::gil_scoped_acquire gil_acquire;
          return py::make_tuple(alignment, words, like, per_frame_loglikes, ans, retried);
        }

        GetLinearSymbolSequence(decoded, &alignment, &words, &weight);
        like = -(weight.Value1()+weight.Value2()) / acoustic_scale;
        GetPerFrameAcousticCosts(decoded, &per_frame_loglikes);
        per_frame_loglikes.Scale(-1 / acoustic_scale);
          py::gil_scoped_acquire gil_acquire;
        return py::make_tuple(alignment, words, like, per_frame_loglikes, ans, retried);
          },
        py::arg("trans_model"),
        py::arg("am_gmm"),
        py::arg("decode_fst"),
        py::arg("features"),
        py::arg("acoustic_scale") = 1.0,
        py::arg("transition_scale") = 1.0,
        py::arg("self_loop_scale") = 1.0,
        py::arg("beam") = 10.0,
        py::arg("retry_beam") = 40.0,
        py::arg("careful") = false
        );

  m.def("gmm_init_mono",
          [](
            const HmmTopology &topo,
      const std::vector<Matrix<BaseFloat> > &feats,
      const std::vector<std::vector<int32> > &shared_phones,
    const std::string model_filename,
    const std::string tree_filename,
      BaseFloat perturb_factor = 0.0){

    int32 dim = feats[0].NumCols();
    Vector<BaseFloat> glob_inv_var(dim);
    glob_inv_var.Set(1.0);
    Vector<BaseFloat> glob_mean(dim);
    glob_mean.Set(1.0);
      double count = 0.0;
      Vector<double> var_stats(dim);
      Vector<double> mean_stats(dim);
        for (int32 j = 0; j < feats.size(); j++){

        for (int32 i = 0; i < feats[j].NumRows(); i++) {
          count += 1.0;
          var_stats.AddVec2(1.0, feats[j].Row(i));
          mean_stats.AddVec(1.0, feats[j].Row(i));
        }
        }
      if (count == 0) { KALDI_ERR << "no features were seen."; }
      var_stats.Scale(1.0/count);
      mean_stats.Scale(1.0/count);
      var_stats.AddVec2(-1.0, mean_stats);
      if (var_stats.Min() <= 0.0)
        KALDI_ERR << "bad variance";
      var_stats.InvertElements();
      glob_inv_var.CopyFromVec(var_stats);
      glob_mean.CopyFromVec(mean_stats);

    const std::vector<int32> &phones = topo.GetPhones();

    std::vector<int32> phone2num_pdf_classes (1+phones.back());
    for (size_t i = 0; i < phones.size(); i++)
      phone2num_pdf_classes[phones[i]] = topo.NumPdfClasses(phones[i]);

    // Now the tree [not really a tree at this point]:
    ContextDependency *ctx_dep = NULL;
      ctx_dep = MonophoneContextDependencyShared(shared_phones, phone2num_pdf_classes);


    int32 num_pdfs = ctx_dep->NumPdfs();

    AmDiagGmm am_gmm;
    DiagGmm gmm;
    gmm.Resize(1, dim);
    {  // Initialize the gmm.
      Matrix<BaseFloat> inv_var(1, dim);
      inv_var.Row(0).CopyFromVec(glob_inv_var);
      Matrix<BaseFloat> mu(1, dim);
      mu.Row(0).CopyFromVec(glob_mean);
      Vector<BaseFloat> weights(1);
      weights.Set(1.0);
      gmm.SetInvVarsAndMeans(inv_var, mu);
      gmm.SetWeights(weights);
      gmm.ComputeGconsts();
    }

    for (int i = 0; i < num_pdfs; i++)
      am_gmm.AddPdf(gmm);

    if (perturb_factor != 0.0) {
      for (int i = 0; i < num_pdfs; i++)
        am_gmm.GetPdf(i).Perturb(perturb_factor);
    }

    // Now the transition model:
    TransitionModel trans_model(*ctx_dep, topo);
    bool binary = true;
    {
      Output ko(model_filename, binary);
      trans_model.Write(ko.Stream(), binary);
      am_gmm.Write(ko.Stream(), binary);
    }

    // Now write the tree.
    ctx_dep->Write(Output(tree_filename, binary).Stream(),
                   binary);

    delete ctx_dep;
          },
        py::arg("topo"),
        py::arg("feats"),
        py::arg("shared_phones"),
        py::arg("model_filename"),
        py::arg("tree_filename"),
        py::arg("perturb_factor") = 0.0
        );

  m.def("gmm_init_model",
          [](
            const HmmTopology &topo,
            const ContextDependency &ctx_dep,
    const BuildTreeStatsType &stats,
    const std::string model_filename,
    double var_floor = 0.01,
    int32 mixup = 0,
    int32 mixdown = 0,
    BaseFloat perturb_factor = 0.01,
    BaseFloat power = 0.2,
    BaseFloat min_count = 20.0){

    bool binary = true;
    const EventMap &to_pdf_map = ctx_dep.ToPdfMap();  // not owned here.

    TransitionModel trans_model(ctx_dep, topo);

    // Now, the summed_stats will be used to initialize the GMM.
    AmDiagGmm am_gmm;
  // Get stats split by tree-leaf ( == pdf):
  std::vector<BuildTreeStatsType> split_stats;
  SplitStatsByMap(stats, to_pdf_map, &split_stats);

  split_stats.resize(to_pdf_map.MaxResult() + 1); // ensure that
  // if the last leaf had no stats, this vector still has the right size.

  // Make sure each leaf has stats.
  for (size_t i = 0; i < split_stats.size(); i++) {
    if (split_stats[i].empty()) {
      std::vector<int32> bad_pdfs(1, i), bad_phones;
      GetPhonesForPdfs(trans_model, bad_pdfs, &bad_phones);
      std::ostringstream ss;
      for (int32 idx = 0; idx < bad_phones.size(); idx ++)
        ss << bad_phones[idx] << ' ';
      KALDI_WARN << "Tree has pdf-id " << i
          << " with no stats; corresponding phone list: " << ss.str();
      /*
        This probably means you have phones that were unseen in training
        and were not shared with other phones in the roots file.
        You should modify your roots file as necessary to fix this.
        (i.e. share that phone with a similar but seen phone on one line
        of the roots file). Be sure to regenerate roots.int from roots.txt,
        if using s5 scripts. To work out the phone, search for
        pdf-id  i  in the output of show-transitions (for this model). */
    }
  }
  std::vector<Clusterable*> summed_stats;
  SumStatsVec(split_stats, &summed_stats);
  Clusterable *avg_stats = SumClusterable(summed_stats);
  KALDI_ASSERT(avg_stats != NULL && "No stats available in gmm-init-model.");
  for (size_t i = 0; i < summed_stats.size(); i++) {
    GaussClusterable *c =
        static_cast<GaussClusterable*>(summed_stats[i] != NULL ? summed_stats[i] : avg_stats);
    DiagGmm gmm(*c, var_floor);
    am_gmm.AddPdf(gmm);
    BaseFloat count = c->count();
    if (count < 100) {
      std::vector<int32> bad_pdfs(1, i), bad_phones;
      GetPhonesForPdfs(trans_model, bad_pdfs, &bad_phones);
      std::ostringstream ss;
      for (int32 idx = 0; idx < bad_phones.size(); idx ++)
        ss << bad_phones[idx] << ' ';
      KALDI_WARN << "Very small count for state " << i << ": "
                 << count << "; corresponding phone list: " << ss.str();
    }
  }
  DeletePointers(&summed_stats);
  delete avg_stats;


    if (mixup != 0 || mixdown != 0) {

      Vector<BaseFloat> occs;
      GetOccs(stats, to_pdf_map, &occs);
      if (occs.Dim() != am_gmm.NumPdfs())
        KALDI_ERR << "Dimension of state occupancies " << occs.Dim()
                   << " does not match num-pdfs " << am_gmm.NumPdfs();

      if (mixdown != 0)
        am_gmm.MergeByCount(occs, mixdown, power, min_count);

      if (mixup != 0)
        am_gmm.SplitByCount(occs, mixup, perturb_factor,
                            power, min_count);
    }
    {
      Output ko(model_filename, binary);
      trans_model.Write(ko.Stream(), binary);
      am_gmm.Write(ko.Stream(), binary);
    }
          },
        py::arg("topo"),
        py::arg("ctx_dep"),
        py::arg("stats"),
        py::arg("model_filename"),
        py::arg("var_floor") = 0.01,
        py::arg("mixup") = 0,
        py::arg("mixdown") = 0,
        py::arg("perturb_factor") = 0.1,
        py::arg("power") = 0.2,
        py::arg("min_count") = 20.0
        );

  m.def("gmm_init_model_from_previous",
          [](
            const HmmTopology &topo,
            const ContextDependency &ctx_dep,
    const BuildTreeStatsType &stats,
  const AmDiagGmm &old_am_gmm,
  const TransitionModel &old_trans_model,
  const ContextDependency &old_tree,
    const std::string model_filename,
    double var_floor = 0.01,
    int32 mixup = 0,
    int32 mixdown = 0,
    BaseFloat perturb_factor = 0.01,
    BaseFloat power = 0.2,
    BaseFloat min_count = 20.0){

    bool binary = true;
    const EventMap &to_pdf_map = ctx_dep.ToPdfMap();  // not owned here.
    int32 N = ctx_dep.ContextWidth();
    int32 P = ctx_dep.CentralPosition();
    TransitionModel trans_model(ctx_dep, topo);

    // Now, the summed_stats will be used to initialize the GMM.
    AmDiagGmm am_gmm;

  // Get stats split by (new) tree-leaf ( == pdf):
  std::vector<BuildTreeStatsType> split_stats;
  SplitStatsByMap(stats, to_pdf_map, &split_stats);
  // Make sure each leaf has stats.
  for (size_t i = 0; i < split_stats.size(); i++) {
    if (split_stats[i].empty()) {
      KALDI_WARN << "Leaf " << i << " of new tree has no stats.";
    }
  }
  if (static_cast<int32>(split_stats.size()) != to_pdf_map.MaxResult() + 1) {
    KALDI_ASSERT(static_cast<int32>(split_stats.size()) <
                 to_pdf_map.MaxResult() + 1);
    KALDI_WARN << "Tree may have final leaf with no stats.";
    split_stats.resize(to_pdf_map.MaxResult() + 1);
    // avoid indexing errors later.
  }

  int32 oldN = old_tree.ContextWidth(), oldP = old_tree.CentralPosition();

  // avg_stats will be used for leaves that have no stats.
  Clusterable *avg_stats = SumStats(stats);
  GaussClusterable *avg_stats_gc = dynamic_cast<GaussClusterable*>(avg_stats);
  KALDI_ASSERT(avg_stats_gc != NULL && "Empty stats input.");
  DiagGmm avg_gmm(*avg_stats_gc, var_floor);
  delete avg_stats;
  avg_stats = NULL;
  avg_stats_gc = NULL;

  const EventMap &old_map = old_tree.ToPdfMap();

  KALDI_ASSERT(am_gmm.NumPdfs() == 0);
  int32 num_pdfs = static_cast<int32>(split_stats.size());
  for (int32 pdf = 0; pdf < num_pdfs; pdf++) {
    BuildTreeStatsType &my_stats = split_stats[pdf];
    // The next statement converts the stats to a possibly narrower older
    // context-width (e.g. triphone -> monophone).
    // note: don't get confused by the "old" and "new" in the parameters
    // to ConvertStats.  The next line is correct.
    bool ret = ConvertStats(N, P, oldN, oldP, &my_stats);
    if (!ret)
      KALDI_ERR << "InitAmGmmFromOld: old system has wider context "
          "so cannot convert stats.";
    // oldpdf_to_count works out a map from old pdf-id to count (for stats
    // that align to this "new" pdf... we'll use it to work out the old pdf-id
    // that's "closest" in stats overlap to this new pdf ("pdf").
    std::map<int32, BaseFloat> oldpdf_to_count;
    for (size_t i = 0; i < my_stats.size(); i++) {
      EventType evec = my_stats[i].first;
      EventAnswerType ans;
      bool ret = old_map.Map(evec, &ans);
      if (!ret) { KALDI_ERR << "Could not map context using old tree."; }
      KALDI_ASSERT(my_stats[i].second != NULL);
      BaseFloat stats_count = my_stats[i].second->Normalizer();
      if (oldpdf_to_count.count(ans) == 0) oldpdf_to_count[ans] = stats_count;
      else oldpdf_to_count[ans] += stats_count;
    }
    BaseFloat max_count = 0; int32 max_old_pdf = -1;
    for (std::map<int32, BaseFloat>::const_iterator iter = oldpdf_to_count.begin();
        iter != oldpdf_to_count.end();
        ++iter) {
      if (iter->second > max_count) {
        max_count = iter->second;
        max_old_pdf = iter->first;
      }
    }
    if (max_count == 0) {  // no overlap - probably a leaf with no stats at all.
      KALDI_WARN << "Leaf " << pdf << " of new tree being initialized with "
                 << "globally averaged stats.";
      am_gmm.AddPdf(avg_gmm);
    } else {
      am_gmm.AddPdf(old_am_gmm.GetPdf(max_old_pdf));  // Here is where we copy the relevant old PDF.
    }
  }


    if (mixup != 0 || mixdown != 0) {

      Vector<BaseFloat> occs;
      GetOccs(stats, to_pdf_map, &occs);
      if (occs.Dim() != am_gmm.NumPdfs())
        KALDI_ERR << "Dimension of state occupancies " << occs.Dim()
                   << " does not match num-pdfs " << am_gmm.NumPdfs();

      if (mixdown != 0)
        am_gmm.MergeByCount(occs, mixdown, power, min_count);

      if (mixup != 0)
        am_gmm.SplitByCount(occs, mixup, perturb_factor,
                            power, min_count);
    }
    {
      Output ko(model_filename, binary);
      trans_model.Write(ko.Stream(), binary);
      am_gmm.Write(ko.Stream(), binary);
    }
          },
        py::arg("topo"),
        py::arg("ctx_dep"),
        py::arg("stats"),
        py::arg("old_am_gmm"),
        py::arg("old_trans_model"),
        py::arg("old_tree"),
        py::arg("model_filename"),
        py::arg("var_floor") = 0.01,
        py::arg("mixup") = 0,
        py::arg("mixdown") = 0,
        py::arg("perturb_factor") = 0.1,
        py::arg("power") = 0.2,
        py::arg("min_count") = 20.0
        );

  m.def("pdf_post_to_gpost",
        [](const Posterior &pdf_post,
                         const AmDiagGmm &am_gmm,
                         const Matrix<BaseFloat> &feats,
                         BaseFloat rand_prune = 0.0){

          py::gil_scoped_release release;
          GaussPost gpost(pdf_post.size());
          BaseFloat tot_like_this_file = 0.0, tot_weight = 0.0;
          for (size_t i = 0; i < pdf_post.size(); i++) {
               gpost[i].reserve(pdf_post[i].size());
               for (size_t j = 0; j < pdf_post[i].size(); j++) {
               int32 pdf_id = pdf_post[i][j].first;
               BaseFloat weight = pdf_post[i][j].second;
               const DiagGmm &gmm = am_gmm.GetPdf(pdf_id);
               Vector<BaseFloat> this_post_vec;
               BaseFloat like =
                    gmm.ComponentPosteriors(feats.Row(i), &this_post_vec);
               this_post_vec.Scale(weight);
               if (rand_prune > 0.0)
               for (int32 k = 0; k < this_post_vec.Dim(); k++)
                    this_post_vec(k) = RandPrune(this_post_vec(k),
                                                  rand_prune);
               if (!this_post_vec.IsZero())
               gpost[i].push_back(std::make_pair(pdf_id, this_post_vec));
               tot_like_this_file += like * weight;
               tot_weight += weight;
               }
          }
          return gpost;
                       },
        py::arg("pdf_post"),
        py::arg("am_gmm"),
        py::arg("feats"),
        py::arg("rand_prune") = 0.0,
        py::return_value_policy::take_ownership
        );

  m.def("gmm_compute_likes",
        [](const AmDiagGmm &am_gmm,
                         const Matrix<BaseFloat> &features){

          py::gil_scoped_release release;
          Matrix<BaseFloat> loglikes(features.NumRows(), am_gmm.NumPdfs());
          for (int32 i = 0; i < features.NumRows(); i++) {
            for (int32 j = 0; j < am_gmm.NumPdfs(); j++) {
              SubVector<BaseFloat> feat_row(features, i);
              loglikes(i, j) = am_gmm.LogLikelihood(j, feat_row);
            }
          }
          return loglikes;
                       },
        py::arg("am_gmm"),
        py::arg("features"),
        py::return_value_policy::take_ownership
        );
}
