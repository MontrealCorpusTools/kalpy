#include "transform/pybind_transform.h"
#include "transform/basis-fmllr-diag-gmm.h"
#include "transform/cmvn.h"
#include "transform/compressed-transform-stats.h"
#include "transform/decodable-am-diag-gmm-regtree.h"
#include "transform/fmllr-diag-gmm.h"
#include "transform/fmllr-raw.h"
#include "transform/fmpe.h"
#include "transform/lda-estimate.h"
#include "transform/lvtln.h"
#include "transform/mllt.h"
#include "transform/regression-tree.h"
#include "transform/regtree-fmllr-diag-gmm.h"
#include "transform/regtree-mllr-diag-gmm.h"
#include "transform/transform-common.h"
#include "lat/kaldi-lattice.h"
#include "lat/lattice-functions.h"
#include <pybind11/stl.h>


using namespace kaldi;

void pybind_basis_fmllr_diag_gmm(py::module &m) {

  {
    using PyClass = BasisFmllrOptions;

    auto basis_fmllr_options = py::class_<PyClass>(
        m, "BasisFmllrOptions");

    basis_fmllr_options.def(py::init<>())
      .def_readwrite("num_iters", &PyClass::num_iters)
      .def_readwrite("size_scale", &PyClass::size_scale)
      .def_readwrite("min_count", &PyClass::min_count)
      .def_readwrite("step_size_iters", &PyClass::step_size_iters);
  }
  {
     using PyClass = BasisFmllrAccus;

     auto basis_fmllr_accus = py::class_<PyClass>(
        m, "BasisFmllrAccus");

     basis_fmllr_accus.def(py::init<>())
        .def("ResizeAccus", &PyClass::ResizeAccus, py::arg("dim"))
        .def("Write", &PyClass::Write, py::arg("os"), py::arg("binary"),
      py::call_guard<py::gil_scoped_release>())
        .def("Read", &PyClass::Read, py::arg("is"), py::arg("binary"), py::arg("add") = false,
      py::call_guard<py::gil_scoped_release>())
        .def("AccuGradientScatter",
          &PyClass::AccuGradientScatter,
          "Accumulate gradient scatter for one (training) speaker. "
          "To finish the process, we need to traverse the whole training "
          "set. Parallelization works if the speaker list is splitted, and "
          "stats are summed up by setting add=true in BasisFmllrEstimate:: "
          "ReadBasis. See section 5.2 of the paper.",
          py::arg("spk_stats"));
        ;
  }
  {
     using PyClass = BasisFmllrEstimate;

     auto basis_fmllr_estimate = py::class_<PyClass>(
        m, "BasisFmllrEstimate");

     basis_fmllr_estimate.def(py::init<>())
        .def("Write", &PyClass::Write, py::arg("os"), py::arg("binary"),
      py::call_guard<py::gil_scoped_release>())
        .def("Read", &PyClass::Read, py::arg("is"), py::arg("binary"),
      py::call_guard<py::gil_scoped_release>())
        .def("EstimateFmllrBasis",
          &PyClass::EstimateFmllrBasis,
          "Estimate the base matrices efficiently in a Maximum Likelihood manner. "
          "It takes diagonal GMM as argument, which will be used for preconditioner "
          "computation. The total number of bases is fixed to "
          "N = (dim + 1) * dim "
          "Note that SVD is performed in the normalized space. The base matrices "
          "are finally converted back to the unnormalized space.",
          py::arg("am_gmm"),
          py::arg("basis_accus"))
        .def("ComputeAmDiagPrecond",
          &PyClass::ComputeAmDiagPrecond,
          "This function computes the preconditioner matrix, prior to base matrices "
          "estimation. Since the expected values of G statistics are used, it "
          "takes the acoustic model as the argument, rather than the actual "
          "accumulations AffineXformStats "
          "See section 5.1 of the paper.",
          py::arg("am_gmm"),
          py::arg("pre_cond"))
        .def("Dim", &PyClass::Dim)
        .def("BasisSize", &PyClass::BasisSize)
        .def("ComputeTransform",
          &PyClass::ComputeTransform,
          "This function performs speaker adaptation, computing the fMLLR matrix "
          "based on speaker statistics. It takes fMLLR stats as argument. "
          "The basis weights (d_{1}, d_{2}, ..., d_{N}) are also optimized "
          "explicitly. Finally, it returns objective function improvement over "
          "all the iterations, compared with the value at the initial value of "
          "\"out_xform\" (or the unit transform if not provided). "
          "The coefficients are output to \"coefficients\" only if the vector is "
          "provided. "
          "See section 5.3 of the paper for more details.",
          py::arg("spk_stats"),
          py::arg("out_xform"),
          py::arg("coefficients"),
          py::arg("options"));
        ;
  }
}

void pybind_cmvn(py::module &m) {

  m.def("InitCmvnStats",
        &InitCmvnStats,
        "This function initializes the matrix to dimension 2 by (dim+1); "
          "1st \"dim\" elements of 1st row are mean stats, 1st \"dim\" elements "
          "of 2nd row are var stats, last element of 1st row is count, "
          "last element of 2nd row is zero.",
        py::arg("dim"), py::arg("stats"),
        py::call_guard<py::gil_scoped_release>());

  m.def("AccCmvnStats",
        py::overload_cast<const VectorBase<BaseFloat> &,
                  BaseFloat ,
                  MatrixBase<double> *>(&AccCmvnStats),
        "Accumulation from a single frame (weighted).",
        py::arg("feat"), py::arg("weight"), py::arg("stats"));
  m.def("AccCmvnStats",
        py::overload_cast<const MatrixBase<BaseFloat> &,
                  const VectorBase<BaseFloat> *,
                  MatrixBase<double> *>(&AccCmvnStats),
        "Accumulation from a feature file (possibly weighted-- useful in excluding silence).",
        py::arg("feats"), py::arg("weights"), py::arg("stats"),
        py::call_guard<py::gil_scoped_release>());
     m.def("calculate_cmvn",
     [](const std::vector<std::string> &uttlist,
          RandomAccessBaseFloatMatrixReader &feat_reader){

          int32 num_done = 0, num_err = 0;
          bool is_init = false;
          Matrix<double> stats;
          for (size_t i = 0; i < uttlist.size(); i++) {
            std::string utt = uttlist[i];
            if (!feat_reader.HasKey(utt)) {
              KALDI_WARN << "Did not find features for utterance " << utt;
              num_err++;
              continue;
            }
            const Matrix<BaseFloat> &feats = feat_reader.Value(utt);
            if (!is_init) {
              InitCmvnStats(feats.NumCols(), &stats);
              is_init = true;
            }
               AccCmvnStats(feats, NULL, &stats);
               num_done++;

          }
          return py::make_tuple(stats, num_done, num_err);
     },
     "Calculate CMVN from a speaker's utterances",
     py::arg("uttlist"),
     py::arg("feat_reader"));

  m.def("ApplyCmvn",
        &ApplyCmvn,
        "Apply cepstral mean and variance normalization to a matrix of features. "
          "If norm_vars == true, expects stats to be of dimension 2 by (dim+1), but "
          "if norm_vars == false, will accept stats of dimension 1 by (dim+1); these "
          "are produced by the balanced-cmvn code when it computes an offset and "
          "represents it as \"fake stats\".",
        py::arg("stats"), py::arg("norm_vars"), py::arg("feats"),
        py::call_guard<py::gil_scoped_release>());

  m.def("apply_transform",
        [](
               const Matrix<BaseFloat> &feat,
               const Matrix<BaseFloat> &trans
          ){
          py::gil_scoped_release release;
          int32 transform_rows = trans.NumRows(),
               transform_cols = trans.NumCols(),
               feat_dim = feat.NumCols();
      Matrix<BaseFloat> feat_out(feat.NumRows(), transform_rows);

          if (transform_cols == feat_dim) {
        feat_out.AddMatMat(1.0, feat, kNoTrans, trans, kTrans, 0.0);
      } else if (transform_cols == feat_dim + 1) {
        // append the implicit 1.0 to the input features.
        SubMatrix<BaseFloat> linear_part(trans, 0, transform_rows, 0, feat_dim);
        feat_out.AddMatMat(1.0, feat, kNoTrans, linear_part, kTrans, 0.0);
        Vector<BaseFloat> offset(transform_rows);
        offset.CopyColFromMat(trans, feat_dim);
        feat_out.AddVecToRows(1.0, offset);
      }
      return feat_out;
        },
        py::arg("feat"), py::arg("trans"));
  m.def("ApplyCmvnReverse",
        &ApplyCmvnReverse,
        "This is as ApplyCmvn, but does so in the reverse sense, i.e. applies a transform "
          "that would take zero-mean, unit-variance input and turn it into output with the "
          "stats of \"stats\".  This can be useful if you trained without CMVN but later want "
          "to correct a mismatch, so you would first apply CMVN and then do the \"reverse\" "
          "CMVN with the summed stats of your training data.",
        py::arg("stats"), py::arg("norm_vars"), py::arg("feats"),
        py::call_guard<py::gil_scoped_release>());
  m.def("FakeStatsForSomeDims",
        &FakeStatsForSomeDims,
        "Modify the stats so that for some dimensions (specified in \"dims\"), we "
          "replace them with \"fake\" stats that have zero mean and unit variance; this "
          "is done to disable CMVN for those dimensions.",
        py::arg("dims"), py::arg("stats"));
}

void pybind_compressed_transform_stats(py::module &m) {
     {

      using PyClass = CompressedAffineXformStats;
      auto compressed_affine_xform_stats = py::class_<PyClass>(
          m, "CompressedAffineXformStats");
      compressed_affine_xform_stats
          .def(py::init<>())
          .def(py::init<const AffineXformStats &>(),
               py::arg("input"))
          .def("CopyFromAffineXformStats", &PyClass::CopyFromAffineXformStats,
               py::arg("input"))
          .def("CopyToAffineXformStats", &PyClass::CopyToAffineXformStats,
               py::arg("input"))
          .def("Read", &PyClass::Read,
               py::arg("in_stream"), py::arg("binary"),
      py::call_guard<py::gil_scoped_release>())
          .def("Write", &PyClass::Write,
               py::arg("out_stream"), py::arg("binary"),
      py::call_guard<py::gil_scoped_release>());
    }
}

void pybind_decodable_am_diag_gmm_regtree(py::module &m) {

     {

      using PyClass = DecodableAmDiagGmmRegtreeFmllr;
      auto decodable_am_diag_gmm_regtree_fmllr = py::class_<PyClass, DecodableAmDiagGmmUnmapped>(
          m, "DecodableAmDiagGmmRegtreeFmllr");
      decodable_am_diag_gmm_regtree_fmllr
          .def(py::init<const AmDiagGmm &,
                                 const TransitionModel &,
                                 const Matrix<BaseFloat> &,
                                 const RegtreeFmllrDiagGmm &,
                                 const RegressionTree &,
                                 BaseFloat ,
                                 BaseFloat>(),
               py::arg("am"),
               py::arg("tm"),
               py::arg("feats"),
               py::arg("fmllr_xform"),
               py::arg("regtree"),
               py::arg("scale"),
               py::arg("log_sum_exp_prune") = -1.0)
          .def("LogLikelihood", &PyClass::LogLikelihood,
               py::arg("frame"),
               py::arg("tid"))
          .def("NumFramesReady", &PyClass::NumFramesReady)
          .def("NumIndices", &PyClass::NumIndices);
    }
     {

      using PyClass = DecodableAmDiagGmmRegtreeMllr;
      auto decodable_am_diag_gmm_regtree_mllr = py::class_<PyClass, DecodableAmDiagGmmUnmapped>(
          m, "DecodableAmDiagGmmRegtreeMllr");
      decodable_am_diag_gmm_regtree_mllr
          .def(py::init<const AmDiagGmm &,
                                 const TransitionModel &,
                                 const Matrix<BaseFloat> &,
                                 const RegtreeMllrDiagGmm &,
                                 const RegressionTree &,
                                 BaseFloat ,
                                 BaseFloat>(),
               py::arg("am"),
               py::arg("tm"),
               py::arg("feats"),
               py::arg("mllr_xform"),
               py::arg("regtree"),
               py::arg("scale"),
               py::arg("log_sum_exp_prune") = -1.0)
          .def("LogLikelihood", &PyClass::LogLikelihood,
               py::arg("frame"),
               py::arg("tid"))
          .def("NumFramesReady", &PyClass::NumFramesReady)
          .def("NumIndices", &PyClass::NumIndices)
          .def("TransModel", &PyClass::TransModel);
    }
}

void pybind_fmllr_diag_gmm(py::module &m) {

  py::class_<FmllrOptions>(m, "FmllrOptions")
      .def(py::init<>())
      .def_readwrite("update_type", &FmllrOptions::update_type)
      .def_readwrite("min_count", &FmllrOptions::min_count)
      .def_readwrite("num_iters", &FmllrOptions::num_iters);
     {

      using PyClass = FmllrDiagGmmAccs;
      auto fmllr_diag_gmm_accs = py::class_<PyClass, AffineXformStats>(
          m, "FmllrDiagGmmAccs");
      fmllr_diag_gmm_accs
          .def(py::init<const FmllrOptions &>(),
               py::arg("opts") = FmllrOptions())
          .def(py::init<int32 , const FmllrOptions &>(),
               py::arg("dim"),
               py::arg("opts") = FmllrOptions())
          .def(py::init<const FmllrDiagGmmAccs &>(),
               py::arg("other"))
          .def(py::init<const DiagGmm &, const AccumFullGmm &>(),
               py::arg("gmm"),
               py::arg("fgmm_accs"))
          .def("Init", &PyClass::Init,
               py::arg("dim"))
          .def("Read", &PyClass::Read,
               py::arg("in"),
               py::arg("binary"),
               py::arg("add"),
      py::call_guard<py::gil_scoped_release>())
          .def("AccumulateForGmm", &PyClass::AccumulateForGmm,
               py::arg("gmm"),
               py::arg("data"),
               py::arg("weight"),
      py::call_guard<py::gil_scoped_release>())
          .def("AccumulateForGmmPreselect", &PyClass::AccumulateForGmmPreselect,
               py::arg("gmm"),
               py::arg("gselect"),
               py::arg("data"),
               py::arg("weight"))
          .def("AccumulateFromPosteriors", &PyClass::AccumulateFromPosteriors,
               py::arg("gmm"),
               py::arg("data"),
               py::arg("posteriors"),
      py::call_guard<py::gil_scoped_release>())
          .def("accumulate_from_pdf_post",
               [](PyClass& spk_stats,
               const Posterior &pdf_post,
                         const AmDiagGmm &am_gmm,
                         const Matrix<BaseFloat> &feats
               ){
          py::gil_scoped_release gil_release;
                    for (size_t i = 0; i < pdf_post.size(); i++) {
                    for (size_t j = 0; j < pdf_post[i].size(); j++) {
                         int32 pdf_id = pdf_post[i][j].first;
                         spk_stats.AccumulateForGmm(am_gmm.GetPdf(pdf_id),
                                                  feats.Row(i),
                                                  pdf_post[i][j].second);
                    }
                    }

          },
               py::arg("pdf_post"),
               py::arg("am_gmm"),
               py::arg("feats"))
          .def("accumulate_from_gpost",
               [](PyClass& spk_stats,
                         const GaussPost &gpost,
                         const AmDiagGmm &am_gmm,
                         const Matrix<BaseFloat> &feats
               ){
          py::gil_scoped_release gil_release;
               Vector<BaseFloat> posterior;
               for (size_t i = 0; i < gpost.size(); i++) {
                    for (size_t j = 0; j < gpost[i].size(); j++) {
                         int32 pdf_id = gpost[i][j].first;
                         posterior = Vector<BaseFloat>(gpost[i][j].second);

                         spk_stats.AccumulateFromPosteriors(am_gmm.GetPdf(pdf_id),
                                                            feats.Row(i), posterior);


                    }
                    }
          },
               py::arg("gpost"),
               py::arg("am_gmm"),
               py::arg("feats"))
          .def("accumulate_from_alignment",
               [](PyClass& spk_stats,
                         const TransitionModel &trans_model,
                         const AmDiagGmm &am_gmm,
                         const Matrix<BaseFloat> &feats,
                         const std::vector<int32> &ali,
                       const ConstIntegerSet<int32> &silence_set,
                       BaseFloat silence_scale,
                         BaseFloat rand_prune = 0.0,
                       bool distributed = false,
                       bool two_models = false
               ){
                    py::gil_scoped_release gil_release;
                    Posterior pdf_post;
                    Posterior post;

                    AlignmentToPosterior(ali, &post);
                    if (distributed)
                    WeightSilencePostDistributed(trans_model, silence_set,
                                                  silence_scale, &post);
                    else
                         WeightSilencePost(trans_model, silence_set,
                                   silence_scale, &post);
                         ConvertPosteriorToPdfs(trans_model, post, &pdf_post);

                    if (!two_models){
                         for (size_t i = 0; i < pdf_post.size(); i++) {
                         for (size_t j = 0; j < pdf_post[i].size(); j++) {
                              int32 pdf_id = pdf_post[i][j].first;
                              spk_stats.AccumulateForGmm(am_gmm.GetPdf(pdf_id),
                                                       feats.Row(i),
                                                       pdf_post[i][j].second);
                         }
                         }
                    }
                    else{


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
                         for (size_t i = 0; i < gpost.size(); i++) {
                              for (size_t j = 0; j < gpost[i].size(); j++) {
                                   int32 pdf_id = gpost[i][j].first;
                                   const Vector<BaseFloat> & posterior(gpost[i][j].second);

                                   spk_stats.AccumulateFromPosteriors(am_gmm.GetPdf(pdf_id),
                                                                      feats.Row(i), posterior);

                              }
                              }
                    }
          },
               py::arg("trans_model"),
               py::arg("am_gmm"),
               py::arg("feats"),
               py::arg("ali"),
               py::arg("silence_set"),
               py::arg("silence_scale"),
               py::arg("rand_prune") = 0.0,
               py::arg("distributed") = false,
               py::arg("two_models") = false)
          .def("accumulate_from_lattice",
               [](PyClass* spk_stats,
                         const TransitionModel &trans_model,
                         const AmDiagGmm &am_gmm,
                         const Matrix<BaseFloat> &feats,
                         Lattice &lat,
                       const ConstIntegerSet<int32> &silence_set,
                       BaseFloat silence_scale,
                       BaseFloat lm_scale = 1.0,
                       BaseFloat acoustic_scale = 1.0,
                         BaseFloat rand_prune = 0.0,
                       bool distributed = false,
                       bool two_models = false
               ){
          py::gil_scoped_release gil_release;
                    if (acoustic_scale != 1.0 || lm_scale != 1.0)
                         fst::ScaleLattice(fst::LatticeScale(lm_scale, acoustic_scale), &lat);

                    kaldi::uint64 props = lat.Properties(fst::kFstProperties, false);
                    if (!(props & fst::kTopSorted)) {
                         if (fst::TopSort(&lat) == false)
                         {

                              py::gil_scoped_acquire acquire;
                              KALDI_ERR << "Cycles detected in lattice.";
                              return;
                         }
                         }
                    Posterior post;
                    double lat_like = LatticeForwardBackward(lat, &post);
                    if (distributed)
                    WeightSilencePostDistributed(trans_model, silence_set,
                                                  silence_scale, &post);
                    else
                    WeightSilencePost(trans_model, silence_set,
                              silence_scale, &post);
                    Posterior pdf_post;
                    ConvertPosteriorToPdfs(trans_model, post, &pdf_post);
                    if (!two_models){
                         for (size_t i = 0; i < post.size(); i++) {
                         for (size_t j = 0; j < pdf_post[i].size(); j++) {
                              int32 pdf_id = pdf_post[i][j].first;
                              spk_stats->AccumulateForGmm(am_gmm.GetPdf(pdf_id),
                                                       feats.Row(i),
                                                       pdf_post[i][j].second);
                         }
                         }
                    }
                    else{


                         GaussPost gpost(post.size());
                         BaseFloat tot_like_this_file = 0.0, tot_weight = 0.0;
                         for (size_t i = 0; i < post.size(); i++) {
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
                         for (size_t i = 0; i < gpost.size(); i++) {
                              for (size_t j = 0; j < gpost[i].size(); j++) {
                                   int32 pdf_id = gpost[i][j].first;
                                   const Vector<BaseFloat> & posterior(gpost[i][j].second);
                                   spk_stats->AccumulateFromPosteriors(am_gmm.GetPdf(pdf_id),
                                                                      feats.Row(i), posterior);
                              }
                              }
                    }
          },
               py::arg("trans_model"),
               py::arg("am_gmm"),
               py::arg("feats"),
               py::arg("lat"),
               py::arg("silence_set"),
               py::arg("silence_scale"),
               py::arg("lm_scale") = 1.0,
               py::arg("acoustic_scale") = 1.0,
               py::arg("rand_prune") = 0.0,
               py::arg("distributed") = false,
               py::arg("two_models") = false)
          .def("AccumulateFromPosteriorsPreselect", &PyClass::AccumulateFromPosteriorsPreselect,
               py::arg("gmm"),
               py::arg("gselect"),
               py::arg("data"),
               py::arg("posteriors"))
          .def("Update",
               &PyClass::Update,
               py::arg("opts"),
               py::arg("fmllr_mat"),
               py::arg("objf_impr"),
               py::arg("count"))
          .def("compute_transform",
               [](PyClass& f, const AmDiagGmm &am_gmm,
               const FmllrOptions &fmllr_opts){
                    py::gil_scoped_release gil_release;
                    BaseFloat impr, tot_t;
                    Matrix<BaseFloat> transform(am_gmm.Dim(), am_gmm.Dim()+1);
                    {
                    transform.SetUnit();
                    f.Update(fmllr_opts, &transform, &impr, &tot_t);
                    return transform;
                    }
               },
               py::arg("am_gmm"),
               py::arg("fmllr_opts"));
    }
  m.def("InitFmllr",
        &InitFmllr,
        "Initializes the FMLLR matrix to its default values.",
        py::arg("dim"),
        py::arg("out_fmllr"));
  m.def("ComputeFmllrLogDet",
        &ComputeFmllrLogDet,
        py::arg("fmllr_mat"));
  m.def("ComputeFmllrMatrixDiagGmmFull",
        &ComputeFmllrMatrixDiagGmmFull,
        "Updates the FMLLR matrix using Mark Gales' row-by-row update. "
          "Uses full fMLLR matrix (no structure).  Returns the "
          "objective function improvement, not normalized by number of frames.",
        py::arg("in_xform"),
        py::arg("stats"),
        py::arg("num_iters"),
        py::arg("out_xform"));
  m.def("ComputeFmllrMatrixDiagGmmDiagonal",
        &ComputeFmllrMatrixDiagGmmDiagonal,
        "This does diagonal fMLLR (i.e. only estimate an offset and scale per "
          "dimension).  The format of the output is the same as for the full case.  Of "
          "course, these statistics are unnecessarily large for this case.  Returns the "
          "objective function improvement, not normalized by number of frames.",
        py::arg("in_xform"),
        py::arg("stats"),
        py::arg("out_xform"));
  m.def("ComputeFmllrMatrixDiagGmmOffset",
        &ComputeFmllrMatrixDiagGmmOffset,
        "This does offset-only fMLLR, i.e. it only estimates an offset.",
        py::arg("in_xform"),
        py::arg("stats"),
        py::arg("out_xform"));
  m.def("ComputeFmllrMatrixDiagGmm",
        &ComputeFmllrMatrixDiagGmm,
        "This function internally calls ComputeFmllrMatrixDiagGmm{Full, Diagonal, Offset}, "
          "depending on \"fmllr_type\".",
        py::arg("in_xform"),
        py::arg("stats"),
        py::arg("fmllr_type"),
        py::arg("num_iters"),
        py::arg("out_xform"));
  m.def("FmllrAuxFuncDiagGmm",
        py::overload_cast<const MatrixBase<float> &,
                          const AffineXformStats &>(&FmllrAuxFuncDiagGmm),
        "Returns the (diagonal-GMM) FMLLR auxiliary function value given the transform "
          "and the stats.",
        py::arg("xform"),
        py::arg("stats"));
  m.def("FmllrAuxFuncDiagGmm",
        py::overload_cast<const MatrixBase<double> &,
                          const AffineXformStats &>(&FmllrAuxFuncDiagGmm),
        "Returns the (diagonal-GMM) FMLLR auxiliary function value given the transform "
          "and the stats.",
        py::arg("xform"),
        py::arg("stats"));
  m.def("FmllrAuxfGradient",
        &FmllrAuxfGradient,
        "Returns the (diagonal-GMM) FMLLR auxiliary function value given the transform "
          "and the stats.",
        py::arg("xform"),
        py::arg("stats"),
        py::arg("grad_out"));
  m.def("ApplyFeatureTransformToStats",
        &ApplyFeatureTransformToStats,
        "This function applies a feature-level transform to stats (useful for "
          "certain techniques based on fMLLR).  Assumes the stats are of the "
          "standard diagonal sort. "
          "The transform \"xform\" may be either dim x dim (linear), "
          "dim x dim+1 (affine), or dim+1 x dim+1 (affine with the "
          "last row equal to 0 0 0 .. 0 1).",
        py::arg("xform"),
        py::arg("stats"));
  m.def("ApplyModelTransformToStats",
        &ApplyModelTransformToStats,
        "ApplyModelTransformToStats takes a transform \"xform\", which must be diagonal "
          "(i.e. of the form T = [ D; b ] where D is diagonal), and applies it to the "
          "stats as if we had made it a model-space transform (note that the transform "
          "applied to the model means is the inverse transform of T).  Thus, if we are "
          "estimating a transform T U, and we get stats valid for estimating T U and we "
          "estimate T, we can then call this function (treating T as a model-space "
          "transform) and will get stats valid for estimating U.  This only works if T is "
          "diagonal, because otherwise the standard stats format is not valid.  xform must "
          "be of dimension d x d+1",
        py::arg("xform"),
        py::arg("stats"));
  m.def("FmllrInnerUpdate",
        &FmllrInnerUpdate,
        "This function does one row of the inner-loop fMLLR transform update. "
          "We export it because it's needed in the RawFmllr code. "
          "Here, if inv_G is the inverse of the G matrix indexed by this row, "
          "and k is the corresponding row of the K matrix.",
        py::arg("inv_G"),
        py::arg("k"),
        py::arg("beta"),
        py::arg("row"),
        py::arg("transform"));
}

void pybind_fmllr_raw(py::module &m) {

  py::class_<FmllrRawOptions>(m, "FmllrRawOptions")
      .def(py::init<>())
      .def_readwrite("min_count", &FmllrRawOptions::min_count)
      .def_readwrite("num_iters", &FmllrRawOptions::num_iters);
     {

      using PyClass = FmllrRawAccs;
      auto fmllr_raw_accs = py::class_<PyClass>(
          m, "FmllrRawAccs");
      fmllr_raw_accs
          .def(py::init<>())
          .def(py::init<int32 ,
               int32 ,
               const Matrix<BaseFloat> &>(),
               py::arg("raw_dim"),
               py::arg("model_dim"),
               py::arg("full_transform"))
          .def("RawDim", &PyClass::RawDim)
          .def("FullDim", &PyClass::FullDim)
          .def("SpliceWidth", &PyClass::SpliceWidth)
          .def("ModelDim", &PyClass::ModelDim)
          .def("AccumulateForGmm", &PyClass::AccumulateForGmm,
               "Accumulate stats for a single GMM in the model; returns log likelihood. "
               "Here, \"data\" will typically be of larger dimension than the model. "
               "Note: \"data\" is the original, spliced features-- before LDA+MLLT. "
               "Returns log-like for this data given this GMM, including rejected "
               "dimensions (not multiplied by weight).",
               py::arg("gmm"),
               py::arg("data"),
               py::arg("weight"))
          .def("AccumulateFromPosteriors", &PyClass::AccumulateFromPosteriors,
          "Accumulate stats for a GMM, given supplied posteriors.  Note: \"data\" is "
          "the original, spliced features-- before LDA+MLLT. ",
               py::arg("gmm"),
               py::arg("data"),
               py::arg("posteriors"))
          .def("Update", &PyClass::Update,
               "Update \"raw_fmllr_mat\"; it should have the correct dimension and "
               "reasonable values at entry (see the function InitFmllr in fmllr-diag-gmm.h "
               "for how to initialize it.) "
               "The only reason this function is not const is because we may have "
               "to call CommitSingleFrameStats().",
               py::arg("opts"),
               py::arg("fmllr_mat"),
               py::arg("objf_impr"),
               py::arg("count"))
          .def("SetZero", &PyClass::SetZero);
    }
}

void pybind_fmpe(py::module &m) {

  py::class_<FmpeOptions>(m, "FmpeOptions")
      .def(py::init<>())
      .def_readwrite("context_expansion", &FmpeOptions::context_expansion)
      .def_readwrite("post_scale", &FmpeOptions::post_scale)
          .def("Write", &FmpeOptions::Write,
               py::arg("os"),
               py::arg("binary"),
      py::call_guard<py::gil_scoped_release>())
          .def("Read", &FmpeOptions::Read,
               py::arg("is"),
               py::arg("binary"),
      py::call_guard<py::gil_scoped_release>());

  py::class_<FmpeUpdateOptions>(m, "FmpeUpdateOptions")
      .def(py::init<>())
      .def_readwrite("learning_rate", &FmpeUpdateOptions::learning_rate)
      .def_readwrite("l2_weight", &FmpeUpdateOptions::l2_weight);

  py::class_<FmpeStats>(m, "FmpeStats")
      .def(py::init<>())
      .def(py::init<const Fmpe &>(),
               py::arg("fmpe"))
      .def("Init", &FmpeStats::Init)
     .def("Write", &FmpeStats::Write,
          py::arg("os"),
          py::arg("binary"),
      py::call_guard<py::gil_scoped_release>())
     .def("Read", &FmpeStats::Read,
          py::arg("is"),
          py::arg("binary"),
          py::arg("add") = false,
      py::call_guard<py::gil_scoped_release>())
      .def("DerivPlus", &FmpeStats::DerivPlus)
      .def("DerivMinus", &FmpeStats::DerivMinus)
      .def("AccumulateChecks", &FmpeStats::AccumulateChecks,
          "If we're using the indirect differential, accumulates certain quantities "
          "that will be used in the update phase to verify that the computation "
          "of the indirect differential was done correctly",
          py::arg("feats"),
          py::arg("direct_deriv"),
          py::arg("indirect_deriv"))
      .def("DoChecks", &FmpeStats::DoChecks);
     {

      using PyClass = Fmpe;
      auto fmpe = py::class_<PyClass>(
          m, "Fmpe");
      fmpe
          .def(py::init<>())
          .def(py::init<const DiagGmm &, const FmpeOptions &>(),
               py::arg("gmm"),
               py::arg("config"))
          .def("FeatDim", &PyClass::FeatDim)
          .def("NumGauss", &PyClass::NumGauss)
          .def("NumContexts", &PyClass::NumContexts)
          .def("ProjectionTNumRows", &PyClass::ProjectionTNumRows)
          .def("ProjectionTNumCols", &PyClass::ProjectionTNumCols)
          .def("ComputeFeatures", &PyClass::ComputeFeatures,
               "Computes the fMPE feature offsets and outputs them. "
               "You can add feat_in to this afterwards, if you want. "
               "Requires the Gaussian-selection info, which would normally "
               "be computed by a separate program-- this consists of "
               "lists of the top-scoring Gaussians for these features.",
               py::arg("feat_in"),
               py::arg("gselect"),
               py::arg("feat_out"))
          .def("AccStats", &PyClass::AccStats,
          "For training-- compute the derivative w.r.t the projection matrix "
          "(we keep the positive and negative parts separately to help "
          "set the learning rates).",
               py::arg("feat_in"),
               py::arg("gselect"),
               py::arg("direct_feat_deriv"),
               py::arg("indirect_feat_deriv"),
               py::arg("stats"))
          .def("Write", &PyClass::Write,
               py::arg("os"),
               py::arg("binary"),
      py::call_guard<py::gil_scoped_release>())
          .def("Read", &PyClass::Read,
               py::arg("is"),
               py::arg("binary"),
      py::call_guard<py::gil_scoped_release>())
          .def("Update", &PyClass::Update,
               "Returns total objf improvement, based on linear assumption.",
               py::arg("config"),
               py::arg("stats"));
    }

  m.def("ComputeAmGmmFeatureDeriv",
        &ComputeAmGmmFeatureDeriv,
        "Computes derivatives of the likelihood of these states (weighted), "
          "w.r.t. the feature values.  Used in fMPE training.  Note, the "
          "weights \"posterior\" may be positive or negative-- for MMI, MPE, "
          "etc., they will typically be of both signs.  Will resize \"deriv\". "
          "Returns the sum of (GMM likelihood * weight), which may be used "
          "as an approximation to the objective function. "
          "Last two parameters are optional.  See GetStatsDerivative() for "
          "or fMPE paper (ICASSP, 2005) more info on indirect derivative. "
          "Caution: if you supply the last two parameters, this function only "
          "works in the MMI case as it assumes the stats with positive weight "
          "are numerator == ml stats-- this is only the same thing in the MMI "
          "case, not fMPE.",
        py::arg("am_gmm"),
        py::arg("trans_model"),
        py::arg("posterior"),
        py::arg("features"),
        py::arg("direct_deriv"),
        py::arg("model_diff") = NULL,
        py::arg("indirect_deriv") = NULL);
}

void pybind_lda_estimate(py::module &m) {


  py::class_<LdaEstimateOptions>(m, "LdaEstimateOptions")
      .def(py::init<>())
      .def_readwrite("remove_offset", &LdaEstimateOptions::remove_offset)
      .def_readwrite("dim", &LdaEstimateOptions::dim)
      .def_readwrite("allow_large_dim", &LdaEstimateOptions::allow_large_dim)
      .def_readwrite("within_class_factor", &LdaEstimateOptions::within_class_factor);
  {

      using PyClass = LdaEstimate;
      auto lda_estimate = py::class_<PyClass>(
          m, "LdaEstimate");
      lda_estimate
          .def(py::init<>())
          .def("Init", &PyClass::Init,
               py::arg("num_classes"),
               py::arg("dimension"))
          .def("NumClasses", &PyClass::NumClasses)
          .def("Dim", &PyClass::Dim)
          .def("ZeroAccumulators", &PyClass::ZeroAccumulators)
          .def("Scale", &PyClass::Scale, py::arg("f"))
          .def("TotCount", &PyClass::TotCount)
          .def("Accumulate", &PyClass::Accumulate,
               py::arg("data"), py::arg("class_id"), py::arg("weight") = 1.0)
          .def("Estimate", &PyClass::Estimate,
               "Estimates the LDA transform matrix m.  If Mfull != NULL, it also outputs "
               "the full matrix (without dimensionality reduction), which is useful for "
               "some purposes.  If opts.remove_offset == true, it will output both matrices "
               "with an extra column which corresponds to mean-offset removal (the matrix "
               "should be multiplied by the feature with a 1 appended to give the correct "
               "result, as with other Kaldi transforms.) "
               "The \"remove_offset\" argument is new and should be set to false for back "
               "compatibility.",
               py::arg("opts"), py::arg("M"), py::arg("Mfull") = NULL)
          .def("estimate",
          [](PyClass &lda, const LdaEstimateOptions &opts){

               Matrix<BaseFloat> lda_mat;
               Matrix<BaseFloat> full_lda_mat;
               lda.Estimate(opts, &lda_mat, &full_lda_mat);
               return py::make_tuple(lda_mat, full_lda_mat);
          }, py::arg("opts"))
          .def("Read", &PyClass::Read,
               py::arg("in_stream"), py::arg("binary"), py::arg("add"),
      py::call_guard<py::gil_scoped_release>())
          .def("Write", &PyClass::Write,
               py::arg("out_stream"), py::arg("binary"),
      py::call_guard<py::gil_scoped_release>())
          .def("Add",
               [](
                  PyClass &lda,
                  const PyClass &other
               ){
               py::gil_scoped_release gil_release;

               std::ostringstream os;
               bool binary = true;
               other.Write(os, binary);
               std::istringstream str(os.str());
               lda.Read(str, true, true);
               },
               py::arg("other")
               )
          .def("acc_lda",
                    [](PyClass &lda,
                    const TransitionModel &trans_model,
                                   const std::vector<int32> &ali,
                                   const Matrix<BaseFloat> &feats,
                                   const ConstIntegerSet<int32> &silence_set,
                                   BaseFloat rand_prune = 0.0,
                                   BaseFloat silence_weight = 0.0){
          py::gil_scoped_release gil_release;

               Posterior post;
               AlignmentToPosterior(ali, &post);
               WeightSilencePost(trans_model, silence_set,
                                   silence_weight, &post);
               Posterior pdf_post;
               ConvertPosteriorToPdfs(trans_model, post, &pdf_post);
               for (int32 i = 0; i < feats.NumRows(); i++) {
               SubVector<BaseFloat> feat(feats, i);
               for (size_t j = 0; j < pdf_post[i].size(); j++) {
                    int32 pdf_id = pdf_post[i][j].first;
                    BaseFloat weight = RandPrune(pdf_post[i][j].second, rand_prune);
                    if (weight != 0.0) {
                    lda.Accumulate(feat, pdf_id, weight);
                    }
               }
               }
                    },
               py::arg("trans_model"),
               py::arg("ali"),
               py::arg("feats"),
               py::arg("silence_set"),
               py::arg("rand_prune") = 0.0,
               py::arg("silence_weight") = 0.0
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


}

void pybind_lvtln(py::module &m) {
    {

      using PyClass = LinearVtln;
      auto linear_vtln = py::class_<PyClass>(
          m, "LinearVtln");
      linear_vtln
          .def(py::init<>())
          .def(py::init<int32 , int32 , int32 >(),
               py::arg("dim"),
               py::arg("num_classes"),
               py::arg("default_class"))
          .def("SetTransform", &PyClass::SetTransform,
               "SetTransform is used when we initialize it as \"normal\" VTLN. "
               "It's not necessary to ever call this function.  \"transform\" is \"A\", "
               "the square part of the transform matrix.",
               py::arg("i"),
               py::arg("transform"))
          .def("SetWarp", &PyClass::SetWarp,
               py::arg("i"),
               py::arg("warp"))
          .def("GetWarp", &PyClass::GetWarp,
               py::arg("i"))
          .def("GetTransform", &PyClass::GetTransform,
               py::arg("i"),
               py::arg("transform"))
          .def("ComputeTransform", &PyClass::ComputeTransform,
               "Compute the transform for the speaker.",
               py::arg("accs"),
               py::arg("norm_type"),
               py::arg("logdet_scale"),
               py::arg("Ws"),
               py::arg("class_idx"),
               py::arg("logdet_out"),
               py::arg("objf_impr") = NULL,
               py::arg("count") = NULL)
          .def("Read", &PyClass::Read,
               py::arg("in_stream"), py::arg("binary"),
      py::call_guard<py::gil_scoped_release>())
          .def("Write", &PyClass::Write,
               py::arg("out_stream"), py::arg("binary"),
      py::call_guard<py::gil_scoped_release>())
          .def("NumClasses", &PyClass::NumClasses)
          .def("Dim", &PyClass::Dim);
    }
}

void pybind_mllt(py::module &m) {

    {

      using PyClass = MlltAccs;
      auto mllt_accs = py::class_<PyClass>(
          m, "MlltAccs");
      mllt_accs
          .def(py::init<>())
          .def(py::init<int32 , BaseFloat >(),
               py::arg("dim"),
               py::arg("rand_prune") = 0.25)
          .def("Init", &PyClass::Init,
               py::arg("dim"),
               py::arg("rand_prune") = 0.25)
          .def("Read", &PyClass::Read,
               py::arg("in_stream"), py::arg("binary"), py::arg("add") = false,
      py::call_guard<py::gil_scoped_release>())
          .def("Write", &PyClass::Write,
               py::arg("out_stream"), py::arg("binary"),
      py::call_guard<py::gil_scoped_release>())
          .def("Dim", &PyClass::Dim)
          .def("Update", static_cast< void (PyClass::*)(MatrixBase<BaseFloat> *,
              BaseFloat *,
              BaseFloat *) const>(&PyClass::Update),
               "The Update function does the ML update; it requires that M has the "
               "right size. "
               " @param [in, out] M  The output transform, will be of dimension Dim() x Dim(). "
               "                  At input, should be the unit transform (the objective function "
               "                  improvement is measured relative to this value). "
               " @param [out] objf_impr_out  The objective function improvement "
               " @param [out] count_out  The data-count",
               py::arg("M"),
               py::arg("objf_impr_out"),
               py::arg("count_out"))
          .def("update",
               [](
                  PyClass &mllt_accs){

               Matrix<BaseFloat> mat(mllt_accs.Dim(), mllt_accs.Dim());
               mat.SetUnit();
               BaseFloat objf_impr, count;
               mllt_accs.Update(&mat, &objf_impr, &count);
               return py::make_tuple(mat, objf_impr, count);
               },
               "The Update function does the ML update; it requires that M has the "
               "right size. "
               " @param [in, out] M  The output transform, will be of dimension Dim() x Dim(). "
               "                  At input, should be the unit transform (the objective function "
               "                  improvement is measured relative to this value). "
               " @param [out] objf_impr_out  The objective function improvement "
               " @param [out] count_out  The data-count")
          .def("Add",
               [](
                  PyClass &accs,
                  const PyClass &other
               ){
          py::gil_scoped_release gil_release;

               std::ostringstream os;
               bool binary = true;
               other.Write(os, binary);
               std::istringstream str(os.str());
               accs.Read(str, true, true);
               },
               py::arg("other")
               )
          .def_static("Update_", py::overload_cast<double ,
                     const std::vector<SpMatrix<double> > &,
                     MatrixBase<BaseFloat> *,
                     BaseFloat *,
                     BaseFloat *>(&PyClass::Update),
               "A static version of the Update function, so it can "
               "be called externally, given the right stats.",
               py::arg("beta"),
               py::arg("G"),
               py::arg("M"),
               py::arg("objf_impr_out"),
               py::arg("count_out"))
          .def("AccumulateFromGmm", &PyClass::AccumulateFromGmm,
               py::arg("gmm"),
               py::arg("data"),
               py::arg("weight"))
          .def("AccumulateFromGmmPreselect", &PyClass::AccumulateFromGmmPreselect,
               py::arg("gmm"),
               py::arg("gselect"),
               py::arg("data"),
               py::arg("weight"))
          .def("AccumulateFromPosteriors", &PyClass::AccumulateFromPosteriors,
               py::arg("gmm"),
               py::arg("data"),
               py::arg("posteriors"))
          .def("gmm_acc_mllt",
               [](PyClass &mllt_accs,
                  const AmDiagGmm &am_gmm,
                    const TransitionModel &trans_model,
                                   const std::vector<int32> &ali,
                                  const Matrix<BaseFloat> &mat,
                                   const ConstIntegerSet<int32> &silence_set,
                                   BaseFloat silence_weight = 0.0){

                    py::gil_scoped_release release;
                    Posterior post;
                    AlignmentToPosterior(ali, &post);
                    WeightSilencePost(trans_model, silence_set,
                                        silence_weight, &post);
                    Posterior pdf_posterior;
                    ConvertPosteriorToPdfs(trans_model, post, &pdf_posterior);
                    BaseFloat tot_like_this_file = 0.0, tot_weight = 0.0;
                    for (size_t i = 0; i < pdf_posterior.size(); i++) {
                         for (size_t j = 0; j < pdf_posterior[i].size(); j++) {
                         int32 pdf_id = pdf_posterior[i][j].first;
                         BaseFloat weight = pdf_posterior[i][j].second;

                         tot_like_this_file += mllt_accs.AccumulateFromGmm(am_gmm.GetPdf(pdf_id),
                                                                           mat.Row(i),
                                                                           weight) * weight;
                         tot_weight += weight;
                         }
                    }
                    py::gil_scoped_acquire acquire;
                    return py::make_tuple(tot_like_this_file, tot_weight);
               },
               py::arg("gmm"),
               py::arg("trans_model"),
               py::arg("ali"),
               py::arg("mat"),
               py::arg("silence_set"),
               py::arg("silence_weight") = 0.0)
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
}

void pybind_regression_tree(py::module &m) {

    {

      using PyClass = RegressionTree;
      auto regression_tree = py::class_<PyClass>(
          m, "RegressionTree");
      regression_tree
          .def(py::init<>())
          .def("BuildTree", &PyClass::BuildTree,
               "Top-down clustering of the Gaussians in a model based on their means. "
               "If sil_indices is nonempty, will put silence in a special class "
               "using a top-level split.",
               py::arg("state_occs"),
               py::arg("sil_indices"),
               py::arg("am"),
               py::arg("randmax_clusters_prune"))
          .def("GatherStats", &PyClass::GatherStats,
               "Parses the regression tree and finds the nodes whose occupancies (read "
               "from stats_in) are greater than min_count. The regclass_out vector has "
               "size equal to number of baseclasses, and contains the regression class "
               "index for each baseclass. The stats_out vector has size equal to number "
               "of regression classes. Return value is true if at least one regression "
               "class passed the count cutoff, false otherwise.",
               py::arg("stats_in"),
               py::arg("min_count"),
               py::arg("regclasses_out"),
               py::arg("stats_out"))
          .def("Read", &PyClass::Read,
               py::arg("in_stream"), py::arg("binary"), py::arg("am"),
      py::call_guard<py::gil_scoped_release>())
          .def("Write", &PyClass::Write,
               py::arg("out_stream"), py::arg("binary"),
      py::call_guard<py::gil_scoped_release>())
          .def("NumBaseclasses", &PyClass::NumBaseclasses)
          .def("GetBaseclass", &PyClass::GetBaseclass, py::arg("bclass"))
          .def("Gauss2BaseclassId", &PyClass::Gauss2BaseclassId, py::arg("pdf_id"), py::arg("gauss_id"));
    }
}

void pybind_regtree_fmllr_diag_gmm(py::module &m) {

  py::class_<RegtreeFmllrOptions>(m, "RegtreeFmllrOptions")
      .def(py::init<>())
      .def_readwrite("update_type", &RegtreeFmllrOptions::update_type)
      .def_readwrite("min_count", &RegtreeFmllrOptions::min_count)
      .def_readwrite("num_iters", &RegtreeFmllrOptions::num_iters)
      .def_readwrite("use_regtree", &RegtreeFmllrOptions::use_regtree);

    {

      using PyClass = RegtreeFmllrDiagGmm;
      auto regtree_fmllr_diag_gmm = py::class_<PyClass>(
          m, "RegtreeFmllrDiagGmm",
          "An FMLLR (feature-space MLLR) transformation, also called CMLLR "
          "(constrained MLLR) is an affine transformation of the feature vectors. "
          "This class supports multiple transforms, and a regression tree. "
          "For a single, feature-level transformation see fmllr-diag-gmm-global.h "
          "Note: the \"regression classes\" are the classes after tree-clustering, "
          "which are smaller in number than the \"base classes\"  (these correspond "
          "to the leaves of the tree).");
      regtree_fmllr_diag_gmm
          .def(py::init<>())
          .def(py::init<const RegtreeFmllrDiagGmm &>(),
               py::arg("other"))
          .def("Init", &PyClass::Init,
               py::arg("num_xforms"),
               py::arg("dim"))
          .def("Validate", &PyClass::Validate)
          .def("SetUnit", &PyClass::SetUnit)
          .def("ComputeLogDets", &PyClass::ComputeLogDets)
          .def("TransformFeature", &PyClass::TransformFeature,
               "Get the transformed features for each of the transforms.",
               py::arg("in"),
               py::arg("out"))
          .def("Read", &PyClass::Read,
               py::arg("in_stream"), py::arg("binary"),
      py::call_guard<py::gil_scoped_release>())
          .def("Write", &PyClass::Write,
               py::arg("out_stream"), py::arg("binary"),
      py::call_guard<py::gil_scoped_release>())
          .def("Dim", &PyClass::Dim)
          .def("NumBaseClasses", &PyClass::NumBaseClasses)
          .def("NumRegClasses", &PyClass::NumRegClasses)
          .def("GetXformMatrix", &PyClass::GetXformMatrix, py::arg("xform_index"), py::arg("out"))
          .def("GetLogDets", &PyClass::GetLogDets, py::arg("out"))
          .def("Base2RegClass", &PyClass::Base2RegClass, py::arg("bclass"))
          .def("SetParameters", &PyClass::SetParameters, py::arg("mat"), py::arg("regclass"))
          .def("set_bclass2xforms", &PyClass::set_bclass2xforms, py::arg("in"));
    }
    {

      using PyClass = RegtreeFmllrDiagGmmAccs;
      auto regtree_fmllr_diag_gmm_accs = py::class_<PyClass>(
          m, "RegtreeFmllrDiagGmmAccs",
          "Class for computing the accumulators needed for the maximum-likelihood "
          "estimate of FMLLR transforms for an acoustic model that uses diagonal "
          "Gaussian mixture models as emission densities.");
      regtree_fmllr_diag_gmm_accs
          .def(py::init<>())
          .def("Init", &PyClass::Init,
               py::arg("num_bclass"),
               py::arg("dim"))
          .def("SetZero", &PyClass::SetZero)
          .def("AccumulateForGmm", &PyClass::AccumulateForGmm,
               "Accumulate stats for a single GMM in the model; returns log likelihood. "
               "This does not work if the features have already been transformed "
               "with multiple feature transforms (so you can't use use this to "
               "do a 2nd pass of regression-tree fMLLR estimation, which as I write "
               "(Dan, 2016) I'm not sure that this framework even supports.",
               py::arg("regtree"),
               py::arg("am"),
               py::arg("data"),
               py::arg("pdf_index"),
               py::arg("weight"))
          .def("AccumulateForGaussian", &PyClass::AccumulateForGaussian,
               "Accumulate stats for a single Gaussian component in the model.",
               py::arg("regtree"),
               py::arg("am"),
               py::arg("data"),
               py::arg("pdf_index"),
               py::arg("gauss_index"),
               py::arg("weight"))
          .def("Update", &PyClass::Update,
               py::arg("regtree"),
               py::arg("opts"),
               py::arg("out_fmllr"),
               py::arg("auxf_impr"),
               py::arg("tot_t"))
          .def("Read", &PyClass::Read,
               py::arg("in_stream"), py::arg("binary"), py::arg("add"),
      py::call_guard<py::gil_scoped_release>())
          .def("Write", &PyClass::Write,
               py::arg("out_stream"), py::arg("binary"),
      py::call_guard<py::gil_scoped_release>())
          .def("Dim", &PyClass::Dim)
          .def("NumBaseClasses", &PyClass::NumBaseClasses)
          .def("baseclass_stats", &PyClass::baseclass_stats);
    }
}

void pybind_regtree_mllr_diag_gmm(py::module &m) {

  py::class_<RegtreeMllrOptions>(m, "ProcessPitchOptions")
      .def(py::init<>())
      .def_readwrite("min_count", &RegtreeMllrOptions::min_count)
      .def_readwrite("use_regtree", &RegtreeMllrOptions::use_regtree);
    {

      using PyClass = RegtreeMllrDiagGmm;
      auto regtree_mllr_diag_gmm = py::class_<PyClass>(
          m, "RegtreeMllrDiagGmm",
          "Class for computing the accumulators needed for the maximum-likelihood "
          "estimate of FMLLR transforms for an acoustic model that uses diagonal "
          "Gaussian mixture models as emission densities.");
      regtree_mllr_diag_gmm
          .def(py::init<>())
          .def("Init", &PyClass::Init,
               py::arg("num_xforms"),
               py::arg("dim"))
          .def("SetUnit", &PyClass::SetUnit)
          .def("TransformModel", &PyClass::TransformModel,
               py::arg("regtree"),
               py::arg("am"))
          .def("GetTransformedMeans", &PyClass::GetTransformedMeans,
               "Get all the transformed means for a given pdf.",
               py::arg("regtree"),
               py::arg("am"),
               py::arg("pdf_index"),
               py::arg("out"))
          .def("Read", &PyClass::Read,
               py::arg("in_stream"), py::arg("binary"),
      py::call_guard<py::gil_scoped_release>())
          .def("Write", &PyClass::Write,
               py::arg("out_stream"), py::arg("binary"),
      py::call_guard<py::gil_scoped_release>())
          .def("SetParameters", &PyClass::SetParameters,
               py::arg("mat"),
               py::arg("regclass"))
          .def("set_bclass2xforms", &PyClass::set_bclass2xforms,
               py::arg("in"));
    }
    {

      using PyClass = RegtreeMllrDiagGmmAccs;
      auto regtree_mllr_diag_gmm_accs = py::class_<PyClass>(
          m, "RegtreeMllrDiagGmmAccs",
          "Class for computing the maximum-likelihood estimates of the parameters of "
          "an acoustic model that uses diagonal Gaussian mixture models as emission "
          "densities.");
      regtree_mllr_diag_gmm_accs
          .def(py::init<>())
          .def("Init", &PyClass::Init,
               py::arg("num_bclass"),
               py::arg("dim"))
          .def("SetZero", &PyClass::SetZero)
          .def("AccumulateForGmm", &PyClass::AccumulateForGmm,
               "Accumulate stats for a single GMM in the model; returns log likelihood. "
          "This does not work with multiple feature transforms.",
               py::arg("regtree"),
               py::arg("am"),
               py::arg("data"),
               py::arg("pdf_index"),
               py::arg("weight"))
          .def("AccumulateForGaussian", &PyClass::AccumulateForGaussian,
               "Accumulate stats for a single Gaussian component in the model.",
               py::arg("regtree"),
               py::arg("am"),
               py::arg("data"),
               py::arg("pdf_index"),
               py::arg("gauss_index"),
               py::arg("weight"))
          .def("Update", &PyClass::Update,
               py::arg("regtree"),
               py::arg("opts"),
               py::arg("out_mllr"),
               py::arg("auxf_impr"),
               py::arg("t"))
          .def("Read", &PyClass::Read,
               py::arg("in_stream"), py::arg("binary"), py::arg("add"),
      py::call_guard<py::gil_scoped_release>())
          .def("Write", &PyClass::Write,
               py::arg("out_stream"), py::arg("binary"),
      py::call_guard<py::gil_scoped_release>())
          .def("Dim", &PyClass::Dim)
          .def("NumBaseClasses", &PyClass::NumBaseClasses)
          .def("baseclass_stats", &PyClass::baseclass_stats);
    }
}

void pybind_transform_common(py::module &m) {

    {

      using PyClass = AffineXformStats;
      auto affine_xform_stats = py::class_<PyClass>(
          m, "AffineXformStats");
      affine_xform_stats
          .def(py::init<>())
          .def(py::init<const AffineXformStats &>(),
               py::arg("other"))
          .def_readwrite("beta_", &PyClass::beta_)
          .def_readwrite("K_", &PyClass::K_)
          .def_readwrite("G_", &PyClass::G_)
          .def_readwrite("dim_", &PyClass::dim_)
          .def("Init", &PyClass::Init,
               py::arg("num_bclass"),
               py::arg("dim"))
          .def("Dim", &PyClass::Dim)
          .def("SetZero", &PyClass::SetZero)
          .def("CopyStats", &PyClass::CopyStats,
               py::arg("other"))
          .def("Add", &PyClass::Add,
      py::call_guard<py::gil_scoped_release>(),
               py::arg("other"))
          .def("Read", &PyClass::Read,
               py::arg("in_stream"), py::arg("binary"), py::arg("add"),
      py::call_guard<py::gil_scoped_release>())
          .def("Write", &PyClass::Write,
               py::arg("out_stream"), py::arg("binary"),
      py::call_guard<py::gil_scoped_release>());
    }
  m.def("ComposeTransforms",
        &ComposeTransforms,
        py::arg("a"),
        py::arg("b"),
        py::arg("b_is_affine"),
        py::arg("c"));
  m.def("compose_transforms",
          [](const Matrix<BaseFloat> &a, const Matrix<BaseFloat> &b,
                       bool b_is_affine){
          py::gil_scoped_release gil_release;
                Matrix<BaseFloat> c;
                ComposeTransforms(a, b, b_is_affine, &c);
                return c;
                       },
        py::arg("a"),
        py::arg("b"),
        py::arg("b_is_affine"));
  m.def("ApplyAffineTransform",
        &ApplyAffineTransform,
        "Applies the affine transform 'xform' to the vector 'vec' and overwrites the "
          "contents of 'vec'.",
        py::arg("xform"),
        py::arg("vec"));
}

void init_transform(py::module &_m) {
     py::module m = _m.def_submodule("transform", "transform pybind for Kaldi");

     pybind_transform_common(m);
     pybind_basis_fmllr_diag_gmm(m);
     pybind_cmvn(m);
     pybind_compressed_transform_stats(m);
     pybind_decodable_am_diag_gmm_regtree(m);
     pybind_fmllr_diag_gmm(m);
     pybind_fmllr_raw(m);
     pybind_fmpe(m);
     pybind_lda_estimate(m);
     pybind_lvtln(m);
     pybind_mllt(m);
     pybind_regression_tree(m);
     pybind_regtree_fmllr_diag_gmm(m);
     pybind_regtree_mllr_diag_gmm(m);

     m.def("accumulate_from_alignment",
               [](
                         const TransitionModel &trans_model,
                         const AmDiagGmm &am_gmm,
                         const Matrix<BaseFloat> &feats,
                         const std::vector<int32> &ali,
                       const ConstIntegerSet<int32> &silence_set,
                            FmllrDiagGmmAccs *spk_stats,
                       BaseFloat silence_scale,
                         BaseFloat rand_prune = 0.0,
                       bool distributed = false,
                       bool two_models = false
               ){
          py::gil_scoped_release gil_release;
               Posterior pdf_post;
               Posterior post;
               AlignmentToPosterior(ali, &post);
               if (distributed)
               WeightSilencePostDistributed(trans_model, silence_set,
                                             silence_scale, &post);
               else
                    WeightSilencePost(trans_model, silence_set,
                              silence_scale, &post);
                    ConvertPosteriorToPdfs(trans_model, post, &pdf_post);
                    if (!two_models){
                         for (size_t i = 0; i < pdf_post.size(); i++) {
                         for (size_t j = 0; j < pdf_post[i].size(); j++) {
                              int32 pdf_id = pdf_post[i][j].first;
                              spk_stats->AccumulateForGmm(am_gmm.GetPdf(pdf_id),
                                                       feats.Row(i),
                                                       pdf_post[i][j].second);
                         }
                         }
                    }
                    else{


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
                         for (size_t i = 0; i < gpost.size(); i++) {
                              for (size_t j = 0; j < gpost[i].size(); j++) {
                                   int32 pdf_id = gpost[i][j].first;
                                   const Vector<BaseFloat> & posterior(gpost[i][j].second);

                                   spk_stats->AccumulateFromPosteriors(am_gmm.GetPdf(pdf_id),
                                                                      feats.Row(i), posterior);

                              }
                              }
                    }
          },
               py::arg("trans_model"),
               py::arg("am_gmm"),
               py::arg("feats"),
               py::arg("ali"),
               py::arg("silence_set"),
               py::arg("spk_stats"),
               py::arg("silence_scale"),
               py::arg("rand_prune") = 0.0,
               py::arg("distributed") = false,
               py::arg("two_models") = false);

          m.def("compute_fmllr_transform",
               [](FmllrDiagGmmAccs* f, int32 dim,
               const FmllrOptions &fmllr_opts){
          py::gil_scoped_release gil_release;

                    BaseFloat impr, tot_t;
                    Matrix<BaseFloat> transform(dim, dim+1);
                    transform.SetUnit();
                    f->Update(fmllr_opts, &transform, &impr, &tot_t);
                    return transform;
               },
               py::arg("f"),
               py::arg("am_gmm"),
               py::arg("fmllr_opts"));
}
