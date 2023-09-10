
#include "ivector/pybind_ivector.h"
#include "ivector/agglomerative-clustering.h"
#include "ivector/ivector-extractor.h"
#include "ivector/logistic-regression.h"
#include "ivector/plda.h"
#include "ivector/voice-activity-detection.h"
#include "base/kaldi-math.h"

using namespace kaldi;

void pybind_agglomerative_clustering(py::module &m) {

  {
    using PyClass = AhcCluster;

    auto ahc_cluster = py::class_<PyClass>(
        m, "AhcCluster");
    ahc_cluster.def(py::init<int32, int32, int32, std::vector<int32>>(),
        py::arg("id"),
        py::arg("p1"),
        py::arg("p2"),
        py::arg("utts"));
  }
  {
    using PyClass = AgglomerativeClusterer;

    auto agglomerative_clusterer = py::class_<PyClass>(
        m, "AgglomerativeClusterer");
    agglomerative_clusterer.def(py::init<
                const Matrix<BaseFloat> &,
                BaseFloat,
                int32,
                int32,
                BaseFloat,
                std::vector<int32> *>(),
        py::arg("costs"),
        py::arg("threshold"),
        py::arg("min_clusters"),
        py::arg("first_pass_max_points"),
        py::arg("max_cluster_fraction"),
        py::arg("assignments_out"))
      .def("Cluster",
        &PyClass::Cluster,
        "Clusters points. Chooses single pass or two pass algorithm.")
      .def("ClusterSinglePass",
        &PyClass::ClusterSinglePass,
        "Clusters points using single pass algorithm.")
      .def("ClusterTwoPass",
        &PyClass::ClusterTwoPass,
        "Clusters points using two pass algorithm.");
  }
  m.def("AgglomerativeCluster",
        &AgglomerativeCluster,
        "This is the function that is called to perform the agglomerative "
        "clustering. It takes the following arguments: "
        " - A matrix of all pairwise costs, with each row/column corresponding "
        "    to an utterance ID, and the elements of the matrix containing the "
        "    cost for pairing the utterances for its row and column "
        " - A threshold which is used as the stopping criterion for the clusters "
        " - A minimum number of clusters that will not be merged past "
        " - A maximum fraction of points that can be in a cluster "
        " - A vector which will be filled with integer IDs corresponding to each "
        "    of the rows/columns of the score matrix. "
        "\n"
        "The basic algorithm is as follows: "
        "\\code "
        "    while (num-clusters > min-clusters && smallest-merge-cost <= threshold) "
        "        if (size-of-new-cluster <= max-cluster-size) "
        "            merge the two clusters with lowest cost "
        "\\endcode "
        "\n"
        "The cost between two clusters is the average cost of all pairwise "
        "costs between points across the two clusters. "
        "\n"
        "The algorithm takes advantage of the fact that the sum of the pairwise "
        "costs between the points of clusters I and J is equiavlent to the "
        "sum of the pairwise costs between cluster I and the parents of cluster "
        "J. In other words, the total cost between I and J is the sum of the "
        "costs between clusters I and M and clusters I and N, where "
        "cluster J was formed by merging clusters M and N. "
        "\n"
        "If the number of points to cluster is larger than first-pass-max-points, "
        "then clustering is done in two passes. In the first pass, input points are "
        "divided into contiguous subsets of size at most first-pass-max-points and "
        "each subset is clustered separately. In the second pass, the first pass "
        "clusters are merged into the final set of clusters.",
        py::arg("costs"),
        py::arg("threshold"),
        py::arg("min_clusters"),
        py::arg("first_pass_max_points"),
        py::arg("max_cluster_fraction"),
        py::arg("assignments_out"));
}

void pybind_ivector_extractor(py::module &m) {

  {
    using PyClass = IvectorEstimationOptions;

    auto ivector_estimation_options = py::class_<PyClass>(
        m, "IvectorEstimationOptions");
    ivector_estimation_options.def(py::init<>())
      .def_readwrite("acoustic_weight", &PyClass::acoustic_weight)
      .def_readwrite("max_count", &PyClass::max_count)
      .def(py::pickle(
        [](const PyClass &p) { // __getstate__
            /* Return a tuple that fully encodes the state of the object */
            return py::make_tuple(
                p.acoustic_weight,
                p.max_count);
        },
        [](py::tuple t) { // __setstate__
            if (t.size() != 2)
                throw std::runtime_error("Invalid state!");

            /* Create a new C++ instance */
            PyClass opts;

            /* Assign any additional state */
            opts.acoustic_weight = t[0].cast<double>();
            opts.max_count = t[1].cast<double>();

            return opts;
        }
    ));
  }

  {
    using PyClass = IvectorExtractorUtteranceStats;

    auto ivector_extractor_utterance_stats = py::class_<PyClass>(
        m, "IvectorExtractorUtteranceStats");
    ivector_extractor_utterance_stats.def(py::init<int32, int32,
                                 bool >(),
        py::arg("num_gauss"),
        py::arg("feat_dim"),
        py::arg("need_2nd_order_stats"))
      .def("AccStats",
        &PyClass::AccStats,
        py::arg("feats"),
        py::arg("post"))
      .def("Scale",
        &PyClass::Scale,
        py::arg("scale"))
      .def("NumFrames",
        &PyClass::NumFrames);
  }

  {
    using PyClass = IvectorExtractorOptions;

    auto ivector_extractor_options = py::class_<PyClass>(
        m, "IvectorExtractorOptions");
    ivector_extractor_options.def(py::init<>())
      .def_readwrite("ivector_dim", &PyClass::ivector_dim)
      .def_readwrite("num_iters", &PyClass::num_iters)
      .def_readwrite("use_weights", &PyClass::use_weights)
      .def(py::pickle(
        [](const PyClass &p) { // __getstate__
            /* Return a tuple that fully encodes the state of the object */
            return py::make_tuple(
                p.ivector_dim,
                p.num_iters,
                p.use_weights);
        },
        [](py::tuple t) { // __setstate__
            if (t.size() != 2)
                throw std::runtime_error("Invalid state!");

            /* Create a new C++ instance */
            PyClass opts;

            /* Assign any additional state */
            opts.ivector_dim = t[0].cast<int>();
            opts.num_iters = t[1].cast<int>();
            opts.use_weights = t[1].cast<bool>();

            return opts;
        }
    ));
  }

  {
    using PyClass = IvectorExtractor;

    auto ivector_extractor = py::class_<PyClass>(
        m, "IvectorExtractor");
    ivector_extractor.def(py::init<>())
      .def(py::init<const IvectorExtractorOptions &,
      const FullGmm &>(),
        py::arg("opts"),
        py::arg("fgmm"))
      .def("GetIvectorDistribution",
        &PyClass::GetIvectorDistribution,
        "Gets the distribution over ivectors (or at least, a Gaussian approximation "
        "to it).  The output \"var\" may be NULL if you don't need it.  \"mean\", and "
        "\"var\", if present, must be the correct dimension (this->IvectorDim()). "
        "If you only need a point estimate of the iVector, get the mean only.",
        py::arg("utt_stats"),
        py::arg("mean"),
        py::arg("var"))
      .def("PriorOffset",
        &PyClass::PriorOffset)
      .def("GetAuxf",
        &PyClass::GetAuxf,
        "Returns the log-likelihood objective function, summed over frames, "
        "for this distribution of iVectors (a point distribution, if var == NULL).",
        py::arg("utt_stats"),
        py::arg("mean"),
        py::arg("var"))
      .def("GetAcousticAuxf",
        &PyClass::GetAcousticAuxf,
        "Returns the data-dependent part of the log-likelihood objective function, "
        "summed over frames.  If variance pointer is NULL, uses point value.",
        py::arg("utt_stats"),
        py::arg("mean"),
        py::arg("var") = NULL)
      .def("GetPriorAuxf",
        &PyClass::GetPriorAuxf,
        "Returns the prior-related part of the log-likelihood objective function. "
        "Note: if var != NULL, this quantity is a *probability*, otherwise it is "
        "a likelihood (and the corresponding probability is zero).",
        py::arg("mean"),
        py::arg("var") = NULL)
      .def("GetAcousticAuxfVariance",
        &PyClass::GetAcousticAuxfVariance,
        "This returns just the part of the acoustic auxf that relates to the "
        "variance of the utt_stats (i.e. which would be zero if the utt_stats had "
        "zero variance.  This does not depend on the iVector, it's included as an "
        "aid to debugging.  We can only get this if we stored the S statistics.  If "
        "not we assume the variance is generated from the model.",
        py::arg("utt_stats"))
      .def("GetAcousticAuxfMean",
        &PyClass::GetAcousticAuxfMean,
        "This returns just the part of the acoustic auxf that relates to the "
        "variance of the utt_stats (i.e. which would be zero if the utt_stats had "
        "zero variance.  This does not depend on the iVector, it's included as an "
        "aid to debugging.  We can only get this if we stored the S statistics.  If "
        "not we assume the variance is generated from the model.",
        py::arg("utt_stats"),
        py::arg("mean"),
        py::arg("var") = NULL)
      .def("GetAcousticAuxfGconst",
        &PyClass::GetAcousticAuxfGconst,
        "This returns the part of the acoustic auxf that relates to the "
        "gconsts of the Gaussians.",
        py::arg("utt_stats"))
      .def("GetAcousticAuxfWeight",
        &PyClass::GetAcousticAuxfWeight,
        "This returns the part of the acoustic auxf that relates to the "
        "Gaussian-specific weights.  (impacted by the iVector only if "
        "we are using w_).",
        py::arg("utt_stats"),
        py::arg("mean"),
        py::arg("var") = NULL)
      .def("GetIvectorDistMean",
        &PyClass::GetIvectorDistMean,
        "Gets the linear and quadratic terms in the distribution over iVectors, but "
        "only the terms arising from the Gaussian means (i.e. not the weights "
        "or the priors). "
        "Setup is log p(x) \\propto x^T linear -0.5 x^T quadratic x. "
        "This function *adds to* the output rather than setting it.",
        py::arg("utt_stats"),
        py::arg("linear"),
        py::arg("quadratic"))
      .def("GetIvectorDistPrior",
        &PyClass::GetIvectorDistPrior,
        "Gets the linear and quadratic terms in the distribution over "
        "iVectors, that arise from the prior.  Adds to the outputs, "
        "rather than setting them.",
        py::arg("utt_stats"),
        py::arg("linear"),
        py::arg("quadratic"))
      .def("GetIvectorDistWeight",
        &PyClass::GetIvectorDistWeight,
        "Gets the linear and quadratic terms in the distribution over "
        "iVectors, that arise from the weights (if applicable).  The "
        "\"mean\" parameter is the iVector point that we compute "
        "the expansion around (it's a quadratic approximation of a "
        "nonlinear function, but with a \"safety factor\" (the \"max\" stuff). "
        "Adds to the outputs, rather than setting them.",
        py::arg("utt_stats"),
        py::arg("mean"),
        py::arg("linear"),
        py::arg("quadratic") = NULL)
      .def("FeatDim",
        &PyClass::FeatDim)
      .def("IvectorDim",
        &PyClass::IvectorDim)
      .def("NumGauss",
        &PyClass::NumGauss)
      .def("IvectorDependentWeights",
        &PyClass::IvectorDependentWeights)
      .def("Read", &PyClass::Read,
        py::arg("is"),
        py::arg("binary"),
      py::call_guard<py::gil_scoped_release>())
      .def("Write", &PyClass::Write,
        py::arg("os"),
        py::arg("binary"),
      py::call_guard<py::gil_scoped_release>());
  }

  {
    using PyClass = OnlineIvectorEstimationStats;

    auto online_ivector_estimation_stats = py::class_<PyClass>(
        m, "OnlineIvectorEstimationStats");
    online_ivector_estimation_stats.def(py::init<int32,
                               BaseFloat,
                               BaseFloat>(),
        py::arg("ivector_dim"),
        py::arg("prior_offset"),
        py::arg("max_count"))
      .def(py::init<const OnlineIvectorEstimationStats &>(),
        py::arg("other"))
      .def("AccStats",
        py::overload_cast<const IvectorExtractor &,
                const VectorBase<BaseFloat> &,
                const std::vector<std::pair<int32, BaseFloat> > &>(&PyClass::AccStats),
        "Accumulate stats for one frame.",
        py::arg("extractor"),
        py::arg("feature"),
        py::arg("gauss_post"))
        .def("AccStats",
        py::overload_cast<const IvectorExtractor &,
                const MatrixBase<BaseFloat> &,
                const std::vector<std::vector<std::pair<int32, BaseFloat> > > &>(&PyClass::AccStats),
        "Accumulate stats for a sequence (or collection) of frames.",
        py::arg("extractor"),
        py::arg("features"),
        py::arg("gauss_post"))
      .def("IvectorDim",
        &PyClass::IvectorDim)
      .def("GetIvector",
        &PyClass::GetIvector,
        py::arg("num_cg_iters"),
        py::arg("ivector"))
      .def("NumFrames",
        &PyClass::NumFrames)
      .def("PriorOffset",
        &PyClass::PriorOffset)
      .def("ObjfChange",
        &PyClass::ObjfChange,
        py::arg("ivector"))
      .def("Count",
        &PyClass::Count)
      .def("Scale",
        &PyClass::Scale,
        py::arg("scale"))
      .def("Read", &PyClass::Read,
        py::arg("is"),
        py::arg("binary"),
      py::call_guard<py::gil_scoped_release>())
      .def("Write", &PyClass::Write,
        py::arg("os"),
        py::arg("binary"),
      py::call_guard<py::gil_scoped_release>());
  }
  m.def("EstimateIvectorsOnline",
        &EstimateIvectorsOnline,
        "This code obtains periodically (for each \"ivector_period\" frames, e.g. 10 "
        "frames), an estimate of the iVector including all frames up to that point. "
        "This emulates what you could do in an online/streaming algorithm; its use is "
        "for neural network training in a way that's matched to online decoding. "
        "[note: I don't believe we are currently using the program, "
        "ivector-extract-online.cc, that calls this function, in any of the scripts.]. "
        "Caution: this program outputs the raw iVectors, where the first component "
        "will generally be very positive.  You probably want to subtract PriorOffset() "
        "from the first element of each row of the output before writing it out. "
        "For num_cg_iters, we suggest 15.  It can be a positive number (more -> more "
        "exact, less -> faster), or if it's negative it will do the optimization "
        "exactly each time which is slower. "
        "It returns the objective function improvement per frame from the \"default\" iVector to "
        "the last iVector estimated.",
        py::arg("feats"),
        py::arg("post"),
        py::arg("extractor"),
        py::arg("ivector_period"),
        py::arg("num_cg_iters"),
        py::arg("max_count"),
        py::arg("ivectors"));


  {
    using PyClass = IvectorExtractorStatsOptions;

    auto ivector_extractor_stats_options = py::class_<PyClass>(
        m, "IvectorExtractorStatsOptions");
    ivector_extractor_stats_options.def(py::init<>())
      .def_readwrite("update_variances", &PyClass::update_variances)
      .def_readwrite("compute_auxf", &PyClass::compute_auxf)
      .def_readwrite("num_samples_for_weights", &PyClass::num_samples_for_weights)
      .def_readwrite("cache_size", &PyClass::cache_size)
      .def(py::pickle(
        [](const PyClass &p) { // __getstate__
            /* Return a tuple that fully encodes the state of the object */
            return py::make_tuple(
                p.update_variances,
                p.compute_auxf,
                p.num_samples_for_weights,
                p.cache_size);
        },
        [](py::tuple t) { // __setstate__
            if (t.size() != 4)
                throw std::runtime_error("Invalid state!");

            /* Create a new C++ instance */
            PyClass opts;

            /* Assign any additional state */
            opts.update_variances = t[0].cast<bool>();
            opts.compute_auxf = t[1].cast<bool>();
            opts.num_samples_for_weights = t[2].cast<int32>();
            opts.cache_size = t[3].cast<bool>();

            return opts;
        }
    ));
  }
  {
    using PyClass = IvectorExtractorEstimationOptions;

    auto ivector_extractor_estimation_options = py::class_<PyClass>(
        m, "IvectorExtractorEstimationOptions");
    ivector_extractor_estimation_options.def(py::init<>())
      .def_readwrite("variance_floor_factor", &PyClass::variance_floor_factor)
      .def_readwrite("gaussian_min_count", &PyClass::gaussian_min_count)
      .def_readwrite("num_threads", &PyClass::num_threads)
      .def_readwrite("diagonalize", &PyClass::diagonalize)
      .def(py::pickle(
        [](const PyClass &p) { // __getstate__
            /* Return a tuple that fully encodes the state of the object */
            return py::make_tuple(
                p.variance_floor_factor,
                p.gaussian_min_count,
                p.num_threads,
                p.diagonalize);
        },
        [](py::tuple t) { // __setstate__
            if (t.size() != 4)
                throw std::runtime_error("Invalid state!");

            /* Create a new C++ instance */
            PyClass opts;

            /* Assign any additional state */
            opts.variance_floor_factor = t[0].cast<double>();
            opts.gaussian_min_count = t[1].cast<double>();
            opts.num_threads = t[2].cast<int32>();
            opts.diagonalize = t[3].cast<bool>();

            return opts;
        }
    ));
  }
  {
    using PyClass = IvectorExtractorStats;

    auto ivector_extractor_stats = py::class_<PyClass>(
        m, "IvectorExtractorStats");
    ivector_extractor_stats.def(py::init<>())
      .def(py::init<const IvectorExtractor &,
                        const IvectorExtractorStatsOptions &>(),
        py::arg("extractor"),
        py::arg("stats_opts"))
      .def("Add",
        &PyClass::Add,
        py::arg("other"))
      .def("AccStatsForUtterance",
        py::overload_cast<const IvectorExtractor &,
                            const MatrixBase<BaseFloat> &,
                            const Posterior &>(&PyClass::AccStatsForUtterance),
        py::arg("extractor"),
        py::arg("feats"),
        py::arg("post"),
      py::call_guard<py::gil_scoped_release>())
      .def("AccStatsForUtterance",
        py::overload_cast<const IvectorExtractor &,
                              const MatrixBase<BaseFloat> &,
                              const FullGmm &>(&PyClass::AccStatsForUtterance),
        py::arg("extractor"),
        py::arg("feats"),
        py::arg("fgmm"),
      py::call_guard<py::gil_scoped_release>())
      .def("Read", &PyClass::Read,
        py::arg("is"),
        py::arg("binary"),
        py::arg("add") = false,
      py::call_guard<py::gil_scoped_release>())
      .def("Write",
      py::overload_cast<std::ostream &, bool>(&PyClass::Write),
        py::arg("os"),
        py::arg("binary"),
      py::call_guard<py::gil_scoped_release>())
      .def("Update", &PyClass::Update,
      "Returns the objf improvement per frame.",
        py::arg("opts"),
        py::arg("extractor"),
      py::call_guard<py::gil_scoped_release>())
      .def("update", [](
        PyClass &stats,
        IvectorExtractor &extractor,
        double variance_floor_factor=0.1,
        double gaussian_min_count=100.0,
        int32 num_threads=1,
        bool diagonalize=true
      ){

      IvectorExtractorEstimationOptions update_opts;
      update_opts.variance_floor_factor = variance_floor_factor;
      update_opts.gaussian_min_count = gaussian_min_count;
      update_opts.num_threads = num_threads;
      update_opts.diagonalize = diagonalize;
      stats.Update(update_opts, &extractor);
      },
      "Returns the objf improvement per frame.",
        py::arg("extractor"),
        py::arg("variance_floor_factor")=0.1,
        py::arg("gaussian_min_count")=100.0,
        py::arg("num_threads")=1,
        py::arg("diagonalize")=true)
      .def("AuxfPerFrame",
        &PyClass::AuxfPerFrame)
      .def("IvectorVarianceDiagnostic",
        &PyClass::IvectorVarianceDiagnostic,
        "Prints the proportion of the variance explained by "
        "the Ivector model versus the Gaussians",
        py::arg("extractor"));
  }
}

void pybind_logistic_regression(py::module &m) {

  {
    using PyClass = LogisticRegressionConfig;

    auto logistic_regression_config = py::class_<PyClass>(
        m, "LogisticRegressionConfig");
    logistic_regression_config.def(py::init<>())
      .def_readwrite("max_steps", &PyClass::max_steps)
      .def_readwrite("mix_up", &PyClass::mix_up)
      .def_readwrite("normalizer", &PyClass::normalizer)
      .def_readwrite("power", &PyClass::power)
      .def(py::pickle(
        [](const PyClass &p) { // __getstate__
            /* Return a tuple that fully encodes the state of the object */
            return py::make_tuple(
                p.max_steps,
                p.mix_up,
                p.normalizer,
                p.power);
        },
        [](py::tuple t) { // __setstate__
            if (t.size() != 4)
                throw std::runtime_error("Invalid state!");

            /* Create a new C++ instance */
            PyClass opts;

            /* Assign any additional state */
            opts.max_steps = t[0].cast<int32>();
            opts.mix_up = t[1].cast<int32>();
            opts.normalizer = t[2].cast<double>();
            opts.power = t[2].cast<double>();

            return opts;
        }
    ));
  }
  {
    using PyClass = LogisticRegression;

    auto logistic_regression = py::class_<PyClass>(
        m, "LogisticRegression");
    logistic_regression.def(py::init<>())
      .def("Train",
        &PyClass::Train,
        "xs and ys are the training data. Each row of xs is a vector "
        "corresponding to the class label in the same row of ys.",
        py::arg("xs"),
        py::arg("ys"),
        py::arg("conf"))
      .def("GetLogPosteriors",
        py::overload_cast<const Matrix<BaseFloat> &,
                        Matrix<BaseFloat> *>(&PyClass::GetLogPosteriors),
        "Calculates the log posterior of the class label given the input xs. "
        "The rows of log_posteriors corresponds to the rows of xs: the "
        "individual data points to be evaluated. The columns of "
        "log_posteriors are the integer class labels.",
        py::arg("xs"),
        py::arg("log_posteriors"))
      .def("GetLogPosteriors",
        py::overload_cast<const Vector<BaseFloat> &,
                        Vector<BaseFloat> *>(&PyClass::GetLogPosteriors),
        "Calculates the log posterior of the class label given the input x. "
        "The indices of log_posteriors are the class labels.",
        py::arg("x"),
        py::arg("log_posteriors"))
      .def("Read", &PyClass::Read,
        py::arg("is"),
        py::arg("binary"),
      py::call_guard<py::gil_scoped_release>())
      .def("Write", &PyClass::Write,
        py::arg("os"),
        py::arg("binary"))
      .def("ScalePriors", &PyClass::ScalePriors,
        py::arg("prior_scales"),
      py::call_guard<py::gil_scoped_release>());
  }
}

void pybind_plda(py::module &m) {

  {
    using PyClass = PldaConfig;

    auto plda_config = py::class_<PyClass>(
        m, "PldaConfig");
    plda_config.def(py::init<>())
      .def_readwrite("normalize_length", &PyClass::normalize_length)
      .def_readwrite("simple_length_norm", &PyClass::simple_length_norm)
      .def(py::pickle(
        [](const PyClass &p) { // __getstate__
            /* Return a tuple that fully encodes the state of the object */
            return py::make_tuple(
                p.normalize_length,
                p.simple_length_norm);
        },
        [](py::tuple t) { // __setstate__
            if (t.size() != 2)
                throw std::runtime_error("Invalid state!");

            /* Create a new C++ instance */
            PyClass opts;

            /* Assign any additional state */
            opts.normalize_length = t[0].cast<bool>();
            opts.simple_length_norm = t[1].cast<bool>();

            return opts;
        }
    ));
  }
  {
    using PyClass = Plda;

    auto plda = py::class_<PyClass>(
        m, "Plda");
    plda.def(py::init<>())
      .def(py::init<const Plda &>(),
        py::arg("other"))
      .def("transform_ivector",
          [](
            PyClass &plda,
                             const Vector<double> &ivector,
                             int32 num_examples
          ){
          py::gil_scoped_release gil_release;
          PldaConfig plda_config;
          Vector<double> *transformed_ivector = new Vector<double>(plda.Dim());
          plda.TransformIvector(plda_config, ivector,
                                                      num_examples,
                                                      transformed_ivector);
          return transformed_ivector;
          },
        py::arg("ivector"),
        py::arg("num_examples")
        )
      .def("transform_ivector",
          [](
            PyClass &plda,
                             py::array_t<double> ivector,
                             int32 num_examples
          ){
          py::gil_scoped_release gil_release;
          PldaConfig plda_config;
          Vector<double> *transformed_ivector = new Vector<double>(plda.Dim());
          Vector<double> ivector_dbl;
          auto r_one = ivector.unchecked<1>();
          ivector_dbl.Resize(r_one.shape(0));
          for (py::size_t i = 0; i < r_one.shape(0); i++)
            ivector_dbl(i) = r_one(i);
          plda.TransformIvector(plda_config, ivector_dbl,
                                                      num_examples,
                                                      transformed_ivector);
          return transformed_ivector;
          },
        py::arg("ivector"),
        py::arg("num_examples")
        )
      .def("classify_utterance",
      [](
            PyClass &plda,
            const Vector<double> &utterance_ivector,
            const std::vector<Vector<double>> &speaker_ivectors,
            const std::vector<int32> &speaker_counts
      ){
          py::gil_scoped_release gil_release;
        Vector<float> scores;
        scores.Resize(speaker_counts.size());
        for (int32 i = 0; i < speaker_counts.size(); i++) {
          scores(i) = plda.LogLikelihoodRatio(speaker_ivectors[i],
                                                speaker_counts[i],
                                                utterance_ivector);
      }
      int32 index;
      BaseFloat score = scores.Max(&index);
      return std::make_pair(index, score);
      },
        py::arg("utterance_ivector"),
        py::arg("speaker_ivectors"),
        py::arg("speaker_counts")
      )
      .def("classify_utterance",
      [](
            PyClass &plda,
            py::array_t<double> utterance_ivector,
            py::array_t<double> speaker_ivectors,
            py::array_t<int32> speaker_counts
      ){
          py::gil_scoped_release gil_release;
        Vector<double> utterance_ivector_dbl;
        auto r_one = utterance_ivector.unchecked<1>();
        utterance_ivector_dbl.Resize(r_one.shape(0));
        for (py::size_t i = 0; i < r_one.shape(0); i++)
          utterance_ivector_dbl(i) = r_one(i);

        Vector<float> scores;
        auto r_counts = speaker_counts.unchecked<1>();
        scores.Resize(r_counts.shape(0));

        auto r_speaker = speaker_ivectors.unchecked<2>();

        for (int32 i = 0; i < r_counts.shape(0); i++) {
          Vector<double> speaker_ivector_dbl;
          speaker_ivector_dbl.Resize(r_one.shape(0));
          for (py::size_t j = 0; j < r_one.shape(0); j++)
            speaker_ivector_dbl(j) = r_speaker(i, j);

          scores(i) = plda.LogLikelihoodRatio(speaker_ivector_dbl,
                                                r_counts(i),
                                                utterance_ivector_dbl);
      }
      int32 index;
      BaseFloat score = scores.Max(&index);
      return std::make_pair(index, score);
      },
        py::arg("utterance_ivector"),
        py::arg("speaker_ivectors"),
        py::arg("speaker_counts")
      )
      .def("generate_affinity_matrix",
      [](
            PyClass &plda,
            const std::vector<Vector<double>> &ivectors
      ){
          py::gil_scoped_release gil_release;
        Matrix<float> scores;
        scores.Resize(ivectors.size(), ivectors.size());
        for (int32 i = 0; i < ivectors.size(); i++) {
          for (int32 j = 0; j < ivectors.size(); j++) {
            scores(i, j) = plda.LogLikelihoodRatio(ivectors[i],
                                                  1,
                                                  ivectors[j]);
          }
        }
      return scores;
      },
        py::arg("ivectors")
      )
      .def("transform_ivectors",
          [](
            PyClass &plda,
            py::array_t<double> ivectors,
            py::array_t<int32> num_examples
          ){
          py::gil_scoped_release gil_release;
          PldaConfig plda_config;

        auto r = ivectors.unchecked<2>();
        auto r_counts = num_examples.unchecked<1>();

          std::vector<const Vector<double> *> transformed_ivectors(r_counts.shape(0));

          for (int32 i = 0; i < r_counts.shape(0); i++) {
            Vector<double> *transformed_ivector = new Vector<double>(plda.Dim());
            Vector<double> ivector;
            ivector.Resize(r.shape(1));
            for (py::size_t j = 0; j < r.shape(1); j++)
              ivector(j) = r(i, j);
            plda.TransformIvector(plda_config, ivector,
                                                        r_counts(i),
                                                        transformed_ivector);
            transformed_ivectors[i] = transformed_ivector;
          }
          return transformed_ivectors;
          },
        py::arg("ivectors"),
        py::arg("num_examples")
        )
      .def("transform_ivectors",
          [](
            PyClass &plda,
            const std::vector<Vector<double>> &ivectors,
            std::vector<int32> num_examples
          ){
          py::gil_scoped_release gil_release;
          PldaConfig plda_config;

          std::vector<const Vector<double> *> transformed_ivectors(ivectors.size());

          for (int32 i = 0; i < ivectors.size(); i++) {
            Vector<double> *transformed_ivector = new Vector<double>(plda.Dim());
            plda.TransformIvector(plda_config, ivectors[i],
                                                        num_examples[i],
                                                        transformed_ivector);
            transformed_ivectors[i] = transformed_ivector;
          }
          return transformed_ivectors;
          },
        py::arg("ivectors"),
        py::arg("num_examples")
        )
      .def("score",
        [](
            PyClass &plda,
            const VectorBase<double> & utterance_ivector,
            const std::vector<Vector<double>> &transformed_enrolled_ivectors,
            std::vector<int32> num_enroll_utts
        ){
          py::gil_scoped_release gil_release;
          PldaConfig plda_config;

          std::vector<BaseFloat> scores;

          for (int32 j = 0; j < transformed_enrolled_ivectors.size(); j++) {
            scores.push_back(plda.LogLikelihoodRatio(utterance_ivector,
                                                  num_enroll_utts[j],
                                                  transformed_enrolled_ivectors[j]));
          }
          return scores;

        },
        py::arg("utterance_ivector"),
        py::arg("transformed_enrolled_ivectors"),
        py::arg("num_enroll_utts"))
      .def("log_likelihood_distance",
        [](
            PyClass &plda,
          py::array_t<double> utterance_one_ivector,
          py::array_t<double> utterance_two_ivector
        ){
          py::gil_scoped_release gil_release;
          Vector<double> ivector_one_dbl;
        auto r_one = utterance_one_ivector.unchecked<1>();
        ivector_one_dbl.Resize(r_one.shape(0));
        for (py::size_t i = 0; i < r_one.shape(0); i++)
          ivector_one_dbl(i) = r_one(i);

          Vector<double> ivector_two_dbl;
        auto r_two = utterance_two_ivector.unchecked<1>();
        ivector_two_dbl.Resize(r_two.shape(0));
        for (py::size_t i = 0; i < r_two.shape(0); i++)
          ivector_two_dbl(i) = r_two(i);

        BaseFloat score = 1.0 / Exp(plda.LogLikelihoodRatio(ivector_one_dbl,
                                                  1,
                                                  ivector_two_dbl));
        return score;
        },
        py::arg("utterance_one_ivector"),
        py::arg("utterance_two_ivector"))
      .def("score",
        [](
            PyClass &plda,
            const VectorBase<float> & utterance_ivector,
            const std::vector<Vector<float>> &transformed_enrolled_ivectors
        ){
          py::gil_scoped_release gil_release;
          PldaConfig plda_config;
          Vector<double> ivector_one_dbl(utterance_ivector);

          std::vector<BaseFloat> scores;

          for (int32 j = 0; j < transformed_enrolled_ivectors.size(); j++) {
            Vector<double> ivector_two_dbl(transformed_enrolled_ivectors[j]);
            scores.push_back(plda.LogLikelihoodRatio(ivector_one_dbl,
                                                  1,
                                                  ivector_two_dbl));
          }
          return scores;

        },
        py::arg("utterance_ivector"),
        py::arg("transformed_enrolled_ivectors"))
      .def("TransformIvector",
        py::overload_cast<const PldaConfig &,
                          const VectorBase<double> &,
                          int32,
                          VectorBase<double> *>(&PyClass::TransformIvector, py::const_),
        "Transforms an iVector into a space where the within-class variance "
        "is unit and between-class variance is diagonalized.  The only "
        "anticipated use of this function is to pre-transform iVectors "
        "before giving them to the function LogLikelihoodRatio (it's "
        "done this way for efficiency because a given iVector may be "
        "used multiple times in LogLikelihoodRatio and we don't want "
        "to repeat the matrix multiplication "
        "\n"
        "If config.normalize_length == true, it will also normalize the iVector's "
        "length by multiplying by a scalar that ensures that ivector^T inv_var "
        "ivector = dim.  In this case, \"num_enroll_examples\" comes into play because it "
        "affects the expected covariance matrix of the iVector.  The normalization "
        "factor is returned, even if config.normalize_length == false, in which "
        "case the normalization factor is computed but not applied. "
        "If config.simple_length_normalization == true, then an alternative "
        "normalization factor is computed that causes the iVector length "
        "to be equal to the square root of the iVector dimension.",
        py::arg("config"),
        py::arg("ivector"),
        py::arg("num_enroll_examples"),
        py::arg("transformed_ivector"))
      .def("TransformIvector",
        py::overload_cast<const PldaConfig &,
                          const VectorBase<float> &,
                          int32,
                          VectorBase<float> *>(&PyClass::TransformIvector, py::const_),
        "Transforms an iVector into a space where the within-class variance "
        "is unit and between-class variance is diagonalized.  The only "
        "anticipated use of this function is to pre-transform iVectors "
        "before giving them to the function LogLikelihoodRatio (it's "
        "done this way for efficiency because a given iVector may be "
        "used multiple times in LogLikelihoodRatio and we don't want "
        "to repeat the matrix multiplication "
        "\n"
        "If config.normalize_length == true, it will also normalize the iVector's "
        "length by multiplying by a scalar that ensures that ivector^T inv_var "
        "ivector = dim.  In this case, \"num_enroll_examples\" comes into play because it "
        "affects the expected covariance matrix of the iVector.  The normalization "
        "factor is returned, even if config.normalize_length == false, in which "
        "case the normalization factor is computed but not applied. "
        "If config.simple_length_normalization == true, then an alternative "
        "normalization factor is computed that causes the iVector length "
        "to be equal to the square root of the iVector dimension.",
        py::arg("config"),
        py::arg("ivector"),
        py::arg("num_enroll_examples"),
        py::arg("transformed_ivector"))
      .def("LogLikelihoodRatio",
        &PyClass::LogLikelihoodRatio,
        "Returns the log-likelihood ratio "
        "log (p(test_ivector | same) / p(test_ivector | different)). "
        "transformed_enroll_ivector is an average over utterances for "
        "that speaker.  Both transformed_enroll_vector and transformed_test_ivector "
        "are assumed to have been transformed by the function TransformIvector(). "
        "Note: any length normalization will have been done while computing "
        "the transformed iVectors.",
        py::arg("transformed_enroll_ivector"),
        py::arg("num_enroll_utts"),
        py::arg("transformed_test_ivector"),
      py::call_guard<py::gil_scoped_release>())
      .def("LogLikelihoodRatio",
        [](

            PyClass &plda,
            const VectorBase<float> & transformed_enroll_ivector,
            int32 num_enroll_utts,
            const VectorBase<float> & transformed_test_ivector
        ){
          py::gil_scoped_release gil_release;
          Vector<double> ivector_one_dbl(transformed_enroll_ivector);
          Vector<double> ivector_two_dbl(transformed_test_ivector);
          return plda.LogLikelihoodRatio(ivector_one_dbl, num_enroll_utts, ivector_two_dbl);

        },
        "Returns the log-likelihood ratio "
        "log (p(test_ivector | same) / p(test_ivector | different)). "
        "transformed_enroll_ivector is an average over utterances for "
        "that speaker.  Both transformed_enroll_vector and transformed_test_ivector "
        "are assumed to have been transformed by the function TransformIvector(). "
        "Note: any length normalization will have been done while computing "
        "the transformed iVectors.",
        py::arg("transformed_enroll_ivector"),
        py::arg("num_enroll_utts"),
        py::arg("transformed_test_ivector"))
      .def("LogLikelihoodRatio",
        [](

            PyClass &plda,
            py::array_t<double> & transformed_enroll_ivector,
            int32 num_enroll_utts,
            py::array_t<double> & transformed_test_ivector
        ){
          py::gil_scoped_release gil_release;
          Vector<double> ivector_one_dbl;
          auto r1 = transformed_enroll_ivector.unchecked<1>();
          ivector_one_dbl.Resize(r1.shape(0));
          for (py::size_t i = 0; i < r1.shape(0); i++)
            ivector_one_dbl(i) = r1(i);

          Vector<double> ivector_two_dbl;
          auto r2 = transformed_test_ivector.unchecked<1>();
          ivector_two_dbl.Resize(r2.shape(0));
          for (py::size_t i = 0; i < r2.shape(0); i++)
            ivector_two_dbl(i) = r2(i);

          return plda.LogLikelihoodRatio(ivector_one_dbl, num_enroll_utts, ivector_two_dbl);

        },
        "Returns the log-likelihood ratio "
        "log (p(test_ivector | same) / p(test_ivector | different)). "
        "transformed_enroll_ivector is an average over utterances for "
        "that speaker.  Both transformed_enroll_vector and transformed_test_ivector "
        "are assumed to have been transformed by the function TransformIvector(). "
        "Note: any length normalization will have been done while computing "
        "the transformed iVectors.",
        py::arg("transformed_enroll_ivector"),
        py::arg("num_enroll_utts"),
        py::arg("transformed_test_ivector"))
      .def("SmoothWithinClassCovariance",
        &PyClass::SmoothWithinClassCovariance,
        "This function smooths the within-class covariance by adding to it, "
        "smoothing_factor (e.g. 0.1) times the between-class covariance (it's "
        "implemented by modifying transform_).  This is to compensate for "
        "situations where there were too few utterances per speaker get a good "
        "estimate of the within-class covariance, and where the leading elements of "
        "psi_ were as a result very large.",
        py::arg("smoothing_factor"))
      .def("ApplyTransform",
        &PyClass::ApplyTransform,
        "Apply a transform to the PLDA model.  This is mostly used for "
        "projecting the parameters of the model into a lower dimensional space, "
        "i.e. in_transform.NumRows() <= in_transform.NumCols(), typically for "
        "speaker diarization with a PCA transform.",
        py::arg("in_transform"))
      .def("Dim",
        &PyClass::Dim)
      .def("Read", &PyClass::Read,
        py::arg("is"),
        py::arg("binary"),
      py::call_guard<py::gil_scoped_release>())
      .def("Write", &PyClass::Write,
        py::arg("os"),
        py::arg("binary"),
      py::call_guard<py::gil_scoped_release>());
  }
  {
    using PyClass = PldaStats;

    auto plda_stats = py::class_<PyClass>(
        m, "PldaStats");
    plda_stats.def(py::init<>())
      .def("AddSamples",
        &PyClass::AddSamples,
        "This function adds training samples corresponding to "
        "one class (e.g. a speaker).  Each row is a separate "
        "sample from this group.  The \"weight\" would normally "
        "be 1.0, but you can set it to other values if you want "
        "to weight your training samples.",
        py::arg("weight"),
        py::arg("group"))
      .def("Dim",
        &PyClass::Dim)
      .def("Init",
        &PyClass::Init,
        py::arg("dim"))
      .def("Sort",
        &PyClass::Sort)
      .def("IsSorted",
        &PyClass::IsSorted);
  }
  {
    using PyClass = PldaEstimationConfig;

    auto plda_estimation_config = py::class_<PyClass>(
        m, "PldaEstimationConfig");
    plda_estimation_config.def(py::init<>())
      .def_readwrite("num_em_iters", &PyClass::num_em_iters)
      .def(py::pickle(
        [](const PyClass &p) { // __getstate__
            /* Return a tuple that fully encodes the state of the object */
            return py::make_tuple(
                p.num_em_iters);
        },
        [](py::tuple t) { // __setstate__
            if (t.size() != 1)
                throw std::runtime_error("Invalid state!");

            /* Create a new C++ instance */
            PyClass opts;

            /* Assign any additional state */
            opts.num_em_iters = t[0].cast<int32>();

            return opts;
        }
    ));
  }
  {
    using PyClass = PldaEstimator;

    auto plda_estimator = py::class_<PyClass>(
        m, "PldaEstimator");
    plda_estimator.def(py::init<const PldaStats &>(),
        py::arg("stats"))
      .def("Estimate",
        &PyClass::Estimate,
        py::arg("config"),
        py::arg("output"))
      .def("estimate",
        [](
          PyClass &plda_estimator,
          const PldaEstimationConfig &plda_config
        ){

        Plda plda;
        plda_estimator.Estimate(plda_config, &plda);
        return &plda;
        },
        py::arg("config"),
        py::return_value_policy::reference);
  }
  {
    using PyClass = PldaUnsupervisedAdaptorConfig;

    auto plda_unsupervised_adaptor_config = py::class_<PyClass>(
        m, "PldaUnsupervisedAdaptorConfig");
    plda_unsupervised_adaptor_config.def(py::init<>())
      .def_readwrite("mean_diff_scale", &PyClass::mean_diff_scale)
      .def_readwrite("within_covar_scale", &PyClass::within_covar_scale)
      .def_readwrite("between_covar_scale", &PyClass::between_covar_scale)
      .def(py::pickle(
        [](const PyClass &p) { // __getstate__
            /* Return a tuple that fully encodes the state of the object */
            return py::make_tuple(
                p.mean_diff_scale,
                p.within_covar_scale,
                p.between_covar_scale);
        },
        [](py::tuple t) { // __setstate__
            if (t.size() != 3)
                throw std::runtime_error("Invalid state!");

            /* Create a new C++ instance */
            PyClass opts;

            /* Assign any additional state */
            opts.mean_diff_scale = t[0].cast<BaseFloat>();
            opts.within_covar_scale = t[1].cast<BaseFloat>();
            opts.between_covar_scale = t[2].cast<BaseFloat>();

            return opts;
        }
    ));
  }
  {
    using PyClass = PldaUnsupervisedAdaptor;

    auto plda_unsupervised_adaptor = py::class_<PyClass>(
        m, "PldaUnsupervisedAdaptor");
    plda_unsupervised_adaptor.def(py::init<>())
      .def("AddStats",
        py::overload_cast<double, const Vector<double> &>(&PyClass::AddStats),
        "Add stats to this class.  Normally the weight will be 1.0.",
        py::arg("weight"),
        py::arg("ivector"))
      .def("AddStats",
        py::overload_cast<double, const Vector<float> &>(&PyClass::AddStats),
        "Add stats to this class.  Normally the weight will be 1.0.",
        py::arg("weight"),
        py::arg("ivector"))
      .def("UpdatePlda",
        &PyClass::UpdatePlda,
        "Add stats to this class.  Normally the weight will be 1.0.",
        py::arg("config"),
        py::arg("plda"));
  }
}

void pybind_voice_activity_detection(py::module &m) {

  {
    using PyClass = VadEnergyOptions;

    auto vad_energy_options = py::class_<PyClass>(
        m, "VadEnergyOptions");
    vad_energy_options.def(py::init<>())
      .def_readwrite("vad_energy_threshold", &PyClass::vad_energy_threshold)
      .def_readwrite("vad_energy_mean_scale", &PyClass::vad_energy_mean_scale)
      .def_readwrite("vad_frames_context", &PyClass::vad_frames_context)
      .def_readwrite("vad_proportion_threshold", &PyClass::vad_proportion_threshold)
      .def(py::pickle(
        [](const PyClass &p) { // __getstate__
            /* Return a tuple that fully encodes the state of the object */
            return py::make_tuple(
                p.vad_energy_threshold,
                p.vad_energy_mean_scale,
                p.vad_frames_context,
                p.vad_proportion_threshold);
        },
        [](py::tuple t) { // __setstate__
            if (t.size() != 4)
                throw std::runtime_error("Invalid state!");

            /* Create a new C++ instance */
            PyClass opts;

            /* Assign any additional state */
            opts.vad_energy_threshold = t[0].cast<BaseFloat>();
            opts.vad_energy_mean_scale = t[1].cast<BaseFloat>();
            opts.vad_frames_context = t[2].cast<int32>();
            opts.vad_proportion_threshold = t[3].cast<BaseFloat>();

            return opts;
        }
    ));
  }
  m.def("ComputeVadEnergy",
      [](const VadEnergyOptions &opts,
                      const MatrixBase<BaseFloat> &input_features){
      Vector<BaseFloat> vad_result(input_features.NumRows());

      ComputeVadEnergy(opts, input_features, &vad_result);
      return vad_result;
      },
        "Compute voice-activity vector for a file: 1 if we judge the frame as "
        "voiced, 0 otherwise.  There are no continuity constraints. "
        "This method is a very simple energy-based method which only looks "
        "at the first coefficient of \"input_features\", which is assumed to "
        "be a log-energy or something similar.  A cutoff is set-- we use  "
        "a formula of the general type: cutoff = 5.0 + 0.5 * (average log-energy "
        "in this file), and for each frame the decision is based on the "
        "proportion of frames in a context window around the current frame, "
        "which are above this cutoff.",
        py::arg("opts"),
        py::arg("input_features"));
}

void init_ivector(py::module &_m) {
  py::module m = _m.def_submodule("ivector", "ivector pybind for Kaldi");

  pybind_agglomerative_clustering(m);
  pybind_ivector_extractor(m);
  pybind_logistic_regression(m);
  pybind_plda(m);
  pybind_voice_activity_detection(m);

  m.def("ivector_extract",
    [](
    const DiagGmm &gmm,
      const IvectorExtractor &extractor,
      const Matrix<BaseFloat> &feats,

      double acoustic_weight = 1.0,
      double max_count = 0.0,
      int32 num_post = 50,
      BaseFloat min_post = 0.0
    ) -> Vector<double> {
      IvectorEstimationOptions opts;
      opts.acoustic_weight = acoustic_weight;
      opts.max_count = max_count;
      int32 T = feats.NumRows();
      std::vector<std::vector<int32> > gselect(T);

      Matrix<BaseFloat> loglikes;

      gmm.LogLikelihoods(feats, &loglikes);

      Posterior posterior(T);

      double log_like_this_file = 0.0;
      for (int32 t = 0; t < T; t++) {
        log_like_this_file +=
            VectorToPosteriorEntry(loglikes.Row(t), num_post,
                                   min_post, &(posterior[t]));
      }


        double this_t = opts.acoustic_weight * TotalPosterior(posterior),
            max_count_scale = 1.0;
        if (opts.max_count > 0 && this_t > opts.max_count) {
          max_count_scale = opts.max_count / this_t;
          KALDI_LOG << "Scaling stats for utterance by scale "
                    << max_count_scale << " due to --max-count="
                    << opts.max_count;
          this_t = opts.max_count;
        }
        ScalePosterior(opts.acoustic_weight * max_count_scale,
                        &posterior);

      bool need_2nd_order_stats = false;

      IvectorExtractorUtteranceStats utt_stats(extractor.NumGauss(),
                                              extractor.FeatDim(),
                                              need_2nd_order_stats);

    utt_stats.AccStats(feats, posterior);

    Vector<double> ivector;
    ivector.Resize(extractor.IvectorDim());
    ivector(0) = extractor.PriorOffset();
    extractor.GetIvectorDistribution(utt_stats, &ivector, NULL);

    return ivector;
    },
        py::arg("gmm"),
        py::arg("extractor"),
        py::arg("feats"),
        py::arg("acoustic_weight") = 1.0,
        py::arg("max_count") = 0.0,
        py::arg("num_post") = 50,
        py::arg("min_post") = 0.0);

  m.def("ivector_normalize_length",
    [](
    Vector<BaseFloat>* ivector,
      bool normalize = true,
      bool scaleup = true
    ) {
          py::gil_scoped_release gil_release;
      BaseFloat norm = ivector->Norm(2.0);
      BaseFloat ratio = norm / sqrt(ivector->Dim());
      if (!scaleup) ratio = norm;
      if (normalize) ivector->Scale(1.0 / ratio);
    },
        py::arg("ivector"),
        py::arg("normalize") = true,
        py::arg("scaleup") = true);

  m.def("ivector_subtract_mean",
    [](
        std::vector<Vector<float>*> &ivectors
    ) {
          py::gil_scoped_release gil_release;
      Vector<double> sum;

        for (size_t i = 0; i < ivectors.size(); i++) {
          if (sum.Dim() == 0) sum.Resize(ivectors[i]->Dim());
          sum.AddVec(1.0, *ivectors[i]);
        }
        for (size_t i = 0; i < ivectors.size(); i++) {
          Vector<BaseFloat> *ivector = ivectors[i];
            ivector->AddVec(-1.0 / ivectors.size(), sum);
        }
    },
        py::arg("ivectors"));
}
