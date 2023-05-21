#include "pybind/kaldi_pybind.h"
#include "chain/pybind_chain.h"
#include "cudamatrix/pybind_cudamatrix.h"
#include "hmm/pybind_hmm.h"
#include "gmm/pybind_gmm.h"
#include "decoder/pybind_decoder.h"
#include "feat/pybind_feat.h"
#include "fstext/pybind_fstext.h"
#include "kws/pybind_kws.h"
#include "lat/pybind_lat.h"
#include "lm/pybind_lm.h"
#include "nnet/pybind_nnet.h"
#include "nnet2/pybind_nnet2.h"
#include "nnet3/pybind_nnet3.h"
#include "itf/pybind_itf.h"
#include "ivector/pybind_ivector.h"
//#include "rnnlm/pybind_rnnlm.h"
#include "online/pybind_online.h"
#include "online2/pybind_online2.h"
#include "transform/pybind_transform.h"
#include "tree/pybind_tree.h"
#include "matrix/pybind_matrix.h"
#include "util/pybind_util.h"



PYBIND11_MODULE(_kalpy, m){
    m.attr("__version__") = "dev";
    init_itf(m);
    //init_base(m);
    init_util(m);
    init_matrix(m);
    init_cudamatrix(m);
    init_fstext(m);
    init_tree(m);
    init_feat(m);
    init_hmm(m);
    init_gmm(m);
    init_lat(m);
    init_transform(m);
    init_decoder(m);
    init_chain(m);
    init_ivector(m);
    init_kws(m);
    //init_rnnlm(m);
    init_lm(m);
    init_online(m);
    init_online2(m);
    init_nnet(m);
    init_nnet2(m);
    init_nnet3(m);
}
