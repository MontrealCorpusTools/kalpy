// pybind/kaldi_pybind.h

// Copyright 2019   Daniel Povey
//           2019   Dongji Gao
//           2019   Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)
//
// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#ifndef KALDI_PYBIND_KALDI_PYBIND_H_
#define KALDI_PYBIND_KALDI_PYBIND_H_

#define PYBIND11_DETAILED_ERROR_MESSAGES
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/cast.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
#include <mutex>
#include <iostream>
#include "fst/vector-fst.h"
#include "fst/fst.h"
#include "fst/fstlib.h"
#include "fst/fst-decl.h"
#include <fst/script/fst-class.h>
#include <fst/extensions/far/far-class.h>
#include "_pywrapfst.h"


static std::mutex cout_mutex;

namespace py = pybind11;

template <typename... Args>
using overload_cast_ = py::detail::overload_cast_impl<Args...>;


#endif  // KALDI_PYBIND_KALDI_PYBIND_H_
