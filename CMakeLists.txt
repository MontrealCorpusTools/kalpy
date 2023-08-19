cmake_minimum_required(VERSION 3.18)
project(_kalpy)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_EXTENSIONS OFF)
option(BUILD_SHARED_LIBS "Build shared libraries" ON)
if(KALDI_ROOT)
    message(STATUS "Adding KALDI_ROOT directories: ${KALDI_ROOT}")
    set(CMAKE_INCLUDE_PATH "${KALDI_ROOT}/include")
    set(CMAKE_LIBRARY_PATH "${KALDI_ROOT}/lib")
    link_directories("${KALDI_ROOT}/lib")
    include_directories("${KALDI_ROOT}/include")
    include_directories("${KALDI_ROOT}/include/kaldi")
    if (NOT CUDA_TOOLKIT_ROOT_DIR)
        set(CUDA_TOOLKIT_ROOT_DIR "${KALDI_ROOT}")
    endif()

endif()
add_definitions(-DKALDI_NO_PORTAUDIO=1)
if (MSVC)
    find_package(dlfcn-win32 REQUIRED)
    set(CMAKE_DL_LIBS dlfcn-win32::dl)

    #   set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    add_definitions(-DWIN32_LEAN_AND_MEAN)
    add_definitions(-DNOMINMAX)
    add_definitions(-D_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS)
    add_definitions(-D_USE_MATH_DEFINES)
    add_compile_options(/permissive- /FS /wd4819 /EHsc /bigobj)

    # some warnings related with fst
    add_compile_options(/wd4018 /wd4244 /wd4267 /wd4291 /wd4305)

    set(CompilerFlags
        CMAKE_CXX_FLAGS
        CMAKE_CXX_FLAGS_DEBUG
        CMAKE_CXX_FLAGS_RELEASE
        CMAKE_C_FLAGS
        CMAKE_C_FLAGS_DEBUG
        CMAKE_C_FLAGS_RELEASE
        )
    foreach(CompilerFlag ${CompilerFlags})
      string(REPLACE "/MD" "/MT" ${CompilerFlag} "${${CompilerFlag}}")
    endforeach()
 elseif(APPLE)
     set(CMAKE_INSTALL_RPATH "@loader_path")
 else()
     set(CMAKE_INSTALL_RPATH "$ORIGIN;$ORIGIN/../lib;$ORIGIN/../../tools/openfst/lib")
endif ()

find_package(CUDA)

find_package(pybind11 REQUIRED)
include_directories(extensions)
pybind11_add_module(_kalpy extensions/_kalpy.cpp
                            extensions/chain/chain.cpp
                            extensions/cudamatrix/cudamatrix.cpp
                            extensions/decoder/decoder.cpp
                            extensions/feat/feat.cpp
                            extensions/fstext/fstext.cpp
                            extensions/gmm/gmm.cpp
                            extensions/hmm/hmm.cpp
                            extensions/itf/itf.cpp
                            extensions/ivector/ivector.cpp
                            extensions/kws/kws.cpp
                            extensions/lat/lat.cpp
                            extensions/lm/lm.cpp
                            #extensions/rnnlm/rnnlm.cpp
                            extensions/online/online.cpp
                            extensions/online2/online2.cpp
                            extensions/matrix/matrix.cpp
                            extensions/nnet/nnet.cpp
                            extensions/nnet2/nnet2.cpp
                            extensions/nnet3/nnet3.cpp
                            extensions/transform/transform.cpp
                            extensions/tree/tree.cpp
                            extensions/util/util.cpp
                            )
target_link_libraries(_kalpy PUBLIC kaldi-base kaldi-chain
                                    kaldi-matrix
                                    kaldi-cudamatrix
                                    kaldi-hmm
                                    kaldi-online kaldi-online2 kaldi-rnnlm
                                    kaldi-nnet3
                                    kaldi-nnet2 kaldi-nnet
                                    kaldi-kws
                                    kaldi-decoder
                                    kaldi-lat
                                    kaldi-ivector kaldi-lm
                                    kaldi-fstext kaldi-feat
                                    kaldi-transform kaldi-gmm
                                    kaldi-tree
                                    kaldi-util
                                    fst
                                    fstscript
                                    )

if(CUDA_FOUND)

   target_link_libraries(_kalpy PUBLIC kaldi-cudadecoder kaldi-cudafeat
                                    )
endif()
target_compile_definitions(_kalpy
                           PRIVATE VERSION_INFO="5.5.1068")
if(MSVC)
set_target_properties(_kalpy PROPERTIES
    DEFINE_SYMBOL "KALDI_DLL_IMPORTS"
                "KALDI_CUMATRIX_DLL_IMPORTS"
                "KALDI_UTIL_DLL_IMPORTS")
endif(MSVC)
