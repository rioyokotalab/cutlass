# Copyright (c) 2017 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

function(cutlass_apply_cuda_gencode_flags TARGET)
  set_property(TARGET ${TARGET} PROPERTY CUDA_ARCHITECTURES 90a-real;90a-virtual)
endfunction()

function(cutlass_apply_standard_compile_options TARGET)
  target_compile_options(${TARGET} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;-DCUTLASS_TEST_LEVEL=0;-DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1;-Xcompiler=-Wconversion;-Xcompiler=-fno-strict-aliasing;-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1>)
endfunction()

set(CUTLASS_NATIVE_CUDA ON CACHE BOOL "Utilize the CMake native CUDA flow")

enable_language(CUDA)
get_filename_component(CUDA_TOOLKIT_ROOT_DIR "${CMAKE_CUDA_COMPILER}/../.." ABSOLUTE)
find_library(
  CUDA_DRIVER_LIBRARY cuda
  PATHS
  ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES
  lib/x86_64-linux-gnu
  lib/x64
  lib64
  lib
  lib64/stubs
  lib/stubs
  NO_DEFAULT_PATH
  )
message(STATUS "CUDA Driver: ${CUDA_DRIVER_LIBRARY}")
add_library(cuda_driver SHARED IMPORTED GLOBAL)
set_property(
  TARGET cuda_driver
  PROPERTY IMPORTED_LOCATION
  ${CUDA_DRIVER_LIBRARY}
  )

set(CUTLASS_UNITY_BUILD_ENABLED OFF CACHE BOOL "Enable combined source compilation")
set(CUTLASS_UNITY_BUILD_BATCH_SIZE 16 CACHE STRING "Batch size for unified source files")

function(cutlass_unify_source_files TARGET_ARGS_VAR)
  set(${TARGET_ARGS_VAR} ${__UNPARSED_ARGUMENTS} PARENT_SCOPE)
endfunction()

function(cutlass_add_library NAME)

  set(options SKIP_GENCODE_FLAGS)
  set(oneValueArgs EXPORT_NAME)
  set(multiValueArgs)
  cmake_parse_arguments(_ "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  cutlass_unify_source_files(TARGET_SOURCE_ARGS ${__UNPARSED_ARGUMENTS})

  add_library(${NAME} ${TARGET_SOURCE_ARGS} "")

  cutlass_apply_standard_compile_options(${NAME})
  if (NOT __SKIP_GENCODE_FLAGS)
  cutlass_apply_cuda_gencode_flags(${NAME})
  endif()

  target_compile_features(
   ${NAME}
   INTERFACE
   cxx_std_11
   )

  get_target_property(TARGET_TYPE ${NAME} TYPE)

  if (TARGET_TYPE MATCHES "SHARED")
    set_target_properties(${NAME} PROPERTIES CUDA_RUNTIME_LIBRARY Shared)
  elseif(TARGET_TYPE MATCHES "STATIC")
    set_target_properties(${NAME} PROPERTIES CUDA_RUNTIME_LIBRARY Static)
  endif()

  if(__EXPORT_NAME)
    add_library(nvidia::cutlass::${__EXPORT_NAME} ALIAS ${NAME})
    set_target_properties(${NAME} PROPERTIES EXPORT_NAME ${__EXPORT_NAME})
  endif()

endfunction()

function(cutlass_add_executable NAME)

  set(options)
  set(oneValueArgs CUDA_RUNTIME_LIBRARY)
  set(multiValueArgs)
  cmake_parse_arguments(_ "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if (NOT DEFINED __CUDA_RUNTIME_LIBRARY)
    set(__CUDA_RUNTIME_LIBRARY Shared)
  endif()

  set(__CUDA_RUNTIME_LIBRARY_ALLOWED None Shared Static)
  if (NOT __CUDA_RUNTIME_LIBRARY IN_LIST __CUDA_RUNTIME_LIBRARY_ALLOWED)
    message(FATAL_ERROR "CUDA_RUNTIME_LIBRARY value '${__CUDA_RUNTIME_LIBRARY}' is not in allowed list of '${__CUDA_RUNTIME_LIBRARY_ALLOWED}'")
  endif()

  cutlass_unify_source_files(TARGET_SOURCE_ARGS ${__UNPARSED_ARGUMENTS})

  add_executable(${NAME} ${TARGET_SOURCE_ARGS})

  cutlass_apply_standard_compile_options(${NAME})
  cutlass_apply_cuda_gencode_flags(${NAME})

  target_compile_features(
   ${NAME}
   INTERFACE
   cxx_std_11
   )

  set_target_properties(${NAME} PROPERTIES CUDA_RUNTIME_LIBRARY ${__CUDA_RUNTIME_LIBRARY})

endfunction()

function(cutlass_target_sources NAME)

  set(options)
  set(oneValueArgs)
  set(multiValueArgs)
  cmake_parse_arguments(_ "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  cutlass_unify_source_files(TARGET_SOURCE_ARGS ${__UNPARSED_ARGUMENTS})
  target_sources(${NAME} ${TARGET_SOURCE_ARGS})

endfunction()
