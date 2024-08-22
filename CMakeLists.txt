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

cmake_minimum_required(VERSION 3.19 FATAL_ERROR)
cmake_policy(SET CMP0112 NEW)
cmake_policy(SET CMP0104 NEW)
project(CUTLASS VERSION 3.5.0 LANGUAGES CXX)
include(${CMAKE_CURRENT_SOURCE_DIR}/CUDA.cmake)
set(CMAKE_INSTALL_PREFIX install CACHE PATH "Default installation location." FORCE)
find_package(Python3 3.5 COMPONENTS Interpreter REQUIRED)
include(GNUInstallDirs)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

function(cutlass_apply_cuda_gencode_flags TARGET)
  set_property(TARGET ${TARGET} PROPERTY CUDA_ARCHITECTURES 90a-real;90a-virtual)
endfunction()

function(cutlass_apply_standard_compile_options TARGET)
  target_compile_options(${TARGET} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;-DCUTLASS_TEST_LEVEL=0;-DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1;-Xcompiler=-Wconversion;-Xcompiler=-fno-strict-aliasing;-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1>)
endfunction()

add_library(CUTLASS INTERFACE)
add_library(nvidia::cutlass::cutlass ALIAS CUTLASS)
set(CUTLASS_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include CACHE PATH "CUTLASS Header Library")
set(CUTLASS_GENERATOR_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tools/library CACHE INTERNAL "Location of generator scripts")
set(CUTLASS_TOOLS_UTIL_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tools/util/include CACHE INTERNAL "")
include_directories(${CUTLASS_INCLUDE_DIR})
include(${CMAKE_CURRENT_SOURCE_DIR}/cuBLAS.cmake)
add_subdirectory(tools)
include(CMakePackageConfigHelpers)

#configure_file(
#  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/NvidiaCutlassConfig.cmake.in
#  ${CMAKE_CURRENT_BINARY_DIR}/NvidiaCutlassConfig.cmake  
#  @ONLY
#  )

install(
  FILES 
    ${CMAKE_CURRENT_BINARY_DIR}/NvidiaCutlassConfig.cmake  
    ${CMAKE_CURRENT_BINARY_DIR}/NvidiaCutlassConfigVersion.cmake 
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/NvidiaCutlass/
  )

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/NvidiaCutlassPackageConfig.cmake)

