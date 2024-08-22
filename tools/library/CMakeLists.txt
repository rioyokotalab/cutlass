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

include(GNUInstallDirs)

add_library(cutlass_library_includes INTERFACE)

target_include_directories(
  cutlass_library_includes
  INTERFACE
  $<INSTALL_INTERFACE:include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  )

target_link_libraries(
  cutlass_library_includes 
  INTERFACE 
  CUTLASS
  cutlass_tools_util_includes
  )

install(
  TARGETS cutlass_library_includes
  EXPORT NvidiaCutlass
  )

install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/
  )

add_library(cutlass_library_internal_interface INTERFACE)

target_include_directories(
  cutlass_library_internal_interface
  INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
  )

target_link_libraries(
  cutlass_library_internal_interface
  INTERFACE
  cutlass_library_includes
  )

################################################################################

function(cutlass_add_cutlass_library)
  set(options)
  set(oneValueArgs SUFFIX)
  set(multiValueArgs)
  cmake_parse_arguments(_ "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  set(DEFAULT_NAME cutlass_library)

  set(__NAME ${DEFAULT_NAME})
  set(__OUTPUT_NAME cutlass)
  set(__EXPORT_NAME library)

  if (__SUFFIX)
    string(APPEND __NAME _${__SUFFIX})
    string(APPEND __OUTPUT_NAME _${__SUFFIX})
    string(APPEND __EXPORT_NAME _${__SUFFIX})
  endif()

  cutlass_add_library(
    ${__NAME}_objs
    OBJECT
    ${__UNPARSED_ARGUMENTS}
    )  

  target_link_libraries(${__NAME}_objs
    PUBLIC cutlass_library_includes
    PRIVATE cutlass_library_internal_interface
    )

  cutlass_add_library(
    ${__NAME} 
    SHARED
    EXPORT_NAME ${__EXPORT_NAME}
    ""
    )

  target_link_libraries(
    ${__NAME}
    PUBLIC cutlass_library_includes
    PRIVATE $<BUILD_INTERFACE:${__NAME}_objs>
    cuda_driver
    )
    
  if (__SUFFIX)
    target_link_libraries(${DEFAULT_NAME} PUBLIC ${__NAME})
  endif()

endfunction()

################################################################################

cutlass_add_cutlass_library(
  src/handle.cu
  src/manifest.cpp
  src/operation_table.cu
  src/singleton.cu
  src/util.cu
  )

# For backward compatibility with the old name
add_library(cutlass_lib ALIAS cutlass_library)

execute_process(
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../python/cutlass_library
  COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CUTLASS_LIBRARY_PACKAGE_DIR}
    ${Python3_EXECUTABLE} ${CUTLASS_SOURCE_DIR}/python/cutlass_library/generator.py
    --operations "all" 
    --build-dir ${PROJECT_BINARY_DIR}
    --curr-build-dir ${CMAKE_CURRENT_BINARY_DIR}
    --generator-target library
    --architectures "90a"
    --kernels "cutlass_tensorop_s16816gemm_f16_128x128_32x5_nt_align8"
    --ignore-kernels ""
    --kernel-filter-file ""
    --selected-kernel-list "${CMAKE_CURRENT_BINARY_DIR}/generated_kernels.txt"
    --cuda-version "${CMAKE_CUDA_COMPILER_VERSION}"
    --log-level DEBUG
    --disable-cutlass-package-imports
  RESULT_VARIABLE cutlass_lib_INSTANCE_GENERATION_RESULT
  OUTPUT_VARIABLE cutlass_lib_INSTANCE_GENERATION_OUTPUT
  OUTPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/library_instance_generation.log
  ERROR_FILE ${CMAKE_CURRENT_BINARY_DIR}/library_instance_generation.log
)
include(${CMAKE_CURRENT_BINARY_DIR}/generated/manifest.cmake)

