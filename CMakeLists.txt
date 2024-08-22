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

###################################################################################################

#source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/include REGULAR_EXPRESSION ".*\.h")

add_library(CUTLASS INTERFACE)
add_library(nvidia::cutlass::cutlass ALIAS CUTLASS)
set_target_properties(CUTLASS PROPERTIES EXPORT_NAME cutlass)

set(CUTLASS_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include CACHE PATH "CUTLASS Header Library")

set(CUTLASS_GENERATOR_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tools/library CACHE INTERNAL "Location of generator scripts")

# The following utility directory is needed even if the tools build is disabled, so it exists here.
set(CUTLASS_TOOLS_UTIL_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tools/util/include CACHE INTERNAL "")

include_directories(${CUTLASS_INCLUDE_DIR})

target_compile_features(CUTLASS INTERFACE cxx_std_11)

if (NOT DEFINED CUTLASS_REVISION)

  find_package(Git QUIET)

  execute_process(
    COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
    RESULT_VARIABLE CUTLASS_REVISION_RESULT
    OUTPUT_VARIABLE CUTLASS_REVISION
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  if (CUTLASS_REVISION_RESULT)
    message(STATUS "CUTLASS Revision: Unable to detect, Git returned code ${CUTLASS_REVISION_RESULT}.")
  else()
    message(STATUS "CUTLASS Revision: ${CUTLASS_REVISION}")
  endif()

endif()

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/version_extended.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/cutlass/version_extended.h
  @ONLY)

target_include_directories(
  CUTLASS
  INTERFACE
  $<INSTALL_INTERFACE:include>
  $<BUILD_INTERFACE:${CUTLASS_INCLUDE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
  $<BUILD_INTERFACE:${cute_SOURCE_DIR}/include>
  )

# Mark CTK headers as system to supress warnings from them
target_include_directories(
  CUTLASS
  SYSTEM INTERFACE
  $<BUILD_INTERFACE:${CUDA_TOOLKIT_ROOT_DIR}/include>
  )

install(
  DIRECTORY
  ${CUTLASS_INCLUDE_DIR}/
  ${CMAKE_CURRENT_BINARY_DIR}/include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )

install(
  TARGETS CUTLASS
  EXPORT NvidiaCutlass
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )

include(CTest)
enable_testing()


if (NOT TARGET test_all)
  add_custom_target(test_all)
endif()

set(CUTLASS_INSTALL_TESTS ON CACHE BOOL "Install test executables")
set(CUTLASS_TEST_EXECUTION_ENVIRONMENT "" CACHE BOOL "Environment in which to invoke unit test executables")

set(CMAKE_TEST_INSTALL_PREFIX test CACHE STRING "Test root install location, relative to CMAKE_INSTALL_PREFIX.")
set(CUTLASS_TEST_INSTALL_PREFIX ${CMAKE_TEST_INSTALL_PREFIX}/cutlass CACHE STRING "Test root install location, relative to CMAKE_INSTALL_PREFIX.")
set(CUTLASS_TEST_INSTALL_BINDIR ${CUTLASS_TEST_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR} CACHE STRING "Test root install location, relative to CMAKE_INSTALL_PREFIX.")
set(CUTLASS_TEST_INSTALL_LIBDIR ${CUTLASS_TEST_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR} CACHE STRING "Test root install location, relative to CMAKE_INSTALL_PREFIX.")

install(DIRECTORY DESTINATION ${CUTLASS_TEST_INSTALL_PREFIX})
install(DIRECTORY DESTINATION ${CUTLASS_TEST_INSTALL_BINDIR})
install(DIRECTORY DESTINATION ${CUTLASS_TEST_INSTALL_LIBDIR})
install(DIRECTORY DESTINATION ${CUTLASS_TEST_INSTALL_PREFIX}/ctest)

################################################################################

set(CUTLASS_ENABLE_CUBLAS OFF CACHE BOOL "cuBLAS usage for tests")
set(CUTLASS_ENABLE_CUDNN OFF CACHE BOOL "cuDNN usage for tests")

include(${CMAKE_CURRENT_SOURCE_DIR}/cuBLAS.cmake)

if (CUTLASS_ENABLE_CUBLAS)
  target_compile_definitions(CUTLASS INTERFACE CUTLASS_ENABLE_CUBLAS=1)
endif()


if (CUTLASS_ENABLE_CUDNN)
  target_compile_definitions(CUTLASS INTERFACE CUTLASS_ENABLE_CUDNN=1)
endif()

################################################################################

set(CUTLASS_DEFAULT_ACTIVE_TEST_SETS "default" CACHE STRING "Default
  activated test sets. In `make test` mode, this string determines the
  active set of tests. In `ctest` mode, this value can be overriden
  with CUTLASS_TEST_SETS environment variable when running the ctest
  executable.")

set(CUTLASS_CTEST_TEMPLATE_FILE ${CMAKE_CURRENT_LIST_DIR}/cmake/CTestTestfile.configure.cmake)
set(CUTLASS_CTEST_GENERATED_FILES "" CACHE INTERNAL "")

function(cutlass_add_executable_tests NAME TARGET)
# 
# Generates test rules for `make test`, `make test_all`, and `ctest` invoked from either the 
# <CMAKE_BINARY_DIR> or the <CMAKE_INSTALL_PREFIX>/<CUTLASS_TEST_INSTALL_PREFIX> after installation.
# 
# NAME: The base name for the test. Can be run with `make <NAME>` or `ctest -R 'c<NAME>'`.
# TARGET: The target corresponding to the executable under test.
# DISABLE_EXECUTABLE_INSTALL_RULE: An option, if given, that disables creating an install rule for TARGET.
# DEPENDS: A list of targets or files on which this test is dependent.
# DEPENDEES: A list of targets which should depend on this test.
# TEST_COMMAND_OPTIONS: A list of variables (i.e. by reference params) which contain command line arguments
#   to pass to the test executable. A unique test is generated for each set of 
#   options given. If this option is not used, a single test with no arguments is generated.
# TEST_COMMAND_OPTIONS_PREFIX: If provided, is added as a prefix to each TEST_COMMAND_OPTIONS value for 
#   generating the full variable name to be referenced.
# RESULT_CACHE_FILE: A file to be installed alongside the test executable with pre-computed
#   test results to speed up test runtime.
# TEST_SETS_SUPPORTED: A list of test set names these tests support. 
# 

  set(options DISABLE_EXECUTABLE_INSTALL_RULE)
  set(oneValueArgs DISABLE_TESTS RESULT_CACHE_FILE TEST_COMMAND_OPTIONS_PREFIX)
  set(multiValueArgs DEPENDS DEPENDEES TEST_COMMAND_OPTIONS TEST_SETS_SUPPORTED)
  cmake_parse_arguments(_ "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if (NOT DEFINED __DISABLE_TESTS)
    set(__DISABLE_TESTS OFF)
  endif()

  set(TEST_EXE $<TARGET_FILE_NAME:${TARGET}>)
  set(TEST_EXE_WORKING_DIRECTORY ./${CMAKE_INSTALL_BINDIR})

  if (NOT DEFINED __TEST_SETS_SUPPORTED)
    set(__TEST_SETS_SUPPORTED ${CUTLASS_DEFAULT_ACTIVE_TEST_SETS})
  endif()

  set(TEST_SETS_SUPPORTED ${__TEST_SETS_SUPPORTED})

  if (__RESULT_CACHE_FILE)

    add_custom_command(
      TARGET ${TARGET}
      POST_BUILD
      COMMAND ${CMAKE_COMMAND}
      ARGS -E copy ${__RESULT_CACHE_FILE} "$<TARGET_FILE_DIR:${TARGET}>"
      )

  endif()

  if (NOT __TEST_COMMAND_OPTIONS)
    set(__TEST_COMMAND_OPTIONS " ")
  endif()

  list(LENGTH __TEST_COMMAND_OPTIONS CMD_COUNT)

  if (CMD_COUNT GREATER 1)
    add_custom_target(${NAME} DEPENDS ${TARGET} ${__DEPENDS})
    foreach(DEPENDEE ${__DEPENDEES})
      add_dependencies(${DEPENDEE} ${NAME})
    endforeach()
  endif()

  set(TEST_GROUP_NAME ${NAME})

  foreach(CMD_OPTIONS_VAR IN LISTS __TEST_COMMAND_OPTIONS)

    if (CMD_COUNT GREATER 1)
      string(TOLOWER "${NAME}_${CMD_OPTIONS_VAR}" TEST_NAME)
    else()
      string(TOLOWER "${NAME}" TEST_NAME)
    endif()

    # The following rigmarole is needed to deal with spaces and possible quotes in 
    # command line arguments. The options are passed "by reference" as the actual
    # variable names holding the real options. We then expand these in a way that
    # preserves any quotes. Note, they have to be in this order for it to work for 
    # all the use cases below.

    set(TEST_COMMAND_OPTIONS ${${__TEST_COMMAND_OPTIONS_PREFIX}${CMD_OPTIONS_VAR}})
    list(JOIN TEST_COMMAND_OPTIONS " " TEST_COMMAND_OPTIONS)
    separate_arguments(TEST_COMMAND_OPTIONS)

    add_custom_target(
      ${TEST_NAME}
      COMMAND
      ${CUTLASS_TEST_EXECUTION_ENVIRONMENT} $<TARGET_FILE:${TARGET}> ${TEST_COMMAND_OPTIONS}
      DEPENDS
      ${TARGET}
      )

    if (CMD_COUNT GREATER 1)
      add_dependencies(${NAME} ${TEST_NAME})
    endif()

    foreach(DEPENDEE ${__DEPENDEES})
      add_dependencies(${DEPENDEE} ${TEST_NAME})
    endforeach()

    set(TEST_NAME c${TEST_NAME})
    string(CONFIGURE "${_INLINE_PER_TEST_CODE_TEMPLATE}" _TEST_CODE @ONLY)
    string(APPEND _INLINE_PER_TEST_CODE "${_TEST_CODE}")

  endforeach()

  # To run the tests from an install package with tests enabled, we need to generate test files
  # that don't rely on the current directory structure in build.  

  set(TEST_NAME c${NAME})
  set(TEST_GEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/ctest/${TEST_NAME})
  file(MAKE_DIRECTORY ${TEST_GEN_DIR})

  set(TEST_EXE_PATH $<TARGET_FILE:${TARGET}>)
  set(TEST_USE_EXTENDED_FORMAT ON)
  configure_file("${CUTLASS_CTEST_TEMPLATE_FILE}" "${TEST_GEN_DIR}/CTestTestfile.${TEST_NAME}.cmake" @ONLY)

  set(TEST_EXE_PATH $<TARGET_FILE_NAME:${TARGET}>)
  set(TEST_USE_EXTENDED_FORMAT OFF) # ctest does not support extended add_test format.
  configure_file("${CUTLASS_CTEST_TEMPLATE_FILE}" "${TEST_GEN_DIR}/CTestTestfile.${TEST_NAME}.install.cmake.in" @ONLY)

  # The following line imports the tests for immediate run via `make test`.

  include(${TEST_GEN_DIR}/CTestTestfile.${TEST_NAME}.cmake)
 
  set(CUTLASS_CTEST_GENERATED_FILES ${CUTLASS_CTEST_GENERATED_FILES};ctest/${TEST_NAME}/CTestTestfile.${TEST_NAME}.cmake CACHE INTERNAL "")

endfunction()

add_subdirectory(tools)
add_dependencies(test_all test_profiler)


################################################################################

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/NvidiaCutlassConfigVersion.cmake 
  COMPATIBILITY AnyNewerVersion)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/NvidiaCutlassConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/NvidiaCutlassConfig.cmake  
  @ONLY
  )

install(
  FILES 
    ${CMAKE_CURRENT_BINARY_DIR}/NvidiaCutlassConfig.cmake  
    ${CMAKE_CURRENT_BINARY_DIR}/NvidiaCutlassConfigVersion.cmake 
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/NvidiaCutlass/
  )

install(
  EXPORT NvidiaCutlass
  NAMESPACE nvidia::cutlass::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/NvidiaCutlass/
  FILE NvidiaCutlassTargets.cmake
  )

################################################################################

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/NvidiaCutlassPackageConfig.cmake)

