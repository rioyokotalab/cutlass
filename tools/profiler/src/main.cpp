/***************************************************************************************************
 * Copyright (c) 2017 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/* \file
   \brief 
*/

#include <iostream>

#include "cutlass/profiler/options.h"
#include "cutlass/library/singleton.h"
#include "cutlass/profiler/gemm_operation_profiler.h"

#include "cutlass/profiler/operation_profiler.h"
#include "cutlass/profiler/gpu_timer.h"

#include "cutlass/trace.h"
///////////////////////////////////////////////////////////////////////////////////////////////////

bool find_string_matches_( //used
  std::string const &filter_string,
  std::string const &operation_name) {
  // Returns true if all substrings appear in the operation_name in order

  // Split filter_string of the format "gemm*f32*nt" to tokens ["gemm", "f32", "nt"]
  std::string item;
  std::istringstream iss(filter_string);
  std::vector<std::string> filter_tokens;
  while (std::getline(iss, item, '*')) {
    filter_tokens.push_back(item);
  }

  // Search filter_tokens in operation_name in order
  size_t start = 0, idx = 0;
  for (auto & token : filter_tokens) {
    // Check if characters left to be parsed in operation_name
    if (start < operation_name.length()) {
      // Find token in operation_name[start:]
      idx = operation_name.substr(start).find(token);
      if (idx == std::string::npos) {
        return false;
      }
    }
    start += (idx + token.length());
  }

  // All tokens in filter_string found in operation_name
  return true;
}


int main(int argc, char const *arg[]) {
  cutlass::CommandLine cmdline(argc, arg);
  cutlass::profiler::Options options(cmdline);
  cutlass::profiler::DeviceContext device_context;
  auto profiler = new cutlass::profiler::GemmOperationProfiler(options);
  // return profiler->profile_all(options, cutlass::library::Singleton::get().manifest, device_context);
  //--------------------------------constrcut profiler-----------------------------//
  //
  cutlass::profiler::ArgumentDescriptionVector tile_description_arguments{
    {cutlass::profiler::ArgumentTypeID::kEnumerated, {"op_class", "opcode-class"}, "Class of math instruction (simt, tensorop, wmmatensorop, wmma)"},
    {cutlass::profiler::ArgumentTypeID::kEnumerated, {"accum", "accumulator-type"}, "Math instruction accumulator data type"},
    {cutlass::profiler::ArgumentTypeID::kInteger, {"cta_m", "threadblock-shape::m"}, "Threadblock shape in the M dimension"},
    {cutlass::profiler::ArgumentTypeID::kInteger, {"cta_n", "threadblock-shape::n"}, "Threadblock shape in the N dimension"},
    {cutlass::profiler::ArgumentTypeID::kInteger, {"cta_k", "threadblock-shape::k"}, "Threadblock shape in the K dimension"},
    {cutlass::profiler::ArgumentTypeID::kInteger, {"cluster_m", "cluster-shape::m"}, "Cluster shape in the M dimension"},
    {cutlass::profiler::ArgumentTypeID::kInteger, {"cluster_n", "cluster-shape::n"}, "Cluster shape in the N dimension"},
    {cutlass::profiler::ArgumentTypeID::kInteger, {"cluster_k", "cluster-shape::k"}, "Cluster shape in the K dimension"},
    {cutlass::profiler::ArgumentTypeID::kInteger, {"stages", "threadblock-stages"}, "Number of stages of threadblock-scoped matrix multiply"},
    {cutlass::profiler::ArgumentTypeID::kInteger, {"warps_m", "warp-count::m"}, "Number of warps within threadblock along the M dimension"},
    {cutlass::profiler::ArgumentTypeID::kInteger, {"warps_n", "warp-count::n"}, "Number of warps within threadblock along the N dimension"},
    {cutlass::profiler::ArgumentTypeID::kInteger, {"warps_k", "warp-count::k"}, "Number of warps within threadblock along the K dimension"},
    {cutlass::profiler::ArgumentTypeID::kInteger, {"inst_m", "instruction-shape::m"}, "Math instruction shape in the M dimension"},
    {cutlass::profiler::ArgumentTypeID::kInteger, {"inst_n", "instruction-shape::n"}, "Math instruction shape in the N dimension"},
    {cutlass::profiler::ArgumentTypeID::kInteger, {"inst_k", "instruction-shape::k"}, "Math instruction shape in the K dimension"},
    {cutlass::profiler::ArgumentTypeID::kInteger, {"min_cc", "minimum-compute-capability"}, "Minimum device compute capability"},
    {cutlass::profiler::ArgumentTypeID::kInteger, {"max_cc", "maximum-compute-capability"}, "Maximum device compute capability"}
  };

  profiler->arguments_.insert(profiler->arguments_.end(), tile_description_arguments.begin(), tile_description_arguments.end());
  //---------------------------------profile_all()---------------------------------//
  // cutlass::library::Manifest &manifest = cutlass::library::Singleton::get().manifest;
  const cutlass::library::Manifest &manifest = cutlass::library::Singleton::get().manifest;

  cutlass::profiler::ProblemSpace problem_space(profiler->arguments_, options.cmdline);
   // 1. Construct performance report
  cutlass::profiler::PerformanceReport report(options, problem_space.argument_names(), profiler->kind_);

  // 2. For each problem in problem space
  cutlass::profiler::ProblemSpace::Iterator problem_it = problem_space.begin();
  cutlass::profiler::ProblemSpace::Iterator problem_end = problem_space.end();

  bool continue_profiling = true;
  int retval = 0;

  // For each problem in problem space
  cutlass::profiler::ProblemSpace::Problem problem = problem_it.at();
  report.next_problem();

    // For each operation in manifest
  int matched_operation_count = 0;
  auto operation_ptr = manifest.begin();
  cutlass::library::Operation const *operation = operation_ptr->get();

  auto min_cc = operation->description().tile_description.minimum_compute_capability;
  auto max_cc = operation->description().tile_description.maximum_compute_capability;


  device_context.free();

  std::string operation_name(operation->description().name);
  // Filter kernels by name
  /*
  bool filtered_by_name = options.operation_names.empty();
  if (!filtered_by_name) {//must have something in operationnames , but no output in for loop below
    
    for (auto const & op_name : options.operation_names) {
      if (find_string_matches_(op_name, operation_name)) {
	filtered_by_name = true;
	break;
      }
    }
  }
  */

  // we have found a kernel match, so increment the counter for match kernels
  ++matched_operation_count;

  // A. Initialize configuration
  cutlass::Status status = profiler->initialize_configuration(
    options,
    report,
    device_context,
    operation,
    problem_space,
    problem);

  if (continue_profiling) {
    status = profiler->initialize_workspace(
      options,
      report,
      device_context,
      operation,
      problem_space,
      problem);


  }

  //
  // Profile CUTLASS if it is enabled
  //

  // B. Verify CUTLASS
  if (continue_profiling && options.profiling.provider_enabled(cutlass::library::Provider::kCUTLASS)) {

    continue_profiling = profiler->verify_cutlass(
      options,
      report,
      device_context,
      operation,
      problem_space,
      problem);

    retval |= (not continue_profiling);
  }

  //
  // D. Profile
  //

  if (continue_profiling && options.profiling.enabled) {

    continue_profiling = profiler->profile(
      options,
      report,
      device_context,
      operation,
      problem_space,
      problem);
  }

  report.append_results(profiler->results_);
  profiler->results_.clear();

  return retval;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
