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
   \brief Command line options for performance test program
*/

#include <algorithm>

#include "cutlass/cutlass.h"
#include "cutlass/version.h"

#include "cutlass/library/util.h"

#include "cutlass/profiler/options.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace profiler {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Newline and indent for help strings
static char const *end_of_line = "\n                                             ";

/////////////////////////////////////////////////////////////////////////////////////////////////

Options::Device::Device(cutlass::CommandLine const &cmdline) { //used

  cmdline.get_cmd_line_argument("device", device, 0);

  cudaError_t result;
  result = cudaGetDeviceProperties(&properties, device);

  if (result != cudaSuccess) {
    throw std::runtime_error("cudaGetDeviceProperties() failed for given device");
  }

  result = cudaSetDevice(device);
  if (result != cudaSuccess) {
    throw std::runtime_error("cudaSetDevice() failed for given device.");
  }

  // Permit overriding the compute capability
  if (cmdline.check_cmd_line_flag("compute-capability")) {
    int cc = compute_capability();
    cmdline.get_cmd_line_argument("compute-capability", cc, cc);
    properties.major = cc / 10;
    properties.minor = cc % 10;
  }
  
  // Permit overriding the L2 cache capacity
  if (cmdline.check_cmd_line_flag("llc-capacity")) {
    int llc_capacity = 0;
    cmdline.get_cmd_line_argument("llc-capacity", llc_capacity, 0);

    if (llc_capacity >= 0) {
      properties.l2CacheSize = (llc_capacity << 10);
    }
  }

}

//
///// Returns the compute capability of the listed device (e.g. 61, 60, 70, 75)
int Options::Device::compute_capability() const {
  return properties.major * 10 + properties.minor;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

Options::Initialization::Initialization(cutlass::CommandLine const &cmdline) { //used

  cmdline.get_cmd_line_argument("initialization-enabled", enabled, true);

  if (cmdline.check_cmd_line_flag("initialization-provider")) {
//    std::string str;
//    cmdline.get_cmd_line_argument("initialization-provider", str);
//    provider = library::from_string<library::Provider>(str);
//    if () {
//      enabled = false;
//    }
//    else if (provider != library::Provider::kReferenceHost && provider != library::Provider::kReferenceDevice) {
//      throw std::runtime_error("Unsupported initialization provider specified.");
//    }
  }
  else {
    provider = library::Provider::kReferenceDevice;
  }

  cmdline.get_cmd_line_argument("seed", seed, 2019);

//  if (cmdline.check_cmd_line_flag("dist")) {
//    // user has set the data distribution (fix data distribution once set)
//    fix_data_distribution = true;
//    // set user provided data distribution
//    get_distribution(cmdline, "dist", data_distribution);
//  }
//  else {
//    // profiler chosen data distribution (allowed to change based on numeric types)
//    fix_data_distribution = false;
//    // set uniform data distribution with range [-4, 4] 
//    data_distribution.set_uniform(-4, 4, 0);
//  }
  

}


Options::Library::Library(cutlass::CommandLine const &cmdline) { //used

  algorithm_mode = AlgorithmMode::kDefault;

//  if (cmdline.check_cmd_line_flag("library-algo-mode")) {
//    std::string mode = "default";
//    cmdline.get_cmd_line_argument("library-algo-mode", mode);
//    algorithm_mode = from_string<AlgorithmMode>(mode);
//  }  

  if (cmdline.check_cmd_line_flag("library-algos")) {

    // If algorithms are specified, override as kBest.
    algorithm_mode = AlgorithmMode::kBest;

    std::vector<std::string> tokens;
    cmdline.get_cmd_line_arguments("library-algos", tokens);

    algorithms.reserve(tokens.size());

    for (auto const & token : tokens) {
      if (token.find(":")) {
        // TODO: tokenized range
      }
      else {
        int algo;
        std::stringstream ss; 

        ss << token;
        ss >> algo;

        algorithms.push_back(algo);
      }
    }
  }
}

Options::Profiling::Profiling(cutlass::CommandLine const &cmdline) { //used

  cmdline.get_cmd_line_argument("workspace-count", workspace_count, 0);  
  cmdline.get_cmd_line_argument("warmup-iterations", warmup_iterations, 10);
  cmdline.get_cmd_line_argument("profiling-iterations", iterations, 100);
  cmdline.get_cmd_line_argument("sleep-duration", sleep_duration, 50);
  cmdline.get_cmd_line_argument("profiling-enabled", enabled, true);
  
  if (cmdline.check_cmd_line_flag("providers")) {

    std::vector<std::string> tokens;
    cmdline.get_cmd_line_arguments("providers", tokens);

    providers.clear();

//    for (auto const &token : tokens) {
//      providers.push_back(library::from_string<library::Provider>(token));
//    }
  }
  else {
    providers.push_back(library::Provider::kCUTLASS);
    providers.push_back(library::Provider::kCUBLAS);
    providers.push_back(library::Provider::kCUDNN);      
  }
}


/// Returns true if a provider is enabled
bool Options::Profiling::provider_enabled(library::Provider provider) const { //use 4 times
  return std::find(providers.begin(), providers.end(), provider) != providers.end();
}

/// Returns the index of a provider if its enabled


/////////////////////////////////////////////////////////////////////////////////////////////////

Options::Verification::Verification(cutlass::CommandLine const &cmdline) { //used
  
  cmdline.get_cmd_line_argument("verification-enabled", enabled, true);
  if (enabled) {
    cmdline.get_cmd_line_argument("verification-required", required, false);
  }

  cmdline.get_cmd_line_argument("epsilon", epsilon, 0.05);

  cmdline.get_cmd_line_argument("nonzero-floor", nonzero_floor, 1.0 / 256.0);

  if (cmdline.check_cmd_line_flag("save-workspace")) {
//    std::string value;
//    cmdline.get_cmd_line_argument("save-workspace", value);
//    save_workspace = from_string<SaveWorkspace>(value);
  }
  else {
    save_workspace = SaveWorkspace::kNever;
  }

  if (cmdline.check_cmd_line_flag("verification-providers")) {
    
//    std::vector<std::string> tokens;
//    cmdline.get_cmd_line_arguments("verification-providers", tokens);
//
//    providers.clear();
//
//    for (auto const &token : tokens) {
//      library::Provider provider = library::from_string<library::Provider>(token);
//      if (provider != library::Provider::kInvalid) {
//        providers.push_back(provider);
//      }
//    }
  }
  else {
    providers.push_back(library::Provider::kCUBLAS);
    providers.push_back(library::Provider::kReferenceDevice);
    providers.push_back(library::Provider::kCUDNN);      
  }
}

/// Returns true if a provider is enabled
bool Options::Verification::provider_enabled(library::Provider provider) const {
  return std::find(providers.begin(), providers.end(), provider) != providers.end();
}

/////////////////////////////////////////////////////////////////////////////////////////////////

Options::Report::Report(cutlass::CommandLine const &cmdline) {//used
  
  cmdline.get_cmd_line_argument("append", append, false);
  cmdline.get_cmd_line_argument("output", output_path);
  cmdline.get_cmd_line_argument("junit-output", junit_output_path);
 
  if (cmdline.check_cmd_line_flag("tags")) {
    cmdline.get_cmd_line_argument_pairs("tags", pivot_tags);
  }

  cmdline.get_cmd_line_argument("report-not-run", report_not_run, false);

  cmdline.get_cmd_line_argument("verbose", verbose, true);

  cmdline.get_cmd_line_argument("sort-results", sort_results, false);

  cmdline.get_cmd_line_argument("print-kernel-before-running", print_kernel_before_running, false);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

Options::About::About(cutlass::CommandLine const &cmdline) { //used
  help = cmdline.check_cmd_line_flag("help");
  version = cmdline.check_cmd_line_flag("version");
  device_info = cmdline.check_cmd_line_flag("device-info");
}



/////////////////////////////////////////////////////////////////////////////////////////////////

Options::Options(cutlass::CommandLine const &cmdline)://if delete this function, will output nothing
  cmdline(cmdline),
  device(cmdline),
  initialization(cmdline),
  library(cmdline),
  profiling(cmdline), 
  verification(cmdline), 
  report(cmdline),
  about(cmdline) {
  
  if (cmdline.check_cmd_line_flag("mode")) {
//    std::string token;
//    cmdline.get_cmd_line_argument("mode", token);
//    execution_mode = from_string<ExecutionMode>(token);
  }
  else {
    execution_mode = ExecutionMode::kProfile;
  }

  // Enumerating kernels is equivalent to a dry run.
  if (execution_mode == ExecutionMode::kEnumerate) {
    execution_mode = ExecutionMode::kDryRun;
  }

  if (cmdline.check_cmd_line_flag("operation")) {
//    std::string str;
//    cmdline.get_cmd_line_argument("operation", str);
//    operation_kind = library::from_string<library::OperationKind>(str);
  }
  else if (cmdline.check_cmd_line_flag("function")) {
//    std::string str;
//    cmdline.get_cmd_line_argument("function", str);
//    operation_kind = library::from_string<library::OperationKind>(str);
  }
  else {
    operation_kind = library::OperationKind::kInvalid;
  }

  if (cmdline.check_cmd_line_flag("operation_names")) {
    cmdline.get_cmd_line_arguments("operation_names", operation_names);
  }
  else if (cmdline.check_cmd_line_flag("kernels")) {
    cmdline.get_cmd_line_arguments("kernels", operation_names);
    profiling.error_on_no_match = cmdline.check_cmd_line_flag("error-on-no-match");
  }

  if (cmdline.check_cmd_line_flag("ignore-kernels")) {
    cmdline.get_cmd_line_arguments("ignore-kernels", excluded_operation_names);
    profiling.error_on_no_match = cmdline.check_cmd_line_flag("error-on-no-match");
  }

  // Prevent launches on the device for anything other than CUTLASS operation
  // Allow verification only on host
  if (execution_mode == ExecutionMode::kTrace) {
    initialization.provider = library::Provider::kReferenceHost;
    verification.providers = {library::Provider::kReferenceHost};
    profiling.enabled = false;
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace profiler
} // namespace cutlass
