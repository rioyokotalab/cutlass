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

int main(int argc, char const *arg[]) {
  cutlass::CommandLine cmdline(argc, arg);
  cutlass::profiler::Options options(cmdline);
  cutlass::profiler::DeviceContext device_context;
  auto profiler = new cutlass::profiler::GemmOperationProfiler(options);
  // return profiler->profile_all(options, cutlass::library::Singleton::get().manifest, device_context);
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
  for (; continue_profiling && problem_it != problem_end; ++problem_it) {
    cutlass::profiler::ProblemSpace::Problem problem = problem_it.at();
    report.next_problem();

    // For each operation in manifest
    int matched_operation_count = 0;
    for (auto const& operation_ptr : manifest) {

      cutlass::library::Operation const *operation = operation_ptr.get();

      auto min_cc = operation->description().tile_description.minimum_compute_capability;
      auto max_cc = operation->description().tile_description.maximum_compute_capability;


      device_context.free();

      // Execute compatible cutlass operations if they satisfy the current device's compute capability
      if (operation->description().kind == profiler->kind_ &&
          operation->description().provider == cutlass::library::Provider::kCUTLASS &&
          options.device.compute_capability() >= min_cc &&
          options.device.compute_capability() <= max_cc) {

        std::string operation_name(operation->description().name);
        // Filter kernels by name
        bool filtered_by_name = options.operation_names.empty();
        if (!filtered_by_name) {

          for (auto const & op_name : options.operation_names) {
            if (profiler->find_string_matches_(op_name, operation_name)) {
              filtered_by_name = true;
              break;
            }
          }
        }

        for (auto const & op_name : options.excluded_operation_names) {
          if (profiler->find_string_matches_(op_name, operation_name)) {
            filtered_by_name = false;
            break;
          }
        }

        if (!filtered_by_name || !profiler->satisfies(operation->description(), problem_space, problem)) {
          continue;
        }

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

        if (status == cutlass::Status::kErrorInternal) {

          // If there was an internal error, consume the CUDA error and move to the next operation.
          (void)cudaGetLastError();

          report.append_results(profiler->results_);
          continue;
        }
        else if (status != cutlass::Status::kSuccess) {
          // If the workspace could not be initialized for any other reason, continue to
          // the next operation.
          continue;
        }

        if (continue_profiling) {

          if (options.report.print_kernel_before_running) {
            std::cout << "Profiling kernel for JUnit test " << options.report.junit_output_path << ": "
                      << operation_name << std::endl;
          }

          status = profiler->initialize_workspace(
            options,
            report,
            device_context,
            operation,
            problem_space,
            problem);

          if (status == cutlass::Status::kErrorInternal) {

            // If there was an internal error, consume the CUDA error and move to the next operation.
            (void)cudaGetLastError();

            report.append_results(profiler->results_);
            continue;
          }
          else if (status != cutlass::Status::kSuccess) {
            // If the workspace could not be initialized for any other reason, continue to
            // the next operation.
            continue;
          }
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

        if (options.execution_mode == cutlass::profiler::ExecutionMode::kDryRun) {
          report.append_results(profiler->results_);
          profiler->results_.clear();
          continue;
        }

        //
        // C. Optionally save workspace
        //

        if (options.verification.save_workspace == cutlass::profiler::SaveWorkspace::kAlways) {
          profiler->save_workspace(
            device_context,
            options,
            operation->description(),
            cutlass::library::Provider::kCUTLASS);
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
      }

      if (!continue_profiling) {
        break;
      }
    }

    // If we did not find any kernels that match our filters and error_on_no_match was set, report an error
    if (options.profiling.error_on_no_match && matched_operation_count <= 0) {
      #if !NDEBUG
      std::cout << "Error: No matching kernels found with kernel selection filters [--error_on_no_match]" << std::endl;
      #endif
      retval = 1;
    }
  }

  return retval;
}

// int OperationProfiler::profile_all(
//   Options const &options,
//   library::Manifest const &manifest,
//   DeviceContext &device_context) {
//   ProblemSpace problem_space(arguments_, options.cmdline);
//
//   // 1. Construct performance report
//   PerformanceReport report(options, problem_space.argument_names(), kind_);
//
//   // 2. For each problem in problem space
//   ProblemSpace::Iterator problem_it = problem_space.begin();
//   ProblemSpace::Iterator problem_end = problem_space.end();
//
//   bool continue_profiling = true;
//   int retval = 0;
//
//   // For each problem in problem space
//   for (; continue_profiling && problem_it != problem_end; ++problem_it) {
//     ProblemSpace::Problem problem = problem_it.at();
//     report.next_problem();
//
//     // For each operation in manifest
//     int matched_operation_count = 0;
//     for (auto const& operation_ptr : manifest) {
//
//       library::Operation const *operation = operation_ptr.get();
// #if defined(CUTLASS_DEBUG_TRACE_LEVEL) && (CUTLASS_DEBUG_TRACE_LEVEL > 1)
//       std::cerr << "  Operation: " << typeid(*operation).name() << "\n"
//                 << "    name: " << operation->description().name << "\n"
//                 << "    kind: " << operation->description().kind << "\n"
//                 << "    provider: " << operation->description().provider << "\n";
// #endif // CUTLASS_DEBUG_TRACE_LEVEL
//
//       auto min_cc = operation->description().tile_description.minimum_compute_capability;
//       auto max_cc = operation->description().tile_description.maximum_compute_capability;
//
// #if defined(CUTLASS_DEBUG_TRACE_LEVEL) && (CUTLASS_DEBUG_TRACE_LEVEL > 1)
//       std::cerr << "    min_cc: " << min_cc << "\n";
//       std::cerr << "    max_cc: " << min_cc << "\n";
// #endif
//
//       // Clear named allocations
//       device_context.free();
//
// #if defined(CUTLASS_DEBUG_TRACE_LEVEL) && (CUTLASS_DEBUG_TRACE_LEVEL > 1)
//       if (operation->description().kind != kind_) {
//         std::cerr << "    @ kind " << operation->description().kind
//                   << " != kind_ " << kind_ << "\n";
//       }
//       if (operation->description().provider != library::Provider::kCUTLASS) {
//         std::cerr << "    @ provider " << operation->description().provider
//                   << " != library::Provider::kCUTLASS\n";
//       }
//       if (options.device.compute_capability() < min_cc) {
//         std::cerr << "    @ compute_capability "
//                   << options.device.compute_capability()
//                   << " < min_cc " << min_cc << "\n";
//       }
//       if (options.device.compute_capability() > max_cc) {
//         std::cerr << "    @ compute_capability "
//                   << options.device.compute_capability()
//                   << " > max_cc " << max_cc << "\n";
//       }
// #endif
//
//       // Execute compatible cutlass operations if they satisfy the current device's compute capability
//       if (operation->description().kind == kind_ &&
//           operation->description().provider == library::Provider::kCUTLASS &&
//           options.device.compute_capability() >= min_cc &&
//           options.device.compute_capability() <= max_cc) {
//
//         std::string operation_name(operation->description().name);
//         // Filter kernels by name
//         bool filtered_by_name = options.operation_names.empty();
//         if (!filtered_by_name) {
//
//           for (auto const & op_name : options.operation_names) {
//             if (find_string_matches_(op_name, operation_name)) {
//               filtered_by_name = true;
//               break;
//             }
//           }
//         }
//
//         for (auto const & op_name : options.excluded_operation_names) {
//           if (find_string_matches_(op_name, operation_name)) {
//             filtered_by_name = false;
//             break;
//           }
//         }
//
//         if (!filtered_by_name || !satisfies(operation->description(), problem_space, problem)) {
//           continue;
//         }
//
//         // we have found a kernel match, so increment the counter for match kernels
//         ++matched_operation_count;
//
//         // A. Initialize configuration
//         Status status = this->initialize_configuration(
//           options,
//           report,
//           device_context,
//           operation,
//           problem_space,
//           problem);
//
//         if (status == Status::kErrorInternal) {
//
//           // If there was an internal error, consume the CUDA error and move to the next operation.
//           (void)cudaGetLastError();
//
//           report.append_results(results_);
//           continue;
//         }
//         else if (status != Status::kSuccess) {
//           // If the workspace could not be initialized for any other reason, continue to
//           // the next operation.
//           continue;
//         }
//
//         if (continue_profiling) {
//
//           if (options.report.print_kernel_before_running) {
//             std::cout << "Profiling kernel for JUnit test " << options.report.junit_output_path << ": "
//                       << operation_name << std::endl;
//           }
//
//           status = this->initialize_workspace(
//             options,
//             report,
//             device_context,
//             operation,
//             problem_space,
//             problem);
//
//           if (status == Status::kErrorInternal) {
//
//             // If there was an internal error, consume the CUDA error and move to the next operation.
//             (void)cudaGetLastError();
//
//             report.append_results(results_);
//             continue;
//           }
//           else if (status != Status::kSuccess) {
//             // If the workspace could not be initialized for any other reason, continue to
//             // the next operation.
//             continue;
//           }
//         }
//
//         //
//         // Profile CUTLASS if it is enabled
//         //
//
//         // B. Verify CUTLASS
//         if (continue_profiling && options.profiling.provider_enabled(library::Provider::kCUTLASS)) {
//
//           continue_profiling = this->verify_cutlass(
//             options,
//             report,
//             device_context,
//             operation,
//             problem_space,
//             problem);
//
//           retval |= (not continue_profiling);
//         }
//
//         if (options.execution_mode == ExecutionMode::kDryRun) {
//           report.append_results(results_);
//           results_.clear();
//           continue;
//         }
//
//         //
//         // C. Optionally save workspace
//         //
//
//         if (options.verification.save_workspace == SaveWorkspace::kAlways) {
//           save_workspace(
//             device_context,
//             options,
//             operation->description(),
//             library::Provider::kCUTLASS);
//         }
//
//         //
//         // D. Profile
//         //
//
//         if (continue_profiling && options.profiling.enabled) {
//
//           continue_profiling = this->profile(
//             options,
//             report,
//             device_context,
//             operation,
//             problem_space,
//             problem);
//         }
//
//         report.append_results(results_);
//         results_.clear();
//       }
//
//       if (!continue_profiling) {
//         break;
//       }
//     }
//
//     // If we did not find any kernels that match our filters and error_on_no_match was set, report an error
//     if (options.profiling.error_on_no_match && matched_operation_count <= 0) {
//       #if !NDEBUG
//       std::cout << "Error: No matching kernels found with kernel selection filters [--error_on_no_match]" << std::endl;
//       #endif
//       retval = 1;
//     }
//   }
//
//   return retval;
// }
///////////////////////////////////////////////////////////////////////////////////////////////////
