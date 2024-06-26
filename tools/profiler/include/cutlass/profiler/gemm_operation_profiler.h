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
   \brief Gemm Profiler
*/

#pragma once

#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include <unordered_map>

// CUTLASS Library includes
#include "cutlass/library/library.h"
#include "cutlass/library/util.h"
#include "cutlass/library/manifest.h"

// Profiler includes
#include "options.h"
#include "device_context.h"
#include "performance_result.h"
#include "performance_report.h"
#include "problem_space.h"

namespace cutlass {
namespace profiler {

class GemmOperationProfiler {
public:

  struct GemmProblem {
    cutlass::library::GemmUniversalMode mode{library::GemmUniversalMode::kGemm};
    int64_t m{16};
    int64_t n{16};
    int64_t k{16};
    int64_t lda{0};
    int64_t ldb{0};
    int64_t ldc{0};
    std::vector<uint8_t> alpha;
    std::vector<uint8_t> beta;
    cutlass::library::SplitKMode split_k_mode{library::SplitKMode::kNone};
    int split_k_slices{1};
    int batch_count{1};
    cutlass::library::RasterOrder raster_order{cutlass::library::RasterOrder::kHeuristic};
    std::vector<uint8_t> alpha_one;
    std::vector<uint8_t> beta_zero;

    void initialize_result(
      PerformanceResult &result,
      library::GemmDescription const &operation_desc,
      ProblemSpace const &problem_space);
  };

  struct GemmWorkspace {
    DeviceAllocation *A{nullptr};
    DeviceAllocation *B{nullptr};
    DeviceAllocation *C{nullptr};
    DeviceAllocation *Computed{nullptr};
    DeviceAllocation *Reference{nullptr};
    int problem_count{1};
    library::GemmUniversalConfiguration configuration;
    library::GemmUniversalArguments arguments;
    std::vector<uint8_t> host_workspace;
    DeviceAllocation device_workspace;
    library::ReductionConfiguration reduction_configuration;
    library::ReductionArguments reduction_arguments;
    std::vector<uint8_t> reduction_host_workspace;
  };

  GemmProblem problem_;
  GemmWorkspace gemm_workspace_;

  PerformanceResult model_result_;
  PerformanceResultVector results_;
  library::Operation const *reduction_op_;
  library::OperationKind kind_;
  ArgumentDescriptionVector arguments_;

  GemmOperationProfiler(
    library::OperationKind kind, 
    ArgumentDescriptionVector const &arguments = ArgumentDescriptionVector());

  ~GemmOperationProfiler();

  GemmProblem const& problem() const { return problem_; }

  int64_t bytes(library::GemmDescription const &operation_desc) const;

  int64_t flops() const;

  static void set_argument(  
    PerformanceResult &result,
    char const *name,
    ProblemSpace const &problem_space,
    std::string const &value);

  static void set_argument(  
    PerformanceResult &result,
    char const *name,
    ProblemSpace const &problem_space,
    int64_t value);

  void initialize_configuration(
    Options const &options,
    PerformanceReport &report,
    DeviceContext &device_context,
    library::Operation const *operation,
    ProblemSpace const &problem_space,
    ProblemSpace::Problem const &problem);

  void initialize_workspace(
    Options const &options,
    PerformanceReport &report,
    DeviceContext &device_context,
    library::Operation const *operation,
    ProblemSpace const &problem_space,
    ProblemSpace::Problem const &problem);

  void profile(
    Options const &options,
    PerformanceReport &report,
    DeviceContext &device_context,
    library::Operation const *operation,
    ProblemSpace const &problem_space,
    ProblemSpace::Problem const &problem);
};

} // namespace profiler
} // namespace cutlass
