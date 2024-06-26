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
   \brief Defines a math function
*/

#pragma once

#include <vector>
#include <string>
#include <memory>
#include <unordered_map>

// CUTLASS includes
#include "cutlass/trace.h"

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
#include "debug.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace profiler {

/////////////////////////////////////////////////////////////////////////////////////////////////

class OperationProfiler {

public:
  /// Top-level operation kind
  library::OperationKind kind_;

  /// Human readable description
  std::string description_;

  /// Arguments parsed from command line
  ArgumentDescriptionVector arguments_;

  /// List of providers used to verify and compare each result
  ProviderVector verification_providers_;

  /// Model performance result initialized by the operation profiler with workload statistics
  /// and reasonable default state.
  PerformanceResult model_result_;

  /// Performance result vector constructed by profiling the operation
  PerformanceResultVector results_;

  OperationProfiler();

  OperationProfiler(
    library::OperationKind kind, 
    ArgumentDescriptionVector const &arguments = ArgumentDescriptionVector(),
    ProviderVector const & verification_providers = ProviderVector());

  /// Destructor
  virtual ~OperationProfiler();

  /// Obtains the operation kind
  library::OperationKind kind() const { return kind_; }

  /// Gets the schema description
  std::string const &description() const;

  /// Returns a reference to the arguments
  ArgumentDescriptionVector const &arguments() const { return arguments_; }

  /// Sleep for a given duration in ms
  static void sleep(int sleep_duration);

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
};

using OperationProfilerVector = std::vector<std::unique_ptr<OperationProfiler>>;

} // namespace profiler
} // namespace cutlass
