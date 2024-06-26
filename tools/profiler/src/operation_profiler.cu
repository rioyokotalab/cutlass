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

#include <algorithm>
#include <stdexcept>
#include <iomanip>
#include <cstring>
#include <fstream>
#include <sstream>

#include "cutlass/profiler/options.h"
#include "cutlass/profiler/operation_profiler.h"
#include "cutlass/profiler/gpu_timer.h"
#include "cutlass/trace.h"

namespace cutlass {
namespace profiler {

OperationProfiler::OperationProfiler(
  library::OperationKind kind,
  ArgumentDescriptionVector const &arguments
):
  kind_(kind), arguments_(arguments) {}

OperationProfiler::~OperationProfiler() {}

std::string const & OperationProfiler::description() const {
  return description_;
}

void OperationProfiler::set_argument(
  PerformanceResult &result,
  char const *name,
  ProblemSpace const &problem_space,
  std::string const &value) {
  result.arguments.at(problem_space.argument_index(name)) = make_pair(std::string(name), value);
}

void OperationProfiler::set_argument(
  PerformanceResult &result,
  char const *name,
  ProblemSpace const &problem_space,
  int64_t value) {
  result.arguments.at(problem_space.argument_index(name)) = make_pair(std::string(name), library::lexical_cast(value));
}

} // namespace profiler
} // namespace cutlass
