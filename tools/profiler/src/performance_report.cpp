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

#include <iostream>
#include <stdexcept>
#include <iomanip>
#include <algorithm>
#include <cstring>

#include "cutlass/library/util.h"
#include "cutlass/profiler/performance_report.h"
#include "cutlass/profiler/debug.h"

namespace cutlass {
namespace profiler {

PerformanceReport::PerformanceReport(
  Options const &options,
  std::vector<std::string> const &argument_names,
  library::OperationKind const &op_kind
):
  options_(options), argument_names_(argument_names), good_(true), op_kind_(op_kind) {}

PerformanceReport::~PerformanceReport() {}

void PerformanceReport::append_result(PerformanceResult result) {
  std::cout
    << "=============================\n"
    << "        Provider: " << library::to_string(result.provider, true) << "\n"
    << "   OperationKind: " << library::to_string(result.op_kind) << "\n"
    << "       Operation: " << result.operation_name << "\n\n"
    << "       Arguments:";

  int column_idx = 0;
  for (auto const &arg : result.arguments) {
    if (!arg.second.empty()) {
      std::cout << " --" << arg.first << "=" << arg.second; 
      column_idx += int(4 + arg.first.size() + arg.second.size());
      if (column_idx > 98) {
	std::cout << "  \\\n                 ";
        column_idx = 0;
      }
    }
  }
  std::cout
    << "\n"
    << "         Runtime: " << result.runtime << "  ms\n"
    << "          Memory: " << result.gbytes_per_sec() << " GiB/s\n"
    << "            Math: " << result.gflops_per_sec() << " GFLOP/s\n";
}

} // namespace profiler
} // namespace cutlass
