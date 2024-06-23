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

#include <algorithm>
#include <stdexcept>
#include <iomanip>
#include <cstring>
#include <fstream>
#include <sstream>

#ifdef __unix__
#include <unistd.h>
#elif defined(_WIN32) || defined(WIN32)
#include <windows.h>
#else
// sleep not supported
#endif

#include "cutlass/profiler/options.h"
#include "cutlass/profiler/operation_profiler.h"
#include "cutlass/profiler/gpu_timer.h"

#include "cutlass/trace.h"

///////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace profiler {

///////////////////////////////////////////////////////////////////////////////////////////////////

OperationProfiler::OperationProfiler(): kind_(library::OperationKind::kInvalid) { }

/// Ctor
OperationProfiler::OperationProfiler(
  Options const &options,
  library::OperationKind kind,
  ArgumentDescriptionVector const &arguments,
  ProviderVector const & verification_providers
):
  kind_(kind), arguments_(arguments) {

}

/// Destructor
OperationProfiler::~OperationProfiler() {}

/// Gets the schema description
std::string const & OperationProfiler::description() const {
  return description_;
}

/// Prints usage statement for the math function

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Returns true if the current operation description satisfies the problem space
bool OperationProfiler::satisfies( //used
  library::OperationDescription const &op_desc,
  ProblemSpace const &problem_space,
  ProblemSpace::Problem const &problem) {
  library::OpcodeClassID opcode_class;
  if (arg_as_OpcodeClassID(opcode_class, "op_class", problem_space, problem)) {
    if (opcode_class != op_desc.tile_description.math_instruction.opcode_class) {
      return false;
    }
  }
  int64_t int_value;

  if (arg_as_int(int_value, "inst_m", problem_space, problem)) {
    if (int64_t(op_desc.tile_description.math_instruction.instruction_shape.m()) != int_value) {
      return false;
    }
  }

  if (arg_as_int(int_value, "inst_n", problem_space, problem)) {
    if (int64_t(op_desc.tile_description.math_instruction.instruction_shape.n()) != int_value) {
      return false;
    }
  }

  if (arg_as_int(int_value, "inst_k", problem_space, problem)) {
    if (int64_t(op_desc.tile_description.math_instruction.instruction_shape.k()) != int_value) {
      return false;
    }
  }

  if (arg_as_int(int_value, "cta_m", problem_space, problem)) {
    if (int64_t(op_desc.tile_description.threadblock_shape.m()) != int_value) {
      return false;
    }
  }

  if (arg_as_int(int_value, "cta_n", problem_space, problem)) {
    if (int64_t(op_desc.tile_description.threadblock_shape.n()) != int_value) {
      return false;
    }
  }

  if (arg_as_int(int_value, "cta_k", problem_space, problem)) {
    if (int64_t(op_desc.tile_description.threadblock_shape.k()) != int_value) {
      return false;
    }
  }

  if (arg_as_int(int_value, "cluster_m", problem_space, problem)) {
    if (int64_t(op_desc.tile_description.cluster_shape.m()) != int_value) {
      return false;
    }
  }

  if (arg_as_int(int_value, "cluster_n", problem_space, problem)) {
    if (int64_t(op_desc.tile_description.cluster_shape.n()) != int_value) {
      return false;
    }
  }

  if (arg_as_int(int_value, "cluster_k", problem_space, problem)) {
    if (int64_t(op_desc.tile_description.cluster_shape.k()) != int_value) {
      return false;
    }
  }

  if (arg_as_int(int_value, "stages", problem_space, problem)) {
    if (int64_t(op_desc.tile_description.threadblock_stages) != int_value) {
      return false;
    }
  }

  if (arg_as_int(int_value, "warps_m", problem_space, problem)) {
    if (int64_t(op_desc.tile_description.warp_count.m()) != int_value) {
      return false;
    }
  }

  if (arg_as_int(int_value, "warps_n", problem_space, problem)) {
    if (int64_t(op_desc.tile_description.warp_count.n()) != int_value) {
      return false;
    }
  }

  if (arg_as_int(int_value, "warps_k", problem_space, problem)) {
    if (int64_t(op_desc.tile_description.warp_count.k()) != int_value) {
      return false;
    }
  }

  library::NumericTypeID numeric_type;
  if (arg_as_NumericTypeID(numeric_type, "accum", problem_space, problem)) {
    if (numeric_type != op_desc.tile_description.math_instruction.element_accumulator) {
      return false;
    }
  }

  return true;
}

#if defined(CUTLASS_DEBUG_TRACE_LEVEL) && (CUTLASS_DEBUG_TRACE_LEVEL > 1)
#endif // defined(CUTLASS_DEBUG_TRACE_LEVEL) && (CUTLASS_DEBUG_TRACE_LEVEL > 1)

/// Entry point to profile all operations in the manifest

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Sleep for a given duration in ms
void OperationProfiler::sleep(int sleep_duration) { //used
  if (sleep_duration) {
    #ifdef __unix__
    usleep(sleep_duration * 1000);
    #elif defined(_WIN32) || defined(WIN32)
    SleepEx(sleep_duration, false);
    #else
    // sleep not supported
    #endif
  }
}


/// Compares tensors for equality
Disposition OperationProfiler::compare_tensors( //used
  Options const &options,
  DeviceAllocation &experimental,
  DeviceAllocation &reference,
  int64_t count) {

  if (experimental.type() != reference.type()) {
    return Disposition::kIncorrect;
  }

  bool passed = false;

  if (count == 0) {
    count = reference.capacity();
  }


  return passed ? Disposition::kPassed : Disposition::kIncorrect;
}

/// Saves the workspace
// void OperationProfiler::save_workspace( //no output but called
//   DeviceContext &device_context,
//   Options const &options,
//   library::OperationDescription const &desc,
//   library::Provider provider,
//   library::Provider verification_provider) {
//
//
// }
//

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Method to profile a CUTLASS Operation

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Sets operation description
void OperationProfiler::initialize_result_( //used
  PerformanceResult &result,
  library::OperationDescription const &operation_desc,
  ProblemSpace const &problem_space) {

  set_argument(result, "op_class", problem_space,
    library::to_string(operation_desc.tile_description.math_instruction.opcode_class));

  set_argument(result, "accum", problem_space,
    library::to_string(operation_desc.tile_description.math_instruction.element_accumulator));

  set_argument(result, "cta_m", problem_space, operation_desc.tile_description.threadblock_shape.m());
  set_argument(result, "cta_n", problem_space, operation_desc.tile_description.threadblock_shape.n());
  set_argument(result, "cta_k", problem_space, operation_desc.tile_description.threadblock_shape.k());
  set_argument(result, "cluster_m", problem_space, operation_desc.tile_description.cluster_shape.m());
  set_argument(result, "cluster_n", problem_space, operation_desc.tile_description.cluster_shape.n());
  set_argument(result, "cluster_k", problem_space, operation_desc.tile_description.cluster_shape.k());
  set_argument(result, "stages", problem_space, operation_desc.tile_description.threadblock_stages);
  set_argument(result, "warps_m", problem_space, operation_desc.tile_description.warp_count.m());
  set_argument(result, "warps_n", problem_space, operation_desc.tile_description.warp_count.n());
  set_argument(result, "warps_k", problem_space, operation_desc.tile_description.warp_count.k());
  set_argument(result, "inst_m", problem_space, operation_desc.tile_description.math_instruction.instruction_shape.m());
  set_argument(result, "inst_n", problem_space, operation_desc.tile_description.math_instruction.instruction_shape.n());
  set_argument(result, "inst_k", problem_space, operation_desc.tile_description.math_instruction.instruction_shape.k());
  set_argument(result, "min_cc", problem_space, operation_desc.tile_description.minimum_compute_capability);
  set_argument(result, "max_cc", problem_space, operation_desc.tile_description.maximum_compute_capability);
}

/// Helper
void OperationProfiler::set_argument( //used
  PerformanceResult &result,
  char const *name,
  ProblemSpace const &problem_space,
  std::string const &value) {

  result.arguments.at(problem_space.argument_index(name)) = make_pair(std::string(name), value);
}

void OperationProfiler::set_argument( //used
  PerformanceResult &result,
  char const *name,
  ProblemSpace const &problem_space,
  int64_t value) {

  result.arguments.at(problem_space.argument_index(name)) = make_pair(std::string(name), library::lexical_cast(value));
}


/// finds string matches filter_string in operation_name
bool OperationProfiler::find_string_matches_( //used
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

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace profiler
} // namespace cutlass

///////////////////////////////////////////////////////////////////////////////////////////////////
