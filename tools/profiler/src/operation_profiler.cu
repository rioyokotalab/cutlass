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

// OperationProfiler::OperationProfiler(): kind_(library::OperationKind::kInvalid) { }

/// Ctor
OperationProfiler::OperationProfiler(
  Options const &options,
  library::OperationKind kind,
  ArgumentDescriptionVector const &arguments,
  ProviderVector const & verification_providers
)
   {

}

/// Destructor
OperationProfiler::~OperationProfiler() {}

/// Gets the schema description

/// Prints usage statement for the math function

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Returns true if the current operation description satisfies the problem space
#if defined(CUTLASS_DEBUG_TRACE_LEVEL) && (CUTLASS_DEBUG_TRACE_LEVEL > 1)
#endif // defined(CUTLASS_DEBUG_TRACE_LEVEL) && (CUTLASS_DEBUG_TRACE_LEVEL > 1)

/// Entry point to profile all operations in the manifest

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Sleep for a given duration in ms
// void OperationProfiler::sleep(int sleep_duration) { //used
//   if (sleep_duration) {
//     #ifdef __unix__
//     usleep(sleep_duration * 1000);
//     #elif defined(_WIN32) || defined(WIN32)
//     SleepEx(sleep_duration, false);
//     #else
//     // sleep not supported
//     #endif
//   }
// }
//
//
// /// Compares tensors for equality
// Disposition OperationProfiler::compare_tensors( //used
//   Options const &options,
//   DeviceAllocation &experimental,
//   DeviceAllocation &reference,
//   int64_t count) {
//
//   if (experimental.type() != reference.type()) {
//     return Disposition::kIncorrect;
//   }
//
//   bool passed = false;
//
//   if (count == 0) {
//     count = reference.capacity();
//   }
//
//
//   return passed ? Disposition::kPassed : Disposition::kIncorrect;
// }
//
//
// /// Helper
// void OperationProfiler::set_argument( //used
//   PerformanceResult &result,
//   char const *name,
//   ProblemSpace const &problem_space,
//   std::string const &value) {
//
//   result.arguments.at(problem_space.argument_index(name)) = make_pair(std::string(name), value);
// }
//
// void OperationProfiler::set_argument( //used
//   PerformanceResult &result,
//   char const *name,
//   ProblemSpace const &problem_space,
//   int64_t value) {
//
//   result.arguments.at(problem_space.argument_index(name)) = make_pair(std::string(name), library::lexical_cast(value));
// }


///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace profiler
} // namespace cutlass

///////////////////////////////////////////////////////////////////////////////////////////////////
