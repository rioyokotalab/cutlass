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
   \brief Execution environment
*/

#include <iostream>
#include <stdexcept>
#include <iomanip>
#include <ios>
#include <vector>

#include "cutlass/core_io.h"

#include "cutlass/profiler/cublas_helpers.h"
#include "cutlass/profiler/gemm_operation_profiler.h"
#include "cutlass/profiler/gpu_timer.h"
#include "cutlass/library/singleton.h"
#include "cutlass/library/library.h"
#include "cutlass/library/handle.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace profiler {


/////////////////////////////////////////////////////////////////////////////////////////////////

/// Ctor
GemmOperationProfiler::GemmOperationProfiler(Options const &options):
  OperationProfiler(
    options,
    library::OperationKind::kGemm,
    {
      {ArgumentTypeID::kEnumerated, {"gemm_kind"}, "Variant of GEMM (universal, gemm, planar_complex, planar_complex_array)"},
      {ArgumentTypeID::kInteger, {"m", "problem-size::m"}, "M dimension of the GEMM problem space"},
      {ArgumentTypeID::kInteger, {"n", "problem-size::n"}, "N dimension of the GEMM problem space"},
      {ArgumentTypeID::kInteger, {"k", "problem-size::k"}, "K dimension of the GEMM problem space"},
      {ArgumentTypeID::kTensor, {"A"}, "Tensor storing the A operand"},
      {ArgumentTypeID::kTensor, {"B"}, "Tensor storing the B operand"},
      {ArgumentTypeID::kTensor, {"C"}, "Tensor storing the C operand"},
      {ArgumentTypeID::kTensor, {"D"}, "Tensor storing the D output"},
      {ArgumentTypeID::kScalar, {"alpha", "epilogue::alpha"}, "Epilogue scalar alpha"},
      {ArgumentTypeID::kScalar, {"beta", "epilogue::beta"}, "Epilogue scalar beta"},
      {ArgumentTypeID::kEnumerated, {"split_k_mode", "split-k-mode"}, "Variant of split K mode(serial, parallel)"},
      {ArgumentTypeID::kInteger, {"split_k_slices", "split-k-slices"}, "Number of partitions of K dimension"},
      {ArgumentTypeID::kInteger, {"batch_count", "batch-count"}, "Number of GEMMs computed in one batch"},
      {ArgumentTypeID::kEnumerated, {"raster_order", "raster-order"}, "Raster order (heuristic, along_n, along_m)"},
    },
    { library::Provider::kCUBLAS}
  ) {

  description_ = "      General matrix-matrix product. D = alpha * A*B + beta * C";
}

/// Destructor
GemmOperationProfiler::~GemmOperationProfiler() {}

/// Total number of bytes loaded
int64_t GemmOperationProfiler::GemmProblem::bytes(library::GemmDescription const &operation_desc) const {//used
  int64_t bytes =
    int64_t(library::sizeof_bits(operation_desc.A.element) * m / 8) * k +
    int64_t(library::sizeof_bits(operation_desc.B.element) * n / 8) * k +
    int64_t(library::sizeof_bits(operation_desc.C.element) * m / 8) * n;
  bytes *= batch_count;
  return bytes;
}

/// Total number of flops computed
int64_t GemmOperationProfiler::GemmProblem::flops(library::GemmDescription const &operation_desc) const {//used
  int64_t flops_ = (int64_t(m) * n * k + m * n) * 2 * batch_count;
  return flops_;
}

/// Extracts the problem dimensions
Status GemmOperationProfiler::initialize_configuration(//used
  Options const &options,
  PerformanceReport &report,
  DeviceContext &device_context,
  library::Operation const *operation,
  ProblemSpace const &problem_space,
  ProblemSpace::Problem const &problem) {

  library::GemmDescription const &operation_desc =
    static_cast<library::GemmDescription const &>(operation->description());

  problem_.mode = library::GemmUniversalMode::kGemm;
  problem_.m = 3456;
  problem_.n = 4096;
  problem_.k = 4096;
  problem_.split_k_mode = library::SplitKMode::kSerial;
  problem_.mode = library::GemmUniversalMode::kGemm;
  problem_.split_k_slices = 1;
  problem_.batch_count = 1;
  problem_.raster_order = library::RasterOrder::kHeuristic;
  cast_from_double(problem_.alpha, operation_desc.element_epilogue, 1);
  cast_from_double(problem_.beta, operation_desc.element_epilogue, 0);
  problem_.lda = DeviceAllocation::get_packed_layout(
    operation_desc.A.layout, {int(problem_.m), int(problem_.k)}).front();
  problem_.ldb = DeviceAllocation::get_packed_layout(
    operation_desc.B.layout, {int(problem_.k), int(problem_.n)}).front();
  problem_.ldc = DeviceAllocation::get_packed_layout(
    operation_desc.C.layout, {int(problem_.m), int(problem_.n)}).front();

  gemm_workspace_.configuration.mode = problem_.mode;
  gemm_workspace_.configuration.problem_size.m() = int(problem_.m);
  gemm_workspace_.configuration.problem_size.n() = int(problem_.n);
  gemm_workspace_.configuration.problem_size.k() = int(problem_.k);
  gemm_workspace_.configuration.lda = problem_.lda;
  gemm_workspace_.configuration.ldb = problem_.ldb;
  gemm_workspace_.configuration.ldc = problem_.ldc;
  gemm_workspace_.configuration.ldd = problem_.ldc;
  gemm_workspace_.configuration.batch_count = problem_.split_k_slices;
  gemm_workspace_.arguments.A = nullptr;
  gemm_workspace_.arguments.B = nullptr;
  gemm_workspace_.arguments.C = nullptr;
  gemm_workspace_.arguments.D = nullptr;
  gemm_workspace_.arguments.alpha = problem_.alpha.data();
  gemm_workspace_.arguments.beta = problem_.beta.data();
  gemm_workspace_.arguments.pointer_mode = library::ScalarPointerMode::kHost;
  gemm_workspace_.arguments.raster_order = problem_.raster_order;

  PerformanceResult &result = this->model_result_;
  result.provider = library::Provider::kCUTLASS;
  result.disposition = Disposition::kNotRun;
  result.status = Status::kSuccess;
  result.operation_name = operation_desc.name;
  result.arguments.resize(problem_space.rank());
  set_argument(result, "gemm_kind", problem_space, library::to_string(operation_desc.gemm_kind));
  set_argument(result, "A", problem_space,
    std::string(library::to_string(operation_desc.A.element)) + ":" + library::to_string(operation_desc.A.layout));
  set_argument(result, "B", problem_space,
    std::string(library::to_string(operation_desc.B.element)) + ":" + library::to_string(operation_desc.B.layout));
  set_argument(result, "C", problem_space,
    std::string(library::to_string(operation_desc.C.element)) + ":" + library::to_string(operation_desc.C.layout));
  set_argument(result, "D", problem_space,
    std::string(library::to_string(operation_desc.D.element)) + ":" + library::to_string(operation_desc.D.layout));
  set_argument(result, "m", problem_space, problem_.m);
  set_argument(result, "n", problem_space, problem_.n);
  set_argument(result, "k", problem_space, problem_.k);
  set_argument(result, "split_k_mode", problem_space, library::to_string(problem_.split_k_mode));
  set_argument(result, "split_k_slices", problem_space, problem_.split_k_slices);
  set_argument(result, "batch_count", problem_space, problem_.batch_count);
  set_argument(result, "raster_order", problem_space, library::to_string(problem_.raster_order));
  set_argument(result, "alpha", problem_space,
    library::lexical_cast(problem_.alpha, operation_desc.element_epilogue));
  set_argument(result, "beta", problem_space,
    library::lexical_cast(problem_.beta, operation_desc.element_epilogue));
  set_argument(result, "op_class", problem_space, library::to_string(operation_desc.tile_description.math_instruction.opcode_class));
  set_argument(result, "accum", problem_space, library::to_string(operation_desc.tile_description.math_instruction.element_accumulator));
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
  result.bytes = problem_.bytes(operation_desc);
  result.flops = problem_.flops(operation_desc);
  result.runtime = 0;

  return operation->can_implement(&gemm_workspace_.configuration, &gemm_workspace_.arguments);
}

Status GemmOperationProfiler::initialize_workspace(
  Options const &options,
  PerformanceReport &report,
  DeviceContext &device_context,
  library::Operation const *operation,
  ProblemSpace const &problem_space,
  ProblemSpace::Problem const &problem) {

  library::GemmDescription const &operation_desc =
    static_cast<library::GemmDescription const &>(operation->description());
  int64_t bytes = problem_.bytes(operation_desc);
  gemm_workspace_.problem_count =
    1 + int((3 * int64_t(options.device.properties.l2CacheSize)) / bytes);
  int seed_shift = 0;
  gemm_workspace_.A = device_context.allocate_tensor(
    options,
    "A",
    operation_desc.A.element,
    operation_desc.A.layout,
    {int(problem_.m), int(problem_.k)},
    {int(problem_.lda)},
    problem_.batch_count * gemm_workspace_.problem_count,
    seed_shift++
  );

  gemm_workspace_.B = device_context.allocate_tensor(
    options,
    "B",
    operation_desc.B.element,
    operation_desc.B.layout,
    {int(problem_.k), int(problem_.n)},
    {int(problem_.ldb)},
    problem_.batch_count * gemm_workspace_.problem_count,
    seed_shift++
  );

  gemm_workspace_.C = device_context.allocate_tensor(
    options,
    "C",
    operation_desc.C.element,
    operation_desc.C.layout,
    {int(problem_.m), int(problem_.n)},
    {int(problem_.ldc)},
    problem_.batch_count * gemm_workspace_.problem_count,
    seed_shift++
  );

  gemm_workspace_.Computed = device_context.allocate_tensor(
    "D",
    operation_desc.D.element,
    operation_desc.D.layout,
    {int(problem_.m), int(problem_.n)},
    {int(problem_.ldc)},
    problem_.batch_count * gemm_workspace_.problem_count
  );

  gemm_workspace_.Reference = device_context.allocate_tensor(
    "Reference",
    operation_desc.D.element,
    operation_desc.D.layout,
    {int(problem_.m), int(problem_.n)},
    {int(problem_.ldc)},
    problem_.batch_count * gemm_workspace_.problem_count
  );

  gemm_workspace_.arguments.problem_size = {int(problem_.m), int(problem_.n), int(problem_.k)};
  gemm_workspace_.arguments.batch_count = problem_.batch_count;
  gemm_workspace_.arguments.lda = problem_.lda;
  gemm_workspace_.arguments.ldb = problem_.ldb;
  gemm_workspace_.arguments.ldc = problem_.ldc;
  gemm_workspace_.arguments.ldd = problem_.ldc;
  gemm_workspace_.arguments.batch_stride_A = gemm_workspace_.A->batch_stride();
  gemm_workspace_.arguments.batch_stride_B = gemm_workspace_.B->batch_stride();
  gemm_workspace_.arguments.batch_stride_C = gemm_workspace_.C->batch_stride();
  gemm_workspace_.arguments.batch_stride_D = gemm_workspace_.Computed->batch_stride();
  gemm_workspace_.arguments.sm_count = options.device.properties.multiProcessorCount;

  Status status = Status::kSuccess;
  uint64_t workspace_size = operation->get_host_workspace_size(&gemm_workspace_.configuration);
  gemm_workspace_.host_workspace.resize(workspace_size, 0);
  workspace_size = operation->get_device_workspace_size(&gemm_workspace_.configuration,
							&gemm_workspace_.arguments);
  gemm_workspace_.device_workspace.reset(library::NumericTypeID::kU8, workspace_size);
  status = operation->initialize(
    &gemm_workspace_.configuration,
    gemm_workspace_.host_workspace.data(),
    gemm_workspace_.device_workspace.data());
  results_.push_back(model_result_);
  results_.back().provider = library::Provider::kCUTLASS;
  results_.back().op_kind = library::OperationKind::kGemm;
  results_.back().disposition = Disposition::kNotRun;
  for (auto provider : verification_providers_) {
    results_.back().verification_map[provider] = Disposition::kNotRun;
  }
  return status;
}

bool GemmOperationProfiler::profile( //used
  Options const &options,
  PerformanceReport &report,
  DeviceContext &device_context,
  library::Operation const *operation,
  ProblemSpace const &problem_space,
  ProblemSpace::Problem const &problem) {

  gemm_workspace_.arguments.A = gemm_workspace_.A->data();
  gemm_workspace_.arguments.B = gemm_workspace_.B->data();
  gemm_workspace_.arguments.C = gemm_workspace_.C->data();
  gemm_workspace_.arguments.D = gemm_workspace_.Computed->data();
  gemm_workspace_.arguments.alpha = problem_.alpha.data();
  gemm_workspace_.arguments.beta = problem_.beta.data();
  gemm_workspace_.arguments.pointer_mode = library::ScalarPointerMode::kHost;
  gemm_workspace_.arguments.batch_stride_A = gemm_workspace_.A->batch_stride();
  gemm_workspace_.arguments.batch_stride_B = gemm_workspace_.B->batch_stride();
  gemm_workspace_.arguments.batch_stride_C = gemm_workspace_.C->batch_stride();
  gemm_workspace_.arguments.batch_stride_D = gemm_workspace_.Computed->batch_stride();

  GpuTimer timer;
  sleep(options.profiling.sleep_duration);

  for (int iteration = 0; iteration < options.profiling.warmup_iterations; ++iteration) {
    int problem_idx = (iteration % gemm_workspace_.problem_count) * problem_.batch_count;
    gemm_workspace_.arguments.A = gemm_workspace_.A->batch_data(problem_idx);
    gemm_workspace_.arguments.B = gemm_workspace_.B->batch_data(problem_idx);
    gemm_workspace_.arguments.C = gemm_workspace_.C->batch_data(problem_idx);
    gemm_workspace_.arguments.D = gemm_workspace_.Computed->batch_data(problem_idx);
    operation->run(
      &gemm_workspace_.arguments,
      gemm_workspace_.host_workspace.data(),
      gemm_workspace_.device_workspace.data());
  }

  timer.start();
  int Iterations = options.profiling.iterations;
  int iteration = 0;
  for (; iteration < Iterations; ++iteration) {
    int workspace_idx = options.profiling.warmup_iterations + iteration;
    int problem_idx = (workspace_idx % gemm_workspace_.problem_count) * problem_.batch_count;
    gemm_workspace_.arguments.A = gemm_workspace_.A->batch_data(problem_idx);
    gemm_workspace_.arguments.B = gemm_workspace_.B->batch_data(problem_idx);
    gemm_workspace_.arguments.C = gemm_workspace_.C->batch_data(problem_idx);
    gemm_workspace_.arguments.D = gemm_workspace_.Computed->batch_data(problem_idx);
    operation->run(
      &gemm_workspace_.arguments,
      gemm_workspace_.host_workspace.data(),
      gemm_workspace_.device_workspace.data());
  }
  timer.stop_and_wait();
  results_.back().runtime = timer.duration(iteration);
  return true;
}

} // namespace profiler
} // namespace cutlass

