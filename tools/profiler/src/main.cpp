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
#include "cutlass/profiler/options.h"
#include "cutlass/library/singleton.h"
#include "cutlass/library/library.h"
#include "cutlass/profiler/gemm_operation_profiler.h"
#include "cutlass/profiler/operation_profiler.h"
#include "cutlass/profiler/gpu_timer.h"
#include "cutlass/trace.h"

using cutlass;

int main(int argc, char const *arg[]) {
  CommandLine cmdline(argc, arg);
  profiler::Options options(cmdline);
  profiler::DeviceContext device_context;
  auto profiler = new profiler::GemmOperationProfiler(options);
  profiler::ArgumentDescriptionVector tile_description_arguments{
    {profiler::ArgumentTypeID::kEnumerated, {"op_class", "opcode-class"}, "Class of math instruction (simt, tensorop, wmmatensorop, wmma)"},
    {profiler::ArgumentTypeID::kEnumerated, {"accum", "accumulator-type"}, "Math instruction accumulator data type"},
    {profiler::ArgumentTypeID::kInteger, {"cta_m", "threadblock-shape::m"}, "Threadblock shape in the M dimension"},
    {profiler::ArgumentTypeID::kInteger, {"cta_n", "threadblock-shape::n"}, "Threadblock shape in the N dimension"},
    {profiler::ArgumentTypeID::kInteger, {"cta_k", "threadblock-shape::k"}, "Threadblock shape in the K dimension"},
    {profiler::ArgumentTypeID::kInteger, {"cluster_m", "cluster-shape::m"}, "Cluster shape in the M dimension"},
    {profiler::ArgumentTypeID::kInteger, {"cluster_n", "cluster-shape::n"}, "Cluster shape in the N dimension"},
    {profiler::ArgumentTypeID::kInteger, {"cluster_k", "cluster-shape::k"}, "Cluster shape in the K dimension"},
    {profiler::ArgumentTypeID::kInteger, {"stages", "threadblock-stages"}, "Number of stages of threadblock-scoped matrix multiply"},
    {profiler::ArgumentTypeID::kInteger, {"warps_m", "warp-count::m"}, "Number of warps within threadblock along the M dimension"},
    {profiler::ArgumentTypeID::kInteger, {"warps_n", "warp-count::n"}, "Number of warps within threadblock along the N dimension"},
    {profiler::ArgumentTypeID::kInteger, {"warps_k", "warp-count::k"}, "Number of warps within threadblock along the K dimension"},
    {profiler::ArgumentTypeID::kInteger, {"inst_m", "instruction-shape::m"}, "Math instruction shape in the M dimension"},
    {profiler::ArgumentTypeID::kInteger, {"inst_n", "instruction-shape::n"}, "Math instruction shape in the N dimension"},
    {profiler::ArgumentTypeID::kInteger, {"inst_k", "instruction-shape::k"}, "Math instruction shape in the K dimension"},
    {profiler::ArgumentTypeID::kInteger, {"min_cc", "minimum-compute-capability"}, "Minimum device compute capability"},
    {profiler::ArgumentTypeID::kInteger, {"max_cc", "maximum-compute-capability"}, "Maximum device compute capability"}
  };

  profiler->arguments_.insert(profiler->arguments_.end(), tile_description_arguments.begin(), tile_description_arguments.end());
  const library::Manifest &manifest = library::Singleton::get().manifest;
  profiler::ProblemSpace problem_space(profiler->arguments_, options.cmdline);
  profiler::PerformanceReport report(options, problem_space.argument_names(), profiler->kind_);
  profiler::ProblemSpace::Iterator problem_it = problem_space.begin();
  profiler::ProblemSpace::Iterator problem_end = problem_space.end();
  profiler::ProblemSpace::Problem problem = problem_it.at();
  report.next_problem();
  auto operation_ptr = manifest.begin();
  library::Operation const *operation = operation_ptr->get();
  device_context.free();
  std::string operation_name(operation->description().name);

  //profiler->initialize_configuration(options, report, device_context, operation, problem_space, problem);
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
  problem_.initialize_result(result, operation_desc, problem_space);
  set_argument(result, "op_class", problem_space, library::to_string(operation_desc.tile_description.math_instruction.opcode_class));
  set_argument(result, "accum", problem_space, library::to_string(operation_desc.tile_description.math_instruction.element_accumulator));
  set_argument(result, "cta_m", problem_space, operation_desc.tile_description.threadblock_shape.m());
  set_argument(result, "cta_n", problem_space, operation_desc.tile_description.threadblock_shape.n());
  set_argument(result, "cta_k", problem_space, operation_desc.tile_description.threadblock_shape.k());

  profiler->initialize_workspace(options, report, device_context, operation, problem_space, problem);

  profiler->verify_cutlass(options, report, device_context, operation, problem_space, problem);

  profiler->profile(options, report, device_context, operation, problem_space, problem);

  report.append_results(profiler->results_);
  profiler->results_.clear();
}

///////////////////////////////////////////////////////////////////////////////////////////////////
