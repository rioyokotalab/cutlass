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
#include "cutlass/profiler/gemm_operation_profiler.h"
#include "cutlass/profiler/gpu_timer.h"
#include "cutlass/trace.h"

using namespace cutlass;

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

  profiler->problem_.mode = library::GemmUniversalMode::kGemm;
  profiler->problem_.m = 3456;
  profiler->problem_.n = 4096;
  profiler->problem_.k = 4096;
  profiler->problem_.split_k_mode = library::SplitKMode::kSerial;
  profiler->problem_.mode = library::GemmUniversalMode::kGemm;
  profiler->problem_.split_k_slices = 1;
  profiler->problem_.batch_count = 1;
  profiler->problem_.raster_order = library::RasterOrder::kHeuristic;
  cast_from_double(profiler->problem_.alpha, operation_desc.element_epilogue, 1);
  cast_from_double(profiler->problem_.beta, operation_desc.element_epilogue, 0);
  profiler->problem_.lda = DeviceAllocation::get_packed_layout(
    operation_desc.A.layout, {int(profiler->problem_.m), int(profiler->problem_.k)}).front();
  profiler->problem_.ldb = DeviceAllocation::get_packed_layout(
    operation_desc.B.layout, {int(profiler->problem_.k), int(profiler->problem_.n)}).front();
  profiler->problem_.ldc = DeviceAllocation::get_packed_layout(
    operation_desc.C.layout, {int(profiler->problem_.m), int(profiler->problem_.n)}).front();

  profiler->gemm_workspace_.configuration.mode = profiler->problem_.mode;
  profiler->gemm_workspace_.configuration.problem_size.m() = int(profiler->problem_.m);
  profiler->gemm_workspace_.configuration.problem_size.n() = int(profiler->problem_.n);
  profiler->gemm_workspace_.configuration.problem_size.k() = int(profiler->problem_.k);
  profiler->gemm_workspace_.configuration.lda = profiler->problem_.lda;
  profiler->gemm_workspace_.configuration.ldb = profiler->problem_.ldb;
  profiler->gemm_workspace_.configuration.ldc = profiler->problem_.ldc;
  profiler->gemm_workspace_.configuration.ldd = profiler->problem_.ldc;
  profiler->gemm_workspace_.configuration.batch_count = profiler->problem_.split_k_slices;
  profiler->gemm_workspace_.arguments.A = nullptr;
  profiler->gemm_workspace_.arguments.B = nullptr;
  profiler->gemm_workspace_.arguments.C = nullptr;
  profiler->gemm_workspace_.arguments.D = nullptr;
  profiler->gemm_workspace_.arguments.alpha = profiler->problem_.alpha.data();
  profiler->gemm_workspace_.arguments.beta = profiler->problem_.beta.data();
  profiler->gemm_workspace_.arguments.pointer_mode = library::ScalarPointerMode::kHost;
  profiler->gemm_workspace_.arguments.raster_order = profiler->problem_.raster_order;

  PerformanceResult &result = profiler->model_result_;
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

  profiler->initialize_workspace(options, report, device_context, operation, problem_space, problem);

  profiler->profile(options, report, device_context, operation, problem_space, problem);

  report.append_results(profiler->results_);
  profiler->results_.clear();
}

///////////////////////////////////////////////////////////////////////////////////////////////////
