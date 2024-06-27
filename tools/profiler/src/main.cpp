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
  auto profiler = new profiler::GemmOperationProfiler();
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
  profiler::ProblemSpace::Iterator problem_it = problem_space.begin();
  profiler::ProblemSpace::Iterator problem_end = problem_space.end();
  profiler::ProblemSpace::Problem problem = problem_it.at();
  auto operation_ptr = manifest.begin();
  library::Operation const *operation = operation_ptr->get();
  device_context.free();
  std::string operation_name(operation->description().name);

  profiler->initialize_configuration(options, device_context, operation, problem_space, problem);

  profiler->initialize_workspace(options, device_context, operation, problem_space, problem);

  profiler->profile(options, device_context, operation, problem_space, problem);

  profiler::PerformanceResult result = profiler->results_.front();
  library::GemmDescription const &operation_desc =
    static_cast<library::GemmDescription const &>(operation->description());

  result.provider = library::Provider::kCUTLASS;
  result.disposition = profiler::Disposition::kNotRun;
  result.status = Status::kSuccess;
  result.operation_name = operation_desc.name;
  result.arguments.resize(problem_space.rank());
  profiler->set_argument(result, "gemm_kind", problem_space, library::to_string(operation_desc.gemm_kind));
  profiler->set_argument(result, "A", problem_space,
    std::string(library::to_string(operation_desc.A.element)) + ":" + library::to_string(operation_desc.A.layout));
  profiler->set_argument(result, "B", problem_space,
    std::string(library::to_string(operation_desc.B.element)) + ":" + library::to_string(operation_desc.B.layout));
  profiler->set_argument(result, "C", problem_space,
    std::string(library::to_string(operation_desc.C.element)) + ":" + library::to_string(operation_desc.C.layout));
  profiler->set_argument(result, "D", problem_space,
    std::string(library::to_string(operation_desc.D.element)) + ":" + library::to_string(operation_desc.D.layout));
  profiler->set_argument(result, "m", problem_space, profiler->problem_.m);
  profiler->set_argument(result, "n", problem_space, profiler->problem_.n);
  profiler->set_argument(result, "k", problem_space, profiler->problem_.k);
  profiler->set_argument(result, "split_k_mode", problem_space, library::to_string(profiler->problem_.split_k_mode));
  profiler->set_argument(result, "split_k_slices", problem_space, profiler->problem_.split_k_slices);
  profiler->set_argument(result, "batch_count", problem_space, profiler->problem_.batch_count);
  profiler->set_argument(result, "raster_order", problem_space, library::to_string(profiler->problem_.raster_order));
  profiler->set_argument(result, "alpha", problem_space,
    library::lexical_cast(profiler->problem_.alpha, operation_desc.element_epilogue));
  profiler->set_argument(result, "beta", problem_space,
    library::lexical_cast(profiler->problem_.beta, operation_desc.element_epilogue));
  profiler->set_argument(result, "op_class", problem_space, library::to_string(operation_desc.tile_description.math_instruction.opcode_class));
  profiler->set_argument(result, "accum", problem_space, library::to_string(operation_desc.tile_description.math_instruction.element_accumulator));
  profiler->set_argument(result, "cta_m", problem_space, operation_desc.tile_description.threadblock_shape.m());
  profiler->set_argument(result, "cta_n", problem_space, operation_desc.tile_description.threadblock_shape.n());
  profiler->set_argument(result, "cta_k", problem_space, operation_desc.tile_description.threadblock_shape.k());
  profiler->set_argument(result, "cluster_m", problem_space, operation_desc.tile_description.cluster_shape.m());
  profiler->set_argument(result, "cluster_n", problem_space, operation_desc.tile_description.cluster_shape.n());
  profiler->set_argument(result, "cluster_k", problem_space, operation_desc.tile_description.cluster_shape.k());
  profiler->set_argument(result, "stages", problem_space, operation_desc.tile_description.threadblock_stages);
  profiler->set_argument(result, "warps_m", problem_space, operation_desc.tile_description.warp_count.m());
  profiler->set_argument(result, "warps_n", problem_space, operation_desc.tile_description.warp_count.n());
  profiler->set_argument(result, "warps_k", problem_space, operation_desc.tile_description.warp_count.k());
  profiler->set_argument(result, "inst_m", problem_space, operation_desc.tile_description.math_instruction.instruction_shape.m());
  profiler->set_argument(result, "inst_n", problem_space, operation_desc.tile_description.math_instruction.instruction_shape.n());
  profiler->set_argument(result, "inst_k", problem_space, operation_desc.tile_description.math_instruction.instruction_shape.k());
  profiler->set_argument(result, "min_cc", problem_space, operation_desc.tile_description.minimum_compute_capability);
  profiler->set_argument(result, "max_cc", problem_space, operation_desc.tile_description.maximum_compute_capability);
  result.bytes = profiler->bytes(operation_desc);
  result.flops = profiler->flops();
  result.runtime = 0;
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

///////////////////////////////////////////////////////////////////////////////////////////////////
