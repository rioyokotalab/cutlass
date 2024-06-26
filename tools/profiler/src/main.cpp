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

  profiler->initialize_configuration(device_context, operation, problem_space, problem);

  profiler->initialize_workspace(options, device_context, operation, problem_space, problem);

  double runtime = profiler->profile(options, device_context, operation, problem_space, problem);

  library::GemmDescription const &operation_desc =
    static_cast<library::GemmDescription const &>(operation->description());

  std::cout
    << "=============================\n"
    << "       Arguments:";
  std::cout << " --gemm_kind=" << library::to_string(operation_desc.gemm_kind);
  std::cout << " --m=" << profiler->problem_.m;
  std::cout << " --n=" << profiler->problem_.n;
  std::cout << " --k=" << profiler->problem_.k;
  std::cout << " --A=" << library::to_string(operation_desc.A.element) << ":" << library::to_string(operation_desc.A.layout);
  std::cout << " --B=" << library::to_string(operation_desc.B.element) << ":" << library::to_string(operation_desc.B.layout);
  std::cout << " --C=" << library::to_string(operation_desc.C.element) << ":" << library::to_string(operation_desc.C.layout);
  std::cout << " --D=" << library::to_string(operation_desc.D.element) << ":" << library::to_string(operation_desc.D.layout);
  std::cout << "  \\\n                 ";
  std::cout << " --alpha=" << library::lexical_cast(profiler->problem_.alpha, operation_desc.element_epilogue);
  std::cout << " --beta=" << library::lexical_cast(profiler->problem_.beta, operation_desc.element_epilogue);
  std::cout << " --split_k_mode=" << library::to_string(profiler->problem_.split_k_mode);
  std::cout << " --split_k_slices=" << profiler->problem_.split_k_slices;
  std::cout << " --batch_count=" << profiler->problem_.batch_count;
  std::cout << " --raster_order=" << library::to_string(profiler->problem_.raster_order);
  std::cout << "  \\\n                 ";
  std::cout << " --op_class=" << library::to_string(operation_desc.tile_description.math_instruction.opcode_class);
  std::cout << " --accum=" << library::to_string(operation_desc.tile_description.math_instruction.element_accumulator);
  std::cout << " --cta_m=" << operation_desc.tile_description.threadblock_shape.m();
  std::cout << " --cta_n=" << operation_desc.tile_description.threadblock_shape.n();
  std::cout << " --cta_k=" << operation_desc.tile_description.threadblock_shape.k();
  std::cout << " --cluster_m=" << operation_desc.tile_description.cluster_shape.m();
  std::cout << " --cluster_n=" << operation_desc.tile_description.cluster_shape.n();
  std::cout << " --cluster_k=" << operation_desc.tile_description.cluster_shape.k();
  std::cout << "  \\\n                 ";
  std::cout << " --stages=" << operation_desc.tile_description.threadblock_stages;
  std::cout << " --warps_m=" << operation_desc.tile_description.warp_count.m();
  std::cout << " --warps_n=" << operation_desc.tile_description.warp_count.n();
  std::cout << " --warps_k=" << operation_desc.tile_description.warp_count.k();
  std::cout << " --inst_m=" << operation_desc.tile_description.math_instruction.instruction_shape.m();
  std::cout << " --inst_n=" << operation_desc.tile_description.math_instruction.instruction_shape.n();
  std::cout << " --inst_k=" << operation_desc.tile_description.math_instruction.instruction_shape.k();
  std::cout << " --min_cc=" << operation_desc.tile_description.minimum_compute_capability;
  std::cout << " --max_cc=" << operation_desc.tile_description.maximum_compute_capability;
  std::cout << "  \\\n                 ";
  double bytes = profiler->bytes(operation_desc, profiler->problem_);
  double flops = profiler->flops(profiler->problem_);
  std::cout
    << "\n"
    << "         Runtime: " << runtime << "  ms\n"
    << "          Memory: " << bytes / double(1 << 30) / runtime * 1000.0 << " GiB/s\n"
    << "            Math: " << flops / runtime / 1.0e6 << " GFLOP/s\n";
}
