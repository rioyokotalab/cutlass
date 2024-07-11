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

   // library::GemmUniversalArguments arguments;

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


  cutlass::profiler::ArgumentDescriptionVector  arguments_(
    {
      {profiler::ArgumentTypeID::kEnumerated, {"gemm_kind"}, "Variant of GEMM (universal, gemm, planar_complex, planar_complex_array)"},
      {profiler::ArgumentTypeID::kInteger, {"m", "problem-size::m"}, "M dimension of the GEMM problem space"},
      {profiler::ArgumentTypeID::kInteger, {"n", "problem-size::n"}, "N dimension of the GEMM problem space"},
      {profiler::ArgumentTypeID::kInteger, {"k", "problem-size::k"}, "K dimension of the GEMM problem space"},
      {profiler::ArgumentTypeID::kTensor, {"A"}, "Tensor storing the A operand"},
      {profiler::ArgumentTypeID::kTensor, {"B"}, "Tensor storing the B operand"},
      {profiler::ArgumentTypeID::kTensor, {"C"}, "Tensor storing the C operand"},
      {profiler::ArgumentTypeID::kTensor, {"D"}, "Tensor storing the D output"},
      {profiler::ArgumentTypeID::kScalar, {"alpha", "epilogue::alpha"}, "Epilogue scalar alpha"},
      {profiler::ArgumentTypeID::kScalar, {"beta", "epilogue::beta"}, "Epilogue scalar beta"},
      {profiler::ArgumentTypeID::kEnumerated, {"split_k_mode", "split-k-mode"}, "Variant of split K mode(serial, parallel)"},
      {profiler::ArgumentTypeID::kInteger, {"split_k_slices", "split-k-slices"}, "Number of partitions of K dimension"},
      {profiler::ArgumentTypeID::kInteger, {"batch_count", "batch-count"}, "Number of GEMMs computed in one batch"},
      {profiler::ArgumentTypeID::kEnumerated, {"raster_order", "raster-order"}, "Raster order (heuristic, along_n, along_m)"},
    });

  arguments_.insert(arguments_.end(), tile_description_arguments.begin(), tile_description_arguments.end());

   // arguments.A = nullptr;
   // arguments.B = nullptr;
   // arguments.C = nullptr;
   // arguments.D = nullptr;
   // arguments.alpha = profiler->problem_.alpha.data();
   // arguments.beta = profiler->problem_.beta.data();
   // arguments.pointer_mode = library::ScalarPointerMode::kHost;
   // arguments.raster_order = profiler->problem_.raster_order;

  const library::Manifest &manifest = library::Singleton::get().manifest;
  // profiler::ProblemSpace problem_space(profiler->arguments_, options.cmdline);
  profiler::ProblemSpace problem_space(arguments_, options.cmdline);
  profiler::ProblemSpace::Iterator problem_it = problem_space.begin();
  profiler::ProblemSpace::Iterator problem_end = problem_space.end();
  profiler::ProblemSpace::Problem problem = problem_it.at();
  auto operation_ptr = manifest.begin();
  library::Operation const *operation = operation_ptr->get();
  device_context.free(); //??why
  std::string operation_name(operation->description().name);
///---------------------------------------------------------------------//
   profiler::DeviceAllocation *A{nullptr};
   profiler::DeviceAllocation *B{nullptr};
   profiler::DeviceAllocation *C{nullptr};
   profiler::DeviceAllocation *Computed{nullptr};
   profiler::DeviceAllocation *Reference{nullptr};
  //int problem_count{1}; //maynot? check it
   library::GemmUniversalConfiguration configuration;
   library::GemmUniversalArguments arguments;
   std::vector<uint8_t> host_workspace;
   cutlass::profiler::DeviceAllocation device_workspace;
   library::ReductionConfiguration reduction_configuration;
   library::ReductionArguments reduction_arguments;
   std::vector<uint8_t> reduction_host_workspace;
//initial
   int m = 3456;  
   int n = 4096;
   int k = 4096;
  library::GemmDescription const &operation_desc =
    static_cast<library::GemmDescription const &>(operation->description());
  double bytes = profiler->bytes(operation_desc, profiler->problem_);
  configuration.mode = library::GemmUniversalMode::kGemm;
  configuration.problem_size.m() = int(m);
  configuration.problem_size.n() = int(n);
  configuration.problem_size.k() = int(k);
  configuration.lda = cutlass::profiler::DeviceAllocation::get_packed_layout( //problem_.lda;
    operation_desc.A.layout, {int(m), int(k)}).front();
  //gemm_workspace_.configuration.lda 
  configuration.ldb = cutlass::profiler::DeviceAllocation::get_packed_layout( //problem_.ldb;
    operation_desc.B.layout, {int(k), int(n)}).front();
  configuration.ldc = cutlass::profiler::DeviceAllocation::get_packed_layout(//problem_.ldc;
    operation_desc.C.layout, {int(m), int(n)}).front();
//
  configuration.ldd = configuration.ldc;//problem_.ldc;
  configuration.batch_count = 1;// problem_.split_k_slices;
  arguments.A = nullptr;
  arguments.B = nullptr;
  arguments.C = nullptr;
  arguments.D = nullptr;
  //arguments.alpha = 1;//problem_.alpha.data();
  arguments.alpha = reinterpret_cast<void*>(1);
  arguments.beta = reinterpret_cast<void*>(0);
  //= reinterpret_cast<void*>(1)
  //arguments.beta = 0;//problem_.beta.data();
  // cast_from_double(arguments.alpha, operation_desc.element_epilogue, 1);
  // cast_from_double(arguments.beta, operation_desc.element_epilogue, 0);
  arguments.pointer_mode = library::ScalarPointerMode::kHost;
  arguments.raster_order = library::RasterOrder::kHeuristic;//problem_.raster_order;
//
  int problem_count =
    1 + int((3 * int64_t(options.device.properties.l2CacheSize)) / bytes);
  int batch_count = 1;//problem_.batch_count = 1;
  // A = device_context.allocate_tensor(
  //   "A",
  //   operation_desc.A.element,
  //   operation_desc.A.layout,
  //   {int(m), int(k)},
  //   //{int(problem_.lda)},
  //   {int(configuration.lda)},
  //   batch_count * problem_count);
//
//   B = device_context.allocate_tensor(
//     "B",
//     operation_desc.B.element,
//     operation_desc.B.layout,
//     {int(k), int(n)},
//     {int(configuration.ldb)},
//     batch_count * problem_count);
//
//   C = device_context.allocate_tensor(
//     "C",
//     operation_desc.C.element,
//     operation_desc.C.layout,
//     {int(m), int(n)},
//     {int(configuration.ldc)},
//     batch_count * problem_count);
//
//   Computed = device_context.allocate_tensor(
//     "D",
//     operation_desc.D.element,
//     operation_desc.D.layout,
//     {int(m), int(n)},
//     {int(configuration.ldc)},
//     batch_count * problem_count);
//
//   Reference = device_context.allocate_tensor(
//     "Reference",
//     operation_desc.D.element,
//     operation_desc.D.layout,
//     {int(m), int(n)},
//     {int(configuration.ldc)},
//     batch_count * problem_count);
//
//   arguments.problem_size = {int(m), int(n), int(k)};
//   arguments.batch_count = batch_count;
//   arguments.lda =configuration.lda;
//   arguments.ldb =configuration.ldb;
//   arguments.ldc =configuration.ldc;
//   arguments.ldd =configuration.ldc;
//   arguments.batch_stride_A = A->batch_stride();
//   arguments.batch_stride_B = B->batch_stride();
//   arguments.batch_stride_C = C->batch_stride();
//   arguments.batch_stride_D = Computed->batch_stride();
//   arguments.sm_count = options.device.properties.multiProcessorCount;
//
//   uint64_t workspace_size = operation->get_host_workspace_size(&configuration);
//   host_workspace.resize(workspace_size, 0);
//   workspace_size = operation->get_device_workspace_size(&configuration,
// 							&arguments);
//   device_workspace.reset(library::NumericTypeID::kU8, workspace_size);
//   operation->initialize(
//     &configuration,
//     host_workspace.data(),
//     device_workspace.data());
  //---------------------------------------------------------------------------//
  profiler->initialize_configuration(device_context, operation, problem_space, problem);

  profiler->initialize_workspace(options, device_context, operation, problem_space, problem);
  //options, problem_space, problem
  double runtime = profiler->profile(options, device_context, operation, problem_space, problem); //todo:remove these things ,unused

  // library::GemmDescription const &operation_desc =
  //   static_cast<library::GemmDescription const &>(operation->description());

  std::cout
    << "=============================\n"
    << "       Arguments:";
  std::cout << " --gemm_kind=" << library::to_string(operation_desc.gemm_kind);
  std::cout << " --m=" << profiler->problem_.m; //todo:arguments.m put everything printed to arguments
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
//  double bytes = profiler->bytes(operation_desc, profiler->problem_);
  double flops = profiler->flops(profiler->problem_);
  std::cout
    << "\n"
    << "         Runtime: " << runtime << "  ms\n"
    << "          Memory: " << bytes / double(1 << 30) / runtime * 1000.0 << " GiB/s\n"
    << "            Math: " << flops / runtime / 1.0e6 << " GFLOP/s\n";
}
