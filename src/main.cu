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
#include "cutlass/profiler/gpu_timer.h"
#include "cutlass/trace.h"
#include "cutlass/library/operation_table.h"
#include "cutlass/library/library.h"
#include "cutlass/profiler/device_context.h"
#include "cutlass/profiler/problem_space.h"

#include "cutlass/cutlass.h"
#include "library_internal.h"
#include "gemm_operation.h"
//#include "gemm_operation_3x.hpp"
#include "cutlass/arch/wmma.h"
#include "cutlass/numeric_types.h"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/default_gemm_universal.h"
using namespace cutlass;
using kernel =
  typename gemm::kernel::DefaultGemmUniversal<
    half_t, layout::ColumnMajor, ComplexTransform::kNone, 8,    // transposed B operand
    half_t, layout::RowMajor, ComplexTransform::kNone, 8,    // transposed A operand
    float, layout::RowMajor,
    float,
    arch::OpClassTensorOp,
    arch::Sm80,
    gemm::GemmShape<128, 128, 32>,
    gemm::GemmShape<64, 64, 32>,
    gemm::GemmShape<16, 8, 16>,
    
    epilogue::thread::LinearCombination<
      float,
      4,
      float,
      float
    >
,
    gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    5,
    arch::OpMultiplyAdd
>::GemmKernel;
// Profiler includes
int main(int argc, char const *arg[]) {
  CommandLine cmdline(argc, arg);
  profiler::Options options(cmdline);
  profiler::DeviceContext device_context;
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

  profiler::ProblemSpace problem_space(arguments_, options.cmdline);
  profiler::ProblemSpace::Iterator problem_it = problem_space.begin();
  profiler::ProblemSpace::Iterator problem_end = problem_space.end();
  profiler::ProblemSpace::Problem problem = problem_it.at();
  std::unique_ptr<library::GemmUniversalOperation<gemm::device::GemmUniversalAdapter<kernel> > > operation;
  
  operation = std::unique_ptr<library::GemmUniversalOperation<
      gemm::device::GemmUniversalAdapter<kernel>
  >>(new library::GemmUniversalOperation<
      gemm::device::GemmUniversalAdapter<kernel>
  >("kernel"));
  // initialize_all(operation); //in kernel.cu 
  device_context.free(); //??why
   profiler::DeviceAllocation *A{nullptr};
   profiler::DeviceAllocation *B{nullptr};
   profiler::DeviceAllocation *C{nullptr};
   profiler::DeviceAllocation *Computed{nullptr};
   library::GemmUniversalConfiguration configuration;
   library::GemmUniversalArguments arguments;
   std::vector<uint8_t> host_workspace;
   cutlass::profiler::DeviceAllocation device_workspace;
   std::vector<uint8_t> reduction_host_workspace;
   int m = 3456;  
   int n = 4096;
   int k = 4096;
   int split_k_slices{1};
  library::GemmDescription const &operation_desc =
    static_cast<library::GemmDescription const &>(operation->description()); //easy to use desc in gemmoperation

  int batch_count = 1;//problem_.batch_count = 1;
  int64_t bytes =
    int64_t(library::sizeof_bits(operation_desc.A.element) * m / 8) * k +
    int64_t(library::sizeof_bits(operation_desc.B.element) * n / 8) * k +
    int64_t(library::sizeof_bits(operation_desc.C.element) * m / 8) * n;
  bytes *= batch_count;
  configuration.mode = library::GemmUniversalMode::kGemm;
  configuration.problem_size.m() = int(m);
  configuration.problem_size.n() = int(n);
  configuration.problem_size.k() = int(k);
  configuration.lda = cutlass::profiler::DeviceAllocation::get_packed_layout( //problem_.lda;
    operation_desc.A.layout, {int(m), int(k)}).front();
  configuration.ldb = cutlass::profiler::DeviceAllocation::get_packed_layout( //problem_.ldb;
    operation_desc.B.layout, {int(k), int(n)}).front();
  configuration.ldc = cutlass::profiler::DeviceAllocation::get_packed_layout(//problem_.ldc;
    operation_desc.C.layout, {int(m), int(n)}).front();
  configuration.ldd = configuration.ldc;//problem_.ldc;
  configuration.batch_count = 1;// problem_.split_k_slices;
  arguments.A = nullptr;
  arguments.B = nullptr;
  arguments.C = nullptr;
  arguments.D = nullptr;
  std::vector<uint8_t>alpha; 
  std::vector<uint8_t>beta;
  cast_from_double(alpha, operation_desc.element_epilogue, 1);
  cast_from_double(beta, operation_desc.element_epilogue, 0);
  arguments.alpha = alpha.data(); 
  arguments.beta = beta.data();
  arguments.pointer_mode = library::ScalarPointerMode::kHost;
  arguments.raster_order = library::RasterOrder::kHeuristic;//problem_.raster_order;
  int problem_count =
    1 + int((3 * int64_t(options.device.properties.l2CacheSize)) / bytes);
  A = device_context.allocate_tensor(
    "A",
    operation_desc.A.element,
    operation_desc.A.layout,
    {int(m), int(k)},
    {int(configuration.lda)},
    batch_count * problem_count);

  B = device_context.allocate_tensor(
    "B",
    operation_desc.B.element,
    operation_desc.B.layout,
    {int(k), int(n)},
    {int(configuration.ldb)},
    batch_count * problem_count);

  C = device_context.allocate_tensor(
    "C",
    operation_desc.C.element,
    operation_desc.C.layout,
    {int(m), int(n)},
    {int(configuration.ldc)},
    batch_count * problem_count);

  Computed = device_context.allocate_tensor(
    "D",
    operation_desc.D.element,
    operation_desc.D.layout,
    {int(m), int(n)},
    {int(configuration.ldc)},
    batch_count * problem_count);


  arguments.problem_size = {int(m), int(n), int(k)};
  arguments.batch_count = batch_count;
  arguments.lda =configuration.lda;
  arguments.ldb =configuration.ldb;
  arguments.ldc =configuration.ldc;
  arguments.ldd =configuration.ldc;
  arguments.batch_stride_A = A->batch_stride();
  arguments.batch_stride_B = B->batch_stride();
  arguments.batch_stride_C = C->batch_stride();
  arguments.batch_stride_D = Computed->batch_stride();
  arguments.sm_count = options.device.properties.multiProcessorCount;

  uint64_t workspace_size = operation->get_host_workspace_size(&configuration);
  host_workspace.resize(workspace_size, 0);
  workspace_size = operation->get_device_workspace_size(&configuration,&arguments); //Segmentation fault
  device_workspace.reset(library::NumericTypeID::kU8, workspace_size);
  operation->initialize(
    &configuration,
    host_workspace.data(),
    device_workspace.data());
  arguments.A = A->data();
  arguments.B = B->data();
  arguments.C = C->data();
  arguments.D = Computed->data();
  arguments.pointer_mode = library::ScalarPointerMode::kHost;
  arguments.batch_stride_A = A->batch_stride();
  arguments.batch_stride_B = B->batch_stride();
  arguments.batch_stride_C = C->batch_stride();
  arguments.batch_stride_D = Computed->batch_stride();
  for (int iteration = 0; iteration < options.profiling.warmup_iterations; ++iteration) {
    int problem_idx = (iteration % problem_count) * batch_count;
    arguments.A = A->batch_data(problem_idx);
    arguments.B = B->batch_data(problem_idx);
    arguments.C = C->batch_data(problem_idx);
    arguments.D = Computed->batch_data(problem_idx);
    operation->run( //todo:move to main //Segmentation fault
      &arguments,
      host_workspace.data(),
      device_workspace.data());
  }
  cutlass::profiler::GpuTimer timer;
  timer.start();
  int Iterations = options.profiling.iterations;
  int iteration = 0;
  for (; iteration < Iterations; ++iteration) {
    int workspace_idx = options.profiling.warmup_iterations + iteration;
    int problem_idx = (workspace_idx % problem_count) * batch_count;
    arguments.A = A->batch_data(problem_idx);
    arguments.B = B->batch_data(problem_idx);
    arguments.C = C->batch_data(problem_idx);
    arguments.D = Computed->batch_data(problem_idx);
    operation->run(
      &arguments,
      host_workspace.data(),
      device_workspace.data());
  }
  timer.stop_and_wait();
  double runtime = timer.duration(iteration);
  std::cout
    << "=============================\n"
    << "       Arguments:";
  std::cout << " --gemm_kind=" << library::to_string(operation_desc.gemm_kind);
  std::cout << " --m=" << m; //todo:arguments.m put everything printed to arguments
  std::cout << " --n=" << n;
  std::cout << " --k=" << k;
  std::cout << " --A=" << library::to_string(operation_desc.A.element) << ":" << library::to_string(operation_desc.A.layout);
  std::cout << " --B=" << library::to_string(operation_desc.B.element) << ":" << library::to_string(operation_desc.B.layout);
  std::cout << " --C=" << library::to_string(operation_desc.C.element) << ":" << library::to_string(operation_desc.C.layout);
  std::cout << " --D=" << library::to_string(operation_desc.D.element) << ":" << library::to_string(operation_desc.D.layout);
  std::cout << "  \\\n                 ";
  std::cout << " --alpha=" << library::lexical_cast(alpha, operation_desc.element_epilogue);
  std::cout << " --beta=" << library::lexical_cast(beta, operation_desc.element_epilogue);
  std::cout << " --split_k_slices=" << split_k_slices;
  std::cout << " --batch_count=" << batch_count;
  std::cout << " --raster_order=" << library::to_string(arguments.raster_order);
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
   double flops = (double)((double)m*n*k+m*n)*2*batch_count;
  std::cout
    << "\n"
    << "         Runtime: " << runtime << "  ms\n"
    << "          Memory: " << (double)bytes / double(1 << 30) / runtime * 1000.0 << " GiB/s\n"
    << "            Math: " << flops / runtime / 1.0e6 << " GFLOP/s\n";
}
