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

#pragma once

// common
#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/detail/layout.hpp"
#include "cutlass/detail/mma.hpp"
#include "cutlass/cuda_host_adapter.hpp"

#if !defined(__CUDACC_RTC__)
#include "cutlass/cluster_launch.hpp"
#include "cutlass/trace.h"
#endif // !defined(__CUDACC_RTC__)

#include "cutlass/gemm/kernel/gemm_transpose_operands.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/epilogue/threadblock/epilogue_with_visitor_callbacks.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::device {

////////////////////////////////////////////////////////////////////////////////

/*!
  GemmUniversalAdapter is a stateful, reusable GEMM handle built around a kernel
  of type cutlass::gemm::kernel::Gemm or cutlass::gemm::kernel::GemmUniversal.

  It manages the lifetime of the underlying `kernel::Params` struct, and exposes APIs
  to create it from the host facing arguments. For power users, new static methods
  are exposed in 3.x APIs that bypass the stateful methods or args->params lowering.

  It supports kernel types that implement both the 2.x and 3.0 APIs,
  however, this is done by specializing the implementation of GemmUniversalAdapter
  on the two kernel API types, and thus, GemmUniversalAdapter's behaviour might
  differ between the two specializations.
*/

////////////////////////////////////////////////////////////////////////////////
////////////////////////////// CUTLASS 2.x API /////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
template <class GemmKernel_>
class GemmUniversalAdapter
{
public:

  using GemmKernel = GemmKernel_;

  using ThreadblockShape = typename GemmKernel::Mma::Shape;
  using WarpShape = typename GemmKernel::WarpShape;
  using InstructionShape = typename GemmKernel::InstructionShape;

  // warp-level, arch-level (instruction), math operator
  using WarpMmaOperator = typename GemmKernel::Mma::Policy::Operator;
  using ArchMmaOperator = typename WarpMmaOperator::ArchMmaOperator;
  using MathOperator = typename WarpMmaOperator::MathOperator;

  // Operator class and arch tag extract bottom-up
  // set it for top-level gemm device-level template
  using OperatorClass = typename WarpMmaOperator::OperatorClass;
  using ArchTag = typename WarpMmaOperator::ArchTag;

  // Type, layout, and complex transform deliberately exchanged with B
  using MapArguments = kernel::detail::MapArguments<
    typename GemmKernel::ElementA,
    typename GemmKernel::LayoutA,
    GemmKernel::kTransformA,
    GemmKernel::kAlignmentA,
    typename GemmKernel::ElementB,
    typename GemmKernel::LayoutB,
    GemmKernel::kTransformB,
    GemmKernel::kAlignmentB,
    typename GemmKernel::LayoutC,
    true
  >;

  using ElementA = typename MapArguments::ElementA;
  using LayoutA = typename MapArguments::LayoutA;
  static ComplexTransform const kTransformA = MapArguments::kTransformA;
  static int const kAlignmentA = MapArguments::kAlignmentA;

  using ElementB = typename MapArguments::ElementB;
  using LayoutB = typename MapArguments::LayoutB;
  static ComplexTransform const kTransformB = MapArguments::kTransformB;
  static int const kAlignmentB = MapArguments::kAlignmentB;

  using ElementC = typename GemmKernel::ElementC;
  using LayoutC = typename MapArguments::LayoutC;
  static int const kAlignmentC = GemmKernel::kAlignmentC;

  // C and D same type for 2.x kernel
  using ElementD = ElementC;
  using LayoutD = LayoutC;
  static int const kStages = GemmKernel::Mma::kStages;

  using EpilogueOutputOp = typename GemmKernel::EpilogueOutputOp;
  using ElementAccumulator = typename EpilogueOutputOp::ElementAccumulator;
  using ThreadblockSwizzle = typename GemmKernel::ThreadblockSwizzle;
  using Arguments = typename GemmKernel::Arguments;

  typename GemmKernel::Params params_;

  static constexpr size_t kSharedStorageSize = sizeof(typename GemmKernel::SharedStorage);

  /// Device SM count
  CUTLASS_THREAD_LOCAL static int device_sms_;

  /// Kernel SM occupancy (in thread blocks)
  CUTLASS_THREAD_LOCAL static int sm_occupancy_;

  /// Constructs the GEMM.
  GemmUniversalAdapter() { }

  /// Initialize static thread-local members for the thread's current device,
  /// if necessary.
  static Status init_device_props()
  {
    int current_ordinal;
    cudaGetDevice(&current_ordinal);
    cudaDeviceGetAttribute (&device_sms_, cudaDevAttrMultiProcessorCount, current_ordinal);

    cudaFuncSetAttribute(
      Kernel2<GemmKernel>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      kSharedStorageSize);

    cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
      &sm_occupancy_,
      Kernel2<GemmKernel>,
      GemmKernel::kThreadCount,
      kSharedStorageSize,
      cudaOccupancyDisableCachingOverride);

    return Status::kSuccess;
  }

  /// Helper to construct a transposed equivalent for the underying GEMM operator
  static Arguments to_underlying_arguments(Arguments const &args_) {
    Arguments args(args_);
    std::swap(args.problem_size.m(), args.problem_size.n());
    std::swap(args.ptr_A, args.ptr_B);
    std::swap(args.lda, args.ldb);
    std::swap(args.stride_a, args.stride_b);
    std::swap(args.batch_stride_A, args.batch_stride_B);
    std::swap(args.ptr_gather_A_indices, args.ptr_gather_B_indices);
    return args;
  }

  /// Determines whether the GEMM can execute the given problem.
  static Status can_implement(Arguments const &args, CudaHostAdapter *cuda_adapter = nullptr) {
    return GemmKernel::can_implement(to_underlying_arguments(args));
  }

  /// Gets the workspace size
  static size_t get_workspace_size(Arguments const &args, CudaHostAdapter *cuda_adapter = nullptr) {
    GemmUniversalAdapter base;
    base.init_device_props();
    typename GemmKernel::Params params = typename GemmKernel::Params(to_underlying_arguments(args), base.device_sms_, base.sm_occupancy_);
    return params.get_workspace_size();
  }

  /// Initializes GEMM state from arguments.
  Status initialize(
    Arguments const &args,
    void *workspace = nullptr,
    cudaStream_t stream = nullptr,
    CudaHostAdapter *cuda_adapter = nullptr
  ) {

    init_device_props();
    params_ = typename GemmKernel::Params();
    params_.problem_size = cutlass::gemm::GemmCoord(3456,4096,4096);
    params_.mode = args.mode;
    params_.batch_count = args.batch_count;
    params_.init_grid_tiled_shape();
    params_.grid_tiled_shape = cutlass::gemm::GemmCoord(32,27,1);
    typename GemmKernel::Mma::IteratorA::Params params_A(make_Coord(long(4096)));
    typename GemmKernel::Mma::IteratorB::Params params_B(make_Coord(long(3456)));
    typename GemmKernel::Epilogue::OutputTileIterator::Params params_C(make_Coord(long(3456)));
    typename GemmKernel::Epilogue::OutputTileIterator::Params params_D(make_Coord(long(3456)));
    params_.params_A = params_A;
    params_.params_B = params_B;
    params_.params_C = params_C;
    params_.params_D = params_D;
    params_.ptr_A = const_cast<void *>(args.ptr_A);
    params_.ptr_B = const_cast<void *>(args.ptr_B);
    params_.ptr_C = const_cast<void *>(args.ptr_C);
    params_.ptr_D = args.ptr_D;
    params_.batch_stride_A = args.batch_stride_A;
    params_.batch_stride_B = args.batch_stride_B;
    params_.batch_stride_C = args.batch_stride_C;
    params_.batch_stride_D = args.batch_stride_D;
    params_.ptr_gather_A_indices = const_cast<int *>(args.ptr_gather_A_indices);
    params_.ptr_gather_B_indices = const_cast<int *>(args.ptr_gather_B_indices);
    params_.ptr_scatter_D_indices = const_cast<int *>(args.ptr_scatter_D_indices);
    params_.semaphore = static_cast<int *>(workspace);
    return Status::kSuccess;
  }

  /// Lightweight update given a subset of arguments.
  Status update(Arguments const &args) {
    params_.update(to_underlying_arguments(args));
    return Status::kSuccess;
  }

  /// Runs the kernel using initialized state.
  Status run( //run here
    cudaStream_t stream = nullptr,
    CudaHostAdapter *cuda_adapter = nullptr) {

    dim3 block(128, 1, 1);
    dim3 grid(256, 4, 1);
    Kernel2<GemmKernel><<<grid, block, 81920, stream>>>(params_);

    cudaError_t result = cudaGetLastError();
    if (result != cudaSuccess) {
      CUTLASS_TRACE_HOST("  grid launch failed with error " << cudaGetErrorString(result));
      return Status::kErrorInternal;
    }
    return Status::kSuccess;
  }

};

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Static initializers
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Device SM count
template <typename GemmKernel_>
CUTLASS_THREAD_LOCAL int GemmUniversalAdapter<GemmKernel_>::device_sms_ = -1;

/// Kernel SM occupancy (in thread blocks)
template <typename GemmKernel_>
CUTLASS_THREAD_LOCAL int GemmUniversalAdapter<GemmKernel_>::sm_occupancy_ = -1;

////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::device

////////////////////////////////////////////////////////////////////////////////
