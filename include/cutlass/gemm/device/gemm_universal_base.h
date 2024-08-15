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
/*!
  \file
  \brief The universal GEMM accommodates streamk, batched strided, and batched array variants.
*/


#pragma once

#if defined(__CUDACC_RTC__)
#include <cuda/std/limits>
#else
#include <limits>
#endif

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/arch/arch.h"
#include "cutlass/device_kernel.h"
#include "cutlass/cuda_host_adapter.hpp"

#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/gemm_universal.h"

#include "cutlass/gemm/kernel/default_gemm_universal.h"
#include "cutlass/gemm/device/default_gemm_configuration.h"

#include "cutlass/trace.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace device {

/////////////////////////////////////////////////////////////////////////////////////////////////


template <typename GemmKernel_>
class GemmUniversalBase {
public:

  using GemmKernel = GemmKernel_;

  /// Boolean indicating whether the CudaHostAdapter is enabled
  static bool const kEnableCudaHostAdapter = CUTLASS_ENABLE_CUDA_HOST_ADAPTER;

  using ThreadblockShape = typename GemmKernel::Mma::Shape;

  using ElementA = typename GemmKernel::ElementA;
  using LayoutA = typename GemmKernel::LayoutA;
  using TensorRefA = TensorRef<ElementA const, LayoutA>;
  static ComplexTransform const kTransformA = GemmKernel::kTransformA;

  using ElementB = typename GemmKernel::ElementB;
  using LayoutB = typename GemmKernel::LayoutB;
  using TensorRefB = TensorRef<ElementB const, LayoutB>;
  static ComplexTransform const kTransformB = GemmKernel::kTransformB;

  using ElementC = typename GemmKernel::ElementC;
  using LayoutC = typename GemmKernel::LayoutC;
  using TensorRefC = TensorRef<ElementC const, LayoutC>;
  using TensorRefD = TensorRef<ElementC, LayoutC>;

  /// Numerical accumulation element type
  using ElementAccumulator = typename GemmKernel::Mma::ElementC;

  using EpilogueOutputOp = typename GemmKernel::EpilogueOutputOp;
  using ThreadblockSwizzle = typename GemmKernel::ThreadblockSwizzle;
  using Operator = typename GemmKernel::Operator;

  /// Argument structure
  using Arguments = typename GemmKernel::Arguments;


  /// Index of the GEMM Kernel within the CudaHostAdapter
  static int32_t const kGemmKernelIndex = 0;

  /// Kernel dynamic shared memory allocation requirement
  /// Update the kernel function's shared memory configuration for the current device
  static constexpr size_t kSharedStorageSize = sizeof(typename GemmKernel::SharedStorage);

  //
  // Device properties (uniform across all instances of the current thread)
  //

  // Device ordinal
  CUTLASS_THREAD_LOCAL static int device_ordinal_;

  /// Device SM count
  CUTLASS_THREAD_LOCAL static int device_sms_;

  /// Kernel SM occupancy (in thread blocks)
  CUTLASS_THREAD_LOCAL static int sm_occupancy_;

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

    device_ordinal_ = current_ordinal;
    return Status::kSuccess;
  }


  //
  // Instance data members
  //

  /// Kernel parameters
  typename GemmKernel::Params params_;


  /// Initialize params member
  Status init_params(Arguments const &args, CudaHostAdapter *cuda_adapter = nullptr)
  {
    init_device_props();
    params_ = typename GemmKernel::Params(args, device_sms_, sm_occupancy_);
    return Status::kSuccess;
  }

  //---------------------------------------------------------------------------------------------
  // Stateless API
  //---------------------------------------------------------------------------------------------

  /// Determines whether the GEMM can execute the given problem.
  static Status can_implement(Arguments const &args, CudaHostAdapter *cuda_adapter = nullptr)
  {
    return GemmKernel::can_implement(args);
  }

  /// Returns the workspace size (in bytes) needed for the problem
  /// geometry expressed by these arguments
  static size_t get_workspace_size(Arguments const &args, CudaHostAdapter *cuda_adapter = nullptr)
  {
    GemmUniversalBase base;
    base.init_params(args, cuda_adapter);
    return base.params_.get_workspace_size();
  }

  //---------------------------------------------------------------------------------------------
  // Stateful API
  //---------------------------------------------------------------------------------------------

  /// Initializes GEMM state from arguments and workspace memory
  Status initialize(
    Arguments const &args,
    void *workspace = nullptr,
    cudaStream_t stream = nullptr,
    CudaHostAdapter *cuda_adapter = nullptr)
  {
    init_device_props();
    params_ = typename GemmKernel::Params(args, device_sms_, sm_occupancy_);
    return params_.init_workspace(workspace, stream);
  }


  /// Lightweight update given a subset of arguments.
  Status update(Arguments const &args)
  {
    CUTLASS_TRACE_HOST("GemmUniversalBase()::update()");
    params_.update(args);
    return Status::kSuccess;
  }

  /// Runs the kernel using initialized state.
  Status run(cudaStream_t stream = nullptr, CudaHostAdapter *cuda_adapter = nullptr)
  {
    CUTLASS_TRACE_HOST("GemmUniversalBase::run()");

    // Configure grid and block dimensions
    dim3 block(GemmKernel::kThreadCount, 1, 1);
    dim3 grid = params_.get_grid_dims();

    // Launch kernel
    CUTLASS_TRACE_HOST("  "
      "grid: (" << grid << "), "
      "block: (" << block << "), "
      "SMEM: (" << kSharedStorageSize << ")");

    CUTLASS_ASSERT(cuda_adapter == nullptr);

    Kernel2<GemmKernel><<<grid, block, kSharedStorageSize, stream>>>(params_);

      // Query for errors
    cudaError_t result = cudaGetLastError();
    if (result != cudaSuccess) {
      CUTLASS_TRACE_HOST("  grid launch failed with error " << cudaGetErrorString(result));
      return Status::kErrorInternal;
    }

    return Status::kSuccess;
  }


  /// Runs the kernel using initialized state.
  Status operator()(cudaStream_t stream = nullptr, CudaHostAdapter *cuda_adapter = nullptr)
  {
    return run(stream, cuda_adapter);
  }


  /// Runs the kernel using initialized state.
  Status operator()(
    Arguments const &args, 
    void *workspace = nullptr, 
    cudaStream_t stream = nullptr,
    CudaHostAdapter *cuda_adapter = nullptr)
  {
    Status status = initialize(args, workspace, stream, cuda_adapter);

    if (status == Status::kSuccess) {
      status = run(stream, cuda_adapter);
    }

    return status;
  }
};


/////////////////////////////////////////////////////////////////////////////////////////////////
/// Static initializers
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Device ordinal
template <typename GemmKernel_>
CUTLASS_THREAD_LOCAL int GemmUniversalBase<GemmKernel_>::device_ordinal_ = -1;

/// Device SM count
template <typename GemmKernel_>
CUTLASS_THREAD_LOCAL int GemmUniversalBase<GemmKernel_>::device_sms_ = -1;

/// Kernel SM occupancy (in thread blocks)
template <typename GemmKernel_>
CUTLASS_THREAD_LOCAL int GemmUniversalBase<GemmKernel_>::sm_occupancy_ = -1;

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace device
} // namespace gemm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
