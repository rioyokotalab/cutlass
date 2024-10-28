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
   \brief Defines operations for all GEMM operation kinds in CUTLASS Library.
*/

#pragma once
#include "cutlass/cutlass.h"

#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_batched.h"
#include "cutlass/gemm/device/gemm_array.h"
#include "cutlass/gemm/kernel/default_gemm_universal.h"

#include "cutlass/library/library.h"
#include "library_internal.h"

// common
#include "cutlass/device_kernel.h"
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

///////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace library {


///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Operator_>
class GemmUniversalOperation {
public:
  // assuming all tensors use same type for StrideIndex 

  using Operator = Operator_;
  /*
  using ElementA = typename Operator::ElementA;
  using LayoutA = typename Operator::LayoutA;
  using ElementB = typename Operator::ElementB;
  using LayoutB = typename Operator::LayoutB;
  using ElementC = typename Operator::ElementC;
  using LayoutC = typename Operator::LayoutC;
  using ElementD = ElementC;
  using LayoutD = LayoutC;
  */
  using StrideIndex = typename Operator::LayoutA::Index;
  //using ElementAccumulator = typename Operator::ElementAccumulator;
  using ElementCompute = typename Operator::EpilogueOutputOp::ElementCompute;

  using OperatorArguments = typename Operator::Arguments;
  //-------------------------------------------Adapter-------------------------------//
  //using GemmKernel = GemmKernel_;
  using GemmKernel = typename Operator_::GemmKernel;

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
  //using MapArguments = kernel::detail::MapArguments<
  using MapArguments = cutlass::gemm::kernel::detail::MapArguments<
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
  //----------------------------------------------Adapter finish---------------------------//

protected:

  GemmDescription description_;

public:

  /// Constructor

  GemmUniversalOperation(char const *name = "unknown_gemm") {

    description_.name = name;
    description_.provider = Provider::kCUTLASS;
    description_.kind = OperationKind::kGemm;
    //description_.gemm_kind = GemmKind::kGemm; //in operation base
    description_.gemm_kind = GemmKind::kUniversal;

    description_.tile_description.threadblock_shape = make_Coord(
      Operator::ThreadblockShape::kM,
      Operator::ThreadblockShape::kN,
      Operator::ThreadblockShape::kK);

    description_.tile_description.threadblock_stages = Operator::kStages;

    description_.tile_description.warp_count = make_Coord(
      Operator::GemmKernel::WarpCount::kM,
      Operator::GemmKernel::WarpCount::kN,
      Operator::GemmKernel::WarpCount::kK);
    
    description_.tile_description.math_instruction.instruction_shape = make_Coord(
      Operator::InstructionShape::kM,
      Operator::InstructionShape::kN,
      Operator::InstructionShape::kK);

    description_.tile_description.math_instruction.element_accumulator = 
      NumericTypeMap<ElementAccumulator>::kId;

    description_.tile_description.math_instruction.opcode_class = 
      OpcodeClassMap<typename Operator::OperatorClass>::kId;

    description_.tile_description.math_instruction.math_operation =
      MathOperationMap<typename Operator::MathOperator>::kId;

    description_.tile_description.minimum_compute_capability = 
      ArchMap<typename Operator::ArchTag, typename Operator::OperatorClass>::kMin;

    description_.tile_description.maximum_compute_capability = 
      ArchMap<typename Operator::ArchTag, typename Operator::OperatorClass>::kMax;
    
    description_.A = make_TensorDescription<ElementA, LayoutA>(Operator::kAlignmentA);
    description_.B = make_TensorDescription<ElementB, LayoutB>(Operator::kAlignmentB);
    description_.C = make_TensorDescription<ElementC, LayoutC>(Operator::kAlignmentC);
    description_.D = make_TensorDescription<ElementD, LayoutD>(Operator::kAlignmentC);
    description_.element_epilogue = NumericTypeMap<ElementCompute>::kId;

    description_.split_k_mode = SplitKMode::kNone;
    description_.transform_A = ComplexTransformMap<Operator::kTransformA>::kId;
    description_.transform_B = ComplexTransformMap<Operator::kTransformB>::kId;
  }
  
  /// Returns the description of the GEMM operation
  virtual OperationDescription const & description() const {
    return description_;
  }
protected:

  /// Constructs the arguments structure given the configuration and arguments
  static Status construct_arguments_(
    OperatorArguments &operator_args,
    GemmUniversalConfiguration const *configuration) {

    operator_args.mode = configuration->mode;

    operator_args.problem_size = configuration->problem_size;
    operator_args.batch_count = configuration->batch_count;

    operator_args.lda = (configuration->lda);
    operator_args.ldb = (configuration->ldb);
    operator_args.ldc = (configuration->ldc);
    operator_args.ldd = (configuration->ldd);

    return Status::kSuccess;
  }

  /// Constructs the arguments structure given the configuration and arguments
  static Status update_arguments_(
    OperatorArguments &operator_args,
    GemmUniversalArguments const *arguments) {
    
    // printf("update6\n");
    //printf("%"
    if (arguments->pointer_mode == ScalarPointerMode::kHost) {//segmentation here
      typename Operator::EpilogueOutputOp::Params params(
        *static_cast<ElementCompute const *>(arguments->alpha),
        *static_cast<ElementCompute const *>(arguments->beta)
      );
      // printf("enter if\n");
      operator_args.epilogue = params;
    }
    else if (arguments->pointer_mode == ScalarPointerMode::kDevice){
      typename Operator::EpilogueOutputOp::Params params(
        static_cast<ElementCompute const *>(arguments->alpha),
        static_cast<ElementCompute const *>(arguments->beta)
      );
      operator_args.epilogue = params; 
    }
    else {
      return Status::kErrorInvalidProblem;
    }

    // update arguments
    operator_args.ptr_A = arguments->A;
    operator_args.ptr_B = arguments->B;
    operator_args.ptr_C = arguments->C;
    operator_args.ptr_D = arguments->D;

    operator_args.batch_stride_A = arguments->batch_stride_A;
    operator_args.batch_stride_B = arguments->batch_stride_B;
    operator_args.batch_stride_C = arguments->batch_stride_C;
    operator_args.batch_stride_D = arguments->batch_stride_D;
    
    return Status::kSuccess;
  }

public:

  //----------------------------Adapter---------------------------------//
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
//-----------------------------------------Adapter-------------------------------------//
  /// Determines whether the GEMM can execute the given problem.
  static Status can_implement(Arguments const &args, CudaHostAdapter *cuda_adapter = nullptr) {
    return GemmKernel::can_implement(to_underlying_arguments(args));
  }

  /// Returns success if the operation can proceed
  /// Gets the host-side workspace
  virtual uint64_t get_host_workspace_size(
    void const *configuration) const {

    return sizeof(Operator);
  }
  
  /// Gets the device-side workspace
  virtual uint64_t get_device_workspace_size(
    void const *configuration_ptr,
    void const *arguments_ptr) const {

    OperatorArguments args;

    Status status = construct_arguments_(
      args, 
      static_cast<GemmUniversalConfiguration const *>(configuration_ptr));
    if (status != Status::kSuccess) {
      return 0;
    }

    status = update_arguments_(
      args,
      static_cast<GemmUniversalArguments const *>(arguments_ptr));
    if (status != Status::kSuccess) {
      return 0;
    }

    uint64_t size = Operator::get_workspace_size(args);

    return size;
  }
  
  /// Initializes the workspace
  virtual Status initialize(
    void const *configuration_ptr, 
    void *host_workspace, 
    void *device_workspace, 
    cudaStream_t stream = nullptr) const {

    OperatorArguments args;

    Status status = construct_arguments_(
      args, 
      static_cast<GemmUniversalConfiguration const *>(configuration_ptr));

    if (status != Status::kSuccess) {
      return status;
    }

    Operator *op = new (host_workspace) Operator;

    status = op->initialize(args, device_workspace, stream);
    
    return status;
  }

  /// Runs the kernel
  virtual Status run(//used
    void const *arguments_ptr,
    void *host_workspace, 
    void *device_workspace = nullptr, 
    cudaStream_t stream = nullptr) const {

    OperatorArguments args;
    
    Status status = update_arguments_(
      args, 
      static_cast<GemmUniversalArguments const *>(arguments_ptr));

    if (status != Status::kSuccess) {
      return status;
    }
    
    Operator *op = static_cast<Operator *>(host_workspace); //new gemm::device::GemmUniversalAdapter<kernel>

    status = op->update(args);

    if (status != Status::kSuccess) {
      return status;
    }
    
    status = op->run(stream);
    
    return status;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace library
} // namespace cutlass

///////////////////////////////////////////////////////////////////////////////////////////////////
