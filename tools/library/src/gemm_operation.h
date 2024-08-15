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
#include "cutlass/gemm/kernel/default_gemm_planar_complex_universal.h"

#include "cutlass/library/library.h"
#include "library_internal.h"

///////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace library {

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Operator_>
class GemmOperationBase : public Operation {
public:
  using Operator = Operator_;
  using ElementA = typename Operator::ElementA;
  using LayoutA = typename Operator::LayoutA;
  using ElementB = typename Operator::ElementB;
  using LayoutB = typename Operator::LayoutB;
  using ElementC = typename Operator::ElementC;
  using LayoutC = typename Operator::LayoutC;
  using ElementD = ElementC;
  using LayoutD = LayoutC;
  // assuming all tensors use same type for StrideIndex 
  using StrideIndex = typename Operator::LayoutA::Index;
  using ElementAccumulator = typename Operator::ElementAccumulator;
  using ElementCompute = typename Operator::EpilogueOutputOp::ElementCompute;

  using OperatorArguments = typename Operator::Arguments;

protected:

  /// 
  GemmDescription description_;

public:

  /// Constructor
  GemmOperationBase(char const *name = "unknown_gemm") {

    description_.name = name;
    description_.provider = Provider::kCUTLASS;
    description_.kind = OperationKind::kGemm;
    description_.gemm_kind = GemmKind::kGemm;

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
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Operator_>
class GemmUniversalOperation : public GemmOperationBase<Operator_> {
public:

  using Operator = Operator_;
  using ElementA = typename Operator::ElementA;
  using LayoutA = typename Operator::LayoutA;
  using ElementB = typename Operator::ElementB;
  using LayoutB = typename Operator::LayoutB;
  using ElementC = typename Operator::ElementC;
  using LayoutC = typename Operator::LayoutC;
  using ElementD = ElementC;
  using LayoutD = LayoutC;
  using ElementAccumulator = typename Operator::ElementAccumulator;
  using ElementCompute = typename Operator::EpilogueOutputOp::ElementCompute;

  using OperatorArguments = typename Operator::Arguments;

public:

  /// Constructor
  GemmUniversalOperation(char const *name = "unknown_gemm"): 
    GemmOperationBase<Operator_>(name) {

    this->description_.gemm_kind = GemmKind::kUniversal;
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

  /// Returns success if the operation can proceed
  virtual Status can_implement(
    void const *configuration_ptr, 
    void const *arguments_ptr) const {
    
    GemmUniversalConfiguration const *configuration = 
      static_cast<GemmUniversalConfiguration const *>(configuration_ptr);

    GemmUniversalArguments const *arguments = 
      static_cast<GemmUniversalArguments const *>(arguments_ptr);

    OperatorArguments args;

    Status status = construct_arguments_(args, configuration);

    if (status != Status::kSuccess) {
      return status;
    }

    status = update_arguments_(args, arguments);

    if (status != Status::kSuccess) {
      return status;
    }

    return Operator::can_implement(args);
  }
  
  /// Gets the host-side workspace
  virtual uint64_t get_host_workspace_size(
    void const *configuration) const {

    return sizeof(Operator);
  }
  
  /// Gets the device-side workspace
  virtual uint64_t get_device_workspace_size(
    void const *configuration_ptr,
    void const *arguments_ptr) const {

    //printf("2\n");
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
    
    Operator *op = static_cast<Operator *>(host_workspace);

    status = op->update(args);

    if (status != Status::kSuccess) {
      return status;
    }
    
    status = op->run(stream);
    
    return status;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Operator_>
class GemmPlanarComplexOperation : public GemmOperationBase<Operator_> {
public:

  using Operator = Operator_;
  using ElementA = typename Operator::ElementA;
  using LayoutA = typename Operator::LayoutA;
  using ElementB = typename Operator::ElementB;
  using LayoutB = typename Operator::LayoutB;
  using ElementC = typename Operator::ElementC;
  using LayoutC = typename Operator::LayoutC;
  using ElementD = ElementC;
  using LayoutD = LayoutC;
  using ElementAccumulator = typename Operator::ElementAccumulator;
  using ElementCompute = typename Operator::EpilogueOutputOp::ElementCompute;

  using OperatorArguments = typename Operator::Arguments;

public:

  /// Constructor
  GemmPlanarComplexOperation(char const *name = "unknown_gemm"): GemmOperationBase<Operator_>(name) {

    this->description_.gemm_kind = GemmKind::kPlanarComplex;
  }

protected:

  /// Constructs the arguments structure given the configuration and arguments
  static Status construct_arguments_(
    OperatorArguments &operator_args,
    GemmPlanarComplexConfiguration const *configuration) {

    operator_args.mode = cutlass::gemm::GemmUniversalMode::kBatched;
    operator_args.problem_size = configuration->problem_size;
    operator_args.batch_count = configuration->batch_count;


    operator_args.lda_real = configuration->lda_real;
    operator_args.lda_imag = configuration->lda_imag;
    operator_args.ldb_real = configuration->ldb_real;
    operator_args.ldb_imag = configuration->ldb_imag;
    operator_args.ldc_real = configuration->ldc_real;
    operator_args.ldc_imag = configuration->ldc_imag;
    operator_args.ldd_real = configuration->ldd_real;
    operator_args.ldd_imag = configuration->ldd_imag;

    return Status::kSuccess;
  }

  /// Constructs the arguments structure given the configuration and arguments
  static Status update_arguments_(
    OperatorArguments &operator_args,
    GemmPlanarComplexArguments const *arguments) {
    
    printf("enter update");
    if (arguments->pointer_mode == ScalarPointerMode::kHost) {
      typename Operator::EpilogueOutputOp::Params params(
        *static_cast<cutlass::complex<ElementCompute> const *>(arguments->alpha),
        *static_cast<cutlass::complex<ElementCompute> const *>(arguments->beta)
      );
      operator_args.epilogue = params;
    }
    else if (arguments->pointer_mode == ScalarPointerMode::kDevice){
      typename Operator::EpilogueOutputOp::Params params(
        static_cast<cutlass::complex<ElementCompute> const *>(arguments->alpha),
        static_cast<cutlass::complex<ElementCompute> const *>(arguments->beta)
      );
      operator_args.epilogue = params; 
    }
    else {
      return Status::kErrorInvalidProblem;
    }

    // update arguments
    operator_args.ptr_A_real = arguments->A_real;
    operator_args.ptr_A_imag = arguments->A_imag;
    operator_args.ptr_B_real = arguments->B_real;
    operator_args.ptr_B_imag = arguments->B_imag;
    operator_args.ptr_C_real = arguments->C_real;
    operator_args.ptr_C_imag = arguments->C_imag;
    operator_args.ptr_D_real = arguments->D_real;
    operator_args.ptr_D_imag = arguments->D_imag;

    operator_args.batch_stride_A = arguments->batch_stride_A_real;
    operator_args.batch_stride_A_imag = arguments->batch_stride_A_imag;
    operator_args.batch_stride_B = arguments->batch_stride_B_real;
    operator_args.batch_stride_B_imag = arguments->batch_stride_B_imag;
    operator_args.batch_stride_C = arguments->batch_stride_C_real;
    operator_args.batch_stride_C_imag = arguments->batch_stride_C_imag;
    operator_args.batch_stride_D = arguments->batch_stride_D_real;
    operator_args.batch_stride_D_imag = arguments->batch_stride_D_imag;
    
    return Status::kSuccess;
  }

public:

  /// Returns success if the operation can proceed
  virtual Status can_implement(
    void const *configuration_ptr, 
    void const *arguments_ptr) const {
    
    GemmPlanarComplexConfiguration const *configuration = 
      static_cast<GemmPlanarComplexConfiguration const *>(configuration_ptr);

    GemmPlanarComplexArguments const *arguments = 
      static_cast<GemmPlanarComplexArguments const *>(arguments_ptr);

    OperatorArguments args;

    Status status = construct_arguments_(args, configuration);

    if (status != Status::kSuccess) {
      return status;
    }

    status = update_arguments_(args, arguments);

    if (status != Status::kSuccess) {
      return status;
    }

    return Operator::can_implement(args);
  }
  
  /// Gets the host-side workspace
  virtual uint64_t get_host_workspace_size(
    void const *configuration) const {

    return sizeof(Operator);
  }
  
  /// Gets the device-side workspace
  virtual uint64_t get_device_workspace_size(
    void const *configuration_ptr,
    void const *arguments_ptr = nullptr) const {

    printf("3\n");
    OperatorArguments args;

    Status status = construct_arguments_(
      args, 
      static_cast<GemmPlanarComplexConfiguration const *>(configuration_ptr));

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
      static_cast<GemmPlanarComplexConfiguration const *>(configuration_ptr));

    if (status != Status::kSuccess) {
      return status;
    }

    Operator *op = new (host_workspace) Operator;

    status = op->initialize(args, device_workspace, stream);
    
    return status;
  }

  /// Runs the kernel
  virtual Status run(
    void const *arguments_ptr,
    void *host_workspace,
    void *device_workspace = nullptr,
    cudaStream_t stream = nullptr) const {

    OperatorArguments args;

    Status status = update_arguments_(
      args,
      static_cast<GemmPlanarComplexArguments const *>(arguments_ptr));

    if (status != Status::kSuccess) {
      return status;
    }

    Operator *op = static_cast<Operator *>(host_workspace);

    status = op->update(args);

    if (status != Status::kSuccess) {
      return status;
    }

    status = op->run(stream);

    return status;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Operator_>
class GemmPlanarComplexArrayOperation : public GemmOperationBase<Operator_> {
public:

  using Operator = Operator_;
  using ElementA = typename Operator::ElementA;
  using LayoutA = typename Operator::LayoutA;
  using ElementB = typename Operator::ElementB;
  using LayoutB = typename Operator::LayoutB;
  using ElementC = typename Operator::ElementC;
  using LayoutC = typename Operator::LayoutC;
  using ElementD = ElementC;
  using LayoutD = LayoutC;
  using ElementAccumulator = typename Operator::ElementAccumulator;
  using ElementCompute = typename Operator::EpilogueOutputOp::ElementCompute;

  using OperatorArguments = typename Operator::Arguments;

public:

  /// Constructor
  GemmPlanarComplexArrayOperation(char const *name = "unknown_gemm"): GemmOperationBase<Operator_>(name) {

    this->description_.gemm_kind = GemmKind::kPlanarComplexArray;
  }

protected:

  /// Constructs the arguments structure given the configuration and arguments
  static Status construct_arguments_(
    OperatorArguments &operator_args,
    GemmPlanarComplexArrayConfiguration const *configuration) {

    operator_args.mode = cutlass::gemm::GemmUniversalMode::kArray;
    operator_args.problem_size = configuration->problem_size;
    operator_args.batch_count = configuration->batch_count;

    operator_args.lda_real = configuration->lda_real;
    operator_args.lda_imag = configuration->lda_imag;
    operator_args.ldb_real = configuration->ldb_real;
    operator_args.ldb_imag = configuration->ldb_imag;
    operator_args.ldc_real = configuration->ldc_real;
    operator_args.ldc_imag = configuration->ldc_imag;
    operator_args.ldd_real = configuration->ldd_real;
    operator_args.ldd_imag = configuration->ldd_imag;

    return Status::kSuccess;
  }

  /// Constructs the arguments structure given the configuration and arguments
  static Status update_arguments_(
    OperatorArguments &operator_args,
    GemmPlanarComplexArrayArguments const *arguments) {
    
    printf("update2\n");
    if (arguments->pointer_mode == ScalarPointerMode::kHost) {
      typename Operator::EpilogueOutputOp::Params params(
        *static_cast<cutlass::complex<ElementCompute> const *>(arguments->alpha),
        *static_cast<cutlass::complex<ElementCompute> const *>(arguments->beta)
      );
      operator_args.epilogue = params;
    }
    else if (arguments->pointer_mode == ScalarPointerMode::kDevice){
      typename Operator::EpilogueOutputOp::Params params(
        static_cast<cutlass::complex<ElementCompute> const *>(arguments->alpha),
        static_cast<cutlass::complex<ElementCompute> const *>(arguments->beta)
      );
      operator_args.epilogue = params; 
    }
    else {
      return Status::kErrorInvalidProblem;
    }

    // update arguments
    operator_args.ptr_A_real = arguments->A_real;
    operator_args.ptr_A_imag = arguments->A_imag;
    operator_args.ptr_B_real = arguments->B_real;
    operator_args.ptr_B_imag = arguments->B_imag;
    operator_args.ptr_C_real = arguments->C_real;
    operator_args.ptr_C_imag = arguments->C_imag;
    operator_args.ptr_D_real = arguments->D_real;
    operator_args.ptr_D_imag = arguments->D_imag;

    operator_args.ptr_M = arguments->M;
    operator_args.ptr_N = arguments->N;
    operator_args.ptr_K = arguments->K;
    
    return Status::kSuccess;
  }

public:

  /// Returns success if the operation can proceed
  virtual Status can_implement(
    void const *configuration_ptr, 
    void const *arguments_ptr) const {
    
    GemmPlanarComplexArrayConfiguration const *configuration = 
      static_cast<GemmPlanarComplexArrayConfiguration const *>(configuration_ptr);

    GemmPlanarComplexArrayArguments const *arguments = 
      static_cast<GemmPlanarComplexArrayArguments const *>(arguments_ptr);

    OperatorArguments args;

    Status status = construct_arguments_(args, configuration);

    if (status != Status::kSuccess) {
      return status;
    }

    status = update_arguments_(args, arguments);

    if (status != Status::kSuccess) {
      return status;
    }

    return Operator::can_implement(args);
  }
  
  /// Gets the host-side workspace
  virtual uint64_t get_host_workspace_size(
    void const *configuration) const {

    return sizeof(Operator);
  }
  
  /// Gets the device-side workspace
  virtual uint64_t get_device_workspace_size(
    void const *configuration_ptr,
    void const *arguments_ptr = nullptr) const {

    printf("4\n");
    OperatorArguments args;

    Status status = construct_arguments_(
      args, 
      static_cast<GemmPlanarComplexArrayConfiguration const *>(configuration_ptr));

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
      static_cast<GemmPlanarComplexArrayConfiguration const *>(configuration_ptr));

    if (status != Status::kSuccess) {
      return status;
    }

    Operator *op = new (host_workspace) Operator;

    status = op->initialize(args, device_workspace, stream);
    
    return status;
  }

  /// Runs the kernel
  virtual Status run(
    void const *arguments_ptr,
    void *host_workspace, 
    void *device_workspace = nullptr, 
    cudaStream_t stream = nullptr) const {

    OperatorArguments args;
    
    Status status = update_arguments_(
      args, 
      static_cast<GemmPlanarComplexArrayArguments const *>(arguments_ptr));

    if (status != Status::kSuccess) {
      return status;
    }
    
    Operator *op = static_cast<Operator *>(host_workspace);
    
    status = op->update(args);

    if (status != Status::kSuccess) {
      return status;
    }
    
    status = op->run(stream);
    
    return status;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Operator_>
class GemmGroupedOperation : public GemmOperationBase<Operator_> {
public:

  using Operator = Operator_;
  using ElementA = typename Operator::ElementA;
  using LayoutA = typename Operator::LayoutA;
  using ElementB = typename Operator::ElementB;
  using LayoutB = typename Operator::LayoutB;
  using ElementC = typename Operator::ElementC;
  using LayoutC = typename Operator::LayoutC;
  using ElementD = ElementC;
  using LayoutD = LayoutC;
  using ElementAccumulator = typename Operator::ElementAccumulator;
  using ElementCompute = typename Operator::EpilogueOutputOp::ElementCompute;

  using OperatorArguments = typename Operator::Arguments;

public:

  /// Constructor
  GemmGroupedOperation(char const *name = "unknown_gemm"):
    GemmOperationBase<Operator_>(name) {

    this->description_.gemm_kind = GemmKind::kGrouped;
  }

protected:

  /// Constructs the arguments structure given the configuration and arguments
  static Status construct_arguments_(
    OperatorArguments &op_args,
    GemmGroupedConfiguration const *config) {

    op_args.problem_count = config->problem_count;
    op_args.threadblock_count = config->threadblock_count;

    return Status::kSuccess;
  }

  /// Constructs the arguments structure given the configuration and arguments
  static Status update_arguments_(
    OperatorArguments &op_args,
    GemmGroupedArguments const *arguments) {

    printf("update3\n");
    if (arguments->pointer_mode == ScalarPointerMode::kHost) {

      typename Operator::EpilogueOutputOp::Params params(
        *static_cast<ElementCompute const *>(arguments->alpha),
        *static_cast<ElementCompute const *>(arguments->beta)
      );

      op_args.output_op = params;
    }
    else if (arguments->pointer_mode == ScalarPointerMode::kDevice) {

      typename Operator::EpilogueOutputOp::Params params(
        static_cast<ElementCompute const *>(arguments->alpha),
        static_cast<ElementCompute const *>(arguments->beta)
      );

      op_args.output_op = params;
    }
    else {
      return Status::kErrorInvalidProblem;
    }

    op_args.problem_sizes = arguments->problem_sizes;

    op_args.ptr_A         = static_cast<ElementA **>(arguments->ptr_A);
    op_args.ptr_B         = static_cast<ElementB **>(arguments->ptr_B);
    op_args.ptr_C         = static_cast<ElementC **>(arguments->ptr_C);
    op_args.ptr_D         = static_cast<ElementD **>(arguments->ptr_D);

    op_args.lda           = arguments->lda;
    op_args.ldb           = arguments->ldb;
    op_args.ldc           = arguments->ldc;
    op_args.ldd           = arguments->ldd;

    return Status::kSuccess;
  }

public:

  /// Returns success if the operation can proceed
  virtual Status can_implement(
    void const *configuration_ptr,
    void const *arguments_ptr) const {

    GemmGroupedConfiguration const *configuration =
      static_cast<GemmGroupedConfiguration const *>(configuration_ptr);

    GemmGroupedArguments const *arguments =
      static_cast<GemmGroupedArguments const *>(arguments_ptr);

    OperatorArguments args;

    Status status = construct_arguments_(args, configuration);

    if (status != Status::kSuccess) {
      return status;
    }

    status = update_arguments_(args, arguments);

    if (status != Status::kSuccess) {
      return status;
    }

    return Operator::can_implement(args);
  }

  /// Gets the host-side workspace
  virtual uint64_t get_host_workspace_size(
    void const *configuration) const {

    return sizeof(Operator);
  }

  /// Gets the device-side workspace
  virtual uint64_t get_device_workspace_size(
    void const *configuration_ptr,
    void const *arguments_ptr) const {

    printf("5\n");
    OperatorArguments args;

    Status status = construct_arguments_(
      args,
      static_cast<GemmGroupedConfiguration const *>(configuration_ptr));

    if (status != Status::kSuccess) {
      return 0;
    }

    status = update_arguments_(
      args,
      static_cast<GemmGroupedArguments const *>(arguments_ptr));

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
      static_cast<GemmGroupedConfiguration const *>(configuration_ptr));

    if (status != Status::kSuccess) {
      return status;
    }

    Operator *op = new (host_workspace) Operator;

    status = op->initialize(args, device_workspace, stream);

    return status;
  }

  /// Runs the kernel
  virtual Status run(
    void const *arguments_ptr,
    void *host_workspace,
    void *device_workspace = nullptr,
    cudaStream_t stream = nullptr) const {

    OperatorArguments args;

    Status status = update_arguments_(
      args,
      static_cast<GemmGroupedArguments const *>(arguments_ptr));

    if (status != Status::kSuccess) {
      return status;
    }

    Operator *op = static_cast<Operator *>(host_workspace);

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
