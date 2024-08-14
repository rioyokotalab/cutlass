#################################################################################################
#
# Copyright (c) 2017 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#################################################################################################

"""
Utilities for enumerating CUTLASS library kernels
"""

import argparse
import enum
from itertools import product
import logging
import os.path
import shutil
import sys
import copy
from typing import Any, Optional, Sequence, Tuple

_LOGGER = logging.getLogger(__name__)

# Certain usecases of cutlass_library nearly always prefer to run as scripts with
# relative imports, rather than via an installed Python package. An example of this
# is using CUTLASS's CMake system to generate a library of kernels to be profiled.
# To make it easy to use these use cases when an existing installation of cutlass_library
# exists, this global flag can be set to true (via command-line arguments) to ensure
# that package-based installations are not used.

# Create a temporary argument parser to check only for the availability of the
# --disable-cutlass-package-imports argument, which controls whether package-based
# imports are disabled.
def _add_package_disablement_flag(argparser):
  argparser.add_argument("--disable-cutlass-package-imports", action='store_true', required=False,
                     help="Disable use of cutlass_library from Python package")

_parser = argparse.ArgumentParser()
_add_package_disablement_flag(_parser)
_args, _ = _parser.parse_known_args()

# Add `CUTLASS_IGNORE_PACKAGE` to `builtins` so that it is visible for gating future
# imports without requiring importing another module. Ideally, we would just place this
# as a global variable in a module to that could be imported and checked (e.g.,
# utils.CUTLASS_IGNORE_PACKAGE). However, this raises the issue of determining
# where this module should be sourced (from the cutlass_library package or from
# a relative import), which is the problem this variable is being used to solve in the
# first place.
import builtins
builtins.CUTLASS_IGNORE_PACKAGE = _args.disable_cutlass_package_imports

try:
  if CUTLASS_IGNORE_PACKAGE:
    raise ImportError("Disabling attempt to import cutlass_library")
  from cutlass_library.library import *
  from cutlass_library.manifest import *
except ImportError:
  from library import *
  from manifest import *
###################################################################################################

#
def CudaToolkitVersionSatisfies(semantic_ver_string, major, minor, patch = 0):

  # by default, use the latest CUDA Toolkit version
  cuda_version = [11, 0, 132]

  # Update cuda_version based on parsed string
  if semantic_ver_string != '':
    for i, x in enumerate([int(x) for x in semantic_ver_string.split('.')]):
      if i < len(cuda_version):
        cuda_version[i] = x
      else:
        cuda_version.append(x)
  return cuda_version >= [major, minor, patch]


###################################################################################################
###################################################################################################

def DefaultSwizzlingFunctor():
    return SwizzlingFunctor.Identity8
    # To use StreamK decomposition for basic GEMMs, set `swizzling_functor = SwizzlingFunctor.StreamK`

#
def CreateGemmOperator(manifest, layouts, tile_descriptions, data_type, \
  alignment_constraints, complex_transforms = None, epilogue_functor = EpilogueFunctor.LinearCombination, \
  swizzling_functor = DefaultSwizzlingFunctor()):

  if complex_transforms is None:
    complex_transforms = [(ComplexTransform.none, ComplexTransform.none),]

  element_a, element_b, element_c, element_epilogue = data_type

  operations = []

  # by default, only generate the largest tile and largest alignment
  if manifest.kernel_filter == '':
    tile_descriptions = [tile_descriptions[0],]
    alignment_constraints = [alignment_constraints[0],]

  for layout in layouts:
    for tile_description in tile_descriptions:
      for alignment in alignment_constraints:
        for complex_transform in complex_transforms:

            # If alignment is a tuple or a list, then we have different alignments for A and B
            alignment_a = alignment if isinstance(alignment, int) else alignment[0]
            alignment_b = alignment if isinstance(alignment, int) else alignment[1]
            alignment_c = min(8, alignment_a) if isinstance(alignment, int) else alignment[2]

            A = TensorDescription(element_a, layout[0], alignment_a, complex_transform[0])
            B = TensorDescription(element_b, layout[1], alignment_b, complex_transform[1])
            C = TensorDescription(element_c, layout[2], alignment_c)

            new_operation = GemmOperation(GemmKind.Universal, tile_description.minimum_compute_capability, \
              tile_description, A, B, C, element_epilogue, epilogue_functor, swizzling_functor)

            manifest.append(new_operation)
            operations.append(new_operation)

  return operations

def CreateGemmGroupedOperator(manifest, layouts, tile_descriptions, data_type, \
  alignment_constraints, complex_transforms = None, epilogue_functor = EpilogueFunctor.LinearCombination, \
  swizzling_functor = SwizzlingFunctor.Identity8):

  if complex_transforms is None:
    complex_transforms = [(ComplexTransform.none, ComplexTransform.none),]

  element_a, element_b, element_c, element_epilogue = data_type

  operations = []

  # by default, only generate the largest tile and largest alignment
  if manifest.kernel_filter == '':
    tile_descriptions = [tile_descriptions[0],]
    alignment_constraints = [alignment_constraints[0],]

  for layout in layouts:
    for tile_description in tile_descriptions:
      for alignment in alignment_constraints:
        for complex_transform in complex_transforms:

            alignment_c = min(8, alignment)

            A = TensorDescription(element_a, layout[0], alignment, complex_transform[0])
            B = TensorDescription(element_b, layout[1], alignment, complex_transform[1])
            C = TensorDescription(element_c, layout[2], alignment_c)

            new_operation = GroupedGemmOperation(GemmKind.Grouped, tile_description.minimum_compute_capability, \
              tile_description, A, B, C, element_epilogue, epilogue_functor, swizzling_functor)

            manifest.append(new_operation)
            operations.append(new_operation)

  return operations

#
#

###########################################################################################################
#   ConvolutionOperator support variations
#        ____________________________________________________________________
#         ConvolutionalOperator |      Analytic          |    Optimized
#        ____________________________________________________________________
#        |       Fprop          |     (strided)          |    (strided)
#        |       Dgrad          |     (strided, unity*)  |    (strided, unity)
#        |       Wgrad          |     (strided)          |    (strided)
#        ____________________________________________________________________
#
# Note :  Operator marked (*) are supported but not generated to keep the instantiated kernel count low
###########################################################################################################
# Convolution for 2D operations

###################################################################################################
###################################################################################################

#

###################################################################################################
###################################################################################################

#
def GenerateSM80_TensorOp_16816(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  math_instructions = [
    MathInstruction(                                  \
      [16, 8, 16],                                    \
      DataType.f16, DataType.f16, DataType.f32,       \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add),
    MathInstruction(                                  \
      [16, 8, 16],                                    \
      DataType.f16, DataType.f16, DataType.f16,       \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add),
    MathInstruction(                                  \
      [16, 8, 16],                                    \
      DataType.bf16, DataType.bf16, DataType.f32,     \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add),
  ]

  min_cc = 80
  max_cc = 1024

  alignment_constraints = [8, 4, 2]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([256, 128, 32],  3, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256, 32],  3, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64, 32],  3, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64, 32],  4, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 256, 32],  4, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 32],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 32],  4, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 32],  5, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 32],  6, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 32],  6, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 32], 10, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([256, 128, 64],  3, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256, 64],  3, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64, 64],  4, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 256, 64],  4, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 64],  4, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64, 64],  3, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 256, 64],  3, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 64],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 64],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 64],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 64],  5, [2, 2, 1], math_inst, min_cc, max_cc),
    ]

    data_type = [
      math_inst.element_a,
      math_inst.element_b,
      math_inst.element_accumulator,
      math_inst.element_accumulator,
    ]

    CreateGemmOperator(manifest, layouts, tile_descriptions, \
      data_type, alignment_constraints)

    CreateGemmGroupedOperator(manifest, layouts, tile_descriptions, data_type, alignment_constraints)

    conv_layout = (LayoutType.TensorNHWC, LayoutType.TensorNHWC, LayoutType.TensorNHWC)

    # Avoid emitting two kernels if the accumulator type does not differ from the input type (e.g. F16 accumulation)
    if math_inst.element_a != math_inst.element_accumulator:

      data_type_mixed = [
        math_inst.element_a,
        math_inst.element_b,
        math_inst.element_a,
        math_inst.element_accumulator,
      ]

      CreateGemmOperator(manifest, layouts, tile_descriptions, \
        data_type_mixed, alignment_constraints)

###################################################################################################

#
def GenerateSM80(manifest, cuda_version):
  GenerateSM80_TensorOp_16816(manifest, cuda_version)

###################################################################################################

def numeric_log_level(log_level: str) -> int:
  """
  Converts the string identifier of the log level into the numeric identifier used
  in setting the log level

  :param x: string representation of log level (e.g., 'INFO', 'DEBUG')
  :type x: str

  :return: numeric representation of log level
  :rtype: int
  """
  numeric_level = getattr(logging, log_level.upper(), None)
  if not isinstance(numeric_level, int):
    raise ValueError(f'Invalid log level: {log_level}')
  return numeric_level


# This function for defining the ArgumentParser is used to make it easy for the CUTLASS Python interface
# to leverage the functionality in this file without running this script via a shell prompt.
def define_parser():
  parser = argparse.ArgumentParser(description="Generates device kernel registration code for CUTLASS Kernels")
  parser.add_argument("--operations", default="all", help="Specifies the operation to generate (gemm, all)")
  parser.add_argument("--build-dir", default=".", required=False, help="CUTLASS top-level build directory")
  parser.add_argument("--curr-build-dir", default=".", help="CUTLASS current build directory. cmake files will be emitted in this directory")
  parser.add_argument("--generator-target", default='library', help="Target of CUTLASS Library Generator.")
  parser.add_argument("--architectures", default='53;60;61;70;75;80;90', help="Target compute architectures")
  parser.add_argument("--kernels", default='', help='Comma delimited list to filter kernels by name.')
  parser.add_argument("--ignore-kernels", default='', help='Comma delimited list of kernels to exclude from build.')
  parser.add_argument("--filter-by-cc", default='True', type=str, help='If enabled, kernels whose compute capability range is not satisfied by the build target are excluded.')
  parser.add_argument("--cuda-version", default="11.0.0", help="Semantic version string of CUDA Toolkit")
  parser.add_argument('--kernel-filter-file',   type=str, default=None, required=False, help='Full path of filter file')
  parser.add_argument('--selected-kernel-list',   type=str, default=None, required=False,
                        help='Specify the output log file containing all enabled kernels in this build')
  parser.add_argument("--interface-dir", default=None, required=False, help="Interface header to kernels")
  parser.add_argument("--disable-full-archs-compilation", action="store_true", required=False, help="Disable compilation for every archs in --architectures")
  parser.add_argument("--log-level", default='info', type=numeric_log_level, required=False,
                      help='Logging level to be used by the generator script')
  _add_package_disablement_flag(parser)
  return parser


if __name__ == "__main__":
  parser = define_parser()
  args = parser.parse_args()

  # Set the logging level based on the user-provided `--log-level` command-line option
  logging.basicConfig(level=args.log_level)

  manifest = Manifest(args)

  GenerateSM80(manifest, args.cuda_version)
  if 'library' in args.generator_target.split(','):
    manifest.emit(GeneratorTarget.Library)

  if args.selected_kernel_list is not None:
    if len(manifest.selected_kernels) > 0:
      with open(args.selected_kernel_list, 'w') as file_writer:
        for line in manifest.selected_kernels:
          file_writer.write("%s\n" % line)

###################################################################################################
