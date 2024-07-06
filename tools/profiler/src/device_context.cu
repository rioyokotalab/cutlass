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
   \brief 
*/

#include "cutlass/profiler/device_context.h"

namespace cutlass {
namespace profiler {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Allocates memory of a given type, capacity (elements), and name
DeviceAllocation *DeviceContext::allocate_tensor(//used
  std::string const &name,
  library::NumericTypeID type, 
  library::LayoutTypeID layout_id, 
  std::vector<int> const &extent, 
  std::vector<int64_t> const &stride,
  int batch_count) {

  device_memory_.emplace_back(type, layout_id, extent, stride, batch_count);
  DeviceAllocation *allocation = &device_memory_.back();
  
  allocations_[name] = allocation;
  return allocation;
}

/// Allocates memory of a given type, capacity (elements), and name
DeviceAllocation *DeviceContext::allocate_tensor( //used
  Options const &options,
  std::string const &name,
  library::NumericTypeID type, 
  library::LayoutTypeID layout_id, 
  std::vector<int> const &extent, 
  std::vector<int64_t> const &stride,
  int batch_count,
  int seed_shift) {

  DeviceAllocation *allocation = 
    allocate_tensor(name, type, layout_id, extent, stride, batch_count);

  // if (options.initialization.enabled) { // print the value and replace here  options.initialization.enabled = 1
    // Distribution data_distribution = options.initialization.data_distribution; 
    // data_distribution.set_uniform(-3, 3, 0);

  // }

  // std::cout << options.initialization.enabled << std::endl;
   // Distribution data_distribution = options.initialization.data_distribution; 
  Distribution data_distribution;
   data_distribution.set_uniform(-3, 3, 0);

  return allocation;
}

/// Allocates memory for sparse meta data 
DeviceAllocation *DeviceContext::allocate_sparsemeta_tensor(
  Options const &options,
  std::string const &name,
  library::NumericTypeID type, 
  library::LayoutTypeID layout_id, 
  library::NumericTypeID type_a,
  std::vector<int> const &extent, 
  std::vector<int64_t> const &stride,
  int batch_count,
  int seed_shift) {

  DeviceAllocation *allocation = 
    allocate_tensor(name, type, layout_id, extent, stride, batch_count);
  return allocation;
}

/// Frees all device memory allocations
void DeviceContext::free() { //used
  allocations_.clear();
  device_memory_.clear();
}


/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace profiler
} // namespace cutlass
