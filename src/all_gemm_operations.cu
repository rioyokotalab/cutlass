
/*
 Generated by manifest.py - Do not edit.
*/

#include "cutlass/cutlass.h"
#include "cutlass/library/library.h"
#include "cutlass/library/manifest.h"

namespace cutlass {
namespace library {

///////////////////////////////////////////////////////////////////////////////////////////////////

void initialize_cutlass_tensorop_s16816gemm_f16_128x128_32x5_nt_align8(Manifest &manifest);


//
// Entry point to construct operations
//
void initialize_all_gemm_operations(Manifest &manifest) {
  initialize_cutlass_tensorop_s16816gemm_f16_128x128_32x5_nt_align8(manifest);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace library
} // namespace cutlass

