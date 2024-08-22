rm -rf build
mkdir build
cd build
cmake .. -DCUTLASS_LIBRARY_KERNELS=cutlass_tensorop_s16816gemm_f16_128x128_32x5_nt_align8
make cutlass_profiler -j16
./tools/profiler/cutlass_profiler

