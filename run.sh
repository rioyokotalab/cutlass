rm -rf build
mkdir build
cd build
cmake .. -DCUTLASS_NVCC_ARCHS='90a' -DCUTLASS_LIBRARY_KERNELS=cutlass_tensorop_s16832spgemm_f16_128x128_64x3_nt_align8

make cutlass_profiler -j16

./tools/profiler/cutlass_profiler --kernels=cutlass_tensorop_s16832spgemm_f16_128x128_64x3_nt_align8 --m=3456 --n=4096 --k=4096
