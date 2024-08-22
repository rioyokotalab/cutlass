rm -rf build
mkdir build
cd build
cmake ..
make cutlass_profiler -j16
./tools/profiler/cutlass_profiler

