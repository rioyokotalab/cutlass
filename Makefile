.SUFFIXES: .cxx .cu .o


CXX = gcc 
CUDA = nvcc
INC_FLAGS = -Iinclude -Itools/library/include -Itools/library/src -Itools/util/include -Itools/profiler/include
CXX_FLAGS = -fPIC
CUDA_FLAGS = --generate-code=arch=compute_90a,code=[sm_90a] --generate-code=arch=compute_90a,code=[compute_90a] -Xcompiler=-fPIC --expt-relaxed-constexpr -std=c++17 -Xcompiler="-fPIC -Wconversion -fno-strict-aliasing"
CUTLASS_FLAGS = -DCUTLASS_TEST_LEVEL=0 -DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1 -DCUTLASS_ENABLE_TENSOR_CORE_MMA=2

%.cpp.o: %.cpp
	$(CXX) $(INC_FLAGS) $(CXX_FLAGS) -c $? -o $@
%.cu.o: %.cu
	$(CUDA) $(INC_FLAGS) $(CUDA_FLAGS) $(CUTLASS_FLAGS) -c $? -o $@

SOURCE	= $(shell find src -name '*.cpp' -or -name '*.cu')

OBJECT	= $(SOURCE:%=%.o)

main: $(OBJECT)
	$(CUDA) $?
	./a.out
clean:
	$(RM) src/*.o a.out

