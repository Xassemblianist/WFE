CXX      := g++
NVCC     := nvcc

# sm_75=RTX2060 (laptop), sm_86=RTX3080, sm_89=RTX4090
# sm_120=RTX5070Ti (Blackwell) requires CUDA 12.8+ — enable on main PC
CUDA_ARCH := -gencode arch=compute_75,code=sm_75 \
             -gencode arch=compute_86,code=sm_86 \
             -gencode arch=compute_89,code=sm_89

CXXFLAGS := -std=c++20 -O3 -march=native -DNDEBUG -I.
NVCCFLAGS := -std=c++17 -O3 --use_fast_math $(CUDA_ARCH) \
             --expt-relaxed-constexpr --expt-extended-lambda \
             -I. -DNDEBUG

CUDART   := -L/usr/lib/x86_64-linux-gnu -lcudart

BUILD    := build
BIN      := $(BUILD)/cppwrf
BIN2D    := $(BUILD)/cppwrf2d

SRCS_CXX := src/main.cpp \
             src/io/output.cpp \
             src/solver/swe1d.cpp

SRCS_CU  := src/solver/cuda/swe1d_gpu.cu

SRCS2D_CXX := src/main2d.cpp
SRCS2D_CU  := src/solver/cuda/euler2d_gpu.cu

OBJS_CXX := $(patsubst %.cpp,$(BUILD)/%.o,$(SRCS_CXX))
OBJS_CU  := $(patsubst %.cu,$(BUILD)/%.o,$(SRCS_CU))

OBJS2D_CXX := $(patsubst %.cpp,$(BUILD)/%.o,$(SRCS2D_CXX))
OBJS2D_CU  := $(patsubst %.cu,$(BUILD)/%.o,$(SRCS2D_CU))

.PHONY: all clean

all: $(BIN) $(BIN2D)

$(BIN): $(OBJS_CXX) $(OBJS_CU)
	@mkdir -p $(@D)
	$(NVCC) $(CUDA_ARCH) --expt-relaxed-constexpr $^ -o $@ $(CUDART)

$(BIN2D): $(OBJS2D_CXX) $(OBJS2D_CU)
	@mkdir -p $(@D)
	$(NVCC) $(CUDA_ARCH) --expt-relaxed-constexpr $^ -o $@ $(CUDART)

$(BUILD)/%.o: %.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/%.o: %.cu
	@mkdir -p $(@D)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD)
