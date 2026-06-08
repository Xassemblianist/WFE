CXX      := g++
NVCC     := nvcc

# sm_75=RTX2060 (laptop), sm_86=RTX3080, sm_89=RTX4090
# sm_120=RTX5070Ti (Blackwell) requires CUDA 12.8+
CUDA_ARCH := -gencode arch=compute_75,code=sm_75 \
             -gencode arch=compute_86,code=sm_86 \
             -gencode arch=compute_89,code=sm_89

CXXFLAGS := -std=c++20 -O3 -march=native -DNDEBUG -I.
NVCCFLAGS := -std=c++17 -O3 --use_fast_math $(CUDA_ARCH) \
             --expt-relaxed-constexpr --expt-extended-lambda \
             -I. -DNDEBUG

CUDART   := -L/usr/lib/x86_64-linux-gnu -lcudart

BUILD    := build
BIN      := $(BUILD)/wfe
BIN2D    := $(BUILD)/wfe2d
BIN_MW   := $(BUILD)/wfe_mw
BIN3D    := $(BUILD)/wfe3d

SRCS_CXX := src/main.cpp src/io/output.cpp src/solver/swe1d.cpp
SRCS_CU  := src/solver/cuda/swe1d_gpu.cu

SRCS2D_CXX := src/main2d.cpp
SRCS_MW_CXX := src/main_mw.cpp
SRCS2D_CU  := src/solver/cuda/euler2d_gpu.cu


SRCS3D_CXX := src/main3d.cpp
SRCS3D_CU  := src/solver/cuda/euler3d_gpu.cu

OBJS_CXX := $(patsubst %.cpp,$(BUILD)/%.o,$(SRCS_CXX))
OBJS_CU  := $(patsubst %.cu,$(BUILD)/%.o,$(SRCS_CU))

OBJS2D_CXX := $(patsubst %.cpp,$(BUILD)/%.o,$(SRCS2D_CXX))
OBJS_MW_CXX := $(patsubst %.cpp,$(BUILD)/%.o,$(SRCS_MW_CXX))
OBJS2D_CU  := $(patsubst %.cu,$(BUILD)/%.o,$(SRCS2D_CU))

OBJS3D_CXX := $(patsubst %.cpp,$(BUILD)/%.o,$(SRCS3D_CXX))
OBJS3D_CU  := $(patsubst %.cu,$(BUILD)/%.o,$(SRCS3D_CU))


.PHONY: all clean

all: $(BIN) $(BIN2D) $(BIN_MW) $(BIN3D)

$(BIN): $(OBJS_CXX) $(OBJS_CU)
	@mkdir -p $(@D)
	$(NVCC) $(CUDA_ARCH) --expt-relaxed-constexpr $^ -o $@ $(CUDART)

$(BIN2D): $(OBJS2D_CXX) $(OBJS2D_CU)
	@mkdir -p $(@D)
	$(NVCC) $(CUDA_ARCH) --expt-relaxed-constexpr $^ -o $@ $(CUDART)

$(BIN_MW): $(OBJS_MW_CXX) $(OBJS2D_CU)
	@mkdir -p $(@D)
	$(NVCC) $(CUDA_ARCH) --expt-relaxed-constexpr $^ -o $@ $(CUDART)

$(BIN3D): $(OBJS3D_CXX) $(OBJS3D_CU)
	@mkdir -p $(@D)
	$(NVCC) $(CUDA_ARCH) --expt-relaxed-constexpr $^ -o $@ $(CUDART)



$(BUILD)/%.o: %.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@

$(BUILD)/%.o: %.cu
	@mkdir -p $(@D)
	$(NVCC) $(NVCCFLAGS) -MMD -MP -c $< -o $@

-include $(wildcard $(BUILD)/src/*.d $(BUILD)/src/**/*.d)

clean:
	rm -rf $(BUILD)
