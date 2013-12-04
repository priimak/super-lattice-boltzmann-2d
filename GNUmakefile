all: bin/boltzmann_solver bin/boltzmann_c_solver

NVCC            ?= $(CUDA_BIN_PATH)/nvcc
GCC             ?= gcc

.PHONY: clean check-env check-nvcc

CUDA_PATH       ?= /usr/local/cuda-5.5
CUDA_INC_PATH   ?= $(CUDA_PATH)/include
CUDA_BIN_PATH   ?= $(CUDA_PATH)/bin
CUDA_LIB_PATH   ?= $(CUDA_PATH)/lib64
NVCC            ?= $(CUDA_BIN_PATH)/nvcc

GENCODE_FLAGS := -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35

NVCCFLAGS := -m64
CCFLAGS := -m64
LDFLAGS   := -L$(CUDA_LIB_PATH) -lcudart -lgsl -lgslcblas

check-env:
ifndef BLTZM_KERNEL
	$(error You must define env. variable BLTZM_KERNEL)
endif

check-nvcc:
ifeq (,$(wildcard $(NVCC)))
    $(error Unable to find $(NVCC). You need to specify env. variable CUDA_PATH)
endif

src/boltzmann_gpu.o: check-env check-nvcc

src/boltzmann_gpu.o: src/boltzmann_gpu.cu src/boltzmann_gpu.h
	$(NVCC) $(NVCCFLAGS) -I$(CUDA_INC_PATH) $(GENCODE_FLAGS) -o $@ -c src/boltzmann_gpu.cu -DBLTZM_KERNEL=$(BLTZM_KERNEL)

bin/boltzmann_solver: src/boltzmann_gpu.o src/boltzmann_cli.c src/boltzmann_cli.h src/boltzmann.h src/boltzmann_solver.c src/boltzmann_solver.h
	$(GCC) $(CCFLAGS) -O3 -std=gnu99 -o $@ $+ $(LDFLAGS) $(EXTRA_LDFLAGS) -I$(CUDA_INC_PATH) -DBLTZM_KERNEL=$(BLTZM_KERNEL) && rm -f src/boltzmann_gpu.o

bin/boltzmann_c_solver: src/boltzmann_c_solver.c src/boltzmann_cli.c src/boltzmann_cli.h src/boltzmann.h src/boltzmann_solver.h 
	$(GCC) -std=gnu99 $+ -o $@ -lm -lgsl -lgslcblas

clean: 
	rm -f bin/boltzmann_c_solver bin/boltzmann_solver src/boltzmann_gpu.o && find . -type f -name '*~' -exec rm -f {} \;

