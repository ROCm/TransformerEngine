# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information
# KOSMOS build (gfx950, ROCm >= 7.2 with hipcc, rocBLAS, hipBLASLt, RCCL, OpenMPI); targets in README.md.

KOSMOS_ROOT  := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
KOSMOS_BUILD ?= $(KOSMOS_ROOT)/build
ROCM_PATH    ?= $(shell if [ -d /opt/rocm ]; then echo /opt/rocm; else hipconfig --rocmpath 2>/dev/null; fi)
OMPI_HOME    ?= /opt/ompi
HIPCXX       ?= $(ROCM_PATH)/bin/hipcc
HOSTCXX      ?= $(ROCM_PATH)/llvm/bin/clang++
CHECK_ARCH   ?= -march=native
GPU_ARCH     ?= gfx950
HK_INC       := $(KOSMOS_ROOT)/3rdparty/hipkittens/include

MOE_FLAGS := -std=c++20 --offload-arch=$(GPU_ARCH) -DKITTENS_CDNA4 -fno-gpu-rdc -O3 \
             -Wno-unused-value -Wno-deprecated-declarations \
             -I$(HK_INC) -I$(ROCM_PATH)/include/hip -I$(ROCM_PATH)/include -I$(OMPI_HOME)/include
MOE_LIBS  := -L$(OMPI_HOME)/lib -lmpi -Wl,-rpath,$(OMPI_HOME)/lib \
             -L$(ROCM_PATH)/lib -lrocblas -lamdhip64 -Wl,-rpath,$(ROCM_PATH)/lib
CHECK_FLAGS := -std=c++17 -O3 $(CHECK_ARCH) -fopenmp -I$(OMPI_HOME)/include
CHECK_LIBS  := -fopenmp -Wl,-rpath,$(ROCM_PATH)/llvm/lib

TPSP_FLAGS := -std=c++20 -x hip --offload-arch=$(GPU_ARCH) -DKITTENS_CDNA4 -fno-gpu-rdc -O3 \
              -Wno-unused-value -gline-tables-only -fPIC \
              -I$(HK_INC) -I$(ROCM_PATH)/include/hip -I$(ROCM_PATH)/include

COMM_FLAGS := -std=c++20 -O3 -fPIC -I$(ROCM_PATH)/include
COMM_OBJ   := $(KOSMOS_BUILD)/obj/kosmos_comm.o

MOE_LIB_FLAGS := $(filter-out -I$(OMPI_HOME)/include,$(MOE_FLAGS)) -fPIC
MOE_LIB_SRC   := $(wildcard $(KOSMOS_ROOT)/src/moe/*.h $(KOSMOS_ROOT)/src/moe/*.cuh $(KOSMOS_ROOT)/src/moe/*.inc \
                           $(KOSMOS_ROOT)/src/utils/*.h $(KOSMOS_ROOT)/src/utils/*.cuh)
BENCH_COMMON  := $(KOSMOS_ROOT)/bench/common/kosmos_mpi.h $(KOSMOS_ROOT)/bench/common/moe_inputs.cuh \
                 $(KOSMOS_ROOT)/bench/common/moe_shapes.h $(KOSMOS_ROOT)/bench/common/moe_check.h
CHECK_OBJ     := $(KOSMOS_BUILD)/obj/moe_check.o
# The training API's kernel translation units, built once per activation (MoeAct): SwiGLU, and with -DMOE_ACT_CLAMP10
# into *_c10.o.
KERN_OBJ := $(foreach t,moe_train_api moe_train_api_ks,$(KOSMOS_BUILD)/obj/$(t).o $(KOSMOS_BUILD)/obj/$(t)_c10.o)
TRAIN_SRC := $(KOSMOS_ROOT)/bench/moe_train/src/moe_train.cu $(MOE_LIB_SRC) $(BENCH_COMMON)
PYTHON    ?= python3
PY_INC     = $(shell $(PYTHON) -m pybind11 --includes)
PY_SUFFIX  = $(shell $(PYTHON) -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')
PY_MODULE  = $(KOSMOS_BUILD)/python/kosmos$(PY_SUFFIX)
TPSP_SRC := $(wildcard $(KOSMOS_ROOT)/src/tpsp/*.cuh $(KOSMOS_ROOT)/src/tpsp/*.inc \
                       $(KOSMOS_ROOT)/src/tpsp/*.h $(KOSMOS_ROOT)/src/utils/*.cuh $(KOSMOS_ROOT)/src/utils/*.h)

all: moe_train moe tpsp overlap report

moe_train: $(KOSMOS_BUILD)/moe_train
moe:     $(KOSMOS_BUILD)/libkosmos_moe.a $(KOSMOS_BUILD)/moe_linkcheck
tpsp:    $(KOSMOS_BUILD)/libkosmos_tpsp.a $(KOSMOS_BUILD)/tpsp_linkcheck
report:  $(KOSMOS_BUILD)/kosmos_report

$(CHECK_OBJ): $(KOSMOS_ROOT)/bench/common/moe_check.cpp $(KOSMOS_ROOT)/bench/common/moe_check.h
	@mkdir -p $(KOSMOS_BUILD)/obj
	$(HOSTCXX) $(CHECK_FLAGS) -c $< -o $@

$(COMM_OBJ): $(KOSMOS_ROOT)/src/utils/kosmos_comm.cpp $(KOSMOS_ROOT)/src/utils/kosmos_comm.h \
             $(KOSMOS_ROOT)/src/utils/kosmos_comm_impl.h
	@mkdir -p $(KOSMOS_BUILD)/obj
	$(HIPCXX) $(COMM_FLAGS) -c $< -o $@

$(KOSMOS_BUILD)/moe_train: $(TRAIN_SRC) $(CHECK_OBJ) $(COMM_OBJ) $(KERN_OBJ)
	@mkdir -p $(KOSMOS_BUILD)/obj
	rm -f $@
	$(HIPCXX) $(MOE_FLAGS) -c $(KOSMOS_ROOT)/bench/moe_train/src/moe_train.cu -o $(KOSMOS_BUILD)/obj/moe_train.o
	$(HIPCXX) -o $@ $(KOSMOS_BUILD)/obj/moe_train.o $(KERN_OBJ) $(COMM_OBJ) \
	    $(CHECK_OBJ) $(MOE_LIBS) $(CHECK_LIBS)

$(KOSMOS_BUILD)/obj/moe.o: $(KOSMOS_ROOT)/src/moe/moe.cu $(MOE_LIB_SRC)
	@mkdir -p $(KOSMOS_BUILD)/obj
	$(HIPCXX) $(MOE_LIB_FLAGS) -c $< -o $@

$(KOSMOS_BUILD)/obj/moe_train_api.o: $(KOSMOS_ROOT)/src/moe/moe_train_api.cu $(MOE_LIB_SRC)
	@mkdir -p $(KOSMOS_BUILD)/obj
	$(HIPCXX) $(MOE_LIB_FLAGS) -c $< -o $@

$(KOSMOS_BUILD)/obj/%_c10.o: $(KOSMOS_ROOT)/src/moe/%.cu $(MOE_LIB_SRC)
	@mkdir -p $(KOSMOS_BUILD)/obj
	$(HIPCXX) $(MOE_LIB_FLAGS) -DMOE_ACT_CLAMP10=1 -c $< -o $@

$(KOSMOS_BUILD)/obj/%_ks.o: $(KOSMOS_ROOT)/src/moe/%_ks.cu $(MOE_LIB_SRC)
	@mkdir -p $(KOSMOS_BUILD)/obj
	$(HIPCXX) $(MOE_LIB_FLAGS) -c $< -o $@

$(KOSMOS_BUILD)/libkosmos_moe.a: $(KOSMOS_BUILD)/obj/moe.o $(KERN_OBJ) $(COMM_OBJ)
	rm -f $@
	ar rcs $@ $^

$(KOSMOS_BUILD)/moe_linkcheck: $(KOSMOS_ROOT)/src/moe/moe_linkcheck.cpp $(KOSMOS_BUILD)/libkosmos_moe.a
	$(HIPCXX) -std=c++20 --offload-arch=$(GPU_ARCH) -I$(KOSMOS_ROOT)/src/moe \
	  -o $@ $< -x none $(KOSMOS_BUILD)/libkosmos_moe.a -L$(ROCM_PATH)/lib -lamdhip64 -Wl,-rpath,$(ROCM_PATH)/lib

python: $(KOSMOS_BUILD)/libkosmos_moe.a $(KOSMOS_ROOT)/python/kosmos.cpp $(KOSMOS_ROOT)/src/moe/moe.h
	@mkdir -p $(KOSMOS_BUILD)/python
	$(HIPCXX) -std=c++20 -O2 -fPIC -shared -fvisibility=hidden --offload-arch=$(GPU_ARCH) $(PY_INC) \
	  -I$(KOSMOS_ROOT)/src/moe -o $(PY_MODULE) $(KOSMOS_ROOT)/python/kosmos.cpp -x none \
	  $(KOSMOS_BUILD)/libkosmos_moe.a -Wl,--exclude-libs,ALL -L$(ROCM_PATH)/lib -lamdhip64 -Wl,-rpath,$(ROCM_PATH)/lib

$(KOSMOS_BUILD)/obj/comm_gemm.o: $(KOSMOS_ROOT)/src/tpsp/comm_gemm.cpp $(TPSP_SRC)
	@mkdir -p $(KOSMOS_BUILD)/obj
	$(HIPCXX) $(TPSP_FLAGS) -c $< -o $@

$(KOSMOS_BUILD)/libkosmos_tpsp.a: $(KOSMOS_BUILD)/obj/comm_gemm.o $(COMM_OBJ)
	rm -f $@
	ar rcs $@ $^

$(KOSMOS_BUILD)/tpsp_linkcheck: $(KOSMOS_ROOT)/src/tpsp/tpsp_linkcheck.cpp $(KOSMOS_BUILD)/libkosmos_tpsp.a
	$(HIPCXX) -std=c++20 --offload-arch=$(GPU_ARCH) -I$(KOSMOS_ROOT)/src/tpsp \
	  -o $@ $< -x none $(KOSMOS_BUILD)/libkosmos_tpsp.a -L$(ROCM_PATH)/lib -lamdhip64 -Wl,-rpath,$(ROCM_PATH)/lib

overlap: $(COMM_OBJ)
	$(MAKE) -C $(KOSMOS_ROOT)/bench/tpsp KOSMOS_BUILD=$(KOSMOS_BUILD) ROCM_PATH=$(ROCM_PATH) \
	  OMPI_HOME=$(OMPI_HOME) GPU_ARCH=$(GPU_ARCH)

$(KOSMOS_BUILD)/kosmos_report: $(KOSMOS_ROOT)/bench/report.cpp
	@mkdir -p $(KOSMOS_BUILD)
	$(HOSTCXX) -std=c++17 -O2 -o $@ $<

clean:
	rm -rf $(KOSMOS_BUILD)

.PHONY: all moe_train moe tpsp overlap report python clean
