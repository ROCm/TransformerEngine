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
                 $(KOSMOS_ROOT)/bench/common/moe_shapes.h $(KOSMOS_ROOT)/bench/common/moe_check.h \
                 $(KOSMOS_ROOT)/bench/common/moe_warm.cuh
CHECK_OBJ     := $(KOSMOS_BUILD)/obj/moe_check.o
# The kernel translation units built once per activation (MoeAct): SwiGLU, and with -DMOE_ACT_CLAMP10 into *_c10.o
# (moe_train_api.cu: the training API's);
# the router's also once per score function (MoeScore): softmax, and with -DMOE_ROUTER_SCORE=n into *_s<n>.o; the
# shared expert's gated backward from *_bwdg.cu; the MXFP8 kernels' families (MoeMxV) from *_ks.cu and *_ks_pin.cu.
ROUTER_SCORES := 1 2 3
PLAIN_OBJ     := $(KOSMOS_BUILD)/obj/moe_plain.o $(KOSMOS_BUILD)/obj/moe_plain_c10.o \
                 $(KOSMOS_BUILD)/obj/moe_train_api.o $(KOSMOS_BUILD)/obj/moe_train_api_c10.o \
                 $(KOSMOS_BUILD)/obj/moe_train_api_ks.o $(KOSMOS_BUILD)/obj/moe_train_api_ks_c10.o
MX_RT_TUS     := moe_mx_router moe_mx_router_ks moe_mx_router_q8 moe_mx_router_q8_ks moe_mx_shared_router \
                 moe_mx_shared_router_ks moe_mx_shared_router_q8 moe_mx_shared_router_q8_ks moe_mx_shared_router_bwdg \
                 moe_mx_shared_router_q8_bwdg
ROUTER_TUS    := moe_router moe_router_c10 moe_shared_router moe_shared_router_c10 moe_shared_router_bwdg \
                 moe_shared_router_bwdg_c10 moe_router_ks moe_router_ks_c10 moe_shared_router_ks \
                 moe_shared_router_ks_c10 $(foreach t,$(MX_RT_TUS),$(t) $(t)_c10)
ROUTER_OBJ    := $(foreach t,$(ROUTER_TUS),$(KOSMOS_BUILD)/obj/$(t).o \
                   $(foreach n,$(ROUTER_SCORES),$(KOSMOS_BUILD)/obj/$(t)_s$(n).o))
SHARED_OBJ    := $(KOSMOS_BUILD)/obj/moe_shared.o $(KOSMOS_BUILD)/obj/moe_shared_c10.o \
                 $(KOSMOS_BUILD)/obj/moe_shared_bwdg.o $(KOSMOS_BUILD)/obj/moe_shared_bwdg_c10.o
MX_SH_TUS     := moe_mx_shared moe_mx_shared_ks moe_mx_shared_ks_pin moe_mx_shared_bwdg moe_mx_shared_q8 \
                 moe_mx_shared_q8_ks moe_mx_shared_q8_ks_pin moe_mx_shared_q8_bwdg
MX_OBJ        := $(foreach t,moe_mx moe_mx_q8 moe_mx_ks moe_mx_q8_ks moe_mx_ks_pin moe_mx_q8_ks_pin $(MX_SH_TUS), \
                   $(KOSMOS_BUILD)/obj/$(t).o $(KOSMOS_BUILD)/obj/$(t)_c10.o)
KS_OBJ        := $(foreach t,moe_ks moe_shared_ks moe_ks_pin moe_shared_ks_pin,$(KOSMOS_BUILD)/obj/$(t).o \
                   $(KOSMOS_BUILD)/obj/$(t)_c10.o)
KERN_OBJ      := $(PLAIN_OBJ) $(ROUTER_OBJ) $(SHARED_OBJ) $(KS_OBJ) $(MX_OBJ)
# Backward kernels at the register limit whose round loop (over-capacity calls) spills under MachineLICM's hoisted
# constants: their translation units are built without it (some of bf16's, all of MXFP8's).
NOLICM_OBJ    := $(filter $(KOSMOS_BUILD)/obj/moe_ks_pin% $(KOSMOS_BUILD)/obj/moe_shared_ks_pin% \
                   $(KOSMOS_BUILD)/obj/moe_shared_router% $(KOSMOS_BUILD)/obj/moe_mx%,$(KERN_OBJ))
$(NOLICM_OBJ): MOE_LIB_FLAGS += -mllvm -disable-machine-licm
BWD_SRC  := $(KOSMOS_ROOT)/bench/moe_bwd/src/moe_bwd_dx.cu $(MOE_LIB_SRC) $(BENCH_COMMON)
BENCH_SRC     := $(KOSMOS_ROOT)/bench/moe/src
# The unfused ablation's objects (moe_bench --unfused), <tag>:<mx>:<V> per precision and kernel family (V 0: k in {6,
# 8, 10}, 1: the small-k kernels, 2: probs on input); the backward's also per activation (c10) and part.
PART_VARIANTS := part:0:0 kspart:0:1 pinpart:0:2 mxpart:1:0 mxkspart:1:1 mxpinpart:1:2
pv             = $(word $(2),$(subst :, ,$(1)))
FWD_PART_OBJ  := $(foreach v,$(PART_VARIANTS),$(KOSMOS_BUILD)/obj/moe_fwd_$(call pv,$(v),1).o)
BWD_PART_OBJ  := $(foreach v,$(PART_VARIANTS),$(foreach a,_ _c10,$(foreach p,1 2 3 4 5, \
                   $(KOSMOS_BUILD)/obj/moe_bwd$(a)$(call pv,$(v),1)$(p).o)))
TRAIN_SRC := $(KOSMOS_ROOT)/bench/moe_train/src/moe_train.cu $(MOE_LIB_SRC) $(BENCH_COMMON)
PYTHON    ?= python3
PY_INC     = $(shell $(PYTHON) -m pybind11 --includes)
PY_SUFFIX  = $(shell $(PYTHON) -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')
PY_MODULE  = $(KOSMOS_BUILD)/python/kosmos$(PY_SUFFIX)
TPSP_SRC := $(wildcard $(KOSMOS_ROOT)/src/tpsp/*.cuh $(KOSMOS_ROOT)/src/tpsp/*.inc \
                       $(KOSMOS_ROOT)/src/tpsp/*.h $(KOSMOS_ROOT)/src/utils/*.cuh $(KOSMOS_ROOT)/src/utils/*.h)

all: moe_fwd moe_bwd moe_train moe_bench moe tpsp overlap routing report

moe_fwd: $(KOSMOS_BUILD)/moe_fwd
moe_bwd: $(KOSMOS_BUILD)/moe_bwd_dx
moe_train: $(KOSMOS_BUILD)/moe_train
moe_bench: $(KOSMOS_BUILD)/moe_bench
moe:     $(KOSMOS_BUILD)/libkosmos_moe.a $(KOSMOS_BUILD)/moe_linkcheck
tpsp:    $(KOSMOS_BUILD)/libkosmos_tpsp.a $(KOSMOS_BUILD)/tpsp_linkcheck
routing: $(KOSMOS_BUILD)/gen_routing_ref
report:  $(KOSMOS_BUILD)/kosmos_report

$(CHECK_OBJ): $(KOSMOS_ROOT)/bench/common/moe_check.cpp $(KOSMOS_ROOT)/bench/common/moe_check.h
	@mkdir -p $(KOSMOS_BUILD)/obj
	$(HOSTCXX) $(CHECK_FLAGS) -c $< -o $@

$(COMM_OBJ): $(KOSMOS_ROOT)/src/utils/kosmos_comm.cpp $(KOSMOS_ROOT)/src/utils/kosmos_comm.h \
             $(KOSMOS_ROOT)/src/utils/kosmos_comm_impl.h
	@mkdir -p $(KOSMOS_BUILD)/obj
	$(HIPCXX) $(COMM_FLAGS) -c $< -o $@

$(KOSMOS_BUILD)/moe_fwd: $(KOSMOS_ROOT)/bench/moe_fwd/src/moe_fwd.cu $(MOE_LIB_SRC) $(BENCH_COMMON) $(CHECK_OBJ) \
                         $(COMM_OBJ) $(KERN_OBJ)
	@mkdir -p $(KOSMOS_BUILD)/obj
	rm -f $@
	$(HIPCXX) $(MOE_FLAGS) -c $< -o $(KOSMOS_BUILD)/obj/moe_fwd.o
	$(HIPCXX) -o $@ $(KOSMOS_BUILD)/obj/moe_fwd.o $(KERN_OBJ) $(COMM_OBJ) \
	    $(CHECK_OBJ) $(MOE_LIBS) $(CHECK_LIBS)

moe_fwd_shexp: $(KOSMOS_BUILD)/moe_fwd_shexp

$(KOSMOS_BUILD)/moe_fwd_shexp: $(KOSMOS_ROOT)/bench/moe_fwd/src/moe_fwd_shexp.cu $(MOE_LIB_SRC) $(BENCH_COMMON) \
                               $(CHECK_OBJ) $(COMM_OBJ) $(KERN_OBJ)
	@mkdir -p $(KOSMOS_BUILD)/obj
	$(HIPCXX) $(MOE_FLAGS) -c $< -o $(KOSMOS_BUILD)/obj/moe_fwd_shexp.o
	$(HIPCXX) -o $@ $(KOSMOS_BUILD)/obj/moe_fwd_shexp.o $(KERN_OBJ) $(COMM_OBJ) \
	    $(CHECK_OBJ) $(MOE_LIBS) -lhipblaslt $(CHECK_LIBS)

moe_shexp_ab: $(KOSMOS_BUILD)/moe_shexp_ab

$(KOSMOS_BUILD)/moe_shexp_ab: $(KOSMOS_ROOT)/bench/moe_train/src/moe_shexp_ab.cu $(MOE_LIB_SRC) $(BENCH_COMMON) \
                              $(COMM_OBJ) $(KERN_OBJ)
	@mkdir -p $(KOSMOS_BUILD)/obj
	$(HIPCXX) $(MOE_FLAGS) -c $< -o $(KOSMOS_BUILD)/obj/moe_shexp_ab.o
	$(HIPCXX) -o $@ $(KOSMOS_BUILD)/obj/moe_shexp_ab.o $(KERN_OBJ) $(COMM_OBJ) \
	    $(MOE_LIBS) -lhipblaslt

$(KOSMOS_BUILD)/moe_bwd_dx: $(BWD_SRC) $(CHECK_OBJ) $(COMM_OBJ) $(KERN_OBJ)
	@mkdir -p $(KOSMOS_BUILD)/obj
	rm -f $@
	$(HIPCXX) $(MOE_FLAGS) -c $(KOSMOS_ROOT)/bench/moe_bwd/src/moe_bwd_dx.cu -o $(KOSMOS_BUILD)/obj/moe_bwd_dx.o
	$(HIPCXX) -o $@ $(KOSMOS_BUILD)/obj/moe_bwd_dx.o $(KERN_OBJ) $(COMM_OBJ) \
	    $(CHECK_OBJ) $(MOE_LIBS) $(CHECK_LIBS)

$(KOSMOS_BUILD)/moe_train: $(TRAIN_SRC) $(CHECK_OBJ) $(COMM_OBJ) $(KERN_OBJ)
	@mkdir -p $(KOSMOS_BUILD)/obj
	rm -f $@
	$(HIPCXX) $(MOE_FLAGS) -c $(KOSMOS_ROOT)/bench/moe_train/src/moe_train.cu -o $(KOSMOS_BUILD)/obj/moe_train.o
	$(HIPCXX) -o $@ $(KOSMOS_BUILD)/obj/moe_train.o $(KERN_OBJ) $(COMM_OBJ) \
	    $(CHECK_OBJ) $(MOE_LIBS) $(CHECK_LIBS)

$(KOSMOS_BUILD)/moe_bench: $(BENCH_SRC)/moe_bench.cu $(MOE_LIB_SRC) $(BENCH_COMMON) $(CHECK_OBJ) $(COMM_OBJ) \
                           $(KERN_OBJ) $(FWD_PART_OBJ) $(BWD_PART_OBJ)
	@mkdir -p $(KOSMOS_BUILD)/obj
	rm -f $@
	$(HIPCXX) $(MOE_FLAGS) -c $< -o $(KOSMOS_BUILD)/obj/moe_bench.o
	$(HIPCXX) -o $@ $(KOSMOS_BUILD)/obj/moe_bench.o $(KERN_OBJ) $(FWD_PART_OBJ) \
	  $(BWD_PART_OBJ) $(COMM_OBJ) $(CHECK_OBJ) $(MOE_LIBS) $(CHECK_LIBS)

define PART_RULES
$(KOSMOS_BUILD)/obj/moe_fwd_$(1).o: $(BENCH_SRC)/moe_fwd_part.cu $(MOE_LIB_SRC)
	@mkdir -p $(KOSMOS_BUILD)/obj
	$(HIPCXX) $(MOE_FLAGS) -DMOE_FWD_PART_MX=$(2) -DMOE_FWD_PART_V=$(3) -c $$< -o $$@

$(KOSMOS_BUILD)/obj/moe_bwd_$(1)%.o: $(BENCH_SRC)/moe_bwd_part.cu $(MOE_LIB_SRC)
	@mkdir -p $(KOSMOS_BUILD)/obj
	$(HIPCXX) $(MOE_FLAGS) -DMOE_BWD_PART=$$* -DMOE_BWD_PART_MX=$(2) -DMOE_BWD_PART_V=$(3) -c $$< -o $$@

$(KOSMOS_BUILD)/obj/moe_bwd_c10$(1)%.o: $(BENCH_SRC)/moe_bwd_part.cu $(MOE_LIB_SRC)
	@mkdir -p $(KOSMOS_BUILD)/obj
	$(HIPCXX) $(MOE_FLAGS) -DMOE_BWD_PART=$$* -DMOE_BWD_PART_MX=$(2) -DMOE_BWD_PART_V=$(3) -DMOE_ACT_CLAMP10=1 \
	  -c $$< -o $$@
endef
$(foreach v,$(PART_VARIANTS),$(eval $(call PART_RULES,$(call pv,$(v),1),$(call pv,$(v),2),$(call pv,$(v),3))))

$(KOSMOS_BUILD)/obj/moe.o: $(KOSMOS_ROOT)/src/moe/moe.cu $(MOE_LIB_SRC)
	@mkdir -p $(KOSMOS_BUILD)/obj
	$(HIPCXX) $(MOE_LIB_FLAGS) -c $< -o $@

$(KOSMOS_BUILD)/obj/moe_plain.o: $(KOSMOS_ROOT)/src/moe/moe_plain.cu $(MOE_LIB_SRC)
	@mkdir -p $(KOSMOS_BUILD)/obj
	$(HIPCXX) $(MOE_LIB_FLAGS) -c $< -o $@

$(KOSMOS_BUILD)/obj/moe_train_api.o: $(KOSMOS_ROOT)/src/moe/moe_train_api.cu $(MOE_LIB_SRC)
	@mkdir -p $(KOSMOS_BUILD)/obj
	$(HIPCXX) $(MOE_LIB_FLAGS) -c $< -o $@

$(KOSMOS_BUILD)/obj/%_c10.o: $(KOSMOS_ROOT)/src/moe/%.cu $(MOE_LIB_SRC)
	@mkdir -p $(KOSMOS_BUILD)/obj
	$(HIPCXX) $(MOE_LIB_FLAGS) -DMOE_ACT_CLAMP10=1 -c $< -o $@

define ROUTER_SCORE_RULES
$(KOSMOS_BUILD)/obj/%_s$(1).o: $(KOSMOS_ROOT)/src/moe/%.cu $(MOE_LIB_SRC)
	@mkdir -p $(KOSMOS_BUILD)/obj
	$(HIPCXX) $$(MOE_LIB_FLAGS) -DMOE_ROUTER_SCORE=$(1) -c $$< -o $$@

$(KOSMOS_BUILD)/obj/%_c10_s$(1).o: $(KOSMOS_ROOT)/src/moe/%.cu $(MOE_LIB_SRC)
	@mkdir -p $(KOSMOS_BUILD)/obj
	$(HIPCXX) $$(MOE_LIB_FLAGS) -DMOE_ACT_CLAMP10=1 -DMOE_ROUTER_SCORE=$(1) -c $$< -o $$@
endef
$(foreach n,$(ROUTER_SCORES),$(eval $(call ROUTER_SCORE_RULES,$(n))))

$(KOSMOS_BUILD)/obj/moe_shared.o: $(KOSMOS_ROOT)/src/moe/moe_shared.cu $(MOE_LIB_SRC)
	@mkdir -p $(KOSMOS_BUILD)/obj
	$(HIPCXX) $(MOE_LIB_FLAGS) -c $< -o $@

$(KOSMOS_BUILD)/obj/moe_shared_router.o: $(KOSMOS_ROOT)/src/moe/moe_shared_router.cu $(MOE_LIB_SRC)
	@mkdir -p $(KOSMOS_BUILD)/obj
	$(HIPCXX) $(MOE_LIB_FLAGS) -c $< -o $@

$(KOSMOS_BUILD)/obj/%_bwdg.o: $(KOSMOS_ROOT)/src/moe/%_bwdg.cu $(MOE_LIB_SRC)
	@mkdir -p $(KOSMOS_BUILD)/obj
	$(HIPCXX) $(MOE_LIB_FLAGS) -c $< -o $@

$(filter $(KOSMOS_BUILD)/obj/moe_shared_bwdg%,$(SHARED_OBJ)): $(KOSMOS_ROOT)/src/moe/moe_shared.cu
$(filter $(KOSMOS_BUILD)/obj/moe_shared_router_bwdg%,$(ROUTER_OBJ)): $(KOSMOS_ROOT)/src/moe/moe_shared_router.cu
$(filter $(KOSMOS_BUILD)/obj/moe_mx_ks%,$(MX_OBJ)): $(KOSMOS_ROOT)/src/moe/moe_mx.cu
$(filter $(KOSMOS_BUILD)/obj/moe_mx_q8_ks%,$(MX_OBJ)): $(KOSMOS_ROOT)/src/moe/moe_mx_q8.cu
$(filter $(KOSMOS_BUILD)/obj/moe_mx_shared_%,$(MX_OBJ)): $(KOSMOS_ROOT)/src/moe/moe_mx_shared.cu
$(filter $(KOSMOS_BUILD)/obj/moe_mx_router_% $(KOSMOS_BUILD)/obj/moe_mx_shared_router%,$(ROUTER_OBJ)): \
  $(KOSMOS_ROOT)/src/moe/moe_mx_router.cu

$(KOSMOS_BUILD)/obj/moe_router.o: $(KOSMOS_ROOT)/src/moe/moe_router.cu $(MOE_LIB_SRC)
	@mkdir -p $(KOSMOS_BUILD)/obj
	$(HIPCXX) $(MOE_LIB_FLAGS) -c $< -o $@

$(KOSMOS_BUILD)/obj/moe_mx.o: $(KOSMOS_ROOT)/src/moe/moe_mx.cu $(MOE_LIB_SRC)
	@mkdir -p $(KOSMOS_BUILD)/obj
	$(HIPCXX) $(MOE_LIB_FLAGS) -c $< -o $@

$(KOSMOS_BUILD)/obj/moe_mx_q8.o: $(KOSMOS_ROOT)/src/moe/moe_mx_q8.cu $(MOE_LIB_SRC)
	@mkdir -p $(KOSMOS_BUILD)/obj
	$(HIPCXX) $(MOE_LIB_FLAGS) -c $< -o $@

$(KOSMOS_BUILD)/obj/moe_mx_shared.o: $(KOSMOS_ROOT)/src/moe/moe_mx_shared.cu $(MOE_LIB_SRC)
	@mkdir -p $(KOSMOS_BUILD)/obj
	$(HIPCXX) $(MOE_LIB_FLAGS) -c $< -o $@

$(KOSMOS_BUILD)/obj/moe_mx_shared_q8.o: $(KOSMOS_ROOT)/src/moe/moe_mx_shared_q8.cu $(MOE_LIB_SRC)
	@mkdir -p $(KOSMOS_BUILD)/obj
	$(HIPCXX) $(MOE_LIB_FLAGS) -c $< -o $@

$(foreach t,moe_mx_router moe_mx_router_q8 moe_mx_shared_router moe_mx_shared_router_q8,$(KOSMOS_BUILD)/obj/$(t).o): \
  $(KOSMOS_BUILD)/obj/%.o: $(KOSMOS_ROOT)/src/moe/%.cu $(MOE_LIB_SRC)
	@mkdir -p $(KOSMOS_BUILD)/obj
	$(HIPCXX) $(MOE_LIB_FLAGS) -c $< -o $@

$(KOSMOS_BUILD)/obj/%_ks.o: $(KOSMOS_ROOT)/src/moe/%_ks.cu $(MOE_LIB_SRC)
	@mkdir -p $(KOSMOS_BUILD)/obj
	$(HIPCXX) $(MOE_LIB_FLAGS) -c $< -o $@

$(KOSMOS_BUILD)/obj/%_ks_pin.o: $(KOSMOS_ROOT)/src/moe/%_ks_pin.cu $(MOE_LIB_SRC)
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

$(KOSMOS_BUILD)/gen_routing_ref: $(KOSMOS_ROOT)/bench/megamoe/gen_routing_ref.cpp $(KOSMOS_ROOT)/bench/common/moe_shapes.h
	@mkdir -p $(KOSMOS_BUILD)
	g++ -O2 -std=c++17 -o $@ $<

$(KOSMOS_BUILD)/kosmos_report: $(KOSMOS_ROOT)/bench/report.cpp $(KOSMOS_ROOT)/bench/common/moe_shapes.h
	@mkdir -p $(KOSMOS_BUILD)
	$(HOSTCXX) -std=c++17 -O2 -o $@ $<

clean:
	rm -rf $(KOSMOS_BUILD)

.PHONY: all moe_fwd moe_bwd moe_train moe_bench moe tpsp overlap routing report python clean
