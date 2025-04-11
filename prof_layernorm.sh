#!/usr/bin/env bash

shapes=(
    "2048 12288"
    "768 1024"
    "256 65536"
    "128 6144"
    "64 2304"
    "229 541"
    "71 3571"
    "29 17389"
    "76800 1600"
)

py_src='tests/pytorch/triton_kernels/test_layernorm_triton.py'

function prof_triton_fwd() {
    for we in 1 2 4; do  # waves_per_eu
	for wa in 4 8 16; do  # num_warps
	    for shape in "${shapes[@]}"; do
		read -r m n <<< "${shape}"

		prof_kernel.sh \
		    -r _layernorm_fwd_triton \
		    -o "prof_triton_fwd/we${we}_wa${wa}/${m}_${n}" \
		    -- python "${py_src}" triton "${m}" "${n}" fwd "${we}" "${wa}"
	    done
	done
    done
}

function prof_te_fwd() {
    for shape in "${shapes[@]}"; do
	read -r m n <<< "${shape}"

	# Kernel name can be 'ln_fwd_tuned_kernel' or 'ln_fwd_general_kernel'.
	prof_kernel.sh \
	    -r 'ln_fwd_\(tuned\|general\)_kernel' \
	    -o "prof_te_fwd/${m}_${n}" \
	    -- python "${py_src}" te "${m}" "${n}" fwd
    done
}

function prof_triton_bwd() {
    we=2
    wa=8

    for shape in "${shapes[@]}"; do
	read -r m n <<< "${shape}"

	prof_kernel.sh \
	    -r '_layernorm_bwd_\(dx_fused\|dwdb\)_triton' \
	    -o "prof_triton_bwd/${m}_${n}" \
	    -- python "${py_src}" triton "${m}" "${n}" bwd "${we}" "${wa}"
    done
}

### ENTRY POINT

# prof_triton_fwd
# prof_te_fwd
prof_triton_bwd
# TODO: prof_te_bwd
