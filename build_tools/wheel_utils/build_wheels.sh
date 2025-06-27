# This file was modified for portability to AMDGPU
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

set -e

PLATFORM=${1:-manylinux_2_28_x86_64}
BUILD_METAPACKAGE=${2:-true}
BUILD_COMMON=${3:-true}
BUILD_PYTORCH=${4:-true}
BUILD_JAX=${5:-true}

export NVTE_RELEASE_BUILD=1
export TARGET_BRANCH=${TARGET_BRANCH:-}
mkdir -p /wheelhouse/logs

# Generate wheels for common library.
git config --global --add safe.directory /TransformerEngine
cd /TransformerEngine

#If there is default Python installation, use it
PYTHON=`which python || true`
if [ -z "$PYTHON" ]; then
        PYBINDIR=/opt/python/cp310-cp310/bin/
else
        PYBINDIR="" #python bindir is already in PATHs
fi

ROCM_BUILD=`${PYBINDIR}python -c "import build_tools.utils as u; print(int(u.rocm_build()))"`

if [ "$ROCM_BUILD" = "1" ]; then
        git pull
fi
git checkout $TARGET_BRANCH
git submodule update --init --recursive

if $BUILD_METAPACKAGE ; then
        cd /TransformerEngine
        if [ "$ROCM_BUILD" != "1" ]; then
                PYBINDIR=/opt/python/cp310-cp310/bin/
        fi
        NVTE_BUILD_METAPACKAGE=1 ${PYBINDIR}python setup.py bdist_wheel 2>&1 | tee /wheelhouse/logs/metapackage.txt
        mv dist/* /wheelhouse/
fi

if $BUILD_COMMON ; then
        VERSION=`cat build_tools/VERSION.txt`
        WHL_BASE="transformer_engine-${VERSION}"
        if [ "$ROCM_BUILD" = "1" ]; then
                TE_CUDA_VERS="rocm"
                ${PYBINDIR}pip install ninja
        else
                TE_CUDA_VERS="cu12"
                PYBINDIR=/opt/python/cp38-cp38/bin/
        fi

        # Create the wheel.
        ${PYBINDIR}python setup.py bdist_wheel --verbose --python-tag=py3 --plat-name=$PLATFORM 2>&1 | tee /wheelhouse/logs/common.txt

        # Repack the wheel for cuda specific package, i.e. cu12.
        ${PYBINDIR}wheel unpack dist/*
        # From python 3.10 to 3.11, the package name delimiter in metadata got changed from - (hyphen) to _ (underscore).
        sed -i "s/Name: transformer-engine/Name: transformer-engine-${TE_CUDA_VERS}/g" "transformer_engine-${VERSION}/transformer_engine-${VERSION}.dist-info/METADATA"
        sed -i "s/Name: transformer_engine/Name: transformer_engine_${TE_CUDA_VERS}/g" "transformer_engine-${VERSION}/transformer_engine-${VERSION}.dist-info/METADATA"
        mv "${WHL_BASE}/${WHL_BASE}.dist-info" "${WHL_BASE}/transformer_engine_${TE_CUDA_VERS}-${VERSION}.dist-info"
        ${PYBINDIR}wheel pack ${WHL_BASE}

        # Rename the wheel to make it python version agnostic.
        whl_name=$(basename dist/*)
        IFS='-' read -ra whl_parts <<< "$whl_name"
        whl_name_target="${whl_parts[0]}_${TE_CUDA_VERS}-${whl_parts[1]}-py3-none-${whl_parts[4]}"
        rm -rf $WHL_BASE dist
        mv *.whl /wheelhouse/"$whl_name_target"
fi

if $BUILD_PYTORCH ; then
	cd /TransformerEngine/transformer_engine/pytorch
	if [ "$ROCM_BUILD" = "1" ]; then
                ${PYBINDIR}pip install torch --index-url https://download.pytorch.org/whl/rocm6.3
        else
                PYBINDIR=/opt/python/cp38-cp38/bin/
                ${PYBINDIR}pip install torch
        fi
        ${PYBINDIR}python setup.py sdist 2>&1 | tee /wheelhouse/logs/torch.txt
	cp dist/* /wheelhouse/
fi

if $BUILD_JAX ; then
	cd /TransformerEngine/transformer_engine/jax
	if [ "$ROCM_BUILD" = "1" ]; then
                ${PYBINDIR}pip install jax
        else
                PYBINDIR=/opt/python/cp310-cp310/bin/
                ${PYBINDIR}pip install "jax[cuda12_local]" jaxlib
        fi
	${PYBINDIR}python setup.py sdist 2>&1 | tee /wheelhouse/logs/jax.txt
	cp dist/* /wheelhouse/
fi
