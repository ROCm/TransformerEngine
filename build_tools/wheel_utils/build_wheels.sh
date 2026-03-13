# This file was modified for portability to AMDGPU
# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

set -e

PLATFORM=${1:-manylinux_2_28_x86_64}
BUILD_METAPACKAGE=${2:-true}
BUILD_COMMON=${3:-true}
BUILD_PYTORCH=${4:-true}
BUILD_JAX=${5:-true}
CUDA_MAJOR=${6:-12}

export NVTE_RELEASE_BUILD=1
export PIP_CONSTRAINT=""
export TARGET_BRANCH=${TARGET_BRANCH:-}
mkdir -p /wheelhouse/logs

# Generate wheels for common library.
git config --global --add safe.directory /TransformerEngine
cd /TransformerEngine

#If there is default Python installation, use it
PYTHON=`which python || true`
if [ -z "$PYTHON" ]; then
        PYBINDIR=/opt/python/cp310-cp310/bin/
        #hipify expects python in PATH, also ninja may be installed to python bindir
        PATH="$PYBINDIR:$PATH"
else
        PYBINDIR="" #python bindir is already in PATHs
fi

ROCM_BUILD=$(${PYBINDIR}python -c "import build_tools.utils as u; print('true' if u.rocm_build() else 'false')")

if [ "$LOCAL_TREE_BUILD" != "1" ]; then
        if $ROCM_BUILD ; then
                git pull
        fi
        git checkout $TARGET_BRANCH
        git submodule update --init --recursive
else
        git submodule status --recursive | cut -d' ' -f3 | xargs -l -P1 -I_SUB_ git config --global --add safe.directory /TransformerEngine/_SUB_
fi

if $ROCM_BUILD ; then
  #dataclasses, psutil are needed for AITER
  ${PYBINDIR}pip install pybind11[global] ninja setuptools dataclasses psutil
else
  PYBINDIR=/opt/python/cp310-cp310/bin/
  /opt/python/cp310-cp310/bin/pip install cmake pybind11[global] ninja setuptools wheel
fi

if $BUILD_METAPACKAGE ; then
        cd /TransformerEngine
        NVTE_BUILD_METAPACKAGE=1 ${PYBINDIR}python setup.py bdist_wheel 2>&1 | tee /wheelhouse/logs/metapackage.txt
        mv dist/* /wheelhouse/
fi

if $BUILD_COMMON -a $ROCM_BUILD ; then
        # Create the wheel.
        ${PYBINDIR}python setup.py bdist_wheel --verbose --plat-name=$PLATFORM 2>&1 | tee /wheelhouse/logs/common.txt

        # Rename the wheel to make it python version agnostic.
        whl_name=$(basename dist/*)
        IFS='-' read -ra whl_parts <<< "$whl_name"
        whl_name_target="${whl_parts[0]}-${whl_parts[1]}-py3-none-${whl_parts[4]}"
        mv dist/*.whl /wheelhouse/"$whl_name_target"
elif $BUILD_COMMON ; then
        VERSION=`cat build_tools/VERSION.txt`
        WHL_BASE="transformer_engine-${VERSION}"

        # Create the wheel.
        /opt/python/cp310-cp310/bin/python setup.py bdist_wheel --verbose --python-tag=py3 --plat-name=$PLATFORM 2>&1 | tee /wheelhouse/logs/common.txt

        # Repack the wheel for specific cuda version.
        /opt/python/cp310-cp310/bin/wheel unpack dist/*
        # From python 3.10 to 3.11, the package name delimiter in metadata got changed from - (hyphen) to _ (underscore).
        sed -i "s/Name: transformer-engine/Name: transformer-engine-cu${CUDA_MAJOR}/g" "transformer_engine-${VERSION}/transformer_engine-${VERSION}.dist-info/METADATA"
        sed -i "s/Name: transformer_engine/Name: transformer_engine_cu${CUDA_MAJOR}/g" "transformer_engine-${VERSION}/transformer_engine-${VERSION}.dist-info/METADATA"
        mv "${WHL_BASE}/${WHL_BASE}.dist-info" "${WHL_BASE}/transformer_engine_cu${CUDA_MAJOR}-${VERSION}.dist-info"
        /opt/python/cp310-cp310/bin/wheel pack ${WHL_BASE}

        # Rename the wheel to make it python version agnostic.
        whl_name=$(basename dist/*)
        IFS='-' read -ra whl_parts <<< "$whl_name"
        whl_name_target="${whl_parts[0]}_cu${CUDA_MAJOR}-${whl_parts[1]}-py3-none-${whl_parts[4]}"
        rm -rf $WHL_BASE dist
        mv *.whl /wheelhouse/"$whl_name_target"
fi

if $BUILD_PYTORCH -a $ROCM_BUILD ; then
        cd /TransformerEngine/transformer_engine/pytorch
        #Only need torch for creating sdist, install CPU version to avoid installing CUDA/ROCm dependencies
        ${PYBINDIR}pip install torch --index-url https://download.pytorch.org/whl/cpu
        ${PYBINDIR}python setup.py sdist 2>&1 | tee /wheelhouse/logs/torch.txt
        mv dist/* /wheelhouse/
elif $BUILD_PYTORCH ; then
	cd /TransformerEngine/transformer_engine/pytorch
	/opt/python/cp310-cp310/bin/pip install torch
	/opt/python/cp310-cp310/bin/python setup.py sdist 2>&1 | tee /wheelhouse/logs/torch.txt
	cp dist/* /wheelhouse/
fi

if $BUILD_JAX -a $ROCM_BUILD ; then
        cd /TransformerEngine/transformer_engine/jax
        ${PYBINDIR}pip install jax
        ${PYBINDIR}python setup.py sdist 2>&1 | tee /wheelhouse/logs/jax.txt
        mv dist/* /wheelhouse/
elif $BUILD_JAX ; then
	cd /TransformerEngine/transformer_engine/jax
	/opt/python/cp310-cp310/bin/pip install "jax[cuda${CUDA_MAJOR}_local]" jaxlib
	/opt/python/cp310-cp310/bin/python setup.py sdist 2>&1 | tee /wheelhouse/logs/jax.txt
	cp dist/* /wheelhouse/
fi
