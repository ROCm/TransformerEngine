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

export NVTE_RELEASE_BUILD=1
export TARGET_BRANCH=${TARGET_BRANCH:-}

WHEEL_ROOT=${WHEEL_ROOT:-/wheelhouse}
mkdir -p $WHEEL_ROOT/logs

# Generate wheels for common library.
TE_ROOT=${TE_ROOT:-/TransformerEngine}
cd $TE_ROOT

#If there is default Python installation, use it
PYTHON=`which python || true`
if [ -z "$PYTHON" ]; then
        PYBINDIR=/opt/python/cp310-cp310/bin/
else
        PYBINDIR="" #python bindir is already in PATHs
fi

ROCM_BUILD=`${PYBINDIR}python -c "import build_tools.utils as u; print(int(u.rocm_build()))"`
if [ "$ROCM_BUILD" = "1" ]; then
        ROCM_BUILD=true
else
        ROCM_BUILD=false
fi

if [ "$LOCAL_TREE_BUILD" != "1" ]; then
        git config --global --add safe.directory $TE_ROOT
        if [ "$SKIP_REPO_UPDATE" = "1" ]; then
                git submodule status --recursive | cut -d' ' -f3 | xargs -l -P1 -I_SUB_ git config --global --add safe.directory $TE_ROOT/_SUB_
        else
                if [ $ROCM_BUILD ]; then
                        git pull
                fi
                git checkout $TARGET_BRANCH
                git submodule update --init --recursive
        fi
fi

# Install deps
if [ $ROCM_BUILD ]; then
  ${PYBINDIR}pip install setuptools wheel pybind11[global] ninja
else
  ${PYBINDIR}pip install cmake pybind11[global] ninja
fi

if $BUILD_METAPACKAGE ; then
        cd $TE_ROOT
        if [ ! $ROCM_BUILD ]; then
                PYBINDIR=/opt/python/cp310-cp310/bin/
        fi
        NVTE_BUILD_METAPACKAGE=1 ${PYBINDIR}python setup.py bdist_wheel 2>&1 | tee $WHEEL_ROOT/logs/metapackage.txt
        mv dist/* $WHEEL_ROOT/
fi

if $BUILD_COMMON -a $ROCM_BUILD; then
        VERSION=`cat build_tools/VERSION.txt`
        WHL_BASE="transformer_engine_rocm-${VERSION}"
        #dataclasses, psutil are needed for AITER
        ${PYBINDIR}pip install dataclasses psutil
        #hipify expects python in PATH, also ninja may be installed to python bindir
        test -n "$PYBINDIR" && PATH="$PYBINDIR:$PATH" || true

        # Create the wheel.
        ${PYBINDIR}python setup.py bdist_wheel --verbose --python-tag=py3 --plat-name=$PLATFORM 2>&1 | tee $WHEEL_ROOT/logs/common.txt

        # Rename the wheel to make it python version agnostic.
        whl_name=$(basename dist/*)
        IFS='-' read -ra whl_parts <<< "$whl_name"
        whl_name_target="${whl_parts[0]}-${whl_parts[1]}-py3-none-${whl_parts[4]}"
        mv dist/*.whl $WHEEL_ROOT/"$whl_name_target"

elif $BUILD_COMMON; then
        VERSION=`cat build_tools/VERSION.txt`
        WHL_BASE="transformer_engine-${VERSION}"
        PYBINDIR=/opt/python/cp38-cp38/bin/

        # Create the wheel.
        ${PYBINDIR}python setup.py bdist_wheel --verbose --python-tag=py3 --plat-name=$PLATFORM 2>&1 | tee $WHEEL_ROOT/logs/common.txt

        # Repack the wheel for cuda specific package, i.e. cu12.
        ${PYBINDIR}wheel unpack dist/*
        # From python 3.10 to 3.11, the package name delimiter in metadata got changed from - (hyphen) to _ (underscore).
        sed -i "s/Name: transformer-engine/Name: transformer-engine-cu12/g" "transformer_engine-${VERSION}/transformer_engine-${VERSION}.dist-info/METADATA"
        sed -i "s/Name: transformer_engine/Name: transformer_engine_cu12/g" "transformer_engine-${VERSION}/transformer_engine-${VERSION}.dist-info/METADATA"
        mv "${WHL_BASE}/${WHL_BASE}.dist-info" "${WHL_BASE}/transformer_engine_cu12-${VERSION}.dist-info"
        ${PYBINDIR}wheel pack ${WHL_BASE}

        # Rename the wheel to make it python version agnostic.
        whl_name=$(basename dist/*)
        IFS='-' read -ra whl_parts <<< "$whl_name"
        whl_name_target="${whl_parts[0]}_cu12-${whl_parts[1]}-py3-none-${whl_parts[4]}"
        rm -rf $WHL_BASE dist
        mv *.whl $WHEEL_ROOT/"$whl_name_target"
fi

if $BUILD_PYTORCH ; then
  cd $TE_ROOT/transformer_engine/pytorch
  if [ $ROCM_BUILD ]; then
    ${PYBINDIR}pip install torch --index-url https://download.pytorch.org/whl/cpu
  else
    PYBINDIR=/opt/python/cp38-cp38/bin/
    ${PYBINDIR}pip install torch
  fi
  ${PYBINDIR}python setup.py sdist 2>&1 | tee $WHEEL_ROOT/logs/torch.txt
  cp dist/* $WHEEL_ROOT/
fi

if $BUILD_JAX ; then
  cd $TE_ROOT/transformer_engine/jax
  if [ $ROCM_BUILD ]; then
    ${PYBINDIR}pip install jax
  else
    PYBINDIR=/opt/python/cp310-cp310/bin/
    ${PYBINDIR}pip install "jax[cuda12_local]" jaxlib
  fi
  ${PYBINDIR}python setup.py sdist 2>&1 | tee $WHEEL_ROOT/logs/jax.txt
  cp dist/* $WHEEL_ROOT/
fi
