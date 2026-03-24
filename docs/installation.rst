..
    This file was modified to include portability information to AMDGPU.
    Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

Installation on AMD GPUs
========================

Prerequisites
-------------
1. `AMD Instinct GPU <https://www.amd.com/en/products/accelerators/instinct.html>`__. Other GPUs are not supported while they can still work.
2. Linux x86_64
3. `ROCm stack <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/index.html>`__. For ROCm TheRock (ROCm 7.11 and newer), install amdrocm-core-sdk* package

Additional Prerequisites
^^^^^^^^^^^^^^^^^^^^^^^^

1. [For PyTorch support] `Pytorch <https://rocm.docs.amd.com/projects/install-on-linux/en/develop/install/3rd-party/pytorch-install.html>`__
2. [For JAX support] `JAX <https://rocm.docs.amd.com/projects/install-on-linux/en/develop/install/3rd-party/jax-install.html>`__

If HIP compiler complains it cannot detect the platform set `HIP_PLATFORM=amd` in the environment.
If ROCm is installed in a non-standard location, set `ROCM_PATH` to the root of the ROCm installation in the environment, e.g. `ROCM_PATH=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel` and additionally set the following environment variables:

- HIP_DEVICE_LIB_PATH=$ROCM_PATH/llvm/amdgcn/bitcode/
- CMAKE_PREFIX_PATH=$ROCM_PATH/lib/cmake/

pip - from wheels
-----------------

Transformer Engine for ROCm 7.0 and newer can be installed from `Manylinux wheels <https://repo.radeon.com/rocm/manylinux>`__. Four files related to Transformer Engine can be found there:

- transformer_engine-\*-py3-none-any.whl - the main TE pure Python metapackage.
- transformer_engine_rocm-\*-py3-none-manylinux_2_28_x86_64.whl - the core library package.
- transformer_engine_jax-\*.tar.gz - source tarball (sdist) for the JAX extension.
- transformer_engine_torch-\*.tar.gz - source tarball (sdist) for the Pytorch extension.

Below are the example commands to download and install the wheels published with ROCm 7.2. They install both Pytorch and JAX extensions on the system where both frameworks are installed.

.. code-block:: bash

  wget -r -l1 -nd -A 'transformer_engine*' https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/
  pip install ./transformer_engine* --no-build-isolation

Starting from version 2.10, core library wheel can be installed by itself or as an extension for TE Python metapackage.

Example of installing ROCm core library wheel without any framework extensions:

.. code-block:: bash

  pip install transformer_engine_rocm7-2.10.0-py3-none-manylinux_2_28_x86_64.whl

Additionally install framework extensions using ROCm package name and pip extras syntax.

.. code-block:: bash

  pip install --find-links <url/or/local/directory/with/te/wheels/> transformer_engine_rocm7[pytorch,jax] --no-build-isolation

Installing the common library and frameworks extensions as extras for TE Python metapackage

.. code-block:: bash

  pip install ./transformer_engine-2.10.0-py3-none-any.whl[rocm7,rocm_pytorch,rocm_jax] --no-build-isolation

It is not recommended to install TE Python metapackage using just package name transformer_engine because of possible installing of the NVIDIA GPU version. It is recommended to use either transformer_engine_rocm7 or wheel file name to make sure the correct common library is installed.


Installation from source
^^^^^^^^^^^^^^^^^^^^^^^^^^
Execute the following commands to install Transformer Engine from source:

.. code-block:: bash

  # Clone repository, checkout stable branch, clone submodules
  git clone --recursive https://github.com/ROCm/TransformerEngine.git

  cd TransformerEngine
  export NVTE_FRAMEWORK=pytorch,jax     # Optionally set framework(s)
  export NVTE_ROCM_ARCH="gfx942;gfx950" # Optionally set target GPU architectures; gfx942 for MI300/MI325, and gfx950 for MI350
  export NVTE_USE_ROCM=1                # Optionally force building for ROCm, useful when both ROCm and CUDA build environments are installed. If set to 0, it will force building for CUDA.
  pip3 install --no-build-isolation .   # Build and install

Or instead of immediate installation, create wheel file to install it later:

.. code-block:: bash

  pip wheel . --no-build-isolation
  pip3 install ./transformer_engine-*.whl

If the Git repository has already been cloned, make sure the submodules do not have any local changes, otherwise the build will try to reset them unless `NVTE_SKIP_SUBMODULE_CHECKS_DURING_BUILD=1` is set.

Extra dependencies for testing can be installed by setting the "test" option:

.. code-block:: bash

  pip3 install --no-build-isolation .[test]

To build the C++ extensions with debug symbols, e.g. with the `-g` flag:

.. code-block:: bash

  NVTE_BUILD_DEBUG=1 pip3 install --no-build-isolation .


Switching between Installation from Source and Installation from Wheels
-----------------------------------------------------------------------
Sometimes, issues might occur when installing from source on a system where a previous installation with wheels, or vice versa. It is safe to uninstall TE first before 
switching between installing from source and installing from wheels. Here is the example command:

.. code-block:: bash

  # The package name pattern might be transformer_engine or transformer-engine depending on Setuptools version
  pip list | grep transformer.engine | cut -f' ' -d1 | xargs pip uninstall -y


Installation on NVIDIA GPUs
===========================

Prerequisites
-------------
.. |driver link| replace:: NVIDIA Driver
.. _driver link: https://www.nvidia.com/drivers

1. Linux x86_64
2. `CUDA 12.1+ (12.8+ for Blackwell support) <https://developer.nvidia.com/cuda-downloads>`__
3. |driver link|_ supporting CUDA 12.1 or later.
4. `cuDNN 9.3 <https://developer.nvidia.com/cudnn>`__ or later.

If the CUDA Toolkit headers are not available at runtime in a standard
installation path, e.g. within `CUDA_HOME`, set
`NVTE_CUDA_INCLUDE_PATH` in the environment.

Transformer Engine in NGC Containers
------------------------------------

Transformer Engine library is preinstalled in the PyTorch container in versions 22.09 and later
on `NVIDIA GPU Cloud <https://ngc.nvidia.com>`_.


pip - from PyPI
---------------

Transformer Engine can be directly installed from `our PyPI <https://pypi.org/project/transformer-engine/>`_, e.g.

.. code-block:: bash

    pip3 install --no-build-isolation transformer_engine[pytorch]

To obtain the necessary Python bindings for Transformer Engine, the frameworks needed must be explicitly specified as extra dependencies in a comma-separated list (e.g. [jax,pytorch]). Transformer Engine ships wheels for the core library. Source distributions are shipped for the JAX and PyTorch extensions.

The core package from Transformer Engine (without any framework extensions) can be installed via:

.. code-block:: bash

    pip3 install transformer_engine[core]

By default, this will install the core library compiled for CUDA 12. The cuda major version can be specified by modified the extra dependency to `core_cu12` or `core_cu13`.

pip - from GitHub
-----------------

Additional Prerequisites
^^^^^^^^^^^^^^^^^^^^^^^^

1. [For PyTorch support] `PyTorch <https://pytorch.org/>`__ with GPU support.
2. [For JAX support] `JAX <https://github.com/google/jax/>`__ with GPU support, version >= 0.4.7.

Installation (stable release)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Execute the following command to install the latest stable version of Transformer Engine:

.. code-block:: bash

  pip3 install --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@stable

This will automatically detect if any supported deep learning frameworks are installed and build Transformer Engine support for them. To explicitly specify frameworks, set the environment variable `NVTE_FRAMEWORK` to a comma-separated list (e.g. `NVTE_FRAMEWORK=jax,pytorch`).

Installation (development build)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. warning::

   While the development build of Transformer Engine could contain new features not available in
   the official build yet, it is not supported and so its usage is not recommended for general
   use.

Execute the following command to install the latest development build of Transformer Engine:

.. code-block:: bash

  pip3 install --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@main

This will automatically detect if any supported deep learning frameworks are installed and build Transformer Engine support for them. To explicitly specify frameworks, set the environment variable `NVTE_FRAMEWORK` to a comma-separated list (e.g. `NVTE_FRAMEWORK=jax,pytorch`). To only build the framework-agnostic C++ API, set `NVTE_FRAMEWORK=none`.

In order to install a specific PR, execute (after changing NNN to the PR number):

.. code-block:: bash

  pip3 install --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@refs/pull/NNN/merge


Installation (from source)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Execute the following commands to install Transformer Engine from source:

.. code-block:: bash

  # Clone repository, checkout stable branch, clone submodules
  git clone --branch stable --recursive https://github.com/NVIDIA/TransformerEngine.git

  cd TransformerEngine
  export NVTE_FRAMEWORK=pytorch         # Optionally set framework
  pip3 install --no-build-isolation .   # Build and install

If the Git repository has already been cloned, make sure to also clone the submodules:

.. code-block:: bash

  git submodule update --init --recursive

Extra dependencies for testing can be installed by setting the "test" option:

.. code-block:: bash

  pip3 install --no-build-isolation .[test]

To build the C++ extensions with debug symbols, e.g. with the `-g` flag:

.. code-block:: bash

  NVTE_BUILD_DEBUG=1 pip3 install --no-build-isolation .

.. include:: ../README.rst
   :start-after: troubleshooting-begin-marker-do-not-remove
   :end-before: troubleshooting-end-marker-do-not-remove
