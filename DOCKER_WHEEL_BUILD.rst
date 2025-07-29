..
    Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
    See LICENSE for license information.

|License|



Building manylinux TransformerEngine wheels using Docker
********************************************************

Step 1: Create Builder Image
============================

Obtain Manylinux_2_28_x86_64 based docker image with ROCm installed on it

In the TransformerEngine source tree at build_tool/wheel_utils run:
.. code-block:: bash
  docker build -f Dockerfile.rocm.manylinux.x86 --build-arg BASE_IMAGE={base_image_tag} -t {builder_image_tag} .

This command uses the following parameters:

{base_image_tag} - image from step 1.1

{builder_image_tag} - tag for generated image


TARGET_BRANCH - parameter is branch name; if specified, use named branch instead of default one

GPU_TARGETS - parameter is semicolon separated list of AMD GPU architectures to build TE for; default is gfx942

Step 2: Run Builder
===================

After the image is built, run the following command to create and run a container from the builder image:

.. code-block:: bash
  docker run --rm -v {host_wheel_destination}:/wheelhouse {builder_image_tag}

This command generates TE wheels and puts them to {host_wheel_destination}.

Four packages are generated: 2 wheels and 2 sdist packages:

transformer_engine-{version}-py3-none-any.whl  - meta package that contains dependencies and declares JAX and Pytorch extras

transformer_engine_rocm-{version}-py3-none-manylinux_2_28_x86_64.whl - TE library core package

transformer_engine_jax-{version}.tar.gz - JAX extra that should only be installed on targets with JAX framework; builds during installation

transformer_engine_torch-{version}.tar.gz - Pytorch extra that should only be installed on targets with Pytorch framework; builds during installation

Step 3: Install TransformerEngine
=================================

Take the wheel and tarball archives created by step 2, and run
.. code-block:: bash
    pip install /wheelhouse/transformer_engine*

This will build the target-specific code on the target machine

Note that it may be necessary to include the flag --no-build-isolation for this installation to be successful

.. |License| image:: https://img.shields.io/badge/License-Apache%202.0-blue.svg
   :target: https://opensource.org/licenses/Apache-2.0
