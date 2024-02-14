..
		This file was modified to include portability information to AMDGPU.

    Copyright (c) 2023-2024, Advanced Micro Devices, Inc. All rights reserved.

    Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

|License|

Transformer Engine On ROCm and AMDGPU
=====================================

This repository enables Transformer Engine (TE) on ROCm as a library to accelerate Transformer models on AMD GPUs, including using 8-bit floating point (FP8) precision on MI300 GPUs, to provide better performance with lower memory utilization in both training and inference. 
One of the missions is to provide an alternative to accelerate Transformer models that were previously run on NVIDIA GPUs like Hopper with best efforts to make the migration frictionless. 
Moreover, we add optimizations specific to AMD GPUs to get the best performance benefits out of AMD GPUs.

Feature Support Status
----------------------

* Activation, cast, fused softmax, layernorm, rmsnorm, transpose, HipRTC: fully supported
* GEMM: partially supported with following input/output types: (fp32/fp32), (fp16/fp16), (bf16/bf16), (fp8, bf8/fp16, bf16, fp32)
* Attention (Flash Attention, Fused Multihead Attention): not supported
* HipGraph, HipTX: partially supported

Installation
------------
Execute the following commands to install ROCm Transformer Engine from source on AMDGPUs:

.. code-block:: bash

  # Clone TE repo and submodules
  git clone --recursive https://github.com/ROCmSoftwarePlatform/TransformerEngine-private.git
  
  cd TransformerEngine-private
  export NVTE_FRAMEWORK=pytorch #optionally set framework, currently only support pytorch, jax, and tensorflow
  pip install .


The default installation above will use rocblas in GEMM computation. The hipBlasLt alternative can be selected by setting the environment variable `NVTE_USE_HIPBLASLT` before the `pip install` as:

.. code-block:: bash

  export NVTE_USE_HIPBLASLT=1

The hipBlasLt alternative has not yet supported all the GEMM configurations in the pytorch unit tests. When hipBlasLt is fully support, we will switch to hipBlasLt as the default path for GEMM computation.

Test
----

Framework Agnostic C++ library unittests
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After a successful Transformer Engine installation via `pip install`, execute the following commands to build and test the framework agnostic C++ library:

.. code-block:: bash

  cd tests/cpp
  cmake .
  make
  make test

Framework Integration pytests
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Pytorch

The following two Pytorch integration pytests are supported: 

.. code-block:: bash

  tests/pytorch/test_sanity.py
  tests/pytorch/test_numerics.py

Execute the following command to test them after a successfuly installation with Pytorch. 

.. code-block:: bash

  ROCBLAS_STREAM_ORDER_ALLOC=1 NVTE_FUSED_ATTN=0 NVTE_FLASH_ATTN=0 pytest tests/pytorch/<testname>

`ROCBLAS_STREAM_ORDER_ALLOC=1` can be dropped when the hipGraph feature is fully supported in Pytorch on AMDGPUs. 
The other environmental variables are required since our ROCm Transformer Engine has not supported fused attention or flash attention yet. 

Jax

All jax pytests except for test_fused_attn.py are supported. 

Examples
--------
Pytorch
^^^^^^^
MNIST with optional FP8

.. code-block:: bash
  
  cd examples/pytorch/mnist
  python main.py
  python main.py --use-te   # Linear layers from TransformerEngine
  python main.py --use-fp8  # FP8 + TransformerEngine for Linear layers

Sort with minGPT

.. code-block:: bash
  
  cd examples/pytorch/minGPT
  python gptSort.py --use-te # Linear and layernorm from TransformerEngine
  python gptSort.py --use-te --ln-mlp # In addition, use LayernormMLP from transformer engine
  python gptSort.py --use-te --ln-mlp --use-fp8 # In addition, use fp8

Jax
^^^
Flax

.. code-block:: python

  import jax
  import jax.numpy as jnp
  import transformer_engine.jax as te
  import transformer_engine.jax.flax as te_flax
  from transformer_engine.common import recipe

  BATCH = 32
  SEQLEN = 128
  HIDDEN = 1024

  # Initialize RNG and inputs.
  rng = jax.random.PRNGKey(0)
  init_rng, data_rng = jax.random.split(rng)
  inp = jax.random.normal(data_rng, [BATCH, SEQLEN, HIDDEN], jnp.float32)

  # Create an FP8 recipe. Note: All input args are optional.
  fp8_recipe = recipe.DelayedScaling(margin=0, interval=1, fp8_format=recipe.Format.HYBRID)

  # Enable autocasting for the forward pass
  with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
      model = te_flax.DenseGeneral(features=HIDDEN)

      def loss_fn(params, other_vars, inp):
        out = model.apply({'params':params, **other_vars}, inp)
        return jnp.mean(out)

      # Initialize models.
      variables = model.init(init_rng, inp)
      other_variables, params = variables.pop('params')

      # Construct the forward and backward function
      fwd_bwd_fn = jax.value_and_grad(loss_fn, argnums=(0, 1))

      for _ in range(10):
        loss, (param_grads, other_grads) = fwd_bwd_fn(params, other_variables, inp)
        # Update FP8 metas
        other_variables = te.update_fp8_metas(other_grads)

MNIST

.. code-block:: bash
  
  cd examples/jax/mnist
  python test_single_gpu_mnist.py # Use Flax to train MNIST with BF16 as usual
  python test_single_gpu_mnist.py --use-te # Use `te.DenseGeneral` provided by Transformer Engine to train MNIST with BF16
  python test_single_gpu_mnist.py --use-fp8 # Use `te.DenseGeneral` provided by Transformer Engine to train MNIST and enable FP8 training and evaluation.

Encoder

.. code-block:: bash
  
  cd examples/jax/encoder
  python test_single_gpu_encoder.py
  python test_single_gpu_encoder.py --use-fp8

Transformer Engine
==================

`Quickstart <#examples>`_ | `Installation <#installation>`_ | `User Guide <https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html>`_ | `Examples <https://github.com/NVIDIA/TransformerEngine/tree/main/examples>`_ | `Model Support <#model-support>`_ | `Integrations <#integrations>`_ | `Release notes <https://docs.nvidia.com/deeplearning/transformer-engine/release-notes/index.html>`_

Latest News
==================

* [04/2023] `Benchmarking Large Language Models on NVIDIA H100 GPUs with CoreWeave (Part 1) <https://www.mosaicml.com/blog/coreweave-nvidia-h100-part-1>`_


What is Transformer Engine?
==================
.. overview-begin-marker-do-not-remove

Transformer Engine (TE) is a library for accelerating Transformer models on NVIDIA GPUs, including
using 8-bit floating point (FP8) precision on Hopper GPUs, to provide better performance with lower
memory utilization in both training and inference. TE provides a collection of highly optimized
building blocks for popular Transformer architectures and an automatic mixed precision-like API that
can be used seamlessly with your framework-specific code. TE also includes a framework agnostic
C++ API that can be integrated with other deep learning libraries to enable FP8 support for Transformers.

As the number of parameters in Transformer models continues to grow, training and inference for
architectures such as BERT, GPT and T5 become very memory and compute intensive. Most deep learning
frameworks train with FP32 by default. This is not essential, however, to achieve full accuracy for
many deep learning models. Using mixed-precision training, which combines single-precision (FP32)
with lower precision (e.g. FP16) format when training a model, results in significant speedups with
minimal differences in accuracy as compared to FP32 training. With Hopper GPU
architecture FP8 precision was introduced, which offers improved performance over FP16 with no
degradation in accuracy. Although all major deep learning frameworks support FP16, FP8 support is
not available natively in frameworks today.

TE addresses the problem of FP8 support by providing APIs that integrate with popular Large Language
Model (LLM) libraries. It provides a Python API consisting of modules to easily build a Transformer
layer as well as a framework agnostic library in C++ including structs and kernels needed for FP8 support.
Modules provided by TE internally maintain scaling factors and other values needed for FP8 training, greatly
simplifying mixed precision training for users.

Highlights
----------

* Easy-to-use modules for building Transformer layers with FP8 support 
* Optimizations (e.g. fused kernels) for Transformer models 
* Support for FP8 on NVIDIA Hopper and NVIDIA Ada GPUs
* Support for optimizations across all precisions (FP16, BF16) on NVIDIA Ampere GPU architecture generations and later

Examples
----------

PyTorch
^^^^^^^

.. code-block:: python

  import torch
  import transformer_engine.pytorch as te
  from transformer_engine.common import recipe

  # Set dimensions.
  in_features = 768
  out_features = 3072
  hidden_size = 2048

  # Initialize model and inputs.
  model = te.Linear(in_features, out_features, bias=True)
  inp = torch.randn(hidden_size, in_features, device="cuda")

  # Create an FP8 recipe. Note: All input args are optional.
  fp8_recipe = recipe.DelayedScaling(margin=0, interval=1, fp8_format=recipe.Format.E4M3)

  # Enable autocasting for the forward pass
  with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
      out = model(inp)

  loss = out.sum()
  loss.backward()


JAX
^^^

Flax
~~~~

.. code-block:: python

  import jax
  import jax.numpy as jnp
  import transformer_engine.jax as te
  import transformer_engine.jax.flax as te_flax
  from transformer_engine.common import recipe

  BATCH = 32
  SEQLEN = 128
  HIDDEN = 1024

  # Initialize RNG and inputs.
  rng = jax.random.PRNGKey(0)
  init_rng, data_rng = jax.random.split(rng)
  inp = jax.random.normal(data_rng, [BATCH, SEQLEN, HIDDEN], jnp.float32)

  # Create an FP8 recipe. Note: All input args are optional.
  fp8_recipe = recipe.DelayedScaling(margin=0, interval=1, fp8_format=recipe.Format.HYBRID)

  # Enable autocasting for the forward pass
  with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
      model = te_flax.DenseGeneral(features=HIDDEN)

      def loss_fn(params, other_vars, inp):
        out = model.apply({'params':params, **other_vars}, inp)
        return jnp.mean(out)

      # Initialize models.
      variables = model.init(init_rng, inp)
      other_variables, params = variables.pop('params')

      # Construct the forward and backward function
      fwd_bwd_fn = jax.value_and_grad(loss_fn, argnums=(0, 1))

      for _ in range(10):
        loss, (param_grads, other_grads) = fwd_bwd_fn(params, other_variables, inp)
        # Update FP8 metas
        other_variables = te.update_fp8_metas(other_grads)

TensorFlow
^^^^^^^^^^

.. code-block:: python

  import tensorflow as tf
  import transformer_engine.tensorflow as te
  from transformer_engine.common import recipe
  
  # Set dimensions.
  in_features = 768
  out_features = 3072
  hidden_size = 2048
  
  # Initialize model and inputs.
  model = te.Dense(out_features, use_bias=True)
  inp = tf.random.normal((hidden_size, in_features))
  
  optimizer = tf.keras.optimizers.Adam(0.001)
  
  # Create FP8 recipe. Note: All input args are optional.
  fp8_recipe = recipe.DelayedScaling(margin=0, interval=1, fp8_format=recipe.Format.E4M3)
  
  with tf.GradientTape(persistent=True) as tape:
      # Enables autocasting for the forward pass
      with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
          out = model(inp)
      loss = tf.reduce_sum(out)
  grads = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(grads, model.trainable_variables))

.. overview-end-marker-do-not-remove

Installation
----------
.. installation

In the NGC container
^^^^^^^^^^^^^^^^^^^^

The quickest way to get started with Transformer Engine is the NGC PyTorch container on
`NVIDIA GPU Cloud Catalog <https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch>`_ (versions 22.09 and later).

.. code-block:: bash

    docker run --gpus all -it --rm nvcr.io/nvidia/pytorch:23.04-py3

Where 23.04 is the container version. For example, 23.04 for April 2023 release.

Pre-requisites
^^^^^^^^^^^^^^^^^^^^
* Linux x86_64
* CUDA 11.8 or later
* NVIDIA Driver supporting CUDA 11.8 or later
* cuDNN 8.1 or later
* For fused attention, CUDA 12.1 or later, NVIDIA Driver supporting CUDA 12.1 or later, and cuDNN 8.9 or later.

From source
^^^^^^^^^^^

`See the installation guide <https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/installation.html>`_.

Compiling with Flash Attention 2
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TransformerEngine release v0.11.0 adds support for Flash Attention 2.0 for improved performance. It is a known issue that Flash Attention 2.0 compilation is
resource intensive and requires a large amount of RAM (see `bug <https://github.com/Dao-AILab/flash-attention/issues/358>`_), which may lead to out of memory
errors during the installation of TransformerEngine. To circumvent the issue, please try setting **MAX_JOBS=1** in the environment. If the errors persist, then
proceed to install a supported version of Flash Attention 1 (v1.0.6 to v1.0.9).

Model Support
----------

While the more granular modules in Transformer Engine allow building any Transformer architecture,
the `TransformerLayer` API of Transformer Engine is flexible enough to build multiple major
Transformer model architectures.

Transformer Engine supports the following DL frameworks: PyTorch, JAX (Flax, Praxis), and TensorFlow.

NOTE: For simplicity, we only show PyTorch examples below. For the usage of `TransformerLayer`
of all supported frameworks, refer to `examples <https://github.com/NVIDIA/TransformerEngine/tree/main/examples>`_.

GPT
^^^

`GPT` architecture has `LayerNorm` at the input side (before `QKV Gemm`) and the residual connection
is taken from the input of that `LayerNorm`. In TE this can be achieved by setting the following
arguments in the `TransformerLayer` API.

.. code-block:: python

  transformer_engine.pytorch.TransformerLayer(
          ...,
          ...,
          apply_residual_connection_post_layernorm=False,
          output_layernorm=False,
          layer_type="encoder",
  )

BERT
^^^^

`BERT` architecture has `LayerNorm` at the output side (after the final `BiasDropoutAdd`) and the
residual connection is taken from the output of that `LayerNorm`. In TE this can be achieved by
setting the following arguments in the `TransformerLayer` API.

.. code-block:: python

  transformer_engine.pytorch.TransformerLayer(
          ...,
          ...,
          apply_residual_connection_post_layernorm=True,
          output_layernorm=True,
          layer_type="encoder",
  )

T5
^^

`T5` architecture has an additional `cross-attention` + `BiasDropoutAdd` + `LayerNorm` block before
the `MLP` layer. In TE this can be added by setting the `layer_type` to `decoder` in the
`TransformerLayer` API.

.. code-block:: python

  transformer_engine.pytorch.TransformerLayer(
          ...,
          ...,
          layer_type="decoder",
  )

Integrations
==================

Transformer Engine has been integrated with several popular open-source DL frameworks such as:

* `DeepSpeed <https://github.com/microsoft/DeepSpeed/pull/3731>`_ 
* `Hugging Face Accelerate <https://github.com/huggingface/accelerate/releases/tag/v0.17.0>`_ 
* `Megatron-LM <https://github.com/NVIDIA/Megatron-LM>`_ 
* `MosaicML Composer <https://github.com/mosaicml/composer/releases/tag/v0.13.1>`_ 
* `NVIDIA NeMo <https://github.com/NVIDIA/NeMo>`_ 
* `Amazon SageMaker Model Parallel Library <https://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel.html>`_ - Coming soon!
* `Colossal-AI <https://github.com/hpcaitech/ColossalAI>`_ - Coming soon!
* `Lightning <https://github.com/Lightning-AI/lightning/issues/17172>`_ - Coming soon!
* `PeriFlow <https://github.com/friendliai/periflow-python-sdk>`_ - Coming soon!


Contributing
==================

We welcome contributions to Transformer Engine! To contribute to Transformer Engine and make pull requests,
follow the guidelines outlined in the `<CONTRIBUTING.rst>`_ guide. 

Papers
==================

* `Attention original paper <https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf>`_
* `Megatron-LM tensor parallel <https://arxiv.org/pdf/1909.08053.pdf>`_
* `Megatron-LM sequence parallel <https://arxiv.org/pdf/2205.05198.pdf>`_
* `FP8 Formats for Deep Learning <https://arxiv.org/abs/2209.05433>`_

Videos
==================

* `FP8 Training with Transformer Engine <https://www.nvidia.com/en-us/on-demand/session/gtcspring23-s51393>`_  
* `FP8 for Deep Learning <https://www.nvidia.com/en-us/on-demand/session/gtcspring23-s52166/>`_  
* `Inside the Hopper Architecture <https://www.nvidia.com/en-us/on-demand/session/gtcspring22-s42663/>`_  

.. |License| image:: https://img.shields.io/badge/License-Apache%202.0-blue.svg
   :target: https://opensource.org/licenses/Apache-2.0
