..
    This file was modified to include portability information to AMDGPU.

    Copyright (c) 2023-2024, Advanced Micro Devices, Inc. All rights reserved.

    Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

|License|

Transformer Engine On ROCm and AMDGPU
*************************************

This repository enables Transformer Engine (TE) on ROCm as a library to accelerate Transformer models on AMD GPUs, including using 8-bit floating point (FP8) precision on MI300 GPUs, to provide better performance with lower memory utilization in both training and inference. 
One of the missions is to provide an alternative to accelerate Transformer models that were previously run on NVIDIA GPUs like Hopper with best efforts to make the migration frictionless. 
Moreover, we add optimizations specific to AMD GPUs to get the best performance benefits out of AMD GPUs.

Feature Support Status
======================

* Activation, cast, fused softmax, layernorm, rmsnorm, transpose, fused rope, fp8 recipe, HipRTC: fully supported
* GEMM: partially supported with following input/output types: (fp32/fp32), (fp16/fp16), (bf16/bf16), (fp8, bf8/fp16, bf16, fp32)
* Attention (Flash Attention, Fused Multihead Attention): partially supported: Fused Attention with AOTriton and CK backends
* HipGraph, HipTX: partially supported
* Tensor Parallelism, Sequence Parallelism, Context Parallelism: supported

Installation
============

Execute the following commands to install ROCm Transformer Engine from source on AMDGPUs:

.. code-block:: bash

  # Clone TE repo and submodules
  git clone --recursive https://github.com/ROCm/TransformerEngine.git
  
  cd TransformerEngine
  export NVTE_FRAMEWORK=pytorch,jax #optionally set framework, currently only support pytorch and jax; if not set will try to detect installed frameworks
  export NVTE_ROCM_ARCH=gfx942 # CK fused attn only support MI200 and MI300 and fp8 features are only supported on MI300
  pip install .

The default installation above supports both rocBlas and hipBlasLt in GEMM computation. Building with single backend support can be done by setting `NVTE_USE_HIPBLASLT` or `NVTE_USE_ROCBLAS` environment variable before `pip install` as:

.. code-block:: bash

  export NVTE_USE_HIPBLASLT=1
  export NVTE_USE_ROCBLAS=1

When both GEMM backends are supported the aforementioned env variables can be used to select which backend to use. If none is set hipBlasLt is used by default. The hipBlasLt backed has not yet supported all the GEMM configurations in the pytorch unit tests. 

Test
====

Framework Agnostic C++ library unittests
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After a successful Transformer Engine installation via `pip install`, execute the following commands to build and test the framework agnostic C++ library:

.. code-block:: bash

  cd tests/cpp
  mkdir build
  cd build
  cmake ../
  make
  make test

Note that some of operator unit tests fail in hipBLASLt config due to limited input data configurations support

Pytorch framework integration tests
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Pytorch integration pytests under tests/pytorch/ and tests/pytorch/fused_attn/ are supported 
Except the following tests that are not supported in rocBLAS configuration

* tests/pytorch/test_cuda_graph.py 
* tests/pytorch/test_sanity.py::test_gpt_guda_graph

Env `ROCBLAS_STREAM_ORDER_ALLOC=1` should be used when run tests in pytorch-rocblas configuration. 

Also test_onnx_export.py does not support FP8 dues to absence of custom QDQ operatrs library

Jax framework integration tests
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All JAX pytests are supported. 

Examples
========

Pytorch
^^^^^^^
MNIST with optional FP8
~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: bash
  
  cd examples/pytorch/mnist
  python main.py
  python main.py --use-te   # Linear layers from TransformerEngine
  python main.py --use-fp8  # FP8 + TransformerEngine for Linear layers

Sort with minGPT
~~~~~~~~~~~~~~~~
.. code-block:: bash
  
  cd examples/pytorch/minGPT
  python gptSort.py --use-te # Linear and layernorm from TransformerEngine
  python gptSort.py --use-te --ln-mlp # In addition, use LayernormMLP from transformer engine
  python gptSort.py --use-te --ln-mlp --use-fp8 # In addition, use fp8

Jax
^^^
Flax
~~~~
.. code-block:: python
  
  import flax
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
      other_variables, params = flax.core.pop(variables, 'params')

      # Construct the forward and backward function
      fwd_bwd_fn = jax.value_and_grad(loss_fn, argnums=(0, 1))

      for _ in range(10):
        loss, (param_grads, other_grads) = fwd_bwd_fn(params, other_variables, inp)
        # Update FP8 metas
        other_variables = te.update_fp8_metas(other_grads)

MNIST
~~~~~
.. code-block:: bash
  
  cd examples/jax/mnist
  python test_single_gpu_mnist.py # Use Flax to train MNIST with BF16 as usual
  python test_single_gpu_mnist.py --use-te # Use `te.DenseGeneral` provided by Transformer Engine to train MNIST with BF16
  python test_single_gpu_mnist.py --use-fp8 # Use `te.DenseGeneral` provided by Transformer Engine to train MNIST and enable FP8 training and evaluation.

Encoder
~~~~~~~
.. code-block:: bash
  
  cd examples/jax/encoder
  python test_single_gpu_encoder.py
  python test_single_gpu_encoder.py --use-fp8

Features on ROCm Platform
=========================

GEMM tuning with hipBlasLt
^^^^^^^^^^^^^^^^^^^^^^^^^^
When using GEMM with hipBlasLt, TE provides an ability to manually or automatically select GPU algorithm to use from a list generated by hipBlasLt. Selected algorithms info can be stored to file and read on further applications run.
This ability is controlled by environment variables when call GEMM operation with specific config for the first time.

* TE_HIPBLASLT_ALGO_SELECTION - algorithm index to use in the list returned by hipBlasLt for the config or the first algorithm to select from if auto-selection is enabled; default=0.
* TE_HIPBLASLT_TUNING_RUN_COUNT - number of profiling loops for algorithm auto-selection; default=0 which means no auto-selection. For small tasks where run-to-run time variation is relatively high, using higher number of loops may give better auto-selection results.
* TE_HIPBLASLT_TUNING_ALGO_COUNT - maximal number of algorithms to check when auto-selection is enabled; default=16.
* TE_HIPBLASLT_ALGO_LOAD - filename of algorithm selection data saved by previous GEMM operation runs; if file does not exist, algorithm selection logic proceeds as if no filename were specified
* TE_HIPBLASLT_ALGO_SAVE - filename to save algorithm selection data to; can be the same as a filename to load in which case the file will be read first and then overwritten with updated results

It is not guaranteed that algorithm selection data file created with one version of TE or hipBlasLt will work with other versions. Even if it works, it is highly recommended to perform algorithm selection tuning again when switch to new libraries versions because new hipBLASLt may have new optimized algorithms.

Typical usage is the following:

1. Run single iteration of training enabling algorithm selection autotuning and saving:

.. code-block:: bash

  export TE_HIPBLASLT_TUNING_RUN_COUNT=20
  export TE_HIPBLASLT_TUNING_ALGO_COUNT=400
  export TE_HIPBLASLT_ALGO_SAVE=algo_tune.csv
  some_training_app

2. Use resulting algo_tune.csv for further training runs

.. code-block:: bash

  unset TE_HIPBLASLT_TUNING_RUN_COUNT TE_HIPBLASLT_TUNING_ALGO_COUNT TE_HIPBLASLT_ALGO_SAVE #these variables are not needed anymore
  export TE_HIPBLASLT_ALGO_LOAD=algo_tune.csv
  some_training_app

If you want to check that only previously tuned algorithms are used by your application, it can be done by keeping selection data saving enabled. 

.. code-block:: bash

  export TE_HIPBLASLT_ALGO_SAVE=algo_tune_check.csv
  export TE_HIPBLASLT_ALGO_LOAD=algo_tune.csv
  some_training_app
  #If the files are different, some not previously cached algorithms are probably used
  diff algo_tune.csv algo_tune_check.csv


Fused Attention Backends on ROCm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Currently ROCm TE supports two backends, AOTriton and CK, for fused attention. 
To enable specific backends, the following environment variables can be used:

* NVTE_FUSED_ATTN - enable the fused attention, default = 1;
* NVTE_FUSED_ATTN_CK - enable the CK backend, default = 1;
* NVTE_FUSED_ATTN_AOTRITON - enable the AOTriton backend, default = 1.

NVTE_FUSED_ATTN has higher priority than NVTE_FUSED_ATTN_CK and NVTE_FUSED_ATTN_AOTRITON. 
NVTE_FUSED_ATTN=0 will use the TE unfused attention even if NVTE_FUSED_ATTN_CK or NVTE_FUSED_ATTN_AOTRITON is set. 
Fused attention backends are chosen according to the match results between the actual problem config and the support matrix of the specific backend. 
For the scenario that both backends are enabled and match the problem configuration, the CK backend will be chosen with higher priority. 

Transformer Engine
******************

`Quickstart <#examples>`_ | `Installation <#installation>`_ | `User Guide <https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html>`_ | `Examples <https://github.com/NVIDIA/TransformerEngine/tree/main/examples>`_ | `FP8 Convergence <#fp8-convergence>`_ | `Integrations <#integrations>`_ | `Release notes <https://docs.nvidia.com/deeplearning/transformer-engine/release-notes/index.html>`_

Latest News
===========

* [03/2024] `Turbocharged Training: Optimizing the Databricks Mosaic AI stack with FP8 <https://www.databricks.com/blog/turbocharged-training-optimizing-databricks-mosaic-ai-stack-fp8>`_
* [03/2024] `FP8 Training Support in SageMaker Model Parallelism Library <https://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel-release-notes.html>`_
* [12/2023] `New NVIDIA NeMo Framework Features and NVIDIA H200 <https://developer.nvidia.com/blog/new-nvidia-nemo-framework-features-and-nvidia-h200-supercharge-llm-training-performance-and-versatility/>`_

.. image:: docs/examples/H200-NeMo-performance.png
  :width: 600
  :alt: H200

* [11/2023] `Inflection-2: The Next Step Up <https://inflection.ai/inflection-2>`_
* [11/2023] `Unleashing The Power Of Transformers With NVIDIA Transformer Engine <https://lambdalabs.com/blog/unleashing-the-power-of-transformers-with-nvidia-transformer-engine>`_
* [11/2023] `Accelerating PyTorch Training Workloads with FP8 <https://towardsdatascience.com/accelerating-pytorch-training-workloads-with-fp8-5a5123aec7d7>`_
* [09/2023] `Transformer Engine added to AWS DL Container for PyTorch Training <https://github.com/aws/deep-learning-containers/pull/3315>`_
* [06/2023] `Breaking MLPerf Training Records with NVIDIA H100 GPUs <https://developer.nvidia.com/blog/breaking-mlperf-training-records-with-nvidia-h100-gpus/>`_
* [04/2023] `Benchmarking Large Language Models on NVIDIA H100 GPUs with CoreWeave (Part 1) <https://www.mosaicml.com/blog/coreweave-nvidia-h100-part-1>`_

What is Transformer Engine?
===========================
.. overview-begin-marker-do-not-remove

Transformer Engine (TE) is a library for accelerating Transformer models on NVIDIA GPUs, including
using 8-bit floating point (FP8) precision on Hopper GPUs, to provide better performance with lower
memory utilization in both training and inference. TE provides a collection of highly optimized
building blocks for popular Transformer architectures and an automatic mixed precision-like API that
can be used seamlessly with your framework-specific code. TE also includes a framework agnostic
C++ API that can be integrated with other deep learning libraries to enable FP8 support for Transformers.

As the number of parameters in Transformer models continues to grow, training and inference for
architectures such as BERT, GPT and T5 become very memory and compute-intensive. Most deep learning
frameworks train with FP32 by default. This is not essential, however, to achieve full accuracy for
many deep learning models. Using mixed-precision training, which combines single-precision (FP32)
with lower precision (e.g. FP16) format when training a model, results in significant speedups with
minimal differences in accuracy as compared to FP32 training. With Hopper GPU
architecture FP8 precision was introduced, which offers improved performance over FP16 with no
degradation in accuracy. Although all major deep learning frameworks support FP16, FP8 support is
not available natively in frameworks today.

TE addresses the problem of FP8 support by providing APIs that integrate with popular Large Language
Model (LLM) libraries. It provides a Python API consisting of modules to easily build a Transformer
layer as well as a framework-agnostic library in C++ including structs and kernels needed for FP8 support.
Modules provided by TE internally maintain scaling factors and other values needed for FP8 training, greatly
simplifying mixed precision training for users.

Highlights
==========

* Easy-to-use modules for building Transformer layers with FP8 support
* Optimizations (e.g. fused kernels) for Transformer models
* Support for FP8 on NVIDIA Hopper and NVIDIA Ada GPUs
* Support for optimizations across all precisions (FP16, BF16) on NVIDIA Ampere GPU architecture generations and later

Examples
========

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
  fp8_recipe = recipe.DelayedScaling(margin=0, fp8_format=recipe.Format.E4M3)

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

  import flax
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
  fp8_recipe = recipe.DelayedScaling(margin=0, fp8_format=recipe.Format.HYBRID)

  # Enable autocasting for the forward pass
  with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
      model = te_flax.DenseGeneral(features=HIDDEN)

      def loss_fn(params, other_vars, inp):
        out = model.apply({'params':params, **other_vars}, inp)
        return jnp.mean(out)

      # Initialize models.
      variables = model.init(init_rng, inp)
      other_variables, params = flax.core.pop(variables, 'params')

      # Construct the forward and backward function
      fwd_bwd_fn = jax.value_and_grad(loss_fn, argnums=(0, 1))

      for _ in range(10):
        loss, (param_grads, other_grads) = fwd_bwd_fn(params, other_variables, inp)

.. overview-end-marker-do-not-remove

Installation
============
.. installation

Pre-requisites
^^^^^^^^^^^^^^^^^^^^
* Linux x86_64
* CUDA 11.8+ for Hopper and CUDA 12.1+ for Ada
* NVIDIA Driver supporting CUDA 11.8 or later
* cuDNN 8.1 or later
* For fused attention, CUDA 12.1 or later, NVIDIA Driver supporting CUDA 12.1 or later, and cuDNN 8.9 or later.

Docker
^^^^^^^^^^^^^^^^^^^^

The quickest way to get started with Transformer Engine is by using Docker images on
`NVIDIA GPU Cloud (NGC) Catalog <https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch>`_. For example to use the NGC PyTorch container interactively,

.. code-block:: bash

    docker run --gpus all -it --rm nvcr.io/nvidia/pytorch:23.10-py3

Where 23.10 is the container version. For example, 23.10 for the October 2023 release.

pip
^^^^^^^^^^^^^^^^^^^^
To install the latest stable version of Transformer Engine,

.. code-block:: bash

    pip install git+https://github.com/NVIDIA/TransformerEngine.git@stable

This will automatically detect if any supported deep learning frameworks are installed and build Transformer Engine support for them. To explicitly specify frameworks, set the environment variable NVTE_FRAMEWORK to a comma-separated list (e.g. NVTE_FRAMEWORK=jax,pytorch).

From source
^^^^^^^^^^^
`See the installation guide <https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/installation.html#installation-from-source>`_.

Compiling with FlashAttention-2
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Transformer Engine release v0.11.0 adds support for FlashAttention-2 in PyTorch for improved performance. 

It is a known issue that FlashAttention-2 compilation is resource-intensive and requires a large amount of RAM (see `bug <https://github.com/Dao-AILab/flash-attention/issues/358>`_), which may lead to out of memory errors during the installation of Transformer Engine. Please try setting **MAX_JOBS=1** in the environment to circumvent the issue.

Note that NGC PyTorch 23.08+ containers include FlashAttention-2.

Breaking Changes
================

v1.7: Padding mask definition for PyTorch
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In an effort to unify the definition and usage of the attention mask across all three frameworks in Transformer Engine, the padding mask has changed from `True` meaning inclusion of the corresponding position in attention to exclusion of that position in our PyTorch implementation. Since v1.7, all attention mask types follow the same definition where `True` means masking out the corresponding position and `False` means including that position in attention calculation.

An example of this change is,

.. code-block:: bash

    # for a batch of 3 sequences where `a`s, `b`s and `c`s are the useful tokens
    # and `0`s are the padding tokens,
    [a, a, a, 0, 0,
     b, b, 0, 0, 0,
     c, c, c, c, 0]
    # the padding mask for this batch before v1.7 is,
    [ True,  True,  True, False, False,
      True,  True, False, False, False,
      True,  True,  True,  True, False]
    # and for v1.7 onwards it should be,
    [False, False, False,  True,  True,
     False, False,  True,  True,  True,
     False, False, False, False,  True]

FP8 Convergence
===============

FP8 has been tested extensively across different model architectures and configurations and we found **no significant difference** between FP8 and BF16 training loss curves. FP8 has also been validated for accuracy on downstream LLM tasks (e.g. LAMBADA and WikiText). Below are examples of models tested for convergence across different frameworks.

+------------+------------------+---------------------------------------------------------------------------------------------------------+
| Model      | Framework        | Source                                                                                                  |
+============+==================+=========================================================================================================+
| T5-770M    |  JAX/T5x         | https://github.com/NVIDIA/JAX-Toolbox/tree/main/rosetta/rosetta/projects/t5x#convergence-and-performance|
+------------+------------------+---------------------------------------------------------------------------------------------------------+
| MPT-1.3B   |  Mosaic Composer | https://www.mosaicml.com/blog/coreweave-nvidia-h100-part-1                                              |
+------------+------------------+---------------------------------------------------------------------------------------------------------+
| GPT-5B     |  JAX/Paxml       | https://github.com/NVIDIA/JAX-Toolbox/tree/main/rosetta/rosetta/projects/pax#h100-results               |
+------------+------------------+---------------------------------------------------------------------------------------------------------+
| GPT-5B     |  NeMo Framework  | Available on request                                                                                    |
+------------+------------------+---------------------------------------------------------------------------------------------------------+
| LLama2-7B  |  Alibaba Pai     | https://mp.weixin.qq.com/s/NQT0uKXLbXyh5031zBdeBQ                                                       |
+------------+------------------+---------------------------------------------------------------------------------------------------------+
| T5-11B     |  JAX/T5x         | Available on request                                                                                    |
+------------+------------------+---------------------------------------------------------------------------------------------------------+
| MPT-13B    |  Mosaic Composer | https://www.databricks.com/blog/turbocharged-training-optimizing-databricks-mosaic-ai-stack-fp8         |
+------------+------------------+---------------------------------------------------------------------------------------------------------+
| GPT-22B    |  NeMo Framework  | Available on request                                                                                    |
+------------+------------------+---------------------------------------------------------------------------------------------------------+
| LLama2-70B |  Alibaba Pai     | https://mp.weixin.qq.com/s/NQT0uKXLbXyh5031zBdeBQ                                                       |
+------------+------------------+---------------------------------------------------------------------------------------------------------+
| GPT-175B   |  JAX/Paxml       | https://github.com/NVIDIA/JAX-Toolbox/tree/main/rosetta/rosetta/projects/pax#h100-results               |
+------------+------------------+---------------------------------------------------------------------------------------------------------+

Integrations
============

Transformer Engine has been integrated with popular LLM frameworks such as:

* `DeepSpeed <https://github.com/microsoft/DeepSpeed/pull/3731>`_
* `Hugging Face Accelerate <https://github.com/huggingface/accelerate/releases/tag/v0.17.0>`_
* `Lightning <https://github.com/Lightning-AI/lightning/issues/17172>`_
* `MosaicML Composer <https://github.com/mosaicml/composer/releases/tag/v0.13.1>`_
* `NVIDIA JAX Toolbox <https://github.com/NVIDIA/JAX-Toolbox>`_
* `NVIDIA Megatron-LM <https://github.com/NVIDIA/Megatron-LM>`_
* `NVIDIA NeMo Framework <https://github.com/NVIDIA/NeMo-Megatron-Launcher>`_
* `Amazon SageMaker Model Parallel Library <https://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel-core-features-v2-tensor-parallelism.html>`_
* `Levanter <https://github.com/stanford-crfm/levanter>`_
* `Hugging Face Nanotron <https://github.com/huggingface/nanotron>`_ - Coming soon!
* `Colossal-AI <https://github.com/hpcaitech/ColossalAI>`_ - Coming soon!
* `PeriFlow <https://github.com/friendliai/periflow-python-sdk>`_ - Coming soon!
* `GPT-NeoX <https://github.com/EleutherAI/gpt-neox>`_ - Coming soon!


Contributing
============

We welcome contributions to Transformer Engine! To contribute to Transformer Engine and make pull requests,
follow the guidelines outlined in the `<CONTRIBUTING.rst>`_ guide.

Papers
======

* `Attention original paper <https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf>`_
* `Megatron-LM tensor parallel <https://arxiv.org/pdf/1909.08053.pdf>`_
* `Megatron-LM sequence parallel <https://arxiv.org/pdf/2205.05198.pdf>`_
* `FP8 Formats for Deep Learning <https://arxiv.org/abs/2209.05433>`_

Videos
======

* `What's New in Transformer Engine and FP8 Training | GTC 2024 <https://www.nvidia.com/en-us/on-demand/session/gtc24-s62457/>`_
* `FP8 Training with Transformer Engine | GTC 2023 <https://www.nvidia.com/en-us/on-demand/session/gtcspring23-s51393>`_
* `FP8 for Deep Learning | GTC 2023 <https://www.nvidia.com/en-us/on-demand/session/gtcspring23-s52166/>`_
* `Inside the Hopper Architecture <https://www.nvidia.com/en-us/on-demand/session/gtcspring22-s42663/>`_

.. |License| image:: https://img.shields.io/badge/License-Apache%202.0-blue.svg
   :target: https://opensource.org/licenses/Apache-2.0
