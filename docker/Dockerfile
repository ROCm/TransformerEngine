# CONTEXT {'gpu_vendor': 'AMD', 'guest_os': 'UBUNTU'}
ARG BASE_DOCKER=rocm/pytorch:rocm6.4_ubuntu22.04_py3.10_pytorch_release_2.5.1
FROM $BASE_DOCKER
WORKDIR /var/lib/jenkins

RUN apt update \
    && apt install -y nano wget ninja-build \
    && apt install -y python3 python3-pip git \
    && apt install -y sqlite3 libsqlite3-dev libfmt-dev libmsgpack-dev libsuitesparse-dev

RUN python3 -m pip install --upgrade pip
RUN pip install ninja cmake setuptools wheel
RUN pip install uv tabulate
RUN pip install ipython pytest fire pydantic pybind11

RUN pip uninstall -y torch

RUN apt --fix-broken install -y
RUN apt install -y libzstd-dev
RUN apt install -y libibverbs-dev

ENV LLVM_SYMBOLIZER_PATH=/opt/rocm/llvm/bin/llvm-symbolizer
ENV PATH=$PATH:/opt/rocm/bin:
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib/:

# Install pytorch
ARG PYTORCH_COMMIT="f929e0d602a71aa393ca2e6097674b210bdf321c"
ENV PYTORCH_ROCM_ARCH=gfx942
RUN rm -fr pytorch \
    && git clone https://github.com/pytorch/pytorch \
    && cd pytorch \
    && git fetch origin ${PYTORCH_COMMIT} \
    && git checkout -q ${PYTORCH_COMMIT} \
    && git submodule update --recursive --init \
    && ./tools/amd_build/build_amd.py \
    && BUILD_TEST=0 python3 setup.py install 

WORKDIR /var/lib/jenkins

# Install flash-attention
ENV GPU_ARCHS=${PYTORCH_ROCM_ARCH}
RUN git clone https://github.com/ROCm/flash-attention.git \
    && cd flash-attention \
    && git checkout v2.7.3-cktile \
    && pip install . 

WORKDIR /var/lib/jenkins

# Install jax 
RUN git clone -b rocm-jaxlib-v0.4.35-qa https://github.com/ROCm/jax.git \
    && git clone -b rocm-jaxlib-v0.4.35-qa https://github.com/ROCm/xla.git \
    && cd jax \
    && echo 'run:rocm_plugin --copt=-DLEGACY_HIPBLAS_DIRECT' > .bazelrc.user \
    && python3 ./build/build.py --enable_rocm \
									 --build_gpu_plugin \
									 --use_clang=true \
									 --clang_path=/opt/rocm-6.4.0/lib/llvm/bin/clang \
									 --gpu_plugin_rocm_version=60 \
									 --rocm_path=/opt/rocm-6.4.0/ \
									 --rocm_amdgpu_targets=gfx942 \
									 --bazel_options=--override_repository=xla=/var/lib/jenkins/xla \
    && pip install jax==0.4.35 \
    && python3 setup.py develop --user && python3 -m pip install dist/*.whl \
    && pip install jax==0.4.35

WORKDIR /workspace/
CMD ["/bin/bash"]