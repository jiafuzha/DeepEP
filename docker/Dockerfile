# =============================================================================
# Dockerfile to reproduce the "deepep_jiafuzha" image
#
# Target: DeepEP test environment on Intel GPU (XPU)
#   - Ubuntu 24.04 (Noble)
#   - Intel oneAPI 2025.3 Deep Learning Essentials (DPC++, MKL, oneCCL, MPI,
#     oneDNN, TBB, oneDPL, PTI, UMF)
#   - Intel GPU compute runtime (Level-Zero, OpenCL ICD, IGC, xpu-smi)
#   - RDMA / libibverbs stack for Mellanox ConnectX-8 NICs
#     (rdma-core, ibverbs, librdmacm, perftest, infiniband-diags, UCX)
#   - PyTorch (XPU) built from source + DeepEP / ISHMEM-IBGDA (built at runtime
#     from the bind-mounted /root/jiafuzha source tree — see notes at bottom)
#
# NOTE ON DRIVERS/HARDWARE:
#   RDMA (libibverbs / verbs) and the Mellanox kernel drivers (mlx5) live in the
#   HOST kernel, NOT in the image. This image only ships the *user-space* verbs
#   libraries and tools. At runtime the container must be started with access to
#   the host RDMA and GPU devices, e.g.:
#
#     docker run -itd --privileged --net=host \
#       --device=/dev/dri --device=/dev/infiniband \
#       --ulimit memlock=-1 --shm-size=32g ...
#
#   (see docker/startup-container.sh)
# =============================================================================

FROM ubuntu:24.04

LABEL maintainer="jiafuzha" \
      description="DeepEP on Intel GPU (XPU) with oneAPI 2025.3 + RDMA/verbs for Mellanox ConnectX-8"

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    LANG=C.UTF-8

# -----------------------------------------------------------------------------
# 1. Base build tools, utilities and the RDMA / libibverbs (verbs) user stack
#    for the Mellanox ConnectX-8 NICs.
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        # --- build toolchain ---
        build-essential \
        gcc g++ make \
        autoconf automake libtool \
        pkg-config \
        python3-dev python3-pip \
        # --- source control / download ---
        git wget curl ca-certificates gnupg gpg-agent \
        software-properties-common lsb-release \
        # --- general utilities / debugging ---
        vim gdb bc file lsof less \
        net-tools iproute2 pciutils \
        numactl libnuma-dev \
        hwloc libhwloc-dev libhwloc-plugins \
        openssh-client openssh-server \
        ffmpeg \
        # --- RDMA / InfiniBand / verbs user-space stack (Mellanox mlx5) ---
        rdma-core \
        ibverbs-providers \
        ibverbs-utils \
        libibverbs1 libibverbs-dev \
        librdmacm1t64 librdmacm-dev \
        libibmad5 libibmad-dev \
        libibumad3 libibumad-dev \
        infiniband-diags \
        ibutils \
        perftest \
        libucx0 \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# 2. GitHub CLI (gh) repository + install
# -----------------------------------------------------------------------------
RUN install -m 0755 -d /etc/apt/keyrings \
    && curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
         -o /etc/apt/keyrings/githubcli-archive-keyring.gpg \
    && chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
         > /etc/apt/sources.list.d/github-cli.list \
    && apt-get update && apt-get install -y --no-install-recommends gh \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# 3. Intel oneAPI 2025.3 - Deep Learning Essentials
#    (DPC++/C++ compiler, MKL, oneCCL, MPI, oneDNN, TBB, oneDPL, PTI, UMF)
# -----------------------------------------------------------------------------
RUN curl -fsSL https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
        | gpg --dearmor -o /usr/share/keyrings/oneapi-archive-keyring.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
        > /etc/apt/sources.list.d/oneAPI.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        intel-deep-learning-essentials \
        intel-oneapi-compiler-dpcpp-cpp \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# 4. Intel GPU compute runtime (Level-Zero, OpenCL ICD, IGC, xpu-smi)
#    from the Intel graphics repository.
# -----------------------------------------------------------------------------
RUN curl -fsSL https://repositories.intel.com/gpu/intel-graphics.key \
        | gpg --dearmor -o /usr/share/keyrings/intel-graphics-archive-keyring.gpg \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics-archive-keyring.gpg] https://repositories.intel.com/gpu/ubuntu noble unified" \
        > /etc/apt/sources.list.d/intel-graphics.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        intel-opencl-icd \
        intel-igc-core-2 \
        intel-igc-opencl-2 \
        intel-ocloc \
        level-zero libze1 libze-dev \
        libze-intel-gpu1 \
        libigdgmm12 \
        xpu-smi \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# 5. Python packages (installed to the system interpreter, as in the source
#    image). cmake / ninja come from pip and land in /usr/local/bin.
# -----------------------------------------------------------------------------
RUN pip3 install --no-cache-dir --break-system-packages \
        cmake \
        ninja \
        numpy \
        pyyaml \
        setuptools \
        wheel \
        packaging \
        filelock \
        fsspec \
        jinja2 \
        networkx \
        sympy \
        typing_extensions \
        requests

# -----------------------------------------------------------------------------
# 6. Make the oneAPI environment available in every (login/interactive) shell.
#    setvars.sh populates PATH, LD_LIBRARY_PATH, CPATH, CMAKE_PREFIX_PATH,
#    MKLROOT, CCL_ROOT, I_MPI_ROOT, FI_PROVIDER_PATH, etc.
# -----------------------------------------------------------------------------
RUN echo 'if [ -f /opt/intel/oneapi/setvars.sh ]; then source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; fi' \
        >> /root/.bashrc

# Entry-point wrapper that sources oneAPI before handing control to the command.
RUN printf '%s\n' \
    '#!/bin/bash' \
    'if [ -f /opt/intel/oneapi/setvars.sh ]; then' \
    '  source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1' \
    'fi' \
    'exec "$@"' \
    > /usr/local/bin/entrypoint.sh \
    && chmod +x /usr/local/bin/entrypoint.sh

WORKDIR /root/jiafuzha

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["/bin/bash"]

# =============================================================================
# 7. Building the from-source components (NOT baked into the image because they
#    live in the bind-mounted /root/jiafuzha source tree).
#
#    After starting the container (see docker/startup-container.sh), run:
#
#      source /opt/intel/oneapi/setvars.sh
#
#      # PyTorch (XPU build)  -> /root/jiafuzha/code-repo/pytorch  (commit b14e6fb)
#      cd /root/jiafuzha/code-repo/pytorch
#      pip3 install -r requirements.txt --break-system-packages
#      USE_XPU=1 python3 setup.py develop
#
#      # intel_gpu_uar_bridge
#      cd /root/jiafuzha/code-repo/intel_gpu_uar_bridge && ./build.sh
#
#      # ISHMEM (IBGDA)
#      cd /root/jiafuzha/ishmem_ibgda && ./scripts/...   # per its README
#
#      # DeepEP
#      cd /root/jiafuzha/code-repo/zjf2012/DeepEP && ./install.sh
#
#    These build steps depend on the specific host GPU/NIC and the mounted
#    source, so they are intentionally left as runtime steps.
# =============================================================================
