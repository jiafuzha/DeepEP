# Minimal container image for DeepEP 2-node simulation.
# All heavy dependencies (oneAPI, conda, repos) are bind-mounted from the host.
FROM ubuntu:22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libibverbs1 \
    librdmacm1 \
    libibumad3 \
    ibverbs-providers \
    rdma-core \
    numactl \
    openssh-server \
    openssh-client \
    iproute2 \
    pciutils \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /var/run/sshd \
    && echo "PermitRootLogin yes" >> /etc/ssh/sshd_config \
    && echo "StrictHostKeyChecking no" >> /etc/ssh/ssh_config

# MPI hydra expects /usr/bin/env bash
RUN ln -sf /bin/bash /usr/bin/bash

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
