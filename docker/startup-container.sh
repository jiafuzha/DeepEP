#!/bin/bash
# Start the DeepEP (Intel XPU + RDMA/Mellanox) test container.
#
# For the two-node simulation on one host, use docker-compose.yml together with:
#   docker/scripts/up_sim_nodes.sh
#   docker/scripts/run_pytorch_distributed.sh
#
#   --device=/dev/dri          -> Intel GPU (XPU) render nodes
#   --device=/dev/infiniband   -> RDMA verbs / Mellanox ConnectX-8 devices
#   --ulimit memlock=-1        -> required so verbs can pin (register) memory
#   --cap-add=IPC_LOCK         -> allow memory locking for RDMA
#   --net=host                 -> use host networking (RDMA/IB fabric)
#   --shm-size=32g             -> large shared memory for oneCCL/DeepEP
docker run -itd \
    --privileged \
    --net=host \
    --device=/dev/dri \
    --device=/dev/infiniband \
    --ulimit memlock=-1 \
    --cap-add=IPC_LOCK \
    --name=deepep_test \
    -v /root/jiafuzha:/root/jiafuzha \
    -e no_proxy=localhost,127.0.0.1 \
    -e http_proxy=$http_proxy \
    -e https_proxy=$https_proxy \
    --shm-size="32g" \
    --entrypoint /bin/bash \
    deepep_jiafuzha
