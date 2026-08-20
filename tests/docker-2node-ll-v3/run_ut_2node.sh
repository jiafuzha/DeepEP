#!/bin/bash
# Run the faithful ibgda_doorbell UT (proof or stress) ACROSS the 2 containers
# (4 ranks: 2 per node) over the exact same RDMA transport the LL test uses.
# Mirrors run.sh's cross-container `mpirun -launcher ssh` but launches the UT
# binary instead of the python LL test.
#
# Env knobs (optional):
#   UT_BIN     which unit test  (default ibgda_doorbell_stress)
#   ISHMEM_PROOF_NPUTS / _ELEMS / _WORKITEMS / _KITERS  (stress scaling)
#   ISHMEM_SYMMETRIC_SIZE  (default 67108864)
#   TIMEOUT_SEC (default 120)
set -uo pipefail

NODE0_CONTAINER=deepep-ll-v3-node0
NODE1_CONTAINER=deepep-ll-v3-node1
ISHMEM_DIR="${ISHMEM_DIR:-/root/jiafuzha/code-repo/ishmem_ibgda/build/_install}"
DEEP_EP_DIR=/root/jiafuzha/code-repo/zjf2012/DeepEP
WRAPPER_PATH="$DEEP_EP_DIR/tests/docker-2node-ll-v3/ut_wrapper.sh"
UT_BIN="${UT_BIN:-/root/jiafuzha/code-repo/ishmem_ibgda/build/test/unit/ibgda_doorbell_stress}"
TIMEOUT_SEC="${TIMEOUT_SEC:-120}"
NUM_PROCESSES=2   # ranks per node
TOTAL_RANKS=4

docker exec "$NODE0_CONTAINER" chmod +x "$WRAPPER_PATH" 2>/dev/null || true
docker exec "$NODE1_CONTAINER" chmod +x "$WRAPPER_PATH" 2>/dev/null || true

set +e
docker exec \
    -e ISHMEM_DIR="$ISHMEM_DIR" \
    -e ISHMEM_DEBUG="${ISHMEM_DEBUG:-0}" \
    -e ISHMEM_PROOF_NPUTS="${ISHMEM_PROOF_NPUTS:-64}" \
    -e ISHMEM_PROOF_ELEMS="${ISHMEM_PROOF_ELEMS:-1}" \
    -e ISHMEM_PROOF_WORKITEMS="${ISHMEM_PROOF_WORKITEMS:-1}" \
    -e ISHMEM_PROOF_KITERS="${ISHMEM_PROOF_KITERS:-1}" \
    "$NODE0_CONTAINER" \
    bash -lc "
        source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
        export ISHMEM_DIR=$ISHMEM_DIR
        export LD_LIBRARY_PATH=\$ISHMEM_DIR/lib:\${LD_LIBRARY_PATH:-}
        cd $DEEP_EP_DIR
        timeout $TIMEOUT_SEC mpirun \
            -n $TOTAL_RANKS -ppn $NUM_PROCESSES \
            -hosts $NODE0_CONTAINER,$NODE1_CONTAINER \
            -genv ISHMEM_IB_ENABLE_IBGDA 1 \
            -genv ISHMEM_IBGDA_DIRECT_DOORBELL 1 \
            -genv ISHMEM_ENABLE_GPU_IPC 0 \
            -genv ISHMEM_ENABLE_ACCESSIBLE_HOST_HEAP 0 \
            -genv ISHMEM_SYMMETRIC_SIZE ${ISHMEM_SYMMETRIC_SIZE:-67108864} \
            -genv ZE_ENABLE_PCI_ID_DEVICE_ORDER 1 \
            -genv ISHMEM_IBGDA_QPS_PER_PE ${ISHMEM_IBGDA_QPS_PER_PE:-1} \
            -genv ISHMEM_IBGDA_DB_BATCH_SIZE ${ISHMEM_IBGDA_DB_BATCH_SIZE:-0} \
            -genv ISHMEM_IBGDA_BAR_BACKEND igub \
            -genv ISHMEM_DEBUG \"\${ISHMEM_DEBUG:-0}\" \
            -genv ISHMEM_PROOF_NPUTS \"\${ISHMEM_PROOF_NPUTS:-64}\" \
            -genv ISHMEM_PROOF_ELEMS \"\${ISHMEM_PROOF_ELEMS:-1}\" \
            -genv ISHMEM_PROOF_WORKITEMS \"\${ISHMEM_PROOF_WORKITEMS:-1}\" \
            -genv ISHMEM_PROOF_KITERS \"\${ISHMEM_PROOF_KITERS:-1}\" \
            -genv I_MPI_FABRICS shm:ofi \
            -genv FI_PROVIDER tcp \
            -genv ISHMEM_DIR $ISHMEM_DIR \
            -genv MASTER_ADDR $NODE0_CONTAINER \
            -genv WORLD_SIZE 2 \
            -launcher ssh \
            -bootstrap-exec-args '-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i /root/.ssh/id_rsa' \
            $WRAPPER_PATH \
            $UT_BIN
    "
rc=$?
set -e
echo "===== UT cross-container rc=$rc ====="
exit $rc
