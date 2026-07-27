RDMA_ENV='CCL_ATL_TRANSPORT=ofi FI_PROVIDER=verbs FI_SHM_DISABLE=1'
for NT in 32 64 128 256 1024 2048 4096; do
     PORT=$((29700 + NT % 200))
     # node_rank=1 on b70-hq-2 (background)
     ssh b70-hq-2 "source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; \
        source /root/jiafuzha/code-repo/oneccl/build/_install/env/setvars.sh; \
       cd /root/jiafuzha/code-repo/zjf2012/DeepEP; \
       $RDMA_ENV nohup torchrun --nnodes=2 --nproc-per-node=2 --node_rank=1 \
         --master_addr=10.239.11.55 --master_port=$PORT \
         tests/xpu_allgather_reducescatter_4rank.py \
         --num-ranks 4 --num-tensors $NT --tensor-dtype bf16 --tensor-dimension 7168 \
         --benchmark --bench-iters 20 > /tmp/n1.log 2>&1 &"
     sleep 1
     # node_rank=0 on b70-hq-1 (foreground)
     source /root/jiafuzha/code-repo/oneccl/build/_install/env/setvars.sh
     env $RDMA_ENV torchrun --nnodes=2 --nproc-per-node=2 --node_rank=0 \
       --master_addr=10.239.11.55 --master_port=$PORT \
       tests/xpu_allgather_reducescatter_4rank.py \
       --num-ranks 4 --num-tensors $NT --tensor-dtype bf16 --tensor-dimension 7168 \
       --benchmark --bench-iters 20
     sleep 2
   done
