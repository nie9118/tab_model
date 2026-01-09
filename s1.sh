#!/bin/bash
#SBATCH --job-name=golan
#SBATCH --partition=faculty
#SBATCH --account=test-acc
#SBATCH --qos=bgqos
#SBATCH --nodes=2                     # <<< 多机：按需修改节点数
#SBATCH --ntasks-per-node=1            # 每节点仅启 1 个“启动器任务”
#SBATCH --gpus-per-node=8              # 每节点 GPU 数
#SBATCH --cpus-per-task=128
#SBATCH --mem=240G
#SBATCH --time=3-00:00:00
#SBATCH --output=/vast/users/guangyi.chen/causal_group/zijian.li/slurm_tools/logs/%x-%j.out
#SBATCH --error=/vast/users/guangyi.chen/causal_group/zijian.li/slurm_tools/logs/%x-%j.err                              #SBATCH --export=ALL                                                                                                    #SBATCH --exclude auh7-1b-gpu-257,auh7-1b-gpu-258,auh7-1b-gpu-259,auh7-1b-gpu-260,,auh7-1b-gpu-261,auh7-1b-gpu-262,auh7-1b-gpu-263,auh7-1b-gpu-264,auh7-1b-gpu-265,auh7-1b-gpu-266,auh7-1b-gpu-267,auh7-1b-gpu-268,auh7-1b-gpu-269,auh7-1b-gpu-270,auh7-1b-gpu-271,auh7-1b-gpu-272,auh7-1b-gpu-230,auh7-1b-gpu-231,auh7-1b-gpu-232,auh7-1b-gpu-233,auh7-1b-gpu-234,auh7-1b-gpu-235,auh7-1b-gpu-236,auh7-1b-gpu-237


echo "[$(date)] Running on host: $(hostname)"
echo "[$(date)] SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "[$(date)] Starting distributed training job..."

########################################
# 环境设置
########################################
source ~/.bashrc
source ~/slurm_tools/mi.sh
conda activate ORI

########################################
# 分布式环境变量
########################################
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=${MASTER_PORT:-29500}
GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-8}
NUM_NODES=${SLURM_NNODES}
WORLD_SIZE=$(( NUM_NODES * GPUS_PER_NODE ))

export MASTER_ADDR MASTER_PORT WORLD_SIZE
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
# export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-^lo,docker0}

export PROJECT_HOME="/vast/users/guangyi.chen/causal_group/zijian.li/dmir_crl/tab_model/Orion-MSP-main"
export PYTHONPATH="$PROJECT_HOME:$PYTHONPATH"

echo "[$(date)] MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
echo "[$(date)] NUM_NODES=$NUM_NODES GPUS_PER_NODE=$GPUS_PER_NODE WORLD_SIZE=$WORLD_SIZE"

########################################
# 启动训练
########################################
srun --ntasks=${NUM_NODES} --ntasks-per-node=1 bash -lc '
  NODE_RANK=${SLURM_NODEID}
  echo "[${HOSTNAME}] NODE_RANK=${NODE_RANK}"

  GPUS_PER_NODE=${SLURM_GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-8}}
  echo "[${HOSTNAME}] GPUS_PER_NODE=$GPUS_PER_NODE"
  echo "[${HOSTNAME}] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

  source ~/.bashrc
  source ~/slurm_tools/mi.sh
  conda activate ORI

   torchrun  --nproc_per_node=${GPUS_PER_NODE} --nnodes='"${NUM_NODES}"' --node_rank=${NODE_RANK} --master_addr='"${MASTER_ADDR}"' --master_port='"${MASTER_PORT}"'   /vast/users/guangyi.chen/causal_group/zijian.li/dmir_crl/tab_model/Orion-MSP-main/src/orion_msp/train/run.py \
            --wandb_log True \
            --wandb_project TabICL \
            --wandb_name ldm_2 \
            --wandb_dir /vast/users/guangyi.chen/causal_group/zijian.li/dmir_crl/tab_model/Orion-MSP-main/wandb/dir \
            --wandb_mode offline \
            --device cuda \
            --dtype float32 \
            --np_seed 42 \
            --torch_seed 42 \
            --max_steps 25000 \
            --batch_size 512 \
            --micro_batch_size 4 \
            --lr 1e-4 \
            --scheduler cosine_warmup \
            --warmup_proportion 0.02 \
            --gradient_clipping 1.0 \
            --prior_type mix_scm \
            --prior_device cpu \
            --batch_size_per_gp 4 \
            --min_features 2 \
            --max_features 100 \
            --max_classes 10 \
            --max_seq_len 1024 \
            --min_train_size 0.1 \
            --max_train_size 0.9 \
            --embed_dim 128 \
            --col_num_blocks 3 \
            --col_nhead 4 \
            --col_num_inds 128 \
            --row_num_blocks 3 \
            --row_nhead 8 \
            --row_num_cls 4 \
            --row_rope_base 100000 \
            --icl_num_blocks 12 \
            --icl_nhead 4 \
            --ff_factor 2 \
            --norm_first True \
            --checkpoint_dir /vast/users/guangyi.chen/causal_group/zijian.li/dmir_crl/tab_model/Orion-MSP-main/tab_1/stage1/checkpoint/dir \
            --save_temp_every 50 \
            --save_perm_every 5000

'

echo "[$(date)] Job finished."
