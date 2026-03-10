#!/bin/bash
# 参考训练任务multi_planner_6
echo "开始训练-----"
RUNTIME_SCRIPT_DIR=/root/dl/runtime/script

cd $RUNTIME_SCRIPT_DIR
DIR=`pwd`

set -e

echo $NODE_IP_LIST > env.txt 2>&1 &
echo "${LOCAL_IP} slots=8" > "hostfile"
sed "s/:/ slots=/g" env.txt | sed "s/,/\n/g" > "hostfile"
sed "s/:.//g" env.txt | sed "s/,/\n/g" > "pssh.hosts"

share_pssh_host=${RUNTIME_SCRIPT_SHARE_DIR}/pssh.hosts
cp pssh.hosts $share_pssh_host

total_nodes_num=$(cat $share_pssh_host | wc -l )

remote_script="remote_scripts.sh"
cat > $remote_script <<EOF
#!/bin/bash

source ~/.bashrc

export NCCL_P2P_LEVEL=NVL
export NCCL_IB_TIMEOUT=24
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=INFO
export NCCL_MPI_PROFILE_PRIMS_ENABLE=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export NCCL_IB_GID_INDEX=3
export NCCL_IB_SL=3
export NCCL_CHECK_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export NCCL_LL_THRESHOLD=16384
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=3600000 

export NCCL_SOCKET_IFNAME=bond1
export UCX_NET_DEVICES=bond1
export NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6

export NCCL_COLLNET_ENABLE=0
export SHARP_COLL_ENABLE_SAT=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=160
export NCCL_PXN_DISABLE=1

# export MODELSCOPE_CACHE=/shared/packing_cache

master_port=29501
# master_addr=\${LOCAL_IP}
master_addr=\${CHIEF_IP}
# total_nodes_num=16

# log file dir
TENSORBOARD_DIR=\$HUNYUAN_TF_EVENTS_PATH
LOG_FILE=\$(dirname "\$TENSORBOARD_DIR")"/log/"
mkdir -p \${LOG_FILE}
current_time=\$(date "+%Y.%m.%d-%H.%M.%S")
current_log_file=\${LOG_FILE}/\${current_time}-\${INDEX}.txt
echo "current_log_file: \${current_log_file}"

mkdir -p \${HUNYUAN_CKPT_PATH}
mkdir -p \${HUNYUAN_TF_EVENTS_PATH}

# assert PP * TP * EP == NNODES * 8
PP=4
TP=8
EP=1

# python data_process.py
# source ~/.bashrc
pip install "qwen_vl_utils>=0.0.14" -U
export NNODES=${total_nodes_num}
export NODE_RANK=\${INDEX}
export MASTER_ADDR=\${master_addr}
export MASTER_PORT=29501
export NPROC_PER_NODE=8

echo "NODE_RANK:"
echo \${NODE_RANK}

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
IMAGE_MAX_TOKEN_NUM=16384 \
VIDEO_MAX_TOKEN_NUM=128 \
FPS_MAX_FRAMES=16 \

swift sft \
    --model \$HUNYUAN_MODEL_PATH \
    --train_type full \
    --dataset /apdcephfs_szcf/share_303378293/hunyuan/eiraouyang/repo_gongfeng/posttraing/project/multi_agent/data/训练数据/20260202/merge/merged_output_new_sp_fliter.jsonl \
    --load_from_cache_file true \
    --torch_dtype bfloat16 \
    --num_train_epochs 3 \
    --split_dataset_ratio 0.001 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --deepspeed zero2 \
    --gradient_accumulation_steps 2 \
    --eval_steps 100 \
    --save_steps 100 \
    --packing true \
    --save_total_limit 2 \
    --logging_steps 1 \
    --max_length 32768 \
    --output_dir \${HUNYUAN_CKPT_PATH} \
    --warmup_ratio 0.1 \
    --dataloader_num_workers 4 \
    --attn_impl flash_attn \

2>&1 | tee \${current_log_file}

exit_code=\$?
exit \$exit_code
EOF

# pssh -h ${share_pssh_host} -i "scp $DIR/$remote_script \$USER@\${HOSTNAME}:$RUNTIME_SCRIPT_DIR/$remote_script"

while IFS= read -r HOST; do
    echo "scp $RUNTIME_SCRIPT_DIR/$remote_script root@${HOST}:$RUNTIME_SCRIPT_DIR/$remote_script"
    scp $RUNTIME_SCRIPT_DIR/$remote_script "root@${HOST}:$RUNTIME_SCRIPT_DIR/$remote_script"
    if [ $? -eq 0 ]; then
        echo "复制到主机 $HOST 成功"
    else
        echo "复制到主机 $HOST 失败"
    fi
done < "$share_pssh_host"

# to avoid git clone
# pssh -i -t 0 -h pssh.hosts "sed -i '6d' ~/.bashrc"

pssh -h ${share_pssh_host} -i "[ -f $RUNTIME_SCRIPT_DIR/$remote_script ] && echo 'File exists' || echo 'File does not exist'"

# pssh -h ${share_pssh_host} -i "nohup bash $RUNTIME_SCRIPT_DIR/$remote_script > /dev/null 2>&1 &"
pssh -t 0 -h ${share_pssh_host} -i "bash $RUNTIME_SCRIPT_DIR/$remote_script"

# # pssh -t 0 -h pssh.hosts -i "pkill -f megatron"
exit_code=$?

# patch model file
# find "$HUNYUAN_CKPT_PATH" -type d -name "iter_*" | while read dir; do
#     echo "start merge ${dir}/../"

#     iter_name=$(basename "$dir")
#     iter_num=${iter_name#iter_}

#     swift export \
#         --mcore_model ${dir}/../ \
#         --device_map cpu \
#         --to_hf true \
#         --torch_dtype bfloat16 \
#         --output_dir ${HUNYUAN_CKPT_PATH}/checkpoint-${iter_num} \
#         --exist_ok true 

#     echo "end merge ${dir}/../"
# done

exit $exit_code
