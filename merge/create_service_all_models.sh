#!/bin/bash
### Note!!!!!!#########
rm -rf ./logs_fastchat/
set -e

HOST_ADDR=0.0.0.0
CONTROLER_PORT=28777
WORKER_BASE_PORT=30010

echo PYTHON_EXECUTABLE=$(which python3)
PYTHON_EXECUTABLE=$(which python3)

# 定义模型路径
MODEL_BASE=/data2/OpenLLMs
CUDA_DEVICE_BASE=4

# 定义各个模型名称
POLICY_MODEL_NAME=peiyi9979/mistral-7b-sft
VALUE_MODEL_NAME=peiyi9979/math-shepherd-mistral-7b-prm
MERGE_MODEL_NAME=/data1/home/jiawei/llmmcts/merge/results/longformer_results/checkpoint-24265

# 完整模型路径
MODEL_PATH=$MODEL_BASE/$POLICY_MODEL_NAME
VALUE_MODEL_PATH=$MODEL_BASE/$VALUE_MODEL_NAME
MERGE_MODEL_PATH=$MERGE_MODEL_NAME  # 直接使用完整路径

# 输出类型，可选 binary, similarity, qwen
MERGE_OUTPUT_TYPE="binary"

# 日志目录
LOGDIR=logs_fastchat

# 创建目录（如果不存在）
mkdir -p merge

# 部署工作节点数量
NUM_LM_WORKER=2
NUM_RM_WORKER=1
NUM_MERGE_WORKER=1

# 启动tmux session
tmux start-server
tmux new-session -s fastchat2 -n controller -d
tmux send-keys "export LOGDIR=${LOGDIR}" Enter
tmux send-keys "$PYTHON_EXECUTABLE -m fastchat.serve.controller --port ${CONTROLER_PORT} --host $HOST_ADDR" Enter

echo "等待10秒以启动控制器..."
sleep 5

# 为policy model定义GPU设备
POLICY_CUDA_DEVICES="1 2"  # 在GPU 2上运行

echo "启动policy模型工作节点..."
for i in $(seq 0 $((NUM_LM_WORKER-1)))
do
  WORKER_PORT=$((WORKER_BASE_PORT+i))
  # 获取对应的GPU ID
  CUDA_DEVICE=$(echo $POLICY_CUDA_DEVICES | cut -d' ' -f$((i+1)))
  tmux new-window -n policy_worker_$i
  tmux send-keys "export LOGDIR=${LOGDIR}" Enter
  tmux send-keys "CUDA_VISIBLE_DEVICES=$CUDA_DEVICE $PYTHON_EXECUTABLE -m reason.llm_service.workers.vllm_worker --model-path $MODEL_PATH --controller-address http://$HOST_ADDR:$CONTROLER_PORT --host $HOST_ADDR --port $WORKER_PORT --worker-address http://$HOST_ADDR:$WORKER_PORT --dtype bfloat16" Enter
done

# 为reward model定义GPU设备
REWARD_CUDA_DEVICES="3"  # 在GPU 3上运行

echo "启动reward模型工作节点..."
for i in $(seq 0 $((NUM_RM_WORKER-1)))
do
  WORKER_PORT=$((i+WORKER_BASE_PORT+NUM_LM_WORKER))
  # 获取对应的GPU ID
  REWARD_CUDA_DEVICE=$(echo $REWARD_CUDA_DEVICES | cut -d' ' -f$((i+1)))
  tmux new-window -n reward_worker_$i
  tmux send-keys "export LOGDIR=${LOGDIR}" Enter
  tmux send-keys "CUDA_VISIBLE_DEVICES=$REWARD_CUDA_DEVICE $PYTHON_EXECUTABLE -m reason.llm_service.workers.reward_model_worker --model-path $VALUE_MODEL_PATH --controller-address http://$HOST_ADDR:$CONTROLER_PORT --host $HOST_ADDR --port $WORKER_PORT --worker-address http://$HOST_ADDR:$WORKER_PORT" Enter
done

# 为merge model定义GPU设备
MERGE_CUDA_DEVICES="2"  # 在GPU 1上运行

echo "启动merge模型工作节点..."
for i in $(seq 0 $((NUM_MERGE_WORKER-1)))
do
  WORKER_PORT=$((i+WORKER_BASE_PORT+NUM_LM_WORKER+NUM_RM_WORKER))
  # 获取对应的GPU ID
  MERGE_CUDA_DEVICE=$(echo $MERGE_CUDA_DEVICES | cut -d' ' -f$((i+1)))
  tmux new-window -n merge_worker_$i
  tmux send-keys "export LOGDIR=${LOGDIR}" Enter
  tmux send-keys "CUDA_VISIBLE_DEVICES=$MERGE_CUDA_DEVICE $PYTHON_EXECUTABLE -m merge.merge_model_worker --model-path $MERGE_MODEL_PATH --output-type $MERGE_OUTPUT_TYPE --controller-address http://$HOST_ADDR:$CONTROLER_PORT --host $HOST_ADDR --port $WORKER_PORT --worker-address http://$HOST_ADDR:$WORKER_PORT --model-names merge-model" Enter
done

echo "所有模型服务已部署完成！"
echo "使用 tmux attach -t fastchat2 查看服务状态"