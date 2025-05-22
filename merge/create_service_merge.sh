#!/bin/bash
set -e

export USE_REMOTE_MERGE_MODEL=true
export MERGE_MODEL_NAME=merge-model
CONTROLLER_ADDR=http://0.0.0.0:28777
HOST_ADDR=0.0.0.0
CONTROLER_PORT=28777
WORKER_BASE_PORT=30010

echo PYTHON_EXECUTABLE=$(which python3)
PYTHON_EXECUTABLE=$(which python3)

# Change MODEL_BASE POLICY_MODEL_NAME VALUE_MODEL_NAME
MODEL_BASE=/data2/OpenLLMs
POLICY_MODEL_NAME=Qwen2.5-Math-7B-Instruct
# POLICY_MODEL_NAME=peiyi9979/mistral-7b-sft
VALUE_MODEL_NAME=peiyi9979/math-shepherd-mistral-7b-prm

MODEL_PATH=$MODEL_BASE/$POLICY_MODEL_NAME
VALUE_MODEL_PATH=$MODEL_BASE/$VALUE_MODEL_NAME
MERGE_MODEL_PATH=./checkpoint-24265  # Change to your merge model path

# output type: binary, similarity, qwen
MERGE_OUTPUT_TYPE="binary"

# log directory
LOGDIR=logs_fastchat

# create directory (if not exists)
mkdir -p merge

# Change NUM_LM_WORKER NUM_RM_WORKER
NUM_LM_WORKER=2
NUM_RM_WORKER=2
NUM_MERGE_WORKER=2

# 启动tmux session
tmux start-server
tmux new-session -s fastchat2 -n controller -d
tmux send-keys "export LOGDIR=${LOGDIR}" Enter
tmux send-keys "$PYTHON_EXECUTABLE -m fastchat.serve.controller --port ${CONTROLER_PORT} --host $HOST_ADDR" Enter

echo "Wait 10 seconds ..."
sleep 5

# Define GPU devices for policy model
POLICY_CUDA_DEVICES="0 1"  # Run on GPU 2

echo "Start policy model worker nodes..."
for i in $(seq 0 $((NUM_LM_WORKER-1)))
do
  WORKER_PORT=$((WORKER_BASE_PORT+i))
  # Get corresponding GPU ID
  CUDA_DEVICE=$(echo $POLICY_CUDA_DEVICES | cut -d' ' -f$((i+1)))
  tmux new-window -n policy_worker_$i
  tmux send-keys "export LOGDIR=${LOGDIR}" Enter
  tmux send-keys "CUDA_VISIBLE_DEVICES=$CUDA_DEVICE $PYTHON_EXECUTABLE -m reason.llm_service.workers.vllm_worker --model-path $MODEL_PATH --controller-address http://$HOST_ADDR:$CONTROLER_PORT --host $HOST_ADDR --port $WORKER_PORT --worker-address http://$HOST_ADDR:$WORKER_PORT --dtype bfloat16" Enter
done

# Define GPU devices for reward model
REWARD_CUDA_DEVICES="2 3"  # Run on GPU 2,3

echo "Start reward model worker nodes..."
for i in $(seq 0 $((NUM_RM_WORKER-1)))
do
  WORKER_PORT=$((i+WORKER_BASE_PORT+NUM_LM_WORKER))
  # Get corresponding GPU ID
  REWARD_CUDA_DEVICE=$(echo $REWARD_CUDA_DEVICES | cut -d' ' -f$((i+1)))
  tmux new-window -n reward_worker_$i
  tmux send-keys "export LOGDIR=${LOGDIR}" Enter
  tmux send-keys "CUDA_VISIBLE_DEVICES=$REWARD_CUDA_DEVICE $PYTHON_EXECUTABLE -m reason.llm_service.workers.reward_model_worker --model-path $VALUE_MODEL_PATH --controller-address http://$HOST_ADDR:$CONTROLER_PORT --host $HOST_ADDR --port $WORKER_PORT --worker-address http://$HOST_ADDR:$WORKER_PORT" Enter
done
# Define GPU devices for merge model
MERGE_CUDA_DEVICES="4 5"  # Run on GPU 4,5

echo "Start merge model worker nodes..."
for i in $(seq 0 $((NUM_MERGE_WORKER-1)))
do
  WORKER_PORT=$((i+WORKER_BASE_PORT+NUM_LM_WORKER+NUM_RM_WORKER))
  # Get corresponding GPU ID
  MERGE_CUDA_DEVICE=$(echo $MERGE_CUDA_DEVICES | cut -d' ' -f$((i+1)))
  tmux new-window -n merge_worker_$i
  tmux send-keys "export LOGDIR=${LOGDIR}" Enter
  tmux send-keys "CUDA_VISIBLE_DEVICES=$MERGE_CUDA_DEVICE $PYTHON_EXECUTABLE -m merge.merge_model_worker --model-path $MERGE_MODEL_PATH --output-type $MERGE_OUTPUT_TYPE --controller-address http://$HOST_ADDR:$CONTROLER_PORT --host $HOST_ADDR --port $WORKER_PORT --worker-address http://$HOST_ADDR:$WORKER_PORT --model-names merge-model" Enter
done

echo "All model services deployed!"
echo "Use tmux attach -t fastchat2 to view service status"