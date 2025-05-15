### Note!!!!!!#########
rm -rf ./logs_fastchat/

set -e

HOST_ADDR=0.0.0.0
CONTROLER_PORT=28777
WORKER_BASE_PORT=30010

echo PYTHON_EXECUTABLE=$(which python3)
PYTHON_EXECUTABLE=$(which python3)

# Change MODEL_BASE POLICY_MODEL_NAME VALUE_MODEL_NAME
MODEL_BASE=/data1/home/jiawei/llm_rl/critic-rl/model/
CUDA_DEVICE_BASE=5
POLICY_MODEL_NAME=qwen25-math-sft-alltrain-1epoch-1_5b/checkpoint-12

# VALUE_MODEL_NAME=peiyi9979/math-shepherd-mistral-7b-prm
MODEL_PATH=$MODEL_BASE/$POLICY_MODEL_NAME
# VALUE_MODEL_PATH=$MODEL_BASE/$VALUE_MODEL_NAME

LOGDIR=logs_fastchat

tmux start-server
tmux new-session -s fastchat1 -n controller -d
tmux send-keys "export LOGDIR=${LOGDIR}" Enter
tmux send-keys "$PYTHON_EXECUTABLE -m fastchat.serve.controller --port ${CONTROLER_PORT} --host $HOST_ADDR" Enter

# Change NUM_LM_WORKER
NUM_LM_WORKER=1

echo "Wait 5 seconds ..."
sleep 5

# 定义GPU设备ID数组，要和NUM_LM_WORKER=？一致
CUDA_DEVICES="0"  # 比如在GPU 0和GPU 2上运行

echo "Starting workers"
for i in $(seq 0 $((NUM_LM_WORKER-1)))
do
  WORKER_PORT=$((WORKER_BASE_PORT+i))
  # 使用cut命令来获取对应的GPU ID
  CUDA_DEVICE=$(echo $CUDA_DEVICES | cut -d' ' -f$((i+1)))
  tmux new-window -n policy_worker_$i
  tmux send-keys "export LOGDIR=${LOGDIR}" Enter
  tmux send-keys "CUDA_VISIBLE_DEVICES=$CUDA_DEVICE $PYTHON_EXECUTABLE -m reason.llm_service.workers.vllm_worker --model-path $MODEL_PATH --controller-address http://$HOST_ADDR:$CONTROLER_PORT --host $HOST_ADDR --port $WORKER_PORT --worker-address http://$HOST_ADDR:$WORKER_PORT --dtype bfloat16" Enter
done
