# #!/bin/bash

# # 检查是否提供了模型路径参数
# if [ -z "$1" ]; then
#     echo "请提供merge模型路径，例如: ./run_with_remote_merge.sh /path/to/merge/model"
#     exit 1
# fi

# # 设置模型路径和输出类型
# export MERGE_MODEL_PATH=$1
# export MERGE_OUTPUT_TYPE=${2:-"binary"}  # 默认为binary
# export USE_REMOTE_MERGE_MODEL=true
# export CONTROLLER_ADDR=http://0.0.0.0:28777
# export MERGE_MODEL_NAME=merge-model

# # 运行部署服务脚本
# echo "开始部署所有模型服务..."
# bash ./create_service_all_models.sh

# # 等待服务启动
# echo "等待服务完全启动..."
# sleep 10

# # 运行测试脚本
# echo "开始运行evaluation..."
# cd ../scripts/eval/
# bash ./simple_mcts.sh

# # 完成后提示
# echo "运行完成！"
# echo "您可以使用'tmux attach -t fastchat2'查看服务状态"
# echo "使用'tmux kill-session -t fastchat2'关闭所有服务"