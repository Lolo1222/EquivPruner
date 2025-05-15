在终端中运行：
   chmod +x run_with_remote_merge.sh
   ./run_with_remote_merge.sh /data1/home/jiawei/llmmcts/merge/results/longformer_results/checkpoint-24265 binary
这将部署三个服务：policy model、reward model和merge model，并设置环境变量使is_similar_str_pair函数使用远程服务而不是本地模型。
然后它会自动运行scripts/eval/simple_mcts.sh脚本，该脚本将使用远程merge模型进行相似度比较，而不是本地加载模型。
这样实现的好处是：
不修改原有代码，只是添加了新的功能
可以轻松切换回本地模式（只需设置USE_REMOTE_MERGE_MODEL=false）
实现了类似于policy model和reward model的高效通信方式
提供了错误重试和优雅降级机制

merge_infer_fns.py文件中有prompt