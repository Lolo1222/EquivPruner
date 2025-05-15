export PYTHONPATH=$(pwd)
python reason/evaluation/evaluate.py \
    --LM Qwen2.5-Math-7B-Instruct \
    --RM math-shepherd-mistral-7b-prm \
    --task_name MATH \
    --temperature 0.7 \
    --max_new_tokens 1024 \
    --num_sequence 1 \
    --tree_max_width 10 \
    --tree_max_depth 50 \
    --save_dir qwen_mprm_result \
    --method simple_mcts \
    --num_worker 32 \
    --num_simulations 20 \
    --is_merge \
    --metric model_merge \
    --controller_addr http://0.0.0.0:28777

    # --resume_dir /data1/home/jiawei/llmmcts/openr/merge_qwen_mprm_result/MATH/simple_mcts/20250415_152135 \