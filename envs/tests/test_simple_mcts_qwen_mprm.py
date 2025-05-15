import sys
import os

# 获取当前文件的绝对路径
current_file_path = os.path.abspath(__file__)
# 获取项目根目录的路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
# 将项目根目录添加到 sys.path
sys.path.append(project_root)

from envs.MATH import (
    Env,
    get_train_test_dataset,
    extract_groundtruth,
    judge_correct,
    extract_answer,
)

from transformers import AutoTokenizer
from reason.evaluation.methods import SimpleMCTSConfig

from reason.inference.lm_call import VLLMRemoteCaller
from reason.inference.rm_call import DummyRewardModelCaller,RMRemoteCaller,RemoteRewardModelConfig

import functools

from reason.evaluation.evaluator import SolutionOutput, Task, TreeSearchSolutionOutput
from reason.guided_search.tree import SearchTree

from reason.evaluation.methods import cot
import jsonlines
import tree
import numpy as np
from tqdm import tqdm

# Import evaluate to ensure logging is configured
import reason.evaluation.evaluate
import logging

# 获取全局 logger
logger = logging.getLogger('reason.evaluation.evaluate')
logger.setLevel(logging.DEBUG)
# logger.setLevel(logging.INFO)
# 创建文件处理器
file_handler = logging.FileHandler('logs_terminal/debug.log')
file_handler.setLevel(logging.DEBUG)

# 创建格式化器
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# 将处理器添加到logger
logger.addHandler(file_handler)

if __name__ == "__main__":

    method_config = SimpleMCTSConfig(
                task_name='MATH',
                tree_max_depth=50,
                tree_max_width=4,
                select_by_prior=False,
                num_path=1,
                num_simulations=4,
            )    
    _, ds = get_train_test_dataset()
    print(len(ds))
    # print(ds[1])
    problem = ds[1]
    lm_call = VLLMRemoteCaller("Qwen2.5-Math-7B-Instruct", "http://0.0.0.0:28777")
    env = Env(
        config={
            "max_actions": method_config.tree_max_width,
            "max_length": 50,
            "generation_config": {
                "max_new_tokens": 512,
                "temperature": 0.8,
                "top_p": 1.0,
                "top_k": 50,
            },
        },
        math_problems=[
            {
                "question": problem["question"],
                "answer": extract_groundtruth(problem["answer"]),
            }
        ],
        # llm_gen_fn=VLLMRemoteCaller("Qwen2.5-Math-1.5B-Instruct", "http://0.0.0.0:28777"),
        llm_gen_fn=lm_call,
        reset=False,
    )

    rm_config = RemoteRewardModelConfig(
        step_tag="ки\n",
        format_str="{question} {answer}",
        model_name='math-shepherd-mistral-7b-prm',
        controller_addr='http://0.0.0.0:28777',
    )
    rm_call = RMRemoteCaller(rm_config)    
    # task = Task("MATH")

    search_tree = SearchTree(
        cfg={
            "pb_c_base": method_config.pb_c_base,
            "pb_c_init": method_config.pb_c_init,
            "init_critic_value": method_config.init_critic_value,
            "num_simulations": method_config.num_simulations,
        }
    )
    rm_call_fn = functools.partial(rm_call, lm_step_tag=lm_call.lm_step_tag)
    traj_list = search_tree.simple_mcts(
        simulate_env=env,
        num_path=method_config.num_path,
        reward_model_fn=rm_call_fn,
        select_by_prior=False
    )

    # print(TreeSearchSolutionOutput(
    #     solutions=[t["text"] for t in traj_list],
    #     completion_tokens=[t["api_completion_tokens"] for t in traj_list],
    #     tree_completion_tokens=[t["tree_completion_tokens"] for t in traj_list],
    # ))