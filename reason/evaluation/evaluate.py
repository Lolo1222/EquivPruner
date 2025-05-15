from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
from config.config_utils import str2bool
from reason.inference.lm_call import LMCallingConfig, VLLMRemoteCaller
from reason.inference.rm_call import (
    RMRemoteCaller,
    DummyRewardModelCaller,
    RewardModelBaseConfig,
    RemoteRewardModelConfig,
)
from reason.evaluation.evaluator import SolutionOutput, Task, RemoteMathEvaluator
import torch
from functools import partial
import json
import jsonlines
import time
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
import os
import random
from multiprocessing import Pool
import tree
from ray.util.actor_pool import ActorPool
from reason.evaluation.methods import *
import ray
from envs.human_eval.evaluator import consolidate_outputs, evaluate_human_eval_samples, save_results

import logging

# 配置 logging
# 创建处理器
file_handler = logging.FileHandler('logs_terminal/math500_vmcts_qwen_mprm_width10.log')
file_handler.setLevel(logging.INFO)  # 文件处理器接收 DEBUG 及以上级别
# file_handler.setLevel(logging.DEBUG)  # 文件处理器接收 DEBUG 及以上级别

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)  # 控制台处理器只接收 INFO 及以上级别

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        file_handler,  # 文件处理器
        console_handler  # 控制台处理器
    ]
)
logger = logging.getLogger(__name__)
logging.getLogger('urllib3').setLevel(logging.INFO)
logging.getLogger('filelock').setLevel(logging.INFO)

# 配置 logging
# logging.basicConfig(
#     level=logging.DEBUG,  # 设置根logger的级别为DEBUG
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler('logs_terminal/mathtrain6500_simple_mcts_qwen_mprm_width10.log', encoding='utf-8'),  # 文件处理器
#         logging.StreamHandler()  # 控制台处理器
#     ]
# )

# logger = logging.getLogger(__name__)
# logging.getLogger().setLevel(logging.DEBUG)  # 确保该logger的级别为DEBUG
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--LM", type=str, required=True)
    parser.add_argument("--RM", type=str, default="dummy")
    parser.add_argument("--controller_addr", type=str, default="http://0.0.0.0:28778")
    # task config
    parser.add_argument("--task_name", type=str, default="gsm8k")
    parser.add_argument("--test", type=str2bool, default=True)
    parser.add_argument("--is_few_shot", type=str2bool, default=False)
    parser.add_argument("--seed", type=int, default=0)
    # method config
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--num_sequence", type=int, default=1)
    # LM gen config
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    # Tree construction config
    parser.add_argument("--tree_max_depth", type=int, default=None)
    parser.add_argument("--tree_max_width", type=int, default=None)
    # ckpg config
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--resume_dir", type=str, default=None)
    parser.add_argument("--output", action="store_true", default=False)
    parser.add_argument("--output_dir", type=str, default=None)
    # parallel config
    parser.add_argument("--local", action="store_true", default=False)
    parser.add_argument("--num_worker", type=int, default=32)
    # simple mcts config
    parser.add_argument("--num_simulations", type=int, default=3)
    parser.add_argument("--is_merge", action="store_true", default=False)
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--metric", type=str, default="levenshtein", choices=["model_merge", "levenshtein"])
    # HumanEval特定配置
    parser.add_argument("--k_values", type=str, default="1,10,100", help="HumanEval中的pass@k值列表，用逗号分隔")
    config = parser.parse_args()

    setup_seed(config.seed)
    # output_dir = "beam_train_math_output_files"
    if config.output:
        os.makedirs(config.output_dir, exist_ok=True)      
    if config.local:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("run in pure local mode for debug only")
        config.num_worker = 1
        ray.init(local_mode=True)
    else:
        ###########Notice!!!!!!!!!!!!!!!!!!!
        # logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger().setLevel(logging.INFO)
    # TODO(ziyu): move into some configuration file
    if "math-shepherd" in config.RM.lower():
        prm_step_tag = "ки\n"
    else:
        # assume qwen
        prm_step_tag = "\n\n\n\n\n "
    prm_format_str = "{question} {answer}"

    if "qwen" in config.LM.lower():
        lm_step_tag = "\n\n"
    else:
        lm_step_tag = "ки\n"

    llm_gen_fn = VLLMRemoteCaller(
        config.LM, config.controller_addr, lm_step_tag=lm_step_tag
    )
    if config.RM == "dummy":
        rm_config = RewardModelBaseConfig(
            step_tag=prm_step_tag, format_str=prm_format_str
        )
        rm_call = DummyRewardModelCaller(rm_config)
    else:
        rm_config = RemoteRewardModelConfig(
            step_tag=prm_step_tag,
            format_str=prm_format_str,
            model_name=config.RM,
            controller_addr=config.controller_addr,
        )
        rm_call = RMRemoteCaller(rm_config)

    task = Task(task_name=config.task_name, is_few_shot=config.is_few_shot)

    def parallel_evaluate_test_dataset(
        method_name: str, solver_fn: Callable, save_dir: Optional[Path] = None
    ) -> List[Dict[str, Any]]:
        if save_dir is not None:
            record_writer = jsonlines.open(save_dir / f"record.jsonl", mode="w")
        else:
            record_writer = None

        test_ds = task.test_ds
        # test_ds = [test_ds[i] for i in range(32)]

        results = []
        problem_insts = []
        outputs = []
        
        if config.resume_dir is not None:
            answered_questions = set()
            with jsonlines.open(
                Path(config.resume_dir) / "record.jsonl", "r"
            ) as reader:
                cnt = 0
                for obj in reader:
                    results.append(obj["result"])
                    problem_insts.append({"question": obj["question"], "answer": obj["groundtruth"]})
                    outputs.append({"output": obj["output"]})
                    answered_questions.add(obj["question"])
                    if record_writer is not None:
                        record_writer.write(obj)
                        cnt += 1
            print(f"Resumed {cnt} questions from {config.resume_dir}")
            total_cnt = len(test_ds)
            test_ds = [
                problem_inst
                for problem_inst in test_ds
                if problem_inst["question"] not in answered_questions
            ]
            new_cnt = len(test_ds)
            print(
                f"After resuming, there are {new_cnt}/{total_cnt} new questions to answer."
            )

        actor_pool = ActorPool(
            [
                RemoteMathEvaluator.remote(config.task_name, llm_gen_fn, rm_call)
                for _ in range(config.num_worker)
            ]
        )
        res_q = actor_pool.map_unordered(
            lambda p, x: p.evaluate_problem.remote(x, solver_fn), test_ds
        )       # Distributes tasks from the test_ds dataset across the worker pool asynchronously and
                # collects results in any order as they complete. Every worker has a new searching tree as we reset the
                # tree in solver_fn
        if config.output is not True:
            # Lolo1222: Dont output candidate
            print("Dont output candidate.")
            for i, (problem_inst, result, output) in enumerate(
                tqdm(res_q, total=len(test_ds))
            ):
                results.append(result)
                problem_insts.append(problem_inst)
                outputs.append({"output": output[0]["text"] if output else ""})
                if record_writer:
                    obj = {
                        # "i": i,
                        "question": problem_inst["question"],
                        "groundtruth": problem_inst["answer"],
                        "result": result,
                        "output": output,
                    }
                    record_writer.write(obj)
        else:
            # Lolo1222: output candidate in output_dir.
            for i, (problem_inst, result, output, candidate_list) in enumerate(
                tqdm(res_q, total=len(test_ds))
            ):
                results.append(result)
                problem_insts.append(problem_inst)
                outputs.append({"output": output[0]["text"] if output else ""})
                output_file = os.path.join(config.output_dir, f"question_{i}.json")
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(candidate_list, f, ensure_ascii=False)
                    # for sublist in candidate_list:
                    #     f.write(json.dumps(sublist, ensure_ascii=False) + '\n')                    
                # with open(output_file, 'r') as f:
                #     loaded_data = json.load(f)                
                if record_writer:
                    obj = {
                        # "i": i,
                        "question": problem_inst["question"],
                        "groundtruth": problem_inst["answer"],
                        "result": result,
                        "output": output,
                    }
                    record_writer.write(obj)
        
        # 计算平均结果
        avg_res = (tree.map_structure(lambda *xs: np.mean(xs), *results),)
        if record_writer:
            json.dump(avg_res, open(save_dir / "avg_result.json", "w"))
        print("Method: {}. Average result: {}".format(method_name, avg_res))
        
        # 如果是HumanEval，额外计算pass@k指标
        if config.task_name == "human_eval" and save_dir is not None:
            try:
                from envs.human_eval.evaluator import consolidate_outputs, evaluate_human_eval_samples, save_results
                
                # 将所有problem_insts和outputs处理为HumanEval格式
                print("Processing HumanEval outputs for pass@k evaluation...")
                
                # 处理已有的数据
                solutions = consolidate_outputs(outputs, problem_insts)
                
                # 计算pass@k指标
                k_values = [int(k) for k in config.k_values.split(",")]
                print(f"Calculating pass@k for k values: {k_values}")
                try:
                    eval_results, detailed_results = evaluate_human_eval_samples(solutions, k_values)
                    
                    # 保存结果
                    human_eval_metrics_dir = save_dir / "human_eval_metrics"
                    summary = save_results(eval_results, detailed_results, human_eval_metrics_dir)
                    print(f"HumanEval评估结果: {summary}")
                except Exception as e:
                    print(f"Error evaluating HumanEval samples: {e}")
            except Exception as e:
                print(f"Error during HumanEval pass@k calculation: {e}")
        
        return results

    solver_fns = {"cot": cot, "best_of_n": best_of_n}

    cfg_dict_record = dict()
    # XXX: qwen-2.5 requires add more stop words
    # not do it now.
    # stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]
    gen_config = LMCallingConfig(
        n=config.num_sequence,
        temperature=config.temperature,
        top_k=config.top_k,
        top_p=config.top_p,
        max_new_tokens=config.max_new_tokens,
    )
    cfg_dict_record["gen_config"] = gen_config.__dict__

    if config.method == "cot":
        method_config = CoTConfig(config.task_name)
        solver_fn = partial(cot, method_config, gen_config)
    elif config.method == "best_of_n":
        method_config = BestOfNConfig(
            config.task_name, num_sequence=config.num_sequence
        )
        solver_fn = partial(best_of_n, method_config, gen_config)
    elif config.method == "beam_search":
        method_config = BeamSearchConfig(
            task_name=config.task_name,
            tree_max_depth=config.tree_max_depth,
            tree_max_width=config.tree_max_width,
            beam_size=config.num_sequence,
        )
        solver_fn = partial(beam_search, method_config, gen_config)
    elif config.method == "vanila_mcts":
        method_config = VanilaMCTSConfig(
            task_name=config.task_name,
            tree_max_depth=config.tree_max_depth,
            tree_max_width=config.tree_max_width,
            select_by_prior=False,
            num_path=config.num_sequence,
        )
        solver_fn = partial(vanila_mcts, method_config, gen_config)
    elif config.method == "simple_mcts":
        method_config = SimpleMCTSConfig(
            task_name=config.task_name,
            tree_max_depth=config.tree_max_depth,
            tree_max_width=config.tree_max_width,
            select_by_prior=False,
            num_path=config.num_sequence,
            num_simulations=config.num_simulations,
            is_merge=config.is_merge,
            threshold=config.threshold,
            metric=config.metric,
        )
        solver_fn = partial(simple_mcts, method_config, gen_config)
    elif config.method == "rstar_mcts":
        # XXX(Lolo1222): Note! Why this is not RStarMCTSConfig?
        method_config = RStarMCTSConfig(
            task_name=config.task_name,
            tree_max_depth=config.tree_max_depth,
            tree_max_width=config.tree_max_width,
            select_by_prior=False,
            num_path=config.num_sequence,
            is_merge=config.is_merge,
            threshold=config.threshold,
            metric=config.metric,            
        )
        solver_fn = partial(rstar_mcts, method_config, gen_config)

    else:
        raise ValueError(f"Unknown method: {config.method}")
    cfg_dict_record["method"] = config.method
    cfg_dict_record["method_config"] = method_config.__dict__

    if config.save_dir is not None:
        datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = Path(config.save_dir) / task.task_name / config.method / datetime_str
        save_dir.mkdir(parents=True)
        record_writer = jsonlines.open(save_dir / f"record.jsonl", mode="w")
        cfg_dict_record["LM"] = config.LM
        cfg_dict_record["RM"] = config.RM
        json.dump(cfg_dict_record, open(save_dir / "config.json", "w"))
    else:
        save_dir = None

    parallel_evaluate_test_dataset(config.method, solver_fn, save_dir)
