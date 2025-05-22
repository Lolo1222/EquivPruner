"""
A model worker for merge models that execute similarity judgments.
"""

import argparse
import gc
import os
import uuid
import functools
import torch
import asyncio
import time
import requests
import threading
import json
from typing import List, Optional, Dict, Any, Union
import uvicorn

from fastchat.constants import ErrorCode, SERVER_ERROR_MSG
from fastchat.model.model_adapter import load_model, add_model_args
from fastchat.modules.awq import AWQConfig
from fastchat.modules.exllama import ExllamaConfig
from fastchat.modules.xfastertransformer import XftConfig
from fastchat.modules.gptq import GptqConfig
from reason.llm_service.workers.base_model_worker import BaseModelWorker, app
from fastchat.utils import build_logger, get_context_length, str_to_torch_dtype
from merge.merge_infer_fns import get_merge_model_infer_fn

worker_id = str(uuid.uuid4())[:8]
logger = build_logger("merge_model_worker", f"merge_model_worker_{worker_id}.log")

class ModelWorker(BaseModelWorker):
    def __init__(
        self,
        controller_addr: str,
        worker_addr: str,
        worker_id: str,
        model_path: str,
        model_names: List[str],
        limit_worker_concurrency: int,
        no_register: bool,
        device: str,
        num_gpus: int,
        max_gpu_memory: str,
        output_type: str = "binary",
        dtype: Optional[torch.dtype] = None,
        load_8bit: bool = False,
        cpu_offloading: bool = False,
        gptq_config: Optional[GptqConfig] = None,
        awq_config: Optional[AWQConfig] = None,
        exllama_config: Optional[ExllamaConfig] = None,
        xft_config: Optional[XftConfig] = None,
        conv_template: Optional[str] = None,
        embed_in_truncate: bool = False,
        seed: Optional[int] = None,
        debug: bool = False,
        **kwargs,
    ):
        super().__init__(
            controller_addr,
            worker_addr,
            worker_id,
            model_path,
            model_names,
            limit_worker_concurrency,
            conv_template=conv_template,
        )

        logger.info(f"Loading the merge model {self.model_names} on worker {worker_id} ...")
        if 'qwen' not in model_path and 'llama' not in model_path:
            from transformers import LongformerTokenizer, LongformerForSequenceClassification
            self.model = LongformerForSequenceClassification.from_pretrained(model_path).to(device)
            self.tokenizer = LongformerTokenizer.from_pretrained(model_path)
        else:
            self.model, self.tokenizer = load_model(
                model_path,
                device=device,
                num_gpus=num_gpus,
                max_gpu_memory=max_gpu_memory,
                dtype=dtype,
                load_8bit=load_8bit,
                cpu_offloading=cpu_offloading,
                gptq_config=gptq_config,
                awq_config=awq_config,
                exllama_config=exllama_config,
                xft_config=xft_config,
                debug=debug,
            )
        self.device = device
        self.seed = seed
        self.output_type = output_type

        if not no_register:
            self.init_heart_beat()

        # 获取对应的推理函数
        self.infer_fn = functools.partial(
            get_merge_model_infer_fn(model_path, output_type),
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device
        )

    @torch.inference_mode()
    def merge_model_inference(self, params):
        """处理相似度判断请求"""
        max_tries = 3
        tries = 0
        
        while tries < max_tries:
            tries += 1
            try:
                input_pairs = params["input_pairs"]
                is_single_pair = isinstance(input_pairs[0], str)
                
                if is_single_pair:
                    # 单对字符串输入
                    results = self.infer_fn(input_pairs)
                    # 根据输出类型处理结果
                    if self.output_type == "binary":
                        # 二分类结果，返回预测标签
                        predicted_label = torch.argmax(results, dim=1).item()
                        ret = {"input": input_pairs, "is_similar": bool(predicted_label)}
                    elif self.output_type == "similarity":
                        # 相似度分数
                        similarity_score = results.item()
                        ret = {"input": input_pairs, "similarity_score": similarity_score}
                    else:  # qwen输出
                        predicted_label = torch.argmax(results, dim=1).item()
                        ret = {"input": input_pairs, "is_similar": bool(predicted_label)}
                else:
                    # 多对字符串输入
                    all_results = []
                    for pair in input_pairs:
                        result = self.infer_fn(pair)
                        
                        if self.output_type == "binary":
                            predicted_label = torch.argmax(result, dim=1).item()
                            all_results.append(bool(predicted_label))
                        elif self.output_type == "similarity":
                            similarity_score = result.item()
                            all_results.append(similarity_score)
                        else:  # qwen输出
                            predicted_label = torch.argmax(result, dim=1).item()
                            all_results.append(bool(predicted_label))
                    
                    if self.output_type == "binary" or self.output_type == "qwen":
                        ret = {"input": input_pairs, "is_similar_list": all_results}
                    else:
                        ret = {"input": input_pairs, "similarity_scores": all_results}
                
                # 成功处理请求，清理内存
                gc.collect()
                torch.cuda.empty_cache()
                return ret
            
            except torch.cuda.OutOfMemoryError as e:
                if tries < max_tries:
                    # 清理内存并重试
                    gc.collect()
                    torch.cuda.empty_cache()
                    time.sleep(1)
                    logger.warning(f"CUDA OOM error, retrying ({tries}/{max_tries}): {e}")
                else:
                    # 达到最大重试次数
                    ret = {
                        "text": f"{SERVER_ERROR_MSG}\n\n({e})",
                        "error_code": ErrorCode.CUDA_OUT_OF_MEMORY,
                    }
                    return ret
            
            except Exception as e:
                if tries < max_tries:
                    # 其他错误也重试
                    logger.warning(f"Error occurred, retrying ({tries}/{max_tries}): {e}")
                    time.sleep(1)
                else:
                    ret = {
                        "text": f"{SERVER_ERROR_MSG}\n\n({e})",
                        "error_code": ErrorCode.INTERNAL_ERROR,
                    }
                    return ret


def create_model_worker():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21002)
    parser.add_argument("--worker-address", type=str, default="http://localhost:21002")
    parser.add_argument(
        "--controller-address", type=str, default="http://localhost:21001"
    )
    add_model_args(parser)
    parser.add_argument(
        "--model-names",
        type=lambda s: s.split(","),
        help="Optional display comma separated names",
    )
    parser.add_argument(
        "--output-type",
        type=str,
        default="binary",
        choices=["binary", "similarity", "qwen"],
        help="Output type of merge model: binary (two-class probabilities), similarity (similarity score), or qwen (binary classification)"
    )
    parser.add_argument(
        "--conv-template", type=str, default=None, help="Conversation prompt template."
    )
    parser.add_argument("--embed-in-truncate", action="store_true")
    parser.add_argument(
        "--limit-worker-concurrency",
        type=int,
        default=5,
        help="Limit the model concurrency to prevent OOM.",
    )
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Overwrite the random seed for each generation.",
    )
    parser.add_argument(
        "--debug", type=bool, default=False, help="Print debugging messages"
    )
    parser.add_argument(
        "--ssl",
        action="store_true",
        required=False,
        default=False,
        help="Enable SSL. Requires OS Environment variables 'SSL_KEYFILE' and 'SSL_CERTFILE'.",
    )
    args = parser.parse_args()
    logger.info(f"args: {args}")

    if args.gpus:
        if len(args.gpus.split(",")) < args.num_gpus:
            raise ValueError(
                f"Larger --num-gpus ({args.num_gpus}) than --gpus {args.gpus}!"
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    if args.enable_exllama:
        exllama_config = ExllamaConfig(
            max_seq_len=args.exllama_max_seq_len,
            gpu_split=args.exllama_gpu_split,
            cache_8bit=args.exllama_cache_8bit,
        )
    else:
        exllama_config = None
    if args.enable_xft:
        xft_config = XftConfig(
            max_seq_len=args.xft_max_seq_len,
            data_type=args.xft_dtype,
        )
        if args.device != "cpu":
            print("xFasterTransformer now is only support CPUs. Reset device to CPU")
            args.device = "cpu"
    else:
        xft_config = None        
    worker = ModelWorker(
        controller_addr=args.controller_address,
        worker_addr=args.worker_address,
        worker_id=str(uuid.uuid4())[:8],
        model_path=args.model_path,
        model_names=args.model_names,
        limit_worker_concurrency=args.limit_worker_concurrency,
        no_register=args.no_register,
        device=args.device,
        num_gpus=args.num_gpus,
        max_gpu_memory=args.max_gpu_memory,
        output_type=args.output_type,
        dtype=str_to_torch_dtype(args.dtype),
        load_8bit=args.load_8bit,
        cpu_offloading=args.cpu_offloading,
        gptq_config=GptqConfig(
            ckpt=args.gptq_ckpt or args.model_path,
            wbits=args.gptq_wbits,
            groupsize=args.gptq_groupsize,
            act_order=args.gptq_act_order,
        ),
        awq_config=AWQConfig(
            ckpt=args.awq_ckpt or args.model_path,
            wbits=args.awq_wbits,
            groupsize=args.awq_groupsize,
        ),
        exllama_config=exllama_config,
        xft_config=xft_config,
        seed=args.seed,
        debug=args.debug,
    )

    return args, worker

def acquire_worker_semaphore():
    if worker.semaphore is None:
        worker.semaphore = asyncio.Semaphore(worker.limit_worker_concurrency)
    return worker.semaphore.acquire()
# 添加特定路由
@app.post("/merge_model_inference")
async def merge_model_inference(request: Dict[str, Any]):
    await acquire_worker_semaphore()
    try:
        ret = worker.merge_model_inference(request)
        return ret
    finally:
        worker.semaphore.release()


if __name__ == "__main__":
    args, worker = create_model_worker()
    if args.ssl:
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level="info",
            ssl_keyfile=os.environ["SSL_KEYFILE"],
            ssl_certfile=os.environ["SSL_CERTFILE"],
        )
    else:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")