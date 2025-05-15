import json
import os
import requests
import time
from typing import List, Optional, Dict, Any, Union, Tuple

class MergeModelClient:
    """
    用于调用远程merge model服务的客户端
    """
    def __init__(
        self,
        model_name: str,
        controller_address: str = "http://localhost:21001",
        max_retry: int = 3,
    ):
        self.model_name = model_name
        self.controller_address = controller_address
        self.max_retry = max_retry
        
    def get_worker_address(self) -> str:
        """获取可用的worker地址"""
        controller_address = self.controller_address
        
        url = controller_address + "/get_worker_address"
        data = {
            "model": self.model_name
        }
        
        for retry in range(self.max_retry):
            try:
                response = requests.post(url, json=data, timeout=10)
                if response.status_code == 200:
                    return response.json()["address"]
                else:
                    error_msg = f"Get worker address failed with status code: {response.status_code}"
                    if retry == self.max_retry - 1:
                        raise RuntimeError(error_msg)
                    else:
                        print(f"Warning: {error_msg}, retrying...")
                        time.sleep(1)
            except Exception as e:
                if retry == self.max_retry - 1:
                    raise RuntimeError(f"Failed to get worker address: {e}")
                else:
                    print(f"Warning: Failed to get worker address: {e}, retrying...")
                    time.sleep(1)
        
        raise RuntimeError("Failed to get worker address after retries")
    
    def is_similar(self, str1: str, str2: str) -> bool:
        """
        判断两个字符串是否相似
        
        返回:
            True表示相似，False表示不相似
        """
        worker_addr = self.get_worker_address()
        url = worker_addr + "/merge_model_inference"
        
        data = {
            "input_pairs": [str1, str2]
        }
        
        for retry in range(self.max_retry):
            try:
                response = requests.post(url, json=data, timeout=30)
                if response.status_code == 200:
                    result = response.json()
                    if "is_similar" in result:
                        return result["is_similar"]
                    elif "similarity_score" in result:
                        # 如果是相似度分数，使用0.5作为阈值
                        return result["similarity_score"] > 0.5
                    else:
                        raise RuntimeError(f"Unexpected response format: {result}")
                else:
                    error_msg = f"Merge model inference failed with status code: {response.status_code}"
                    if retry == self.max_retry - 1:
                        raise RuntimeError(error_msg)
                    else:
                        print(f"Warning: {error_msg}, retrying...")
                        time.sleep(1)
            except Exception as e:
                if retry == self.max_retry - 1:
                    raise RuntimeError(f"Failed to perform merge model inference: {e}")
                else:
                    print(f"Warning: Failed to perform merge model inference: {e}, retrying...")
                    time.sleep(1)
        
        # 如果所有重试都失败，默认认为不相似
        return False
    
    def batch_is_similar(self, str_pairs: List[Tuple[str, str]]) -> List[bool]:
        """
        批量判断字符串对是否相似
        
        参数:
            str_pairs: 字符串对列表，每个元素是(str1, str2)元组
            
        返回:
            布尔值列表，每个元素表示对应字符串对是否相似
        """
        worker_addr = self.get_worker_address()
        url = worker_addr + "/merge_model_inference"
        
        data = {
            "input_pairs": str_pairs
        }
        
        for retry in range(self.max_retry):
            try:
                response = requests.post(url, json=data, timeout=60)
                if response.status_code == 200:
                    result = response.json()
                    if "is_similar_list" in result:
                        return result["is_similar_list"]
                    elif "similarity_scores" in result:
                        # 如果是相似度分数，使用0.5作为阈值
                        return [score > 0.5 for score in result["similarity_scores"]]
                    else:
                        raise RuntimeError(f"Unexpected response format: {result}")
                else:
                    error_msg = f"Batch merge model inference failed with status code: {response.status_code}"
                    if retry == self.max_retry - 1:
                        raise RuntimeError(error_msg)
                    else:
                        print(f"Warning: {error_msg}, retrying...")
                        time.sleep(1)
            except Exception as e:
                if retry == self.max_retry - 1:
                    raise RuntimeError(f"Failed to perform batch merge model inference: {e}")
                else:
                    print(f"Warning: Failed to perform batch merge model inference: {e}, retrying...")
                    time.sleep(1)
        
        # 如果所有重试都失败，默认所有对都不相似
        return [False] * len(str_pairs)