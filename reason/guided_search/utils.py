from Levenshtein import distance
from sentence_transformers import SentenceTransformer, util
from transformers import LongformerTokenizer, LongformerForSequenceClassification
import torch
import torch.nn.functional as F
import os

# 全局配置参数
USE_REMOTE_MERGE_MODEL = os.environ.get('USE_REMOTE_MERGE_MODEL', 'false').lower() == 'true'
MERGE_MODEL_NAME = os.environ.get('MERGE_MODEL_NAME', 'merge-model')
CONTROLLER_ADDR = os.environ.get('CONTROLLER_ADDR', 'http://0.0.0.0:28777')

# 保留原有全局变量用于本地调用
MODEL = None
TOKENIZER = None

# 懒加载模型和tokenizer
def get_model_and_tokenizer():
    global MODEL, TOKENIZER
    if MODEL is None or TOKENIZER is None:
        MODEL = LongformerForSequenceClassification.from_pretrained("/data1/home/jiawei/llmmcts/merge/results/longformer_results/checkpoint-24265")
        TOKENIZER = LongformerTokenizer.from_pretrained("/data1/home/jiawei/llmmcts/merge/results/longformer_results/checkpoint-24265")
    return MODEL, TOKENIZER

def preprocess_function(Sentence1, Sentence2, tokenizer):
    return tokenizer(Sentence1, Sentence2, truncation='longest_first', padding='max_length', max_length=2048, return_tensors="pt")

# 导入远程客户端，仅当需要时才导入
def get_merge_client():
    from merge.merge_model_client import MergeModelClient
    return MergeModelClient(MERGE_MODEL_NAME, CONTROLLER_ADDR)

# Lolo1222: for merge similar node
def is_similar_str_pair(str1: str, str2: str, metric="levenshtein_ratio", threshold=0.95) -> bool:
    ratio = 1 - distance(str1, str2) / max(len(str1), len(str2))
    if ratio <= 0.75:
        return False
    
    if metric == "levenshtein_ratio":
        return ratio > threshold
    elif metric == "model_merge":
        # 如果使用远程模型，则调用远程服务
        if USE_REMOTE_MERGE_MODEL:
            try:
                client = get_merge_client()
                return client.is_similar(str1, str2)
            except Exception as e:
                print(f"远程调用失败: {e}，降级为本地调用")
                # 远程调用失败时降级为本地调用
                model, tokenizer = get_model_and_tokenizer()
                inputs = preprocess_function(str1, str2, tokenizer)
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits 
                    probabilities = torch.softmax(logits, dim=-1)
                predicted_label = torch.argmax(probabilities, dim=1).item()        
                return True if predicted_label else False
        else:
            # 使用本地模型
            model, tokenizer = get_model_and_tokenizer()
            inputs = preprocess_function(str1, str2, tokenizer)
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits 
                probabilities = torch.softmax(logits, dim=-1)
            predicted_label = torch.argmax(probabilities, dim=1).item()        
            return True if predicted_label else False
    else:
        raise NotImplementedError("未实现的merge method")
    return False

if __name__ == '__main__':
    # 测试代码
    sentence1 = "The result is a+b=1"
    sentence2 = "The result is a+b=-1"

    # 尝试使用远程服务
    if USE_REMOTE_MERGE_MODEL:
        try:
            client = get_merge_client()
            result = client.is_similar(sentence1, sentence2)
            print(f"远程服务结果: {result}")
        except Exception as e:
            print(f"远程调用失败: {e}")
    
    # 使用本地模型
    model, tokenizer = get_model_and_tokenizer()
    inputs = preprocess_function(sentence1, sentence2, tokenizer)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits 
        probabilities = torch.softmax(logits, dim=-1)
    predicted_label = torch.argmax(probabilities, dim=1).item()
    similarity_score = probabilities[0][1].item()
    print(f"本地模型结果: label={predicted_label}, score={similarity_score}")
    print(f"logits: {logits}")