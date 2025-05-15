import torch
from typing import Tuple, List, Optional, Dict, Any, Union
from transformers import AutoTokenizer, AutoModelForSequenceClassification

@torch.inference_mode()
def _longformer_binary_infer_fn(input_pair: List[str], model, tokenizer, device):
    """
    处理Longformer模型的二分类推理函数
    返回二维向量 [label0_prob, label1_prob]
    """
    str1, str2 = input_pair
    inputs = tokenizer(str1, str2, truncation='longest_first', padding='max_length', 
                      max_length=2048, return_tensors="pt").to(device)
    
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=-1)
    
    # 返回二维概率向量
    return probabilities.cpu()

@torch.inference_mode()
def _longformer_similarity_infer_fn(input_pair: List[str], model, tokenizer, device):
    """
    处理Longformer模型的相似度推理函数
    返回单一的相似度值
    """
    str1, str2 = input_pair
    inputs = tokenizer(str1, str2, truncation='longest_first', padding='max_length', 
                      max_length=2048, return_tensors="pt").to(device)
    
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=-1)
    
    # 返回label=1的概率（表示相似）
    return probabilities[:, 1].cpu()

@torch.inference_mode()
def _qwen_binary_infer_fn(input_pair: List[str], model, tokenizer, device):
    """
    处理Qwen模型的二分类推理函数
    返回单一的0或1值
    """
    str1, str2 = input_pair
    
    # 构建提示词
    prompt = f"Determine if the following two sentences are semantically equivalent. Output only '0' (not equivalent) or '1' (equivalent).\nSentence 1: {str1}\nSentence 2: {str2}\nOutput:"
    
    # 编码并推理
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=False
        )
    
    # 解码结果
    result = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    
    # 尝试提取数字结果
    try:
        # 尝试从结果中提取数字（0或1）
        if "1" in result:
        # if "1" in result or "一" in result or "相似" in result:
            return torch.tensor([[0.0, 1.0]])
        else:
            return torch.tensor([[1.0, 0.0]])
    except:
        # 默认返回不相似
        return torch.tensor([[1.0, 0.0]])

def get_merge_model_infer_fn(model_path: str, output_type: str = "binary"):
    """
    根据模型路径和输出类型选择对应的推理函数
    
    参数:
        model_path: 模型路径
        output_type: 输出类型，可选 "binary"（二分类结果）, "similarity"（相似度）, "qwen"（Qwen模型）
    
    返回:
        对应的推理函数
    """
    if "longformer" in model_path.lower():
        if output_type == "binary":
            return _longformer_binary_infer_fn
        elif output_type == "similarity":
            return _longformer_similarity_infer_fn
        else:
            raise ValueError(f"Unsupported output_type {output_type} for longformer model")
    elif "qwen" in model_path.lower():
        return _qwen_binary_infer_fn
    else:
        raise ValueError(f"Model type not recognized: {model_path}")