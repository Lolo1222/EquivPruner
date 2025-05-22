from Levenshtein import distance
from sentence_transformers import SentenceTransformer, util
from transformers import LongformerTokenizer, LongformerForSequenceClassification
import torch
import torch.nn.functional as F
import os

# global config parameters
USE_REMOTE_MERGE_MODEL = os.environ.get('USE_REMOTE_MERGE_MODEL', 'false').lower() == 'true'
MERGE_MODEL_NAME = os.environ.get('MERGE_MODEL_NAME', 'merge-model')
CONTROLLER_ADDR = os.environ.get('CONTROLLER_ADDR', 'http://0.0.0.0:28777')

# keep original global variables for local use
MODEL = None
TOKENIZER = None

# lazy load model and tokenizer
def get_model_and_tokenizer():
    global MODEL, TOKENIZER
    if MODEL is None or TOKENIZER is None:
        # MODEL = LongformerForSequenceClassification.from_pretrained("allenai/longformer-base-4096")
        # TOKENIZER = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
        MODEL = LongformerForSequenceClassification.from_pretrained("/data1/home/jiawei/llmmcts/merge/results/longformer_results/checkpoint-24265")
        TOKENIZER = LongformerTokenizer.from_pretrained("/data1/home/jiawei/llmmcts/merge/results/longformer_results/checkpoint-24265")
    return MODEL, TOKENIZER

def preprocess_function(Sentence1, Sentence2, tokenizer):
    return tokenizer(Sentence1, Sentence2, truncation='longest_first', padding='max_length', max_length=2048, return_tensors="pt")

# import remote client, only when needed
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
        # if use remote model, call remote service
        if USE_REMOTE_MERGE_MODEL:
            try:
                client = get_merge_client()
                # print(f"remote merge model call success")
                return client.is_similar(str1, str2)
            except Exception as e:
                # print(f"remote merge model call failed: {e}, use local model")
                # remote merge model call failed, use local model
                model, tokenizer = get_model_and_tokenizer()
                inputs = preprocess_function(str1, str2, tokenizer)
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits 
                    probabilities = torch.softmax(logits, dim=-1)
                predicted_label = torch.argmax(probabilities, dim=1).item()        
                return True if predicted_label else False
        else:
            print("Use local model")
            # use local model
            model, tokenizer = get_model_and_tokenizer()
            inputs = preprocess_function(str1, str2, tokenizer)
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits 
                probabilities = torch.softmax(logits, dim=-1)
            predicted_label = torch.argmax(probabilities, dim=1).item()        
            return True if predicted_label else False
    else:
        raise NotImplementedError("Unknown merge method")
    return False