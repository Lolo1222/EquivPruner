from Levenshtein import distance
from sentence_transformers import SentenceTransformer, util
# from transformers import LongformerTokenizer, LongformerModel
from transformers import LongformerTokenizer, LongformerForSequenceClassification
import torch
import torch.nn.functional as F
# model = SentenceTransformer('/data2/OpenLLMs/sentence-transformers/paraphrase-MiniLM-L6-v2')
# model = SentenceTransformer('/data2/OpenLLMs/math-similarity/Bert-MLM_arXiv-MP-class_zbMath')
MODEL = LongformerForSequenceClassification.from_pretrained("/data1/home/jiawei/llmmcts/merge/results/longformer_results/checkpoint-24265")
TOKENIZER = LongformerTokenizer.from_pretrained("/data1/home/jiawei/llmmcts/merge/results/longformer_results/checkpoint-24265")
# def compute_similarity(sentences_pair):
#     embeddings = model.encode(sentences_pair)
#     embedding_1= model.encode(sentences_pair[0], convert_to_tensor=True)
#     embedding_2 = model.encode(sentences_pair[1], convert_to_tensor=True)
#     return util.pytorch_cos_sim(embedding_1, embedding_2).cpu().numpy()[0][0]
def preprocess_function(Sentence1, Sentence2, tokenizer):
    return tokenizer(Sentence1, Sentence2, truncation='longest_first', padding='max_length', max_length=2048, return_tensors="pt")
# Lolo1222: for merge similar node
def is_similar_str_pair(str1: str, str2: str, metric="levenshtein_ratio", threshold=0.95) -> bool:
    ratio = 1 - distance(str1, str2) / max(len(str1), len(str2))
    if ratio <= 0.75:
        return False
    if metric == "levenshtein_ratio":
        # ratio = 1 - distance(str1, str2) / max(len(str1), len(str2))
        return ratio > threshold
    elif metric == "model_merge":
        inputs = preprocess_function(str1, str2, TOKENIZER)
        with torch.no_grad():
            outputs = MODEL(**inputs)
            logits = outputs.logits 
            probabilities = torch.softmax(logits, dim=-1)
        predicted_label = torch.argmax(probabilities, dim=1).item()        
        return True if predicted_label else False
    else:
        raise NotImplementedError("未实现的merge method")
    return False

if __name__ == '__main__':
    # model = LongformerForSequenceClassification.from_pretrained("/data1/home/jiawei/llmmcts/merge/results/longformer_results/checkpoint-24265")
    # tokenizer = LongformerTokenizer.from_pretrained("/data1/home/jiawei/llmmcts/merge/results/longformer_results/checkpoint-24265")

    sentence1 = "The result is a+b=1"
    sentence2 = "The result is a+b=-1"

    inputs = preprocess_function(sentence1, sentence2, TOKENIZER)
    with torch.no_grad():
        outputs = MODEL(**inputs)
        logits = outputs.logits 
        probabilities = torch.softmax(logits, dim=-1)
    predicted_label = torch.argmax(probabilities, dim=1).item()
    similarity_score = probabilities[0][1].item()
    print(predicted_label, similarity_score)
    print(logits)
