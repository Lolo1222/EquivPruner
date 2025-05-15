from collections import Counter, defaultdict
from typing import List
# from reason.guided_search.utils_bk import MODEL, TOKENIZER, preprocess_function, is_similar_str_pair

MAJORITY_VOTE = "majority_vote"
ORM_VOTE = "orm_vote"
ORM_MAX = "orm_max"
PRM_MIN_MAX = "prm_min_max"
PRM_MIN_VOTE = "prm_min_vote"
PRM_LAST_MAX = "prm_last_max"
PRM_LAST_VOTE = "prm_last_vote"
# MERGED_MAJORITY_VOTE = "merged_majority_vote"
# def remove_similar_strings(strings: List[str], metric="model_merge", threshold=0.95) -> List[str]:
#     unique_strings = []
    
#     for s in strings:
#         if not any(is_similar_str_pair(s, unique, metric=metric, threshold=threshold) for unique in unique_strings):
#             unique_strings.append(s)
    
#     return unique_strings
# def normalize_similar_strings(strings: List[str], metric="model_merge", threshold=0.95) -> List[str]:
#     result = strings.copy()  # 创建输入列表的副本
#     n = len(strings)
    
#     # 遍历所有可能的字符串对
#     for i in range(n):
#         for j in range(i + 1, n):
#             # 如果两个字符串相似
#             if is_similar_str_pair(result[i], result[j], metric=metric, threshold=threshold):
#                 # 将后面的字符串改为和前面的字符串相同
#                 result[j] = result[i]
    
#     return result

def _agg_majority_vote(x_list: List[str], unused_v_list: List[float]):
    counts = Counter(x_list)
    most_common = max(counts, key=counts.get)
    return most_common

# def _agg_merged_majority_vote(x_list: List[str], unused_v_list: List[float]):
#     y_list = normalize_similar_strings(x_list)
#     counts = Counter(y_list)
#     most_common = max(counts, key=counts.get)
#     return most_common



def _agg_orm_vote(x_list: List[str], v_list: List[float]):
    assert len(x_list) == len(v_list)
    x_dict = defaultdict(lambda: 0.0)
    for x, v in zip(x_list, v_list):
        x_dict[x] += v

    highest_x = max(x_dict, key=x_dict.get)
    return highest_x


def _agg_orm_max(x_list: List[str], v_list: List[float]):
    text_max = x_list[v_list.index(max(v_list))]
    return text_max


def _agg_prm_min_max(x_list: List[str], v_list: List[List[float]]):
    v_list = [min(v) if v else -1.0 for v in v_list]
    text_max = x_list[v_list.index(max(v_list))]
    return text_max


def _agg_prm_last_max(x_list: List[str], v_list: List[List[float]]):
    v_list = [v[-1] if v else -1.0 for v in v_list]
    text_max = x_list[v_list.index(max(v_list))]
    return text_max


def _agg_prm_min_vote(x_list: List[str], v_list: List[List[float]]):
    v_list = [min(v) if v else -1.0 for v in v_list]
    return _agg_orm_vote(x_list, v_list)


def _agg_prm_last_vote(x_list: List[str], v_list: List[List[float]]):
    v_list = [v[-1] if v else -1.0 for v in v_list]
    return _agg_orm_vote(x_list, v_list)


AGG_FN_MAP = {
    MAJORITY_VOTE: _agg_majority_vote,
    # ORM_VOTE: _agg_orm_vote,
    # ORM_MAX: _agg_orm_max,
    PRM_MIN_MAX: _agg_prm_min_max,
    PRM_MIN_VOTE: _agg_prm_min_vote,
    PRM_LAST_MAX: _agg_prm_last_max,
    PRM_LAST_VOTE: _agg_prm_last_vote,
}
