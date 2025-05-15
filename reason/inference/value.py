import torch
from typing import Union, List
from transformers import AutoTokenizer
import re
import numpy as np
import requests


def _value_inference_fastchat(
    model_name: str,
    input_str: Union[List[str], str],
    controller_addr="http://0.0.0.0:28777",
):
    ret = requests.post(
        controller_addr + "/get_worker_address", json={"model": model_name}
    )
    worker_addr = ret.json()["address"]
    if not worker_addr:
        raise ValueError("Value Model name {} does not exist.".format(model_name))

    headers = {"User-Agent": "FastChat Client"}
    gen_params = {"input_str": input_str}
    response = requests.post(
        worker_addr + "/worker_value_inference",
        headers=headers,
        json=gen_params,
        stream=True,
    )
    results = response.json()
    try:
        value = results["value"]
    except Exception as e:
        print('Error value results is : ', results)
        print("gen_params is : ", gen_params)
        value = [[0] for _ in range(len(gen_params["input_str"]))]
    return value
