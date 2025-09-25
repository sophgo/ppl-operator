import numpy as np
import torch
from tool.test_process import test_processor

@test_processor
def embedding(tpu_out, **kwargs):
    outer_num = kwargs.get('outer_num', 1)
    select_num = kwargs.get('select_num', 1)
    inner_num = kwargs.get('inner_num', 1)
    index_num = kwargs.get('index_num', 1)
    weight = torch.tensor(tpu_out["1"].reshape([select_num, inner_num]))
    indices = torch.tensor(tpu_out["2"].reshape([index_num])).int()
    dic = {}
    out = torch.nn.functional.embedding(indices, weight).numpy()
    dic["0"] = out
    return dic

# embedding(dir="embedding", outer_num=1 , select_num = 32000, inner_num = 4096, index_num = 6)
embedding(dir="embedding", outer_num=1 , select_num = 15, inner_num = 64, index_num = 6)
