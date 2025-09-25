import numpy as np
import torch
from tool.test_process import test_processor


@test_processor
def mlp(tpu_out, **kwargs):
    dic = {}
    b = kwargs.get('b', 1)
    w = kwargs.get('w', 4096)
    middle_w = kwargs.get('middle_w', 11008)
    input = torch.tensor(tpu_out['0'].reshape([b, w]))
    w0 = torch.tensor(tpu_out['1'].reshape([w, middle_w]))
    w1 = torch.tensor(tpu_out['2'].reshape([w, middle_w]))
    w2 = torch.tensor(tpu_out['3'].reshape([middle_w, w]))
    m1 = torch.matmul(input, w1)
    silu = torch.nn.SiLU()(m1)
    m0 = torch.matmul(input, w0)
    m0 = silu * m0
    out = torch.matmul(m0, w2)
    dic["4"] = out.numpy()
    return dic

mlp(dir="mlp_multicore", b=1, w=4096, middle_w=1024)
# mlp(dir="mlp_multicore", b=1024, w=4096, middle_w=11008)
