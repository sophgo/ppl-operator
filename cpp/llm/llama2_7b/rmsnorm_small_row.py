import numpy as np
import torch
from tool.test_process import test_processor


@test_processor
def rms_norm(tpu_out, shapes, eps=1e-5, **kwargs):
    N ,C, H, W = shapes
    x = tpu_out["1"]
    weight = tpu_out["2"]
    bias = tpu_out["3"]
    t_in = torch.tensor(x.reshape([1, N * C, 1, H * W]), dtype=torch.float)
    t_weight = torch.tensor(weight.reshape([1, 1, 1, H * W]),
                            dtype=torch.float)
    t_bias = torch.tensor(bias.reshape([1, 1, 1, H * W]),
                            dtype=torch.float)
    variance = t_in.pow(2).mean(-1, keepdim=True)
    t_in = t_in * torch.rsqrt(variance + eps)
    return {"0": (t_in * t_weight + t_bias).numpy()}

rms_norm(dir="rmsnorm_small_row", shapes=(1,8,1,4096), eps=0.000001)

