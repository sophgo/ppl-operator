import torch
from tool.test_process import test_processor

@test_processor
def matmul(tpu_out, batch, M, K, N, **kwargs):
    res_out = tpu_out["0"]
    left = tpu_out["1"]
    right = tpu_out["2"]
    tensor_left = torch.tensor(left.reshape([batch, M, K]), dtype=torch.float)
    tensor_right = torch.tensor(right.reshape([batch, N, K]), dtype=torch.float)
    torch_out = torch.matmul(tensor_left, tensor_right.transpose(1,2))
    return {"0": torch_out.numpy()}

matmul(dir="matmul", batch=2, M=6, K=1024, N=6)
