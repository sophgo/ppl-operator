import torch
import numpy as np
import torch.nn.functional as F
from tool.test_process import test_processor

@test_processor
def matmul(tpu_out, M, K, N, group_size, **kwargs):
    res_out = tpu_out["0"]
    left = tpu_out["1"]
    right = tpu_out["2"]
    scale_zp = tpu_out["3"]
    tensor_left = torch.tensor(left.reshape([M, K]), dtype=torch.float)

    tensor_right = torch.tensor(right.reshape([N, K//2]), dtype=torch.float)
    # 将 float 数据转换回 uint8 数据
    tensor_right_uint8 = tensor_right.to(torch.uint8)
    # 提取 uint8 数据的高 4 位和低 4 位
    right_high_4_bits = tensor_right_uint8 >> 4
    right_low_4_bits = tensor_right_uint8 & 0x0F
    # 合并高 4 位和低 4 位到一个张量中
    right_uint4_tensor = torch.stack((right_low_4_bits, right_high_4_bits), dim=-1).reshape([N, K])
    # 将 uint4 数据转换为 uint8 数据
    right_int8_tensor = right_uint4_tensor.to(torch.int8)

    group = K // group_size
    scale_zp_w = group * 5 // 2
    tensor_scale_zp = torch.tensor(scale_zp.reshape([N, scale_zp_w]), dtype=torch.float)
    # 将 float 数据转换回 uint8 数据
    scale_zp_uint8_tensor = tensor_scale_zp.to(torch.uint8)

    # 取前 [N, 2 * group] 数据：两个 uint8 存一个 fp16
    first_part_uint8 = scale_zp_uint8_tensor[:, :2 * group]
    # 将张量转换为 NumPy 数组
    first_part_uint8_numpy = first_part_uint8.numpy()
    # 重解释为 float16，注意字节顺序（假设是小端序）
    scale_fp16_numpy = first_part_uint8_numpy.view(dtype=np.float16)
    # 将 NumPy 数组转换回 PyTorch 张量
    scale_fp16_tensor = torch.from_numpy(scale_fp16_numpy)

    # 取剩余的 [N, 0.5 * group] 数据：一个 uint8 存两个 uint4
    remaining_uint8 = scale_zp_uint8_tensor[:, 2 * group:]
    # 获取高四位和低四位
    high_nibble = remaining_uint8 >> 4
    low_nibble = remaining_uint8 & 0x0F
    # 将高四位和低四位拼接成一个 [N, group] 的张量
    zp_int8_tensor = torch.stack((low_nibble, high_nibble), dim=-1).view(N, group).to(torch.int8)

    sub_res_tensor = right_int8_tensor.reshape([N, group, group_size]) - zp_int8_tensor.reshape([N, group, 1])
    sub_res_f16_tensor = sub_res_tensor.to(torch.float16)
    mul_res_f16 = sub_res_f16_tensor.reshape([N, group, group_size]) * scale_fp16_tensor.reshape([N, group, 1])
    torch_out = F.linear(tensor_left, mul_res_f16.reshape([N, K]).to(torch.float))

    return {"0": torch_out.numpy()}

matmul(dir="w4a16_matmul", M=128, K=4096, N=4096, group_size=128)
