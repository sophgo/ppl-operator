import torch

import ppl
import ppl.language as pl
import struct
import numpy as np
import functools

def rmsnorm_small_tiling(fn, ptr_output,
                            ptr_input,
                            ptr_weight,
                            ptr_bias,
                            eps,
                            with_weight,
                            with_bias,
                            row,
                            col,
                            row_slice,
                            block_w):
  while block_w > 0:
      ret = fn(ptr_output, ptr_input, ptr_weight, ptr_bias, eps, with_weight, with_bias, row, col, row_slice, block_w)
      if ret.errorCode == 0:
          print("block_w:%d"%block_w)
          return ret
      else:
          block_w = block_w // 2
  raise RuntimeError("[!Error]: rmsnorm tiling failed")

@ppl.jit(tiling=rmsnorm_small_tiling, debug=False)
def rmsnorm_small_row_kernel(ptr_output,
                            ptr_input,
                            ptr_weight,
                            ptr_bias,
                            eps,
                            with_weight:pl.constexpr,
                            with_bias:pl.constexpr,
                            row:pl.constexpr,
                            col,
                            row_slice,
                            block_w:pl.constexpr):
    core_idx = pl.get_core_index()
    core_num = pl.get_core_num()

    row_slice = 1
    for i in range(2, pl.LANE_NUM()):
        if col % i == 0:
            row_slice = i

    if core_idx < core_num:
        r_per_core = pl.cdiv(row, core_num)
        r_start = core_idx * r_per_core
        r_end = min(r_start + r_per_core, row)

        col_split = col // row_slice

        global_shape = [row, row_slice, 1, col_split]
        global_weight_shape = [1, row_slice, 1, col_split]
        in_gtensor = pl.gtensor(global_shape, pl.GLOBAL, ptr_input)
        weight_gtensor = pl.gtensor(global_weight_shape, pl.GLOBAL, ptr_weight)
        bias_gtensor = pl.gtensor(global_weight_shape, pl.GLOBAL, ptr_bias)
        out_gtensor = pl.gtensor(global_shape, pl.GLOBAL, ptr_output)

        local_in_block_shape = [r_per_core, pl.LANE_NUM(), 1, block_w]
        local_avg_block_shape = [r_per_core, pl.LANE_NUM(), 1, 1]
        local_weight_shape = [1, pl.LANE_NUM(), 1, block_w]

        is_input_split = block_w < col_split

        local_in_total = pl.make_tensor(local_in_block_shape, pl.float16)

        weight_in_total = pl.make_tensor(local_weight_shape, pl.float16, global_weight_shape)
        if (not is_input_split):
            in_global_offset = [r_start, 0, 0, 0]
            local_in_real_shape = [r_per_core, row_slice, 1, col_split]
            pl.dma.load(local_in_total.view(local_in_real_shape),
                    in_gtensor.sub_view(local_in_real_shape, in_global_offset))
            pl.dma.load(weight_in_total, weight_gtensor)

        for r_idx in range(r_start, r_end, r_per_core):
            avg_buffer = pl.make_tensor(local_avg_block_shape, pl.float32)
            pl.tiu.fill(avg_buffer, eps)
            for w_idx in range(0, col_split, block_w):
                pl.enable_pipeline()
                w = min(block_w, col_split - w_idx)
                local_in_shape = [r_per_core, row_slice, 1, w]
                input_global_offset = [r_start, 0, 0, w_idx]

                local_in = pl.make_tensor(local_in_block_shape, pl.float16, local_in_shape)
                if (is_input_split):
                    pl.dma.load(local_in,
                                in_gtensor.sub_view(local_in_shape, input_global_offset))

                local_in_fp32 = pl.make_tensor(local_in_block_shape, pl.float32, local_in_shape)
                if (is_input_split):
                    pl.tiu.cast(local_in_fp32, local_in, type=pl.float32)
                else:
                    pl.tiu.cast(local_in_fp32, local_in_total, type=pl.float32)

                local_in_tmp = pl.make_tensor(local_in_block_shape, pl.float32, local_in_shape)
                pl.tiu.fmul(local_in_tmp, local_in_fp32, local_in_fp32)
                sub_avg_shape = [r_per_core, row_slice, 1, 1]
                sub_avg = pl.make_tensor(local_avg_block_shape, pl.float32, sub_avg_shape)

                kernel_pool = [1, w]
                pad = [0, 0, 0, 0]
                stride = [1, 1]
                dilation = [1, 1]

                pl.tiu.pool_avg(sub_avg, local_in_tmp, kernel_pool, pad, stride, dilation, 1.0)
                kernel_conv = [1, 1]
                weight_c = 1.0 / col
                pl.tiu.fconv(avg_buffer, sub_avg, weight_c, kernel_conv, stride,
                        dilation, pad, oc=pl.LANE_NUM(), result_add=True, out_dtype=pl.float32)

            local_mu = pl.make_tensor(local_avg_block_shape, pl.float32)
            pl.tiu.frsqrt(local_mu, avg_buffer, num_iter=4)

            stride_n, stride_c, stride_h, stride_w = pl.aligned_stride_4d(local_avg_block_shape[0],
                                                    local_avg_block_shape[1],
                                                    local_avg_block_shape[2],
                                                    local_avg_block_shape[3],
                                                    pl.TPU_ALIGN, 0, pl.float32)
            avg_stride = [stride_n, stride_c, stride_h, stride_w]
            avg_bc_stride = [stride_n, 0, 0, 0]
            avg_bc_shape = [r_per_core, row_slice, 1, 1]

            for w_idx in range(0, col_split, block_w):
                pl.enable_pipeline()
                w = min(block_w, col_split - w_idx)
                local_in_shape = [r_per_core, row_slice, 1, w]
                input_global_offset = [r_start, 0, 0, w_idx]

                local_in = pl.make_tensor(local_in_block_shape, pl.float16, local_in_shape)
                if (is_input_split):
                    pl.dma.load(local_in,
                            in_gtensor.sub_view(local_in_shape, input_global_offset))

                local_in_fp32 = pl.make_tensor(local_in_block_shape, pl.float32, local_in_shape)
                if (is_input_split):
                    pl.tiu.cast(local_in_fp32, local_in, type=pl.float32)
                else:
                    pl.tiu.cast(local_in_fp32, local_in_total, type=pl.float32)

                out_fp32 = pl.make_tensor(local_in_block_shape, pl.float32, local_in_shape)
                pl.tiu.fmul(out_fp32, local_in_fp32,
                        local_mu.view(avg_bc_shape, avg_bc_stride))

                weight_real_shape = [1, row_slice, 1, w]
                local_weight_sub = pl.make_tensor(local_weight_shape, pl.float16, weight_real_shape)
                if with_weight and is_input_split:
                    weight_offset = [0, 0, 0, w_idx]
                    pl.dma.load(local_weight_sub,
                                weight_gtensor.sub_view(weight_real_shape, weight_offset))

                if (with_weight):
                    local_weight_sub_fp32 = pl.make_tensor(local_weight_shape, pl.float32, weight_real_shape)
                    if (is_input_split):
                        pl.tiu.cast(local_weight_sub_fp32, local_weight_sub, type=pl.float32)
                    else:
                        pl.tiu.cast(local_weight_sub_fp32, weight_in_total, type=pl.float32)

                    pl.tiu.fmul(out_fp32, out_fp32, local_weight_sub_fp32)

                if (with_bias):
                    bias_offset = [0, 0, 0, w_idx]
                    bias_real_shape = [1, row_slice, 1, w]
                    local_bias_sub = pl.make_tensor(local_weight_shape, pl.float16, bias_real_shape)
                    pl.dma.load(local_bias_sub,
                                bias_gtensor.sub_view(bias_real_shape, bias_offset))
                    local_bias_sub_fp32 = pl.make_tensor(local_weight_shape, pl.float32, bias_real_shape)
                    pl.tiu.cast(local_bias_sub_fp32, local_bias_sub, type=pl.float32)
                    pl.tiu.fadd(out_fp32, out_fp32, local_bias_sub_fp32)

                out = pl.make_tensor(local_in_block_shape, pl.float16, local_in_shape)
                pl.tiu.cast(out, out_fp32, type=pl.float16)
                pl.dma.store(out_gtensor.sub_view(local_in_shape, input_global_offset), out)

def rms_norm(input, eps=1e-5):
    variance = input.pow(2).mean(-1, keepdim=True)
    output = input * torch.rsqrt(variance + eps)
    return output

@functools.lru_cache()
def get_max_common_div(v, max_v):
    for i in range(max_v, 0, -1):
        if v % i == 0:
          return i
    return 1

delta = 0.005

atol = 1e-3  # 绝对容忍度
rtol = 1e-3  # 相对容忍度
torch.manual_seed(0)
input_shape = [8, 4096]
input_tensor = torch.randn(input_shape, dtype = torch.half)

eps=1e-5

output_torch = rms_norm(input_tensor)
output_tpu = torch.empty(input_shape, dtype = torch.half)

row_slice = 1
block_w = 1000000


rmsnorm_small_row_kernel[(1,8)](output_tpu, input_tensor, None, None, eps, False, False, input_shape[0], input_shape[1], row_slice, block_w)
# print(output_tpu)
# print(output_torch)
assert torch.allclose(output_torch, output_tpu, atol=atol, rtol=rtol) , "v2 result cmp failed"

