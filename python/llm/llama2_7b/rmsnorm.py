import torch

import ppl
import ppl.language as pl
import struct
import numpy as np

def rms_tiling(fn, ptr_out,
                    ptr_input,
                    ptr_weight,
                    ptr_bias,
                    eps,
                    with_weight,
                    with_bias,
                    _N,
                    _C,
                    _H,
                    _W,
                    block_w):
  while block_w > 0:
      ret = fn(ptr_out, ptr_input, ptr_weight, ptr_bias, eps, with_weight, with_bias, _N, _C, _H, _W, block_w)
      if ret.errorCode == 0:
          return ret
      else:
          block_w = block_w // 2
  raise RuntimeError("[!Error]: rmsnorm tiling failed")

@ppl.jit(tiling=rms_tiling)
def rms_norm_kernel(ptr_out,
                    ptr_input,
                    ptr_weight,
                    ptr_bias,
                    eps,
                    with_weight:pl.constexpr,
                    with_bias:pl.constexpr,
                    _N,
                    _C,
                    _H,
                    _W,
                    block_w:pl.constexpr):
    """
    rms_norm

        .. code-block:: python

            rms_norm_kernel(ptr_out, ptr_input, ptr_weight, ptr_bias, eps, with_weight, with_bias, _N, _C, _H, _W, block_w)

    参数:
        - ``ptr_out`` (`ppl.language.tensor`): output张量

        - ``ptr_input`` (`ppl.language.tensor`): input张量

        - ``ptr_weight`` (`ppl.language.tensor`): weight张量

        - ``ptr_bias`` (`ppl.language.tensor`): bias张量

        - ``eps`` (`标量`): eps

        - ``with_weight`` (`bool`): 是否有weight

        - ``with_bias`` (`bool`): 是否有bias

        - ``_N`` (`int`): input张量的N

        - ``_C`` (`int`): input张量的C

        - ``_H`` (`int`): input张量的H

        - ``_W`` (`int`): input张量的W

        - ``block_w`` (`int`): input张量的切分W

    返回值:
        无

    注意事项:
        无
    """
    core_idx = pl.get_core_index()
    core_num = pl.get_core_num()

    if core_idx < core_num:
      C = _N * _C
      W = _H * _W
      c_per_core = (C + core_num - 1) // core_num
      c_start = core_idx * c_per_core
      c_end = min(c_start + c_per_core, C)
      block_c = pl.LANE_NUM()

      global_shape = [1, c_per_core, 1, W]
      global_weight_shape = [1,1,1,W]
      in_gtensor = pl.gtensor(global_shape, pl.GLOBAL, ptr_input)
      weight_gtensor = pl.gtensor(global_weight_shape, pl.GLOBAL, ptr_weight)
      bias_gtensor = pl.gtensor(global_weight_shape, pl.GLOBAL, ptr_bias)
      out_gtensor = pl.gtensor(global_shape, pl.GLOBAL, ptr_out)

      local_in_block_shape = [1, block_c, 1, block_w]
      local_avg_block_shape = [1, block_c, 1, 1]
      local_weight_shape = [1,1,1,block_w]
      for c_idx in range(c_start, c_end, block_c):
        c = min(block_c, c_end - c_idx)
        local_avg_shape = [1,c,1,1]
        avg_buffer = pl.make_tensor(local_avg_block_shape, pl.float32, local_avg_shape)
        pl.tiu.fill(avg_buffer, eps)

        for w_idx in range(0, W, block_w):
          pl.enable_pipeline()
          w = min(block_w, W - w_idx)
          local_in_shape = [1,c,1,w]
          input_global_offset = [0, c_idx, 0, w_idx]


          local_in = pl.make_tensor(local_in_block_shape, ptr_input.dtype, local_in_shape)
          pl.dma.load(local_in, in_gtensor.sub_view(local_in_shape, input_global_offset))

          local_in_fp32 = pl.make_tensor(local_in_block_shape, pl.float32, local_in_shape)
          pl.tiu.cast(local_in_fp32, local_in)

          local_in_tmp = pl.make_tensor(local_in_block_shape, pl.float32, local_in_shape)
          pl.tiu.fmul(local_in_tmp, local_in_fp32, local_in_fp32)
          sub_avg = pl.make_tensor(local_avg_block_shape, pl.float32, local_avg_shape)

          pl.quick_pooling(sub_avg, local_in_tmp, 1, block_c, 1, block_w, 1, c, 1, w, 0, 1, 1.0 / W)
          pl.tiu.fadd(avg_buffer, avg_buffer, sub_avg)

        local_mu = pl.make_tensor(local_avg_block_shape, pl.float32, local_avg_shape)
        pl.tiu.frsqrt(local_mu, avg_buffer, num_iter = 4)

        for w_idx in range(0, W, block_w):
          pl.enable_pipeline()
          w = min(block_w, W - w_idx)
          local_in_shape = [1,c,1,w]
          input_global_offset = [0, c_idx, 0, w_idx]

          local_in = pl.make_tensor(local_in_block_shape, ptr_input.dtype, local_in_shape)
          pl.dma.load(local_in, in_gtensor.sub_view(local_in_shape, input_global_offset))

          local_in_fp32_2 = pl.make_tensor(local_in_block_shape, pl.float32, local_in_shape)
          pl.tiu.cast(local_in_fp32_2, local_in)

          out_fp32 = pl.make_tensor(local_in_block_shape, pl.float32, local_in_shape)
          pl.tiu.fmul(out_fp32, local_in_fp32_2, local_mu)

          if with_weight:
            weight_offset = [0, 0, 0, w_idx]
            weight_real_shape = [1, 1, 1, w]
            local_weight_sub = pl.make_tensor(local_weight_shape, ptr_weight.dtype, weight_real_shape)
            pl.dma.load(local_weight_sub, weight_gtensor.sub_view(weight_real_shape, weight_offset))
            local_weight_sub_fp32 = pl.make_tensor(local_weight_shape, pl.float32, weight_real_shape)
            pl.tiu.cast(local_weight_sub_fp32, local_weight_sub)
            tmp_fp32 = pl.make_tensor(local_in_block_shape, pl.float32, local_in_shape)
            pl.tiu.broadcast(tmp_fp32, local_weight_sub_fp32)
            pl.tiu.fmul(out_fp32, out_fp32, tmp_fp32)

          if with_bias:
            bias_offset = [0, 0, 0 ,w_idx]
            bias_real_shape = [1, 1, 1, w]
            local_bias_sub = pl.make_tensor(local_weight_shape, ptr_bias.dtype, bias_real_shape)
            pl.dma.load(local_bias_sub, bias_gtensor.sub_view(bias_real_shape, bias_offset))
            local_bias_sub_fp32 = pl.make_tensor(local_weight_shape, pl.float32, bias_real_shape)
            pl.tiu.cast(local_bias_sub_fp32, local_bias_sub)
            tmp_fp32 = pl.make_tensor(local_in_block_shape, pl.float32, local_in_shape)
            pl.tiu.broadcast(tmp_fp32, local_bias_sub_fp32)
            pl.tiu.fadd(out_fp32, out_fp32, tmp_fp32)
          out = pl.make_tensor(local_in_block_shape, ptr_out.dtype, local_in_shape)
          pl.tiu.cast(out, out_fp32)
          pl.dma.store(out_gtensor.sub_view(local_in_shape, input_global_offset), out)

N ,C, H, W = [1,1024,1,1024]
in_shape = [N, C, H, W]
weight_shape = [1,1,1,H*W]

input = torch.rand(in_shape, dtype = torch.half)
weight = torch.rand(weight_shape, dtype = torch.half)
bias = torch.rand(weight_shape, dtype = torch.half)
output_ppl = torch.zeros(in_shape, dtype = torch.half)

variance = input.pow(2).mean(-1, keepdim=True).to(torch.float32)
tmp = input * torch.rsqrt(variance + 1e-5)
output_torch = (tmp * weight + bias).to(torch.half)

rms_norm_kernel[(1,8,)](output_ppl, input, weight, bias, 1e-5, True, True, N, C, H, W, 256)

atol = 1e-3  # 绝对容忍度
rtol = 1e-3  # 相对容忍度
# print(output_ppl)
# print(output_torch)
assert torch.allclose(output_torch, output_ppl, atol=atol, rtol=rtol) , "result cmp failed"

