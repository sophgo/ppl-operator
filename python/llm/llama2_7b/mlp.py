import torch

import ppl
import ppl.language as pl
import struct
import numpy as np


@ppl.jit
def mlp_kernel(out_ptr, input_ptr, weight0_ptr, weight1_ptr, weight2_ptr,
               batch, input_w, middle_w, block_b: pl.constexpr,
               block_iw: pl.constexpr, block_w: pl.constexpr):
    """
    mlp

        .. code-block:: python

            mlp_kernel(out_ptr, input_ptr, weight0_ptr, weight1_ptr, weight2_ptr, batch, input_w, middle_w, block_b, block_iw, block_w)

    参数:
        - ``out_ptr`` (`ppl.language.tensor`): output张量

        - ``input_ptr`` (`ppl.language.tensor`): input张量

        - ``weight0_ptr`` (`ppl.language.tensor`): weight0张量

        - ``weight1_ptr`` (`ppl.language.tensor`): weight1张量

        - ``weight2_ptr`` (`ppl.language.tensor`): weight2张量

        - ``batch`` (`int`): input张量的C

        - ``input_w`` (`int`): input张量的W

        - ``middle_w`` (`int`): weight0张量的W

        - ``block_b`` (`int`): input张量的切分C

        - ``block_iw`` (`int`): input张量的切分W

        - ``block_w`` (`int`): weight0张量的切分W

    返回值:
        无

    注意事项:
        无
    """
    core_num = pl.get_core_num()
    core_idx = pl.get_core_index()
    b_loop = pl.cdiv(batch, pl.LANE_NUM())
    for b_idx in range(b_loop):
        b_offset = b_idx * pl.LANE_NUM()
        b_slice = min(pl.LANE_NUM(), batch - b_offset)
        g_input_shape = [1, batch, 1, input_w]
        g_offset = [0, b_offset, 0, 0]

        slice_per_core = pl.cdiv(middle_w, core_num)
        core_offset = slice_per_core * core_idx
        slice_per_core = min(slice_per_core, middle_w - core_offset)
        input_shape = [1, b_slice, 1, input_w]
        input_block_shape = [1, block_b, 1, block_iw]
        l2_out_tensor = pl.gtensor(input_block_shape, pl.L2,
                                   dtype=pl.float16).view(shape=input_shape)
        if (slice_per_core > 0):
            weight0_global_shape = [1, input_w, 1, middle_w]
            weight2_global_shape = [1, middle_w, 1, input_w]

            weight0_block_shape = [1, block_iw, 1, block_w]
            weight2_block_shape = [1, block_w, 1, block_iw]
            middle_buffer_shape = [1, block_b, 1, block_w]

            input_gtensor = pl.gtensor(g_input_shape, pl.GLOBAL,
                                       input_ptr).sub_view(
                                           input_shape, g_offset)

            weight0_gtensor = pl.gtensor(weight0_global_shape, pl.GLOBAL,
                                         weight0_ptr)
            weight1_gtensor = pl.gtensor(weight0_global_shape, pl.GLOBAL,
                                         weight1_ptr)
            weight2_gtensor = pl.gtensor(weight2_global_shape, pl.GLOBAL,
                                         weight2_ptr)

            input_local = pl.make_tensor(input_block_shape, input_ptr.dtype,
                                         input_shape)
            out_f32_local = pl.make_tensor(input_block_shape, pl.float32,
                                           input_shape)

            pl.dma.load(input_local, input_gtensor)
            pl.dma.fill(out_f32_local, 0)

            for w_idx in range(0, slice_per_core, block_w):
                pl.enable_pipeline()
                middle_slice = min(block_w, slice_per_core - w_idx)
                weight0_shape = [1, input_w, 1, middle_slice]
                weight2_shape = [1, middle_slice, 1, input_w]
                weight0_offset = [0, 0, 0, core_offset + w_idx]
                weight2_offset = [0, core_offset + w_idx, 0, 0]
                weight0_local = pl.make_tensor(weight0_block_shape, pl.float16,
                                               weight0_shape)
                weight1_local = pl.make_tensor(weight0_block_shape, pl.float16,
                                               weight0_shape)
                weight2_local = pl.make_tensor(weight2_block_shape, pl.float16,
                                               weight2_shape)

                pl.dma.load(
                    weight0_local,
                    weight0_gtensor.sub_view(weight0_shape, weight0_offset))
                pl.dma.load(
                    weight1_local,
                    weight1_gtensor.sub_view(weight0_shape, weight0_offset))
                pl.dma.load(
                    weight2_local,
                    weight2_gtensor.sub_view(weight2_shape, weight2_offset))
                middle_real_shape = [1, b_slice, 1, middle_slice]
                middle_buffer_f16_local_1 = pl.make_tensor(
                    middle_buffer_shape, pl.float16, middle_real_shape)
                middle_buffer_f16_local_2 = pl.make_tensor(
                    middle_buffer_shape, pl.float16, middle_real_shape)
                middle_buffer_f32_local = pl.make_tensor(
                    middle_buffer_shape, pl.float32, middle_real_shape)

                #matmul -> x f16
                pl.tiu.fmm2(middle_buffer_f16_local_1,
                            input_local,
                            weight1_local,
                            out_dtype=pl.float16)
                # neg -> -x f16
                pl.tiu.fmul(middle_buffer_f16_local_2,
                            middle_buffer_f16_local_1, -1.0)
                # cast -> f16 2 f32
                pl.tiu.cast(middle_buffer_f32_local,
                            middle_buffer_f16_local_2,
                            type=pl.float32)
                # exp -> exp^-x f32
                middle_buffer_f32_local = pl.exp_no_overflow(
                    middle_buffer_f32_local, 1, block_b, 1, block_w, 1,
                    b_slice, 1, middle_slice)

                # add -> 1 + exp^-x f32
                pl.tiu.fadd(middle_buffer_f32_local, middle_buffer_f32_local,
                            1.0)
                # div -> 1/(1 + exp^-x)  f32
                pl.tiu.fdiv(middle_buffer_f32_local,
                            1.0,
                            middle_buffer_f32_local,
                            num_iter=4)
                # cast -> f32 2 f16
                pl.tiu.cast(middle_buffer_f16_local_2,
                            middle_buffer_f32_local,
                            type=pl.float16)
                # mul -> x/(1+exp^-x) f16
                pl.tiu.fmul(middle_buffer_f16_local_1,
                            middle_buffer_f16_local_1,
                            middle_buffer_f16_local_2)
                # matmul -> x1 f16
                pl.tiu.fmm2(middle_buffer_f16_local_2,
                            input_local,
                            weight0_local,
                            out_dtype=pl.float16)
                # mul -> x1 * x/(1+exp^-x) f16
                pl.tiu.fmul(middle_buffer_f16_local_1,
                            middle_buffer_f16_local_1,
                            middle_buffer_f16_local_2)
                # matmul -> out f32
                pl.tiu.fmm2(out_f32_local,
                            middle_buffer_f16_local_1,
                            weight2_local,
                            result_add=True,
                            out_dtype=pl.float32)

            pl.tiu.cast(input_local, out_f32_local, type=input_ptr.dtype)
            pl.dma.zero(l2_out_tensor)
            pl.sync()
            pl.dma.reduce(l2_out_tensor, input_local, pl.ALL_REDUCE_PSUM_WR,
                          pl.ALL_REDUCE_ADD)

        ele_num = b_slice * input_w
        slice_per_core = pl.cdiv(ele_num, core_num)
        core_offset = slice_per_core * core_idx
        slice_per_core = min(slice_per_core,
                             ele_num - core_idx * slice_per_core)
        if (slice_per_core > 0):
            output_shape = [1, 1, 1, b_slice * input_w]
            output_gtensor = pl.gtensor(g_input_shape, pl.GLOBAL,
                                        out_ptr).sub_view(
                                            output_shape, g_offset)
            real_shape = [1, 1, 1, slice_per_core]
            offset = [0, 0, 0, core_offset]
            pl.sync()
            pl.sdma.move(
                output_gtensor.sub_view(real_shape, offset),
                l2_out_tensor.view(shape=output_shape).sub_view(
                    real_shape, offset))
            pl.sync()


if __name__ == '__main__':
    B = 1
    W = 1024
    MIDDLE_W = 2048
    core_num = 8
    a, b = -1, 1
    a_, b_ = -0.5, 0.5
    input = torch.randn((1, B, 1, W), device='cpu', dtype=torch.half)
    input = a + (b - a) * (input - input.min()) / (input.max() - input.min())
    output = torch.empty((1, B, 1, W), device='cpu', dtype=torch.half)
    weight0 = torch.randn((1, W, 1, MIDDLE_W), device='cpu', dtype=torch.half)
    weight1 = torch.randn((1, W, 1, MIDDLE_W), device='cpu', dtype=torch.half)
    weight2 = torch.randn((1, MIDDLE_W, 1, W), device='cpu', dtype=torch.half)
    weight0 = a_ + (b_ - a_) * (weight0 - weight0.min()) / (weight0.max() -
                                                            weight0.min())
    weight1 = a_ + (b_ - a_) * (weight1 - weight1.min()) / (weight1.max() -
                                                            weight1.min())
    weight2 = a_ + (b_ - a_) * (weight2 - weight2.min()) / (weight2.max() -
                                                            weight2.min())
    m1 = torch.matmul(
        input.reshape([B, W]).to(torch.float),
        weight1.reshape([W, MIDDLE_W]).to(torch.float))
    silu = torch.nn.SiLU()(m1)
    m0 = torch.matmul(
        input.reshape([B, W]).to(torch.float),
        weight0.reshape([W, MIDDLE_W]).to(torch.float))
    m0 = silu * m0
    cpu_res = torch.matmul(m0, weight2.reshape([MIDDLE_W, W]).to(torch.float))
    mlp_kernel[(1, core_num)](output, input, weight0, weight1, weight2, B,
                                 W, MIDDLE_W, B, W, 288)

    atol = 1e-1  # 绝对容忍度
    rtol = 1e-1  # 相对容忍度
    assert torch.allclose(cpu_res.reshape([1, B, 1, W]).to(output.dtype),
                          output,
                          atol=atol,
                          rtol=rtol), "result cmp failed"
