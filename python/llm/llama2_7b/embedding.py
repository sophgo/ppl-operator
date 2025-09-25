import torch

import ppl
import ppl.language as pl
import struct
import numpy as np


@ppl.jit(debug=False)
def embedding_kernel(ptr_out, ptr_param, ptr_index, outer_num,
                     inner_num: pl.constexpr, select_num: pl.constexpr,
                     index_num):
    """
    embedding计算
        .. code-block:: python

             embedding_kernel(ptr_out, ptr_param, ptr_index, outer_num, inner_num, select_num, index_num)

    参数:
        - ``ptr_out`` (`ppl.language.tensor`): output张量

        - ``ptr_param`` (`ppl.language.tensor`): param张量

        - ``ptr_index`` (`ppl.language.tensor`): index张量

        - ``outer_num`` (`int`): outer_num

        - ``inner_num`` (`int`): inner_num

        - ``select_num`` (`int`): select_num

        - ``index_num`` (`int`): index_num

    返回值:
        无

    注意事项:
        无
    """
    core_idx = pl.get_core_index()
    core_num = pl.get_core_num()

    output_shape = [1, 1, index_num, inner_num]
    param_shape = [1, 1, select_num, inner_num]
    index_shape = [1, 1, index_num, 1]

    ptr_index.set_dtype(pl.pu32_t)

    output_g = pl.gtensor(output_shape, pl.GLOBAL, ptr_out)
    param_g = pl.gtensor(param_shape, pl.GLOBAL, ptr_param)
    index_g = pl.gtensor(index_shape, pl.GLOBAL, ptr_index)

    if (select_num < inner_num):
        inner_slice = pl.cast((inner_num + core_num - 1) // core_num, pl.int32)
        real_inner_slice = min(inner_slice, inner_num - core_idx * inner_slice)
        if real_inner_slice > 0:
            output_slice_shape = [1, 1, index_num, real_inner_slice]
            param_slice_shape = [1, 1, select_num, real_inner_slice]
            offset = [0, 0, 0, core_idx * inner_slice]
            pl.dma.gather_h(output_g.sub_view(output_slice_shape, offset),
                            param_g.sub_view(param_slice_shape, offset),
                            index_g, 0)
    else:
        index_slice = pl.cast((index_num + core_num - 1) // core_num, pl.int32)
        allocated_core = pl.cast((index_num + index_slice - 1) // index_slice,
                                 pl.int32)
        real_index_slice = min(index_slice, index_num - core_idx * index_slice)

        if core_idx < allocated_core:
            output_slice_shape = [1, 1, real_index_slice, inner_num]
            index_slice_shape = [1, 1, real_index_slice, 1]
            offset = [0, 0, core_idx * index_slice, 0]

            pl.dma.gather_h(output_g.sub_view(output_slice_shape,
                                              offset), param_g,
                            index_g.sub_view(index_slice_shape, offset))


if __name__ == '__main__':
    outer_num = 1
    select_num = 1024
    inner_num = 128
    index_num = 6

    param = torch.rand((select_num, inner_num), dtype=torch.float)
    index = torch.randint(0, 10, (1, index_num), dtype=torch.int32)
    index_cpu = index.to(torch.int32)
    output_ppl = torch.zeros((index_num, inner_num), dtype=torch.float)

    embedding_kernel[(
        1,
        8,
    )](output_ppl, param, index, outer_num, inner_num, select_num, index_num)

    output_torch = torch.nn.functional.embedding(index_cpu, param)

    atol = 1e-3  # 绝对容忍度
    rtol = 1e-3  # 相对容忍度
    # print(output_torch)
    # print(output_ppl)
    assert torch.allclose(output_torch, output_ppl, atol=atol,
                          rtol=rtol), "result cmp failed"
