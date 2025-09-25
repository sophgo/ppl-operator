import torch

import ppl
import ppl.language as pl
import struct
import numpy as np

@ppl.jit
def matmul_rtrans_mc_kernel(out_ptr, left_ptr, right_ptr,
                     batch: pl.constexpr,
                     M,
                     K:pl.constexpr,
                     N,
                     m_slice: pl.constexpr,
                     n_slice: pl.constexpr):
    """
    w16a16_matmul

        .. code-block:: python

            matmul_rtrans_mc_kernel(out_ptr, left_ptr, right_ptr, batch, M, K, N, m_slice, k_slice, n_slice)

    参数:
        - ``out_ptr`` (`ppl.language.tensor`): output张量

        - ``left_ptr`` (`ppl.language.tensor`): 左矩阵张量

        - ``right_ptr`` (`ppl.language.tensor`): 右矩阵张量(转置)

        - ``batch`` (`int`): output矩阵的batch

        - ``M`` (`int`): output矩阵的M

        - ``K`` (`int`): 左矩阵的K

        - ``N`` (`int`): output矩阵的N

        - ``m_slice`` (`int`): M的切分slice

        - ``n_slice`` (`int`): N的切分slice

    返回值:
        无

    注意事项:
        输入右矩阵转置
    """
    core_num = pl.get_core_num()
    index = pl.get_core_index()

    n_core_slice = pl.cdiv(N, core_num)
    n_per_core = min(n_core_slice, N - index * n_core_slice)
    if (n_per_core <= 0):
        pass

    n_core_offset = index * n_core_slice
    left_global_shape = [batch, M, 1, K]
    right_global_shape = [batch, N, 1, K]
    res_global_shape = [batch, M, 1, N]

    left_block_shape = [1, m_slice, 1, K]
    right_block_shape = [1, n_slice, 1, K]
    res_block_shape = [1, m_slice, 1, n_slice]

    left_gtensor = pl.gtensor(left_global_shape, pl.GLOBAL, left_ptr)
    right_gtensor = pl.gtensor(right_global_shape, pl.GLOBAL, right_ptr)
    res_gtensor = pl.gtensor(res_global_shape, pl.GLOBAL, out_ptr)

    m_secs = pl.cdiv(M, m_slice)
    n_secs = pl.cdiv(n_per_core, n_slice)

    m_stride = n_secs
    n_stride = 1
    for i in range(batch):
        for count in range(m_stride * m_secs):
            # pl.enable_pipeline()
            remain = count
            m_count = remain // m_stride
            remain %= m_stride
            n_count = remain // n_stride

            idx_m = m_count * m_slice
            idx_n = n_count * n_slice

            cur_m = min(m_slice, M - idx_m)
            cur_n = min(n_slice, n_per_core - idx_n)

            left_real_shape = [1, cur_m, 1, K]
            right_real_shape = [1, cur_n, 1, K]
            res_real_shape = [1, cur_m, 1, cur_n]

            left_local = pl.make_tensor(left_block_shape, left_ptr.dtype, left_real_shape)
            right_local = pl.make_tensor(right_block_shape, right_ptr.dtype, right_real_shape)
            res_local = pl.make_tensor(res_block_shape, out_ptr.dtype, res_real_shape)
            left_offset = [i, idx_m, 0, 0]
            right_offset = [i, idx_n + n_core_offset, 0, 0]
            res_offset = [i, idx_m, 0, idx_n + n_core_offset]
            pl.dma.load(left_local,
                left_gtensor[i:i+1, idx_m:idx_m+cur_m, :, :])
            pl.dma.load(right_local,
                right_gtensor.sub_view(right_real_shape, right_offset))

            pl.tiu.dot(res_local, left_local, right_local, rtrans=True)
            pl.dma.store(res_gtensor.sub_view(res_real_shape, res_offset),
                 res_local)
batch = 2
M = 6
K = 1024
N = 6
m_slice = M
n_slice = N
left = torch.rand((batch, M, 1, K), device='cpu', dtype=torch.half)
right = torch.rand((batch, N, 1, K), device='cpu', dtype=torch.half)
output = torch.empty((batch, M, 1, N), device='cpu', dtype=torch.half)
cpu_res = torch.matmul(left.reshape([batch, M, K]).to(torch.float), right.reshape([batch, N, K]).transpose(1,2).to(torch.float))
matmul_rtrans_mc_kernel[(1,8)](output, left, right, batch, M, K, N, m_slice, n_slice)
atol = 1e-4  # 绝对容忍度
rtol = 1e-5  # 相对容忍度
assert torch.allclose(cpu_res.reshape([batch, M, 1, N]).to(output.dtype), output, atol=atol, rtol=rtol) , "result cmp failed"
