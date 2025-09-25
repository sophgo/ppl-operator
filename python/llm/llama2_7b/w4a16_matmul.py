import torch
import functools

import ppl
import ppl.language as pl
import struct
import numpy as np
import os
import torch.nn.functional as F


def matmul_tiling(fn, output, left, right, scale_zp, M, K, N, group_size,
                  m_slice, n_slice, smallM):
    if smallM:
        n_slice = 128
        while m_slice > 0:
            ret = fn(output, left, right, scale_zp, M, K, N, group_size,
                     m_slice, n_slice, smallM)
            if ret.errorCode == 0:
                print("m_slice:%d, n_slice:%d" % (m_slice, n_slice))
                return ret
            else:
                m_slice = m_slice // 2
    else:
        n_slice_start = n_slice
        while m_slice > 0:
            n_slice = n_slice_start
            while n_slice > 0:
                ret = fn(output, left, right, scale_zp, M, K, N, group_size,
                         m_slice, n_slice, smallM)
                if ret.errorCode == 0:
                    print("m_slice:%d, n_slice:%d" % (m_slice, n_slice))
                    return ret
                else:
                    n_slice = n_slice // 2
            m_slice = m_slice // 2

    raise RuntimeError("[!Error]: matmul tiling failed")


@ppl.jit(tiling=matmul_tiling, debug=False)
def matmul_w4a16_rtrans_mc_kernel(out_ptr, left_ptr, right_ptr, scale_zp_ptr,
                                  M, K: pl.constexpr, N: pl.constexpr,
                                  group_size: pl.constexpr,
                                  m_slice: pl.constexpr, n_slice: pl.constexpr,
                                  smallM: pl.constexpr):
    """
    w4a16_matmul

        .. code-block:: python

            matmul_w4a16_rtrans_mc_kernel(out_ptr, left_ptr, right_ptr, scale_zp_ptr, M, K, N, group_size, m_slice, n_slice)

    参数:
        - ``out_ptr`` (`ppl.language.tensor`): output张量

        - ``left_ptr`` (`ppl.language.tensor`): 左矩阵张量

        - ``right_ptr`` (`ppl.language.tensor`): 右矩阵张量

        - ``scale_zp_ptr`` (`ppl.language.tensor`): 右矩阵scale_zp 张量

        - ``M`` (`int`): output矩阵的M

        - ``K`` (`int`): 左矩阵的K

        - ``N`` (`int`): output矩阵的N

        - ``group_size`` (`int`): group_size

        - ``m_slice`` (`int`): M的切分slice

        - ``n_slice`` (`int`): N的切分slice

        - ``smallM`` (`bool`): M是否比较小

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

    groups = K // group_size
    scale_zp_w = pl.cdiv(groups * 5, 2)
    scale_w_u8 = groups * 2

    left_global_shape = [1, M, 1, K]
    right_global_shape = [1, N, 1, K // 2]
    scale_zp_global_shape = [1, N, 1, scale_zp_w]
    res_global_shape = [1, M, 1, N]

    left_block_shape = [1, m_slice, 1, K]
    right_block_shape = [1, n_slice, 1, K]
    right_u8_block_shape = [1, n_slice, 1, K // 2]
    zp_block_shape = [1, n_slice, 1, groups]
    res_block_shape = [1, m_slice, 1, n_slice]

    left_gtensor = pl.gtensor(left_global_shape, pl.GLOBAL, left_ptr)

    right_gtensor = pl.gtensor(right_global_shape, pl.GLOBAL, right_ptr)
    scale_zp_gtensor = pl.gtensor(scale_zp_global_shape, pl.GLOBAL,
                                  scale_zp_ptr)
    res_gtensor = pl.gtensor(res_global_shape, pl.GLOBAL, out_ptr)

    scale_zp_block_shape = [1, n_core_slice, 1, scale_zp_w]
    scale_zp_real_shape = [1, n_per_core, 1, scale_zp_w]
    scale_zp_offset = [0, n_core_offset, 0, 0]
    scale_local = pl.make_tensor(scale_zp_block_shape, scale_zp_ptr.dtype,
                                 scale_zp_real_shape)
    pl.dma.load(
        scale_local,
        scale_zp_gtensor[:, n_core_offset:n_core_offset + n_per_core, :, :])

    m_secs = pl.cdiv(M, m_slice)
    n_secs = pl.cdiv(n_per_core, n_slice)
    is_left_slice = m_slice < M
    left_total_shape = [1, M, 1, K]
    left_total = pl.make_tensor(left_block_shape, left_ptr.dtype,
                                left_total_shape)
    if smallM:
        if not is_left_slice:
            pl.dma.load(left_total, left_gtensor)

    enable_core_interleave = n_secs * n_slice == n_per_core

    for count in range(0, m_secs * n_secs, 1):
        pl.enable_pipeline()
        idx_m = (count // n_secs) * m_slice

        n_count = count % n_secs
        tmp_n_idx = (n_count +
                     index) % n_secs if enable_core_interleave else n_count
        idx_n = tmp_n_idx * n_slice
        cur_m = min(m_slice, M - idx_m)
        cur_n = min(n_slice, n_per_core - idx_n)
        left_real_shape = [1, cur_m, 1, K]
        right_u8_real_shape = [1, cur_n, 1, K // 2]
        right_real_shape = [1, cur_n, 1, K]
        scale_zp_u8_block_shape = [1, n_slice, 1, scale_zp_w]
        scale_zp_u8_real_shape = [1, cur_n, 1, scale_zp_w]

        left_local = pl.make_tensor(left_block_shape, left_ptr.dtype,
                                    left_real_shape)

        right_local_u8_in = pl.make_tensor(right_u8_block_shape,
                                           right_ptr.dtype,
                                           right_u8_real_shape)
        left_offset = [0, idx_m, 0, 0]
        right_offset = [0, idx_n + n_core_offset, 0, 0]
        scale_zp_u8_offset = [0, idx_n, 0, 0]
        if smallM:
            if (is_left_slice):
                pl.dma.load(left_local,
                            left_gtensor[:, idx_m:idx_m + cur_m, :, :])
        else:
            pl.dma.load(left_local, left_gtensor[:, idx_m:idx_m + cur_m, :, :])

        pl.dma.load(
            right_local_u8_in, right_gtensor[:, idx_n + n_core_offset:idx_n +
                                             n_core_offset + cur_n, :, :])
        scale_zp_local_u8 = scale_local[:, idx_n:idx_n + cur_n, :, :]
        # right u4 -> u8
        right_local_u4 = right_local_u8_in.view(dtype=pl.uint4)

        right_local_u8 = pl.make_tensor(right_block_shape, pl.int8,
                                        right_real_shape)
        pl.tiu.cast(right_local_u8,
                    right_local_u4,
                    type=pl.int8,
                    mode=pl.RM_DOWN)
        # gather f16 scale
        scale_zp_f16_shape = [1, cur_n, scale_zp_w // 2, 1]
        scale_zp_local_f16 = scale_zp_local_u8.view(shape=scale_zp_f16_shape,
                                                    dtype=pl.float16)
        scale_real_shape = [1, cur_n, groups, 1]

        stride_n, stride_c, stride_h, stride_w = \
                    pl.aligned_stride_4d(1, cur_n, scale_zp_w // 2, 1, pl.TPU_ALIGN, 0, pl.float16)
        scale_stride = [stride_n, stride_c, stride_h, stride_w]
        scale_local_f16 = scale_zp_local_f16.view(shape=scale_real_shape,
                                                  stride=scale_stride)

        #gather u4 zp
        zp_u8_shape = [1, cur_n, 1, groups // 2]
        zp_shape = [1, cur_n, 1, groups]
        zp_offset = [0, idx_n, 0, scale_w_u8]

        zp_stride_n, zp_stride_c, _, zp_stride_w = pl.aligned_stride_4d(
            1, cur_n, 1, scale_zp_w, pl.TPU_ALIGN, 0, pl.uint8)
        zp_stride = [zp_stride_n, zp_stride_c, groups // 2, zp_stride_w]
        zp_local_u8 = scale_local[:, idx_n:idx_n + cur_n, :,
                                  scale_w_u8:scale_w_u8 + groups // 2].view(
                                      shape=zp_u8_shape, stride=zp_stride)
        #zp u4 -> i8
        zp_local_u8_ = pl.make_tensor(zp_block_shape, pl.uint8, zp_shape)

        zp_local_u8_stride_n, zp_local_u8_stride_c, zp_local_u8_stride_h, _  = \
                        pl.aligned_stride_4d(1, cur_n, 1, groups, pl.TPU_ALIGN, 0, pl.uint8)
        zp_local_u8_stride = [
            zp_local_u8_stride_n, zp_local_u8_stride_c, zp_local_u8_stride_h, 2
        ]
        zp_local_u8_high_offset = [0, 0, 0, 0]
        zp_local_u8_low_offset = [0, 0, 0, 1]
        #lower 4bit to uint8
        pl.tiu.bitwise_and(
            zp_local_u8_.sub_view(zp_u8_shape, zp_local_u8_high_offset).view(
                shape=zp_u8_shape, stride=zp_local_u8_stride), zp_local_u8,
            0xf)

        #higher 4bit to uint8
        pl.tiu.shift(zp_local_u8_.sub_view(zp_u8_shape,
                                           zp_local_u8_low_offset).view(
                                               shape=zp_u8_shape,
                                               stride=zp_local_u8_stride),
                     zp_local_u8,
                     shift=-4,
                     r_mode=pl.RM_TOWARDS_ZERO)

        # sub_res = right_i8 - zp_i8
        zp_shape_ = [1, cur_n, groups, 1]
        right_group_shape = [1, cur_n, groups, group_size]
        sub_res_u8 = pl.make_tensor(right_block_shape, pl.int8,
                                    right_real_shape)
        pl.tiu.sub(sub_res_u8.view(shape=right_group_shape),
                   right_local_u8.view(shape=right_group_shape),
                   zp_local_u8_.view(shape=zp_shape_),
                   shift=0,
                   mode=pl.RM_HALF_AWAY_FROM_ZERO,
                   saturation=True)

        #sub_res i8 -> f16
        right_local_f16 = pl.make_tensor(right_block_shape, pl.float16,
                                         right_real_shape)
        pl.tiu.cast(right_local_f16,
                    sub_res_u8,
                    type=pl.float16,
                    mode=pl.RM_HALF_AWAY_FROM_ZERO)

        #right_f16 = sub_res_f16 * scale_f16
        mul_res_f16 = pl.make_tensor(right_block_shape, pl.float16,
                                     right_real_shape)
        pl.tiu.fmul(mul_res_f16.view(shape=right_group_shape),
                    right_local_f16.view(shape=right_group_shape),
                    scale_local_f16)

        l_trans = False
        r_trans = True
        rst_trans = False
        res_real_shape = [1, cur_m, 1, cur_n]
        res_offset = [0, idx_m, 0, idx_n + n_core_offset]
        res_trans_block_shape = [1, n_slice, 1, m_slice]
        res_trans_real_shape = [1, cur_n, 1, cur_m]
        res_trans_local = pl.make_tensor(res_trans_block_shape, pl.float16,
                                         res_trans_real_shape)

        if smallM:
            if (is_left_slice):
                pl.tiu.fmm2(res_trans_local,
                            mul_res_f16,
                            left_local,
                            ltrans=l_trans,
                            rtrans=r_trans,
                            rst_trans=rst_trans)
            else:
                pl.tiu.fmm2(res_trans_local,
                            mul_res_f16,
                            left_total,
                            ltrans=l_trans,
                            rtrans=r_trans,
                            rst_trans=rst_trans)
        else:
            pl.tiu.fmm2(res_trans_local,
                        mul_res_f16,
                        left_local,
                        ltrans=l_trans,
                        rtrans=r_trans,
                        rst_trans=rst_trans,
                        out_dtype=pl.float16)

        res_local = pl.make_tensor(res_block_shape, pl.float16, res_real_shape)
        pl.tiu.transpose_wc(res_local, res_trans_local)
        pl.dma.store(
            res_gtensor[:, idx_m:idx_m + cur_m, :,
                        idx_n + n_core_offset:idx_n + n_core_offset + cur_n],
            res_local)


#case 2(if need autoting or use at tgi/tpu-train, pls follow case 2): ok
def matmul_v2(left, right, scale_zp, M, K, N, group_size, m_slice, n_slice):
    output = torch.empty((1, M, 1, N), device='cpu', dtype=left.dtype)
    matmul_w4a16_rtrans_mc_kernel[(1, 8)](output, left, right, scale_zp, M, K,
                                          N, group_size, m_slice, n_slice,
                                          False)
    return output


def matmul_v1(left, right, scale_zp, M, K, N, group_size, m_slice, n_slice):
    output = torch.empty((1, M, 1, N), device='cpu', dtype=left.dtype)
    matmul_w4a16_rtrans_mc_kernel[(1, 8)](output, left, right, scale_zp, M, K,
                                          N, group_size, m_slice, n_slice,
                                          True)
    return output


def cpu_matmul(left, right, scale_zp, M, K, N, group_size):
    tensor_left = torch.tensor(left.reshape([M, K]), dtype=torch.float)
    tensor_right = torch.tensor(right.reshape([N, K // 2]), dtype=torch.float)
    tensor_right_uint8 = tensor_right.to(torch.uint8)
    right_high_4_bits = tensor_right_uint8 >> 4
    right_low_4_bits = tensor_right_uint8 & 0x0F
    right_uint4_tensor = torch.stack((right_low_4_bits, right_high_4_bits),
                                     dim=-1).reshape([N, K])
    right_int8_tensor = right_uint4_tensor.to(torch.int8)
    group = K // group_size
    scale_zp_w = -(-group * 5 // 2)
    tensor_scale_zp = torch.tensor(scale_zp.reshape([N, scale_zp_w]),
                                   dtype=torch.float)
    scale_zp_uint8_tensor = tensor_scale_zp.to(torch.uint8)
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
    zp_int8_tensor = torch.stack((low_nibble, high_nibble),
                                 dim=-1).view(N, group).to(torch.int8)

    sub_res_tensor = right_int8_tensor.reshape(
        [N, group, group_size]) - zp_int8_tensor.reshape([N, group, 1])
    sub_res_f16_tensor = sub_res_tensor.to(torch.float16)
    mul_res_f16 = sub_res_f16_tensor.reshape(
        [N, group, group_size]) * scale_fp16_tensor.reshape([N, group, 1])
    torch_out = F.linear(tensor_left,
                         mul_res_f16.reshape([N, K]).to(torch.float)).reshape(
                             [1, M, 1, N]).to(torch.half)
    return torch_out


torch.manual_seed(0)
delta = 0.005
atol = 1e-4  # 绝对容忍度
rtol = 1e-5  # 相对容忍度
M = 128
K = 4096
N = 4096
m_slice = M
n_slice = N
group_size = 128
group = K // group_size
scale_zp_w = -(-group * 5 // 2)
a, b = -1, 1
left = torch.randn((1, M, 1, K), device='cpu', dtype=torch.half)
left = a + (b - a) * (left - left.min()) / (left.max() - left.min())
right = torch.randint(16,
                      32, (1, N, 1, K // 2),
                      device='cpu',
                      dtype=torch.uint8)
scale_zp = torch.randint(0,
                         2, (1, N, 1, scale_zp_w),
                         device='cpu',
                         dtype=torch.uint8)

cpu_res = cpu_matmul(left, right, scale_zp, M, K, N, group_size)
ppl_res_v2 = matmul_v2(left, right, scale_zp, M, K, N, group_size, m_slice,
                       n_slice)
assert torch.allclose(cpu_res, ppl_res_v2, atol=atol,
                      rtol=rtol), "v2 result cmp failed"

M = 8
K = 4096
N = 4096
m_slice = M
n_slice = N
group_size = 128
group = K // group_size
scale_zp_w = -(-group * 5 // 2)
left = torch.randn((1, M, 1, K), device='cpu', dtype=torch.half)
right = torch.randint(16,
                      32, (1, N, 1, K // 2),
                      device='cpu',
                      dtype=torch.uint8)
scale_zp = torch.randint(0,
                         2, (1, N, 1, scale_zp_w),
                         device='cpu',
                         dtype=torch.uint8)
cpu_res_v1 = cpu_matmul(left, right, scale_zp, M, K, N, group_size)
ppl_res_v1 = matmul_v1(left, right, scale_zp, M, K, N, group_size, m_slice,
                       n_slice)
assert torch.allclose(cpu_res_v1, ppl_res_v1, atol=atol,
                      rtol=rtol), "v1 result cmp failed"
