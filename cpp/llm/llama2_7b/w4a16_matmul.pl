#include "ppl.h"

using namespace ppl;

// __KERNEL__ void matmul_w4a16_rtrans_mc(fp16 *ptr_res, fp16 *ptr_left,
//                                        uint8 *ptr_right, uint8 *ptr_scale_zp,
//                                        fp16 *ptr_test, int8 *ptr_test_u8, int
//                                        M, int K, int N, int group_size, const
//                                        int m_slice, const int n_slice) {
template <bool IsSmallM>
void matmul_w4a16_rtrans_inner(fp16 *ptr_res, fp16 *ptr_left, uint8 *ptr_right,
                               uint8 *ptr_scale_zp, int M, const int K,
                               const int N, const int group_size,
                               const int m_slice, const int n_slice) {
  // assert(n_slice % LANE_NUM == 0);
  set_core_num(8);
  int core_num = get_core_num();
  int index = get_core_index();

  int n_core_slice = div_up(N, core_num);
  int n_per_core = min(n_core_slice, N - index * n_core_slice);

  if (n_per_core <= 0) {
    return;
  }
  int n_core_offset = index * n_core_slice;

  assert(K % group_size == 0);
  const int groups = K / group_size;
  const int scale_zp_w = div_up(groups * 5, 2);
  const int scale_w_u8 = groups * 2;

  // must be right trans
  dim4 left_global_shape = {1, M, 1, K};
  dim4 right_global_shape = {1, N, 1, K / 2};
  dim4 scale_zp_global_shape = {1, N, 1, scale_zp_w};
  dim4 res_global_shape = {1, M, 1, N};

  dim4 left_block_shape = {1, m_slice, 1, K};
  dim4 right_block_shape = {1, n_slice, 1, K};
  dim4 right_u8_block_shape = {1, n_slice, 1, K / 2};
  dim4 zp_block_shape = {1, n_slice, 1, groups};
  dim4 res_block_shape = {1, m_slice, 1, n_slice};

  // dim4 test_scale_shape = {1, N, 1, groups};
  // dim4 test_sub_shape = {1, N, 1, K};
  // auto test_gtensor = gtensor<fp16>(test_sub_shape, GLOBAL, ptr_test);
  // auto test_u8_gtensor = gtensor<int8>(test_sub_shape, GLOBAL, ptr_test_u8);

  auto left_gtensor = gtensor<fp16>(left_global_shape, GLOBAL, ptr_left);
  auto right_gtensor = gtensor<uint8>(right_global_shape, GLOBAL, ptr_right);
  auto scale_zp_gtensor =
      gtensor<uint8>(scale_zp_global_shape, GLOBAL, ptr_scale_zp);
  auto res_gtensor = gtensor<fp16>(res_global_shape, GLOBAL, ptr_res);

  dim4 scale_zp_block_shape = {1, n_core_slice, 1, scale_zp_w};
  dim4 scale_zp_real_shape = {1, n_per_core, 1, scale_zp_w};
  dim4 scale_zp_offset = {0, n_core_offset, 0, 0};
  auto scale_local =
      make_tensor<uint8>(scale_zp_block_shape, scale_zp_real_shape);
  dma::load(scale_local,
            scale_zp_gtensor.sub_view(scale_zp_real_shape, scale_zp_offset));

  int m_secs = div_up(M, m_slice);
  int n_secs = div_up(n_per_core, n_slice);
  bool is_left_slice = m_slice < M;
  dim4 left_total_shape = {1, M, 1, K};
  auto left_total = make_tensor<fp16>(left_block_shape, left_total_shape);
  if constexpr (IsSmallM) {
    if (!is_left_slice) {
      dma::load(left_total, left_gtensor);
    }
  }

  bool enable_core_interleave = n_secs * n_slice == n_per_core;

  for (int count = 0; count < m_secs * n_secs; ++count) {
    enable_pipeline();
    int idx_m = (count / n_secs) * m_slice;
    // int idx_n = (count % n_secs) * n_slice;
    int n_count = count % n_secs;
    int tmp_n_idx =
        enable_core_interleave ? (n_count + index) % n_secs : n_count;
    int idx_n = tmp_n_idx * n_slice;
    int cur_m = min(m_slice, M - idx_m);
    int cur_n = min(n_slice, n_per_core - idx_n);
    dim4 left_real_shape = {1, cur_m, 1, K};
    dim4 right_u8_real_shape = {1, cur_n, 1, K / 2};
    dim4 right_real_shape = {1, cur_n, 1, K};
    dim4 scale_zp_u8_block_shape = {1, n_slice, 1, scale_zp_w};
    dim4 scale_zp_u8_real_shape = {1, cur_n, 1, scale_zp_w};

    auto left_local = make_tensor<fp16>(left_block_shape, left_real_shape);
    auto right_local_u8_in =
        make_tensor<uint8>(right_u8_block_shape, right_u8_real_shape);

    dim4 left_offset = {0, idx_m, 0, 0};
    dim4 right_offset = {0, idx_n + n_core_offset, 0, 0};
    dim4 scale_zp_u8_offset = {0, idx_n, 0, 0};
    if constexpr (IsSmallM) {
      if (is_left_slice) {
        dma::load(left_local,
                  left_gtensor.sub_view(left_real_shape, left_offset));
      }
    } else {
      dma::load(left_local,
                left_gtensor.sub_view(left_real_shape, left_offset));
    }
    dma::load(right_local_u8_in,
              right_gtensor.sub_view(right_u8_real_shape, right_offset));
    auto scale_zp_local_u8 =
        scale_local.sub_view(scale_zp_u8_real_shape, scale_zp_u8_offset);
    // right u4 -> u8
    auto right_local_u4 = right_local_u8_in.view<uint4>();
    auto right_local_u8 =
        make_tensor<int8>(right_block_shape, right_real_shape);
    tiu::cast(right_local_u8, right_local_u4, RM_DOWN);
    // // 对齐
    // dma::store(test_u8_gtensor.sub_view(right_real_shape, right_offset),
    //            right_local_u8);

    // gather f16 scale
    dim4 scale_zp_f16_shape = {1, cur_n, scale_zp_w / 2, 1};
    auto scale_zp_local_f16 = scale_zp_local_u8.view<fp16>(scale_zp_f16_shape);
    dim4 scale_real_shape = {1, cur_n, groups, 1};
    dim4 scale_stride;
    aligned_stride_4d(&scale_stride, &scale_zp_f16_shape, 0, sizeof(fp16));
    auto scale_local_f16 =
        scale_zp_local_f16.view(scale_real_shape, scale_stride);
    // // 对齐
    // dim4 test_shape = {1, cur_n, 1, groups};
    // dma::store(test_gtensor.sub_view(test_shape, right_offset),
    //            scale_local_f16);

    // gather u4 zp
    dim4 zp_u8_shape = {1, cur_n, 1, groups / 2};
    dim4 zp_shape = {1, cur_n, 1, groups};
    dim4 zp_offset = {0, idx_n, 0, scale_w_u8};
    dim4 zp_stride;
    aligned_stride_4d(&zp_stride, &scale_zp_u8_real_shape, 0, sizeof(uint8));
    zp_stride.h = groups / 2;
    auto zp_local_u8 = scale_local.sub_view(zp_u8_shape, zp_offset)
                           .view(zp_u8_shape, zp_stride);
    // zp u4 -> i8
    auto zp_local_u8_ = make_tensor<uint8>(zp_block_shape, zp_shape);
    dim4 zp_local_u8_stride;
    aligned_stride_4d(&zp_local_u8_stride, &zp_shape, 0, sizeof(uint8));
    zp_local_u8_stride.w = 2;
    dim4 zp_local_u8_high_offset = {0, 0, 0, 0};
    dim4 zp_local_u8_low_offset = {0, 0, 0, 1};
    // lower 4bit to uint8

    tiu::bitwise_and(zp_local_u8_.sub_view(zp_u8_shape, zp_local_u8_high_offset)
                         .view(zp_u8_shape, zp_local_u8_stride),
                     zp_local_u8, 0xf);
    // higher 4bit to uint8
    tiu::shift(zp_local_u8_.sub_view(zp_u8_shape, zp_local_u8_low_offset)
                   .view(zp_u8_shape, zp_local_u8_stride),
               zp_local_u8, -4, RM_TOWARDS_ZERO);
    // //   对齐
    // dma::store(test_u8_gtensor.sub_view(test_shape, right_offset),
    //            zp_local_u8_.view(test_shape));

    // sub_res = right_i8 - zp_i8
    dim4 zp_shape_ = {1, cur_n, groups, 1};
    dim4 right_group_shape = {1, cur_n, groups, group_size};
    auto sub_res_u8 = make_tensor<int8>(right_block_shape, right_real_shape);
    tiu::sub(sub_res_u8.view(right_group_shape),
             right_local_u8.view(right_group_shape),
             zp_local_u8_.view(zp_shape_), 0, RM_HALF_AWAY_FROM_ZERO, true);
    // // 对齐
    // dma::store(test_u8_gtensor.sub_view(right_real_shape, right_offset),
    //            sub_res_u8);

    // sub_res i8 -> f16
    auto right_local_f16 =
        make_tensor<fp16>(right_block_shape, right_real_shape);
    tiu::cast(right_local_f16, sub_res_u8, RM_HALF_AWAY_FROM_ZERO);
    // // 对齐
    // dma::store(test_gtensor.sub_view(right_real_shape, right_offset),
    //            right_local_f16);

    // right_f16 = sub_res_f16 * scale_f16
    auto mul_res_f16 = make_tensor<fp16>(right_block_shape, right_real_shape);
    tiu::fmul(mul_res_f16.view(right_group_shape),
              right_local_f16.view(right_group_shape), scale_local_f16);
    // // 对齐
    // dma::store(test_gtensor.sub_view(right_real_shape, right_offset),
    //            mul_res_f16);

    bool l_trans = false;
    bool r_trans = true;
    bool rst_trans = false;
    dim4 res_real_shape = {1, cur_m, 1, cur_n};
    dim4 res_offset = {0, idx_m, 0, idx_n + n_core_offset};
    dim4 res_trans_block_shape = {1, n_slice, 1, m_slice};
    dim4 res_trans_real_shape = {1, cur_n, 1, cur_m};
    auto res_trans_local =
        make_tensor<fp16>(res_trans_block_shape, res_trans_real_shape);
    if constexpr (IsSmallM) {
      if (is_left_slice) {
        tiu::fmm2(res_trans_local, mul_res_f16, left_local, l_trans, r_trans,
                  rst_trans);
      } else {
        tiu::fmm2(res_trans_local, mul_res_f16, left_total, l_trans, r_trans,
                  rst_trans);
      }
    } else {
      tiu::fmm2(res_trans_local, mul_res_f16, left_local, l_trans, r_trans,
                rst_trans);
    }
    auto res_local = make_tensor<fp16>(res_block_shape, res_real_shape);
    tiu::transpose_wc(res_local, res_trans_local);
    // 对齐
    dma::store(res_gtensor.sub_view(res_real_shape, res_offset), res_local);
  }
}

__KERNEL__ void
matmul_w4a16_rtrans_small_m(fp16 *ptr_res, fp16 *ptr_left, uint8 *ptr_right,
                            uint8 *ptr_scale_zp, int M, const int K,
                            const int N, const int group_size,
                            const int m_slice, const int n_slice) {
  matmul_w4a16_rtrans_inner<true>(ptr_res, ptr_left, ptr_right, ptr_scale_zp, M,
                                  K, N, group_size, m_slice, n_slice);
}

__KERNEL__ void matmul_w4a16_rtrans(fp16 *ptr_res, fp16 *ptr_left,
                                    uint8 *ptr_right, uint8 *ptr_scale_zp,
                                    int M, const int K, const int N,
                                    const int group_size, const int m_slice,
                                    const int n_slice) {
  matmul_w4a16_rtrans_inner<false>(ptr_res, ptr_left, ptr_right, ptr_scale_zp,
                                   M, K, N, group_size, m_slice, n_slice);
}

__TEST__ void matmul_w4a16() {
  const int M = 128;
  const int K = 4096;
  const int N = 4096;
  const int m_slice = 500;
  const int n_slice = 128;
  const int group_size = 128;
  const int group = K / group_size;
  const int scale_zp_w = div_up(group * 5, 2); // 80
  dim4 res_shape = {1, M, 1, N};
  dim4 left_shape = {1, M, 1, K}; // non transpose
  dim4 right_shape = {1, N, 1, K / 2};
  dim4 scale_zp_shape = {1, N, 1, scale_zp_w};
  fp16 *res = rand<fp16>(&res_shape);
  fp16 *left = rand<fp16>(&left_shape, -1.0, 1.0);
  uint8 *right = rand<uint8>(&right_shape, 16, 32);
  uint8 *scale_zp = rand<uint8>(&scale_zp_shape, 0, 2);

  // dim4 scale_test_shape = {1, N, 1, group};
  // dim4 sub_test_shape = {1, N, 1, K};
  // fp16 *test = rand<fp16>(&sub_test_shape);
  // int8 *test_u8 = rand<int8>(&sub_test_shape);
  // matmul_w4a16_rtrans_mc(res, left, right, scale_zp, test, test_u8, M, K, N,
  //                        group_size, m_slice, n_slice);
  matmul_w4a16_rtrans_small_m(res, left, right, scale_zp, M, K, N, group_size,
                              m_slice, n_slice);
}
