#include "ppl.h"

using namespace ppl;
__KERNEL__ void matmul_rtrans_mc(fp16 *ptr_res, fp16 *ptr_left, fp16 *ptr_right,
                                 int batch, int M, const int K, int N,
                                 const int m_slice, const int n_slice) {
  // const int m_slice = 6;
  // const int n_slice = 768;
  set_core_num(8);
  int core_num = get_core_num();
  int index = get_core_index();

  int n_core_slice = div_up(N, core_num);
  int n_per_core = min(n_core_slice, N - index * n_core_slice);

  if (n_per_core <= 0) {
    return;
  }

  int n_core_offset = index * n_core_slice;

  // must be right trans
  dim4 left_global_shape = {batch, M, 1, K};
  dim4 right_global_shape = {batch, N, 1, K};
  dim4 res_global_shape = {batch, M, 1, N};

  dim4 left_block_shape = {1, m_slice, 1, K};
  dim4 right_block_shape = {1, n_slice, 1, K};
  dim4 res_block_shape = {1, m_slice, 1, n_slice};

  auto left_gtensor = gtensor<fp16>(left_global_shape, GLOBAL, ptr_left);
  auto right_gtensor = gtensor<fp16>(right_global_shape, GLOBAL, ptr_right);
  auto res_gtensor = gtensor<fp16>(res_global_shape, GLOBAL, ptr_res);

  int m_secs = div_up(M, m_slice);
  int n_secs = div_up(n_per_core, n_slice);

  int m_stride = n_secs;
  int n_stride = 1;
  for (int i = 0; i < batch; ++i) {
    for (int count = 0; count < m_stride * m_secs; ++count) {
      ppl::enable_pipeline();
      int remain = count;
      int m_count = remain / m_stride;
      remain %= m_stride;
      int n_count = remain / n_stride;
      int k_count = remain % n_stride;

      int idx_m = m_count * m_slice;
      int idx_n = n_count * n_slice;

      int cur_m = min(m_slice, M - idx_m);
      int cur_n = min(n_slice, n_per_core - idx_n);
      int cur_k = K;

      bool result_add = false;
      data_type_t dtype = DT_FP16;

      dim4 left_real_shape = {1, cur_m, 1, cur_k};
      dim4 right_real_shape = {1, cur_n, 1, cur_k};
      dim4 res_real_shape = {1, cur_m, 1, cur_n};

      auto left_local = make_tensor<fp16>(left_block_shape, left_real_shape);
      auto right_local = make_tensor<fp16>(right_block_shape, right_real_shape);
      auto res_local = make_tensor<fp16>(res_block_shape, res_real_shape);

      dim4 left_offset = {i, idx_m, 0, 0};
      dim4 right_offset = {i, idx_n + n_core_offset, 0, 0};
      dim4 res_offset = {i, idx_m, 0, idx_n + n_core_offset};
      dma::load(left_local,
                left_gtensor.sub_view(left_real_shape, left_offset));
      dma::load(right_local,
                right_gtensor.sub_view(right_real_shape, right_offset));

      tiu::fmm2(res_local, left_local, right_local, false, true, false, false,
                result_add, dtype);
      dma::store(res_gtensor.sub_view(res_real_shape, res_offset),
                 res_local.view<fp16>());
    }
  }
}

__TEST__ void matmul() {
  int batch = 2;
  const int M = 6;
  const int K = 1024;
  const int N = 6;
  const int m_slice = 6;
  const int n_slice = 6;
  dim4 res_shape = {batch, M, 1, N};
  dim4 left_shape = {batch, M, 1, K}; // non transpose
  dim4 right_shape = {batch, N, 1, K};
  fp16 *res = rand<fp16>(&res_shape, -1.0, 1.0);
  fp16 *left = rand<fp16>(&left_shape, -1.0, 1.0);
  fp16 *right = rand<fp16>(&right_shape, -1.0, 1.0);

  matmul_rtrans_mc(res, left, right, batch, M, K, N, m_slice, n_slice);
}
