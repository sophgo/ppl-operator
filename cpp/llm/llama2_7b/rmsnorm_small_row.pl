#include "ppl.h"
#include "ppl_wrapper_func.h"

using namespace ppl;

// int get_max_common_div(int v, int max_v) {
//   for (int i = max_v; i > 0; i--) {
//     if (v % i == 0)
//       return i;
//   }
//   return 1;
// }

__KERNEL__ void rmsnorm_small_row(fp16 *ptr_output, fp16 *ptr_input,
                                  fp16 *ptr_weight, fp16 *ptr_bias, float eps,
                                  bool with_weight, bool with_bias,
                                  const int row, int col, int row_slice,
                                  const int block_w) {
  ppl::set_core_num(8);
  // assert(row < LANE_NUM);
  int core_num = get_core_num();
  int core_index = get_core_index();
  if (core_index >= core_num)
    return;

  int r_per_core = div_up(row, core_num);
  int r_start = core_index * r_per_core;
  int r_end = min(r_start + r_per_core, row);

  // int row_slice = get_max_common_div(col, LANE_NUM);
  int col_split = col / row_slice;

  // int N = 1;
  // int C = row;
  // int H = 1;
  // int W = col;

  dim4 global_shape = {row, row_slice, 1, col_split};
  dim4 global_weight_shape = {1, row_slice, 1, col_split};
  auto in_gtensor = gtensor<fp16>(global_shape, GLOBAL, ptr_input);
  auto weight_gtensor = gtensor<fp16>(global_weight_shape, GLOBAL, ptr_weight);
  auto bias_gtensor = gtensor<fp16>(global_weight_shape, GLOBAL, ptr_bias);
  auto out_gtensor = gtensor<fp16>(global_shape, GLOBAL, ptr_output);

  dim4 local_in_block_shape = {r_per_core, LANE_NUM, 1, block_w};
  dim4 local_avg_block_shape = {r_per_core, LANE_NUM, 1, 1};
  dim4 local_weight_shape = {1, LANE_NUM, 1, block_w};

  bool is_input_split = block_w < col_split;

  auto local_in_total = tensor<fp16>(local_in_block_shape);
  auto weight_in_total =
      make_tensor<fp16>(local_weight_shape, global_weight_shape);
  if (!is_input_split) {
    dim4 in_global_offset = {r_start, 0, 0, 0};
    dim4 local_in_real_shape = {r_per_core, row_slice, 1, col_split};
    dma::load(local_in_total.view(local_in_real_shape),
              in_gtensor.sub_view(local_in_real_shape, in_global_offset));
    dma::load(weight_in_total, weight_gtensor);
  }

  for (auto r_idx = r_start; r_idx < r_end; r_idx += r_per_core) {
    auto avg_buffer = tensor<fp32>(local_avg_block_shape);
    tiu::fill(avg_buffer, eps);
    for (auto w_idx = 0; w_idx < col_split; w_idx += block_w) {
      enable_pipeline();
      int w = min(block_w, col_split - w_idx);
      dim4 local_in_shape = {r_per_core, row_slice, 1, w};
      dim4 input_global_offset = {r_start, 0, 0, w_idx};

      auto local_in = make_tensor<fp16>(local_in_block_shape, local_in_shape);
      if (is_input_split) {
        dma::load(local_in,
                  in_gtensor.sub_view(local_in_shape, input_global_offset));
      }
      auto local_in_fp32 =
          make_tensor<fp32>(local_in_block_shape, local_in_shape);
      if (is_input_split) {
        tiu::cast(local_in_fp32, local_in);
      } else {
        tiu::cast(local_in_fp32, local_in_total);
      }

      // tmp = x^2
      auto local_in_tmp =
          make_tensor<fp32>(local_in_block_shape, local_in_shape);
      tiu::fmul(local_in_tmp, local_in_fp32, local_in_fp32);
      dim4 sub_avg_shape = {r_per_core, row_slice, 1, 1};
      auto sub_avg = make_tensor<fp32>(local_avg_block_shape, sub_avg_shape);
      // avg(x^2)
      dim2 kernel_pool = {1, w};
      padding_t pad = {0, 0, 0, 0};
      dim2 stride = {1, 1};
      dim2 dilation = {1, 1};
      tiu::fpool_avg(sub_avg, local_in_tmp, &kernel_pool, &pad, &stride,
                     &dilation, 1.f);
      dim2 kernel_conv = {1, 1};
      float weight_c = 1.f / col;
      tiu::fconv(avg_buffer, sub_avg, weight_c, LANE_NUM, &kernel_conv, &stride,
                 &dilation, &pad, true, DT_FP32);
    }
    // avg(x^2) + exp

    // 1/sqrt(avg(x^2) + exp)
    auto local_mu = tensor<fp32>(local_avg_block_shape);
    tiu::frsqrt(local_mu, avg_buffer, 4);
    dim4 avg_stride;
    get_stride<fp32>(&avg_stride, &local_avg_block_shape, TPU_ALIGN);
    dim4 avg_bc_stride = {avg_stride.n, 0, 0, 0};
    dim4 avg_bc_shape = {r_per_core, row_slice, 1, 1};

    for (auto w_idx = 0; w_idx < col_split; w_idx += block_w) {
      enable_pipeline();
      int w = min(block_w, col_split - w_idx);
      dim4 local_in_shape = {r_per_core, row_slice, 1, w};
      dim4 input_global_offset = {r_start, 0, 0, w_idx};

      auto local_in = make_tensor<fp16>(local_in_block_shape, local_in_shape);
      if (is_input_split) {
        dma::load(local_in,
                  in_gtensor.sub_view(local_in_shape, input_global_offset));
      }
      auto local_in_fp32 =
          make_tensor<fp32>(local_in_block_shape, local_in_shape);
      if (is_input_split) {
        tiu::cast(local_in_fp32, local_in);
      } else {
        tiu::cast(local_in_fp32, local_in_total);
      }

      auto out_fp32 = make_tensor<fp32>(local_in_block_shape, local_in_shape);
      // 1/sqrt(avg(x^2)) * x
      tiu::fmul(out_fp32, local_in_fp32,
                local_mu.view(avg_bc_shape, avg_bc_stride));

      dim4 weight_real_shape = {1, row_slice, 1, w};
      auto local_weight_sub =
          make_tensor<fp16>(local_weight_shape, weight_real_shape);
      if (with_weight && is_input_split) {
        dim4 weight_offset = {0, 0, 0, w_idx};
        dma::load(local_weight_sub,
                  weight_gtensor.sub_view(weight_real_shape, weight_offset));
      }
      if (with_weight) {
        auto local_weight_sub_fp32 =
            make_tensor<fp32>(local_weight_shape, weight_real_shape);
        if (is_input_split) {
          tiu::cast(local_weight_sub_fp32, local_weight_sub);
        } else {
          tiu::cast(local_weight_sub_fp32, weight_in_total);
        }
        tiu::fmul(out_fp32, out_fp32, local_weight_sub_fp32);
      }

      if (with_bias) {
        dim4 bias_offset = {0, 0, 0, w_idx};
        dim4 bias_real_shape = {1, row_slice, 1, w};
        auto local_bias_sub =
            make_tensor<fp16>(local_weight_shape, bias_real_shape);
        dma::load(local_bias_sub,
                  bias_gtensor.sub_view(bias_real_shape, bias_offset));
        auto local_bias_sub_fp32 =
            make_tensor<fp32>(local_weight_shape, bias_real_shape);
        tiu::cast(local_bias_sub_fp32, local_bias_sub);
        tiu::fadd(out_fp32, out_fp32, local_bias_sub_fp32);
      }

      auto out = make_tensor<fp16>(local_in_block_shape, local_in_shape);
      tiu::cast(out, out_fp32);
      dma::store(out_gtensor.sub_view(local_in_shape, input_global_offset),
                 out);
    }
  }
}

__TEST__ void rms_norm() {
  const int N = 1;
  const int C = 8;
  const int H = 1;
  const int W = 4096;
  const int block_w = 1024;
  dim4 in_shape = {1, N * C, 1, H * W};
  fp16 *output = rand<fp16>(&in_shape);
  fp16 *input = rand<fp16>(&in_shape, -5.0f, 5.0f);
  dim4 weight_shape = {1, 1, 1, H * W};
  fp16 *weight = rand<fp16>(&weight_shape, -5.0f, 5.0f);
  fp16 *bias = rand<fp16>(&weight_shape, -5.0f, 5.0f);
  int row_slice = 64;
  rmsnorm_small_row(output, input, weight, bias, 0.000001, true, true, N * C,
                    H * W, row_slice, block_w);
}
