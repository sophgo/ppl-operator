#include "ppl.h"
#include "ppl_wrapper_func.h"

using namespace ppl;

__KERNEL__ void rms_norm_kernel(fp16 *ptr_output, fp16 *ptr_input,
                                fp16 *ptr_weight, fp16 *ptr_bias, float eps,
                                bool with_weight, bool with_bias, int _N,
                                int _C, int _H, int _W, const int block_w) {
  ppl::set_core_num(8);
  int core_num = get_core_num();
  int core_index = get_core_index();
  if (core_index >= core_num)
    return;

  int N = 1;
  int C = _N * _C;
  int H = 1;
  int W = _H * _W;
  int c_per_core = div_up(C, core_num);
  int c_start = core_index * c_per_core;
  int c_end = min(c_start + c_per_core, C);
  int block_c = LANE_NUM;

  dim4 global_shape = {1, c_per_core, 1, W};
  dim4 global_weight_shape = {1, 1, 1, W};
  auto in_gtensor = gtensor<fp16>(global_shape, GLOBAL, ptr_input);
  auto weight_gtensor = gtensor<fp16>(global_weight_shape, GLOBAL, ptr_weight);
  auto bias_gtensor = gtensor<fp16>(global_weight_shape, GLOBAL, ptr_bias);
  auto out_gtensor = gtensor<fp16>(global_shape, GLOBAL, ptr_output);

  dim4 local_in_block_shape = {1, block_c, 1, block_w};
  dim4 local_avg_block_shape = {1, block_c, 1, 1};
  dim4 local_weight_shape = {1, 1, 1, block_w};

  for (auto c_idx = c_start; c_idx < c_end; c_idx += block_c) {
    int c = min(block_c, c_end - c_idx);
    dim4 local_avg_shape = {1, c, 1, 1};
    auto avg_buffer = make_tensor<fp32>(local_avg_block_shape, local_avg_shape);
    tiu::fill(avg_buffer, eps);

    for (auto w_idx = 0; w_idx < W; w_idx += block_w) {
      enable_pipeline();
      int w = min(block_w, W - w_idx);
      dim4 local_in_shape = {1, c, 1, w};
      dim4 input_global_offset = {0, c_idx, 0, w_idx};

      auto local_in = make_tensor<fp16>(local_in_block_shape, local_in_shape);
      dma::load(local_in,
                in_gtensor.sub_view(local_in_shape, input_global_offset));
      auto local_in_fp32 =
          make_tensor<fp32>(local_in_block_shape, local_in_shape);
      tiu::cast(local_in_fp32, local_in);

      // tmp = x^2
      auto local_in_tmp =
          make_tensor<fp32>(local_in_block_shape, local_in_shape);
      tiu::fmul(local_in_tmp, local_in_fp32, local_in_fp32);
      auto sub_avg = make_tensor<fp32>(local_avg_block_shape, local_avg_shape);
      // avg(x^2)
      quick_pooling(sub_avg, local_in_tmp, &local_in_block_shape,
                    &local_in_shape, 0, 1, 1.f / W);
      // avg(x^2) + exp
      tiu::fadd(avg_buffer, avg_buffer, sub_avg);
    }

    // 1/sqrt(avg(x^2) + exp)
    auto local_mu = make_tensor<fp32>(local_avg_block_shape, local_avg_shape);
    tiu::frsqrt(local_mu, avg_buffer, 4);

    for (auto w_idx = 0; w_idx < W; w_idx += block_w) {
      enable_pipeline();
      int w = min(block_w, W - w_idx);
      dim4 local_in_shape = {1, c, 1, w};
      dim4 input_global_offset = {0, c_idx, 0, w_idx};

      auto local_in = make_tensor<fp16>(local_in_block_shape, local_in_shape);
      dma::load(local_in,
                in_gtensor.sub_view(local_in_shape, input_global_offset));
      auto local_in_fp32 =
          make_tensor<fp32>(local_in_block_shape, local_in_shape);
      tiu::cast(local_in_fp32, local_in);

      auto out_fp32 = make_tensor<fp32>(local_in_block_shape, local_in_shape);
      // 1/sqrt(avg(x^2)) * x
      tiu::fmul(out_fp32, local_in_fp32, local_mu);

      if (with_weight) {
        dim4 weight_offset = {0, 0, 0, w_idx};
        dim4 weight_real_shape = {1, 1, 1, w};
        auto local_weight_sub =
            make_tensor<fp16>(local_weight_shape, weight_real_shape);
        dma::load(local_weight_sub,
                  weight_gtensor.sub_view(weight_real_shape, weight_offset));
        auto local_weight_sub_fp32 =
            make_tensor<fp32>(local_weight_shape, weight_real_shape);
        tiu::cast(local_weight_sub_fp32, local_weight_sub);
        tiu::broadcast(local_in_fp32, local_weight_sub_fp32);
        tiu::fmul(out_fp32, out_fp32, local_in_fp32);
      }

      if (with_bias) {
        dim4 bias_offset = {0, 0, 0, w_idx};
        dim4 bias_real_shape = {1, 1, 1, w};
        auto local_bias_sub =
            make_tensor<fp16>(local_weight_shape, bias_real_shape);
        dma::load(local_bias_sub,
                  bias_gtensor.sub_view(bias_real_shape, bias_offset));
        auto local_bias_sub_fp32 =
            make_tensor<fp32>(local_weight_shape, bias_real_shape);
        tiu::cast(local_bias_sub_fp32, local_bias_sub);
        tiu::broadcast(local_in_fp32, local_bias_sub_fp32);
        tiu::fadd(out_fp32, out_fp32, local_in_fp32);
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
  const int C = 1024;
  const int H = 1;
  const int W = 4096;
  const int block_w = 2048;
  dim4 in_shape = {1, N * C, 1, H * W};
  fp16 *output = rand<fp16>(&in_shape);
  fp16 *input = rand<fp16>(&in_shape, -5.0f, 5.0f);
  dim4 weight_shape = {1, 1, 1, H * W};
  fp16 *weight = rand<fp16>(&weight_shape, -5.0f, 5.0f);
  fp16 *bias = rand<fp16>(&weight_shape, -5.0f, 5.0f);
  rms_norm_kernel(output, input, weight, bias, 0.000001, true, true, N, C, H, W,
                  block_w);
}
