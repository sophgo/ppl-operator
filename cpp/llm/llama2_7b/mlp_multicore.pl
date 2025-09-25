#include "ppl.h"
#include "ppl_wrapper_func.h"

using namespace ppl;

#define CORE_NUM 8

__KERNEL__ void llma_mlp_mulit_core_kernel_bf16(
    fp16 *ptr_input, fp16 *ptr_weight0, fp16 *ptr_weight1, fp16 *ptr_weight2,
    fp16 *ptr_output, int batch, int input_w, int middle_w,
    const int g_core_num, const int block_b, const int block_iw,
    const int block_w) {

  ppl::set_core_num(g_core_num);
  int core_num = get_core_num();
  int core_idx = get_core_index();
  int b_loop = div_up(batch, NPU_NUM);
  for (int b_idx = 0; b_idx < b_loop; b_idx++) {
    int b_offset = b_idx * NPU_NUM;
    int b_slice = min(NPU_NUM, batch - b_offset);
    dim4 g_input_shape = {1, batch, 1, input_w};
    dim4 g_offset = {0, b_offset, 0, 0};

    int slice_per_core = div_up(middle_w, core_num);
    int core_offset = slice_per_core * core_idx;
    slice_per_core = min(slice_per_core, middle_w - core_offset);
    dim4 input_shape = {1, b_slice, 1, input_w};
    dim4 input_block_shape = {1, block_b, 1, block_iw};
    auto _l2_out_tensor = gtensor<fp16>(input_block_shape, L2);
    auto l2_out_tensor = _l2_out_tensor.view(input_shape);

    if (slice_per_core > 0) {
      dim4 weight0_global_shape = {1, input_w, 1, middle_w};
      dim4 weight2_global_shape = {1, middle_w, 1, input_w};

      dim4 weight0_block_shape = {1, block_iw, 1, block_w};
      dim4 weight2_block_shape = {1, block_w, 1, block_iw};
      dim4 middle_buffer_shape = {1, block_b, 1, block_w};

      auto _input_gtensor = gtensor<fp16>(g_input_shape, GLOBAL, ptr_input);
      auto input_gtensor = _input_gtensor.sub_view(input_shape, g_offset);

      auto weight0_gtensor =
          gtensor<fp16>(weight0_global_shape, GLOBAL, ptr_weight0);
      auto weight1_gtensor =
          gtensor<fp16>(weight0_global_shape, GLOBAL, ptr_weight1);
      auto weight2_gtensor =
          gtensor<fp16>(weight2_global_shape, GLOBAL, ptr_weight2);

      auto input_local = make_tensor<fp16>(input_block_shape, input_shape);
      auto out_f32_local = make_tensor<fp32>(input_block_shape, input_shape);

      dma::load(input_local, input_gtensor);
      dma::fill(out_f32_local, 0);

      for (auto w_idx = 0; w_idx < slice_per_core; w_idx += block_w) {
        ppl::enable_pipeline();
        int middle_slice = min(block_w, slice_per_core - w_idx);
        dim4 weight0_shape = {1, input_w, 1, middle_slice};
        dim4 weight2_shape = {1, middle_slice, 1, input_w};
        dim4 weight0_offset = {0, 0, 0, core_offset + w_idx};
        dim4 weight2_offset = {0, core_offset + w_idx, 0, 0};
        auto weight0_local =
            make_tensor<fp16>(weight0_block_shape, weight0_shape);
        auto weight1_local =
            make_tensor<fp16>(weight0_block_shape, weight0_shape);
        auto weight2_local =
            make_tensor<fp16>(weight2_block_shape, weight2_shape);

        dma::load(weight0_local,
                  weight0_gtensor.sub_view(weight0_shape, weight0_offset));
        dma::load(weight1_local,
                  weight1_gtensor.sub_view(weight0_shape, weight0_offset));
        dma::load(weight2_local,
                  weight2_gtensor.sub_view(weight2_shape, weight2_offset));
        dim4 middle_real_shape = {1, b_slice, 1, middle_slice};
        auto middle_buffer_f16_local_1 =
            make_tensor<fp16>(middle_buffer_shape, middle_real_shape);
        auto middle_buffer_f16_local_2 =
            make_tensor<fp16>(middle_buffer_shape, middle_real_shape);
        auto middle_buffer_f32_local =
            make_tensor<fp32>(middle_buffer_shape, middle_real_shape);

        // matmul -> x f16
        tiu::fmm2(middle_buffer_f16_local_1, input_local, weight1_local, false,
                  false, false, false, false, DT_FP16);
        // neg -> -x f16
        tiu::fmul(middle_buffer_f16_local_2, middle_buffer_f16_local_1, -1.0);
        // cast -> f16 2 f32
        tiu::cast(middle_buffer_f32_local, middle_buffer_f16_local_2);
        // exp -> exp^-x f32
        exp_no_overflow(middle_buffer_f32_local, middle_buffer_f32_local,
                        &middle_buffer_shape, &middle_real_shape);
        // add -> 1 + exp^-x f32
        tiu::fadd(middle_buffer_f32_local, middle_buffer_f32_local, 1.0);
        // div -> 1/(1 + exp^-x)  f32
        tiu::fdiv(middle_buffer_f32_local, 1.0, middle_buffer_f32_local, 4);
        // cast -> f32 2 f16
        tiu::cast(middle_buffer_f16_local_2, middle_buffer_f32_local);
        // mul -> x/(1+exp^-x) f16
        tiu::fmul(middle_buffer_f16_local_1, middle_buffer_f16_local_1,
                  middle_buffer_f16_local_2);
        // matmul -> x1 f16
        tiu::fmm2(middle_buffer_f16_local_2, input_local, weight0_local, false,
                  false, false, false, false, DT_FP16);
        // mul -> x1 * x/(1+exp^-x) f16
        tiu::fmul(middle_buffer_f16_local_1, middle_buffer_f16_local_1,
                  middle_buffer_f16_local_2);
        // matmul -> out f32
        tiu::fmm2(out_f32_local, middle_buffer_f16_local_1, weight2_local,
                  false, false, false, false, true, DT_FP32);
      }
      tiu::cast(input_local, out_f32_local);
      int psum_op = ALL_REDUCE_PSUM_WR;
      int op_code = ALL_REDUCE_ADD;
      dma::zero(l2_out_tensor);
      // sync();
      dma::reduce(l2_out_tensor, input_local, ALL_REDUCE_PSUM_WR,
                  ALL_REDUCE_ADD);
    }
    int ele_num = b_slice * input_w;
    slice_per_core = div_up(ele_num, core_num);
    core_offset = slice_per_core * core_idx;
    slice_per_core = min(slice_per_core, ele_num - core_idx * slice_per_core);
    if (slice_per_core > 0) {
      dim4 output_shape = {1, 1, 1, b_slice * input_w};
      auto _output_gtensor = gtensor<fp16>(g_input_shape, GLOBAL, ptr_output);
      auto output_gtensor = _output_gtensor.sub_view(output_shape, g_offset);
      dim4 real_shape = {1, 1, 1, slice_per_core};
      dim4 offset = {0, 0, 0, core_offset};
      // sync();
      sdma::move(output_gtensor.sub_view(real_shape, offset),
                 l2_out_tensor.view(output_shape).sub_view(real_shape, offset));
      // sync();
    }
  }
}

using DTYPE = fp16;

__TEST__ void mlp_multicore_main() {
  // const int B = 1024;
  const int B = 1;
  const int W = 4096;
  const int MIDDLE_W = 1024;
  const int core_num = 8; // 2

  dim4 in_shape = {1, B, 1, W};
  DTYPE *output = rand<DTYPE>(&in_shape);
  DTYPE *input = rand<DTYPE>(&in_shape, -1.0f, 1.0f);
  dim4 weight0_shape = {1, W, 1, MIDDLE_W};
  dim4 weight2_shape = {1, MIDDLE_W, 1, W};
  DTYPE *weight0 = rand<DTYPE>(&weight0_shape, -0.5f, 0.5f);
  DTYPE *weight1 = rand<DTYPE>(&weight0_shape, -0.5f, 0.5f);
  DTYPE *weight2 = rand<DTYPE>(&weight2_shape, -0.5f, 0.5f);
  llma_mlp_mulit_core_kernel_bf16(input, weight0, weight1, weight2, output, B,
                                  W, MIDDLE_W, core_num, B, W, 288);
}

// __AUTOTUNE__ void mlp_multicore_main() {
//   // const int B = 1024;
//   const int core_num = 8;
//   const int B = 1;
//   const int W = 4096;
//   const int MIDDLE_W = 11008;
//   // llma_mlp_mulit_core_kernel_bf16(input, weight0, weight1, weight2,
//   output, B, W, MIDDLE_W); for (int i = 64; i < 258; i += 64)
//     llma_mlp_mulit_core_kernel_bf16(nullptr, nullptr, nullptr, nullptr,
//     nullptr, B, W, MIDDLE_W, core_num, B, W, i);
// }
