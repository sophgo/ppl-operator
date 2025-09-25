#include "ppl.h"

using namespace ppl;
__KERNEL__ void embedding_kernel(fp16 *ptr_output, fp16 *ptr_param,
                                 uint32 *ptr_index, int outer_num,
                                 int inner_num, int select_num, int index_num,
                                 int const_val) {
  set_core_num(8);
  int core_num = get_core_num();
  int core_idx = get_core_index();
  assert(outer_num == 1);
  if (select_num < inner_num) {
    int inner_slice = div_up(inner_num, core_num);
    int real_inner_slice = min(inner_slice, inner_num - core_idx * inner_slice);

    if (real_inner_slice <= 0) {
      return;
    }
    dim4 output_shape = {1, 1, index_num, inner_num};
    dim4 output_slice_shape = {1, 1, index_num, real_inner_slice};
    dim4 param_shape = {1, 1, select_num, inner_num};
    dim4 param_slice_shape = {1, 1, select_num, real_inner_slice};
    dim4 index_shape = {1, 1, index_num, 1};
    dim4 offset = {0, 0, 0, core_idx * inner_slice};

    auto output_g = gtensor<fp16>(output_shape, GLOBAL, ptr_output);
    auto param_g = gtensor<fp16>(param_shape, GLOBAL, ptr_param);
    auto index_g = gtensor<uint32>(index_shape, GLOBAL, ptr_index);

    dma::gather_h(output_g.sub_view(output_slice_shape, offset),
                  param_g.sub_view(param_slice_shape, offset), index_g, 0);

  } else {
    int index_slice = div_up(index_num, core_num);
    int allocated_core = div_up(index_num, index_slice);
    int real_index_slice = min(index_slice, index_num - core_idx * index_slice);

    if (core_idx >= allocated_core) {
      return;
    }
    dim4 output_shape = {1, 1, index_num, inner_num};
    dim4 output_slice_shape = {1, 1, real_index_slice, inner_num};
    dim4 param_shape = {1, 1, select_num, inner_num};
    dim4 index_shape = {1, 1, index_num, 1};
    dim4 index_slice_shape = {1, 1, real_index_slice, 1};
    dim4 offset = {0, 0, core_idx * index_slice, 0};

    auto output_g = gtensor<fp16>(output_shape, GLOBAL, ptr_output);
    auto param_g = gtensor<fp16>(param_shape, GLOBAL, ptr_param);
    auto index_g = gtensor<uint32>(index_shape, GLOBAL, ptr_index);

    dma::gather_h(output_g.sub_view(output_slice_shape, offset), param_g,
                  index_g.sub_view(index_slice_shape, offset), 0);
  }
}

__TEST__ void gather_h_s2s_index_global() {
#if 0
  int outer_num = 1;
  int select_num = 32000;
  int inner_num = 4096;
  int index_num = 6;
#else
  int outer_num = 1;
  int select_num = 15;
  int inner_num = 64;
  int index_num = 6;
#endif
  dim4 output_shape = {1, outer_num, index_num, inner_num};
  auto output = ppl::malloc<fp16>(&output_shape);

  dim4 param_shape = {1, outer_num, select_num, inner_num};
  auto param = ppl::malloc<fp16>(&param_shape);
  ppl::rand(param, &param_shape, -32, 32);

  dim4 index_shape = {1, outer_num, index_num, 1};
  auto index = ppl::malloc<uint32>(&index_shape);
  ppl::rand(index, &index_shape, 0, select_num -1);

  embedding_kernel(output, param, index, outer_num, inner_num, select_num,
                   index_num, 0);
}
