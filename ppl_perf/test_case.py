#!/usr/bin/env python3

Y, N = True, False

full_list = {
    # (filename,                                              bm1684x, bm1688, bm1690, sg2260e, sg2262, 2262rv, masr3, bm1684xe)
    #                                                         [base, full]
    "cpp/llm/llama2_7b":
      [
       ("embedding.pl",                                          Y,       Y,      Y,      N,      N,      N,     N,      N),
       ("matmul.pl",                                             Y,       Y,      Y,      N,      N,      N,     N,      N),
      #  ("mlp_multicore.pl",                                      Y,       Y,      Y,      N,      N,      N,     N,      N),
       ("rmsnorm_small_row.pl",                                  Y,       Y,      Y,      N,      N,      N,     N,      N),
       ("rmsnorm.pl",                                            N,       Y,      Y,      N,      N,      N,     N,      N),
       ("w4a16_matmul.pl",                                       N,       Y,      Y,      N,      N,      N,     N,      N),
      ],
}
