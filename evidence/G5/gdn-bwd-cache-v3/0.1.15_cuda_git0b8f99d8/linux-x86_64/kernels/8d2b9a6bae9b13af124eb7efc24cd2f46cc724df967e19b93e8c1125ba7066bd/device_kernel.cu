#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/scan.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void tilelang_zero_bwd_padding_kernel_kernel(const int* __restrict__ cu_seqlens, float* __restrict__ db, float* __restrict__ dg, bfloat16_t* __restrict__ dk, bfloat16_t* __restrict__ dq, bfloat16_t* __restrict__ dv, int num_tokens, int real_batch_size);
extern "C" __global__ void __launch_bounds__(128, 1) tilelang_zero_bwd_padding_kernel_kernel(const int* __restrict__ cu_seqlens, float* __restrict__ db, float* __restrict__ dg, bfloat16_t* __restrict__ dk, bfloat16_t* __restrict__ dq, bfloat16_t* __restrict__ dv, int num_tokens, int real_batch_size) {
  int seq_end_idx = 0;
  int condval;
  if ((0 <= real_batch_size)) {
    condval = cu_seqlens[((int64_t)real_batch_size)];
  } else {
    condval = 0;
  }
  seq_end_idx = condval;
  for (int i = 0; i < (num_tokens - seq_end_idx); ++i) {
    if (0 <= (seq_end_idx + i)) {
      dq[((((((int64_t)seq_end_idx) * (int64_t)512) + (((int64_t)i) * (int64_t)512)) + (((int64_t)((int)blockIdx.x)) * (int64_t)128)) + ((int64_t)((int)threadIdx.x)))] = bfloat16_t(0x0p+0f/*0.000000e+00*/);
    }
    if (0 <= (seq_end_idx + i)) {
      dk[((((((int64_t)seq_end_idx) * (int64_t)512) + (((int64_t)i) * (int64_t)512)) + (((int64_t)((int)blockIdx.x)) * (int64_t)128)) + ((int64_t)((int)threadIdx.x)))] = bfloat16_t(0x0p+0f/*0.000000e+00*/);
    }
    if (0 <= (seq_end_idx + i)) {
      dv[((((((int64_t)seq_end_idx) * (int64_t)512) + (((int64_t)i) * (int64_t)512)) + (((int64_t)((int)blockIdx.x)) * (int64_t)128)) + ((int64_t)((int)threadIdx.x)))] = bfloat16_t(0x0p+0f/*0.000000e+00*/);
    }
    if (((int)threadIdx.x) == 0) {
      if (0 <= (seq_end_idx + i)) {
        dg[(((((int64_t)seq_end_idx) * (int64_t)4) + (((int64_t)i) * (int64_t)4)) + ((int64_t)((int)blockIdx.x)))] = 0x0p+0f/*0.000000e+00*/;
      }
      if (0 <= (seq_end_idx + i)) {
        db[(((((int64_t)seq_end_idx) * (int64_t)4) + (((int64_t)i) * (int64_t)4)) + ((int64_t)((int)blockIdx.x)))] = 0x0p+0f/*0.000000e+00*/;
      }
    }
  }
}

