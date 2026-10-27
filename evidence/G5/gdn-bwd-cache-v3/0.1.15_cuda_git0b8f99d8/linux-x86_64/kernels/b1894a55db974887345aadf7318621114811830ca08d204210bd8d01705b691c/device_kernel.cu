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

extern "C" __global__ void tilelang_chunk_local_cumsum_kernel_kernel(const int* __restrict__ chunk_indices, const int* __restrict__ cu_seqlens, float* __restrict__ g_cumsum, const float* __restrict__ g_raw, int num_chunks, int num_tokens, int real_batch_size);
extern "C" __global__ void __launch_bounds__(128, 1) tilelang_chunk_local_cumsum_kernel_kernel(const int* __restrict__ chunk_indices, const int* __restrict__ cu_seqlens, float* __restrict__ g_cumsum, const float* __restrict__ g_raw, int num_chunks, int num_tokens, int real_batch_size) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* gT_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* scan_smem = ((void*)((char*)buf_dyn_shmem + 640));
  float gT_fragment[1];
  float src_buffer[1];
  int batch_idx = chunk_indices[(((int64_t)((int)blockIdx.x)) * (int64_t)2)];
  int chunk_idx = chunk_indices[((((int64_t)((int)blockIdx.x)) * (int64_t)2) + (int64_t)1)];
  int condval;
  if (((batch_idx <= real_batch_size) && (0 <= batch_idx))) {
    condval = cu_seqlens[((int64_t)batch_idx)];
  } else {
    condval = 0;
  }
  int seq_start_idx = condval;
  int condval_1;
  if (((batch_idx < real_batch_size) && (-1 <= batch_idx))) {
    condval_1 = cu_seqlens[(((int64_t)batch_idx) + (int64_t)1)];
  } else {
    condval_1 = 0;
  }
  int seq_end_idx = condval_1;
  if ((((chunk_idx * 32) + seq_start_idx) + 32) <= seq_end_idx) {
    float condval_2;
    if ((((((chunk_idx * 32) + (((int)threadIdx.x) >> 2)) + seq_start_idx) < num_tokens) && (0 <= (((chunk_idx * 32) + (((int)threadIdx.x) >> 2)) + seq_start_idx)))) {
      condval_2 = g_raw[(((((int64_t)chunk_idx) * (int64_t)128) + (((int64_t)seq_start_idx) * (int64_t)4)) + ((int64_t)((int)threadIdx.x)))];
    } else {
      condval_2 = 0x0p+0f/*0.000000e+00*/;
    }
    gT_fragment[0] = condval_2;
  } else {
    if ((((chunk_idx * 32) + (((int)threadIdx.x) >> 2)) + seq_start_idx) < seq_end_idx) {
      float condval_3;
      if (((0 <= (((chunk_idx * 32) + (((int)threadIdx.x) >> 2)) + seq_start_idx)) && ((((chunk_idx * 32) + (((int)threadIdx.x) >> 2)) + seq_start_idx) < num_tokens))) {
        condval_3 = g_raw[(((((int64_t)chunk_idx) * (int64_t)128) + (((int64_t)seq_start_idx) * (int64_t)4)) + ((int64_t)((int)threadIdx.x)))];
      } else {
        condval_3 = 0x0p+0f/*0.000000e+00*/;
      }
      gT_fragment[0] = condval_3;
    } else {
      gT_fragment[0] = 0x0p+0f/*0.000000e+00*/;
    }
  }
  ((float*)gT_shared)[(((((int)threadIdx.x) >> 2) * 5) + (((int)threadIdx.x) & 3))] = gT_fragment[0];
  __syncthreads();
  src_buffer[0] = ((float*)gT_shared)[(((((int)threadIdx.x) & 31) * 5) + (((int)threadIdx.x) >> 5))];
  ((float*)scan_smem)[((int)threadIdx.x)] = src_buffer[0];
  __syncthreads();
  tl::CumSum2D<128, 1, true, 0>::run((&(((float*)scan_smem)[0])), (&(((float*)scan_smem)[0])), 4, 32, 32, 32);
  __syncthreads();
  src_buffer[0] = ((float*)scan_smem)[((int)threadIdx.x)];
  ((float*)gT_shared)[(((((int)threadIdx.x) & 31) * 5) + (((int)threadIdx.x) >> 5))] = src_buffer[0];
  __syncthreads();
  gT_fragment[0] = ((float*)gT_shared)[(((((int)threadIdx.x) >> 2) * 5) + (((int)threadIdx.x) & 3))];
  if ((((chunk_idx * 32) + seq_start_idx) + 32) <= seq_end_idx) {
    if (((((chunk_idx * 32) + (((int)threadIdx.x) >> 2)) + seq_start_idx) < num_tokens) && (0 <= (((chunk_idx * 32) + (((int)threadIdx.x) >> 2)) + seq_start_idx))) {
      g_cumsum[(((((int64_t)chunk_idx) * (int64_t)128) + (((int64_t)seq_start_idx) * (int64_t)4)) + ((int64_t)((int)threadIdx.x)))] = gT_fragment[0];
    }
  } else {
    if ((((chunk_idx * 32) + (((int)threadIdx.x) >> 2)) + seq_start_idx) < seq_end_idx) {
      if (0 <= (((chunk_idx * 32) + (((int)threadIdx.x) >> 2)) + seq_start_idx)) {
        if ((((chunk_idx * 32) + (((int)threadIdx.x) >> 2)) + seq_start_idx) < num_tokens) {
          g_cumsum[(((((int64_t)chunk_idx) * (int64_t)128) + (((int64_t)seq_start_idx) * (int64_t)4)) + ((int64_t)((int)threadIdx.x)))] = gT_fragment[0];
        }
      }
    }
  }
  if ((batch_idx == (real_batch_size - 1)) && (seq_end_idx <= (((chunk_idx * 32) + seq_start_idx) + 32))) {
    if (((((int)threadIdx.x) >> 2) + seq_end_idx) < num_tokens) {
      if (0 <= ((((int)threadIdx.x) >> 2) + seq_end_idx)) {
        g_cumsum[((((int64_t)seq_end_idx) * (int64_t)4) + ((int64_t)((int)threadIdx.x)))] = 0x0p+0f/*0.000000e+00*/;
      }
    }
  }
}

