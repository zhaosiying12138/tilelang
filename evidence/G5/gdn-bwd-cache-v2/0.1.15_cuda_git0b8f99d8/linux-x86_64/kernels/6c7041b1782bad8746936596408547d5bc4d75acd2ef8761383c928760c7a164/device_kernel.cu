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

extern "C" __global__ void tilelang_chunk_local_cumsum_kernel_kernel(float* __restrict__ g_cumsum, const float* __restrict__ g_raw, int data_batch_size, int num_chunks, int seq_end_idx);
extern "C" __global__ void __launch_bounds__(128, 1) tilelang_chunk_local_cumsum_kernel_kernel(float* __restrict__ g_cumsum, const float* __restrict__ g_raw, int data_batch_size, int num_chunks, int seq_end_idx) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* gT_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* scan_smem = ((void*)((char*)buf_dyn_shmem + 640));
  float gT_fragment[1];
  float src_buffer[1];
  int rmod = (((int)blockIdx.x) % data_batch_size);
  int rdiv = (((int)blockIdx.x) / data_batch_size);
  if (((((((0 <= data_batch_size) && (0 <= rmod)) || ((data_batch_size < 0) && (rmod <= 0))) ? rdiv : (rdiv - 1)) * 32) + 32) <= seq_end_idx) {
    int rmod_1 = (((int)blockIdx.x) % data_batch_size);
    float condval;
    if ((((((0 <= data_batch_size) && (0 <= rmod_1)) || ((data_batch_size < 0) && (rmod_1 <= 0))) ? rmod_1 : (rmod_1 + data_batch_size)) < data_batch_size)) {
      int64_t rmod_2 = (((int64_t)((int)blockIdx.x)) % ((int64_t)data_batch_size));
      int64_t rdiv_1 = (((int64_t)((int)blockIdx.x)) / ((int64_t)data_batch_size));
      int64_t rmod_3 = (((int64_t)((int)blockIdx.x)) % ((int64_t)data_batch_size));
      int64_t rmod_4 = (((int64_t)((int)blockIdx.x)) % ((int64_t)data_batch_size));
      int64_t rdiv_2 = (((int64_t)((int)blockIdx.x)) / ((int64_t)data_batch_size));
      int64_t rmod_5 = (((int64_t)((int)blockIdx.x)) % ((int64_t)data_batch_size));
      condval = g_raw[((((((((int64_t)0 <= ((int64_t)data_batch_size)) && ((int64_t)0 <= rmod_4)) || ((((int64_t)data_batch_size) < (int64_t)0) && (rmod_4 <= (int64_t)0))) ? rdiv_2 : (rdiv_2 - (int64_t)1)) * (int64_t)128) + (((((((int64_t)0 <= ((int64_t)data_batch_size)) && ((int64_t)0 <= rmod_5)) || ((((int64_t)data_batch_size) < (int64_t)0) && (rmod_5 <= (int64_t)0))) ? rmod_5 : (rmod_5 + ((int64_t)data_batch_size))) * ((int64_t)seq_end_idx)) * (int64_t)4)) + ((int64_t)((int)threadIdx.x)))];
    } else {
      condval = 0x0p+0f/*0.000000e+00*/;
    }
    gT_fragment[0] = condval;
  } else {
    int rmod_6 = (((int)blockIdx.x) % data_batch_size);
    int rdiv_3 = (((int)blockIdx.x) / data_batch_size);
    if (((((((0 <= data_batch_size) && (0 <= rmod_6)) || ((data_batch_size < 0) && (rmod_6 <= 0))) ? rdiv_3 : (rdiv_3 - 1)) * 32) + (((int)threadIdx.x) >> 2)) < seq_end_idx) {
      int rmod_7 = (((int)blockIdx.x) % data_batch_size);
      float condval_1;
      if ((((((0 <= data_batch_size) && (0 <= rmod_7)) || ((data_batch_size < 0) && (rmod_7 <= 0))) ? rmod_7 : (rmod_7 + data_batch_size)) < data_batch_size)) {
        int64_t rmod_8 = (((int64_t)((int)blockIdx.x)) % ((int64_t)data_batch_size));
        int64_t rdiv_4 = (((int64_t)((int)blockIdx.x)) / ((int64_t)data_batch_size));
        int64_t rmod_9 = (((int64_t)((int)blockIdx.x)) % ((int64_t)data_batch_size));
        int64_t rmod_10 = (((int64_t)((int)blockIdx.x)) % ((int64_t)data_batch_size));
        int64_t rdiv_5 = (((int64_t)((int)blockIdx.x)) / ((int64_t)data_batch_size));
        int64_t rmod_11 = (((int64_t)((int)blockIdx.x)) % ((int64_t)data_batch_size));
        condval_1 = g_raw[((((((((int64_t)0 <= ((int64_t)data_batch_size)) && ((int64_t)0 <= rmod_10)) || ((((int64_t)data_batch_size) < (int64_t)0) && (rmod_10 <= (int64_t)0))) ? rdiv_5 : (rdiv_5 - (int64_t)1)) * (int64_t)128) + (((((((int64_t)0 <= ((int64_t)data_batch_size)) && ((int64_t)0 <= rmod_11)) || ((((int64_t)data_batch_size) < (int64_t)0) && (rmod_11 <= (int64_t)0))) ? rmod_11 : (rmod_11 + ((int64_t)data_batch_size))) * ((int64_t)seq_end_idx)) * (int64_t)4)) + ((int64_t)((int)threadIdx.x)))];
      } else {
        condval_1 = 0x0p+0f/*0.000000e+00*/;
      }
      gT_fragment[0] = condval_1;
    } else {
      gT_fragment[0] = 0x0p+0f/*0.000000e+00*/;
    }
  }
  ((float*)gT_shared)[(((((int)threadIdx.x) >> 2) * 5) + (((int)threadIdx.x) & 3))] = gT_fragment[0];
  __syncthreads();
  src_buffer[0] = ((float*)gT_shared)[(((((int)threadIdx.x) & 31) * 5) + (((int)threadIdx.x) >> 5))];
  ((float*)scan_smem)[((int)threadIdx.x)] = src_buffer[0];
  __syncthreads();
  tl::CumSum2D<128, 1, false, 0>::run((&(((float*)scan_smem)[0])), (&(((float*)scan_smem)[0])), 4, 32, 32, 32);
  __syncthreads();
  src_buffer[0] = ((float*)scan_smem)[((int)threadIdx.x)];
  ((float*)gT_shared)[(((((int)threadIdx.x) & 31) * 5) + (((int)threadIdx.x) >> 5))] = src_buffer[0];
  __syncthreads();
  gT_fragment[0] = ((float*)gT_shared)[(((((int)threadIdx.x) >> 2) * 5) + (((int)threadIdx.x) & 3))];
  int rmod_12 = (((int)blockIdx.x) % data_batch_size);
  int rdiv_6 = (((int)blockIdx.x) / data_batch_size);
  if (((((((0 <= data_batch_size) && (0 <= rmod_12)) || ((data_batch_size < 0) && (rmod_12 <= 0))) ? rdiv_6 : (rdiv_6 - 1)) * 32) + 32) <= seq_end_idx) {
    int64_t rmod_13 = (((int64_t)((int)blockIdx.x)) % ((int64_t)data_batch_size));
    int64_t rdiv_7 = (((int64_t)((int)blockIdx.x)) / ((int64_t)data_batch_size));
    int64_t rmod_14 = (((int64_t)((int)blockIdx.x)) % ((int64_t)data_batch_size));
    int64_t rmod_15 = (((int64_t)((int)blockIdx.x)) % ((int64_t)data_batch_size));
    int64_t rdiv_8 = (((int64_t)((int)blockIdx.x)) / ((int64_t)data_batch_size));
    int64_t rmod_16 = (((int64_t)((int)blockIdx.x)) % ((int64_t)data_batch_size));
    g_cumsum[((((((((int64_t)0 <= ((int64_t)data_batch_size)) && ((int64_t)0 <= rmod_15)) || ((((int64_t)data_batch_size) < (int64_t)0) && (rmod_15 <= (int64_t)0))) ? rdiv_8 : (rdiv_8 - (int64_t)1)) * (int64_t)128) + (((((((int64_t)0 <= ((int64_t)data_batch_size)) && ((int64_t)0 <= rmod_16)) || ((((int64_t)data_batch_size) < (int64_t)0) && (rmod_16 <= (int64_t)0))) ? rmod_16 : (rmod_16 + ((int64_t)data_batch_size))) * ((int64_t)seq_end_idx)) * (int64_t)4)) + ((int64_t)((int)threadIdx.x)))] = gT_fragment[0];
  } else {
    int rmod_17 = (((int)blockIdx.x) % data_batch_size);
    int rdiv_9 = (((int)blockIdx.x) / data_batch_size);
    if (((((((0 <= data_batch_size) && (0 <= rmod_17)) || ((data_batch_size < 0) && (rmod_17 <= 0))) ? rdiv_9 : (rdiv_9 - 1)) * 32) + (((int)threadIdx.x) >> 2)) < seq_end_idx) {
      int64_t rmod_18 = (((int64_t)((int)blockIdx.x)) % ((int64_t)data_batch_size));
      int64_t rdiv_10 = (((int64_t)((int)blockIdx.x)) / ((int64_t)data_batch_size));
      int64_t rmod_19 = (((int64_t)((int)blockIdx.x)) % ((int64_t)data_batch_size));
      int64_t rmod_20 = (((int64_t)((int)blockIdx.x)) % ((int64_t)data_batch_size));
      int64_t rdiv_11 = (((int64_t)((int)blockIdx.x)) / ((int64_t)data_batch_size));
      int64_t rmod_21 = (((int64_t)((int)blockIdx.x)) % ((int64_t)data_batch_size));
      g_cumsum[((((((((int64_t)0 <= ((int64_t)data_batch_size)) && ((int64_t)0 <= rmod_20)) || ((((int64_t)data_batch_size) < (int64_t)0) && (rmod_20 <= (int64_t)0))) ? rdiv_11 : (rdiv_11 - (int64_t)1)) * (int64_t)128) + (((((((int64_t)0 <= ((int64_t)data_batch_size)) && ((int64_t)0 <= rmod_21)) || ((((int64_t)data_batch_size) < (int64_t)0) && (rmod_21 <= (int64_t)0))) ? rmod_21 : (rmod_21 + ((int64_t)data_batch_size))) * ((int64_t)seq_end_idx)) * (int64_t)4)) + ((int64_t)((int)threadIdx.x)))] = gT_fragment[0];
    }
  }
}

