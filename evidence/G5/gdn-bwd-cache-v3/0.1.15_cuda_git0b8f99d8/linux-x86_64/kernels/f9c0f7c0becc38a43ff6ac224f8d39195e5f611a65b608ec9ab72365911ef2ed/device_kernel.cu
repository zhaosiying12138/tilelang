#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/instruction/mma.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/scan.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void tilelang_kkt_solve_kernel_kernel(bfloat16_t* __restrict__ a, const float* __restrict__ b, const int* __restrict__ chunk_indices, const int* __restrict__ cu_seqlens, const bfloat16_t* __restrict__ k, int num_chunks, int num_tokens, int real_batch_size);
extern "C" __global__ void __launch_bounds__(128, 1) tilelang_kkt_solve_kernel_kernel(bfloat16_t* __restrict__ a, const float* __restrict__ b, const int* __restrict__ chunk_indices, const int* __restrict__ cu_seqlens, const bfloat16_t* __restrict__ k, int num_chunks, int num_tokens, int real_batch_size) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* a16i_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* k_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* a16o_shared = ((void*)((char*)buf_dyn_shmem + 2176));
  void* a32_shared = ((void*)((char*)buf_dyn_shmem + 2176));
  int batch_idx = 0;
  int chunk_idx = 0;
  int seq_start_idx = 0;
  int seq_end_idx = 0;
  __shared__ __align__(16) float b_shared[32];
  float a32_fragment[8];
  float a16i_row[16];
  float a16i_sum[1];
  float a16o_fragment[2];
  batch_idx = chunk_indices[((((int64_t)((int)blockIdx.x)) >> (int64_t)2) * (int64_t)2)];
  chunk_idx = chunk_indices[(((((int64_t)((int)blockIdx.x)) >> (int64_t)2) * (int64_t)2) + (int64_t)1)];
  int condval;
  if (((0 <= batch_idx) && (batch_idx <= real_batch_size))) {
    condval = cu_seqlens[batch_idx];
  } else {
    condval = 0;
  }
  seq_start_idx = condval;
  int condval_1;
  if (((-1 <= batch_idx) && (batch_idx < real_batch_size))) {
    condval_1 = cu_seqlens[(((int64_t)batch_idx) + (int64_t)1)];
  } else {
    condval_1 = 0;
  }
  seq_end_idx = condval_1;
  int chunk_idx_1 = chunk_idx;
  int seq_start_idx_1 = seq_start_idx;
  int seq_end_idx_1 = seq_end_idx;
  if ((((chunk_idx_1 * 32) + seq_start_idx_1) + 32) <= seq_end_idx_1) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      tl::cp_async_gs_conditional<16>((&(((bfloat16_t*)k_shared)[((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(k[((((((((int64_t)chunk_idx_1) * (int64_t)8192) + (((int64_t)i) * (int64_t)2048)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)256)) + (((int64_t)seq_start_idx_1) * (int64_t)256)) + (((((int64_t)((int)blockIdx.x)) & (int64_t)3) >> (int64_t)1) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8))])), ((((((((chunk_idx_1 * 32) + (i * 8)) + (((int)threadIdx.x) >> 4)) + seq_start_idx_1) < num_tokens) && (0 <= ((((chunk_idx_1 * 32) + (i * 8)) + (((int)threadIdx.x) >> 4)) + seq_start_idx_1))) && (((((chunk_idx_1 * 32) + (i * 8)) + (((int)threadIdx.x) >> 4)) + seq_start_idx_1) < num_tokens)) && (0 <= ((((chunk_idx_1 * 32) + (i * 8)) + (((int)threadIdx.x) >> 4)) + seq_start_idx_1))));
    }
    tl::cp_async_commit();
  } else {
    #pragma unroll
    for (int i_1 = 0; i_1 < 4; ++i_1) {
      if (((((chunk_idx_1 * 32) + (i_1 * 8)) + (((int)threadIdx.x) >> 4)) + seq_start_idx_1) < seq_end_idx_1) {
        bfloat16_t broadcast_var = bfloat16_t(0x0p+0f/*0.000000e+00*/);
        uint4 condval_2;
        if (((0 <= ((((chunk_idx_1 * 32) + (i_1 * 8)) + (((int)threadIdx.x) >> 4)) + seq_start_idx_1)) && (((((chunk_idx_1 * 32) + (i_1 * 8)) + (((int)threadIdx.x) >> 4)) + seq_start_idx_1) < num_tokens))) {
          condval_2 = *(uint4*)(k + ((((((((int64_t)chunk_idx_1) * (int64_t)8192) + (((int64_t)i_1) * (int64_t)2048)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)256)) + (((int64_t)seq_start_idx_1) * (int64_t)256)) + (((((int64_t)((int)blockIdx.x)) & (int64_t)3) >> (int64_t)1) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
        } else {
          condval_2 = make_uint4(__pack_nv_bfloat162(broadcast_var, broadcast_var), __pack_nv_bfloat162(broadcast_var, broadcast_var), __pack_nv_bfloat162(broadcast_var, broadcast_var), __pack_nv_bfloat162(broadcast_var, broadcast_var));
        }
        *(uint4*)(((bfloat16_t*)k_shared) + ((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_1 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval_2;
      } else {
        bfloat16_t broadcast_var_1 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
        *(uint4*)(((bfloat16_t*)k_shared) + ((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_1 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = make_uint4(__pack_nv_bfloat162(broadcast_var_1, broadcast_var_1), __pack_nv_bfloat162(broadcast_var_1, broadcast_var_1), __pack_nv_bfloat162(broadcast_var_1, broadcast_var_1), __pack_nv_bfloat162(broadcast_var_1, broadcast_var_1));
      }
    }
  }
  if ((((chunk_idx_1 * 32) + seq_start_idx_1) + 32) <= seq_end_idx_1) {
    if (((int)threadIdx.x) < 32) {
      float condval_3;
      if ((((((chunk_idx_1 * 32) + seq_start_idx_1) + ((int)threadIdx.x)) < num_tokens) && (0 <= (((chunk_idx_1 * 32) + seq_start_idx_1) + ((int)threadIdx.x))))) {
        condval_3 = b[((((((int64_t)chunk_idx_1) * (int64_t)128) + (((int64_t)seq_start_idx_1) * (int64_t)4)) + (((int64_t)((int)threadIdx.x)) * (int64_t)4)) + (((int64_t)((int)blockIdx.x)) & (int64_t)3))];
      } else {
        condval_3 = 0x0p+0f/*0.000000e+00*/;
      }
      b_shared[((int)threadIdx.x)] = condval_3;
    }
  } else {
    if (((int)threadIdx.x) < 32) {
      if ((((chunk_idx_1 * 32) + seq_start_idx_1) + ((int)threadIdx.x)) < seq_end_idx_1) {
        float condval_4;
        if ((((((chunk_idx_1 * 32) + seq_start_idx_1) + ((int)threadIdx.x)) < num_tokens) && (0 <= (((chunk_idx_1 * 32) + seq_start_idx_1) + ((int)threadIdx.x))))) {
          condval_4 = b[((((((int64_t)chunk_idx_1) * (int64_t)128) + (((int64_t)seq_start_idx_1) * (int64_t)4)) + (((int64_t)((int)threadIdx.x)) * (int64_t)4)) + (((int64_t)((int)blockIdx.x)) & (int64_t)3))];
        } else {
          condval_4 = 0x0p+0f/*0.000000e+00*/;
        }
        b_shared[((int)threadIdx.x)] = condval_4;
      } else {
        b_shared[((int)threadIdx.x)] = 0x0p+0f/*0.000000e+00*/;
      }
    }
  }
  if ((((chunk_idx_1 * 32) + seq_start_idx_1) + 32) <= seq_end_idx_1) {
    tl::cp_async_wait<0>();
  }
  __syncthreads();
  {
    bfloat16_t A_local[8];
    bfloat16_t B_local[8];
    #pragma unroll
    for (int i_2 = 0; i_2 < 2; ++i_2) {
      float broadcast_var_2 = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(a32_fragment + (i_2 * 4)) = make_float4(broadcast_var_2, broadcast_var_2, broadcast_var_2, broadcast_var_2);
    }
    for (int ki = 0; ki < 8; ++ki) {
      tl::ptx_ldmatrix_x4((&(((bfloat16_t*)k_shared)[(((((((ki >> 2) * 2048) + (((((int)threadIdx.x) & 63) >> 5) * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local[0])));
      tl::ptx_ldmatrix_x4((&(((bfloat16_t*)k_shared)[((((((((ki >> 2) * 2048) + ((((int)threadIdx.x) >> 6) * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local[0])));
      tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(a32_fragment + 0), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + 0));
      tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(a32_fragment + 4), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + 4));
    }
  }
  #pragma unroll
  for (int i_3 = 0; i_3 < 4; ++i_3) {
    float2 __1;
      float2 v_ = *(float2*)(a32_fragment + (i_3 * 2));
      float2 v__1 = make_float2(b_shared[(((((((int)threadIdx.x) & 63) >> 5) * 16) + ((i_3 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))], b_shared[(((((((int)threadIdx.x) & 63) >> 5) * 16) + ((i_3 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))]);
      *(float2*)(&(__1.x)) = tl::mul2(*(float2*)(&(v_.x)), *(float2*)(&(v__1.x)));
    *(float2*)(a32_fragment + (i_3 * 2)) = __1;
  }
  #pragma unroll
  for (int i_4 = 0; i_4 < 8; ++i_4) {
    if ((((((((int)threadIdx.x) & 63) >> 5) * 16) + (((i_4 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) < (((((((int)threadIdx.x) >> 6) * 16) + ((i_4 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_4 & 1))) {
      a32_fragment[i_4] = 0x0p+0f/*0.000000e+00*/;
    } else {
      if ((((((((int)threadIdx.x) & 63) >> 5) * 16) + (((i_4 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) == (((((((int)threadIdx.x) >> 6) * 16) + ((i_4 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_4 & 1))) {
        a32_fragment[i_4] = 0x1p+0f/*1.000000e+00*/;
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int i_5 = 0; i_5 < 4; ++i_5) {
    if (((((int)threadIdx.x) & 63) >> 5) == ((((int)threadIdx.x) >> 6) + 1)) {
      float broadcast_var_3 = -0x1p+0f/*-1.000000e+00*/;
      float2 __2;
        float2 v__2 = *(float2*)(a32_fragment + (i_5 * 2));
        float2 v__3 = make_float2(broadcast_var_3, broadcast_var_3);
        *(float2*)(&(__2.x)) = tl::mul2(*(float2*)(&(v__2.x)), *(float2*)(&(v__3.x)));
      *(float2*)(((float*)a16o_shared) + (((((i_5 & 1) * 128) + (((((int)threadIdx.x) & 31) >> 2) * 16)) + ((i_5 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = __2;
    } else {
      if (((((int)threadIdx.x) & 63) >> 5) == (((int)threadIdx.x) >> 6)) {
        *(float2*)(((float*)a16i_shared) + (((((((((int)threadIdx.x) & 63) >> 5) * 272) + ((i_5 & 1) * 128)) + (((((int)threadIdx.x) & 31) >> 2) * 16)) + ((i_5 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(float2*)(a32_fragment + (i_5 * 2));
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int k_s = 1; k_s < 16; ++k_s) {
    #pragma unroll
    for (int i_6 = 0; i_6 < 16; ++i_6) {
      if (i_6 < k_s) {
        a16i_row[i_6] = ((float*)a16i_shared)[(((((((int)threadIdx.x) & 31) >> 4) * 272) + (k_s * 16)) + i_6)];
      }
    }
    a16i_sum[0] = 0x0p+0f/*0.000000e+00*/;
    #pragma unroll
    for (int k_r = 0; k_r < k_s; ++k_r) {
      a16i_sum[0] = (a16i_sum[0] - (((float*)a16i_shared)[(((((((int)threadIdx.x) & 31) >> 4) * 272) + (k_r * 16)) + (((int)threadIdx.x) & 15))] * a16i_row[k_r]));
    }
    __syncthreads();
    if ((((int)threadIdx.x) >> 5) == 0) {
      if ((((int)threadIdx.x) & 15) < k_s) {
        ((float*)a16i_shared)[(((((((int)threadIdx.x) & 31) >> 4) * 272) + (k_s * 16)) + (((int)threadIdx.x) & 15))] = a16i_sum[0];
      }
    }
  }
  float broadcast_var_4 = 0x0p+0f/*0.000000e+00*/;
  *(float2*)(a16o_fragment + 0) = make_float2(broadcast_var_4, broadcast_var_4);
  __syncthreads();
  #pragma unroll
  for (int k_r_1 = 0; k_r_1 < 16; ++k_r_1) {
    float2 __3;
      float2 v__4 = make_float2(((float*)a16i_shared)[((((((int)threadIdx.x) >> 3) * 16) + k_r_1) + 272)], ((float*)a16i_shared)[((((((int)threadIdx.x) >> 3) * 16) + k_r_1) + 272)]);
      float2 v__5 = *(float2*)(((float*)a16o_shared) + ((k_r_1 * 16) + ((((int)threadIdx.x) & 7) * 2)));
      float2 v__6 = *(float2*)(a16o_fragment + 0);
      *(float2*)(&(__3.x)) = tl::fma2(*(float2*)(&(v__4.x)), *(float2*)(&(v__5.x)), *(float2*)(&(v__6.x)));
    *(float2*)(a16o_fragment + 0) = __3;
  }
  __syncthreads();
  #pragma unroll
  for (int i_7 = 0; i_7 < 2; ++i_7) {
    ((float*)a16o_shared)[((((((int)threadIdx.x) & 7) * 32) + (i_7 * 16)) + (((int)threadIdx.x) >> 3))] = a16o_fragment[i_7];
  }
  float broadcast_var_5 = 0x0p+0f/*0.000000e+00*/;
  *(float2*)(a16o_fragment + 0) = make_float2(broadcast_var_5, broadcast_var_5);
  __syncthreads();
  #pragma unroll
  for (int k_r_2 = 0; k_r_2 < 16; ++k_r_2) {
    float2 __4;
      float2 v__7 = make_float2(((float*)a16o_shared)[((k_r_2 * 16) + (((int)threadIdx.x) >> 3))], ((float*)a16o_shared)[((k_r_2 * 16) + (((int)threadIdx.x) >> 3))]);
      float2 v__8 = *(float2*)(((float*)a16i_shared) + ((k_r_2 * 16) + ((((int)threadIdx.x) & 7) * 2)));
      float2 v__9 = *(float2*)(a16o_fragment + 0);
      *(float2*)(&(__4.x)) = tl::fma2(*(float2*)(&(v__7.x)), *(float2*)(&(v__8.x)), *(float2*)(&(v__9.x)));
    *(float2*)(a16o_fragment + 0) = __4;
  }
  __syncthreads();
  *(float2*)(((float*)a16o_shared) + (((int)threadIdx.x) * 2)) = *(float2*)(a16o_fragment + 0);
  __syncthreads();
  *(float2*)(((float*)a32_shared) + (((((int)threadIdx.x) >> 3) * 32) + ((((int)threadIdx.x) & 7) * 2))) = *(float2*)(((float*)a16i_shared) + (((int)threadIdx.x) * 2));
  float broadcast_var_6 = 0x0p+0f/*0.000000e+00*/;
  *(float2*)(((float*)a32_shared) + ((((((int)threadIdx.x) >> 3) * 32) + ((((int)threadIdx.x) & 7) * 2)) + 16)) = make_float2(broadcast_var_6, broadcast_var_6);
  *(float2*)(((float*)a32_shared) + ((((((int)threadIdx.x) >> 3) * 32) + ((((int)threadIdx.x) & 7) * 2)) + 512)) = *(float2*)(a16o_fragment + 0);
  *(float2*)(((float*)a32_shared) + ((((((int)threadIdx.x) >> 3) * 32) + ((((int)threadIdx.x) & 7) * 2)) + 528)) = *(float2*)(((float*)a16i_shared) + ((((int)threadIdx.x) * 2) + 272));
  __syncthreads();
  if ((((chunk_idx_1 * 32) + seq_start_idx_1) + 32) <= seq_end_idx_1) {
    #pragma unroll
    for (int i_8 = 0; i_8 < 2; ++i_8) {
      float a32_shared_local_cast_1[4];
      bfloat16_t a_local_cast[4];
      if ((((((chunk_idx_1 * 32) + (i_8 * 16)) + (((int)threadIdx.x) >> 3)) + seq_start_idx_1) < num_tokens) && (0 <= ((((chunk_idx_1 * 32) + (i_8 * 16)) + (((int)threadIdx.x) >> 3)) + seq_start_idx_1))) {
        *(float4*)(a32_shared_local_cast_1 + 0) = *(float4*)(((float*)a32_shared) + ((i_8 * 512) + (((int)threadIdx.x) * 4)));
        uint2 __5;
        float4 v__10 = *(float4*)(a32_shared_local_cast_1 + 0);
        (reinterpret_cast<__nv_bfloat162*>(&__5))[0] = __float22bfloat162_rn(((float2*)(&v__10))[0]);
        (reinterpret_cast<__nv_bfloat162*>(&__5))[1] = __float22bfloat162_rn(((float2*)(&v__10))[1]);
        *(uint2*)(a_local_cast + 0) = __5;
        *(uint2*)(a + ((((((((int64_t)chunk_idx_1) * (int64_t)4096) + (((int64_t)i_8) * (int64_t)2048)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)3) * (int64_t)128)) + (((int64_t)seq_start_idx_1) * (int64_t)128)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)3) * (int64_t)32)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)7) * (int64_t)4))) = *(uint2*)(a_local_cast + 0);
      }
    }
  } else {
    #pragma unroll
    for (int i_9 = 0; i_9 < 2; ++i_9) {
      float a32_shared_local_cast_3[4];
      bfloat16_t a_local_cast_2[4];
      if (((((chunk_idx_1 * 32) + (i_9 * 16)) + (((int)threadIdx.x) >> 3)) + seq_start_idx_1) < seq_end_idx_1) {
        *(float4*)(a32_shared_local_cast_3 + 0) = *(float4*)(((float*)a32_shared) + ((i_9 * 512) + (((int)threadIdx.x) * 4)));
        uint2 __6;
        float4 v__11 = *(float4*)(a32_shared_local_cast_3 + 0);
        (reinterpret_cast<__nv_bfloat162*>(&__6))[0] = __float22bfloat162_rn(((float2*)(&v__11))[0]);
        (reinterpret_cast<__nv_bfloat162*>(&__6))[1] = __float22bfloat162_rn(((float2*)(&v__11))[1]);
        *(uint2*)(a_local_cast_2 + 0) = __6;
        if (0 <= ((((chunk_idx_1 * 32) + (i_9 * 16)) + (((int)threadIdx.x) >> 3)) + seq_start_idx_1)) {
          if (((((chunk_idx_1 * 32) + (i_9 * 16)) + (((int)threadIdx.x) >> 3)) + seq_start_idx_1) < num_tokens) {
            *(uint2*)(a + ((((((((int64_t)chunk_idx_1) * (int64_t)4096) + (((int64_t)i_9) * (int64_t)2048)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)3) * (int64_t)128)) + (((int64_t)seq_start_idx_1) * (int64_t)128)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)3) * (int64_t)32)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)7) * (int64_t)4))) = *(uint2*)(a_local_cast_2 + 0);
          }
        }
      }
    }
  }
}

