#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/copy_sm90.h>
#include <tl_templates/cuda/copy_sm100.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/scan.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void tilelang_group_reduce_vector_kernel_kernel(const bfloat16_t* __restrict__ buffer, bfloat16_t* __restrict__ result, int batch_size, int num_tokens);
extern "C" __global__ void __launch_bounds__(128, 1) tilelang_group_reduce_vector_kernel_kernel(const bfloat16_t* __restrict__ buffer, bfloat16_t* __restrict__ result, int batch_size, int num_tokens) {
  float result_fragment[16];
  float buffer_fragment[16];
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(result_fragment + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  for (int i_1 = 0; i_1 < 2; ++i_1) {
    {
      bfloat16_t buffer_local_cast[16];
      bfloat16_t broadcast_var_1 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
      ulonglong4 condval;
      if ((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3)) < num_tokens)) {
        condval = tl::load_global_256(&(*(ulonglong4*)(buffer + ((((((((int64_t)((int)blockIdx.x)) * (int64_t)8192) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)3) * (int64_t)512)) + ((((int64_t)((int)blockIdx.z)) * ((int64_t)num_tokens)) * (int64_t)512)) + (((int64_t)((int)blockIdx.y)) * (int64_t)256)) + (((int64_t)i_1) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)7) * (int64_t)16)))));
      } else {
        condval = make_ulonglong4(tl::pack_bfloat16x4(broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1), tl::pack_bfloat16x4(broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1), tl::pack_bfloat16x4(broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1), tl::pack_bfloat16x4(broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1));
      }
      *(ulonglong4*)(buffer_local_cast + 0) = condval;
      for (int vec = 0; vec < 4; ++vec) {
        float broadcast_var_2 = 0x0p+0f/*0.000000e+00*/;
        float4 condval_1;
        if ((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3)) < num_tokens)) {
          float4 __1;
          uint2 v_ = *(uint2*)(buffer_local_cast + (vec * 4));
          ((float2*)(&__1))[0] = tl::__tl_cvt_bf16x2_to_f32x2((reinterpret_cast<__nv_bfloat162*>(&v_))[0]);
          ((float2*)(&__1))[1] = tl::__tl_cvt_bf16x2_to_f32x2((reinterpret_cast<__nv_bfloat162*>(&v_))[1]);
          condval_1 = __1;
        } else {
          condval_1 = make_float4(broadcast_var_2, broadcast_var_2, broadcast_var_2, broadcast_var_2);
        }
        *(float4*)(buffer_fragment + (vec * 4)) = condval_1;
      }
    }
    #pragma unroll
    for (int i_2 = 0; i_2 < 4; ++i_2) {
      float4 __2;
        float4 v__1 = *(float4*)(result_fragment + (i_2 * 4));
        float4 v__2 = *(float4*)(buffer_fragment + (i_2 * 4));
        *(float2*)(&(__2.x)) = tl::add2(*(float2*)(&(v__1.x)), *(float2*)(&(v__2.x)));
        *(float2*)(&(__2.z)) = tl::add2(*(float2*)(&(v__1.z)), *(float2*)(&(v__2.z)));
      *(float4*)(result_fragment + (i_2 * 4)) = __2;
    }
  }
  {
    bfloat16_t result_local_cast_1[16];
    if (((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3)) < num_tokens) {
      for (int vec_1 = 0; vec_1 < 4; ++vec_1) {
        uint2 __3;
        float4 v__3 = *(float4*)(result_fragment + (vec_1 * 4));
        (reinterpret_cast<__nv_bfloat162*>(&__3))[0] = __float22bfloat162_rn(((float2*)(&v__3))[0]);
        (reinterpret_cast<__nv_bfloat162*>(&__3))[1] = __float22bfloat162_rn(((float2*)(&v__3))[1]);
        *(uint2*)(result_local_cast_1 + (vec_1 * 4)) = __3;
      }
      tl::store_global_256(&(*(ulonglong4*)(result + (((((((int64_t)((int)blockIdx.x)) * (int64_t)4096) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)3) * (int64_t)256)) + ((((int64_t)((int)blockIdx.z)) * ((int64_t)num_tokens)) * (int64_t)256)) + (((int64_t)((int)blockIdx.y)) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)7) * (int64_t)16)))), *(ulonglong4*)(result_local_cast_1 + 0));
    }
  }
}

