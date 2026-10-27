#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/instruction/mma.h>
#include <tl_templates/cuda/intrin.h>
#include <tl_templates/cuda/barrier.h>
#include <tl_templates/cuda/copy_sm90.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/scan.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void tilelang_fused_chunk_gdr_fwd_kernel_kernel(const bfloat16_t* __restrict__ a, __grid_constant__ const CUtensorMap a_desc, const float* __restrict__ b, const int* __restrict__ chunk_offsets, const int* __restrict__ cu_seqlens, const float* __restrict__ g, const float* __restrict__ h0, float* __restrict__ ht, const bfloat16_t* __restrict__ k, __grid_constant__ const CUtensorMap k_desc, bfloat16_t* __restrict__ o, __grid_constant__ const CUtensorMap o_desc, const bfloat16_t* __restrict__ q, __grid_constant__ const CUtensorMap q_desc, const bfloat16_t* __restrict__ v, __grid_constant__ const CUtensorMap v_desc, int batch_size, int num_tokens, int raw_batch_size);
extern "C" __global__ void __launch_bounds__(512, 1) tilelang_fused_chunk_gdr_fwd_kernel_kernel(const bfloat16_t* __restrict__ a, __grid_constant__ const CUtensorMap a_desc, const float* __restrict__ b, const int* __restrict__ chunk_offsets, const int* __restrict__ cu_seqlens, const float* __restrict__ g, const float* __restrict__ h0, float* __restrict__ ht, const bfloat16_t* __restrict__ k, __grid_constant__ const CUtensorMap k_desc, bfloat16_t* __restrict__ o, __grid_constant__ const CUtensorMap o_desc, const bfloat16_t* __restrict__ q, __grid_constant__ const CUtensorMap q_desc, const bfloat16_t* __restrict__ v, __grid_constant__ const CUtensorMap v_desc, int batch_size, int num_tokens, int raw_batch_size) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* h_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* k_shared = ((void*)((char*)buf_dyn_shmem + 16384));
  void* q_shared = ((void*)((char*)buf_dyn_shmem + 32768));
  void* v_shared = ((void*)((char*)buf_dyn_shmem + 49152));
  void* a_shared = ((void*)((char*)buf_dyn_shmem + 57344));
  void* o_shared = ((void*)((char*)buf_dyn_shmem + 61440));
  void* vd_shared = ((void*)((char*)buf_dyn_shmem + 65536));
  void* vn_shared = ((void*)((char*)buf_dyn_shmem + 69632));
  void* p_shared = ((void*)((char*)buf_dyn_shmem + 73728));
  __shared__ __align__(16) uint64_t data_is_ready_mem[2];
  auto data_is_ready = reinterpret_cast<Barrier*>(data_is_ready_mem);
  __shared__ __align__(16) uint64_t data_is_free_mem[2];
  auto data_is_free = reinterpret_cast<Barrier*>(data_is_free_mem);
  __shared__ __align__(16) uint64_t bar_o_mem[1];
  auto bar_o = reinterpret_cast<Barrier*>(bar_o_mem);
  __shared__ __align__(16) uint64_t bar_0_mem[1];
  auto bar_0 = reinterpret_cast<Barrier*>(bar_0_mem);
  __shared__ __align__(16) uint64_t bar_1_mem[1];
  auto bar_1 = reinterpret_cast<Barrier*>(bar_1_mem);
  __shared__ __align__(16) uint64_t _bar_2_mem[1];
  auto _bar_2 = reinterpret_cast<Barrier*>(_bar_2_mem);
  __shared__ __align__(16) uint64_t bar_3_mem[1];
  auto bar_3 = reinterpret_cast<Barrier*>(bar_3_mem);
  __shared__ __align__(16) uint64_t bar_4_mem[1];
  auto bar_4 = reinterpret_cast<Barrier*>(bar_4_mem);
  __shared__ __align__(16) uint64_t bar_5_mem[1];
  auto bar_5 = reinterpret_cast<Barrier*>(bar_5_mem);
  int batch_idx = 0;
  int seq_start_idx = 0;
  int seq_end_idx = 0;
  int chunk_start_idx = 0;
  int raw_batch_idx = 0;
  int raw_seq_end_idx = 0;
  signed char need_store_final_state = (signed char)0;
  int num_iters = 0;
  int num_unmasked_iters = 0;
  float h_fragment[64];
  __shared__ __align__(16) float g_exp_shared[32];
  __shared__ __align__(16) float g_shared[64];
  __shared__ __align__(16) float b_shared[64];
  int seq_split_idx = 0;
  int chunk_split_idx = 0;
  float g_last_local[1];
  __shared__ __align__(16) float g_rev_exp_shared[32];
  float u_fragment[16];
  float v_fragment[16];
  float p_fragment[8];
  float g_fragment[8];
  float a_fragment[8];
  float o_fragment[16];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(q_desc);
    tl::prefetch_tma_descriptor(k_desc);
    tl::prefetch_tma_descriptor(v_desc);
    tl::prefetch_tma_descriptor(a_desc);
    tl::prefetch_tma_descriptor(o_desc);
  }
  if (tl::tl_shuffle_elect<0>()) {
    data_is_ready[0].init(96);
    data_is_ready[1].init(96);
    data_is_free[0].init(384);
    data_is_free[1].init(384);
    bar_o[0].init(128);
    bar_0[0].init(416);
    bar_1[0].init(256);
    _bar_2[0].init(128);
    bar_3[0].init(128);
    bar_4[0].init(128);
    bar_5[0].init(416);
  }
  tl::fence_barrier_init();
  __syncthreads();
  batch_idx = 0;
  seq_start_idx = cu_seqlens[(((int64_t)((int)blockIdx.x)) >> (int64_t)3)];
  seq_end_idx = cu_seqlens[((((int64_t)((int)blockIdx.x)) >> (int64_t)3) + (int64_t)1)];
  chunk_start_idx = chunk_offsets[(((int64_t)((int)blockIdx.x)) >> (int64_t)3)];
  raw_batch_idx = (((int)blockIdx.x) >> 3);
  raw_seq_end_idx = seq_end_idx;
  need_store_final_state = ((signed char)((bool)1 & (raw_seq_end_idx == seq_end_idx)));
  num_iters = (((seq_end_idx + 31) - seq_start_idx) >> 5);
  num_unmasked_iters = ((seq_end_idx - seq_start_idx) >> 5);
  const dim3 blockIdx = tl::rasterization2DRow<10>();
  if (((int)threadIdx.x) < 128) {
    tl::warpgroup_reg_alloc<128>();
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
      *(float2*)(h_fragment + (i * 2)) = *(float2*)(h0 + ((((((((((((int64_t)((int)blockIdx.x)) >> (int64_t)1) * (int64_t)16384) + (((((int64_t)((int)threadIdx.x)) & (int64_t)63) >> (int64_t)5) * (int64_t)8192)) + ((((int64_t)i) >> (int64_t)3) * (int64_t)2048)) + ((((int64_t)i) & (int64_t)1) * (int64_t)1024)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2) * (int64_t)128)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)1) * (int64_t)64)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)6) * (int64_t)32)) + (((((int64_t)i) & (int64_t)7) >> (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2)));
    }
    for (int i_s = 0; i_s < num_iters; ++i_s) {
      data_is_ready[(i_s & 1)].wait(((i_s & 3) >> 1));
      bar_0[0].arrive();
      bar_0[0].wait((i_s & 1));
      tl::__sync_thread_partial(4, 128);
      #pragma unroll
      for (int i_1 = 0; i_1 < 8; ++i_1) {
        tl::ptx_stmatrix_m8n8_x4((&(((bfloat16_t*)h_shared)[((((((((((int)threadIdx.x) & 63) >> 5) * 4096) + ((i_1 >> 1) * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_1 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), __pack_half2(((bfloat16_t)h_fragment[(i_1 * 8)]), ((bfloat16_t)h_fragment[((i_1 * 8) + 1)])), __pack_half2(((bfloat16_t)h_fragment[((i_1 * 8) + 2)]), ((bfloat16_t)h_fragment[((i_1 * 8) + 3)])), __pack_half2(((bfloat16_t)h_fragment[((i_1 * 8) + 4)]), ((bfloat16_t)h_fragment[((i_1 * 8) + 5)])), __pack_half2(((bfloat16_t)h_fragment[((i_1 * 8) + 6)]), ((bfloat16_t)h_fragment[((i_1 * 8) + 7)])));
      }
      bar_1[0].arrive();
      bar_1[0].wait((i_s & 1));
      g_last_local[0] = g_exp_shared[31];
      #pragma unroll
      for (int i_2 = 0; i_2 < 16; ++i_2) {
        float4 __1;
          float4 v_ = *(float4*)(h_fragment + (i_2 * 4));
          float4 v__1 = make_float4(g_last_local[0], g_last_local[0], g_last_local[0], g_last_local[0]);
          *(float2*)(&(__1.x)) = tl::mul2(*(float2*)(&(v_.x)), *(float2*)(&(v__1.x)));
          *(float2*)(&(__1.z)) = tl::mul2(*(float2*)(&(v_.z)), *(float2*)(&(v__1.z)));
        *(float4*)(h_fragment + (i_2 * 4)) = __1;
      }
      bar_5[0].arrive();
      bar_5[0].wait((i_s & 1));
      {
        bfloat16_t A_local[32];
        bfloat16_t B_local[16];
        for (int ki = 0; ki < 2; ++ki) {
          #pragma unroll
          for (int i_3 = 0; i_3 < 4; ++i_3) {
            tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)k_shared)[(((((((((i_s & 1) * 4096) + (((((int)threadIdx.x) & 63) >> 5) * 2048)) + (ki * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_3 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_3 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local[(i_3 * 8)])));
          }
          #pragma unroll
          for (int i_4 = 0; i_4 < 2; ++i_4) {
            tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)vn_shared)[(((((ki * 1024) + ((((int)threadIdx.x) & 15) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_4) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local[(i_4 * 8)])));
          }
          for (int i_5 = 0; i_5 < 4; ++i_5) {
            for (int j = 0; j < 2; ++j) {
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(h_fragment + ((i_5 * 16) + (j * 8))), reinterpret_cast<const unsigned*>(A_local + (i_5 * 8)), reinterpret_cast<const unsigned*>(B_local + (j * 8)));
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(h_fragment + (((i_5 * 16) + (j * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local + (i_5 * 8)), reinterpret_cast<const unsigned*>(B_local + ((j * 8) + 4)));
            }
          }
        }
      }
      data_is_free[(i_s & 1)].arrive();
    }
    if ((bool)need_store_final_state) {
      if (0 <= raw_batch_idx) {
        #pragma unroll
        for (int i_6 = 0; i_6 < 32; ++i_6) {
          if (raw_batch_idx < raw_batch_size) {
            *(float2*)(ht + ((((((((((((int64_t)raw_batch_idx) * (int64_t)65536) + (((((int64_t)((int)blockIdx.x)) & (int64_t)7) >> (int64_t)1) * (int64_t)16384)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)63) >> (int64_t)5) * (int64_t)8192)) + ((((int64_t)i_6) >> (int64_t)3) * (int64_t)2048)) + ((((int64_t)i_6) & (int64_t)1) * (int64_t)1024)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2) * (int64_t)128)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)1) * (int64_t)64)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)6) * (int64_t)32)) + (((((int64_t)i_6) & (int64_t)7) >> (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2))) = *(float2*)(h_fragment + (i_6 * 2));
          }
        }
      }
    }
  } else {
    if (((int)threadIdx.x) < 256) {
      tl::warpgroup_reg_alloc<128>();
      for (int i_s_1 = 0; i_s_1 < num_iters; ++i_s_1) {
        data_is_ready[(i_s_1 & 1)].wait(((i_s_1 & 3) >> 1));
        bar_0[0].arrive();
        bar_0[0].wait((i_s_1 & 1));
        tl::__sync_thread_partial(3, 128);
        if (((int)threadIdx.x) < 160) {
          g_exp_shared[(((int)threadIdx.x) - 128)] = exp2f((g_shared[((((i_s_1 & 1) * 32) + ((int)threadIdx.x)) - 128)] * 0x1.715475a31a4bep+0f/*1.442695e+00*/));
          float condval;
          if (((((i_s_1 * 32) + seq_start_idx) + ((int)threadIdx.x)) < (seq_end_idx + 128))) {
            condval = exp2f(((g_shared[(((i_s_1 & 1) * 32) + 31)] - g_shared[((((i_s_1 & 1) * 32) + ((int)threadIdx.x)) - 128)]) * 0x1.715475a31a4bep+0f/*1.442695e+00*/));
          } else {
            condval = 0x0p+0f/*0.000000e+00*/;
          }
          g_rev_exp_shared[(((int)threadIdx.x) - 128)] = condval;
        }
        bar_1[0].arrive();
        bar_1[0].wait((i_s_1 & 1));
        {
          bfloat16_t A_local_1[8];
          bfloat16_t B_local_1[16];
          #pragma unroll
          for (int i_7 = 0; i_7 < 4; ++i_7) {
            float broadcast_var = 0x0p+0f/*0.000000e+00*/;
            *(float4*)(u_fragment + (i_7 * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
          }
          for (int ki_1 = 0; ki_1 < 8; ++ki_1) {
            tl::ptx_ldmatrix_x4((&(((bfloat16_t*)k_shared)[((((((((i_s_1 & 1) * 4096) + ((ki_1 >> 2) * 2048)) + (((((int)threadIdx.x) & 63) >> 5) * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_1 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_1 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local_1[0])));
            #pragma unroll
            for (int i_8 = 0; i_8 < 2; ++i_8) {
              tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)h_shared)[(((((ki_1 * 1024) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_8) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_1[(i_8 * 8)])));
            }
            for (int j_1 = 0; j_1 < 2; ++j_1) {
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(u_fragment + (j_1 * 8)), reinterpret_cast<const unsigned*>(A_local_1 + 0), reinterpret_cast<const unsigned*>(B_local_1 + (j_1 * 8)));
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(u_fragment + ((j_1 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_1 + 0), reinterpret_cast<const unsigned*>(B_local_1 + ((j_1 * 8) + 4)));
            }
          }
        }
        tl::__sync_thread_partial(3, 128);
        #pragma unroll
        for (int i_9 = 0; i_9 < 8; ++i_9) {
          float2 __2;
            float2 v__2 = *(float2*)(u_fragment + (i_9 * 2));
            float2 v__3 = make_float2((g_exp_shared[(((((((int)threadIdx.x) & 63) >> 5) * 16) + ((i_9 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))] * -0x1p+0f/*-1.000000e+00*/), (g_exp_shared[(((((((int)threadIdx.x) & 63) >> 5) * 16) + ((i_9 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))] * -0x1p+0f/*-1.000000e+00*/));
            *(float2*)(&(__2.x)) = tl::mul2(*(float2*)(&(v__2.x)), *(float2*)(&(v__3.x)));
          *(float2*)(u_fragment + (i_9 * 2)) = __2;
        }
        #pragma unroll
        for (int i_10 = 0; i_10 < 8; ++i_10) {
          bfloat16_t v_shared_local_cast[2];
          *(uint1*)(v_shared_local_cast + 0) = *(uint1*)(((bfloat16_t*)v_shared) + (((((((((i_s_1 & 1) * 2048) + (((((int)threadIdx.x) & 63) >> 5) * 1024)) + ((i_10 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (i_10 >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_10 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
          float2 __3;
            float2 v__4 = *(float2*)(u_fragment + (i_10 * 2));
            float2 __4;
            uint1 v__5 = *(uint1*)(v_shared_local_cast + 0);
            ((float2*)(&__4))[0] = tl::__tl_cvt_bf16x2_to_f32x2((reinterpret_cast<__nv_bfloat162*>(&v__5))[0]);
            *(float2*)(&(__3.x)) = tl::add2(*(float2*)(&(v__4.x)), *(float2*)(&(__4.x)));
          *(float2*)(u_fragment + (i_10 * 2)) = __3;
        }
        tl::__sync_thread_partial(5, 128);
        #pragma unroll
        for (int i_11 = 0; i_11 < 8; ++i_11) {
          bfloat16_t v_shared_local_cast_1[2];
          uint1 __5;
          float2 v__6 = *(float2*)(u_fragment + (i_11 * 2));
          (reinterpret_cast<__nv_bfloat162*>(&__5))[0] = __float22bfloat162_rn(((float2*)(&v__6))[0]);
          *(uint1*)(v_shared_local_cast_1 + 0) = __5;
          *(uint1*)(((bfloat16_t*)v_shared) + (((((((((i_s_1 & 1) * 2048) + (((((int)threadIdx.x) & 63) >> 5) * 1024)) + ((i_11 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (i_11 >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_11 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(v_shared_local_cast_1 + 0);
        }
        bar_3[0].wait((i_s_1 & 1));
        {
          bfloat16_t A_local_2[8];
          bfloat16_t B_local_2[16];
          #pragma unroll
          for (int i_12 = 0; i_12 < 4; ++i_12) {
            float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
            *(float4*)(v_fragment + (i_12 * 4)) = make_float4(broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1);
          }
          tl::__sync_thread_partial(5, 128);
          for (int ki_2 = 0; ki_2 < 2; ++ki_2) {
            tl::ptx_ldmatrix_x4((&(((bfloat16_t*)a_shared)[((((((i_s_1 & 1) * 1024) + (((((int)threadIdx.x) & 63) >> 5) * 512)) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki_2) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), (&(A_local_2[0])));
            #pragma unroll
            for (int i_13 = 0; i_13 < 2; ++i_13) {
              tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)v_shared)[(((((((i_s_1 & 1) * 2048) + (ki_2 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_13) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_2[(i_13 * 8)])));
            }
            for (int j_2 = 0; j_2 < 2; ++j_2) {
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(v_fragment + (j_2 * 8)), reinterpret_cast<const unsigned*>(A_local_2 + 0), reinterpret_cast<const unsigned*>(B_local_2 + (j_2 * 8)));
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(v_fragment + ((j_2 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_2 + 0), reinterpret_cast<const unsigned*>(B_local_2 + ((j_2 * 8) + 4)));
            }
          }
        }
        tl::__sync_thread_partial(5, 128);
        #pragma unroll
        for (int i_14 = 0; i_14 < 2; ++i_14) {
          tl::ptx_stmatrix_m8n8_x4((&(((bfloat16_t*)vd_shared)[(((((((((int)threadIdx.x) & 63) >> 5) * 1024) + ((((int)threadIdx.x) & 15) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_14) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), __pack_half2(((bfloat16_t)v_fragment[(i_14 * 8)]), ((bfloat16_t)v_fragment[((i_14 * 8) + 1)])), __pack_half2(((bfloat16_t)v_fragment[((i_14 * 8) + 2)]), ((bfloat16_t)v_fragment[((i_14 * 8) + 3)])), __pack_half2(((bfloat16_t)v_fragment[((i_14 * 8) + 4)]), ((bfloat16_t)v_fragment[((i_14 * 8) + 5)])), __pack_half2(((bfloat16_t)v_fragment[((i_14 * 8) + 6)]), ((bfloat16_t)v_fragment[((i_14 * 8) + 7)])));
        }
        bar_4[0].arrive();
        #pragma unroll
        for (int i_15 = 0; i_15 < 8; ++i_15) {
          float2 __6;
            float2 v__7 = *(float2*)(v_fragment + (i_15 * 2));
            float2 v__8 = make_float2(g_rev_exp_shared[(((((((int)threadIdx.x) & 63) >> 5) * 16) + ((i_15 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))], g_rev_exp_shared[(((((((int)threadIdx.x) & 63) >> 5) * 16) + ((i_15 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))]);
            *(float2*)(&(__6.x)) = tl::mul2(*(float2*)(&(v__7.x)), *(float2*)(&(v__8.x)));
          *(float2*)(v_fragment + (i_15 * 2)) = __6;
        }
        tl::__sync_thread_partial(5, 128);
        #pragma unroll
        for (int i_16 = 0; i_16 < 2; ++i_16) {
          tl::ptx_stmatrix_m8n8_x4((&(((bfloat16_t*)vn_shared)[(((((((((int)threadIdx.x) & 63) >> 5) * 1024) + ((((int)threadIdx.x) & 15) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_16) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), __pack_half2(((bfloat16_t)v_fragment[(i_16 * 8)]), ((bfloat16_t)v_fragment[((i_16 * 8) + 1)])), __pack_half2(((bfloat16_t)v_fragment[((i_16 * 8) + 2)]), ((bfloat16_t)v_fragment[((i_16 * 8) + 3)])), __pack_half2(((bfloat16_t)v_fragment[((i_16 * 8) + 4)]), ((bfloat16_t)v_fragment[((i_16 * 8) + 5)])), __pack_half2(((bfloat16_t)v_fragment[((i_16 * 8) + 6)]), ((bfloat16_t)v_fragment[((i_16 * 8) + 7)])));
        }
        bar_5[0].arrive();
        bar_5[0].wait((i_s_1 & 1));
        data_is_free[(i_s_1 & 1)].arrive();
      }
    } else {
      if (((int)threadIdx.x) < 384) {
        tl::warpgroup_reg_alloc<128>();
        for (int i_s_2 = 0; i_s_2 < num_iters; ++i_s_2) {
          data_is_ready[(i_s_2 & 1)].wait(((i_s_2 & 3) >> 1));
          bar_0[0].arrive();
          bar_0[0].wait((i_s_2 & 1));
          {
            bfloat16_t A_local_3[8];
            bfloat16_t B_local_3[8];
            #pragma unroll
            for (int i_17 = 0; i_17 < 2; ++i_17) {
              float broadcast_var_2 = 0x0p+0f/*0.000000e+00*/;
              *(float4*)(p_fragment + (i_17 * 4)) = make_float4(broadcast_var_2, broadcast_var_2, broadcast_var_2, broadcast_var_2);
            }
            for (int ki_3 = 0; ki_3 < 8; ++ki_3) {
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)q_shared)[((((((((i_s_2 & 1) * 4096) + ((ki_3 >> 2) * 2048)) + (((((int)threadIdx.x) & 63) >> 5) * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_3 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_3 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local_3[0])));
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)k_shared)[(((((((((i_s_2 & 1) * 4096) + ((ki_3 >> 2) * 2048)) + (((((int)threadIdx.x) & 127) >> 6) * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_3 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_3 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_3[0])));
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(p_fragment + 0), reinterpret_cast<const unsigned*>(A_local_3 + 0), reinterpret_cast<const unsigned*>(B_local_3 + 0));
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(p_fragment + 4), reinterpret_cast<const unsigned*>(A_local_3 + 0), reinterpret_cast<const unsigned*>(B_local_3 + 4));
            }
          }
          #pragma unroll
          for (int i_18 = 0; i_18 < 4; ++i_18) {
            float2 __7;
              float2 v__9 = make_float2(g_shared[(((((i_s_2 & 1) * 32) + (((((int)threadIdx.x) & 63) >> 5) * 16)) + ((i_18 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))], g_shared[(((((i_s_2 & 1) * 32) + (((((int)threadIdx.x) & 63) >> 5) * 16)) + ((i_18 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))]);
              float2 v__10 = *(float2*)(g_shared + ((((((i_s_2 & 1) * 32) + ((((int)threadIdx.x) >> 6) * 16)) + ((i_18 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 64));
              *(float2*)(&(__7.x)) = tl::sub2(*(float2*)(&(v__9.x)), *(float2*)(&(v__10.x)));
            *(float2*)(g_fragment + (i_18 * 2)) = __7;
          }
          #pragma unroll
          for (int i_19 = 0; i_19 < 8; ++i_19) {
            if ((((((((int)threadIdx.x) >> 6) * 16) + ((i_19 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_19 & 1)) <= ((((((((int)threadIdx.x) & 63) >> 5) * 16) + (((i_19 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + 64)) {
              g_fragment[i_19] = exp2f((g_fragment[i_19] * 0x1.715475a31a4bep+0f/*1.442695e+00*/));
            } else {
              g_fragment[i_19] = 0x0p+0f/*0.000000e+00*/;
            }
          }
          tl::__sync_thread_partial(6, 128);
          #pragma unroll
          for (int i_20 = 0; i_20 < 4; ++i_20) {
            bfloat16_t a_shared_local_cast_2[2];
            *(uint1*)(a_shared_local_cast_2 + 0) = *(uint1*)(((bfloat16_t*)a_shared) + ((((((((i_s_2 & 1) * 1024) + (((((int)threadIdx.x) & 63) >> 5) * 512)) + ((i_20 & 1) * 256)) + (((((int)threadIdx.x) & 31) >> 2) * 32)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (i_20 >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
            float2 __8;
            uint1 v__11 = *(uint1*)(a_shared_local_cast_2 + 0);
            ((float2*)(&__8))[0] = tl::__tl_cvt_bf16x2_to_f32x2((reinterpret_cast<__nv_bfloat162*>(&v__11))[0]);
            *(float2*)(a_fragment + (i_20 * 2)) = __8;
          }
          #pragma unroll
          for (int i_21 = 0; i_21 < 2; ++i_21) {
            float4 __9;
              float4 v__12 = *(float4*)(a_fragment + (i_21 * 4));
              float4 v__13 = *(float4*)(g_fragment + (i_21 * 4));
              *(float2*)(&(__9.x)) = tl::mul2(*(float2*)(&(v__12.x)), *(float2*)(&(v__13.x)));
              *(float2*)(&(__9.z)) = tl::mul2(*(float2*)(&(v__12.z)), *(float2*)(&(v__13.z)));
            *(float4*)(a_fragment + (i_21 * 4)) = __9;
          }
          #pragma unroll
          for (int i_22 = 0; i_22 < 4; ++i_22) {
            float2 __10;
              float2 v__14 = *(float2*)(a_fragment + (i_22 * 2));
              float2 v__15 = *(float2*)(b_shared + ((((((i_s_2 & 1) * 32) + ((((int)threadIdx.x) >> 6) * 16)) + ((i_22 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 64));
              *(float2*)(&(__10.x)) = tl::mul2(*(float2*)(&(v__14.x)), *(float2*)(&(v__15.x)));
            *(float2*)(a_fragment + (i_22 * 2)) = __10;
          }
          tl::__sync_thread_partial(6, 128);
          #pragma unroll
          for (int i_23 = 0; i_23 < 4; ++i_23) {
            bfloat16_t a_shared_local_cast_3[2];
            uint1 __11;
            float2 v__16 = *(float2*)(a_fragment + (i_23 * 2));
            (reinterpret_cast<__nv_bfloat162*>(&__11))[0] = __float22bfloat162_rn(((float2*)(&v__16))[0]);
            *(uint1*)(a_shared_local_cast_3 + 0) = __11;
            *(uint1*)(((bfloat16_t*)a_shared) + ((((((((i_s_2 & 1) * 1024) + (((((int)threadIdx.x) & 63) >> 5) * 512)) + ((i_23 & 1) * 256)) + (((((int)threadIdx.x) & 31) >> 2) * 32)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (i_23 >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(a_shared_local_cast_3 + 0);
          }
          bar_1[0].wait((i_s_2 & 1));
          {
            bfloat16_t A_local_4[8];
            bfloat16_t B_local_4[16];
            #pragma unroll
            for (int i_24 = 0; i_24 < 4; ++i_24) {
              float broadcast_var_3 = 0x0p+0f/*0.000000e+00*/;
              *(float4*)(o_fragment + (i_24 * 4)) = make_float4(broadcast_var_3, broadcast_var_3, broadcast_var_3, broadcast_var_3);
            }
            tl::__sync_thread_partial(6, 128);
            for (int ki_4 = 0; ki_4 < 8; ++ki_4) {
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)q_shared)[((((((((i_s_2 & 1) * 4096) + ((ki_4 >> 2) * 2048)) + (((((int)threadIdx.x) & 63) >> 5) * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_4 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_4 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local_4[0])));
              #pragma unroll
              for (int i_25 = 0; i_25 < 2; ++i_25) {
                tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)h_shared)[(((((ki_4 * 1024) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_25) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_4[(i_25 * 8)])));
              }
              for (int j_3 = 0; j_3 < 2; ++j_3) {
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(o_fragment + (j_3 * 8)), reinterpret_cast<const unsigned*>(A_local_4 + 0), reinterpret_cast<const unsigned*>(B_local_4 + (j_3 * 8)));
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(o_fragment + ((j_3 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_4 + 0), reinterpret_cast<const unsigned*>(B_local_4 + ((j_3 * 8) + 4)));
              }
            }
          }
          #pragma unroll
          for (int i_26 = 0; i_26 < 2; ++i_26) {
            float broadcast_var_4 = 0x1.6a09e667f3bcdp-4f/*8.838835e-02*/;
            float4 __12;
              float4 v__17 = *(float4*)(p_fragment + (i_26 * 4));
              float4 __13;
                float4 v__18 = make_float4(broadcast_var_4, broadcast_var_4, broadcast_var_4, broadcast_var_4);
                float4 v__19 = *(float4*)(g_fragment + (i_26 * 4));
                *(float2*)(&(__13.x)) = tl::mul2(*(float2*)(&(v__18.x)), *(float2*)(&(v__19.x)));
                *(float2*)(&(__13.z)) = tl::mul2(*(float2*)(&(v__18.z)), *(float2*)(&(v__19.z)));
              *(float2*)(&(__12.x)) = tl::mul2(*(float2*)(&(v__17.x)), *(float2*)(&(__13.x)));
              *(float2*)(&(__12.z)) = tl::mul2(*(float2*)(&(v__17.z)), *(float2*)(&(__13.z)));
            *(float4*)(p_fragment + (i_26 * 4)) = __12;
          }
          tl::ptx_stmatrix_m8n8_x4((&(((bfloat16_t*)p_shared)[((((((((int)threadIdx.x) & 63) >> 5) * 512) + ((((int)threadIdx.x) & 15) * 32)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), __pack_half2(((bfloat16_t)p_fragment[0]), ((bfloat16_t)p_fragment[1])), __pack_half2(((bfloat16_t)p_fragment[2]), ((bfloat16_t)p_fragment[3])), __pack_half2(((bfloat16_t)p_fragment[4]), ((bfloat16_t)p_fragment[5])), __pack_half2(((bfloat16_t)p_fragment[6]), ((bfloat16_t)p_fragment[7])));
          bar_3[0].arrive();
          #pragma unroll
          for (int i_27 = 0; i_27 < 8; ++i_27) {
            float2 __14;
              float2 v__20 = *(float2*)(o_fragment + (i_27 * 2));
              float2 v__21 = make_float2((0x1.6a09e667f3bcdp-4f/*8.838835e-02*/ * g_exp_shared[(((((((int)threadIdx.x) & 63) >> 5) * 16) + ((i_27 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))]), (0x1.6a09e667f3bcdp-4f/*8.838835e-02*/ * g_exp_shared[(((((((int)threadIdx.x) & 63) >> 5) * 16) + ((i_27 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))]));
              *(float2*)(&(__14.x)) = tl::mul2(*(float2*)(&(v__20.x)), *(float2*)(&(v__21.x)));
            *(float2*)(o_fragment + (i_27 * 2)) = __14;
          }
          bar_4[0].wait((i_s_2 & 1));
          {
            bfloat16_t A_local_5[8];
            bfloat16_t B_local_5[16];
            tl::__sync_thread_partial(6, 128);
            for (int ki_5 = 0; ki_5 < 2; ++ki_5) {
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)p_shared)[((((((((int)threadIdx.x) & 63) >> 5) * 512) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki_5) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), (&(A_local_5[0])));
              #pragma unroll
              for (int i_28 = 0; i_28 < 2; ++i_28) {
                tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)vd_shared)[(((((ki_5 * 1024) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_28) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_5[(i_28 * 8)])));
              }
              for (int j_4 = 0; j_4 < 2; ++j_4) {
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(o_fragment + (j_4 * 8)), reinterpret_cast<const unsigned*>(A_local_5 + 0), reinterpret_cast<const unsigned*>(B_local_5 + (j_4 * 8)));
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(o_fragment + ((j_4 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_5 + 0), reinterpret_cast<const unsigned*>(B_local_5 + ((j_4 * 8) + 4)));
              }
            }
          }
          bar_5[0].arrive();
          bar_5[0].wait((i_s_2 & 1));
          tl::__sync_thread_partial(6, 128);
          #pragma unroll
          for (int i_29 = 0; i_29 < 2; ++i_29) {
            tl::ptx_stmatrix_m8n8_x4((&(((bfloat16_t*)o_shared)[(((((((((int)threadIdx.x) & 63) >> 5) * 1024) + ((((int)threadIdx.x) & 15) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_29) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), __pack_half2(((bfloat16_t)o_fragment[(i_29 * 8)]), ((bfloat16_t)o_fragment[((i_29 * 8) + 1)])), __pack_half2(((bfloat16_t)o_fragment[((i_29 * 8) + 2)]), ((bfloat16_t)o_fragment[((i_29 * 8) + 3)])), __pack_half2(((bfloat16_t)o_fragment[((i_29 * 8) + 4)]), ((bfloat16_t)o_fragment[((i_29 * 8) + 5)])), __pack_half2(((bfloat16_t)o_fragment[((i_29 * 8) + 6)]), ((bfloat16_t)o_fragment[((i_29 * 8) + 7)])));
          }
          data_is_free[(i_s_2 & 1)].arrive();
        }
        bar_o[0].arrive();
      } else {
        tl::warpgroup_reg_dealloc<32>();
        if (((int)threadIdx.x) < 416) {
          tl::__sync_thread_partial(7, 32);
          for (int i_s_3 = 0; i_s_3 < num_iters; ++i_s_3) {
            data_is_free[(i_s_3 & 1)].wait((((i_s_3 >> 1) + 1) & 1));
            int left = ((i_s_3 * 32) + seq_start_idx);
            if ((left + 32) <= seq_end_idx) {
              if (tl::tl_shuffle_elect<32>()) {
                data_is_ready[(i_s_3 & 1)].expect_transaction(8192);
                tl::fence_proxy_async();
                #pragma unroll
                for (int i_30 = 0; i_30 < 2; ++i_30) {
                  tl::tma_load(q_desc, data_is_ready[(i_s_3 & 1)], (&(((bfloat16_t*)q_shared)[(((i_s_3 & 1) * 4096) + (i_30 * 2048))])), (i_30 * 64), left, ((((int)blockIdx.x) & 7) >> 2), batch_idx);
                }
              }
            } else {
              tl::__sync_thread_partial(7, 32);
              #pragma unroll
              for (int i_31 = 0; i_31 < 16; ++i_31) {
                if ((((i_31 * 2) + (((int)threadIdx.x) >> 4)) + left) < (seq_end_idx + 24)) {
                  bfloat16_t broadcast_var_5 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
                  uint4 condval_1;
                  if (((((12 <= ((((((int)threadIdx.x) >> 4) + left) >> 1) + i_31)) && ((((i_31 * 2) + (((int)threadIdx.x) >> 4)) + left) < (num_tokens + 24))) && (0 <= batch_idx)) && (batch_idx < 1))) {
                    condval_1 = *(uint4*)(q + (((((((((int64_t)i_31) * (int64_t)512) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)256)) + (((int64_t)left) * (int64_t)256)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)256)) + (((((int64_t)((int)blockIdx.x)) & (int64_t)7) >> (int64_t)2) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)) - (int64_t)6144));
                  } else {
                    condval_1 = make_uint4(__pack_nv_bfloat162(broadcast_var_5, broadcast_var_5), __pack_nv_bfloat162(broadcast_var_5, broadcast_var_5), __pack_nv_bfloat162(broadcast_var_5, broadcast_var_5), __pack_nv_bfloat162(broadcast_var_5, broadcast_var_5));
                  }
                  *(uint4*)(((bfloat16_t*)q_shared) + (((((((((i_s_3 & 1) * 4096) + (((((int)threadIdx.x) & 15) >> 3) * 2048)) + ((((i_31 * 256) + (((int)threadIdx.x) * 8)) >> 10) * 512)) + ((((i_31 * 2) + (((int)threadIdx.x) >> 4)) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_31 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_31 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) - 1536)) = condval_1;
                } else {
                  bfloat16_t broadcast_var_6 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
                  *(uint4*)(((bfloat16_t*)q_shared) + (((((((((i_s_3 & 1) * 4096) + (((((int)threadIdx.x) & 15) >> 3) * 2048)) + ((((i_31 * 256) + (((int)threadIdx.x) * 8)) >> 10) * 512)) + ((((i_31 * 2) + (((int)threadIdx.x) >> 4)) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_31 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_31 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) - 1536)) = make_uint4(__pack_nv_bfloat162(broadcast_var_6, broadcast_var_6), __pack_nv_bfloat162(broadcast_var_6, broadcast_var_6), __pack_nv_bfloat162(broadcast_var_6, broadcast_var_6), __pack_nv_bfloat162(broadcast_var_6, broadcast_var_6));
                }
              }
            }
            tl::__sync_thread_partial(7, 32);
            if ((left + 32) <= seq_end_idx) {
              if (tl::tl_shuffle_elect<32>()) {
                data_is_ready[(i_s_3 & 1)].expect_transaction(8192);
                tl::fence_proxy_async();
                #pragma unroll
                for (int i_32 = 0; i_32 < 2; ++i_32) {
                  tl::tma_load(k_desc, data_is_ready[(i_s_3 & 1)], (&(((bfloat16_t*)k_shared)[(((i_s_3 & 1) * 4096) + (i_32 * 2048))])), (i_32 * 64), left, ((((int)blockIdx.x) & 7) >> 2), batch_idx);
                }
              }
            } else {
              tl::__sync_thread_partial(7, 32);
              #pragma unroll
              for (int i_33 = 0; i_33 < 16; ++i_33) {
                if ((((i_33 * 2) + (((int)threadIdx.x) >> 4)) + left) < (seq_end_idx + 24)) {
                  bfloat16_t broadcast_var_7 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
                  uint4 condval_2;
                  if (((((12 <= ((((((int)threadIdx.x) >> 4) + left) >> 1) + i_33)) && ((((i_33 * 2) + (((int)threadIdx.x) >> 4)) + left) < (num_tokens + 24))) && (0 <= batch_idx)) && (batch_idx < 1))) {
                    condval_2 = *(uint4*)(k + (((((((((int64_t)i_33) * (int64_t)512) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)256)) + (((int64_t)left) * (int64_t)256)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)256)) + (((((int64_t)((int)blockIdx.x)) & (int64_t)7) >> (int64_t)2) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)) - (int64_t)6144));
                  } else {
                    condval_2 = make_uint4(__pack_nv_bfloat162(broadcast_var_7, broadcast_var_7), __pack_nv_bfloat162(broadcast_var_7, broadcast_var_7), __pack_nv_bfloat162(broadcast_var_7, broadcast_var_7), __pack_nv_bfloat162(broadcast_var_7, broadcast_var_7));
                  }
                  *(uint4*)(((bfloat16_t*)k_shared) + (((((((((i_s_3 & 1) * 4096) + (((((int)threadIdx.x) & 15) >> 3) * 2048)) + ((((i_33 * 256) + (((int)threadIdx.x) * 8)) >> 10) * 512)) + ((((i_33 * 2) + (((int)threadIdx.x) >> 4)) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_33 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_33 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) - 1536)) = condval_2;
                } else {
                  bfloat16_t broadcast_var_8 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
                  *(uint4*)(((bfloat16_t*)k_shared) + (((((((((i_s_3 & 1) * 4096) + (((((int)threadIdx.x) & 15) >> 3) * 2048)) + ((((i_33 * 256) + (((int)threadIdx.x) * 8)) >> 10) * 512)) + ((((i_33 * 2) + (((int)threadIdx.x) >> 4)) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_33 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_33 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) - 1536)) = make_uint4(__pack_nv_bfloat162(broadcast_var_8, broadcast_var_8), __pack_nv_bfloat162(broadcast_var_8, broadcast_var_8), __pack_nv_bfloat162(broadcast_var_8, broadcast_var_8), __pack_nv_bfloat162(broadcast_var_8, broadcast_var_8));
                }
              }
            }
            data_is_ready[(i_s_3 & 1)].arrive();
          }
        } else {
          if (((int)threadIdx.x) < 448) {
            tl::__sync_thread_partial(8, 32);
            for (int i_s_4 = 0; i_s_4 < num_iters; ++i_s_4) {
              data_is_free[(i_s_4 & 1)].wait((((i_s_4 >> 1) + 1) & 1));
              int left_1 = ((i_s_4 * 32) + seq_start_idx);
              if ((left_1 + 32) <= seq_end_idx) {
                if (tl::tl_shuffle_elect<32>()) {
                  data_is_ready[(i_s_4 & 1)].expect_transaction(4096);
                  tl::fence_proxy_async();
                  tl::tma_load(v_desc, data_is_ready[(i_s_4 & 1)], (&(((bfloat16_t*)v_shared)[((i_s_4 & 1) * 2048)])), ((((int)blockIdx.x) & 1) * 64), left_1, ((((int)blockIdx.x) & 7) >> 1), batch_idx);
                }
              } else {
                #pragma unroll
                for (int i_34 = 0; i_34 < 8; ++i_34) {
                  if ((((i_34 * 4) + (((int)threadIdx.x) >> 3)) + left_1) < (seq_end_idx + 52)) {
                    bfloat16_t broadcast_var_9 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
                    uint4 condval_3;
                    if (((((13 <= ((((((int)threadIdx.x) >> 3) + left_1) >> 2) + i_34)) && ((((i_34 * 4) + (((int)threadIdx.x) >> 3)) + left_1) < (num_tokens + 52))) && (0 <= batch_idx)) && (batch_idx < 1))) {
                      condval_3 = *(uint4*)(v + (((((((((int64_t)i_34) * (int64_t)2048) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)3) * (int64_t)512)) + (((int64_t)left_1) * (int64_t)512)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)512)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)7) * (int64_t)64)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)7) * (int64_t)8)) - (int64_t)26624));
                    } else {
                      condval_3 = make_uint4(__pack_nv_bfloat162(broadcast_var_9, broadcast_var_9), __pack_nv_bfloat162(broadcast_var_9, broadcast_var_9), __pack_nv_bfloat162(broadcast_var_9, broadcast_var_9), __pack_nv_bfloat162(broadcast_var_9, broadcast_var_9));
                    }
                    *(uint4*)(((bfloat16_t*)v_shared) + (((((((i_s_4 & 1) * 2048) + ((i_34 >> 1) * 512)) + (((((i_34 * 4) + (((int)threadIdx.x) >> 3)) + 4) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_34 & 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval_3;
                  } else {
                    bfloat16_t broadcast_var_10 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
                    *(uint4*)(((bfloat16_t*)v_shared) + (((((((i_s_4 & 1) * 2048) + ((i_34 >> 1) * 512)) + (((((i_34 * 4) + (((int)threadIdx.x) >> 3)) + 4) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_34 & 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = make_uint4(__pack_nv_bfloat162(broadcast_var_10, broadcast_var_10), __pack_nv_bfloat162(broadcast_var_10, broadcast_var_10), __pack_nv_bfloat162(broadcast_var_10, broadcast_var_10), __pack_nv_bfloat162(broadcast_var_10, broadcast_var_10));
                  }
                }
              }
              if ((left_1 + 32) <= seq_end_idx) {
                float condval_4;
                if (((((416 <= (left_1 + ((int)threadIdx.x))) && ((left_1 + ((int)threadIdx.x)) < (num_tokens + 416))) && (0 <= batch_idx)) && (batch_idx < 1))) {
                  condval_4 = b[(((((((int64_t)left_1) * (int64_t)4) + (((int64_t)((int)threadIdx.x)) * (int64_t)4)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)4)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)7) >> (int64_t)1)) - (int64_t)1664)];
                } else {
                  condval_4 = 0x0p+0f/*0.000000e+00*/;
                }
                b_shared[((((i_s_4 & 1) * 32) + ((int)threadIdx.x)) - 416)] = condval_4;
              } else {
                if ((left_1 + ((int)threadIdx.x)) < (seq_end_idx + 416)) {
                  float condval_5;
                  if (((((416 <= (left_1 + ((int)threadIdx.x))) && ((left_1 + ((int)threadIdx.x)) < (num_tokens + 416))) && (0 <= batch_idx)) && (batch_idx < 1))) {
                    condval_5 = b[(((((((int64_t)left_1) * (int64_t)4) + (((int64_t)((int)threadIdx.x)) * (int64_t)4)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)4)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)7) >> (int64_t)1)) - (int64_t)1664)];
                  } else {
                    condval_5 = 0x0p+0f/*0.000000e+00*/;
                  }
                  b_shared[((((i_s_4 & 1) * 32) + ((int)threadIdx.x)) - 416)] = condval_5;
                } else {
                  b_shared[((((i_s_4 & 1) * 32) + ((int)threadIdx.x)) - 416)] = 0x0p+0f/*0.000000e+00*/;
                }
              }
              data_is_ready[(i_s_4 & 1)].arrive();
            }
          } else {
            if (((int)threadIdx.x) < 480) {
              tl::__sync_thread_partial(9, 32);
              for (int i_s_5 = 0; i_s_5 < num_iters; ++i_s_5) {
                data_is_free[(i_s_5 & 1)].wait((((i_s_5 >> 1) + 1) & 1));
                int left_2 = ((i_s_5 * 32) + seq_start_idx);
                if ((left_2 + 32) <= seq_end_idx) {
                  if (tl::tl_shuffle_elect<32>()) {
                    data_is_ready[(i_s_5 & 1)].expect_transaction(2048);
                    tl::fence_proxy_async();
                    tl::tma_load(a_desc, data_is_ready[(i_s_5 & 1)], (&(((bfloat16_t*)a_shared)[((i_s_5 & 1) * 1024)])), 0, left_2, ((((int)blockIdx.x) & 7) >> 1), batch_idx);
                  }
                } else {
                  #pragma unroll
                  for (int i_35 = 0; i_35 < 4; ++i_35) {
                    if ((((i_35 * 8) + (((int)threadIdx.x) >> 2)) + left_2) < (seq_end_idx + 112)) {
                      bfloat16_t broadcast_var_11 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
                      uint4 condval_6;
                      if (((((14 <= ((((((int)threadIdx.x) >> 2) + left_2) >> 3) + i_35)) && ((((i_35 * 8) + (((int)threadIdx.x) >> 2)) + left_2) < (num_tokens + 112))) && (0 <= batch_idx)) && (batch_idx < 1))) {
                        condval_6 = *(uint4*)(a + (((((((((int64_t)i_35) * (int64_t)1024) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)2) * (int64_t)128)) + (((int64_t)left_2) * (int64_t)128)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)128)) + (((((int64_t)((int)blockIdx.x)) & (int64_t)7) >> (int64_t)1) * (int64_t)32)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)8)) - (int64_t)14336));
                      } else {
                        condval_6 = make_uint4(__pack_nv_bfloat162(broadcast_var_11, broadcast_var_11), __pack_nv_bfloat162(broadcast_var_11, broadcast_var_11), __pack_nv_bfloat162(broadcast_var_11, broadcast_var_11), __pack_nv_bfloat162(broadcast_var_11, broadcast_var_11));
                      }
                      *(uint4*)(((bfloat16_t*)a_shared) + ((((((i_s_5 & 1) * 1024) + (i_35 * 256)) + (((((int)threadIdx.x) & 31) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval_6;
                    } else {
                      bfloat16_t broadcast_var_12 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
                      *(uint4*)(((bfloat16_t*)a_shared) + ((((((i_s_5 & 1) * 1024) + (i_35 * 256)) + (((((int)threadIdx.x) & 31) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = make_uint4(__pack_nv_bfloat162(broadcast_var_12, broadcast_var_12), __pack_nv_bfloat162(broadcast_var_12, broadcast_var_12), __pack_nv_bfloat162(broadcast_var_12, broadcast_var_12), __pack_nv_bfloat162(broadcast_var_12, broadcast_var_12));
                    }
                  }
                }
                if ((left_2 + 32) <= seq_end_idx) {
                  float condval_7;
                  if (((((448 <= (left_2 + ((int)threadIdx.x))) && ((left_2 + ((int)threadIdx.x)) < (num_tokens + 448))) && (0 <= batch_idx)) && (batch_idx < 1))) {
                    condval_7 = g[(((((((int64_t)left_2) * (int64_t)4) + (((int64_t)((int)threadIdx.x)) * (int64_t)4)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)4)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)7) >> (int64_t)1)) - (int64_t)1792)];
                  } else {
                    condval_7 = 0x0p+0f/*0.000000e+00*/;
                  }
                  g_shared[((((i_s_5 & 1) * 32) + ((int)threadIdx.x)) - 448)] = condval_7;
                } else {
                  if ((left_2 + ((int)threadIdx.x)) < (seq_end_idx + 448)) {
                    float condval_8;
                    if (((((448 <= (left_2 + ((int)threadIdx.x))) && ((left_2 + ((int)threadIdx.x)) < (num_tokens + 448))) && (0 <= batch_idx)) && (batch_idx < 1))) {
                      condval_8 = g[(((((((int64_t)left_2) * (int64_t)4) + (((int64_t)((int)threadIdx.x)) * (int64_t)4)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)4)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)7) >> (int64_t)1)) - (int64_t)1792)];
                    } else {
                      condval_8 = 0x0p+0f/*0.000000e+00*/;
                    }
                    g_shared[((((i_s_5 & 1) * 32) + ((int)threadIdx.x)) - 448)] = condval_8;
                  } else {
                    float condval_9;
                    if (((((1 <= seq_end_idx) && (seq_end_idx <= num_tokens)) && (0 <= batch_idx)) && (batch_idx < 1))) {
                      condval_9 = g[((((((int64_t)seq_end_idx) * (int64_t)4) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)4)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)7) >> (int64_t)1)) - (int64_t)4)];
                    } else {
                      condval_9 = 0x0p+0f/*0.000000e+00*/;
                    }
                    g_shared[((((i_s_5 & 1) * 32) + ((int)threadIdx.x)) - 448)] = condval_9;
                  }
                }
                data_is_ready[(i_s_5 & 1)].arrive();
              }
            } else {
              for (int i_s_6 = 0; i_s_6 < num_unmasked_iters; ++i_s_6) {
                int right = ((i_s_6 * 32) + seq_start_idx);
                bar_0[0].arrive();
                bar_0[0].wait((i_s_6 & 1));
                if (0 < i_s_6) {
                  if (tl::tl_shuffle_elect<32>()) {
                    tl::tma_store(o_desc, (&(((bfloat16_t*)o_shared)[0])), ((((int)blockIdx.x) & 1) * 64), (right - 32), ((((int)blockIdx.x) & 7) >> 1), batch_idx);
                    tl::tma_store_arrive();
                    tl::tma_store_wait<0, true>();
                  }
                }
                bar_5[0].arrive();
                bar_1[0].wait((i_s_6 & 1));
              }
              if (num_unmasked_iters < num_iters) {
                seq_split_idx = ((num_unmasked_iters * 32) + seq_start_idx);
              }
              if (num_unmasked_iters < num_iters) {
                chunk_split_idx = (chunk_start_idx + num_unmasked_iters);
              }
              if (num_unmasked_iters < num_iters) {
                int right_1 = seq_split_idx;
                bar_0[0].arrive();
                bar_0[0].wait((num_unmasked_iters & 1));
                if (0 < num_unmasked_iters) {
                  if (tl::tl_shuffle_elect<32>()) {
                    tl::tma_store(o_desc, (&(((bfloat16_t*)o_shared)[0])), ((((int)blockIdx.x) & 1) * 64), (right_1 - 32), ((((int)blockIdx.x) & 7) >> 1), batch_idx);
                    tl::tma_store_arrive();
                    tl::tma_store_wait<0, true>();
                  }
                }
                bar_5[0].arrive();
                bar_1[0].wait((num_unmasked_iters & 1));
              }
              seq_split_idx = (((num_iters * 32) + seq_start_idx) - 32);
              bar_o[0].wait(0);
              if (0 < num_iters) {
                if (0 <= batch_idx) {
                  #pragma unroll
                  for (int i_36 = 0; i_36 < 8; ++i_36) {
                    if ((((i_36 * 4) + (((int)threadIdx.x) >> 3)) + seq_split_idx) < (seq_end_idx + 60)) {
                      if (15 <= ((((((int)threadIdx.x) >> 3) + seq_split_idx) >> 2) + i_36)) {
                        if ((((i_36 * 4) + (((int)threadIdx.x) >> 3)) + seq_split_idx) < (num_tokens + 60)) {
                          if (batch_idx < 1) {
                            *(uint4*)(o + (((((((((int64_t)i_36) * (int64_t)2048) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)3) * (int64_t)512)) + (((int64_t)seq_split_idx) * (int64_t)512)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)512)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)7) * (int64_t)64)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)7) * (int64_t)8)) - (int64_t)30720)) = *(uint4*)(((bfloat16_t*)o_shared) + ((((((i_36 >> 1) * 512) + (((((i_36 * 4) + (((int)threadIdx.x) >> 3)) + 4) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_36 & 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)));
                          }
                        }
                      }
                    }
                  }
                }
              }
              if ((((int)blockIdx.x) >> 3) == (batch_size - 1)) {
                if (0 <= batch_idx) {
                  for (int i_pad = 0; i_pad < (num_tokens - seq_end_idx); ++i_pad) {
                    if (0 <= (seq_end_idx + i_pad)) {
                      if (batch_idx < 1) {
                        bfloat16_t broadcast_var_13 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
                        *(uint1*)(o + ((((((((int64_t)seq_end_idx) * (int64_t)512) + (((int64_t)i_pad) * (int64_t)512)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)512)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)7) * (int64_t)64)) + (((int64_t)((int)threadIdx.x)) * (int64_t)2)) - (int64_t)960)) = make_uint1(__pack_nv_bfloat162(broadcast_var_13, broadcast_var_13));
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

