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

extern "C" __global__ void tilelang_fused_chunk_gdr_bwd_kernel_kernel(const bfloat16_t* __restrict__ a, __grid_constant__ const CUtensorMap a_desc, const float* __restrict__ b, float* __restrict__ db, float* __restrict__ dg, bfloat16_t* __restrict__ dk, __grid_constant__ const CUtensorMap dk_desc, const bfloat16_t* __restrict__ do_1, __grid_constant__ const CUtensorMap do_desc, bfloat16_t* __restrict__ dq, __grid_constant__ const CUtensorMap dq_desc, bfloat16_t* __restrict__ dv, __grid_constant__ const CUtensorMap dv_desc, const float* __restrict__ g, const bfloat16_t* __restrict__ h, __grid_constant__ const CUtensorMap h_desc, const bfloat16_t* __restrict__ k, __grid_constant__ const CUtensorMap k_desc, const bfloat16_t* __restrict__ q, __grid_constant__ const CUtensorMap q_desc, const bfloat16_t* __restrict__ v, int batch_size, int num_chunks, int num_tokens);
extern "C" __global__ void __launch_bounds__(512, 1) tilelang_fused_chunk_gdr_bwd_kernel_kernel(const bfloat16_t* __restrict__ a, __grid_constant__ const CUtensorMap a_desc, const float* __restrict__ b, float* __restrict__ db, float* __restrict__ dg, bfloat16_t* __restrict__ dk, __grid_constant__ const CUtensorMap dk_desc, const bfloat16_t* __restrict__ do_1, __grid_constant__ const CUtensorMap do_desc, bfloat16_t* __restrict__ dq, __grid_constant__ const CUtensorMap dq_desc, bfloat16_t* __restrict__ dv, __grid_constant__ const CUtensorMap dv_desc, const float* __restrict__ g, const bfloat16_t* __restrict__ h, __grid_constant__ const CUtensorMap h_desc, const bfloat16_t* __restrict__ k, __grid_constant__ const CUtensorMap k_desc, const bfloat16_t* __restrict__ q, __grid_constant__ const CUtensorMap q_desc, const bfloat16_t* __restrict__ v, int batch_size, int num_chunks, int num_tokens) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* h_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* tmp_shared_2_1 = ((void*)((char*)buf_dyn_shmem + 16384));
  void* k_shared = ((void*)((char*)buf_dyn_shmem + 24576));
  void* a_shared = ((void*)((char*)buf_dyn_shmem + 32768));
  void* do_shared = ((void*)((char*)buf_dyn_shmem + 34816));
  void* tmp_shared_4_1 = ((void*)((char*)buf_dyn_shmem + 43008));
  void* dqkv_shared = ((void*)((char*)buf_dyn_shmem + 59392));
  void* tmp_shared_2_2 = ((void*)((char*)buf_dyn_shmem + 67584));
  void* tmp_shared_2_3 = ((void*)((char*)buf_dyn_shmem + 75776));
  void* tmp_shared_1_1 = ((void*)((char*)buf_dyn_shmem + 83968));
  void* tmp_shared_1_2 = ((void*)((char*)buf_dyn_shmem + 86016));
  void* tmp_shared_1_3 = ((void*)((char*)buf_dyn_shmem + 88064));
  void* workspace = ((void*)((char*)buf_dyn_shmem + 90112));
  void* workspace_1 = ((void*)((char*)buf_dyn_shmem + 90624));
  void* workspace_2 = ((void*)((char*)buf_dyn_shmem + 91136));
  void* workspace_3 = ((void*)((char*)buf_dyn_shmem + 91648));
  void* workspace_4 = ((void*)((char*)buf_dyn_shmem + 92160));
  void* workspace_5 = ((void*)((char*)buf_dyn_shmem + 92672));
  void* workspace_6 = ((void*)((char*)buf_dyn_shmem + 93184));
  __shared__ __align__(16) uint64_t bar_00_mem[1];
  auto bar_00 = reinterpret_cast<Barrier*>(bar_00_mem);
  __shared__ __align__(16) uint64_t bar_01_mem[1];
  auto bar_01 = reinterpret_cast<Barrier*>(bar_01_mem);
  __shared__ __align__(16) uint64_t bar_02_mem[1];
  auto bar_02 = reinterpret_cast<Barrier*>(bar_02_mem);
  __shared__ __align__(16) uint64_t bar_03_mem[1];
  auto bar_03 = reinterpret_cast<Barrier*>(bar_03_mem);
  __shared__ __align__(16) uint64_t bar_04_mem[1];
  auto bar_04 = reinterpret_cast<Barrier*>(bar_04_mem);
  __shared__ __align__(16) uint64_t bar_05_mem[1];
  auto bar_05 = reinterpret_cast<Barrier*>(bar_05_mem);
  __shared__ __align__(16) uint64_t bar_06_mem[1];
  auto bar_06 = reinterpret_cast<Barrier*>(bar_06_mem);
  __shared__ __align__(16) uint64_t bar_07_mem[1];
  auto bar_07 = reinterpret_cast<Barrier*>(bar_07_mem);
  __shared__ __align__(16) uint64_t bar_08_1_mem[1];
  auto bar_08_1 = reinterpret_cast<Barrier*>(bar_08_1_mem);
  __shared__ __align__(16) uint64_t bar_08_2_mem[1];
  auto bar_08_2 = reinterpret_cast<Barrier*>(bar_08_2_mem);
  __shared__ __align__(16) uint64_t bar_08_3_mem[1];
  auto bar_08_3 = reinterpret_cast<Barrier*>(bar_08_3_mem);
  __shared__ __align__(16) uint64_t bar_k_left_ready_mem[1];
  auto bar_k_left_ready = reinterpret_cast<Barrier*>(bar_k_left_ready_mem);
  __shared__ __align__(16) uint64_t bar_dhtmp_ready_mem[1];
  auto bar_dhtmp_ready = reinterpret_cast<Barrier*>(bar_dhtmp_ready_mem);
  __shared__ __align__(16) uint64_t bar_dh_right_ready_mem[1];
  auto bar_dh_right_ready = reinterpret_cast<Barrier*>(bar_dh_right_ready_mem);
  __shared__ __align__(16) uint64_t bar_s4_dot_a_mem[1];
  auto bar_s4_dot_a = reinterpret_cast<Barrier*>(bar_s4_dot_a_mem);
  __shared__ __align__(16) uint64_t bar_dg_k_init_mem[1];
  auto bar_dg_k_init = reinterpret_cast<Barrier*>(bar_dg_k_init_mem);
  __shared__ __align__(16) uint64_t bar_dg_k_rows_mem[1];
  auto bar_dg_k_rows = reinterpret_cast<Barrier*>(bar_dg_k_rows_mem);
  __shared__ __align__(16) uint64_t bar_dg_a_first_mem[1];
  auto bar_dg_a_first = reinterpret_cast<Barrier*>(bar_dg_a_first_mem);
  __shared__ __align__(16) uint64_t bar_09_mem[1];
  auto bar_09 = reinterpret_cast<Barrier*>(bar_09_mem);
  __shared__ __align__(16) uint64_t bar_10_mem[1];
  auto bar_10 = reinterpret_cast<Barrier*>(bar_10_mem);
  __shared__ __align__(16) uint64_t bar_11_mem[1];
  auto bar_11 = reinterpret_cast<Barrier*>(bar_11_mem);
  __shared__ __align__(16) uint64_t bar_12_mem[1];
  auto bar_12 = reinterpret_cast<Barrier*>(bar_12_mem);
  __shared__ __align__(16) uint64_t bar_13_mem[1];
  auto bar_13 = reinterpret_cast<Barrier*>(bar_13_mem);
  __shared__ __align__(16) uint64_t bar_14_mem[1];
  auto bar_14 = reinterpret_cast<Barrier*>(bar_14_mem);
  __shared__ __align__(16) uint64_t bar_15_mem[1];
  auto bar_15 = reinterpret_cast<Barrier*>(bar_15_mem);
  int batch_idx = 0;
  int seq_start_idx = 0;
  int seq_end_idx = 0;
  int chunk_start_idx = 0;
  int num_iters = 0;
  __shared__ __align__(16) float g_shared[32];
  __shared__ __align__(16) float b_shared[32];
  float dh_fragment_0[64];
  float dh_fragment_1[64];
  __shared__ __align__(16) float g_exp_shared[32];
  __shared__ __align__(16) float g_rev_exp_shared[32];
  __shared__ __align__(16) float dg_shared[32];
  float dk_frag_l[16];
  float dk_frag_r[16];
  __shared__ __align__(16) float db_shared[32];
  float g_last_local_3[1];
  float reduce_fragment[2];
  float2 dg_last_local_3_pack[1];
  float dg_last_local_3[1];
  bfloat16_t dh_left_fragment[64];
  float dv_fragment[32];
  float odot_fragment_1[32];
  float dg_fragment_1[2];
  float2 dg_last_local_1_pack[1];
  float dg_last_local_1[1];
  float p_fragment[8];
  float mask_fragment[8];
  float a_fragment[8];
  float odot_fragment_2[32];
  float u_fragment[32];
  float da_fragment[8];
  float dp_fragment[8];
  float dq_frag_r[16];
  float dq_frag_l[16];
  float2 dg_fragment_2_pack[4];
  float dg_fragment_2[4];
  ushort lo_fragment[8];
  ushort hi_fragment[8];
  uint uint32_fragment[8];
  float dg_fragment_2b[2];
  float db_fragment[2];
  float db_fragment_clear[2];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(k_desc);
    tl::prefetch_tma_descriptor(q_desc);
    tl::prefetch_tma_descriptor(dk_desc);
    tl::prefetch_tma_descriptor(dv_desc);
    tl::prefetch_tma_descriptor(dq_desc);
    tl::prefetch_tma_descriptor(h_desc);
    tl::prefetch_tma_descriptor(a_desc);
    tl::prefetch_tma_descriptor(do_desc);
  }
  if (tl::tl_shuffle_elect<0>()) {
    bar_00[0].init(448);
    bar_01[0].init(384);
    bar_02[0].init(288);
    bar_03[0].init(256);
    bar_04[0].init(416);
    bar_05[0].init(288);
    bar_06[0].init(256);
    bar_07[0].init(256);
    bar_08_1[0].init(256);
    bar_08_2[0].init(384);
    bar_08_3[0].init(128);
    bar_k_left_ready[0].init(128);
    bar_dhtmp_ready[0].init(128);
    bar_dh_right_ready[0].init(128);
    bar_s4_dot_a[0].init(128);
    bar_dg_k_init[0].init(128);
    bar_dg_k_rows[0].init(128);
    bar_dg_a_first[0].init(128);
    bar_09[0].init(256);
    bar_10[0].init(288);
    bar_11[0].init(256);
    bar_12[0].init(128);
    bar_13[0].init(256);
    bar_14[0].init(256);
    bar_15[0].init(256);
  }
  tl::fence_barrier_init();
  __syncthreads();
  batch_idx = (((int)blockIdx.x) >> 2);
  seq_start_idx = 0;
  seq_end_idx = num_tokens;
  chunk_start_idx = 0;
  num_iters = (((seq_end_idx + 31) - seq_start_idx) >> 5);
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    bfloat16_t broadcast_var = bfloat16_t(0x0p+0f/*0.000000e+00*/);
    uint4 condval;
    if (((((1 <= (chunk_start_idx + num_iters)) && ((chunk_start_idx + num_iters) <= num_chunks)) && (0 <= batch_idx)) && (batch_idx < batch_size))) {
      condval = *(uint4*)(h + (((((((((int64_t)chunk_start_idx) * (int64_t)65536) + (((int64_t)num_iters) * (int64_t)65536)) + ((((int64_t)batch_idx) * ((int64_t)num_chunks)) * (int64_t)65536)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)3) * (int64_t)16384)) + (((int64_t)i) * (int64_t)4096)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)) - (int64_t)65536));
    } else {
      condval = make_uint4(__pack_nv_bfloat162(broadcast_var, broadcast_var), __pack_nv_bfloat162(broadcast_var, broadcast_var), __pack_nv_bfloat162(broadcast_var, broadcast_var), __pack_nv_bfloat162(broadcast_var, broadcast_var));
    }
    *(uint4*)(((bfloat16_t*)h_shared) + ((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i * 2048)) + ((((int)threadIdx.x) >> 4) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval;
  }
  if ((((num_iters * 32) + (((int)threadIdx.x) >> 4)) + seq_start_idx) < (seq_end_idx + 32)) {
    bfloat16_t broadcast_var_1 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
    uint4 condval_1;
    if (((((1 <= ((((((int)threadIdx.x) >> 4) + seq_start_idx) >> 5) + num_iters)) && ((((num_iters * 32) + (((int)threadIdx.x) >> 4)) + seq_start_idx) < (num_tokens + 32))) && (0 <= batch_idx)) && (batch_idx < batch_size))) {
      condval_1 = *(uint4*)(q + (((((((((int64_t)num_iters) * (int64_t)8192) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)256)) + (((int64_t)seq_start_idx) * (int64_t)256)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)256)) + (((((int64_t)((int)blockIdx.x)) & (int64_t)3) >> (int64_t)1) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)) - (int64_t)8192));
    } else {
      condval_1 = make_uint4(__pack_nv_bfloat162(broadcast_var_1, broadcast_var_1), __pack_nv_bfloat162(broadcast_var_1, broadcast_var_1), __pack_nv_bfloat162(broadcast_var_1, broadcast_var_1), __pack_nv_bfloat162(broadcast_var_1, broadcast_var_1));
    }
    *(uint4*)(((bfloat16_t*)tmp_shared_2_1) + (((((((((int)threadIdx.x) & 15) >> 3) * 2048) + ((((int)threadIdx.x) >> 4) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval_1;
  } else {
    bfloat16_t broadcast_var_2 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
    *(uint4*)(((bfloat16_t*)tmp_shared_2_1) + (((((((((int)threadIdx.x) & 15) >> 3) * 2048) + ((((int)threadIdx.x) >> 4) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = make_uint4(__pack_nv_bfloat162(broadcast_var_2, broadcast_var_2), __pack_nv_bfloat162(broadcast_var_2, broadcast_var_2), __pack_nv_bfloat162(broadcast_var_2, broadcast_var_2), __pack_nv_bfloat162(broadcast_var_2, broadcast_var_2));
  }
  if ((((num_iters * 32) + (((int)threadIdx.x) >> 4)) + seq_start_idx) < (seq_end_idx + 32)) {
    bfloat16_t broadcast_var_3 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
    uint4 condval_2;
    if (((((1 <= ((((((int)threadIdx.x) >> 4) + seq_start_idx) >> 5) + num_iters)) && ((((num_iters * 32) + (((int)threadIdx.x) >> 4)) + seq_start_idx) < (num_tokens + 32))) && (0 <= batch_idx)) && (batch_idx < batch_size))) {
      condval_2 = *(uint4*)(k + (((((((((int64_t)num_iters) * (int64_t)8192) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)256)) + (((int64_t)seq_start_idx) * (int64_t)256)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)256)) + (((((int64_t)((int)blockIdx.x)) & (int64_t)3) >> (int64_t)1) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)) - (int64_t)8192));
    } else {
      condval_2 = make_uint4(__pack_nv_bfloat162(broadcast_var_3, broadcast_var_3), __pack_nv_bfloat162(broadcast_var_3, broadcast_var_3), __pack_nv_bfloat162(broadcast_var_3, broadcast_var_3), __pack_nv_bfloat162(broadcast_var_3, broadcast_var_3));
    }
    *(uint4*)(((bfloat16_t*)k_shared) + (((((((((int)threadIdx.x) & 15) >> 3) * 2048) + ((((int)threadIdx.x) >> 4) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval_2;
  } else {
    bfloat16_t broadcast_var_4 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
    *(uint4*)(((bfloat16_t*)k_shared) + (((((((((int)threadIdx.x) & 15) >> 3) * 2048) + ((((int)threadIdx.x) >> 4) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = make_uint4(__pack_nv_bfloat162(broadcast_var_4, broadcast_var_4), __pack_nv_bfloat162(broadcast_var_4, broadcast_var_4), __pack_nv_bfloat162(broadcast_var_4, broadcast_var_4), __pack_nv_bfloat162(broadcast_var_4, broadcast_var_4));
  }
  if ((((num_iters * 32) + (((int)threadIdx.x) >> 4)) + seq_start_idx) < (seq_end_idx + 32)) {
    bfloat16_t broadcast_var_5 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
    uint1 condval_3;
    if (((((1 <= ((((((int)threadIdx.x) >> 4) + seq_start_idx) >> 5) + num_iters)) && ((((num_iters * 32) + (((int)threadIdx.x) >> 4)) + seq_start_idx) < (num_tokens + 32))) && (0 <= batch_idx)) && (batch_idx < batch_size))) {
      condval_3 = *(uint1*)(a + (((((((((int64_t)num_iters) * (int64_t)4096) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)128)) + (((int64_t)seq_start_idx) * (int64_t)128)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)128)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)3) * (int64_t)32)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)2)) - (int64_t)4096));
    } else {
      condval_3 = make_uint1(__pack_nv_bfloat162(broadcast_var_5, broadcast_var_5));
    }
    *(uint1*)(((bfloat16_t*)a_shared) + (((((((int)threadIdx.x) >> 4) * 32) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = condval_3;
  } else {
    bfloat16_t broadcast_var_6 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
    *(uint1*)(((bfloat16_t*)a_shared) + (((((((int)threadIdx.x) >> 4) * 32) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = make_uint1(__pack_nv_bfloat162(broadcast_var_6, broadcast_var_6));
  }
  if ((((num_iters * 32) + (((int)threadIdx.x) >> 4)) + seq_start_idx) < (seq_end_idx + 32)) {
    bfloat16_t broadcast_var_7 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
    uint4 condval_4;
    if (((((1 <= ((((((int)threadIdx.x) >> 4) + seq_start_idx) >> 5) + num_iters)) && ((((num_iters * 32) + (((int)threadIdx.x) >> 4)) + seq_start_idx) < (num_tokens + 32))) && (0 <= batch_idx)) && (batch_idx < batch_size))) {
      condval_4 = *(uint4*)(do_1 + (((((((((int64_t)num_iters) * (int64_t)16384) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)512)) + (((int64_t)seq_start_idx) * (int64_t)512)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)512)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)3) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)) - (int64_t)16384));
    } else {
      condval_4 = make_uint4(__pack_nv_bfloat162(broadcast_var_7, broadcast_var_7), __pack_nv_bfloat162(broadcast_var_7, broadcast_var_7), __pack_nv_bfloat162(broadcast_var_7, broadcast_var_7), __pack_nv_bfloat162(broadcast_var_7, broadcast_var_7));
    }
    *(uint4*)(((bfloat16_t*)do_shared) + (((((((((int)threadIdx.x) & 15) >> 3) * 2048) + ((((int)threadIdx.x) >> 4) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval_4;
  } else {
    bfloat16_t broadcast_var_8 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
    *(uint4*)(((bfloat16_t*)do_shared) + (((((((((int)threadIdx.x) & 15) >> 3) * 2048) + ((((int)threadIdx.x) >> 4) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = make_uint4(__pack_nv_bfloat162(broadcast_var_8, broadcast_var_8), __pack_nv_bfloat162(broadcast_var_8, broadcast_var_8), __pack_nv_bfloat162(broadcast_var_8, broadcast_var_8), __pack_nv_bfloat162(broadcast_var_8, broadcast_var_8));
  }
  if (((int)threadIdx.x) < 32) {
    if ((((num_iters * 32) + seq_start_idx) + ((int)threadIdx.x)) < (seq_end_idx + 32)) {
      float condval_5;
      if (((((1 <= (((seq_start_idx + ((int)threadIdx.x)) >> 5) + num_iters)) && ((((num_iters * 32) + seq_start_idx) + ((int)threadIdx.x)) < (num_tokens + 32))) && (0 <= batch_idx)) && (batch_idx < batch_size))) {
        condval_5 = g[((((((((int64_t)num_iters) * (int64_t)128) + (((int64_t)seq_start_idx) * (int64_t)4)) + (((int64_t)((int)threadIdx.x)) * (int64_t)4)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)4)) + (((int64_t)((int)blockIdx.x)) & (int64_t)3)) - (int64_t)128)];
      } else {
        condval_5 = 0x0p+0f/*0.000000e+00*/;
      }
      g_shared[((int)threadIdx.x)] = condval_5;
    } else {
      float condval_6;
      if (((((1 <= seq_end_idx) && (seq_end_idx <= num_tokens)) && (0 <= batch_idx)) && (batch_idx < batch_size))) {
        condval_6 = g[((((((int64_t)seq_end_idx) * (int64_t)4) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)4)) + (((int64_t)((int)blockIdx.x)) & (int64_t)3)) - (int64_t)4)];
      } else {
        condval_6 = 0x0p+0f/*0.000000e+00*/;
      }
      g_shared[((int)threadIdx.x)] = condval_6;
    }
    if ((((num_iters * 32) + seq_start_idx) + ((int)threadIdx.x)) < (seq_end_idx + 32)) {
      float condval_7;
      if (((((1 <= (((seq_start_idx + ((int)threadIdx.x)) >> 5) + num_iters)) && ((((num_iters * 32) + seq_start_idx) + ((int)threadIdx.x)) < (num_tokens + 32))) && (0 <= batch_idx)) && (batch_idx < batch_size))) {
        condval_7 = b[((((((((int64_t)num_iters) * (int64_t)128) + (((int64_t)seq_start_idx) * (int64_t)4)) + (((int64_t)((int)threadIdx.x)) * (int64_t)4)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)4)) + (((int64_t)((int)blockIdx.x)) & (int64_t)3)) - (int64_t)128)];
      } else {
        condval_7 = 0x0p+0f/*0.000000e+00*/;
      }
      b_shared[((int)threadIdx.x)] = condval_7;
    } else {
      b_shared[((int)threadIdx.x)] = 0x0p+0f/*0.000000e+00*/;
    }
  }
  if (((int)threadIdx.x) < 128) {
    tl::warpgroup_reg_alloc<160>();
    #pragma unroll
    for (int i_1 = 0; i_1 < 16; ++i_1) {
      float broadcast_var_9 = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(dh_fragment_0 + (i_1 * 4)) = make_float4(broadcast_var_9, broadcast_var_9, broadcast_var_9, broadcast_var_9);
    }
    #pragma unroll
    for (int i_2 = 0; i_2 < 16; ++i_2) {
      float broadcast_var_10 = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(dh_fragment_1 + (i_2 * 4)) = make_float4(broadcast_var_10, broadcast_var_10, broadcast_var_10, broadcast_var_10);
    }
    bar_dhtmp_ready[0].arrive();
    for (int i_s = 0; i_s < num_iters; ++i_s) {
      int cur_idx = (((chunk_start_idx + num_iters) - i_s) - 1);
      bar_00[0].arrive();
      bar_00[0].wait((i_s & 1));
      tl::__sync_thread_partial(3, 128);
      if (((int)threadIdx.x) < 32) {
        g_exp_shared[((int)threadIdx.x)] = exp2f((g_shared[((int)threadIdx.x)] * 0x1.715475a31a4bep+0f/*1.442695e+00*/));
        g_rev_exp_shared[((int)threadIdx.x)] = exp2f(((g_shared[31] - g_shared[((int)threadIdx.x)]) * 0x1.715475a31a4bep+0f/*1.442695e+00*/));
      }
      bar_01[0].arrive();
      bar_01[0].wait((i_s & 1));
      if (0 < i_s) {
        bar_k_left_ready[0].wait(((i_s + 1) & 1));
        tl::__sync_thread_partial(6, 128);
        #pragma unroll
        for (int i_3 = 0; i_3 < 8; ++i_3) {
          tl::ptx_stmatrix_m8n8_x4((&(((bfloat16_t*)tmp_shared_4_1)[(((((((((int)threadIdx.x) >> 5) * 2048) + ((i_3 >> 2) * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_3 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_3 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), __pack_half2(((bfloat16_t)dh_fragment_1[(i_3 * 8)]), ((bfloat16_t)dh_fragment_1[((i_3 * 8) + 1)])), __pack_half2(((bfloat16_t)dh_fragment_1[((i_3 * 8) + 2)]), ((bfloat16_t)dh_fragment_1[((i_3 * 8) + 3)])), __pack_half2(((bfloat16_t)dh_fragment_1[((i_3 * 8) + 4)]), ((bfloat16_t)dh_fragment_1[((i_3 * 8) + 5)])), __pack_half2(((bfloat16_t)dh_fragment_1[((i_3 * 8) + 6)]), ((bfloat16_t)dh_fragment_1[((i_3 * 8) + 7)])));
        }
        tl::fence_proxy_async();
        bar_dh_right_ready[0].arrive();
      }
      tl::__sync_thread_partial(3, 128);
      g_last_local_3[0] = g_exp_shared[31];
      #pragma unroll
      for (int i_4 = 0; i_4 < 16; ++i_4) {
        float4 __1;
          float4 v_ = *(float4*)(dh_fragment_0 + (i_4 * 4));
          float4 v__1 = make_float4(g_last_local_3[0], g_last_local_3[0], g_last_local_3[0], g_last_local_3[0]);
          *(float2*)(&(__1.x)) = tl::mul2(*(float2*)(&(v_.x)), *(float2*)(&(v__1.x)));
          *(float2*)(&(__1.z)) = tl::mul2(*(float2*)(&(v_.z)), *(float2*)(&(v__1.z)));
        *(float4*)(dh_fragment_0 + (i_4 * 4)) = __1;
        float4 __2;
          float4 v__2 = *(float4*)(dh_fragment_1 + (i_4 * 4));
          float4 v__3 = make_float4(g_last_local_3[0], g_last_local_3[0], g_last_local_3[0], g_last_local_3[0]);
          *(float2*)(&(__2.x)) = tl::mul2(*(float2*)(&(v__2.x)), *(float2*)(&(v__3.x)));
          *(float2*)(&(__2.z)) = tl::mul2(*(float2*)(&(v__2.z)), *(float2*)(&(v__3.z)));
        *(float4*)(dh_fragment_1 + (i_4 * 4)) = __2;
      }
      bar_04[0].arrive();
      bar_04[0].wait((i_s & 1));
      float broadcast_var_11 = 0x0p+0f/*0.000000e+00*/;
      *(float2*)(reduce_fragment + 0) = make_float2(broadcast_var_11, broadcast_var_11);
      #pragma unroll
      for (int i_5 = 0; i_5 < 32; ++i_5) {
        bfloat16_t h_shared_local_cast[2];
        *(uint1*)(h_shared_local_cast + 0) = *(uint1*)(((bfloat16_t*)h_shared) + (((((((((((int)threadIdx.x) >> 5) * 2048) + ((i_5 >> 4) * 1024)) + ((i_5 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((i_5 & 15) >> 3)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_5 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_5 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
        float2 __3;
          float2 v__4 = *(float2*)(dh_fragment_1 + (i_5 * 2));
          float2 __4;
          uint1 v__5 = *(uint1*)(h_shared_local_cast + 0);
          ((float2*)(&__4))[0] = tl::__tl_cvt_bf16x2_to_f32x2((reinterpret_cast<__nv_bfloat162*>(&v__5))[0]);
          float2 v__6 = *(float2*)(reduce_fragment + 0);
          *(float2*)(&(__3.x)) = tl::fma2(*(float2*)(&(v__4.x)), *(float2*)(&(__4.x)), *(float2*)(&(v__6.x)));
        *(float2*)(reduce_fragment + 0) = __3;
      }
      bar_08_2[0].arrive();
      bar_08_2[0].wait((i_s & 1));
      tl::__sync_thread_partial(6, 128);
      #pragma unroll
      for (int i_6 = 0; i_6 < 8; ++i_6) {
        bfloat16_t broadcast_var_12 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
        uint4 condval_8;
        if (((((0 <= cur_idx) && (cur_idx < num_chunks)) && (0 <= batch_idx)) && (batch_idx < batch_size))) {
          condval_8 = *(uint4*)(h + (((((((int64_t)cur_idx) * (int64_t)65536) + ((((int64_t)batch_idx) * ((int64_t)num_chunks)) * (int64_t)65536)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)3) * (int64_t)16384)) + (((int64_t)i_6) * (int64_t)1024)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)));
        } else {
          condval_8 = make_uint4(__pack_nv_bfloat162(broadcast_var_12, broadcast_var_12), __pack_nv_bfloat162(broadcast_var_12, broadcast_var_12), __pack_nv_bfloat162(broadcast_var_12, broadcast_var_12), __pack_nv_bfloat162(broadcast_var_12, broadcast_var_12));
        }
        *(uint4*)(((bfloat16_t*)h_shared) + ((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_6 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval_8;
      }
      bar_08_3[0].arrive();
      bar_08_3[0].wait((i_s & 1));
      tl::__sync_thread_partial(6, 128);
      #pragma unroll
      for (int i_7 = 0; i_7 < 32; ++i_7) {
        bfloat16_t h_shared_local_cast_1[2];
        *(uint1*)(h_shared_local_cast_1 + 0) = *(uint1*)(((bfloat16_t*)h_shared) + ((((((((((int)threadIdx.x) >> 5) * 2048) + ((i_7 >> 3) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((i_7 & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_7 & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_7 & 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
        float2 __5;
          float2 v__7 = *(float2*)(dh_fragment_0 + ((((i_7 >> 4) * 32) + ((i_7 & 7) * 4)) + (((i_7 & 15) >> 3) * 2)));
          float2 __6;
          uint1 v__8 = *(uint1*)(h_shared_local_cast_1 + 0);
          ((float2*)(&__6))[0] = tl::__tl_cvt_bf16x2_to_f32x2((reinterpret_cast<__nv_bfloat162*>(&v__8))[0]);
          float2 v__9 = *(float2*)(reduce_fragment + 0);
          *(float2*)(&(__5.x)) = tl::fma2(*(float2*)(&(v__7.x)), *(float2*)(&(__6.x)), *(float2*)(&(v__9.x)));
        *(float2*)(reduce_fragment + 0) = __5;
      }
      bar_09[0].arrive();
      bar_09[0].wait((i_s & 1));
      bar_10[0].wait((i_s & 1));
      float broadcast_var_13 = 0x0p+0f/*0.000000e+00*/;
      dg_last_local_3_pack[0] = make_float2(broadcast_var_13, broadcast_var_13);
      dg_last_local_3_pack[0] = tl::add2(dg_last_local_3_pack[0], *(float2*)(reduce_fragment + 0));
      dg_last_local_3[0] = ((dg_last_local_3_pack[0]).x + (dg_last_local_3_pack[0]).y);
      tl::__sync_thread_partial(6, 128);
      dg_last_local_3[0] = tl::AllReduce<tl::SumOp, 128, 1, 0, tl::NamedBarrier<128>>::run(dg_last_local_3[0], (&(((float*)workspace_2)[0])));
      if (((int)threadIdx.x) == 0) {
        dg_shared[31] = (dg_shared[31] + dg_last_local_3[0]);
      }
      bar_11[0].arrive();
      bar_11[0].wait((i_s & 1));
      {
        bfloat16_t A_local[16];
        bfloat16_t B_local[32];
        for (int ki = 0; ki < 2; ++ki) {
          #pragma unroll
          for (int i_8 = 0; i_8 < 2; ++i_8) {
            tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)tmp_shared_2_2)[((((((ki * 1024) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_8) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local[(i_8 * 8)])));
          }
          #pragma unroll
          for (int i_9 = 0; i_9 < 4; ++i_9) {
            tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)tmp_shared_2_3)[(((((((((int)threadIdx.x) >> 6) * 2048) + (ki * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_9 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_9 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local[(i_9 * 8)])));
          }
          for (int i_10 = 0; i_10 < 2; ++i_10) {
            for (int j = 0; j < 4; ++j) {
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dh_fragment_0 + ((i_10 * 32) + (j * 8))), reinterpret_cast<const unsigned*>(A_local + (i_10 * 8)), reinterpret_cast<const unsigned*>(B_local + (j * 8)));
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dh_fragment_0 + (((i_10 * 32) + (j * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local + (i_10 * 8)), reinterpret_cast<const unsigned*>(B_local + ((j * 8) + 4)));
            }
          }
        }
      }
      {
        bfloat16_t A_local_1[16];
        bfloat16_t B_local_1[32];
        for (int ki_1 = 0; ki_1 < 2; ++ki_1) {
          #pragma unroll
          for (int i_11 = 0; i_11 < 2; ++i_11) {
            tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)tmp_shared_2_2)[(((((((ki_1 * 1024) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_11) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 2048)])), (&(A_local_1[(i_11 * 8)])));
          }
          #pragma unroll
          for (int i_12 = 0; i_12 < 4; ++i_12) {
            tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)tmp_shared_2_3)[(((((((((int)threadIdx.x) >> 6) * 2048) + (ki_1 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_12 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_12 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_1[(i_12 * 8)])));
          }
          for (int i_13 = 0; i_13 < 2; ++i_13) {
            for (int j_1 = 0; j_1 < 4; ++j_1) {
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dh_fragment_1 + ((i_13 * 32) + (j_1 * 8))), reinterpret_cast<const unsigned*>(A_local_1 + (i_13 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + (j_1 * 8)));
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dh_fragment_1 + (((i_13 * 32) + (j_1 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_1 + (i_13 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + ((j_1 * 8) + 4)));
            }
          }
        }
      }
      bar_12[0].arrive();
      bar_12[0].wait((i_s & 1));
      bar_13[0].wait((i_s & 1));
      tl::__sync_thread_partial(6, 128);
      #pragma unroll
      for (int i_14 = 0; i_14 < 4; ++i_14) {
        bfloat16_t do_shared_local_cast_3[8];
        float g_exp_shared_local_cast_4[8];
        bfloat16_t tmp_shared_2_3_local_cast_2[8];
        *(uint4*)(do_shared_local_cast_3 + 0) = *(uint4*)(((bfloat16_t*)do_shared) + ((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_14 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)));
        for (int vec_copy = 0; vec_copy < 2; ++vec_copy) {
          *(float4*)(g_exp_shared_local_cast_4 + (vec_copy * 4)) = make_float4(g_exp_shared[((i_14 * 8) + (((int)threadIdx.x) >> 4))], g_exp_shared[((i_14 * 8) + (((int)threadIdx.x) >> 4))], g_exp_shared[((i_14 * 8) + (((int)threadIdx.x) >> 4))], g_exp_shared[((i_14 * 8) + (((int)threadIdx.x) >> 4))]);
        }
        for (int vec = 0; vec < 2; ++vec) {
          float broadcast_var_14 = 0x1.6a09e667f3bcdp-4f/*8.838835e-02*/;
          uint2 __7;
          float4 __8;
            float4 __9;
              float4 v__10 = make_float4(broadcast_var_14, broadcast_var_14, broadcast_var_14, broadcast_var_14);
              float4 __10;
              uint2 v__11 = *(uint2*)(do_shared_local_cast_3 + (vec * 4));
              ((float2*)(&__10))[0] = tl::__tl_cvt_bf16x2_to_f32x2((reinterpret_cast<__nv_bfloat162*>(&v__11))[0]);
              ((float2*)(&__10))[1] = tl::__tl_cvt_bf16x2_to_f32x2((reinterpret_cast<__nv_bfloat162*>(&v__11))[1]);
              *(float2*)(&(__9.x)) = tl::mul2(*(float2*)(&(v__10.x)), *(float2*)(&(__10.x)));
              *(float2*)(&(__9.z)) = tl::mul2(*(float2*)(&(v__10.z)), *(float2*)(&(__10.z)));
            float4 v__12 = *(float4*)(g_exp_shared_local_cast_4 + (vec * 4));
            *(float2*)(&(__8.x)) = tl::mul2(*(float2*)(&(__9.x)), *(float2*)(&(v__12.x)));
            *(float2*)(&(__8.z)) = tl::mul2(*(float2*)(&(__9.z)), *(float2*)(&(v__12.z)));
          (reinterpret_cast<__nv_bfloat162*>(&__7))[0] = __float22bfloat162_rn(((float2*)(&__8))[0]);
          (reinterpret_cast<__nv_bfloat162*>(&__7))[1] = __float22bfloat162_rn(((float2*)(&__8))[1]);
          *(uint2*)(tmp_shared_2_3_local_cast_2 + (vec * 4)) = __7;
        }
        *(uint4*)(((bfloat16_t*)tmp_shared_2_3) + ((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_14 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(tmp_shared_2_3_local_cast_2 + 0);
      }
      bar_14[0].arrive();
      bar_14[0].wait((i_s & 1));
      {
        bfloat16_t A_local_2[16];
        bfloat16_t B_local_2[32];
        tl::__sync_thread_partial(6, 128);
        for (int ki_2 = 0; ki_2 < 2; ++ki_2) {
          #pragma unroll
          for (int i_15 = 0; i_15 < 2; ++i_15) {
            tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)tmp_shared_2_1)[((((((ki_2 * 1024) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_15) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local_2[(i_15 * 8)])));
          }
          #pragma unroll
          for (int i_16 = 0; i_16 < 4; ++i_16) {
            tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)tmp_shared_2_3)[(((((((((int)threadIdx.x) >> 6) * 2048) + (ki_2 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_16 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_16 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_2[(i_16 * 8)])));
          }
          for (int i_17 = 0; i_17 < 2; ++i_17) {
            for (int j_2 = 0; j_2 < 4; ++j_2) {
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dh_fragment_0 + ((i_17 * 32) + (j_2 * 8))), reinterpret_cast<const unsigned*>(A_local_2 + (i_17 * 8)), reinterpret_cast<const unsigned*>(B_local_2 + (j_2 * 8)));
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dh_fragment_0 + (((i_17 * 32) + (j_2 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_2 + (i_17 * 8)), reinterpret_cast<const unsigned*>(B_local_2 + ((j_2 * 8) + 4)));
            }
          }
        }
      }
      {
        bfloat16_t A_local_3[16];
        bfloat16_t B_local_3[32];
        for (int ki_3 = 0; ki_3 < 2; ++ki_3) {
          #pragma unroll
          for (int i_18 = 0; i_18 < 2; ++i_18) {
            tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)tmp_shared_2_1)[(((((((ki_3 * 1024) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_18) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 2048)])), (&(A_local_3[(i_18 * 8)])));
          }
          #pragma unroll
          for (int i_19 = 0; i_19 < 4; ++i_19) {
            tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)tmp_shared_2_3)[(((((((((int)threadIdx.x) >> 6) * 2048) + (ki_3 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_19 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_19 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_3[(i_19 * 8)])));
          }
          for (int i_20 = 0; i_20 < 2; ++i_20) {
            for (int j_3 = 0; j_3 < 4; ++j_3) {
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dh_fragment_1 + ((i_20 * 32) + (j_3 * 8))), reinterpret_cast<const unsigned*>(A_local_3 + (i_20 * 8)), reinterpret_cast<const unsigned*>(B_local_3 + (j_3 * 8)));
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dh_fragment_1 + (((i_20 * 32) + (j_3 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_3 + (i_20 * 8)), reinterpret_cast<const unsigned*>(B_local_3 + ((j_3 * 8) + 4)));
            }
          }
        }
      }
      bar_15[0].arrive();
      bar_15[0].wait((i_s & 1));
      tl::__sync_thread_partial(6, 128);
      #pragma unroll
      for (int i_21 = 0; i_21 < 8; ++i_21) {
        tl::ptx_stmatrix_m8n8_x4((&(((bfloat16_t*)tmp_shared_4_1)[(((((((((int)threadIdx.x) >> 5) * 2048) + ((i_21 >> 2) * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_21 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_21 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), __pack_half2(((bfloat16_t)dh_fragment_0[(i_21 * 8)]), ((bfloat16_t)dh_fragment_0[((i_21 * 8) + 1)])), __pack_half2(((bfloat16_t)dh_fragment_0[((i_21 * 8) + 2)]), ((bfloat16_t)dh_fragment_0[((i_21 * 8) + 3)])), __pack_half2(((bfloat16_t)dh_fragment_0[((i_21 * 8) + 4)]), ((bfloat16_t)dh_fragment_0[((i_21 * 8) + 5)])), __pack_half2(((bfloat16_t)dh_fragment_0[((i_21 * 8) + 6)]), ((bfloat16_t)dh_fragment_0[((i_21 * 8) + 7)])));
      }
      tl::fence_proxy_async();
      bar_dhtmp_ready[0].arrive();
    }
  } else {
    if (((int)threadIdx.x) < 256) {
      tl::warpgroup_reg_alloc<168>();
      for (int i_s_1 = 0; i_s_1 < num_iters; ++i_s_1) {
        bar_00[0].arrive();
        bar_00[0].wait((i_s_1 & 1));
        if (0 < i_s_1) {
          tl::__sync_thread_partial(7, 128);
          #pragma unroll
          for (int i_22 = 0; i_22 < 2; ++i_22) {
            tl::ptx_stmatrix_m8n8_x4((&(((bfloat16_t*)dqkv_shared)[(((((((((int)threadIdx.x) & 63) >> 5) * 1024) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_22) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), __pack_half2(((bfloat16_t)dk_frag_l[(i_22 * 8)]), ((bfloat16_t)dk_frag_l[((i_22 * 8) + 1)])), __pack_half2(((bfloat16_t)dk_frag_l[((i_22 * 8) + 2)]), ((bfloat16_t)dk_frag_l[((i_22 * 8) + 3)])), __pack_half2(((bfloat16_t)dk_frag_l[((i_22 * 8) + 4)]), ((bfloat16_t)dk_frag_l[((i_22 * 8) + 5)])), __pack_half2(((bfloat16_t)dk_frag_l[((i_22 * 8) + 6)]), ((bfloat16_t)dk_frag_l[((i_22 * 8) + 7)])));
          }
          tl::__sync_thread_partial(7, 128);
          #pragma unroll
          for (int i_23 = 0; i_23 < 2; ++i_23) {
            tl::ptx_stmatrix_m8n8_x4((&(((bfloat16_t*)dqkv_shared)[((((((((((int)threadIdx.x) & 63) >> 5) * 1024) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_23) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 2048)])), __pack_half2(((bfloat16_t)dk_frag_r[(i_23 * 8)]), ((bfloat16_t)dk_frag_r[((i_23 * 8) + 1)])), __pack_half2(((bfloat16_t)dk_frag_r[((i_23 * 8) + 2)]), ((bfloat16_t)dk_frag_r[((i_23 * 8) + 3)])), __pack_half2(((bfloat16_t)dk_frag_r[((i_23 * 8) + 4)]), ((bfloat16_t)dk_frag_r[((i_23 * 8) + 5)])), __pack_half2(((bfloat16_t)dk_frag_r[((i_23 * 8) + 6)]), ((bfloat16_t)dk_frag_r[((i_23 * 8) + 7)])));
          }
        }
        bar_01[0].arrive();
        bar_01[0].wait((i_s_1 & 1));
        bar_dhtmp_ready[0].wait((i_s_1 & 1));
        tl::__sync_thread_partial(7, 128);
        if (i_s_1 == 0) {
          #pragma unroll
          for (int i_24 = 0; i_24 < 8; ++i_24) {
            bfloat16_t broadcast_var_15 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
            *(uint4*)(((bfloat16_t*)tmp_shared_4_1) + ((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_24 * 512)) + (((((int)threadIdx.x) & 127) >> 4) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = make_uint4(__pack_nv_bfloat162(broadcast_var_15, broadcast_var_15), __pack_nv_bfloat162(broadcast_var_15, broadcast_var_15), __pack_nv_bfloat162(broadcast_var_15, broadcast_var_15), __pack_nv_bfloat162(broadcast_var_15, broadcast_var_15));
          }
          tl::fence_proxy_async();
        }
        tl::__sync_thread_partial(7, 128);
        #pragma unroll
        for (int i_25 = 0; i_25 < 8; ++i_25) {
          *(uint4*)(dh_left_fragment + (i_25 * 8)) = *(uint4*)(((bfloat16_t*)tmp_shared_4_1) + ((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_25 * 512)) + (((((int)threadIdx.x) & 127) >> 4) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)));
        }
        {
          bfloat16_t A_local_4[8];
          bfloat16_t B_local_4[32];
          #pragma unroll
          for (int i_26 = 0; i_26 < 8; ++i_26) {
            float broadcast_var_16 = 0x0p+0f/*0.000000e+00*/;
            *(float4*)(dv_fragment + (i_26 * 4)) = make_float4(broadcast_var_16, broadcast_var_16, broadcast_var_16, broadcast_var_16);
          }
          for (int ki_4 = 0; ki_4 < 4; ++ki_4) {
            tl::ptx_ldmatrix_x4((&(((bfloat16_t*)k_shared)[(((((((((int)threadIdx.x) & 63) >> 5) * 1024) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_4 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_4 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local_4[0])));
            #pragma unroll
            for (int i_27 = 0; i_27 < 4; ++i_27) {
              tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)tmp_shared_4_1)[((((((((((int)threadIdx.x) & 127) >> 6) * 4096) + (ki_4 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_27 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_27 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_4[(i_27 * 8)])));
            }
            for (int j_4 = 0; j_4 < 4; ++j_4) {
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dv_fragment + (j_4 * 8)), reinterpret_cast<const unsigned*>(A_local_4 + 0), reinterpret_cast<const unsigned*>(B_local_4 + (j_4 * 8)));
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dv_fragment + ((j_4 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_4 + 0), reinterpret_cast<const unsigned*>(B_local_4 + ((j_4 * 8) + 4)));
            }
          }
        }
        if (0 < i_s_1) {
          bar_k_left_ready[0].arrive();
        }
        tl::__sync_thread_partial(7, 128);
        if (i_s_1 == 0) {
          #pragma unroll
          for (int i_28 = 0; i_28 < 8; ++i_28) {
            bfloat16_t broadcast_var_17 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
            *(uint4*)(((bfloat16_t*)tmp_shared_4_1) + ((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_28 * 512)) + (((((int)threadIdx.x) & 127) >> 4) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = make_uint4(__pack_nv_bfloat162(broadcast_var_17, broadcast_var_17), __pack_nv_bfloat162(broadcast_var_17, broadcast_var_17), __pack_nv_bfloat162(broadcast_var_17, broadcast_var_17), __pack_nv_bfloat162(broadcast_var_17, broadcast_var_17));
          }
        } else {
          bar_dh_right_ready[0].wait(((i_s_1 + 1) & 1));
        }
        tl::fence_proxy_async();
        {
          bfloat16_t A_local_5[8];
          bfloat16_t B_local_5[32];
          tl::__sync_thread_partial(7, 128);
          for (int ki_5 = 0; ki_5 < 4; ++ki_5) {
            tl::ptx_ldmatrix_x4((&(((bfloat16_t*)k_shared)[((((((((((int)threadIdx.x) & 63) >> 5) * 1024) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_5 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_5 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 2048)])), (&(A_local_5[0])));
            #pragma unroll
            for (int i_29 = 0; i_29 < 4; ++i_29) {
              tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)tmp_shared_4_1)[((((((((((int)threadIdx.x) & 127) >> 6) * 4096) + (ki_5 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_29 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_29 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_5[(i_29 * 8)])));
            }
            for (int j_5 = 0; j_5 < 4; ++j_5) {
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dv_fragment + (j_5 * 8)), reinterpret_cast<const unsigned*>(A_local_5 + 0), reinterpret_cast<const unsigned*>(B_local_5 + (j_5 * 8)));
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dv_fragment + ((j_5 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_5 + 0), reinterpret_cast<const unsigned*>(B_local_5 + ((j_5 * 8) + 4)));
            }
          }
        }
        #pragma unroll
        for (int i_30 = 0; i_30 < 16; ++i_30) {
          float2 __11;
            float2 v__13 = *(float2*)(dv_fragment + (i_30 * 2));
            float2 v__14 = make_float2(g_rev_exp_shared[(((((((int)threadIdx.x) & 63) >> 5) * 16) + ((i_30 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))], g_rev_exp_shared[(((((((int)threadIdx.x) & 63) >> 5) * 16) + ((i_30 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))]);
            *(float2*)(&(__11.x)) = tl::mul2(*(float2*)(&(v__13.x)), *(float2*)(&(v__14.x)));
          *(float2*)(dv_fragment + (i_30 * 2)) = __11;
        }
        bar_02[0].arrive();
        bar_02[0].wait((i_s_1 & 1));
        {
          bfloat16_t A_local_6[8];
          bfloat16_t B_local_6[32];
          for (int ki_6 = 0; ki_6 < 2; ++ki_6) {
            tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)tmp_shared_1_1)[(((((ki_6 * 512) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 7) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), (&(A_local_6[0])));
            #pragma unroll
            for (int i_31 = 0; i_31 < 4; ++i_31) {
              tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)do_shared)[((((((((((int)threadIdx.x) & 127) >> 6) * 2048) + (ki_6 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_31 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_31 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_6[(i_31 * 8)])));
            }
            for (int j_6 = 0; j_6 < 4; ++j_6) {
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dv_fragment + (j_6 * 8)), reinterpret_cast<const unsigned*>(A_local_6 + 0), reinterpret_cast<const unsigned*>(B_local_6 + (j_6 * 8)));
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dv_fragment + ((j_6 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_6 + 0), reinterpret_cast<const unsigned*>(B_local_6 + ((j_6 * 8) + 4)));
            }
          }
        }
        bar_03[0].arrive();
        bar_03[0].wait((i_s_1 & 1));
        tl::__sync_thread_partial(7, 128);
        #pragma unroll
        for (int i_32 = 0; i_32 < 4; ++i_32) {
          tl::ptx_stmatrix_m8n8_x4((&(((bfloat16_t*)tmp_shared_2_1)[(((((((((int)threadIdx.x) >> 5) * 1024) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_32 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_32 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) - 4096)])), __pack_half2(((bfloat16_t)dv_fragment[(i_32 * 8)]), ((bfloat16_t)dv_fragment[((i_32 * 8) + 1)])), __pack_half2(((bfloat16_t)dv_fragment[((i_32 * 8) + 2)]), ((bfloat16_t)dv_fragment[((i_32 * 8) + 3)])), __pack_half2(((bfloat16_t)dv_fragment[((i_32 * 8) + 4)]), ((bfloat16_t)dv_fragment[((i_32 * 8) + 5)])), __pack_half2(((bfloat16_t)dv_fragment[((i_32 * 8) + 6)]), ((bfloat16_t)dv_fragment[((i_32 * 8) + 7)])));
        }
        bar_04[0].arrive();
        bar_04[0].wait((i_s_1 & 1));
        {
          bfloat16_t A_local_7[8];
          bfloat16_t B_local_7[32];
          #pragma unroll
          for (int i_33 = 0; i_33 < 8; ++i_33) {
            float broadcast_var_18 = 0x0p+0f/*0.000000e+00*/;
            *(float4*)(dv_fragment + (i_33 * 4)) = make_float4(broadcast_var_18, broadcast_var_18, broadcast_var_18, broadcast_var_18);
          }
          tl::__sync_thread_partial(7, 128);
          for (int ki_7 = 0; ki_7 < 2; ++ki_7) {
            tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)tmp_shared_1_2)[(((((ki_7 * 512) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 7) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), (&(A_local_7[0])));
            #pragma unroll
            for (int i_34 = 0; i_34 < 4; ++i_34) {
              tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)tmp_shared_2_1)[((((((((((int)threadIdx.x) & 127) >> 6) * 2048) + (ki_7 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_34 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_34 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_7[(i_34 * 8)])));
            }
            for (int j_7 = 0; j_7 < 4; ++j_7) {
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dv_fragment + (j_7 * 8)), reinterpret_cast<const unsigned*>(A_local_7 + 0), reinterpret_cast<const unsigned*>(B_local_7 + (j_7 * 8)));
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dv_fragment + ((j_7 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_7 + 0), reinterpret_cast<const unsigned*>(B_local_7 + ((j_7 * 8) + 4)));
            }
          }
        }
        tl::__sync_thread_partial(7, 128);
        #pragma unroll
        for (int i_35 = 0; i_35 < 4; ++i_35) {
          tl::ptx_stmatrix_m8n8_x4((&(((bfloat16_t*)dqkv_shared)[(((((((((int)threadIdx.x) >> 5) * 1024) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_35 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_35 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) - 4096)])), __pack_half2(((bfloat16_t)dv_fragment[(i_35 * 8)]), ((bfloat16_t)dv_fragment[((i_35 * 8) + 1)])), __pack_half2(((bfloat16_t)dv_fragment[((i_35 * 8) + 2)]), ((bfloat16_t)dv_fragment[((i_35 * 8) + 3)])), __pack_half2(((bfloat16_t)dv_fragment[((i_35 * 8) + 4)]), ((bfloat16_t)dv_fragment[((i_35 * 8) + 5)])), __pack_half2(((bfloat16_t)dv_fragment[((i_35 * 8) + 6)]), ((bfloat16_t)dv_fragment[((i_35 * 8) + 7)])));
        }
        bar_05[0].arrive();
        bar_05[0].wait((i_s_1 & 1));
        #pragma unroll
        for (int i_36 = 0; i_36 < 16; ++i_36) {
          float broadcast_var_19 = -0x1p+0f/*-1.000000e+00*/;
          float2 __12;
            float2 __13;
              float2 v__15 = *(float2*)(dv_fragment + (i_36 * 2));
              float2 v__16 = make_float2(broadcast_var_19, broadcast_var_19);
              *(float2*)(&(__13.x)) = tl::mul2(*(float2*)(&(v__15.x)), *(float2*)(&(v__16.x)));
            float2 v__17 = make_float2(g_exp_shared[(((((((int)threadIdx.x) & 63) >> 5) * 16) + ((i_36 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))], g_exp_shared[(((((((int)threadIdx.x) & 63) >> 5) * 16) + ((i_36 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))]);
            *(float2*)(&(__12.x)) = tl::mul2(*(float2*)(&(__13.x)), *(float2*)(&(v__17.x)));
          *(float2*)(dv_fragment + (i_36 * 2)) = __12;
        }
        tl::__sync_thread_partial(7, 128);
        #pragma unroll
        for (int i_37 = 0; i_37 < 16; ++i_37) {
          bfloat16_t tmp_shared_2_3_local_cast_5[2];
          *(uint1*)(tmp_shared_2_3_local_cast_5 + 0) = *(uint1*)(((bfloat16_t*)tmp_shared_2_3) + (((((((((((int)threadIdx.x) >> 5) * 1024) + ((i_37 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + (i_37 >> 3)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_37 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_37 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 4096));
          float2 __14;
          uint1 v__18 = *(uint1*)(tmp_shared_2_3_local_cast_5 + 0);
          ((float2*)(&__14))[0] = tl::__tl_cvt_bf16x2_to_f32x2((reinterpret_cast<__nv_bfloat162*>(&v__18))[0]);
          *(float2*)(odot_fragment_1 + (i_37 * 2)) = __14;
        }
        #pragma unroll
        for (int i_38 = 0; i_38 < 8; ++i_38) {
          float4 __15;
            float4 v__19 = *(float4*)(odot_fragment_1 + (i_38 * 4));
            float4 v__20 = *(float4*)(dv_fragment + (i_38 * 4));
            *(float2*)(&(__15.x)) = tl::mul2(*(float2*)(&(v__19.x)), *(float2*)(&(v__20.x)));
            *(float2*)(&(__15.z)) = tl::mul2(*(float2*)(&(v__19.z)), *(float2*)(&(v__20.z)));
          *(float4*)(odot_fragment_1 + (i_38 * 4)) = __15;
        }
        #pragma unroll
        for (int i_39 = 0; i_39 < 2; ++i_39) {
          dg_fragment_1[i_39] = 0x0p+0f/*0.000000e+00*/;
          #pragma unroll
          for (int rv = 0; rv < 16; ++rv) {
            dg_fragment_1[i_39] = (dg_fragment_1[i_39] + odot_fragment_1[((((rv & 7) * 4) + (i_39 * 2)) + (rv >> 3))]);
          }
          tl::__sync_thread_partial(7, 128);
          dg_fragment_1[i_39] = tl::AllReduce<tl::SumOp, 128, 64, 128, tl::NamedBarrier<128>>::run(dg_fragment_1[i_39], (&(((float*)workspace)[0])));
          dg_fragment_1[i_39] = tl::AllReduce<tl::SumOp, 4, 1, 128, tl::NamedBarrier<128>>::run(dg_fragment_1[i_39]);
        }
        tl::__sync_thread_partial(4, 128);
        if ((((((int)threadIdx.x) & 3) * 2) + (((int)threadIdx.x) >> 6)) == 2) {
          #pragma unroll
          for (int i_40 = 0; i_40 < 2; ++i_40) {
            dg_shared[(((((((int)threadIdx.x) & 63) >> 5) * 16) + (i_40 * 8)) + ((((int)threadIdx.x) & 31) >> 2))] = dg_fragment_1[i_40];
          }
        }
        bar_dg_k_init[0].arrive();
        bar_dg_k_init[0].wait((i_s_1 & 1));
        tl::__sync_thread_partial(7, 128);
        #pragma unroll
        for (int i_41 = 0; i_41 < 4; ++i_41) {
          tl::ptx_stmatrix_m8n8_x4((&(((bfloat16_t*)tmp_shared_2_3)[(((((((((int)threadIdx.x) >> 5) * 1024) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_41 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_41 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) - 4096)])), __pack_half2(((bfloat16_t)dv_fragment[(i_41 * 8)]), ((bfloat16_t)dv_fragment[((i_41 * 8) + 1)])), __pack_half2(((bfloat16_t)dv_fragment[((i_41 * 8) + 2)]), ((bfloat16_t)dv_fragment[((i_41 * 8) + 3)])), __pack_half2(((bfloat16_t)dv_fragment[((i_41 * 8) + 4)]), ((bfloat16_t)dv_fragment[((i_41 * 8) + 5)])), __pack_half2(((bfloat16_t)dv_fragment[((i_41 * 8) + 6)]), ((bfloat16_t)dv_fragment[((i_41 * 8) + 7)])));
        }
        bar_06[0].arrive();
        bar_06[0].wait((i_s_1 & 1));
        tl::__sync_thread_partial(7, 128);
        #pragma unroll
        for (int i_42 = 0; i_42 < 16; ++i_42) {
          bfloat16_t k_shared_local_cast_6[2];
          *(uint1*)(k_shared_local_cast_6 + 0) = *(uint1*)(((bfloat16_t*)k_shared) + (((((((((((int)threadIdx.x) >> 5) * 1024) + ((i_42 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + (i_42 >> 3)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_42 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_42 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 4096));
          float2 __16;
          uint1 v__21 = *(uint1*)(k_shared_local_cast_6 + 0);
          ((float2*)(&__16))[0] = tl::__tl_cvt_bf16x2_to_f32x2((reinterpret_cast<__nv_bfloat162*>(&v__21))[0]);
          *(float2*)(odot_fragment_1 + (i_42 * 2)) = __16;
        }
        tl::__sync_thread_partial(7, 128);
        #pragma unroll
        for (int i_43 = 0; i_43 < 4; ++i_43) {
          tl::ptx_stmatrix_m8n8_x4((&(((bfloat16_t*)tmp_shared_2_2)[(((((((((int)threadIdx.x) >> 5) * 1024) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_43 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_43 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) - 4096)])), __pack_half2(((bfloat16_t)odot_fragment_1[(i_43 * 8)]), ((bfloat16_t)odot_fragment_1[((i_43 * 8) + 1)])), __pack_half2(((bfloat16_t)odot_fragment_1[((i_43 * 8) + 2)]), ((bfloat16_t)odot_fragment_1[((i_43 * 8) + 3)])), __pack_half2(((bfloat16_t)odot_fragment_1[((i_43 * 8) + 4)]), ((bfloat16_t)odot_fragment_1[((i_43 * 8) + 5)])), __pack_half2(((bfloat16_t)odot_fragment_1[((i_43 * 8) + 6)]), ((bfloat16_t)odot_fragment_1[((i_43 * 8) + 7)])));
        }
        bar_07[0].arrive();
        bar_07[0].wait((i_s_1 & 1));
        {
          bfloat16_t A_local_8[8];
          bfloat16_t B_local_8[16];
          #pragma unroll
          for (int i_44 = 0; i_44 < 4; ++i_44) {
            float broadcast_var_20 = 0x0p+0f/*0.000000e+00*/;
            *(float4*)(dk_frag_r + (i_44 * 4)) = make_float4(broadcast_var_20, broadcast_var_20, broadcast_var_20, broadcast_var_20);
          }
          for (int ki_8 = 0; ki_8 < 8; ++ki_8) {
            tl::ptx_ldmatrix_x4((&(((bfloat16_t*)tmp_shared_2_1)[(((((((ki_8 >> 2) * 2048) + (((((int)threadIdx.x) & 63) >> 5) * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_8 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_8 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local_8[0])));
            #pragma unroll
            for (int i_45 = 0; i_45 < 2; ++i_45) {
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)tmp_shared_4_1)[(((((((((ki_8 >> 2) * 4096) + (((((int)threadIdx.x) & 127) >> 6) * 2048)) + (i_45 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_8 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_8 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_8[(i_45 * 8)])));
            }
            for (int j_8 = 0; j_8 < 2; ++j_8) {
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dk_frag_r + (j_8 * 8)), reinterpret_cast<const unsigned*>(A_local_8 + 0), reinterpret_cast<const unsigned*>(B_local_8 + (j_8 * 8)));
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dk_frag_r + ((j_8 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_8 + 0), reinterpret_cast<const unsigned*>(B_local_8 + ((j_8 * 8) + 4)));
            }
          }
        }
        tl::__sync_thread_partial(7, 128);
        #pragma unroll
        for (int i_46 = 0; i_46 < 8; ++i_46) {
          *(uint4*)(((bfloat16_t*)tmp_shared_4_1) + ((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_46 * 512)) + (((((int)threadIdx.x) & 127) >> 4) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(dh_left_fragment + (i_46 * 8));
        }
        tl::fence_proxy_async();
        {
          bfloat16_t A_local_9[8];
          bfloat16_t B_local_9[16];
          #pragma unroll
          for (int i_47 = 0; i_47 < 4; ++i_47) {
            float broadcast_var_21 = 0x0p+0f/*0.000000e+00*/;
            *(float4*)(dk_frag_l + (i_47 * 4)) = make_float4(broadcast_var_21, broadcast_var_21, broadcast_var_21, broadcast_var_21);
          }
          tl::__sync_thread_partial(7, 128);
          for (int ki_9 = 0; ki_9 < 8; ++ki_9) {
            tl::ptx_ldmatrix_x4((&(((bfloat16_t*)tmp_shared_2_1)[(((((((ki_9 >> 2) * 2048) + (((((int)threadIdx.x) & 63) >> 5) * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_9 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_9 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local_9[0])));
            #pragma unroll
            for (int i_48 = 0; i_48 < 2; ++i_48) {
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)tmp_shared_4_1)[(((((((((ki_9 >> 2) * 4096) + (((((int)threadIdx.x) & 127) >> 6) * 2048)) + (i_48 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_9 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_9 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_9[(i_48 * 8)])));
            }
            for (int j_9 = 0; j_9 < 2; ++j_9) {
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dk_frag_l + (j_9 * 8)), reinterpret_cast<const unsigned*>(A_local_9 + 0), reinterpret_cast<const unsigned*>(B_local_9 + (j_9 * 8)));
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dk_frag_l + ((j_9 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_9 + 0), reinterpret_cast<const unsigned*>(B_local_9 + ((j_9 * 8) + 4)));
            }
          }
        }
        #pragma unroll
        for (int i_49 = 0; i_49 < 8; ++i_49) {
          float2 __17;
            float2 v__22 = *(float2*)(dk_frag_r + (i_49 * 2));
            float2 v__23 = make_float2(g_rev_exp_shared[(((((((int)threadIdx.x) & 63) >> 5) * 16) + ((i_49 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))], g_rev_exp_shared[(((((((int)threadIdx.x) & 63) >> 5) * 16) + ((i_49 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))]);
            *(float2*)(&(__17.x)) = tl::mul2(*(float2*)(&(v__22.x)), *(float2*)(&(v__23.x)));
          *(float2*)(dk_frag_r + (i_49 * 2)) = __17;
        }
        #pragma unroll
        for (int i_50 = 0; i_50 < 8; ++i_50) {
          float2 __18;
            float2 v__24 = *(float2*)(dk_frag_l + (i_50 * 2));
            float2 v__25 = make_float2(g_rev_exp_shared[(((((((int)threadIdx.x) & 63) >> 5) * 16) + ((i_50 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))], g_rev_exp_shared[(((((((int)threadIdx.x) & 63) >> 5) * 16) + ((i_50 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))]);
            *(float2*)(&(__18.x)) = tl::mul2(*(float2*)(&(v__24.x)), *(float2*)(&(v__25.x)));
          *(float2*)(dk_frag_l + (i_50 * 2)) = __18;
        }
        tl::__sync_thread_partial(7, 128);
        #pragma unroll
        for (int i_51 = 0; i_51 < 2; ++i_51) {
          tl::ptx_stmatrix_m8n8_x4((&(((bfloat16_t*)dqkv_shared)[(((((((((int)threadIdx.x) & 63) >> 5) * 1024) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_51) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), __pack_half2(((bfloat16_t)dk_frag_l[(i_51 * 8)]), ((bfloat16_t)dk_frag_l[((i_51 * 8) + 1)])), __pack_half2(((bfloat16_t)dk_frag_l[((i_51 * 8) + 2)]), ((bfloat16_t)dk_frag_l[((i_51 * 8) + 3)])), __pack_half2(((bfloat16_t)dk_frag_l[((i_51 * 8) + 4)]), ((bfloat16_t)dk_frag_l[((i_51 * 8) + 5)])), __pack_half2(((bfloat16_t)dk_frag_l[((i_51 * 8) + 6)]), ((bfloat16_t)dk_frag_l[((i_51 * 8) + 7)])));
        }
        tl::__sync_thread_partial(7, 128);
        #pragma unroll
        for (int i_52 = 0; i_52 < 2; ++i_52) {
          tl::ptx_stmatrix_m8n8_x4((&(((bfloat16_t*)dqkv_shared)[((((((((((int)threadIdx.x) & 63) >> 5) * 1024) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_52) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 2048)])), __pack_half2(((bfloat16_t)dk_frag_r[(i_52 * 8)]), ((bfloat16_t)dk_frag_r[((i_52 * 8) + 1)])), __pack_half2(((bfloat16_t)dk_frag_r[((i_52 * 8) + 2)]), ((bfloat16_t)dk_frag_r[((i_52 * 8) + 3)])), __pack_half2(((bfloat16_t)dk_frag_r[((i_52 * 8) + 4)]), ((bfloat16_t)dk_frag_r[((i_52 * 8) + 5)])), __pack_half2(((bfloat16_t)dk_frag_r[((i_52 * 8) + 6)]), ((bfloat16_t)dk_frag_r[((i_52 * 8) + 7)])));
        }
        tl::fence_proxy_async();
        tl::__sync_thread_partial(7, 128);
        #pragma unroll
        for (int i_53 = 0; i_53 < 16; ++i_53) {
          bfloat16_t dqkv_shared_local_cast_7[2];
          *(uint1*)(dqkv_shared_local_cast_7 + 0) = *(uint1*)(((bfloat16_t*)dqkv_shared) + (((((((((((int)threadIdx.x) >> 5) * 1024) + ((i_53 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + (i_53 >> 3)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_53 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_53 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 4096));
          bfloat16_t broadcast_var_22 = bfloat16_t(-0x1p+0f/*-1.000000e+00*/);
          float2 __19;
            float2 v__26 = *(float2*)(odot_fragment_1 + (i_53 * 2));
            float2 __20;
            uint1 __21;
              uint1 v__27 = *(uint1*)(dqkv_shared_local_cast_7 + 0);
              uint1 v__28 = make_uint1(__pack_nv_bfloat162(broadcast_var_22, broadcast_var_22));
              *(uint1*)(&(__21.x)) = tl::to_uint1(tl::mul2(tl::from_uint1<__nv_bfloat162>(*(uint1*)(&(v__27.x))), tl::from_uint1<__nv_bfloat162>(*(uint1*)(&(v__28.x)))));
            ((float2*)(&__20))[0] = tl::__tl_cvt_bf16x2_to_f32x2((reinterpret_cast<__nv_bfloat162*>(&__21))[0]);
            *(float2*)(&(__19.x)) = tl::mul2(*(float2*)(&(v__26.x)), *(float2*)(&(__20.x)));
          *(float2*)(odot_fragment_1 + (i_53 * 2)) = __19;
        }
        #pragma unroll
        for (int i_54 = 0; i_54 < 2; ++i_54) {
          dg_fragment_1[i_54] = 0x0p+0f/*0.000000e+00*/;
          #pragma unroll
          for (int rv_1 = 0; rv_1 < 16; ++rv_1) {
            dg_fragment_1[i_54] = (dg_fragment_1[i_54] + odot_fragment_1[((((rv_1 & 7) * 4) + (i_54 * 2)) + (rv_1 >> 3))]);
          }
          tl::__sync_thread_partial(7, 128);
          dg_fragment_1[i_54] = tl::AllReduce<tl::SumOp, 128, 64, 128, tl::NamedBarrier<128>>::run(dg_fragment_1[i_54], (&(((float*)workspace_3)[0])));
          dg_fragment_1[i_54] = tl::AllReduce<tl::SumOp, 4, 1, 128, tl::NamedBarrier<128>>::run(dg_fragment_1[i_54]);
        }
        tl::__sync_thread_partial(4, 128);
        if ((((((int)threadIdx.x) & 3) * 2) + (((int)threadIdx.x) >> 6)) == 2) {
          #pragma unroll
          for (int i_55 = 0; i_55 < 2; ++i_55) {
            dg_shared[(((((((int)threadIdx.x) & 63) >> 5) * 16) + (i_55 * 8)) + ((((int)threadIdx.x) & 31) >> 2))] = (dg_shared[(((((((int)threadIdx.x) & 63) >> 5) * 16) + (i_55 * 8)) + ((((int)threadIdx.x) & 31) >> 2))] + dg_fragment_1[i_55]);
          }
        }
        float broadcast_var_23 = 0x0p+0f/*0.000000e+00*/;
        dg_last_local_1_pack[0] = make_float2(broadcast_var_23, broadcast_var_23);
        dg_last_local_1_pack[0] = tl::add2(dg_last_local_1_pack[0], *(float2*)(dg_fragment_1 + 0));
        dg_last_local_1[0] = ((dg_last_local_1_pack[0]).x + (dg_last_local_1_pack[0]).y);
        dg_last_local_1[0] = tl::AllReduce<tl::SumOp, 64, 32, 128, tl::NamedBarrier<128>>::run(dg_last_local_1[0], (&(((float*)workspace_4)[0])));
        dg_last_local_1[0] = tl::AllReduce<tl::SumOp, 32, 4, 128, tl::NamedBarrier<128>>::run(dg_last_local_1[0]);
        bar_dg_k_rows[0].arrive();
        bar_dg_k_rows[0].wait((i_s_1 & 1));
        tl::__sync_thread_partial(4, 128);
        if (((int)threadIdx.x) == 128) {
          dg_shared[31] = (dg_shared[31] - dg_last_local_1[0]);
        }
        bar_08_1[0].arrive();
        bar_08_1[0].wait((i_s_1 & 1));
        {
          bfloat16_t A_local_10[8];
          bfloat16_t B_local_10[16];
          for (int ki_10 = 0; ki_10 < 8; ++ki_10) {
            tl::ptx_ldmatrix_x4((&(((bfloat16_t*)tmp_shared_2_3)[(((((((ki_10 >> 2) * 2048) + (((((int)threadIdx.x) & 63) >> 5) * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_10 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_10 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local_10[0])));
            #pragma unroll
            for (int i_56 = 0; i_56 < 2; ++i_56) {
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)h_shared)[(((((((((ki_10 >> 2) * 4096) + (((((int)threadIdx.x) & 127) >> 6) * 2048)) + (i_56 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_10 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_10 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_10[(i_56 * 8)])));
            }
            for (int j_10 = 0; j_10 < 2; ++j_10) {
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dk_frag_r + (j_10 * 8)), reinterpret_cast<const unsigned*>(A_local_10 + 0), reinterpret_cast<const unsigned*>(B_local_10 + (j_10 * 8)));
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dk_frag_r + ((j_10 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_10 + 0), reinterpret_cast<const unsigned*>(B_local_10 + ((j_10 * 8) + 4)));
            }
          }
        }
        bar_08_2[0].arrive();
        bar_08_3[0].wait((i_s_1 & 1));
        {
          bfloat16_t A_local_11[8];
          bfloat16_t B_local_11[16];
          for (int ki_11 = 0; ki_11 < 8; ++ki_11) {
            tl::ptx_ldmatrix_x4((&(((bfloat16_t*)tmp_shared_2_3)[(((((((ki_11 >> 2) * 2048) + (((((int)threadIdx.x) & 63) >> 5) * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_11 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_11 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local_11[0])));
            #pragma unroll
            for (int i_57 = 0; i_57 < 2; ++i_57) {
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)h_shared)[(((((((((ki_11 >> 2) * 4096) + (((((int)threadIdx.x) & 127) >> 6) * 2048)) + (i_57 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_11 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_11 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_11[(i_57 * 8)])));
            }
            for (int j_11 = 0; j_11 < 2; ++j_11) {
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dk_frag_l + (j_11 * 8)), reinterpret_cast<const unsigned*>(A_local_11 + 0), reinterpret_cast<const unsigned*>(B_local_11 + (j_11 * 8)));
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dk_frag_l + ((j_11 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_11 + 0), reinterpret_cast<const unsigned*>(B_local_11 + ((j_11 * 8) + 4)));
            }
          }
        }
        bar_10[0].arrive();
        bar_10[0].wait((i_s_1 & 1));
        bar_12[0].wait((i_s_1 & 1));
        {
          bfloat16_t A_local_12[8];
          bfloat16_t B_local_12[16];
          for (int ki_12 = 0; ki_12 < 2; ++ki_12) {
            tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)tmp_shared_1_1)[(((((ki_12 * 512) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 7) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), (&(A_local_12[0])));
            #pragma unroll
            for (int i_58 = 0; i_58 < 2; ++i_58) {
              tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)tmp_shared_2_1)[(((((ki_12 * 1024) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_58) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_12[(i_58 * 8)])));
            }
            for (int j_12 = 0; j_12 < 2; ++j_12) {
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dk_frag_l + (j_12 * 8)), reinterpret_cast<const unsigned*>(A_local_12 + 0), reinterpret_cast<const unsigned*>(B_local_12 + (j_12 * 8)));
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dk_frag_l + ((j_12 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_12 + 0), reinterpret_cast<const unsigned*>(B_local_12 + ((j_12 * 8) + 4)));
            }
          }
        }
        {
          bfloat16_t A_local_13[8];
          bfloat16_t B_local_13[16];
          for (int ki_13 = 0; ki_13 < 2; ++ki_13) {
            tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)tmp_shared_1_1)[(((((ki_13 * 512) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 7) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), (&(A_local_13[0])));
            #pragma unroll
            for (int i_59 = 0; i_59 < 2; ++i_59) {
              tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)tmp_shared_2_1)[((((((ki_13 * 1024) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_59) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 2048)])), (&(B_local_13[(i_59 * 8)])));
            }
            for (int j_13 = 0; j_13 < 2; ++j_13) {
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dk_frag_r + (j_13 * 8)), reinterpret_cast<const unsigned*>(A_local_13 + 0), reinterpret_cast<const unsigned*>(B_local_13 + (j_13 * 8)));
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dk_frag_r + ((j_13 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_13 + 0), reinterpret_cast<const unsigned*>(B_local_13 + ((j_13 * 8) + 4)));
            }
          }
        }
        bar_13[0].arrive();
        bar_13[0].wait((i_s_1 & 1));
        bar_15[0].wait((i_s_1 & 1));
        {
          bfloat16_t A_local_14[8];
          bfloat16_t B_local_14[16];
          for (int ki_14 = 0; ki_14 < 2; ++ki_14) {
            tl::ptx_ldmatrix_x4((&(((bfloat16_t*)tmp_shared_1_2)[((((((((int)threadIdx.x) & 63) >> 5) * 512) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki_14) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), (&(A_local_14[0])));
            #pragma unroll
            for (int i_60 = 0; i_60 < 2; ++i_60) {
              tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)tmp_shared_2_2)[(((((ki_14 * 1024) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_60) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_14[(i_60 * 8)])));
            }
            for (int j_14 = 0; j_14 < 2; ++j_14) {
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dk_frag_l + (j_14 * 8)), reinterpret_cast<const unsigned*>(A_local_14 + 0), reinterpret_cast<const unsigned*>(B_local_14 + (j_14 * 8)));
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dk_frag_l + ((j_14 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_14 + 0), reinterpret_cast<const unsigned*>(B_local_14 + ((j_14 * 8) + 4)));
            }
          }
        }
        {
          bfloat16_t A_local_15[8];
          bfloat16_t B_local_15[16];
          for (int ki_15 = 0; ki_15 < 2; ++ki_15) {
            tl::ptx_ldmatrix_x4((&(((bfloat16_t*)tmp_shared_1_2)[((((((((int)threadIdx.x) & 63) >> 5) * 512) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki_15) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), (&(A_local_15[0])));
            #pragma unroll
            for (int i_61 = 0; i_61 < 2; ++i_61) {
              tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)tmp_shared_2_2)[((((((ki_15 * 1024) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_61) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 2048)])), (&(B_local_15[(i_61 * 8)])));
            }
            for (int j_15 = 0; j_15 < 2; ++j_15) {
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dk_frag_r + (j_15 * 8)), reinterpret_cast<const unsigned*>(A_local_15 + 0), reinterpret_cast<const unsigned*>(B_local_15 + (j_15 * 8)));
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dk_frag_r + ((j_15 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_15 + 0), reinterpret_cast<const unsigned*>(B_local_15 + ((j_15 * 8) + 4)));
            }
          }
        }
      }
      if (0 <= batch_idx) {
        #pragma unroll
        for (int i_62 = 0; i_62 < 8; ++i_62) {
          if (((((((((int)threadIdx.x) & 63) >> 5) * 16) + ((i_62 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + seq_start_idx) < seq_end_idx) {
            if (0 <= ((((((((int)threadIdx.x) & 63) >> 5) * 16) + ((i_62 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + seq_start_idx)) {
              if (((((((((int)threadIdx.x) & 63) >> 5) * 16) + ((i_62 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + seq_start_idx) < num_tokens) {
                if (batch_idx < batch_size) {
                  uint1 __22;
                  float2 v__29 = *(float2*)(dk_frag_l + (i_62 * 2));
                  (reinterpret_cast<__nv_bfloat162*>(&__22))[0] = __float22bfloat162_rn(((float2*)(&v__29))[0]);
                  *(uint1*)(dk + ((((((((((((((int64_t)((int)threadIdx.x)) & (int64_t)63) >> (int64_t)5) * (int64_t)8192) + ((((int64_t)i_62) & (int64_t)1) * (int64_t)4096)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2) * (int64_t)512)) + (((int64_t)seq_start_idx) * (int64_t)512)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)512)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)3) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)6) * (int64_t)32)) + ((((int64_t)i_62) >> (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2)) - (int64_t)64)) = __22;
                }
              }
            }
          }
        }
      }
      if (0 <= batch_idx) {
        #pragma unroll
        for (int i_63 = 0; i_63 < 8; ++i_63) {
          if (((((((((int)threadIdx.x) & 63) >> 5) * 16) + ((i_63 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + seq_start_idx) < seq_end_idx) {
            if (0 <= ((((((((int)threadIdx.x) & 63) >> 5) * 16) + ((i_63 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + seq_start_idx)) {
              if (((((((((int)threadIdx.x) & 63) >> 5) * 16) + ((i_63 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + seq_start_idx) < num_tokens) {
                if (batch_idx < batch_size) {
                  uint1 __23;
                  float2 v__30 = *(float2*)(dk_frag_r + (i_63 * 2));
                  (reinterpret_cast<__nv_bfloat162*>(&__23))[0] = __float22bfloat162_rn(((float2*)(&v__30))[0]);
                  *(uint1*)(dk + (((((((((((((int64_t)((int)threadIdx.x)) & (int64_t)63) >> (int64_t)5) * (int64_t)8192) + ((((int64_t)i_63) & (int64_t)1) * (int64_t)4096)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2) * (int64_t)512)) + (((int64_t)seq_start_idx) * (int64_t)512)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)512)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)3) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)6) * (int64_t)32)) + ((((int64_t)i_63) >> (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2))) = __23;
                }
              }
            }
          }
        }
      }
    } else {
      if (((int)threadIdx.x) < 384) {
        tl::warpgroup_reg_alloc<160>();
        for (int i_s_2 = 0; i_s_2 < num_iters; ++i_s_2) {
          int cur_idx_1 = (((chunk_start_idx + num_iters) - i_s_2) - 1);
          bar_00[0].arrive();
          bar_00[0].wait((i_s_2 & 1));
          {
            bfloat16_t A_local_16[8];
            bfloat16_t B_local_16[8];
            #pragma unroll
            for (int i_64 = 0; i_64 < 2; ++i_64) {
              float broadcast_var_24 = 0x0p+0f/*0.000000e+00*/;
              *(float4*)(p_fragment + (i_64 * 4)) = make_float4(broadcast_var_24, broadcast_var_24, broadcast_var_24, broadcast_var_24);
            }
            for (int ki_16 = 0; ki_16 < 8; ++ki_16) {
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)tmp_shared_2_1)[(((((((ki_16 >> 2) * 2048) + (((((int)threadIdx.x) & 63) >> 5) * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_16 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_16 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local_16[0])));
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)k_shared)[((((((((ki_16 >> 2) * 2048) + (((((int)threadIdx.x) & 127) >> 6) * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_16 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_16 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_16[0])));
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(p_fragment + 0), reinterpret_cast<const unsigned*>(A_local_16 + 0), reinterpret_cast<const unsigned*>(B_local_16 + 0));
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(p_fragment + 4), reinterpret_cast<const unsigned*>(A_local_16 + 0), reinterpret_cast<const unsigned*>(B_local_16 + 4));
            }
          }
          bar_01[0].arrive();
          bar_01[0].wait((i_s_2 & 1));
          #pragma unroll
          for (int i_65 = 0; i_65 < 4; ++i_65) {
            float2 __24;
              float2 v__31 = make_float2(g_shared[(((((((int)threadIdx.x) & 63) >> 5) * 16) + ((i_65 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))], g_shared[(((((((int)threadIdx.x) & 63) >> 5) * 16) + ((i_65 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))]);
              float2 v__32 = *(float2*)(g_shared + (((((((int)threadIdx.x) >> 6) * 16) + ((i_65 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 64));
              *(float2*)(&(__24.x)) = tl::sub2(*(float2*)(&(v__31.x)), *(float2*)(&(v__32.x)));
            *(float2*)(mask_fragment + (i_65 * 2)) = __24;
          }
          #pragma unroll
          for (int i_66 = 0; i_66 < 8; ++i_66) {
            if ((((((((int)threadIdx.x) >> 6) * 16) + ((i_66 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_66 & 1)) <= ((((((((int)threadIdx.x) & 63) >> 5) * 16) + (((i_66 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + 64)) {
              mask_fragment[i_66] = exp2f((mask_fragment[i_66] * 0x1.715475a31a4bep+0f/*1.442695e+00*/));
            } else {
              mask_fragment[i_66] = 0x0p+0f/*0.000000e+00*/;
            }
          }
          #pragma unroll
          for (int i_67 = 0; i_67 < 2; ++i_67) {
            float4 __25;
              float4 v__33 = *(float4*)(p_fragment + (i_67 * 4));
              float4 v__34 = *(float4*)(mask_fragment + (i_67 * 4));
              *(float2*)(&(__25.x)) = tl::mul2(*(float2*)(&(v__33.x)), *(float2*)(&(v__34.x)));
              *(float2*)(&(__25.z)) = tl::mul2(*(float2*)(&(v__33.z)), *(float2*)(&(v__34.z)));
            *(float4*)(p_fragment + (i_67 * 4)) = __25;
          }
          #pragma unroll
          for (int i_68 = 0; i_68 < 2; ++i_68) {
            float broadcast_var_25 = 0x1.6a09e667f3bcdp-4f/*8.838835e-02*/;
            float4 __26;
              float4 v__35 = *(float4*)(p_fragment + (i_68 * 4));
              float4 v__36 = make_float4(broadcast_var_25, broadcast_var_25, broadcast_var_25, broadcast_var_25);
              *(float2*)(&(__26.x)) = tl::mul2(*(float2*)(&(v__35.x)), *(float2*)(&(v__36.x)));
              *(float2*)(&(__26.z)) = tl::mul2(*(float2*)(&(v__35.z)), *(float2*)(&(v__36.z)));
            *(float4*)(p_fragment + (i_68 * 4)) = __26;
          }
          tl::ptx_stmatrix_m8n8_x4((&(((bfloat16_t*)tmp_shared_1_1)[((((((((int)threadIdx.x) & 63) >> 5) * 512) + ((((int)threadIdx.x) & 15) * 32)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), __pack_half2(((bfloat16_t)p_fragment[0]), ((bfloat16_t)p_fragment[1])), __pack_half2(((bfloat16_t)p_fragment[2]), ((bfloat16_t)p_fragment[3])), __pack_half2(((bfloat16_t)p_fragment[4]), ((bfloat16_t)p_fragment[5])), __pack_half2(((bfloat16_t)p_fragment[6]), ((bfloat16_t)p_fragment[7])));
          bar_02[0].arrive();
          bar_02[0].wait((i_s_2 & 1));
          tl::__sync_thread_partial(8, 128);
          #pragma unroll
          for (int i_69 = 0; i_69 < 4; ++i_69) {
            bfloat16_t a_shared_local_cast_8[2];
            *(uint1*)(a_shared_local_cast_8 + 0) = *(uint1*)(((bfloat16_t*)a_shared) + ((((((((((int)threadIdx.x) & 63) >> 5) * 512) + ((i_69 & 1) * 256)) + (((((int)threadIdx.x) & 31) >> 2) * 32)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (i_69 >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
            float2 __27;
            uint1 v__37 = *(uint1*)(a_shared_local_cast_8 + 0);
            ((float2*)(&__27))[0] = tl::__tl_cvt_bf16x2_to_f32x2((reinterpret_cast<__nv_bfloat162*>(&v__37))[0]);
            *(float2*)(a_fragment + (i_69 * 2)) = __27;
          }
          #pragma unroll
          for (int i_70 = 0; i_70 < 4; ++i_70) {
            float2 __28;
              float2 v__38 = *(float2*)(a_fragment + (i_70 * 2));
              float2 v__39 = *(float2*)(b_shared + (((((((int)threadIdx.x) >> 6) * 16) + ((i_70 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 64));
              *(float2*)(&(__28.x)) = tl::mul2(*(float2*)(&(v__38.x)), *(float2*)(&(v__39.x)));
            *(float2*)(a_fragment + (i_70 * 2)) = __28;
          }
          #pragma unroll
          for (int i_71 = 0; i_71 < 2; ++i_71) {
            float4 __29;
              float4 v__40 = *(float4*)(a_fragment + (i_71 * 4));
              float4 v__41 = *(float4*)(mask_fragment + (i_71 * 4));
              *(float2*)(&(__29.x)) = tl::mul2(*(float2*)(&(v__40.x)), *(float2*)(&(v__41.x)));
              *(float2*)(&(__29.z)) = tl::mul2(*(float2*)(&(v__40.z)), *(float2*)(&(v__41.z)));
            *(float4*)(a_fragment + (i_71 * 4)) = __29;
          }
          tl::__sync_thread_partial(8, 128);
          tl::ptx_stmatrix_m8n8_x4((&(((bfloat16_t*)tmp_shared_1_2)[((((((((int)threadIdx.x) & 63) >> 5) * 512) + ((((int)threadIdx.x) & 15) * 32)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), __pack_half2(((bfloat16_t)a_fragment[0]), ((bfloat16_t)a_fragment[1])), __pack_half2(((bfloat16_t)a_fragment[2]), ((bfloat16_t)a_fragment[3])), __pack_half2(((bfloat16_t)a_fragment[4]), ((bfloat16_t)a_fragment[5])), __pack_half2(((bfloat16_t)a_fragment[6]), ((bfloat16_t)a_fragment[7])));
          tl::__sync_thread_partial(8, 128);
          #pragma unroll
          for (int i_72 = 0; i_72 < 4; ++i_72) {
            bfloat16_t tmp_shared_2_1_local_cast_9[8];
            *(uint4*)(tmp_shared_2_1_local_cast_9 + 0) = *(uint4*)(((bfloat16_t*)tmp_shared_2_1) + ((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_72 * 512)) + (((((int)threadIdx.x) & 127) >> 4) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)));
            for (int vec_1 = 0; vec_1 < 2; ++vec_1) {
              float4 __30;
              uint2 v__42 = *(uint2*)(tmp_shared_2_1_local_cast_9 + (vec_1 * 4));
              ((float2*)(&__30))[0] = tl::__tl_cvt_bf16x2_to_f32x2((reinterpret_cast<__nv_bfloat162*>(&v__42))[0]);
              ((float2*)(&__30))[1] = tl::__tl_cvt_bf16x2_to_f32x2((reinterpret_cast<__nv_bfloat162*>(&v__42))[1]);
              *(float4*)(odot_fragment_2 + ((i_72 * 8) + (vec_1 * 4))) = __30;
            }
          }
          bar_03[0].arrive();
          bar_03[0].wait((i_s_2 & 1));
          {
            bfloat16_t A_local_17[8];
            bfloat16_t B_local_17[32];
            #pragma unroll
            for (int i_73 = 0; i_73 < 8; ++i_73) {
              float broadcast_var_26 = 0x0p+0f/*0.000000e+00*/;
              *(float4*)(u_fragment + (i_73 * 4)) = make_float4(broadcast_var_26, broadcast_var_26, broadcast_var_26, broadcast_var_26);
            }
            for (int ki_17 = 0; ki_17 < 4; ++ki_17) {
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)k_shared)[(((((((((int)threadIdx.x) & 63) >> 5) * 1024) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_17 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_17 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local_17[0])));
              #pragma unroll
              for (int i_74 = 0; i_74 < 4; ++i_74) {
                tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)h_shared)[((((((((((int)threadIdx.x) & 127) >> 6) * 4096) + (ki_17 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_74 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_74 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_17[(i_74 * 8)])));
              }
              for (int j_16 = 0; j_16 < 4; ++j_16) {
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(u_fragment + (j_16 * 8)), reinterpret_cast<const unsigned*>(A_local_17 + 0), reinterpret_cast<const unsigned*>(B_local_17 + (j_16 * 8)));
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(u_fragment + ((j_16 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_17 + 0), reinterpret_cast<const unsigned*>(B_local_17 + ((j_16 * 8) + 4)));
              }
            }
          }
          tl::__sync_thread_partial(8, 128);
          #pragma unroll
          for (int i_75 = 0; i_75 < 8; ++i_75) {
            bfloat16_t broadcast_var_27 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
            uint4 condval_9;
            if (((((0 <= cur_idx_1) && (cur_idx_1 < num_chunks)) && (0 <= batch_idx)) && (batch_idx < batch_size))) {
              condval_9 = *(uint4*)(h + ((((((((int64_t)cur_idx_1) * (int64_t)65536) + ((((int64_t)batch_idx) * ((int64_t)num_chunks)) * (int64_t)65536)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)3) * (int64_t)16384)) + (((int64_t)i_75) * (int64_t)1024)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)) + (int64_t)6144));
            } else {
              condval_9 = make_uint4(__pack_nv_bfloat162(broadcast_var_27, broadcast_var_27), __pack_nv_bfloat162(broadcast_var_27, broadcast_var_27), __pack_nv_bfloat162(broadcast_var_27, broadcast_var_27), __pack_nv_bfloat162(broadcast_var_27, broadcast_var_27));
            }
            *(uint4*)(((bfloat16_t*)h_shared) + ((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_75 * 512)) + (((((int)threadIdx.x) & 127) >> 4) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval_9;
          }
          {
            bfloat16_t A_local_18[8];
            bfloat16_t B_local_18[32];
            tl::__sync_thread_partial(8, 128);
            for (int ki_18 = 0; ki_18 < 4; ++ki_18) {
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)k_shared)[((((((((((int)threadIdx.x) & 63) >> 5) * 1024) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_18 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_18 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 2048)])), (&(A_local_18[0])));
              #pragma unroll
              for (int i_76 = 0; i_76 < 4; ++i_76) {
                tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)h_shared)[((((((((((int)threadIdx.x) & 127) >> 6) * 4096) + (ki_18 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_76 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_76 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_18[(i_76 * 8)])));
              }
              for (int j_17 = 0; j_17 < 4; ++j_17) {
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(u_fragment + (j_17 * 8)), reinterpret_cast<const unsigned*>(A_local_18 + 0), reinterpret_cast<const unsigned*>(B_local_18 + (j_17 * 8)));
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(u_fragment + ((j_17 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_18 + 0), reinterpret_cast<const unsigned*>(B_local_18 + ((j_17 * 8) + 4)));
              }
            }
          }
          bar_04[0].arrive();
          bar_04[0].wait((i_s_2 & 1));
          tl::__sync_thread_partial(8, 128);
          #pragma unroll
          for (int i_77 = 0; i_77 < 4; ++i_77) {
            tl::ptx_stmatrix_m8n8_x4((&(((bfloat16_t*)tmp_shared_2_3)[(((((((((int)threadIdx.x) >> 5) * 1024) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_77 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_77 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) - 8192)])), __pack_half2(((bfloat16_t)u_fragment[(i_77 * 8)]), ((bfloat16_t)u_fragment[((i_77 * 8) + 1)])), __pack_half2(((bfloat16_t)u_fragment[((i_77 * 8) + 2)]), ((bfloat16_t)u_fragment[((i_77 * 8) + 3)])), __pack_half2(((bfloat16_t)u_fragment[((i_77 * 8) + 4)]), ((bfloat16_t)u_fragment[((i_77 * 8) + 5)])), __pack_half2(((bfloat16_t)u_fragment[((i_77 * 8) + 6)]), ((bfloat16_t)u_fragment[((i_77 * 8) + 7)])));
          }
          #pragma unroll
          for (int i_78 = 0; i_78 < 16; ++i_78) {
            float2 __31;
              float2 v__43 = *(float2*)(u_fragment + (i_78 * 2));
              float2 v__44 = make_float2((g_exp_shared[(((((((int)threadIdx.x) & 63) >> 5) * 16) + ((i_78 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))] * -0x1p+0f/*-1.000000e+00*/), (g_exp_shared[(((((((int)threadIdx.x) & 63) >> 5) * 16) + ((i_78 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))] * -0x1p+0f/*-1.000000e+00*/));
              *(float2*)(&(__31.x)) = tl::mul2(*(float2*)(&(v__43.x)), *(float2*)(&(v__44.x)));
            *(float2*)(u_fragment + (i_78 * 2)) = __31;
          }
          #pragma unroll
          for (int i_79 = 0; i_79 < 16; ++i_79) {
            if (((((((num_iters * 32) + (((((int)threadIdx.x) & 63) >> 5) * 16)) + ((i_79 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + seq_start_idx) - (i_s_2 * 32)) < (seq_end_idx + 32)) {
              bfloat16_t broadcast_var_28 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
              float2 __32;
                float2 v__45 = *(float2*)(u_fragment + (i_79 * 2));
                float2 __33;
                uint1 condval_10;
                if (((((1 <= (((((((((((int)threadIdx.x) & 63) >> 5) * 16) + ((i_79 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + seq_start_idx) >> 5) + num_iters) - i_s_2)) && (((((((num_iters * 32) + (((((int)threadIdx.x) & 63) >> 5) * 16)) + ((i_79 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + seq_start_idx) - (i_s_2 * 32)) < (num_tokens + 32))) && (0 <= batch_idx)) && (batch_idx < batch_size))) {
                  condval_10 = *(uint1*)(v + ((((((((((((((int64_t)num_iters) * (int64_t)16384) + (((((int64_t)((int)threadIdx.x)) & (int64_t)63) >> (int64_t)5) * (int64_t)8192)) + ((((int64_t)i_79) & (int64_t)1) * (int64_t)4096)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2) * (int64_t)512)) + (((int64_t)seq_start_idx) * (int64_t)512)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)512)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)3) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)6) * (int64_t)64)) + ((((int64_t)i_79) >> (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2)) - (((int64_t)i_s_2) * (int64_t)16384)) - (int64_t)16640));
                } else {
                  condval_10 = make_uint1(__pack_nv_bfloat162(broadcast_var_28, broadcast_var_28));
                }
                ((float2*)(&__33))[0] = tl::__tl_cvt_bf16x2_to_f32x2((reinterpret_cast<__nv_bfloat162*>(&condval_10))[0]);
                *(float2*)(&(__32.x)) = tl::add2(*(float2*)(&(v__45.x)), *(float2*)(&(__33.x)));
              *(float2*)(u_fragment + (i_79 * 2)) = __32;
            }
          }
          tl::__sync_thread_partial(8, 128);
          #pragma unroll
          for (int i_80 = 0; i_80 < 4; ++i_80) {
            tl::ptx_stmatrix_m8n8_x4((&(((bfloat16_t*)tmp_shared_2_2)[(((((((((int)threadIdx.x) >> 5) * 1024) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_80 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_80 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) - 8192)])), __pack_half2(((bfloat16_t)u_fragment[(i_80 * 8)]), ((bfloat16_t)u_fragment[((i_80 * 8) + 1)])), __pack_half2(((bfloat16_t)u_fragment[((i_80 * 8) + 2)]), ((bfloat16_t)u_fragment[((i_80 * 8) + 3)])), __pack_half2(((bfloat16_t)u_fragment[((i_80 * 8) + 4)]), ((bfloat16_t)u_fragment[((i_80 * 8) + 5)])), __pack_half2(((bfloat16_t)u_fragment[((i_80 * 8) + 6)]), ((bfloat16_t)u_fragment[((i_80 * 8) + 7)])));
          }
          bar_05[0].arrive();
          bar_05[0].wait((i_s_2 & 1));
          {
            bfloat16_t A_local_19[8];
            bfloat16_t B_local_19[8];
            #pragma unroll
            for (int i_81 = 0; i_81 < 2; ++i_81) {
              float broadcast_var_29 = 0x0p+0f/*0.000000e+00*/;
              *(float4*)(da_fragment + (i_81 * 4)) = make_float4(broadcast_var_29, broadcast_var_29, broadcast_var_29, broadcast_var_29);
            }
            tl::__sync_thread_partial(8, 128);
            for (int ki_19 = 0; ki_19 < 8; ++ki_19) {
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)tmp_shared_2_1)[(((((((ki_19 >> 2) * 2048) + (((((int)threadIdx.x) & 63) >> 5) * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_19 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_19 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local_19[0])));
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)tmp_shared_2_2)[((((((((ki_19 >> 2) * 2048) + (((((int)threadIdx.x) & 127) >> 6) * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_19 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_19 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_19[0])));
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(da_fragment + 0), reinterpret_cast<const unsigned*>(A_local_19 + 0), reinterpret_cast<const unsigned*>(B_local_19 + 0));
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(da_fragment + 4), reinterpret_cast<const unsigned*>(A_local_19 + 0), reinterpret_cast<const unsigned*>(B_local_19 + 4));
            }
          }
          {
            bfloat16_t A_local_20[8];
            bfloat16_t B_local_20[32];
            #pragma unroll
            for (int i_82 = 0; i_82 < 8; ++i_82) {
              float broadcast_var_30 = 0x0p+0f/*0.000000e+00*/;
              *(float4*)(u_fragment + (i_82 * 4)) = make_float4(broadcast_var_30, broadcast_var_30, broadcast_var_30, broadcast_var_30);
            }
            for (int ki_20 = 0; ki_20 < 2; ++ki_20) {
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)tmp_shared_1_2)[((((((((int)threadIdx.x) & 63) >> 5) * 512) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki_20) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), (&(A_local_20[0])));
              #pragma unroll
              for (int i_83 = 0; i_83 < 4; ++i_83) {
                tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)tmp_shared_2_2)[((((((((((int)threadIdx.x) & 127) >> 6) * 2048) + (ki_20 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_83 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_83 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_20[(i_83 * 8)])));
              }
              for (int j_18 = 0; j_18 < 4; ++j_18) {
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(u_fragment + (j_18 * 8)), reinterpret_cast<const unsigned*>(A_local_20 + 0), reinterpret_cast<const unsigned*>(B_local_20 + (j_18 * 8)));
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(u_fragment + ((j_18 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_20 + 0), reinterpret_cast<const unsigned*>(B_local_20 + ((j_18 * 8) + 4)));
              }
            }
          }
          tl::__sync_thread_partial(8, 128);
          #pragma unroll
          for (int i_84 = 0; i_84 < 4; ++i_84) {
            tl::ptx_stmatrix_m8n8_x4((&(((bfloat16_t*)tmp_shared_2_1)[(((((((((int)threadIdx.x) >> 5) * 1024) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_84 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_84 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) - 8192)])), __pack_half2(((bfloat16_t)u_fragment[(i_84 * 8)]), ((bfloat16_t)u_fragment[((i_84 * 8) + 1)])), __pack_half2(((bfloat16_t)u_fragment[((i_84 * 8) + 2)]), ((bfloat16_t)u_fragment[((i_84 * 8) + 3)])), __pack_half2(((bfloat16_t)u_fragment[((i_84 * 8) + 4)]), ((bfloat16_t)u_fragment[((i_84 * 8) + 5)])), __pack_half2(((bfloat16_t)u_fragment[((i_84 * 8) + 6)]), ((bfloat16_t)u_fragment[((i_84 * 8) + 7)])));
          }
          bar_06[0].arrive();
          bar_06[0].wait((i_s_2 & 1));
          {
            bfloat16_t A_local_21[8];
            bfloat16_t B_local_21[8];
            #pragma unroll
            for (int i_85 = 0; i_85 < 2; ++i_85) {
              float broadcast_var_31 = 0x0p+0f/*0.000000e+00*/;
              *(float4*)(dp_fragment + (i_85 * 4)) = make_float4(broadcast_var_31, broadcast_var_31, broadcast_var_31, broadcast_var_31);
            }
            tl::__sync_thread_partial(8, 128);
            for (int ki_21 = 0; ki_21 < 8; ++ki_21) {
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)do_shared)[(((((((ki_21 >> 2) * 2048) + (((((int)threadIdx.x) & 63) >> 5) * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_21 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_21 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local_21[0])));
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)tmp_shared_2_1)[((((((((ki_21 >> 2) * 2048) + (((((int)threadIdx.x) & 127) >> 6) * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_21 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_21 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_21[0])));
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dp_fragment + 0), reinterpret_cast<const unsigned*>(A_local_21 + 0), reinterpret_cast<const unsigned*>(B_local_21 + 0));
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dp_fragment + 4), reinterpret_cast<const unsigned*>(A_local_21 + 0), reinterpret_cast<const unsigned*>(B_local_21 + 4));
            }
          }
          bar_07[0].arrive();
          bar_07[0].wait((i_s_2 & 1));
          #pragma unroll
          for (int i_86 = 0; i_86 < 2; ++i_86) {
            float4 __34;
              float4 v__46 = *(float4*)(da_fragment + (i_86 * 4));
              float4 v__47 = *(float4*)(mask_fragment + (i_86 * 4));
              *(float2*)(&(__34.x)) = tl::mul2(*(float2*)(&(v__46.x)), *(float2*)(&(v__47.x)));
              *(float2*)(&(__34.z)) = tl::mul2(*(float2*)(&(v__46.z)), *(float2*)(&(v__47.z)));
            *(float4*)(da_fragment + (i_86 * 4)) = __34;
          }
          #pragma unroll
          for (int i_87 = 0; i_87 < 4; ++i_87) {
            bfloat16_t tmp_shared_1_1_local_cast_10[2];
            *(uint1*)(tmp_shared_1_1_local_cast_10 + 0) = *(uint1*)(((bfloat16_t*)tmp_shared_1_1) + ((((((((((int)threadIdx.x) & 63) >> 5) * 512) + ((i_87 & 1) * 256)) + (((((int)threadIdx.x) & 31) >> 2) * 32)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (i_87 >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
            float2 __35;
            uint1 v__48 = *(uint1*)(tmp_shared_1_1_local_cast_10 + 0);
            ((float2*)(&__35))[0] = tl::__tl_cvt_bf16x2_to_f32x2((reinterpret_cast<__nv_bfloat162*>(&v__48))[0]);
            *(float2*)(p_fragment + (i_87 * 2)) = __35;
          }
          #pragma unroll
          for (int i_88 = 0; i_88 < 2; ++i_88) {
            float4 __36;
              float4 v__49 = *(float4*)(p_fragment + (i_88 * 4));
              float4 v__50 = *(float4*)(dp_fragment + (i_88 * 4));
              *(float2*)(&(__36.x)) = tl::mul2(*(float2*)(&(v__49.x)), *(float2*)(&(v__50.x)));
              *(float2*)(&(__36.z)) = tl::mul2(*(float2*)(&(v__49.z)), *(float2*)(&(v__50.z)));
            *(float4*)(p_fragment + (i_88 * 4)) = __36;
          }
          #pragma unroll
          for (int i_89 = 0; i_89 < 2; ++i_89) {
            float4 __37;
              float4 v__51 = *(float4*)(dp_fragment + (i_89 * 4));
              float4 v__52 = *(float4*)(mask_fragment + (i_89 * 4));
              *(float2*)(&(__37.x)) = tl::mul2(*(float2*)(&(v__51.x)), *(float2*)(&(v__52.x)));
              *(float2*)(&(__37.z)) = tl::mul2(*(float2*)(&(v__51.z)), *(float2*)(&(v__52.z)));
            *(float4*)(dp_fragment + (i_89 * 4)) = __37;
          }
          #pragma unroll
          for (int i_90 = 0; i_90 < 2; ++i_90) {
            float broadcast_var_32 = 0x1.6a09e667f3bcdp-4f/*8.838835e-02*/;
            float4 __38;
              float4 v__53 = *(float4*)(dp_fragment + (i_90 * 4));
              float4 v__54 = make_float4(broadcast_var_32, broadcast_var_32, broadcast_var_32, broadcast_var_32);
              *(float2*)(&(__38.x)) = tl::mul2(*(float2*)(&(v__53.x)), *(float2*)(&(v__54.x)));
              *(float2*)(&(__38.z)) = tl::mul2(*(float2*)(&(v__53.z)), *(float2*)(&(v__54.z)));
            *(float4*)(dp_fragment + (i_90 * 4)) = __38;
          }
          tl::__sync_thread_partial(8, 128);
          tl::ptx_stmatrix_m8n8_x4((&(((bfloat16_t*)tmp_shared_1_1)[((((((((int)threadIdx.x) & 63) >> 5) * 512) + ((((int)threadIdx.x) & 15) * 32)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), __pack_half2(((bfloat16_t)dp_fragment[0]), ((bfloat16_t)dp_fragment[1])), __pack_half2(((bfloat16_t)dp_fragment[2]), ((bfloat16_t)dp_fragment[3])), __pack_half2(((bfloat16_t)dp_fragment[4]), ((bfloat16_t)dp_fragment[5])), __pack_half2(((bfloat16_t)dp_fragment[6]), ((bfloat16_t)dp_fragment[7])));
          bar_08_1[0].arrive();
          bar_08_1[0].wait((i_s_2 & 1));
          {
            bfloat16_t A_local_22[8];
            bfloat16_t B_local_22[16];
            #pragma unroll
            for (int i_91 = 0; i_91 < 4; ++i_91) {
              float broadcast_var_33 = 0x0p+0f/*0.000000e+00*/;
              *(float4*)(dq_frag_r + (i_91 * 4)) = make_float4(broadcast_var_33, broadcast_var_33, broadcast_var_33, broadcast_var_33);
            }
            for (int ki_22 = 0; ki_22 < 8; ++ki_22) {
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)do_shared)[(((((((ki_22 >> 2) * 2048) + (((((int)threadIdx.x) & 63) >> 5) * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_22 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_22 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local_22[0])));
              #pragma unroll
              for (int i_92 = 0; i_92 < 2; ++i_92) {
                tl::ptx_ldmatrix_x4((&(((bfloat16_t*)h_shared)[(((((((((ki_22 >> 2) * 4096) + (((((int)threadIdx.x) & 127) >> 6) * 2048)) + (i_92 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_22 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_22 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_22[(i_92 * 8)])));
              }
              for (int j_19 = 0; j_19 < 2; ++j_19) {
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dq_frag_r + (j_19 * 8)), reinterpret_cast<const unsigned*>(A_local_22 + 0), reinterpret_cast<const unsigned*>(B_local_22 + (j_19 * 8)));
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dq_frag_r + ((j_19 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_22 + 0), reinterpret_cast<const unsigned*>(B_local_22 + ((j_19 * 8) + 4)));
              }
            }
          }
          bar_08_2[0].arrive();
          bar_08_3[0].wait((i_s_2 & 1));
          {
            bfloat16_t A_local_23[8];
            bfloat16_t B_local_23[16];
            #pragma unroll
            for (int i_93 = 0; i_93 < 4; ++i_93) {
              float broadcast_var_34 = 0x0p+0f/*0.000000e+00*/;
              *(float4*)(dq_frag_l + (i_93 * 4)) = make_float4(broadcast_var_34, broadcast_var_34, broadcast_var_34, broadcast_var_34);
            }
            for (int ki_23 = 0; ki_23 < 8; ++ki_23) {
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)do_shared)[(((((((ki_23 >> 2) * 2048) + (((((int)threadIdx.x) & 63) >> 5) * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_23 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_23 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local_23[0])));
              #pragma unroll
              for (int i_94 = 0; i_94 < 2; ++i_94) {
                tl::ptx_ldmatrix_x4((&(((bfloat16_t*)h_shared)[(((((((((ki_23 >> 2) * 4096) + (((((int)threadIdx.x) & 127) >> 6) * 2048)) + (i_94 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_23 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_23 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_23[(i_94 * 8)])));
              }
              for (int j_20 = 0; j_20 < 2; ++j_20) {
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dq_frag_l + (j_20 * 8)), reinterpret_cast<const unsigned*>(A_local_23 + 0), reinterpret_cast<const unsigned*>(B_local_23 + (j_20 * 8)));
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dq_frag_l + ((j_20 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_23 + 0), reinterpret_cast<const unsigned*>(B_local_23 + ((j_20 * 8) + 4)));
              }
            }
          }
          bar_09[0].arrive();
          bar_09[0].wait((i_s_2 & 1));
          #pragma unroll
          for (int i_95 = 0; i_95 < 8; ++i_95) {
            float2 __39;
              float2 v__55 = *(float2*)(dq_frag_r + (i_95 * 2));
              float2 v__56 = make_float2(g_exp_shared[(((((((int)threadIdx.x) & 63) >> 5) * 16) + ((i_95 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))], g_exp_shared[(((((((int)threadIdx.x) & 63) >> 5) * 16) + ((i_95 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))]);
              *(float2*)(&(__39.x)) = tl::mul2(*(float2*)(&(v__55.x)), *(float2*)(&(v__56.x)));
            *(float2*)(dq_frag_r + (i_95 * 2)) = __39;
            float2 __40;
              float2 v__57 = *(float2*)(dq_frag_l + (i_95 * 2));
              float2 v__58 = make_float2(g_exp_shared[(((((((int)threadIdx.x) & 63) >> 5) * 16) + ((i_95 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))], g_exp_shared[(((((((int)threadIdx.x) & 63) >> 5) * 16) + ((i_95 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))]);
              *(float2*)(&(__40.x)) = tl::mul2(*(float2*)(&(v__57.x)), *(float2*)(&(v__58.x)));
            *(float2*)(dq_frag_l + (i_95 * 2)) = __40;
          }
          #pragma unroll
          for (int i_96 = 0; i_96 < 4; ++i_96) {
            float broadcast_var_35 = 0x1.6a09e667f3bcdp-4f/*8.838835e-02*/;
            float4 __41;
              float4 v__59 = *(float4*)(dq_frag_r + (i_96 * 4));
              float4 v__60 = make_float4(broadcast_var_35, broadcast_var_35, broadcast_var_35, broadcast_var_35);
              *(float2*)(&(__41.x)) = tl::mul2(*(float2*)(&(v__59.x)), *(float2*)(&(v__60.x)));
              *(float2*)(&(__41.z)) = tl::mul2(*(float2*)(&(v__59.z)), *(float2*)(&(v__60.z)));
            *(float4*)(dq_frag_r + (i_96 * 4)) = __41;
            float broadcast_var_36 = 0x1.6a09e667f3bcdp-4f/*8.838835e-02*/;
            float4 __42;
              float4 v__61 = *(float4*)(dq_frag_l + (i_96 * 4));
              float4 v__62 = make_float4(broadcast_var_36, broadcast_var_36, broadcast_var_36, broadcast_var_36);
              *(float2*)(&(__42.x)) = tl::mul2(*(float2*)(&(v__61.x)), *(float2*)(&(v__62.x)));
              *(float2*)(&(__42.z)) = tl::mul2(*(float2*)(&(v__61.z)), *(float2*)(&(v__62.z)));
            *(float4*)(dq_frag_l + (i_96 * 4)) = __42;
          }
          tl::__sync_thread_partial(8, 128);
          #pragma unroll
          for (int i_97 = 0; i_97 < 4; ++i_97) {
            bfloat16_t tmp_shared_2_1_local_cast_11[8];
            for (int vec_2 = 0; vec_2 < 2; ++vec_2) {
              uint2 __43;
              float4 v__63 = *(float4*)(odot_fragment_2 + ((i_97 * 8) + (vec_2 * 4)));
              (reinterpret_cast<__nv_bfloat162*>(&__43))[0] = __float22bfloat162_rn(((float2*)(&v__63))[0]);
              (reinterpret_cast<__nv_bfloat162*>(&__43))[1] = __float22bfloat162_rn(((float2*)(&v__63))[1]);
              *(uint2*)(tmp_shared_2_1_local_cast_11 + (vec_2 * 4)) = __43;
            }
            *(uint4*)(((bfloat16_t*)tmp_shared_2_1) + ((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_97 * 512)) + (((((int)threadIdx.x) & 127) >> 4) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(tmp_shared_2_1_local_cast_11 + 0);
          }
          bar_10[0].arrive();
          bar_10[0].wait((i_s_2 & 1));
          tl::__sync_thread_partial(8, 128);
          #pragma unroll
          for (int i_98 = 0; i_98 < 2; ++i_98) {
            tl::ptx_stmatrix_m8n8_x4((&(((bfloat16_t*)dqkv_shared)[(((((((((int)threadIdx.x) & 63) >> 5) * 1024) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_98) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), __pack_half2(((bfloat16_t)dq_frag_l[(i_98 * 8)]), ((bfloat16_t)dq_frag_l[((i_98 * 8) + 1)])), __pack_half2(((bfloat16_t)dq_frag_l[((i_98 * 8) + 2)]), ((bfloat16_t)dq_frag_l[((i_98 * 8) + 3)])), __pack_half2(((bfloat16_t)dq_frag_l[((i_98 * 8) + 4)]), ((bfloat16_t)dq_frag_l[((i_98 * 8) + 5)])), __pack_half2(((bfloat16_t)dq_frag_l[((i_98 * 8) + 6)]), ((bfloat16_t)dq_frag_l[((i_98 * 8) + 7)])));
          }
          tl::__sync_thread_partial(8, 128);
          #pragma unroll
          for (int i_99 = 0; i_99 < 2; ++i_99) {
            tl::ptx_stmatrix_m8n8_x4((&(((bfloat16_t*)dqkv_shared)[((((((((((int)threadIdx.x) & 63) >> 5) * 1024) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_99) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 2048)])), __pack_half2(((bfloat16_t)dq_frag_r[(i_99 * 8)]), ((bfloat16_t)dq_frag_r[((i_99 * 8) + 1)])), __pack_half2(((bfloat16_t)dq_frag_r[((i_99 * 8) + 2)]), ((bfloat16_t)dq_frag_r[((i_99 * 8) + 3)])), __pack_half2(((bfloat16_t)dq_frag_r[((i_99 * 8) + 4)]), ((bfloat16_t)dq_frag_r[((i_99 * 8) + 5)])), __pack_half2(((bfloat16_t)dq_frag_r[((i_99 * 8) + 6)]), ((bfloat16_t)dq_frag_r[((i_99 * 8) + 7)])));
          }
          bar_s4_dot_a[0].arrive();
          bar_s4_dot_a[0].wait((i_s_2 & 1));
          tl::__sync_thread_partial(8, 128);
          #pragma unroll
          for (int i_100 = 0; i_100 < 4; ++i_100) {
            bfloat16_t dqkv_shared_local_cast_12[8];
            *(uint4*)(dqkv_shared_local_cast_12 + 0) = *(uint4*)(((bfloat16_t*)dqkv_shared) + ((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_100 * 512)) + (((((int)threadIdx.x) & 127) >> 4) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)));
            for (int vec_3 = 0; vec_3 < 2; ++vec_3) {
              float4 __44;
                float4 v__64 = *(float4*)(odot_fragment_2 + ((i_100 * 8) + (vec_3 * 4)));
                float4 __45;
                uint2 v__65 = *(uint2*)(dqkv_shared_local_cast_12 + (vec_3 * 4));
                ((float2*)(&__45))[0] = tl::__tl_cvt_bf16x2_to_f32x2((reinterpret_cast<__nv_bfloat162*>(&v__65))[0]);
                ((float2*)(&__45))[1] = tl::__tl_cvt_bf16x2_to_f32x2((reinterpret_cast<__nv_bfloat162*>(&v__65))[1]);
                *(float2*)(&(__44.x)) = tl::mul2(*(float2*)(&(v__64.x)), *(float2*)(&(__45.x)));
                *(float2*)(&(__44.z)) = tl::mul2(*(float2*)(&(v__64.z)), *(float2*)(&(__45.z)));
              *(float4*)(odot_fragment_2 + ((i_100 * 8) + (vec_3 * 4))) = __44;
            }
          }
          #pragma unroll
          for (int i_101 = 0; i_101 < 4; ++i_101) {
            float broadcast_var_37 = 0x0p+0f/*0.000000e+00*/;
            dg_fragment_2_pack[i_101] = make_float2(broadcast_var_37, broadcast_var_37);
            #pragma unroll
            for (int rv_2 = 0; rv_2 < 4; ++rv_2) {
              dg_fragment_2_pack[i_101] = tl::add2(dg_fragment_2_pack[i_101], *(float2*)(odot_fragment_2 + ((i_101 * 8) + (rv_2 * 2))));
            }
            dg_fragment_2[i_101] = ((dg_fragment_2_pack[i_101]).x + (dg_fragment_2_pack[i_101]).y);
            dg_fragment_2[i_101] = tl::AllReduce<tl::SumOp, 16, 1, 256, tl::NamedBarrier<128>>::run(dg_fragment_2[i_101]);
          }
          {
            bfloat16_t A_local_24[8];
            bfloat16_t B_local_24[16];
            for (int ki_24 = 0; ki_24 < 2; ++ki_24) {
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)tmp_shared_1_1)[((((((((int)threadIdx.x) & 63) >> 5) * 512) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki_24) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), (&(A_local_24[0])));
              #pragma unroll
              for (int i_102 = 0; i_102 < 2; ++i_102) {
                tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)tmp_shared_2_2)[(((((ki_24 * 1024) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_102) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_24[(i_102 * 8)])));
              }
              for (int j_21 = 0; j_21 < 2; ++j_21) {
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dq_frag_l + (j_21 * 8)), reinterpret_cast<const unsigned*>(A_local_24 + 0), reinterpret_cast<const unsigned*>(B_local_24 + (j_21 * 8)));
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dq_frag_l + ((j_21 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_24 + 0), reinterpret_cast<const unsigned*>(B_local_24 + ((j_21 * 8) + 4)));
              }
            }
          }
          {
            bfloat16_t A_local_25[8];
            bfloat16_t B_local_25[16];
            for (int ki_25 = 0; ki_25 < 2; ++ki_25) {
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)tmp_shared_1_1)[((((((((int)threadIdx.x) & 63) >> 5) * 512) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki_25) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), (&(A_local_25[0])));
              #pragma unroll
              for (int i_103 = 0; i_103 < 2; ++i_103) {
                tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)tmp_shared_2_2)[((((((ki_25 * 1024) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_103) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 2048)])), (&(B_local_25[(i_103 * 8)])));
              }
              for (int j_22 = 0; j_22 < 2; ++j_22) {
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dq_frag_r + (j_22 * 8)), reinterpret_cast<const unsigned*>(A_local_25 + 0), reinterpret_cast<const unsigned*>(B_local_25 + (j_22 * 8)));
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dq_frag_r + ((j_22 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_25 + 0), reinterpret_cast<const unsigned*>(B_local_25 + ((j_22 * 8) + 4)));
              }
            }
          }
          tl::__sync_thread_partial(8, 128);
          #pragma unroll
          for (int i_104 = 0; i_104 < 2; ++i_104) {
            tl::ptx_stmatrix_m8n8_x4((&(((bfloat16_t*)dqkv_shared)[(((((((((int)threadIdx.x) & 63) >> 5) * 1024) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_104) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), __pack_half2(((bfloat16_t)dq_frag_l[(i_104 * 8)]), ((bfloat16_t)dq_frag_l[((i_104 * 8) + 1)])), __pack_half2(((bfloat16_t)dq_frag_l[((i_104 * 8) + 2)]), ((bfloat16_t)dq_frag_l[((i_104 * 8) + 3)])), __pack_half2(((bfloat16_t)dq_frag_l[((i_104 * 8) + 4)]), ((bfloat16_t)dq_frag_l[((i_104 * 8) + 5)])), __pack_half2(((bfloat16_t)dq_frag_l[((i_104 * 8) + 6)]), ((bfloat16_t)dq_frag_l[((i_104 * 8) + 7)])));
          }
          tl::__sync_thread_partial(8, 128);
          #pragma unroll
          for (int i_105 = 0; i_105 < 2; ++i_105) {
            tl::ptx_stmatrix_m8n8_x4((&(((bfloat16_t*)dqkv_shared)[((((((((((int)threadIdx.x) & 63) >> 5) * 1024) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_105) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 2048)])), __pack_half2(((bfloat16_t)dq_frag_r[(i_105 * 8)]), ((bfloat16_t)dq_frag_r[((i_105 * 8) + 1)])), __pack_half2(((bfloat16_t)dq_frag_r[((i_105 * 8) + 2)]), ((bfloat16_t)dq_frag_r[((i_105 * 8) + 3)])), __pack_half2(((bfloat16_t)dq_frag_r[((i_105 * 8) + 4)]), ((bfloat16_t)dq_frag_r[((i_105 * 8) + 5)])), __pack_half2(((bfloat16_t)dq_frag_r[((i_105 * 8) + 6)]), ((bfloat16_t)dq_frag_r[((i_105 * 8) + 7)])));
          }
          bar_11[0].arrive();
          bar_11[0].wait((i_s_2 & 1));
          tl::__sync_thread_partial(8, 128);
          #pragma unroll
          for (int i_106 = 0; i_106 < 4; ++i_106) {
            bfloat16_t a_shared_local_cast_13[2];
            *(uint1*)(a_shared_local_cast_13 + 0) = *(uint1*)(((bfloat16_t*)a_shared) + ((((((((((int)threadIdx.x) & 63) >> 5) * 512) + ((i_106 & 1) * 256)) + (((((int)threadIdx.x) & 31) >> 2) * 32)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (i_106 >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
            float2 __46;
            uint1 v__66 = *(uint1*)(a_shared_local_cast_13 + 0);
            ((float2*)(&__46))[0] = tl::__tl_cvt_bf16x2_to_f32x2((reinterpret_cast<__nv_bfloat162*>(&v__66))[0]);
            *(float2*)(a_fragment + (i_106 * 2)) = __46;
          }
          #pragma unroll
          for (int i_107 = 0; i_107 < 2; ++i_107) {
            float4 __47;
              float4 v__67 = *(float4*)(a_fragment + (i_107 * 4));
              float4 v__68 = *(float4*)(da_fragment + (i_107 * 4));
              *(float2*)(&(__47.x)) = tl::mul2(*(float2*)(&(v__67.x)), *(float2*)(&(v__68.x)));
              *(float2*)(&(__47.z)) = tl::mul2(*(float2*)(&(v__67.z)), *(float2*)(&(v__68.z)));
            *(float4*)(a_fragment + (i_107 * 4)) = __47;
          }
          tl::__sync_thread_partial(8, 128);
          tl::ptx_stmatrix_m8n8_x4((&(((bfloat16_t*)tmp_shared_1_3)[((((((((int)threadIdx.x) & 63) >> 5) * 512) + ((((int)threadIdx.x) & 15) * 32)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), __pack_half2(((bfloat16_t)a_fragment[0]), ((bfloat16_t)a_fragment[1])), __pack_half2(((bfloat16_t)a_fragment[2]), ((bfloat16_t)a_fragment[3])), __pack_half2(((bfloat16_t)a_fragment[4]), ((bfloat16_t)a_fragment[5])), __pack_half2(((bfloat16_t)a_fragment[6]), ((bfloat16_t)a_fragment[7])));
          #pragma unroll
          for (int i_108 = 0; i_108 < 4; ++i_108) {
            float2 __48;
              float2 v__69 = *(float2*)(a_fragment + (i_108 * 2));
              float2 v__70 = *(float2*)(b_shared + (((((((int)threadIdx.x) >> 6) * 16) + ((i_108 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 64));
              *(float2*)(&(__48.x)) = tl::mul2(*(float2*)(&(v__69.x)), *(float2*)(&(v__70.x)));
            *(float2*)(a_fragment + (i_108 * 2)) = __48;
          }
          #pragma unroll
          for (int i_109 = 0; i_109 < 2; ++i_109) {
            float4 __49;
              float4 v__71 = *(float4*)(a_fragment + (i_109 * 4));
              float4 v__72 = *(float4*)(p_fragment + (i_109 * 4));
              *(float2*)(&(__49.x)) = tl::add2(*(float2*)(&(v__71.x)), *(float2*)(&(v__72.x)));
              *(float2*)(&(__49.z)) = tl::add2(*(float2*)(&(v__71.z)), *(float2*)(&(v__72.z)));
            *(float4*)(a_fragment + (i_109 * 4)) = __49;
          }
          #pragma unroll
          for (int i_110 = 0; i_110 < 8; ++i_110) {
            float v__73 = a_fragment[i_110];
            uint x = (*(uint *)(&(v__73)));
            lo_fragment[i_110] = ((ushort)(x & (uint)65535));
            hi_fragment[i_110] = ((ushort)(x >> (uint)16));
          }
          tl::__sync_thread_partial(8, 128);
          #pragma unroll
          for (int i_111 = 0; i_111 < 4; ++i_111) {
            for (int j_t_vec = 0; j_t_vec < 2; ++j_t_vec) {
              ushort v__74 = hi_fragment[((((i_111 & 1) * 4) + ((i_111 >> 1) * 2)) + j_t_vec)];
              ((bfloat16_t*)tmp_shared_1_2)[(((((((((((int)threadIdx.x) & 63) >> 5) * 512) + ((i_111 >> 1) * 256)) + (((((int)threadIdx.x) & 31) >> 2) * 32)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (i_111 & 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + j_t_vec)] = (*(bfloat16_t *)(&(v__74)));
            }
          }
          tl::__sync_thread_partial(8, 128);
          #pragma unroll
          for (int i_112 = 0; i_112 < 4; ++i_112) {
            for (int j_t_vec_1 = 0; j_t_vec_1 < 2; ++j_t_vec_1) {
              bfloat16_t v__75 = ((bfloat16_t*)tmp_shared_1_2)[(((((((((((int)threadIdx.x) >> 6) * 512) + ((i_112 & 1) * 256)) + ((((int)threadIdx.x) & 3) * 64)) + (j_t_vec_1 * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + ((((i_112 >> 1) + (((int)threadIdx.x) & 1)) & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) - 2048)];
              hi_fragment[((((i_112 & 1) * 4) + ((i_112 >> 1) * 2)) + j_t_vec_1)] = (*(ushort *)(&(v__75)));
            }
          }
          tl::__sync_thread_partial(8, 128);
          #pragma unroll
          for (int i_113 = 0; i_113 < 4; ++i_113) {
            for (int j_t_vec_2 = 0; j_t_vec_2 < 2; ++j_t_vec_2) {
              ushort v__76 = lo_fragment[((((i_113 & 1) * 4) + ((i_113 >> 1) * 2)) + j_t_vec_2)];
              ((bfloat16_t*)tmp_shared_1_2)[(((((((((((int)threadIdx.x) & 63) >> 5) * 512) + ((i_113 >> 1) * 256)) + (((((int)threadIdx.x) & 31) >> 2) * 32)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (i_113 & 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + j_t_vec_2)] = (*(bfloat16_t *)(&(v__76)));
            }
          }
          tl::__sync_thread_partial(8, 128);
          #pragma unroll
          for (int i_114 = 0; i_114 < 4; ++i_114) {
            for (int j_t_vec_3 = 0; j_t_vec_3 < 2; ++j_t_vec_3) {
              bfloat16_t v__77 = ((bfloat16_t*)tmp_shared_1_2)[(((((((((((int)threadIdx.x) >> 6) * 512) + ((i_114 & 1) * 256)) + ((((int)threadIdx.x) & 3) * 64)) + (j_t_vec_3 * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + ((((i_114 >> 1) + (((int)threadIdx.x) & 1)) & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) - 2048)];
              lo_fragment[((((i_114 & 1) * 4) + ((i_114 >> 1) * 2)) + j_t_vec_3)] = (*(ushort *)(&(v__77)));
            }
          }
          #pragma unroll
          for (int i_115 = 0; i_115 < 8; ++i_115) {
            uint32_fragment[i_115] = ((uint)((hi_fragment[i_115] << (ushort)16) + lo_fragment[i_115]));
            uint v__78 = uint32_fragment[i_115];
            p_fragment[i_115] = (*(float *)(&(v__78)));
          }
          #pragma unroll
          for (int i_116 = 0; i_116 < 2; ++i_116) {
            float4 __50;
              float4 v__79 = *(float4*)(a_fragment + (i_116 * 4));
              float4 v__80 = *(float4*)(p_fragment + (i_116 * 4));
              *(float2*)(&(__50.x)) = tl::sub2(*(float2*)(&(v__79.x)), *(float2*)(&(v__80.x)));
              *(float2*)(&(__50.z)) = tl::sub2(*(float2*)(&(v__79.z)), *(float2*)(&(v__80.z)));
            *(float4*)(a_fragment + (i_116 * 4)) = __50;
          }
          #pragma unroll
          for (int i_117 = 0; i_117 < 2; ++i_117) {
            dg_fragment_2b[i_117] = 0x0p+0f/*0.000000e+00*/;
            #pragma unroll
            for (int rv_3 = 0; rv_3 < 4; ++rv_3) {
              dg_fragment_2b[i_117] = (dg_fragment_2b[i_117] + a_fragment[((((rv_3 & 1) * 4) + (i_117 * 2)) + (rv_3 >> 1))]);
            }
            tl::__sync_thread_partial(8, 128);
            dg_fragment_2b[i_117] = tl::AllReduce<tl::SumOp, 128, 64, 256, tl::NamedBarrier<128>>::run(dg_fragment_2b[i_117], (&(((float*)workspace_5)[0])));
            dg_fragment_2b[i_117] = tl::AllReduce<tl::SumOp, 4, 1, 256, tl::NamedBarrier<128>>::run(dg_fragment_2b[i_117]);
          }
          tl::__sync_thread_partial(5, 128);
          if ((((int)threadIdx.x) % 16) == 0) {
            #pragma unroll
            for (int i_118 = 0; i_118 < 4; ++i_118) {
              dg_shared[(((i_118 * 8) + (((int)threadIdx.x) >> 4)) - 16)] = (dg_shared[(((i_118 * 8) + (((int)threadIdx.x) >> 4)) - 16)] + dg_fragment_2[i_118]);
            }
          }
          bar_dg_a_first[0].arrive();
          bar_dg_a_first[0].wait((i_s_2 & 1));
          tl::__sync_thread_partial(5, 128);
          if ((((((int)threadIdx.x) & 3) * 2) + (((int)threadIdx.x) >> 6)) == 4) {
            #pragma unroll
            for (int i_119 = 0; i_119 < 2; ++i_119) {
              dg_shared[(((((((int)threadIdx.x) & 63) >> 5) * 16) + (i_119 * 8)) + ((((int)threadIdx.x) & 31) >> 2))] = (dg_shared[(((((((int)threadIdx.x) & 63) >> 5) * 16) + (i_119 * 8)) + ((((int)threadIdx.x) & 31) >> 2))] + dg_fragment_2b[i_119]);
            }
          }
          #pragma unroll
          for (int i_120 = 0; i_120 < 8; ++i_120) {
            a_fragment[i_120] = ((float)((bfloat16_t*)tmp_shared_1_3)[(((((((((((int)threadIdx.x) >> 6) * 512) + ((i_120 >> 2) * 256)) + ((((int)threadIdx.x) & 3) * 64)) + ((i_120 & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((i_120 & 3) >> 1) + (((int)threadIdx.x) & 1)) & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) - 2048)]);
          }
          #pragma unroll
          for (int i_121 = 0; i_121 < 2; ++i_121) {
            db_fragment[i_121] = 0x0p+0f/*0.000000e+00*/;
            #pragma unroll
            for (int rv_4 = 0; rv_4 < 4; ++rv_4) {
              db_fragment[i_121] = (db_fragment[i_121] + a_fragment[((((rv_4 & 1) * 4) + (i_121 * 2)) + (rv_4 >> 1))]);
            }
            tl::__sync_thread_partial(8, 128);
            db_fragment[i_121] = tl::AllReduce<tl::SumOp, 128, 64, 256, tl::NamedBarrier<128>>::run(db_fragment[i_121], (&(((float*)workspace_6)[0])));
            db_fragment[i_121] = tl::AllReduce<tl::SumOp, 4, 1, 256, tl::NamedBarrier<128>>::run(db_fragment[i_121]);
          }
          #pragma unroll
          for (int i_122 = 0; i_122 < 4; ++i_122) {
            float2 __51;
              float2 v__81 = *(float2*)(da_fragment + (i_122 * 2));
              float2 v__82 = *(float2*)(b_shared + (((((((int)threadIdx.x) >> 6) * 16) + ((i_122 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 64));
              *(float2*)(&(__51.x)) = tl::mul2(*(float2*)(&(v__81.x)), *(float2*)(&(v__82.x)));
            *(float2*)(da_fragment + (i_122 * 2)) = __51;
          }
          tl::ptx_stmatrix_m8n8_x4((&(((bfloat16_t*)tmp_shared_1_2)[((((((((int)threadIdx.x) & 63) >> 5) * 512) + ((((int)threadIdx.x) & 15) * 32)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), __pack_half2(((bfloat16_t)da_fragment[0]), ((bfloat16_t)da_fragment[1])), __pack_half2(((bfloat16_t)da_fragment[2]), ((bfloat16_t)da_fragment[3])), __pack_half2(((bfloat16_t)da_fragment[4]), ((bfloat16_t)da_fragment[5])), __pack_half2(((bfloat16_t)da_fragment[6]), ((bfloat16_t)da_fragment[7])));
          bar_13[0].arrive();
          bar_13[0].wait((i_s_2 & 1));
          {
            bfloat16_t A_local_26[8];
            bfloat16_t B_local_26[8];
            #pragma unroll
            for (int i_123 = 0; i_123 < 2; ++i_123) {
              float broadcast_var_38 = 0x0p+0f/*0.000000e+00*/;
              *(float4*)(da_fragment + (i_123 * 4)) = make_float4(broadcast_var_38, broadcast_var_38, broadcast_var_38, broadcast_var_38);
            }
            tl::__sync_thread_partial(8, 128);
            for (int ki_26 = 0; ki_26 < 2; ++ki_26) {
              tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)a_shared)[(((((ki_26 * 512) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 7) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), (&(A_local_26[0])));
              tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)tmp_shared_1_2)[((((ki_26 * 512) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), (&(B_local_26[0])));
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(da_fragment + 0), reinterpret_cast<const unsigned*>(A_local_26 + 0), reinterpret_cast<const unsigned*>(B_local_26 + 0));
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(da_fragment + 4), reinterpret_cast<const unsigned*>(A_local_26 + 0), reinterpret_cast<const unsigned*>(B_local_26 + 4));
            }
          }
          tl::__sync_thread_partial(8, 128);
          tl::ptx_stmatrix_m8n8_x4((&(((bfloat16_t*)tmp_shared_1_2)[((((((((int)threadIdx.x) & 63) >> 5) * 512) + ((((int)threadIdx.x) & 15) * 32)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), __pack_half2(((bfloat16_t)da_fragment[0]), ((bfloat16_t)da_fragment[1])), __pack_half2(((bfloat16_t)da_fragment[2]), ((bfloat16_t)da_fragment[3])), __pack_half2(((bfloat16_t)da_fragment[4]), ((bfloat16_t)da_fragment[5])), __pack_half2(((bfloat16_t)da_fragment[6]), ((bfloat16_t)da_fragment[7])));
          {
            bfloat16_t A_local_27[8];
            bfloat16_t B_local_27[8];
            #pragma unroll
            for (int i_124 = 0; i_124 < 2; ++i_124) {
              float broadcast_var_39 = 0x0p+0f/*0.000000e+00*/;
              *(float4*)(da_fragment + (i_124 * 4)) = make_float4(broadcast_var_39, broadcast_var_39, broadcast_var_39, broadcast_var_39);
            }
            tl::__sync_thread_partial(8, 128);
            for (int ki_27 = 0; ki_27 < 2; ++ki_27) {
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)tmp_shared_1_2)[((((((((int)threadIdx.x) & 63) >> 5) * 512) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki_27) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), (&(A_local_27[0])));
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)a_shared)[(((((((((int)threadIdx.x) & 127) >> 6) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 7) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki_27) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), (&(B_local_27[0])));
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(da_fragment + 0), reinterpret_cast<const unsigned*>(A_local_27 + 0), reinterpret_cast<const unsigned*>(B_local_27 + 0));
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(da_fragment + 4), reinterpret_cast<const unsigned*>(A_local_27 + 0), reinterpret_cast<const unsigned*>(B_local_27 + 4));
            }
          }
          {
            bfloat16_t A_local_28[8];
            bfloat16_t B_local_28[8];
            #pragma unroll
            for (int i_125 = 0; i_125 < 2; ++i_125) {
              float broadcast_var_40 = 0x0p+0f/*0.000000e+00*/;
              *(float4*)(a_fragment + (i_125 * 4)) = make_float4(broadcast_var_40, broadcast_var_40, broadcast_var_40, broadcast_var_40);
            }
            for (int ki_28 = 0; ki_28 < 8; ++ki_28) {
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)tmp_shared_2_2)[(((((((ki_28 >> 2) * 2048) + (((((int)threadIdx.x) & 63) >> 5) * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_28 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_28 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local_28[0])));
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)tmp_shared_2_2)[((((((((ki_28 >> 2) * 2048) + (((((int)threadIdx.x) & 127) >> 6) * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_28 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_28 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_28[0])));
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(a_fragment + 0), reinterpret_cast<const unsigned*>(A_local_28 + 0), reinterpret_cast<const unsigned*>(B_local_28 + 0));
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(a_fragment + 4), reinterpret_cast<const unsigned*>(A_local_28 + 0), reinterpret_cast<const unsigned*>(B_local_28 + 4));
            }
          }
          bar_14[0].arrive();
          bar_14[0].wait((i_s_2 & 1));
          #pragma unroll
          for (int i_126 = 0; i_126 < 8; ++i_126) {
            if (((((((((int)threadIdx.x) & 63) >> 5) * 16) + (((i_126 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + 64) <= (((((((int)threadIdx.x) >> 6) * 16) + ((i_126 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_126 & 1))) {
              da_fragment[i_126] = 0x0p+0f/*0.000000e+00*/;
            } else {
              da_fragment[i_126] = (da_fragment[i_126] * -0x1p+0f/*-1.000000e+00*/);
            }
          }
          #pragma unroll
          for (int i_127 = 0; i_127 < 2; ++i_127) {
            float4 __52;
              float4 v__83 = *(float4*)(a_fragment + (i_127 * 4));
              float4 v__84 = *(float4*)(da_fragment + (i_127 * 4));
              *(float2*)(&(__52.x)) = tl::mul2(*(float2*)(&(v__83.x)), *(float2*)(&(v__84.x)));
              *(float2*)(&(__52.z)) = tl::mul2(*(float2*)(&(v__83.z)), *(float2*)(&(v__84.z)));
            *(float4*)(a_fragment + (i_127 * 4)) = __52;
          }
          #pragma unroll
          for (int i_128 = 0; i_128 < 2; ++i_128) {
            db_fragment_clear[i_128] = 0x0p+0f/*0.000000e+00*/;
            #pragma unroll
            for (int rv_5 = 0; rv_5 < 4; ++rv_5) {
              db_fragment_clear[i_128] = (db_fragment_clear[i_128] + a_fragment[((((rv_5 & 1) * 4) + (i_128 * 2)) + (rv_5 >> 1))]);
            }
            tl::__sync_thread_partial(8, 128);
            db_fragment_clear[i_128] = tl::AllReduce<tl::SumOp, 128, 64, 256, tl::NamedBarrier<128>>::run(db_fragment_clear[i_128], (&(((float*)workspace_1)[0])));
            db_fragment_clear[i_128] = tl::AllReduce<tl::SumOp, 4, 1, 256, tl::NamedBarrier<128>>::run(db_fragment_clear[i_128]);
            db_fragment[i_128] = (db_fragment[i_128] + db_fragment_clear[i_128]);
          }
          if ((((((int)threadIdx.x) & 3) * 2) + (((int)threadIdx.x) >> 6)) == 4) {
            #pragma unroll
            for (int i_129 = 0; i_129 < 2; ++i_129) {
              db_shared[(((((((int)threadIdx.x) & 63) >> 5) * 16) + (i_129 * 8)) + ((((int)threadIdx.x) & 31) >> 2))] = db_fragment[i_129];
            }
          }
          #pragma unroll
          for (int i_130 = 0; i_130 < 4; ++i_130) {
            float2 __53;
              float2 v__85 = *(float2*)(da_fragment + (i_130 * 2));
              float2 v__86 = make_float2(b_shared[(((((((int)threadIdx.x) & 63) >> 5) * 16) + ((i_130 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))], b_shared[(((((((int)threadIdx.x) & 63) >> 5) * 16) + ((i_130 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))]);
              *(float2*)(&(__53.x)) = tl::mul2(*(float2*)(&(v__85.x)), *(float2*)(&(v__86.x)));
            *(float2*)(da_fragment + (i_130 * 2)) = __53;
          }
          tl::ptx_stmatrix_m8n8_x4((&(((bfloat16_t*)tmp_shared_1_2)[((((((((int)threadIdx.x) & 63) >> 5) * 512) + ((((int)threadIdx.x) & 15) * 32)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), __pack_half2(((bfloat16_t)da_fragment[0]), ((bfloat16_t)da_fragment[1])), __pack_half2(((bfloat16_t)da_fragment[2]), ((bfloat16_t)da_fragment[3])), __pack_half2(((bfloat16_t)da_fragment[4]), ((bfloat16_t)da_fragment[5])), __pack_half2(((bfloat16_t)da_fragment[6]), ((bfloat16_t)da_fragment[7])));
          tl::__sync_thread_partial(8, 128);
          #pragma unroll
          for (int i_131 = 0; i_131 < 8; ++i_131) {
            da_fragment[i_131] = (da_fragment[i_131] + ((float)((bfloat16_t*)tmp_shared_1_2)[(((((((((((int)threadIdx.x) >> 6) * 512) + ((i_131 >> 2) * 256)) + ((((int)threadIdx.x) & 3) * 64)) + ((i_131 & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((i_131 & 3) >> 1) + (((int)threadIdx.x) & 1)) & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) - 2048)]));
          }
          tl::__sync_thread_partial(8, 128);
          tl::ptx_stmatrix_m8n8_x4((&(((bfloat16_t*)tmp_shared_1_2)[((((((((int)threadIdx.x) & 63) >> 5) * 512) + ((((int)threadIdx.x) & 15) * 32)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), __pack_half2(((bfloat16_t)da_fragment[0]), ((bfloat16_t)da_fragment[1])), __pack_half2(((bfloat16_t)da_fragment[2]), ((bfloat16_t)da_fragment[3])), __pack_half2(((bfloat16_t)da_fragment[4]), ((bfloat16_t)da_fragment[5])), __pack_half2(((bfloat16_t)da_fragment[6]), ((bfloat16_t)da_fragment[7])));
          bar_15[0].arrive();
          bar_15[0].wait((i_s_2 & 1));
        }
      } else {
        tl::warpgroup_reg_dealloc<24>();
        if (((int)threadIdx.x) < 416) {
          for (int i_s_3 = 0; i_s_3 < (num_iters - 1); ++i_s_3) {
            int chunk_idx = ((num_iters - i_s_3) - 2);
            int left = ((chunk_idx * 32) + seq_start_idx);
            bar_00[0].arrive();
            bar_00[0].wait((i_s_3 & 1));
            bar_03[0].wait((i_s_3 & 1));
            float condval_11;
            if (((((384 <= (left + ((int)threadIdx.x))) && ((left + ((int)threadIdx.x)) < (num_tokens + 384))) && (0 <= batch_idx)) && (batch_idx < batch_size))) {
              condval_11 = g[(((((((int64_t)left) * (int64_t)4) + (((int64_t)((int)threadIdx.x)) * (int64_t)4)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)4)) + (((int64_t)((int)blockIdx.x)) & (int64_t)3)) - (int64_t)1536)];
            } else {
              condval_11 = 0x0p+0f/*0.000000e+00*/;
            }
            g_shared[(((int)threadIdx.x) - 384)] = condval_11;
            bar_07[0].wait((i_s_3 & 1));
            if (tl::tl_shuffle_elect<32>()) {
              bar_00[0].expect_transaction(8192);
              tl::fence_proxy_async();
              #pragma unroll
              for (int i_132 = 0; i_132 < 2; ++i_132) {
                tl::tma_load(k_desc, bar_00[0], (&(((bfloat16_t*)k_shared)[(i_132 * 2048)])), (i_132 * 64), left, ((((int)blockIdx.x) & 3) >> 1), batch_idx);
              }
            }
            bar_15[0].wait((i_s_3 & 1));
            if (tl::tl_shuffle_elect<32>()) {
              bar_00[0].expect_transaction(8192);
              tl::fence_proxy_async();
              #pragma unroll
              for (int i_133 = 0; i_133 < 2; ++i_133) {
                tl::tma_load(q_desc, bar_00[0], (&(((bfloat16_t*)tmp_shared_2_1)[(i_133 * 2048)])), (i_133 * 64), left, ((((int)blockIdx.x) & 3) >> 1), batch_idx);
              }
            }
          }
          if (0 < num_iters) {
            bar_00[0].arrive();
          }
        } else {
          if (((int)threadIdx.x) < 448) {
            tl::fence_proxy_async();
            for (int i_s_4 = 0; i_s_4 < num_iters; ++i_s_4) {
              int left_1 = ((((num_iters * 32) + seq_start_idx) - (i_s_4 * 32)) - 32);
              bar_00[0].arrive();
              bar_00[0].wait((i_s_4 & 1));
              bar_01[0].wait((i_s_4 & 1));
              if (i_s_4 == 1) {
                if (0 <= batch_idx) {
                  #pragma unroll
                  for (int i_134 = 0; i_134 < 16; ++i_134) {
                    if (((((i_134 * 2) + (((int)threadIdx.x) >> 4)) + left_1) + 6) < seq_end_idx) {
                      if (-6 <= (((i_134 * 2) + (((int)threadIdx.x) >> 4)) + left_1)) {
                        if (((((i_134 * 2) + (((int)threadIdx.x) >> 4)) + left_1) + 6) < num_tokens) {
                          if (batch_idx < batch_size) {
                            *(uint4*)(dk + (((((((((int64_t)i_134) * (int64_t)1024) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)512)) + (((int64_t)left_1) * (int64_t)512)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)512)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)3) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)) + (int64_t)3072)) = *(uint4*)(((bfloat16_t*)dqkv_shared) + ((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + ((i_134 >> 2) * 512)) + (((((i_134 * 2) + (((int)threadIdx.x) >> 4)) + 6) & 7) * 64)) + ((((((((i_134 * 256) + (((int)threadIdx.x) * 8)) + 768) & 1023) >> 9) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_134 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)));
                          }
                        }
                      }
                    }
                  }
                }
              } else {
                if (1 < i_s_4) {
                  if (tl::tl_shuffle_elect<32>()) {
                    #pragma unroll
                    for (int i_135 = 0; i_135 < 2; ++i_135) {
                      tl::tma_store(dk_desc, (&(((bfloat16_t*)dqkv_shared)[(i_135 * 2048)])), (i_135 * 64), (left_1 + 32), (((int)blockIdx.x) & 3), batch_idx);
                    }
                    tl::tma_store_arrive();
                    tl::tma_store_wait<0, true>();
                  }
                }
              }
              bar_04[0].arrive();
              bar_04[0].wait((i_s_4 & 1));
              bar_05[0].wait((i_s_4 & 1));
              if (i_s_4 == 0) {
                if (0 <= batch_idx) {
                  #pragma unroll
                  for (int i_136 = 0; i_136 < 16; ++i_136) {
                    if ((((i_136 * 2) + (((int)threadIdx.x) >> 4)) + left_1) < (seq_end_idx + 26)) {
                      if (13 <= ((((((int)threadIdx.x) >> 4) + left_1) >> 1) + i_136)) {
                        if ((((i_136 * 2) + (((int)threadIdx.x) >> 4)) + left_1) < (num_tokens + 26)) {
                          if (batch_idx < batch_size) {
                            *(uint4*)(dv + (((((((((int64_t)i_136) * (int64_t)1024) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)512)) + (((int64_t)left_1) * (int64_t)512)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)512)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)3) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)) - (int64_t)13312)) = *(uint4*)(((bfloat16_t*)dqkv_shared) + ((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + ((i_136 >> 2) * 512)) + (((((i_136 * 2) + (((int)threadIdx.x) >> 4)) + 6) & 7) * 64)) + ((((((((i_136 * 256) + (((int)threadIdx.x) * 8)) + 768) & 1023) >> 9) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_136 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)));
                          }
                        }
                      }
                    }
                  }
                }
              } else {
                if (tl::tl_shuffle_elect<32>()) {
                  #pragma unroll
                  for (int i_137 = 0; i_137 < 2; ++i_137) {
                    tl::tma_store(dv_desc, (&(((bfloat16_t*)dqkv_shared)[(i_137 * 2048)])), (i_137 * 64), left_1, (((int)blockIdx.x) & 3), batch_idx);
                  }
                  tl::tma_store_arrive();
                  tl::tma_store_wait<0, true>();
                }
              }
              bar_10[0].arrive();
              bar_10[0].wait((i_s_4 & 1));
              bar_11[0].wait((i_s_4 & 1));
              if (i_s_4 == 0) {
                if (0 <= batch_idx) {
                  #pragma unroll
                  for (int i_138 = 0; i_138 < 16; ++i_138) {
                    if ((((i_138 * 2) + (((int)threadIdx.x) >> 4)) + left_1) < (seq_end_idx + 26)) {
                      if (13 <= ((((((int)threadIdx.x) >> 4) + left_1) >> 1) + i_138)) {
                        if ((((i_138 * 2) + (((int)threadIdx.x) >> 4)) + left_1) < (num_tokens + 26)) {
                          if (batch_idx < batch_size) {
                            *(uint4*)(dq + (((((((((int64_t)i_138) * (int64_t)1024) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)512)) + (((int64_t)left_1) * (int64_t)512)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)512)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)3) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)) - (int64_t)13312)) = *(uint4*)(((bfloat16_t*)dqkv_shared) + ((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + ((i_138 >> 2) * 512)) + (((((i_138 * 2) + (((int)threadIdx.x) >> 4)) + 6) & 7) * 64)) + ((((((((i_138 * 256) + (((int)threadIdx.x) * 8)) + 768) & 1023) >> 9) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_138 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)));
                          }
                        }
                      }
                    }
                  }
                }
              } else {
                if (tl::tl_shuffle_elect<32>()) {
                  #pragma unroll
                  for (int i_139 = 0; i_139 < 2; ++i_139) {
                    tl::tma_store(dq_desc, (&(((bfloat16_t*)dqkv_shared)[(i_139 * 2048)])), (i_139 * 64), left_1, (((int)blockIdx.x) & 3), batch_idx);
                  }
                  tl::tma_store_arrive();
                  tl::tma_store_wait<0, true>();
                }
              }
            }
          } else {
            if (((int)threadIdx.x) < 480) {
              for (int i_s_5 = 0; i_s_5 < (num_iters - 1); ++i_s_5) {
                int chunk_idx_1 = ((num_iters - i_s_5) - 2);
                int left_2 = ((chunk_idx_1 * 32) + seq_start_idx);
                bar_02[0].arrive();
                bar_02[0].wait((i_s_5 & 1));
                bar_10[0].wait((i_s_5 & 1));
                if (tl::tl_shuffle_elect<32>()) {
                  bar_02[0].expect_transaction(16384);
                  tl::fence_proxy_async();
                  #pragma unroll
                  for (int i_140 = 0; i_140 < 2; ++i_140) {
                    tl::tma_load(h_desc, bar_02[0], (&(((bfloat16_t*)h_shared)[(i_140 * 4096)])), (i_140 * 64), 0, (((int)blockIdx.x) & 3), (chunk_start_idx + chunk_idx_1), batch_idx);
                  }
                }
                bar_14[0].wait((i_s_5 & 1));
                if (tl::tl_shuffle_elect<32>()) {
                  bar_02[0].expect_transaction(2048);
                  tl::fence_proxy_async();
                  tl::tma_load(a_desc, bar_02[0], (&(((bfloat16_t*)a_shared)[0])), 0, left_2, (((int)blockIdx.x) & 3), batch_idx);
                  bar_02[0].expect_transaction(8192);
                  #pragma unroll
                  for (int i_141 = 0; i_141 < 2; ++i_141) {
                    tl::tma_load(do_desc, bar_02[0], (&(((bfloat16_t*)do_shared)[(i_141 * 2048)])), (i_141 * 64), left_2, (((int)blockIdx.x) & 3), batch_idx);
                  }
                }
                bar_15[0].wait((i_s_5 & 1));
                float condval_12;
                if (((((448 <= (left_2 + ((int)threadIdx.x))) && ((left_2 + ((int)threadIdx.x)) < (num_tokens + 448))) && (0 <= batch_idx)) && (batch_idx < batch_size))) {
                  condval_12 = b[(((((((int64_t)left_2) * (int64_t)4) + (((int64_t)((int)threadIdx.x)) * (int64_t)4)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)4)) + (((int64_t)((int)blockIdx.x)) & (int64_t)3)) - (int64_t)1792)];
                } else {
                  condval_12 = 0x0p+0f/*0.000000e+00*/;
                }
                b_shared[(((int)threadIdx.x) - 448)] = condval_12;
              }
              if (0 < num_iters) {
                bar_00[0].wait(((num_iters + 1) & 1));
              }
              if (0 < num_iters) {
                bar_02[0].arrive();
              }
            } else {
              for (int i_s_6 = 0; i_s_6 < num_iters; ++i_s_6) {
                int left_3 = ((((num_iters * 32) + seq_start_idx) - (i_s_6 * 32)) - 32);
                bar_05[0].arrive();
                bar_05[0].wait((i_s_6 & 1));
                bar_15[0].wait((i_s_6 & 1));
                if (i_s_6 == 0) {
                  if ((left_3 + ((int)threadIdx.x)) < (seq_end_idx + 480)) {
                    if (480 <= (left_3 + ((int)threadIdx.x))) {
                      if ((left_3 + ((int)threadIdx.x)) < (num_tokens + 480)) {
                        if (0 <= batch_idx) {
                          if (batch_idx < batch_size) {
                            dg[(((((((int64_t)left_3) * (int64_t)4) + (((int64_t)((int)threadIdx.x)) * (int64_t)4)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)4)) + (((int64_t)((int)blockIdx.x)) & (int64_t)3)) - (int64_t)1920)] = dg_shared[(((int)threadIdx.x) - 480)];
                          }
                        }
                      }
                    }
                  }
                  if (0 < ((seq_end_idx - seq_start_idx) & 31)) {
                    if (1 <= seq_end_idx) {
                      if (seq_end_idx <= num_tokens) {
                        if (0 <= batch_idx) {
                          if (batch_idx < batch_size) {
                            float condval_13;
                            if (((((1 <= seq_end_idx) && (seq_end_idx <= num_tokens)) && (0 <= batch_idx)) && (batch_idx < batch_size))) {
                              condval_13 = dg[((((((int64_t)seq_end_idx) * (int64_t)4) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)4)) + (((int64_t)((int)blockIdx.x)) & (int64_t)3)) - (int64_t)4)];
                            } else {
                              condval_13 = 0x0p+0f/*0.000000e+00*/;
                            }
                            dg[((((((int64_t)seq_end_idx) * (int64_t)4) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)4)) + (((int64_t)((int)blockIdx.x)) & (int64_t)3)) - (int64_t)4)] = (condval_13 + dg_shared[31]);
                          }
                        }
                      }
                    }
                  }
                } else {
                  if (480 <= (left_3 + ((int)threadIdx.x))) {
                    if ((left_3 + ((int)threadIdx.x)) < (num_tokens + 480)) {
                      if (0 <= batch_idx) {
                        if (batch_idx < batch_size) {
                          dg[(((((((int64_t)left_3) * (int64_t)4) + (((int64_t)((int)threadIdx.x)) * (int64_t)4)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)4)) + (((int64_t)((int)blockIdx.x)) & (int64_t)3)) - (int64_t)1920)] = dg_shared[(((int64_t)((int)threadIdx.x)) - (int64_t)480)];
                        }
                      }
                    }
                  }
                }
                if (i_s_6 == 0) {
                  if ((left_3 + ((int)threadIdx.x)) < (seq_end_idx + 480)) {
                    if (480 <= (left_3 + ((int)threadIdx.x))) {
                      if ((left_3 + ((int)threadIdx.x)) < (num_tokens + 480)) {
                        if (0 <= batch_idx) {
                          if (batch_idx < batch_size) {
                            db[(((((((int64_t)left_3) * (int64_t)4) + (((int64_t)((int)threadIdx.x)) * (int64_t)4)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)4)) + (((int64_t)((int)blockIdx.x)) & (int64_t)3)) - (int64_t)1920)] = db_shared[(((int)threadIdx.x) - 480)];
                          }
                        }
                      }
                    }
                  }
                } else {
                  if (480 <= (left_3 + ((int)threadIdx.x))) {
                    if ((left_3 + ((int)threadIdx.x)) < (num_tokens + 480)) {
                      if (0 <= batch_idx) {
                        if (batch_idx < batch_size) {
                          db[(((((((int64_t)left_3) * (int64_t)4) + (((int64_t)((int)threadIdx.x)) * (int64_t)4)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)4)) + (((int64_t)((int)blockIdx.x)) & (int64_t)3)) - (int64_t)1920)] = db_shared[(((int64_t)((int)threadIdx.x)) - (int64_t)480)];
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
}

