// tilelang target: {"kind":"c","tag":"","keys":["cpu"]}
#define TVM_EXPORTS
#include "tvm/runtime/base.h"
#include "tvm/runtime/c_backend_api.h"
#include "tvm/ffi/c_api.h"
#include <math.h>
#include <stdio.h>
#include <stdbool.h>
#if defined(_MSC_VER)
#define TL_ALIGN(N) __declspec(align(N))
#else
#define TL_ALIGN(N) __attribute__((aligned(N)))
#endif
#ifdef __OBJC__
#include "tvm/runtime/device_api.h"
#include "tvm/ffi/function.h"
#include <Metal/Metal.h>
#include <Foundation/Foundation.h>
#include <torch/mps.h>
#endif
void* __tvm_ffi__library_ctx = NULL;
static void* __tvm_set_device_packed = NULL;
static void* __tvm_tensormap_create_tiled_packed = NULL;
static void* tilelang_fused_chunk_gdr_fwd_kernel_kernel_packed = NULL;
#ifdef __cplusplus
extern "C"
#endif
int32_t __tvm_ffi_tilelang_fused_chunk_gdr_fwd_kernel(void* self_handle, void* args, int32_t num_args, void* result);
#ifdef __cplusplus
extern "C"
#endif
int32_t __tvm_ffi_tilelang_fused_chunk_gdr_fwd_kernel(void* self_handle, void* args, int32_t num_args, void* result) {
  TL_ALIGN(128) TVMFFIAny stack[25];
  void* stack_ffi_any = stack;
  if (!((num_args == 14))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "tilelang_fused_chunk_gdr_fwd_kernel: num_args should be 14", (long long)(num_args), (long long)(14));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(!(args == NULL))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "tilelang_fused_chunk_gdr_fwd_kernel: args pointer is NULL");
    return -1;
  }
  int32_t q_handle_type_index = (((TVMFFIAny*)args)[0].type_index);
  if (!(((((q_handle_type_index == 0) || (q_handle_type_index == 4)) || (q_handle_type_index == 7)) || (64 <= q_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input q expected pointer or tensor handle");
    return -1;
  }
  int32_t k_handle_type_index = (((TVMFFIAny*)args)[1].type_index);
  if (!(((((k_handle_type_index == 0) || (k_handle_type_index == 4)) || (k_handle_type_index == 7)) || (64 <= k_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input k expected pointer or tensor handle");
    return -1;
  }
  int32_t v_handle_type_index = (((TVMFFIAny*)args)[2].type_index);
  if (!(((((v_handle_type_index == 0) || (v_handle_type_index == 4)) || (v_handle_type_index == 7)) || (64 <= v_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input v expected pointer or tensor handle");
    return -1;
  }
  int32_t a_handle_type_index = (((TVMFFIAny*)args)[3].type_index);
  if (!(((((a_handle_type_index == 0) || (a_handle_type_index == 4)) || (a_handle_type_index == 7)) || (64 <= a_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input a expected pointer or tensor handle");
    return -1;
  }
  int32_t g_handle_type_index = (((TVMFFIAny*)args)[4].type_index);
  if (!(((((g_handle_type_index == 0) || (g_handle_type_index == 4)) || (g_handle_type_index == 7)) || (64 <= g_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input g expected pointer or tensor handle");
    return -1;
  }
  int32_t b_handle_type_index = (((TVMFFIAny*)args)[5].type_index);
  if (!(((((b_handle_type_index == 0) || (b_handle_type_index == 4)) || (b_handle_type_index == 7)) || (64 <= b_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input b expected pointer or tensor handle");
    return -1;
  }
  int32_t h0_handle_type_index = (((TVMFFIAny*)args)[6].type_index);
  if (!(((((h0_handle_type_index == 0) || (h0_handle_type_index == 4)) || (h0_handle_type_index == 7)) || (64 <= h0_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input h0 expected pointer or tensor handle");
    return -1;
  }
  int32_t cu_seqlens_handle_type_index = (((TVMFFIAny*)args)[7].type_index);
  if (!(((((cu_seqlens_handle_type_index == 0) || (cu_seqlens_handle_type_index == 4)) || (cu_seqlens_handle_type_index == 7)) || (64 <= cu_seqlens_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input cu_seqlens expected pointer or tensor handle");
    return -1;
  }
  int32_t chunk_offsets_handle_type_index = (((TVMFFIAny*)args)[8].type_index);
  if (!(((((chunk_offsets_handle_type_index == 0) || (chunk_offsets_handle_type_index == 4)) || (chunk_offsets_handle_type_index == 7)) || (64 <= chunk_offsets_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input chunk_offsets expected pointer or tensor handle");
    return -1;
  }
  int32_t cp_seq_map_handle_type_index = (((TVMFFIAny*)args)[9].type_index);
  if (!(((((cp_seq_map_handle_type_index == 0) || (cp_seq_map_handle_type_index == 4)) || (cp_seq_map_handle_type_index == 7)) || (64 <= cp_seq_map_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input cp_seq_map expected pointer or tensor handle");
    return -1;
  }
  int32_t raw_cu_seqlens_handle_type_index = (((TVMFFIAny*)args)[10].type_index);
  if (!(((((raw_cu_seqlens_handle_type_index == 0) || (raw_cu_seqlens_handle_type_index == 4)) || (raw_cu_seqlens_handle_type_index == 7)) || (64 <= raw_cu_seqlens_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input raw_cu_seqlens expected pointer or tensor handle");
    return -1;
  }
  int32_t o_handle_type_index = (((TVMFFIAny*)args)[11].type_index);
  if (!(((((o_handle_type_index == 0) || (o_handle_type_index == 4)) || (o_handle_type_index == 7)) || (64 <= o_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input o expected pointer or tensor handle");
    return -1;
  }
  int32_t h_handle_type_index = (((TVMFFIAny*)args)[12].type_index);
  if (!(((((h_handle_type_index == 0) || (h_handle_type_index == 4)) || (h_handle_type_index == 7)) || (64 <= h_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input h expected pointer or tensor handle");
    return -1;
  }
  int32_t ht_handle_type_index = (((TVMFFIAny*)args)[13].type_index);
  if (!(((((ht_handle_type_index == 0) || (ht_handle_type_index == 4)) || (ht_handle_type_index == 7)) || (64 <= ht_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input ht expected pointer or tensor handle");
    return -1;
  }
  void* q_handle = ((q_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[0].v_ptr) + 24)) : (((TVMFFIAny*)args)[0].v_ptr));
  void* k_handle = ((k_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[1].v_ptr) + 24)) : (((TVMFFIAny*)args)[1].v_ptr));
  void* v_handle = ((v_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[2].v_ptr) + 24)) : (((TVMFFIAny*)args)[2].v_ptr));
  void* a_handle = ((a_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[3].v_ptr) + 24)) : (((TVMFFIAny*)args)[3].v_ptr));
  void* g_handle = ((g_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[4].v_ptr) + 24)) : (((TVMFFIAny*)args)[4].v_ptr));
  void* b_handle = ((b_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[5].v_ptr) + 24)) : (((TVMFFIAny*)args)[5].v_ptr));
  void* h0_handle = ((h0_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[6].v_ptr) + 24)) : (((TVMFFIAny*)args)[6].v_ptr));
  void* cu_seqlens_handle = ((cu_seqlens_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[7].v_ptr) + 24)) : (((TVMFFIAny*)args)[7].v_ptr));
  void* chunk_offsets_handle = ((chunk_offsets_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[8].v_ptr) + 24)) : (((TVMFFIAny*)args)[8].v_ptr));
  void* cp_seq_map_handle = ((cp_seq_map_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[9].v_ptr) + 24)) : (((TVMFFIAny*)args)[9].v_ptr));
  void* raw_cu_seqlens_handle = ((raw_cu_seqlens_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[10].v_ptr) + 24)) : (((TVMFFIAny*)args)[10].v_ptr));
  void* o_handle = ((o_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[11].v_ptr) + 24)) : (((TVMFFIAny*)args)[11].v_ptr));
  void* h_handle = ((h_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[12].v_ptr) + 24)) : (((TVMFFIAny*)args)[12].v_ptr));
  void* ht_handle = ((ht_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[13].v_ptr) + 24)) : (((TVMFFIAny*)args)[13].v_ptr));
  bool tilelang_fused_chunk_gdr_fwd_kernel_q_is_null = (q_handle == NULL);
  if (!(!tilelang_fused_chunk_gdr_fwd_kernel_q_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "tilelang_fused_chunk_gdr_fwd_kernel.q is expected to have non-NULL pointer");
    return -1;
  }
  bool tilelang_fused_chunk_gdr_fwd_kernel_k_is_null = (k_handle == NULL);
  if (!(!tilelang_fused_chunk_gdr_fwd_kernel_k_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "tilelang_fused_chunk_gdr_fwd_kernel.k is expected to have non-NULL pointer");
    return -1;
  }
  bool tilelang_fused_chunk_gdr_fwd_kernel_v_is_null = (v_handle == NULL);
  if (!(!tilelang_fused_chunk_gdr_fwd_kernel_v_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "tilelang_fused_chunk_gdr_fwd_kernel.v is expected to have non-NULL pointer");
    return -1;
  }
  bool tilelang_fused_chunk_gdr_fwd_kernel_a_is_null = (a_handle == NULL);
  if (!(!tilelang_fused_chunk_gdr_fwd_kernel_a_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "tilelang_fused_chunk_gdr_fwd_kernel.a is expected to have non-NULL pointer");
    return -1;
  }
  bool tilelang_fused_chunk_gdr_fwd_kernel_g_is_null = (g_handle == NULL);
  if (!(!tilelang_fused_chunk_gdr_fwd_kernel_g_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "tilelang_fused_chunk_gdr_fwd_kernel.g is expected to have non-NULL pointer");
    return -1;
  }
  bool tilelang_fused_chunk_gdr_fwd_kernel_b_is_null = (b_handle == NULL);
  if (!(!tilelang_fused_chunk_gdr_fwd_kernel_b_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "tilelang_fused_chunk_gdr_fwd_kernel.b is expected to have non-NULL pointer");
    return -1;
  }
  bool tilelang_fused_chunk_gdr_fwd_kernel_h0_is_null = (h0_handle == NULL);
  bool tilelang_fused_chunk_gdr_fwd_kernel_cu_seqlens_is_null = (cu_seqlens_handle == NULL);
  bool tilelang_fused_chunk_gdr_fwd_kernel_chunk_offsets_is_null = (chunk_offsets_handle == NULL);
  bool tilelang_fused_chunk_gdr_fwd_kernel_cp_seq_map_is_null = (cp_seq_map_handle == NULL);
  bool tilelang_fused_chunk_gdr_fwd_kernel_raw_cu_seqlens_is_null = (raw_cu_seqlens_handle == NULL);
  bool tilelang_fused_chunk_gdr_fwd_kernel_o_is_null = (o_handle == NULL);
  if (!(!tilelang_fused_chunk_gdr_fwd_kernel_o_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "tilelang_fused_chunk_gdr_fwd_kernel.o is expected to have non-NULL pointer");
    return -1;
  }
  bool tilelang_fused_chunk_gdr_fwd_kernel_h_is_null = (h_handle == NULL);
  bool tilelang_fused_chunk_gdr_fwd_kernel_ht_is_null = (ht_handle == NULL);
  if (!(!tilelang_fused_chunk_gdr_fwd_kernel_ht_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "tilelang_fused_chunk_gdr_fwd_kernel.ht is expected to have non-NULL pointer");
    return -1;
  }
  void* tilelang_fused_chunk_gdr_fwd_kernel_q_shape = (((DLTensor*)q_handle)[0].shape);
  void* tilelang_fused_chunk_gdr_fwd_kernel_k_shape = (((DLTensor*)k_handle)[0].shape);
  void* tilelang_fused_chunk_gdr_fwd_kernel_v_shape = (((DLTensor*)v_handle)[0].shape);
  void* tilelang_fused_chunk_gdr_fwd_kernel_a_shape = (((DLTensor*)a_handle)[0].shape);
  void* tilelang_fused_chunk_gdr_fwd_kernel_g_shape = (((DLTensor*)g_handle)[0].shape);
  void* tilelang_fused_chunk_gdr_fwd_kernel_b_shape = (((DLTensor*)b_handle)[0].shape);
  void* condval;
  if (!tilelang_fused_chunk_gdr_fwd_kernel_h0_is_null) {
    condval = (((DLTensor*)h0_handle)[0].shape);
  } else {
    uint64_t v_ = (uint64_t)0;
    condval = (*(void* *)(&(v_)));
  }
  void* tilelang_fused_chunk_gdr_fwd_kernel_h0_shape = condval;
  void* condval_1;
  if (!tilelang_fused_chunk_gdr_fwd_kernel_cu_seqlens_is_null) {
    condval_1 = (((DLTensor*)cu_seqlens_handle)[0].shape);
  } else {
    uint64_t v__1 = (uint64_t)0;
    condval_1 = (*(void* *)(&(v__1)));
  }
  void* tilelang_fused_chunk_gdr_fwd_kernel_cu_seqlens_shape = condval_1;
  void* condval_2;
  if (!tilelang_fused_chunk_gdr_fwd_kernel_chunk_offsets_is_null) {
    condval_2 = (((DLTensor*)chunk_offsets_handle)[0].shape);
  } else {
    uint64_t v__2 = (uint64_t)0;
    condval_2 = (*(void* *)(&(v__2)));
  }
  void* tilelang_fused_chunk_gdr_fwd_kernel_chunk_offsets_shape = condval_2;
  void* condval_3;
  if (!tilelang_fused_chunk_gdr_fwd_kernel_cp_seq_map_is_null) {
    condval_3 = (((DLTensor*)cp_seq_map_handle)[0].shape);
  } else {
    uint64_t v__3 = (uint64_t)0;
    condval_3 = (*(void* *)(&(v__3)));
  }
  void* tilelang_fused_chunk_gdr_fwd_kernel_cp_seq_map_shape = condval_3;
  void* condval_4;
  if (!tilelang_fused_chunk_gdr_fwd_kernel_raw_cu_seqlens_is_null) {
    condval_4 = (((DLTensor*)raw_cu_seqlens_handle)[0].shape);
  } else {
    uint64_t v__4 = (uint64_t)0;
    condval_4 = (*(void* *)(&(v__4)));
  }
  void* tilelang_fused_chunk_gdr_fwd_kernel_raw_cu_seqlens_shape = condval_4;
  void* tilelang_fused_chunk_gdr_fwd_kernel_o_shape = (((DLTensor*)o_handle)[0].shape);
  void* condval_5;
  if (!tilelang_fused_chunk_gdr_fwd_kernel_h_is_null) {
    condval_5 = (((DLTensor*)h_handle)[0].shape);
  } else {
    uint64_t v__5 = (uint64_t)0;
    condval_5 = (*(void* *)(&(v__5)));
  }
  void* tilelang_fused_chunk_gdr_fwd_kernel_h_shape = condval_5;
  void* tilelang_fused_chunk_gdr_fwd_kernel_ht_shape = (((DLTensor*)ht_handle)[0].shape);
  if (!(((((DLTensor*)q_handle)[0].ndim) == 4))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_fwd_kernel input q ndim mismatch, expected 4", (long long)((((DLTensor*)q_handle)[0].ndim)), (long long)(4));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((bool)1)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "Symbolic shape variable batch_size requires at least one non-null buffer among: tilelang_fused_chunk_gdr_fwd_kernel.q, tilelang_fused_chunk_gdr_fwd_kernel.k, tilelang_fused_chunk_gdr_fwd_kernel.v, tilelang_fused_chunk_gdr_fwd_kernel.a, tilelang_fused_chunk_gdr_fwd_kernel.g, tilelang_fused_chunk_gdr_fwd_kernel.b, tilelang_fused_chunk_gdr_fwd_kernel.h0, tilelang_fused_chunk_gdr_fwd_kernel.cp_seq_map, tilelang_fused_chunk_gdr_fwd_kernel.o, tilelang_fused_chunk_gdr_fwd_kernel.h");
    return -1;
  }
  int32_t batch_size = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_q_shape)[0]);
  if (!((bool)1)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "Symbolic shape variable num_tokens requires at least one non-null buffer among: tilelang_fused_chunk_gdr_fwd_kernel.q, tilelang_fused_chunk_gdr_fwd_kernel.k, tilelang_fused_chunk_gdr_fwd_kernel.v, tilelang_fused_chunk_gdr_fwd_kernel.a, tilelang_fused_chunk_gdr_fwd_kernel.g, tilelang_fused_chunk_gdr_fwd_kernel.b, tilelang_fused_chunk_gdr_fwd_kernel.o");
    return -1;
  }
  int32_t num_tokens = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_q_shape)[1]);
  void* tilelang_fused_chunk_gdr_fwd_kernel_q_strides = (((DLTensor*)q_handle)[0].strides);
  int32_t dev_id = (((DLTensor*)q_handle)[0].device.device_id);
  void* q = (((DLTensor*)q_handle)[0].data);
  if (!(((((DLTensor*)k_handle)[0].ndim) == 4))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_fwd_kernel input k ndim mismatch, expected 4", (long long)((((DLTensor*)k_handle)[0].ndim)), (long long)(4));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  void* tilelang_fused_chunk_gdr_fwd_kernel_k_strides = (((DLTensor*)k_handle)[0].strides);
  void* k = (((DLTensor*)k_handle)[0].data);
  if (!(((((DLTensor*)v_handle)[0].ndim) == 4))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_fwd_kernel input v ndim mismatch, expected 4", (long long)((((DLTensor*)v_handle)[0].ndim)), (long long)(4));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  void* tilelang_fused_chunk_gdr_fwd_kernel_v_strides = (((DLTensor*)v_handle)[0].strides);
  void* v = (((DLTensor*)v_handle)[0].data);
  if (!(((((DLTensor*)a_handle)[0].ndim) == 4))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_fwd_kernel input a ndim mismatch, expected 4", (long long)((((DLTensor*)a_handle)[0].ndim)), (long long)(4));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  void* tilelang_fused_chunk_gdr_fwd_kernel_a_strides = (((DLTensor*)a_handle)[0].strides);
  void* a = (((DLTensor*)a_handle)[0].data);
  if (!(((((DLTensor*)g_handle)[0].ndim) == 3))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_fwd_kernel input g ndim mismatch, expected 3", (long long)((((DLTensor*)g_handle)[0].ndim)), (long long)(3));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  void* tilelang_fused_chunk_gdr_fwd_kernel_g_strides = (((DLTensor*)g_handle)[0].strides);
  void* g = (((DLTensor*)g_handle)[0].data);
  if (!(((((DLTensor*)b_handle)[0].ndim) == 3))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_fwd_kernel input b ndim mismatch, expected 3", (long long)((((DLTensor*)b_handle)[0].ndim)), (long long)(3));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  void* tilelang_fused_chunk_gdr_fwd_kernel_b_strides = (((DLTensor*)b_handle)[0].strides);
  void* b = (((DLTensor*)b_handle)[0].data);
  if (!((tilelang_fused_chunk_gdr_fwd_kernel_h0_is_null || ((((DLTensor*)h0_handle)[0].ndim) == 4)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input h0 ndim mismatch, expected 4");
    return -1;
  }
  void* condval_6;
  if (!tilelang_fused_chunk_gdr_fwd_kernel_h0_is_null) {
    condval_6 = (((DLTensor*)h0_handle)[0].strides);
  } else {
    uint64_t v__6 = (uint64_t)0;
    condval_6 = (*(void* *)(&(v__6)));
  }
  void* tilelang_fused_chunk_gdr_fwd_kernel_h0_strides = condval_6;
  void* condval_7;
  if (!tilelang_fused_chunk_gdr_fwd_kernel_h0_is_null) {
    condval_7 = (((DLTensor*)h0_handle)[0].data);
  } else {
    uint64_t v__7 = (uint64_t)0;
    condval_7 = (*(void* *)(&(v__7)));
  }
  void* h0 = condval_7;
  if (!((tilelang_fused_chunk_gdr_fwd_kernel_cu_seqlens_is_null || ((((DLTensor*)cu_seqlens_handle)[0].ndim) == 1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input cu_seqlens ndim mismatch, expected 1");
    return -1;
  }
  void* condval_8;
  if (!tilelang_fused_chunk_gdr_fwd_kernel_cu_seqlens_is_null) {
    condval_8 = (((DLTensor*)cu_seqlens_handle)[0].strides);
  } else {
    uint64_t v__8 = (uint64_t)0;
    condval_8 = (*(void* *)(&(v__8)));
  }
  void* tilelang_fused_chunk_gdr_fwd_kernel_cu_seqlens_strides = condval_8;
  void* condval_9;
  if (!tilelang_fused_chunk_gdr_fwd_kernel_cu_seqlens_is_null) {
    condval_9 = (((DLTensor*)cu_seqlens_handle)[0].data);
  } else {
    uint64_t v__9 = (uint64_t)0;
    condval_9 = (*(void* *)(&(v__9)));
  }
  void* cu_seqlens = condval_9;
  if (!((tilelang_fused_chunk_gdr_fwd_kernel_chunk_offsets_is_null || ((((DLTensor*)chunk_offsets_handle)[0].ndim) == 1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input chunk_offsets ndim mismatch, expected 1");
    return -1;
  }
  void* condval_10;
  if (!tilelang_fused_chunk_gdr_fwd_kernel_chunk_offsets_is_null) {
    condval_10 = (((DLTensor*)chunk_offsets_handle)[0].strides);
  } else {
    uint64_t v__10 = (uint64_t)0;
    condval_10 = (*(void* *)(&(v__10)));
  }
  void* tilelang_fused_chunk_gdr_fwd_kernel_chunk_offsets_strides = condval_10;
  void* condval_11;
  if (!tilelang_fused_chunk_gdr_fwd_kernel_chunk_offsets_is_null) {
    condval_11 = (((DLTensor*)chunk_offsets_handle)[0].data);
  } else {
    uint64_t v__11 = (uint64_t)0;
    condval_11 = (*(void* *)(&(v__11)));
  }
  void* chunk_offsets = condval_11;
  if (!((tilelang_fused_chunk_gdr_fwd_kernel_cp_seq_map_is_null || ((((DLTensor*)cp_seq_map_handle)[0].ndim) == 1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input cp_seq_map ndim mismatch, expected 1");
    return -1;
  }
  void* condval_12;
  if (!tilelang_fused_chunk_gdr_fwd_kernel_cp_seq_map_is_null) {
    condval_12 = (((DLTensor*)cp_seq_map_handle)[0].strides);
  } else {
    uint64_t v__12 = (uint64_t)0;
    condval_12 = (*(void* *)(&(v__12)));
  }
  void* tilelang_fused_chunk_gdr_fwd_kernel_cp_seq_map_strides = condval_12;
  void* condval_13;
  if (!tilelang_fused_chunk_gdr_fwd_kernel_cp_seq_map_is_null) {
    condval_13 = (((DLTensor*)cp_seq_map_handle)[0].data);
  } else {
    uint64_t v__13 = (uint64_t)0;
    condval_13 = (*(void* *)(&(v__13)));
  }
  void* cp_seq_map = condval_13;
  if (!((tilelang_fused_chunk_gdr_fwd_kernel_raw_cu_seqlens_is_null || ((((DLTensor*)raw_cu_seqlens_handle)[0].ndim) == 1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input raw_cu_seqlens ndim mismatch, expected 1");
    return -1;
  }
  int32_t condval_14;
  if (!tilelang_fused_chunk_gdr_fwd_kernel_raw_cu_seqlens_is_null) {
    condval_14 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_raw_cu_seqlens_shape)[0]);
  } else {
    condval_14 = 0;
  }
  int32_t raw_batch_size = (condval_14 - 1);
  void* condval_15;
  if (!tilelang_fused_chunk_gdr_fwd_kernel_raw_cu_seqlens_is_null) {
    condval_15 = (((DLTensor*)raw_cu_seqlens_handle)[0].strides);
  } else {
    uint64_t v__14 = (uint64_t)0;
    condval_15 = (*(void* *)(&(v__14)));
  }
  void* tilelang_fused_chunk_gdr_fwd_kernel_raw_cu_seqlens_strides = condval_15;
  void* condval_16;
  if (!tilelang_fused_chunk_gdr_fwd_kernel_raw_cu_seqlens_is_null) {
    condval_16 = (((DLTensor*)raw_cu_seqlens_handle)[0].data);
  } else {
    uint64_t v__15 = (uint64_t)0;
    condval_16 = (*(void* *)(&(v__15)));
  }
  void* raw_cu_seqlens = condval_16;
  if (!(((((DLTensor*)o_handle)[0].ndim) == 4))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_fwd_kernel input o ndim mismatch, expected 4", (long long)((((DLTensor*)o_handle)[0].ndim)), (long long)(4));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  void* tilelang_fused_chunk_gdr_fwd_kernel_o_strides = (((DLTensor*)o_handle)[0].strides);
  void* o = (((DLTensor*)o_handle)[0].data);
  if (!((tilelang_fused_chunk_gdr_fwd_kernel_h_is_null || ((((DLTensor*)h_handle)[0].ndim) == 5)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input h ndim mismatch, expected 5");
    return -1;
  }
  int32_t condval_17;
  if (!tilelang_fused_chunk_gdr_fwd_kernel_h_is_null) {
    condval_17 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_h_shape)[1]);
  } else {
    condval_17 = 0;
  }
  int32_t num_chunks = condval_17;
  void* condval_18;
  if (!tilelang_fused_chunk_gdr_fwd_kernel_h_is_null) {
    condval_18 = (((DLTensor*)h_handle)[0].strides);
  } else {
    uint64_t v__16 = (uint64_t)0;
    condval_18 = (*(void* *)(&(v__16)));
  }
  void* tilelang_fused_chunk_gdr_fwd_kernel_h_strides = condval_18;
  void* condval_19;
  if (!tilelang_fused_chunk_gdr_fwd_kernel_h_is_null) {
    condval_19 = (((DLTensor*)h_handle)[0].data);
  } else {
    uint64_t v__17 = (uint64_t)0;
    condval_19 = (*(void* *)(&(v__17)));
  }
  void* h = condval_19;
  if (!(((((DLTensor*)ht_handle)[0].ndim) == 4))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_fwd_kernel input ht ndim mismatch, expected 4", (long long)((((DLTensor*)ht_handle)[0].ndim)), (long long)(4));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  void* tilelang_fused_chunk_gdr_fwd_kernel_ht_strides = (((DLTensor*)ht_handle)[0].strides);
  void* ht = (((DLTensor*)ht_handle)[0].data);
  if (!(((((((DLTensor*)q_handle)[0].dtype.code) == (uint8_t)4) && ((((DLTensor*)q_handle)[0].dtype.bits) == (uint8_t)16)) && ((((DLTensor*)q_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input q dtype mismatch, expected bfloat16");
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_q_shape)[2]) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_fwd_kernel input q shape[2] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_q_shape)[2])), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_q_shape)[3]) == 128))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_fwd_kernel input q shape[3] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_q_shape)[3])), (long long)(128));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_20;
  if ((tilelang_fused_chunk_gdr_fwd_kernel_q_strides == NULL)) {
    condval_20 = 1;
  } else {
    condval_20 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_q_strides)[3]);
  }
  if (!((condval_20 == 1))) {
    int32_t condval_21;
    if ((tilelang_fused_chunk_gdr_fwd_kernel_q_strides == NULL)) {
      condval_21 = 1;
    } else {
      condval_21 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_q_strides)[3]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_fwd_kernel input q strides[3] violates packed ABI constraint", (long long)(condval_21), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_22;
  if ((tilelang_fused_chunk_gdr_fwd_kernel_q_strides == NULL)) {
    condval_22 = 1;
  } else {
    condval_22 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_q_strides)[2]);
  }
  if (!((condval_22 == 128))) {
    int32_t condval_23;
    if ((tilelang_fused_chunk_gdr_fwd_kernel_q_strides == NULL)) {
      condval_23 = 1;
    } else {
      condval_23 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_q_strides)[2]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_fwd_kernel input q strides[2] violates packed ABI constraint", (long long)(condval_23), (long long)(128));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_24;
  if ((tilelang_fused_chunk_gdr_fwd_kernel_q_strides == NULL)) {
    condval_24 = 1;
  } else {
    condval_24 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_q_strides)[1]);
  }
  if (!(((condval_24 == 256) || (num_tokens == 1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input q strides[1] violates packed ABI constraint");
    return -1;
  }
  int32_t condval_25;
  if ((tilelang_fused_chunk_gdr_fwd_kernel_q_strides == NULL)) {
    condval_25 = 1;
  } else {
    condval_25 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_q_strides)[0]);
  }
  if (!(((((num_tokens * 256) == condval_25) || (num_tokens == 0)) || (batch_size == 1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input q strides[0] violates packed ABI constraint");
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)q_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_fwd_kernel input q byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)q_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)q_handle)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_fwd_kernel input q device_type mismatch, expected cuda", (long long)((((DLTensor*)q_handle)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((batch_size == 0) || (num_tokens == 0)) || !(q == NULL)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input q data pointer is NULL");
    return -1;
  }
  if (!(((((((DLTensor*)k_handle)[0].dtype.code) == (uint8_t)4) && ((((DLTensor*)k_handle)[0].dtype.bits) == (uint8_t)16)) && ((((DLTensor*)k_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input k dtype mismatch, expected bfloat16");
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_q_shape)[0]) == ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_k_shape)[0])))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_fwd_kernel input k shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_q_shape)[0])), (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_k_shape)[0])));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_q_shape)[1]) == ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_k_shape)[1])))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_fwd_kernel input k shape[1] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_q_shape)[1])), (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_k_shape)[1])));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_k_shape)[2]) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_fwd_kernel input k shape[2] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_k_shape)[2])), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_k_shape)[3]) == 128))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_fwd_kernel input k shape[3] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_k_shape)[3])), (long long)(128));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_26;
  if ((tilelang_fused_chunk_gdr_fwd_kernel_k_strides == NULL)) {
    condval_26 = 1;
  } else {
    condval_26 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_k_strides)[3]);
  }
  if (!((condval_26 == 1))) {
    int32_t condval_27;
    if ((tilelang_fused_chunk_gdr_fwd_kernel_k_strides == NULL)) {
      condval_27 = 1;
    } else {
      condval_27 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_k_strides)[3]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_fwd_kernel input k strides[3] violates packed ABI constraint", (long long)(condval_27), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_28;
  if ((tilelang_fused_chunk_gdr_fwd_kernel_k_strides == NULL)) {
    condval_28 = 1;
  } else {
    condval_28 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_k_strides)[2]);
  }
  if (!((condval_28 == 128))) {
    int32_t condval_29;
    if ((tilelang_fused_chunk_gdr_fwd_kernel_k_strides == NULL)) {
      condval_29 = 1;
    } else {
      condval_29 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_k_strides)[2]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_fwd_kernel input k strides[2] violates packed ABI constraint", (long long)(condval_29), (long long)(128));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_30;
  if ((tilelang_fused_chunk_gdr_fwd_kernel_k_strides == NULL)) {
    condval_30 = 1;
  } else {
    condval_30 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_k_strides)[1]);
  }
  if (!(((condval_30 == 256) || (num_tokens == 1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input k strides[1] violates packed ABI constraint");
    return -1;
  }
  int32_t condval_31;
  if ((tilelang_fused_chunk_gdr_fwd_kernel_k_strides == NULL)) {
    condval_31 = 1;
  } else {
    condval_31 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_k_strides)[0]);
  }
  if (!(((((num_tokens * 256) == condval_31) || (num_tokens == 0)) || (batch_size == 1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input k strides[0] violates packed ABI constraint");
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)k_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_fwd_kernel input k byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)k_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)k_handle)[0].device.device_id) == (((DLTensor*)q_handle)[0].device.device_id)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_fwd_kernel input k device_id violates packed ABI constraint", (long long)((((DLTensor*)k_handle)[0].device.device_id)), (long long)((((DLTensor*)q_handle)[0].device.device_id)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)k_handle)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_fwd_kernel input k device_type mismatch, expected cuda", (long long)((((DLTensor*)k_handle)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((batch_size == 0) || (num_tokens == 0)) || !(k == NULL)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input k data pointer is NULL");
    return -1;
  }
  if (!(((((((DLTensor*)v_handle)[0].dtype.code) == (uint8_t)4) && ((((DLTensor*)v_handle)[0].dtype.bits) == (uint8_t)16)) && ((((DLTensor*)v_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input v dtype mismatch, expected bfloat16");
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_q_shape)[0]) == ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_v_shape)[0])))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_fwd_kernel input v shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_q_shape)[0])), (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_v_shape)[0])));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_q_shape)[1]) == ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_v_shape)[1])))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_fwd_kernel input v shape[1] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_q_shape)[1])), (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_v_shape)[1])));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_v_shape)[2]) == 4))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_fwd_kernel input v shape[2] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_v_shape)[2])), (long long)(4));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_v_shape)[3]) == 128))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_fwd_kernel input v shape[3] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_v_shape)[3])), (long long)(128));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_32;
  if ((tilelang_fused_chunk_gdr_fwd_kernel_v_strides == NULL)) {
    condval_32 = 1;
  } else {
    condval_32 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_v_strides)[3]);
  }
  if (!((condval_32 == 1))) {
    int32_t condval_33;
    if ((tilelang_fused_chunk_gdr_fwd_kernel_v_strides == NULL)) {
      condval_33 = 1;
    } else {
      condval_33 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_v_strides)[3]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_fwd_kernel input v strides[3] violates packed ABI constraint", (long long)(condval_33), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_34;
  if ((tilelang_fused_chunk_gdr_fwd_kernel_v_strides == NULL)) {
    condval_34 = 1;
  } else {
    condval_34 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_v_strides)[2]);
  }
  if (!((condval_34 == 128))) {
    int32_t condval_35;
    if ((tilelang_fused_chunk_gdr_fwd_kernel_v_strides == NULL)) {
      condval_35 = 1;
    } else {
      condval_35 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_v_strides)[2]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_fwd_kernel input v strides[2] violates packed ABI constraint", (long long)(condval_35), (long long)(128));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_36;
  if ((tilelang_fused_chunk_gdr_fwd_kernel_v_strides == NULL)) {
    condval_36 = 1;
  } else {
    condval_36 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_v_strides)[1]);
  }
  if (!(((condval_36 == 512) || (num_tokens == 1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input v strides[1] violates packed ABI constraint");
    return -1;
  }
  int32_t condval_37;
  if ((tilelang_fused_chunk_gdr_fwd_kernel_v_strides == NULL)) {
    condval_37 = 1;
  } else {
    condval_37 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_v_strides)[0]);
  }
  if (!(((((num_tokens * 512) == condval_37) || (num_tokens == 0)) || (batch_size == 1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input v strides[0] violates packed ABI constraint");
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)v_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_fwd_kernel input v byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)v_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)v_handle)[0].device.device_id) == (((DLTensor*)q_handle)[0].device.device_id)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_fwd_kernel input v device_id violates packed ABI constraint", (long long)((((DLTensor*)v_handle)[0].device.device_id)), (long long)((((DLTensor*)q_handle)[0].device.device_id)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)v_handle)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_fwd_kernel input v device_type mismatch, expected cuda", (long long)((((DLTensor*)v_handle)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((batch_size == 0) || (num_tokens == 0)) || !(v == NULL)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input v data pointer is NULL");
    return -1;
  }
  if (!(((((((DLTensor*)a_handle)[0].dtype.code) == (uint8_t)4) && ((((DLTensor*)a_handle)[0].dtype.bits) == (uint8_t)16)) && ((((DLTensor*)a_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input a dtype mismatch, expected bfloat16");
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_q_shape)[0]) == ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_a_shape)[0])))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_fwd_kernel input a shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_q_shape)[0])), (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_a_shape)[0])));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_q_shape)[1]) == ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_a_shape)[1])))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_fwd_kernel input a shape[1] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_q_shape)[1])), (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_a_shape)[1])));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_a_shape)[2]) == 4))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_fwd_kernel input a shape[2] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_a_shape)[2])), (long long)(4));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_a_shape)[3]) == 32))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_fwd_kernel input a shape[3] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_a_shape)[3])), (long long)(32));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_38;
  if ((tilelang_fused_chunk_gdr_fwd_kernel_a_strides == NULL)) {
    condval_38 = 1;
  } else {
    condval_38 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_a_strides)[3]);
  }
  if (!((condval_38 == 1))) {
    int32_t condval_39;
    if ((tilelang_fused_chunk_gdr_fwd_kernel_a_strides == NULL)) {
      condval_39 = 1;
    } else {
      condval_39 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_a_strides)[3]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_fwd_kernel input a strides[3] violates packed ABI constraint", (long long)(condval_39), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_40;
  if ((tilelang_fused_chunk_gdr_fwd_kernel_a_strides == NULL)) {
    condval_40 = 1;
  } else {
    condval_40 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_a_strides)[2]);
  }
  if (!((condval_40 == 32))) {
    int32_t condval_41;
    if ((tilelang_fused_chunk_gdr_fwd_kernel_a_strides == NULL)) {
      condval_41 = 1;
    } else {
      condval_41 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_a_strides)[2]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_fwd_kernel input a strides[2] violates packed ABI constraint", (long long)(condval_41), (long long)(32));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_42;
  if ((tilelang_fused_chunk_gdr_fwd_kernel_a_strides == NULL)) {
    condval_42 = 1;
  } else {
    condval_42 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_a_strides)[1]);
  }
  if (!(((condval_42 == 128) || (num_tokens == 1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input a strides[1] violates packed ABI constraint");
    return -1;
  }
  int32_t condval_43;
  if ((tilelang_fused_chunk_gdr_fwd_kernel_a_strides == NULL)) {
    condval_43 = 1;
  } else {
    condval_43 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_a_strides)[0]);
  }
  if (!(((((num_tokens * 128) == condval_43) || (num_tokens == 0)) || (batch_size == 1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input a strides[0] violates packed ABI constraint");
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)a_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_fwd_kernel input a byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)a_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)a_handle)[0].device.device_id) == (((DLTensor*)q_handle)[0].device.device_id)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_fwd_kernel input a device_id violates packed ABI constraint", (long long)((((DLTensor*)a_handle)[0].device.device_id)), (long long)((((DLTensor*)q_handle)[0].device.device_id)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)a_handle)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_fwd_kernel input a device_type mismatch, expected cuda", (long long)((((DLTensor*)a_handle)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((batch_size == 0) || (num_tokens == 0)) || !(a == NULL)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input a data pointer is NULL");
    return -1;
  }
  if (!(((((((DLTensor*)g_handle)[0].dtype.code) == (uint8_t)2) && ((((DLTensor*)g_handle)[0].dtype.bits) == (uint8_t)32)) && ((((DLTensor*)g_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input g dtype mismatch, expected float32");
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_q_shape)[0]) == ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_g_shape)[0])))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_fwd_kernel input g shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_q_shape)[0])), (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_g_shape)[0])));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_q_shape)[1]) == ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_g_shape)[1])))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_fwd_kernel input g shape[1] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_q_shape)[1])), (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_g_shape)[1])));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_g_shape)[2]) == 4))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_fwd_kernel input g shape[2] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_g_shape)[2])), (long long)(4));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_44;
  if ((tilelang_fused_chunk_gdr_fwd_kernel_g_strides == NULL)) {
    condval_44 = 1;
  } else {
    condval_44 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_g_strides)[2]);
  }
  if (!((condval_44 == 1))) {
    int32_t condval_45;
    if ((tilelang_fused_chunk_gdr_fwd_kernel_g_strides == NULL)) {
      condval_45 = 1;
    } else {
      condval_45 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_g_strides)[2]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_fwd_kernel input g strides[2] violates packed ABI constraint", (long long)(condval_45), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_46;
  if ((tilelang_fused_chunk_gdr_fwd_kernel_g_strides == NULL)) {
    condval_46 = 1;
  } else {
    condval_46 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_g_strides)[1]);
  }
  if (!(((condval_46 == 4) || (num_tokens == 1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input g strides[1] violates packed ABI constraint");
    return -1;
  }
  int32_t condval_47;
  if ((tilelang_fused_chunk_gdr_fwd_kernel_g_strides == NULL)) {
    condval_47 = 1;
  } else {
    condval_47 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_g_strides)[0]);
  }
  if (!(((((num_tokens * 4) == condval_47) || (num_tokens == 0)) || (batch_size == 1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input g strides[0] violates packed ABI constraint");
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)g_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_fwd_kernel input g byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)g_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)g_handle)[0].device.device_id) == (((DLTensor*)q_handle)[0].device.device_id)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_fwd_kernel input g device_id violates packed ABI constraint", (long long)((((DLTensor*)g_handle)[0].device.device_id)), (long long)((((DLTensor*)q_handle)[0].device.device_id)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)g_handle)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_fwd_kernel input g device_type mismatch, expected cuda", (long long)((((DLTensor*)g_handle)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((batch_size == 0) || (num_tokens == 0)) || !(g == NULL)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input g data pointer is NULL");
    return -1;
  }
  if (!(((((((DLTensor*)b_handle)[0].dtype.code) == (uint8_t)2) && ((((DLTensor*)b_handle)[0].dtype.bits) == (uint8_t)32)) && ((((DLTensor*)b_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input b dtype mismatch, expected float32");
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_q_shape)[0]) == ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_b_shape)[0])))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_fwd_kernel input b shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_q_shape)[0])), (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_b_shape)[0])));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_q_shape)[1]) == ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_b_shape)[1])))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_fwd_kernel input b shape[1] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_q_shape)[1])), (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_b_shape)[1])));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_b_shape)[2]) == 4))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_fwd_kernel input b shape[2] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_b_shape)[2])), (long long)(4));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_48;
  if ((tilelang_fused_chunk_gdr_fwd_kernel_b_strides == NULL)) {
    condval_48 = 1;
  } else {
    condval_48 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_b_strides)[2]);
  }
  if (!((condval_48 == 1))) {
    int32_t condval_49;
    if ((tilelang_fused_chunk_gdr_fwd_kernel_b_strides == NULL)) {
      condval_49 = 1;
    } else {
      condval_49 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_b_strides)[2]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_fwd_kernel input b strides[2] violates packed ABI constraint", (long long)(condval_49), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_50;
  if ((tilelang_fused_chunk_gdr_fwd_kernel_b_strides == NULL)) {
    condval_50 = 1;
  } else {
    condval_50 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_b_strides)[1]);
  }
  if (!(((condval_50 == 4) || (num_tokens == 1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input b strides[1] violates packed ABI constraint");
    return -1;
  }
  int32_t condval_51;
  if ((tilelang_fused_chunk_gdr_fwd_kernel_b_strides == NULL)) {
    condval_51 = 1;
  } else {
    condval_51 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_b_strides)[0]);
  }
  if (!(((((num_tokens * 4) == condval_51) || (num_tokens == 0)) || (batch_size == 1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input b strides[0] violates packed ABI constraint");
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)b_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_fwd_kernel input b byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)b_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)b_handle)[0].device.device_id) == (((DLTensor*)q_handle)[0].device.device_id)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_fwd_kernel input b device_id violates packed ABI constraint", (long long)((((DLTensor*)b_handle)[0].device.device_id)), (long long)((((DLTensor*)q_handle)[0].device.device_id)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)b_handle)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_fwd_kernel input b device_type mismatch, expected cuda", (long long)((((DLTensor*)b_handle)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((batch_size == 0) || (num_tokens == 0)) || !(b == NULL)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input b data pointer is NULL");
    return -1;
  }
  uint8_t condval_52;
  if (!tilelang_fused_chunk_gdr_fwd_kernel_h0_is_null) {
    condval_52 = (((DLTensor*)h0_handle)[0].dtype.code);
  } else {
    condval_52 = (uint8_t)2;
  }
  uint8_t condval_53;
  if (!tilelang_fused_chunk_gdr_fwd_kernel_h0_is_null) {
    condval_53 = (((DLTensor*)h0_handle)[0].dtype.bits);
  } else {
    condval_53 = (uint8_t)32;
  }
  uint16_t condval_54;
  if (!tilelang_fused_chunk_gdr_fwd_kernel_h0_is_null) {
    condval_54 = (((DLTensor*)h0_handle)[0].dtype.lanes);
  } else {
    condval_54 = (uint16_t)1;
  }
  if (!((tilelang_fused_chunk_gdr_fwd_kernel_h0_is_null || (((condval_52 == (uint8_t)2) && (condval_53 == (uint8_t)32)) && (condval_54 == (uint16_t)1))))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input h0 dtype mismatch, expected float32");
    return -1;
  }
  int32_t condval_55;
  if (!tilelang_fused_chunk_gdr_fwd_kernel_h0_is_null) {
    condval_55 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_h0_shape)[0]);
  } else {
    condval_55 = 0;
  }
  if (!((tilelang_fused_chunk_gdr_fwd_kernel_h0_is_null || (((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_q_shape)[0]) == condval_55)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input h0 shape[0] violates packed ABI constraint");
    return -1;
  }
  int32_t condval_56;
  if (!tilelang_fused_chunk_gdr_fwd_kernel_h0_is_null) {
    condval_56 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_h0_shape)[1]);
  } else {
    condval_56 = 0;
  }
  if (!((tilelang_fused_chunk_gdr_fwd_kernel_h0_is_null || (condval_56 == 4)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input h0 shape[1] violates packed ABI constraint");
    return -1;
  }
  int32_t condval_57;
  if (!tilelang_fused_chunk_gdr_fwd_kernel_h0_is_null) {
    condval_57 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_h0_shape)[2]);
  } else {
    condval_57 = 0;
  }
  if (!((tilelang_fused_chunk_gdr_fwd_kernel_h0_is_null || (condval_57 == 128)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input h0 shape[2] violates packed ABI constraint");
    return -1;
  }
  int32_t condval_58;
  if (!tilelang_fused_chunk_gdr_fwd_kernel_h0_is_null) {
    condval_58 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_h0_shape)[3]);
  } else {
    condval_58 = 0;
  }
  if (!((tilelang_fused_chunk_gdr_fwd_kernel_h0_is_null || (condval_58 == 128)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input h0 shape[3] violates packed ABI constraint");
    return -1;
  }
  int32_t condval_59;
  if ((tilelang_fused_chunk_gdr_fwd_kernel_h0_strides == NULL)) {
    condval_59 = 1;
  } else {
    condval_59 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_h0_strides)[3]);
  }
  if (!((tilelang_fused_chunk_gdr_fwd_kernel_h0_is_null || (condval_59 == 1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input h0 strides[3] violates packed ABI constraint");
    return -1;
  }
  int32_t condval_60;
  if ((tilelang_fused_chunk_gdr_fwd_kernel_h0_strides == NULL)) {
    condval_60 = 1;
  } else {
    condval_60 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_h0_strides)[2]);
  }
  if (!((tilelang_fused_chunk_gdr_fwd_kernel_h0_is_null || (condval_60 == 128)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input h0 strides[2] violates packed ABI constraint");
    return -1;
  }
  int32_t condval_61;
  if ((tilelang_fused_chunk_gdr_fwd_kernel_h0_strides == NULL)) {
    condval_61 = 1;
  } else {
    condval_61 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_h0_strides)[1]);
  }
  if (!((tilelang_fused_chunk_gdr_fwd_kernel_h0_is_null || (condval_61 == 16384)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input h0 strides[1] violates packed ABI constraint");
    return -1;
  }
  int32_t condval_62;
  if ((tilelang_fused_chunk_gdr_fwd_kernel_h0_strides == NULL)) {
    condval_62 = 1;
  } else {
    condval_62 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_h0_strides)[0]);
  }
  if (!(((tilelang_fused_chunk_gdr_fwd_kernel_h0_is_null || (condval_62 == 65536)) || (batch_size == 1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input h0 strides[0] violates packed ABI constraint");
    return -1;
  }
  uint64_t condval_63;
  if (!tilelang_fused_chunk_gdr_fwd_kernel_h0_is_null) {
    condval_63 = (((DLTensor*)h0_handle)[0].byte_offset);
  } else {
    condval_63 = (uint64_t)0;
  }
  if (!((tilelang_fused_chunk_gdr_fwd_kernel_h0_is_null || ((uint64_t)0 == condval_63)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input h0 byte_offset violates packed ABI constraint");
    return -1;
  }
  int32_t condval_64;
  if (!tilelang_fused_chunk_gdr_fwd_kernel_h0_is_null) {
    condval_64 = (((DLTensor*)h0_handle)[0].device.device_id);
  } else {
    condval_64 = 0;
  }
  if (!((tilelang_fused_chunk_gdr_fwd_kernel_h0_is_null || (condval_64 == (((DLTensor*)q_handle)[0].device.device_id))))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input h0 device_id violates packed ABI constraint");
    return -1;
  }
  int32_t condval_65;
  if (!tilelang_fused_chunk_gdr_fwd_kernel_h0_is_null) {
    condval_65 = (((DLTensor*)h0_handle)[0].device.device_type);
  } else {
    condval_65 = 0;
  }
  if (!((tilelang_fused_chunk_gdr_fwd_kernel_h0_is_null || (condval_65 == 2)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input h0 device_type mismatch, expected cuda");
    return -1;
  }
  if (!(((tilelang_fused_chunk_gdr_fwd_kernel_h0_is_null || (batch_size == 0)) || !(h0 == NULL)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input h0 data pointer is NULL");
    return -1;
  }
  uint8_t condval_66;
  if (!tilelang_fused_chunk_gdr_fwd_kernel_cu_seqlens_is_null) {
    condval_66 = (((DLTensor*)cu_seqlens_handle)[0].dtype.code);
  } else {
    condval_66 = (uint8_t)0;
  }
  uint8_t condval_67;
  if (!tilelang_fused_chunk_gdr_fwd_kernel_cu_seqlens_is_null) {
    condval_67 = (((DLTensor*)cu_seqlens_handle)[0].dtype.bits);
  } else {
    condval_67 = (uint8_t)32;
  }
  uint16_t condval_68;
  if (!tilelang_fused_chunk_gdr_fwd_kernel_cu_seqlens_is_null) {
    condval_68 = (((DLTensor*)cu_seqlens_handle)[0].dtype.lanes);
  } else {
    condval_68 = (uint16_t)1;
  }
  if (!((tilelang_fused_chunk_gdr_fwd_kernel_cu_seqlens_is_null || (((condval_66 == (uint8_t)0) && (condval_67 == (uint8_t)32)) && (condval_68 == (uint16_t)1))))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input cu_seqlens dtype mismatch, expected int32");
    return -1;
  }
  int32_t condval_69;
  if (!tilelang_fused_chunk_gdr_fwd_kernel_cu_seqlens_is_null) {
    condval_69 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_cu_seqlens_shape)[0]);
  } else {
    condval_69 = 0;
  }
  if (!((tilelang_fused_chunk_gdr_fwd_kernel_cu_seqlens_is_null || (condval_69 == (batch_size + 1))))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input cu_seqlens shape[0] violates packed ABI constraint");
    return -1;
  }
  int32_t condval_70;
  if ((tilelang_fused_chunk_gdr_fwd_kernel_cu_seqlens_strides == NULL)) {
    condval_70 = 1;
  } else {
    condval_70 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_cu_seqlens_strides)[0]);
  }
  if (!(((tilelang_fused_chunk_gdr_fwd_kernel_cu_seqlens_is_null || (condval_70 == 1)) || (batch_size == 0)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input cu_seqlens strides[0] violates packed ABI constraint");
    return -1;
  }
  uint64_t condval_71;
  if (!tilelang_fused_chunk_gdr_fwd_kernel_cu_seqlens_is_null) {
    condval_71 = (((DLTensor*)cu_seqlens_handle)[0].byte_offset);
  } else {
    condval_71 = (uint64_t)0;
  }
  if (!((tilelang_fused_chunk_gdr_fwd_kernel_cu_seqlens_is_null || ((uint64_t)0 == condval_71)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input cu_seqlens byte_offset violates packed ABI constraint");
    return -1;
  }
  int32_t condval_72;
  if (!tilelang_fused_chunk_gdr_fwd_kernel_cu_seqlens_is_null) {
    condval_72 = (((DLTensor*)cu_seqlens_handle)[0].device.device_id);
  } else {
    condval_72 = 0;
  }
  if (!((tilelang_fused_chunk_gdr_fwd_kernel_cu_seqlens_is_null || (condval_72 == (((DLTensor*)q_handle)[0].device.device_id))))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input cu_seqlens device_id violates packed ABI constraint");
    return -1;
  }
  int32_t condval_73;
  if (!tilelang_fused_chunk_gdr_fwd_kernel_cu_seqlens_is_null) {
    condval_73 = (((DLTensor*)cu_seqlens_handle)[0].device.device_type);
  } else {
    condval_73 = 0;
  }
  if (!((tilelang_fused_chunk_gdr_fwd_kernel_cu_seqlens_is_null || (condval_73 == 2)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input cu_seqlens device_type mismatch, expected cuda");
    return -1;
  }
  if (!(((tilelang_fused_chunk_gdr_fwd_kernel_cu_seqlens_is_null || (batch_size == -1)) || !(cu_seqlens == NULL)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input cu_seqlens data pointer is NULL");
    return -1;
  }
  uint8_t condval_74;
  if (!tilelang_fused_chunk_gdr_fwd_kernel_chunk_offsets_is_null) {
    condval_74 = (((DLTensor*)chunk_offsets_handle)[0].dtype.code);
  } else {
    condval_74 = (uint8_t)0;
  }
  uint8_t condval_75;
  if (!tilelang_fused_chunk_gdr_fwd_kernel_chunk_offsets_is_null) {
    condval_75 = (((DLTensor*)chunk_offsets_handle)[0].dtype.bits);
  } else {
    condval_75 = (uint8_t)32;
  }
  uint16_t condval_76;
  if (!tilelang_fused_chunk_gdr_fwd_kernel_chunk_offsets_is_null) {
    condval_76 = (((DLTensor*)chunk_offsets_handle)[0].dtype.lanes);
  } else {
    condval_76 = (uint16_t)1;
  }
  if (!((tilelang_fused_chunk_gdr_fwd_kernel_chunk_offsets_is_null || (((condval_74 == (uint8_t)0) && (condval_75 == (uint8_t)32)) && (condval_76 == (uint16_t)1))))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input chunk_offsets dtype mismatch, expected int32");
    return -1;
  }
  int32_t condval_77;
  if (!tilelang_fused_chunk_gdr_fwd_kernel_chunk_offsets_is_null) {
    condval_77 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_chunk_offsets_shape)[0]);
  } else {
    condval_77 = 0;
  }
  if (!((tilelang_fused_chunk_gdr_fwd_kernel_chunk_offsets_is_null || (condval_77 == (batch_size + 1))))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input chunk_offsets shape[0] violates packed ABI constraint");
    return -1;
  }
  int32_t condval_78;
  if ((tilelang_fused_chunk_gdr_fwd_kernel_chunk_offsets_strides == NULL)) {
    condval_78 = 1;
  } else {
    condval_78 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_chunk_offsets_strides)[0]);
  }
  if (!(((tilelang_fused_chunk_gdr_fwd_kernel_chunk_offsets_is_null || (condval_78 == 1)) || (batch_size == 0)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input chunk_offsets strides[0] violates packed ABI constraint");
    return -1;
  }
  uint64_t condval_79;
  if (!tilelang_fused_chunk_gdr_fwd_kernel_chunk_offsets_is_null) {
    condval_79 = (((DLTensor*)chunk_offsets_handle)[0].byte_offset);
  } else {
    condval_79 = (uint64_t)0;
  }
  if (!((tilelang_fused_chunk_gdr_fwd_kernel_chunk_offsets_is_null || ((uint64_t)0 == condval_79)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input chunk_offsets byte_offset violates packed ABI constraint");
    return -1;
  }
  int32_t condval_80;
  if (!tilelang_fused_chunk_gdr_fwd_kernel_chunk_offsets_is_null) {
    condval_80 = (((DLTensor*)chunk_offsets_handle)[0].device.device_id);
  } else {
    condval_80 = 0;
  }
  if (!((tilelang_fused_chunk_gdr_fwd_kernel_chunk_offsets_is_null || (condval_80 == (((DLTensor*)q_handle)[0].device.device_id))))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input chunk_offsets device_id violates packed ABI constraint");
    return -1;
  }
  int32_t condval_81;
  if (!tilelang_fused_chunk_gdr_fwd_kernel_chunk_offsets_is_null) {
    condval_81 = (((DLTensor*)chunk_offsets_handle)[0].device.device_type);
  } else {
    condval_81 = 0;
  }
  if (!((tilelang_fused_chunk_gdr_fwd_kernel_chunk_offsets_is_null || (condval_81 == 2)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input chunk_offsets device_type mismatch, expected cuda");
    return -1;
  }
  if (!(((tilelang_fused_chunk_gdr_fwd_kernel_chunk_offsets_is_null || (batch_size == -1)) || !(chunk_offsets == NULL)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input chunk_offsets data pointer is NULL");
    return -1;
  }
  uint8_t condval_82;
  if (!tilelang_fused_chunk_gdr_fwd_kernel_cp_seq_map_is_null) {
    condval_82 = (((DLTensor*)cp_seq_map_handle)[0].dtype.code);
  } else {
    condval_82 = (uint8_t)0;
  }
  uint8_t condval_83;
  if (!tilelang_fused_chunk_gdr_fwd_kernel_cp_seq_map_is_null) {
    condval_83 = (((DLTensor*)cp_seq_map_handle)[0].dtype.bits);
  } else {
    condval_83 = (uint8_t)32;
  }
  uint16_t condval_84;
  if (!tilelang_fused_chunk_gdr_fwd_kernel_cp_seq_map_is_null) {
    condval_84 = (((DLTensor*)cp_seq_map_handle)[0].dtype.lanes);
  } else {
    condval_84 = (uint16_t)1;
  }
  if (!((tilelang_fused_chunk_gdr_fwd_kernel_cp_seq_map_is_null || (((condval_82 == (uint8_t)0) && (condval_83 == (uint8_t)32)) && (condval_84 == (uint16_t)1))))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input cp_seq_map dtype mismatch, expected int32");
    return -1;
  }
  int32_t condval_85;
  if (!tilelang_fused_chunk_gdr_fwd_kernel_cp_seq_map_is_null) {
    condval_85 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_cp_seq_map_shape)[0]);
  } else {
    condval_85 = 0;
  }
  if (!((tilelang_fused_chunk_gdr_fwd_kernel_cp_seq_map_is_null || (((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_q_shape)[0]) == condval_85)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input cp_seq_map shape[0] violates packed ABI constraint");
    return -1;
  }
  int32_t condval_86;
  if ((tilelang_fused_chunk_gdr_fwd_kernel_cp_seq_map_strides == NULL)) {
    condval_86 = 1;
  } else {
    condval_86 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_cp_seq_map_strides)[0]);
  }
  if (!(((tilelang_fused_chunk_gdr_fwd_kernel_cp_seq_map_is_null || (condval_86 == 1)) || (batch_size == 1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input cp_seq_map strides[0] violates packed ABI constraint");
    return -1;
  }
  uint64_t condval_87;
  if (!tilelang_fused_chunk_gdr_fwd_kernel_cp_seq_map_is_null) {
    condval_87 = (((DLTensor*)cp_seq_map_handle)[0].byte_offset);
  } else {
    condval_87 = (uint64_t)0;
  }
  if (!((tilelang_fused_chunk_gdr_fwd_kernel_cp_seq_map_is_null || ((uint64_t)0 == condval_87)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input cp_seq_map byte_offset violates packed ABI constraint");
    return -1;
  }
  int32_t condval_88;
  if (!tilelang_fused_chunk_gdr_fwd_kernel_cp_seq_map_is_null) {
    condval_88 = (((DLTensor*)cp_seq_map_handle)[0].device.device_id);
  } else {
    condval_88 = 0;
  }
  if (!((tilelang_fused_chunk_gdr_fwd_kernel_cp_seq_map_is_null || (condval_88 == (((DLTensor*)q_handle)[0].device.device_id))))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input cp_seq_map device_id violates packed ABI constraint");
    return -1;
  }
  int32_t condval_89;
  if (!tilelang_fused_chunk_gdr_fwd_kernel_cp_seq_map_is_null) {
    condval_89 = (((DLTensor*)cp_seq_map_handle)[0].device.device_type);
  } else {
    condval_89 = 0;
  }
  if (!((tilelang_fused_chunk_gdr_fwd_kernel_cp_seq_map_is_null || (condval_89 == 2)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input cp_seq_map device_type mismatch, expected cuda");
    return -1;
  }
  if (!(((tilelang_fused_chunk_gdr_fwd_kernel_cp_seq_map_is_null || (batch_size == 0)) || !(cp_seq_map == NULL)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input cp_seq_map data pointer is NULL");
    return -1;
  }
  uint8_t condval_90;
  if (!tilelang_fused_chunk_gdr_fwd_kernel_raw_cu_seqlens_is_null) {
    condval_90 = (((DLTensor*)raw_cu_seqlens_handle)[0].dtype.code);
  } else {
    condval_90 = (uint8_t)0;
  }
  uint8_t condval_91;
  if (!tilelang_fused_chunk_gdr_fwd_kernel_raw_cu_seqlens_is_null) {
    condval_91 = (((DLTensor*)raw_cu_seqlens_handle)[0].dtype.bits);
  } else {
    condval_91 = (uint8_t)32;
  }
  uint16_t condval_92;
  if (!tilelang_fused_chunk_gdr_fwd_kernel_raw_cu_seqlens_is_null) {
    condval_92 = (((DLTensor*)raw_cu_seqlens_handle)[0].dtype.lanes);
  } else {
    condval_92 = (uint16_t)1;
  }
  if (!((tilelang_fused_chunk_gdr_fwd_kernel_raw_cu_seqlens_is_null || (((condval_90 == (uint8_t)0) && (condval_91 == (uint8_t)32)) && (condval_92 == (uint16_t)1))))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input raw_cu_seqlens dtype mismatch, expected int32");
    return -1;
  }
  int32_t condval_93;
  if (!tilelang_fused_chunk_gdr_fwd_kernel_raw_cu_seqlens_is_null) {
    condval_93 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_raw_cu_seqlens_shape)[0]);
  } else {
    condval_93 = 0;
  }
  if (!((tilelang_fused_chunk_gdr_fwd_kernel_raw_cu_seqlens_is_null || (condval_93 == (raw_batch_size + 1))))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input raw_cu_seqlens shape[0] violates packed ABI constraint");
    return -1;
  }
  int32_t condval_94;
  if ((tilelang_fused_chunk_gdr_fwd_kernel_raw_cu_seqlens_strides == NULL)) {
    condval_94 = 1;
  } else {
    condval_94 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_raw_cu_seqlens_strides)[0]);
  }
  if (!(((tilelang_fused_chunk_gdr_fwd_kernel_raw_cu_seqlens_is_null || (condval_94 == 1)) || (raw_batch_size == 0)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input raw_cu_seqlens strides[0] violates packed ABI constraint");
    return -1;
  }
  uint64_t condval_95;
  if (!tilelang_fused_chunk_gdr_fwd_kernel_raw_cu_seqlens_is_null) {
    condval_95 = (((DLTensor*)raw_cu_seqlens_handle)[0].byte_offset);
  } else {
    condval_95 = (uint64_t)0;
  }
  if (!((tilelang_fused_chunk_gdr_fwd_kernel_raw_cu_seqlens_is_null || ((uint64_t)0 == condval_95)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input raw_cu_seqlens byte_offset violates packed ABI constraint");
    return -1;
  }
  int32_t condval_96;
  if (!tilelang_fused_chunk_gdr_fwd_kernel_raw_cu_seqlens_is_null) {
    condval_96 = (((DLTensor*)raw_cu_seqlens_handle)[0].device.device_id);
  } else {
    condval_96 = 0;
  }
  if (!((tilelang_fused_chunk_gdr_fwd_kernel_raw_cu_seqlens_is_null || (condval_96 == (((DLTensor*)q_handle)[0].device.device_id))))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input raw_cu_seqlens device_id violates packed ABI constraint");
    return -1;
  }
  int32_t condval_97;
  if (!tilelang_fused_chunk_gdr_fwd_kernel_raw_cu_seqlens_is_null) {
    condval_97 = (((DLTensor*)raw_cu_seqlens_handle)[0].device.device_type);
  } else {
    condval_97 = 0;
  }
  if (!((tilelang_fused_chunk_gdr_fwd_kernel_raw_cu_seqlens_is_null || (condval_97 == 2)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input raw_cu_seqlens device_type mismatch, expected cuda");
    return -1;
  }
  if (!(((tilelang_fused_chunk_gdr_fwd_kernel_raw_cu_seqlens_is_null || (raw_batch_size == -1)) || !(raw_cu_seqlens == NULL)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input raw_cu_seqlens data pointer is NULL");
    return -1;
  }
  if (!(((((((DLTensor*)o_handle)[0].dtype.code) == (uint8_t)4) && ((((DLTensor*)o_handle)[0].dtype.bits) == (uint8_t)16)) && ((((DLTensor*)o_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input o dtype mismatch, expected bfloat16");
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_q_shape)[0]) == ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_o_shape)[0])))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_fwd_kernel input o shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_q_shape)[0])), (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_o_shape)[0])));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_q_shape)[1]) == ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_o_shape)[1])))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_fwd_kernel input o shape[1] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_q_shape)[1])), (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_o_shape)[1])));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_o_shape)[2]) == 4))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_fwd_kernel input o shape[2] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_o_shape)[2])), (long long)(4));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_o_shape)[3]) == 128))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_fwd_kernel input o shape[3] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_o_shape)[3])), (long long)(128));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_98;
  if ((tilelang_fused_chunk_gdr_fwd_kernel_o_strides == NULL)) {
    condval_98 = 1;
  } else {
    condval_98 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_o_strides)[3]);
  }
  if (!((condval_98 == 1))) {
    int32_t condval_99;
    if ((tilelang_fused_chunk_gdr_fwd_kernel_o_strides == NULL)) {
      condval_99 = 1;
    } else {
      condval_99 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_o_strides)[3]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_fwd_kernel input o strides[3] violates packed ABI constraint", (long long)(condval_99), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_100;
  if ((tilelang_fused_chunk_gdr_fwd_kernel_o_strides == NULL)) {
    condval_100 = 1;
  } else {
    condval_100 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_o_strides)[2]);
  }
  if (!((condval_100 == 128))) {
    int32_t condval_101;
    if ((tilelang_fused_chunk_gdr_fwd_kernel_o_strides == NULL)) {
      condval_101 = 1;
    } else {
      condval_101 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_o_strides)[2]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_fwd_kernel input o strides[2] violates packed ABI constraint", (long long)(condval_101), (long long)(128));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_102;
  if ((tilelang_fused_chunk_gdr_fwd_kernel_o_strides == NULL)) {
    condval_102 = 1;
  } else {
    condval_102 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_o_strides)[1]);
  }
  if (!(((condval_102 == 512) || (num_tokens == 1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input o strides[1] violates packed ABI constraint");
    return -1;
  }
  int32_t condval_103;
  if ((tilelang_fused_chunk_gdr_fwd_kernel_o_strides == NULL)) {
    condval_103 = 1;
  } else {
    condval_103 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_o_strides)[0]);
  }
  if (!(((((num_tokens * 512) == condval_103) || (num_tokens == 0)) || (batch_size == 1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input o strides[0] violates packed ABI constraint");
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)o_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_fwd_kernel input o byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)o_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)o_handle)[0].device.device_id) == (((DLTensor*)q_handle)[0].device.device_id)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_fwd_kernel input o device_id violates packed ABI constraint", (long long)((((DLTensor*)o_handle)[0].device.device_id)), (long long)((((DLTensor*)q_handle)[0].device.device_id)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)o_handle)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_fwd_kernel input o device_type mismatch, expected cuda", (long long)((((DLTensor*)o_handle)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((batch_size == 0) || (num_tokens == 0)) || !(o == NULL)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input o data pointer is NULL");
    return -1;
  }
  uint8_t condval_104;
  if (!tilelang_fused_chunk_gdr_fwd_kernel_h_is_null) {
    condval_104 = (((DLTensor*)h_handle)[0].dtype.code);
  } else {
    condval_104 = (uint8_t)4;
  }
  uint8_t condval_105;
  if (!tilelang_fused_chunk_gdr_fwd_kernel_h_is_null) {
    condval_105 = (((DLTensor*)h_handle)[0].dtype.bits);
  } else {
    condval_105 = (uint8_t)16;
  }
  uint16_t condval_106;
  if (!tilelang_fused_chunk_gdr_fwd_kernel_h_is_null) {
    condval_106 = (((DLTensor*)h_handle)[0].dtype.lanes);
  } else {
    condval_106 = (uint16_t)1;
  }
  if (!((tilelang_fused_chunk_gdr_fwd_kernel_h_is_null || (((condval_104 == (uint8_t)4) && (condval_105 == (uint8_t)16)) && (condval_106 == (uint16_t)1))))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input h dtype mismatch, expected bfloat16");
    return -1;
  }
  int32_t condval_107;
  if (!tilelang_fused_chunk_gdr_fwd_kernel_h_is_null) {
    condval_107 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_h_shape)[0]);
  } else {
    condval_107 = 0;
  }
  if (!((tilelang_fused_chunk_gdr_fwd_kernel_h_is_null || (((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_q_shape)[0]) == condval_107)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input h shape[0] violates packed ABI constraint");
    return -1;
  }
  int32_t condval_108;
  if (!tilelang_fused_chunk_gdr_fwd_kernel_h_is_null) {
    condval_108 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_h_shape)[2]);
  } else {
    condval_108 = 0;
  }
  if (!((tilelang_fused_chunk_gdr_fwd_kernel_h_is_null || (condval_108 == 4)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input h shape[2] violates packed ABI constraint");
    return -1;
  }
  int32_t condval_109;
  if (!tilelang_fused_chunk_gdr_fwd_kernel_h_is_null) {
    condval_109 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_h_shape)[3]);
  } else {
    condval_109 = 0;
  }
  if (!((tilelang_fused_chunk_gdr_fwd_kernel_h_is_null || (condval_109 == 128)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input h shape[3] violates packed ABI constraint");
    return -1;
  }
  int32_t condval_110;
  if (!tilelang_fused_chunk_gdr_fwd_kernel_h_is_null) {
    condval_110 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_h_shape)[4]);
  } else {
    condval_110 = 0;
  }
  if (!((tilelang_fused_chunk_gdr_fwd_kernel_h_is_null || (condval_110 == 128)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input h shape[4] violates packed ABI constraint");
    return -1;
  }
  int32_t condval_111;
  if ((tilelang_fused_chunk_gdr_fwd_kernel_h_strides == NULL)) {
    condval_111 = 1;
  } else {
    condval_111 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_h_strides)[4]);
  }
  if (!((tilelang_fused_chunk_gdr_fwd_kernel_h_is_null || (condval_111 == 1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input h strides[4] violates packed ABI constraint");
    return -1;
  }
  int32_t condval_112;
  if ((tilelang_fused_chunk_gdr_fwd_kernel_h_strides == NULL)) {
    condval_112 = 1;
  } else {
    condval_112 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_h_strides)[3]);
  }
  if (!((tilelang_fused_chunk_gdr_fwd_kernel_h_is_null || (condval_112 == 128)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input h strides[3] violates packed ABI constraint");
    return -1;
  }
  int32_t condval_113;
  if ((tilelang_fused_chunk_gdr_fwd_kernel_h_strides == NULL)) {
    condval_113 = 1;
  } else {
    condval_113 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_h_strides)[2]);
  }
  if (!((tilelang_fused_chunk_gdr_fwd_kernel_h_is_null || (condval_113 == 16384)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input h strides[2] violates packed ABI constraint");
    return -1;
  }
  int32_t condval_114;
  if ((tilelang_fused_chunk_gdr_fwd_kernel_h_strides == NULL)) {
    condval_114 = 1;
  } else {
    condval_114 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_h_strides)[1]);
  }
  if (!(((tilelang_fused_chunk_gdr_fwd_kernel_h_is_null || (condval_114 == 65536)) || (num_chunks == 1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input h strides[1] violates packed ABI constraint");
    return -1;
  }
  int32_t condval_115;
  if ((tilelang_fused_chunk_gdr_fwd_kernel_h_strides == NULL)) {
    condval_115 = 1;
  } else {
    condval_115 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_h_strides)[0]);
  }
  if (!((((tilelang_fused_chunk_gdr_fwd_kernel_h_is_null || ((num_chunks * 65536) == condval_115)) || (num_chunks == 0)) || (batch_size == 1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input h strides[0] violates packed ABI constraint");
    return -1;
  }
  uint64_t condval_116;
  if (!tilelang_fused_chunk_gdr_fwd_kernel_h_is_null) {
    condval_116 = (((DLTensor*)h_handle)[0].byte_offset);
  } else {
    condval_116 = (uint64_t)0;
  }
  if (!((tilelang_fused_chunk_gdr_fwd_kernel_h_is_null || ((uint64_t)0 == condval_116)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input h byte_offset violates packed ABI constraint");
    return -1;
  }
  int32_t condval_117;
  if (!tilelang_fused_chunk_gdr_fwd_kernel_h_is_null) {
    condval_117 = (((DLTensor*)h_handle)[0].device.device_id);
  } else {
    condval_117 = 0;
  }
  if (!((tilelang_fused_chunk_gdr_fwd_kernel_h_is_null || (condval_117 == (((DLTensor*)q_handle)[0].device.device_id))))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input h device_id violates packed ABI constraint");
    return -1;
  }
  int32_t condval_118;
  if (!tilelang_fused_chunk_gdr_fwd_kernel_h_is_null) {
    condval_118 = (((DLTensor*)h_handle)[0].device.device_type);
  } else {
    condval_118 = 0;
  }
  if (!((tilelang_fused_chunk_gdr_fwd_kernel_h_is_null || (condval_118 == 2)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input h device_type mismatch, expected cuda");
    return -1;
  }
  if (!((((tilelang_fused_chunk_gdr_fwd_kernel_h_is_null || (batch_size == 0)) || (num_chunks == 0)) || !(h == NULL)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input h data pointer is NULL");
    return -1;
  }
  if (!(((((((DLTensor*)ht_handle)[0].dtype.code) == (uint8_t)2) && ((((DLTensor*)ht_handle)[0].dtype.bits) == (uint8_t)32)) && ((((DLTensor*)ht_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input ht dtype mismatch, expected float32");
    return -1;
  }
  int32_t condval_119;
  if (!tilelang_fused_chunk_gdr_fwd_kernel_raw_cu_seqlens_is_null) {
    condval_119 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_raw_cu_seqlens_shape)[0]);
  } else {
    condval_119 = 0;
  }
  if (!(((condval_119 - 1) == ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_ht_shape)[0])))) {
    int32_t condval_120;
    if (!tilelang_fused_chunk_gdr_fwd_kernel_raw_cu_seqlens_is_null) {
      condval_120 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_raw_cu_seqlens_shape)[0]);
    } else {
      condval_120 = 0;
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_fwd_kernel input ht shape[0] violates packed ABI constraint", (long long)((condval_120 - 1)), (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_ht_shape)[0])));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_ht_shape)[1]) == 4))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_fwd_kernel input ht shape[1] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_ht_shape)[1])), (long long)(4));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_ht_shape)[2]) == 128))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_fwd_kernel input ht shape[2] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_ht_shape)[2])), (long long)(128));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_ht_shape)[3]) == 128))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_fwd_kernel input ht shape[3] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_ht_shape)[3])), (long long)(128));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_121;
  if ((tilelang_fused_chunk_gdr_fwd_kernel_ht_strides == NULL)) {
    condval_121 = 1;
  } else {
    condval_121 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_ht_strides)[3]);
  }
  if (!((condval_121 == 1))) {
    int32_t condval_122;
    if ((tilelang_fused_chunk_gdr_fwd_kernel_ht_strides == NULL)) {
      condval_122 = 1;
    } else {
      condval_122 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_ht_strides)[3]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_fwd_kernel input ht strides[3] violates packed ABI constraint", (long long)(condval_122), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_123;
  if ((tilelang_fused_chunk_gdr_fwd_kernel_ht_strides == NULL)) {
    condval_123 = 1;
  } else {
    condval_123 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_ht_strides)[2]);
  }
  if (!((condval_123 == 128))) {
    int32_t condval_124;
    if ((tilelang_fused_chunk_gdr_fwd_kernel_ht_strides == NULL)) {
      condval_124 = 1;
    } else {
      condval_124 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_ht_strides)[2]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_fwd_kernel input ht strides[2] violates packed ABI constraint", (long long)(condval_124), (long long)(128));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_125;
  if ((tilelang_fused_chunk_gdr_fwd_kernel_ht_strides == NULL)) {
    condval_125 = 1;
  } else {
    condval_125 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_ht_strides)[1]);
  }
  if (!((condval_125 == 16384))) {
    int32_t condval_126;
    if ((tilelang_fused_chunk_gdr_fwd_kernel_ht_strides == NULL)) {
      condval_126 = 1;
    } else {
      condval_126 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_ht_strides)[1]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_fwd_kernel input ht strides[1] violates packed ABI constraint", (long long)(condval_126), (long long)(16384));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_127;
  if ((tilelang_fused_chunk_gdr_fwd_kernel_ht_strides == NULL)) {
    condval_127 = 1;
  } else {
    condval_127 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_fwd_kernel_ht_strides)[0]);
  }
  if (!(((condval_127 == 65536) || (raw_batch_size == 1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input ht strides[0] violates packed ABI constraint");
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)ht_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_fwd_kernel input ht byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)ht_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)ht_handle)[0].device.device_id) == (((DLTensor*)q_handle)[0].device.device_id)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_fwd_kernel input ht device_id violates packed ABI constraint", (long long)((((DLTensor*)ht_handle)[0].device.device_id)), (long long)((((DLTensor*)q_handle)[0].device.device_id)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)ht_handle)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_fwd_kernel input ht device_type mismatch, expected cuda", (long long)((((DLTensor*)ht_handle)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((raw_batch_size == 0) || !(ht == NULL)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_fwd_kernel input ht data pointer is NULL");
    return -1;
  }
  (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = ((int64_t)2);
  (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = ((int64_t)dev_id);
  (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 0;
  (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = (int64_t)0;
  if (__tvm_set_device_packed == NULL) {
    if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_set_device", &__tvm_set_device_packed) != 0) {
      return -1;
    }
  }
  TVMFFIAny result_1;
  result_1.type_index = kTVMFFINone;
  result_1.zero_padding = 0;
  result_1.v_int64 = 0;
  if (TVMFFIFunctionCall(__tvm_set_device_packed, (TVMFFIAny*) stack_ffi_any, 2, &result_1) != 0) {
    return -1;
  }
  TL_ALIGN(128) TVMFFIAny stack_1[16];
  void* q_desc = stack_1;
  if (q_desc == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = q_desc;
  (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = ((int64_t)9);
  (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = ((int64_t)4);
  if (q == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[3].v_ptr) = q;
  (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)128);
  (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = ((int64_t)num_tokens);
  (((TVMFFIAny*)stack_ffi_any)[6].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[6].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[6].v_int64) = ((int64_t)2);
  (((TVMFFIAny*)stack_ffi_any)[7].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[7].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[7].v_int64) = ((int64_t)batch_size);
  (((TVMFFIAny*)stack_ffi_any)[8].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[8].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[8].v_int64) = ((int64_t)2);
  (((TVMFFIAny*)stack_ffi_any)[9].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[9].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[9].v_int64) = ((int64_t)512);
  (((TVMFFIAny*)stack_ffi_any)[10].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[10].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[10].v_int64) = ((int64_t)256);
  (((TVMFFIAny*)stack_ffi_any)[11].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[11].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[11].v_int64) = (((int64_t)num_tokens) * (int64_t)512);
  (((TVMFFIAny*)stack_ffi_any)[12].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[12].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[12].v_int64) = ((int64_t)64);
  (((TVMFFIAny*)stack_ffi_any)[13].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[13].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[13].v_int64) = ((int64_t)32);
  (((TVMFFIAny*)stack_ffi_any)[14].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[14].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[14].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[15].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[15].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[15].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[16].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[16].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[16].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[17].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[17].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[17].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[18].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[18].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[18].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[19].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[19].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[19].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[20].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[20].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[20].v_int64) = ((int64_t)0);
  (((TVMFFIAny*)stack_ffi_any)[21].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[21].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[21].v_int64) = ((int64_t)3);
  (((TVMFFIAny*)stack_ffi_any)[22].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[22].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[22].v_int64) = ((int64_t)2);
  (((TVMFFIAny*)stack_ffi_any)[23].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[23].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[23].v_int64) = ((int64_t)0);
  (((TVMFFIAny*)stack_ffi_any)[24].type_index) = 0;
  (((TVMFFIAny*)stack_ffi_any)[24].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[24].v_int64) = (int64_t)0;
  if (__tvm_tensormap_create_tiled_packed == NULL) {
    if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_tensormap_create_tiled", &__tvm_tensormap_create_tiled_packed) != 0) {
      return -1;
    }
  }
  TVMFFIAny result_2;
  result_2.type_index = kTVMFFINone;
  result_2.zero_padding = 0;
  result_2.v_int64 = 0;
  if (TVMFFIFunctionCall(__tvm_tensormap_create_tiled_packed, (TVMFFIAny*) stack_ffi_any, 24, &result_2) != 0) {
    return -1;
  }
  TL_ALIGN(128) TVMFFIAny stack_2[16];
  void* k_desc = stack_2;
  if (k_desc == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = k_desc;
  (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = ((int64_t)9);
  (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = ((int64_t)4);
  if (k == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[3].v_ptr) = k;
  (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)128);
  (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = ((int64_t)num_tokens);
  (((TVMFFIAny*)stack_ffi_any)[6].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[6].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[6].v_int64) = ((int64_t)2);
  (((TVMFFIAny*)stack_ffi_any)[7].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[7].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[7].v_int64) = ((int64_t)batch_size);
  (((TVMFFIAny*)stack_ffi_any)[8].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[8].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[8].v_int64) = ((int64_t)2);
  (((TVMFFIAny*)stack_ffi_any)[9].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[9].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[9].v_int64) = ((int64_t)512);
  (((TVMFFIAny*)stack_ffi_any)[10].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[10].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[10].v_int64) = ((int64_t)256);
  (((TVMFFIAny*)stack_ffi_any)[11].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[11].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[11].v_int64) = (((int64_t)num_tokens) * (int64_t)512);
  (((TVMFFIAny*)stack_ffi_any)[12].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[12].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[12].v_int64) = ((int64_t)64);
  (((TVMFFIAny*)stack_ffi_any)[13].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[13].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[13].v_int64) = ((int64_t)32);
  (((TVMFFIAny*)stack_ffi_any)[14].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[14].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[14].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[15].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[15].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[15].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[16].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[16].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[16].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[17].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[17].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[17].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[18].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[18].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[18].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[19].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[19].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[19].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[20].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[20].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[20].v_int64) = ((int64_t)0);
  (((TVMFFIAny*)stack_ffi_any)[21].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[21].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[21].v_int64) = ((int64_t)3);
  (((TVMFFIAny*)stack_ffi_any)[22].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[22].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[22].v_int64) = ((int64_t)2);
  (((TVMFFIAny*)stack_ffi_any)[23].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[23].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[23].v_int64) = ((int64_t)0);
  (((TVMFFIAny*)stack_ffi_any)[24].type_index) = 0;
  (((TVMFFIAny*)stack_ffi_any)[24].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[24].v_int64) = (int64_t)0;
  if (__tvm_tensormap_create_tiled_packed == NULL) {
    if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_tensormap_create_tiled", &__tvm_tensormap_create_tiled_packed) != 0) {
      return -1;
    }
  }
  TVMFFIAny result_3;
  result_3.type_index = kTVMFFINone;
  result_3.zero_padding = 0;
  result_3.v_int64 = 0;
  if (TVMFFIFunctionCall(__tvm_tensormap_create_tiled_packed, (TVMFFIAny*) stack_ffi_any, 24, &result_3) != 0) {
    return -1;
  }
  TL_ALIGN(128) TVMFFIAny stack_3[16];
  void* v_desc = stack_3;
  if (v_desc == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = v_desc;
  (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = ((int64_t)9);
  (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = ((int64_t)4);
  if (v == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[3].v_ptr) = v;
  (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)128);
  (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = ((int64_t)num_tokens);
  (((TVMFFIAny*)stack_ffi_any)[6].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[6].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[6].v_int64) = ((int64_t)4);
  (((TVMFFIAny*)stack_ffi_any)[7].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[7].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[7].v_int64) = ((int64_t)batch_size);
  (((TVMFFIAny*)stack_ffi_any)[8].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[8].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[8].v_int64) = ((int64_t)2);
  (((TVMFFIAny*)stack_ffi_any)[9].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[9].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[9].v_int64) = ((int64_t)1024);
  (((TVMFFIAny*)stack_ffi_any)[10].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[10].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[10].v_int64) = ((int64_t)256);
  (((TVMFFIAny*)stack_ffi_any)[11].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[11].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[11].v_int64) = (((int64_t)num_tokens) * (int64_t)1024);
  (((TVMFFIAny*)stack_ffi_any)[12].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[12].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[12].v_int64) = ((int64_t)64);
  (((TVMFFIAny*)stack_ffi_any)[13].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[13].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[13].v_int64) = ((int64_t)32);
  (((TVMFFIAny*)stack_ffi_any)[14].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[14].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[14].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[15].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[15].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[15].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[16].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[16].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[16].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[17].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[17].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[17].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[18].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[18].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[18].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[19].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[19].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[19].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[20].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[20].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[20].v_int64) = ((int64_t)0);
  (((TVMFFIAny*)stack_ffi_any)[21].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[21].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[21].v_int64) = ((int64_t)3);
  (((TVMFFIAny*)stack_ffi_any)[22].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[22].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[22].v_int64) = ((int64_t)2);
  (((TVMFFIAny*)stack_ffi_any)[23].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[23].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[23].v_int64) = ((int64_t)0);
  (((TVMFFIAny*)stack_ffi_any)[24].type_index) = 0;
  (((TVMFFIAny*)stack_ffi_any)[24].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[24].v_int64) = (int64_t)0;
  if (__tvm_tensormap_create_tiled_packed == NULL) {
    if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_tensormap_create_tiled", &__tvm_tensormap_create_tiled_packed) != 0) {
      return -1;
    }
  }
  TVMFFIAny result_4;
  result_4.type_index = kTVMFFINone;
  result_4.zero_padding = 0;
  result_4.v_int64 = 0;
  if (TVMFFIFunctionCall(__tvm_tensormap_create_tiled_packed, (TVMFFIAny*) stack_ffi_any, 24, &result_4) != 0) {
    return -1;
  }
  TL_ALIGN(128) TVMFFIAny stack_4[16];
  void* a_desc = stack_4;
  if (a_desc == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = a_desc;
  (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = ((int64_t)9);
  (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = ((int64_t)4);
  if (a == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[3].v_ptr) = a;
  (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)32);
  (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = ((int64_t)num_tokens);
  (((TVMFFIAny*)stack_ffi_any)[6].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[6].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[6].v_int64) = ((int64_t)4);
  (((TVMFFIAny*)stack_ffi_any)[7].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[7].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[7].v_int64) = ((int64_t)batch_size);
  (((TVMFFIAny*)stack_ffi_any)[8].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[8].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[8].v_int64) = ((int64_t)2);
  (((TVMFFIAny*)stack_ffi_any)[9].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[9].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[9].v_int64) = ((int64_t)256);
  (((TVMFFIAny*)stack_ffi_any)[10].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[10].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[10].v_int64) = ((int64_t)64);
  (((TVMFFIAny*)stack_ffi_any)[11].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[11].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[11].v_int64) = (((int64_t)num_tokens) * (int64_t)256);
  (((TVMFFIAny*)stack_ffi_any)[12].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[12].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[12].v_int64) = ((int64_t)32);
  (((TVMFFIAny*)stack_ffi_any)[13].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[13].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[13].v_int64) = ((int64_t)32);
  (((TVMFFIAny*)stack_ffi_any)[14].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[14].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[14].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[15].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[15].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[15].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[16].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[16].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[16].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[17].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[17].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[17].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[18].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[18].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[18].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[19].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[19].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[19].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[20].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[20].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[20].v_int64) = ((int64_t)0);
  (((TVMFFIAny*)stack_ffi_any)[21].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[21].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[21].v_int64) = ((int64_t)2);
  (((TVMFFIAny*)stack_ffi_any)[22].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[22].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[22].v_int64) = ((int64_t)2);
  (((TVMFFIAny*)stack_ffi_any)[23].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[23].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[23].v_int64) = ((int64_t)0);
  (((TVMFFIAny*)stack_ffi_any)[24].type_index) = 0;
  (((TVMFFIAny*)stack_ffi_any)[24].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[24].v_int64) = (int64_t)0;
  if (__tvm_tensormap_create_tiled_packed == NULL) {
    if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_tensormap_create_tiled", &__tvm_tensormap_create_tiled_packed) != 0) {
      return -1;
    }
  }
  TVMFFIAny result_5;
  result_5.type_index = kTVMFFINone;
  result_5.zero_padding = 0;
  result_5.v_int64 = 0;
  if (TVMFFIFunctionCall(__tvm_tensormap_create_tiled_packed, (TVMFFIAny*) stack_ffi_any, 24, &result_5) != 0) {
    return -1;
  }
  TL_ALIGN(128) TVMFFIAny stack_5[16];
  void* o_desc = stack_5;
  if (o_desc == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = o_desc;
  (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = ((int64_t)9);
  (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = ((int64_t)4);
  if (o == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[3].v_ptr) = o;
  (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)128);
  (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = ((int64_t)num_tokens);
  (((TVMFFIAny*)stack_ffi_any)[6].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[6].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[6].v_int64) = ((int64_t)4);
  (((TVMFFIAny*)stack_ffi_any)[7].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[7].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[7].v_int64) = ((int64_t)batch_size);
  (((TVMFFIAny*)stack_ffi_any)[8].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[8].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[8].v_int64) = ((int64_t)2);
  (((TVMFFIAny*)stack_ffi_any)[9].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[9].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[9].v_int64) = ((int64_t)1024);
  (((TVMFFIAny*)stack_ffi_any)[10].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[10].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[10].v_int64) = ((int64_t)256);
  (((TVMFFIAny*)stack_ffi_any)[11].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[11].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[11].v_int64) = (((int64_t)num_tokens) * (int64_t)1024);
  (((TVMFFIAny*)stack_ffi_any)[12].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[12].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[12].v_int64) = ((int64_t)64);
  (((TVMFFIAny*)stack_ffi_any)[13].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[13].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[13].v_int64) = ((int64_t)32);
  (((TVMFFIAny*)stack_ffi_any)[14].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[14].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[14].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[15].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[15].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[15].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[16].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[16].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[16].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[17].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[17].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[17].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[18].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[18].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[18].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[19].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[19].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[19].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[20].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[20].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[20].v_int64) = ((int64_t)0);
  (((TVMFFIAny*)stack_ffi_any)[21].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[21].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[21].v_int64) = ((int64_t)3);
  (((TVMFFIAny*)stack_ffi_any)[22].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[22].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[22].v_int64) = ((int64_t)2);
  (((TVMFFIAny*)stack_ffi_any)[23].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[23].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[23].v_int64) = ((int64_t)0);
  (((TVMFFIAny*)stack_ffi_any)[24].type_index) = 0;
  (((TVMFFIAny*)stack_ffi_any)[24].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[24].v_int64) = (int64_t)0;
  if (__tvm_tensormap_create_tiled_packed == NULL) {
    if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_tensormap_create_tiled", &__tvm_tensormap_create_tiled_packed) != 0) {
      return -1;
    }
  }
  TVMFFIAny result_6;
  result_6.type_index = kTVMFFINone;
  result_6.zero_padding = 0;
  result_6.v_int64 = 0;
  if (TVMFFIFunctionCall(__tvm_tensormap_create_tiled_packed, (TVMFFIAny*) stack_ffi_any, 24, &result_6) != 0) {
    return -1;
  }
  if (a == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = a;
  if (a_desc == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = a_desc;
  if (b == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = b;
  if (g == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[3].v_ptr) = g;
  if (ht == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[4].v_ptr) = ht;
  if (k == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[5].v_ptr) = k;
  if (k_desc == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[6].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[6].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[6].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[6].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[6].v_ptr) = k_desc;
  if (o == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[7].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[7].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[7].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[7].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[7].v_ptr) = o;
  if (o_desc == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[8].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[8].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[8].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[8].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[8].v_ptr) = o_desc;
  if (q == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[9].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[9].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[9].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[9].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[9].v_ptr) = q;
  if (q_desc == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[10].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[10].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[10].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[10].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[10].v_ptr) = q_desc;
  if (v == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[11].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[11].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[11].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[11].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[11].v_ptr) = v;
  if (v_desc == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[12].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[12].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[12].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[12].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[12].v_ptr) = v_desc;
  (((TVMFFIAny*)stack_ffi_any)[13].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[13].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[13].v_int64) = ((int64_t)batch_size);
  (((TVMFFIAny*)stack_ffi_any)[14].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[14].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[14].v_int64) = ((int64_t)num_tokens);
  (((TVMFFIAny*)stack_ffi_any)[15].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[15].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[15].v_int64) = ((int64_t)raw_batch_size);
  (((TVMFFIAny*)stack_ffi_any)[16].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[16].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[16].v_int64) = ((int64_t)(batch_size * 8));
  (((TVMFFIAny*)stack_ffi_any)[17].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[17].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[17].v_int64) = ((int64_t)512);
  (((TVMFFIAny*)stack_ffi_any)[18].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[18].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[18].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[19].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[19].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[19].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[20].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[20].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[20].v_int64) = ((int64_t)75776);
  (((TVMFFIAny*)stack_ffi_any)[21].type_index) = 0;
  (((TVMFFIAny*)stack_ffi_any)[21].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[21].v_int64) = (int64_t)0;
  if (tilelang_fused_chunk_gdr_fwd_kernel_kernel_packed == NULL) {
    if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "tilelang_fused_chunk_gdr_fwd_kernel_kernel", &tilelang_fused_chunk_gdr_fwd_kernel_kernel_packed) != 0) {
      return -1;
    }
  }
  TVMFFIAny result_7;
  result_7.type_index = kTVMFFINone;
  result_7.zero_padding = 0;
  result_7.v_int64 = 0;
  if (TVMFFIFunctionCall(tilelang_fused_chunk_gdr_fwd_kernel_kernel_packed, (TVMFFIAny*) stack_ffi_any, 21, &result_7) != 0) {
    return -1;
  }
  return 0;
}

// CodegenC: NOTE: Auto-generated entry function
#ifdef __cplusplus
extern "C"
#endif
int32_t __tvm_ffi_main(void* self, void* args,int num_args, void* result) {
  return __tvm_ffi_tilelang_fused_chunk_gdr_fwd_kernel(self, args, num_args, result);
}
