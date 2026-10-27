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
static void* tilelang_fused_chunk_gdr_bwd_kernel_kernel_packed = NULL;
#ifdef __cplusplus
extern "C"
#endif
int32_t __tvm_ffi_tilelang_fused_chunk_gdr_bwd_kernel(void* self_handle, void* args, int32_t num_args, void* result);
#ifdef __cplusplus
extern "C"
#endif
int32_t __tvm_ffi_tilelang_fused_chunk_gdr_bwd_kernel(void* self_handle, void* args, int32_t num_args, void* result) {
  TL_ALIGN(128) TVMFFIAny stack[34];
  void* stack_ffi_any = stack;
  if (!((num_args == 17))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "tilelang_fused_chunk_gdr_bwd_kernel: num_args should be 17", (long long)(num_args), (long long)(17));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(!(args == NULL))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "tilelang_fused_chunk_gdr_bwd_kernel: args pointer is NULL");
    return -1;
  }
  int32_t do_handle_type_index = (((TVMFFIAny*)args)[0].type_index);
  if (!(((((do_handle_type_index == 0) || (do_handle_type_index == 4)) || (do_handle_type_index == 7)) || (64 <= do_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_bwd_kernel input do expected pointer or tensor handle");
    return -1;
  }
  int32_t dht_handle_type_index = (((TVMFFIAny*)args)[1].type_index);
  if (!(((((dht_handle_type_index == 0) || (dht_handle_type_index == 4)) || (dht_handle_type_index == 7)) || (64 <= dht_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_bwd_kernel input dht expected pointer or tensor handle");
    return -1;
  }
  int32_t q_handle_type_index = (((TVMFFIAny*)args)[2].type_index);
  if (!(((((q_handle_type_index == 0) || (q_handle_type_index == 4)) || (q_handle_type_index == 7)) || (64 <= q_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_bwd_kernel input q expected pointer or tensor handle");
    return -1;
  }
  int32_t k_handle_type_index = (((TVMFFIAny*)args)[3].type_index);
  if (!(((((k_handle_type_index == 0) || (k_handle_type_index == 4)) || (k_handle_type_index == 7)) || (64 <= k_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_bwd_kernel input k expected pointer or tensor handle");
    return -1;
  }
  int32_t v_handle_type_index = (((TVMFFIAny*)args)[4].type_index);
  if (!(((((v_handle_type_index == 0) || (v_handle_type_index == 4)) || (v_handle_type_index == 7)) || (64 <= v_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_bwd_kernel input v expected pointer or tensor handle");
    return -1;
  }
  int32_t a_handle_type_index = (((TVMFFIAny*)args)[5].type_index);
  if (!(((((a_handle_type_index == 0) || (a_handle_type_index == 4)) || (a_handle_type_index == 7)) || (64 <= a_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_bwd_kernel input a expected pointer or tensor handle");
    return -1;
  }
  int32_t g_handle_type_index = (((TVMFFIAny*)args)[6].type_index);
  if (!(((((g_handle_type_index == 0) || (g_handle_type_index == 4)) || (g_handle_type_index == 7)) || (64 <= g_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_bwd_kernel input g expected pointer or tensor handle");
    return -1;
  }
  int32_t b_handle_type_index = (((TVMFFIAny*)args)[7].type_index);
  if (!(((((b_handle_type_index == 0) || (b_handle_type_index == 4)) || (b_handle_type_index == 7)) || (64 <= b_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_bwd_kernel input b expected pointer or tensor handle");
    return -1;
  }
  int32_t h_handle_type_index = (((TVMFFIAny*)args)[8].type_index);
  if (!(((((h_handle_type_index == 0) || (h_handle_type_index == 4)) || (h_handle_type_index == 7)) || (64 <= h_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_bwd_kernel input h expected pointer or tensor handle");
    return -1;
  }
  int32_t cu_seqlens_handle_type_index = (((TVMFFIAny*)args)[9].type_index);
  if (!(((((cu_seqlens_handle_type_index == 0) || (cu_seqlens_handle_type_index == 4)) || (cu_seqlens_handle_type_index == 7)) || (64 <= cu_seqlens_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_bwd_kernel input cu_seqlens expected pointer or tensor handle");
    return -1;
  }
  int32_t chunk_offsets_handle_type_index = (((TVMFFIAny*)args)[10].type_index);
  if (!(((((chunk_offsets_handle_type_index == 0) || (chunk_offsets_handle_type_index == 4)) || (chunk_offsets_handle_type_index == 7)) || (64 <= chunk_offsets_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_bwd_kernel input chunk_offsets expected pointer or tensor handle");
    return -1;
  }
  int32_t dq_handle_type_index = (((TVMFFIAny*)args)[11].type_index);
  if (!(((((dq_handle_type_index == 0) || (dq_handle_type_index == 4)) || (dq_handle_type_index == 7)) || (64 <= dq_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_bwd_kernel input dq expected pointer or tensor handle");
    return -1;
  }
  int32_t dk_handle_type_index = (((TVMFFIAny*)args)[12].type_index);
  if (!(((((dk_handle_type_index == 0) || (dk_handle_type_index == 4)) || (dk_handle_type_index == 7)) || (64 <= dk_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_bwd_kernel input dk expected pointer or tensor handle");
    return -1;
  }
  int32_t dv_handle_type_index = (((TVMFFIAny*)args)[13].type_index);
  if (!(((((dv_handle_type_index == 0) || (dv_handle_type_index == 4)) || (dv_handle_type_index == 7)) || (64 <= dv_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_bwd_kernel input dv expected pointer or tensor handle");
    return -1;
  }
  int32_t dg_handle_type_index = (((TVMFFIAny*)args)[14].type_index);
  if (!(((((dg_handle_type_index == 0) || (dg_handle_type_index == 4)) || (dg_handle_type_index == 7)) || (64 <= dg_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_bwd_kernel input dg expected pointer or tensor handle");
    return -1;
  }
  int32_t db_handle_type_index = (((TVMFFIAny*)args)[15].type_index);
  if (!(((((db_handle_type_index == 0) || (db_handle_type_index == 4)) || (db_handle_type_index == 7)) || (64 <= db_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_bwd_kernel input db expected pointer or tensor handle");
    return -1;
  }
  int32_t dh0_handle_type_index = (((TVMFFIAny*)args)[16].type_index);
  if (!(((((dh0_handle_type_index == 0) || (dh0_handle_type_index == 4)) || (dh0_handle_type_index == 7)) || (64 <= dh0_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_bwd_kernel input dh0 expected pointer or tensor handle");
    return -1;
  }
  void* do_handle = ((do_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[0].v_ptr) + 24)) : (((TVMFFIAny*)args)[0].v_ptr));
  void* dht_handle = ((dht_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[1].v_ptr) + 24)) : (((TVMFFIAny*)args)[1].v_ptr));
  void* q_handle = ((q_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[2].v_ptr) + 24)) : (((TVMFFIAny*)args)[2].v_ptr));
  void* k_handle = ((k_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[3].v_ptr) + 24)) : (((TVMFFIAny*)args)[3].v_ptr));
  void* v_handle = ((v_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[4].v_ptr) + 24)) : (((TVMFFIAny*)args)[4].v_ptr));
  void* a_handle = ((a_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[5].v_ptr) + 24)) : (((TVMFFIAny*)args)[5].v_ptr));
  void* g_handle = ((g_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[6].v_ptr) + 24)) : (((TVMFFIAny*)args)[6].v_ptr));
  void* b_handle = ((b_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[7].v_ptr) + 24)) : (((TVMFFIAny*)args)[7].v_ptr));
  void* h_handle = ((h_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[8].v_ptr) + 24)) : (((TVMFFIAny*)args)[8].v_ptr));
  void* cu_seqlens_handle = ((cu_seqlens_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[9].v_ptr) + 24)) : (((TVMFFIAny*)args)[9].v_ptr));
  void* chunk_offsets_handle = ((chunk_offsets_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[10].v_ptr) + 24)) : (((TVMFFIAny*)args)[10].v_ptr));
  void* dq_handle = ((dq_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[11].v_ptr) + 24)) : (((TVMFFIAny*)args)[11].v_ptr));
  void* dk_handle = ((dk_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[12].v_ptr) + 24)) : (((TVMFFIAny*)args)[12].v_ptr));
  void* dv_handle = ((dv_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[13].v_ptr) + 24)) : (((TVMFFIAny*)args)[13].v_ptr));
  void* dg_handle = ((dg_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[14].v_ptr) + 24)) : (((TVMFFIAny*)args)[14].v_ptr));
  void* db_handle = ((db_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[15].v_ptr) + 24)) : (((TVMFFIAny*)args)[15].v_ptr));
  void* dh0_handle = ((dh0_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[16].v_ptr) + 24)) : (((TVMFFIAny*)args)[16].v_ptr));
  bool tilelang_fused_chunk_gdr_bwd_kernel_do_is_null = (do_handle == NULL);
  if (!(!tilelang_fused_chunk_gdr_bwd_kernel_do_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "tilelang_fused_chunk_gdr_bwd_kernel.do is expected to have non-NULL pointer");
    return -1;
  }
  bool tilelang_fused_chunk_gdr_bwd_kernel_dht_is_null = (dht_handle == NULL);
  if (!(!tilelang_fused_chunk_gdr_bwd_kernel_dht_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "tilelang_fused_chunk_gdr_bwd_kernel.dht is expected to have non-NULL pointer");
    return -1;
  }
  bool tilelang_fused_chunk_gdr_bwd_kernel_q_is_null = (q_handle == NULL);
  if (!(!tilelang_fused_chunk_gdr_bwd_kernel_q_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "tilelang_fused_chunk_gdr_bwd_kernel.q is expected to have non-NULL pointer");
    return -1;
  }
  bool tilelang_fused_chunk_gdr_bwd_kernel_k_is_null = (k_handle == NULL);
  if (!(!tilelang_fused_chunk_gdr_bwd_kernel_k_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "tilelang_fused_chunk_gdr_bwd_kernel.k is expected to have non-NULL pointer");
    return -1;
  }
  bool tilelang_fused_chunk_gdr_bwd_kernel_v_is_null = (v_handle == NULL);
  if (!(!tilelang_fused_chunk_gdr_bwd_kernel_v_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "tilelang_fused_chunk_gdr_bwd_kernel.v is expected to have non-NULL pointer");
    return -1;
  }
  bool tilelang_fused_chunk_gdr_bwd_kernel_a_is_null = (a_handle == NULL);
  if (!(!tilelang_fused_chunk_gdr_bwd_kernel_a_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "tilelang_fused_chunk_gdr_bwd_kernel.a is expected to have non-NULL pointer");
    return -1;
  }
  bool tilelang_fused_chunk_gdr_bwd_kernel_g_is_null = (g_handle == NULL);
  if (!(!tilelang_fused_chunk_gdr_bwd_kernel_g_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "tilelang_fused_chunk_gdr_bwd_kernel.g is expected to have non-NULL pointer");
    return -1;
  }
  bool tilelang_fused_chunk_gdr_bwd_kernel_b_is_null = (b_handle == NULL);
  if (!(!tilelang_fused_chunk_gdr_bwd_kernel_b_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "tilelang_fused_chunk_gdr_bwd_kernel.b is expected to have non-NULL pointer");
    return -1;
  }
  bool tilelang_fused_chunk_gdr_bwd_kernel_h_is_null = (h_handle == NULL);
  if (!(!tilelang_fused_chunk_gdr_bwd_kernel_h_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "tilelang_fused_chunk_gdr_bwd_kernel.h is expected to have non-NULL pointer");
    return -1;
  }
  bool tilelang_fused_chunk_gdr_bwd_kernel_cu_seqlens_is_null = (cu_seqlens_handle == NULL);
  if (!(!tilelang_fused_chunk_gdr_bwd_kernel_cu_seqlens_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "tilelang_fused_chunk_gdr_bwd_kernel.cu_seqlens is expected to have non-NULL pointer");
    return -1;
  }
  bool tilelang_fused_chunk_gdr_bwd_kernel_chunk_offsets_is_null = (chunk_offsets_handle == NULL);
  if (!(!tilelang_fused_chunk_gdr_bwd_kernel_chunk_offsets_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "tilelang_fused_chunk_gdr_bwd_kernel.chunk_offsets is expected to have non-NULL pointer");
    return -1;
  }
  bool tilelang_fused_chunk_gdr_bwd_kernel_dq_is_null = (dq_handle == NULL);
  if (!(!tilelang_fused_chunk_gdr_bwd_kernel_dq_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "tilelang_fused_chunk_gdr_bwd_kernel.dq is expected to have non-NULL pointer");
    return -1;
  }
  bool tilelang_fused_chunk_gdr_bwd_kernel_dk_is_null = (dk_handle == NULL);
  if (!(!tilelang_fused_chunk_gdr_bwd_kernel_dk_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "tilelang_fused_chunk_gdr_bwd_kernel.dk is expected to have non-NULL pointer");
    return -1;
  }
  bool tilelang_fused_chunk_gdr_bwd_kernel_dv_is_null = (dv_handle == NULL);
  if (!(!tilelang_fused_chunk_gdr_bwd_kernel_dv_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "tilelang_fused_chunk_gdr_bwd_kernel.dv is expected to have non-NULL pointer");
    return -1;
  }
  bool tilelang_fused_chunk_gdr_bwd_kernel_dg_is_null = (dg_handle == NULL);
  if (!(!tilelang_fused_chunk_gdr_bwd_kernel_dg_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "tilelang_fused_chunk_gdr_bwd_kernel.dg is expected to have non-NULL pointer");
    return -1;
  }
  bool tilelang_fused_chunk_gdr_bwd_kernel_db_is_null = (db_handle == NULL);
  if (!(!tilelang_fused_chunk_gdr_bwd_kernel_db_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "tilelang_fused_chunk_gdr_bwd_kernel.db is expected to have non-NULL pointer");
    return -1;
  }
  bool tilelang_fused_chunk_gdr_bwd_kernel_dh0_is_null = (dh0_handle == NULL);
  if (!(!tilelang_fused_chunk_gdr_bwd_kernel_dh0_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "tilelang_fused_chunk_gdr_bwd_kernel.dh0 is expected to have non-NULL pointer");
    return -1;
  }
  void* tilelang_fused_chunk_gdr_bwd_kernel_do_shape = (((DLTensor*)do_handle)[0].shape);
  void* tilelang_fused_chunk_gdr_bwd_kernel_dht_shape = (((DLTensor*)dht_handle)[0].shape);
  void* tilelang_fused_chunk_gdr_bwd_kernel_q_shape = (((DLTensor*)q_handle)[0].shape);
  void* tilelang_fused_chunk_gdr_bwd_kernel_k_shape = (((DLTensor*)k_handle)[0].shape);
  void* tilelang_fused_chunk_gdr_bwd_kernel_v_shape = (((DLTensor*)v_handle)[0].shape);
  void* tilelang_fused_chunk_gdr_bwd_kernel_a_shape = (((DLTensor*)a_handle)[0].shape);
  void* tilelang_fused_chunk_gdr_bwd_kernel_g_shape = (((DLTensor*)g_handle)[0].shape);
  void* tilelang_fused_chunk_gdr_bwd_kernel_b_shape = (((DLTensor*)b_handle)[0].shape);
  void* tilelang_fused_chunk_gdr_bwd_kernel_h_shape = (((DLTensor*)h_handle)[0].shape);
  void* tilelang_fused_chunk_gdr_bwd_kernel_cu_seqlens_shape = (((DLTensor*)cu_seqlens_handle)[0].shape);
  void* tilelang_fused_chunk_gdr_bwd_kernel_chunk_offsets_shape = (((DLTensor*)chunk_offsets_handle)[0].shape);
  void* tilelang_fused_chunk_gdr_bwd_kernel_dq_shape = (((DLTensor*)dq_handle)[0].shape);
  void* tilelang_fused_chunk_gdr_bwd_kernel_dk_shape = (((DLTensor*)dk_handle)[0].shape);
  void* tilelang_fused_chunk_gdr_bwd_kernel_dv_shape = (((DLTensor*)dv_handle)[0].shape);
  void* tilelang_fused_chunk_gdr_bwd_kernel_dg_shape = (((DLTensor*)dg_handle)[0].shape);
  void* tilelang_fused_chunk_gdr_bwd_kernel_db_shape = (((DLTensor*)db_handle)[0].shape);
  void* tilelang_fused_chunk_gdr_bwd_kernel_dh0_shape = (((DLTensor*)dh0_handle)[0].shape);
  if (!(((((DLTensor*)do_handle)[0].ndim) == 4))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input do ndim mismatch, expected 4", (long long)((((DLTensor*)do_handle)[0].ndim)), (long long)(4));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((bool)1)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "Symbolic shape variable num_tokens requires at least one non-null buffer among: tilelang_fused_chunk_gdr_bwd_kernel.do, tilelang_fused_chunk_gdr_bwd_kernel.q, tilelang_fused_chunk_gdr_bwd_kernel.k, tilelang_fused_chunk_gdr_bwd_kernel.v, tilelang_fused_chunk_gdr_bwd_kernel.a, tilelang_fused_chunk_gdr_bwd_kernel.g, tilelang_fused_chunk_gdr_bwd_kernel.b, tilelang_fused_chunk_gdr_bwd_kernel.dq, tilelang_fused_chunk_gdr_bwd_kernel.dk, tilelang_fused_chunk_gdr_bwd_kernel.dv, tilelang_fused_chunk_gdr_bwd_kernel.dg, tilelang_fused_chunk_gdr_bwd_kernel.db");
    return -1;
  }
  int32_t num_tokens = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_do_shape)[1]);
  void* tilelang_fused_chunk_gdr_bwd_kernel_do_strides = (((DLTensor*)do_handle)[0].strides);
  int32_t dev_id = (((DLTensor*)do_handle)[0].device.device_id);
  void* do_1 = (((DLTensor*)do_handle)[0].data);
  if (!(((((DLTensor*)dht_handle)[0].ndim) == 4))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input dht ndim mismatch, expected 4", (long long)((((DLTensor*)dht_handle)[0].ndim)), (long long)(4));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((bool)1)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "Symbolic shape variable batch_size requires at least one non-null buffer among: tilelang_fused_chunk_gdr_bwd_kernel.dht, tilelang_fused_chunk_gdr_bwd_kernel.dh0");
    return -1;
  }
  int32_t batch_size = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_dht_shape)[0]);
  void* tilelang_fused_chunk_gdr_bwd_kernel_dht_strides = (((DLTensor*)dht_handle)[0].strides);
  void* dht = (((DLTensor*)dht_handle)[0].data);
  if (!(((((DLTensor*)q_handle)[0].ndim) == 4))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input q ndim mismatch, expected 4", (long long)((((DLTensor*)q_handle)[0].ndim)), (long long)(4));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  void* tilelang_fused_chunk_gdr_bwd_kernel_q_strides = (((DLTensor*)q_handle)[0].strides);
  void* q = (((DLTensor*)q_handle)[0].data);
  if (!(((((DLTensor*)k_handle)[0].ndim) == 4))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input k ndim mismatch, expected 4", (long long)((((DLTensor*)k_handle)[0].ndim)), (long long)(4));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  void* tilelang_fused_chunk_gdr_bwd_kernel_k_strides = (((DLTensor*)k_handle)[0].strides);
  void* k = (((DLTensor*)k_handle)[0].data);
  if (!(((((DLTensor*)v_handle)[0].ndim) == 4))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input v ndim mismatch, expected 4", (long long)((((DLTensor*)v_handle)[0].ndim)), (long long)(4));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  void* tilelang_fused_chunk_gdr_bwd_kernel_v_strides = (((DLTensor*)v_handle)[0].strides);
  void* v = (((DLTensor*)v_handle)[0].data);
  if (!(((((DLTensor*)a_handle)[0].ndim) == 4))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input a ndim mismatch, expected 4", (long long)((((DLTensor*)a_handle)[0].ndim)), (long long)(4));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  void* tilelang_fused_chunk_gdr_bwd_kernel_a_strides = (((DLTensor*)a_handle)[0].strides);
  void* a = (((DLTensor*)a_handle)[0].data);
  if (!(((((DLTensor*)g_handle)[0].ndim) == 3))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input g ndim mismatch, expected 3", (long long)((((DLTensor*)g_handle)[0].ndim)), (long long)(3));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  void* tilelang_fused_chunk_gdr_bwd_kernel_g_strides = (((DLTensor*)g_handle)[0].strides);
  void* g = (((DLTensor*)g_handle)[0].data);
  if (!(((((DLTensor*)b_handle)[0].ndim) == 3))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input b ndim mismatch, expected 3", (long long)((((DLTensor*)b_handle)[0].ndim)), (long long)(3));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  void* tilelang_fused_chunk_gdr_bwd_kernel_b_strides = (((DLTensor*)b_handle)[0].strides);
  void* b = (((DLTensor*)b_handle)[0].data);
  if (!(((((DLTensor*)h_handle)[0].ndim) == 5))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input h ndim mismatch, expected 5", (long long)((((DLTensor*)h_handle)[0].ndim)), (long long)(5));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t num_chunks = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_h_shape)[1]);
  void* tilelang_fused_chunk_gdr_bwd_kernel_h_strides = (((DLTensor*)h_handle)[0].strides);
  void* h = (((DLTensor*)h_handle)[0].data);
  if (!(((((DLTensor*)cu_seqlens_handle)[0].ndim) == 1))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input cu_seqlens ndim mismatch, expected 1", (long long)((((DLTensor*)cu_seqlens_handle)[0].ndim)), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  void* tilelang_fused_chunk_gdr_bwd_kernel_cu_seqlens_strides = (((DLTensor*)cu_seqlens_handle)[0].strides);
  void* cu_seqlens = (((DLTensor*)cu_seqlens_handle)[0].data);
  if (!(((((DLTensor*)chunk_offsets_handle)[0].ndim) == 1))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input chunk_offsets ndim mismatch, expected 1", (long long)((((DLTensor*)chunk_offsets_handle)[0].ndim)), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  void* tilelang_fused_chunk_gdr_bwd_kernel_chunk_offsets_strides = (((DLTensor*)chunk_offsets_handle)[0].strides);
  void* chunk_offsets = (((DLTensor*)chunk_offsets_handle)[0].data);
  if (!(((((DLTensor*)dq_handle)[0].ndim) == 4))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input dq ndim mismatch, expected 4", (long long)((((DLTensor*)dq_handle)[0].ndim)), (long long)(4));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  void* tilelang_fused_chunk_gdr_bwd_kernel_dq_strides = (((DLTensor*)dq_handle)[0].strides);
  void* dq = (((DLTensor*)dq_handle)[0].data);
  if (!(((((DLTensor*)dk_handle)[0].ndim) == 4))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input dk ndim mismatch, expected 4", (long long)((((DLTensor*)dk_handle)[0].ndim)), (long long)(4));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  void* tilelang_fused_chunk_gdr_bwd_kernel_dk_strides = (((DLTensor*)dk_handle)[0].strides);
  void* dk = (((DLTensor*)dk_handle)[0].data);
  if (!(((((DLTensor*)dv_handle)[0].ndim) == 4))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input dv ndim mismatch, expected 4", (long long)((((DLTensor*)dv_handle)[0].ndim)), (long long)(4));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  void* tilelang_fused_chunk_gdr_bwd_kernel_dv_strides = (((DLTensor*)dv_handle)[0].strides);
  void* dv = (((DLTensor*)dv_handle)[0].data);
  if (!(((((DLTensor*)dg_handle)[0].ndim) == 3))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input dg ndim mismatch, expected 3", (long long)((((DLTensor*)dg_handle)[0].ndim)), (long long)(3));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  void* tilelang_fused_chunk_gdr_bwd_kernel_dg_strides = (((DLTensor*)dg_handle)[0].strides);
  void* dg = (((DLTensor*)dg_handle)[0].data);
  if (!(((((DLTensor*)db_handle)[0].ndim) == 3))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input db ndim mismatch, expected 3", (long long)((((DLTensor*)db_handle)[0].ndim)), (long long)(3));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  void* tilelang_fused_chunk_gdr_bwd_kernel_db_strides = (((DLTensor*)db_handle)[0].strides);
  void* db = (((DLTensor*)db_handle)[0].data);
  if (!(((((DLTensor*)dh0_handle)[0].ndim) == 4))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input dh0 ndim mismatch, expected 4", (long long)((((DLTensor*)dh0_handle)[0].ndim)), (long long)(4));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  void* tilelang_fused_chunk_gdr_bwd_kernel_dh0_strides = (((DLTensor*)dh0_handle)[0].strides);
  void* dh0 = (((DLTensor*)dh0_handle)[0].data);
  if (!(((((((DLTensor*)do_handle)[0].dtype.code) == (uint8_t)4) && ((((DLTensor*)do_handle)[0].dtype.bits) == (uint8_t)16)) && ((((DLTensor*)do_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_bwd_kernel input do dtype mismatch, expected bfloat16");
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_do_shape)[0]) == 1))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input do shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_do_shape)[0])), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_do_shape)[2]) == 4))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input do shape[2] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_do_shape)[2])), (long long)(4));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_do_shape)[3]) == 128))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input do shape[3] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_do_shape)[3])), (long long)(128));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval;
  if ((tilelang_fused_chunk_gdr_bwd_kernel_do_strides == NULL)) {
    condval = 1;
  } else {
    condval = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_do_strides)[3]);
  }
  if (!((condval == 1))) {
    int32_t condval_1;
    if ((tilelang_fused_chunk_gdr_bwd_kernel_do_strides == NULL)) {
      condval_1 = 1;
    } else {
      condval_1 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_do_strides)[3]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input do strides[3] violates packed ABI constraint", (long long)(condval_1), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_2;
  if ((tilelang_fused_chunk_gdr_bwd_kernel_do_strides == NULL)) {
    condval_2 = 1;
  } else {
    condval_2 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_do_strides)[2]);
  }
  if (!((condval_2 == 128))) {
    int32_t condval_3;
    if ((tilelang_fused_chunk_gdr_bwd_kernel_do_strides == NULL)) {
      condval_3 = 1;
    } else {
      condval_3 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_do_strides)[2]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input do strides[2] violates packed ABI constraint", (long long)(condval_3), (long long)(128));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_4;
  if ((tilelang_fused_chunk_gdr_bwd_kernel_do_strides == NULL)) {
    condval_4 = 1;
  } else {
    condval_4 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_do_strides)[1]);
  }
  if (!(((condval_4 == 512) || (num_tokens == 1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_bwd_kernel input do strides[1] violates packed ABI constraint");
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)do_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input do byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)do_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)do_handle)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input do device_type mismatch, expected cuda", (long long)((((DLTensor*)do_handle)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((num_tokens == 0) || !(do_1 == NULL)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_bwd_kernel input do data pointer is NULL");
    return -1;
  }
  if (!(((((((DLTensor*)dht_handle)[0].dtype.code) == (uint8_t)2) && ((((DLTensor*)dht_handle)[0].dtype.bits) == (uint8_t)32)) && ((((DLTensor*)dht_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_bwd_kernel input dht dtype mismatch, expected float32");
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_dht_shape)[1]) == 4))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input dht shape[1] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_dht_shape)[1])), (long long)(4));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_dht_shape)[2]) == 128))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input dht shape[2] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_dht_shape)[2])), (long long)(128));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_dht_shape)[3]) == 128))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input dht shape[3] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_dht_shape)[3])), (long long)(128));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_5;
  if ((tilelang_fused_chunk_gdr_bwd_kernel_dht_strides == NULL)) {
    condval_5 = 1;
  } else {
    condval_5 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_dht_strides)[3]);
  }
  if (!((condval_5 == 1))) {
    int32_t condval_6;
    if ((tilelang_fused_chunk_gdr_bwd_kernel_dht_strides == NULL)) {
      condval_6 = 1;
    } else {
      condval_6 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_dht_strides)[3]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input dht strides[3] violates packed ABI constraint", (long long)(condval_6), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_7;
  if ((tilelang_fused_chunk_gdr_bwd_kernel_dht_strides == NULL)) {
    condval_7 = 1;
  } else {
    condval_7 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_dht_strides)[2]);
  }
  if (!((condval_7 == 128))) {
    int32_t condval_8;
    if ((tilelang_fused_chunk_gdr_bwd_kernel_dht_strides == NULL)) {
      condval_8 = 1;
    } else {
      condval_8 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_dht_strides)[2]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input dht strides[2] violates packed ABI constraint", (long long)(condval_8), (long long)(128));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_9;
  if ((tilelang_fused_chunk_gdr_bwd_kernel_dht_strides == NULL)) {
    condval_9 = 1;
  } else {
    condval_9 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_dht_strides)[1]);
  }
  if (!((condval_9 == 16384))) {
    int32_t condval_10;
    if ((tilelang_fused_chunk_gdr_bwd_kernel_dht_strides == NULL)) {
      condval_10 = 1;
    } else {
      condval_10 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_dht_strides)[1]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input dht strides[1] violates packed ABI constraint", (long long)(condval_10), (long long)(16384));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_11;
  if ((tilelang_fused_chunk_gdr_bwd_kernel_dht_strides == NULL)) {
    condval_11 = 1;
  } else {
    condval_11 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_dht_strides)[0]);
  }
  if (!(((condval_11 == 65536) || (batch_size == 1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_bwd_kernel input dht strides[0] violates packed ABI constraint");
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)dht_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input dht byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)dht_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)dht_handle)[0].device.device_id) == (((DLTensor*)do_handle)[0].device.device_id)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input dht device_id violates packed ABI constraint", (long long)((((DLTensor*)dht_handle)[0].device.device_id)), (long long)((((DLTensor*)do_handle)[0].device.device_id)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)dht_handle)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input dht device_type mismatch, expected cuda", (long long)((((DLTensor*)dht_handle)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((batch_size == 0) || !(dht == NULL)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_bwd_kernel input dht data pointer is NULL");
    return -1;
  }
  if (!(((((((DLTensor*)q_handle)[0].dtype.code) == (uint8_t)4) && ((((DLTensor*)q_handle)[0].dtype.bits) == (uint8_t)16)) && ((((DLTensor*)q_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_bwd_kernel input q dtype mismatch, expected bfloat16");
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_q_shape)[0]) == 1))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input q shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_q_shape)[0])), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_do_shape)[1]) == ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_q_shape)[1])))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input q shape[1] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_do_shape)[1])), (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_q_shape)[1])));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_q_shape)[2]) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input q shape[2] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_q_shape)[2])), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_q_shape)[3]) == 128))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input q shape[3] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_q_shape)[3])), (long long)(128));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_12;
  if ((tilelang_fused_chunk_gdr_bwd_kernel_q_strides == NULL)) {
    condval_12 = 1;
  } else {
    condval_12 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_q_strides)[3]);
  }
  if (!((condval_12 == 1))) {
    int32_t condval_13;
    if ((tilelang_fused_chunk_gdr_bwd_kernel_q_strides == NULL)) {
      condval_13 = 1;
    } else {
      condval_13 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_q_strides)[3]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input q strides[3] violates packed ABI constraint", (long long)(condval_13), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_14;
  if ((tilelang_fused_chunk_gdr_bwd_kernel_q_strides == NULL)) {
    condval_14 = 1;
  } else {
    condval_14 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_q_strides)[2]);
  }
  if (!((condval_14 == 128))) {
    int32_t condval_15;
    if ((tilelang_fused_chunk_gdr_bwd_kernel_q_strides == NULL)) {
      condval_15 = 1;
    } else {
      condval_15 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_q_strides)[2]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input q strides[2] violates packed ABI constraint", (long long)(condval_15), (long long)(128));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_16;
  if ((tilelang_fused_chunk_gdr_bwd_kernel_q_strides == NULL)) {
    condval_16 = 1;
  } else {
    condval_16 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_q_strides)[1]);
  }
  if (!(((condval_16 == 256) || (num_tokens == 1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_bwd_kernel input q strides[1] violates packed ABI constraint");
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)q_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input q byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)q_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)q_handle)[0].device.device_id) == (((DLTensor*)do_handle)[0].device.device_id)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input q device_id violates packed ABI constraint", (long long)((((DLTensor*)q_handle)[0].device.device_id)), (long long)((((DLTensor*)do_handle)[0].device.device_id)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)q_handle)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input q device_type mismatch, expected cuda", (long long)((((DLTensor*)q_handle)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((num_tokens == 0) || !(q == NULL)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_bwd_kernel input q data pointer is NULL");
    return -1;
  }
  if (!(((((((DLTensor*)k_handle)[0].dtype.code) == (uint8_t)4) && ((((DLTensor*)k_handle)[0].dtype.bits) == (uint8_t)16)) && ((((DLTensor*)k_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_bwd_kernel input k dtype mismatch, expected bfloat16");
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_k_shape)[0]) == 1))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input k shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_k_shape)[0])), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_do_shape)[1]) == ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_k_shape)[1])))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input k shape[1] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_do_shape)[1])), (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_k_shape)[1])));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_k_shape)[2]) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input k shape[2] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_k_shape)[2])), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_k_shape)[3]) == 128))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input k shape[3] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_k_shape)[3])), (long long)(128));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_17;
  if ((tilelang_fused_chunk_gdr_bwd_kernel_k_strides == NULL)) {
    condval_17 = 1;
  } else {
    condval_17 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_k_strides)[3]);
  }
  if (!((condval_17 == 1))) {
    int32_t condval_18;
    if ((tilelang_fused_chunk_gdr_bwd_kernel_k_strides == NULL)) {
      condval_18 = 1;
    } else {
      condval_18 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_k_strides)[3]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input k strides[3] violates packed ABI constraint", (long long)(condval_18), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_19;
  if ((tilelang_fused_chunk_gdr_bwd_kernel_k_strides == NULL)) {
    condval_19 = 1;
  } else {
    condval_19 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_k_strides)[2]);
  }
  if (!((condval_19 == 128))) {
    int32_t condval_20;
    if ((tilelang_fused_chunk_gdr_bwd_kernel_k_strides == NULL)) {
      condval_20 = 1;
    } else {
      condval_20 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_k_strides)[2]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input k strides[2] violates packed ABI constraint", (long long)(condval_20), (long long)(128));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_21;
  if ((tilelang_fused_chunk_gdr_bwd_kernel_k_strides == NULL)) {
    condval_21 = 1;
  } else {
    condval_21 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_k_strides)[1]);
  }
  if (!(((condval_21 == 256) || (num_tokens == 1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_bwd_kernel input k strides[1] violates packed ABI constraint");
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)k_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input k byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)k_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)k_handle)[0].device.device_id) == (((DLTensor*)do_handle)[0].device.device_id)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input k device_id violates packed ABI constraint", (long long)((((DLTensor*)k_handle)[0].device.device_id)), (long long)((((DLTensor*)do_handle)[0].device.device_id)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)k_handle)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input k device_type mismatch, expected cuda", (long long)((((DLTensor*)k_handle)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((num_tokens == 0) || !(k == NULL)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_bwd_kernel input k data pointer is NULL");
    return -1;
  }
  if (!(((((((DLTensor*)v_handle)[0].dtype.code) == (uint8_t)4) && ((((DLTensor*)v_handle)[0].dtype.bits) == (uint8_t)16)) && ((((DLTensor*)v_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_bwd_kernel input v dtype mismatch, expected bfloat16");
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_v_shape)[0]) == 1))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input v shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_v_shape)[0])), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_do_shape)[1]) == ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_v_shape)[1])))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input v shape[1] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_do_shape)[1])), (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_v_shape)[1])));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_v_shape)[2]) == 4))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input v shape[2] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_v_shape)[2])), (long long)(4));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_v_shape)[3]) == 128))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input v shape[3] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_v_shape)[3])), (long long)(128));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_22;
  if ((tilelang_fused_chunk_gdr_bwd_kernel_v_strides == NULL)) {
    condval_22 = 1;
  } else {
    condval_22 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_v_strides)[3]);
  }
  if (!((condval_22 == 1))) {
    int32_t condval_23;
    if ((tilelang_fused_chunk_gdr_bwd_kernel_v_strides == NULL)) {
      condval_23 = 1;
    } else {
      condval_23 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_v_strides)[3]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input v strides[3] violates packed ABI constraint", (long long)(condval_23), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_24;
  if ((tilelang_fused_chunk_gdr_bwd_kernel_v_strides == NULL)) {
    condval_24 = 1;
  } else {
    condval_24 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_v_strides)[2]);
  }
  if (!((condval_24 == 128))) {
    int32_t condval_25;
    if ((tilelang_fused_chunk_gdr_bwd_kernel_v_strides == NULL)) {
      condval_25 = 1;
    } else {
      condval_25 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_v_strides)[2]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input v strides[2] violates packed ABI constraint", (long long)(condval_25), (long long)(128));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_26;
  if ((tilelang_fused_chunk_gdr_bwd_kernel_v_strides == NULL)) {
    condval_26 = 1;
  } else {
    condval_26 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_v_strides)[1]);
  }
  if (!(((condval_26 == 512) || (num_tokens == 1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_bwd_kernel input v strides[1] violates packed ABI constraint");
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)v_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input v byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)v_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)v_handle)[0].device.device_id) == (((DLTensor*)do_handle)[0].device.device_id)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input v device_id violates packed ABI constraint", (long long)((((DLTensor*)v_handle)[0].device.device_id)), (long long)((((DLTensor*)do_handle)[0].device.device_id)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)v_handle)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input v device_type mismatch, expected cuda", (long long)((((DLTensor*)v_handle)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((num_tokens == 0) || !(v == NULL)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_bwd_kernel input v data pointer is NULL");
    return -1;
  }
  if (!(((((((DLTensor*)a_handle)[0].dtype.code) == (uint8_t)4) && ((((DLTensor*)a_handle)[0].dtype.bits) == (uint8_t)16)) && ((((DLTensor*)a_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_bwd_kernel input a dtype mismatch, expected bfloat16");
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_a_shape)[0]) == 1))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input a shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_a_shape)[0])), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_do_shape)[1]) == ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_a_shape)[1])))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input a shape[1] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_do_shape)[1])), (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_a_shape)[1])));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_a_shape)[2]) == 4))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input a shape[2] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_a_shape)[2])), (long long)(4));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_a_shape)[3]) == 32))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input a shape[3] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_a_shape)[3])), (long long)(32));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_27;
  if ((tilelang_fused_chunk_gdr_bwd_kernel_a_strides == NULL)) {
    condval_27 = 1;
  } else {
    condval_27 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_a_strides)[3]);
  }
  if (!((condval_27 == 1))) {
    int32_t condval_28;
    if ((tilelang_fused_chunk_gdr_bwd_kernel_a_strides == NULL)) {
      condval_28 = 1;
    } else {
      condval_28 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_a_strides)[3]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input a strides[3] violates packed ABI constraint", (long long)(condval_28), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_29;
  if ((tilelang_fused_chunk_gdr_bwd_kernel_a_strides == NULL)) {
    condval_29 = 1;
  } else {
    condval_29 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_a_strides)[2]);
  }
  if (!((condval_29 == 32))) {
    int32_t condval_30;
    if ((tilelang_fused_chunk_gdr_bwd_kernel_a_strides == NULL)) {
      condval_30 = 1;
    } else {
      condval_30 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_a_strides)[2]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input a strides[2] violates packed ABI constraint", (long long)(condval_30), (long long)(32));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_31;
  if ((tilelang_fused_chunk_gdr_bwd_kernel_a_strides == NULL)) {
    condval_31 = 1;
  } else {
    condval_31 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_a_strides)[1]);
  }
  if (!(((condval_31 == 128) || (num_tokens == 1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_bwd_kernel input a strides[1] violates packed ABI constraint");
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)a_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input a byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)a_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)a_handle)[0].device.device_id) == (((DLTensor*)do_handle)[0].device.device_id)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input a device_id violates packed ABI constraint", (long long)((((DLTensor*)a_handle)[0].device.device_id)), (long long)((((DLTensor*)do_handle)[0].device.device_id)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)a_handle)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input a device_type mismatch, expected cuda", (long long)((((DLTensor*)a_handle)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((num_tokens == 0) || !(a == NULL)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_bwd_kernel input a data pointer is NULL");
    return -1;
  }
  if (!(((((((DLTensor*)g_handle)[0].dtype.code) == (uint8_t)2) && ((((DLTensor*)g_handle)[0].dtype.bits) == (uint8_t)32)) && ((((DLTensor*)g_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_bwd_kernel input g dtype mismatch, expected float32");
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_g_shape)[0]) == 1))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input g shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_g_shape)[0])), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_do_shape)[1]) == ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_g_shape)[1])))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input g shape[1] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_do_shape)[1])), (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_g_shape)[1])));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_g_shape)[2]) == 4))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input g shape[2] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_g_shape)[2])), (long long)(4));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_32;
  if ((tilelang_fused_chunk_gdr_bwd_kernel_g_strides == NULL)) {
    condval_32 = 1;
  } else {
    condval_32 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_g_strides)[2]);
  }
  if (!((condval_32 == 1))) {
    int32_t condval_33;
    if ((tilelang_fused_chunk_gdr_bwd_kernel_g_strides == NULL)) {
      condval_33 = 1;
    } else {
      condval_33 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_g_strides)[2]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input g strides[2] violates packed ABI constraint", (long long)(condval_33), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_34;
  if ((tilelang_fused_chunk_gdr_bwd_kernel_g_strides == NULL)) {
    condval_34 = 1;
  } else {
    condval_34 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_g_strides)[1]);
  }
  if (!(((condval_34 == 4) || (num_tokens == 1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_bwd_kernel input g strides[1] violates packed ABI constraint");
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)g_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input g byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)g_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)g_handle)[0].device.device_id) == (((DLTensor*)do_handle)[0].device.device_id)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input g device_id violates packed ABI constraint", (long long)((((DLTensor*)g_handle)[0].device.device_id)), (long long)((((DLTensor*)do_handle)[0].device.device_id)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)g_handle)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input g device_type mismatch, expected cuda", (long long)((((DLTensor*)g_handle)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((num_tokens == 0) || !(g == NULL)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_bwd_kernel input g data pointer is NULL");
    return -1;
  }
  if (!(((((((DLTensor*)b_handle)[0].dtype.code) == (uint8_t)2) && ((((DLTensor*)b_handle)[0].dtype.bits) == (uint8_t)32)) && ((((DLTensor*)b_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_bwd_kernel input b dtype mismatch, expected float32");
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_b_shape)[0]) == 1))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input b shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_b_shape)[0])), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_do_shape)[1]) == ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_b_shape)[1])))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input b shape[1] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_do_shape)[1])), (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_b_shape)[1])));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_b_shape)[2]) == 4))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input b shape[2] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_b_shape)[2])), (long long)(4));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_35;
  if ((tilelang_fused_chunk_gdr_bwd_kernel_b_strides == NULL)) {
    condval_35 = 1;
  } else {
    condval_35 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_b_strides)[2]);
  }
  if (!((condval_35 == 1))) {
    int32_t condval_36;
    if ((tilelang_fused_chunk_gdr_bwd_kernel_b_strides == NULL)) {
      condval_36 = 1;
    } else {
      condval_36 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_b_strides)[2]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input b strides[2] violates packed ABI constraint", (long long)(condval_36), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_37;
  if ((tilelang_fused_chunk_gdr_bwd_kernel_b_strides == NULL)) {
    condval_37 = 1;
  } else {
    condval_37 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_b_strides)[1]);
  }
  if (!(((condval_37 == 4) || (num_tokens == 1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_bwd_kernel input b strides[1] violates packed ABI constraint");
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)b_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input b byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)b_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)b_handle)[0].device.device_id) == (((DLTensor*)do_handle)[0].device.device_id)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input b device_id violates packed ABI constraint", (long long)((((DLTensor*)b_handle)[0].device.device_id)), (long long)((((DLTensor*)do_handle)[0].device.device_id)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)b_handle)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input b device_type mismatch, expected cuda", (long long)((((DLTensor*)b_handle)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((num_tokens == 0) || !(b == NULL)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_bwd_kernel input b data pointer is NULL");
    return -1;
  }
  if (!(((((((DLTensor*)h_handle)[0].dtype.code) == (uint8_t)4) && ((((DLTensor*)h_handle)[0].dtype.bits) == (uint8_t)16)) && ((((DLTensor*)h_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_bwd_kernel input h dtype mismatch, expected bfloat16");
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_h_shape)[0]) == 1))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input h shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_h_shape)[0])), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_h_shape)[2]) == 4))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input h shape[2] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_h_shape)[2])), (long long)(4));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_h_shape)[3]) == 128))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input h shape[3] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_h_shape)[3])), (long long)(128));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_h_shape)[4]) == 128))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input h shape[4] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_h_shape)[4])), (long long)(128));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_38;
  if ((tilelang_fused_chunk_gdr_bwd_kernel_h_strides == NULL)) {
    condval_38 = 1;
  } else {
    condval_38 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_h_strides)[4]);
  }
  if (!((condval_38 == 1))) {
    int32_t condval_39;
    if ((tilelang_fused_chunk_gdr_bwd_kernel_h_strides == NULL)) {
      condval_39 = 1;
    } else {
      condval_39 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_h_strides)[4]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input h strides[4] violates packed ABI constraint", (long long)(condval_39), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_40;
  if ((tilelang_fused_chunk_gdr_bwd_kernel_h_strides == NULL)) {
    condval_40 = 1;
  } else {
    condval_40 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_h_strides)[3]);
  }
  if (!((condval_40 == 128))) {
    int32_t condval_41;
    if ((tilelang_fused_chunk_gdr_bwd_kernel_h_strides == NULL)) {
      condval_41 = 1;
    } else {
      condval_41 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_h_strides)[3]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input h strides[3] violates packed ABI constraint", (long long)(condval_41), (long long)(128));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_42;
  if ((tilelang_fused_chunk_gdr_bwd_kernel_h_strides == NULL)) {
    condval_42 = 1;
  } else {
    condval_42 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_h_strides)[2]);
  }
  if (!((condval_42 == 16384))) {
    int32_t condval_43;
    if ((tilelang_fused_chunk_gdr_bwd_kernel_h_strides == NULL)) {
      condval_43 = 1;
    } else {
      condval_43 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_h_strides)[2]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input h strides[2] violates packed ABI constraint", (long long)(condval_43), (long long)(16384));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_44;
  if ((tilelang_fused_chunk_gdr_bwd_kernel_h_strides == NULL)) {
    condval_44 = 1;
  } else {
    condval_44 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_h_strides)[1]);
  }
  if (!(((condval_44 == 65536) || (num_chunks == 1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_bwd_kernel input h strides[1] violates packed ABI constraint");
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)h_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input h byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)h_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)h_handle)[0].device.device_id) == (((DLTensor*)do_handle)[0].device.device_id)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input h device_id violates packed ABI constraint", (long long)((((DLTensor*)h_handle)[0].device.device_id)), (long long)((((DLTensor*)do_handle)[0].device.device_id)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)h_handle)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input h device_type mismatch, expected cuda", (long long)((((DLTensor*)h_handle)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((num_chunks == 0) || !(h == NULL)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_bwd_kernel input h data pointer is NULL");
    return -1;
  }
  if (!(((((((DLTensor*)cu_seqlens_handle)[0].dtype.code) == (uint8_t)0) && ((((DLTensor*)cu_seqlens_handle)[0].dtype.bits) == (uint8_t)32)) && ((((DLTensor*)cu_seqlens_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_bwd_kernel input cu_seqlens dtype mismatch, expected int32");
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_cu_seqlens_shape)[0]) == (batch_size + 1)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input cu_seqlens shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_cu_seqlens_shape)[0])), (long long)((batch_size + 1)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_45;
  if ((tilelang_fused_chunk_gdr_bwd_kernel_cu_seqlens_strides == NULL)) {
    condval_45 = 1;
  } else {
    condval_45 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_cu_seqlens_strides)[0]);
  }
  if (!(((condval_45 == 1) || (batch_size == 0)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_bwd_kernel input cu_seqlens strides[0] violates packed ABI constraint");
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)cu_seqlens_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input cu_seqlens byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)cu_seqlens_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)cu_seqlens_handle)[0].device.device_id) == (((DLTensor*)do_handle)[0].device.device_id)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input cu_seqlens device_id violates packed ABI constraint", (long long)((((DLTensor*)cu_seqlens_handle)[0].device.device_id)), (long long)((((DLTensor*)do_handle)[0].device.device_id)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)cu_seqlens_handle)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input cu_seqlens device_type mismatch, expected cuda", (long long)((((DLTensor*)cu_seqlens_handle)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((batch_size == -1) || !(cu_seqlens == NULL)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_bwd_kernel input cu_seqlens data pointer is NULL");
    return -1;
  }
  if (!(((((((DLTensor*)chunk_offsets_handle)[0].dtype.code) == (uint8_t)0) && ((((DLTensor*)chunk_offsets_handle)[0].dtype.bits) == (uint8_t)32)) && ((((DLTensor*)chunk_offsets_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_bwd_kernel input chunk_offsets dtype mismatch, expected int32");
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_chunk_offsets_shape)[0]) == (batch_size + 1)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input chunk_offsets shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_chunk_offsets_shape)[0])), (long long)((batch_size + 1)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_46;
  if ((tilelang_fused_chunk_gdr_bwd_kernel_chunk_offsets_strides == NULL)) {
    condval_46 = 1;
  } else {
    condval_46 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_chunk_offsets_strides)[0]);
  }
  if (!(((condval_46 == 1) || (batch_size == 0)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_bwd_kernel input chunk_offsets strides[0] violates packed ABI constraint");
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)chunk_offsets_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input chunk_offsets byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)chunk_offsets_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)chunk_offsets_handle)[0].device.device_id) == (((DLTensor*)do_handle)[0].device.device_id)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input chunk_offsets device_id violates packed ABI constraint", (long long)((((DLTensor*)chunk_offsets_handle)[0].device.device_id)), (long long)((((DLTensor*)do_handle)[0].device.device_id)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)chunk_offsets_handle)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input chunk_offsets device_type mismatch, expected cuda", (long long)((((DLTensor*)chunk_offsets_handle)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((batch_size == -1) || !(chunk_offsets == NULL)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_bwd_kernel input chunk_offsets data pointer is NULL");
    return -1;
  }
  if (!(((((((DLTensor*)dq_handle)[0].dtype.code) == (uint8_t)4) && ((((DLTensor*)dq_handle)[0].dtype.bits) == (uint8_t)16)) && ((((DLTensor*)dq_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_bwd_kernel input dq dtype mismatch, expected bfloat16");
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_dq_shape)[0]) == 1))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input dq shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_dq_shape)[0])), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_do_shape)[1]) == ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_dq_shape)[1])))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input dq shape[1] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_do_shape)[1])), (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_dq_shape)[1])));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_dq_shape)[2]) == 4))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input dq shape[2] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_dq_shape)[2])), (long long)(4));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_dq_shape)[3]) == 128))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input dq shape[3] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_dq_shape)[3])), (long long)(128));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_47;
  if ((tilelang_fused_chunk_gdr_bwd_kernel_dq_strides == NULL)) {
    condval_47 = 1;
  } else {
    condval_47 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_dq_strides)[3]);
  }
  if (!((condval_47 == 1))) {
    int32_t condval_48;
    if ((tilelang_fused_chunk_gdr_bwd_kernel_dq_strides == NULL)) {
      condval_48 = 1;
    } else {
      condval_48 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_dq_strides)[3]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input dq strides[3] violates packed ABI constraint", (long long)(condval_48), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_49;
  if ((tilelang_fused_chunk_gdr_bwd_kernel_dq_strides == NULL)) {
    condval_49 = 1;
  } else {
    condval_49 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_dq_strides)[2]);
  }
  if (!((condval_49 == 128))) {
    int32_t condval_50;
    if ((tilelang_fused_chunk_gdr_bwd_kernel_dq_strides == NULL)) {
      condval_50 = 1;
    } else {
      condval_50 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_dq_strides)[2]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input dq strides[2] violates packed ABI constraint", (long long)(condval_50), (long long)(128));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_51;
  if ((tilelang_fused_chunk_gdr_bwd_kernel_dq_strides == NULL)) {
    condval_51 = 1;
  } else {
    condval_51 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_dq_strides)[1]);
  }
  if (!(((condval_51 == 512) || (num_tokens == 1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_bwd_kernel input dq strides[1] violates packed ABI constraint");
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)dq_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input dq byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)dq_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)dq_handle)[0].device.device_id) == (((DLTensor*)do_handle)[0].device.device_id)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input dq device_id violates packed ABI constraint", (long long)((((DLTensor*)dq_handle)[0].device.device_id)), (long long)((((DLTensor*)do_handle)[0].device.device_id)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)dq_handle)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input dq device_type mismatch, expected cuda", (long long)((((DLTensor*)dq_handle)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((num_tokens == 0) || !(dq == NULL)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_bwd_kernel input dq data pointer is NULL");
    return -1;
  }
  if (!(((((((DLTensor*)dk_handle)[0].dtype.code) == (uint8_t)4) && ((((DLTensor*)dk_handle)[0].dtype.bits) == (uint8_t)16)) && ((((DLTensor*)dk_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_bwd_kernel input dk dtype mismatch, expected bfloat16");
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_dk_shape)[0]) == 1))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input dk shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_dk_shape)[0])), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_do_shape)[1]) == ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_dk_shape)[1])))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input dk shape[1] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_do_shape)[1])), (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_dk_shape)[1])));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_dk_shape)[2]) == 4))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input dk shape[2] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_dk_shape)[2])), (long long)(4));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_dk_shape)[3]) == 128))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input dk shape[3] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_dk_shape)[3])), (long long)(128));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_52;
  if ((tilelang_fused_chunk_gdr_bwd_kernel_dk_strides == NULL)) {
    condval_52 = 1;
  } else {
    condval_52 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_dk_strides)[3]);
  }
  if (!((condval_52 == 1))) {
    int32_t condval_53;
    if ((tilelang_fused_chunk_gdr_bwd_kernel_dk_strides == NULL)) {
      condval_53 = 1;
    } else {
      condval_53 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_dk_strides)[3]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input dk strides[3] violates packed ABI constraint", (long long)(condval_53), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_54;
  if ((tilelang_fused_chunk_gdr_bwd_kernel_dk_strides == NULL)) {
    condval_54 = 1;
  } else {
    condval_54 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_dk_strides)[2]);
  }
  if (!((condval_54 == 128))) {
    int32_t condval_55;
    if ((tilelang_fused_chunk_gdr_bwd_kernel_dk_strides == NULL)) {
      condval_55 = 1;
    } else {
      condval_55 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_dk_strides)[2]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input dk strides[2] violates packed ABI constraint", (long long)(condval_55), (long long)(128));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_56;
  if ((tilelang_fused_chunk_gdr_bwd_kernel_dk_strides == NULL)) {
    condval_56 = 1;
  } else {
    condval_56 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_dk_strides)[1]);
  }
  if (!(((condval_56 == 512) || (num_tokens == 1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_bwd_kernel input dk strides[1] violates packed ABI constraint");
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)dk_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input dk byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)dk_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)dk_handle)[0].device.device_id) == (((DLTensor*)do_handle)[0].device.device_id)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input dk device_id violates packed ABI constraint", (long long)((((DLTensor*)dk_handle)[0].device.device_id)), (long long)((((DLTensor*)do_handle)[0].device.device_id)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)dk_handle)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input dk device_type mismatch, expected cuda", (long long)((((DLTensor*)dk_handle)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((num_tokens == 0) || !(dk == NULL)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_bwd_kernel input dk data pointer is NULL");
    return -1;
  }
  if (!(((((((DLTensor*)dv_handle)[0].dtype.code) == (uint8_t)4) && ((((DLTensor*)dv_handle)[0].dtype.bits) == (uint8_t)16)) && ((((DLTensor*)dv_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_bwd_kernel input dv dtype mismatch, expected bfloat16");
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_dv_shape)[0]) == 1))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input dv shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_dv_shape)[0])), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_do_shape)[1]) == ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_dv_shape)[1])))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input dv shape[1] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_do_shape)[1])), (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_dv_shape)[1])));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_dv_shape)[2]) == 4))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input dv shape[2] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_dv_shape)[2])), (long long)(4));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_dv_shape)[3]) == 128))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input dv shape[3] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_dv_shape)[3])), (long long)(128));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_57;
  if ((tilelang_fused_chunk_gdr_bwd_kernel_dv_strides == NULL)) {
    condval_57 = 1;
  } else {
    condval_57 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_dv_strides)[3]);
  }
  if (!((condval_57 == 1))) {
    int32_t condval_58;
    if ((tilelang_fused_chunk_gdr_bwd_kernel_dv_strides == NULL)) {
      condval_58 = 1;
    } else {
      condval_58 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_dv_strides)[3]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input dv strides[3] violates packed ABI constraint", (long long)(condval_58), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_59;
  if ((tilelang_fused_chunk_gdr_bwd_kernel_dv_strides == NULL)) {
    condval_59 = 1;
  } else {
    condval_59 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_dv_strides)[2]);
  }
  if (!((condval_59 == 128))) {
    int32_t condval_60;
    if ((tilelang_fused_chunk_gdr_bwd_kernel_dv_strides == NULL)) {
      condval_60 = 1;
    } else {
      condval_60 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_dv_strides)[2]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input dv strides[2] violates packed ABI constraint", (long long)(condval_60), (long long)(128));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_61;
  if ((tilelang_fused_chunk_gdr_bwd_kernel_dv_strides == NULL)) {
    condval_61 = 1;
  } else {
    condval_61 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_dv_strides)[1]);
  }
  if (!(((condval_61 == 512) || (num_tokens == 1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_bwd_kernel input dv strides[1] violates packed ABI constraint");
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)dv_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input dv byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)dv_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)dv_handle)[0].device.device_id) == (((DLTensor*)do_handle)[0].device.device_id)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input dv device_id violates packed ABI constraint", (long long)((((DLTensor*)dv_handle)[0].device.device_id)), (long long)((((DLTensor*)do_handle)[0].device.device_id)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)dv_handle)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input dv device_type mismatch, expected cuda", (long long)((((DLTensor*)dv_handle)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((num_tokens == 0) || !(dv == NULL)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_bwd_kernel input dv data pointer is NULL");
    return -1;
  }
  if (!(((((((DLTensor*)dg_handle)[0].dtype.code) == (uint8_t)2) && ((((DLTensor*)dg_handle)[0].dtype.bits) == (uint8_t)32)) && ((((DLTensor*)dg_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_bwd_kernel input dg dtype mismatch, expected float32");
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_dg_shape)[0]) == 1))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input dg shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_dg_shape)[0])), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_do_shape)[1]) == ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_dg_shape)[1])))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input dg shape[1] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_do_shape)[1])), (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_dg_shape)[1])));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_dg_shape)[2]) == 4))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input dg shape[2] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_dg_shape)[2])), (long long)(4));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_62;
  if ((tilelang_fused_chunk_gdr_bwd_kernel_dg_strides == NULL)) {
    condval_62 = 1;
  } else {
    condval_62 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_dg_strides)[2]);
  }
  if (!((condval_62 == 1))) {
    int32_t condval_63;
    if ((tilelang_fused_chunk_gdr_bwd_kernel_dg_strides == NULL)) {
      condval_63 = 1;
    } else {
      condval_63 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_dg_strides)[2]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input dg strides[2] violates packed ABI constraint", (long long)(condval_63), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_64;
  if ((tilelang_fused_chunk_gdr_bwd_kernel_dg_strides == NULL)) {
    condval_64 = 1;
  } else {
    condval_64 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_dg_strides)[1]);
  }
  if (!(((condval_64 == 4) || (num_tokens == 1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_bwd_kernel input dg strides[1] violates packed ABI constraint");
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)dg_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input dg byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)dg_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)dg_handle)[0].device.device_id) == (((DLTensor*)do_handle)[0].device.device_id)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input dg device_id violates packed ABI constraint", (long long)((((DLTensor*)dg_handle)[0].device.device_id)), (long long)((((DLTensor*)do_handle)[0].device.device_id)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)dg_handle)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input dg device_type mismatch, expected cuda", (long long)((((DLTensor*)dg_handle)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((num_tokens == 0) || !(dg == NULL)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_bwd_kernel input dg data pointer is NULL");
    return -1;
  }
  if (!(((((((DLTensor*)db_handle)[0].dtype.code) == (uint8_t)2) && ((((DLTensor*)db_handle)[0].dtype.bits) == (uint8_t)32)) && ((((DLTensor*)db_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_bwd_kernel input db dtype mismatch, expected float32");
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_db_shape)[0]) == 1))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input db shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_db_shape)[0])), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_do_shape)[1]) == ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_db_shape)[1])))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input db shape[1] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_do_shape)[1])), (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_db_shape)[1])));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_db_shape)[2]) == 4))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input db shape[2] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_db_shape)[2])), (long long)(4));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_65;
  if ((tilelang_fused_chunk_gdr_bwd_kernel_db_strides == NULL)) {
    condval_65 = 1;
  } else {
    condval_65 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_db_strides)[2]);
  }
  if (!((condval_65 == 1))) {
    int32_t condval_66;
    if ((tilelang_fused_chunk_gdr_bwd_kernel_db_strides == NULL)) {
      condval_66 = 1;
    } else {
      condval_66 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_db_strides)[2]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input db strides[2] violates packed ABI constraint", (long long)(condval_66), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_67;
  if ((tilelang_fused_chunk_gdr_bwd_kernel_db_strides == NULL)) {
    condval_67 = 1;
  } else {
    condval_67 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_db_strides)[1]);
  }
  if (!(((condval_67 == 4) || (num_tokens == 1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_bwd_kernel input db strides[1] violates packed ABI constraint");
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)db_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input db byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)db_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)db_handle)[0].device.device_id) == (((DLTensor*)do_handle)[0].device.device_id)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input db device_id violates packed ABI constraint", (long long)((((DLTensor*)db_handle)[0].device.device_id)), (long long)((((DLTensor*)do_handle)[0].device.device_id)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)db_handle)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input db device_type mismatch, expected cuda", (long long)((((DLTensor*)db_handle)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((num_tokens == 0) || !(db == NULL)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_bwd_kernel input db data pointer is NULL");
    return -1;
  }
  if (!(((((((DLTensor*)dh0_handle)[0].dtype.code) == (uint8_t)2) && ((((DLTensor*)dh0_handle)[0].dtype.bits) == (uint8_t)32)) && ((((DLTensor*)dh0_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_bwd_kernel input dh0 dtype mismatch, expected float32");
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_dht_shape)[0]) == ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_dh0_shape)[0])))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input dh0 shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_dht_shape)[0])), (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_dh0_shape)[0])));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_dh0_shape)[1]) == 4))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input dh0 shape[1] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_dh0_shape)[1])), (long long)(4));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_dh0_shape)[2]) == 128))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input dh0 shape[2] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_dh0_shape)[2])), (long long)(128));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_dh0_shape)[3]) == 128))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input dh0 shape[3] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_dh0_shape)[3])), (long long)(128));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_68;
  if ((tilelang_fused_chunk_gdr_bwd_kernel_dh0_strides == NULL)) {
    condval_68 = 1;
  } else {
    condval_68 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_dh0_strides)[3]);
  }
  if (!((condval_68 == 1))) {
    int32_t condval_69;
    if ((tilelang_fused_chunk_gdr_bwd_kernel_dh0_strides == NULL)) {
      condval_69 = 1;
    } else {
      condval_69 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_dh0_strides)[3]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input dh0 strides[3] violates packed ABI constraint", (long long)(condval_69), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_70;
  if ((tilelang_fused_chunk_gdr_bwd_kernel_dh0_strides == NULL)) {
    condval_70 = 1;
  } else {
    condval_70 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_dh0_strides)[2]);
  }
  if (!((condval_70 == 128))) {
    int32_t condval_71;
    if ((tilelang_fused_chunk_gdr_bwd_kernel_dh0_strides == NULL)) {
      condval_71 = 1;
    } else {
      condval_71 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_dh0_strides)[2]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input dh0 strides[2] violates packed ABI constraint", (long long)(condval_71), (long long)(128));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_72;
  if ((tilelang_fused_chunk_gdr_bwd_kernel_dh0_strides == NULL)) {
    condval_72 = 1;
  } else {
    condval_72 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_dh0_strides)[1]);
  }
  if (!((condval_72 == 16384))) {
    int32_t condval_73;
    if ((tilelang_fused_chunk_gdr_bwd_kernel_dh0_strides == NULL)) {
      condval_73 = 1;
    } else {
      condval_73 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_dh0_strides)[1]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input dh0 strides[1] violates packed ABI constraint", (long long)(condval_73), (long long)(16384));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_74;
  if ((tilelang_fused_chunk_gdr_bwd_kernel_dh0_strides == NULL)) {
    condval_74 = 1;
  } else {
    condval_74 = ((int32_t)((int64_t*)tilelang_fused_chunk_gdr_bwd_kernel_dh0_strides)[0]);
  }
  if (!(((condval_74 == 65536) || (batch_size == 1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_bwd_kernel input dh0 strides[0] violates packed ABI constraint");
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)dh0_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input dh0 byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)dh0_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)dh0_handle)[0].device.device_id) == (((DLTensor*)do_handle)[0].device.device_id)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input dh0 device_id violates packed ABI constraint", (long long)((((DLTensor*)dh0_handle)[0].device.device_id)), (long long)((((DLTensor*)do_handle)[0].device.device_id)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)dh0_handle)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_fused_chunk_gdr_bwd_kernel input dh0 device_type mismatch, expected cuda", (long long)((((DLTensor*)dh0_handle)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((batch_size == 0) || !(dh0 == NULL)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_fused_chunk_gdr_bwd_kernel input dh0 data pointer is NULL");
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
  void* k_desc = stack_1;
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
  (((TVMFFIAny*)stack_ffi_any)[7].v_int64) = ((int64_t)1);
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
  void* q_desc = stack_2;
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
  (((TVMFFIAny*)stack_ffi_any)[7].v_int64) = ((int64_t)1);
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
  void* dk_desc = stack_3;
  if (dk_desc == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = dk_desc;
  (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = ((int64_t)9);
  (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = ((int64_t)4);
  if (dk == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[3].v_ptr) = dk;
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
  (((TVMFFIAny*)stack_ffi_any)[7].v_int64) = ((int64_t)1);
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
  void* dv_desc = stack_4;
  if (dv_desc == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = dv_desc;
  (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = ((int64_t)9);
  (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = ((int64_t)4);
  if (dv == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[3].v_ptr) = dv;
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
  (((TVMFFIAny*)stack_ffi_any)[7].v_int64) = ((int64_t)1);
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
  TVMFFIAny result_5;
  result_5.type_index = kTVMFFINone;
  result_5.zero_padding = 0;
  result_5.v_int64 = 0;
  if (TVMFFIFunctionCall(__tvm_tensormap_create_tiled_packed, (TVMFFIAny*) stack_ffi_any, 24, &result_5) != 0) {
    return -1;
  }
  TL_ALIGN(128) TVMFFIAny stack_5[16];
  void* dq_desc = stack_5;
  if (dq_desc == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = dq_desc;
  (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = ((int64_t)9);
  (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = ((int64_t)4);
  if (dq == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[3].v_ptr) = dq;
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
  (((TVMFFIAny*)stack_ffi_any)[7].v_int64) = ((int64_t)1);
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
  TL_ALIGN(128) TVMFFIAny stack_6[16];
  void* h_desc = stack_6;
  if (h_desc == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = h_desc;
  (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = ((int64_t)9);
  (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = ((int64_t)5);
  if (h == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[3].v_ptr) = h;
  (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)128);
  (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = ((int64_t)128);
  (((TVMFFIAny*)stack_ffi_any)[6].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[6].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[6].v_int64) = ((int64_t)4);
  (((TVMFFIAny*)stack_ffi_any)[7].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[7].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[7].v_int64) = ((int64_t)num_chunks);
  (((TVMFFIAny*)stack_ffi_any)[8].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[8].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[8].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[9].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[9].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[9].v_int64) = ((int64_t)2);
  (((TVMFFIAny*)stack_ffi_any)[10].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[10].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[10].v_int64) = ((int64_t)256);
  (((TVMFFIAny*)stack_ffi_any)[11].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[11].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[11].v_int64) = ((int64_t)32768);
  (((TVMFFIAny*)stack_ffi_any)[12].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[12].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[12].v_int64) = ((int64_t)131072);
  (((TVMFFIAny*)stack_ffi_any)[13].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[13].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[13].v_int64) = (((int64_t)num_chunks) * (int64_t)131072);
  (((TVMFFIAny*)stack_ffi_any)[14].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[14].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[14].v_int64) = ((int64_t)64);
  (((TVMFFIAny*)stack_ffi_any)[15].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[15].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[15].v_int64) = ((int64_t)64);
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
  (((TVMFFIAny*)stack_ffi_any)[20].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[21].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[21].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[21].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[22].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[22].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[22].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[23].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[23].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[23].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[24].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[24].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[24].v_int64) = ((int64_t)0);
  (((TVMFFIAny*)stack_ffi_any)[25].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[25].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[25].v_int64) = ((int64_t)3);
  (((TVMFFIAny*)stack_ffi_any)[26].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[26].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[26].v_int64) = ((int64_t)2);
  (((TVMFFIAny*)stack_ffi_any)[27].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[27].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[27].v_int64) = ((int64_t)0);
  (((TVMFFIAny*)stack_ffi_any)[28].type_index) = 0;
  (((TVMFFIAny*)stack_ffi_any)[28].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[28].v_int64) = (int64_t)0;
  if (__tvm_tensormap_create_tiled_packed == NULL) {
    if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_tensormap_create_tiled", &__tvm_tensormap_create_tiled_packed) != 0) {
      return -1;
    }
  }
  TVMFFIAny result_7;
  result_7.type_index = kTVMFFINone;
  result_7.zero_padding = 0;
  result_7.v_int64 = 0;
  if (TVMFFIFunctionCall(__tvm_tensormap_create_tiled_packed, (TVMFFIAny*) stack_ffi_any, 28, &result_7) != 0) {
    return -1;
  }
  TL_ALIGN(128) TVMFFIAny stack_7[16];
  void* a_desc = stack_7;
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
  (((TVMFFIAny*)stack_ffi_any)[7].v_int64) = ((int64_t)1);
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
  TVMFFIAny result_8;
  result_8.type_index = kTVMFFINone;
  result_8.zero_padding = 0;
  result_8.v_int64 = 0;
  if (TVMFFIFunctionCall(__tvm_tensormap_create_tiled_packed, (TVMFFIAny*) stack_ffi_any, 24, &result_8) != 0) {
    return -1;
  }
  TL_ALIGN(128) TVMFFIAny stack_8[16];
  void* do_desc = stack_8;
  if (do_desc == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = do_desc;
  (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = ((int64_t)9);
  (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = ((int64_t)4);
  if (do_1 == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[3].v_ptr) = do_1;
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
  (((TVMFFIAny*)stack_ffi_any)[7].v_int64) = ((int64_t)1);
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
  TVMFFIAny result_9;
  result_9.type_index = kTVMFFINone;
  result_9.zero_padding = 0;
  result_9.v_int64 = 0;
  if (TVMFFIFunctionCall(__tvm_tensormap_create_tiled_packed, (TVMFFIAny*) stack_ffi_any, 24, &result_9) != 0) {
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
  if (chunk_offsets == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[3].v_ptr) = chunk_offsets;
  if (cu_seqlens == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[4].v_ptr) = cu_seqlens;
  if (db == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[5].v_ptr) = db;
  if (dg == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[6].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[6].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[6].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[6].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[6].v_ptr) = dg;
  if (dh0 == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[7].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[7].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[7].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[7].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[7].v_ptr) = dh0;
  if (dht == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[8].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[8].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[8].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[8].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[8].v_ptr) = dht;
  if (dk == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[9].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[9].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[9].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[9].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[9].v_ptr) = dk;
  if (dk_desc == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[10].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[10].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[10].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[10].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[10].v_ptr) = dk_desc;
  if (do_1 == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[11].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[11].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[11].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[11].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[11].v_ptr) = do_1;
  if (do_desc == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[12].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[12].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[12].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[12].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[12].v_ptr) = do_desc;
  if (dq == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[13].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[13].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[13].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[13].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[13].v_ptr) = dq;
  if (dq_desc == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[14].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[14].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[14].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[14].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[14].v_ptr) = dq_desc;
  if (dv == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[15].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[15].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[15].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[15].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[15].v_ptr) = dv;
  if (dv_desc == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[16].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[16].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[16].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[16].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[16].v_ptr) = dv_desc;
  if (g == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[17].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[17].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[17].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[17].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[17].v_ptr) = g;
  if (h == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[18].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[18].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[18].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[18].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[18].v_ptr) = h;
  if (h_desc == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[19].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[19].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[19].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[19].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[19].v_ptr) = h_desc;
  if (k == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[20].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[20].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[20].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[20].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[20].v_ptr) = k;
  if (k_desc == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[21].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[21].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[21].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[21].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[21].v_ptr) = k_desc;
  if (q == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[22].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[22].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[22].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[22].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[22].v_ptr) = q;
  if (q_desc == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[23].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[23].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[23].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[23].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[23].v_ptr) = q_desc;
  if (v == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[24].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[24].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[24].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[24].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[24].v_ptr) = v;
  (((TVMFFIAny*)stack_ffi_any)[25].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[25].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[25].v_int64) = ((int64_t)batch_size);
  (((TVMFFIAny*)stack_ffi_any)[26].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[26].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[26].v_int64) = ((int64_t)num_chunks);
  (((TVMFFIAny*)stack_ffi_any)[27].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[27].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[27].v_int64) = ((int64_t)num_tokens);
  (((TVMFFIAny*)stack_ffi_any)[28].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[28].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[28].v_int64) = ((int64_t)(batch_size * 4));
  (((TVMFFIAny*)stack_ffi_any)[29].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[29].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[29].v_int64) = ((int64_t)512);
  (((TVMFFIAny*)stack_ffi_any)[30].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[30].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[30].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[31].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[31].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[31].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[32].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[32].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[32].v_int64) = ((int64_t)93696);
  (((TVMFFIAny*)stack_ffi_any)[33].type_index) = 0;
  (((TVMFFIAny*)stack_ffi_any)[33].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[33].v_int64) = (int64_t)0;
  if (tilelang_fused_chunk_gdr_bwd_kernel_kernel_packed == NULL) {
    if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "tilelang_fused_chunk_gdr_bwd_kernel_kernel", &tilelang_fused_chunk_gdr_bwd_kernel_kernel_packed) != 0) {
      return -1;
    }
  }
  TVMFFIAny result_10;
  result_10.type_index = kTVMFFINone;
  result_10.zero_padding = 0;
  result_10.v_int64 = 0;
  if (TVMFFIFunctionCall(tilelang_fused_chunk_gdr_bwd_kernel_kernel_packed, (TVMFFIAny*) stack_ffi_any, 33, &result_10) != 0) {
    return -1;
  }
  return 0;
}

// CodegenC: NOTE: Auto-generated entry function
#ifdef __cplusplus
extern "C"
#endif
int32_t __tvm_ffi_main(void* self, void* args,int num_args, void* result) {
  return __tvm_ffi_tilelang_fused_chunk_gdr_bwd_kernel(self, args, num_args, result);
}
