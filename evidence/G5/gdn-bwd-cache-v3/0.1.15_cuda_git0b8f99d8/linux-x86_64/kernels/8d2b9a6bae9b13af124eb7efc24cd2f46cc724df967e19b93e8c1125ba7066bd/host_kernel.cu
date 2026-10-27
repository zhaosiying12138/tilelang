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
static void* tilelang_zero_bwd_padding_kernel_kernel_packed = NULL;
#ifdef __cplusplus
extern "C"
#endif
int32_t __tvm_ffi_tilelang_zero_bwd_padding_kernel(void* self_handle, void* args, int32_t num_args, void* result);
#ifdef __cplusplus
extern "C"
#endif
int32_t __tvm_ffi_tilelang_zero_bwd_padding_kernel(void* self_handle, void* args, int32_t num_args, void* result) {
  TL_ALIGN(128) TVMFFIAny stack[13];
  void* stack_ffi_any = stack;
  if (!((num_args == 6))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "tilelang_zero_bwd_padding_kernel: num_args should be 6", (long long)(num_args), (long long)(6));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(!(args == NULL))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "tilelang_zero_bwd_padding_kernel: args pointer is NULL");
    return -1;
  }
  int32_t cu_seqlens_handle_type_index = (((TVMFFIAny*)args)[0].type_index);
  if (!(((((cu_seqlens_handle_type_index == 0) || (cu_seqlens_handle_type_index == 4)) || (cu_seqlens_handle_type_index == 7)) || (64 <= cu_seqlens_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_zero_bwd_padding_kernel input cu_seqlens expected pointer or tensor handle");
    return -1;
  }
  int32_t dq_handle_type_index = (((TVMFFIAny*)args)[1].type_index);
  if (!(((((dq_handle_type_index == 0) || (dq_handle_type_index == 4)) || (dq_handle_type_index == 7)) || (64 <= dq_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_zero_bwd_padding_kernel input dq expected pointer or tensor handle");
    return -1;
  }
  int32_t dk_handle_type_index = (((TVMFFIAny*)args)[2].type_index);
  if (!(((((dk_handle_type_index == 0) || (dk_handle_type_index == 4)) || (dk_handle_type_index == 7)) || (64 <= dk_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_zero_bwd_padding_kernel input dk expected pointer or tensor handle");
    return -1;
  }
  int32_t dv_handle_type_index = (((TVMFFIAny*)args)[3].type_index);
  if (!(((((dv_handle_type_index == 0) || (dv_handle_type_index == 4)) || (dv_handle_type_index == 7)) || (64 <= dv_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_zero_bwd_padding_kernel input dv expected pointer or tensor handle");
    return -1;
  }
  int32_t dg_handle_type_index = (((TVMFFIAny*)args)[4].type_index);
  if (!(((((dg_handle_type_index == 0) || (dg_handle_type_index == 4)) || (dg_handle_type_index == 7)) || (64 <= dg_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_zero_bwd_padding_kernel input dg expected pointer or tensor handle");
    return -1;
  }
  int32_t db_handle_type_index = (((TVMFFIAny*)args)[5].type_index);
  if (!(((((db_handle_type_index == 0) || (db_handle_type_index == 4)) || (db_handle_type_index == 7)) || (64 <= db_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_zero_bwd_padding_kernel input db expected pointer or tensor handle");
    return -1;
  }
  void* cu_seqlens_handle = ((cu_seqlens_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[0].v_ptr) + 24)) : (((TVMFFIAny*)args)[0].v_ptr));
  void* dq_handle = ((dq_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[1].v_ptr) + 24)) : (((TVMFFIAny*)args)[1].v_ptr));
  void* dk_handle = ((dk_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[2].v_ptr) + 24)) : (((TVMFFIAny*)args)[2].v_ptr));
  void* dv_handle = ((dv_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[3].v_ptr) + 24)) : (((TVMFFIAny*)args)[3].v_ptr));
  void* dg_handle = ((dg_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[4].v_ptr) + 24)) : (((TVMFFIAny*)args)[4].v_ptr));
  void* db_handle = ((db_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[5].v_ptr) + 24)) : (((TVMFFIAny*)args)[5].v_ptr));
  bool tilelang_zero_bwd_padding_kernel_cu_seqlens_is_null = (cu_seqlens_handle == NULL);
  if (!(!tilelang_zero_bwd_padding_kernel_cu_seqlens_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "tilelang_zero_bwd_padding_kernel.cu_seqlens is expected to have non-NULL pointer");
    return -1;
  }
  bool tilelang_zero_bwd_padding_kernel_dq_is_null = (dq_handle == NULL);
  if (!(!tilelang_zero_bwd_padding_kernel_dq_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "tilelang_zero_bwd_padding_kernel.dq is expected to have non-NULL pointer");
    return -1;
  }
  bool tilelang_zero_bwd_padding_kernel_dk_is_null = (dk_handle == NULL);
  if (!(!tilelang_zero_bwd_padding_kernel_dk_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "tilelang_zero_bwd_padding_kernel.dk is expected to have non-NULL pointer");
    return -1;
  }
  bool tilelang_zero_bwd_padding_kernel_dv_is_null = (dv_handle == NULL);
  if (!(!tilelang_zero_bwd_padding_kernel_dv_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "tilelang_zero_bwd_padding_kernel.dv is expected to have non-NULL pointer");
    return -1;
  }
  bool tilelang_zero_bwd_padding_kernel_dg_is_null = (dg_handle == NULL);
  if (!(!tilelang_zero_bwd_padding_kernel_dg_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "tilelang_zero_bwd_padding_kernel.dg is expected to have non-NULL pointer");
    return -1;
  }
  bool tilelang_zero_bwd_padding_kernel_db_is_null = (db_handle == NULL);
  if (!(!tilelang_zero_bwd_padding_kernel_db_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "tilelang_zero_bwd_padding_kernel.db is expected to have non-NULL pointer");
    return -1;
  }
  void* tilelang_zero_bwd_padding_kernel_cu_seqlens_shape = (((DLTensor*)cu_seqlens_handle)[0].shape);
  void* tilelang_zero_bwd_padding_kernel_dq_shape = (((DLTensor*)dq_handle)[0].shape);
  void* tilelang_zero_bwd_padding_kernel_dk_shape = (((DLTensor*)dk_handle)[0].shape);
  void* tilelang_zero_bwd_padding_kernel_dv_shape = (((DLTensor*)dv_handle)[0].shape);
  void* tilelang_zero_bwd_padding_kernel_dg_shape = (((DLTensor*)dg_handle)[0].shape);
  void* tilelang_zero_bwd_padding_kernel_db_shape = (((DLTensor*)db_handle)[0].shape);
  if (!(((((DLTensor*)cu_seqlens_handle)[0].ndim) == 1))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_zero_bwd_padding_kernel input cu_seqlens ndim mismatch, expected 1", (long long)((((DLTensor*)cu_seqlens_handle)[0].ndim)), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t real_batch_size = (((int32_t)((int64_t*)tilelang_zero_bwd_padding_kernel_cu_seqlens_shape)[0]) - 1);
  void* tilelang_zero_bwd_padding_kernel_cu_seqlens_strides = (((DLTensor*)cu_seqlens_handle)[0].strides);
  int32_t dev_id = (((DLTensor*)cu_seqlens_handle)[0].device.device_id);
  void* cu_seqlens = (((DLTensor*)cu_seqlens_handle)[0].data);
  if (!(((((DLTensor*)dq_handle)[0].ndim) == 4))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_zero_bwd_padding_kernel input dq ndim mismatch, expected 4", (long long)((((DLTensor*)dq_handle)[0].ndim)), (long long)(4));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((bool)1)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "Symbolic shape variable num_tokens requires at least one non-null buffer among: tilelang_zero_bwd_padding_kernel.dq, tilelang_zero_bwd_padding_kernel.dk, tilelang_zero_bwd_padding_kernel.dv, tilelang_zero_bwd_padding_kernel.dg, tilelang_zero_bwd_padding_kernel.db");
    return -1;
  }
  int32_t num_tokens = ((int32_t)((int64_t*)tilelang_zero_bwd_padding_kernel_dq_shape)[1]);
  void* tilelang_zero_bwd_padding_kernel_dq_strides = (((DLTensor*)dq_handle)[0].strides);
  void* dq = (((DLTensor*)dq_handle)[0].data);
  if (!(((((DLTensor*)dk_handle)[0].ndim) == 4))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_zero_bwd_padding_kernel input dk ndim mismatch, expected 4", (long long)((((DLTensor*)dk_handle)[0].ndim)), (long long)(4));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  void* tilelang_zero_bwd_padding_kernel_dk_strides = (((DLTensor*)dk_handle)[0].strides);
  void* dk = (((DLTensor*)dk_handle)[0].data);
  if (!(((((DLTensor*)dv_handle)[0].ndim) == 4))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_zero_bwd_padding_kernel input dv ndim mismatch, expected 4", (long long)((((DLTensor*)dv_handle)[0].ndim)), (long long)(4));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  void* tilelang_zero_bwd_padding_kernel_dv_strides = (((DLTensor*)dv_handle)[0].strides);
  void* dv = (((DLTensor*)dv_handle)[0].data);
  if (!(((((DLTensor*)dg_handle)[0].ndim) == 3))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_zero_bwd_padding_kernel input dg ndim mismatch, expected 3", (long long)((((DLTensor*)dg_handle)[0].ndim)), (long long)(3));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  void* tilelang_zero_bwd_padding_kernel_dg_strides = (((DLTensor*)dg_handle)[0].strides);
  void* dg = (((DLTensor*)dg_handle)[0].data);
  if (!(((((DLTensor*)db_handle)[0].ndim) == 3))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_zero_bwd_padding_kernel input db ndim mismatch, expected 3", (long long)((((DLTensor*)db_handle)[0].ndim)), (long long)(3));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  void* tilelang_zero_bwd_padding_kernel_db_strides = (((DLTensor*)db_handle)[0].strides);
  void* db = (((DLTensor*)db_handle)[0].data);
  if (!(((((((DLTensor*)cu_seqlens_handle)[0].dtype.code) == (uint8_t)0) && ((((DLTensor*)cu_seqlens_handle)[0].dtype.bits) == (uint8_t)32)) && ((((DLTensor*)cu_seqlens_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_zero_bwd_padding_kernel input cu_seqlens dtype mismatch, expected int32");
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_zero_bwd_padding_kernel_cu_seqlens_shape)[0]) == (real_batch_size + 1)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_zero_bwd_padding_kernel input cu_seqlens shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_zero_bwd_padding_kernel_cu_seqlens_shape)[0])), (long long)((real_batch_size + 1)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval;
  if ((tilelang_zero_bwd_padding_kernel_cu_seqlens_strides == NULL)) {
    condval = 1;
  } else {
    condval = ((int32_t)((int64_t*)tilelang_zero_bwd_padding_kernel_cu_seqlens_strides)[0]);
  }
  if (!(((condval == 1) || (real_batch_size == 0)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_zero_bwd_padding_kernel input cu_seqlens strides[0] violates packed ABI constraint");
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)cu_seqlens_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_zero_bwd_padding_kernel input cu_seqlens byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)cu_seqlens_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)cu_seqlens_handle)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_zero_bwd_padding_kernel input cu_seqlens device_type mismatch, expected cuda", (long long)((((DLTensor*)cu_seqlens_handle)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((real_batch_size == -1) || !(cu_seqlens == NULL)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_zero_bwd_padding_kernel input cu_seqlens data pointer is NULL");
    return -1;
  }
  if (!(((((((DLTensor*)dq_handle)[0].dtype.code) == (uint8_t)4) && ((((DLTensor*)dq_handle)[0].dtype.bits) == (uint8_t)16)) && ((((DLTensor*)dq_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_zero_bwd_padding_kernel input dq dtype mismatch, expected bfloat16");
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_zero_bwd_padding_kernel_dq_shape)[0]) == 1))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_zero_bwd_padding_kernel input dq shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_zero_bwd_padding_kernel_dq_shape)[0])), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_zero_bwd_padding_kernel_dq_shape)[2]) == 4))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_zero_bwd_padding_kernel input dq shape[2] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_zero_bwd_padding_kernel_dq_shape)[2])), (long long)(4));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_zero_bwd_padding_kernel_dq_shape)[3]) == 128))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_zero_bwd_padding_kernel input dq shape[3] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_zero_bwd_padding_kernel_dq_shape)[3])), (long long)(128));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_1;
  if ((tilelang_zero_bwd_padding_kernel_dq_strides == NULL)) {
    condval_1 = 1;
  } else {
    condval_1 = ((int32_t)((int64_t*)tilelang_zero_bwd_padding_kernel_dq_strides)[3]);
  }
  if (!((condval_1 == 1))) {
    int32_t condval_2;
    if ((tilelang_zero_bwd_padding_kernel_dq_strides == NULL)) {
      condval_2 = 1;
    } else {
      condval_2 = ((int32_t)((int64_t*)tilelang_zero_bwd_padding_kernel_dq_strides)[3]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_zero_bwd_padding_kernel input dq strides[3] violates packed ABI constraint", (long long)(condval_2), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_3;
  if ((tilelang_zero_bwd_padding_kernel_dq_strides == NULL)) {
    condval_3 = 1;
  } else {
    condval_3 = ((int32_t)((int64_t*)tilelang_zero_bwd_padding_kernel_dq_strides)[2]);
  }
  if (!((condval_3 == 128))) {
    int32_t condval_4;
    if ((tilelang_zero_bwd_padding_kernel_dq_strides == NULL)) {
      condval_4 = 1;
    } else {
      condval_4 = ((int32_t)((int64_t*)tilelang_zero_bwd_padding_kernel_dq_strides)[2]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_zero_bwd_padding_kernel input dq strides[2] violates packed ABI constraint", (long long)(condval_4), (long long)(128));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_5;
  if ((tilelang_zero_bwd_padding_kernel_dq_strides == NULL)) {
    condval_5 = 1;
  } else {
    condval_5 = ((int32_t)((int64_t*)tilelang_zero_bwd_padding_kernel_dq_strides)[1]);
  }
  if (!(((condval_5 == 512) || (num_tokens == 1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_zero_bwd_padding_kernel input dq strides[1] violates packed ABI constraint");
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)dq_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_zero_bwd_padding_kernel input dq byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)dq_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)dq_handle)[0].device.device_id) == (((DLTensor*)cu_seqlens_handle)[0].device.device_id)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_zero_bwd_padding_kernel input dq device_id violates packed ABI constraint", (long long)((((DLTensor*)dq_handle)[0].device.device_id)), (long long)((((DLTensor*)cu_seqlens_handle)[0].device.device_id)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)dq_handle)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_zero_bwd_padding_kernel input dq device_type mismatch, expected cuda", (long long)((((DLTensor*)dq_handle)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((num_tokens == 0) || !(dq == NULL)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_zero_bwd_padding_kernel input dq data pointer is NULL");
    return -1;
  }
  if (!(((((((DLTensor*)dk_handle)[0].dtype.code) == (uint8_t)4) && ((((DLTensor*)dk_handle)[0].dtype.bits) == (uint8_t)16)) && ((((DLTensor*)dk_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_zero_bwd_padding_kernel input dk dtype mismatch, expected bfloat16");
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_zero_bwd_padding_kernel_dk_shape)[0]) == 1))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_zero_bwd_padding_kernel input dk shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_zero_bwd_padding_kernel_dk_shape)[0])), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_zero_bwd_padding_kernel_dq_shape)[1]) == ((int32_t)((int64_t*)tilelang_zero_bwd_padding_kernel_dk_shape)[1])))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_zero_bwd_padding_kernel input dk shape[1] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_zero_bwd_padding_kernel_dq_shape)[1])), (long long)(((int32_t)((int64_t*)tilelang_zero_bwd_padding_kernel_dk_shape)[1])));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_zero_bwd_padding_kernel_dk_shape)[2]) == 4))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_zero_bwd_padding_kernel input dk shape[2] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_zero_bwd_padding_kernel_dk_shape)[2])), (long long)(4));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_zero_bwd_padding_kernel_dk_shape)[3]) == 128))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_zero_bwd_padding_kernel input dk shape[3] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_zero_bwd_padding_kernel_dk_shape)[3])), (long long)(128));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_6;
  if ((tilelang_zero_bwd_padding_kernel_dk_strides == NULL)) {
    condval_6 = 1;
  } else {
    condval_6 = ((int32_t)((int64_t*)tilelang_zero_bwd_padding_kernel_dk_strides)[3]);
  }
  if (!((condval_6 == 1))) {
    int32_t condval_7;
    if ((tilelang_zero_bwd_padding_kernel_dk_strides == NULL)) {
      condval_7 = 1;
    } else {
      condval_7 = ((int32_t)((int64_t*)tilelang_zero_bwd_padding_kernel_dk_strides)[3]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_zero_bwd_padding_kernel input dk strides[3] violates packed ABI constraint", (long long)(condval_7), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_8;
  if ((tilelang_zero_bwd_padding_kernel_dk_strides == NULL)) {
    condval_8 = 1;
  } else {
    condval_8 = ((int32_t)((int64_t*)tilelang_zero_bwd_padding_kernel_dk_strides)[2]);
  }
  if (!((condval_8 == 128))) {
    int32_t condval_9;
    if ((tilelang_zero_bwd_padding_kernel_dk_strides == NULL)) {
      condval_9 = 1;
    } else {
      condval_9 = ((int32_t)((int64_t*)tilelang_zero_bwd_padding_kernel_dk_strides)[2]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_zero_bwd_padding_kernel input dk strides[2] violates packed ABI constraint", (long long)(condval_9), (long long)(128));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_10;
  if ((tilelang_zero_bwd_padding_kernel_dk_strides == NULL)) {
    condval_10 = 1;
  } else {
    condval_10 = ((int32_t)((int64_t*)tilelang_zero_bwd_padding_kernel_dk_strides)[1]);
  }
  if (!(((condval_10 == 512) || (num_tokens == 1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_zero_bwd_padding_kernel input dk strides[1] violates packed ABI constraint");
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)dk_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_zero_bwd_padding_kernel input dk byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)dk_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)dk_handle)[0].device.device_id) == (((DLTensor*)cu_seqlens_handle)[0].device.device_id)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_zero_bwd_padding_kernel input dk device_id violates packed ABI constraint", (long long)((((DLTensor*)dk_handle)[0].device.device_id)), (long long)((((DLTensor*)cu_seqlens_handle)[0].device.device_id)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)dk_handle)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_zero_bwd_padding_kernel input dk device_type mismatch, expected cuda", (long long)((((DLTensor*)dk_handle)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((num_tokens == 0) || !(dk == NULL)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_zero_bwd_padding_kernel input dk data pointer is NULL");
    return -1;
  }
  if (!(((((((DLTensor*)dv_handle)[0].dtype.code) == (uint8_t)4) && ((((DLTensor*)dv_handle)[0].dtype.bits) == (uint8_t)16)) && ((((DLTensor*)dv_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_zero_bwd_padding_kernel input dv dtype mismatch, expected bfloat16");
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_zero_bwd_padding_kernel_dv_shape)[0]) == 1))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_zero_bwd_padding_kernel input dv shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_zero_bwd_padding_kernel_dv_shape)[0])), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_zero_bwd_padding_kernel_dq_shape)[1]) == ((int32_t)((int64_t*)tilelang_zero_bwd_padding_kernel_dv_shape)[1])))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_zero_bwd_padding_kernel input dv shape[1] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_zero_bwd_padding_kernel_dq_shape)[1])), (long long)(((int32_t)((int64_t*)tilelang_zero_bwd_padding_kernel_dv_shape)[1])));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_zero_bwd_padding_kernel_dv_shape)[2]) == 4))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_zero_bwd_padding_kernel input dv shape[2] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_zero_bwd_padding_kernel_dv_shape)[2])), (long long)(4));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_zero_bwd_padding_kernel_dv_shape)[3]) == 128))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_zero_bwd_padding_kernel input dv shape[3] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_zero_bwd_padding_kernel_dv_shape)[3])), (long long)(128));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_11;
  if ((tilelang_zero_bwd_padding_kernel_dv_strides == NULL)) {
    condval_11 = 1;
  } else {
    condval_11 = ((int32_t)((int64_t*)tilelang_zero_bwd_padding_kernel_dv_strides)[3]);
  }
  if (!((condval_11 == 1))) {
    int32_t condval_12;
    if ((tilelang_zero_bwd_padding_kernel_dv_strides == NULL)) {
      condval_12 = 1;
    } else {
      condval_12 = ((int32_t)((int64_t*)tilelang_zero_bwd_padding_kernel_dv_strides)[3]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_zero_bwd_padding_kernel input dv strides[3] violates packed ABI constraint", (long long)(condval_12), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_13;
  if ((tilelang_zero_bwd_padding_kernel_dv_strides == NULL)) {
    condval_13 = 1;
  } else {
    condval_13 = ((int32_t)((int64_t*)tilelang_zero_bwd_padding_kernel_dv_strides)[2]);
  }
  if (!((condval_13 == 128))) {
    int32_t condval_14;
    if ((tilelang_zero_bwd_padding_kernel_dv_strides == NULL)) {
      condval_14 = 1;
    } else {
      condval_14 = ((int32_t)((int64_t*)tilelang_zero_bwd_padding_kernel_dv_strides)[2]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_zero_bwd_padding_kernel input dv strides[2] violates packed ABI constraint", (long long)(condval_14), (long long)(128));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_15;
  if ((tilelang_zero_bwd_padding_kernel_dv_strides == NULL)) {
    condval_15 = 1;
  } else {
    condval_15 = ((int32_t)((int64_t*)tilelang_zero_bwd_padding_kernel_dv_strides)[1]);
  }
  if (!(((condval_15 == 512) || (num_tokens == 1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_zero_bwd_padding_kernel input dv strides[1] violates packed ABI constraint");
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)dv_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_zero_bwd_padding_kernel input dv byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)dv_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)dv_handle)[0].device.device_id) == (((DLTensor*)cu_seqlens_handle)[0].device.device_id)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_zero_bwd_padding_kernel input dv device_id violates packed ABI constraint", (long long)((((DLTensor*)dv_handle)[0].device.device_id)), (long long)((((DLTensor*)cu_seqlens_handle)[0].device.device_id)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)dv_handle)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_zero_bwd_padding_kernel input dv device_type mismatch, expected cuda", (long long)((((DLTensor*)dv_handle)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((num_tokens == 0) || !(dv == NULL)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_zero_bwd_padding_kernel input dv data pointer is NULL");
    return -1;
  }
  if (!(((((((DLTensor*)dg_handle)[0].dtype.code) == (uint8_t)2) && ((((DLTensor*)dg_handle)[0].dtype.bits) == (uint8_t)32)) && ((((DLTensor*)dg_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_zero_bwd_padding_kernel input dg dtype mismatch, expected float32");
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_zero_bwd_padding_kernel_dg_shape)[0]) == 1))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_zero_bwd_padding_kernel input dg shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_zero_bwd_padding_kernel_dg_shape)[0])), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_zero_bwd_padding_kernel_dq_shape)[1]) == ((int32_t)((int64_t*)tilelang_zero_bwd_padding_kernel_dg_shape)[1])))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_zero_bwd_padding_kernel input dg shape[1] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_zero_bwd_padding_kernel_dq_shape)[1])), (long long)(((int32_t)((int64_t*)tilelang_zero_bwd_padding_kernel_dg_shape)[1])));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_zero_bwd_padding_kernel_dg_shape)[2]) == 4))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_zero_bwd_padding_kernel input dg shape[2] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_zero_bwd_padding_kernel_dg_shape)[2])), (long long)(4));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_16;
  if ((tilelang_zero_bwd_padding_kernel_dg_strides == NULL)) {
    condval_16 = 1;
  } else {
    condval_16 = ((int32_t)((int64_t*)tilelang_zero_bwd_padding_kernel_dg_strides)[2]);
  }
  if (!((condval_16 == 1))) {
    int32_t condval_17;
    if ((tilelang_zero_bwd_padding_kernel_dg_strides == NULL)) {
      condval_17 = 1;
    } else {
      condval_17 = ((int32_t)((int64_t*)tilelang_zero_bwd_padding_kernel_dg_strides)[2]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_zero_bwd_padding_kernel input dg strides[2] violates packed ABI constraint", (long long)(condval_17), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_18;
  if ((tilelang_zero_bwd_padding_kernel_dg_strides == NULL)) {
    condval_18 = 1;
  } else {
    condval_18 = ((int32_t)((int64_t*)tilelang_zero_bwd_padding_kernel_dg_strides)[1]);
  }
  if (!(((condval_18 == 4) || (num_tokens == 1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_zero_bwd_padding_kernel input dg strides[1] violates packed ABI constraint");
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)dg_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_zero_bwd_padding_kernel input dg byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)dg_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)dg_handle)[0].device.device_id) == (((DLTensor*)cu_seqlens_handle)[0].device.device_id)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_zero_bwd_padding_kernel input dg device_id violates packed ABI constraint", (long long)((((DLTensor*)dg_handle)[0].device.device_id)), (long long)((((DLTensor*)cu_seqlens_handle)[0].device.device_id)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)dg_handle)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_zero_bwd_padding_kernel input dg device_type mismatch, expected cuda", (long long)((((DLTensor*)dg_handle)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((num_tokens == 0) || !(dg == NULL)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_zero_bwd_padding_kernel input dg data pointer is NULL");
    return -1;
  }
  if (!(((((((DLTensor*)db_handle)[0].dtype.code) == (uint8_t)2) && ((((DLTensor*)db_handle)[0].dtype.bits) == (uint8_t)32)) && ((((DLTensor*)db_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_zero_bwd_padding_kernel input db dtype mismatch, expected float32");
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_zero_bwd_padding_kernel_db_shape)[0]) == 1))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_zero_bwd_padding_kernel input db shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_zero_bwd_padding_kernel_db_shape)[0])), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_zero_bwd_padding_kernel_dq_shape)[1]) == ((int32_t)((int64_t*)tilelang_zero_bwd_padding_kernel_db_shape)[1])))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_zero_bwd_padding_kernel input db shape[1] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_zero_bwd_padding_kernel_dq_shape)[1])), (long long)(((int32_t)((int64_t*)tilelang_zero_bwd_padding_kernel_db_shape)[1])));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_zero_bwd_padding_kernel_db_shape)[2]) == 4))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_zero_bwd_padding_kernel input db shape[2] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_zero_bwd_padding_kernel_db_shape)[2])), (long long)(4));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_19;
  if ((tilelang_zero_bwd_padding_kernel_db_strides == NULL)) {
    condval_19 = 1;
  } else {
    condval_19 = ((int32_t)((int64_t*)tilelang_zero_bwd_padding_kernel_db_strides)[2]);
  }
  if (!((condval_19 == 1))) {
    int32_t condval_20;
    if ((tilelang_zero_bwd_padding_kernel_db_strides == NULL)) {
      condval_20 = 1;
    } else {
      condval_20 = ((int32_t)((int64_t*)tilelang_zero_bwd_padding_kernel_db_strides)[2]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_zero_bwd_padding_kernel input db strides[2] violates packed ABI constraint", (long long)(condval_20), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_21;
  if ((tilelang_zero_bwd_padding_kernel_db_strides == NULL)) {
    condval_21 = 1;
  } else {
    condval_21 = ((int32_t)((int64_t*)tilelang_zero_bwd_padding_kernel_db_strides)[1]);
  }
  if (!(((condval_21 == 4) || (num_tokens == 1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_zero_bwd_padding_kernel input db strides[1] violates packed ABI constraint");
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)db_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_zero_bwd_padding_kernel input db byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)db_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)db_handle)[0].device.device_id) == (((DLTensor*)cu_seqlens_handle)[0].device.device_id)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_zero_bwd_padding_kernel input db device_id violates packed ABI constraint", (long long)((((DLTensor*)db_handle)[0].device.device_id)), (long long)((((DLTensor*)cu_seqlens_handle)[0].device.device_id)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)db_handle)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_zero_bwd_padding_kernel input db device_type mismatch, expected cuda", (long long)((((DLTensor*)db_handle)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((num_tokens == 0) || !(db == NULL)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_zero_bwd_padding_kernel input db data pointer is NULL");
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
  if (cu_seqlens == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = cu_seqlens;
  if (db == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = db;
  if (dg == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = dg;
  if (dk == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[3].v_ptr) = dk;
  if (dq == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[4].v_ptr) = dq;
  if (dv == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[5].v_ptr) = dv;
  (((TVMFFIAny*)stack_ffi_any)[6].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[6].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[6].v_int64) = ((int64_t)num_tokens);
  (((TVMFFIAny*)stack_ffi_any)[7].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[7].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[7].v_int64) = ((int64_t)real_batch_size);
  (((TVMFFIAny*)stack_ffi_any)[8].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[8].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[8].v_int64) = ((int64_t)4);
  (((TVMFFIAny*)stack_ffi_any)[9].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[9].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[9].v_int64) = ((int64_t)128);
  (((TVMFFIAny*)stack_ffi_any)[10].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[10].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[10].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[11].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[11].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[11].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[12].type_index) = 0;
  (((TVMFFIAny*)stack_ffi_any)[12].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[12].v_int64) = (int64_t)0;
  if (tilelang_zero_bwd_padding_kernel_kernel_packed == NULL) {
    if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "tilelang_zero_bwd_padding_kernel_kernel", &tilelang_zero_bwd_padding_kernel_kernel_packed) != 0) {
      return -1;
    }
  }
  TVMFFIAny result_2;
  result_2.type_index = kTVMFFINone;
  result_2.zero_padding = 0;
  result_2.v_int64 = 0;
  if (TVMFFIFunctionCall(tilelang_zero_bwd_padding_kernel_kernel_packed, (TVMFFIAny*) stack_ffi_any, 12, &result_2) != 0) {
    return -1;
  }
  return 0;
}

// CodegenC: NOTE: Auto-generated entry function
#ifdef __cplusplus
extern "C"
#endif
int32_t __tvm_ffi_main(void* self, void* args,int num_args, void* result) {
  return __tvm_ffi_tilelang_zero_bwd_padding_kernel(self, args, num_args, result);
}
