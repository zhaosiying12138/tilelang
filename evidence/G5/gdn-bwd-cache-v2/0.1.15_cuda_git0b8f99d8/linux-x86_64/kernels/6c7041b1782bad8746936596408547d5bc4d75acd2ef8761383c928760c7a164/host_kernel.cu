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
static void* tilelang_chunk_local_cumsum_kernel_kernel_packed = NULL;
#ifdef __cplusplus
extern "C"
#endif
int32_t __tvm_ffi_tilelang_chunk_local_cumsum_kernel(void* self_handle, void* args, int32_t num_args, void* result);
#ifdef __cplusplus
extern "C"
#endif
int32_t __tvm_ffi_tilelang_chunk_local_cumsum_kernel(void* self_handle, void* args, int32_t num_args, void* result) {
  TL_ALIGN(128) TVMFFIAny stack[11];
  void* stack_ffi_any = stack;
  if (!((num_args == 3))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "tilelang_chunk_local_cumsum_kernel: num_args should be 3", (long long)(num_args), (long long)(3));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(!(args == NULL))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "tilelang_chunk_local_cumsum_kernel: args pointer is NULL");
    return -1;
  }
  int32_t g_raw_handle_type_index = (((TVMFFIAny*)args)[0].type_index);
  if (!(((((g_raw_handle_type_index == 0) || (g_raw_handle_type_index == 4)) || (g_raw_handle_type_index == 7)) || (64 <= g_raw_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_chunk_local_cumsum_kernel input g_raw expected pointer or tensor handle");
    return -1;
  }
  int32_t g_cumsum_handle_type_index = (((TVMFFIAny*)args)[1].type_index);
  if (!(((((g_cumsum_handle_type_index == 0) || (g_cumsum_handle_type_index == 4)) || (g_cumsum_handle_type_index == 7)) || (64 <= g_cumsum_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_chunk_local_cumsum_kernel input g_cumsum expected pointer or tensor handle");
    return -1;
  }
  int32_t num_chunks_type_index = (((TVMFFIAny*)args)[2].type_index);
  if (!(((num_chunks_type_index == 1) || (num_chunks_type_index == 2)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_chunk_local_cumsum_kernel scalar num_chunks expected integer");
    return -1;
  }
  void* g_raw_handle = ((g_raw_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[0].v_ptr) + 24)) : (((TVMFFIAny*)args)[0].v_ptr));
  void* g_cumsum_handle = ((g_cumsum_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[1].v_ptr) + 24)) : (((TVMFFIAny*)args)[1].v_ptr));
  int32_t num_chunks = ((int32_t)(((TVMFFIAny*)args)[2].v_int64));
  bool tilelang_chunk_local_cumsum_kernel_g_raw_is_null = (g_raw_handle == NULL);
  if (!(!tilelang_chunk_local_cumsum_kernel_g_raw_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "tilelang_chunk_local_cumsum_kernel.g_raw is expected to have non-NULL pointer");
    return -1;
  }
  bool tilelang_chunk_local_cumsum_kernel_g_cumsum_is_null = (g_cumsum_handle == NULL);
  if (!(!tilelang_chunk_local_cumsum_kernel_g_cumsum_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "tilelang_chunk_local_cumsum_kernel.g_cumsum is expected to have non-NULL pointer");
    return -1;
  }
  void* tilelang_chunk_local_cumsum_kernel_g_raw_shape = (((DLTensor*)g_raw_handle)[0].shape);
  void* tilelang_chunk_local_cumsum_kernel_g_cumsum_shape = (((DLTensor*)g_cumsum_handle)[0].shape);
  if (!(((((DLTensor*)g_raw_handle)[0].ndim) == 3))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_chunk_local_cumsum_kernel input g_raw ndim mismatch, expected 3", (long long)((((DLTensor*)g_raw_handle)[0].ndim)), (long long)(3));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((bool)1)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "Symbolic shape variable data_batch_size requires at least one non-null buffer among: tilelang_chunk_local_cumsum_kernel.g_raw, tilelang_chunk_local_cumsum_kernel.g_cumsum");
    return -1;
  }
  int32_t data_batch_size = ((int32_t)((int64_t*)tilelang_chunk_local_cumsum_kernel_g_raw_shape)[0]);
  if (!((bool)1)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "Symbolic shape variable seq_end_idx requires at least one non-null buffer among: tilelang_chunk_local_cumsum_kernel.g_raw, tilelang_chunk_local_cumsum_kernel.g_cumsum");
    return -1;
  }
  int32_t seq_end_idx = ((int32_t)((int64_t*)tilelang_chunk_local_cumsum_kernel_g_raw_shape)[1]);
  void* tilelang_chunk_local_cumsum_kernel_g_raw_strides = (((DLTensor*)g_raw_handle)[0].strides);
  int32_t dev_id = (((DLTensor*)g_raw_handle)[0].device.device_id);
  void* g_raw = (((DLTensor*)g_raw_handle)[0].data);
  if (!(((((DLTensor*)g_cumsum_handle)[0].ndim) == 3))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_chunk_local_cumsum_kernel input g_cumsum ndim mismatch, expected 3", (long long)((((DLTensor*)g_cumsum_handle)[0].ndim)), (long long)(3));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  void* tilelang_chunk_local_cumsum_kernel_g_cumsum_strides = (((DLTensor*)g_cumsum_handle)[0].strides);
  void* g_cumsum = (((DLTensor*)g_cumsum_handle)[0].data);
  if (!(((((((DLTensor*)g_raw_handle)[0].dtype.code) == (uint8_t)2) && ((((DLTensor*)g_raw_handle)[0].dtype.bits) == (uint8_t)32)) && ((((DLTensor*)g_raw_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_chunk_local_cumsum_kernel input g_raw dtype mismatch, expected float32");
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_chunk_local_cumsum_kernel_g_raw_shape)[2]) == 4))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_chunk_local_cumsum_kernel input g_raw shape[2] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_chunk_local_cumsum_kernel_g_raw_shape)[2])), (long long)(4));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval;
  if ((tilelang_chunk_local_cumsum_kernel_g_raw_strides == NULL)) {
    condval = 1;
  } else {
    condval = ((int32_t)((int64_t*)tilelang_chunk_local_cumsum_kernel_g_raw_strides)[2]);
  }
  if (!((condval == 1))) {
    int32_t condval_1;
    if ((tilelang_chunk_local_cumsum_kernel_g_raw_strides == NULL)) {
      condval_1 = 1;
    } else {
      condval_1 = ((int32_t)((int64_t*)tilelang_chunk_local_cumsum_kernel_g_raw_strides)[2]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_chunk_local_cumsum_kernel input g_raw strides[2] violates packed ABI constraint", (long long)(condval_1), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_2;
  if ((tilelang_chunk_local_cumsum_kernel_g_raw_strides == NULL)) {
    condval_2 = 1;
  } else {
    condval_2 = ((int32_t)((int64_t*)tilelang_chunk_local_cumsum_kernel_g_raw_strides)[1]);
  }
  if (!(((condval_2 == 4) || (seq_end_idx == 1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_chunk_local_cumsum_kernel input g_raw strides[1] violates packed ABI constraint");
    return -1;
  }
  int32_t condval_3;
  if ((tilelang_chunk_local_cumsum_kernel_g_raw_strides == NULL)) {
    condval_3 = 1;
  } else {
    condval_3 = ((int32_t)((int64_t*)tilelang_chunk_local_cumsum_kernel_g_raw_strides)[0]);
  }
  if (!(((((seq_end_idx * 4) == condval_3) || (seq_end_idx == 0)) || (data_batch_size == 1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_chunk_local_cumsum_kernel input g_raw strides[0] violates packed ABI constraint");
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)g_raw_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_chunk_local_cumsum_kernel input g_raw byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)g_raw_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)g_raw_handle)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_chunk_local_cumsum_kernel input g_raw device_type mismatch, expected cuda", (long long)((((DLTensor*)g_raw_handle)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((data_batch_size == 0) || (seq_end_idx == 0)) || !(g_raw == NULL)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_chunk_local_cumsum_kernel input g_raw data pointer is NULL");
    return -1;
  }
  if (!(((((((DLTensor*)g_cumsum_handle)[0].dtype.code) == (uint8_t)2) && ((((DLTensor*)g_cumsum_handle)[0].dtype.bits) == (uint8_t)32)) && ((((DLTensor*)g_cumsum_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_chunk_local_cumsum_kernel input g_cumsum dtype mismatch, expected float32");
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_chunk_local_cumsum_kernel_g_raw_shape)[0]) == ((int32_t)((int64_t*)tilelang_chunk_local_cumsum_kernel_g_cumsum_shape)[0])))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_chunk_local_cumsum_kernel input g_cumsum shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_chunk_local_cumsum_kernel_g_raw_shape)[0])), (long long)(((int32_t)((int64_t*)tilelang_chunk_local_cumsum_kernel_g_cumsum_shape)[0])));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_chunk_local_cumsum_kernel_g_raw_shape)[1]) == ((int32_t)((int64_t*)tilelang_chunk_local_cumsum_kernel_g_cumsum_shape)[1])))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_chunk_local_cumsum_kernel input g_cumsum shape[1] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_chunk_local_cumsum_kernel_g_raw_shape)[1])), (long long)(((int32_t)((int64_t*)tilelang_chunk_local_cumsum_kernel_g_cumsum_shape)[1])));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_chunk_local_cumsum_kernel_g_cumsum_shape)[2]) == 4))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_chunk_local_cumsum_kernel input g_cumsum shape[2] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_chunk_local_cumsum_kernel_g_cumsum_shape)[2])), (long long)(4));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_4;
  if ((tilelang_chunk_local_cumsum_kernel_g_cumsum_strides == NULL)) {
    condval_4 = 1;
  } else {
    condval_4 = ((int32_t)((int64_t*)tilelang_chunk_local_cumsum_kernel_g_cumsum_strides)[2]);
  }
  if (!((condval_4 == 1))) {
    int32_t condval_5;
    if ((tilelang_chunk_local_cumsum_kernel_g_cumsum_strides == NULL)) {
      condval_5 = 1;
    } else {
      condval_5 = ((int32_t)((int64_t*)tilelang_chunk_local_cumsum_kernel_g_cumsum_strides)[2]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_chunk_local_cumsum_kernel input g_cumsum strides[2] violates packed ABI constraint", (long long)(condval_5), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_6;
  if ((tilelang_chunk_local_cumsum_kernel_g_cumsum_strides == NULL)) {
    condval_6 = 1;
  } else {
    condval_6 = ((int32_t)((int64_t*)tilelang_chunk_local_cumsum_kernel_g_cumsum_strides)[1]);
  }
  if (!(((condval_6 == 4) || (seq_end_idx == 1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_chunk_local_cumsum_kernel input g_cumsum strides[1] violates packed ABI constraint");
    return -1;
  }
  int32_t condval_7;
  if ((tilelang_chunk_local_cumsum_kernel_g_cumsum_strides == NULL)) {
    condval_7 = 1;
  } else {
    condval_7 = ((int32_t)((int64_t*)tilelang_chunk_local_cumsum_kernel_g_cumsum_strides)[0]);
  }
  if (!(((((seq_end_idx * 4) == condval_7) || (seq_end_idx == 0)) || (data_batch_size == 1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_chunk_local_cumsum_kernel input g_cumsum strides[0] violates packed ABI constraint");
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)g_cumsum_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_chunk_local_cumsum_kernel input g_cumsum byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)g_cumsum_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)g_cumsum_handle)[0].device.device_id) == (((DLTensor*)g_raw_handle)[0].device.device_id)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_chunk_local_cumsum_kernel input g_cumsum device_id violates packed ABI constraint", (long long)((((DLTensor*)g_cumsum_handle)[0].device.device_id)), (long long)((((DLTensor*)g_raw_handle)[0].device.device_id)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)g_cumsum_handle)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_chunk_local_cumsum_kernel input g_cumsum device_type mismatch, expected cuda", (long long)((((DLTensor*)g_cumsum_handle)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((data_batch_size == 0) || (seq_end_idx == 0)) || !(g_cumsum == NULL)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_chunk_local_cumsum_kernel input g_cumsum data pointer is NULL");
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
  if (g_cumsum == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = g_cumsum;
  if (g_raw == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = g_raw;
  (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = ((int64_t)data_batch_size);
  (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)num_chunks);
  (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)seq_end_idx);
  (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = ((int64_t)num_chunks);
  (((TVMFFIAny*)stack_ffi_any)[6].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[6].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[6].v_int64) = ((int64_t)128);
  (((TVMFFIAny*)stack_ffi_any)[7].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[7].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[7].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[8].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[8].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[8].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[9].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[9].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[9].v_int64) = ((int64_t)1152);
  (((TVMFFIAny*)stack_ffi_any)[10].type_index) = 0;
  (((TVMFFIAny*)stack_ffi_any)[10].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[10].v_int64) = (int64_t)0;
  if (tilelang_chunk_local_cumsum_kernel_kernel_packed == NULL) {
    if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "tilelang_chunk_local_cumsum_kernel_kernel", &tilelang_chunk_local_cumsum_kernel_kernel_packed) != 0) {
      return -1;
    }
  }
  TVMFFIAny result_2;
  result_2.type_index = kTVMFFINone;
  result_2.zero_padding = 0;
  result_2.v_int64 = 0;
  if (TVMFFIFunctionCall(tilelang_chunk_local_cumsum_kernel_kernel_packed, (TVMFFIAny*) stack_ffi_any, 10, &result_2) != 0) {
    return -1;
  }
  return 0;
}

// CodegenC: NOTE: Auto-generated entry function
#ifdef __cplusplus
extern "C"
#endif
int32_t __tvm_ffi_main(void* self, void* args,int num_args, void* result) {
  return __tvm_ffi_tilelang_chunk_local_cumsum_kernel(self, args, num_args, result);
}
