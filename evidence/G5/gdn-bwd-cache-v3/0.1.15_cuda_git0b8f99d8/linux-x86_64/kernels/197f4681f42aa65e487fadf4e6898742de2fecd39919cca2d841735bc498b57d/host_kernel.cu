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
static void* tilelang_group_reduce_vector_kernel_kernel_packed = NULL;
#ifdef __cplusplus
extern "C"
#endif
int32_t __tvm_ffi_tilelang_group_reduce_vector_kernel(void* self_handle, void* args, int32_t num_args, void* result);
#ifdef __cplusplus
extern "C"
#endif
int32_t __tvm_ffi_tilelang_group_reduce_vector_kernel(void* self_handle, void* args, int32_t num_args, void* result) {
  TL_ALIGN(128) TVMFFIAny stack[11];
  void* stack_ffi_any = stack;
  if (!((num_args == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "tilelang_group_reduce_vector_kernel: num_args should be 2", (long long)(num_args), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(!(args == NULL))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "tilelang_group_reduce_vector_kernel: args pointer is NULL");
    return -1;
  }
  int32_t buffer_handle_type_index = (((TVMFFIAny*)args)[0].type_index);
  if (!(((((buffer_handle_type_index == 0) || (buffer_handle_type_index == 4)) || (buffer_handle_type_index == 7)) || (64 <= buffer_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_group_reduce_vector_kernel input buffer expected pointer or tensor handle");
    return -1;
  }
  int32_t result_handle_type_index = (((TVMFFIAny*)args)[1].type_index);
  if (!(((((result_handle_type_index == 0) || (result_handle_type_index == 4)) || (result_handle_type_index == 7)) || (64 <= result_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_group_reduce_vector_kernel input result expected pointer or tensor handle");
    return -1;
  }
  void* buffer_handle = ((buffer_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[0].v_ptr) + 24)) : (((TVMFFIAny*)args)[0].v_ptr));
  void* result_handle = ((result_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[1].v_ptr) + 24)) : (((TVMFFIAny*)args)[1].v_ptr));
  bool tilelang_group_reduce_vector_kernel_buffer_is_null = (buffer_handle == NULL);
  if (!(!tilelang_group_reduce_vector_kernel_buffer_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "tilelang_group_reduce_vector_kernel.buffer is expected to have non-NULL pointer");
    return -1;
  }
  bool tilelang_group_reduce_vector_kernel_result_is_null = (result_handle == NULL);
  if (!(!tilelang_group_reduce_vector_kernel_result_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "tilelang_group_reduce_vector_kernel.result is expected to have non-NULL pointer");
    return -1;
  }
  void* tilelang_group_reduce_vector_kernel_buffer_shape = (((DLTensor*)buffer_handle)[0].shape);
  void* tilelang_group_reduce_vector_kernel_result_shape = (((DLTensor*)result_handle)[0].shape);
  if (!(((((DLTensor*)buffer_handle)[0].ndim) == 4))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_group_reduce_vector_kernel input buffer ndim mismatch, expected 4", (long long)((((DLTensor*)buffer_handle)[0].ndim)), (long long)(4));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((bool)1)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "Symbolic shape variable batch_size requires at least one non-null buffer among: tilelang_group_reduce_vector_kernel.buffer, tilelang_group_reduce_vector_kernel.result");
    return -1;
  }
  int32_t batch_size = ((int32_t)((int64_t*)tilelang_group_reduce_vector_kernel_buffer_shape)[0]);
  if (!((bool)1)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "Symbolic shape variable num_tokens requires at least one non-null buffer among: tilelang_group_reduce_vector_kernel.buffer, tilelang_group_reduce_vector_kernel.result");
    return -1;
  }
  int32_t num_tokens = ((int32_t)((int64_t*)tilelang_group_reduce_vector_kernel_buffer_shape)[1]);
  void* tilelang_group_reduce_vector_kernel_buffer_strides = (((DLTensor*)buffer_handle)[0].strides);
  int32_t dev_id = (((DLTensor*)buffer_handle)[0].device.device_id);
  void* buffer = (((DLTensor*)buffer_handle)[0].data);
  if (!(((((DLTensor*)result_handle)[0].ndim) == 4))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_group_reduce_vector_kernel input result ndim mismatch, expected 4", (long long)((((DLTensor*)result_handle)[0].ndim)), (long long)(4));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  void* tilelang_group_reduce_vector_kernel_result_strides = (((DLTensor*)result_handle)[0].strides);
  void* result_1 = (((DLTensor*)result_handle)[0].data);
  if (!(((((((DLTensor*)buffer_handle)[0].dtype.code) == (uint8_t)4) && ((((DLTensor*)buffer_handle)[0].dtype.bits) == (uint8_t)16)) && ((((DLTensor*)buffer_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_group_reduce_vector_kernel input buffer dtype mismatch, expected bfloat16");
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_group_reduce_vector_kernel_buffer_shape)[2]) == 4))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_group_reduce_vector_kernel input buffer shape[2] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_group_reduce_vector_kernel_buffer_shape)[2])), (long long)(4));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_group_reduce_vector_kernel_buffer_shape)[3]) == 128))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_group_reduce_vector_kernel input buffer shape[3] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_group_reduce_vector_kernel_buffer_shape)[3])), (long long)(128));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval;
  if ((tilelang_group_reduce_vector_kernel_buffer_strides == NULL)) {
    condval = 1;
  } else {
    condval = ((int32_t)((int64_t*)tilelang_group_reduce_vector_kernel_buffer_strides)[3]);
  }
  if (!((condval == 1))) {
    int32_t condval_1;
    if ((tilelang_group_reduce_vector_kernel_buffer_strides == NULL)) {
      condval_1 = 1;
    } else {
      condval_1 = ((int32_t)((int64_t*)tilelang_group_reduce_vector_kernel_buffer_strides)[3]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_group_reduce_vector_kernel input buffer strides[3] violates packed ABI constraint", (long long)(condval_1), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_2;
  if ((tilelang_group_reduce_vector_kernel_buffer_strides == NULL)) {
    condval_2 = 1;
  } else {
    condval_2 = ((int32_t)((int64_t*)tilelang_group_reduce_vector_kernel_buffer_strides)[2]);
  }
  if (!((condval_2 == 128))) {
    int32_t condval_3;
    if ((tilelang_group_reduce_vector_kernel_buffer_strides == NULL)) {
      condval_3 = 1;
    } else {
      condval_3 = ((int32_t)((int64_t*)tilelang_group_reduce_vector_kernel_buffer_strides)[2]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_group_reduce_vector_kernel input buffer strides[2] violates packed ABI constraint", (long long)(condval_3), (long long)(128));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_4;
  if ((tilelang_group_reduce_vector_kernel_buffer_strides == NULL)) {
    condval_4 = 1;
  } else {
    condval_4 = ((int32_t)((int64_t*)tilelang_group_reduce_vector_kernel_buffer_strides)[1]);
  }
  if (!(((condval_4 == 512) || (num_tokens == 1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_group_reduce_vector_kernel input buffer strides[1] violates packed ABI constraint");
    return -1;
  }
  int32_t condval_5;
  if ((tilelang_group_reduce_vector_kernel_buffer_strides == NULL)) {
    condval_5 = 1;
  } else {
    condval_5 = ((int32_t)((int64_t*)tilelang_group_reduce_vector_kernel_buffer_strides)[0]);
  }
  if (!(((((num_tokens * 512) == condval_5) || (num_tokens == 0)) || (batch_size == 1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_group_reduce_vector_kernel input buffer strides[0] violates packed ABI constraint");
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)buffer_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_group_reduce_vector_kernel input buffer byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)buffer_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)buffer_handle)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_group_reduce_vector_kernel input buffer device_type mismatch, expected cuda", (long long)((((DLTensor*)buffer_handle)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((batch_size == 0) || (num_tokens == 0)) || !(buffer == NULL)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_group_reduce_vector_kernel input buffer data pointer is NULL");
    return -1;
  }
  if (!(((((((DLTensor*)result_handle)[0].dtype.code) == (uint8_t)4) && ((((DLTensor*)result_handle)[0].dtype.bits) == (uint8_t)16)) && ((((DLTensor*)result_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_group_reduce_vector_kernel input result dtype mismatch, expected bfloat16");
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_group_reduce_vector_kernel_buffer_shape)[0]) == ((int32_t)((int64_t*)tilelang_group_reduce_vector_kernel_result_shape)[0])))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_group_reduce_vector_kernel input result shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_group_reduce_vector_kernel_buffer_shape)[0])), (long long)(((int32_t)((int64_t*)tilelang_group_reduce_vector_kernel_result_shape)[0])));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_group_reduce_vector_kernel_buffer_shape)[1]) == ((int32_t)((int64_t*)tilelang_group_reduce_vector_kernel_result_shape)[1])))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_group_reduce_vector_kernel input result shape[1] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_group_reduce_vector_kernel_buffer_shape)[1])), (long long)(((int32_t)((int64_t*)tilelang_group_reduce_vector_kernel_result_shape)[1])));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_group_reduce_vector_kernel_result_shape)[2]) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_group_reduce_vector_kernel input result shape[2] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_group_reduce_vector_kernel_result_shape)[2])), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_group_reduce_vector_kernel_result_shape)[3]) == 128))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_group_reduce_vector_kernel input result shape[3] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_group_reduce_vector_kernel_result_shape)[3])), (long long)(128));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_6;
  if ((tilelang_group_reduce_vector_kernel_result_strides == NULL)) {
    condval_6 = 1;
  } else {
    condval_6 = ((int32_t)((int64_t*)tilelang_group_reduce_vector_kernel_result_strides)[3]);
  }
  if (!((condval_6 == 1))) {
    int32_t condval_7;
    if ((tilelang_group_reduce_vector_kernel_result_strides == NULL)) {
      condval_7 = 1;
    } else {
      condval_7 = ((int32_t)((int64_t*)tilelang_group_reduce_vector_kernel_result_strides)[3]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_group_reduce_vector_kernel input result strides[3] violates packed ABI constraint", (long long)(condval_7), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_8;
  if ((tilelang_group_reduce_vector_kernel_result_strides == NULL)) {
    condval_8 = 1;
  } else {
    condval_8 = ((int32_t)((int64_t*)tilelang_group_reduce_vector_kernel_result_strides)[2]);
  }
  if (!((condval_8 == 128))) {
    int32_t condval_9;
    if ((tilelang_group_reduce_vector_kernel_result_strides == NULL)) {
      condval_9 = 1;
    } else {
      condval_9 = ((int32_t)((int64_t*)tilelang_group_reduce_vector_kernel_result_strides)[2]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_group_reduce_vector_kernel input result strides[2] violates packed ABI constraint", (long long)(condval_9), (long long)(128));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_10;
  if ((tilelang_group_reduce_vector_kernel_result_strides == NULL)) {
    condval_10 = 1;
  } else {
    condval_10 = ((int32_t)((int64_t*)tilelang_group_reduce_vector_kernel_result_strides)[1]);
  }
  if (!(((condval_10 == 256) || (num_tokens == 1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_group_reduce_vector_kernel input result strides[1] violates packed ABI constraint");
    return -1;
  }
  int32_t condval_11;
  if ((tilelang_group_reduce_vector_kernel_result_strides == NULL)) {
    condval_11 = 1;
  } else {
    condval_11 = ((int32_t)((int64_t*)tilelang_group_reduce_vector_kernel_result_strides)[0]);
  }
  if (!(((((num_tokens * 256) == condval_11) || (num_tokens == 0)) || (batch_size == 1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_group_reduce_vector_kernel input result strides[0] violates packed ABI constraint");
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)result_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_group_reduce_vector_kernel input result byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)result_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)result_handle)[0].device.device_id) == (((DLTensor*)buffer_handle)[0].device.device_id)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_group_reduce_vector_kernel input result device_id violates packed ABI constraint", (long long)((((DLTensor*)result_handle)[0].device.device_id)), (long long)((((DLTensor*)buffer_handle)[0].device.device_id)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)result_handle)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_group_reduce_vector_kernel input result device_type mismatch, expected cuda", (long long)((((DLTensor*)result_handle)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((batch_size == 0) || (num_tokens == 0)) || !(result_1 == NULL)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_group_reduce_vector_kernel input result data pointer is NULL");
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
  TVMFFIAny result_2;
  result_2.type_index = kTVMFFINone;
  result_2.zero_padding = 0;
  result_2.v_int64 = 0;
  if (TVMFFIFunctionCall(__tvm_set_device_packed, (TVMFFIAny*) stack_ffi_any, 2, &result_2) != 0) {
    return -1;
  }
  if (buffer == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = buffer;
  if (result_1 == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = result_1;
  (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = ((int64_t)batch_size);
  (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)num_tokens);
  (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)((num_tokens + 15) >> 4));
  (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = ((int64_t)2);
  (((TVMFFIAny*)stack_ffi_any)[6].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[6].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[6].v_int64) = ((int64_t)batch_size);
  (((TVMFFIAny*)stack_ffi_any)[7].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[7].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[7].v_int64) = ((int64_t)128);
  (((TVMFFIAny*)stack_ffi_any)[8].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[8].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[8].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[9].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[9].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[9].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[10].type_index) = 0;
  (((TVMFFIAny*)stack_ffi_any)[10].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[10].v_int64) = (int64_t)0;
  if (tilelang_group_reduce_vector_kernel_kernel_packed == NULL) {
    if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "tilelang_group_reduce_vector_kernel_kernel", &tilelang_group_reduce_vector_kernel_kernel_packed) != 0) {
      return -1;
    }
  }
  TVMFFIAny result_3;
  result_3.type_index = kTVMFFINone;
  result_3.zero_padding = 0;
  result_3.v_int64 = 0;
  if (TVMFFIFunctionCall(tilelang_group_reduce_vector_kernel_kernel_packed, (TVMFFIAny*) stack_ffi_any, 10, &result_3) != 0) {
    return -1;
  }
  return 0;
}

// CodegenC: NOTE: Auto-generated entry function
#ifdef __cplusplus
extern "C"
#endif
int32_t __tvm_ffi_main(void* self, void* args,int num_args, void* result) {
  return __tvm_ffi_tilelang_group_reduce_vector_kernel(self, args, num_args, result);
}
