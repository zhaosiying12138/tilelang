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
static void* tilelang_kkt_solve_kernel_kernel_packed = NULL;
#ifdef __cplusplus
extern "C"
#endif
int32_t __tvm_ffi_tilelang_kkt_solve_kernel(void* self_handle, void* args, int32_t num_args, void* result);
#ifdef __cplusplus
extern "C"
#endif
int32_t __tvm_ffi_tilelang_kkt_solve_kernel(void* self_handle, void* args, int32_t num_args, void* result) {
  TL_ALIGN(128) TVMFFIAny stack[12];
  void* stack_ffi_any = stack;
  if (!((num_args == 4))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "tilelang_kkt_solve_kernel: num_args should be 4", (long long)(num_args), (long long)(4));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(!(args == NULL))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "tilelang_kkt_solve_kernel: args pointer is NULL");
    return -1;
  }
  int32_t k_handle_type_index = (((TVMFFIAny*)args)[0].type_index);
  if (!(((((k_handle_type_index == 0) || (k_handle_type_index == 4)) || (k_handle_type_index == 7)) || (64 <= k_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_kkt_solve_kernel input k expected pointer or tensor handle");
    return -1;
  }
  int32_t b_handle_type_index = (((TVMFFIAny*)args)[1].type_index);
  if (!(((((b_handle_type_index == 0) || (b_handle_type_index == 4)) || (b_handle_type_index == 7)) || (64 <= b_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_kkt_solve_kernel input b expected pointer or tensor handle");
    return -1;
  }
  int32_t a_handle_type_index = (((TVMFFIAny*)args)[2].type_index);
  if (!(((((a_handle_type_index == 0) || (a_handle_type_index == 4)) || (a_handle_type_index == 7)) || (64 <= a_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_kkt_solve_kernel input a expected pointer or tensor handle");
    return -1;
  }
  int32_t num_chunks_type_index = (((TVMFFIAny*)args)[3].type_index);
  if (!(((num_chunks_type_index == 1) || (num_chunks_type_index == 2)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_kkt_solve_kernel scalar num_chunks expected integer");
    return -1;
  }
  void* k_handle = ((k_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[0].v_ptr) + 24)) : (((TVMFFIAny*)args)[0].v_ptr));
  void* b_handle = ((b_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[1].v_ptr) + 24)) : (((TVMFFIAny*)args)[1].v_ptr));
  void* a_handle = ((a_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[2].v_ptr) + 24)) : (((TVMFFIAny*)args)[2].v_ptr));
  int32_t num_chunks = ((int32_t)(((TVMFFIAny*)args)[3].v_int64));
  bool tilelang_kkt_solve_kernel_k_is_null = (k_handle == NULL);
  if (!(!tilelang_kkt_solve_kernel_k_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "tilelang_kkt_solve_kernel.k is expected to have non-NULL pointer");
    return -1;
  }
  bool tilelang_kkt_solve_kernel_b_is_null = (b_handle == NULL);
  if (!(!tilelang_kkt_solve_kernel_b_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "tilelang_kkt_solve_kernel.b is expected to have non-NULL pointer");
    return -1;
  }
  bool tilelang_kkt_solve_kernel_a_is_null = (a_handle == NULL);
  if (!(!tilelang_kkt_solve_kernel_a_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "tilelang_kkt_solve_kernel.a is expected to have non-NULL pointer");
    return -1;
  }
  void* tilelang_kkt_solve_kernel_k_shape = (((DLTensor*)k_handle)[0].shape);
  void* tilelang_kkt_solve_kernel_b_shape = (((DLTensor*)b_handle)[0].shape);
  void* tilelang_kkt_solve_kernel_a_shape = (((DLTensor*)a_handle)[0].shape);
  if (!(((((DLTensor*)k_handle)[0].ndim) == 4))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_kkt_solve_kernel input k ndim mismatch, expected 4", (long long)((((DLTensor*)k_handle)[0].ndim)), (long long)(4));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((bool)1)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "Symbolic shape variable data_batch_size requires at least one non-null buffer among: tilelang_kkt_solve_kernel.k, tilelang_kkt_solve_kernel.b, tilelang_kkt_solve_kernel.a");
    return -1;
  }
  int32_t data_batch_size = ((int32_t)((int64_t*)tilelang_kkt_solve_kernel_k_shape)[0]);
  if (!((bool)1)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "Symbolic shape variable num_tokens requires at least one non-null buffer among: tilelang_kkt_solve_kernel.k, tilelang_kkt_solve_kernel.b, tilelang_kkt_solve_kernel.a");
    return -1;
  }
  int32_t num_tokens = ((int32_t)((int64_t*)tilelang_kkt_solve_kernel_k_shape)[1]);
  void* tilelang_kkt_solve_kernel_k_strides = (((DLTensor*)k_handle)[0].strides);
  int32_t dev_id = (((DLTensor*)k_handle)[0].device.device_id);
  void* k = (((DLTensor*)k_handle)[0].data);
  if (!(((((DLTensor*)b_handle)[0].ndim) == 3))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_kkt_solve_kernel input b ndim mismatch, expected 3", (long long)((((DLTensor*)b_handle)[0].ndim)), (long long)(3));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  void* tilelang_kkt_solve_kernel_b_strides = (((DLTensor*)b_handle)[0].strides);
  void* b = (((DLTensor*)b_handle)[0].data);
  if (!(((((DLTensor*)a_handle)[0].ndim) == 4))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_kkt_solve_kernel input a ndim mismatch, expected 4", (long long)((((DLTensor*)a_handle)[0].ndim)), (long long)(4));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  void* tilelang_kkt_solve_kernel_a_strides = (((DLTensor*)a_handle)[0].strides);
  void* a = (((DLTensor*)a_handle)[0].data);
  if (!(((((((DLTensor*)k_handle)[0].dtype.code) == (uint8_t)4) && ((((DLTensor*)k_handle)[0].dtype.bits) == (uint8_t)16)) && ((((DLTensor*)k_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_kkt_solve_kernel input k dtype mismatch, expected bfloat16");
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_kkt_solve_kernel_k_shape)[2]) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_kkt_solve_kernel input k shape[2] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_kkt_solve_kernel_k_shape)[2])), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_kkt_solve_kernel_k_shape)[3]) == 128))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_kkt_solve_kernel input k shape[3] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_kkt_solve_kernel_k_shape)[3])), (long long)(128));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval;
  if ((tilelang_kkt_solve_kernel_k_strides == NULL)) {
    condval = 1;
  } else {
    condval = ((int32_t)((int64_t*)tilelang_kkt_solve_kernel_k_strides)[3]);
  }
  if (!((condval == 1))) {
    int32_t condval_1;
    if ((tilelang_kkt_solve_kernel_k_strides == NULL)) {
      condval_1 = 1;
    } else {
      condval_1 = ((int32_t)((int64_t*)tilelang_kkt_solve_kernel_k_strides)[3]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_kkt_solve_kernel input k strides[3] violates packed ABI constraint", (long long)(condval_1), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_2;
  if ((tilelang_kkt_solve_kernel_k_strides == NULL)) {
    condval_2 = 1;
  } else {
    condval_2 = ((int32_t)((int64_t*)tilelang_kkt_solve_kernel_k_strides)[2]);
  }
  if (!((condval_2 == 128))) {
    int32_t condval_3;
    if ((tilelang_kkt_solve_kernel_k_strides == NULL)) {
      condval_3 = 1;
    } else {
      condval_3 = ((int32_t)((int64_t*)tilelang_kkt_solve_kernel_k_strides)[2]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_kkt_solve_kernel input k strides[2] violates packed ABI constraint", (long long)(condval_3), (long long)(128));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_4;
  if ((tilelang_kkt_solve_kernel_k_strides == NULL)) {
    condval_4 = 1;
  } else {
    condval_4 = ((int32_t)((int64_t*)tilelang_kkt_solve_kernel_k_strides)[1]);
  }
  if (!(((condval_4 == 256) || (num_tokens == 1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_kkt_solve_kernel input k strides[1] violates packed ABI constraint");
    return -1;
  }
  int32_t condval_5;
  if ((tilelang_kkt_solve_kernel_k_strides == NULL)) {
    condval_5 = 1;
  } else {
    condval_5 = ((int32_t)((int64_t*)tilelang_kkt_solve_kernel_k_strides)[0]);
  }
  if (!(((((num_tokens * 256) == condval_5) || (num_tokens == 0)) || (data_batch_size == 1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_kkt_solve_kernel input k strides[0] violates packed ABI constraint");
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)k_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_kkt_solve_kernel input k byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)k_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)k_handle)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_kkt_solve_kernel input k device_type mismatch, expected cuda", (long long)((((DLTensor*)k_handle)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((data_batch_size == 0) || (num_tokens == 0)) || !(k == NULL)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_kkt_solve_kernel input k data pointer is NULL");
    return -1;
  }
  if (!(((((((DLTensor*)b_handle)[0].dtype.code) == (uint8_t)2) && ((((DLTensor*)b_handle)[0].dtype.bits) == (uint8_t)32)) && ((((DLTensor*)b_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_kkt_solve_kernel input b dtype mismatch, expected float32");
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_kkt_solve_kernel_k_shape)[0]) == ((int32_t)((int64_t*)tilelang_kkt_solve_kernel_b_shape)[0])))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_kkt_solve_kernel input b shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_kkt_solve_kernel_k_shape)[0])), (long long)(((int32_t)((int64_t*)tilelang_kkt_solve_kernel_b_shape)[0])));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_kkt_solve_kernel_k_shape)[1]) == ((int32_t)((int64_t*)tilelang_kkt_solve_kernel_b_shape)[1])))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_kkt_solve_kernel input b shape[1] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_kkt_solve_kernel_k_shape)[1])), (long long)(((int32_t)((int64_t*)tilelang_kkt_solve_kernel_b_shape)[1])));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_kkt_solve_kernel_b_shape)[2]) == 4))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_kkt_solve_kernel input b shape[2] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_kkt_solve_kernel_b_shape)[2])), (long long)(4));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_6;
  if ((tilelang_kkt_solve_kernel_b_strides == NULL)) {
    condval_6 = 1;
  } else {
    condval_6 = ((int32_t)((int64_t*)tilelang_kkt_solve_kernel_b_strides)[2]);
  }
  if (!((condval_6 == 1))) {
    int32_t condval_7;
    if ((tilelang_kkt_solve_kernel_b_strides == NULL)) {
      condval_7 = 1;
    } else {
      condval_7 = ((int32_t)((int64_t*)tilelang_kkt_solve_kernel_b_strides)[2]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_kkt_solve_kernel input b strides[2] violates packed ABI constraint", (long long)(condval_7), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_8;
  if ((tilelang_kkt_solve_kernel_b_strides == NULL)) {
    condval_8 = 1;
  } else {
    condval_8 = ((int32_t)((int64_t*)tilelang_kkt_solve_kernel_b_strides)[1]);
  }
  if (!(((condval_8 == 4) || (num_tokens == 1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_kkt_solve_kernel input b strides[1] violates packed ABI constraint");
    return -1;
  }
  int32_t condval_9;
  if ((tilelang_kkt_solve_kernel_b_strides == NULL)) {
    condval_9 = 1;
  } else {
    condval_9 = ((int32_t)((int64_t*)tilelang_kkt_solve_kernel_b_strides)[0]);
  }
  if (!(((((num_tokens * 4) == condval_9) || (num_tokens == 0)) || (data_batch_size == 1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_kkt_solve_kernel input b strides[0] violates packed ABI constraint");
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)b_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_kkt_solve_kernel input b byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)b_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)b_handle)[0].device.device_id) == (((DLTensor*)k_handle)[0].device.device_id)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_kkt_solve_kernel input b device_id violates packed ABI constraint", (long long)((((DLTensor*)b_handle)[0].device.device_id)), (long long)((((DLTensor*)k_handle)[0].device.device_id)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)b_handle)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_kkt_solve_kernel input b device_type mismatch, expected cuda", (long long)((((DLTensor*)b_handle)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((data_batch_size == 0) || (num_tokens == 0)) || !(b == NULL)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_kkt_solve_kernel input b data pointer is NULL");
    return -1;
  }
  if (!(((((((DLTensor*)a_handle)[0].dtype.code) == (uint8_t)4) && ((((DLTensor*)a_handle)[0].dtype.bits) == (uint8_t)16)) && ((((DLTensor*)a_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_kkt_solve_kernel input a dtype mismatch, expected bfloat16");
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_kkt_solve_kernel_k_shape)[0]) == ((int32_t)((int64_t*)tilelang_kkt_solve_kernel_a_shape)[0])))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_kkt_solve_kernel input a shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_kkt_solve_kernel_k_shape)[0])), (long long)(((int32_t)((int64_t*)tilelang_kkt_solve_kernel_a_shape)[0])));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_kkt_solve_kernel_k_shape)[1]) == ((int32_t)((int64_t*)tilelang_kkt_solve_kernel_a_shape)[1])))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_kkt_solve_kernel input a shape[1] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_kkt_solve_kernel_k_shape)[1])), (long long)(((int32_t)((int64_t*)tilelang_kkt_solve_kernel_a_shape)[1])));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_kkt_solve_kernel_a_shape)[2]) == 4))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_kkt_solve_kernel input a shape[2] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_kkt_solve_kernel_a_shape)[2])), (long long)(4));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)tilelang_kkt_solve_kernel_a_shape)[3]) == 32))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_kkt_solve_kernel input a shape[3] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)tilelang_kkt_solve_kernel_a_shape)[3])), (long long)(32));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_10;
  if ((tilelang_kkt_solve_kernel_a_strides == NULL)) {
    condval_10 = 1;
  } else {
    condval_10 = ((int32_t)((int64_t*)tilelang_kkt_solve_kernel_a_strides)[3]);
  }
  if (!((condval_10 == 1))) {
    int32_t condval_11;
    if ((tilelang_kkt_solve_kernel_a_strides == NULL)) {
      condval_11 = 1;
    } else {
      condval_11 = ((int32_t)((int64_t*)tilelang_kkt_solve_kernel_a_strides)[3]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_kkt_solve_kernel input a strides[3] violates packed ABI constraint", (long long)(condval_11), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_12;
  if ((tilelang_kkt_solve_kernel_a_strides == NULL)) {
    condval_12 = 1;
  } else {
    condval_12 = ((int32_t)((int64_t*)tilelang_kkt_solve_kernel_a_strides)[2]);
  }
  if (!((condval_12 == 32))) {
    int32_t condval_13;
    if ((tilelang_kkt_solve_kernel_a_strides == NULL)) {
      condval_13 = 1;
    } else {
      condval_13 = ((int32_t)((int64_t*)tilelang_kkt_solve_kernel_a_strides)[2]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_kkt_solve_kernel input a strides[2] violates packed ABI constraint", (long long)(condval_13), (long long)(32));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_14;
  if ((tilelang_kkt_solve_kernel_a_strides == NULL)) {
    condval_14 = 1;
  } else {
    condval_14 = ((int32_t)((int64_t*)tilelang_kkt_solve_kernel_a_strides)[1]);
  }
  if (!(((condval_14 == 128) || (num_tokens == 1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_kkt_solve_kernel input a strides[1] violates packed ABI constraint");
    return -1;
  }
  int32_t condval_15;
  if ((tilelang_kkt_solve_kernel_a_strides == NULL)) {
    condval_15 = 1;
  } else {
    condval_15 = ((int32_t)((int64_t*)tilelang_kkt_solve_kernel_a_strides)[0]);
  }
  if (!(((((num_tokens * 128) == condval_15) || (num_tokens == 0)) || (data_batch_size == 1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_kkt_solve_kernel input a strides[0] violates packed ABI constraint");
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)a_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_kkt_solve_kernel input a byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)a_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)a_handle)[0].device.device_id) == (((DLTensor*)k_handle)[0].device.device_id)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_kkt_solve_kernel input a device_id violates packed ABI constraint", (long long)((((DLTensor*)a_handle)[0].device.device_id)), (long long)((((DLTensor*)k_handle)[0].device.device_id)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)a_handle)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel tilelang_kkt_solve_kernel input a device_type mismatch, expected cuda", (long long)((((DLTensor*)a_handle)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((data_batch_size == 0) || (num_tokens == 0)) || !(a == NULL)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tilelang_kkt_solve_kernel input a data pointer is NULL");
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
  if (a == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = a;
  if (b == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = b;
  if (k == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = k;
  (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)data_batch_size);
  (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)num_chunks);
  (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = ((int64_t)num_tokens);
  (((TVMFFIAny*)stack_ffi_any)[6].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[6].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[6].v_int64) = ((int64_t)(num_chunks * 4));
  (((TVMFFIAny*)stack_ffi_any)[7].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[7].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[7].v_int64) = ((int64_t)128);
  (((TVMFFIAny*)stack_ffi_any)[8].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[8].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[8].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[9].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[9].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[9].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[10].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[10].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[10].v_int64) = ((int64_t)8192);
  (((TVMFFIAny*)stack_ffi_any)[11].type_index) = 0;
  (((TVMFFIAny*)stack_ffi_any)[11].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[11].v_int64) = (int64_t)0;
  if (tilelang_kkt_solve_kernel_kernel_packed == NULL) {
    if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "tilelang_kkt_solve_kernel_kernel", &tilelang_kkt_solve_kernel_kernel_packed) != 0) {
      return -1;
    }
  }
  TVMFFIAny result_2;
  result_2.type_index = kTVMFFINone;
  result_2.zero_padding = 0;
  result_2.v_int64 = 0;
  if (TVMFFIFunctionCall(tilelang_kkt_solve_kernel_kernel_packed, (TVMFFIAny*) stack_ffi_any, 11, &result_2) != 0) {
    return -1;
  }
  return 0;
}

// CodegenC: NOTE: Auto-generated entry function
#ifdef __cplusplus
extern "C"
#endif
int32_t __tvm_ffi_main(void* self, void* args,int num_args, void* result) {
  return __tvm_ffi_tilelang_kkt_solve_kernel(self, args, num_args, result);
}
