/*!
 * \file z3_buffer_versions.h
 * \brief Shared metadata contracts for CUDA Z3 physical buffer versions.
 */
#ifndef TVM_TL_TRANSFORM_COMMON_Z3_BUFFER_VERSIONS_H_
#define TVM_TL_TRANSFORM_COMMON_Z3_BUFFER_VERSIONS_H_

#include <tvm/ffi/any.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/string.h>
#include <tvm/ir/cast.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/stmt.h>

namespace tvm {
namespace tl {

using namespace tirx;

// Public, root-scoped request and selected-result maps. Loop-local operational
// copies exist only between normalization and selected-count projection.
static constexpr const char *kZ3BufferVersionRequests =
    "tl.cuda_z3_buffer_version_requests";
static constexpr const char *kZ3BufferVersionModeRequests =
    "tl.cuda_z3_buffer_version_mode_requests";
static constexpr const char *kZ3BufferVersionsSelected =
    "tl.cuda_z3_buffer_versions_selected";

// Read-only provenance. "selected" is the model result; "required" is the
// post-schedule recomputation. The public selected map above is the final count.
static constexpr const char *kZ3AuditVersionRequest =
    "tl_z3_audit_version_request";
static constexpr const char *kZ3AuditVersionModeRequest =
    "tl_z3_audit_version_mode_request";
static constexpr const char *kZ3AuditVersionSelected =
    "tl_z3_audit_version_selected";
static constexpr const char *kZ3AuditVersionRequired =
    "tl_z3_audit_version_required";

using Z3BufferVersionCountMap = ffi::Map<Var, IntImm>;
using Z3BufferVersionModeMap = ffi::Map<Var, ffi::String>;

inline Z3BufferVersionCountMap
ReadZ3BufferVersionCounts(const ffi::Map<ffi::String, ffi::Any> &annotations,
                          const char *key, const char *pass_name) {
  Z3BufferVersionCountMap result;
  auto annotation = annotations.Get(key);
  if (!annotation.has_value()) return result;
  auto raw = annotation.value().as<ffi::Map<ffi::Any, ffi::Any>>();
  ICHECK(raw.has_value())
      << pass_name << ": " << key << " must be Map<Var, int64>";
  for (const auto &[entry_key, value] : raw.value()) {
    auto var = entry_key.as<Var>();
    ICHECK(var.has_value())
        << pass_name << ": buffer-version map has a non-Var key";
    auto count = value.as<IntImm>();
    ICHECK(count.has_value())
        << pass_name << ": buffer version for " << var.value()
        << " must be an IntImm";
    ICHECK_GT(count.value()->value, 0)
        << pass_name << ": buffer version for " << var.value()
        << " must be positive";
    result.Set(var.value(), count.value());
  }
  return result;
}

inline Z3BufferVersionModeMap
ReadZ3BufferVersionModes(const ffi::Map<ffi::String, ffi::Any> &annotations,
                         const char *key, const char *pass_name) {
  Z3BufferVersionModeMap result;
  auto annotation = annotations.Get(key);
  if (!annotation.has_value()) return result;
  auto raw = annotation.value().as<ffi::Map<ffi::Any, ffi::Any>>();
  ICHECK(raw.has_value())
      << pass_name << ": " << key << " must be Map<Var, String>";
  for (const auto &[entry_key, value] : raw.value()) {
    auto var = entry_key.as<Var>();
    auto mode = value.as<ffi::String>();
    ICHECK(var.has_value() && mode.has_value())
        << pass_name << ": " << key << " must be Map<Var, String>";
    ICHECK(mode.value() == "auto" || mode.value() == "iteration" ||
           mode.value() == "counter")
        << pass_name << ": unknown buffer version mode '" << mode.value()
        << "' for " << var.value();
    result.Set(var.value(), mode.value());
  }
  return result;
}

inline bool IsZ3BareOneRequest(const Var &storage,
                               const Z3BufferVersionCountMap &requests,
                               const Z3BufferVersionModeMap &modes) {
  auto count = requests.Get(storage);
  return count.has_value() && count.value()->value == 1 &&
         !modes.count(storage);
}

} // namespace tl
} // namespace tvm

#endif // TVM_TL_TRANSFORM_COMMON_Z3_BUFFER_VERSIONS_H_
