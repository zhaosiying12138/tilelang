/*!
 * \file z3_schedule_hints.h
 * \brief Shared CUDA Z3 schedule-unit marker keys and atomic-region queries.
 */
#ifndef TVM_TL_TRANSFORM_COMMON_Z3_SCHEDULE_HINTS_H_
#define TVM_TL_TRANSFORM_COMMON_Z3_SCHEDULE_HINTS_H_

#include <tvm/tirx/stmt.h>

namespace tvm {
namespace tl {

using namespace tirx;

static constexpr const char *kZ3CudaTaskAttr = "tl.cuda_z3_task";
static constexpr const char *kZ3CudaStageAttr = "tl.cuda_z3_stage";
static constexpr const char *kZ3CudaWarpRegionAttr =
    "tl.cuda_z3_warp_region";
static constexpr const char *kZ3ExecutionDomains =
    "tl_z3_pipeline_execution_domains";

/*! \brief Whether an attribute is a public schedule-unit marker. */
inline bool IsZ3ScheduleUnitAttrKey(const ffi::String &key) {
  return key == kZ3CudaTaskAttr || key == kZ3CudaStageAttr;
}

/*!
 * \brief Whether pre-schedule explicit unrolling must not enter an attribute.
 *
 * The two existing WS keys are included so owner-ring and handwritten regions
 * retain their established atomic behavior. The warp-region key is registered
 * here ahead of its public P07 implementation to keep the unroller forward
 * compatible without changing its traversal rules again.
 */
inline bool IsZ3AtomicAttrKey(const ffi::String &key) {
  return key == kZ3CudaTaskAttr || key == kZ3CudaWarpRegionAttr ||
         key == "warp_specialize" || key == "kWarpSpecializationScope";
}

inline bool IsZ3TaskMarker(const Stmt &stmt) {
  const auto *attr = stmt.as<AttrStmtNode>();
  return attr != nullptr && attr->attr_key == kZ3CudaTaskAttr;
}

}  // namespace tl
}  // namespace tvm

#endif  // TVM_TL_TRANSFORM_COMMON_Z3_SCHEDULE_HINTS_H_
