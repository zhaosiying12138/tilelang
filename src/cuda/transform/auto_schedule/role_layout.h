/*!
 * \file role_layout.h
 * \brief Shared CUDA warp-role layout construction.
 */
#ifndef TVM_TL_CUDA_TRANSFORM_AUTO_SCHEDULE_ROLE_LAYOUT_H_
#define TVM_TL_CUDA_TRANSFORM_AUTO_SCHEDULE_ROLE_LAYOUT_H_

#include <cstdint>
#include <vector>

namespace tvm {
namespace tl {
namespace cuda {

// Fixed role indices are shared with the recursive CudaRoleTable convention.
enum class CudaScheduleRole : uint8_t { kWorker = 0, kLoad = 1, kMma = 2, kStore = 3 };

constexpr int kCudaAuxiliaryRoleNumRegisters = 24;
constexpr int kCudaCpAsyncLoadNumRegisters = 48;

struct CudaRoleLayout {
  CudaScheduleRole role;
  int warp_lo;
  int warp_hi;
  int max_nreg;
};

// Construct warp ranges and auxiliary register requests in canonical role
// order. Bits 0..3 of active_roles name CudaScheduleRole values. Occupancy-
// dependent Worker budgets remain a scheduler policy applied by the caller.
std::vector<CudaRoleLayout> PlanCudaRoleLayout(uint64_t active_roles, int worker_warps,
                                               bool widen_cpasync_load);

const char* CudaScheduleRoleName(CudaScheduleRole role);

}  // namespace cuda
}  // namespace tl
}  // namespace tvm

#endif  // TVM_TL_CUDA_TRANSFORM_AUTO_SCHEDULE_ROLE_LAYOUT_H_
