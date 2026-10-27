/*!
 * \file role_layout.cc
 * \brief Shared CUDA warp-role layout construction.
 */

#include "./role_layout.h"

#include <tvm/runtime/logging.h>

#include <array>

#include "support/check.h"

namespace tvm {
namespace tl {
namespace cuda {

const char* CudaScheduleRoleName(CudaScheduleRole role) {
  switch (role) {
    case CudaScheduleRole::kWorker:
      return "Worker";
    case CudaScheduleRole::kLoad:
      return "Load";
    case CudaScheduleRole::kMma:
      return "MMA";
    case CudaScheduleRole::kStore:
      return "Store";
  }
  LOG(FATAL) << "Invalid CUDA schedule role";
  return "";
}

std::vector<CudaRoleLayout> PlanCudaRoleLayout(uint64_t active_roles, int worker_warps,
                                               bool widen_cpasync_load) {
  constexpr uint64_t kKnownRoles = 0xf;
  ICHECK_EQ(active_roles & ~kKnownRoles, 0)
      << "CUDA role layout received unknown active-role bits";
  ICHECK(active_roles != 0) << "CUDA role layout requires at least one active role";
  constexpr std::array<CudaScheduleRole, 4> kRoles = {
      CudaScheduleRole::kWorker, CudaScheduleRole::kLoad, CudaScheduleRole::kMma,
      CudaScheduleRole::kStore};
  auto active = [&](CudaScheduleRole role) {
    return (active_roles & (uint64_t{1} << static_cast<unsigned>(role))) != 0;
  };
  if (active(CudaScheduleRole::kWorker)) {
    ICHECK_GT(worker_warps, 0) << "Worker warp count must be positive";
    ICHECK_EQ(worker_warps % 4, 0)
        << "Worker warps must form complete warpgroups, got " << worker_warps;
  }

  std::vector<CudaRoleLayout> result;
  int cursor = active(CudaScheduleRole::kWorker) ? worker_warps : 0;
  for (CudaScheduleRole role : kRoles) {
    if (!active(role)) continue;
    if (role == CudaScheduleRole::kWorker) {
      result.push_back({role, 0, worker_warps, 0});
      continue;
    }
    bool wide = role == CudaScheduleRole::kLoad && widen_cpasync_load;
    int width = wide ? 4 : 1;
    result.push_back({role, cursor, cursor + width,
                      wide ? kCudaCpAsyncLoadNumRegisters : kCudaAuxiliaryRoleNumRegisters});
    cursor += width;
  }
  return result;
}

}  // namespace cuda
}  // namespace tl
}  // namespace tvm
