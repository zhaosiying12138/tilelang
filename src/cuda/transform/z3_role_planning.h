/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file z3_role_planning.h
 * \brief CUDA role capability, input availability, and execution-demand fixed points.
 */

#ifndef TVM_TL_CUDA_TRANSFORM_Z3_ROLE_PLANNING_H_
#define TVM_TL_CUDA_TRANSFORM_Z3_ROLE_PLANNING_H_

#include <tvm/target/target.h>

#include <unordered_map>
#include <vector>

#include "transform/common/warp_specialize.h"
#include "z3_dependency_analysis.h"

namespace tvm::tl::cuda::z3 {

struct CudaRoleTable {
  ffi::Array<WSRole> roles;
  // Zero is the convenient default meaning "all entries in roles". Otherwise
  // this must be a nonempty subset of the bits naming entries in roles.
  RoleMask active_mask{0};
};

// Section 6 plan values are declared here until the ownership implementation
// lands; z3_buffer_ownership.h must reuse these types rather than redefine them.
struct MultiBufferOwnerInfo {
  ControlNode* loop{nullptr};
  PrimExpr active_guard{Bool(true)};
};

enum class ClockKind { kIteration, kCounter };

class MultiBufferInfo {
 public:
  tirx::Var storage;
  int num_versions{1};
  ClockKind clock{ClockKind::kIteration};
  int counter_group_id{0};
  std::vector<MultiBufferOwnerInfo> owners;
  tirx::Buffer counter;
  std::vector<int> owner_stages;

  bool UsesCounter() const { return clock == ClockKind::kCounter; }
  // Null outside every owner; nested ownership for this storage is an error.
  const MultiBufferOwnerInfo* FindOwner(const IRStructure& node) const;
};

class MultiBufferPlan {
 public:
  const std::vector<MultiBufferInfo>& Infos() const { return infos_; }
  const MultiBufferInfo* Find(const tirx::Var& storage) const;
  const MultiBufferInfo* Find(const tirx::Buffer& buffer) const;
  void AddInfo(MultiBufferInfo info);

 private:
  std::vector<MultiBufferInfo> infos_;
};

struct DemandMaps {
  std::unordered_map<tirx::Var, RoleMask, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> scalar;
  std::unordered_map<tirx::Var, RoleMask, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> registers;
  std::unordered_map<const TaskNode*, RoleMask> special_writers;
  std::unordered_map<const TaskNode*, RoleMask> execution;
};

// RoleAvailability answers where values can be evaluated, not where shared
// data happens to reside. Its invariants are:
//  * an externally defined/unwritten value is available on every known active role;
//  * every non-self reaching definition intersects the available role mask;
//  * local/fragment/register buffers are role-private, while shared/TMEM/global
//    payloads remain visible and therefore do not themselves constrain roles;
//  * region indices, scalar control, task guards, and ancestor headers are inputs;
//  * a zero mask is a hard placement error, never permission to pick arbitrarily.
class RoleAvailability {
 public:
  explicit RoleAvailability(const std::vector<TaskNode*>& tasks);

  RoleMask ValueRoles(const tirx::Var& value,
                      const TaskNode* reader = nullptr) const;
  RoleMask ExpressionRoles(const PrimExpr& expr,
                           const TaskNode* reader = nullptr) const;
  RoleMask AccessRoles(const TaskAccessInfo& accesses,
                       const TaskNode* reader = nullptr) const;
  RoleMask TaskInputRoles(const TaskNode* task) const;
  RoleMask AllRoles() const { return all_roles_; }

 private:
  using WriterMap =
      std::unordered_map<tirx::Var, std::vector<const TaskNode*>, ffi::ObjectPtrHash,
                         ffi::ObjectPtrEqual>;

  WriterMap writers_;
  RoleMask all_roles_{0};
};

bool IsRolePrivateBuffer(const tirx::Buffer& buffer);

// Assign widest legal candidates and iterate input availability to a fixed point.
// PerRole groups resolve to exactly one role during ResolveRoleDemands.
void AssignRoleCandidates(ScheduledTIR* scheduled, const CudaRoleTable& role_table,
                          const Target& target);

// Propagate scalar/private/special/header/counter/loop-exit demand monotonically,
// choose deterministic pure-task fallbacks, rerun the fixed point, and publish
// nonempty ResolvedRoles subsets of CandidateRoles on every task/header.
void ResolveRoleDemands(ScheduledTIR* scheduled, const MultiBufferPlan& plan,
                        const CudaRoleTable& role_table);

}  // namespace tvm::tl::cuda::z3

#endif  // TVM_TL_CUDA_TRANSFORM_Z3_ROLE_PLANNING_H_
