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
 * \file z3_schedule_ir.h
 * \brief Internal nested Task/Control IR for recursive CUDA Z3 scheduling.
 */

#ifndef TVM_TL_CUDA_TRANSFORM_Z3_SCHEDULE_IR_H_
#define TVM_TL_CUDA_TRANSFORM_Z3_SCHEDULE_IR_H_

#include <tvm/ffi/any.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/string.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/buffer.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/stmt.h>

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <utility>
#include <vector>

#include "transform/common/constr_visitor.h"

namespace tvm::tl::cuda::z3 {

using RoleId = uint8_t;           // Index into the kernel's WSRole array.
using RoleMask = uint64_t;        // Bit i is WSRole index i.
using SpecialStateMask = uint64_t;
using BufferVersionMap = ffi::Map<tirx::Var, int>;
using BufferAliasMap = ffi::Map<tirx::Var, ffi::Array<tirx::Var>>;

enum class SpecialState : uint64_t {
  kNone = 0,
  kLoopControl = 1ULL << 0,
};

enum class TaskKind { kOperation, kDeclaration, kPerRoleGroup };

struct TaskAccessInfo {
  ffi::Array<tirx::BufferRegion> reads;
  ffi::Array<tirx::BufferRegion> writes;
  std::vector<tirx::Var> read_vars;
  std::vector<tirx::Var> write_vars;
  SpecialStateMask state_reads{0};
  SpecialStateMask state_writes{0};
};

constexpr int kUnscheduledStage = -1;
constexpr const char* kScheduleUnit = "tl.schedule_unit";
constexpr const char* kCudaTask = "tl.cuda_task";
constexpr const char* kCudaPerRoleTask = "tl.cuda_per_role_task";
constexpr const char* kCudaStage = "tl.cuda_stage";
constexpr const char* kWSOpId = "tl.ws_op_id";
constexpr const char* kRequestedBufferVersions = "tl.buffer_versions_map";
constexpr const char* kSelectedBufferVersions = "tl.cuda_z3_selected_buffer_versions";
constexpr const char* kBufferVersionModes = "tl.buffer_version_mode";
constexpr const char* kBufferAliases = "tl.buffer_alias_map";

// A guard delta relative to the nearest surviving structural ancestor.
class Guard {
 public:
  enum class Kind { kCondition, kAttribute };

  virtual ~Guard() = default;
  virtual Kind GetKind() const = 0;
  virtual std::unique_ptr<Guard> Clone() const = 0;
  virtual void SubstituteVar(const tirx::Var& old_var, const tirx::Var& new_var) = 0;
  virtual tirx::Stmt Wrap(tirx::Stmt body) const = 0;

  bool IsCondition() const { return GetKind() == Kind::kCondition; }
  bool IsAttribute() const { return GetKind() == Kind::kAttribute; }
};

class ConditionGuard final : public Guard {
 public:
  explicit ConditionGuard(PrimExpr condition) : condition(std::move(condition)) {}

  Kind GetKind() const final { return Kind::kCondition; }
  std::unique_ptr<Guard> Clone() const final;
  void SubstituteVar(const tirx::Var& old_var, const tirx::Var& new_var) final;
  tirx::Stmt Wrap(tirx::Stmt body) const final;

  PrimExpr condition;
};

class AttributeGuard final : public Guard {
 public:
  AttributeGuard(ffi::Any node, ffi::String key, PrimExpr value, Span span)
      : node(std::move(node)), key(std::move(key)), value(std::move(value)), span(span) {}

  Kind GetKind() const final { return Kind::kAttribute; }
  std::unique_ptr<Guard> Clone() const final;
  void SubstituteVar(const tirx::Var& old_var, const tirx::Var& new_var) final;
  tirx::Stmt Wrap(tirx::Stmt body) const final;

  ffi::Any node;
  ffi::String key;
  PrimExpr value;
  Span span;
};

using GuardList = std::vector<std::unique_ptr<Guard>>;

bool IsScheduleGuardAttribute(const ffi::String& key);
tirx::Stmt WrapScheduledGuards(tirx::Stmt body, const GuardList& guards);

// Pass-local state is cloned and substituted with its owning node. The shared
// IR does not interpret backend analysis or epoch data stored in this interface.
class IRExtraInfo {
 public:
  virtual ~IRExtraInfo() = default;
  virtual std::unique_ptr<IRExtraInfo> Clone() const = 0;
  virtual void SubstituteVar(const tirx::Var&, const tirx::Var&) {}
};

class IRStructure {
 public:
  enum class Kind { kTask, kControl };

  explicit IRStructure(std::unique_ptr<IRExtraInfo> extra_info = nullptr)
      : extra_info_(std::move(extra_info)) {}
  virtual ~IRStructure() = default;

  virtual Kind GetKind() const = 0;
  virtual std::shared_ptr<IRStructure> Clone() const = 0;
  virtual void SubstituteVar(const tirx::Var& old_var, const tirx::Var& new_var) = 0;

  IRStructure* GetParent() { return parent_; }
  const IRStructure* GetParent() const { return parent_; }
  void SetParent(IRStructure* parent);
  bool IsWithin(const IRStructure* ancestor) const;
  IRStructure* GetLowestCommonAncestor(IRStructure* other);
  const IRStructure* GetLowestCommonAncestor(const IRStructure* other) const;
  std::vector<IRStructure*> PathFrom(IRStructure* ancestor = nullptr);
  std::vector<const IRStructure*> PathFrom(const IRStructure* ancestor = nullptr) const;
  IRStructure* GetChildOnPathFrom(IRStructure* strict_ancestor);

  const GuardList& GetGuards() const { return guards_; }
  PrimExpr GetConditionGuard() const;
  bool GuardsTouchStorage(const tirx::Var& storage) const;
  void SetGuards(GuardList guards) { guards_ = std::move(guards); }
  bool HasGuards() const { return !guards_.empty(); }
  void PrependCondition(const PrimExpr& condition);

  int GetStage() const { return stage_; }
  size_t GetIndex() const { return index_; }
  int64_t GetLatency() const { return latency_; }
  int64_t GetII() const { return ii_; }
  int64_t GetSourceId() const { return source_id_; }
  void SetStage(int stage) { stage_ = stage; }
  void SetIndex(size_t index) { index_ = index; }
  void SetSourceId(int64_t source_id) { source_id_ = source_id; }
  void SetSchedule(int stage, size_t index);
  void SetCost(int64_t latency, int64_t ii);

  virtual const TaskAccessInfo& Accesses() const = 0;
  virtual RoleMask CandidateRoles() const = 0;
  virtual RoleMask ResolvedRoles() const = 0;

  bool IsTask() const { return GetKind() == Kind::kTask; }
  bool IsControl() const { return GetKind() == Kind::kControl; }
  bool ReadsStorage(const tirx::Var& storage) const;
  bool WritesStorage(const tirx::Var& storage) const;
  bool TouchesStorage(const tirx::Var& storage) const {
    return ReadsStorage(storage) || WritesStorage(storage);
  }

  template <typename T>
  T* GetExtraInfo() {
    ICHECK(extra_info_ != nullptr) << "Expected pass-local CUDA Z3 IR extra info";
    return static_cast<T*>(extra_info_.get());
  }
  template <typename T>
  const T* GetExtraInfo() const {
    ICHECK(extra_info_ != nullptr) << "Expected pass-local CUDA Z3 IR extra info";
    return static_cast<const T*>(extra_info_.get());
  }

 protected:
  std::unique_ptr<IRExtraInfo> CloneExtraInfo() const {
    return extra_info_ ? extra_info_->Clone() : nullptr;
  }
  GuardList& MutableGuards() { return guards_; }
  void SubstituteVarInGuards(const tirx::Var& old_var, const tirx::Var& new_var);
  void SubstituteVarInExtraInfo(const tirx::Var& old_var, const tirx::Var& new_var) {
    if (extra_info_) {
      extra_info_->SubstituteVar(old_var, new_var);
    }
  }
  void CopyBaseMetadataFrom(const IRStructure& other);

 private:
  IRStructure* parent_{nullptr};  // Non-owning.
  std::unique_ptr<IRExtraInfo> extra_info_;
  GuardList guards_;
  int stage_{kUnscheduledStage};
  size_t index_{0};
  int64_t latency_{0};
  int64_t ii_{0};
  int64_t source_id_{-1};  // Immutable source/solver identity, not sibling position.
};

class TaskNode final : public IRStructure {
 public:
  explicit TaskNode(std::unique_ptr<IRExtraInfo> extra_info = nullptr)
      : IRStructure(std::move(extra_info)) {}

  tirx::Stmt stmt;  // Includes the semantic task marker when relevant.
  ConstrSet outer_ctx;
  TaskKind task_kind{TaskKind::kOperation};
  bool solver_eligible{true};
  ffi::String ws_op_id;
  TaskAccessInfo accesses;

  Kind GetKind() const final { return Kind::kTask; }
  std::shared_ptr<IRStructure> Clone() const final;
  void SubstituteVar(const tirx::Var& old_var, const tirx::Var& new_var) final;

  void SetCandidateRoles(RoleMask roles);
  void SetResolvedRoles(RoleMask roles);
  const ConstrSet& GetContextBeforeConditionGuards() const;

  const TaskAccessInfo& Accesses() const final { return accesses; }
  RoleMask CandidateRoles() const final { return candidate_roles_; }
  RoleMask ResolvedRoles() const final { return resolved_roles_; }

 private:
  RoleMask candidate_roles_{0};
  RoleMask resolved_roles_{0};
  mutable ConstrSet context_before_condition_guards_;
};

class ControlNode final : public IRStructure {
 public:
  explicit ControlNode(std::unique_ptr<IRExtraInfo> extra_info = nullptr);

  tirx::For control;  // Serial or unrolled only.
  std::shared_ptr<TaskNode> task;  // Synthetic min/extent/step + guard accesses.
  std::vector<std::shared_ptr<IRStructure>> children;

  Kind GetKind() const final { return Kind::kControl; }
  std::shared_ptr<IRStructure> Clone() const final;
  void SubstituteVar(const tirx::Var& old_var, const tirx::Var& new_var) final;

  void InitializeLoopBodyContext();
  const ConstrSet& GetLoopBodyContext() const { return loop_body_ctx_; }
  ControlNode* GetParentControl();
  const ControlNode* GetParentControl() const;
  std::vector<const ControlNode*> GetAncestorControls() const;
  bool BodyHasLoopBreak() const;
  bool BodyTouchesStorage(const tirx::Var& storage) const;

  void SetIterationTiming(int64_t ii_per_iter, int64_t first_iteration_span,
                          int64_t total_latency);
  int64_t GetIIperIter() const { return ii_per_iter_; }
  int64_t GetFirstIterationSpan() const { return first_iteration_span_; }
  int64_t GetLastIterationOffset() const;

  const TaskAccessInfo& Accesses() const final;
  RoleMask CandidateRoles() const final;
  RoleMask ResolvedRoles() const final;

 private:
  ConstrSet loop_body_ctx_;
  int64_t ii_per_iter_{0};
  int64_t first_iteration_span_{0};
  // Rebuilt on every Accesses query. It is never an authoritative cache, so
  // child/header/guard substitution and reordering cannot leave stale data.
  mutable TaskAccessInfo access_view_;
};

struct ScheduledTIRMetadata {
  tirx::SBlock kernel_root;
  BufferVersionMap requested_versions;
  BufferVersionMap selected_versions;
  ffi::Map<tirx::Var, ffi::String> version_modes;
  BufferAliasMap buffer_aliases;
  bool has_buffer_aliases{false};
};

struct ScheduledTIR {
  std::vector<std::shared_ptr<IRStructure>> tree;
  ScheduledTIRMetadata metadata;
};

using ScheduledExtraInfoFactory = std::unique_ptr<IRExtraInfo> (*)();

ScheduledTIR DecodeScheduledTIR(const tirx::SBlock& root, const ConstrSet& outer_ctx,
                                ScheduledExtraInfoFactory make_extra_info = nullptr);
tirx::SBlock EncodeScheduledTIR(ScheduledTIR scheduled);
void ValidateScheduledTIR(const ScheduledTIR& scheduled);

// Repair non-owning parent links and current sibling indices after insertion or
// reordering. Passing a null parent marks every node in the list as a root.
void RepairParentsAndIndices(std::vector<std::shared_ptr<IRStructure>>& nodes,
                             IRStructure* parent = nullptr);

int CompareIRStructure(const IRStructure* lhs, const IRStructure* rhs);
PrimExpr CalculateIterationCount(const ControlNode* loop = nullptr);
void CollectAllTaskNodes(IRStructure* node, std::vector<TaskNode*>* tasks,
                         bool include_control_headers = false);
void CollectAllTaskNodes(const std::vector<std::shared_ptr<IRStructure>>& nodes,
                         std::vector<TaskNode*>* tasks, bool include_control_headers = false);

inline RoleMask RoleMaskBit(RoleId role) {
  ICHECK_LT(static_cast<unsigned int>(role), 64U);
  return uint64_t{1} << static_cast<unsigned int>(role);
}

inline RoleMask MakeRoleMask(std::initializer_list<RoleId> roles) {
  RoleMask result = 0;
  for (RoleId role : roles) {
    result |= RoleMaskBit(role);
  }
  return result;
}

inline void ValidateRoleTableSize(size_t num_roles) {
  ICHECK_LE(num_roles, 64U) << "CUDA Z3 role masks support at most 64 WSRole entries";
}

// Masks cross annotation/FFI integer boundaries as two unsigned 32-bit words;
// bit 63 is therefore never routed through a signed 64-bit annotation value.
inline std::pair<uint32_t, uint32_t> RoleMaskToWords(RoleMask mask) {
  return {static_cast<uint32_t>(mask), static_cast<uint32_t>(mask >> 32)};
}

inline RoleMask RoleMaskFromWords(uint32_t low, uint32_t high) {
  return static_cast<RoleMask>(low) | (static_cast<RoleMask>(high) << 32);
}

}  // namespace tvm::tl::cuda::z3

#endif  // TVM_TL_CUDA_TRANSFORM_Z3_SCHEDULE_IR_H_
