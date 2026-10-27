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

#include "z3_role_planning.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>

#include <algorithm>
#include <array>
#include <functional>
#include <sstream>
#include <unordered_set>

#include "op/builtin.h"
#include "op/copy.h"
#include "op/operator.h"
#include "op/utils.h"
#include "ws_analysis.h"

namespace tvm::tl::cuda::z3 {

using namespace tirx;

namespace {

constexpr size_t kRoleBits = std::numeric_limits<RoleMask>::digits;
constexpr SpecialStateMask kLoopControl =
    static_cast<SpecialStateMask>(SpecialState::kLoopControl);
constexpr SpecialStateMask kCrossRoleProtocolState =
    static_cast<SpecialStateMask>(CudaSpecialState::kAsyncProxy) |
    static_cast<SpecialStateMask>(CudaSpecialState::kMBarrierProtocol);

RoleMask LowestRole(RoleMask mask) {
  ICHECK_NE(mask, 0);
  return mask & (~mask + 1);
}

std::string RoleMaskName(RoleMask mask, const CudaRoleTable& table) {
  std::ostringstream os;
  bool first = true;
  for (size_t i = 0; i < table.roles.size(); ++i) {
    if (!(mask & RoleMaskBit(static_cast<RoleId>(i)))) continue;
    if (!first) os << "|";
    os << table.roles[i]->name;
    first = false;
  }
  if (first) os << "<none>";
  return os.str();
}

RoleMask ActiveRoles(const CudaRoleTable& table) {
  ValidateRoleTableSize(table.roles.size());
  ICHECK(!table.roles.empty()) << "CUDA Z3 role planning requires at least one WSRole";
  RoleMask all = table.roles.size() == 64
                     ? std::numeric_limits<RoleMask>::max()
                     : ((RoleMask{1} << table.roles.size()) - 1);
  RoleMask active = table.active_mask == 0 ? all : table.active_mask;
  ICHECK(active != 0 && (active & ~all) == 0)
      << "CudaRoleTable.active_mask must be a nonempty subset of the WSRole entries";
  return active;
}

RoleMask RoleNamed(const CudaRoleTable& table, RoleMask active,
                   const ffi::String& name) {
  RoleMask result = 0;
  for (size_t i = 0; i < table.roles.size(); ++i) {
    if (table.roles[i]->name == name) result |= RoleMaskBit(static_cast<RoleId>(i));
  }
  return result & active;
}

std::string RolePath(const TaskNode* task) {
  std::vector<std::string> components;
  for (const IRStructure* parent = task->GetParent(); parent != nullptr;
       parent = parent->GetParent()) {
    ICHECK(parent->IsControl());
    const auto* control = static_cast<const ControlNode*>(parent);
    if (!control->task->ws_op_id.empty()) components.push_back(control->task->ws_op_id);
  }
  std::ostringstream os;
  for (auto it = components.rbegin(); it != components.rend(); ++it) os << *it << "/";
  if (!task->ws_op_id.empty()) {
    os << task->ws_op_id;
  } else {
    os << "source_" << task->GetSourceId();
  }
  const auto* parent = task->GetParent();
  if (parent != nullptr && parent->IsControl() &&
      static_cast<const ControlNode*>(parent)->task.get() == task) {
    os << "@header";
  }
  return os.str();
}

std::string UnavailableInput(const TaskNode* task,
                             const RoleAvailability& availability,
                             RoleMask candidate) {
  for (const Var& var : task->accesses.read_vars) {
    if ((candidate & availability.ValueRoles(var, task)) == 0) {
      std::ostringstream os;
      os << "unavailable scalar input " << var;
      return os.str();
    }
  }
  for (const BufferRegion& region : task->accesses.reads) {
    if (IsRolePrivateBuffer(region->buffer) &&
        (candidate & availability.ValueRoles(region->buffer->data, task)) == 0) {
      std::ostringstream os;
      os << "unavailable role-private input " << region->buffer->data;
      return os.str();
    }
  }
  for (const IRStructure* parent = task->GetParent(); parent != nullptr;
       parent = parent->GetParent()) {
    const auto* control = static_cast<const ControlNode*>(parent);
    if ((candidate & availability.TaskInputRoles(control->task.get())) == 0) {
      return "unavailable ancestor Control input in " + RolePath(control->task.get());
    }
  }
  return "one or more unavailable scalar/control expressions";
}

class ExternalEffectDetector final : public StmtExprVisitor {
 public:
  static bool HasExternalEffect(const Stmt& stmt) {
    ExternalEffectDetector detector;
    detector.VisitStmt(stmt);
    return detector.external_;
  }

 private:
  void VisitStmt_(const BufferStoreNode* op) final {
    external_ = external_ || !IsRolePrivateBuffer(op->buffer);
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const CallNode* op) final {
    Call call = ffi::GetRef<Call>(op);
    if (op->op.same_as(tl::loop_break())) return;
    if (ParseOperator(call).defined()) {
      external_ = true;
      return;
    }
    external_ = external_ || SideEffect(call) > CallEffectKind::kReadState;
    StmtExprVisitor::VisitExpr_(op);
  }

  bool external_{false};
};

bool PurelyReplicable(const TaskNode* task) {
  return !ExternalEffectDetector::HasExternalEffect(task->stmt);
}

class CandidateAssigner {
 public:
  CandidateAssigner(ScheduledTIR* scheduled, const CudaRoleTable& table,
                    const Target& target, RoleMask active)
      : scheduled_(scheduled), table_(table), target_(target), active_(active) {
    CollectAllTaskNodes(scheduled_->tree, &all_tasks_, true);
    CollectAnnotatedLayouts(scheduled_->metadata.kernel_root, layouts_);
  }

  void Run() {
    for (TaskNode* task : all_tasks_) {
      task->SetCandidateRoles(InitialCandidates(task));
      task->SetResolvedRoles(0);
    }
    RoleAvailability availability(all_tasks_);
    RefineCandidates(availability);
  }

 private:
  RoleMask TmaLayoutCandidates(const Call& call, RoleMask tma_role,
                               RoleMask worker_role, bool store) const {
    auto tile_op = ParseOperator(call);
    const auto* copy = tile_op.defined() ? tile_op.as<CopyNode>() : nullptr;
    if (copy == nullptr) return tma_role;
    const Buffer& shared = store ? copy->src : copy->dst;
    auto found = layouts_.find(shared->data);
    if (found == layouts_.end()) return tma_role;
    const auto& [buffer, layout] = found->second;
    return layout.defined() && IsTmaCompatibleLayout(layout, buffer) ? tma_role
                                                                     : worker_role;
  }

  RoleMask CapabilityForStatement(const Stmt& statement) const {
    RoleMask worker = RoleNamed(table_, active_, "Worker");
    Stmt classified_stmt = statement;
    Optional<Call> simple_call = GetEvaluateCallInSimpleWrapper(statement);
    if (simple_call.defined()) classified_stmt = Evaluate(simple_call.value());
    TileStmtKind kind = ClassifyStmt(classified_stmt, target_);
    switch (kind) {
      case TileStmtKind::kTmaProducer: {
        RoleMask load = RoleNamed(table_, active_, "Load");
        return simple_call.defined()
                   ? TmaLayoutCandidates(simple_call.value(), load, worker, false)
                   : load;
      }
      case TileStmtKind::kCpAsyncProducer:
        return RoleNamed(table_, active_, "Load");
      case TileStmtKind::kTmaStore: {
        RoleMask store = RoleNamed(table_, active_, "Store");
        return simple_call.defined()
                   ? TmaLayoutCandidates(simple_call.value(), store, worker, true)
                   : store;
      }
      case TileStmtKind::kTcgen05Mma:
        return RoleNamed(table_, active_, "MMA");
      case TileStmtKind::kCpAsyncRaw:
      case TileStmtKind::kSimtProducer:
      case TileStmtKind::kConsumer:
        return worker;
    }
    return worker;
  }

  RoleMask PerRoleCandidates(const TaskNode* task) const {
    class Visitor final : public StmtExprVisitor {
     public:
      Visitor(const CandidateAssigner* assigner, RoleMask worker)
          : assigner_(assigner), worker_(worker) {}
      RoleMask Run(const Stmt& stmt) {
        VisitStmt(stmt);
        return candidates_ == 0 ? worker_ : candidates_;
      }

     private:
      void VisitExpr_(const CallNode* op) final {
        Call call = ffi::GetRef<Call>(op);
        if (ParseOperator(call).defined()) {
          candidates_ |= assigner_->CapabilityForStatement(Evaluate(call));
          return;
        }
        if (SideEffect(call) > CallEffectKind::kReadState &&
            !op->op.same_as(tl::loop_break())) {
          candidates_ |= worker_;
        }
        StmtExprVisitor::VisitExpr_(op);
      }
      const CandidateAssigner* assigner_;
      RoleMask worker_;
      RoleMask candidates_{0};
    } visitor(this, RoleNamed(table_, active_, "Worker"));
    return visitor.Run(task->stmt) & active_;
  }

  RoleMask InitialCandidates(const TaskNode* task) const {
    RoleMask result;
    if (PurelyReplicable(task)) {
      result = active_;
    } else if (task->task_kind == TaskKind::kPerRoleGroup) {
      result = PerRoleCandidates(task);
    } else {
      result = CapabilityForStatement(task->stmt);
    }
    ICHECK(result != 0) << "No active WSRole can execute task " << RolePath(task)
                        << "; required instruction role is absent from CudaRoleTable";
    return result;
  }

  void RefineCandidates(const RoleAvailability& availability) {
    bool changed = true;
    while (changed) {
      changed = false;
      for (TaskNode* task : all_tasks_) {
        RoleMask available = availability.TaskInputRoles(task) & active_;
        for (const IRStructure* parent = task->GetParent(); parent != nullptr;
             parent = parent->GetParent()) {
          ICHECK(parent->IsControl());
          const auto* control = static_cast<const ControlNode*>(parent);
          available &= availability.TaskInputRoles(control->task.get());
        }
        RoleMask narrowed = task->CandidateRoles() & available;
        ICHECK(narrowed != 0)
            << "Cannot place task " << RolePath(task) << ": "
            << UnavailableInput(task, availability, task->CandidateRoles())
            << "; legal roles are " << RoleMaskName(task->CandidateRoles(), table_)
            << ": " << task->stmt;
        if (narrowed != task->CandidateRoles()) {
          task->SetCandidateRoles(narrowed);
          changed = true;
        }
      }
    }
  }

  ScheduledTIR* scheduled_;
  const CudaRoleTable& table_;
  Target target_;
  RoleMask active_;
  BufferLayoutMap layouts_;
  std::vector<TaskNode*> all_tasks_;
};

struct PreparedGuard {
  PrimExpr expression;
  TaskAccessInfo accesses;
};

class DemandResolver {
 public:
  DemandResolver(ScheduledTIR* scheduled, const MultiBufferPlan& plan,
                 const CudaRoleTable& table, RoleMask active)
      : scheduled_(scheduled), plan_(plan), table_(table), active_(active),
        availability_(CollectTasks()), execution_(all_tasks_.size(), 0) {
    for (TaskNode* task : all_tasks_) {
      ICHECK(task->CandidateRoles() != 0)
          << "ResolveRoleDemands requires AssignRoleCandidates first: " << RolePath(task);
      candidates_[task] = task->CandidateRoles();
    }
    RegisterProtocolInputs();
    RefineCandidates();
    for (size_t i = 0; i < all_tasks_.size(); ++i) {
      TaskNode* task = all_tasks_[i];
      if (task->task_kind == TaskKind::kPerRoleGroup) {
        execution_[i] = PreferredFallback(task);
      } else if (!PurelyReplicable(task)) {
        execution_[i] = candidates_.at(task);
      }
    }
  }

  void Run() {
    PropagateToFixedPoint();
    for (size_t i = 0; i < all_tasks_.size(); ++i) {
      if (execution_[i] == 0) RequireExecution(all_tasks_[i], PreferredFallback(all_tasks_[i]));
    }
    PropagateToFixedPoint();
    for (size_t i = 0; i < all_tasks_.size(); ++i) {
      TaskNode* task = all_tasks_[i];
      RoleMask resolved = execution_[i];
      ICHECK(resolved != 0 && (resolved & candidates_.at(task)) == resolved)
          << "Role demand resolution left an invalid mask for " << RolePath(task);
      if (task->task_kind == TaskKind::kPerRoleGroup) {
        ICHECK((resolved & (resolved - 1)) == 0)
            << "PerRole task must resolve to exactly one concrete role: " << RolePath(task);
      }
      task->SetResolvedRoles(resolved);
    }
  }

 private:
  std::vector<TaskNode*> CollectTasks() {
    CollectAllTaskNodes(scheduled_->tree, &all_tasks_, true);
    return all_tasks_;
  }

  size_t TaskIndex(const TaskNode* task) const {
    auto found = std::find(all_tasks_.begin(), all_tasks_.end(), task);
    ICHECK(found != all_tasks_.end());
    return static_cast<size_t>(found - all_tasks_.begin());
  }

  RoleMask Execution(const TaskNode* task) const { return execution_[TaskIndex(task)]; }

  RoleMask PreferredFallback(const TaskNode* task) const {
    RoleMask candidate = candidates_.at(task);
    RoleMask worker = RoleNamed(table_, active_, "Worker") & candidate;
    return worker != 0 ? LowestRole(worker) : LowestRole(candidate);
  }

  void RegisterProtocolInputs() {
    std::unordered_map<int, const MultiBufferInfo*> representatives;
    for (const MultiBufferInfo& info : plan_.Infos()) {
      if (!info.UsesCounter()) continue;
      ICHECK_GT(info.counter_group_id, 0);
      ICHECK(info.counter.defined()) << "Counter-clock storage " << info.storage
                                     << " has no physical counter";
      representatives.emplace(info.counter_group_id, &info);
    }
    for (TaskNode* task : all_tasks_) {
      std::unordered_set<int> seen_groups;
      auto append = [&](const ffi::Array<BufferRegion>& regions) {
        for (const BufferRegion& region : regions) {
          const MultiBufferInfo* info = plan_.Find(region->buffer);
          if (info == nullptr || !info->UsesCounter()) continue;
          ICHECK(info->FindOwner(*task) != nullptr)
              << "Counter-versioned task " << RolePath(task)
              << " is outside every owner for storage " << info->storage->name_hint;
          task_counters_[task].push_back(info);
          if (!seen_groups.insert(info->counter_group_id).second) continue;
          const MultiBufferInfo* representative = representatives.at(info->counter_group_id);
          for (const MultiBufferOwnerInfo& owner : representative->owners) {
            if (is_one(owner.active_guard)) continue;
            PreparedGuard guard{owner.active_guard, {}};
            AppendExpressionAccesses(owner.active_guard, &guard.accesses);
            task_guards_[task].push_back(std::move(guard));
          }
        }
      };
      append(task->accesses.reads);
      append(task->accesses.writes);
    }
  }

  void RefineCandidates() {
    bool changed = true;
    while (changed) {
      changed = false;
      for (TaskNode* task : all_tasks_) {
        RoleMask available = availability_.TaskInputRoles(task) & active_;
        for (const IRStructure* parent = task->GetParent(); parent != nullptr;
             parent = parent->GetParent()) {
          const auto* control = static_cast<const ControlNode*>(parent);
          available &= availability_.TaskInputRoles(control->task.get());
        }
        auto guards = task_guards_.find(task);
        if (guards != task_guards_.end()) {
          for (const PreparedGuard& guard : guards->second) {
            RoleMask guard_roles = availability_.AccessRoles(guard.accesses, task) & active_;
            ICHECK((task->CandidateRoles() & guard_roles) != 0)
                << "Cannot evaluate epoch/counter guard " << guard.expression << " for task "
                << RolePath(task) << " on its legal roles";
            available &= guard_roles;
          }
        }
        RoleMask narrowed = candidates_.at(task) & available;
        ICHECK(narrowed != 0)
            << "Cannot resolve role inputs for " << RolePath(task) << ": "
            << UnavailableInput(task, availability_, candidates_.at(task))
            << "; empty candidate intersection";
        if (narrowed != candidates_.at(task)) {
          candidates_[task] = narrowed;
          task->SetCandidateRoles(narrowed);
          changed = true;
        }
      }
    }
  }

  template <typename Map>
  bool AddDemand(Map* demands, const Var& value, RoleMask demand) {
    if (demand == 0) return false;
    RoleMask& slot = (*demands)[value];
    RoleMask updated = slot | demand;
    if (updated == slot) return false;
    slot = updated;
    return true;
  }

  bool AddReadDemand(const TaskAccessInfo& accesses, RoleMask demand) {
    if (demand == 0) return false;
    bool changed = false;
    for (const Var& var : accesses.read_vars) changed |= AddDemand(&demands_.scalar, var, demand);
    for (const BufferRegion& region : accesses.reads) {
      if (IsRolePrivateBuffer(region->buffer)) {
        changed |= AddDemand(&demands_.registers, region->buffer->data, demand);
      }
    }
    return changed;
  }

  bool AddReadDemand(const TaskNode* task, RoleMask demand) {
    return AddReadDemand(task->accesses, demand);
  }

  bool AddProtocolDemand(const TaskNode* task, RoleMask demand) {
    bool changed = false;
    auto counters = task_counters_.find(task);
    if (counters != task_counters_.end()) {
      for (const MultiBufferInfo* info : counters->second) {
        changed |= AddDemand(&demands_.registers, info->counter->data, demand);
      }
    }
    auto guards = task_guards_.find(task);
    if (guards != task_guards_.end()) {
      for (const PreparedGuard& guard : guards->second) {
        ICHECK((availability_.AccessRoles(guard.accesses, task) & demand) == demand)
            << "Role demand cannot evaluate epoch/counter guard " << guard.expression
            << " for " << RolePath(task);
        changed |= AddReadDemand(guard.accesses, demand);
      }
    }
    return changed;
  }

  RoleMask WriteDemand(const TaskNode* task) const {
    RoleMask result = 0;
    auto special = demands_.special_writers.find(task);
    if (special != demands_.special_writers.end()) result |= special->second;
    for (const Var& var : task->accesses.write_vars) {
      auto found = demands_.scalar.find(var);
      if (found != demands_.scalar.end()) result |= found->second;
    }
    for (const BufferRegion& region : task->accesses.writes) {
      if (!IsRolePrivateBuffer(region->buffer)) continue;
      auto found = demands_.registers.find(region->buffer->data);
      if (found != demands_.registers.end()) result |= found->second;
    }
    return result;
  }

  bool RequireExecution(TaskNode* task, RoleMask demand) {
    if (demand == 0) return false;
    RoleMask candidate = candidates_.at(task);
    ICHECK((demand & candidate) == demand)
        << "Cannot resolve role demand for " << RolePath(task) << ": consumers require "
        << RoleMaskName(demand, table_) << ", but candidates are "
        << RoleMaskName(candidate, table_)
        << "; externally visible effects cannot be broadcast and pure producers "
           "cannot read incompatible role-private inputs";
    size_t index = TaskIndex(task);
    RoleMask updated = execution_[index] | demand;
    if (updated == execution_[index]) return false;
    execution_[index] = updated;
    return true;
  }

  RoleMask SubtreeDemand(const IRStructure* node) const {
    if (node->IsTask()) return Execution(static_cast<const TaskNode*>(node));
    RoleMask result = 0;
    const auto* control = static_cast<const ControlNode*>(node);
    for (const auto& child : control->children) result |= SubtreeDemand(child.get());
    return result;
  }

  bool SeedTreeDemand(IRStructure* node, RoleMask enclosing_demand) {
    if (node->IsTask()) {
      auto* task = static_cast<TaskNode*>(node);
      bool changed = false;
      if (task->accesses.state_writes & kLoopControl) {
        changed |= RequireExecution(task, enclosing_demand);
      }
      RoleMask demand = Execution(task);
      changed |= AddReadDemand(task, demand);
      changed |= AddProtocolDemand(task, demand);
      return changed;
    }
    auto* control = static_cast<ControlNode*>(node);
    RoleMask demand = SubtreeDemand(control);
    bool changed = RequireExecution(control->task.get(), demand);
    changed |= AddReadDemand(control->task.get(), demand);
    changed |= AddProtocolDemand(control->task.get(), demand);
    for (auto& child : control->children) changed |= SeedTreeDemand(child.get(), demand);
    return changed;
  }

  struct SpecialScope {
    std::array<RoleMask, kRoleBits> readers{};
    std::array<std::vector<TaskNode*>, kRoleBits> writers;
  };

  SpecialScope CollectSpecialDemand(
      const std::vector<std::shared_ptr<IRStructure>>& nodes) {
    SpecialScope scope;
    auto add_task = [&](TaskNode* task) {
      RoleMask demand = Execution(task);
      // Mbarrier/async protocols intentionally join different roles; their
      // dependency edges do not make an arrive/wait definition role-local.
      // Ordinary special state retains Ascend-style reader-to-writer demand.
      SpecialStateMask reads =
          task->accesses.state_reads & ~(kLoopControl | kCrossRoleProtocolState);
      SpecialStateMask writes =
          task->accesses.state_writes & ~(kLoopControl | kCrossRoleProtocolState);
      for (size_t bit = 0; bit < kRoleBits; ++bit) {
        SpecialStateMask mask = SpecialStateMask{1} << bit;
        if (reads & mask) scope.readers[bit] |= demand;
        if (writes & mask) scope.writers[bit].push_back(task);
      }
    };
    for (const auto& node : nodes) {
      if (node->IsTask()) {
        add_task(static_cast<TaskNode*>(node.get()));
      } else {
        auto* control = static_cast<ControlNode*>(node.get());
        add_task(control->task.get());
        SpecialScope nested = CollectSpecialDemand(control->children);
        for (size_t bit = 0; bit < kRoleBits; ++bit) {
          scope.readers[bit] |= nested.readers[bit];
          scope.writers[bit].insert(scope.writers[bit].end(), nested.writers[bit].begin(),
                                    nested.writers[bit].end());
        }
      }
    }
    for (size_t bit = 0; bit < kRoleBits; ++bit) {
      for (TaskNode* writer : scope.writers[bit]) {
        demands_.special_writers[writer] |= scope.readers[bit];
      }
    }
    return scope;
  }

  void PropagateToFixedPoint() {
    bool changed = true;
    while (changed) {
      changed = false;
      demands_.special_writers.clear();
      CollectSpecialDemand(scheduled_->tree);
      for (auto& node : scheduled_->tree) {
        changed |= SeedTreeDemand(node.get(), SubtreeDemand(node.get()));
      }
      for (TaskNode* task : all_tasks_) changed |= RequireExecution(task, WriteDemand(task));
    }
  }

  ScheduledTIR* scheduled_;
  const MultiBufferPlan& plan_;
  const CudaRoleTable& table_;
  RoleMask active_;
  std::vector<TaskNode*> all_tasks_;
  RoleAvailability availability_;
  std::unordered_map<const TaskNode*, RoleMask> candidates_;
  DemandMaps demands_;
  std::vector<RoleMask> execution_;
  std::unordered_map<const TaskNode*, std::vector<const MultiBufferInfo*>> task_counters_;
  std::unordered_map<const TaskNode*, std::vector<PreparedGuard>> task_guards_;
};

ffi::String InspectTaskName(const TaskNode* task) { return ffi::String(RolePath(task)); }

ffi::Array<ffi::Any> RoleSummary(const ScheduledTIR& scheduled) {
  std::vector<TaskNode*> tasks;
  CollectAllTaskNodes(scheduled.tree, &tasks, true);
  ffi::Array<ffi::Any> result;
  for (const TaskNode* task : tasks) {
    ffi::Map<ffi::String, ffi::Any> item;
    item.Set("task", InspectTaskName(task));
    item.Set("candidate", Integer(static_cast<int64_t>(task->CandidateRoles())));
    item.Set("resolved", Integer(static_cast<int64_t>(task->ResolvedRoles())));
    item.Set("pure", Integer(PurelyReplicable(task) ? 1 : 0));
    result.push_back(item);
  }
  return result;
}

ffi::Array<ffi::Any> InspectRolePlanning(const SBlock& root, ffi::Array<WSRole> roles,
                                         int64_t active_mask, ffi::String target,
                                         bool resolve) {
  ICHECK_GE(active_mask, 0);
  ScheduledTIR scheduled = DecodeScheduledTIR(root, ConstrSet());
  CudaRoleTable table{roles, static_cast<RoleMask>(active_mask)};
  RefreshCudaTaskAccesses(&scheduled);
  AssignRoleCandidates(&scheduled, table, Target(target));
  if (resolve) {
    MultiBufferPlan empty;
    ResolveRoleDemands(&scheduled, empty, table);
  }
  return RoleSummary(scheduled);
}

ffi::Array<ffi::Any> InspectCounterRolePlanning(
    const SBlock& root, ffi::Array<WSRole> roles, int64_t active_mask,
    ffi::String target, ffi::String storage_name, ffi::String counter_name,
    ffi::Array<ffi::String> owner_ids, ffi::Array<PrimExpr> guards) {
  ICHECK_GE(active_mask, 0);
  ICHECK_EQ(owner_ids.size(), guards.size());
  ScheduledTIR scheduled = DecodeScheduledTIR(root, ConstrSet());
  CudaRoleTable table{roles, static_cast<RoleMask>(active_mask)};
  RefreshCudaTaskAccesses(&scheduled);
  AssignRoleCandidates(&scheduled, table, Target(target));

  MultiBufferInfo info;
  info.num_versions = 2;
  info.clock = ClockKind::kCounter;
  info.counter_group_id = 1;
  for (const Buffer& buffer : root->alloc_buffers) {
    if (buffer->name == storage_name) info.storage = buffer->data;
    if (buffer->name == counter_name) info.counter = buffer;
  }
  ICHECK(info.storage.defined() && info.counter.defined());
  std::unordered_map<ffi::String, ControlNode*> controls;
  std::function<void(IRStructure*)> collect = [&](IRStructure* node) {
    if (!node->IsControl()) return;
    auto* control = static_cast<ControlNode*>(node);
    controls.emplace(control->task->ws_op_id, control);
    for (auto& child : control->children) collect(child.get());
  };
  for (auto& node : scheduled.tree) collect(node.get());
  for (size_t i = 0; i < owner_ids.size(); ++i) {
    auto found = controls.find(owner_ids[i]);
    ICHECK(found != controls.end()) << "Unknown counter owner " << owner_ids[i];
    info.owners.push_back({found->second, guards[i]});
    info.owner_stages.push_back(0);
  }
  MultiBufferPlan plan;
  plan.AddInfo(std::move(info));
  ResolveRoleDemands(&scheduled, plan, table);
  return RoleSummary(scheduled);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.cuda.transform.Z3RolePlanning", InspectRolePlanning);
  refl::GlobalDef().def("tl.cuda.transform.Z3CounterRolePlanning",
                        InspectCounterRolePlanning);
}

}  // namespace

bool IsRolePrivateBuffer(const Buffer& buffer) {
  return IsLocalBuffer(buffer, true) || IsFragmentBuffer(buffer) || buffer.scope() == "register";
}

const MultiBufferOwnerInfo* MultiBufferInfo::FindOwner(const IRStructure& node) const {
  const MultiBufferOwnerInfo* result = nullptr;
  for (const MultiBufferOwnerInfo& owner : owners) {
    if (owner.loop == nullptr || !node.IsWithin(owner.loop)) continue;
    ICHECK(result == nullptr) << "Nested ownership for storage " << storage->name_hint;
    result = &owner;
  }
  return result;
}

const MultiBufferInfo* MultiBufferPlan::Find(const Var& storage_value) const {
  for (const MultiBufferInfo& info : infos_) {
    if (info.storage.same_as(storage_value)) return &info;
  }
  return nullptr;
}

const MultiBufferInfo* MultiBufferPlan::Find(const Buffer& buffer) const {
  return Find(buffer->data);
}

void MultiBufferPlan::AddInfo(MultiBufferInfo info) {
  ICHECK(info.storage.defined());
  ICHECK(Find(info.storage) == nullptr)
      << "Duplicate MultiBufferInfo for storage " << info.storage->name_hint;
  ICHECK_GT(info.num_versions, 0);
  if (info.UsesCounter()) {
    ICHECK_GT(info.counter_group_id, 0);
    ICHECK(info.counter.defined());
  } else {
    ICHECK_EQ(info.counter_group_id, 0);
    ICHECK(!info.counter.defined());
  }
  infos_.push_back(std::move(info));
}

RoleAvailability::RoleAvailability(const std::vector<TaskNode*>& tasks) {
  for (const TaskNode* task : tasks) {
    all_roles_ |= task->CandidateRoles();
    for (const Var& var : task->accesses.write_vars) writers_[var].push_back(task);
    for (const BufferRegion& region : task->accesses.writes) {
      if (IsRolePrivateBuffer(region->buffer)) writers_[region->buffer->data].push_back(task);
    }
  }
}

RoleMask RoleAvailability::ValueRoles(const Var& value, const TaskNode* reader) const {
  auto found = writers_.find(value);
  if (found == writers_.end()) return all_roles_;
  RoleMask result = all_roles_;
  bool reaching = false;
  for (const TaskNode* writer : found->second) {
    if (writer == reader) continue;
    reaching = true;
    result &= writer->CandidateRoles();
  }
  return reaching ? result : all_roles_;
}

RoleMask RoleAvailability::ExpressionRoles(const PrimExpr& expr,
                                            const TaskNode* reader) const {
  if (!expr.defined()) return all_roles_;
  TaskAccessInfo accesses;
  AppendExpressionAccesses(expr, &accesses);
  return AccessRoles(accesses, reader);
}

RoleMask RoleAvailability::AccessRoles(const TaskAccessInfo& accesses,
                                       const TaskNode* reader) const {
  RoleMask result = all_roles_;
  for (const Var& var : accesses.read_vars) result &= ValueRoles(var, reader);
  for (const BufferRegion& region : accesses.reads) {
    if (IsRolePrivateBuffer(region->buffer)) result &= ValueRoles(region->buffer->data, reader);
  }
  auto region_indices = [&](const ffi::Array<BufferRegion>& regions) {
    for (const BufferRegion& region : regions) {
      for (const Range& range : region->region) {
        result &= ExpressionRoles(range->min, reader);
        result &= ExpressionRoles(range->extent, reader);
      }
    }
  };
  region_indices(accesses.reads);
  region_indices(accesses.writes);
  return result;
}

RoleMask RoleAvailability::TaskInputRoles(const TaskNode* task) const {
  ICHECK(task != nullptr);
  return AccessRoles(task->accesses, task);
}

void AssignRoleCandidates(ScheduledTIR* scheduled, const CudaRoleTable& role_table,
                          const Target& target) {
  ICHECK(scheduled != nullptr);
  RefreshCudaTaskAccesses(scheduled);
  CandidateAssigner(scheduled, role_table, target, ActiveRoles(role_table)).Run();
}

void ResolveRoleDemands(ScheduledTIR* scheduled, const MultiBufferPlan& plan,
                        const CudaRoleTable& role_table) {
  ICHECK(scheduled != nullptr);
  DemandResolver(scheduled, plan, role_table, ActiveRoles(role_table)).Run();
}

}  // namespace tvm::tl::cuda::z3
