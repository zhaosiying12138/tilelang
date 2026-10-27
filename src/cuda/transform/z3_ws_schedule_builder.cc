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
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file z3_ws_schedule_builder.cc
 * \brief Project resolved recursive CUDA schedules into typed WS metadata.
 */

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "./auto_schedule/role_layout.h"
#include "./z3_buffer_ownership.h"
#include "./z3_lifetime_analysis.h"
#include "./z3_role_planning.h"

namespace tvm::tl::cuda::z3 {

using namespace tirx;

WSSchedule BuildZ3WSSchedule(ScheduledTIR* scheduled, const MultiBufferPlan& plan,
                             const CudaRoleTable& role_table, int worker_warps = 0,
                             bool widen_cpasync_load = false);
CudaSyncModel AdaptScheduledTIRToCudaSyncModel(const ScheduledTIR& scheduled,
                                               const MultiBufferPlan& plan,
                                               const EpochDomainRegistry& registry);

namespace {

using VarSet = std::unordered_set<Var, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>;

std::string NodeName(const IRStructure* node) {
  if (node->IsTask()) {
    const auto* task = static_cast<const TaskNode*>(node);
    if (!task->ws_op_id.empty()) return std::string(task->ws_op_id);
  } else {
    const auto* control = static_cast<const ControlNode*>(node);
    if (control->task != nullptr && !control->task->ws_op_id.empty()) {
      return std::string(control->task->ws_op_id);
    }
  }
  return "source_" + std::to_string(node->GetSourceId());
}

PrimExpr Conjoin(const PrimExpr& lhs, const PrimExpr& rhs) {
  if (!lhs.defined()) return rhs;
  if (!rhs.defined()) return lhs;
  if (is_one(lhs)) return rhs;
  if (is_one(rhs)) return lhs;
  return lhs && rhs;
}

PrimExpr ConjoinAll(const std::vector<PrimExpr>& terms) {
  PrimExpr result = Bool(true);
  for (const PrimExpr& term : terms) result = Conjoin(result, term);
  return result;
}

bool Prove(arith::Analyzer* analyzer, const PrimExpr& expression) {
  return analyzer->CanProve(expression) || analyzer->z3_prover.CanProve(expression);
}

RoleMask ActiveMask(const CudaRoleTable& table) {
  ValidateRoleTableSize(table.roles.size());
  RoleMask all = table.roles.size() == 64
                     ? std::numeric_limits<RoleMask>::max()
                     : (RoleMask{1} << table.roles.size()) - 1;
  RoleMask active = table.active_mask == 0 ? all : table.active_mask;
  ICHECK(active != 0 && (active & ~all) == 0) << "Invalid active CUDA WSRole mask";
  return active;
}

RoleId OnlyRole(RoleMask mask, const char* what) {
  ICHECK(mask != 0 && (mask & (mask - 1)) == 0)
      << "CUDA Z3 " << what << " must resolve to exactly one WSRole";
  return static_cast<RoleId>(__builtin_ctzll(mask));
}

struct PerRoleCandidateInfo {
  ffi::String group_id;
  PrimExpr guard;
};

struct Protocol {
  ffi::String name;
  std::vector<const MultiBufferInfo*> infos;
  std::vector<Buffer> buffers;
  VarSet storages;
  bool counter_clock{false};
  int counter_group{0};
  int depth{1};
  std::vector<ControlNode*> owners;
  std::vector<PrimExpr> guards;
  std::vector<int> stages;
  RoleMask producer_mask{0};
  RoleMask consumer_mask{0};
  RoleId producer{0};
  RoleId consumer{0};
  bool emit{false};
};

class RecursiveWSScheduleBuilder {
 public:
  RecursiveWSScheduleBuilder(ScheduledTIR* scheduled, const MultiBufferPlan& plan,
                             const CudaRoleTable& role_table, int worker_warps,
                             bool widen_cpasync_load)
      : scheduled_(scheduled), plan_(plan), role_table_(role_table),
        worker_warps_(worker_warps), widen_cpasync_load_(widen_cpasync_load) {}

  WSSchedule Run() {
    ClearSyntheticHeaderRoles(scheduled_->tree);
    ValidateScheduledTIR(*scheduled_);
    ValidateMultiBufferPlan(*scheduled_, plan_);
    AssignStableIds(scheduled_->tree);
    ExpandPerRoleGroups(&scheduled_->tree, nullptr);
    RefreshCudaTaskAccesses(scheduled_);
    CollectAllTaskNodes(scheduled_->tree, &tasks_, false);
    BuildProtocols();
    ComputeScopeParticipants();
    EmitRoles();
    EmitPipelines();
    EmitScopes();
    return WSSchedule(num_warps_, ws_roles_, ws_pipelines_, ws_scopes_);
  }

  const std::vector<Protocol>& protocols() const { return protocols_; }

 private:
  void ClearSyntheticHeaderRoles(const std::vector<std::shared_ptr<IRStructure>>& nodes) {
    for (const auto& node : nodes) {
      if (!node->IsControl()) continue;
      auto* control = static_cast<ControlNode*>(node.get());
      // Demand propagation uses headers as analysis endpoints. Their masks are
      // not executable role placement and must not survive into canonical
      // scheduled IR; descendants and real leaf tasks keep their own masks.
      control->task->SetCandidateRoles(0);
      control->task->SetResolvedRoles(0);
      ClearSyntheticHeaderRoles(control->children);
    }
  }

  void AssignStableIds(const std::vector<std::shared_ptr<IRStructure>>& nodes) {
    std::function<void(IRStructure*)> collect = [&](IRStructure* node) {
      TaskNode* identity = node->IsTask() ? static_cast<TaskNode*>(node)
                                         : static_cast<ControlNode*>(node)->task.get();
      if (!identity->ws_op_id.empty()) {
        ICHECK(used_ids_.insert(identity->ws_op_id).second)
            << "Duplicate tl.ws_op_id " << identity->ws_op_id;
      }
      next_source_id_ = std::max(next_source_id_, node->GetSourceId() + 1);
      if (node->IsControl()) {
        for (auto& child : static_cast<ControlNode*>(node)->children) collect(child.get());
      }
    };
    for (const auto& node : nodes) collect(node.get());

    std::function<void(IRStructure*)> assign = [&](IRStructure* node) {
      TaskNode* identity = node->IsTask() ? static_cast<TaskNode*>(node)
                                         : static_cast<ControlNode*>(node)->task.get();
      if (identity->ws_op_id.empty()) {
        std::string base = (node->IsTask() ? "tl.ws_task_" : "tl.ws_control_") +
                           std::to_string(node->GetSourceId());
        std::string candidate = base;
        for (int suffix = 2; !used_ids_.insert(candidate).second; ++suffix) {
          candidate = base + "_" + std::to_string(suffix);
        }
        identity->ws_op_id = ffi::String(candidate);
      }
      if (node->IsControl()) {
        auto* control = static_cast<ControlNode*>(node);
        control->control.CopyOnWrite()->annotations.Set(kWSOpId, control->task->ws_op_id);
        for (auto& child : control->children) assign(child.get());
      }
    };
    for (const auto& node : nodes) assign(node.get());
  }

  struct Candidate {
    const AttrStmtNode* marker;
    PrimExpr guard;
  };

  void CollectPerRoleCandidates(const Stmt& stmt, const PrimExpr& guard,
                                std::vector<Candidate>* out) const {
    if (const auto* sequence = stmt.as<SeqStmtNode>()) {
      for (const Stmt& child : sequence->seq) CollectPerRoleCandidates(child, guard, out);
      return;
    }
    if (const auto* condition = stmt.as<IfThenElseNode>()) {
      CollectPerRoleCandidates(condition->then_case, Conjoin(guard, condition->condition), out);
      if (condition->else_case.defined()) {
        CollectPerRoleCandidates(condition->else_case.value(),
                                 Conjoin(guard, !condition->condition), out);
      }
      return;
    }
    if (const auto* attribute = stmt.as<AttrStmtNode>()) {
      ICHECK(attribute->attr_key == kCudaTask)
          << "PerRole WS migration supports only candidate Tasks and their original guards; "
             "normalize or remove enclosing scheduling attributes before WS construction";
      out->push_back({attribute, guard});
      return;
    }
    if (const auto* evaluate = stmt.as<EvaluateNode>(); evaluate != nullptr && is_zero(evaluate->value)) {
      return;
    }
    LOG(FATAL) << "PerRole WS migration cannot preserve dynamic control around candidates: " << stmt;
  }

  std::vector<std::shared_ptr<IRStructure>> ExpandPerRoleGroup(const TaskNode& group) {
    const auto* marker = group.stmt.as<AttrStmtNode>();
    ICHECK(marker != nullptr && marker->attr_key == kCudaPerRoleTask);
    std::vector<Candidate> candidates;
    CollectPerRoleCandidates(marker->body, Bool(true), &candidates);
    ICHECK(!candidates.empty()) << "PerRole group " << NodeName(&group) << " has no candidates";

    arith::Analyzer analyzer;
    group.outer_ctx.Populate(analyzer);
    for (size_t i = 0; i < candidates.size(); ++i) {
      for (size_t j = i + 1; j < candidates.size(); ++j) {
        ICHECK(!Prove(&analyzer, candidates[i].guard && candidates[j].guard))
            << "PerRole candidates " << i << " and " << j << " in " << NodeName(&group)
            << " are provably nonexclusive";
      }
    }
    PrimExpr any = Bool(false);
    for (const Candidate& candidate : candidates) any = any || candidate.guard;
    ICHECK(!Prove(&analyzer, !any)) << "PerRole group " << NodeName(&group)
                                   << " has a provably nonexhaustive candidate set";

    std::vector<std::shared_ptr<IRStructure>> result;
    for (size_t i = 0; i < candidates.size(); ++i) {
      auto task = std::static_pointer_cast<TaskNode>(group.Clone());
      const Candidate& candidate = candidates[i];
      task->stmt = AttrStmt(candidate.marker->node, kCudaTask, candidate.marker->value,
                            candidate.marker->body, candidate.marker->span);
      task->task_kind = TaskKind::kOperation;
      task->ws_op_id = group.ws_op_id + "__candidate_" + std::to_string(i);
      ICHECK(used_ids_.insert(task->ws_op_id).second)
          << "Duplicate generated PerRole candidate ID " << task->ws_op_id;
      task->SetSourceId(next_source_id_++);
      task->PrependCondition(candidate.guard);
      per_role_candidates_[task.get()] = {group.ws_op_id, candidate.guard};
      result.push_back(std::move(task));
    }
    return result;
  }

  void ExpandPerRoleGroups(std::vector<std::shared_ptr<IRStructure>>* nodes,
                           IRStructure* parent) {
    std::vector<std::shared_ptr<IRStructure>> expanded;
    for (auto& node : *nodes) {
      if (node->IsControl()) {
        auto* control = static_cast<ControlNode*>(node.get());
        ExpandPerRoleGroups(&control->children, control);
        expanded.push_back(std::move(node));
      } else {
        auto* task = static_cast<TaskNode*>(node.get());
        if (task->task_kind == TaskKind::kPerRoleGroup) {
          std::vector<std::shared_ptr<IRStructure>> candidates = ExpandPerRoleGroup(*task);
          expanded.insert(expanded.end(), candidates.begin(), candidates.end());
        } else {
          expanded.push_back(std::move(node));
        }
      }
    }
    *nodes = std::move(expanded);
    RepairParentsAndIndices(*nodes, parent);
  }

  Buffer FindBuffer(const Var& storage) const {
    for (const Buffer& buffer : scheduled_->metadata.kernel_root->alloc_buffers) {
      if (buffer->data.same_as(storage)) return buffer;
    }
    for (const TaskNode* task : tasks_) {
      for (const ffi::Array<BufferRegion>* regions : {&task->accesses.reads, &task->accesses.writes}) {
        for (const BufferRegion& region : *regions) {
          if (region->buffer->data.same_as(storage)) return region->buffer;
        }
      }
    }
    LOG(FATAL) << "CUDA Z3 WS pipeline cannot resolve storage " << storage->name_hint;
    return Buffer();
  }

  void AddProtocolInfo(Protocol* protocol, const MultiBufferInfo& info) {
    if (!protocol->infos.empty()) {
      ICHECK_EQ(protocol->depth, info.num_versions)
          << "CUDA Z3 WS counter group " << protocol->counter_group
          << " mixes buffer depths; one full/empty barrier ring requires a common depth";
      ICHECK_EQ(protocol->owners.size(), info.owners.size());
      for (size_t i = 0; i < info.owners.size(); ++i) {
        ICHECK(protocol->owners[i] == info.owners[i].loop &&
               protocol->stages[i] == info.owner_stages[i] &&
               ffi::StructuralEqual()(protocol->guards[i], info.owners[i].active_guard))
            << "CUDA Z3 WS counter group has incompatible owner geometry";
      }
    } else {
      protocol->depth = info.num_versions;
      protocol->counter_clock = info.UsesCounter();
      protocol->counter_group = info.counter_group_id;
      protocol->name = info.UsesCounter() ? info.counter->name
                                          : info.storage->name_hint + "_ws";
      for (size_t i = 0; i < info.owners.size(); ++i) {
        protocol->owners.push_back(info.owners[i].loop);
        protocol->guards.push_back(info.owners[i].active_guard);
        protocol->stages.push_back(info.owner_stages[i]);
      }
    }
    protocol->infos.push_back(&info);
    protocol->buffers.push_back(FindBuffer(info.storage));
    protocol->storages.insert(info.storage);
    storage_protocol_[info.storage] = static_cast<int>(protocols_.size());
  }

  bool Touches(const TaskAccessInfo& accesses, const Protocol& protocol, bool writes) const {
    const ffi::Array<BufferRegion>& regions = writes ? accesses.writes : accesses.reads;
    for (const BufferRegion& region : regions) {
      if (protocol.storages.count(region->buffer->data)) return true;
    }
    return false;
  }

  bool Touches(const IRStructure& node, const Protocol& protocol) const {
    return Touches(node.Accesses(), protocol, true) || Touches(node.Accesses(), protocol, false);
  }

  void BuildProtocols() {
    for (const MultiBufferInfo& info : plan_.Infos()) {
      Protocol protocol;
      AddProtocolInfo(&protocol, info);
      protocols_.push_back(std::move(protocol));
    }
    // AddProtocolInfo records indices into the final vector. Group members were
    // appended independently above, so coalesce counter groups now and repair
    // the storage map deterministically.
    std::vector<Protocol> grouped;
    std::map<int, int> counter_groups;
    storage_protocol_.clear();
    for (Protocol& protocol : protocols_) {
      int index = -1;
      if (protocol.counter_clock) {
        auto [found, inserted] = counter_groups.emplace(protocol.counter_group, grouped.size());
        index = found->second;
        if (!inserted) {
          Protocol* existing = &grouped[index];
          for (const MultiBufferInfo* info : protocol.infos) AddProtocolInfo(existing, *info);
          continue;
        }
      } else {
        index = static_cast<int>(grouped.size());
      }
      grouped.push_back(std::move(protocol));
      for (const Var& storage : grouped[index].storages) storage_protocol_[storage] = index;
    }
    protocols_ = std::move(grouped);
    storage_protocol_.clear();
    for (size_t i = 0; i < protocols_.size(); ++i) {
      for (const Var& storage : protocols_[i].storages) storage_protocol_[storage] = i;
    }

    for (Protocol& protocol : protocols_) {
      for (const TaskNode* task : tasks_) {
        RoleMask roles = task->ResolvedRoles();
        ICHECK(roles != 0) << "Unresolved task role at WS construction: " << NodeName(task);
        bool writes = Touches(task->accesses, protocol, true);
        bool reads = Touches(task->accesses, protocol, false);
        if (writes) protocol.producer_mask |= roles;
        if (reads) protocol.consumer_mask |= roles;
      }
      protocol.consumer_mask &= ~protocol.producer_mask;
      if (protocol.producer_mask == 0 && protocol.consumer_mask == 0) continue;
      RoleMask all = protocol.producer_mask | protocol.consumer_mask;
      if (all != 0 && (all & (all - 1)) == 0) {
        // Same-role handoffs use the stock clock path; no WS barrier is needed.
        continue;
      }
      protocol.producer = OnlyRole(protocol.producer_mask, "producer frontier");
      protocol.consumer = OnlyRole(protocol.consumer_mask, "consumer frontier");
      ICHECK(protocol.producer != protocol.consumer)
          << "CUDA Z3 WS pipeline " << protocol.name << " needs distinct producer/consumer roles";
      protocol.emit = true;
      for (const TaskNode* task : tasks_) {
        if (!Touches(*task, protocol)) continue;
        RoleMask roles = task->ResolvedRoles();
        ICHECK(roles == RoleMaskBit(protocol.producer) || roles == RoleMaskBit(protocol.consumer))
            << "Task " << NodeName(task) << " touches cross-role WS storage from multiple or "
            << "non-protocol roles; split or replicate only scalar/private work";
      }
    }
    std::map<int, bool> group_emission;
    for (const Protocol& protocol : protocols_) {
      if (!protocol.counter_clock) continue;
      auto [found, inserted] = group_emission.emplace(protocol.counter_group, protocol.emit);
      ICHECK(inserted || found->second == protocol.emit)
          << "Counter group " << protocol.counter_group
          << " mixes same-role and cross-role WS protocols";
    }
  }

  void ComputeScopeParticipants() {
    RoleMask active = ActiveMask(role_table_);
    for (const TaskNode* task : tasks_) {
      if (task->task_kind == TaskKind::kDeclaration) continue;
      ICHECK((task->ResolvedRoles() & ~active) == 0)
          << "Task " << NodeName(task) << " resolves outside the active role table";
      used_roles_ |= task->ResolvedRoles();
    }
    std::function<RoleMask(IRStructure*)> visit = [&](IRStructure* node) -> RoleMask {
      if (node->IsTask()) {
        const auto* task = static_cast<const TaskNode*>(node);
        return task->task_kind == TaskKind::kDeclaration ? 0 : node->ResolvedRoles();
      }
      auto* control = static_cast<ControlNode*>(node);
      RoleMask roles = 0;
      for (auto& child : control->children) roles |= visit(child.get());
      for (const Protocol& protocol : protocols_) {
        if (!protocol.emit) continue;
        for (ControlNode* owner : protocol.owners) {
          if (owner == control) roles |= RoleMaskBit(protocol.producer) | RoleMaskBit(protocol.consumer);
        }
      }
      scope_roles_[control] = roles;
      return roles;
    };
    for (auto& node : scheduled_->tree) (void)visit(node.get());
    ICHECK(used_roles_ != 0) << "CUDA Z3 WS construction found no executable role demand";
  }

  void EmitRoles() {
    uint64_t layout_mask = 0;
    std::unordered_map<std::string, CudaScheduleRole> kinds;
    for (size_t i = 0; i < role_table_.roles.size(); ++i) {
      if ((used_roles_ & RoleMaskBit(static_cast<RoleId>(i))) == 0) continue;
      const WSRole& role = role_table_.roles[i];
      std::string name(role->name);
      std::optional<CudaScheduleRole> kind;
      if (name == "Worker") kind = CudaScheduleRole::kWorker;
      if (name == "Load") kind = CudaScheduleRole::kLoad;
      if (name == "MMA") kind = CudaScheduleRole::kMma;
      if (name == "Store") kind = CudaScheduleRole::kStore;
      ICHECK(kind.has_value()) << "Unknown recursive CUDA WSRole name " << name;
      kinds.emplace(name, *kind);
      layout_mask |= uint64_t{1} << static_cast<unsigned>(*kind);
    }
    if (worker_warps_ == 0) {
      worker_warps_ = 4;
      for (size_t i = 0; i < role_table_.roles.size(); ++i) {
        const WSRole& role = role_table_.roles[i];
        if (role->name == "Worker" && role->warp_hi > role->warp_lo) {
          worker_warps_ = static_cast<int>(role->warp_hi - role->warp_lo);
        }
      }
    }
    std::vector<CudaRoleLayout> layout =
        PlanCudaRoleLayout(layout_mask, worker_warps_, widen_cpasync_load_);
    std::unordered_map<std::string, CudaRoleLayout> by_name;
    for (const CudaRoleLayout& entry : layout) by_name.emplace(CudaScheduleRoleName(entry.role), entry);
    for (size_t i = 0; i < role_table_.roles.size(); ++i) {
      if ((used_roles_ & RoleMaskBit(static_cast<RoleId>(i))) == 0) continue;
      const WSRole& role = role_table_.roles[i];
      const CudaRoleLayout& planned = by_name.at(std::string(role->name));
      ws_roles_.push_back(WSRole(role->name, planned.warp_lo, planned.warp_hi, role->max_nreg));
      role_ids_[role->name] = static_cast<RoleId>(i);
      num_warps_ = std::max(num_warps_, planned.warp_hi);
    }
    num_warps_ = (num_warps_ + 3) / 4 * 4;
  }

  void EmitPipelines() {
    for (const Protocol& protocol : protocols_) {
      if (!protocol.emit) continue;
      ffi::Array<ffi::String> owners;
      if (protocol.counter_clock) {
        for (ControlNode* owner : protocol.owners) owners.push_back(owner->task->ws_op_id);
      }
      ws_pipelines_.push_back(WSPipeline(protocol.name, protocol.buffers, protocol.depth, owners));
    }
  }

  WSSync MakeSync(const Protocol& protocol, size_t owner_index, bool producer, bool open,
                  const PrimExpr& candidate_guard = PrimExpr()) const {
    PrimExpr guard = protocol.counter_clock ? protocol.guards[owner_index] : PrimExpr();
    if (candidate_guard.defined()) guard = Conjoin(guard, candidate_guard);
    ffi::Optional<PrimExpr> optional_guard = guard.defined() ? ffi::Optional<PrimExpr>(guard)
                                                             : ffi::Optional<PrimExpr>(std::nullopt);
    WSSyncKind kind = producer ? (open ? WSSyncKind::ProducerAcquire()
                                       : WSSyncKind::ProducerCommit())
                               : (open ? WSSyncKind::ConsumerWait()
                                       : WSSyncKind::ConsumerRelease());
    return WSSync(kind, protocol.name, protocol.stages[owner_index], optional_guard);
  }

  bool IsPerRoleCandidate(const IRStructure* node, const Protocol& protocol,
                          ffi::String* group, PrimExpr* guard) const {
    if (!node->IsTask()) return false;
    const auto* task = static_cast<const TaskNode*>(node);
    auto found = per_role_candidates_.find(task);
    if (found == per_role_candidates_.end()) return false;
    *group = found->second.group_id;
    *guard = found->second.guard;
    return true;
  }

  ffi::Array<WSInstr> EmitBody(const std::vector<std::shared_ptr<IRStructure>>& children,
                               ControlNode* scope, RoleId role) {
    struct Entry {
      const IRStructure* node;
      bool reference;
    };
    std::vector<Entry> entries;
    for (const auto& child : children) {
      bool participates = child->IsTask()
                              ? (child->ResolvedRoles() & RoleMaskBit(role)) != 0 &&
                                    static_cast<const TaskNode*>(child.get())->task_kind !=
                                        TaskKind::kDeclaration
                              : (scope_roles_.at(static_cast<ControlNode*>(child.get())) &
                                 RoleMaskBit(role)) != 0;
      entries.push_back({child.get(), participates});
    }

    std::vector<std::vector<WSSync>> before(entries.size()), after(entries.size());
    for (const Protocol& protocol : protocols_) {
      if (!protocol.emit || (role != protocol.producer && role != protocol.consumer)) continue;
      bool producer = role == protocol.producer;
      for (size_t owner_index = 0; owner_index < protocol.owners.size(); ++owner_index) {
        if (protocol.owners[owner_index] != scope) continue;
        std::vector<bool> candidate(entries.size(), false);
        std::vector<bool> group_touches;
        std::vector<ffi::String> groups;
        std::vector<PrimExpr> group_guards;
        for (size_t i = 0; i < entries.size(); ++i) {
          ffi::String group;
          PrimExpr guard;
          if (!entries[i].reference ||
              !IsPerRoleCandidate(entries[i].node, protocol, &group, &guard)) {
            continue;
          }
          auto found = std::find(groups.begin(), groups.end(), group);
          size_t group_index = found == groups.end() ? groups.size() : found - groups.begin();
          if (found == groups.end()) {
            groups.push_back(group);
            group_guards.push_back(guard);
            group_touches.push_back(false);
          }
          candidate[i] = true;
          group_touches[group_index] = group_touches[group_index] || Touches(*entries[i].node, protocol);
        }
        for (size_t i = 0; i < entries.size(); ++i) {
          if (!candidate[i]) continue;
          ffi::String group;
          PrimExpr guard;
          (void)IsPerRoleCandidate(entries[i].node, protocol, &group, &guard);
          size_t group_index = std::find(groups.begin(), groups.end(), group) - groups.begin();
          if (!group_touches[group_index]) continue;
          before[i].push_back(MakeSync(protocol, owner_index, producer, true, guard));
          after[i].push_back(MakeSync(protocol, owner_index, producer, false, guard));
        }

        int first = -1, last = -1;
        for (size_t i = 0; i < entries.size(); ++i) {
          if (!entries[i].reference || candidate[i] || !Touches(*entries[i].node, protocol)) continue;
          if (first < 0) first = static_cast<int>(i);
          last = static_cast<int>(i);
        }
        if (first >= 0) {
          before[first].push_back(MakeSync(protocol, owner_index, producer, true));
          after[last].push_back(MakeSync(protocol, owner_index, producer, false));
        } else if (groups.empty() ||
                   !std::any_of(group_touches.begin(), group_touches.end(),
                                [](bool touches) { return touches; })) {
          // Empty epochs still carry the fixed arrival signature and advance
          // every protocol role's continuous phase exactly once.
          before[0].push_back(MakeSync(protocol, owner_index, producer, true));
          after[0].push_back(MakeSync(protocol, owner_index, producer, false));
        }
      }
    }

    ffi::Array<WSInstr> result;
    for (size_t i = 0; i < entries.size(); ++i) {
      for (const WSSync& sync : before[i]) result.push_back(sync);
      if (entries[i].reference) {
        const TaskNode* task = entries[i].node->IsTask()
                                   ? static_cast<const TaskNode*>(entries[i].node)
                                   : static_cast<const ControlNode*>(entries[i].node)->task.get();
        result.push_back(WSOpRef(task->ws_op_id));
      }
      for (const WSSync& sync : after[i]) result.push_back(sync);
    }
    return result;
  }

  WSScope EmitScope(ControlNode* scope) {
    ffi::Map<ffi::String, ffi::Array<WSInstr>> bodies;
    const std::vector<std::shared_ptr<IRStructure>>& children =
        scope == nullptr ? scheduled_->tree : scope->children;
    for (size_t i = 0; i < role_table_.roles.size(); ++i) {
      RoleId role = static_cast<RoleId>(i);
      if ((used_roles_ & RoleMaskBit(role)) == 0) continue;
      if (scope != nullptr && (scope_roles_.at(scope) & RoleMaskBit(role)) == 0) continue;
      ffi::Array<WSInstr> body = EmitBody(children, scope, role);
      if (!body.empty()) bodies.Set(role_table_.roles[i]->name, std::move(body));
    }
    return WSScope(scope == nullptr ? kWSRootScopeId : scope->task->ws_op_id, bodies);
  }

  void EmitScopes() {
    std::function<void(ControlNode*)> emit = [&](ControlNode* control) {
      for (auto& child : control->children) {
        if (child->IsControl()) emit(static_cast<ControlNode*>(child.get()));
      }
      if (scope_roles_.at(control) != 0) ws_scopes_.push_back(EmitScope(control));
    };
    for (auto& node : scheduled_->tree) {
      if (node->IsControl()) emit(static_cast<ControlNode*>(node.get()));
    }
    ws_scopes_.push_back(EmitScope(nullptr));
  }

  ScheduledTIR* scheduled_;
  const MultiBufferPlan& plan_;
  const CudaRoleTable& role_table_;
  int worker_warps_;
  bool widen_cpasync_load_;
  int64_t next_source_id_{0};
  int num_warps_{0};
  RoleMask used_roles_{0};
  std::vector<TaskNode*> tasks_;
  std::vector<Protocol> protocols_;
  std::unordered_map<Var, int, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> storage_protocol_;
  std::unordered_map<const TaskNode*, PerRoleCandidateInfo> per_role_candidates_;
  std::unordered_map<const ControlNode*, RoleMask> scope_roles_;
  std::unordered_map<ffi::String, RoleId> role_ids_;
  std::unordered_set<ffi::String> used_ids_;
  ffi::Array<WSRole> ws_roles_;
  ffi::Array<WSPipeline> ws_pipelines_;
  ffi::Array<WSScope> ws_scopes_;
};

SBlock BuildWSScheduleInspection(const SBlock& root, ffi::Array<WSRole> roles,
                                 int64_t active_mask, ffi::String target,
                                 int64_t worker_warps, bool widen_cpasync_load) {
  ICHECK_GE(active_mask, 0);
  ICHECK_GE(worker_warps, 0);
  ScheduledTIR scheduled = DecodeScheduledTIR(root, ConstrSet());
  CudaRoleTable table{roles, static_cast<RoleMask>(active_mask)};
  RefreshCudaTaskAccesses(&scheduled);
  AssignRoleCandidates(&scheduled, table, Target(target));
  std::vector<TaskNode*> tasks;
  CollectAllTaskNodes(scheduled.tree, &tasks, true);
  RoleAvailability availability(tasks);
  MultiBufferPlan plan = PrepareBufferEpochs(&scheduled, availability);
  ResolveRoleDemands(&scheduled, plan, table);
  ValidateMultiBufferPlan(scheduled, plan);
  WSSchedule schedule = BuildZ3WSSchedule(&scheduled, plan, table,
                                          static_cast<int>(worker_warps), widen_cpasync_load);
  SBlock encoded = EncodeScheduledTIR(std::move(scheduled));
  encoded.CopyOnWrite()->annotations.Set(kWSScheduleKey, std::move(schedule));
  return encoded;
}

}  // namespace

WSSchedule BuildZ3WSSchedule(ScheduledTIR* scheduled, const MultiBufferPlan& plan,
                             const CudaRoleTable& role_table, int worker_warps,
                             bool widen_cpasync_load) {
  ICHECK(scheduled != nullptr);
  WSSchedule result = RecursiveWSScheduleBuilder(scheduled, plan, role_table, worker_warps,
                                                 widen_cpasync_load)
                          .Run();
  // Construction of the graph is a validation gate for every planned async
  // completion and for all concrete owner/domain projections. Alias results
  // remain a separate Slice H analysis product. A single-role/no-pipeline
  // result has no WS synchronization to validate and is discarded by Slice I.
  if (!result->pipelines.empty()) {
    EpochDomainRegistry registry(*scheduled, plan);
    CudaSyncModel model = AdaptScheduledTIRToCudaSyncModel(*scheduled, plan, registry);
    (void)CudaLifetimeGraph(model, &registry);
  }
  return result;
}

// The adapter is defined below the builder so both consume the same validated
// owner geometry. Its graph uses actual access endpoints and planned handshakes;
// all alias mathematics remains in z3_lifetime_analysis.cc.
CudaSyncModel AdaptScheduledTIRToCudaSyncModel(const ScheduledTIR& scheduled,
                                               const MultiBufferPlan& plan,
                                               const EpochDomainRegistry& registry) {
  CudaSyncModel model;
  std::vector<TaskNode*> tasks;
  CollectAllTaskNodes(scheduled.tree, &tasks, false);

  struct SyncSites {
    size_t acquire;
    size_t commit;
    size_t wait;
    size_t release;
    int domain;
  };
  struct ProtocolSites {
    std::vector<SyncSites> owners;
    std::vector<const MultiBufferInfo*> infos;
    int representative_domain{0};
    const ControlNode* lifetime_scope{nullptr};
    bool emit_ws{false};
  };
  std::map<int, ProtocolSites> counter_protocols;
  std::unordered_map<const MultiBufferInfo*, ProtocolSites> iteration_protocols;

  auto add_site = [&](int domain) {
    model.sites.push_back(CudaSyncSite{domain});
    return model.sites.size() - 1;
  };
  auto parent_scope = [](ControlNode* owner) -> const ControlNode* {
    if (owner == nullptr) return nullptr;
    const IRStructure* parent = owner->GetParent();
    return parent == nullptr ? nullptr : static_cast<const ControlNode*>(parent);
  };

  for (const MultiBufferInfo& info : plan.Infos()) {
    ProtocolSites protocol;
    protocol.infos.push_back(&info);
    for (const MultiBufferOwnerInfo& owner : info.owners) {
      int domain = registry.DomainForStorage(info.storage, owner.loop);
      protocol.owners.push_back({add_site(domain), add_site(domain), add_site(domain),
                                 add_site(domain), domain});
    }
    protocol.representative_domain = protocol.owners.front().domain;
    protocol.lifetime_scope = info.UsesCounter() ? parent_scope(info.owners.front().loop)
                                                 : info.owners.front().loop;
    RoleMask producers = 0, consumers = 0;
    for (const TaskNode* task : tasks) {
      for (const BufferRegion& region : task->accesses.writes) {
        if (region->buffer->data.same_as(info.storage)) producers |= task->ResolvedRoles();
      }
      for (const BufferRegion& region : task->accesses.reads) {
        if (region->buffer->data.same_as(info.storage)) consumers |= task->ResolvedRoles();
      }
    }
    consumers &= ~producers;
    RoleMask protocol_roles = producers | consumers;
    protocol.emit_ws = protocol_roles != 0 && (protocol_roles & (protocol_roles - 1)) != 0;
    if (info.UsesCounter()) {
      auto [found, inserted] = counter_protocols.emplace(info.counter_group_id, protocol);
      if (!inserted) {
        ICHECK(found->second.owners.size() == protocol.owners.size())
            << "Lifetime adapter found inconsistent counter-group owner counts";
        found->second.infos.push_back(&info);
        found->second.emit_ws = found->second.emit_ws || protocol.emit_ws;
      }
    } else {
      iteration_protocols.emplace(&info, std::move(protocol));
    }
  }

  auto get_protocol = [&](const MultiBufferInfo& info) -> ProtocolSites* {
    if (info.UsesCounter()) return &counter_protocols.at(info.counter_group_id);
    return &iteration_protocols.at(&info);
  };
  auto add_edge = [&](ProtocolSites* protocol, size_t src, size_t dst, int64_t distance,
                      bool strong) {
    model.edges.push_back(CudaSyncEdge{src, dst, distance, strong,
                                       protocol->representative_domain});
  };

  // Access sites are per storage and task. Issue edges are weak for both
  // synchronous and asynchronous atoms; the concrete close/full-empty
  // handshake below supplies the strong completion edge. Same-role storage has
  // no invented WS handshake and therefore cannot gain aliases from this model.
  for (const MultiBufferInfo& info : plan.Infos()) {
    ProtocolSites* protocol = get_protocol(info);
    CudaStorageLifetime lifetime;
    lifetime.storage = info.storage;
    lifetime.domain = protocol->representative_domain;
    lifetime.domain_kind = info.UsesCounter() ? EpochDomainKind::kCounter
                                              : EpochDomainKind::kLexical;
    lifetime.scope = protocol->lifetime_scope;
    lifetime.periodic = info.num_versions > 1;
    if (!info.UsesCounter() && lifetime.periodic) {
      PrimExpr iterations = CalculateIterationCount(info.owners.front().loop);
      const auto* constant = iterations.as<IntImmNode>();
      ICHECK(constant != nullptr && constant->value > 0)
          << "Lifetime adapter requires a static lexical period for " << info.storage->name_hint;
      lifetime.lexical_period = constant->value;
    }

    std::vector<GenerationLifetime> generations(info.owners.size());
    for (size_t i = 0; i < info.owners.size(); ++i) {
      generations[i].guard = info.owners[i].active_guard;
      // A nontrivial counter guard is conservatively treated as changing unless
      // a later proof establishes iteration invariance; no reuse is invented.
      generations[i].guard_iteration_invariant = is_one(info.owners[i].active_guard);
    }
    for (const TaskNode* task : tasks) {
      const MultiBufferOwnerInfo* owner = info.FindOwner(*task);
      if (owner == nullptr) continue;
      size_t owner_index = std::find_if(info.owners.begin(), info.owners.end(),
                                        [&](const MultiBufferOwnerInfo& candidate) {
                                          return &candidate == owner;
                                        }) - info.owners.begin();
      ICHECK_LT(owner_index, info.owners.size());
      const SyncSites& sync = protocol->owners[owner_index];
      bool writes = false, reads = false;
      for (const BufferRegion& region : task->accesses.writes) {
        writes = writes || region->buffer->data.same_as(info.storage);
      }
      for (const BufferRegion& region : task->accesses.reads) {
        reads = reads || region->buffer->data.same_as(info.storage);
      }
      if (!writes && !reads) continue;
      size_t access = add_site(sync.domain);
      bool producer = writes;
      if (producer) {
        add_edge(protocol, sync.acquire, access, 0, false);
        add_edge(protocol, access, sync.commit, 0, false);
        generations[owner_index].begin_sites.push_back({access, 0});
      } else {
        add_edge(protocol, sync.wait, access, 0, false);
        add_edge(protocol, access, sync.release, 0, false);
      }
      generations[owner_index].end_sites.push_back({access, 0});
    }
    lifetime.generations = std::move(generations);
    model.storages.push_back(std::move(lifetime));
  }

  auto connect_protocol = [&](ProtocolSites* protocol) {
    if (!protocol->emit_ws) return;
    for (const SyncSites& sync : protocol->owners) {
      add_edge(protocol, sync.commit, sync.wait, 0, true);
      add_edge(protocol, sync.release, sync.acquire, 1, true);
    }
    for (size_t i = 1; i < protocol->owners.size(); ++i) {
      add_edge(protocol, protocol->owners[i - 1].release, protocol->owners[i].acquire, 1, true);
      add_edge(protocol, protocol->owners[i - 1].commit, protocol->owners[i].wait, 0, true);
    }
  };
  for (auto& [id, protocol] : counter_protocols) connect_protocol(&protocol);
  for (auto& [info, protocol] : iteration_protocols) connect_protocol(&protocol);
  return model;
}

BufferAliasMap AnalyzeScheduledTIRCudaSyncModel(ScheduledTIR* scheduled,
                                                const MultiBufferPlan& plan) {
  return AnalyzeBufferLifetimesWithAdapter(scheduled, plan,
                                           AdaptScheduledTIRToCudaSyncModel);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.cuda.transform.Z3BuildWSSchedule", BuildWSScheduleInspection);
  refl::GlobalDef().def("tl.cuda.transform.Z3AnalyzeRecursiveLifetimes",
                        [](const SBlock& root, ffi::Array<WSRole> roles,
                           int64_t active_mask, ffi::String target) {
                          ICHECK_GE(active_mask, 0);
                          ScheduledTIR scheduled = DecodeScheduledTIR(root, ConstrSet());
                          CudaRoleTable table{roles, static_cast<RoleMask>(active_mask)};
                          RefreshCudaTaskAccesses(&scheduled);
                          AssignRoleCandidates(&scheduled, table, Target(target));
                          std::vector<TaskNode*> tasks;
                          CollectAllTaskNodes(scheduled.tree, &tasks, true);
                          RoleAvailability availability(tasks);
                          MultiBufferPlan plan = PrepareBufferEpochs(&scheduled, availability);
                          ResolveRoleDemands(&scheduled, plan, table);
                          (void)AnalyzeScheduledTIRCudaSyncModel(&scheduled, plan);
                          return EncodeScheduledTIR(std::move(scheduled));
                        });
}

}  // namespace tvm::tl::cuda::z3
