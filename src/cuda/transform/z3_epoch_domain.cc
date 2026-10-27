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

#include "z3_epoch_domain.h"

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/op.h>

#include <algorithm>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>

namespace tvm::tl::cuda::z3 {

using namespace tirx;

namespace {

using VarSet = std::unordered_set<Var, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>;

bool StructurallyEqual(const PrimExpr& a, const PrimExpr& b) {
  return a.same_as(b) || ffi::StructuralEqual()(a, b);
}

bool GuardImplies(const PrimExpr& premise, const PrimExpr& conclusion,
                  const ConstrSet& context) {
  ICHECK(premise.defined() && conclusion.defined());
  if (is_zero(premise) || is_one(conclusion) || StructurallyEqual(premise, conclusion)) {
    return true;
  }
  arith::Analyzer analyzer;
  context.Populate(analyzer);
  auto exit_constraint = analyzer.EnterConstraint(premise);
  bool result = analyzer.CanProve(conclusion);
  exit_constraint();
  return result;
}

std::string StorageName(const Var& storage) {
  return storage.defined() ? std::string(storage->name_hint) : std::string("<undefined>");
}

std::string ControlName(const ControlNode* control) {
  if (control == nullptr) return "<root>";
  if (control->task != nullptr && !control->task->ws_op_id.empty()) {
    return control->task->ws_op_id;
  }
  if (control->control.defined()) return control->control->loop_var->name_hint;
  return "source_" + std::to_string(control->GetSourceId());
}

std::string TaskName(const TaskNode* task) {
  if (!task->ws_op_id.empty()) return task->ws_op_id;
  return "source_" + std::to_string(task->GetSourceId());
}

void CollectControls(const std::vector<std::shared_ptr<IRStructure>>& nodes,
                     std::vector<ControlNode*>* controls) {
  for (const auto& node : nodes) {
    if (node->IsTask()) continue;
    auto* control = static_cast<ControlNode*>(node.get());
    controls->push_back(control);
    CollectControls(control->children, controls);
  }
}

ffi::Map<ffi::String, ffi::Any> InspectEpochDomainsImpl(const SBlock& root,
                                                        ffi::Array<WSRole> roles,
                                                        int64_t active_mask,
                                                        ffi::String target) {
  ICHECK_GE(active_mask, 0);
  ScheduledTIR scheduled = DecodeScheduledTIR(root, ConstrSet());
  CudaRoleTable table{roles, static_cast<RoleMask>(active_mask)};
  AssignRoleCandidates(&scheduled, table, Target(target));
  std::vector<TaskNode*> tasks;
  CollectAllTaskNodes(scheduled.tree, &tasks, true);
  RoleAvailability availability(tasks);
  MultiBufferPlan plan = PrepareBufferEpochs(&scheduled, availability);
  EpochDomainRegistry registry(scheduled, plan);

  ffi::Map<ffi::String, ffi::Any> result;
  ffi::Array<ffi::Any> domain_summary;
  for (size_t i = 0; i < plan.Infos().size(); ++i) {
    const MultiBufferInfo& info = plan.Infos()[i];
    ffi::Map<ffi::String, ffi::Any> item;
    item.Set("storage", ffi::String(StorageName(info.storage)));
    item.Set("versions", Integer(info.num_versions));
    item.Set("clock", ffi::String(info.UsesCounter() ? "counter" : "iteration"));
    item.Set("counter_group", Integer(info.counter_group_id));
    item.Set("counter", ffi::String(info.counter.defined() ? info.counter->name : ""));
    ffi::Array<ffi::Any> owners;
    for (size_t j = 0; j < info.owners.size(); ++j) {
      ffi::Map<ffi::String, ffi::Any> owner;
      owner.Set("owner", ffi::String(ControlName(info.owners[j].loop)));
      owner.Set("guard", info.owners[j].active_guard);
      owner.Set("stage", Integer(info.owner_stages[j]));
      owners.push_back(std::move(owner));
    }
    item.Set("owners", std::move(owners));
    domain_summary.push_back(std::move(item));
  }
  result.Set("plans", std::move(domain_summary));

  ffi::Array<ffi::Any> domains;
  // Domain IDs are dense in creation order. Compute the actual count from
  // lexical scopes and unique concrete counter owner/guard/counter tuples.
  std::vector<ControlNode*> controls;
  CollectControls(scheduled.tree, &controls);
  size_t domain_count = controls.size() + 1;
  // Counter-group members can intern the same concrete owner/guard/counter
  // domain. Compute the actual count from unique tuples rather than plan size.
  std::vector<std::tuple<ControlNode*, PrimExpr, Buffer>> unique_counter_domains;
  for (const MultiBufferInfo& info : plan.Infos()) {
    if (!info.UsesCounter()) continue;
    for (const MultiBufferOwnerInfo& owner : info.owners) {
      auto key = std::make_tuple(owner.loop, owner.active_guard, info.counter);
      bool found = false;
      for (const auto& existing : unique_counter_domains) {
        if (std::get<0>(existing) == owner.loop &&
            std::get<2>(existing).same_as(info.counter) &&
            StructurallyEqual(std::get<1>(existing), owner.active_guard)) {
          found = true;
          break;
        }
      }
      if (!found) unique_counter_domains.push_back(std::move(key));
    }
  }
  domain_count += unique_counter_domains.size();
  for (size_t id = 1; id <= domain_count; ++id) {
    const EpochDomain& domain = registry.Domain(static_cast<int>(id));
    ffi::Map<ffi::String, ffi::Any> item;
    item.Set("id", Integer(domain.id));
    item.Set("kind", ffi::String(domain.IsCounter() ? "counter" : "lexical"));
    item.Set("owner", ffi::String(ControlName(domain.owner)));
    item.Set("guard", domain.active_guard);
    item.Set("counter", ffi::String(domain.counter.defined() ? domain.counter->name : ""));
    domains.push_back(std::move(item));
  }
  result.Set("domains", std::move(domains));

  std::vector<const ControlNode*> scopes;
  scopes.push_back(nullptr);
  scopes.insert(scopes.end(), controls.begin(), controls.end());
  ffi::Array<ffi::Any> storage_domains;
  for (const MultiBufferInfo& info : plan.Infos()) {
    for (const ControlNode* scope : scopes) {
      ffi::Map<ffi::String, ffi::Any> item;
      item.Set("storage", ffi::String(StorageName(info.storage)));
      item.Set("scope", ffi::String(ControlName(scope)));
      item.Set("domain", Integer(registry.DomainForStorage(info.storage, scope)));
      storage_domains.push_back(std::move(item));
    }
  }
  result.Set("storage_domains", std::move(storage_domains));

  ffi::Array<ffi::Any> task_domains;
  for (TaskNode* task : tasks) {
    // Synthetic headers are valid registry queries too, and including them
    // makes escaped header accesses visible to host tests.
    for (const ControlNode* scope : scopes) {
      ffi::Map<ffi::String, ffi::Any> item;
      item.Set("task", ffi::String(TaskName(task)));
      item.Set("scope", ffi::String(ControlName(scope)));
      ffi::Array<int64_t> ids;
      for (int domain : registry.DomainsForTaskAtScope(*task, scope)) ids.push_back(domain);
      item.Set("domains", std::move(ids));
      task_domains.push_back(std::move(item));
    }
  }
  result.Set("task_domains", std::move(task_domains));

  ffi::Array<ffi::Any> projections;
  for (const ControlNode* scope : scopes) {
    for (size_t source = 1; source <= domain_count; ++source) {
      for (size_t target_id = 1; target_id <= domain_count; ++target_id) {
        if (!registry.CanProjectDomain(static_cast<int>(source), static_cast<int>(target_id),
                                       scope)) {
          continue;
        }
        ffi::Map<ffi::String, ffi::Any> item;
        item.Set("scope", ffi::String(ControlName(scope)));
        item.Set("source", Integer(static_cast<int64_t>(source)));
        item.Set("target", Integer(static_cast<int64_t>(target_id)));
        projections.push_back(std::move(item));
      }
    }
  }
  result.Set("projections", std::move(projections));
  return result;
}

}  // namespace

EpochDomainRegistry::EpochDomainRegistry(const ScheduledTIR& scheduled,
                                         const MultiBufferPlan& plan)
    : scheduled_(scheduled), plan_(plan) {
  AddLexicalScope(nullptr);
  RegisterLexicalScopes(scheduled_.tree);
  RegisterCounterDomains();
  BuildProjectionClosure();
}

const EpochDomain& EpochDomainRegistry::Domain(int id) const {
  ICHECK_GT(id, 0) << "CUDA Z3 epoch-domain IDs are positive";
  ICHECK_LE(static_cast<size_t>(id), domains_.size())
      << "Unknown CUDA Z3 epoch domain " << id;
  const EpochDomain& domain = domains_[id - 1];
  ICHECK_EQ(domain.id, id);
  return domain;
}

int EpochDomainRegistry::UnconditionalLexicalDomain(const ControlNode* scope) const {
  auto found = lexical_domain_ids_.find(scope);
  ICHECK(found != lexical_domain_ids_.end())
      << "Missing unconditional lexical domain for scope " << ControlName(scope);
  return found->second;
}

int EpochDomainRegistry::DomainForStorage(const Var& storage,
                                          const ControlNode* scope) const {
  ICHECK(storage.defined());
  if (scope != nullptr) {
    if (const MultiBufferInfo* info = plan_.Find(storage); info != nullptr && info->UsesCounter()) {
      if (const MultiBufferOwnerInfo* owner = info->FindOwner(*scope); owner != nullptr) {
        int domain = FindCounterBinding(storage, owner->loop);
        ICHECK_GT(domain, 0) << "Missing counter domain for storage " << StorageName(storage)
                             << " at owner " << ControlName(owner->loop);
        return domain;
      }
    }
  }
  return UnconditionalLexicalDomain(scope);
}

bool EpochDomainRegistry::CanProjectDomain(int source, int target,
                                           const ControlNode* scope) const {
  Domain(source);
  Domain(target);
  auto scope_found = projection_targets_.find(scope);
  ICHECK(scope_found != projection_targets_.end())
      << "Missing epoch projection scope " << ControlName(scope);
  auto source_found = scope_found->second.find(source);
  ICHECK(source_found != scope_found->second.end());
  return std::binary_search(source_found->second.begin(), source_found->second.end(), target);
}

std::vector<int> EpochDomainRegistry::DomainsForTaskAtScope(
    const TaskNode& task, const ControlNode* scope) const {
  std::vector<int> result;
  std::unordered_set<int> seen_domains;
  VarSet seen_storages;
  auto append = [&](const ffi::Array<BufferRegion>& regions) {
    for (const BufferRegion& region : regions) {
      if (IsRolePrivateBuffer(region->buffer)) continue;
      const Var& storage = region->buffer->data;
      if (!seen_storages.insert(storage).second) continue;
      if (const MultiBufferInfo* info = plan_.Find(storage); info != nullptr) {
        if (info->FindOwner(task) == nullptr) {
          ICHECK(IsMultiBufferBroadcastFill(&task, storage))
              << "CUDA Z3 task " << TaskName(&task) << " is outside every owner for storage "
              << StorageName(storage);
        }
      }
      int domain = DomainForStorage(storage, scope);
      if (seen_domains.insert(domain).second) result.push_back(domain);
    }
  };
  append(task.accesses.reads);
  append(task.accesses.writes);
  return result;
}

int EpochDomainRegistry::InternDomain(EpochDomainKind kind, ControlNode* owner,
                                      const PrimExpr& active_guard, const Buffer& counter) {
  ICHECK(active_guard.defined());
  ICHECK(kind == EpochDomainKind::kLexical || owner != nullptr);
  ICHECK_EQ(kind == EpochDomainKind::kCounter, counter.defined());
  for (const EpochDomain& domain : domains_) {
    if (domain.kind != kind || domain.owner != owner) continue;
    if (domain.counter.defined() != counter.defined()) continue;
    if (counter.defined() && !domain.counter.same_as(counter)) continue;
    if (StructurallyEqual(domain.active_guard, active_guard)) return domain.id;
  }
  int id = static_cast<int>(domains_.size()) + 1;
  domains_.push_back(EpochDomain{id, kind, owner, active_guard, counter});
  return id;
}

void EpochDomainRegistry::AddLexicalScope(ControlNode* scope) {
  int id = InternDomain(EpochDomainKind::kLexical, scope, Bool(true));
  auto [found, inserted] = lexical_domain_ids_.emplace(scope, id);
  ICHECK(inserted || found->second == id);
}

void EpochDomainRegistry::RegisterLexicalScopes(
    const std::vector<std::shared_ptr<IRStructure>>& nodes) {
  for (const auto& node : nodes) {
    if (node->IsTask()) continue;
    auto* control = static_cast<ControlNode*>(node.get());
    AddLexicalScope(control);
    RegisterLexicalScopes(control->children);
  }
}

void EpochDomainRegistry::RegisterCounterDomains() {
  for (const MultiBufferInfo& info : plan_.Infos()) {
    if (!info.UsesCounter()) continue;
    for (const MultiBufferOwnerInfo& owner : info.owners) {
      int id = InternDomain(EpochDomainKind::kCounter, owner.loop, owner.active_guard,
                            info.counter);
      BindCounterStorage(info.storage, owner.loop, id);
    }
  }
}

void EpochDomainRegistry::BindCounterStorage(const Var& storage, ControlNode* owner,
                                             int domain) {
  auto [found, inserted] = counter_bindings_[storage].emplace(owner, domain);
  ICHECK(inserted || found->second == domain)
      << "Conflicting counter epoch bindings for storage " << StorageName(storage);
}

int EpochDomainRegistry::FindCounterBinding(const Var& storage,
                                            const ControlNode* owner) const {
  auto storage_found = counter_bindings_.find(storage);
  if (storage_found == counter_bindings_.end()) return 0;
  auto owner_found = storage_found->second.find(owner);
  return owner_found == storage_found->second.end() ? 0 : owner_found->second;
}

void EpochDomainRegistry::BuildProjectionClosure() {
  const size_t count = domains_.size();
  const ConstrSet empty_context;
  for (const auto& [scope, unused] : lexical_domain_ids_) {
    (void)unused;
    const ConstrSet& context = scope == nullptr ? empty_context : scope->GetLoopBodyContext();
    std::vector<std::vector<bool>> projection(count, std::vector<bool>(count, false));
    for (size_t source = 0; source < count; ++source) {
      for (size_t target = 0; target < count; ++target) {
        const EpochDomain& from = domains_[source];
        const EpochDomain& to = domains_[target];
        bool allowed = source == target;
        if (!allowed && from.IsCounter() && !to.IsCounter()) {
          // A physical counter phase has no implicit lexical iteration period.
          allowed = false;
        } else if (!allowed && from.IsCounter() && to.IsCounter() &&
                   !from.counter.same_as(to.counter)) {
          allowed = false;
        } else if (!allowed &&
                   (SideEffect(from.active_guard) > CallEffectKind::kPure ||
                    SideEffect(to.active_guard) > CallEffectKind::kPure)) {
          // Distinct mutable-memory snapshots are not interchangeable merely
          // because their expressions have the same syntax.
          allowed = false;
        } else if (!allowed) {
          // Projection source -> target requires the target's active epoch to
          // be covered by the source guard at this concrete sync scope.
          allowed = GuardImplies(to.active_guard, from.active_guard, context);
        }
        projection[source][target] = allowed;
      }
    }
    for (size_t intermediate = 0; intermediate < count; ++intermediate) {
      for (size_t source = 0; source < count; ++source) {
        if (!projection[source][intermediate]) continue;
        for (size_t target = 0; target < count; ++target) {
          projection[source][target] =
              projection[source][target] || projection[intermediate][target];
        }
      }
    }
    auto& targets_by_source = projection_targets_[scope];
    for (size_t source = 0; source < count; ++source) {
      std::vector<int>& targets = targets_by_source[domains_[source].id];
      for (size_t target = 0; target < count; ++target) {
        if (projection[source][target]) targets.push_back(domains_[target].id);
      }
      std::sort(targets.begin(), targets.end());
    }
  }
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.cuda.transform.Z3EpochDomains", InspectEpochDomainsImpl);
}

}  // namespace tvm::tl::cuda::z3
