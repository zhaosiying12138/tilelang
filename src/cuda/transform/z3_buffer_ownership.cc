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

#include "z3_buffer_ownership.h"

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>

#include <algorithm>
#include <functional>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "op/builtin.h"
#include "op/utils.h"
#include "transform/common/attr.h"
#include "transform/common/write_first.h"

namespace tvm::tl::cuda::z3 {

using namespace tirx;

namespace {

using VarSet = std::unordered_set<Var, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>;
using ControlClaims = std::unordered_map<ControlNode*, std::vector<Var>>;

std::string StorageName(const Var& storage) {
  return storage.defined() ? std::string(storage->name_hint) : std::string("<undefined>");
}

std::string ControlName(const ControlNode* control) {
  if (control == nullptr) return "<null>";
  if (control->task != nullptr && !control->task->ws_op_id.empty()) {
    return control->task->ws_op_id;
  }
  if (control->control.defined()) return control->control->loop_var->name_hint;
  return "source_" + std::to_string(control->GetSourceId());
}

std::string TaskName(const TaskNode* task) {
  if (task == nullptr) return "<null>";
  if (!task->ws_op_id.empty()) return task->ws_op_id;
  return "source_" + std::to_string(task->GetSourceId());
}

bool StructurallyEqual(const PrimExpr& a, const PrimExpr& b) {
  return a.same_as(b) || ffi::StructuralEqual()(a, b);
}

bool GuardsEquivalent(const PrimExpr& a, const PrimExpr& b, const ConstrSet& context) {
  ICHECK(a.defined() && b.defined());
  if (StructurallyEqual(a, b)) return true;
  arith::Analyzer analyzer;
  context.Populate(analyzer);
  return analyzer.CanProveEqual(a, b);
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

PrimExpr DisjoinGuards(const PrimExpr& a, const PrimExpr& b, arith::Analyzer* analyzer) {
  if (is_zero(a) || is_one(b)) return b;
  if (is_zero(b) || is_one(a) || StructurallyEqual(a, b)) return a;
  if (analyzer->CanProveEqual(a, b) || GuardImplies(b, a, ConstrSet())) return a;
  if (GuardImplies(a, b, ConstrSet())) return b;
  PrimExpr result = a || b;
  return analyzer->CanProve(result) ? Bool(true) : result;
}

bool CanRewriteGuardAttribute(const ffi::String& key, const ffi::Any& node) {
  return (key == tirx::attr::tilelang_assume || key == tl::attr::kAssumeRequiresRuntimeCheck) &&
         node.as<PrimExprNode>() != nullptr;
}

bool ExprTouchesStorage(const PrimExpr& expr, const Var& storage) {
  if (!expr.defined()) return false;
  TaskAccessInfo accesses = AnalyzeCudaTaskAccesses(Evaluate(expr), {});
  auto touches = [&](const ffi::Array<BufferRegion>& regions) {
    return std::any_of(regions.begin(), regions.end(), [&](const BufferRegion& region) {
      return region->buffer->data.same_as(storage);
    });
  };
  return touches(accesses.reads) || touches(accesses.writes);
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

std::vector<const TaskNode*> CollectRealTasks(const ScheduledTIR& scheduled) {
  std::vector<TaskNode*> tasks;
  CollectAllTaskNodes(scheduled.tree, &tasks, false);
  return {tasks.begin(), tasks.end()};
}

bool IsAutomaticStorage(const Buffer& buffer) {
  const std::string scope = buffer.scope();
  return scope == "shared" || scope == "shared.dyn" || scope == "shared.tmem" ||
         scope == "shared.barrier" || scope == "local" || scope == "local.var" ||
         scope == "local.fragment" || scope == "register";
}

std::vector<Var> NormalizeStorageAnnotation(const ffi::Any& annotation) {
  std::vector<Var> result;
  VarSet seen;
  for (const ffi::Any& item : Downcast<ffi::Array<ffi::Any>>(annotation)) {
    Var storage;
    if (auto buffer = item.try_cast<Buffer>(); buffer.has_value()) {
      storage = buffer.value()->data;
    } else if (auto var = item.try_cast<Var>(); var.has_value()) {
      storage = var.value();
    } else {
      LOG(FATAL) << "CUDA Z3 '" << kMultiBufferEligible
                 << "' entries must be Buffer or Var objects, got " << item;
    }
    if (seen.insert(storage).second) result.push_back(storage);
  }
  return result;
}

ffi::Optional<Bind> GetFlatTaskBind(const Stmt& stmt) {
  Stmt current = stmt;
  while (const auto* attr = current.as<AttrStmtNode>()) current = attr->body;
  if (const auto* bind = current.as<BindNode>()) return ffi::GetRef<Bind>(bind);
  return std::nullopt;
}

bool IsPureScalarDefinition(const TaskNode* task) {
  ffi::Optional<Bind> bind = GetFlatTaskBind(task->stmt);
  return bind.has_value() && SideEffect(bind.value()->value) <= CallEffectKind::kPure;
}

// ---------------------------------------------------------------------------
// Four-state write-first classifier. Buffer aliases deliberately compare only
// Buffer.data; view shape and Buffer object identity do not affect ownership.
// ---------------------------------------------------------------------------

bool IsTerminal(AccessOrder order) {
  return order == AccessOrder::kReadFirst || order == AccessOrder::kWriteFirst;
}

class WriteFirstClassifier final : public StmtExprVisitor {
 public:
  AccessOrder Classify(const Stmt& stmt, const Var& storage) {
    target_storage_ = storage;
    result_ = AccessOrder::kUntouched;
    StmtExprVisitor::VisitStmt(stmt);
    return result_;
  }

  AccessOrder Classify(const PrimExpr& expr, const Var& storage) {
    target_storage_ = storage;
    result_ = AccessOrder::kUntouched;
    StmtExprVisitor::VisitExpr(expr);
    return result_;
  }

 private:
  void VisitStmt(const Stmt& stmt) final {
    if (!IsTerminal(result_)) StmtExprVisitor::VisitStmt(stmt);
  }

  void VisitExpr(const PrimExpr& expr) final {
    if (!IsTerminal(result_)) StmtExprVisitor::VisitExpr(expr);
  }

  template <typename F>
  AccessOrder ClassifyLocal(F visit) {
    AccessOrder saved = result_;
    result_ = AccessOrder::kUntouched;
    visit();
    AccessOrder local = result_;
    result_ = saved;
    return local;
  }

  bool MustExecute(const ForNode* op) const {
    arith::Analyzer analyzer;
    if (!analyzer.CanProve(op->extent > 0)) return false;
    return !op->step.has_value() || analyzer.CanProve(op->step.value() > 0);
  }

  void VisitExpr_(const BufferLoadNode* op) final {
    for (const PrimExpr& index : op->indices) {
      VisitExpr(index);
      if (IsTerminal(result_)) return;
    }
    if (op->predicate.defined()) VisitExpr(op->predicate.value());
    if (target_storage_.same_as(op->buffer->data)) result_ = AccessOrder::kReadFirst;
  }

  void VisitStmt_(const BufferStoreNode* op) final {
    VisitExpr(op->value);
    if (IsTerminal(result_)) return;
    for (const PrimExpr& index : op->indices) {
      VisitExpr(index);
      if (IsTerminal(result_)) return;
    }
    if (op->predicate.defined()) {
      VisitExpr(op->predicate.value());
      if (IsTerminal(result_)) return;
    }
    if (target_storage_.same_as(op->buffer->data)) result_ = AccessOrder::kWriteFirst;
  }

  void VisitStmt_(const SeqStmtNode* op) final {
    for (const Stmt& stmt : op->seq) {
      VisitStmt(stmt);
      if (IsTerminal(result_)) return;
    }
  }

  void VisitStmt_(const EvaluateNode* op) final { VisitExpr(op->value); }

  void VisitStmt_(const ForNode* op) final {
    VisitExpr(op->min);
    if (IsTerminal(result_)) return;
    VisitExpr(op->extent);
    if (IsTerminal(result_)) return;
    if (op->step.has_value()) {
      VisitExpr(op->step.value());
      if (IsTerminal(result_)) return;
    }
    AccessOrder body = ClassifyLocal([&] { VisitStmt(op->body); });
    result_ = SeqCompose(result_, DemoteIfNotMustExecute(body, MustExecute(op)));
  }

  void VisitStmt_(const IfThenElseNode* op) final {
    VisitExpr(op->condition);
    if (IsTerminal(result_)) return;
    AccessOrder then_order = ClassifyLocal([&] { VisitStmt(op->then_case); });
    AccessOrder branch;
    if (op->else_case.defined()) {
      AccessOrder else_order = ClassifyLocal([&] { VisitStmt(op->else_case.value()); });
      branch = MergeIfThenElse(then_order, else_order);
    } else {
      branch = MergeIfThenOnly(then_order);
    }
    result_ = SeqCompose(result_, branch);
  }

  void VisitStmt_(const BindNode* op) final { VisitExpr(op->value); }

  void VisitStmt_(const AttrStmtNode* op) final {
    if (CanRewriteGuardAttribute(op->attr_key, op->node)) {
      VisitExpr(Downcast<PrimExpr>(op->node));
      if (IsTerminal(result_)) return;
    }
    if (op->value.defined()) {
      VisitExpr(op->value);
      if (IsTerminal(result_)) return;
    }
    VisitStmt(op->body);
  }

  void VisitStmt_(const WhileNode* op) final {
    VisitExpr(op->condition);
    if (IsTerminal(result_)) return;
    AccessOrder body = ClassifyLocal([&] { VisitStmt(op->body); });
    result_ = SeqCompose(result_, DemoteIfNotMustExecute(body, false));
  }

  void VisitStmt_(const SBlockNode* op) final {
    if (op->init.defined()) {
      AccessOrder init = ClassifyLocal([&] { VisitStmt(op->init.value()); });
      result_ = SeqCompose(result_, DemoteIfNotMustExecute(init, false));
      if (IsTerminal(result_)) return;
    }
    VisitStmt(op->body);
  }

  void VisitStmt_(const SBlockRealizeNode* op) final {
    VisitExpr(op->predicate);
    if (IsTerminal(result_)) return;
    bool must_execute = false;
    if (const auto* immediate = op->predicate.as<IntImmNode>()) {
      must_execute = immediate->value != 0;
    }
    AccessOrder body = ClassifyLocal([&] { VisitStmt(op->block); });
    result_ = SeqCompose(result_, DemoteIfNotMustExecute(body, must_execute));
  }

  void VisitExpr_(const CallNode* op) final {
    static const Op& region_op = region();
    static const Op& access_ptr_op = tl::access_ptr();
    if (op->op.same_as(region_op)) {
      HandleRegionCall(op);
      return;
    }
    if (op->op.same_as(access_ptr_op)) {
      HandleAccessPtrCall(op);
      return;
    }
    for (const PrimExpr& arg : op->args) {
      VisitExpr(arg);
      if (IsTerminal(result_)) return;
    }
  }

  void HandleRegionCall(const CallNode* op) {
    const auto* load = op->args.size() >= 2 ? op->args[0].as<BufferLoadNode>() : nullptr;
    const auto* mode = op->args.size() >= 2 ? op->args[1].as<IntImmNode>() : nullptr;
    if (load == nullptr || mode == nullptr) {
      for (const PrimExpr& arg : op->args) {
        VisitExpr(arg);
        if (IsTerminal(result_)) return;
      }
      return;
    }
    for (const PrimExpr& index : load->indices) {
      VisitExpr(index);
      if (IsTerminal(result_)) return;
    }
    for (size_t i = 2; i < op->args.size(); ++i) {
      VisitExpr(op->args[i]);
      if (IsTerminal(result_)) return;
    }
    if (target_storage_.same_as(load->buffer->data)) {
      if (mode->value & 1) {
        result_ = AccessOrder::kReadFirst;
      } else if (mode->value & 2) {
        result_ = AccessOrder::kWriteFirst;
      }
    }
  }

  void HandleAccessPtrCall(const CallNode* op) {
    const auto* load = op->args.size() >= 3 ? op->args[0].as<BufferLoadNode>() : nullptr;
    const auto* mask = op->args.size() >= 3 ? op->args[2].as<IntImmNode>() : nullptr;
    if (load == nullptr || mask == nullptr) {
      for (const PrimExpr& arg : op->args) {
        VisitExpr(arg);
        if (IsTerminal(result_)) return;
      }
      return;
    }
    for (const PrimExpr& index : load->indices) {
      VisitExpr(index);
      if (IsTerminal(result_)) return;
    }
    VisitExpr(op->args[1]);
    if (IsTerminal(result_)) return;
    if (target_storage_.same_as(load->buffer->data)) {
      if (mask->value & 1) {
        result_ = AccessOrder::kReadFirst;
      } else if (mask->value & 2) {
        result_ = AccessOrder::kWriteFirst;
      }
    }
  }

  Var target_storage_;
  AccessOrder result_{AccessOrder::kUntouched};
};

int RequestedStage(const IRStructure* node) {
  return node->GetStage() == kUnscheduledStage ? 0 : node->GetStage();
}

std::vector<const IRStructure*> WriteFirstOrder(
    const std::vector<std::shared_ptr<IRStructure>>& nodes) {
  std::vector<const IRStructure*> result;
  result.reserve(nodes.size());
  bool has_stage = false;
  for (const auto& node : nodes) {
    result.push_back(node.get());
    has_stage |= node->GetStage() != kUnscheduledStage;
  }
  if (has_stage) {
    std::stable_sort(result.begin(), result.end(), [](const IRStructure* a, const IRStructure* b) {
      return RequestedStage(a) < RequestedStage(b);
    });
  }
  return result;
}

class IRWriteFirstClassifier {
 public:
  static AccessOrder Classify(const std::vector<std::shared_ptr<IRStructure>>& nodes,
                              const Var& storage, const ConstrSet& outer_context) {
    State state;
    for (const IRStructure* node : WriteFirstOrder(nodes)) {
      ProcessNode(node, storage, outer_context, &state);
      if (state.unsafe) return AccessOrder::kReadFirst;
    }
    if (!state.touched) return AccessOrder::kUntouched;
    return IsCovered(Bool(true), state.written_guard, outer_context)
               ? AccessOrder::kWriteFirst
               : AccessOrder::kMaybeWriteFirst;
  }

 private:
  struct State {
    bool touched{false};
    bool unsafe{false};
    PrimExpr written_guard{Bool(false)};
  };

  static bool IsCovered(const PrimExpr& access_guard, const PrimExpr& written_guard,
                        const ConstrSet& context) {
    return GuardsEquivalent(access_guard, written_guard, context) ||
           GuardImplies(access_guard, written_guard, context);
  }

  static PrimExpr MergeWriteGuard(const PrimExpr& known, const PrimExpr& added,
                                  const ConstrSet& context) {
    if (is_zero(known)) return added;
    if (is_zero(added) || GuardsEquivalent(known, added, context) ||
        GuardImplies(added, known, context)) {
      return known;
    }
    if (GuardImplies(known, added, context)) return added;
    PrimExpr result = known || added;
    return context.CanProve(result) ? Bool(true) : result;
  }

  static void ProcessAccess(AccessOrder order, const PrimExpr& active_guard,
                            const ConstrSet& context, State* state) {
    if (order == AccessOrder::kUntouched || state->unsafe) return;
    state->touched = true;
    if (order == AccessOrder::kReadFirst) {
      state->unsafe = !IsCovered(active_guard, state->written_guard, context);
    } else if (order == AccessOrder::kWriteFirst) {
      state->written_guard = MergeWriteGuard(state->written_guard, active_guard, context);
    }
  }

  static AccessOrder ClassifyPayload(const IRStructure* node, const Var& storage) {
    WriteFirstClassifier classifier;
    if (node->IsTask()) {
      return classifier.Classify(static_cast<const TaskNode*>(node)->stmt, storage);
    }
    const auto* control = static_cast<const ControlNode*>(node);
    AccessOrder header = classifier.Classify(control->task->stmt, storage);
    if (IsTerminal(header)) return header;
    AccessOrder body = Classify(control->children, storage, control->GetLoopBodyContext());
    arith::Analyzer analyzer;
    bool must_execute = analyzer.CanProve(control->control->extent > 0) &&
                        (!control->control->step.has_value() ||
                         analyzer.CanProve(control->control->step.value() > 0));
    return SeqCompose(header, DemoteIfNotMustExecute(body, must_execute));
  }

  static void ProcessNode(const IRStructure* node, const Var& storage,
                          const ConstrSet& context, State* state) {
    PrimExpr active_guard = Bool(true);
    WriteFirstClassifier classifier;
    for (const auto& guard : node->GetGuards()) {
      if (guard->IsCondition()) {
        const PrimExpr& condition = static_cast<const ConditionGuard*>(guard.get())->condition;
        ProcessAccess(classifier.Classify(condition, storage), active_guard, context, state);
        active_guard = is_one(active_guard) ? condition : active_guard && condition;
        continue;
      }
      const auto* attribute = static_cast<const AttributeGuard*>(guard.get());
      if (auto expr = attribute->node.try_cast<PrimExpr>(); expr.has_value()) {
        ProcessAccess(classifier.Classify(expr.value(), storage), active_guard, context, state);
      }
      if (attribute->value.defined()) {
        ProcessAccess(classifier.Classify(attribute->value, storage), active_guard, context,
                      state);
      }
    }
    ProcessAccess(ClassifyPayload(node, storage), active_guard, context, state);
  }
};

// ---------------------------------------------------------------------------
// Complete structural owner coverage.
// ---------------------------------------------------------------------------

enum class Coverage { kUntouched, kCovered, kNeedsOwner };

struct CoveragePlan {
  Coverage coverage{Coverage::kUntouched};
  std::vector<ControlNode*> owners;

  bool IsComplete() const {
    return coverage == Coverage::kUntouched || coverage == Coverage::kCovered;
  }

  void Merge(const CoveragePlan& other) {
    coverage = std::max(coverage, other.coverage);
    owners.insert(owners.end(), other.owners.begin(), other.owners.end());
  }
};

CoveragePlan PlanOwnerNode(IRStructure* node, const Var& storage);

CoveragePlan PlanOwnerList(const std::vector<std::shared_ptr<IRStructure>>& nodes,
                           const Var& storage) {
  CoveragePlan result;
  for (const auto& node : nodes) result.Merge(PlanOwnerNode(node.get(), storage));
  return result;
}

CoveragePlan PlanOwnerNode(IRStructure* node, const Var& storage) {
  if (node->IsTask()) {
    const auto* task = static_cast<const TaskNode*>(node);
    CoveragePlan result;
    if (!task->TouchesStorage(storage)) return result;
    bool broadcast_fill = !task->GuardsTouchStorage(storage) &&
                          CanBroadcastFillToStorage(task->stmt, storage);
    result.coverage = broadcast_fill ? Coverage::kCovered : Coverage::kNeedsOwner;
    return result;
  }

  auto* control = static_cast<ControlNode*>(node);
  CoveragePlan result = PlanOwnerList(control->children, storage);
  if (result.coverage == Coverage::kNeedsOwner) {
    AccessOrder order = IRWriteFirstClassifier::Classify(
        control->children, storage, control->GetLoopBodyContext());
    if (order != AccessOrder::kReadFirst) {
      result.coverage = Coverage::kCovered;
      result.owners = {control};
    }
  }
  // A Control header is evaluated outside its body and therefore cannot be
  // covered by claiming that same Control. It must promote the claim outward.
  if (control->task->TouchesStorage(storage)) result.coverage = Coverage::kNeedsOwner;
  return result;
}

void ValidateOwnerShape(const ScheduledTIR& scheduled, const Var& storage,
                        const std::vector<ControlNode*>& owners) {
  std::vector<ControlNode*> controls;
  CollectControls(scheduled.tree, &controls);
  std::unordered_set<ControlNode*> known(controls.begin(), controls.end());
  for (size_t i = 0; i < owners.size(); ++i) {
    ControlNode* owner = owners[i];
    ICHECK(owner != nullptr && known.count(owner))
        << "CUDA Z3 storage " << StorageName(storage) << " has an unknown owner loop";
    ICHECK(owner->BodyTouchesStorage(storage))
        << "CUDA Z3 owner " << ControlName(owner) << " for storage " << StorageName(storage)
        << " contains no access to that storage";
    AccessOrder order = IRWriteFirstClassifier::Classify(
        owner->children, storage, owner->GetLoopBodyContext());
    ICHECK(order != AccessOrder::kReadFirst)
        << "CUDA Z3 owner " << ControlName(owner) << " for storage " << StorageName(storage)
        << " is read-first; write-first ownership is unprovable";
    for (size_t j = i + 1; j < owners.size(); ++j) {
      ICHECK(owner != owners[j]) << "Duplicate CUDA Z3 owner for storage " << StorageName(storage);
      ICHECK(!owner->IsWithin(owners[j]) && !owners[j]->IsWithin(owner))
          << "CUDA Z3 storage " << StorageName(storage) << " has nested owner loops "
          << ControlName(owner) << " and " << ControlName(owners[j])
          << "; owners must be disjoint";
      ICHECK(CompareIRStructure(owner, owners[j]) < 0)
          << "CUDA Z3 owners for storage " << StorageName(storage)
          << " are not in final schedule order: " << ControlName(owner) << " versus "
          << ControlName(owners[j]);
    }
  }
}

bool StrictlyWithinOneOwner(const ControlNode* control,
                            const std::vector<ControlNode*>& owners) {
  ControlNode* found = nullptr;
  for (ControlNode* owner : owners) {
    if (!control->IsWithin(owner)) continue;
    if (control == owner || found != nullptr) return false;
    found = owner;
  }
  return found != nullptr;
}

ffi::Array<Var> BroadcastFills(const TaskNode* task) {
  if (task == nullptr) return {};
  const auto* semantic = task->stmt.as<AttrStmtNode>();
  if (semantic == nullptr ||
      (semantic->attr_key != kCudaTask && semantic->attr_key != kCudaPerRoleTask)) {
    return {};
  }
  const auto* metadata = semantic->node.as<ffi::MapObj>();
  if (metadata == nullptr) return {};
  for (const auto& [key, value] : *metadata) {
    if (key.cast<ffi::String>() == kMultiBufferBroadcastFill) {
      return value.cast<ffi::Array<Var>>();
    }
  }
  return {};
}

void MarkBroadcastFill(TaskNode* task, const Var& storage) {
  ffi::Array<Var> fills = BroadcastFills(task);
  for (const Var& existing : fills) {
    if (existing.same_as(storage)) return;
  }
  fills.push_back(storage);
  AttrStmt semantic = Downcast<AttrStmt>(task->stmt);
  ffi::Map<ffi::String, ffi::Any> metadata;
  if (const auto* old = semantic->node.as<ffi::MapObj>()) {
    for (const auto& [key, value] : *old) metadata.Set(key.cast<ffi::String>(), value);
  }
  metadata.Set(kMultiBufferBroadcastFill, std::move(fills));
  task->stmt = AttrStmt(std::move(metadata), semantic->attr_key, semantic->value,
                        semantic->body, semantic->span);
}

void ValidateStorageCoverage(ScheduledTIR* scheduled, const Var& storage,
                             const std::vector<ControlNode*>& owners) {
  ValidateOwnerShape(*scheduled, storage, owners);
  for (TaskNode* task : [&] {
         std::vector<TaskNode*> tasks;
         CollectAllTaskNodes(scheduled->tree, &tasks, false);
         return tasks;
       }()) {
    if (!task->TouchesStorage(storage)) continue;
    size_t owner_count = 0;
    for (ControlNode* owner : owners) owner_count += task->IsWithin(owner);
    ICHECK_LE(owner_count, 1U)
        << "CUDA Z3 task " << TaskName(task) << " belongs to nested owners for storage "
        << StorageName(storage);
    if (owner_count == 1) continue;
    ICHECK(!task->GuardsTouchStorage(storage))
        << "CUDA Z3 storage " << StorageName(storage) << " is read by a guard on task "
        << TaskName(task) << " outside its owner";
    ICHECK(CanBroadcastFillToStorage(task->stmt, storage))
        << "CUDA Z3 storage " << StorageName(storage) << " is accessed outside its owner "
        << "frontier by task " << TaskName(task) << "; only a recognized, non-self-reading "
        << "owner-external fill may initialize every physical version";
    MarkBroadcastFill(task, storage);
  }

  std::vector<ControlNode*> controls;
  CollectControls(scheduled->tree, &controls);
  for (ControlNode* control : controls) {
    const For& loop = control->control;
    bool bound_touches = ExprTouchesStorage(loop->min, storage) ||
                         ExprTouchesStorage(loop->extent, storage) ||
                         (loop->step.has_value() && ExprTouchesStorage(loop->step.value(), storage));
    ICHECK(!bound_touches)
        << "CUDA Z3 storage " << StorageName(storage) << " is used by an unnormalized loop "
        << "bound in " << ControlName(control)
        << "; normalize buffer-dependent bounds into schedulable tasks before ownership";
    if (control->GuardsTouchStorage(storage)) {
      ICHECK(StrictlyWithinOneOwner(control, owners))
          << "CUDA Z3 storage " << StorageName(storage) << " is used by a Control guard in "
          << ControlName(control) << " outside a strict descendant of exactly one owner";
    }
  }
}

MultiBufferOwnerMap CurrentOwnerClaims(const ScheduledTIR& scheduled) {
  MultiBufferOwnerMap result;
  std::vector<ControlNode*> controls;
  CollectControls(scheduled.tree, &controls);
  for (ControlNode* control : controls) {
    auto annotation = control->control->annotations.Get(kMultiBufferEligible);
    if (!annotation.has_value()) continue;
    for (const Var& storage : NormalizeStorageAnnotation(annotation.value())) {
      result[storage].push_back(control);
    }
  }
  return result;
}

struct StorageEpochAnalysis {
  PrimExpr active_guard{Bool(true)};
  std::optional<int> stage;
  bool cross_stage{false};
  bool late_guard{false};
  bool mutable_guard{false};
};

StorageEpochAnalysis AnalyzeStorageEpoch(const Var& storage, ControlNode* owner) {
  ICHECK(owner != nullptr);
  std::vector<PrimExpr> guard_terms;
  std::unordered_map<Var, int, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> definitions;
  arith::Analyzer analyzer;
  owner->GetLoopBodyContext().Populate(analyzer);
  StorageEpochAnalysis result;
  int first_access = -1;
  int guard_ready = -1;

  for (const auto& child : owner->children) {
    const TaskNode* direct = child->IsTask()
                                 ? static_cast<const TaskNode*>(child.get())
                                 : static_cast<const ControlNode*>(child.get())->task.get();
    for (const Var& var : direct->accesses.write_vars) {
      definitions[var] = static_cast<int>(child->GetIndex());
    }
    if (!child->TouchesStorage(storage)) continue;
    int position = static_cast<int>(child->GetIndex());
    if (first_access < 0) first_access = position;
    if (!result.stage.has_value()) {
      result.stage = child->GetStage();
    } else if (*result.stage != child->GetStage()) {
      result.cross_stage = true;
    }
    guard_terms.push_back(child->GetConditionGuard());
  }
  ICHECK_GE(first_access, 0) << "CUDA Z3 owner " << ControlName(owner)
                             << " has no epoch access for storage " << StorageName(storage);
  if (result.cross_stage) return result;

  PrimExpr guard = Bool(false);
  for (const PrimExpr& term : guard_terms) guard = DisjoinGuards(guard, term, &analyzer);
  if (analyzer.CanProve(guard)) guard = Bool(true);
  result.active_guard = guard;

  TaskAccessInfo guard_accesses;
  AppendExpressionAccesses(guard, &guard_accesses);
  for (const Var& var : guard_accesses.read_vars) {
    auto definition = definitions.find(var);
    if (definition != definitions.end()) guard_ready = std::max(guard_ready, definition->second);
  }
  if (guard_ready >= first_access) {
    result.active_guard = Bool(true);
    result.late_guard = true;
    return result;
  }

  // A mutable load spelled identically at two positions is not a proof that
  // both snapshots agree. Such guards may have a private counter, but are not
  // merged with another storage's counter without a stronger snapshot proof.
  for (const BufferRegion& region : guard_accesses.reads) {
    const Var& sampled = region->buffer->data;
    for (const auto& child : owner->children) {
      if (child->WritesStorage(sampled)) result.mutable_guard = true;
    }
  }
  if (result.mutable_guard) {
    // The owner-boundary value cannot be identified with every earlier
    // snapshot. Use the unconditional domain locally and never share this
    // unproved snapshot with another storage's physical counter.
    result.active_guard = Bool(true);
  }
  return result;
}

using BindDefinitions =
    std::unordered_map<Var, const TaskNode*, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>;

bool ExprUsesAnyVar(const PrimExpr& expr, const VarSet& vars) {
  return expr.defined() && !vars.empty() &&
         UsesVar(expr, [&](const VarNode* node) { return vars.count(ffi::GetRef<Var>(node)); });
}

bool HasNonAffineLoopPath(const ControlNode* owner, const BindDefinitions& bind_definitions) {
  VarSet outer_loop_vars;
  std::vector<const ControlNode*> outer_loops;
  auto varying = [&](const PrimExpr& expr) {
    if (SideEffect(expr) > CallEffectKind::kPure || ExprUsesAnyVar(expr, outer_loop_vars)) {
      return true;
    }
    return UsesVar(expr, [&](const VarNode* node) {
      auto definition = bind_definitions.find(ffi::GetRef<Var>(node));
      if (definition == bind_definitions.end()) return false;
      return std::any_of(outer_loops.begin(), outer_loops.end(), [&](const ControlNode* loop) {
        return definition->second->IsWithin(loop);
      });
    });
  };

  for (const ControlNode* control : owner->GetAncestorControls()) {
    if (control->HasGuards() || control->BodyHasLoopBreak()) return true;
    const For& loop = control->control;
    if (varying(loop->min) || varying(loop->extent) ||
        (loop->step.has_value() && varying(loop->step.value()))) {
      return true;
    }
    outer_loop_vars.insert(loop->loop_var);
    outer_loops.push_back(control);
  }
  return false;
}

RoleMask StorageProtocolRoles(const ScheduledTIR& scheduled, const MultiBufferInfo& info) {
  RoleMask result = 0;
  for (const TaskNode* task : CollectRealTasks(scheduled)) {
    if (!task->TouchesStorage(info.storage) || info.FindOwner(*task) == nullptr ||
        IsMultiBufferBroadcastFill(task, info.storage)) {
      continue;
    }
    ICHECK(task->CandidateRoles() != 0)
        << "CUDA Z3 clock preparation requires role candidates before task " << TaskName(task);
    result |= task->CandidateRoles();
  }
  ICHECK(result != 0) << "CUDA Z3 counter storage " << StorageName(info.storage)
                      << " has no role-assigned owner access";
  return result;
}

// Conservatively check role-uniform epoch decisions. Shared immutable inputs
// are supported; role-private or mutable memory requires an explicit proof
// that this plan does not have. Scalar definitions are checked separately.
bool GuardRoleUniform(const PrimExpr& guard, RoleMask roles,
                      const VarSet& written_storages) {
  if ((roles & (roles - 1)) == 0) return true;
  TaskAccessInfo accesses;
  AppendExpressionAccesses(guard, &accesses);
  for (const BufferRegion& region : accesses.reads) {
    if (IsRolePrivateBuffer(region->buffer) ||
        written_storages.count(region->buffer->data)) {
      return false;
    }
  }
  return true;
}

struct PreparedStorage {
  MultiBufferInfo info;
  RoleMask protocol_roles{0};
  std::vector<bool> mutable_guards;
};

struct CounterGroup {
  std::vector<MultiBufferOwnerInfo> owners;
  std::vector<int> stages;
  std::vector<bool> mutable_guards;
  RoleMask protocol_roles{0};
  int id{0};
  Buffer counter;
};

class BufferEpochBuilder {
 public:
  BufferEpochBuilder(ScheduledTIR* scheduled, const RoleAvailability& availability)
      : scheduled_(scheduled), availability_(availability) {
    CollectAllTaskNodes(scheduled_->tree, &all_tasks_, true);
    for (const TaskNode* task : all_tasks_) {
      for (const Var& var : task->accesses.write_vars) scalar_writers_.emplace(var, task);
      for (const BufferRegion& region : task->accesses.writes) {
        written_storages_.insert(region->buffer->data);
      }
      if (ffi::Optional<Bind> bind = GetFlatTaskBind(task->stmt); bind.has_value()) {
        bind_definitions_.emplace(bind.value()->var, task);
      }
    }
  }

  MultiBufferPlan Build() {
    MultiBufferOwnerMap owner_map = InferCompleteOwnerFrontier(scheduled_);
    std::vector<std::pair<Var, int>> selected;
    for (const auto& [storage, versions] : scheduled_->metadata.selected_versions) {
      selected.emplace_back(storage, versions);
    }
    std::sort(selected.begin(), selected.end(), [](const auto& a, const auto& b) {
      return StorageName(a.first) < StorageName(b.first);
    });

    std::vector<PreparedStorage> prepared;
    for (const auto& [storage, versions] : selected) {
      auto found = owner_map.find(storage);
      ICHECK(found != owner_map.end() && !found->second.empty())
          << "CUDA Z3 selected storage " << StorageName(storage)
          << " has no complete owner frontier; read-first or owner-external access is not "
             "automatically versionable";
      MultiBufferInfo info;
      info.storage = storage;
      info.num_versions = versions;
      ffi::String mode = "auto";
      if (auto requested = scheduled_->metadata.version_modes.Get(storage); requested.has_value()) {
        mode = requested.value();
      }
      ICHECK(mode == "auto" || mode == "iteration" || mode == "counter")
          << "Unknown CUDA Z3 buffer clock mode '" << mode << "' for " << StorageName(storage);

      bool multiple_owners = found->second.size() > 1;
      bool non_affine = false;
      bool late_guard = false;
      bool cross_stage = false;
      std::vector<StorageEpochAnalysis> analyses;
      for (ControlNode* owner : found->second) {
        StorageEpochAnalysis analysis = AnalyzeStorageEpoch(storage, owner);
        non_affine |= HasNonAffineLoopPath(owner, bind_definitions_);
        late_guard |= analysis.late_guard;
        cross_stage |= analysis.cross_stage;
        info.owners.push_back({owner, analysis.active_guard});
        info.owner_stages.push_back(analysis.stage.value_or(kUnscheduledStage));
        analyses.push_back(std::move(analysis));
      }

      bool stages_available = !cross_stage;
      for (const StorageEpochAnalysis& analysis : analyses) {
        stages_available &= analysis.stage.has_value() && *analysis.stage >= 0;
      }
      bool guarded = std::any_of(analyses.begin(), analyses.end(), [](const auto& analysis) {
        return !is_one(analysis.active_guard);
      });
      bool needs_counter = multiple_owners || non_affine || (!cross_stage && !late_guard && guarded);

      if (mode == "iteration") {
        ICHECK(!multiple_owners)
            << "Explicit iteration clock for storage " << StorageName(storage)
            << " requires exactly one owner";
        ICHECK(!non_affine && !guarded)
            << "Explicit iteration clock for storage " << StorageName(storage)
            << " requires an affine, unconditional owner path; use counter mode";
        info.clock = ClockKind::kIteration;
      } else if (mode == "counter") {
        ICHECK(stages_available)
            << "Explicit counter clock for storage " << StorageName(storage)
            << " requires one nonnegative common stage per owner; accesses at multiple stages "
               "cannot share an active epoch";
        ICHECK(!late_guard)
            << "Explicit counter clock for storage " << StorageName(storage)
            << " has an epoch guard defined at or after its first access; move the definition "
               "earlier or use the lexical iteration domain";
        info.clock = ClockKind::kCounter;
      } else if (needs_counter) {
        ICHECK(stages_available)
            << "CUDA Z3 storage " << StorageName(storage)
            << " requires a counter clock but has unscheduled or cross-stage owner accesses";
        ICHECK(!late_guard)
            << "CUDA Z3 storage " << StorageName(storage)
            << " requires a counter clock, but its guard definition does not dominate the first "
               "storage access";
        info.clock = ClockKind::kCounter;
      } else {
        ICHECK(!multiple_owners && !non_affine)
            << "CUDA Z3 storage " << StorageName(storage)
            << " cannot fall back to an unconditional iteration clock";
        info.clock = ClockKind::kIteration;
      }

      if (!info.UsesCounter()) {
        info.owners.front().active_guard = Bool(true);
        info.counter_group_id = 0;
        info.counter = Buffer();
      }
      PreparedStorage item;
      item.info = std::move(info);
      for (const StorageEpochAnalysis& analysis : analyses) {
        item.mutable_guards.push_back(analysis.mutable_guard);
      }
      if (item.info.UsesCounter()) {
        item.protocol_roles = StorageProtocolRoles(*scheduled_, item.info);
        for (const MultiBufferOwnerInfo& owner : item.info.owners) {
          ICHECK(GuardAvailable(owner.active_guard, item.protocol_roles))
              << "CUDA Z3 epoch guard " << owner.active_guard << " for storage "
              << StorageName(storage) << " is not available on every protocol role";
          ICHECK(GuardRoleUniform(owner.active_guard, item.protocol_roles, written_storages_))
              << "CUDA Z3 epoch guard " << owner.active_guard << " for storage "
              << StorageName(storage)
              << " may denote different active epochs on different roles; a role-uniform "
                 "predicate is required";
          ICHECK(UniformScalarsAvailable(owner.active_guard, item.protocol_roles))
              << "CUDA Z3 scalar inputs to epoch guard " << owner.active_guard
              << " are not demonstrably role-uniform for storage " << StorageName(storage);
        }
      }
      prepared.push_back(std::move(item));
    }

    AssignCounterGroups(&prepared);
    MultiBufferPlan plan;
    for (PreparedStorage& item : prepared) plan.AddInfo(std::move(item.info));
    ValidateMultiBufferPlan(*scheduled_, plan);
    return plan;
  }

 private:
  RoleMask GuardRoles(const PrimExpr& guard) const { return availability_.ExpressionRoles(guard); }

  bool GuardAvailable(const PrimExpr& guard, RoleMask roles) const {
    return (GuardRoles(guard) & roles) == roles;
  }

  bool UniformScalarsAvailable(const PrimExpr& guard, RoleMask roles) const {
    if ((roles & (roles - 1)) == 0) return true;
    TaskAccessInfo accesses;
    AppendExpressionAccesses(guard, &accesses);
    std::function<bool(const Var&, VarSet*)> check = [&](const Var& var, VarSet* active) {
      auto found = scalar_writers_.find(var);
      if (found == scalar_writers_.end()) return true;
      if (!active->insert(var).second) return false;
      const TaskNode* writer = found->second;
      bool valid = (writer->CandidateRoles() & roles) == roles &&
                   writer->accesses.writes.empty() && writer->accesses.state_writes == 0 &&
                   IsPureScalarDefinition(writer);
      for (const Var& input : writer->accesses.read_vars) {
        valid = valid && check(input, active);
      }
      for (const BufferRegion& region : writer->accesses.reads) {
        valid = valid && !IsRolePrivateBuffer(region->buffer);
      }
      active->erase(var);
      return valid;
    };
    VarSet active;
    for (const Var& var : accesses.read_vars) {
      if (!check(var, &active)) return false;
    }
    return true;
  }

  bool Compatible(const CounterGroup& group, const PreparedStorage& storage) const {
    if (group.owners.size() != storage.info.owners.size() ||
        group.stages != storage.info.owner_stages) {
      return false;
    }
    RoleMask combined = group.protocol_roles | storage.protocol_roles;
    for (size_t i = 0; i < group.owners.size(); ++i) {
      if (group.mutable_guards[i] || storage.mutable_guards[i]) return false;
      const MultiBufferOwnerInfo& a = group.owners[i];
      const MultiBufferOwnerInfo& b = storage.info.owners[i];
      if (a.loop != b.loop ||
          !GuardsEquivalent(a.active_guard, b.active_guard, b.loop->GetLoopBodyContext()) ||
          !GuardAvailable(a.active_guard, combined) || !GuardAvailable(b.active_guard, combined) ||
          !GuardRoleUniform(a.active_guard, combined, written_storages_) ||
          !GuardRoleUniform(b.active_guard, combined, written_storages_) ||
          !UniformScalarsAvailable(a.active_guard, combined) ||
          !UniformScalarsAvailable(b.active_guard, combined)) {
        return false;
      }
    }
    return true;
  }

  void AssignCounterGroups(std::vector<PreparedStorage>* storages) {
    std::vector<CounterGroup> groups;
    int next_id = 1;
    for (PreparedStorage& storage : *storages) {
      if (!storage.info.UsesCounter()) continue;
      CounterGroup* selected = nullptr;
      for (CounterGroup& group : groups) {
        if (Compatible(group, storage)) {
          selected = &group;
          break;
        }
      }
      if (selected == nullptr) {
        CounterGroup group;
        group.owners = storage.info.owners;
        group.stages = storage.info.owner_stages;
        group.mutable_guards = storage.mutable_guards;
        group.protocol_roles = storage.protocol_roles;
        group.id = next_id++;
        group.counter = decl_buffer(
            {IntImm(DataType::Int(32), 1)}, DataType::Int(32),
            StorageName(storage.info.storage) + "_version_counter_" + std::to_string(group.id),
            "local.var");
        groups.push_back(std::move(group));
        selected = &groups.back();
      } else {
        selected->protocol_roles |= storage.protocol_roles;
        for (size_t i = 0; i < storage.info.owners.size(); ++i) {
          storage.info.owners[i].active_guard = selected->owners[i].active_guard;
        }
      }
      storage.info.counter_group_id = selected->id;
      storage.info.counter = selected->counter;
    }
  }

  ScheduledTIR* scheduled_;
  const RoleAvailability& availability_;
  std::vector<TaskNode*> all_tasks_;
  BindDefinitions bind_definitions_;
  std::unordered_map<Var, const TaskNode*, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> scalar_writers_;
  VarSet written_storages_;
};

ffi::Map<ffi::String, ffi::Any> PlanSummary(const MultiBufferInfo& info) {
  ffi::Map<ffi::String, ffi::Any> result;
  result.Set("storage", ffi::String(StorageName(info.storage)));
  result.Set("versions", Integer(info.num_versions));
  result.Set("clock", ffi::String(info.UsesCounter() ? "counter" : "iteration"));
  result.Set("counter_group", Integer(info.counter_group_id));
  result.Set("counter", ffi::String(info.counter.defined() ? info.counter->name : ""));
  ffi::Array<ffi::Any> owners;
  for (size_t i = 0; i < info.owners.size(); ++i) {
    ffi::Map<ffi::String, ffi::Any> owner;
    owner.Set("owner", ffi::String(ControlName(info.owners[i].loop)));
    owner.Set("guard", info.owners[i].active_guard);
    owner.Set("stage", Integer(i < info.owner_stages.size() ? info.owner_stages[i]
                                                            : kUnscheduledStage));
    owners.push_back(std::move(owner));
  }
  result.Set("owners", std::move(owners));
  return result;
}

ffi::Array<ffi::Any> InspectOwnership(const SBlock& root) {
  ScheduledTIR scheduled = DecodeScheduledTIR(root, ConstrSet());
  MultiBufferOwnerMap owners = InferCompleteOwnerFrontier(&scheduled);
  std::vector<std::pair<Var, std::vector<ControlNode*>>> ordered(owners.begin(), owners.end());
  std::sort(ordered.begin(), ordered.end(), [](const auto& a, const auto& b) {
    return StorageName(a.first) < StorageName(b.first);
  });
  ffi::Array<ffi::Any> result;
  for (const auto& [storage, loops] : ordered) {
    ffi::Map<ffi::String, ffi::Any> item;
    item.Set("storage", ffi::String(StorageName(storage)));
    ffi::Array<ffi::String> names;
    for (const ControlNode* loop : loops) names.push_back(ffi::String(ControlName(loop)));
    item.Set("owners", std::move(names));
    ffi::Array<ffi::String> broadcasts;
    for (const TaskNode* task : CollectRealTasks(scheduled)) {
      if (IsMultiBufferBroadcastFill(task, storage)) {
        broadcasts.push_back(ffi::String(TaskName(task)));
      }
    }
    item.Set("broadcast_tasks", std::move(broadcasts));
    result.push_back(std::move(item));
  }
  return result;
}

ffi::Array<ffi::Any> InspectEpochs(const SBlock& root, ffi::Array<WSRole> roles,
                                   int64_t active_mask, ffi::String target) {
  ICHECK_GE(active_mask, 0);
  ScheduledTIR scheduled = DecodeScheduledTIR(root, ConstrSet());
  CudaRoleTable table{roles, static_cast<RoleMask>(active_mask)};
  AssignRoleCandidates(&scheduled, table, Target(target));
  std::vector<TaskNode*> tasks;
  CollectAllTaskNodes(scheduled.tree, &tasks, true);
  RoleAvailability availability(tasks);
  MultiBufferPlan plan = PrepareBufferEpochs(&scheduled, availability);
  ffi::Array<ffi::Any> result;
  for (const MultiBufferInfo& info : plan.Infos()) result.push_back(PlanSummary(info));
  return result;
}

}  // namespace

bool CanBroadcastFillToStorage(const Stmt& stmt, const Var& storage) {
  class Detector final : public StmtExprVisitor {
   public:
    explicit Detector(Var storage) : storage_(std::move(storage)) {}

    bool found{false};
    bool valid{true};

   private:
    void VisitStmt_(const AttrStmtNode* op) final {
      if (CanRewriteGuardAttribute(op->attr_key, op->node)) {
        VisitExpr(Downcast<PrimExpr>(op->node));
      }
      if (op->value.defined()) VisitExpr(op->value);
      VisitStmt(op->body);
    }

    void VisitStmt_(const BufferStoreNode* op) final {
      if (op->buffer->data.same_as(storage_)) valid = false;
      StmtExprVisitor::VisitStmt_(op);
    }

    void VisitExpr_(const BufferLoadNode* op) final {
      if (op->buffer->data.same_as(storage_)) valid = false;
      StmtExprVisitor::VisitExpr_(op);
    }

    void VisitExpr_(const VarNode* op) final {
      if (ffi::GetRef<Var>(op).same_as(storage_)) valid = false;
    }

    void VisitExpr_(const CallNode* op) final {
      static const auto fill_op = Op::Get("tl.tileop.fill");
      static const Op& region_op = region();
      if (op->op.same_as(fill_op) && !op->args.empty()) {
        if (const auto* region_call = op->args[0].as<CallNode>();
            region_call != nullptr && region_call->op.same_as(region_op)) {
          BufferRegion destination = NormalizeToBufferRegion(ffi::GetRef<Call>(region_call));
          if (destination->buffer->data.same_as(storage_)) {
            found = true;
            for (const Range& range : destination->region) {
              VisitExpr(range->min);
              VisitExpr(range->extent);
            }
            for (size_t i = 1; i < op->args.size(); ++i) VisitExpr(op->args[i]);
            return;
          }
        }
      }
      if (op->op.same_as(region_op)) {
        BufferRegion accessed = NormalizeToBufferRegion(ffi::GetRef<Call>(op));
        if (accessed->buffer->data.same_as(storage_)) valid = false;
      }
      StmtExprVisitor::VisitExpr_(op);
    }

    Var storage_;
  } detector(storage);
  detector(stmt);
  return detector.found && detector.valid;
}

bool IsMultiBufferBroadcastFill(const TaskNode* task, const Var& storage) {
  for (const Var& candidate : BroadcastFills(task)) {
    if (candidate.same_as(storage)) return true;
  }
  return false;
}

MultiBufferOwnerMap InferCompleteOwnerFrontier(ScheduledTIR* scheduled) {
  ICHECK(scheduled != nullptr);
  RefreshCudaTaskAccesses(scheduled);

  MultiBufferOwnerMap explicit_claims = CurrentOwnerClaims(*scheduled);
  VarSet excluded;
  for (const auto& [storage, _] : explicit_claims) excluded.insert(storage);
  VarSet bare_single_version;
  for (const auto& [storage, versions] : scheduled->metadata.requested_versions) {
    // A present mode makes the request an automatic-plan policy, including a
    // one-version ring. A bare request is a manual ring; bare one is an opt-out.
    if (!scheduled->metadata.version_modes.count(storage)) {
      excluded.insert(storage);
      if (versions == 1) bare_single_version.insert(storage);
    }
  }

  std::vector<TaskNode*> tasks;
  CollectAllTaskNodes(scheduled->tree, &tasks, true);
  std::vector<Var> storages;
  VarSet seen;
  auto append = [&](const ffi::Array<BufferRegion>& regions) {
    for (const BufferRegion& region : regions) {
      const Var& storage = region->buffer->data;
      if (IsAutomaticStorage(region->buffer) && !excluded.count(storage) &&
          seen.insert(storage).second) {
        storages.push_back(storage);
      }
    }
  };
  for (const TaskNode* task : tasks) {
    append(task->accesses.reads);
    append(task->accesses.writes);
  }
  std::sort(storages.begin(), storages.end(), [](const Var& a, const Var& b) {
    return StorageName(a) < StorageName(b);
  });

  MultiBufferOwnerMap result;
  for (const auto& [storage, owners] : explicit_claims) {
    if (!bare_single_version.count(storage)) result.emplace(storage, owners);
  }
  for (const Var& storage : storages) {
    CoveragePlan coverage = PlanOwnerList(scheduled->tree, storage);
    if (coverage.IsComplete() && !coverage.owners.empty()) {
      result.emplace(storage, std::move(coverage.owners));
    }
  }

  for (auto& [storage, owners] : result) {
    std::stable_sort(owners.begin(), owners.end(), [](const ControlNode* a, const ControlNode* b) {
      return CompareIRStructure(a, b) < 0;
    });
  }

  // Serialize both explicit and inferred claims. Bare one-version opt-outs
  // remove claims for that storage across the entire kernel.
  ControlClaims by_control;
  for (const auto& [storage, owners] : result) {
    for (ControlNode* owner : owners) by_control[owner].push_back(storage);
  }
  std::vector<ControlNode*> controls;
  CollectControls(scheduled->tree, &controls);
  for (ControlNode* control : controls) {
    auto& claims = by_control[control];
    std::sort(claims.begin(), claims.end(), [](const Var& a, const Var& b) {
      return StorageName(a) < StorageName(b);
    });
    claims.erase(std::unique(claims.begin(), claims.end(),
                             [](const Var& a, const Var& b) { return a.same_as(b); }),
                 claims.end());
    For loop = control->control;
    if (claims.empty() && !loop->annotations.count(kMultiBufferEligible)) continue;
    ffi::Array<Var> annotation(claims.begin(), claims.end());
    loop.CopyOnWrite()->annotations.Set(kMultiBufferEligible, std::move(annotation));
    control->control = std::move(loop);
  }

  for (const auto& [storage, owners] : result) {
    ValidateStorageCoverage(scheduled, storage, owners);
  }
  return result;
}

MultiBufferPlan PrepareBufferEpochs(ScheduledTIR* scheduled,
                                    const RoleAvailability& availability) {
  ICHECK(scheduled != nullptr);
  return BufferEpochBuilder(scheduled, availability).Build();
}

void ValidateMultiBufferPlan(const ScheduledTIR& scheduled, const MultiBufferPlan& plan) {
  // Coverage validation also recognizes and marks owner-external fills. Keep a
  // mutable tree reference for that metadata-only repair; no schedule geometry
  // is changed by validation.
  auto* mutable_scheduled = const_cast<ScheduledTIR*>(&scheduled);
  struct Group {
    Buffer counter;
    const MultiBufferInfo* representative;
  };
  std::unordered_map<int, Group> groups;
  for (const MultiBufferInfo& info : plan.Infos()) {
    ICHECK(info.storage.defined() && info.num_versions > 0);
    ICHECK(!info.owners.empty()) << "CUDA Z3 storage " << StorageName(info.storage)
                                 << " has an empty owner frontier";
    ICHECK_EQ(info.owner_stages.size(), info.owners.size())
        << "CUDA Z3 storage " << StorageName(info.storage)
        << " must have one recorded stage per owner";
    ValidateStorageCoverage(mutable_scheduled, info.storage,
                            [&] {
                              std::vector<ControlNode*> owners;
                              for (const auto& owner : info.owners) owners.push_back(owner.loop);
                              return owners;
                            }());
    if (!info.UsesCounter()) {
      ICHECK_EQ(info.counter_group_id, 0);
      ICHECK(!info.counter.defined());
      ICHECK_EQ(info.owners.size(), 1U);
      ICHECK(info.owners.front().active_guard.defined() &&
             is_one(info.owners.front().active_guard))
          << "Iteration storage " << StorageName(info.storage)
          << " must have one unconditional lexical owner";
      continue;
    }
    ICHECK_GT(info.counter_group_id, 0);
    ICHECK(info.counter.defined());
    auto [group, inserted] = groups.emplace(info.counter_group_id, Group{info.counter, &info});
    if (!inserted) {
      ICHECK(group->second.counter.same_as(info.counter))
          << "Counter group " << info.counter_group_id << " has inconsistent physical counters";
      ICHECK_EQ(group->second.representative->owners.size(), info.owners.size());
    }
    const MultiBufferInfo& representative = *group->second.representative;
    ICHECK(representative.owner_stages == info.owner_stages)
        << "Counter group " << info.counter_group_id << " has incompatible owner stages";
    for (size_t i = 0; i < info.owners.size(); ++i) {
      ICHECK(info.owners[i].loop != nullptr && info.owners[i].active_guard.defined());
      ICHECK_GE(info.owner_stages[i], 0)
          << "Counter storage " << StorageName(info.storage) << " has an unscheduled owner";
      ICHECK(representative.owners[i].loop == info.owners[i].loop)
          << "Counter group " << info.counter_group_id << " has inconsistent ordered owners";
      ICHECK(GuardsEquivalent(representative.owners[i].active_guard,
                              info.owners[i].active_guard,
                              info.owners[i].loop->GetLoopBodyContext()))
          << "Counter group " << info.counter_group_id << " has incompatible active guards";
    }
  }
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.cuda.transform.Z3BufferOwnership", InspectOwnership);
  refl::GlobalDef().def("tl.cuda.transform.Z3BufferEpochs", InspectEpochs);
}

}  // namespace tvm::tl::cuda::z3
