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
 * \file z3_schedule_units.cc
 * \brief Materialize and validate recursive CUDA Z3 schedule units.
 */

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>
#include <tvm/target/target.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "./z3_schedule_ir.h"
#include "cuda/transform/ws_analysis.h"
#include "op/builtin.h"
#include "op/copy.h"
#include "op/gemm.h"
#include "op/operator.h"
#include "op/parallel.h"
#include "op/utils.h"
#include "transform/common/attr.h"
#include "transform/common/constr_visitor.h"
#include "transform/common/pipeline_utils.h"

namespace tvm::tl::cuda {

using namespace tirx;

namespace {

using Metadata = ffi::Map<ffi::String, ffi::Any>;

constexpr uint32_t kWorkerFamily = 1U << 0;
constexpr uint32_t kLoadFamily = 1U << 1;
constexpr uint32_t kStoreFamily = 1U << 2;
constexpr uint32_t kMmaFamily = 1U << 3;
constexpr const char* kZ3CudaTaskAttr = "tl.cuda_z3_task";
constexpr const char* kZ3CudaStageAttr = "tl.cuda_z3_stage";

Stmt MakeSequence(const std::vector<Stmt>& statements) {
  ICHECK(!statements.empty());
  if (statements.size() == 1) {
    return statements[0];
  }
  ffi::Array<Stmt> sequence;
  sequence.reserve(statements.size());
  for (const Stmt& stmt : statements) {
    sequence.push_back(stmt);
  }
  return SeqStmt::Flatten(sequence);
}

bool IsRecursiveTaskMarker(const ffi::String& key) {
  return key == z3::kCudaTask || key == z3::kCudaPerRoleTask;
}

bool IsAnyTaskOrStageMarker(const ffi::String& key) {
  return IsRecursiveTaskMarker(key) || key == z3::kCudaStage ||
         key == kZ3CudaTaskAttr || key == kZ3CudaStageAttr;
}

bool HasBufferRead(const PrimExpr& expr) {
  bool found = false;
  PostOrderVisit(Evaluate(expr), [&](const ffi::ObjectRef& node) {
    found = found || node.as<BufferLoadNode>() != nullptr ||
            node.as<ProducerLoadNode>() != nullptr;
  });
  return found;
}

bool HasExecutableStatement(const Stmt& stmt) {
  if (IsPipelineDeclarationStmt(stmt)) {
    return false;
  }
  if (const auto* evaluate = stmt.as<EvaluateNode>()) {
    return !is_zero(evaluate->value);
  }
  if (const auto* sequence = stmt.as<SeqStmtNode>()) {
    return std::any_of(sequence->seq.begin(), sequence->seq.end(),
                       [](const Stmt& child) { return HasExecutableStatement(child); });
  }
  if (const auto* attribute = stmt.as<AttrStmtNode>()) {
    return HasExecutableStatement(attribute->body);
  }
  return true;
}

bool IsLoopBreakControl(const Stmt& stmt) {
  const auto* evaluate = stmt.as<EvaluateNode>();
  if (evaluate == nullptr) {
    return false;
  }
  const auto* call = evaluate->value.as<CallNode>();
  return call != nullptr && call->op.same_as(tl::loop_break());
}

class TaskControlValidator final : public StmtExprVisitor {
 public:
  explicit TaskControlValidator(bool inside_per_role) : inside_per_role_(inside_per_role) {}

 private:
  void VisitStmt_(const AttrStmtNode* op) final {
    ICHECK(!IsAnyTaskOrStageMarker(op->attr_key) && op->attr_key != z3::kScheduleUnit)
        << "Nested Task/PerRoleTask/Stage markers are not supported inside an atomic Task; "
        << "place Stage outside the complete task and PerRoleTask outside its candidates";
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const IfThenElseNode* op) final {
    ICHECK(!inside_per_role_)
        << "Control flow selecting a PerRole candidate must remain outside the candidate Task";
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const CallNode* op) final {
    if (auto tile_op = ParseOperator(ffi::GetRef<Call>(op)); tile_op.defined()) {
      if (const auto* parallel = tile_op.as<ParallelOpNode>()) {
        this->VisitStmt(parallel->GetRoot());
      }
      return;
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  bool inside_per_role_{false};
};

struct TaskMetadata {
  std::optional<int64_t> latency;
  std::optional<int64_t> ii;
  bool declaration{false};
};

TaskMetadata ValidateTaskMetadata(const AttrStmtNode* op) {
  ICHECK(is_one(op->value)) << "CUDA task marker value must be 1";
  Metadata metadata;
  const auto* map = op->node.as<ffi::MapObj>();
  ICHECK(map != nullptr) << "CUDA task marker requires a map payload";
  for (const auto& [key, value] : *map) {
    metadata.Set(key.cast<ffi::String>(), value);
  }

  TaskMetadata result;
  if (auto declaration = metadata.Get("declaration"); declaration.has_value()) {
    const auto* value = declaration.value().as<IntImmNode>();
    ICHECK(value != nullptr && value->value == 1)
        << "Internal declaration metadata must be integer 1";
    result.declaration = true;
  }
  for (const auto& [key, value] : metadata) {
    if (key == "latency" || key == "ii") {
      ICHECK(!result.declaration) << "Declarations never carry solver costs";
      const auto* integer = value.as<IntImmNode>();
      ICHECK(integer != nullptr) << "T.Task " << key << " must be a constant integer";
      if (key == "latency") {
        ICHECK_GE(integer->value, 0) << "T.Task latency must be non-negative";
        result.latency = integer->value;
      } else {
        ICHECK_GT(integer->value, 0) << "T.Task ii must be positive";
        result.ii = integer->value;
      }
      continue;
    }
    if (result.declaration && key == "solver_eligible") {
      const auto* integer = value.as<IntImmNode>();
      ICHECK(integer != nullptr && integer->value == 0)
          << "Declarations must have solver_eligible=0";
      continue;
    }
    ICHECK(key == "declaration")
        << "Public T.Task metadata only supports latency and ii; role, solver, and "
           "declaration fields are compiler-owned. Unsupported key: "
        << key;
  }
  if (result.latency.has_value() && result.ii.has_value()) {
    ICHECK_GE(result.latency.value(), result.ii.value())
        << "T.Task latency must be greater than or equal to ii";
  }
  return result;
}

struct AccessSignature {
  std::vector<std::pair<size_t, size_t>> tile_regions;
  size_t scalar_reads{0};
  size_t scalar_writes{0};
  size_t async_arrivals{0};
  size_t async_waits{0};

  bool operator==(const AccessSignature& other) const {
    return tile_regions == other.tile_regions && scalar_reads == other.scalar_reads &&
           scalar_writes == other.scalar_writes && async_arrivals == other.async_arrivals &&
           async_waits == other.async_waits;
  }
};

struct Capability {
  uint32_t family{0};
  AccessSignature access;
  bool producer{false};
};

class CapabilityAnalyzer final : public StmtExprVisitor {
 public:
  static Capability Analyze(const Stmt& stmt, const ffi::Optional<Target>& target) {
    CapabilityAnalyzer analyzer(target);
    analyzer(stmt);
    std::sort(analyzer.result_.access.tile_regions.begin(),
              analyzer.result_.access.tile_regions.end());
    size_t tile_reads = 0;
    size_t tile_writes = 0;
    for (const auto& [read, write] : analyzer.result_.access.tile_regions) {
      tile_reads += read;
      tile_writes += write;
    }
    analyzer.result_.producer =
        (analyzer.result_.family == kLoadFamily) ||
        (analyzer.result_.family == kWorkerFamily &&
         analyzer.result_.access.scalar_writes + tile_writes > 0 &&
         analyzer.result_.access.scalar_reads + tile_reads == 0);
    return std::move(analyzer.result_);
  }

 private:
  explicit CapabilityAnalyzer(const ffi::Optional<Target>& target) : target_(target) {}

  static std::string OpName(const CallNode* op) {
    const auto* op_node = op->op.as<OpNode>();
    return op_node == nullptr ? std::string() : std::string(op_node->name);
  }

  static bool IsRawAsync(const std::string& name) {
    return name.find("cp_async") != std::string::npos ||
           name.find("ptx_commit_group") != std::string::npos ||
           name.find("ptx_wait_group") != std::string::npos;
  }

  void AddFamily(uint32_t family) { result_.family |= family; }

  void AddTileAccess(const TileOperator& tile_op) {
    AccessRegions access = tile_op->GetAccessRegions();
    result_.access.tile_regions.emplace_back(access.reads.size(), access.writes.size());
  }

  uint32_t FallbackFamily(const TileOperator& tile_op) {
    if (const auto* copy = tile_op.as<CopyNode>()) {
      bool src_global = IsGlobalBuffer(copy->src);
      bool dst_global = IsGlobalBuffer(copy->dst);
      if (src_global && !dst_global) {
        return kLoadFamily;
      }
      if (!src_global && dst_global) {
        return kStoreFamily;
      }
      return kWorkerFamily;
    }
    if (auto gemm = tl::GetGemmInfo(tile_op); gemm.has_value()) {
      return IsTmemBuffer(gemm->accumulator) ? kMmaFamily : kWorkerFamily;
    }
    if (tile_op.as<Im2ColOpNode>()) {
      return kLoadFamily;
    }
    return kWorkerFamily;
  }

  uint32_t ClassifyTileOp(const CallNode* call, const TileOperator& tile_op) {
    if (!target_.defined()) {
      return FallbackFamily(tile_op);
    }
    switch (ClassifyStmt(Evaluate(ffi::GetRef<Call>(call)), target_.value())) {
      case TileStmtKind::kTmaProducer:
      case TileStmtKind::kCpAsyncProducer:
      case TileStmtKind::kCpAsyncRaw:
        return kLoadFamily;
      case TileStmtKind::kTmaStore:
        return kStoreFamily;
      case TileStmtKind::kTcgen05Mma:
        return kMmaFamily;
      case TileStmtKind::kSimtProducer:
      case TileStmtKind::kConsumer:
        return kWorkerFamily;
    }
    LOG(FATAL) << "Unknown CUDA tile statement kind";
    return kWorkerFamily;
  }

  void VisitStmt_(const BufferStoreNode* op) final {
    AddFamily(kWorkerFamily);
    ++result_.access.scalar_writes;
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const BindNode* op) final {
    // Scalar Binds are replayable/control endpoints. Their reads remain in the
    // surrounding task's access footprint, but they do not select a data role.
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const EvaluateNode* op) final {
    const auto* call = op->value.as<CallNode>();
    if (!is_zero(op->value) && call != nullptr && !call->op.same_as(tl::loop_break()) &&
        !ParseOperator(ffi::GetRef<Call>(call)).defined() &&
        SideEffect(op->value) > CallEffectKind::kPure) {
      AddFamily(kWorkerFamily);
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const SBlockNode*) final { AddFamily(kWorkerFamily); }

  void VisitExpr_(const BufferLoadNode* op) final {
    ++result_.access.scalar_reads;
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const CallNode* op) final {
    if (auto tile_op = ParseOperator(ffi::GetRef<Call>(op)); tile_op.defined()) {
      AddFamily(ClassifyTileOp(op, tile_op));
      AddTileAccess(tile_op);
      return;
    }
    std::string name = OpName(op);
    if (IsRawAsync(name)) {
      AddFamily(kLoadFamily);
      if (name.find("wait") != std::string::npos) {
        ++result_.access.async_waits;
      } else {
        ++result_.access.async_arrivals;
      }
    } else if (name.find("mbarrier") != std::string::npos ||
               name.find("barrier_arrive") != std::string::npos) {
      if (name.find("wait") != std::string::npos) {
        ++result_.access.async_waits;
      } else {
        ++result_.access.async_arrivals;
      }
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  ffi::Optional<Target> target_;
  Capability result_;
};

class OpaqueSideEffectValidator final : public StmtExprVisitor {
 private:
  void VisitExpr_(const CallNode* op) final {
    if (auto tile_op = ParseOperator(ffi::GetRef<Call>(op)); tile_op.defined()) {
      return;
    }
    const auto* op_node = op->op.as<OpNode>();
    std::string name = op_node == nullptr ? std::string() : std::string(op_node->name);
    bool known_tilelang_contract = name.rfind("tl.", 0) == 0;
    ICHECK(known_tilelang_contract || SideEffect(ffi::GetRef<Call>(op)) == CallEffectKind::kPure)
        << "PerRoleTask candidate contains an unresolved opaque side effect from `" << name
        << "`; use an operation with an explicit CUDA access/arrival contract";
    StmtExprVisitor::VisitExpr_(op);
  }
};

Capability ValidateTask(const AttrStmtNode* op, bool inside_per_role,
                        const ffi::Optional<Target>& target) {
  ICHECK(op->attr_key == z3::kCudaTask);
  TaskMetadata metadata = ValidateTaskMetadata(op);
  if (metadata.declaration) {
    ICHECK(!inside_per_role) << "Declarations are not PerRole candidates";
    ICHECK(op->body.as<AllocBufferNode>() || op->body.as<DeclBufferNode>())
        << "A declaration task must contain AllocBuffer or DeclBuffer";
    return {};
  }
  ICHECK(HasExecutableStatement(op->body))
      << "T.Task must contain an executable statement, not only declarations or no-ops";
  TaskControlValidator control_validator(inside_per_role);
  control_validator(op->body);
  Capability capability = CapabilityAnalyzer::Analyze(ffi::GetRef<Stmt>(op), target);
  ICHECK(capability.family == 0 || (capability.family & (capability.family - 1)) == 0)
      << "T.Task requires one CUDA role-capability family (Worker, Load, Store, or MMA); "
         "split this compound cross-role body into separate tasks. Body="
      << ffi::GetRef<Stmt>(op);
  if (inside_per_role) {
    OpaqueSideEffectValidator side_effect_validator;
    side_effect_validator(op->body);
    ICHECK(capability.family != 0 || op->body.as<BindNode>() || IsLoopBreakControl(op->body) ||
           (op->body.as<EvaluateNode>() && is_zero(op->body.as<EvaluateNode>()->value)))
        << "A PerRole candidate must have one inferable role-capability family";
  }
  return capability;
}

bool IsStaticControlExpression(const PrimExpr& expr) {
  return !HasBufferRead(expr) && SideEffect(expr) == CallEffectKind::kPure;
}

PrimExpr Conjunction(const std::vector<PrimExpr>& expressions) {
  PrimExpr result = Bool(true);
  for (const PrimExpr& expr : expressions) {
    result = is_one(result) ? expr : result && expr;
  }
  return result;
}

struct PerRoleCandidate {
  Capability capability;
  PrimExpr guard;
  bool static_control{true};
};

class PerRoleCandidateCollector final : public StmtExprVisitor {
 public:
  PerRoleCandidateCollector(const ffi::Optional<Target>& target, const ConstrSet& outer_ctx,
                            std::vector<PerRoleCandidate>* candidates)
      : target_(target), candidates_(candidates) {
    for (const Constr& constraint : outer_ctx.constrs_) {
      if (constraint.kind == Constr::kBindValue && IsDynamicExpression(constraint.value)) {
        dynamic_bindings_.push_back(constraint.var);
      }
    }
  }

 private:
  bool UsesDynamicBinding(const PrimExpr& expr) const {
    for (const Var& var : dynamic_bindings_) {
      if (tirx::UsesVar(expr, [&](const VarNode* node) { return node == var.get(); })) {
        return true;
      }
    }
    return false;
  }

  bool IsDynamicExpression(const PrimExpr& expr) const {
    return !IsStaticControlExpression(expr) || UsesDynamicBinding(expr);
  }

  void VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == z3::kCudaTask) {
      Capability capability = ValidateTask(op, /*inside_per_role=*/true, target_);
      if (capability.family == 0) {
        // Dependency-free scalar Bind, loop_break, and no-op tasks are group
        // control, not alternative data-path candidates. Preserve mutable-read
        // dependencies through chains of condition snapshots.
        if (const auto* bind = op->body.as<BindNode>();
            bind != nullptr && IsDynamicExpression(bind->value)) {
          dynamic_bindings_.push_back(bind->var);
        }
        return;
      }
      PrimExpr guard = Conjunction(guards_);
      bool static_control = static_control_ && IsStaticControlExpression(guard) &&
                            !UsesDynamicBinding(guard);
      candidates_->push_back({std::move(capability), guard, static_control});
      return;
    }
    ICHECK(op->attr_key != z3::kCudaPerRoleTask) << "Nested T.PerRoleTask regions are not supported";
    ICHECK(op->attr_key != z3::kCudaStage && op->attr_key != kZ3CudaStageAttr)
        << "T.Stage must be outside T.PerRoleTask";
    ICHECK(op->attr_key != z3::kScheduleUnit)
        << "T.PerRoleTask contains an already-materialized schedule unit";
    ICHECK(op->attr_key != kZ3CudaTaskAttr)
        << "Flat tl.cuda_z3_task markers cannot be mixed into recursive PerRoleTask";
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const IfThenElseNode* op) final {
    guards_.push_back(op->condition);
    this->VisitStmt(op->then_case);
    guards_.back() = Not(op->condition);
    if (op->else_case.defined()) {
      this->VisitStmt(op->else_case.value());
    }
    guards_.pop_back();
  }

  void VisitStmt_(const ForNode* op) final {
    ICHECK(op->kind == ForKind::kSerial || op->kind == ForKind::kUnrolled)
        << "Unmarked parallel loop inside PerRoleTask was not normalized into a candidate";
    const auto* extent = op->extent.as<IntImmNode>();
    if (extent != nullptr) {
      ICHECK(extent->value == 1)
          << "A PerRole candidate under a constant loop extent other than one has a provable "
             "zero/multiple-execution contract";
    } else {
      static_control_ = false;
    }
    this->VisitStmt(op->body);
  }

  void VisitStmt_(const WhileNode*) final {
    LOG(FATAL) << "PerRoleTask contains a bare While; run cuda.NormalizeZ3ControlFlow first";
  }

  ffi::Optional<Target> target_;
  std::vector<PerRoleCandidate>* candidates_;
  std::vector<PrimExpr> guards_;
  std::vector<Var> dynamic_bindings_;
  bool static_control_{true};
};

bool Prove(arith::Analyzer* analyzer, const PrimExpr& expression) {
  return analyzer->CanProve(expression) || analyzer->z3_prover.CanProve(expression);
}

void ValidatePerRoleTask(const AttrStmtNode* op, const ConstrSet& outer_ctx,
                         const ffi::Optional<Target>& target) {
  ICHECK(op->attr_key == z3::kCudaPerRoleTask);
  ICHECK(is_one(op->value)) << "T.PerRoleTask marker value must be 1";
  if (const auto* metadata = op->node.as<ffi::MapObj>()) {
    ICHECK_EQ(metadata->size(), 0U) << "T.PerRoleTask does not accept public metadata";
  } else {
    PrimExpr legacy_payload = op->node.cast<PrimExpr>();
    ICHECK(is_zero(legacy_payload))
        << "T.PerRoleTask requires an empty map or the public scalar-zero payload";
  }
  std::vector<PerRoleCandidate> candidates;
  PerRoleCandidateCollector(target, outer_ctx, &candidates)(op->body);
  ICHECK(!candidates.empty()) << "T.PerRoleTask must contain at least one role-bearing candidate";

  const Capability& expected = candidates.front().capability;
  for (const PerRoleCandidate& candidate : candidates) {
    ICHECK((candidate.capability.family & (candidate.capability.family - 1)) == 0)
        << "Every PerRole candidate must use exactly one role-capability family";
    ICHECK(candidate.capability.family == expected.family)
        << "PerRole candidates must use the same CUDA role-capability family";
    ICHECK(candidate.capability.producer == expected.producer)
        << "PerRole candidates must have the same logical producer/consumer direction";
    ICHECK(candidate.capability.access == expected.access)
        << "PerRole candidates must have compatible access and async arrival signatures";
  }

  arith::Analyzer analyzer;
  outer_ctx.Populate(analyzer);
  for (const PerRoleCandidate& candidate : candidates) {
    ICHECK(!Prove(&analyzer, Not(candidate.guard)))
        << "PerRole candidate is provably unreachable under its outer context";
  }

  bool static_exactly_once = std::all_of(
      candidates.begin(), candidates.end(),
      [](const PerRoleCandidate& candidate) { return candidate.static_control; });
  if (!static_exactly_once) {
    // Dynamic exactly-once remains the explicit user contract. Device-side
    // arrival checking is emitted by the WS/runtime slices, not simulated here.
    return;
  }
  for (size_t i = 0; i < candidates.size(); ++i) {
    for (size_t j = i + 1; j < candidates.size(); ++j) {
      ICHECK(Prove(&analyzer, Not(candidates[i].guard && candidates[j].guard)))
          << "Static PerRole candidate guards must be pairwise exclusive; candidates " << i
          << " and " << j << " may execute together";
    }
  }
  PrimExpr any_candidate = Bool(false);
  for (const PerRoleCandidate& candidate : candidates) {
    any_candidate = any_candidate || candidate.guard;
  }
  ICHECK(Prove(&analyzer, any_candidate))
      << "Static PerRole candidate guards must be collectively exhaustive; a zero-execution "
         "path remains";
}

bool IsTaskBoundary(const Stmt& stmt) {
  if (const auto* loop = stmt.as<ForNode>()) {
    return loop->kind != ForKind::kSerial && loop->kind != ForKind::kUnrolled;
  }
  return stmt.as<EvaluateNode>() || stmt.as<BufferStoreNode>() || stmt.as<BindNode>() ||
         stmt.as<SBlockNode>() || stmt.as<SBlockRealizeNode>() || stmt.as<AssertStmtNode>();
}

AttrStmt MakeTask(Stmt body, bool declaration = false) {
  Metadata metadata;
  if (declaration) {
    metadata.Set("declaration", Integer(1));
    metadata.Set("solver_eligible", Integer(0));
  }
  Span span = body->span;
  return AttrStmt(metadata, z3::kCudaTask, Integer(1), std::move(body), span);
}

class TaskNormalizer final : public StmtMutator {
 public:
  static Stmt Rewrite(const Stmt& body) { return TaskNormalizer()(body); }

 private:
  Stmt VisitStmt(const Stmt& stmt) final {
    if (const auto* attribute = stmt.as<AttrStmtNode>()) {
      if (attribute->attr_key == z3::kCudaStage) {
        ICHECK_EQ(task_depth_, 0) << "T.Stage cannot appear inside T.Task";
        ICHECK_EQ(per_role_depth_, 0) << "T.Stage cannot appear inside T.PerRoleTask";
        ICHECK_EQ(stage_depth_, 0) << "Nested T.Stage scopes are not supported";
        ++stage_depth_;
        Stmt result = StmtMutator::VisitStmt(stmt);
        --stage_depth_;
        return result;
      }
      if (attribute->attr_key == z3::kCudaTask) {
        ICHECK_EQ(task_depth_, 0) << "Nested T.Task regions are not supported";
        ++task_depth_;
        Stmt result = StmtMutator::VisitStmt(stmt);
        --task_depth_;
        return result;
      }
      if (attribute->attr_key == z3::kCudaPerRoleTask) {
        ICHECK_EQ(task_depth_, 0) << "T.PerRoleTask cannot be nested inside T.Task";
        ICHECK_EQ(per_role_depth_, 0) << "Nested T.PerRoleTask regions are not supported";
        ++per_role_depth_;
        Stmt result = StmtMutator::VisitStmt(stmt);
        --per_role_depth_;
        return result;
      }
    }

    if (task_depth_ > 0) {
      return StmtMutator::VisitStmt(stmt);
    }
    if (const auto* evaluate = stmt.as<EvaluateNode>(); evaluate && is_zero(evaluate->value)) {
      return stmt;
    }
    if (IsPipelineDeclarationStmt(stmt)) {
      return MakeTask(stmt, /*declaration=*/true);
    }
    if (IsTaskBoundary(stmt)) {
      return MakeTask(stmt);
    }
    if (const auto* loop = stmt.as<ForNode>();
        loop != nullptr && task_depth_ == 0 &&
        (loop->kind == ForKind::kSerial || loop->kind == ForKind::kUnrolled)) {
      // A Control owns a new child list. Stage scopes may nest across that
      // boundary and select manual mode independently in each list.
      int outer_stage_depth = stage_depth_;
      stage_depth_ = 0;
      Stmt result = StmtMutator::VisitStmt(stmt);
      stage_depth_ = outer_stage_depth;
      return result;
    }
    return StmtMutator::VisitStmt(stmt);
  }

  int task_depth_{0};
  int per_role_depth_{0};
  int stage_depth_{0};
};

class SemanticTaskValidator final : public ConstrVisitor {
 public:
  SemanticTaskValidator(ConstrSet outer_ctx, ffi::Optional<Target> target)
      : target_(std::move(target)) {
    constr_stack_ = std::move(outer_ctx.constrs_);
  }

 private:
  static ffi::Optional<Bind> GetFlatBind(const Stmt& stmt) {
    const auto* marker = stmt.as<AttrStmtNode>();
    if (marker == nullptr || marker->attr_key != z3::kCudaTask) {
      return std::nullopt;
    }
    if (const auto* bind = marker->body.as<BindNode>()) {
      return ffi::GetRef<Bind>(bind);
    }
    return std::nullopt;
  }

  void VisitStmt_(const SeqStmtNode* op) final {
    size_t old_size = constr_stack_.size();
    for (const Stmt& child : op->seq) {
      this->VisitStmt(child);
      if (auto bind = GetFlatBind(child); bind.defined()) {
        constr_stack_.emplace_back(bind.value()->var, bind.value()->value);
      }
    }
    constr_stack_.resize(old_size);
  }

  void VisitStmt_(const AttrStmtNode* op) final {
    ICHECK(op->attr_key != kZ3CudaTaskAttr && op->attr_key != kZ3CudaStageAttr)
        << "The recursive CUDA materializer encountered flat schema marker `" << op->attr_key
        << "`. Use T.Task/T.Stage(schema=\"recursive\") consistently; the flat "
           "NormalizeZ3ScheduleUnits pass owns tl.cuda_z3_* markers.";
    ICHECK(op->attr_key != z3::kScheduleUnit)
        << "cuda.MaterializeZ3ScheduleUnits cannot run on an already-materialized schedule";
    if (op->attr_key == z3::kCudaTask) {
      ValidateTask(op, /*inside_per_role=*/false, target_);
      return;
    }
    if (op->attr_key == z3::kCudaPerRoleTask) {
      ValidatePerRoleTask(op, GetConstrSet(), target_);
      return;
    }
    ConstrVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const WhileNode*) final {
    LOG(FATAL) << "cuda.MaterializeZ3ScheduleUnits found a bare While; run "
                  "cuda.NormalizeZ3ControlFlow first";
  }

  ffi::Optional<Target> target_;
};

class ScheduleUnitMaterializer {
 public:
  Stmt Rewrite(const Stmt& body) { return MakeSequence(MaterializeChildList(body, "root")); }

 private:
  AttrStmt RequireUnit(const Stmt& stmt, const std::string& path) const {
    const auto* unit = stmt.as<AttrStmtNode>();
    ICHECK(unit != nullptr && unit->attr_key == z3::kScheduleUnit)
        << "Expected schedule unit at " << path << ", got " << stmt;
    return ffi::GetRef<AttrStmt>(unit);
  }

  bool IsDeclaration(const Stmt& stmt, const std::string& path) const {
    AttrStmt unit = RequireUnit(stmt, path);
    Stmt body = unit->body;
    while (const auto* wrapper = body.as<AttrStmtNode>()) {
      if (wrapper->attr_key != z3::kWSOpId) {
        break;
      }
      body = wrapper->body;
    }
    const auto* marker = body.as<AttrStmtNode>();
    if (marker == nullptr || marker->attr_key != z3::kCudaTask) {
      return false;
    }
    const auto* map = marker->node.as<ffi::MapObj>();
    if (map == nullptr) {
      return false;
    }
    for (const auto& [key, value] : *map) {
      if (key.cast<ffi::String>() == "declaration") {
        return true;
      }
    }
    return false;
  }

  AttrStmt MakeInitialUnit(Stmt body) {
    Metadata metadata;
    metadata.Set("stage", Integer(z3::kUnscheduledStage));
    metadata.Set("source_id", Integer(next_source_id_++));
    Span span = body->span;
    return AttrStmt(metadata, z3::kScheduleUnit, Integer(1), std::move(body), span);
  }

  Stmt RebuildBody(const AttrStmt& unit, Stmt body) {
    return AttrStmt(unit->node, unit->attr_key, unit->value, std::move(body), unit->span);
  }

  Stmt AssignStage(const Stmt& stmt, int stage, const std::string& path) {
    if (IsDeclaration(stmt, path)) {
      return stmt;
    }
    AttrStmt unit = RequireUnit(stmt, path);
    Metadata metadata = unit->node.cast<Metadata>();
    auto current = metadata.Get("stage");
    ICHECK(current.has_value() && current.value().cast<IntImm>()->value == z3::kUnscheduledStage)
        << "Nested T.Stage scopes are not supported at " << path;
    metadata.Set("stage", Integer(stage));
    return AttrStmt(metadata, unit->attr_key, unit->value, unit->body, unit->span);
  }

  Stmt PrependCondition(const Stmt& stmt, const PrimExpr& condition, const std::string& path) {
    ICHECK(!IsDeclaration(stmt, path))
        << "A buffer declaration cannot be guarded by conditional control at " << path;
    AttrStmt unit = RequireUnit(stmt, path);
    return RebuildBody(unit, IfThenElse(condition, unit->body));
  }

  Stmt PrependAttribute(const Stmt& stmt, const AttrStmtNode* attribute,
                        const std::string& path) {
    ICHECK(!IsDeclaration(stmt, path))
        << "A buffer declaration cannot be guarded by an attribute at " << path;
    AttrStmt unit = RequireUnit(stmt, path);
    return RebuildBody(unit,
                       AttrStmt(attribute->node, attribute->attr_key, attribute->value, unit->body,
                                attribute->span));
  }

  int64_t SourceId(const AttrStmt& unit) const {
    auto value = unit->node.cast<Metadata>().Get("source_id");
    ICHECK(value.has_value());
    return value.value().cast<IntImm>()->value;
  }

  ffi::String RequireOpId(const ffi::Any& value) const {
    if (auto string = value.try_cast<ffi::String>(); string.has_value()) {
      ICHECK(!string.value().empty()) << "tl.ws_op_id must be non-empty";
      return string.value();
    }
    const auto* string = value.as<StringImmNode>();
    ICHECK(string != nullptr && !string->value.empty()) << "tl.ws_op_id must be a non-empty string";
    return string->value;
  }

  void RegisterIdentity(const AttrStmt& unit, const ffi::String& id, const std::string& path) {
    int64_t source_id = SourceId(unit);
    auto [it, inserted] = identities_.emplace(std::string(id), source_id);
    ICHECK(inserted || it->second == source_id)
        << "Duplicate tl.ws_op_id `" << id << "` on different schedule units at " << path;
  }

  std::vector<Stmt> MaterializeChildList(const Stmt& stmt, const std::string& path) {
    std::vector<Stmt> units = MaterializeList(stmt, path);
    if (units.empty()) {
      units.push_back(MakeInitialUnit(MakeTask(Evaluate(0))));
    }
    NormalizeStages(&units, path);
    return units;
  }

  std::vector<Stmt> MaterializeList(const Stmt& stmt, const std::string& path) {
    if (!stmt.defined()) {
      return {};
    }
    if (const auto* sequence = stmt.as<SeqStmtNode>()) {
      std::vector<Stmt> result;
      for (size_t i = 0; i < sequence->seq.size(); ++i) {
        auto nested = MaterializeList(sequence->seq[i], path + "/" + std::to_string(i));
        result.insert(result.end(), nested.begin(), nested.end());
      }
      return result;
    }
    if (const auto* loop = stmt.as<ForNode>()) {
      return MaterializeLoop(loop, path);
    }
    if (const auto* condition = stmt.as<IfThenElseNode>()) {
      return MaterializeCondition(condition, path);
    }
    if (const auto* attribute = stmt.as<AttrStmtNode>()) {
      return MaterializeAttribute(attribute, path);
    }
    if (const auto* evaluate = stmt.as<EvaluateNode>(); evaluate && is_zero(evaluate->value)) {
      return {};
    }
    LOG(FATAL) << "Unmarked non-declaration statement at " << path
               << " after internal task normalization: " << stmt;
    return {};
  }

  std::vector<Stmt> MaterializeLoop(const ForNode* op, const std::string& path) {
    ICHECK(op->kind == ForKind::kSerial || op->kind == ForKind::kUnrolled)
        << "An opaque parallel loop at " << path << " was not wrapped in a Task";
    std::vector<Stmt> body = MaterializeChildList(op->body, path + "/body");
    For loop = ffi::GetRef<For>(op);
    loop.CopyOnWrite()->body = MakeSequence(body);
    AttrStmt unit = MakeInitialUnit(std::move(loop));
    if (auto id = op->annotations.Get(z3::kWSOpId); id.has_value()) {
      RegisterIdentity(unit, RequireOpId(id.value()), path);
    }
    return {unit};
  }

  std::vector<Stmt> MaterializeCondition(const IfThenElseNode* op, const std::string& path) {
    std::vector<Stmt> result = MaterializeList(op->then_case, path + "/then");
    for (Stmt& unit : result) {
      unit = PrependCondition(unit, op->condition, path + "/then");
    }
    if (!op->else_case.defined()) {
      return result;
    }
    std::vector<Stmt> alternatives = MaterializeList(op->else_case.value(), path + "/else");
    for (Stmt& unit : alternatives) {
      unit = PrependCondition(unit, Not(op->condition), path + "/else");
    }
    result.insert(result.end(), alternatives.begin(), alternatives.end());
    return result;
  }

  std::vector<Stmt> MaterializeAttribute(const AttrStmtNode* op, const std::string& path) {
    ICHECK(op->attr_key != z3::kScheduleUnit)
        << "Already-materialized schedule unit at " << path;
    if (op->attr_key == z3::kCudaTask || op->attr_key == z3::kCudaPerRoleTask) {
      Stmt semantic = ffi::GetRef<Stmt>(op);
      if (op->attr_key == z3::kCudaPerRoleTask && op->node.as<ffi::MapObj>() == nullptr) {
        // The public Ascend-style scalar-zero marker is normalized to the
        // canonical map payload required by the scheduled-TIR codec.
        semantic = AttrStmt(Metadata(), op->attr_key, op->value, op->body, op->span);
      }
      return {MakeInitialUnit(std::move(semantic))};
    }
    if (op->attr_key == z3::kCudaStage) {
      ICHECK(is_one(op->value)) << "T.Stage marker value must be 1 at " << path;
      PrimExpr stage_expr = op->node.cast<PrimExpr>();
      const auto* stage = stage_expr.as<IntImmNode>();
      ICHECK(stage != nullptr && stage->value >= 0 &&
             stage->value <= std::numeric_limits<int>::max())
          << "T.Stage requires a non-negative constant integer at " << path << ", got "
          << stage_expr;
      std::vector<Stmt> body = MaterializeList(op->body, path + "/stage");
      ICHECK(!body.empty()) << "T.Stage must wrap at least one complete scheduler task at " << path;
      bool has_non_declaration = false;
      for (Stmt& unit : body) {
        has_non_declaration = has_non_declaration || !IsDeclaration(unit, path + "/stage");
        unit = AssignStage(unit, static_cast<int>(stage->value), path + "/stage");
      }
      ICHECK(has_non_declaration)
          << "T.Stage must wrap a complete scheduler task, not only declarations, at " << path;
      return body;
    }
    if (op->attr_key == z3::kWSOpId) {
      std::vector<Stmt> body = MaterializeList(op->body, path + "/ws_op");
      ICHECK_EQ(body.size(), 1U)
          << "One tl.ws_op_id wrapper must identify exactly one Task or Control at " << path;
      AttrStmt unit = RequireUnit(body[0], path + "/ws_op");
      ffi::String id = RequireOpId(op->value);
      RegisterIdentity(unit, id, path + "/ws_op");
      body[0] = RebuildBody(unit, AttrStmt(op->node, op->attr_key, op->value, unit->body, op->span));
      return body;
    }
    ICHECK(z3::IsScheduleGuardAttribute(op->attr_key))
        << "Unsupported scheduling attribute `" << op->attr_key << "` at " << path
        << "; only tl.assume and tl.assume_requires_runtime_check are guard attributes";
    std::vector<Stmt> body = MaterializeList(op->body, path + "/guard");
    ICHECK(!body.empty()) << "Scheduling guard has an empty body at " << path;
    for (Stmt& unit : body) {
      unit = PrependAttribute(unit, op, path + "/guard");
    }
    return body;
  }

  void NormalizeStages(std::vector<Stmt>* units, const std::string& path) {
    bool manual = false;
    for (const Stmt& stmt : *units) {
      if (IsDeclaration(stmt, path)) {
        continue;
      }
      AttrStmt unit = RequireUnit(stmt, path);
      auto stage = unit->node.cast<Metadata>().Get("stage");
      manual = manual || (stage.has_value() && stage.value().cast<IntImm>()->value >= 0);
    }
    if (!manual) {
      return;
    }
    for (Stmt& stmt : *units) {
      if (IsDeclaration(stmt, path)) {
        continue;
      }
      AttrStmt unit = RequireUnit(stmt, path);
      auto stage = unit->node.cast<Metadata>().Get("stage");
      if (stage.has_value() && stage.value().cast<IntImm>()->value == z3::kUnscheduledStage) {
        stmt = AssignStage(stmt, 0, path);
      }
    }
  }

  int64_t next_source_id_{0};
  std::unordered_map<std::string, int64_t> identities_;
};

struct KernelContext {
  SBlock root;
  ConstrSet outer_ctx;
};

class KernelContextCollector final : public ConstrVisitor {
 public:
  static std::vector<KernelContext> Collect(const Stmt& body) {
    KernelContextCollector collector;
    collector(body);
    return std::move(collector.contexts_);
  }

 private:
  void VisitStmt_(const SBlockNode* op) final {
    if (op->name_hint != "tilelang_root") {
      ConstrVisitor::VisitStmt_(op);
      return;
    }
    contexts_.push_back({ffi::GetRef<SBlock>(op), GetConstrSet()});
  }

  std::vector<KernelContext> contexts_;
};

class KernelRootRewriter final : public StmtMutator {
 public:
  KernelRootRewriter(std::vector<KernelContext> contexts, ffi::Optional<Target> target)
      : contexts_(std::move(contexts)), target_(std::move(target)) {}

  Stmt Rewrite(const Stmt& body) { return VisitStmt(body); }

 private:
  Stmt VisitStmt_(const SBlockNode* op) final {
    if (op->name_hint != "tilelang_root") {
      return StmtMutator::VisitStmt_(op);
    }
    ICHECK_LT(next_root_, contexts_.size());
    const KernelContext& context = contexts_[next_root_++];
    ICHECK(context.root.same_as(ffi::GetRef<SBlock>(op)));

    Stmt normalized = TaskNormalizer::Rewrite(context.root->body);
    SemanticTaskValidator(context.outer_ctx, target_)(normalized);
    SBlock materialized = context.root;
    materialized.CopyOnWrite()->body = ScheduleUnitMaterializer().Rewrite(normalized);

    // Materialization and the canonical codec share one contract. Running a
    // decode/encode cycle here checks declaration scope, guards, stage modes,
    // source IDs, parent links, and task payloads before any planner consumes IR.
    z3::ScheduledTIR scheduled = z3::DecodeScheduledTIR(materialized, context.outer_ctx);
    return z3::EncodeScheduledTIR(std::move(scheduled));
  }

  std::vector<KernelContext> contexts_;
  ffi::Optional<Target> target_;
  size_t next_root_{0};
};

}  // namespace

tvm::transform::Pass MaterializeZ3ScheduleUnits() {
  using namespace tirx::transform;
  auto pass_func = [](PrimFunc func, const IRModule&, const PassContext&) {
    std::vector<KernelContext> contexts = KernelContextCollector::Collect(func->body);
    KernelRootRewriter rewriter(contexts, func->GetAttr<Target>("target"));
    func.CopyOnWrite()->body = rewriter.Rewrite(func->body);
    return func;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.cuda.MaterializeZ3ScheduleUnits", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.cuda.transform.MaterializeZ3ScheduleUnits", MaterializeZ3ScheduleUnits);
}

}  // namespace tvm::tl::cuda
