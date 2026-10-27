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

#include "./z3_schedule_ir.h"

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>

#include <algorithm>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "transform/common/attr.h"
#include "transform/common/iteration_order.h"
#include "transform/common/write_first.h"

namespace tvm::tl::cuda::z3 {

using namespace tirx;

namespace {

using Metadata = ffi::Map<ffi::String, ffi::Any>;
using BufferRemap = std::unordered_map<Buffer, Buffer, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>;

bool IsSemanticTaskMarker(const AttrStmtNode* op) {
  return op->attr_key == kCudaTask || op->attr_key == kCudaPerRoleTask;
}

std::string ExprString(const PrimExpr& expr) {
  std::ostringstream os;
  os << expr;
  return os.str();
}

class Substitution final : public StmtExprMutator {
 public:
  Substitution(const Var& old_var, const Var& new_var) : old_var_(old_var), new_var_(new_var) {}

  Stmt Apply(const Stmt& stmt) { return VisitStmt(stmt); }
  PrimExpr Apply(const PrimExpr& expr) { return VisitExpr(expr); }

  Buffer Apply(const Buffer& buffer) {
    if (auto it = buffer_remap_.find(buffer); it != buffer_remap_.end()) {
      return (*it).second;
    }
    Var data = buffer->data.same_as(old_var_) ? new_var_ : buffer->data;
    auto shape = buffer->shape.Map([this](const PrimExpr& expr) { return VisitExpr(expr); });
    auto strides = buffer->strides.Map([this](const PrimExpr& expr) { return VisitExpr(expr); });
    PrimExpr elem_offset = VisitExpr(buffer->elem_offset);
    if (data.same_as(buffer->data) && shape.same_as(buffer->shape) &&
        strides.same_as(buffer->strides) && elem_offset.same_as(buffer->elem_offset)) {
      buffer_remap_.Set(buffer, buffer);
      return buffer;
    }
    auto node = ffi::make_object<BufferNode>(*buffer.get());
    node->data = std::move(data);
    node->shape = std::move(shape);
    node->strides = std::move(strides);
    node->elem_offset = std::move(elem_offset);
    Buffer result(std::move(node));
    buffer_remap_.Set(buffer, result);
    return result;
  }

  BufferRegion Apply(const BufferRegion& region) {
    Buffer buffer = Apply(region->buffer);
    Region ranges;
    ranges.reserve(region->region.size());
    for (const Range& range : region->region) {
      ranges.push_back(Range::FromMinExtent(VisitExpr(range->min), VisitExpr(range->extent)));
    }
    return BufferRegion(std::move(buffer), std::move(ranges));
  }

  Var ApplyVar(const Var& var) const { return var.same_as(old_var_) ? new_var_ : var; }

 private:
  PrimExpr VisitExpr_(const VarNode* op) final {
    Var var = ffi::GetRef<Var>(op);
    return var.same_as(old_var_) ? PrimExpr(new_var_) : PrimExpr(var);
  }

  Buffer VisitBufferDef(const Buffer& buffer, bool) final { return Apply(buffer); }
  Buffer VisitBufferUse(const Buffer& buffer) final { return Apply(buffer); }

  Stmt VisitStmt_(const BindNode* op) final {
    PrimExpr value = VisitExpr(op->value);
    Var var = ApplyVar(op->var);
    if (value.same_as(op->value) && var.same_as(op->var)) {
      return ffi::GetRef<Stmt>(op);
    }
    return Bind(var, value, op->span);
  }

  Stmt VisitStmt_(const ForNode* op) final {
    For rewritten = Downcast<For>(StmtExprMutator::VisitStmt_(op));
    if (!rewritten->loop_var.same_as(old_var_)) {
      return rewritten;
    }
    auto node = ffi::make_object<ForNode>(*rewritten.get());
    node->loop_var = new_var_;
    return For(std::move(node));
  }

  Var old_var_;
  Var new_var_;
};

void SubstituteGuard(Guard* guard, Substitution* substitution) {
  ICHECK(guard != nullptr && substitution != nullptr);
  if (guard->IsCondition()) {
    auto* condition = static_cast<ConditionGuard*>(guard);
    condition->condition = substitution->Apply(condition->condition);
    return;
  }
  auto* attribute = static_cast<AttributeGuard*>(guard);
  if (auto expr = attribute->node.try_cast<PrimExpr>(); expr.has_value()) {
    attribute->node = substitution->Apply(expr.value());
  }
  if (attribute->value.defined()) {
    attribute->value = substitution->Apply(attribute->value);
  }
}

ConstrSet SubstituteContext(const ConstrSet& context, Substitution* substitution) {
  ConstrSet result;
  result.constrs_.reserve(context.constrs_.size());
  for (const Constr& constraint : context.constrs_) {
    switch (constraint.kind) {
      case Constr::kConstr:
        result.constrs_.emplace_back(substitution->Apply(constraint.value), constraint.is_assume);
        break;
      case Constr::kBindValue:
        result.constrs_.emplace_back(substitution->ApplyVar(constraint.var),
                                     substitution->Apply(constraint.value));
        break;
      case Constr::kBindRange:
        result.constrs_.emplace_back(
            substitution->ApplyVar(constraint.var),
            Range::FromMinExtent(substitution->Apply(constraint.range->min),
                                 substitution->Apply(constraint.range->extent)));
        break;
    }
  }
  return result;
}

class DirectAccessCollector final : public StmtExprVisitor {
 public:
  TaskAccessInfo Analyze(const Stmt& stmt) {
    VisitStmt(stmt);
    return std::move(accesses_);
  }

 private:
  static void AddVar(const Var& var, std::vector<Var>* vars) {
    if (std::find_if(vars->begin(), vars->end(),
                     [&](const Var& existing) { return existing.same_as(var); }) == vars->end()) {
      vars->push_back(var);
    }
  }

  static Region PointRegion(const ffi::Array<PrimExpr>& indices) {
    Region region;
    region.reserve(indices.size());
    for (const PrimExpr& index : indices) {
      region.push_back(Range::FromMinExtent(index, make_const(index.dtype(), 1)));
    }
    return region;
  }

  void VisitExpr_(const VarNode* op) final {
    AddVar(ffi::GetRef<Var>(op), &accesses_.read_vars);
  }

  void VisitExpr_(const BufferLoadNode* op) final {
    accesses_.reads.push_back(BufferRegion(op->buffer, PointRegion(op->indices)));
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const BufferStoreNode* op) final {
    accesses_.writes.push_back(BufferRegion(op->buffer, PointRegion(op->indices)));
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const BindNode* op) final {
    AddVar(op->var, &accesses_.write_vars);
    VisitExpr(op->value);
  }

  TaskAccessInfo accesses_;
};

TaskAccessInfo AnalyzeDirectAccesses(const Stmt& stmt) { return DirectAccessCollector().Analyze(stmt); }

void AppendDirectAccesses(const TaskAccessInfo& input, TaskAccessInfo* output) {
  output->reads.insert(output->reads.end(), input.reads.begin(), input.reads.end());
  output->writes.insert(output->writes.end(), input.writes.begin(), input.writes.end());
  output->read_vars.insert(output->read_vars.end(), input.read_vars.begin(), input.read_vars.end());
  output->write_vars.insert(output->write_vars.end(), input.write_vars.begin(), input.write_vars.end());
  output->state_reads |= input.state_reads;
  output->state_writes |= input.state_writes;
}

void AppendDirectExpressionAccesses(const PrimExpr& expr, TaskAccessInfo* output) {
  AppendDirectAccesses(AnalyzeDirectAccesses(Evaluate(expr)), output);
}

class StorageUseDetector final : public StmtExprVisitor {
 public:
  explicit StorageUseDetector(const Var& storage) : storage_(storage) {}

  bool Contains(const PrimExpr& expr) {
    VisitStmt(Evaluate(expr));
    return found_;
  }

  void VisitBufferDef(const Buffer& buffer, bool alloc_data) final {
    Check(buffer);
    if (!found_) {
      StmtVisitor::VisitBufferDef(buffer, alloc_data);
    }
  }
  void VisitBufferUse(const Buffer& buffer) final {
    Check(buffer);
    if (!found_) {
      StmtVisitor::VisitBufferUse(buffer);
    }
  }

 private:
  void Check(const Buffer& buffer) { found_ = found_ || buffer->data.same_as(storage_); }

  Var storage_;
  bool found_{false};
};

bool ContainsLoopBreak(const Stmt& stmt) {
  static const Op& loop_break_op = Op::Get("tl.loop_break");
  bool found = false;
  PostOrderVisit(stmt, [&](const ffi::ObjectRef& node) {
    if (found) {
      return;
    }
    if (const auto* call = node.as<CallNode>()) {
      found = call->op.same_as(loop_break_op);
    }
  });
  return found;
}

ffi::Optional<Bind> GetFlatTaskBind(const Stmt& stmt) {
  const auto* marker = stmt.as<AttrStmtNode>();
  if (marker == nullptr || !IsSemanticTaskMarker(marker)) {
    return std::nullopt;
  }
  if (const auto* bind = marker->body.as<BindNode>()) {
    return ffi::GetRef<Bind>(bind);
  }
  return std::nullopt;
}

ffi::Optional<Bind> GetScheduledFlatBind(const Stmt& stmt) {
  const auto* unit = stmt.as<AttrStmtNode>();
  if (unit == nullptr || unit->attr_key != kScheduleUnit) {
    return std::nullopt;
  }
  return GetFlatTaskBind(unit->body);
}

std::optional<int64_t> GetMetadataInt(const Metadata& metadata, const ffi::String& key) {
  auto value = metadata.Get(key);
  if (!value.has_value()) {
    return std::nullopt;
  }
  PrimExpr expr = value.value().cast<PrimExpr>();
  const auto* integer = expr.as<IntImmNode>();
  ICHECK(integer != nullptr && integer->dtype.is_int() && integer->dtype.lanes() == 1)
      << "Scheduled metadata `" << key << "` must be a scalar integer, got " << expr;
  return integer->value;
}

uint32_t GetRoleWord(const Metadata& metadata, const char* key) {
  auto value = GetMetadataInt(metadata, key);
  ICHECK(value.has_value()) << "Missing role-mask word `" << key << "`";
  ICHECK_GE(*value, 0);
  ICHECK_LE(*value, static_cast<int64_t>(std::numeric_limits<uint32_t>::max()));
  return static_cast<uint32_t>(*value);
}

RoleMask DecodeRoleMask(const Metadata& metadata, const char* prefix) {
  std::string low_key = std::string(prefix) + "_lo";
  std::string high_key = std::string(prefix) + "_hi";
  bool has_low = metadata.count(low_key) != 0;
  bool has_high = metadata.count(high_key) != 0;
  ICHECK(has_low == has_high) << "Role mask `" << prefix << "` requires two uint32 words";
  if (!has_low) {
    return 0;
  }
  return RoleMaskFromWords(GetRoleWord(metadata, low_key.c_str()),
                           GetRoleWord(metadata, high_key.c_str()));
}

void EncodeRoleMask(RoleMask mask, const char* prefix, Metadata* metadata) {
  if (mask == 0) {
    return;
  }
  auto [low, high] = RoleMaskToWords(mask);
  metadata->Set(std::string(prefix) + "_lo",
                IntImm(DataType::Int(64), static_cast<int64_t>(low)));
  metadata->Set(std::string(prefix) + "_hi",
                IntImm(DataType::Int(64), static_cast<int64_t>(high)));
}

bool GetMetadataFlag(const Metadata& metadata, const char* key, bool default_value) {
  auto value = GetMetadataInt(metadata, key);
  if (!value.has_value()) {
    return default_value;
  }
  ICHECK(*value == 0 || *value == 1) << "Metadata `" << key << "` must be 0 or 1";
  return *value != 0;
}

ffi::String RequireString(const ffi::Any& value, const char* key) {
  if (auto string = value.try_cast<ffi::String>(); string.has_value()) {
    return string.value();
  }
  PrimExpr expr = value.cast<PrimExpr>();
  const auto* string = expr.as<StringImmNode>();
  ICHECK(string != nullptr) << "Metadata `" << key << "` must be a string";
  return string->value;
}

ffi::String GetMetadataString(const Metadata& metadata, const char* key) {
  auto value = metadata.Get(key);
  if (!value.has_value()) {
    return ffi::String();
  }
  return RequireString(value.value(), key);
}

Metadata GetMapPayload(const ffi::Any& node, const char* context) {
  const auto* map = node.as<ffi::MapObj>();
  ICHECK(map != nullptr) << context << " requires a map payload";
  Metadata result;
  for (const auto& [key, value] : *map) {
    result.Set(key.cast<ffi::String>(), value);
  }
  return result;
}

void ValidateKeys(const Metadata& metadata, const std::unordered_set<std::string>& allowed,
                  const char* context) {
  for (const auto& [key, value] : metadata) {
    ICHECK(allowed.count(std::string(key))) << context << " contains unsupported metadata key `" << key << "`";
  }
}

void ValidateBufferVersions(const BufferVersionMap& versions, const char* name) {
  for (const auto& [storage, count] : versions) {
    ICHECK(storage.defined()) << name << " contains an undefined storage variable";
    ICHECK_GT(count, 0) << name << " version count for " << storage << " must be positive";
  }
}

void ValidateVersionModes(const ffi::Map<Var, ffi::String>& modes) {
  for (const auto& [storage, mode] : modes) {
    ICHECK(storage.defined());
    ICHECK(mode == "auto" || mode == "iteration" || mode == "counter")
        << "Buffer version mode for " << storage << " must be auto, iteration, or counter, got "
        << mode;
  }
}

void ValidateAliases(const BufferAliasMap& aliases) {
  for (const auto& [storage, peers] : aliases) {
    ICHECK(storage.defined());
    std::unordered_set<Var, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> seen;
    for (const Var& peer : peers) {
      ICHECK(peer.defined());
      ICHECK(seen.insert(peer).second) << "Duplicate buffer alias " << peer << " for " << storage;
    }
  }
}

class ScheduledTIREncoder {
 public:
  Stmt Encode(const std::vector<std::shared_ptr<IRStructure>>& tree) { return EncodeList(tree); }

 private:
  Stmt EncodeList(const std::vector<std::shared_ptr<IRStructure>>& nodes) {
    ffi::Array<Stmt> statements;
    statements.reserve(nodes.size());
    for (const auto& node : nodes) {
      statements.push_back(EncodeNode(node.get()));
    }
    return SeqStmt::Flatten(statements);
  }

  Stmt EncodeScheduleUnit(const IRStructure* node, Stmt body, const Span& span) {
    Metadata metadata;
    metadata.Set("stage", Integer(node->GetStage()));
    if (node->GetSourceId() >= 0) {
      metadata.Set("source_id", Integer(node->GetSourceId()));
    }
    body = WrapScheduledGuards(std::move(body), node->GetGuards());
    return AttrStmt(metadata, kScheduleUnit, Integer(1), std::move(body), span);
  }

  Stmt EncodeNode(const IRStructure* node) {
    ICHECK(node != nullptr);
    if (node->IsTask()) {
      return EncodeTask(static_cast<const TaskNode*>(node));
    }
    const auto* control = static_cast<const ControlNode*>(node);
    For loop = control->control;
    loop.CopyOnWrite()->body = EncodeList(control->children);
    if (!control->task->ws_op_id.empty()) {
      loop.CopyOnWrite()->annotations.Set(kWSOpId, control->task->ws_op_id);
    }
    return EncodeScheduleUnit(control, std::move(loop), control->control->span);
  }

  Stmt EncodeTask(const TaskNode* task) {
    const auto* marker = task->stmt.as<AttrStmtNode>();
    ICHECK(marker != nullptr && IsSemanticTaskMarker(marker))
        << "EncodeScheduledTIR expected exactly one outer CUDA task marker, got " << task->stmt;

    Metadata metadata;
    if (task->GetII() > 0) {
      if (task->GetLatency() != 0) {
        metadata.Set("latency", IntImm(DataType::Int(64), task->GetLatency()));
      }
      metadata.Set("ii", IntImm(DataType::Int(64), task->GetII()));
    } else if (task->GetLatency() != 0) {
      metadata.Set("latency", IntImm(DataType::Int(64), task->GetLatency()));
    }
    if (task->task_kind == TaskKind::kDeclaration) {
      metadata.Set("declaration", Integer(1));
    }
    if (!task->solver_eligible) {
      metadata.Set("solver_eligible", Integer(0));
    }
    EncodeRoleMask(task->CandidateRoles(), "candidate_roles", &metadata);
    EncodeRoleMask(task->ResolvedRoles(), "resolved_roles", &metadata);
    if (!task->ws_op_id.empty()) {
      metadata.Set("ws_op_id", task->ws_op_id);
    }
    ffi::String key = task->task_kind == TaskKind::kPerRoleGroup ? kCudaPerRoleTask : kCudaTask;
    Stmt body = AttrStmt(metadata, key, marker->value, marker->body, marker->span);
    return EncodeScheduleUnit(task, std::move(body), task->stmt->span);
  }
};

class ScheduledTIRDecoder final : public ConstrVisitor {
 public:
  ScheduledTIRDecoder(const ConstrSet& outer_ctx, ScheduledExtraInfoFactory make_extra_info)
      : make_extra_info_(make_extra_info) {
    constr_stack_ = outer_ctx.constrs_;
  }

  std::vector<std::shared_ptr<IRStructure>> Decode(const Stmt& body) {
    return DecodeList(body, nullptr);
  }

 private:
  std::string PathString() const {
    std::ostringstream os;
    os << "root";
    for (size_t index : path_) {
      os << "/" << index;
    }
    return os.str();
  }

  std::vector<std::shared_ptr<IRStructure>> DecodeList(const Stmt& stmt, IRStructure* parent) {
    if (const auto* evaluate = stmt.as<EvaluateNode>(); evaluate != nullptr && is_zero(evaluate->value)) {
      return {};
    }
    std::vector<std::shared_ptr<IRStructure>> saved;
    saved.swap(result_);
    VisitStmt(stmt);
    std::vector<std::shared_ptr<IRStructure>> decoded;
    decoded.swap(result_);
    result_.swap(saved);
    for (size_t i = 0; i < decoded.size(); ++i) {
      decoded[i]->SetIndex(i);
      decoded[i]->SetParent(parent);
    }
    bool manual = false;
    for (const auto& node : decoded) {
      if (node->GetStage() >= 0) {
        manual = true;
      }
      ICHECK_GE(node->GetStage(), kUnscheduledStage) << "Invalid stage at " << PathString();
    }
    if (manual) {
      for (auto& node : decoded) {
        bool declaration = node->IsTask() &&
                           static_cast<const TaskNode*>(node.get())->task_kind == TaskKind::kDeclaration;
        if (!declaration && node->GetStage() == kUnscheduledStage) {
          node->SetStage(0);
        }
      }
    }
    return decoded;
  }

  void VisitStmt_(const SeqStmtNode* op) final {
    size_t old_size = constr_stack_.size();
    for (const Stmt& stmt : op->seq) {
      VisitStmt(stmt);
      if (auto bind = GetScheduledFlatBind(stmt); bind.defined()) {
        constr_stack_.emplace_back(bind.value()->var, bind.value()->value);
      }
    }
    constr_stack_.resize(old_size);
  }

  void VisitStmt_(const AttrStmtNode* op) final {
    ICHECK(op->attr_key == kScheduleUnit)
        << "DecodeScheduledTIR expected `tl.schedule_unit` at " << PathString() << ", got attr "
        << op->attr_key;
    Metadata metadata = GetMapPayload(op->node, "tl.schedule_unit");
    ValidateKeys(metadata, {"stage", "source_id"}, "tl.schedule_unit");
    std::optional<int64_t> stage = GetMetadataInt(metadata, "stage");
    std::optional<int64_t> source_id = GetMetadataInt(metadata, "source_id");
    if (source_id.has_value()) {
      ICHECK_GE(*source_id, 0);
      next_source_id_ = std::max(next_source_id_, *source_id + 1);
    }

    path_.push_back(result_.size());
    GuardList guards;
    ffi::String ws_op_id;
    auto node = DecodeGuardedNode(op, op->body, stage, source_id, &guards, &ws_op_id);
    path_.pop_back();
    result_.push_back(std::move(node));
  }

  void VisitStmt_(const ForNode*) final { FailUnmarked("For"); }
  void VisitStmt_(const IfThenElseNode*) final { FailUnmarked("IfThenElse"); }
  void VisitStmt_(const WhileNode*) final { FailUnmarked("While"); }
  void VisitStmt_(const SBlockNode*) final { FailUnmarked("SBlock"); }
  void VisitStmt_(const SBlockRealizeNode*) final { FailUnmarked("SBlockRealize"); }
  void VisitStmtDefault_(const ffi::Object*) final { FailUnmarked("statement"); }

  [[noreturn]] void FailUnmarked(const char* kind) const {
    LOG(FATAL) << "DecodeScheduledTIR found an unmarked " << kind << " at " << PathString();
  }

  std::shared_ptr<IRStructure> DecodeGuardedNode(const AttrStmtNode* marker, const Stmt& body,
                                                 std::optional<int64_t> stage,
                                                 std::optional<int64_t> source_id,
                                                 GuardList* guards, ffi::String* ws_op_id) {
    if (const auto* condition = body.as<IfThenElseNode>()) {
      ICHECK(!condition->else_case.defined())
          << "Scheduled condition guard at " << PathString() << " must not have an else branch";
      guards->push_back(std::make_unique<ConditionGuard>(condition->condition));
      auto scope = MakeGuard(condition->condition);
      return DecodeGuardedNode(marker, condition->then_case, stage, source_id, guards, ws_op_id);
    }
    if (const auto* attribute = body.as<AttrStmtNode>()) {
      if (IsSemanticTaskMarker(attribute)) {
        return DecodeNode(marker, body, stage, source_id, std::move(*guards), *ws_op_id);
      }
      if (attribute->attr_key == kWSOpId) {
        ffi::String id = RequireString(attribute->value, kWSOpId);
        ICHECK(!id.empty()) << "Empty tl.ws_op_id at " << PathString();
        ICHECK(ws_op_id->empty() || *ws_op_id == id)
            << "Conflicting tl.ws_op_id wrappers at " << PathString();
        *ws_op_id = id;
        return DecodeGuardedNode(marker, attribute->body, stage, source_id, guards, ws_op_id);
      }
      ICHECK(IsScheduleGuardAttribute(attribute->attr_key))
          << "DecodeScheduledTIR found unsupported guard attribute `" << attribute->attr_key
          << "` at " << PathString();
      guards->push_back(std::make_unique<AttributeGuard>(attribute->node, attribute->attr_key,
                                                         attribute->value, attribute->span));
      if (attribute->attr_key == tirx::attr::tilelang_assume) {
        auto scope = MakeGuard(Downcast<PrimExpr>(attribute->node), true);
        return DecodeGuardedNode(marker, attribute->body, stage, source_id, guards, ws_op_id);
      }
      return DecodeGuardedNode(marker, attribute->body, stage, source_id, guards, ws_op_id);
    }
    return DecodeNode(marker, body, stage, source_id, std::move(*guards), *ws_op_id);
  }

  std::shared_ptr<IRStructure> DecodeNode(const AttrStmtNode* marker, const Stmt& body,
                                          std::optional<int64_t> stage,
                                          std::optional<int64_t> source_id, GuardList guards,
                                          ffi::String ws_op_id) {
    const auto* loop = body.as<ForNode>();
    if (loop == nullptr) {
      return DecodeTask(marker, body, stage, source_id, std::move(guards), std::move(ws_op_id));
    }

    ICHECK(stage.has_value()) << "Control schedule unit at " << PathString() << " is missing stage";
    ICHECK(loop->kind == ForKind::kSerial || loop->kind == ForKind::kUnrolled)
        << "Scheduled control at " << PathString() << " must be serial or unrolled";
    auto control = MakeControlNode();
    control->SetSchedule(static_cast<int>(*stage), 0);
    control->SetSourceId(source_id.has_value() ? *source_id : next_source_id_++);
    control->SetGuards(std::move(guards));
    control->control = ffi::GetRef<For>(loop);
    control->task->stmt = For(loop->loop_var, loop->min, loop->extent, loop->kind, Evaluate(0),
                              loop->thread_binding, loop->annotations, loop->step, loop->span);
    control->task->outer_ctx = GetConstrSet();
    control->task->accesses = AnalyzeDirectAccesses(control->task->stmt);
    ApplyGuardAccesses(*control, control->task.get());
    control->task->ws_op_id = ws_op_id;
    if (auto id = loop->annotations.Get(kWSOpId); id.has_value()) {
      ffi::String annotation_id = RequireString(id.value(), kWSOpId);
      ICHECK(control->task->ws_op_id.empty() || control->task->ws_op_id == annotation_id)
          << "Conflicting control tl.ws_op_id at " << PathString();
      control->task->ws_op_id = annotation_id;
    }
    {
      auto loop_scope = MakeGuard(loop->loop_var, Range::FromMinExtent(loop->min, loop->extent));
      auto non_empty_scope = MakeGuard(loop->extent > 0);
      control->children = DecodeList(loop->body, control.get());
    }
    control->InitializeLoopBodyContext();
    return control;
  }

  std::shared_ptr<TaskNode> DecodeTask(const AttrStmtNode* marker, const Stmt& body,
                                       std::optional<int64_t> stage,
                                       std::optional<int64_t> source_id, GuardList guards,
                                       ffi::String ws_op_id) {
    const auto* semantic = body.as<AttrStmtNode>();
    ICHECK(semantic != nullptr && IsSemanticTaskMarker(semantic))
        << "Leaf schedule unit at " << PathString()
        << " must contain exactly one outer tl.cuda_task/tl.cuda_per_role_task, got " << body;
    Metadata metadata = GetMapPayload(semantic->node, "CUDA task marker");
    ValidateKeys(metadata,
                 {"latency", "ii", "candidate_roles_lo", "candidate_roles_hi", "resolved_roles_lo",
                  "resolved_roles_hi", "ws_op_id", "declaration", "solver_eligible"},
                 "CUDA task marker");

    auto task = MakeTaskNode();
    task->stmt = body;
    task->task_kind = semantic->attr_key == kCudaPerRoleTask ? TaskKind::kPerRoleGroup
                                                              : TaskKind::kOperation;
    bool declaration = GetMetadataFlag(metadata, "declaration", false);
    if (declaration) {
      ICHECK(task->task_kind == TaskKind::kOperation)
          << "A declaration cannot be a PerRole group at " << PathString();
      task->task_kind = TaskKind::kDeclaration;
      ICHECK(semantic->body.as<AllocBufferNode>() != nullptr ||
             semantic->body.as<DeclBufferNode>() != nullptr)
          << "Marked declaration at " << PathString() << " must contain AllocBuffer or DeclBuffer";
    } else {
      ICHECK(stage.has_value()) << "Non-declaration task at " << PathString() << " is missing stage";
    }
    task->solver_eligible = GetMetadataFlag(metadata, "solver_eligible", !declaration);
    if (declaration) {
      ICHECK(!task->solver_eligible) << "Declarations cannot be solver eligible";
    }
    task->SetGuards(std::move(guards));
    task->outer_ctx = GetConstrSet();
    task->ws_op_id = GetMetadataString(metadata, "ws_op_id");
    ICHECK(ws_op_id.empty() || task->ws_op_id.empty() || ws_op_id == task->ws_op_id)
        << "Conflicting task tl.ws_op_id at " << PathString();
    if (!ws_op_id.empty()) {
      task->ws_op_id = ws_op_id;
    }

    auto latency = GetMetadataInt(metadata, "latency");
    auto ii = GetMetadataInt(metadata, "ii");
    if (latency.has_value()) {
      ICHECK_GE(*latency, 0);
    }
    if (ii.has_value()) {
      ICHECK_GT(*ii, 0);
    }
    if (latency.has_value() && ii.has_value()) {
      ICHECK_GE(*latency, *ii) << "Task latency must be greater than or equal to II";
    }
    task->SetCost(latency.value_or(0), ii.value_or(0));
    task->SetCandidateRoles(DecodeRoleMask(metadata, "candidate_roles"));
    task->SetResolvedRoles(DecodeRoleMask(metadata, "resolved_roles"));
    task->SetSchedule(static_cast<int>(stage.value_or(kUnscheduledStage)), 0);
    task->SetSourceId(source_id.has_value() ? *source_id : next_source_id_++);
    task->accesses = AnalyzeDirectAccesses(body);
    task->accesses.state_reads = static_cast<SpecialStateMask>(SpecialState::kLoopControl);
    task->accesses.state_writes =
        ContainsLoopBreak(body) ? static_cast<SpecialStateMask>(SpecialState::kLoopControl) : 0;
    ApplyGuardAccesses(*task, task.get());
    return task;
  }

  void ApplyGuardAccesses(const IRStructure& node, TaskNode* task) {
    for (const auto& guard : node.GetGuards()) {
      if (guard->IsCondition()) {
        AppendDirectExpressionAccesses(static_cast<const ConditionGuard*>(guard.get())->condition,
                                       &task->accesses);
        continue;
      }
      const auto* attribute = static_cast<const AttributeGuard*>(guard.get());
      if (auto expr = attribute->node.try_cast<PrimExpr>(); expr.has_value()) {
        AppendDirectExpressionAccesses(expr.value(), &task->accesses);
      }
      if (attribute->value.defined()) {
        AppendDirectExpressionAccesses(attribute->value, &task->accesses);
      }
    }
  }

  std::shared_ptr<TaskNode> MakeTaskNode() const {
    return std::make_shared<TaskNode>(make_extra_info_ ? make_extra_info_() : nullptr);
  }

  std::shared_ptr<ControlNode> MakeControlNode() const {
    auto control = std::make_shared<ControlNode>(make_extra_info_ ? make_extra_info_() : nullptr);
    if (make_extra_info_) {
      control->task = MakeTaskNode();
      control->task->solver_eligible = false;
      control->task->SetParent(control.get());
    }
    return control;
  }

  std::vector<std::shared_ptr<IRStructure>> result_;
  ScheduledExtraInfoFactory make_extra_info_{nullptr};
  std::vector<size_t> path_;
  int64_t next_source_id_{0};
};

void ValidateNode(const IRStructure* node, const IRStructure* parent, size_t index,
                  std::unordered_set<const IRStructure*>* seen) {
  ICHECK(node != nullptr);
  ICHECK(seen->insert(node).second) << "A scheduled IR node occurs more than once in the tree";
  ICHECK(node->GetParent() == parent) << "Invalid parent link at sibling index " << index;
  ICHECK(node->GetIndex() == index) << "Stale sibling index at position " << index;
  ICHECK_GE(node->GetStage(), kUnscheduledStage);
  ICHECK_GE(node->GetLatency(), 0);
  ICHECK_GE(node->GetII(), 0);
  ICHECK(node->GetII() == 0 || node->GetLatency() == 0 || node->GetLatency() >= node->GetII())
      << "Task latency must be greater than or equal to II when both are present";
  ICHECK_GE(node->GetSourceId(), -1);
  if ((node->ResolvedRoles() != 0) && (node->CandidateRoles() != 0)) {
    ICHECK((node->ResolvedRoles() & ~node->CandidateRoles()) == 0)
        << "Resolved roles must be a subset of nonempty candidate roles";
  }

  if (node->IsTask()) {
    const auto* task = static_cast<const TaskNode*>(node);
    ICHECK(task->stmt.defined()) << "Task has no statement";
    if (task->task_kind == TaskKind::kDeclaration) {
      ICHECK(!task->solver_eligible);
      ICHECK(task->GetStage() == kUnscheduledStage) << "Declarations never receive a solver stage";
      ICHECK(task->GetLatency() == 0 && task->GetII() == 0)
          << "Declarations never receive solver costs";
      ICHECK(task->CandidateRoles() == 0 && task->ResolvedRoles() == 0)
          << "Declarations never receive roles";
      const auto* marker = task->stmt.as<AttrStmtNode>();
      ICHECK(marker != nullptr && IsSemanticTaskMarker(marker));
      ICHECK(marker->body.as<AllocBufferNode>() != nullptr ||
             marker->body.as<DeclBufferNode>() != nullptr)
          << "A declaration task must contain AllocBuffer or DeclBuffer";
    }
    return;
  }

  const auto* control = static_cast<const ControlNode*>(node);
  ICHECK(control->control.defined());
  ICHECK(control->control->kind == ForKind::kSerial ||
         control->control->kind == ForKind::kUnrolled);
  ICHECK(control->task != nullptr) << "Control is missing its synthetic header task";
  ICHECK(!control->task->solver_eligible);
  ICHECK(control->task->CandidateRoles() == 0 && control->task->ResolvedRoles() == 0)
      << "Role placement lives on real tasks, not synthetic control headers";
  ValidateNode(control->task.get(), control, 0, seen);
  for (size_t i = 0; i < control->children.size(); ++i) {
    ValidateNode(control->children[i].get(), control, i, seen);
  }
}

void CollectControlNodes(IRStructure* node, std::vector<ControlNode*>* controls) {
  if (node == nullptr || node->IsTask()) {
    return;
  }
  auto* control = static_cast<ControlNode*>(node);
  controls->push_back(control);
  for (const auto& child : control->children) {
    CollectControlNodes(child.get(), controls);
  }
}

ScheduledTIR DecodeForFFI(const SBlock& root) { return DecodeScheduledTIR(root, ConstrSet()); }

SBlock RoundTrip(const SBlock& root) {
  ScheduledTIR first = DecodeForFFI(root);
  SBlock canonical = EncodeScheduledTIR(std::move(first));
  ScheduledTIR second = DecodeForFFI(canonical);
  return EncodeScheduledTIR(std::move(second));
}

SBlock CloneRoundTrip(const SBlock& root) {
  ScheduledTIR scheduled = DecodeForFFI(root);
  std::vector<std::shared_ptr<IRStructure>> cloned;
  cloned.reserve(scheduled.tree.size());
  for (const auto& node : scheduled.tree) {
    cloned.push_back(node->Clone());
  }
  scheduled.tree = std::move(cloned);
  RepairParentsAndIndices(scheduled.tree);
  ValidateScheduledTIR(scheduled);
  return EncodeScheduledTIR(std::move(scheduled));
}

SBlock SubstituteRoundTrip(const SBlock& root, const ffi::String& old_name,
                           const ffi::String& new_name, bool clone_first) {
  ICHECK(!old_name.empty() && !new_name.empty());
  ScheduledTIR scheduled = DecodeForFFI(root);
  if (clone_first) {
    std::vector<std::shared_ptr<IRStructure>> cloned;
    cloned.reserve(scheduled.tree.size());
    for (const auto& node : scheduled.tree) {
      cloned.push_back(node->Clone());
    }
    scheduled.tree = std::move(cloned);
    RepairParentsAndIndices(scheduled.tree);
  }

  ffi::Optional<Var> old_var{std::nullopt};
  PostOrderVisit(root, [&](const ffi::ObjectRef& node) {
    const auto* var = node.as<VarNode>();
    if (var == nullptr || var->name_hint != old_name) {
      return;
    }
    Var candidate = ffi::GetRef<Var>(var);
    ICHECK(!old_var.has_value() || old_var.value().same_as(candidate))
        << "Substitution variable name `" << old_name << "` is not unique";
    old_var = candidate;
  });
  ICHECK(old_var.has_value()) << "Substitution variable `" << old_name << "` was not found";
  Var new_var = old_var.value().copy_with_name(new_name);
  for (auto& node : scheduled.tree) {
    node->SubstituteVar(old_var.value(), new_var);
  }
  RepairParentsAndIndices(scheduled.tree);
  ValidateScheduledTIR(scheduled);
  return EncodeScheduledTIR(std::move(scheduled));
}

ffi::Map<ffi::String, ffi::Any> Inspect(const SBlock& root) {
  ScheduledTIR scheduled = DecodeForFFI(root);
  std::vector<TaskNode*> tasks;
  std::vector<TaskNode*> with_headers;
  std::vector<ControlNode*> controls;
  for (const auto& node : scheduled.tree) {
    CollectAllTaskNodes(node.get(), &tasks);
    CollectAllTaskNodes(node.get(), &with_headers, true);
    CollectControlNodes(node.get(), &controls);
  }

  ffi::Map<ffi::String, ffi::Any> result;
  result.Set("task_count", Integer(static_cast<int64_t>(tasks.size())));
  result.Set("task_with_header_count", Integer(static_cast<int64_t>(with_headers.size())));
  result.Set("control_count", Integer(static_cast<int64_t>(controls.size())));
  result.Set("null_root_compare", Integer(CompareIRStructure(nullptr, scheduled.tree.front().get())));
  result.Set("self_compare",
             Integer(CompareIRStructure(scheduled.tree.front().get(), scheduled.tree.front().get())));
  if (scheduled.tree.size() >= 2) {
    result.Set("sibling_compare",
               Integer(CompareIRStructure(scheduled.tree[0].get(), scheduled.tree[1].get())));
  }
  if (!controls.empty()) {
    const ControlNode* control = controls.front();
    result.Set("header_parent_valid", Integer(control->task->GetParent() == control));
    result.Set("control_read_regions",
               Integer(static_cast<int64_t>(control->Accesses().reads.size())));
    result.Set("control_write_regions",
               Integer(static_cast<int64_t>(control->Accesses().writes.size())));
    result.Set("body_has_loop_break", Integer(control->BodyHasLoopBreak()));
    if (!control->Accesses().writes.empty()) {
      Var storage = control->Accesses().writes.front()->buffer->data;
      result.Set("body_touches_storage", Integer(control->BodyTouchesStorage(storage)));
    }
    if (!control->children.empty()) {
      const IRStructure* child = control->children.front().get();
      result.Set("ancestor_compare", Integer(CompareIRStructure(control, child)));
      result.Set("descendant_compare", Integer(CompareIRStructure(child, control)));
      result.Set("path_size", Integer(static_cast<int64_t>(child->PathFrom(control).size())));
      result.Set("child_on_path_valid",
                 Integer(const_cast<IRStructure*>(child)->GetChildOnPathFrom(
                             const_cast<ControlNode*>(control)) == child));
    }
    if (control->children.size() >= 2) {
      const IRStructure* lca = control->children[0]->GetLowestCommonAncestor(control->children[1].get());
      result.Set("lca_is_control", Integer(lca == control));
    }
    result.Set("iteration_count", ffi::String(ExprString(CalculateIterationCount(controls.back()))));
  }
  return result;
}

int WriteFirstLattice(const ffi::String& operation, int first, int second) {
  ICHECK_GE(first, 0);
  ICHECK_LE(first, static_cast<int>(AccessOrder::kReadFirst));
  ICHECK_GE(second, 0);
  ICHECK_LE(second, static_cast<int>(AccessOrder::kReadFirst));
  auto a = static_cast<AccessOrder>(first);
  auto b = static_cast<AccessOrder>(second);
  AccessOrder result;
  if (operation == "seq") {
    result = SeqCompose(a, b);
  } else if (operation == "if_else") {
    result = MergeIfThenElse(a, b);
  } else if (operation == "if_only") {
    result = MergeIfThenOnly(a);
  } else if (operation == "demote") {
    result = DemoteIfNotMustExecute(a, second != 0);
  } else {
    LOG(FATAL) << "Unknown write-first lattice operation `" << operation << "`";
  }
  return static_cast<int>(result);
}

ffi::Optional<int64_t> MinimumIterationDistanceFFI(
    int64_t sites, const ffi::Array<int64_t>& sources, const ffi::Array<int64_t>& destinations,
    const ffi::Array<int64_t>& distances, const ffi::Array<int64_t>& strict, int64_t source,
    int64_t destination) {
  ICHECK_GE(sites, 0);
  ICHECK(sources.size() == destinations.size() && sources.size() == distances.size() &&
         sources.size() == strict.size());
  IterationOrder order(static_cast<size_t>(sites));
  for (size_t i = 0; i < sources.size(); ++i) {
    ICHECK(order.AddEdge(static_cast<size_t>(sources[i]), static_cast<size_t>(destinations[i]),
                         distances[i], strict[i] != 0))
        << "Invalid iteration-order edge at index " << i;
  }
  ICHECK_GE(source, 0);
  ICHECK_GE(destination, 0);
  return order.MinimumDistance(static_cast<size_t>(source), static_cast<size_t>(destination));
}

}  // namespace

bool IsScheduleGuardAttribute(const ffi::String& key) {
  return key == tirx::attr::tilelang_assume || key == tl::attr::kAssumeRequiresRuntimeCheck;
}

Stmt WrapScheduledGuards(Stmt body, const GuardList& guards) {
  for (auto it = guards.rbegin(); it != guards.rend(); ++it) {
    body = (*it)->Wrap(std::move(body));
  }
  return body;
}

std::unique_ptr<Guard> ConditionGuard::Clone() const {
  return std::make_unique<ConditionGuard>(condition);
}

void ConditionGuard::SubstituteVar(const Var& old_var, const Var& new_var) {
  Substitution substitution(old_var, new_var);
  condition = substitution.Apply(condition);
}

Stmt ConditionGuard::Wrap(Stmt body) const { return IfThenElse(condition, std::move(body)); }

std::unique_ptr<Guard> AttributeGuard::Clone() const {
  return std::make_unique<AttributeGuard>(node, key, value, span);
}

void AttributeGuard::SubstituteVar(const Var& old_var, const Var& new_var) {
  Substitution substitution(old_var, new_var);
  SubstituteGuard(this, &substitution);
}

Stmt AttributeGuard::Wrap(Stmt body) const { return AttrStmt(node, key, value, std::move(body), span); }

void IRStructure::SetParent(IRStructure* parent) {
  ICHECK(parent != this) << "An IRStructure node cannot parent itself";
  ICHECK(parent == nullptr || !parent->IsWithin(this))
      << "An IRStructure parent cannot be inside the child's subtree";
  parent_ = parent;
}

bool IRStructure::IsWithin(const IRStructure* ancestor) const {
  for (const IRStructure* node = this; node != nullptr; node = node->GetParent()) {
    if (node == ancestor) {
      return true;
    }
  }
  return false;
}

IRStructure* IRStructure::GetLowestCommonAncestor(IRStructure* other) {
  return const_cast<IRStructure*>(
      static_cast<const IRStructure*>(this)->GetLowestCommonAncestor(other));
}

const IRStructure* IRStructure::GetLowestCommonAncestor(const IRStructure* other) const {
  ICHECK(other != nullptr) << "Cannot find an ancestor of a null IRStructure";
  std::unordered_set<const IRStructure*> ancestors;
  for (const IRStructure* node = this; node != nullptr; node = node->GetParent()) {
    ancestors.insert(node);
  }
  for (const IRStructure* node = other; node != nullptr; node = node->GetParent()) {
    if (ancestors.count(node)) {
      return node;
    }
  }
  return nullptr;
}

std::vector<IRStructure*> IRStructure::PathFrom(IRStructure* ancestor) {
  std::vector<IRStructure*> path;
  for (IRStructure* node = this; node != nullptr; node = node->GetParent()) {
    path.push_back(node);
    if (node == ancestor) {
      std::reverse(path.begin(), path.end());
      return path;
    }
  }
  ICHECK(ancestor == nullptr) << "The requested IRStructure path start is not an ancestor";
  std::reverse(path.begin(), path.end());
  return path;
}

std::vector<const IRStructure*> IRStructure::PathFrom(const IRStructure* ancestor) const {
  std::vector<const IRStructure*> path;
  for (const IRStructure* node = this; node != nullptr; node = node->GetParent()) {
    path.push_back(node);
    if (node == ancestor) {
      std::reverse(path.begin(), path.end());
      return path;
    }
  }
  ICHECK(ancestor == nullptr) << "The requested IRStructure path start is not an ancestor";
  std::reverse(path.begin(), path.end());
  return path;
}

IRStructure* IRStructure::GetChildOnPathFrom(IRStructure* strict_ancestor) {
  ICHECK(strict_ancestor != nullptr) << "IRStructure path ancestor cannot be null";
  IRStructure* node = this;
  while (node != nullptr && node->GetParent() != strict_ancestor) {
    node = node->GetParent();
  }
  ICHECK(node != nullptr) << "The requested IRStructure path start is not a strict ancestor";
  return node;
}

PrimExpr IRStructure::GetConditionGuard() const {
  PrimExpr result = Bool(true);
  for (const auto& guard : guards_) {
    if (!guard->IsCondition()) {
      continue;
    }
    const PrimExpr& condition = static_cast<const ConditionGuard*>(guard.get())->condition;
    result = is_one(result) ? condition : result && condition;
  }
  return result;
}

bool IRStructure::GuardsTouchStorage(const Var& storage) const {
  for (const auto& guard : guards_) {
    if (guard->IsCondition()) {
      const auto* condition = static_cast<const ConditionGuard*>(guard.get());
      if (StorageUseDetector(storage).Contains(condition->condition)) {
        return true;
      }
      continue;
    }
    const auto* attribute = static_cast<const AttributeGuard*>(guard.get());
    if (auto expr = attribute->node.try_cast<PrimExpr>();
        expr.has_value() && StorageUseDetector(storage).Contains(expr.value())) {
      return true;
    }
    if (attribute->value.defined() && StorageUseDetector(storage).Contains(attribute->value)) {
      return true;
    }
  }
  return false;
}

void IRStructure::PrependCondition(const PrimExpr& condition) {
  guards_.insert(guards_.begin(), std::make_unique<ConditionGuard>(condition));
}

void IRStructure::SetSchedule(int stage, size_t index) {
  ICHECK_GE(stage, kUnscheduledStage);
  stage_ = stage;
  index_ = index;
}

void IRStructure::SetCost(int64_t latency, int64_t ii) {
  ICHECK_GE(latency, 0);
  ICHECK_GE(ii, 0);
  latency_ = latency;
  ii_ = ii;
}

bool IRStructure::ReadsStorage(const Var& storage) const {
  for (const BufferRegion& region : Accesses().reads) {
    if (region->buffer->data.same_as(storage)) {
      return true;
    }
  }
  return false;
}

bool IRStructure::WritesStorage(const Var& storage) const {
  for (const BufferRegion& region : Accesses().writes) {
    if (region->buffer->data.same_as(storage)) {
      return true;
    }
  }
  return false;
}

void IRStructure::SubstituteVarInGuards(const Var& old_var, const Var& new_var) {
  Substitution substitution(old_var, new_var);
  for (auto& guard : guards_) {
    SubstituteGuard(guard.get(), &substitution);
  }
}

void IRStructure::CopyBaseMetadataFrom(const IRStructure& other) {
  guards_.clear();
  guards_.reserve(other.guards_.size());
  for (const auto& guard : other.guards_) {
    guards_.push_back(guard->Clone());
  }
  stage_ = other.stage_;
  index_ = other.index_;
  latency_ = other.latency_;
  ii_ = other.ii_;
  source_id_ = other.source_id_;
}

std::shared_ptr<IRStructure> TaskNode::Clone() const {
  auto clone = std::make_shared<TaskNode>(CloneExtraInfo());
  clone->stmt = stmt;
  clone->outer_ctx = outer_ctx;
  clone->task_kind = task_kind;
  clone->solver_eligible = solver_eligible;
  clone->ws_op_id = ws_op_id;
  clone->accesses = accesses;
  clone->candidate_roles_ = candidate_roles_;
  clone->resolved_roles_ = resolved_roles_;
  clone->CopyBaseMetadataFrom(*this);
  return clone;
}

void TaskNode::SubstituteVar(const Var& old_var, const Var& new_var) {
  ICHECK(old_var.defined() && new_var.defined());
  if (old_var.same_as(new_var)) {
    return;
  }
  Substitution substitution(old_var, new_var);
  stmt = substitution.Apply(stmt);
  accesses.reads = accesses.reads.Map(
      [&](const BufferRegion& region) { return substitution.Apply(region); });
  accesses.writes = accesses.writes.Map(
      [&](const BufferRegion& region) { return substitution.Apply(region); });
  for (Var& var : accesses.read_vars) {
    var = substitution.ApplyVar(var);
  }
  for (Var& var : accesses.write_vars) {
    var = substitution.ApplyVar(var);
  }
  for (auto& guard : MutableGuards()) {
    SubstituteGuard(guard.get(), &substitution);
  }
  outer_ctx = SubstituteContext(outer_ctx, &substitution);
  SubstituteVarInExtraInfo(old_var, new_var);
}

void TaskNode::SetCandidateRoles(RoleMask roles) {
  ICHECK(task_kind != TaskKind::kDeclaration || roles == 0)
      << "Declarations never receive role candidates";
  candidate_roles_ = roles;
}

void TaskNode::SetResolvedRoles(RoleMask roles) {
  ICHECK(task_kind != TaskKind::kDeclaration || roles == 0)
      << "Declarations never receive resolved roles";
  resolved_roles_ = roles;
}

const ConstrSet& TaskNode::GetContextBeforeConditionGuards() const {
  ConstrSet result = outer_ctx;
  for (auto it = GetGuards().rbegin(); it != GetGuards().rend(); ++it) {
    if (!(*it)->IsCondition()) {
      continue;
    }
    const PrimExpr& condition = static_cast<const ConditionGuard*>(it->get())->condition;
    bool removed = false;
    for (size_t i = result.constrs_.size(); i > 0; --i) {
      const Constr& constraint = result.constrs_[i - 1];
      if (constraint.kind == Constr::kConstr && ffi::StructuralEqual()(constraint.value, condition)) {
        result.constrs_.erase(result.constrs_.begin() + i - 1);
        removed = true;
        break;
      }
    }
    ICHECK(removed) << "Task condition is missing from its constraint context: " << condition;
  }
  context_before_condition_guards_ = std::move(result);
  return context_before_condition_guards_;
}

ControlNode::ControlNode(std::unique_ptr<IRExtraInfo> extra_info)
    : IRStructure(std::move(extra_info)), task(std::make_shared<TaskNode>()) {
  task->solver_eligible = false;
  task->SetParent(this);
}

std::shared_ptr<IRStructure> ControlNode::Clone() const {
  auto clone = std::make_shared<ControlNode>(CloneExtraInfo());
  clone->control = control;
  clone->task = std::static_pointer_cast<TaskNode>(task->Clone());
  clone->task->SetParent(clone.get());
  clone->children.reserve(children.size());
  for (const auto& child : children) {
    clone->children.push_back(child->Clone());
  }
  RepairParentsAndIndices(clone->children, clone.get());
  clone->loop_body_ctx_ = loop_body_ctx_;
  clone->ii_per_iter_ = ii_per_iter_;
  clone->first_iteration_span_ = first_iteration_span_;
  clone->CopyBaseMetadataFrom(*this);
  return clone;
}

void ControlNode::SubstituteVar(const Var& old_var, const Var& new_var) {
  ICHECK(old_var.defined() && new_var.defined());
  if (old_var.same_as(new_var)) {
    return;
  }
  Substitution substitution(old_var, new_var);
  control = Downcast<For>(substitution.Apply(control));
  task->SubstituteVar(old_var, new_var);
  for (auto& child : children) {
    child->SubstituteVar(old_var, new_var);
  }
  for (auto& guard : MutableGuards()) {
    SubstituteGuard(guard.get(), &substitution);
  }
  loop_body_ctx_ = SubstituteContext(loop_body_ctx_, &substitution);
  SubstituteVarInExtraInfo(old_var, new_var);
}

void ControlNode::InitializeLoopBodyContext() {
  ICHECK(task != nullptr && control.defined());
  ConstrSet result = task->outer_ctx;
  result.AddConstr(control->loop_var, Range::FromMinExtent(control->min, control->extent));
  result.AddConstr(control->extent > 0);
  for (const auto& child : children) {
    if (!child->IsTask()) {
      continue;
    }
    const auto* body_task = static_cast<const TaskNode*>(child.get());
    auto bind = GetFlatTaskBind(body_task->stmt);
    if (!bind.defined()) {
      continue;
    }
    PrimExpr definition_guard = body_task->GetConditionGuard();
    if (is_one(definition_guard)) {
      result.AddConstr(bind.value()->var, bind.value()->value);
    } else if (!bind.value()->var.dtype().is_vector()) {
      PrimExpr value = FreshenMutableReads()(bind.value()->value);
      result.AddConstr(!definition_guard || bind.value()->var == value);
    }
  }
  loop_body_ctx_ = std::move(result);
}

ControlNode* ControlNode::GetParentControl() {
  IRStructure* parent = GetParent();
  if (parent == nullptr) {
    return nullptr;
  }
  ICHECK(parent->IsControl());
  return static_cast<ControlNode*>(parent);
}

const ControlNode* ControlNode::GetParentControl() const {
  const IRStructure* parent = GetParent();
  if (parent == nullptr) {
    return nullptr;
  }
  ICHECK(parent->IsControl());
  return static_cast<const ControlNode*>(parent);
}

std::vector<const ControlNode*> ControlNode::GetAncestorControls() const {
  std::vector<const ControlNode*> result;
  for (const ControlNode* current = this; current != nullptr; current = current->GetParentControl()) {
    result.push_back(current);
  }
  std::reverse(result.begin(), result.end());
  return result;
}

bool ControlNode::BodyHasLoopBreak() const {
  constexpr SpecialStateMask loop_control = static_cast<SpecialStateMask>(SpecialState::kLoopControl);
  for (const auto& child : children) {
    if ((child->Accesses().state_writes & loop_control) != 0) {
      return true;
    }
  }
  return false;
}

bool ControlNode::BodyTouchesStorage(const Var& storage) const {
  for (const auto& child : children) {
    if (child->TouchesStorage(storage)) {
      return true;
    }
  }
  return false;
}

void ControlNode::SetIterationTiming(int64_t ii_per_iter, int64_t first_iteration_span,
                                     int64_t total_latency) {
  ICHECK_GE(ii_per_iter, 0);
  ICHECK_GE(first_iteration_span, 0);
  ICHECK_GE(total_latency, 0);
  ICHECK(total_latency == 0 || total_latency >= first_iteration_span)
      << "A nonzero control latency must include its first-iteration span";
  ii_per_iter_ = ii_per_iter;
  first_iteration_span_ = first_iteration_span;
  SetCost(total_latency, total_latency);
}

int64_t ControlNode::GetLastIterationOffset() const {
  if (GetLatency() == 0) {
    return 0;
  }
  ICHECK_GE(GetLatency(), first_iteration_span_);
  return GetLatency() - first_iteration_span_;
}

const TaskAccessInfo& ControlNode::Accesses() const {
  access_view_ = TaskAccessInfo();
  auto append = [this](const TaskAccessInfo& input, bool mask_loop_control) {
    access_view_.reads.insert(access_view_.reads.end(), input.reads.begin(), input.reads.end());
    access_view_.writes.insert(access_view_.writes.end(), input.writes.begin(), input.writes.end());
    access_view_.read_vars.insert(access_view_.read_vars.end(), input.read_vars.begin(),
                                  input.read_vars.end());
    access_view_.write_vars.insert(access_view_.write_vars.end(), input.write_vars.begin(),
                                   input.write_vars.end());
    access_view_.state_reads |= input.state_reads;
    access_view_.state_writes |= input.state_writes;
    if (mask_loop_control) {
      access_view_.state_writes &= ~static_cast<SpecialStateMask>(SpecialState::kLoopControl);
    }
  };
  append(task->Accesses(), false);
  for (const auto& child : children) {
    append(child->Accesses(), true);
  }
  // loop_break is local to this control even when it appears in its header view.
  access_view_.state_writes &= ~static_cast<SpecialStateMask>(SpecialState::kLoopControl);
  return access_view_;
}

RoleMask ControlNode::CandidateRoles() const {
  RoleMask result = task->CandidateRoles();
  for (const auto& child : children) {
    result |= child->CandidateRoles();
  }
  return result;
}

RoleMask ControlNode::ResolvedRoles() const {
  RoleMask result = task->ResolvedRoles();
  for (const auto& child : children) {
    result |= child->ResolvedRoles();
  }
  return result;
}

void RepairParentsAndIndices(std::vector<std::shared_ptr<IRStructure>>& nodes, IRStructure* parent) {
  for (size_t i = 0; i < nodes.size(); ++i) {
    ICHECK(nodes[i] != nullptr) << "Scheduled IR child lists cannot contain null nodes";
    nodes[i]->SetParent(parent);
    nodes[i]->SetIndex(i);
    if (nodes[i]->IsControl()) {
      auto* control = static_cast<ControlNode*>(nodes[i].get());
      ICHECK(control->task != nullptr);
      control->task->SetParent(control);
      control->task->SetIndex(0);
      RepairParentsAndIndices(control->children, control);
    }
  }
}

int CompareIRStructure(const IRStructure* lhs, const IRStructure* rhs) {
  if (lhs == rhs) {
    return 0;
  }
  if (lhs == nullptr) {
    return -2;
  }
  if (rhs == nullptr) {
    return 2;
  }
  auto lhs_path = lhs->PathFrom();
  auto rhs_path = rhs->PathFrom();
  size_t common_size = std::min(lhs_path.size(), rhs_path.size());
  bool same_nodes = true;
  for (size_t i = 0; i < common_size; ++i) {
    same_nodes = same_nodes && lhs_path[i] == rhs_path[i];
    if (lhs_path[i]->GetStage() != rhs_path[i]->GetStage()) {
      return lhs_path[i]->GetStage() < rhs_path[i]->GetStage() ? -1 : 1;
    }
    if (lhs_path[i]->GetIndex() != rhs_path[i]->GetIndex()) {
      return lhs_path[i]->GetIndex() < rhs_path[i]->GetIndex() ? -1 : 1;
    }
  }
  if (lhs_path.size() == rhs_path.size()) {
    return 0;
  }
  if (lhs_path.size() < rhs_path.size()) {
    return same_nodes ? -2 : -1;
  }
  return same_nodes ? 2 : 1;
}

PrimExpr CalculateIterationCount(const ControlNode* loop) {
  DataType i64 = DataType::Int(64);
  PrimExpr total_iter = IntImm(i64, 0);
  PrimExpr total_multiplier = IntImm(i64, 1);
  for (; loop != nullptr; loop = loop->GetParentControl()) {
    const ForNode* control = loop->control.get();
    PrimExpr step = control->step.has_value() ? control->step.value() : IntImm(control->min.dtype(), 1);
    PrimExpr offset = cast(i64, control->loop_var - control->min);
    total_iter = total_iter + indexdiv(offset, cast(i64, step)) * total_multiplier;
    total_multiplier = total_multiplier * ceildiv(cast(i64, control->extent), cast(i64, step));
  }
  return total_iter;
}

void CollectAllTaskNodes(IRStructure* node, std::vector<TaskNode*>* tasks,
                         bool include_control_headers) {
  ICHECK(tasks != nullptr);
  if (node == nullptr) {
    return;
  }
  if (node->IsTask()) {
    tasks->push_back(static_cast<TaskNode*>(node));
    return;
  }
  auto* control = static_cast<ControlNode*>(node);
  if (include_control_headers) {
    tasks->push_back(control->task.get());
  }
  for (const auto& child : control->children) {
    CollectAllTaskNodes(child.get(), tasks, include_control_headers);
  }
}

void CollectAllTaskNodes(const std::vector<std::shared_ptr<IRStructure>>& nodes,
                         std::vector<TaskNode*>* tasks, bool include_control_headers) {
  for (const auto& node : nodes) {
    CollectAllTaskNodes(node.get(), tasks, include_control_headers);
  }
}

void ValidateScheduledTIR(const ScheduledTIR& scheduled) {
  ICHECK(scheduled.metadata.kernel_root.defined())
      << "Scheduled TIR metadata is missing its kernel root shell";
  ValidateBufferVersions(scheduled.metadata.requested_versions, "Requested buffer");
  ValidateBufferVersions(scheduled.metadata.selected_versions, "Selected buffer");
  ValidateVersionModes(scheduled.metadata.version_modes);
  if (scheduled.metadata.has_buffer_aliases) {
    ValidateAliases(scheduled.metadata.buffer_aliases);
  }
  std::unordered_set<const IRStructure*> seen;
  for (size_t i = 0; i < scheduled.tree.size(); ++i) {
    ValidateNode(scheduled.tree[i].get(), nullptr, i, &seen);
  }
}

ScheduledTIR DecodeScheduledTIR(const SBlock& root, const ConstrSet& outer_ctx,
                                ScheduledExtraInfoFactory make_extra_info) {
  ICHECK(root.defined());
  ScheduledTIR scheduled;
  scheduled.tree = ScheduledTIRDecoder(outer_ctx, make_extra_info).Decode(root->body);
  scheduled.metadata.kernel_root = root;
  const auto& annotations = root->annotations;
  if (auto value = annotations.Get(kRequestedBufferVersions); value.has_value()) {
    scheduled.metadata.requested_versions = value.value().cast<BufferVersionMap>();
  }
  if (auto value = annotations.Get(kSelectedBufferVersions); value.has_value()) {
    scheduled.metadata.selected_versions = value.value().cast<BufferVersionMap>();
  }
  if (auto value = annotations.Get(kBufferVersionModes); value.has_value()) {
    scheduled.metadata.version_modes = value.value().cast<ffi::Map<Var, ffi::String>>();
  }
  if (auto value = annotations.Get(kBufferAliases); value.has_value()) {
    scheduled.metadata.buffer_aliases = value.value().cast<BufferAliasMap>();
    scheduled.metadata.has_buffer_aliases = true;
  }
  ValidateScheduledTIR(scheduled);
  return scheduled;
}

SBlock EncodeScheduledTIR(ScheduledTIR scheduled) {
  ValidateScheduledTIR(scheduled);
  Stmt body = ScheduledTIREncoder().Encode(scheduled.tree);
  SBlock rewritten = scheduled.metadata.kernel_root;
  SBlockNode* node = rewritten.CopyOnWrite();
  node->body = std::move(body);
  Metadata annotations = node->annotations;
  annotations.erase(kRequestedBufferVersions);
  annotations.erase(kSelectedBufferVersions);
  annotations.erase(kBufferVersionModes);
  annotations.erase(kBufferAliases);
  if (!scheduled.metadata.requested_versions.empty()) {
    annotations.Set(kRequestedBufferVersions, std::move(scheduled.metadata.requested_versions));
  }
  if (!scheduled.metadata.selected_versions.empty()) {
    annotations.Set(kSelectedBufferVersions, std::move(scheduled.metadata.selected_versions));
  }
  if (!scheduled.metadata.version_modes.empty()) {
    annotations.Set(kBufferVersionModes, std::move(scheduled.metadata.version_modes));
  }
  if (scheduled.metadata.has_buffer_aliases) {
    ValidateAliases(scheduled.metadata.buffer_aliases);
    annotations.Set(kBufferAliases, std::move(scheduled.metadata.buffer_aliases));
  }
  node->annotations = std::move(annotations);
  return rewritten;
}

}  // namespace tvm::tl::cuda::z3

namespace {

// Internal host-inspection entry points. These do not register or run a CUDA
// transform pass and are never called by the default compilation pipeline.
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  using tvm::tirx::SBlock;
  using tvm::tl::cuda::z3::CloneRoundTrip;
  using tvm::tl::cuda::z3::Inspect;
  using tvm::tl::cuda::z3::MinimumIterationDistanceFFI;
  using tvm::tl::cuda::z3::RoundTrip;
  using tvm::tl::cuda::z3::SubstituteRoundTrip;
  using tvm::tl::cuda::z3::WriteFirstLattice;

  refl::GlobalDef().def("tl.cuda.transform.Z3ScheduleIRRoundTrip", RoundTrip);
  refl::GlobalDef().def("tl.cuda.transform.Z3ScheduleIRCloneRoundTrip", CloneRoundTrip);
  refl::GlobalDef().def("tl.cuda.transform.Z3ScheduleIRSubstituteRoundTrip", SubstituteRoundTrip);
  refl::GlobalDef().def("tl.cuda.transform.Z3ScheduleIRInspect", Inspect);
  refl::GlobalDef().def("tl.cuda.transform.Z3WriteFirstLattice", WriteFirstLattice);
  refl::GlobalDef().def("tl.cuda.transform.Z3MinimumIterationDistance", MinimumIterationDistanceFFI);
}

}  // namespace
