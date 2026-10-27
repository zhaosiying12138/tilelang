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
 * \file z3_normalize_control_flow.cc
 * \brief Normalize recursive CUDA Z3 control flow before task materialization.
 */

#include <tvm/ffi/container/map.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>

#include "./z3_schedule_ir.h"
#include "op/builtin.h"

namespace tvm::tl::cuda {

using namespace tirx;

namespace {

constexpr const char* kZ3CudaTaskAttr = "tl.cuda_z3_task";

bool HasBufferRead(const PrimExpr& expr) {
  bool found = false;
  PostOrderVisit(Evaluate(expr), [&](const ffi::ObjectRef& node) {
    found = found || node.as<BufferLoadNode>() != nullptr ||
            node.as<ProducerLoadNode>() != nullptr;
  });
  return found;
}

class ControlFlowNormalizer final : public StmtExprMutator {
 public:
  static Stmt Rewrite(const Stmt& body) { return ControlFlowNormalizer()(body); }

 private:
  static int64_t Gcd(int64_t a, int64_t b) {
    while (b != 0) {
      int64_t next = a % b;
      a = b;
      b = next;
    }
    return a;
  }

  static int64_t MaxNumStages(const Stmt& stmt) {
    int64_t max_stages = 1;
    PostOrderVisit(stmt, [&](const ffi::ObjectRef& node) {
      const auto* loop = node.as<ForNode>();
      if (loop == nullptr) {
        return;
      }
      auto value = loop->annotations.Get("num_stages");
      if (!value.has_value()) {
        return;
      }
      PrimExpr stage_expr = value.value().cast<PrimExpr>();
      const auto* stages = stage_expr.as<IntImmNode>();
      ICHECK(stages != nullptr) << "num_stages must be a constant integer";
      ICHECK_GE(stages->value, 0) << "num_stages must be non-negative";
      max_stages = std::max(max_stages, stages->value);
    });
    return max_stages;
  }

  static int64_t SyntheticWhileExtent(const Stmt& body) {
    int64_t extent = 1;
    int64_t max_stages = MaxNumStages(body);
    for (int64_t factor = 1; factor <= max_stages; ++factor) {
      int64_t reduced = extent / Gcd(extent, factor);
      ICHECK(factor == 0 || reduced <= std::numeric_limits<int64_t>::max() / factor)
          << "Synthetic While bound overflows int64 while computing lcm(1.."
          << max_stages << ")";
      extent = reduced * factor;
    }
    return std::max<int64_t>(extent, 2);
  }

  Stmt VisitStmt_(const WhileNode* op) final {
    Var loop_var("__while_w_" + std::to_string(while_counter_++), DataType::Int(64));
    Stmt visited_body = VisitStmt(op->body);
    int64_t extent = SyntheticWhileExtent(visited_body);

    Stmt body = visited_body;
    if (!is_one(op->condition)) {
      // Route the guard through the If visitor so its condition is snapshotted
      // before either the break or any body mutation can execute.
      Stmt guard = VisitStmt(IfThenElse(
          logical_not(op->condition),
          Evaluate(Call(DataType::Handle(), tl::loop_break(), {}))));
      body = SeqStmt({guard, visited_body});
    }
    ffi::Map<ffi::String, ffi::Any> annotations;
    annotations.Set("synthetic_while", Integer(1));
    // No num_stages annotation: the synthetic loop itself is never pipelined.
    return For(loop_var, IntImm(DataType::Int(64), 0), IntImm(DataType::Int(64), extent),
               ForKind::kSerial, body, /*thread_binding=*/std::nullopt, annotations);
  }

  Stmt VisitStmt_(const IfThenElseNode* op) final {
    PrimExpr condition = VisitExpr(op->condition);
    Stmt then_case = VisitStmt(op->then_case);
    ffi::Optional<Stmt> else_case = op->else_case;
    if (else_case.defined()) {
      else_case = VisitStmt(else_case.value());
    }
    if (condition.as<VarNode>()) {
      return IfThenElse(condition, then_case, else_case, op->span);
    }

    Var snapshot("__cond_" + std::to_string(condition_counter_++), DataType::Bool());
    Stmt branch = IfThenElse(snapshot, then_case, else_case, op->span);
    return SeqStmt({tirx::Bind(snapshot, condition), branch});
  }

  Stmt VisitStmt_(const SeqStmtNode* op) final {
    ffi::Array<Stmt> sequence;
    for (const Stmt& child : op->seq) {
      Stmt rewritten = VisitStmt(child);
      if (!rewritten.defined()) {
        continue;
      }
      if (const auto* nested = rewritten.as<SeqStmtNode>()) {
        sequence.insert(sequence.end(), nested->seq.begin(), nested->seq.end());
      } else {
        sequence.push_back(rewritten);
      }
    }
    return SeqStmt::Flatten(sequence);
  }

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    // Atomic tasks, including the legacy flat schema, are opaque to recursive
    // control normalization. In particular, a While in one of these leaves is
    // emitted unchanged rather than becoming a second scheduler Control.
    if (op->attr_key == z3::kCudaTask || op->attr_key == z3::kCudaPerRoleTask ||
        op->attr_key == z3::kScheduleUnit || op->attr_key == kZ3CudaTaskAttr) {
      return ffi::GetRef<Stmt>(op);
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const ForNode* op) final {
    if (op->kind != ForKind::kSerial && op->kind != ForKind::kUnrolled) {
      return ffi::GetRef<For>(op);
    }

    PrimExpr min = VisitExpr(op->min);
    PrimExpr extent = VisitExpr(op->extent);
    ffi::Optional<PrimExpr> step = op->step;
    if (step.defined()) {
      step = VisitExpr(step.value());
    }
    Stmt body = VisitStmt(op->body);

    ffi::Array<Stmt> prefix;
    auto snapshot_buffer_read = [&](PrimExpr expr) {
      if (!HasBufferRead(expr)) {
        return expr;
      }
      Var snapshot("__loop_bound_" + std::to_string(loop_bound_counter_++), expr.dtype());
      prefix.push_back(tirx::Bind(snapshot, expr));
      return PrimExpr(snapshot);
    };
    min = snapshot_buffer_read(std::move(min));
    extent = snapshot_buffer_read(std::move(extent));
    if (step.defined()) {
      step = snapshot_buffer_read(std::move(step.value()));
    }

    For loop = ffi::GetRef<For>(op);
    ForNode* writer = loop.CopyOnWrite();
    writer->min = std::move(min);
    writer->extent = std::move(extent);
    writer->step = std::move(step);
    writer->body = std::move(body);
    if (prefix.empty()) {
      return loop;
    }
    prefix.push_back(loop);
    return SeqStmt::Flatten(prefix);
  }

  Stmt VisitStmt_(const SBlockNode* op) final {
    // Non-root blocks are opaque task leaves. The root rewriter invokes this
    // visitor directly on tilelang_root.body, so the root itself is not skipped.
    return ffi::GetRef<SBlock>(op);
  }

  int condition_counter_{0};
  int while_counter_{0};
  int loop_bound_counter_{0};
};

class KernelRootRewriter final : public StmtExprMutator {
 public:
  static PrimFunc Rewrite(PrimFunc func) {
    KernelRootRewriter rewriter;
    func.CopyOnWrite()->body = rewriter(func->body);
    return func;
  }

 private:
  Stmt VisitStmt_(const SBlockNode* op) final {
    if (op->name_hint != "tilelang_root") {
      return StmtExprMutator::VisitStmt_(op);
    }
    SBlock root = ffi::GetRef<SBlock>(op);
    root.CopyOnWrite()->body = ControlFlowNormalizer::Rewrite(root->body);
    return root;
  }
};

}  // namespace

tvm::transform::Pass NormalizeZ3ControlFlow() {
  using namespace tirx::transform;
  auto pass_func = [](PrimFunc func, const IRModule&, const PassContext&) {
    return KernelRootRewriter::Rewrite(std::move(func));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.cuda.NormalizeZ3ControlFlow", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.cuda.transform.NormalizeZ3ControlFlow", NormalizeZ3ControlFlow);
}

}  // namespace tvm::tl::cuda
