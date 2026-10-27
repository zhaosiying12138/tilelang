/*!
 * \file unroll_loop_skip_z3_atomic.cc
 * \brief Explicit-only loop unrolling outside CUDA Z3 atomic regions.
 */

#include "support/check.h"

#include <limits>
#include <utility>

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/cast.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include "../op/builtin.h"
#include "../op/operator.h"
#include "common/z3_schedule_hints.h"
#include "tir/transforms/ir_utils.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace {

class TargetedLoopBreakDetector : public StmtExprVisitor {
 public:
  static bool Contains(const Stmt &body) {
    TargetedLoopBreakDetector detector;
    detector(body);
    return detector.found_;
  }

 private:
  void VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(tl::loop_break()) ||
        op->op.same_as(builtin::break_loop())) {
      found_ = true;
      return;
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const ForNode *) final {}
  void VisitStmt_(const WhileNode *) final {}

  bool found_{false};
};

// Renames Var and Buffer definitions in each unrolled body copy so every
// iteration gets a fresh SSA identity.
class UnrolledBodyDefFreshener : public StmtExprMutator {
 public:
  PrimExpr VisitExpr_(const VarNode *op) final {
    Var var = GetRef<Var>(op);
    auto it = var_remap_.find(var);
    if (it != var_remap_.end()) return (*it).second;
    return var;
  }

  Buffer VisitBufferDef(const Buffer &buffer, bool alloc_data) final {
    Var data;
    if (alloc_data) {
      data = FreshVar(buffer->data);
    } else {
      PrimExpr remapped_data = VisitExpr(buffer->data);
      ICHECK(remapped_data.as<VarNode>())
          << "Buffer data must remain a Var after freshening definitions";
      data = Downcast<Var>(remapped_data);
    }

    auto visit_expr = [this](const PrimExpr &expr) {
      return this->VisitExpr(expr);
    };
    Buffer new_buffer = buffer;
    auto writer = new_buffer.CopyOnWrite();
    writer->data = std::move(data);
    writer->shape = buffer->shape.Map(visit_expr);
    writer->strides = buffer->strides.Map(visit_expr);
    writer->elem_offset = visit_expr(buffer->elem_offset);
    buffer_remap_.Set(buffer, new_buffer);
    return new_buffer;
  }

 private:
  Var FreshVar(const Var &var) {
    Var new_var = Var(make_object<VarNode>(*var.get()));
    var_remap_.Set(var, new_var);
    return new_var;
  }

  Map<Var, Var> var_remap_;
};

class ExplicitLoopUnroller : public StmtExprMutator {
 public:
  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (IsZ3AtomicAttrKey(op->attr_key)) return GetRef<Stmt>(op);
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const EvaluateNode *op) final {
    // Tile operators own their internal region/root representation. In
    // particular, a ParallelOp root is one scheduler task, not an unroll site.
    if (const auto *call = op->value.as<CallNode>()) {
      if (ParseOperator(GetRef<Call>(call)).defined()) {
        return GetRef<Stmt>(op);
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const ForNode *op) final {
    // Hand-built host TIR uses a plain parallel For where the frontend may
    // later wrap a ParallelOp. Both forms are one atomic scheduling region.
    if (op->kind == ForKind::kParallel ||
        op->kind == ForKind::kVectorized ||
        op->kind == ForKind::kThreadBinding) {
      return GetRef<Stmt>(op);
    }

    // Post-order traversal materializes explicit requests nested under retained
    // serial/unrolled loops without enabling generic auto-unrolling.
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    const auto *loop = stmt.as<ForNode>();
    ICHECK(loop != nullptr);
    if (!IsExplicitUnroll(loop)) return stmt;
    return Unroll(loop);
  }

 private:
  static bool IsExplicitUnroll(const ForNode *op) {
    if (op->kind != ForKind::kUnrolled) return false;
    auto it = op->annotations.find(tirx::attr::pragma_unroll_explicit);
    if (it == op->annotations.end()) return false;
    if (auto value = (*it).second.as<bool>()) return value.value();
    if (auto value = (*it).second.as<Bool>()) return value.value();
    if (const auto *value = (*it).second.as<IntImmNode>()) {
      ICHECK(value->value == 0 || value->value == 1)
          << "pragma_unroll_explicit must be a boolean";
      return value->value != 0;
    }
    ICHECK(false) << "pragma_unroll_explicit must be a boolean";
    return false;
  }

  int64_t GetStep(const ForNode *op) const {
    if (!op->step.defined()) return 1;
    const int64_t *step = as_const_int(op->step.value());
    ICHECK(step != nullptr)
        << "T.unroll(..., explicit=True) requires a constant loop step";
    ICHECK_GT(*step, 0)
        << "T.unroll(..., explicit=True) requires a positive loop step";
    return *step;
  }

  int GetTripCount(const ForNode *op) {
    PrimExpr extent_expr = analyzer_.Simplify(op->extent);
    const auto *extent = extent_expr.as<IntImmNode>();
    ICHECK(extent != nullptr)
        << "T.unroll(..., explicit=True) requires a constant loop extent";
    ICHECK_GE(extent->value, 0)
        << "T.unroll(..., explicit=True) requires a non-negative loop extent";
    ICHECK_LE(extent->value, std::numeric_limits<int>::max())
        << "T.unroll(..., explicit=True) loop extent is too large";
    if (extent->value == 0) return 0;
    int64_t trips = 1 + (extent->value - 1) / GetStep(op);
    ICHECK_LE(trips, std::numeric_limits<int>::max());
    return static_cast<int>(trips);
  }

  Stmt Unroll(const ForNode *op) {
    CHECK(!TargetedLoopBreakDetector::Contains(op->body), ValueError)
        << "[TileLang Semantic Check] A loop containing T.loop_break() that "
           "targets that loop cannot be fully expanded. Use "
           "T.unroll(..., explicit=False) or T.serial(...) to preserve the "
           "loop.";
    int trips = GetTripCount(op);
    if (trips == 0) return Evaluate(0);
    int64_t step = GetStep(op);

    Map<Var, PrimExpr> substitutions;
    Array<Stmt> unrolled;
    for (int i = 0; i < trips; ++i) {
      PrimExpr iteration =
          op->min + make_const(op->loop_var.dtype(),
                               static_cast<int64_t>(i) * step);
      substitutions.Set(op->loop_var, iteration);
      Stmt body = Substitute(op->body, substitutions);
      body = UnrolledBodyDefFreshener()(std::move(body));
      unrolled.push_back(body);
    }
    return SeqStmt::Flatten(unrolled);
  }

  arith::Analyzer analyzer_;
};

Stmt UnrollLoopSkipZ3Atomic(Stmt stmt) {
  Stmt result = ExplicitLoopUnroller()(stmt);
  return result.same_as(stmt) ? result : ConvertSSA(result);
}

}  // namespace

namespace transform {

using namespace tirx::transform;

Pass UnrollLoopSkipZ3Atomic() {
  auto pass_func = [](PrimFunc func, const IRModule &, PassContext) {
    Stmt body = tl::UnrollLoopSkipZ3Atomic(func->body);
    if (body.same_as(func->body)) return func;
    func.CopyOnWrite()->body = std::move(body);
    return func;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.UnrollLoopSkipZ3Atomic", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = reflection;
  refl::GlobalDef().def("tl.transform.UnrollLoopSkipZ3Atomic",
                        UnrollLoopSkipZ3Atomic);
}

}  // namespace transform
}  // namespace tl
}  // namespace tvm
