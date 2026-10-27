/*!
 * \file bind_utils.h
 * \brief Shared helpers for dependency-closed scalar Bind analysis.
 */
#ifndef TVM_TL_TRANSFORM_COMMON_BIND_UTILS_H_
#define TVM_TL_TRANSFORM_COMMON_BIND_UTILS_H_

#include <tvm/arith/analyzer.h>
#include <tvm/ir/op.h>
#include <tvm/ir/type.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>

#include "pipeline_utils.h"

#include <algorithm>
#include <deque>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

using BufferSet =
    std::unordered_set<Buffer, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>;
using ScalarVarSet =
    std::unordered_set<Var, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>;

/*!
 * \brief Consumer-site scalar clone metadata emitted by Z3PipelinePlanning.
 *
 * kPipelineScalarClonePlan is an Array<Map<String, Any>>. Each entry contains
 * clone_index, clone_task, definition_task, definition_stmt, definition_var,
 * value, consumer_task, consumer_stmt, consumer_order, stage, start,
 * schedule_position, consumer_schedule_position, scalar_inputs, and
 * virtual_reads. clone_task and
 * scalar_inputs use augmented task ids; ids below the original task count are
 * original tasks, and larger ids index this plan. definition_stmt/consumer_stmt
 * index the flattened original body. scalar_inputs omit replayable definitions,
 * which the ordinary replay expansion still handles. schedule_position is the
 * topological position in the full augmented schedule, not an original-length
 * software_pipeline_order value.
 */
static constexpr const char *kPipelineScalarClonePlan =
    "tl_pipeline_scalar_clone_plan";
static constexpr const char *kPipelineScalarCloneOriginalTaskCount =
    "tl_pipeline_scalar_clone_original_task_count";
static constexpr const char *kPipelineScalarCloneOriginalFlatIndices =
    "tl_pipeline_scalar_clone_original_flat_indices";
static constexpr const char *kZ3AuditCloneRounds =
    "tl_z3_pipeline_audit_clone_rounds";

/*! \brief Whether a Bind has a scalar, non-pointer result that can be cloned. */
inline bool HasCloneableScalarBindType(const BindNode *bind) {
  if (bind == nullptr || !bind->var.dtype().is_scalar() ||
      bind->var.dtype().is_handle() ||
      bind->var->type_annotation.as<PointerTypeNode>()) {
    return false;
  }
  return bind->value.dtype().is_scalar() && !bind->value.dtype().is_handle();
}

/*!
 * \brief Prove that duplicating an expression only duplicates pure arithmetic
 * and explicitly represented BufferLoad operations.
 *
 * SideEffect alone is insufficient here: an opaque read-state call (for
 * example a clock or RNG intrinsic) has the same kReadState kind as an ordinary
 * BufferLoad. Inspect every call's own effect attribute as well. Extern and
 * packed calls are conservatively kept single-evaluation even when annotated
 * pure, since their C ABI contract is not part of pipeline dependency analysis.
 */
inline bool IsProvenDuplicatableScalarValue(const PrimExpr &value) {
  if (SideEffect(value) > CallEffectKind::kReadState) {
    return false;
  }
  static auto call_effects =
      Op::GetAttrMap<TCallEffectKind>("TCallEffectKind");
  bool duplicatable = true;
  PostOrderVisit(value, [&](const ObjectRef &node) {
    if (!duplicatable) return;
    if (const auto *var = node.as<VarNode>()) {
      if (var->dtype.is_handle() ||
          var->type_annotation.as<PointerTypeNode>()) {
        duplicatable = false;
      }
      return;
    }
    const auto *call = node.as<CallNode>();
    if (call == nullptr) return;
    auto op = call->op.as<Op>();
    if (!op.has_value()) {
      duplicatable = false;
      return;
    }
    if (static_cast<CallEffectKind>(call_effects[op.value()]->value) !=
        CallEffectKind::kPure) {
      // kExprAnnotation is also not accepted: cloning must not duplicate an
      // annotation whose lowering contract is unknown to this analysis.
      duplicatable = false;
      return;
    }
    std::string name = op.value()->name;
    if (name.find("extern") != std::string::npos ||
        name.find("packed") != std::string::npos ||
        name.find("atomic") != std::string::npos ||
        name.find("random") != std::string::npos ||
        name.find("rng") != std::string::npos) {
      duplicatable = false;
    }
  });
  return duplicatable;
}

/*! \brief Whether a scalar Bind can be recomputed at a consumer site. */
inline bool IsCloneableScalarBind(const Stmt &stmt) {
  const auto *bind = stmt.as<BindNode>();
  return HasCloneableScalarBindType(bind) &&
         IsProvenDuplicatableScalarValue(bind->value);
}

inline void AppendFlatScalarStatements(const Stmt &body, Array<Stmt> *output) {
  if (const auto *seq = body.as<SeqStmtNode>()) {
    for (const Stmt &child : seq->seq) {
      AppendFlatScalarStatements(child, output);
    }
  } else if (body.defined()) {
    output->push_back(body);
  }
}

inline bool IsFlatScalarBindChain(const Stmt &body) {
  Array<Stmt> statements;
  AppendFlatScalarStatements(body, &statements);
  if (statements.empty()) {
    return false;
  }
  for (const Stmt &stmt : statements) {
    if (stmt.as<BindNode>() == nullptr) {
      return false;
    }
  }
  return true;
}

inline const BindNode *TrailingFlatScalarBind(const Stmt &body) {
  if (!IsFlatScalarBindChain(body)) {
    return nullptr;
  }
  Array<Stmt> statements;
  AppendFlatScalarStatements(body, &statements);
  return statements.back().as<BindNode>();
}

inline bool IsReplayableScalarBind(const Stmt &stmt,
                                   const ffi::Array<BufferRegion> &reads,
                                   const BufferSet &write_buffers) {
  const auto *bind = stmt.as<BindNode>();
  if (!IsCloneableScalarBind(stmt)) {
    return false;
  }
  // A bind value that carries a side effect (e.g. an atomic RMW returning the
  // previous value, or any opaque call) must be materialized exactly once.
  // Replaying read-state with no complete BufferLoad contract is rejected by
  // IsProvenDuplicatableScalarValue as well. Known read-only values remain
  // replayable only when their storage is stable across the whole pipeline.
  for (const BufferRegion &read : reads) {
    for (const Buffer &write : write_buffers) {
      // Distinct Buffer views can mutate the same allocation. Region
      // coordinates are not needed for stability: any write to the storage
      // makes replay at a later program point unsafe.
      if (read->buffer->data.same_as(write->data)) {
        return false;
      }
    }
  }
  return true;
}

/*! \brief Whether an expression contains no Var other than the given loop Var. */
inline bool PipelineExprOnlyUsesVar(const PrimExpr &expr, const Var &loop_var) {
  bool supported = true;
  PostOrderVisit(expr, [&](const ObjectRef &node) {
    if (const auto *var = node.as<VarNode>()) {
      supported = supported && GetRef<Var>(var).same_as(loop_var);
    }
  });
  return supported;
}

/*! \brief Conservative proof that two regions at distinct iterations are disjoint. */
inline bool PipelineDistinctIterationsDisjoint(const BufferRegion &left,
                                               const BufferRegion &right,
                                               const ForNode *loop) {
  if (!left->buffer->data.same_as(right->buffer->data)) return true;
  // Region coordinates are view-relative. Without a physical layout proof,
  // distinct views of one allocation must conservatively remain conflicting.
  if (!left->buffer.same_as(right->buffer)) return false;
  if (!left->buffer->strides.empty() || !right->buffer->strides.empty() ||
      left->region.size() != right->region.size()) {
    return false;
  }
  arith::Analyzer analyzer;
  Var delta("z3_distinct_iteration_delta", loop->loop_var.dtype());
  ffi::Map<Var, PrimExpr> shift;
  shift.Set(loop->loop_var, loop->loop_var + delta);
  auto is_zero = [](const PrimExpr &expr) {
    const int64_t *value = as_const_int(expr);
    return value != nullptr && *value == 0;
  };
  auto floor_div = [](__int128 value, __int128 divisor) {
    ICHECK(divisor > 0);
    __int128 quotient = value / divisor;
    if (value % divisor < 0) --quotient;
    return quotient;
  };
  for (size_t dim = 0; dim < left->region.size(); ++dim) {
    const Range &a = left->region[dim];
    const Range &b = right->region[dim];
    const int64_t *a_extent = as_const_int(analyzer.Simplify(a->extent));
    const int64_t *b_extent = as_const_int(analyzer.Simplify(b->extent));
    if (a_extent == nullptr || b_extent == nullptr || *a_extent < 0 ||
        *b_extent < 0 ||
        !PipelineExprOnlyUsesVar(a->min, loop->loop_var) ||
        !PipelineExprOnlyUsesVar(b->min, loop->loop_var)) {
      return false;
    }
    if (*a_extent == 0 || *b_extent == 0) return true;
    PrimExpr unshifted = analyzer.Simplify(b->min - a->min);
    bool contains_loop_var = false;
    PostOrderVisit(unshifted, [&](const ObjectRef &node) {
      if (const auto *var = node.as<VarNode>()) {
        contains_loop_var = contains_loop_var ||
                            GetRef<Var>(var).same_as(loop->loop_var);
      }
    });
    if (contains_loop_var) return false;
    const int64_t *offset = as_const_int(unshifted);
    if (offset == nullptr) return false;

    PrimExpr shifted = tirx::Substitute(b->min, shift);
    PrimExpr slope = analyzer.Simplify(shifted - b->min);
    if (!PipelineExprOnlyUsesVar(slope, delta)) return false;
    ffi::Map<Var, PrimExpr> delta_zero;
    delta_zero.Set(delta, IntImm(delta.dtype(), 0));
    ffi::Map<Var, PrimExpr> delta_one;
    delta_one.Set(delta, IntImm(delta.dtype(), 1));
    if (!is_zero(analyzer.Simplify(tirx::Substitute(slope, delta_zero)))) {
      return false;
    }
    const int64_t *coefficient =
        as_const_int(analyzer.Simplify(tirx::Substitute(slope, delta_one)));
    if (coefficient == nullptr ||
        !is_zero(analyzer.Simplify(
            slope - delta * IntImm(delta.dtype(), *coefficient)))) {
      return false;
    }

    __int128 c = *coefficient;
    __int128 offset_value = *offset;
    __int128 a_size = *a_extent;
    __int128 b_size = *b_extent;
    if (c == 0) {
      if (offset_value <= -b_size || offset_value >= a_size) return true;
      continue;
    }
    // After translating the left minimum to zero, overlap requires
    // -b_size < c * nonzero_delta + offset < a_size. Testing the first
    // integer in that interval, plus the neighbors of excluded delta=0,
    // proves disjointness without mistaking distance one for all distances.
    __int128 lower = -b_size - offset_value;
    __int128 upper = a_size - offset_value;
    if (c < 0) {
      __int128 old_lower = lower;
      lower = -upper;
      upper = -old_lower;
    }
    __int128 step = c < 0 ? -c : c;
    __int128 first = floor_div(lower, step) + 1;
    bool possible_overlap = false;
    for (__int128 candidate : {first, __int128(-1), __int128(1)}) {
      if (candidate == 0) continue;
      __int128 difference = c * candidate + offset_value;
      if (difference > -b_size && difference < a_size) {
        possible_overlap = true;
        break;
      }
    }
    if (!possible_overlap) return true;
  }
  return false;
}

/*! \brief Region conflict check shared by original and virtual clone reads. */
inline bool PipelineLoopRegionsConflict(
    const BufferRegion &left, const BufferRegion &right, const ForNode *loop,
    bool cross_iteration,
    const ffi::Map<Var, arith::IntSet> &thread_domains) {
  if (cross_iteration) {
    return !PipelineDistinctIterationsDisjoint(left, right, loop);
  }
  return PipelineRegionsMayConflict(left, right, thread_domains);
}

/*! \brief A direct scalar use-def edge between flattened statements. */
struct ScalarUseDefEdge {
  size_t definition;
  size_t use;
};

/*! \brief Var-identity scalar graph and dependency-closed Bind classification. */
struct ScalarBindAnalysis {
  std::vector<char> replayable_binds;
  std::vector<char> cloneable_binds;
  std::vector<ScalarVarSet> scalar_defs;
  std::vector<ScalarVarSet> scalar_uses;
  std::unordered_map<Var, size_t, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>
      scalar_def_stmt;
  std::vector<ScalarUseDefEdge> edges;
};

class ScalarUseDefCollector : public StmtExprVisitor {
 public:
  static std::pair<ScalarVarSet, ScalarVarSet> Collect(const Stmt &stmt) {
    ScalarUseDefCollector collector;
    collector(stmt);
    return {std::move(collector.scalar_defs_),
            std::move(collector.scalar_uses_)};
  }

 private:
  void VisitStmt_(const BindNode *op) final {
    this->VisitExpr(op->value);
    scalar_defs_.insert(op->var);
  }

  void VisitExpr_(const VarNode *op) final {
    scalar_uses_.insert(GetRef<Var>(op));
  }

  ScalarVarSet scalar_defs_;
  ScalarVarSet scalar_uses_;
};

/*!
 * \brief Build the scalar graph and close replayability over scheduled defs.
 *
 * A locally pure Bind is not independently replayable when its value refers to
 * a scheduled Bind: replay would reproduce only the Var use, not the defining
 * evaluation. Propagate non-replayability through the complete Let/Bind chain
 * using object identity, never variable names. The resulting mask is shared by
 * task inference, Z3 planning, and stock pipeline planning.
 */
inline ScalarBindAnalysis AnalyzeScalarBinds(
    const ffi::Array<Stmt> &flat_stmts,
    const std::vector<ffi::Array<BufferRegion>> &reads,
    const BufferSet &pipeline_write_buffers) {
  ICHECK_EQ(reads.size(), flat_stmts.size());
  ScalarBindAnalysis analysis;
  const size_t n = flat_stmts.size();
  analysis.replayable_binds.assign(n, 0);
  analysis.cloneable_binds.assign(n, 0);
  analysis.scalar_defs.reserve(n);
  analysis.scalar_uses.reserve(n);

  std::deque<size_t> scheduled_definitions;
  for (size_t i = 0; i < n; ++i) {
    auto collected = ScalarUseDefCollector::Collect(flat_stmts[i]);
    analysis.scalar_defs.push_back(std::move(collected.first));
    analysis.scalar_uses.push_back(std::move(collected.second));
    const auto *bind = flat_stmts[i].as<BindNode>();
    if (bind == nullptr) continue;
    auto inserted = analysis.scalar_def_stmt.emplace(bind->var, i);
    ICHECK(inserted.second)
        << "Duplicate scalar Bind definition for Var " << bind->var
        << "; cross-stage scalar analysis requires unique Var identity";
    analysis.cloneable_binds[i] = IsCloneableScalarBind(flat_stmts[i]) ? 1 : 0;
    analysis.replayable_binds[i] =
        IsReplayableScalarBind(flat_stmts[i], reads[i],
                               pipeline_write_buffers)
            ? 1
            : 0;
    if (!analysis.replayable_binds[i]) scheduled_definitions.push_back(i);
  }

  std::vector<std::vector<size_t>> uses_by_definition(n);
  for (size_t use = 0; use < n; ++use) {
    for (const Var &var : analysis.scalar_uses[use]) {
      auto definition = analysis.scalar_def_stmt.find(var);
      if (definition == analysis.scalar_def_stmt.end()) continue;
      ICHECK_NE(definition->second, use)
          << "Cyclic scalar Bind definition involving Var " << var;
      analysis.edges.push_back({definition->second, use});
      uses_by_definition[definition->second].push_back(use);
    }
  }

  std::sort(analysis.edges.begin(), analysis.edges.end(),
            [](const ScalarUseDefEdge &left, const ScalarUseDefEdge &right) {
              return std::tie(left.definition, left.use) <
                     std::tie(right.definition, right.use);
            });
  while (!scheduled_definitions.empty()) {
    size_t definition = scheduled_definitions.front();
    scheduled_definitions.pop_front();
    for (size_t use : uses_by_definition[definition]) {
      if (flat_stmts[use].as<BindNode>() == nullptr ||
          !analysis.replayable_binds[use]) {
        continue;
      }
      analysis.replayable_binds[use] = 0;
      scheduled_definitions.push_back(use);
    }
  }
  return analysis;
}

}  // namespace tl
}  // namespace tvm

#endif  // TVM_TL_TRANSFORM_COMMON_BIND_UTILS_H_
