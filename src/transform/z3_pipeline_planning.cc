/*!
 * \file z3_pipeline_planning.cc
 * \brief Opt-in adapter from explicit CUDA loop problems to pipeline IR.
 */

#include <algorithm>
#include <array>
#include <cstdint>
#include <deque>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/arith/analyzer.h>
#include <tvm/ir/cast.h>
#include <tvm/runtime/logging.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include "../op/builtin.h"
#include "../op/copy.h"
#include "../op/operator.h"
#include "../op/parallel.h"
#include "../op/utils.h"
#include "../cuda/transform/z3_buffer_ownership.h"
#include "../cuda/transform/z3_dependency_analysis.h"
#include "../cuda/transform/z3_schedule_ir.h"
#include "common/bind_utils.h"
#include "common/pipeline_utils.h"
#include "common/z3_buffer_versions.h"
#include "common/z3_schedule_hints.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace {

constexpr const char *kZ3PipelineConfig = "tl.enable_z3_pipeline_planning";
constexpr const char *kAutoWarpSpecializationConfig =
    "tl.enable_auto_warp_specialization";
constexpr const char *kZ3Pipeline = "tl_z3_pipeline";
constexpr const char *kZ3Latencies = "tl_z3_pipeline_latencies";
constexpr const char *kZ3IIs = "tl_z3_pipeline_iis";
constexpr const char *kZ3ResourceFlags = "tl_z3_pipeline_resource_flags";
constexpr const char *kZ3DataDeps = "tl_z3_pipeline_data_deps";
constexpr const char *kZ3ResourceDeps = "tl_z3_pipeline_resource_deps";
constexpr const char *kZ3OwnerExclusionDeps =
    "tl_z3_pipeline_owner_exclusion_deps";
constexpr const char *kZ3PipeOrderDeps = "tl_z3_pipeline_pipe_order_deps";
constexpr const char *kZ3StageOrderDeps = "tl_z3_pipeline_stage_order_deps";
constexpr const char *kZ3BufferSizes = "tl_z3_pipeline_buffer_sizes";
constexpr const char *kZ3MemoryGroups = "tl_z3_pipeline_memory_groups";
constexpr const char *kZ3BufferVars = "tl_z3_pipeline_buffer_vars";
constexpr const char *kZ3ResourceBits = "tl_z3_pipeline_resource_bits";
constexpr const char *kZ3EnableOffset = "tl_z3_pipeline_enable_offset";
constexpr const char *kZ3ManualStages = "tl_z3_pipeline_manual_stages";
constexpr const char *kZ3ManualSchedule = "tl_z3_pipeline_manual_schedule";
constexpr const char *kZ3AsyncSameStage =
    "tl_z3_pipeline_async_same_stage_deps";
constexpr const char *kZ3Seed = "tl_z3_pipeline_seed";
constexpr const char *kZ3RecalculateBufferVersions =
    "tl_z3_pipeline_recalculate_buffer_versions";
constexpr const char *kZ3Verbose = "tl_z3_pipeline_verbose";
constexpr const char *kZ3DebugDir = "tl_z3_pipeline_debug_dir";
constexpr const char *kZ3AuditII = "tl_z3_pipeline_audit_ii";
constexpr const char *kZ3AuditStarts = "tl_z3_pipeline_audit_starts";
constexpr const char *kZ3MaxCloneRounds =
    "tl_z3_pipeline_max_clone_rounds";
constexpr const char *kZ3ControlTotalLatency =
    "tl_z3_pipeline_total_latency_override";
constexpr const char *kZ3RecursivePrepared = "tl.cuda_z3_recursive_prepared";
constexpr const char *kZ3PreparedClockMap = "tl.cuda_z3_prepared_clock_map";
constexpr const char *kZ3PreparedEpochPlan = "tl.cuda_z3_prepared_buffer_epochs";
constexpr const char *kZ3AuditChildPaths = "tl_z3_pipeline_audit_child_paths";
constexpr const char *kZ3AuditChildStarts = "tl_z3_pipeline_audit_child_starts";
constexpr const char *kZ3AuditChildStages = "tl_z3_pipeline_audit_child_stages";
constexpr const char *kZ3AuditChildResidues = "tl_z3_pipeline_audit_child_residues";
constexpr const char *kZ3AuditFirstSpan = "tl_z3_pipeline_audit_first_span";
constexpr const char *kZ3AuditLastIterOffset =
    "tl_z3_pipeline_audit_last_iter_offset";

const std::vector<const char *> &Z3InputAnnotationKeys() {
  static const std::vector<const char *> keys = {
      kZ3Pipeline,
      kZ3Latencies,
      kZ3IIs,
      kZ3ResourceFlags,
      kZ3DataDeps,
      kZ3ResourceDeps,
      kZ3OwnerExclusionDeps,
      kZ3PipeOrderDeps,
      kZ3StageOrderDeps,
      kZ3BufferSizes,
      kZ3MemoryGroups,
      kZ3BufferVars,
      kZ3ResourceBits,
      kZ3EnableOffset,
      kZ3ManualStages,
      kZ3ManualSchedule,
      kZ3ExecutionDomains,
      kZ3Seed,
      kZ3RecalculateBufferVersions,
      kZ3Verbose,
      kZ3DebugDir,
      kZ3MaxCloneRounds,
      kZ3ControlTotalLatency,
      kZ3BufferVersionRequests,
      kZ3BufferVersionModeRequests,
  };
  return keys;
}

bool IsZ3InputAnnotation(const String &key) {
  for (const char *candidate : Z3InputAnnotationKeys()) {
    if (key == candidate) {
      return true;
    }
  }
  return false;
}

Array<Integer> GetIntegerArray(const Map<String, Any> &annotations,
                               const char *key) {
  auto value = annotations.Get(key);
  if (!value.has_value()) {
    return {};
  }
  return Downcast<Array<Integer>>(value.value());
}

Array<Array<Integer>> GetNestedIntegerArrays(
    const Map<String, Any> &annotations, const char *key) {
  auto value = annotations.Get(key);
  if (!value.has_value()) {
    return {};
  }
  return Downcast<Array<Array<Integer>>>(value.value());
}

int64_t GetInteger(const Map<String, Any> &annotations, const char *key,
                   int64_t default_value) {
  auto value = annotations.Get(key);
  if (!value.has_value()) {
    return default_value;
  }
  const Any &annotation = value.value();
  if (const auto *imm = annotation.as<IntImmNode>(); imm != nullptr) {
    return imm->value;
  }
  // Public recursive annotations can cross FFI as native integers rather than
  // TIR IntImm objects. Keep the established IntImm path for flat planning and
  // reject strings, floats, arrays, and other non-integer payloads.
  auto native = annotation.try_cast<int64_t>();
  ICHECK(native.has_value()) << "Z3PipelinePlanning: " << key
                             << " must be an integer annotation";
  return native.value();
}

bool GetBool(const Map<String, Any> &annotations, const char *key,
             bool default_value) {
  auto value = annotations.Get(key);
  return value.has_value() ? value.value().cast<bool>() : default_value;
}

Array<int64_t> ToInt64(const Array<Integer> &values) {
  Array<int64_t> result;
  result.reserve(values.size());
  for (const Integer &value : values) {
    result.push_back(value->value);
  }
  return result;
}

Array<Array<int64_t>> ToInt64(
    const Array<Array<Integer>> &groups_of_values) {
  Array<Array<int64_t>> result;
  result.reserve(groups_of_values.size());
  for (const Array<Integer> &values : groups_of_values) {
    result.push_back(ToInt64(values));
  }
  return result;
}

class Z3BufferRegionCollector : public StmtExprVisitor {
public:
  static std::pair<Array<BufferRegion>, Array<BufferRegion>>
  Collect(const Stmt &stmt, const Map<Var, Buffer> &buffer_data_to_buffer) {
    SBlock block(/*iter_vars=*/{}, /*reads=*/{}, /*writes=*/{},
                 /*name_hint=*/"", /*body=*/stmt);
    Z3BufferRegionCollector collector(buffer_data_to_buffer);
    collector(block);
    return {collector.reads_, collector.writes_};
  }

private:
  explicit Z3BufferRegionCollector(
      const Map<Var, Buffer> &buffer_data_to_buffer)
      : buffer_data_to_buffer_(buffer_data_to_buffer) {}

  void HandleTileOp(const TileOperator &tile_op) {
    if (const auto *parallel = tile_op.as<ParallelOpNode>()) {
      auto nested = Collect(parallel->GetRoot(), buffer_data_to_buffer_);
      reads_.insert(reads_.end(), nested.first.begin(), nested.first.end());
      writes_.insert(writes_.end(), nested.second.begin(), nested.second.end());
      return;
    }
    AccessRegions access = tile_op->GetAccessRegions();
    reads_.insert(reads_.end(), access.reads.begin(), access.reads.end());
    writes_.insert(writes_.end(), access.writes.begin(), access.writes.end());
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    Array<Range> region;
    for (const PrimExpr &index : op->indices) {
      region.push_back(Range::FromMinExtent(index, 1));
    }
    writes_.push_back(BufferRegion(op->buffer, region));
    this->VisitExpr(op->value);
    for (const PrimExpr &index : op->indices) {
      this->VisitExpr(index);
    }
    if (op->predicate.defined()) {
      this->VisitExpr(op->predicate.value());
    }
  }

  void VisitExpr_(const BufferLoadNode *op) final {
    Array<Range> region;
    for (const PrimExpr &index : op->indices) {
      region.push_back(Range::FromMinExtent(index, 1));
    }
    reads_.push_back(BufferRegion(op->buffer, region));
    for (const PrimExpr &index : op->indices) {
      this->VisitExpr(index);
    }
    if (op->predicate.defined()) {
      this->VisitExpr(op->predicate.value());
    }
  }

  void VisitExpr_(const CallNode *op) final {
    if (auto tile_op = ParseOperator(GetRef<Call>(op)); tile_op.defined()) {
      HandleTileOp(tile_op);
      StmtExprVisitor::VisitExpr_(op);
      return;
    }
    if (op->op.same_as(builtin::address_of())) {
      BufferRegion buffer_region;
      if (const auto *load = op->args[0].as<BufferLoadNode>()) {
        buffer_region = BufferRegion::FullRegion(load->buffer);
      } else if (const auto *var_node = op->args[0].as<VarNode>()) {
        auto it = buffer_data_to_buffer_.find(GetRef<Var>(var_node));
        if (it != buffer_data_to_buffer_.end()) {
          buffer_region = BufferRegion::FullRegion((*it).second);
        }
      }
      if (buffer_region.defined()) {
        reads_.push_back(buffer_region);
      }
      return;
    }
    if (op->op.same_as(tl::access_ptr())) {
      ICHECK_EQ(op->args.size(), 3U);
      const auto *load = op->args[0].as<BufferLoadNode>();
      ICHECK(load != nullptr) << "tl.access_ptr base must be a BufferLoad";
      BufferRegion region = BufferRegion::FullRegion(load->buffer);
      int access_mask = GetConservativeAccessMask(op->args[2]);
      if (access_mask & kAccessRead) {
        reads_.push_back(region);
      }
      if (access_mask & kAccessWrite) {
        writes_.push_back(region);
      }
      for (const PrimExpr &index : load->indices) {
        this->VisitExpr(index);
      }
      if (load->predicate.defined()) {
        this->VisitExpr(load->predicate.value());
      }
      this->VisitExpr(op->args[1]);
      this->VisitExpr(op->args[2]);
      return;
    }
    if (op->op.same_as(builtin::tvm_access_ptr())) {
      const auto *buffer_var = op->args[1].as<VarNode>();
      ICHECK(buffer_var != nullptr);
      auto it = buffer_data_to_buffer_.find(GetRef<Var>(buffer_var));
      if (it != buffer_data_to_buffer_.end()) {
        BufferRegion region = BufferRegion::FullRegion((*it).second);
        int access_mask = op->args.size() == 5U
                              ? GetConservativeAccessMask(op->args[4])
                              : kAccessReadWrite;
        if (access_mask & kAccessRead) {
          reads_.push_back(region);
        }
        if (access_mask & kAccessWrite) {
          writes_.push_back(region);
        }
      }
      for (size_t i = 2; i < op->args.size(); ++i) {
        this->VisitExpr(op->args[i]);
      }
      return;
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  Map<Var, Buffer> buffer_data_to_buffer_;
  Array<BufferRegion> reads_;
  Array<BufferRegion> writes_;
};

class Z3SeqStmtFlattener : public StmtFunctor<Array<Stmt>(const Stmt &)> {
public:
  using Base = StmtFunctor<Array<Stmt>(const Stmt &)>;

  static Array<Stmt> Flatten(const Array<Stmt> &stmts) {
    Z3SeqStmtFlattener flattener;
    Array<Stmt> flattened;
    for (const Stmt &stmt : stmts) {
      Array<Stmt> nested = flattener(stmt);
      flattened.insert(flattened.end(), nested.begin(), nested.end());
    }
    return flattened;
  }

  Array<Stmt> VisitStmt(const Stmt &stmt) final {
    if (!stmt.as<SeqStmtNode>()) {
      return Array<Stmt>{stmt};
    }
    return Base::VisitStmt(stmt);
  }

  Array<Stmt> VisitStmt_(const SeqStmtNode *op) final {
    Array<Stmt> flattened;
    for (const Stmt &stmt : op->seq) {
      Array<Stmt> nested = VisitStmt(stmt);
      flattened.insert(flattened.end(), nested.begin(), nested.end());
    }
    return flattened;
  }

  Array<Stmt> VisitStmtDefault_(const Object *) final { return {}; }
};

struct Z3ScheduledStmtAnalysis {
  Array<Stmt> scheduled_stmts;
  std::vector<size_t> scheduled_indices;
  std::vector<int> scheduled_for_flat;
  std::vector<Array<BufferRegion>> reads;
  std::vector<Array<BufferRegion>> writes;
  ScalarBindAnalysis scalar;
};

Z3ScheduledStmtAnalysis AnalyzeScheduledStmts(
    const Array<Stmt> &flat_stmts,
    const Map<Var, Buffer> &buffer_data_to_buffer) {
  BufferSet pipeline_write_buffers;
  std::vector<std::pair<Array<BufferRegion>, Array<BufferRegion>>> accesses;
  accesses.reserve(flat_stmts.size());
  for (const Stmt &stmt : flat_stmts) {
    if (IsPipelineDeclarationStmt(stmt)) {
      accesses.push_back({});
      continue;
    }
    auto access = Z3BufferRegionCollector::Collect(stmt, buffer_data_to_buffer);
    for (const BufferRegion &write : access.second) {
      pipeline_write_buffers.insert(write->buffer);
    }
    accesses.push_back(std::move(access));
  }

  std::vector<Array<BufferRegion>> flat_reads;
  flat_reads.reserve(flat_stmts.size());
  for (const auto &access : accesses) flat_reads.push_back(access.first);
  Z3ScheduledStmtAnalysis analysis;
  analysis.scalar =
      AnalyzeScalarBinds(flat_stmts, flat_reads, pipeline_write_buffers);
  analysis.scheduled_for_flat.assign(flat_stmts.size(), -1);
  for (size_t i = 0; i < flat_stmts.size(); ++i) {
    const Stmt &stmt = flat_stmts[i];
    if (IsPipelineDeclarationStmt(stmt)) {
      continue;
    }
    if (analysis.scalar.replayable_binds[i]) {
      continue;
    }
    analysis.scheduled_for_flat[i] =
        static_cast<int>(analysis.scheduled_stmts.size());
    analysis.scheduled_indices.push_back(i);
    analysis.scheduled_stmts.push_back(stmt);
    analysis.reads.push_back(std::move(accesses[i].first));
    analysis.writes.push_back(std::move(accesses[i].second));
  }
  return analysis;
}

class Z3FlatReorderSafetyVisitor : public StmtExprVisitor {
public:
  bool has_non_buffer_side_effect() const {
    return has_non_buffer_side_effect_;
  }

private:
  void VisitExpr_(const CallNode *op) final {
    Call call = GetRef<Call>(op);
    if (!ParseOperator(call).defined() &&
        SideEffect(call) > CallEffectKind::kReadState) {
      has_non_buffer_side_effect_ = true;
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  bool has_non_buffer_side_effect_{false};
};

bool RegionsUseSameStorage(const Array<BufferRegion> &lhs,
                           const Array<BufferRegion> &rhs) {
  for (const BufferRegion &left : lhs) {
    for (const BufferRegion &right : rhs) {
      if (left->buffer->data.same_as(right->buffer->data)) {
        return true;
      }
    }
  }
  return false;
}

bool MayHaveBufferDependency(const Z3ScheduledStmtAnalysis &analysis,
                             size_t left, size_t right) {
  return RegionsUseSameStorage(analysis.writes[left], analysis.reads[right]) ||
         RegionsUseSameStorage(analysis.writes[left], analysis.writes[right]) ||
         RegionsUseSameStorage(analysis.writes[right], analysis.reads[left]);
}

using Z3DataDependency = std::array<int64_t, 4>;
using Z3TaskPair = std::pair<int64_t, int64_t>;

struct Z3CloneSpec {
  int definition;
  int consumer;
};

struct Z3ScalarUseEdge {
  int definition;
  int use;
};

struct Z3FixedRecurrence {
  int64_t source;
  int64_t target;
  int64_t latency;
  int64_t fixed_versions;
  Var storage;
};

struct Z3ValidatedSchedule {
  Array<int64_t> starts;
  Array<int64_t> versions;
  Array<int64_t> model_versions;
  Array<int64_t> required_versions;
  std::unordered_map<Var, int64_t, ObjectPtrHash, ObjectPtrEqual> fixed_required;
  int64_t ii{0};
  std::vector<int64_t> raw_stages;
  std::vector<int64_t> residues;
};

struct Z3AugmentedProblem {
  size_t original_tasks{0};
  std::vector<Z3CloneSpec> clones;
  std::vector<int64_t> latencies;
  std::vector<int64_t> iis;
  std::vector<int64_t> resource_flags;
  std::vector<int64_t> manual_stages;
  std::vector<Z3DataDependency> data_deps;
  std::vector<Z3FixedRecurrence> fixed_recurrences;
  std::vector<Z3TaskPair> resource_deps;
  std::vector<Z3TaskPair> pipe_order_deps;
  std::vector<Z3TaskPair> stage_order_deps;
  std::vector<Array<BufferRegion>> reads;
  std::vector<Array<BufferRegion>> writes;
};

void AddZ3DataDependency(std::vector<Z3DataDependency> *dependencies,
                         int64_t source, int64_t target, int64_t distance,
                         int64_t latency) {
  ICHECK(source >= 0 && target >= 0 && latency >= 0);
  for (Z3DataDependency &dependency : *dependencies) {
    if (dependency[0] == source && dependency[1] == target &&
        dependency[2] == distance) {
      dependency[3] = std::max(dependency[3], latency);
      return;
    }
  }
  dependencies->push_back({source, target, distance, latency});
}

void AddZ3TaskPair(std::vector<Z3TaskPair> *pairs, int64_t left,
                   int64_t right, bool unordered = false) {
  if (unordered && left > right) std::swap(left, right);
  if (std::find(pairs->begin(), pairs->end(),
                Z3TaskPair(left, right)) == pairs->end()) {
    pairs->emplace_back(left, right);
  }
}

Array<Array<int64_t>> Z3DataDependenciesToFFI(
    const std::vector<Z3DataDependency> &dependencies) {
  Array<Array<int64_t>> result;
  result.reserve(dependencies.size());
  for (const auto &dependency : dependencies) {
    result.push_back(Array<int64_t>{dependency[0], dependency[1], dependency[2],
                                   dependency[3]});
  }
  return result;
}

Array<Array<int64_t>> Z3TaskPairsToFFI(
    const std::vector<Z3TaskPair> &pairs) {
  Array<Array<int64_t>> result;
  result.reserve(pairs.size());
  for (const auto &pair : pairs) {
    result.push_back(Array<int64_t>{pair.first, pair.second});
  }
  return result;
}

std::vector<int> Z3TopologicalPhysicalOrder(
    size_t n, const std::vector<int64_t> &raw_stages,
    const std::vector<int64_t> &residues,
    const std::vector<Z3TaskPair> &pipe_order_deps,
    const std::vector<Z3DataDependency> &data_deps) {
  std::vector<std::vector<int>> successors(n);
  std::vector<int> indegree(n, 0);
  auto add_tie_edge = [&](int64_t source, int64_t target) {
    if (source == target || raw_stages[source] != raw_stages[target] ||
        residues[source] != residues[target]) {
      return;
    }
    auto &edges = successors[source];
    if (std::find(edges.begin(), edges.end(), static_cast<int>(target)) !=
        edges.end()) {
      return;
    }
    edges.push_back(static_cast<int>(target));
    ++indegree[target];
  };
  for (const auto &pair : pipe_order_deps) {
    add_tie_edge(pair.first, pair.second);
  }
  for (const auto &dependency : data_deps) {
    if (dependency[2] == 0) add_tie_edge(dependency[0], dependency[1]);
  }

  auto higher_priority = [&](int left, int right) {
    if (residues[left] != residues[right]) {
      return residues[left] < residues[right];
    }
    if (raw_stages[left] != raw_stages[right]) {
      return raw_stages[left] > raw_stages[right];
    }
    return left < right;
  };
  std::vector<int> order;
  order.reserve(n);
  std::vector<char> emitted(n, 0);
  for (size_t position = 0; position < n; ++position) {
    int best = -1;
    for (size_t task = 0; task < n; ++task) {
      if (emitted[task] || indegree[task] != 0) continue;
      if (best < 0 || higher_priority(static_cast<int>(task), best)) {
        best = static_cast<int>(task);
      }
    }
    ICHECK_GE(best, 0)
        << "Z3PipelinePlanning: zero-latency same-stage scalar/order "
           "constraints contain a topological cycle";
    emitted[best] = 1;
    order.push_back(best);
    for (int successor : successors[best]) --indegree[successor];
  }
  return order;
}

class Z3PipelinePlanner : public StmtExprMutator {
public:
  static Stmt Substitute(const PrimFunc &func) {
    Z3PipelinePlanner planner;
    for (const auto &[_, buffer] : func->buffer_map) {
      planner.buffer_data_to_buffer_.Set(buffer->data, buffer);
    }
    return planner.VisitStmt(func->body);
  }

private:
  Stmt VisitStmt_(const SBlockNode *op) final {
    for (const Buffer &buffer : op->alloc_buffers) {
      buffer_data_to_buffer_.Set(buffer->data, buffer);
    }
    SBlock block = Downcast<SBlock>(StmtExprMutator::VisitStmt_(op));
    for (const Buffer &buffer : op->alloc_buffers) {
      buffer_data_to_buffer_.erase(buffer->data);
    }
    return block;
  }

  Stmt VisitStmt_(const ForNode *loop) final {
    if (loop->annotations.count("tl_z3_owner_ring")) {
      // Z3OwnerRingPlanning owns these loops and their runtime schedule; do
      // not send their cross-owner dependencies through the per-loop adapter.
      return GetRef<For>(loop);
    }
    if (!loop->annotations.count(kZ3Pipeline)) {
      return StmtExprMutator::VisitStmt_(loop);
    }
    return PlanLoop(loop);
  }

  void ValidateFlatReorder(const Array<Stmt> &flat_stmts,
                           const Z3ScheduledStmtAnalysis &analysis,
                           const std::vector<int> &inverse_order) const {
    for (const Stmt &stmt : flat_stmts) {
      ICHECK(!stmt.as<BindNode>() && !IsPipelineDeclarationStmt(stmt))
          << "Z3PipelinePlanning: flat non-identity reorder with scalar Bind "
             "or buffer declaration statements is unsupported; use a "
             "cross-stage schedule or keep identity order";
      Z3FlatReorderSafetyVisitor safety;
      safety(stmt);
      ICHECK(!safety.has_non_buffer_side_effect())
          << "Z3PipelinePlanning: flat non-identity reorder is unsupported "
             "for opaque or non-buffer side effects";
    }
    ICHECK_EQ(flat_stmts.size(), analysis.scheduled_stmts.size())
        << "Z3PipelinePlanning: flat non-identity reorder currently requires "
           "every non-declaration statement to be scheduled";

    size_t n = analysis.scheduled_stmts.size();
    for (size_t source = 0; source < n; ++source) {
      for (size_t target = source + 1; target < n; ++target) {
        if (inverse_order[source] < inverse_order[target] ||
            !MayHaveBufferDependency(analysis, source, target)) {
          continue;
        }
        LOG(FATAL) << "Z3PipelinePlanning: refusing unsafe flat reorder of "
                      "dependent statements "
                   << source << " and " << target;
      }
    }
  }

  int CloneIndex(const std::vector<Z3CloneSpec> &clones, int definition,
                 int consumer) const {
    for (size_t i = 0; i < clones.size(); ++i) {
      if (clones[i].definition == definition &&
          clones[i].consumer == consumer) {
        return static_cast<int>(i);
      }
    }
    return -1;
  }

  int RootConsumer(const Z3AugmentedProblem &problem, int task) const {
    size_t n = problem.original_tasks;
    return task < static_cast<int64_t>(n)
               ? task
               : problem.clones[task - n].consumer;
  }

  std::vector<Z3ScalarUseEdge> CollectScalarTaskUses(
      const Z3ScheduledStmtAnalysis &analysis,
      const Z3AugmentedProblem &problem) const {
    std::vector<Z3ScalarUseEdge> edges;
    for (const ScalarUseDefEdge &edge : analysis.scalar.edges) {
      int definition = analysis.scheduled_for_flat[edge.definition];
      int use = analysis.scheduled_for_flat[edge.use];
      if (definition < 0) continue;  // A replayable definition is inlined.
      ICHECK_GE(use, 0)
          << "Z3PipelinePlanning: dependency-closed replayability left a "
             "scheduled definition feeding an unscheduled Bind";
      ICHECK_LT(edge.definition, edge.use)
          << "Z3PipelinePlanning: a flat scalar Bind must precede every use";
      edges.push_back({definition, use});
    }
    size_t n = problem.original_tasks;
    for (size_t clone_index = 0; clone_index < problem.clones.size();
         ++clone_index) {
      int definition = problem.clones[clone_index].definition;
      size_t flat = analysis.scheduled_indices[definition];
      std::vector<int> scalar_inputs;
      for (const Var &var : analysis.scalar.scalar_uses[flat]) {
        auto def = analysis.scalar.scalar_def_stmt.find(var);
        if (def == analysis.scalar.scalar_def_stmt.end()) continue;
        int task = analysis.scheduled_for_flat[def->second];
        if (task >= 0) scalar_inputs.push_back(task);
      }
      std::sort(scalar_inputs.begin(), scalar_inputs.end());
      scalar_inputs.erase(
          std::unique(scalar_inputs.begin(), scalar_inputs.end()),
          scalar_inputs.end());
      for (int input : scalar_inputs) {
        edges.push_back({input, static_cast<int>(n + clone_index)});
      }
    }
    std::sort(edges.begin(), edges.end(),
              [](const Z3ScalarUseEdge &left, const Z3ScalarUseEdge &right) {
                return std::tie(left.definition, left.use) <
                       std::tie(right.definition, right.use);
              });
    return edges;
  }

  void AddScalarConstraints(const Z3ScheduledStmtAnalysis &analysis,
                            const Z3ScalarUseEdge &edge,
                            Z3AugmentedProblem *problem) const {
    size_t flat_definition = analysis.scheduled_indices[edge.definition];
    auto add_raw_war_order = [&](int source, int target) {
      AddZ3DataDependency(&problem->data_deps, source, target, 0,
                          problem->latencies[source]);
      AddZ3DataDependency(&problem->data_deps, target, source, 1,
                          problem->latencies[target]);
      // Same-stage physical order is established by the augmented topological
      // sort below. A scalar residue pipe-order would over-constrain valid
      // cross-stage clone snapshots and their next-iteration WAR closure.
    };
    int root = RootConsumer(*problem, edge.use);
    int clone = CloneIndex(problem->clones, edge.definition, root);
    if (clone >= 0) {
      int producer = static_cast<int>(problem->original_tasks) + clone;
      add_raw_war_order(producer, edge.use);
      AddZ3TaskPair(&problem->stage_order_deps, producer, edge.use);
      AddZ3TaskPair(&problem->stage_order_deps, edge.use, producer);
      return;
    }

    // An uncloned use reads the original definition. Once a clone exists, the
    // use is rewired to that fresh definition instead of also requiring the
    // original Bind to complete at the consumer site.
    add_raw_war_order(edge.definition, edge.use);
    if (!analysis.scalar.cloneable_binds[flat_definition]) {
      AddZ3TaskPair(&problem->stage_order_deps, edge.definition, edge.use);
      AddZ3TaskPair(&problem->stage_order_deps, edge.use, edge.definition);
    }
  }

  void AddVirtualCloneBufferDependencies(
      const ForNode *loop, const Array<Var> &buffer_vars,
      const Z3BufferVersionCountMap &fixed_versions,
      Z3AugmentedProblem *problem) const {
    size_t n = problem->original_tasks;
    size_t total = problem->latencies.size();
    if (total == n) return;
    Map<Var, arith::IntSet> thread_domains =
        CollectPipelineThreadDomains(loop->body);
    auto logical_order = [&](size_t task) {
      // Clones execute in the consumer's stage immediately before the original
      // consumer statement. The secondary bit handles a buffer write in that
      // consumer itself without falling back to augmented source indices.
      return task < n
                 ? std::make_pair(static_cast<int>(task), 1)
                 : std::make_pair(problem->clones[task - n].consumer, 0);
    };
    auto buffer_distance = [&](const Var &storage) -> int64_t {
      if (auto fixed = fixed_versions.Get(storage)) return fixed.value()->value;
      for (size_t i = 0; i < buffer_vars.size(); ++i) {
        if (buffer_vars[i].same_as(storage)) {
          return -static_cast<int64_t>(i) - 1;
        }
      }
      return 1;
    };
    auto consider = [&](size_t source, size_t target,
                        const Array<BufferRegion> &left,
                        const Array<BufferRegion> &right) {
      for (bool cross : {false, true}) {
        if (!cross && logical_order(source) >= logical_order(target)) continue;
        for (const BufferRegion &lhs : left) {
          for (const BufferRegion &rhs : right) {
            if (!PipelineLoopRegionsConflict(lhs, rhs, loop, cross,
                                             thread_domains)) {
              continue;
            }
            int64_t distance = cross ? buffer_distance(lhs->buffer->data) : 0;
            if (cross && distance > 0) {
              if (auto fixed = fixed_versions.Get(lhs->buffer->data)) {
                problem->fixed_recurrences.push_back(
                    {static_cast<int64_t>(source), static_cast<int64_t>(target),
                     problem->latencies[source], fixed.value()->value,
                     lhs->buffer->data});
              }
            }
            AddZ3DataDependency(&problem->data_deps, source, target, distance,
                                problem->latencies[source]);
          }
        }
      }
    };
    for (size_t i = 0; i < total; ++i) {
      for (size_t j = 0; j < total; ++j) {
        if (i < n && j < n) continue;  // Original contracts are already present.
        consider(i, j, problem->writes[i], problem->reads[j]);  // RAW
        consider(i, j, problem->reads[i], problem->writes[j]);  // WAR
      }
    }
  }

  Z3AugmentedProblem BuildAugmentedProblem(
      const ForNode *loop, const Z3ScheduledStmtAnalysis &analysis,
      const Array<Var> &buffer_vars,
      const Z3BufferVersionCountMap &fixed_versions,
      const Z3AugmentedProblem &base,
      const std::vector<Z3CloneSpec> &clones, bool manual_schedule) const {
    Z3AugmentedProblem problem = base;
    problem.clones = clones;
    size_t n = problem.original_tasks;
    for (const Z3CloneSpec &clone : clones) {
      problem.latencies.push_back(base.latencies[clone.definition]);
      problem.iis.push_back(base.iis[clone.definition]);
      problem.resource_flags.push_back(base.resource_flags[clone.definition]);
      if (manual_schedule) {
        problem.manual_stages.push_back(base.manual_stages[clone.consumer]);
      }
      problem.reads.push_back(base.reads[clone.definition]);
      problem.writes.push_back({});
      int task = static_cast<int>(problem.latencies.size() - 1);
      AddZ3TaskPair(&problem.stage_order_deps, task, clone.consumer);
      AddZ3TaskPair(&problem.stage_order_deps, clone.consumer, task);
    }
    for (const Z3ScalarUseEdge &edge : CollectScalarTaskUses(analysis, problem)) {
      AddScalarConstraints(analysis, edge, &problem);
    }
    size_t total = problem.latencies.size();
    for (size_t i = 0; i < total; ++i) {
      for (size_t j = i + 1; j < total; ++j) {
        if (i < n && j < n) continue;
        if ((problem.resource_flags[i] & problem.resource_flags[j]) != 0) {
          AddZ3TaskPair(&problem.resource_deps, i, j, /*unordered=*/true);
        }
      }
    }
    AddVirtualCloneBufferDependencies(loop, buffer_vars, fixed_versions, &problem);
    return problem;
  }

  std::vector<Z3CloneSpec> FindNeededClones(
      const Z3ScheduledStmtAnalysis &analysis,
      const Z3AugmentedProblem &problem,
      const std::vector<int64_t> &raw_stages, bool has_async_producers) const {
    std::vector<Z3CloneSpec> needed;
    for (const Z3ScalarUseEdge &edge : CollectScalarTaskUses(analysis, problem)) {
      int root = RootConsumer(problem, edge.use);
      if (CloneIndex(problem.clones, edge.definition, root) >= 0 ||
          raw_stages[edge.definition] == raw_stages[edge.use]) {
        continue;
      }
      size_t flat = analysis.scheduled_indices[edge.definition];
      ICHECK(analysis.scalar.cloneable_binds[flat])
          << "Z3PipelinePlanning: noncloneable scalar dependency from task "
          << edge.definition << " to task " << edge.use
          << " crosses pipeline stages; keep them in the same stage";
      ICHECK(!has_async_producers)
          << "Z3PipelinePlanning: cross-stage scalar clones involving explicit "
             "async producer/consumer ownership are unsupported; use a "
             "same-stage schedule so async guards are not discarded";
      if (CloneIndex(needed, edge.definition, root) < 0) {
        needed.push_back({edge.definition, root});
      }
    }
    return needed;
  }

  Z3ValidatedSchedule ValidateSchedule(
      const Z3AugmentedProblem &problem, const Array<Integer> &buffer_sizes,
      const Array<Array<Integer>> &memory_groups,
      bool recalculate_buffer_versions, Array<int64_t> starts,
      Array<int64_t> returned_versions, int64_t ii,
      bool manual_schedule) const {
    size_t n = problem.latencies.size();
    ICHECK_EQ(starts.size(), n)
        << "Z3PipelinePlanning: solver returned " << starts.size()
        << " start times for " << n << " augmented tasks ("
        << problem.original_tasks << " original + " << problem.clones.size()
        << " clones)";
    ICHECK_EQ(returned_versions.size(), buffer_sizes.size())
        << "Z3PipelinePlanning: solver returned " << returned_versions.size()
        << " buffer versions for " << buffer_sizes.size() << " buffers";
    ICHECK_GT(ii, 0) << "Z3PipelinePlanning: solver returned non-positive II";

    Z3ValidatedSchedule result;
    result.starts = std::move(starts);
    result.ii = ii;
    result.raw_stages.reserve(n);
    result.residues.reserve(n);
    for (size_t i = 0; i < n; ++i) {
      ICHECK_GE(result.starts[i], 0)
          << "Z3PipelinePlanning: solver returned a negative start time";
      result.raw_stages.push_back(result.starts[i] / ii);
      result.residues.push_back(result.starts[i] % ii);
    }
    if (manual_schedule) {
      ICHECK_EQ(problem.manual_stages.size(), n);
      for (size_t i = 0; i < n; ++i) {
        ICHECK_EQ(result.raw_stages[i], problem.manual_stages[i])
            << "Z3PipelinePlanning: solver result violates manual stage for "
            << (i < problem.original_tasks ? "original task " : "clone task ")
            << i;
      }
    }

    std::vector<int64_t> required_versions(buffer_sizes.size(), 1);
    for (const auto &dependency : problem.data_deps) {
      int64_t source = dependency[0];
      int64_t target = dependency[1];
      int64_t distance = dependency[2];
      int64_t latency = dependency[3];
      ICHECK(source >= 0 && source < static_cast<int64_t>(n) && target >= 0 &&
             target < static_cast<int64_t>(n) && latency >= 0);
      if (distance < 0) {
        ICHECK_NE(distance, std::numeric_limits<int64_t>::min());
        int64_t buffer = -distance - 1;
        ICHECK(buffer >= 0 &&
               buffer < static_cast<int64_t>(buffer_sizes.size()))
            << "Z3PipelinePlanning: data dependency references unavailable "
               "buffer "
            << buffer;
        __int128 numerator = static_cast<__int128>(latency) +
                             result.starts[source] - result.starts[target];
        if (numerator > 0) {
          __int128 versions = (numerator + ii - 1) / ii;
          ICHECK(versions <= std::numeric_limits<int64_t>::max())
              << "Z3PipelinePlanning: required buffer version count overflows "
                 "int64";
          required_versions[buffer] =
              std::max(required_versions[buffer],
                       static_cast<int64_t>(versions));
        }
      }
    }
    for (const Z3FixedRecurrence &recurrence : problem.fixed_recurrences) {
      ICHECK(recurrence.source >= 0 &&
             recurrence.source < static_cast<int64_t>(n) &&
             recurrence.target >= 0 &&
             recurrence.target < static_cast<int64_t>(n) &&
             recurrence.latency >= 0 && recurrence.fixed_versions > 0);
      __int128 numerator = static_cast<__int128>(recurrence.latency) +
                           result.starts[recurrence.source] -
                           result.starts[recurrence.target];
      int64_t required = 1;
      if (numerator > 0) {
        __int128 versions = (numerator + ii - 1) / ii;
        ICHECK(versions <= std::numeric_limits<int64_t>::max())
            << "Z3PipelinePlanning: fixed recurrence requirement overflows int64";
        required = static_cast<int64_t>(versions);
      }
      auto found = result.fixed_required.find(recurrence.storage);
      if (found == result.fixed_required.end()) {
        result.fixed_required.emplace(recurrence.storage, required);
      } else {
        found->second = std::max(found->second, required);
      }
      ICHECK_LE(required, recurrence.fixed_versions)
          << "Z3PipelinePlanning: fixed buffer version request for "
          << recurrence.storage << " is under-versioned: schedule requires "
          << required << ", fixed request is " << recurrence.fixed_versions;
    }

    std::vector<int64_t> validated_versions;
    validated_versions.reserve(returned_versions.size());
    for (size_t i = 0; i < returned_versions.size(); ++i) {
      ICHECK_GT(returned_versions[i], 0)
          << "Z3PipelinePlanning: solver returned a non-positive version "
             "count for buffer "
          << i;
      int64_t selected = returned_versions[i];
      if (recalculate_buffer_versions) {
        // The solver normally performs this recomputation itself. Repeat it on
        // the augmented problem so a stale prefix-only result cannot drop
        // clone reads; preserve a larger valid model value for compatibility.
        selected = std::max(selected, required_versions[i]);
      } else {
        ICHECK_GE(selected, required_versions[i])
            << "Z3PipelinePlanning: under-versioned buffer " << i
            << ": schedule requires " << required_versions[i]
            << " versions, solver returned " << selected;
      }
      validated_versions.push_back(selected);
    }
    result.versions = Array<int64_t>(validated_versions);
    result.model_versions = returned_versions;
    result.required_versions = Array<int64_t>(required_versions);

    for (const auto &dependency : problem.data_deps) {
      int64_t source = dependency[0];
      int64_t target = dependency[1];
      int64_t distance = dependency[2];
      __int128 effective_distance = distance;
      if (distance < 0) effective_distance = result.versions[-distance - 1];
      __int128 actual =
          static_cast<__int128>(result.starts[target]) - result.starts[source];
      __int128 required = static_cast<__int128>(dependency[3]) -
                          static_cast<__int128>(ii) * effective_distance;
      ICHECK(actual >= required)
          << "Z3PipelinePlanning: solver result violates augmented data "
             "dependency ("
          << source << ", " << target << ", " << distance << ", "
          << dependency[3] << ")";
    }
    for (const auto &pair : problem.pipe_order_deps) {
      ICHECK_LE(result.residues[pair.first], result.residues[pair.second])
          << "Z3PipelinePlanning: solver result violates scalar/pipe order ("
          << pair.first << ", " << pair.second << ")";
    }
    for (const auto &pair : problem.stage_order_deps) {
      ICHECK_LE(result.raw_stages[pair.first], result.raw_stages[pair.second])
          << "Z3PipelinePlanning: solver result violates stage_order_deps ("
          << pair.first << ", " << pair.second << ")";
    }
    for (const auto &pair : problem.resource_deps) {
      int left = pair.first;
      int right = pair.second;
      __int128 lr = result.residues[left];
      __int128 rr = result.residues[right];
      bool left_first = rr - lr >= problem.iis[left] &&
                        lr - rr + ii >= problem.iis[right];
      bool right_first = lr - rr >= problem.iis[right] &&
                         rr - lr + ii >= problem.iis[left];
      ICHECK(left_first || right_first)
          << "Z3PipelinePlanning: solver result violates augmented resource "
             "exclusion ("
          << left << ", " << right << ")";
    }

    for (const Array<Integer> &group : memory_groups) {
      ICHECK(!group.empty())
          << "Z3PipelinePlanning: memory group must start with a capacity";
      int64_t capacity = group[0]->value;
      ICHECK_GE(capacity, 0);
      __int128 used = 0;
      std::unordered_set<int64_t> members;
      for (size_t i = 1; i < group.size(); ++i) {
        int64_t member = group[i]->value;
        ICHECK(member >= 0 && member < static_cast<int64_t>(buffer_sizes.size()))
            << "Z3PipelinePlanning: memory-group buffer index is out of range";
        ICHECK(members.insert(member).second)
            << "Z3PipelinePlanning: duplicate buffer in memory group";
        ICHECK_GE(buffer_sizes[member]->value, 0);
        used += static_cast<__int128>(result.versions[member]) *
                buffer_sizes[member]->value;
      }
      ICHECK(used <= capacity)
          << "Z3PipelinePlanning: recomputed buffer versions exceed "
             "memory-group capacity "
          << capacity;
    }
    return result;
  }

  Stmt PlanLoop(const ForNode *loop) {
    const Map<String, Any> &annotations = loop->annotations;
    ICHECK(GetBool(annotations, kZ3Pipeline, false))
        << "Z3PipelinePlanning: tl_z3_pipeline must be True when present";
    PostOrderVisit(loop->body, [&](const ObjectRef &node) {
      if (const auto *attr = node.as<AttrStmtNode>()) {
        ICHECK(attr->attr_key != kZ3CudaStageAttr)
            << "Z3PipelinePlanning: unnormalized T.Stage marker; run "
               "NormalizeZ3ScheduleUnits before planning";
      }
    });
    bool manual_schedule = GetBool(annotations, kZ3ManualSchedule, false);
    bool has_handwritten_schedule =
        annotations.count("tl_pipeline_order") ||
        annotations.count("tl_pipeline_stage") ||
        annotations.count("software_pipeline_order") ||
        annotations.count("software_pipeline_stage");
    ICHECK(!has_handwritten_schedule)
        << "Z3PipelinePlanning: "
        << (manual_schedule ? "manual Z3 scheduling" : "Z3 scheduling")
        << " cannot overwrite existing handwritten pipeline order/stage annotations";
    ICHECK(!annotations.count(kPipelineBufferVersions))
        << "Z3PipelinePlanning: tl_pipeline_buffer_versions already exists; "
           "refusing to overwrite a buffer-version contract";
    ICHECK(!annotations.count(kZ3BufferVersionsSelected))
        << "Z3PipelinePlanning: selected buffer versions already exist; "
           "refusing to reinterpret them as a new user request";
    ICHECK(!annotations.count(kPipelineScalarClonePlan))
        << "Z3PipelinePlanning: a scalar clone plan already exists; refusing "
           "to overwrite materialization metadata";

    Array<Integer> latencies = GetIntegerArray(annotations, kZ3Latencies);
    Array<Integer> iis = GetIntegerArray(annotations, kZ3IIs);
    Array<Integer> resource_flags = GetIntegerArray(annotations, kZ3ResourceFlags);
    size_t n = latencies.size();
    ICHECK(n > 0) << "Z3PipelinePlanning: at least one task is required";
    ICHECK_EQ(iis.size(), n)
        << "Z3PipelinePlanning: latencies and iis must have equal lengths";
    ICHECK_EQ(resource_flags.size(), n)
        << "Z3PipelinePlanning: latencies and resource_flags must have equal lengths";
    ICHECK(n <= static_cast<size_t>(std::numeric_limits<int>::max()))
        << "Z3PipelinePlanning: too many tasks for pipeline order annotations";
    if (auto domains = annotations.Get(kZ3ExecutionDomains); domains.has_value()) {
      ICHECK_EQ(Downcast<Array<String>>(domains.value()).size(), n)
          << "Z3PipelinePlanning: execution-domain count does not match the "
             "task count";
    }
    ICHECK(GetNestedIntegerArrays(annotations, kZ3OwnerExclusionDeps).empty())
        << "Z3PipelinePlanning: owner_exclusion_deps are not expressible by "
           "the current CUDA pipeline IR; refusing to discard owner lifetime "
           "and wraparound semantics";

    int64_t num_stages = GetInteger(annotations, "num_stages", 0);
    ICHECK_GT(num_stages, 0)
        << "Z3PipelinePlanning: T.Pipelined num_stages must be positive";
    bool enable_offset = GetBool(annotations, kZ3EnableOffset, false);
    Array<Integer> manual_stages = GetIntegerArray(annotations, kZ3ManualStages);
    if (manual_schedule) {
      ICHECK_EQ(manual_stages.size(), n)
          << "Z3PipelinePlanning: manual_schedule requires one stage per task";
    } else {
      ICHECK(manual_stages.empty())
          << "Z3PipelinePlanning: manual_stages requires manual_schedule=True";
    }

    Array<Integer> original_buffer_sizes = GetIntegerArray(annotations, kZ3BufferSizes);
    Array<Var> original_buffer_vars;
    if (auto value = annotations.Get(kZ3BufferVars); value.has_value()) {
      original_buffer_vars = Downcast<Array<Var>>(value.value());
    }
    ICHECK_EQ(original_buffer_vars.size(), original_buffer_sizes.size())
        << "Z3PipelinePlanning: buffer_vars and buffer_sizes must have equal lengths";
    std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> seen_buffer_vars;
    std::unordered_map<Var, int64_t, ObjectPtrHash, ObjectPtrEqual> original_buffer_ids;
    for (size_t i = 0; i < original_buffer_vars.size(); ++i) {
      const Var &var = original_buffer_vars[i];
      ICHECK(seen_buffer_vars.insert(var).second)
          << "Z3PipelinePlanning: duplicate Buffer.data Var in buffer_vars";
      original_buffer_ids.emplace(var, static_cast<int64_t>(i));
    }
    for (const Integer &size : original_buffer_sizes) {
      ICHECK_GE(size->value, 0) << "Z3PipelinePlanning: negative buffer size";
    }

    Z3BufferVersionCountMap version_requests = ReadZ3BufferVersionCounts(
        annotations, kZ3BufferVersionRequests, "Z3PipelinePlanning");
    Z3BufferVersionModeMap version_mode_requests = ReadZ3BufferVersionModes(
        annotations, kZ3BufferVersionModeRequests, "Z3PipelinePlanning");
    // Every explicit count is exact, including (1, "auto"). Only a bare one
    // has the additional opt-out meaning. Fixed buffers do not occupy solver
    // version-variable positions.
    const Z3BufferVersionCountMap &fixed_versions = version_requests;
    Array<Var> buffer_vars;
    Array<Integer> buffer_sizes;
    std::vector<int64_t> buffer_id_remap(original_buffer_vars.size(),
                                         std::numeric_limits<int64_t>::min());
    for (size_t i = 0; i < original_buffer_vars.size(); ++i) {
      const Var &var = original_buffer_vars[i];
      if (fixed_versions.count(var)) continue;
      buffer_id_remap[i] = static_cast<int64_t>(buffer_vars.size());
      buffer_vars.push_back(var);
      buffer_sizes.push_back(original_buffer_sizes[i]);
    }
    for (const auto &[storage, _] : version_requests) {
      if (IsZ3BareOneRequest(storage, version_requests, version_mode_requests)) {
        continue;
      }
      ICHECK(original_buffer_ids.count(storage))
          << "Z3PipelinePlanning: requested storage " << storage
          << " is missing from buffer_vars after task inference";
    }
    for (const auto &[storage, _] : version_mode_requests) {
      ICHECK(original_buffer_ids.count(storage))
          << "Z3PipelinePlanning: mode-requested storage " << storage
          << " is missing from buffer_vars after task inference";
    }

    Array<Stmt> body_stmts = NormalizePipelineBody(loop->body);
    Array<Stmt> flat_stmts = Z3SeqStmtFlattener::Flatten(body_stmts);
    Z3ScheduledStmtAnalysis analysis =
        AnalyzeScheduledStmts(flat_stmts, buffer_data_to_buffer_);
    ICHECK_EQ(analysis.scheduled_stmts.size(), n)
        << "Z3PipelinePlanning: task count " << n
        << " does not match body scheduled statement count "
        << analysis.scheduled_stmts.size()
        << "; tasks exclude buffer declarations and dependency-closed "
           "replayable scalar Bind statements after SeqStmt flattening";

    auto check_task = [&](int64_t task, const char *what) {
      ICHECK(task >= 0 && task < static_cast<int64_t>(n))
          << "Z3PipelinePlanning: " << what << " task index " << task
          << " is outside [0, " << n << ")";
    };
    Z3AugmentedProblem base;
    base.original_tasks = n;
    for (const Integer &value : latencies) {
      ICHECK_GE(value->value, 0);
      base.latencies.push_back(value->value);
    }
    for (const Integer &value : iis) {
      ICHECK_GT(value->value, 0);
      base.iis.push_back(value->value);
    }
    for (const Integer &value : resource_flags) {
      ICHECK_GE(value->value, 0);
      base.resource_flags.push_back(value->value);
    }
    for (const Integer &value : manual_stages) {
      ICHECK_GE(value->value, 0);
      base.manual_stages.push_back(value->value);
    }
    base.reads = analysis.reads;
    base.writes = analysis.writes;

    for (const Array<Integer> &dep : GetNestedIntegerArrays(annotations, kZ3DataDeps)) {
      ICHECK_EQ(dep.size(), 4U)
          << "Z3PipelinePlanning: data_deps entries must have four fields";
      check_task(dep[0]->value, "data_deps");
      check_task(dep[1]->value, "data_deps");
      ICHECK_GE(dep[3]->value, 0)
          << "Z3PipelinePlanning: data dependency latency must be non-negative";
      int64_t distance = dep[2]->value;
      if (distance < 0) {
        ICHECK_NE(distance, std::numeric_limits<int64_t>::min());
        int64_t original_id = -distance - 1;
        ICHECK(original_id >= 0 &&
               original_id < static_cast<int64_t>(original_buffer_sizes.size()))
            << "Z3PipelinePlanning: data dependency references an unavailable buffer";
        const Var &storage = original_buffer_vars[original_id];
        if (auto fixed = fixed_versions.Get(storage)) {
          distance = fixed.value()->value;
          base.fixed_recurrences.push_back(
              {dep[0]->value, dep[1]->value, dep[3]->value, distance, storage});
        } else {
          int64_t remapped = buffer_id_remap[original_id];
          ICHECK(remapped >= 0);
          distance = -remapped - 1;
        }
      }
      AddZ3DataDependency(&base.data_deps, dep[0]->value, dep[1]->value,
                          distance, dep[3]->value);
    }
    auto parse_pairs = [&](const char *key, bool resource) {
      std::vector<Z3TaskPair> pairs;
      for (const Array<Integer> &pair : GetNestedIntegerArrays(annotations, key)) {
        ICHECK_EQ(pair.size(), 2U)
            << "Z3PipelinePlanning: " << key << " entries must be pairs";
        check_task(pair[0]->value, key);
        check_task(pair[1]->value, key);
        ICHECK(!resource || pair[0]->value != pair[1]->value)
            << "Z3PipelinePlanning: a resource dependency cannot exclude a task from itself";
        AddZ3TaskPair(&pairs, pair[0]->value, pair[1]->value, resource);
      }
      return pairs;
    };
    base.resource_deps = parse_pairs(kZ3ResourceDeps, true);
    if (annotations.count(kZ3PipeOrderDeps)) {
      base.pipe_order_deps = parse_pairs(kZ3PipeOrderDeps, false);
    } else if (manual_schedule) {
      // NormalizeZ3ScheduleUnits removes the helper's empty placeholder for
      // T.Stage. With inference disabled, derive the same source-order pairs
      // here that InferZ3PipelineProblem would have produced.
      for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
          if ((base.resource_flags[i] & base.resource_flags[j]) != 0) {
            AddZ3TaskPair(&base.pipe_order_deps, i, j);
          }
        }
      }
    }
    base.stage_order_deps = parse_pairs(kZ3StageOrderDeps, false);

    auto scheduler = ffi::Function::GetGlobal("tl.cuda.transform.z3_schedule_loop_python");
    ICHECK(scheduler.has_value())
        << "Z3PipelinePlanning: tl.cuda.transform.z3_schedule_loop_python is not registered";
    Array<Array<Integer>> memory_groups;
    for (const Array<Integer> &group :
         GetNestedIntegerArrays(annotations, kZ3MemoryGroups)) {
      ICHECK(!group.empty())
          << "Z3PipelinePlanning: memory group must start with a capacity";
      Array<Integer> normalized{group[0]};
      for (size_t member = 1; member < group.size(); ++member) {
        int64_t original_id = group[member]->value;
        ICHECK(original_id >= 0 &&
               original_id < static_cast<int64_t>(original_buffer_vars.size()))
            << "Z3PipelinePlanning: memory-group buffer index is out of range";
        if (fixed_versions.count(original_buffer_vars[original_id])) continue;
        int64_t remapped = buffer_id_remap[original_id];
        ICHECK(remapped >= 0);
        normalized.push_back(Integer(remapped));
      }
      memory_groups.push_back(std::move(normalized));
    }
    int64_t resource_bits = GetInteger(annotations, kZ3ResourceBits, 7);
    int64_t seed = GetInteger(annotations, kZ3Seed, 42);
    bool recalculate_buffer_versions =
        GetBool(annotations, kZ3RecalculateBufferVersions, true);
    bool verbose = GetBool(annotations, kZ3Verbose, false);
    Optional<String> debug_dir = std::nullopt;
    if (auto value = annotations.Get(kZ3DebugDir); value.has_value()) {
      debug_dir = Downcast<String>(value.value());
    }
    int64_t round_limit = GetInteger(annotations, kZ3MaxCloneRounds,
                                     static_cast<int64_t>(n * n + 1));
    ICHECK_GT(round_limit, 0)
        << "Z3PipelinePlanning: max clone-plan rounds must be positive";
    bool has_async_producers = annotations.count(kPipelineAsyncProducers) ||
                               annotations.count(kPipelineAsyncProducerGroups) ||
                               annotations.count(kZ3AsyncSameStage);

    using SolverResult = ffi::Tuple<ffi::Array<int64_t>, ffi::Array<int64_t>, int64_t>;
    std::vector<Z3CloneSpec> clones;
    Z3AugmentedProblem augmented;
    Z3ValidatedSchedule schedule;
    int64_t clone_rounds = 0;
    for (;;) {
      ICHECK(clone_rounds < round_limit)
          << "Z3PipelinePlanning: scalar clone-plan fixed point did not converge within "
          << round_limit << " solve rounds";
      augmented = BuildAugmentedProblem(loop, analysis, buffer_vars, fixed_versions,
                                        base, clones, manual_schedule);
      ICHECK(augmented.latencies.size() <=
             static_cast<size_t>(std::numeric_limits<int>::max()))
          << "Z3PipelinePlanning: too many augmented clone tasks";
      SolverResult solver_result =
          scheduler.value()(
              num_stages, Array<int64_t>(augmented.latencies),
              Array<int64_t>(augmented.iis), Array<int64_t>(augmented.resource_flags),
              Z3DataDependenciesToFFI(augmented.data_deps),
              Z3TaskPairsToFFI(augmented.resource_deps), Array<Array<int64_t>>(),
              Z3TaskPairsToFFI(augmented.pipe_order_deps), ToInt64(buffer_sizes),
              ToInt64(memory_groups), Z3TaskPairsToFFI(augmented.stage_order_deps),
              enable_offset, Array<int64_t>(augmented.manual_stages), manual_schedule,
              resource_bits, seed, recalculate_buffer_versions, verbose, debug_dir)
              .cast<SolverResult>();
      schedule = ValidateSchedule(
          augmented, buffer_sizes, memory_groups, recalculate_buffer_versions,
          solver_result.get<0>(), solver_result.get<1>(), solver_result.get<2>(),
          manual_schedule);
      ++clone_rounds;
      std::vector<Z3CloneSpec> needed = FindNeededClones(
          analysis, augmented, schedule.raw_stages, has_async_producers);
      if (needed.empty()) break;
      ICHECK(clones.size() + needed.size() <= n * n)
          << "Z3PipelinePlanning: scalar clone-plan fixed point exceeded the "
             "finite definition/consumer plan bound";
      clones.insert(clones.end(), needed.begin(), needed.end());
    }
    size_t total = augmented.latencies.size();
    int64_t min_stage = std::numeric_limits<int64_t>::max();
    for (size_t i = 0; i < total; ++i) {
      min_stage = std::min(min_stage, schedule.raw_stages[i]);
    }
    std::vector<int> stages(total, 0);
    for (size_t i = 0; i < total; ++i) {
      int64_t normalized = schedule.raw_stages[i] - min_stage;
      ICHECK(normalized >= 0 && normalized <= std::numeric_limits<int>::max())
          << "Z3PipelinePlanning: normalized stage does not fit pipeline IR";
      stages[i] = static_cast<int>(normalized);
    }
    std::vector<int> augmented_physical_order = Z3TopologicalPhysicalOrder(
        total, schedule.raw_stages, schedule.residues,
        augmented.pipe_order_deps, augmented.data_deps);
    std::vector<int> augmented_position(total, 0);
    std::vector<size_t> physical_order;
    physical_order.reserve(n);
    for (size_t position = 0; position < augmented_physical_order.size(); ++position) {
      int task = augmented_physical_order[position];
      augmented_position[task] = static_cast<int>(position);
      if (task < static_cast<int>(n)) physical_order.push_back(task);
    }
    ICHECK_EQ(physical_order.size(), n);
    std::vector<int> inverse_order(n, 0);
    for (size_t position = 0; position < n; ++position) {
      inverse_order[physical_order[position]] = static_cast<int>(position);
    }

    bool identity_order = true;
    for (size_t i = 0; i < n; ++i) {
      identity_order = identity_order && physical_order[i] == static_cast<int>(i);
    }
    bool flat_schedule = std::all_of(stages.begin(), stages.end(),
                                     [](int stage) { return stage == 0; });
    if (flat_schedule && !identity_order) {
      ValidateFlatReorder(flat_stmts, analysis, inverse_order);
      Array<Stmt> reordered;
      reordered.reserve(n);
      for (size_t source_index : physical_order) {
        reordered.push_back(analysis.scheduled_stmts[source_index]);
      }
      flat_stmts = std::move(reordered);
      std::iota(inverse_order.begin(), inverse_order.end(), 0);
    }

    int64_t cuda_depth = num_stages;
    for (int stage : stages) {
      cuda_depth = std::max(cuda_depth, static_cast<int64_t>(stage) + 1);
    }
    for (int64_t version : schedule.versions) {
      cuda_depth = std::max(cuda_depth, version);
    }
    Map<Var, IntImm> buffer_version_map;
    Map<Var, IntImm> selected_version_map;
    Map<Var, IntImm> model_version_map;
    Map<Var, IntImm> required_version_map;
    for (size_t i = 0; i < buffer_vars.size(); ++i) {
      buffer_version_map.Set(buffer_vars[i],
                             IntImm(DataType::Int(64), schedule.versions[i]));
      selected_version_map.Set(buffer_vars[i],
                               IntImm(DataType::Int(64), schedule.versions[i]));
      model_version_map.Set(buffer_vars[i],
                            IntImm(DataType::Int(64), schedule.model_versions[i]));
      required_version_map.Set(
          buffer_vars[i], IntImm(DataType::Int(64), schedule.required_versions[i]));
    }
    for (const auto &[storage, count] : fixed_versions) {
      buffer_version_map.Set(storage, count);
      if (IsZ3BareOneRequest(storage, fixed_versions, version_mode_requests)) {
        continue;
      }
      selected_version_map.Set(storage, count);
      cuda_depth = std::max(cuda_depth, static_cast<int64_t>(count->value));
      model_version_map.Set(storage, count);
      int64_t required = 1;
      auto found = schedule.fixed_required.find(storage);
      if (found != schedule.fixed_required.end()) required = found->second;
      required_version_map.Set(storage, IntImm(DataType::Int(64), required));
    }

    Map<String, Any> output_annotations;
    for (const auto &[key, value] : annotations) {
      if (!IsZ3InputAnnotation(key)) output_annotations.Set(key, value);
    }
    Array<Integer> order_annotation;
    Array<Integer> stage_annotation;
    Array<Integer> start_audit;
    order_annotation.reserve(n);
    stage_annotation.reserve(n);
    start_audit.reserve(n);
    for (size_t i = 0; i < n; ++i) {
      // Clone materialization preserves every augmented task as an independent
      // schedule unit. Sparse augmented positions keep the original tasks at
      // their exact solver-ordered positions while leaving room for clones.
      // Without clones, retain the dense original-only order contract.
      int order = clones.empty() ? inverse_order[i] : augmented_position[i];
      order_annotation.push_back(Integer(order));
      stage_annotation.push_back(Integer(stages[i]));
      start_audit.push_back(Integer(IntImm(DataType::Int(64), schedule.starts[i])));
    }
    output_annotations.Set("tl_pipeline_order", order_annotation);
    output_annotations.Set("tl_pipeline_stage", stage_annotation);
    output_annotations.Set("num_stages", IntImm(DataType::Int(64), cuda_depth));
    output_annotations.Set(kPipelineBufferVersions, buffer_version_map);
    output_annotations.Set(kZ3BufferVersionsSelected, selected_version_map);
    output_annotations.Set(kZ3AuditVersionRequest, version_requests);
    if (!version_mode_requests.empty()) {
      output_annotations.Set(kZ3AuditVersionModeRequest, version_mode_requests);
    }
    output_annotations.Set(kZ3AuditVersionSelected, model_version_map);
    output_annotations.Set(kZ3AuditVersionRequired, required_version_map);
    output_annotations.Set(kZ3AuditII, IntImm(DataType::Int(64), schedule.ii));
    output_annotations.Set(kZ3AuditStarts, start_audit);

    if (!clones.empty()) {
      Array<Map<String, Any>> clone_plan;
      for (size_t i = 0; i < clones.size(); ++i) {
        const Z3CloneSpec &clone = clones[i];
        int task = static_cast<int>(n + i);
        size_t definition_flat = analysis.scheduled_indices[clone.definition];
        const auto *bind = flat_stmts[definition_flat].as<BindNode>();
        ICHECK(bind != nullptr);
        std::vector<int64_t> inputs;
        for (const Var &var : analysis.scalar.scalar_uses[definition_flat]) {
          auto definition = analysis.scalar.scalar_def_stmt.find(var);
          if (definition == analysis.scalar.scalar_def_stmt.end()) continue;
          int input = analysis.scheduled_for_flat[definition->second];
          if (input < 0) continue;  // Replayable inputs are expanded as before.
          int input_clone = CloneIndex(clones, input, clone.consumer);
          inputs.push_back(input_clone >= 0
                               ? static_cast<int64_t>(n + input_clone)
                               : static_cast<int64_t>(input));
        }
        std::sort(inputs.begin(), inputs.end());
        inputs.erase(std::unique(inputs.begin(), inputs.end()), inputs.end());
        Array<Integer> scalar_inputs;
        for (int64_t input : inputs) scalar_inputs.push_back(Integer(input));

        Map<String, Any> entry;
        entry.Set("clone_index", Integer(i));
        entry.Set("clone_task", Integer(task));
        entry.Set("definition_task", Integer(clone.definition));
        entry.Set("definition_stmt", Integer(definition_flat));
        entry.Set("definition_var", bind->var);
        entry.Set("value", bind->value);
        entry.Set("consumer_task", Integer(clone.consumer));
        entry.Set("consumer_stmt", Integer(analysis.scheduled_indices[clone.consumer]));
        entry.Set("consumer_order", Integer(augmented_position[clone.consumer]));
        entry.Set("stage", Integer(stages[task]));
        entry.Set("start", Integer(IntImm(DataType::Int(64), schedule.starts[task])));
        entry.Set("schedule_position", Integer(augmented_position[task]));
        entry.Set("consumer_schedule_position",
                  Integer(augmented_position[clone.consumer]));
        entry.Set("scalar_inputs", scalar_inputs);
        entry.Set("virtual_reads", augmented.reads[task]);
        clone_plan.push_back(std::move(entry));
      }
      Array<Integer> original_flat_indices;
      for (size_t index : analysis.scheduled_indices) {
        original_flat_indices.push_back(Integer(index));
      }
      output_annotations.Set(kPipelineScalarClonePlan, clone_plan);
      output_annotations.Set(kPipelineScalarCloneOriginalTaskCount, Integer(n));
      output_annotations.Set(kPipelineScalarCloneOriginalFlatIndices,
                             original_flat_indices);
      output_annotations.Set(kZ3AuditCloneRounds, Integer(clone_rounds));
    }

    return For(loop->loop_var, loop->min, loop->extent, loop->kind,
               MakePipelineBody(flat_stmts), loop->thread_binding,
               output_annotations, loop->step, loop->span);
  }


  Map<Var, Buffer> buffer_data_to_buffer_;
};

// ---------------------------------------------------------------------------
// Recursive scheduled-IR planning. The legacy flat adapter above deliberately
// remains unchanged; this path is selected only by canonical tl.schedule_unit
// nodes produced by cuda.MaterializeZ3ScheduleUnits.
// ---------------------------------------------------------------------------

using cuda::z3::BufferVersionMap;
using cuda::z3::ControlNode;
using cuda::z3::DependencyCache;
using cuda::z3::DependencyKind;
using cuda::z3::DepInfo;
using cuda::z3::IRStructure;
using cuda::z3::MultiBufferOwnerMap;
using cuda::z3::ScheduledTIR;
using cuda::z3::TaskAccessInfo;
using cuda::z3::TaskKind;
using cuda::z3::TaskNode;

class RecursiveTimingExtra final : public cuda::z3::IRExtraInfo {
 public:
  std::unique_ptr<cuda::z3::IRExtraInfo> Clone() const final {
    return std::make_unique<RecursiveTimingExtra>(*this);
  }

  int64_t start{0};
  int manual_stage{cuda::z3::kUnscheduledStage};
  int async_producer{0};
  int async_group{-1};
  int tma_copy{0};
  bool has_async_producer{false};
  bool has_async_group{false};
  bool has_tma_copy{false};
};

std::unique_ptr<cuda::z3::IRExtraInfo> MakeRecursiveTimingExtra() {
  return std::make_unique<RecursiveTimingExtra>();
}

RecursiveTimingExtra *RecursiveExtra(IRStructure *node) {
  ICHECK(node != nullptr);
  return node->GetExtraInfo<RecursiveTimingExtra>();
}

const RecursiveTimingExtra *RecursiveExtra(const IRStructure *node) {
  ICHECK(node != nullptr);
  return node->GetExtraInfo<RecursiveTimingExtra>();
}

int64_t CheckedRecursiveInt64(__int128 value, const char *what) {
  ICHECK(value >= std::numeric_limits<int64_t>::min() &&
         value <= std::numeric_limits<int64_t>::max())
      << "Z3PipelinePlanning recursive: " << what << " overflows int64";
  return static_cast<int64_t>(value);
}

struct RecursiveListProblem {
  std::vector<IRStructure *> nodes;
  ControlNode *loop{nullptr};
  bool manual_schedule{false};
  bool require_stage_zero{false};
  int64_t num_stages{1};
  bool enable_offset{true};
  int64_t resource_bits{7};
  int64_t seed{42};
  bool recalculate_buffer_versions{true};
  bool verbose{false};
  Optional<String> debug_dir{std::nullopt};
  std::vector<int64_t> latencies;
  std::vector<int64_t> iis;
  std::vector<int64_t> resource_flags;
  std::vector<int64_t> manual_stages;
  std::vector<Z3DataDependency> data_deps;
  std::vector<Z3FixedRecurrence> fixed_recurrences;
  std::vector<Z3TaskPair> resource_deps;
  std::vector<Z3DataDependency> owner_exclusion_deps;
  std::vector<Z3TaskPair> pipe_order_deps;
  std::vector<Z3TaskPair> stage_order_deps;
  std::vector<Var> buffer_vars;
  std::vector<int64_t> buffer_sizes;
  std::vector<std::vector<int64_t>> memory_groups;
  std::string audit_path{"root"};
};

struct RecursiveListResult {
  Array<int64_t> starts;
  Array<int64_t> versions;
  Array<int64_t> model_versions;
  Array<int64_t> required_versions;
  std::unordered_map<Var, int64_t, ObjectPtrHash, ObjectPtrEqual> fixed_required;
  int64_t ii{0};
  std::vector<int64_t> raw_stages;
  std::vector<int64_t> residues;
  std::vector<int> physical_order;
};

RecursiveListResult ValidateRecursiveListSchedule(
    const RecursiveListProblem &problem, Array<int64_t> starts,
    Array<int64_t> returned_versions, int64_t ii,
    const std::string &diagnostic_suffix = "") {
  const size_t n = problem.latencies.size();
  ICHECK_EQ(starts.size(), n)
      << "Z3PipelinePlanning recursive: solver returned " << starts.size()
      << " starts for " << n << " tasks at " << problem.audit_path
      << diagnostic_suffix;
  ICHECK_EQ(returned_versions.size(), problem.buffer_sizes.size())
      << "Z3PipelinePlanning recursive: solver returned the wrong version "
         "vector length at "
      << problem.audit_path << diagnostic_suffix;
  ICHECK_GT(ii, 0) << "Z3PipelinePlanning recursive: non-positive II at "
                   << problem.audit_path << diagnostic_suffix;

  RecursiveListResult result;
  result.starts = std::move(starts);
  result.ii = ii;
  result.raw_stages.reserve(n);
  result.residues.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    ICHECK_GE(result.starts[i], 0)
        << "Z3PipelinePlanning recursive: negative start at "
        << problem.audit_path << diagnostic_suffix;
    result.raw_stages.push_back(result.starts[i] / ii);
    result.residues.push_back(result.starts[i] % ii);
    ICHECK(!problem.require_stage_zero || result.raw_stages.back() == 0)
        << "Z3PipelinePlanning recursive: straight-line/owner list task " << i
        << " escaped stage zero at " << problem.audit_path << diagnostic_suffix;
  }
  if (problem.manual_schedule) {
    ICHECK_EQ(problem.manual_stages.size(), n);
    for (size_t i = 0; i < n; ++i) {
      ICHECK_EQ(result.raw_stages[i], problem.manual_stages[i])
          << "Z3PipelinePlanning recursive: manual stage mismatch for task " << i
          << " at " << problem.audit_path << diagnostic_suffix;
    }
  }

  std::vector<int64_t> required_versions(problem.buffer_sizes.size(), 1);
  auto required_from_edge = [&](int64_t source, int64_t target,
                                int64_t latency) -> int64_t {
    __int128 numerator = static_cast<__int128>(latency) +
                         result.starts[source] - result.starts[target];
    if (numerator <= 0) return 1;
    return CheckedRecursiveInt64((numerator + ii - 1) / ii,
                                 "required buffer version count");
  };
  for (const auto &dependency : problem.data_deps) {
    int64_t source = dependency[0];
    int64_t target = dependency[1];
    int64_t distance = dependency[2];
    int64_t latency = dependency[3];
    ICHECK(source >= 0 && source < static_cast<int64_t>(n) && target >= 0 &&
           target < static_cast<int64_t>(n) && latency >= 0);
    if (distance < 0) {
      ICHECK_NE(distance, std::numeric_limits<int64_t>::min());
      int64_t buffer = -distance - 1;
      ICHECK(buffer >= 0 &&
             buffer < static_cast<int64_t>(required_versions.size()));
      required_versions[buffer] =
          std::max(required_versions[buffer],
                   required_from_edge(source, target, latency));
    }
  }
  for (const auto &recurrence : problem.fixed_recurrences) {
    int64_t required = required_from_edge(recurrence.source, recurrence.target,
                                          recurrence.latency);
    auto found = result.fixed_required.find(recurrence.storage);
    if (found == result.fixed_required.end()) {
      result.fixed_required.emplace(recurrence.storage, required);
    } else {
      found->second = std::max(found->second, required);
    }
    ICHECK_LE(required, recurrence.fixed_versions)
        << "Z3PipelinePlanning recursive: exact version request "
        << recurrence.fixed_versions << " for " << recurrence.storage
        << " is below required count " << required << " at "
        << problem.audit_path << diagnostic_suffix;
  }

  std::vector<int64_t> validated_versions;
  validated_versions.reserve(returned_versions.size());
  for (size_t i = 0; i < returned_versions.size(); ++i) {
    ICHECK_GT(returned_versions[i], 0)
        << "Z3PipelinePlanning recursive: non-positive selected version at "
        << problem.audit_path << diagnostic_suffix;
    int64_t selected = returned_versions[i];
    if (problem.recalculate_buffer_versions) {
      selected = std::max(selected, required_versions[i]);
    } else {
      ICHECK_GE(selected, required_versions[i])
          << "Z3PipelinePlanning recursive: under-versioned buffer " << i
          << " at " << problem.audit_path << diagnostic_suffix;
    }
    validated_versions.push_back(selected);
  }
  result.versions = Array<int64_t>(validated_versions);
  result.model_versions = returned_versions;
  result.required_versions = Array<int64_t>(required_versions);

  for (const auto &dependency : problem.data_deps) {
    int64_t source = dependency[0];
    int64_t target = dependency[1];
    int64_t distance = dependency[2];
    __int128 effective_distance = distance;
    if (distance < 0) effective_distance = result.versions[-distance - 1];
    __int128 actual = static_cast<__int128>(result.starts[target]) -
                      result.starts[source];
    __int128 required = static_cast<__int128>(dependency[3]) -
                        static_cast<__int128>(ii) * effective_distance;
    ICHECK(actual >= required)
        << "Z3PipelinePlanning recursive: augmented data inequality ("
        << source << ", " << target << ", " << distance << ", "
        << dependency[3] << ") failed at " << problem.audit_path
        << diagnostic_suffix;
  }
  for (const auto &pair : problem.pipe_order_deps) {
    ICHECK_LE(result.residues[pair.first], result.residues[pair.second])
        << "Z3PipelinePlanning recursive: pipe order failed at "
        << problem.audit_path << diagnostic_suffix;
  }
  for (const auto &pair : problem.stage_order_deps) {
    ICHECK_LE(result.raw_stages[pair.first], result.raw_stages[pair.second])
        << "Z3PipelinePlanning recursive: stage order failed at "
        << problem.audit_path << diagnostic_suffix;
  }
  for (const auto &pair : problem.resource_deps) {
    int left = pair.first;
    int right = pair.second;
    __int128 lr = result.residues[left];
    __int128 rr = result.residues[right];
    bool left_first = rr - lr >= problem.iis[left] &&
                      lr - rr + ii >= problem.iis[right];
    bool right_first = lr - rr >= problem.iis[right] &&
                       rr - lr + ii >= problem.iis[left];
    ICHECK(left_first || right_first)
        << "Z3PipelinePlanning recursive: resource exclusion failed at "
        << problem.audit_path << diagnostic_suffix;
  }
  for (const auto &owner : problem.owner_exclusion_deps) {
    int left = owner[0];
    int right = owner[1];
    __int128 lr = result.residues[left];
    __int128 rr = result.residues[right];
    bool left_first = rr - lr >= owner[2] && lr - rr + ii >= owner[3];
    bool right_first = lr - rr >= owner[3] && rr - lr + ii >= owner[2];
    ICHECK(left_first || right_first)
        << "Z3PipelinePlanning recursive: forward/reverse owner exclusion ("
        << left << ", " << right << ", " << owner[2] << ", " << owner[3]
        << ") failed at " << problem.audit_path << diagnostic_suffix;
  }
  if (!problem.enable_offset) {
    for (size_t left = 0; left < n; ++left) {
      for (size_t right = left + 1; right < n; ++right) {
        if ((problem.resource_flags[left] & problem.resource_flags[right]) == 0) {
          continue;
        }
        __int128 delta = static_cast<__int128>(result.starts[left]) -
                         result.starts[right];
        ICHECK(delta < ii && -delta < ii)
            << "Z3PipelinePlanning recursive: disabled-offset resource window "
               "failed at "
            << problem.audit_path << diagnostic_suffix;
      }
    }
  }

  for (const auto &group : problem.memory_groups) {
    ICHECK(!group.empty()) << "Z3PipelinePlanning recursive: empty memory group";
    int64_t capacity = group[0];
    ICHECK_GE(capacity, 0);
    __int128 used = 0;
    std::unordered_set<int64_t> members;
    for (size_t i = 1; i < group.size(); ++i) {
      int64_t member = group[i];
      ICHECK(member >= 0 &&
             member < static_cast<int64_t>(problem.buffer_sizes.size()));
      ICHECK(members.insert(member).second)
          << "Z3PipelinePlanning recursive: duplicate memory-group member";
      ICHECK_GE(problem.buffer_sizes[member], 0);
      used += static_cast<__int128>(result.versions[member]) *
              problem.buffer_sizes[member];
    }
    ICHECK(used <= capacity)
        << "Z3PipelinePlanning recursive: selected versions use "
        << CheckedRecursiveInt64(used, "memory-group usage")
        << " bytes above capacity " << capacity << " at "
        << problem.audit_path << diagnostic_suffix;
  }
  result.physical_order = Z3TopologicalPhysicalOrder(
      n, result.raw_stages, result.residues, problem.pipe_order_deps,
      problem.data_deps);
  return result;
}

RecursiveListResult SolveRecursiveList(const RecursiveListProblem &problem) {
  const size_t n = problem.latencies.size();
  if (n == 0) {
    RecursiveListResult empty;
    empty.ii = 1;
    empty.versions = Array<int64_t>(problem.buffer_sizes.size(), 1);
    empty.model_versions = empty.versions;
    empty.required_versions = empty.versions;
    return empty;
  }
  ICHECK_EQ(problem.iis.size(), n);
  ICHECK_EQ(problem.resource_flags.size(), n);
  auto scheduler = ffi::Function::GetGlobal(
      "tl.cuda.transform.z3_schedule_loop_python");
  ICHECK(scheduler.has_value())
      << "Z3PipelinePlanning recursive: "
         "tl.cuda.transform.z3_schedule_loop_python is not registered";
  Array<Array<int64_t>> owner_deps;
  for (const auto &dep : problem.owner_exclusion_deps) {
    owner_deps.push_back(Array<int64_t>{dep[0], dep[1], dep[2], dep[3]});
  }
  Array<int64_t> sizes(problem.buffer_sizes);
  Array<Array<int64_t>> groups;
  for (const auto &group : problem.memory_groups) {
    groups.push_back(Array<int64_t>(group));
  }
  using SolverResult = ffi::Tuple<Array<int64_t>, Array<int64_t>, int64_t>;
  SolverResult solved = scheduler.value()(
      problem.num_stages, Array<int64_t>(problem.latencies),
      Array<int64_t>(problem.iis), Array<int64_t>(problem.resource_flags),
      Z3DataDependenciesToFFI(problem.data_deps),
      Z3TaskPairsToFFI(problem.resource_deps), owner_deps,
      Z3TaskPairsToFFI(problem.pipe_order_deps), sizes, groups,
      Z3TaskPairsToFFI(problem.stage_order_deps), problem.enable_offset,
      Array<int64_t>(problem.manual_stages), problem.manual_schedule,
      problem.resource_bits, problem.seed,
      problem.recalculate_buffer_versions, problem.verbose, problem.debug_dir)
                          .cast<SolverResult>();
  // In particular, do not catch an UNSAT/error and substitute a serial tree.
  // The Python solver's legacy fallback is accepted only if every augmented
  // inequality is independently proved by this validator.
  return ValidateRecursiveListSchedule(problem, solved.get<0>(), solved.get<1>(),
                                       solved.get<2>());
}

struct RecursiveCapacityGroup {
  int64_t capacity{0};
  std::vector<Var> members;
  std::string owner_path;
};

struct RecursiveListRecord {
  RecursiveListProblem problem;
  RecursiveListResult result;
  std::vector<RecursiveCapacityGroup> capacity_groups;
};

struct RecursiveListState {
  std::vector<IRStructure *> original_nodes;
  bool manual_schedule{false};
  int64_t num_stages{1};
  bool enable_offset{true};
  int64_t resource_bits{7};
  int64_t seed{42};
  bool recalculate{true};
  bool verbose{false};
  Optional<String> debug_dir{std::nullopt};
  int64_t round_limit{1};
  std::string path{"root"};
};

class RecursiveZ3Planner {
 public:
  explicit RecursiveZ3Planner(ScheduledTIR *scheduled) : scheduled_(scheduled) {
    ICHECK(scheduled_ != nullptr);
    BuildAliasTable();
    CollectStorages();
    for (const auto &[storage, count] : scheduled_->metadata.requested_versions) {
      ICHECK_GT(count, 0);
      known_versions_[Canonical(storage)] = count;
      selected_origins_[Canonical(storage)] = "public exact version request";
    }
    PropagateAliasVersions();
    visible_versions_ = known_versions_;
    for (const auto &[storage, _] : scheduled_->metadata.requested_versions) {
      if (auto count = scheduled_->metadata.requested_versions.Get(storage);
          count.has_value() && count.value() == 1 &&
          !scheduled_->metadata.version_modes.count(storage)) {
        bare_one_requests_.insert(storage);
      }
    }
    std::vector<TaskNode *> tasks;
    cuda::z3::CollectAllTaskNodes(scheduled_->tree, &tasks, true);
    for (const TaskNode *task : tasks) {
      next_source_id_ = std::max(next_source_id_, task->GetSourceId() + 1);
    }
    std::function<void(IRStructure *)> find_controls = [&](IRStructure *node) {
      if (node->IsTask()) return;
      auto *control = static_cast<ControlNode *>(node);
      next_source_id_ = std::max(next_source_id_, control->GetSourceId() + 1);
      for (const auto &child : control->children) find_controls(child.get());
    };
    for (const auto &node : scheduled_->tree) find_controls(node.get());
  }

  void Run() {
    owners_ = cuda::z3::InferCompleteOwnerFrontier(scheduled_);
    ScheduleList(scheduled_->tree, nullptr);
    ValidateGlobalVersionAndCapacityContracts();
    RepairAndRefresh();
    cuda::z3::ValidateScheduledTIR(*scheduled_);
  }

  const std::unordered_map<Var, int64_t, ObjectPtrHash, ObjectPtrEqual> &
  final_versions() const {
    return known_versions_;
  }

  const std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> &
  bare_one_requests() const {
    return bare_one_requests_;
  }

 private:
  using VarMap = std::unordered_map<Var, Var, ObjectPtrHash, ObjectPtrEqual>;

  struct ScalarEdge {
    TaskNode *definition{nullptr};
    TaskNode *reader{nullptr};
    IRStructure *consumer_root{nullptr};
    int definition_index{-1};
    int consumer_index{-1};
    Var var;
  };

  Var Canonical(const Var &storage) const {
    auto found = alias_representative_.find(storage);
    return found == alias_representative_.end() ? storage : found->second;
  }

  std::vector<Var> AliasMembers(const Var &storage) const {
    Var representative = Canonical(storage);
    auto found = alias_members_.find(representative);
    if (found != alias_members_.end()) return found->second;
    return {storage};
  }

  void BuildAliasTable() {
    if (!scheduled_->metadata.has_buffer_aliases) return;
    for (const auto &[storage, peers] : scheduled_->metadata.buffer_aliases) {
      Var representative = storage;
      std::vector<Var> members{storage};
      for (const Var &peer : peers) members.push_back(peer);
      for (const Var &member : members) {
        auto found = alias_representative_.find(member);
        ICHECK(found == alias_representative_.end() ||
               found->second.same_as(representative))
            << "Z3PipelinePlanning recursive: overlapping alias groups for "
            << member;
        alias_representative_[member] = representative;
      }
      alias_members_[representative] = std::move(members);
    }
  }

  static bool VersionableStorage(const Buffer &buffer) {
    if (!buffer.defined()) return false;
    String scope = buffer.scope();
    return scope == "shared" || scope == "shared.dyn" || scope == "local" ||
           scope == "local.var" || scope == "local.fragment" ||
           scope == "register" || scope == "shared.tmem";
  }

  int RegisterStorage(const Var &storage, const Buffer &buffer = Buffer()) {
    Var canonical = Canonical(storage);
    auto found = storage_ids_.find(canonical);
    int id;
    if (found == storage_ids_.end()) {
      id = static_cast<int>(storage_vars_.size());
      storage_ids_.emplace(canonical, id);
      storage_vars_.push_back(canonical);
      storage_buffers_.push_back(Buffer());
      storage_sizes_.push_back(-1);
    } else {
      id = found->second;
    }
    if (buffer.defined() && !storage_buffers_[id].defined()) {
      storage_buffers_[id] = buffer;
    }
    return id;
  }

  int64_t StorageSize(const Var &storage) {
    int id = RegisterStorage(storage);
    if (storage_sizes_[id] >= 0) return storage_sizes_[id];
    const Buffer &buffer = storage_buffers_[id];
    ICHECK(buffer.defined())
        << "Z3PipelinePlanning recursive: no Buffer representative for "
        << storage;
    int64_t elem_bits = buffer->dtype.bits() * buffer->dtype.lanes();
    ICHECK_GT(elem_bits, 0);
    PrimExpr elements = IntImm(DataType::Int(64), 1);
    for (const PrimExpr &dim : buffer->shape) elements = elements * dim;
    arith::Analyzer analyzer;
    PrimExpr static_elements = analyzer.Simplify(elements);
    const auto *count = static_elements.as<IntImmNode>();
    ICHECK(count != nullptr)
        << "Z3PipelinePlanning recursive: dynamic storage size for " << storage
        << " requires an explicit static tl_z3_pipeline_buffer_sizes entry";
    __int128 bytes = (static_cast<__int128>(count->value) * elem_bits + 7) / 8;
    storage_sizes_[id] = CheckedRecursiveInt64(bytes, "buffer byte size");
    return storage_sizes_[id];
  }

  void CollectStorages() {
    if (scheduled_->metadata.kernel_root.defined()) {
      for (const Buffer &buffer : scheduled_->metadata.kernel_root->alloc_buffers) {
        RegisterStorage(buffer->data, buffer);
      }
    }
    std::vector<TaskNode *> tasks;
    cuda::z3::CollectAllTaskNodes(scheduled_->tree, &tasks, true);
    for (const TaskNode *task : tasks) {
      for (const BufferRegion &region : task->accesses.reads) {
        RegisterStorage(region->buffer->data, region->buffer);
      }
      for (const BufferRegion &region : task->accesses.writes) {
        RegisterStorage(region->buffer->data, region->buffer);
      }
      PostOrderVisit(task->stmt, [&](const ObjectRef &node) {
        if (const auto *decl = node.as<DeclBufferNode>()) {
          RegisterStorage(decl->buffer->data, decl->buffer);
        } else if (const auto *alloc = node.as<AllocBufferNode>()) {
          RegisterStorage(alloc->buffer->data, alloc->buffer);
        }
      });
    }
    std::function<void(IRStructure *)> collect_explicit = [&](IRStructure *node) {
      if (node->IsTask()) return;
      auto *control = static_cast<ControlNode *>(node);
      const auto &annotations = control->control->annotations;
      Array<Var> vars;
      if (auto value = annotations.Get(kZ3BufferVars); value.has_value()) {
        vars = Downcast<Array<Var>>(value.value());
      }
      Array<Integer> sizes = GetIntegerArray(annotations, kZ3BufferSizes);
      ICHECK_EQ(vars.size(), sizes.size())
          << "Z3PipelinePlanning recursive: buffer_vars/buffer_sizes length "
             "mismatch at "
          << NodePath(control);
      for (size_t i = 0; i < vars.size(); ++i) {
        int id = RegisterStorage(vars[i]);
        ICHECK_GE(sizes[i]->value, 0);
        if (storage_buffers_[id].defined()) {
          int64_t inferred = StorageSize(vars[i]);
          ICHECK_EQ(inferred, sizes[i]->value)
              << "Z3PipelinePlanning recursive: explicit size differs from the "
                 "static Buffer size for "
              << vars[i];
        }
        storage_sizes_[id] = sizes[i]->value;
      }
      for (const auto &child : control->children) collect_explicit(child.get());
    };
    for (const auto &node : scheduled_->tree) collect_explicit(node.get());
    for (const auto &[storage, _] : scheduled_->metadata.requested_versions) {
      int id = RegisterStorage(storage);
      ICHECK(storage_buffers_[id].defined() || storage_sizes_[id] >= 0)
          << "Z3PipelinePlanning recursive: version request refers to unknown "
             "storage "
          << storage;
      if (storage_buffers_[id].defined()) {
        ICHECK(VersionableStorage(storage_buffers_[id]))
            << "Z3PipelinePlanning recursive: public version request is outside "
               "on-chip storage for "
            << storage;
      }
    }
  }

  std::string NodePath(const IRStructure *node) const {
    if (node == nullptr) return "root";
    std::vector<int64_t> ids;
    for (const IRStructure *current = node; current != nullptr;
         current = current->GetParent()) {
      ids.push_back(current->GetSourceId());
    }
    std::ostringstream os;
    os << "root";
    for (auto it = ids.rbegin(); it != ids.rend(); ++it) os << "/" << *it;
    return os.str();
  }

  static bool IsNoOpTask(const TaskNode *task) {
    const auto *marker = task->stmt.as<AttrStmtNode>();
    if (marker == nullptr || task->HasGuards() || !task->ws_op_id.empty()) return false;
    const auto *evaluate = marker->body.as<EvaluateNode>();
    return evaluate != nullptr && is_zero(evaluate->value);
  }

  static bool SolverEligible(IRStructure *node) {
    if (node->IsControl()) return true;
    auto *task = static_cast<TaskNode *>(node);
    return task->solver_eligible && task->task_kind != TaskKind::kDeclaration &&
           !IsNoOpTask(task);
  }

  std::vector<IRStructure *> EligibleNodes(
      const std::vector<std::shared_ptr<IRStructure>> &children) const {
    std::vector<IRStructure *> result;
    for (const auto &child : children) {
      if (SolverEligible(child.get())) result.push_back(child.get());
    }
    return result;
  }

  RecursiveListState InitializeList(
      std::vector<std::shared_ptr<IRStructure>> &children, ControlNode *loop) {
    RecursiveListState state;
    state.path = NodePath(loop);
    for (const auto &child : children) {
      if (child->IsTask()) {
        auto *task = static_cast<TaskNode *>(child.get());
        if (IsNoOpTask(task)) task->solver_eligible = false;
      }
      RecursiveExtra(child.get())->manual_stage = child->GetStage();
    }
    state.original_nodes = EligibleNodes(children);
    bool has_unscheduled = false;
    for (IRStructure *node : state.original_nodes) {
      state.manual_schedule = state.manual_schedule || node->GetStage() >= 0;
      has_unscheduled = has_unscheduled || node->GetStage() < 0;
    }
    ICHECK(!(state.manual_schedule && has_unscheduled))
        << "Z3PipelinePlanning recursive: mixed staged/unscheduled list at "
        << state.path;
    state.round_limit = std::max<int64_t>(
        1, static_cast<int64_t>(state.original_nodes.size() *
                                    state.original_nodes.size() +
                                1));
    if (loop == nullptr) {
      for (IRStructure *node : state.original_nodes) {
        ICHECK(node->GetStage() <= 0)
            << "Z3PipelinePlanning recursive: nonzero root stage at " << state.path;
      }
      return state;
    }

    const auto &annotations = loop->control->annotations;
    for (const char *key : {"tl_pipeline_order", "tl_pipeline_stage",
                            "software_pipeline_order", "software_pipeline_stage",
                            "tl_pipeline_buffer_versions",
                            "tl_pipeline_scalar_clone_plan"}) {
      ICHECK(!annotations.count(key))
          << "Z3PipelinePlanning recursive: refusing to overwrite existing " << key
          << " at " << state.path;
    }
    int64_t explicit_stages = GetInteger(annotations, "num_stages", 0);
    ICHECK_GE(explicit_stages, 0);
    int max_manual = 0;
    for (IRStructure *node : state.original_nodes) {
      max_manual = std::max(max_manual, node->GetStage());
    }
    state.num_stages = explicit_stages > 0
                           ? explicit_stages
                           : static_cast<int64_t>(max_manual + 1);
    ICHECK_GT(state.num_stages, 0);
    state.enable_offset = GetBool(annotations, kZ3EnableOffset, true);
    state.resource_bits = GetInteger(annotations, kZ3ResourceBits, 7);
    state.seed = GetInteger(annotations, kZ3Seed, 42);
    state.recalculate =
        GetBool(annotations, kZ3RecalculateBufferVersions, true);
    state.verbose = GetBool(annotations, kZ3Verbose, false);
    if (auto value = annotations.Get(kZ3DebugDir); value.has_value()) {
      state.debug_dir = Downcast<String>(value.value());
    }
    state.round_limit =
        GetInteger(annotations, kZ3MaxCloneRounds, state.round_limit);
    ICHECK_GT(state.round_limit, 0);
    ICHECK(!state.manual_schedule || state.enable_offset || max_manual == 0)
        << "Z3PipelinePlanning recursive: nonzero manual stages require offset";

    auto assign_integer_metadata = [&](const char *key, auto setter,
                                       bool required = false) {
      Array<Integer> values = GetIntegerArray(annotations, key);
      if (values.empty()) {
        ICHECK(!required) << "Z3PipelinePlanning recursive: missing " << key
                          << " at " << state.path;
        return;
      }
      if (values.size() == children.size()) {
        for (size_t i = 0; i < children.size(); ++i) setter(children[i].get(), values[i]->value);
        return;
      }
      ICHECK_EQ(values.size(), state.original_nodes.size())
          << "Z3PipelinePlanning recursive: " << key
          << " must align with direct children or solver-eligible children at "
          << state.path;
      for (size_t i = 0; i < values.size(); ++i) {
        setter(state.original_nodes[i], values[i]->value);
      }
    };
    // Resource masks are read into the map used by BuildProblem.
    for (IRStructure *node : state.original_nodes) resource_flags_[node] = 0;
    if (Array<Integer> flags = GetIntegerArray(annotations, kZ3ResourceFlags);
        !flags.empty()) {
      auto assign = [&](IRStructure *node, int64_t value) {
        ICHECK_GE(value, 0);
        resource_flags_[node] = value;
      };
      if (flags.size() == children.size()) {
        for (size_t i = 0; i < flags.size(); ++i) assign(children[i].get(), flags[i]->value);
      } else {
        ICHECK_EQ(flags.size(), state.original_nodes.size());
        for (size_t i = 0; i < flags.size(); ++i) assign(state.original_nodes[i], flags[i]->value);
      }
    }
    assign_integer_metadata(kPipelineAsyncProducers, [](IRStructure *node, int64_t value) {
      auto *extra = RecursiveExtra(node);
      extra->has_async_producer = true;
      extra->async_producer = static_cast<int>(value);
    });
    assign_integer_metadata(kPipelineAsyncProducerGroups, [](IRStructure *node, int64_t value) {
      auto *extra = RecursiveExtra(node);
      extra->has_async_group = true;
      extra->async_group = static_cast<int>(value);
    });
    assign_integer_metadata(kPipelineTmaCopies, [](IRStructure *node, int64_t value) {
      auto *extra = RecursiveExtra(node);
      extra->has_tma_copy = true;
      extra->tma_copy = static_cast<int>(value);
    });
    return state;
  }

  int CurrentIndex(const std::vector<IRStructure *> &nodes,
                   const IRStructure *node) const {
    auto found = std::find(nodes.begin(), nodes.end(), node);
    ICHECK(found != nodes.end());
    return static_cast<int>(found - nodes.begin());
  }

  int OriginalIndex(const RecursiveListState &state,
                    const std::vector<IRStructure *> &nodes,
                    int64_t original) const {
    ICHECK(original >= 0 &&
           original < static_cast<int64_t>(state.original_nodes.size()))
        << "Z3PipelinePlanning recursive: original task index " << original
        << " is out of range at " << state.path;
    return CurrentIndex(nodes, state.original_nodes[original]);
  }

  int64_t NodeResourceFlags(IRStructure *node) const {
    auto explicit_flag = resource_flags_.find(node);
    if (explicit_flag != resource_flags_.end() && explicit_flag->second != 0) {
      return explicit_flag->second;
    }
    if (node->IsTask()) {
      return explicit_flag == resource_flags_.end() ? 0 : explicit_flag->second;
    }
    const auto *control = static_cast<const ControlNode *>(node);
    int64_t result = 0;
    for (const auto &child : control->children) {
      if (SolverEligible(child.get())) result |= NodeResourceFlags(child.get());
    }
    return result;
  }

  int64_t TaskTimestamp(IRStructure *subtree_root, const TaskNode *task,
                        bool producer_side) const {
    ICHECK(subtree_root != nullptr && task != nullptr);
    std::vector<const IRStructure *> path = task->PathFrom(subtree_root);
    ICHECK(!path.empty() && path.front() == subtree_root);
    __int128 timestamp = 0;
    for (size_t i = 0; i + 1 < path.size(); ++i) {
      ICHECK(path[i]->IsControl());
      const auto *control = static_cast<const ControlNode *>(path[i]);
      if (producer_side) timestamp += control->GetLastIterationOffset();
      timestamp += RecursiveExtra(path[i + 1])->start;
    }
    return CheckedRecursiveInt64(timestamp, "nested task timestamp");
  }

  int64_t DependencyLatency(const DepInfo &dep, bool reverse = false) const {
    __int128 latency = 0;
    for (const auto &[forward_producer, forward_consumer] : dep.task_pairs) {
      TaskNode *producer = reverse ? forward_consumer : forward_producer;
      TaskNode *consumer = reverse ? forward_producer : forward_consumer;
      IRStructure *producer_root = reverse ? dep.cons_node : dep.prod_node;
      IRStructure *consumer_root = reverse ? dep.prod_node : dep.cons_node;
      __int128 value = static_cast<__int128>(TaskTimestamp(producer_root, producer, true)) +
                       producer->GetLatency() -
                       TaskTimestamp(consumer_root, consumer, false);
      latency = std::max(latency, value);
    }
    latency = std::max<__int128>(latency, 0);
    return CheckedRecursiveInt64(latency, "nested dependency latency");
  }

  BufferVersionMap KnownVersionsForAnalysis() const {
    BufferVersionMap result;
    for (const auto &[storage, count] : visible_versions_) {
      ICHECK(count > 0 && count <= std::numeric_limits<int>::max());
      result.Set(storage, static_cast<int>(count));
    }
    return result;
  }

  RecursiveListProblem BuildProblem(
      const std::vector<std::shared_ptr<IRStructure>> &children,
      ControlNode *loop, const RecursiveListState &state,
      std::vector<RecursiveCapacityGroup> *capacity_groups) {
    RecursiveListProblem problem;
    problem.nodes = EligibleNodes(children);
    problem.loop = loop;
    problem.manual_schedule = state.manual_schedule;
    problem.require_stage_zero = loop == nullptr;
    problem.num_stages = state.num_stages;
    problem.enable_offset = state.enable_offset;
    problem.resource_bits = state.resource_bits;
    problem.seed = state.seed;
    problem.recalculate_buffer_versions = state.recalculate;
    problem.verbose = state.verbose;
    problem.debug_dir = state.debug_dir;
    problem.audit_path = state.path;
    const size_t n = problem.nodes.size();
    for (IRStructure *node : problem.nodes) {
      if (node->IsTask()) {
        ICHECK_GT(node->GetII(), 0)
            << "Z3PipelinePlanning recursive: task at " << NodePath(node)
            << " has no positive T.Task ii; run nested task inference first";
      }
      ICHECK_GE(node->GetLatency(), 0);
      problem.latencies.push_back(node->GetLatency());
      problem.iis.push_back(node->GetII() > 0 ? node->GetII() : 1);
      problem.resource_flags.push_back(NodeResourceFlags(node));
      if (state.manual_schedule) {
        int stage = RecursiveExtra(node)->manual_stage;
        ICHECK_GE(stage, 0);
        problem.manual_stages.push_back(stage);
      }
    }

    std::unordered_map<Var, int, ObjectPtrHash, ObjectPtrEqual> local_buffer;
    auto add_variable_buffer = [&](const Var &storage) -> int {
      Var canonical = Canonical(storage);
      auto found = local_buffer.find(canonical);
      if (found != local_buffer.end()) return found->second;
      int id = static_cast<int>(problem.buffer_vars.size());
      local_buffer.emplace(canonical, id);
      problem.buffer_vars.push_back(canonical);
      problem.buffer_sizes.push_back(StorageSize(canonical));
      return id;
    };
    auto fixed_count = [&](const Var &storage) -> int64_t {
      auto found = visible_versions_.find(Canonical(storage));
      return found == visible_versions_.end() ? 0 : found->second;
    };
    auto encode_distance = [&](const std::optional<Var> &storage,
                               int source, int target, int64_t latency,
                               int *fixed_count_out = nullptr) -> int64_t {
      if (!storage.has_value()) return 1;
      Var canonical = Canonical(storage.value());
      int id = RegisterStorage(canonical);
      if (storage_buffers_[id].defined() && !VersionableStorage(storage_buffers_[id])) {
        return 1;
      }
      if (int64_t fixed = fixed_count(canonical); fixed > 0) {
        problem.fixed_recurrences.push_back(
            {source, target, latency, fixed, canonical});
        if (fixed_count_out != nullptr) *fixed_count_out = fixed;
        return fixed;
      }
      int local = add_variable_buffer(canonical);
      return -static_cast<int64_t>(local) - 1;
    };

    DependencyCache cache;
    std::vector<DepInfo> deps = cuda::z3::AnalyzeDependencies(
        problem.nodes, loop, KnownVersionsForAnalysis(), owners_, &cache);
    for (const DepInfo &dep : deps) {
      int source = CurrentIndex(problem.nodes, dep.prod_node);
      int target = CurrentIndex(problem.nodes, dep.cons_node);
      int64_t latency = DependencyLatency(dep);
      if (dep.kind == DependencyKind::kOwnerExclusion) {
        int64_t reverse = DependencyLatency(dep, true);
        if (source > target) {
          std::swap(source, target);
          std::swap(latency, reverse);
        }
        problem.owner_exclusion_deps.push_back({source, target, latency, reverse});
        continue;
      }
      int64_t distance = dep.distance;
      if (distance < 0) distance = encode_distance(dep.storage, source, target, latency);
      AddZ3DataDependency(&problem.data_deps, source, target, distance, latency);
    }

    if (loop != nullptr) {
      const auto &annotations = loop->control->annotations;
      Array<Var> original_buffers;
      if (auto value = annotations.Get(kZ3BufferVars); value.has_value()) {
        original_buffers = Downcast<Array<Var>>(value.value());
      }
      for (const Array<Integer> &dep : GetNestedIntegerArrays(annotations, kZ3DataDeps)) {
        ICHECK_EQ(dep.size(), 4U);
        int source = OriginalIndex(state, problem.nodes, dep[0]->value);
        int target = OriginalIndex(state, problem.nodes, dep[1]->value);
        int64_t latency = dep[3]->value;
        ICHECK_GE(latency, 0);
        int64_t distance = dep[2]->value;
        if (distance < 0) {
          ICHECK_NE(distance, std::numeric_limits<int64_t>::min());
          int64_t original = -distance - 1;
          ICHECK(original >= 0 && original < static_cast<int64_t>(original_buffers.size()));
          std::optional<Var> storage = original_buffers[original];
          distance = encode_distance(storage, source, target, latency);
        }
        AddZ3DataDependency(&problem.data_deps, source, target, distance, latency);
      }
      auto parse_pairs = [&](const char *key, bool resource) {
        std::vector<Z3TaskPair> pairs;
        for (const Array<Integer> &pair : GetNestedIntegerArrays(annotations, key)) {
          ICHECK_EQ(pair.size(), 2U);
          int left = OriginalIndex(state, problem.nodes, pair[0]->value);
          int right = OriginalIndex(state, problem.nodes, pair[1]->value);
          ICHECK(!resource || left != right);
          AddZ3TaskPair(&pairs, left, right, resource);
        }
        return pairs;
      };
      for (const Array<Integer> &owner :
           GetNestedIntegerArrays(annotations, kZ3OwnerExclusionDeps)) {
        ICHECK_EQ(owner.size(), 4U);
        int left = OriginalIndex(state, problem.nodes, owner[0]->value);
        int right = OriginalIndex(state, problem.nodes, owner[1]->value);
        ICHECK(left != right && owner[2]->value >= 0 && owner[3]->value >= 0);
        if (left > right) {
          std::swap(left, right);
          problem.owner_exclusion_deps.push_back(
              {left, right, owner[3]->value, owner[2]->value});
        } else {
          problem.owner_exclusion_deps.push_back(
              {left, right, owner[2]->value, owner[3]->value});
        }
      }
      problem.resource_deps = parse_pairs(kZ3ResourceDeps, true);
      problem.pipe_order_deps = parse_pairs(kZ3PipeOrderDeps, false);
      problem.stage_order_deps = parse_pairs(kZ3StageOrderDeps, false);
      for (const Array<Integer> &group : GetNestedIntegerArrays(annotations, kZ3MemoryGroups)) {
        ICHECK(!group.empty());
        RecursiveCapacityGroup global;
        global.capacity = group[0]->value;
        global.owner_path = state.path;
        ICHECK_GE(global.capacity, 0);
        std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> seen;
        for (size_t i = 1; i < group.size(); ++i) {
          int64_t original = group[i]->value;
          ICHECK(original >= 0 && original < static_cast<int64_t>(original_buffers.size()));
          Var storage = Canonical(original_buffers[original]);
          ICHECK(seen.insert(storage).second)
              << "Z3PipelinePlanning recursive: duplicate/alias memory-group member "
              << storage;
          global.members.push_back(storage);
          // Capacity members are versioned allocations even when this list has
          // no recurring edge that would otherwise create a solver variable.
          if (fixed_count(storage) == 0) add_variable_buffer(storage);
        }
        capacity_groups->push_back(std::move(global));
      }
    }

    for (size_t i = 0; i < n; ++i) {
      for (size_t j = i + 1; j < n; ++j) {
        if ((problem.resource_flags[i] & problem.resource_flags[j]) != 0) {
          AddZ3TaskPair(&problem.resource_deps, i, j, true);
          if (state.manual_schedule) AddZ3TaskPair(&problem.pipe_order_deps, i, j);
        }
      }
    }
    for (const ScalarEdge &edge : CollectScalarEdges(problem.nodes)) {
      if (IsCloneableDefinition(edge.definition)) continue;
      AddZ3TaskPair(&problem.stage_order_deps, edge.definition_index,
                    edge.consumer_index);
      AddZ3TaskPair(&problem.stage_order_deps, edge.consumer_index,
                    edge.definition_index);
    }

    // Fixed requests and versions selected by already-scheduled children occupy
    // physical capacity but no solver variable position.
    for (const RecursiveCapacityGroup &global : *capacity_groups) {
      __int128 remaining = global.capacity;
      std::vector<int64_t> local_group{0};
      for (const Var &storage : global.members) {
        if (int64_t fixed = fixed_count(storage); fixed > 0) {
          remaining -= static_cast<__int128>(fixed) * StorageSize(storage);
        } else if (local_buffer.count(storage)) {
          local_group.push_back(local_buffer.at(storage));
        }
      }
      local_group[0] = CheckedRecursiveInt64(remaining, "remaining memory capacity");
      if (local_group.size() > 1 || remaining < 0) {
        problem.memory_groups.push_back(std::move(local_group));
      }
    }
    return problem;
  }

  static ffi::Optional<Bind> DirectTaskBind(const TaskNode *task) {
    if (task == nullptr || task->task_kind != TaskKind::kOperation) return std::nullopt;
    const auto *marker = task->stmt.as<AttrStmtNode>();
    if (marker == nullptr || marker->attr_key != cuda::z3::kCudaTask) return std::nullopt;
    if (const auto *bind = marker->body.as<BindNode>()) return GetRef<Bind>(bind);
    return std::nullopt;
  }

  static bool IsCloneableDefinition(const TaskNode *task) {
    ffi::Optional<Bind> bind = DirectTaskBind(task);
    if (!bind.defined() || !task->solver_eligible) return false;
    if (!task->accesses.writes.empty() || task->accesses.write_vars.size() != 1 ||
        !task->accesses.write_vars.front().same_as(bind.value()->var)) {
      return false;
    }
    bool pure = true;
    PostOrderVisit(bind.value()->value, [&](const ObjectRef &node) {
      if (const auto *call = node.as<CallNode>()) {
        if (!ParseOperator(GetRef<Call>(call)).defined() &&
            SideEffect(GetRef<Call>(call)) > CallEffectKind::kReadState) {
          pure = false;
        }
      }
    });
    return pure;
  }

  std::vector<ScalarEdge> CollectScalarEdges(
      const std::vector<IRStructure *> &nodes) const {
    std::unordered_map<TaskNode *, int> task_root;
    std::unordered_map<Var, TaskNode *, ObjectPtrHash, ObjectPtrEqual> writers;
    for (size_t i = 0; i < nodes.size(); ++i) {
      std::vector<TaskNode *> tasks;
      cuda::z3::CollectAllTaskNodes(nodes[i], &tasks, true);
      for (TaskNode *task : tasks) {
        task_root[task] = static_cast<int>(i);
        for (const Var &var : task->accesses.write_vars) {
          auto [found, inserted] = writers.emplace(var, task);
          ICHECK(inserted) << "Z3PipelinePlanning recursive: duplicate scalar writer for "
                           << var;
        }
      }
    }
    std::vector<ScalarEdge> edges;
    std::set<std::tuple<TaskNode *, TaskNode *, int, int>> seen;
    for (const auto &[reader, consumer] : task_root) {
      for (const Var &var : reader->accesses.read_vars) {
        auto found = writers.find(var);
        if (found == writers.end() || found->second == reader) continue;
        int definition = task_root.at(found->second);
        if (definition == consumer) continue;
        auto key = std::make_tuple(found->second, reader, definition, consumer);
        if (!seen.insert(key).second) continue;
        edges.push_back(
            {found->second, reader, nodes[consumer], definition, consumer, var});
      }
    }
    std::sort(edges.begin(), edges.end(), [](const ScalarEdge &a, const ScalarEdge &b) {
      return std::tie(a.definition_index, a.consumer_index, a.definition, a.reader) <
             std::tie(b.definition_index, b.consumer_index, b.definition, b.reader);
    });
    return edges;
  }

  void PropagateAliasVersions() {
    for (const auto &[representative, members] : alias_members_) {
      int64_t selected = 0;
      for (const Var &member : members) {
        auto found = known_versions_.find(member);
        if (found != known_versions_.end()) selected = std::max(selected, found->second);
      }
      if (selected == 0) continue;
      for (const Var &member : members) known_versions_[member] = selected;
    }
  }

  void MergeSelectedVersion(const Var &storage, int64_t count,
                            const std::string &owner_path) {
    ICHECK_GT(count, 0);
    Var canonical = Canonical(storage);
    int64_t merged = count;
    auto found = known_versions_.find(canonical);
    if (found != known_versions_.end()) merged = std::max(merged, found->second);
    if (found == known_versions_.end() || merged > found->second) {
      selected_origins_[canonical] = owner_path;
    }
    for (const Var &member : AliasMembers(canonical)) {
      known_versions_[member] = merged;
      if (merged == count) selected_origins_[member] = owner_path;
    }
  }

  std::vector<ScalarEdge> FindNeededClones(
      const RecursiveListProblem &problem,
      const RecursiveListResult &result) const {
    std::vector<ScalarEdge> needed;
    std::set<std::pair<TaskNode *, IRStructure *>> seen;
    for (const ScalarEdge &edge : CollectScalarEdges(problem.nodes)) {
      if (result.raw_stages[edge.definition_index] ==
          result.raw_stages[edge.consumer_index]) {
        continue;
      }
      ICHECK(IsCloneableDefinition(edge.definition))
          << "Z3PipelinePlanning recursive: noncloneable scalar definition at "
          << NodePath(problem.nodes[edge.definition_index])
          << " crosses stages into " << NodePath(edge.consumer_root)
          << "; keep definition and consumer in one stage";
      if (seen.emplace(edge.definition, edge.consumer_root).second) {
        needed.push_back(edge);
      }
    }
    return needed;
  }

  size_t MaterializeRequiredScalarClones(
      std::vector<std::shared_ptr<IRStructure>> &children, ControlNode *loop,
      const RecursiveListState &state,
      const std::vector<ScalarEdge> &needed) {
    size_t inserted = 0;
    std::vector<IRStructure *> consumer_order;
    for (const auto &child : children) consumer_order.push_back(child.get());
    for (IRStructure *consumer : consumer_order) {
      std::vector<ScalarEdge> group;
      for (const ScalarEdge &edge : needed) {
        if (edge.consumer_root == consumer) group.push_back(edge);
      }
      if (group.empty()) continue;
      std::vector<std::shared_ptr<TaskNode>> clones;
      std::vector<std::pair<Var, Var>> renames;
      clones.reserve(group.size());
      for (const ScalarEdge &edge : group) {
        ffi::Optional<Bind> bind = DirectTaskBind(edge.definition);
        ICHECK(bind.defined());
        auto clone = std::static_pointer_cast<TaskNode>(edge.definition->Clone());
        Var fresh = bind.value()->var.copy_with_suffix("_z3_clone");
        clone->SubstituteVar(bind.value()->var, fresh);
        clone->SetSourceId(next_source_id_++);
        int stage = state.manual_schedule ? RecursiveExtra(consumer)->manual_stage
                                          : cuda::z3::kUnscheduledStage;
        clone->SetSchedule(stage, 0);
        auto *extra = RecursiveExtra(clone.get());
        extra->start = 0;
        extra->manual_stage = stage;
        extra->async_producer = 0;
        extra->async_group = -1;
        extra->tma_copy = 0;
        extra->has_async_producer = false;
        extra->has_async_group = false;
        extra->has_tma_copy = false;
        renames.emplace_back(bind.value()->var, fresh);
        clones.push_back(std::move(clone));
      }
      // Rewire chained and diamond inputs before any clone enters the tree.
      for (auto &clone : clones) {
        for (const auto &[old_var, fresh] : renames) {
          clone->SubstituteVar(old_var, fresh);
        }
      }
      // Control substitution covers header, guards, descendants and regions.
      for (const auto &[old_var, fresh] : renames) {
        consumer->SubstituteVar(old_var, fresh);
      }

      std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> clone_outputs;
      for (const auto &clone : clones) {
        for (const Var &var : clone->accesses.write_vars) clone_outputs.insert(var);
      }
      std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> available;
      std::vector<std::shared_ptr<TaskNode>> ordered;
      std::vector<bool> emitted(clones.size(), false);
      while (ordered.size() < clones.size()) {
        bool progress = false;
        for (size_t i = 0; i < clones.size(); ++i) {
          if (emitted[i]) continue;
          bool ready = true;
          for (const Var &var : clones[i]->accesses.read_vars) {
            if (clone_outputs.count(var) && !available.count(var)) {
              ready = false;
              break;
            }
          }
          if (!ready) continue;
          for (const Var &var : clones[i]->accesses.write_vars) available.insert(var);
          ordered.push_back(clones[i]);
          emitted[i] = true;
          progress = true;
        }
        ICHECK(progress)
            << "Z3PipelinePlanning recursive: scalar clone dependency cycle at "
            << state.path;
      }
      auto position = std::find_if(children.begin(), children.end(),
                                   [&](const auto &child) {
                                     return child.get() == consumer;
                                   });
      ICHECK(position != children.end());
      children.insert(position, ordered.begin(), ordered.end());
      inserted += ordered.size();
    }
    cuda::z3::RepairParentsAndIndices(children, loop);
    RepairAndRefresh();
    return inserted;
  }

  void ValidateReorderSafety(
      const std::vector<std::shared_ptr<IRStructure>> &children,
      ControlNode *loop, const RecursiveListProblem &problem,
      const std::vector<int> &physical_order) const {
    std::unordered_map<const IRStructure *, size_t> final_position;
    std::vector<IRStructure *> ordered;
    ordered.reserve(problem.nodes.size());
    for (int index : physical_order) ordered.push_back(problem.nodes[index]);
    size_t next = 0;
    for (const auto &child : children) {
      if (SolverEligible(child.get())) {
        final_position[ordered[next++]] = final_position.size();
      } else {
        final_position[child.get()] = final_position.size();
      }
    }
    if (loop == nullptr) {
      bool nonidentity = false;
      for (size_t i = 0; i < physical_order.size(); ++i) {
        nonidentity = nonidentity || physical_order[i] != static_cast<int>(i);
      }
      if (nonidentity) {
        for (const auto &child : children) {
          if (!child->IsTask()) continue;
          Z3FlatReorderSafetyVisitor safety;
          safety(static_cast<const TaskNode *>(child.get())->stmt);
          ICHECK(!safety.has_non_buffer_side_effect())
              << "Z3PipelinePlanning recursive: root nonidentity reorder of an "
                 "opaque/non-buffer side effect is unsafe";
        }
      }
      for (const auto &dep : problem.data_deps) {
        if (dep[2] != 0) continue;
        ICHECK(final_position.at(problem.nodes[dep[0]]) <=
               final_position.at(problem.nodes[dep[1]]))
            << "Z3PipelinePlanning recursive: root buffer/scalar dependency was "
               "reordered unsafely";
      }
    }
    // A declaration must still dominate every direct subtree that touches its
    // storage. Declarations themselves retain their original sibling slot.
    for (const auto &declaration : children) {
      if (!declaration->IsTask()) continue;
      const auto *task = static_cast<const TaskNode *>(declaration.get());
      if (task->task_kind != TaskKind::kDeclaration) continue;
      const auto *marker = task->stmt.as<AttrStmtNode>();
      Buffer buffer;
      if (const auto *alloc = marker->body.as<AllocBufferNode>()) {
        buffer = alloc->buffer;
      } else if (const auto *decl = marker->body.as<DeclBufferNode>()) {
        buffer = decl->buffer;
      }
      ICHECK(buffer.defined());
      for (const auto &other : children) {
        if (other.get() == declaration.get() || !other->TouchesStorage(buffer->data)) {
          continue;
        }
        ICHECK(final_position.at(other.get()) > final_position.at(declaration.get()))
            << "Z3PipelinePlanning recursive: reorder moved a use before its "
               "buffer declaration at "
            << NodePath(declaration.get());
      }
    }
  }

  std::optional<int64_t> StaticTripCount(const ControlNode *control) const {
    arith::Analyzer analyzer;
    PrimExpr extent = analyzer.Simplify(control->control->extent);
    PrimExpr step = control->control->step.defined()
                        ? analyzer.Simplify(control->control->step.value())
                        : IntImm(extent.dtype(), 1);
    const auto *extent_value = extent.as<IntImmNode>();
    const auto *step_value = step.as<IntImmNode>();
    if (extent_value == nullptr || step_value == nullptr) return std::nullopt;
    ICHECK_GT(step_value->value, 0)
        << "Z3PipelinePlanning recursive: static loop step must be positive";
    if (extent_value->value <= 0) return 0;
    return CheckedRecursiveInt64(
        (static_cast<__int128>(extent_value->value) + step_value->value - 1) /
            step_value->value,
        "loop trip count");
  }

  std::pair<int64_t, int64_t> FinalizeControlTiming(
      ControlNode *control, const RecursiveListProblem &problem,
      const RecursiveListResult &result) {
    __int128 first_span = 0;
    for (size_t i = 0; i < problem.nodes.size(); ++i) {
      first_span = std::max(first_span,
                            static_cast<__int128>(result.starts[i]) +
                                problem.nodes[i]->GetLatency());
    }
    int64_t first = CheckedRecursiveInt64(first_span, "first-iteration span");
    std::optional<int64_t> trip = StaticTripCount(control);
    auto override_value = control->control->annotations.Get(kZ3ControlTotalLatency);
    int64_t total = 0;
    if (!trip.has_value()) {
      ICHECK(override_value.has_value())
          << "Z3PipelinePlanning recursive: dynamic-trip control at "
          << NodePath(control) << " requires " << kZ3ControlTotalLatency
          << "; no heuristic trip count is permitted";
      total = Downcast<IntImm>(override_value.value())->value;
      ICHECK_GE(total, first)
          << "Z3PipelinePlanning recursive: dynamic total-latency override is "
             "below the first-iteration span";
    } else {
      if (*trip == 0) {
        total = 0;
      } else {
        total = CheckedRecursiveInt64(
            first_span + static_cast<__int128>(*trip - 1) * result.ii,
            "control total latency");
      }
      if (override_value.has_value()) {
        ICHECK_EQ(Downcast<IntImm>(override_value.value())->value, total)
            << "Z3PipelinePlanning recursive: static total-latency override "
               "does not equal the derived value";
      }
    }
    control->SetIterationTiming(result.ii, first, total);
    return {first, control->GetLastIterationOffset()};
  }

  void AnnotateList(std::vector<std::shared_ptr<IRStructure>> &children,
                    ControlNode *loop, const RecursiveListProblem &problem,
                    const RecursiveListResult &result, int64_t clone_rounds,
                    int64_t first_span, int64_t last_iter_offset) {
    std::unordered_map<const IRStructure *, size_t> source_index;
    for (size_t i = 0; i < problem.nodes.size(); ++i) {
      source_index.emplace(problem.nodes[i], i);
    }
    Array<String> paths;
    Array<Integer> starts;
    Array<Integer> raw_stages;
    Array<Integer> residues;
    Array<Integer> legacy_starts;
    Array<Integer> async_producers;
    Array<Integer> async_groups;
    Array<Integer> tma_copies;
    bool has_async_producers = false;
    bool has_async_groups = false;
    bool has_tma_copies = false;
    for (const auto &child : children) {
      paths.push_back(String(NodePath(child.get())));
      auto found = source_index.find(child.get());
      if (found == source_index.end()) {
        starts.push_back(Integer(IntImm(DataType::Int(64), 0)));
        raw_stages.push_back(Integer(child->GetStage()));
        residues.push_back(Integer(0));
      } else {
        size_t index = found->second;
        starts.push_back(Integer(IntImm(DataType::Int(64), result.starts[index])));
        raw_stages.push_back(Integer(IntImm(DataType::Int(64), result.raw_stages[index])));
        residues.push_back(Integer(IntImm(DataType::Int(64), result.residues[index])));
        legacy_starts.push_back(Integer(IntImm(DataType::Int(64), result.starts[index])));
      }
      if (SolverEligible(child.get())) {
        const auto *extra = RecursiveExtra(child.get());
        has_async_producers |= extra->has_async_producer;
        has_async_groups |= extra->has_async_group;
        has_tma_copies |= extra->has_tma_copy;
        async_producers.push_back(Integer(extra->async_producer));
        async_groups.push_back(Integer(extra->async_group));
        tma_copies.push_back(Integer(extra->tma_copy));
      }
    }
    if (loop == nullptr) return;
    Map<String, Any> annotations = loop->control->annotations;
    annotations.Set(kZ3AuditII, IntImm(DataType::Int(64), result.ii));
    annotations.Set(kZ3AuditStarts, legacy_starts);
    annotations.Set(kZ3AuditChildPaths, paths);
    annotations.Set(kZ3AuditChildStarts, starts);
    annotations.Set(kZ3AuditChildStages, raw_stages);
    annotations.Set(kZ3AuditChildResidues, residues);
    annotations.Set(kZ3AuditFirstSpan,
                    IntImm(DataType::Int(64), first_span));
    annotations.Set(kZ3AuditLastIterOffset,
                    IntImm(DataType::Int(64), last_iter_offset));
    annotations.Set(kZ3AuditCloneRounds, Integer(clone_rounds));
    if (has_async_producers) annotations.Set(kPipelineAsyncProducers, async_producers);
    if (has_async_groups) annotations.Set(kPipelineAsyncProducerGroups, async_groups);
    if (has_tma_copies) annotations.Set(kPipelineTmaCopies, tma_copies);
    loop->control.CopyOnWrite()->annotations = std::move(annotations);
  }

  void ScheduleList(std::vector<std::shared_ptr<IRStructure>> &children,
                    ControlNode *loop) {
    cuda::z3::RepairParentsAndIndices(children, loop);
    // Sibling owners solve independently. Restore the inherited view before
    // each child, then expose every child's selected versions to this parent.
    const auto inherited_versions = visible_versions_;
    RecursiveListState state = InitializeList(children, loop);
    for (auto &child : children) {
      if (child->IsControl()) {
        visible_versions_ = inherited_versions;
        auto *control = static_cast<ControlNode *>(child.get());
        ScheduleList(control->children, control);
      }
    }
    visible_versions_ = known_versions_;

    RecursiveListProblem problem;
    RecursiveListResult result;
    std::vector<RecursiveCapacityGroup> capacity_groups;
    int64_t rounds = 0;
    size_t clone_count = 0;
    const size_t clone_bound = state.original_nodes.size() * state.original_nodes.size();
    for (;;) {
      ICHECK(rounds < state.round_limit)
          << "Z3PipelinePlanning recursive: clone fixed point exceeded "
          << state.round_limit << " rounds at " << state.path;
      capacity_groups.clear();
      problem = BuildProblem(children, loop, state, &capacity_groups);
      result = SolveRecursiveList(problem);
      ++rounds;
      std::vector<ScalarEdge> needed = FindNeededClones(problem, result);
      if (needed.empty()) break;
      ICHECK(clone_count + needed.size() <= clone_bound)
          << "Z3PipelinePlanning recursive: clone plan exceeded the finite "
             "definition/consumer bound "
          << clone_bound << " at " << state.path;
      clone_count += MaterializeRequiredScalarClones(children, loop, state, needed);
    }

    int64_t min_stage = 0;
    if (!result.raw_stages.empty()) {
      min_stage = *std::min_element(result.raw_stages.begin(), result.raw_stages.end());
    }
    for (size_t i = 0; i < problem.nodes.size(); ++i) {
      int64_t stage = result.raw_stages[i] - min_stage;
      ICHECK(stage >= 0 && stage <= std::numeric_limits<int>::max());
      problem.nodes[i]->SetStage(static_cast<int>(stage));
      RecursiveExtra(problem.nodes[i])->start = result.starts[i];
    }
    ValidateReorderSafety(children, loop, problem, result.physical_order);
    std::vector<std::shared_ptr<IRStructure>> ordered;
    ordered.reserve(problem.nodes.size());
    // Recover the existing owning pointers without introducing a second tree.
    for (int index : result.physical_order) {
      IRStructure *wanted = problem.nodes[index];
      auto found = std::find_if(children.begin(), children.end(),
                                [&](const auto &child) { return child.get() == wanted; });
      ICHECK(found != children.end());
      ordered.push_back(*found);
    }
    size_t next = 0;
    for (auto &child : children) {
      if (SolverEligible(child.get())) {
        child = ordered[next++];
      } else if (child->IsTask() &&
                 static_cast<TaskNode *>(child.get())->task_kind != TaskKind::kDeclaration) {
        child->SetStage(0);
        RecursiveExtra(child.get())->start = 0;
      }
    }
    ICHECK_EQ(next, ordered.size());
    cuda::z3::RepairParentsAndIndices(children, loop);

    int64_t first_span = 0;
    int64_t last_offset = 0;
    if (loop != nullptr) {
      std::tie(first_span, last_offset) = FinalizeControlTiming(loop, problem, result);
    }
    AnnotateList(children, loop, problem, result, rounds, first_span, last_offset);
    records_.push_back({problem, result, capacity_groups});
    for (size_t i = 0; i < problem.buffer_vars.size(); ++i) {
      MergeSelectedVersion(problem.buffer_vars[i], result.versions[i], state.path);
    }
  }

  void ValidateGlobalVersionAndCapacityContracts() {
    for (const RecursiveListRecord &record : records_) {
      RecursiveListProblem final_problem = record.problem;
      final_problem.recalculate_buffer_versions = false;
      Array<int64_t> merged;
      std::string origins;
      for (size_t i = 0; i < final_problem.buffer_vars.size(); ++i) {
        const Var &storage = final_problem.buffer_vars[i];
        auto found = known_versions_.find(storage);
        ICHECK(found != known_versions_.end());
        merged.push_back(found->second);
        auto origin = selected_origins_.find(storage);
        if (origin != selected_origins_.end()) origins += "; " + origin->second;
      }
      // This repeats every data/owner/resource inequality with the final
      // globally merged versions; an invalidated parent is never accepted.
      ValidateRecursiveListSchedule(final_problem, record.result.starts, merged,
                                    record.result.ii, origins);
      for (size_t i = 0; i < final_problem.buffer_vars.size(); ++i) {
        required_versions_[final_problem.buffer_vars[i]] =
            std::max(required_versions_[final_problem.buffer_vars[i]],
                     record.result.required_versions[i]);
      }
      for (const auto &[storage, required] : record.result.fixed_required) {
        required_versions_[storage] = std::max(required_versions_[storage], required);
      }
      for (const RecursiveCapacityGroup &group : record.capacity_groups) {
        __int128 used = 0;
        std::string owner_paths = group.owner_path;
        for (const Var &storage : group.members) {
          auto found = known_versions_.find(storage);
          ICHECK(found != known_versions_.end())
              << "Z3PipelinePlanning recursive: capacity group has no selected "
                 "count for "
              << storage;
          used += static_cast<__int128>(found->second) * StorageSize(storage);
          auto origin = selected_origins_.find(storage);
          if (origin != selected_origins_.end()) owner_paths += "; " + origin->second;
        }
        ICHECK(used <= group.capacity)
            << "Z3PipelinePlanning recursive: final merged versions use "
            << CheckedRecursiveInt64(used, "final memory-group usage")
            << " bytes above capacity " << group.capacity
            << "; owner paths: " << owner_paths;
      }
    }
    scheduled_->metadata.selected_versions = BufferVersionMap();
    for (const auto &[storage, count] : known_versions_) {
      for (const Var &member : AliasMembers(storage)) {
        if (bare_one_requests_.count(member)) continue;
        ICHECK(count > 0 && count <= std::numeric_limits<int>::max());
        scheduled_->metadata.selected_versions.Set(member, static_cast<int>(count));
      }
    }
  }

  void RepairAndRefresh() {
    cuda::z3::RepairParentsAndIndices(scheduled_->tree, nullptr);
    cuda::z3::RefreshCudaTaskAccesses(scheduled_);
  }

  ScheduledTIR *scheduled_{nullptr};
  MultiBufferOwnerMap owners_;
  VarMap alias_representative_;
  std::unordered_map<Var, std::vector<Var>, ObjectPtrHash, ObjectPtrEqual> alias_members_;
  std::unordered_map<Var, int, ObjectPtrHash, ObjectPtrEqual> storage_ids_;
  std::vector<Var> storage_vars_;
  std::vector<Buffer> storage_buffers_;
  std::vector<int64_t> storage_sizes_;
  std::unordered_map<IRStructure *, int64_t> resource_flags_;
  std::unordered_map<Var, int64_t, ObjectPtrHash, ObjectPtrEqual> known_versions_;
  std::unordered_map<Var, int64_t, ObjectPtrHash, ObjectPtrEqual> visible_versions_;
  std::unordered_map<Var, int64_t, ObjectPtrHash, ObjectPtrEqual> required_versions_;
  std::unordered_map<Var, std::string, ObjectPtrHash, ObjectPtrEqual> selected_origins_;
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> bare_one_requests_;
  std::vector<RecursiveListRecord> records_;
  int64_t next_source_id_{0};
};

class Z3SelectedVersionProjector : public StmtExprMutator {
public:
  static PrimFunc Rewrite(PrimFunc func) {
    Z3SelectedVersionProjector projector;
    projector.Collect(func);
    if (projector.loops_.empty()) return func;
    func.CopyOnWrite()->body = projector.VisitStmt(func->body);
    if (projector.selected_roots_.empty()) {
      Map<String, Any> attrs = func->attrs.defined() ? func->attrs->dict
                                                     : Map<String, Any>{};
      attrs.Set(kZ3BufferVersionsSelected, projector.selected_);
      func.CopyOnWrite()->attrs = DictAttrs(std::move(attrs));
    }
    return func;
  }

private:
  struct LoopInfo {
    const ForNode *key{nullptr};
    Z3BufferVersionCountMap selected;
    std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> touched;
  };

  void Collect(const PrimFunc &func) {
    std::vector<const SBlockNode *> blocks;
    PostOrderVisit(func->body, [&](const ObjectRef &node) {
      if (const auto *block = node.as<SBlockNode>()) {
        blocks.push_back(block);
        if (block->annotations.count(kZ3AuditVersionRequest)) {
          selected_roots_.insert(block);
        }
        return;
      }
      const auto *loop = node.as<ForNode>();
      if (loop == nullptr || !loop->annotations.count(kZ3BufferVersionsSelected)) {
        return;
      }
      LoopInfo info;
      info.key = loop;
      info.selected = ReadZ3BufferVersionCounts(
          loop->annotations, kZ3BufferVersionsSelected, "Z3PipelinePlanning");
      auto regions = Z3BufferRegionCollector::Collect(loop->body, Map<Var, Buffer>{});
      for (const BufferRegion &region : regions.first) {
        info.touched.insert(region->buffer->data);
      }
      for (const BufferRegion &region : regions.second) {
        info.touched.insert(region->buffer->data);
      }
      for (const auto &[storage, count] : info.selected) {
        if (auto previous = selected_.Get(storage)) {
          ICHECK_EQ(previous.value()->value, count->value)
              << "Z3PipelinePlanning: cross-owner selected counts differ for "
              << storage << "; grouped maximum merging is required before "
                 "physical projection";
        } else {
          selected_.Set(storage, count);
        }
      }
      loops_.push_back(std::move(info));
    });
    if (selected_roots_.empty() && !blocks.empty()) {
      selected_roots_.insert(blocks.back());
    }
  }

  Stmt VisitStmt_(const ForNode *loop) final {
    For rewritten = Downcast<For>(StmtExprMutator::VisitStmt_(loop));
    auto found = std::find_if(loops_.begin(), loops_.end(),
                              [&](const LoopInfo &info) {
                                return info.key == loop;
                              });
    if (found == loops_.end()) return rewritten;
    Z3BufferVersionCountMap physical = ReadZ3BufferVersionCounts(
        rewritten->annotations, kPipelineBufferVersions, "Z3PipelinePlanning");
    for (const auto &[storage, count] : selected_) {
      if (found->selected.count(storage) || found->touched.count(storage)) {
        physical.Set(storage, count);
      }
    }
    Map<String, Any> annotations = rewritten->annotations;
    annotations.erase(kZ3BufferVersionsSelected);
    annotations.Set(kPipelineBufferVersions, physical);
    rewritten.CopyOnWrite()->annotations = std::move(annotations);
    return rewritten;
  }

  Stmt VisitStmt_(const SBlockNode *block) final {
    SBlock rewritten = Downcast<SBlock>(StmtExprMutator::VisitStmt_(block));
    if (!selected_roots_.count(block)) return rewritten;
    rewritten.CopyOnWrite()->annotations.Set(kZ3BufferVersionsSelected,
                                             selected_);
    return rewritten;
  }

  std::vector<LoopInfo> loops_;
  Z3BufferVersionCountMap selected_;
  std::unordered_set<const SBlockNode *> selected_roots_;
};

void MergeRecursivePublicVersionRequests(ScheduledTIR *scheduled) {
  ICHECK(scheduled != nullptr);
  auto merge_counts = [&](const Map<String, Any> &annotations, const char *key) {
    Z3BufferVersionCountMap counts =
        ReadZ3BufferVersionCounts(annotations, key, "Z3PipelinePlanning recursive");
    for (const auto &[storage, count] : counts) {
      ICHECK(count->value > 0 && count->value <= std::numeric_limits<int>::max());
      int value = static_cast<int>(count->value);
      if (auto previous = scheduled->metadata.requested_versions.Get(storage)) {
        ICHECK_EQ(previous.value(), value)
            << "Z3PipelinePlanning recursive: conflicting public version counts "
               "for aliased storage "
            << storage;
      } else {
        scheduled->metadata.requested_versions.Set(storage, value);
      }
    }
  };
  auto merge_modes = [&](const Map<String, Any> &annotations, const char *key) {
    Z3BufferVersionModeMap modes =
        ReadZ3BufferVersionModes(annotations, key, "Z3PipelinePlanning recursive");
    for (const auto &[storage, mode] : modes) {
      if (auto previous = scheduled->metadata.version_modes.Get(storage)) {
        ICHECK_EQ(previous.value(), mode)
            << "Z3PipelinePlanning recursive: conflicting clock modes for "
            << storage;
      } else {
        scheduled->metadata.version_modes.Set(storage, mode);
      }
    }
  };
  auto visit_annotations = [&](const Map<String, Any> &annotations) {
    merge_counts(annotations, kZ3BufferVersionRequests);
    merge_counts(annotations, kZ3AuditVersionRequest);
    merge_modes(annotations, kZ3BufferVersionModeRequests);
    merge_modes(annotations, kZ3AuditVersionModeRequest);
  };
  visit_annotations(scheduled->metadata.kernel_root->annotations);
  std::function<void(IRStructure *)> visit = [&](IRStructure *node) {
    if (node->IsTask()) return;
    auto *control = static_cast<ControlNode *>(node);
    visit_annotations(control->control->annotations);
    for (const auto &child : control->children) visit(child.get());
  };
  for (const auto &node : scheduled->tree) visit(node.get());
}

struct RecursiveKernelContext {
  SBlock root;
  ConstrSet outer_ctx;
};

class RecursiveKernelContextCollector final : public ConstrVisitor {
 public:
  static std::vector<RecursiveKernelContext> Collect(const Stmt &body) {
    RecursiveKernelContextCollector collector;
    collector(body);
    return std::move(collector.contexts_);
  }

 private:
  void VisitStmt_(const SBlockNode *op) final {
    if (op->name_hint != "tilelang_root") {
      ConstrVisitor::VisitStmt_(op);
      return;
    }
    contexts_.push_back({GetRef<SBlock>(op), GetConstrSet()});
  }

  std::vector<RecursiveKernelContext> contexts_;
};

Map<Var, IntImm> RecursivePublicVersionMap(
    const std::unordered_map<Var, int64_t, ObjectPtrHash, ObjectPtrEqual> &versions,
    const std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> &bare_ones,
    bool include_bare_ones) {
  Map<Var, IntImm> result;
  for (const auto &[storage, count] : versions) {
    if (!include_bare_ones && bare_ones.count(storage)) continue;
    result.Set(storage, IntImm(DataType::Int(64), count));
  }
  return result;
}

class RecursiveKernelPlannerRewriter final : public StmtMutator {
 public:
  explicit RecursiveKernelPlannerRewriter(std::vector<RecursiveKernelContext> contexts)
      : contexts_(std::move(contexts)) {}

  Stmt Rewrite(const Stmt &body) { return VisitStmt(body); }

 private:
  Stmt VisitStmt_(const SBlockNode *op) final {
    if (op->name_hint != "tilelang_root") return StmtMutator::VisitStmt_(op);
    ICHECK_LT(next_root_, contexts_.size());
    const RecursiveKernelContext &context = contexts_[next_root_++];
    ICHECK(context.root.same_as(GetRef<SBlock>(op)));
    ScheduledTIR scheduled =
        cuda::z3::DecodeScheduledTIR(context.root, context.outer_ctx,
                                     MakeRecursiveTimingExtra);
    MergeRecursivePublicVersionRequests(&scheduled);
    RecursiveZ3Planner planner(&scheduled);
    planner.Run();
    SBlock rewritten = cuda::z3::EncodeScheduledTIR(std::move(scheduled));
    Map<String, Any> annotations = rewritten->annotations;
    annotations.Set(kZ3BufferVersionsSelected,
                    RecursivePublicVersionMap(planner.final_versions(),
                                              planner.bare_one_requests(), false));
    annotations.Set(kZ3AuditVersionSelected,
                    RecursivePublicVersionMap(planner.final_versions(),
                                              planner.bare_one_requests(), false));
    rewritten.CopyOnWrite()->annotations = std::move(annotations);
    return rewritten;
  }

  std::vector<RecursiveKernelContext> contexts_;
  size_t next_root_{0};
};

Stmt PlanRecursiveKernels(const PrimFunc &func) {
  auto contexts = RecursiveKernelContextCollector::Collect(func->body);
  ICHECK(!contexts.empty())
      << "Z3PipelinePlanning recursive: schedule units require a tilelang_root SBlock";
  return RecursiveKernelPlannerRewriter(std::move(contexts)).Rewrite(func->body);
}

bool ProjectionSolverEligible(const IRStructure *node) {
  if (node->IsControl()) return true;
  const auto *task = static_cast<const TaskNode *>(node);
  return task->solver_eligible && task->task_kind != TaskKind::kDeclaration;
}

std::string ProjectionNodePath(const IRStructure *node) {
  std::vector<int64_t> ids;
  for (const IRStructure *current = node; current != nullptr;
       current = current->GetParent()) {
    ids.push_back(current->GetSourceId());
  }
  std::ostringstream os;
  os << "root";
  for (auto it = ids.rbegin(); it != ids.rend(); ++it) os << "/" << *it;
  return os.str();
}

class RecursiveScheduleUnitLowerer {
 public:
  SBlock Lower(const RecursiveKernelContext &context) {
    ScheduledTIR scheduled = cuda::z3::DecodeScheduledTIR(context.root, context.outer_ctx);
    cuda::z3::RefreshCudaTaskAccesses(&scheduled);
    cuda::z3::MultiBufferPlan plan;
    if (!scheduled.metadata.selected_versions.empty()) {
      std::vector<TaskNode *> tasks;
      cuda::z3::CollectAllTaskNodes(scheduled.tree, &tasks, true);
      cuda::z3::RoleAvailability availability(tasks);
      plan = cuda::z3::PrepareBufferEpochs(&scheduled, availability);
      cuda::z3::ValidateMultiBufferPlan(scheduled, plan);
    }
    Map<Var, IntImm> physical_versions;
    for (const auto &[storage, count] : scheduled.metadata.requested_versions) {
      physical_versions.Set(storage, IntImm(DataType::Int(64), count));
    }
    for (const auto &[storage, count] : scheduled.metadata.selected_versions) {
      physical_versions.Set(storage, IntImm(DataType::Int(64), count));
    }
    physical_versions_ = std::move(physical_versions);
    Stmt body = MakeLoweredBody(LowerList(scheduled.tree));
    SBlock rewritten = scheduled.metadata.kernel_root;
    rewritten.CopyOnWrite()->body = std::move(body);
    PublishPreparedEpochs(plan, &rewritten);
    Map<String, Any> annotations = rewritten->annotations;
    annotations.erase(cuda::z3::kRequestedBufferVersions);
    annotations.erase(cuda::z3::kSelectedBufferVersions);
    annotations.erase(cuda::z3::kBufferVersionModes);
    annotations.Set(kZ3RecursivePrepared, Bool(true));
    rewritten.CopyOnWrite()->annotations = std::move(annotations);
    return rewritten;
  }

 private:
  Stmt MakeLoweredBody(const std::vector<Stmt> &statements) const {
    if (statements.empty()) return Evaluate(0);
    if (statements.size() == 1) return statements.front();
    return SeqStmt::Flatten(Array<Stmt>(statements));
  }

  Stmt RestoreTask(const TaskNode *task) const {
    const auto *marker = task->stmt.as<AttrStmtNode>();
    ICHECK(marker != nullptr);
    Stmt body = marker->body;
    if (!task->ws_op_id.empty()) {
      body = AttrStmt(Integer(0), cuda::z3::kWSOpId, StringImm(task->ws_op_id),
                      std::move(body), task->stmt->span);
    }
    return cuda::z3::WrapScheduledGuards(std::move(body), task->GetGuards());
  }

  std::vector<Stmt> LowerList(const std::vector<std::shared_ptr<IRStructure>> &nodes) {
    std::vector<Stmt> result;
    for (const auto &node : nodes) {
      if (node->IsTask()) {
        const auto *task = static_cast<const TaskNode *>(node.get());
        const auto *marker = task->stmt.as<AttrStmtNode>();
        const auto *evaluate = marker->body.as<EvaluateNode>();
        if (evaluate != nullptr && is_zero(evaluate->value) && !task->HasGuards() &&
            task->ws_op_id.empty()) {
          continue;
        }
        result.push_back(RestoreTask(task));
        continue;
      }
      auto *control = static_cast<ControlNode *>(node.get());
      std::vector<Stmt> children = LowerList(control->children);
      Map<String, Any> annotations;
      for (const auto &[key, value] : control->control->annotations) {
        if (!IsZ3InputAnnotation(key)) annotations.Set(key, value);
      }
      annotations.erase(s_tir::attr::software_pipeline_async_stages);
      Array<Integer> order;
      Array<Integer> stages;
      int64_t depth = GetInteger(control->control->annotations, "num_stages", 1);
      for (const auto &child : control->children) {
        if (!ProjectionSolverEligible(child.get())) continue;
        order.push_back(Integer(order.size()));
        stages.push_back(Integer(child->GetStage()));
        depth = std::max(depth, static_cast<int64_t>(child->GetStage()) + 1);
      }
      Map<Var, IntImm> loop_versions;
      for (const auto &[storage, count] : physical_versions_) {
        if (control->TouchesStorage(storage)) {
          loop_versions.Set(storage, count);
          depth = std::max(depth, static_cast<int64_t>(count->value));
        }
      }
      if (!order.empty()) {
        // Mark only restored Controls that actually carry a prepared stock
        // pipeline plan. The flat planner never enters this lowering path.
        annotations.Set(kZ3RecursivePrepared, Bool(true));
        // LowerList is already in solver physical order. Identity order avoids
        // applying the recursive permutation a second time in stock injection.
        annotations.Set("tl_pipeline_order", order);
        annotations.Set("tl_pipeline_stage", stages);
        annotations.Set("num_stages", IntImm(DataType::Int(64), depth));
        annotations.Set(kPipelineBufferVersions, loop_versions);
      } else {
        annotations.erase("num_stages");
        annotations.erase(kPipelineBufferVersions);
      }
      For loop = control->control;
      loop.CopyOnWrite()->annotations = std::move(annotations);
      loop.CopyOnWrite()->body = MakeLoweredBody(children);
      result.push_back(loop);
    }
    return result;
  }

  void PublishPreparedEpochs(const cuda::z3::MultiBufferPlan &plan,
                             SBlock *root) {
    Map<Var, PrimExpr> clocks;
    Array<Map<String, Any>> records;
    for (const auto &info : plan.Infos()) {
      PrimExpr clock;
      if (info.UsesCounter()) {
        clock = BufferLoad(info.counter, {IntImm(DataType::Int(64), 0)});
      } else {
        ICHECK_EQ(info.owners.size(), 1U);
        clock = cuda::z3::CalculateIterationCount(info.owners.front().loop);
      }
      // This is the unmodified epoch expression. Stock injection applies the
      // selected storage depth; it must not substitute the immediate loop var.
      clocks.Set(info.storage, clock);
      Map<String, Any> record;
      record.Set("storage", info.storage);
      record.Set("versions", Integer(info.num_versions));
      record.Set("clock", String(info.UsesCounter() ? "counter" : "iteration"));
      record.Set("clock_expr", clock);
      record.Set("counter_group", Integer(info.counter_group_id));
      if (info.counter.defined()) {
        record.Set("counter", info.counter);
      } else {
        record.Set("counter", String(""));
      }
      Array<Map<String, Any>> owners;
      for (size_t i = 0; i < info.owners.size(); ++i) {
        const auto &owner = info.owners[i];
        Map<String, Any> entry;
        entry.Set("source_id", Integer(owner.loop->GetSourceId()));
        entry.Set("path", String(ProjectionNodePath(owner.loop)));
        entry.Set("ws_op_id", owner.loop->task->ws_op_id);
        entry.Set("guard", owner.active_guard);
        entry.Set("stage", Integer(info.owner_stages[i]));
        owners.push_back(std::move(entry));
      }
      record.Set("owners", owners);
      records.push_back(std::move(record));
    }
    Map<String, Any> annotations = root->get()->annotations;
    annotations.Set(kZ3PreparedClockMap, clocks);
    annotations.Set(kZ3PreparedEpochPlan, records);
    root->CopyOnWrite()->annotations = std::move(annotations);
  }

  Map<Var, IntImm> physical_versions_;
};

class RecursiveKernelLoweringRewriter final : public StmtMutator {
 public:
  explicit RecursiveKernelLoweringRewriter(std::vector<RecursiveKernelContext> contexts)
      : contexts_(std::move(contexts)) {}

  Stmt Rewrite(const Stmt &body) { return VisitStmt(body); }

 private:
  Stmt VisitStmt_(const SBlockNode *op) final {
    if (op->name_hint != "tilelang_root") return StmtMutator::VisitStmt_(op);
    ICHECK_LT(next_root_, contexts_.size());
    const RecursiveKernelContext &context = contexts_[next_root_++];
    ICHECK(context.root.same_as(GetRef<SBlock>(op)));
    return RecursiveScheduleUnitLowerer().Lower(context);
  }

  std::vector<RecursiveKernelContext> contexts_;
  size_t next_root_{0};
};

bool HasRecursiveScheduleUnits(const PrimFunc &func) {
  bool found = false;
  PostOrderVisit(func->body, [&](const ObjectRef &node) {
    if (const auto *attr = node.as<AttrStmtNode>()) {
      found = found || attr->attr_key == cuda::z3::kScheduleUnit;
    }
  });
  return found;
}

void RejectMixedScheduleUnitSchemas(const PrimFunc &func) {
  PostOrderVisit(func->body, [&](const ObjectRef &node) {
    if (const auto *attr = node.as<AttrStmtNode>()) {
      ICHECK(attr->attr_key != kZ3CudaTaskAttr &&
             attr->attr_key != kZ3CudaStageAttr)
          << "Z3PipelinePlanning: flat and recursive schedule-unit schemas "
             "cannot be mixed in one module";
    }
  });
}

bool HasZ3PipelineAnnotation(const PrimFunc &func) {
  bool found = false;
  PostOrderVisit(func->body, [&](const ObjectRef &node) {
    if (const auto *loop = node.as<ForNode>()) {
      found = found || loop->annotations.count(kZ3Pipeline);
    }
  });
  return found;
}

bool HasExistingWarpSpecialization(const PrimFunc &func) {
  bool found = false;
  PostOrderVisit(func->body, [&](const ObjectRef &node) {
    if (const auto *block = node.as<SBlockNode>()) {
      found = found || block->annotations.count("tl.ws_schedule");
    } else if (const auto *attr = node.as<AttrStmtNode>()) {
      found = found || attr->attr_key == "warp_specialize" ||
              attr->attr_key == "kWarpSpecializationScope";
    }
  });
  return found;
}

bool HasValidatedOwnerRing(const PrimFunc &func) {
  bool found = false;
  PostOrderVisit(func->body, [&](const ObjectRef &node) {
    if (const auto *block = node.as<SBlockNode>()) {
      found = found || block->annotations.count("tl_z3_owner_ring_validated");
    }
  });
  return found;
}

} // namespace

namespace cuda::z3 {

// Shared by z3_owner_ring_planning.cc. This is deliberately not a second
// owner solver: it builds the same recursive list problem and runs the same
// solver invocation and augmented validator as scheduled IR lists.
ffi::Tuple<ffi::Array<int64_t>, ffi::Array<int64_t>, int64_t>
SolveRecursiveOwnerRingList(
    int64_t num_stages, ffi::Array<int64_t> latencies, ffi::Array<int64_t> iis,
    ffi::Array<int64_t> resource_flags,
    ffi::Array<ffi::Array<int64_t>> data_deps,
    ffi::Array<ffi::Array<int64_t>> resource_deps,
    ffi::Array<ffi::Array<int64_t>> owner_exclusion_deps,
    ffi::Array<ffi::Array<int64_t>> pipe_order_deps,
    ffi::Array<int64_t> buffer_sizes,
    ffi::Array<ffi::Array<int64_t>> memory_groups,
    ffi::Array<ffi::Array<int64_t>> stage_order_deps, bool enable_offset,
    ffi::Array<int64_t> manual_stages, bool manual_schedule,
    int64_t resource_bits, int64_t seed, bool recalculate_buffer_versions,
    bool verbose, ffi::Optional<ffi::String> debug_dir) {
  RecursiveListProblem problem;
  problem.require_stage_zero = true;
  problem.num_stages = num_stages;
  problem.enable_offset = enable_offset;
  problem.resource_bits = resource_bits;
  problem.seed = seed;
  problem.recalculate_buffer_versions = recalculate_buffer_versions;
  problem.verbose = verbose;
  problem.debug_dir = std::move(debug_dir);
  problem.manual_schedule = manual_schedule;
  problem.audit_path = "legacy owner-ring adapter";
  auto to_vector = [](const ffi::Array<int64_t> &values) {
    return std::vector<int64_t>(values.begin(), values.end());
  };
  problem.latencies = to_vector(latencies);
  problem.iis = to_vector(iis);
  problem.resource_flags = to_vector(resource_flags);
  problem.manual_stages = to_vector(manual_stages);
  problem.buffer_sizes = to_vector(buffer_sizes);
  for (const auto &dep : data_deps) {
    ICHECK_EQ(dep.size(), 4U);
    problem.data_deps.push_back({dep[0], dep[1], dep[2], dep[3]});
  }
  for (const auto &dep : resource_deps) {
    ICHECK_EQ(dep.size(), 2U);
    problem.resource_deps.emplace_back(dep[0], dep[1]);
  }
  for (const auto &dep : owner_exclusion_deps) {
    ICHECK_EQ(dep.size(), 4U);
    problem.owner_exclusion_deps.push_back({dep[0], dep[1], dep[2], dep[3]});
  }
  for (const auto &dep : pipe_order_deps) {
    ICHECK_EQ(dep.size(), 2U);
    problem.pipe_order_deps.emplace_back(dep[0], dep[1]);
  }
  for (const auto &dep : stage_order_deps) {
    ICHECK_EQ(dep.size(), 2U);
    problem.stage_order_deps.emplace_back(dep[0], dep[1]);
  }
  for (const auto &group : memory_groups) {
    problem.memory_groups.push_back(to_vector(group));
  }
  RecursiveListResult result = SolveRecursiveList(problem);
  for (int64_t versions : result.versions) {
    ICHECK(versions >= 1 && versions <= num_stages)
        << "Z3OwnerRingPlanning: selected version exceeds runtime ring depth";
  }
  return ffi::Tuple<ffi::Array<int64_t>, ffi::Array<int64_t>, int64_t>(
      result.starts, result.versions, result.ii);
}

}  // namespace cuda::z3

tvm::transform::Pass LowerZ3ScheduleUnits() {
  using namespace tirx::transform;
  auto pass_func = [](PrimFunc func, const IRModule &, PassContext) {
    if (!HasRecursiveScheduleUnits(func)) return func;
    auto contexts = RecursiveKernelContextCollector::Collect(func->body);
    ICHECK(!contexts.empty())
        << "LowerZ3ScheduleUnits: schedule units require a tilelang_root SBlock";
    func.CopyOnWrite()->body =
        RecursiveKernelLoweringRewriter(std::move(contexts)).Rewrite(func->body);
    return func;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.cuda.LowerZ3ScheduleUnits", {});
}

tvm::transform::Pass Z3PipelinePlanning() {
  using namespace tirx::transform;
  auto pass_func = [](PrimFunc func, const IRModule &, PassContext ctx) {
    if (!ctx->GetConfig<Bool>(kZ3PipelineConfig, Bool(false)).value()) {
      return func;
    }
    bool recursive = HasRecursiveScheduleUnits(func);
    if (!recursive && !HasZ3PipelineAnnotation(func)) {
      return func;
    }
    if (recursive) RejectMixedScheduleUnitSchemas(func);
    ICHECK(!HasExistingWarpSpecialization(func) || HasValidatedOwnerRing(func))
        << "Z3PipelinePlanning: refusing to overwrite an existing "
           "warp-specialization schedule";
    auto auto_ws =
        ctx->GetConfig<Optional<String>>(kAutoWarpSpecializationConfig);
    ICHECK(!auto_ws.has_value())
        << "Z3PipelinePlanning: automatic warp specialization and CUDA Z3 "
           "pipeline planning cannot schedule the same annotated module";
    if (recursive) {
      func.CopyOnWrite()->body = PlanRecursiveKernels(func);
      return func;
    }
    func.CopyOnWrite()->body = Z3PipelinePlanner::Substitute(func);
    return Z3SelectedVersionProjector::Rewrite(std::move(func));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.Z3PipelinePlanning", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = reflection;
  TVM_REGISTER_PASS_CONFIG_OPTION(kZ3PipelineConfig, Bool);
  refl::GlobalDef().def("tl.transform.Z3PipelinePlanning", Z3PipelinePlanning);
  refl::GlobalDef().def("tl.cuda.transform.LowerZ3ScheduleUnits",
                        LowerZ3ScheduleUnits);
}

} // namespace tl
} // namespace tvm
