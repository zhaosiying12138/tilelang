/*!
 * \file z3_pipeline_planning.cc
 * \brief Opt-in adapter from explicit CUDA loop problems to pipeline IR.
 */

#include <algorithm>
#include <array>
#include <cstdint>
#include <deque>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
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
#include "common/bind_utils.h"
#include "common/pipeline_utils.h"
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
  const auto *imm = value.value().as<IntImmNode>();
  ICHECK(imm != nullptr) << "Z3PipelinePlanning: " << key
                         << " must be an integer annotation";
  return imm->value;
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

struct Z3ValidatedSchedule {
  Array<int64_t> starts;
  Array<int64_t> versions;
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
      const Array<Var> &buffer_vars, const Z3AugmentedProblem &base,
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
    AddVirtualCloneBufferDependencies(loop, buffer_vars, &problem);
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

    Array<Integer> buffer_sizes = GetIntegerArray(annotations, kZ3BufferSizes);
    Array<Var> buffer_vars;
    if (auto value = annotations.Get(kZ3BufferVars); value.has_value()) {
      buffer_vars = Downcast<Array<Var>>(value.value());
    }
    ICHECK_EQ(buffer_vars.size(), buffer_sizes.size())
        << "Z3PipelinePlanning: buffer_vars and buffer_sizes must have equal lengths";
    std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> seen_buffer_vars;
    for (const Var &var : buffer_vars) {
      ICHECK(seen_buffer_vars.insert(var).second)
          << "Z3PipelinePlanning: duplicate Buffer.data Var in buffer_vars";
    }
    for (const Integer &size : buffer_sizes) {
      ICHECK_GE(size->value, 0) << "Z3PipelinePlanning: negative buffer size";
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
        ICHECK(-distance - 1 < static_cast<int64_t>(buffer_sizes.size()))
            << "Z3PipelinePlanning: data dependency references an unavailable buffer";
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
    Array<Array<Integer>> memory_groups =
        GetNestedIntegerArrays(annotations, kZ3MemoryGroups);
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
      augmented = BuildAugmentedProblem(loop, analysis, buffer_vars, base,
                                        clones, manual_schedule);
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
    for (size_t i = 0; i < buffer_vars.size(); ++i) {
      buffer_version_map.Set(buffer_vars[i],
                             IntImm(DataType::Int(64), schedule.versions[i]));
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

tvm::transform::Pass Z3PipelinePlanning() {
  using namespace tirx::transform;
  auto pass_func = [](PrimFunc func, const IRModule &, PassContext ctx) {
    if (!ctx->GetConfig<Bool>(kZ3PipelineConfig, Bool(false)).value()) {
      return func;
    }
    if (!HasZ3PipelineAnnotation(func)) {
      return func;
    }
    ICHECK(!HasExistingWarpSpecialization(func) || HasValidatedOwnerRing(func))
        << "Z3PipelinePlanning: refusing to overwrite an existing "
           "warp-specialization schedule";
    auto auto_ws =
        ctx->GetConfig<Optional<String>>(kAutoWarpSpecializationConfig);
    ICHECK(!auto_ws.has_value())
        << "Z3PipelinePlanning: automatic warp specialization and CUDA Z3 "
           "pipeline planning cannot schedule the same annotated module";
    func.CopyOnWrite()->body = Z3PipelinePlanner::Substitute(func);
    return func;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.Z3PipelinePlanning", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = reflection;
  TVM_REGISTER_PASS_CONFIG_OPTION(kZ3PipelineConfig, Bool);
  refl::GlobalDef().def("tl.transform.Z3PipelinePlanning", Z3PipelinePlanning);
}

} // namespace tl
} // namespace tvm
