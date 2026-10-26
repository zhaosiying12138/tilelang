/*!
 * \file z3_pipeline_planning.cc
 * \brief Opt-in adapter from explicit CUDA loop problems to pipeline IR.
 */

#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <string>
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
constexpr const char *kZ3Seed = "tl_z3_pipeline_seed";
constexpr const char *kZ3RecalculateBufferVersions =
    "tl_z3_pipeline_recalculate_buffer_versions";
constexpr const char *kZ3Verbose = "tl_z3_pipeline_verbose";
constexpr const char *kZ3DebugDir = "tl_z3_pipeline_debug_dir";
constexpr const char *kZ3AuditII = "tl_z3_pipeline_audit_ii";
constexpr const char *kZ3AuditStarts = "tl_z3_pipeline_audit_starts";

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
      kZ3Seed,
      kZ3RecalculateBufferVersions,
      kZ3Verbose,
      kZ3DebugDir,
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
  std::vector<Array<BufferRegion>> reads;
  std::vector<Array<BufferRegion>> writes;
  std::vector<char> replayable_binds;
};

Z3ScheduledStmtAnalysis AnalyzeScheduledStmts(
    const Array<Stmt> &flat_stmts,
    const Map<Var, Buffer> &buffer_data_to_buffer) {
  BufferSet pipeline_write_buffers;
  for (const Stmt &stmt : flat_stmts) {
    if (IsPipelineDeclarationStmt(stmt)) {
      continue;
    }
    auto access = Z3BufferRegionCollector::Collect(stmt, buffer_data_to_buffer);
    for (const BufferRegion &write : access.second) {
      pipeline_write_buffers.insert(write->buffer);
    }
  }

  Z3ScheduledStmtAnalysis analysis;
  analysis.replayable_binds.assign(flat_stmts.size(), 0);
  for (size_t i = 0; i < flat_stmts.size(); ++i) {
    const Stmt &stmt = flat_stmts[i];
    if (IsPipelineDeclarationStmt(stmt)) {
      continue;
    }
    auto access = Z3BufferRegionCollector::Collect(stmt, buffer_data_to_buffer);
    if (IsReplayableScalarBind(stmt, access.first, pipeline_write_buffers)) {
      analysis.replayable_binds[i] = 1;
      continue;
    }
    analysis.scheduled_indices.push_back(i);
    analysis.scheduled_stmts.push_back(stmt);
    analysis.reads.push_back(std::move(access.first));
    analysis.writes.push_back(std::move(access.second));
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

  Stmt PlanLoop(const ForNode *loop) {
    const Map<String, Any> &annotations = loop->annotations;
    ICHECK(GetBool(annotations, kZ3Pipeline, false))
        << "Z3PipelinePlanning: tl_z3_pipeline must be True when present";
    bool manual_schedule = GetBool(annotations, kZ3ManualSchedule, false);
    bool has_handwritten_schedule =
        annotations.count("tl_pipeline_order") ||
        annotations.count("tl_pipeline_stage") ||
        annotations.count("software_pipeline_order") ||
        annotations.count("software_pipeline_stage");
    ICHECK(!has_handwritten_schedule)
        << "Z3PipelinePlanning: "
        << (manual_schedule ? "manual Z3 scheduling" : "Z3 scheduling")
        << " cannot overwrite existing handwritten pipeline order/stage "
           "annotations";
    ICHECK(!annotations.count(kPipelineBufferVersions))
        << "Z3PipelinePlanning: tl_pipeline_buffer_versions already exists; "
           "refusing to overwrite a buffer-version contract";

    Array<Integer> latencies = GetIntegerArray(annotations, kZ3Latencies);
    Array<Integer> iis = GetIntegerArray(annotations, kZ3IIs);
    Array<Integer> resource_flags =
        GetIntegerArray(annotations, kZ3ResourceFlags);
    size_t n = latencies.size();
    ICHECK(n > 0) << "Z3PipelinePlanning: at least one task is required";
    ICHECK_EQ(iis.size(), n)
        << "Z3PipelinePlanning: latencies and iis must have equal lengths";
    ICHECK_EQ(resource_flags.size(), n)
        << "Z3PipelinePlanning: latencies and resource_flags must have equal "
           "lengths";
    ICHECK(n <= static_cast<size_t>(std::numeric_limits<int>::max()))
        << "Z3PipelinePlanning: too many tasks for pipeline order annotations";

    Array<Array<Integer>> owner_exclusion_deps =
        GetNestedIntegerArrays(annotations, kZ3OwnerExclusionDeps);
    ICHECK(owner_exclusion_deps.empty())
        << "Z3PipelinePlanning: owner_exclusion_deps are not expressible by "
           "the current CUDA pipeline IR; refusing to discard owner lifetime "
           "and wraparound semantics";
    Array<Array<Integer>> stage_order_deps =
        GetNestedIntegerArrays(annotations, kZ3StageOrderDeps);
    ICHECK(stage_order_deps.empty())
        << "Z3PipelinePlanning: stage_order_deps are conservatively unsupported "
           "because downstream Let cloning and its stage-order semantics are "
           "not implemented";

    int64_t num_stages = GetInteger(annotations, "num_stages", 0);
    ICHECK_GT(num_stages, 0)
        << "Z3PipelinePlanning: T.Pipelined num_stages must be positive";
    bool enable_offset = GetBool(annotations, kZ3EnableOffset, false);
    Array<Integer> manual_stages =
        GetIntegerArray(annotations, kZ3ManualStages);
    if (manual_schedule) {
      ICHECK_EQ(manual_stages.size(), n)
          << "Z3PipelinePlanning: manual_schedule requires one stage per task";
    } else {
      ICHECK(manual_stages.empty())
          << "Z3PipelinePlanning: manual_stages requires manual_schedule=True";
    }

    Array<Integer> buffer_sizes =
        GetIntegerArray(annotations, kZ3BufferSizes);
    Array<Var> buffer_vars;
    if (auto value = annotations.Get(kZ3BufferVars); value.has_value()) {
      buffer_vars = Downcast<Array<Var>>(value.value());
    }
    ICHECK_EQ(buffer_vars.size(), buffer_sizes.size())
        << "Z3PipelinePlanning: buffer_vars and buffer_sizes must have equal "
           "lengths";
    std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> seen_buffer_vars;
    for (const Var &var : buffer_vars) {
      ICHECK(seen_buffer_vars.insert(var).second)
          << "Z3PipelinePlanning: duplicate Buffer.data Var in buffer_vars";
    }

    Array<Stmt> body_stmts = NormalizePipelineBody(loop->body);
    Array<Stmt> flat_stmts = Z3SeqStmtFlattener::Flatten(body_stmts);
    Z3ScheduledStmtAnalysis analysis =
        AnalyzeScheduledStmts(flat_stmts, buffer_data_to_buffer_);
    ICHECK_EQ(analysis.scheduled_stmts.size(), n)
        << "Z3PipelinePlanning: task count " << n
        << " does not match body scheduled statement count "
        << analysis.scheduled_stmts.size()
        << "; tasks exclude buffer declarations and replayable scalar Bind "
           "statements after SeqStmt flattening";

    auto scheduler =
        ffi::Function::GetGlobal("tl.cuda.transform.z3_schedule_loop_python");
    ICHECK(scheduler.has_value())
        << "Z3PipelinePlanning: tl.cuda.transform.z3_schedule_loop_python is "
           "not registered";

    Array<Array<Integer>> data_deps =
        GetNestedIntegerArrays(annotations, kZ3DataDeps);
    Array<Array<Integer>> resource_deps =
        GetNestedIntegerArrays(annotations, kZ3ResourceDeps);
    Array<Array<Integer>> pipe_order_deps =
        GetNestedIntegerArrays(annotations, kZ3PipeOrderDeps);
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

    using SolverResult =
        ffi::Tuple<ffi::Array<int64_t>, ffi::Array<int64_t>, int64_t>;
    SolverResult result =
        scheduler.value()(num_stages, ToInt64(latencies), ToInt64(iis),
                          ToInt64(resource_flags), ToInt64(data_deps),
                          ToInt64(resource_deps), ToInt64(owner_exclusion_deps),
                          ToInt64(pipe_order_deps), ToInt64(buffer_sizes),
                          ToInt64(memory_groups), ToInt64(stage_order_deps),
                          enable_offset, ToInt64(manual_stages),
                          manual_schedule, resource_bits, seed,
                          recalculate_buffer_versions, verbose, debug_dir)
            .cast<SolverResult>();
    Array<int64_t> starts = result.get<0>();
    Array<int64_t> versions = result.get<1>();
    int64_t ii = result.get<2>();
    ICHECK_EQ(starts.size(), n)
        << "Z3PipelinePlanning: solver returned " << starts.size()
        << " start times for " << n << " tasks";
    ICHECK_EQ(versions.size(), buffer_vars.size())
        << "Z3PipelinePlanning: solver returned " << versions.size()
        << " buffer versions for " << buffer_vars.size() << " buffers";
    ICHECK_GT(ii, 0) << "Z3PipelinePlanning: solver returned non-positive II";
    for (size_t i = 0; i < versions.size(); ++i) {
      ICHECK_GT(versions[i], 0)
          << "Z3PipelinePlanning: solver returned a non-positive version "
             "count for buffer "
          << i;
    }

    std::vector<int64_t> raw_stages(n, 0);
    int64_t min_stage = std::numeric_limits<int64_t>::max();
    for (size_t i = 0; i < n; ++i) {
      ICHECK_GE(starts[i], 0)
          << "Z3PipelinePlanning: solver returned a negative start time";
      raw_stages[i] = starts[i] / ii;
      min_stage = std::min(min_stage, raw_stages[i]);
    }

    std::vector<int> stages(n, 0);
    std::vector<int64_t> residues(n, 0);
    for (size_t i = 0; i < n; ++i) {
      int64_t normalized_stage = raw_stages[i] - min_stage;
      ICHECK(normalized_stage <= std::numeric_limits<int>::max())
          << "Z3PipelinePlanning: normalized stage does not fit pipeline IR";
      stages[i] = static_cast<int>(normalized_stage);
      residues[i] = starts[i] - raw_stages[i] * ii;
    }

    std::vector<size_t> physical_order(n);
    std::iota(physical_order.begin(), physical_order.end(), 0);
    std::sort(physical_order.begin(), physical_order.end(),
              [&](size_t left, size_t right) {
                if (residues[left] != residues[right]) {
                  return residues[left] < residues[right];
                }
                if (stages[left] != stages[right]) {
                  return stages[left] > stages[right];
                }
                return left < right;
              });
    std::vector<int> inverse_order(n, 0);
    for (size_t position = 0; position < n; ++position) {
      inverse_order[physical_order[position]] = static_cast<int>(position);
    }

    bool identity_order = true;
    for (size_t i = 0; i < n; ++i) {
      identity_order = identity_order && physical_order[i] == i;
    }
    bool flat_schedule =
        std::all_of(stages.begin(), stages.end(), [](int stage) {
          return stage == 0;
        });
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
    for (int64_t version : versions) {
      cuda_depth = std::max(cuda_depth, version);
    }

    Map<Var, IntImm> buffer_version_map;
    for (size_t i = 0; i < buffer_vars.size(); ++i) {
      buffer_version_map.Set(
          buffer_vars[i], IntImm(DataType::Int(64), versions[i]));
    }

    Map<String, Any> output_annotations;
    for (const auto &[key, value] : annotations) {
      if (!IsZ3InputAnnotation(key)) {
        output_annotations.Set(key, value);
      }
    }
    Array<Integer> order_annotation;
    Array<Integer> stage_annotation;
    Array<Integer> start_audit;
    order_annotation.reserve(n);
    stage_annotation.reserve(n);
    start_audit.reserve(n);
    for (size_t i = 0; i < n; ++i) {
      order_annotation.push_back(Integer(inverse_order[i]));
      stage_annotation.push_back(Integer(stages[i]));
      start_audit.push_back(
          Integer(IntImm(DataType::Int(64), starts[i])));
    }
    output_annotations.Set("tl_pipeline_order", order_annotation);
    output_annotations.Set("tl_pipeline_stage", stage_annotation);
    output_annotations.Set("num_stages", IntImm(DataType::Int(64), cuda_depth));
    output_annotations.Set(kPipelineBufferVersions, buffer_version_map);
    output_annotations.Set(kZ3AuditII, IntImm(DataType::Int(64), ii));
    output_annotations.Set(kZ3AuditStarts, start_audit);

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
    ICHECK(!HasExistingWarpSpecialization(func))
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
