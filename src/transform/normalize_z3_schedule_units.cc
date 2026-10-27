/*!
 * \file normalize_z3_schedule_units.cc
 * \brief Normalize public CUDA Z3 Task/Stage scheduling boundaries.
 */

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
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
#include "../op/gemm.h"
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

constexpr const char *kZ3Pipeline = "tl_z3_pipeline";
constexpr const char *kZ3ManualStages = "tl_z3_pipeline_manual_stages";
constexpr const char *kZ3ManualSchedule = "tl_z3_pipeline_manual_schedule";
constexpr const char *kZ3EnableOffset = "tl_z3_pipeline_enable_offset";
constexpr const char *kZ3PipeOrderDeps = "tl_z3_pipeline_pipe_order_deps";

bool GetBool(const Map<String, Any> &annotations, const char *key,
             bool default_value = false) {
  auto value = annotations.Get(key);
  return value.has_value() ? value.value().cast<bool>() : default_value;
}

Array<Integer> GetIntegerArray(const Map<String, Any> &annotations,
                               const char *key) {
  auto value = annotations.Get(key);
  return value.has_value() ? Downcast<Array<Integer>>(value.value())
                           : Array<Integer>();
}

Array<Array<Integer>> GetNestedIntegerArrays(
    const Map<String, Any> &annotations, const char *key) {
  auto value = annotations.Get(key);
  return value.has_value() ? Downcast<Array<Array<Integer>>>(value.value())
                           : Array<Array<Integer>>();
}

class ScheduleUnitMarkerDetector : public StmtExprVisitor {
 public:
  static bool Contains(const Stmt &body) {
    ScheduleUnitMarkerDetector detector;
    detector(body);
    return detector.found_;
  }

 private:
  void VisitStmt_(const AttrStmtNode *op) final {
    found_ = found_ || IsZ3ScheduleUnitAttrKey(op->attr_key);
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const CallNode *op) final {
    if (auto tile_op = ParseOperator(GetRef<Call>(op)); tile_op.defined()) {
      if (const auto *parallel = tile_op.as<ParallelOpNode>()) {
        this->VisitStmt(parallel->GetRoot());
      }
      return;
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  bool found_{false};
};

bool ContainsScheduleUnitMarker(const Stmt &body) {
  return ScheduleUnitMarkerDetector::Contains(body);
}

class MarkerScopeValidator : public StmtExprVisitor {
 public:
  static void Validate(const Stmt &body) {
    MarkerScopeValidator validator;
    validator(body);
  }

 private:
  void VisitStmt_(const ForNode *op) final {
    bool z3_loop = op->annotations.count(kZ3Pipeline);
    if (z3_loop) ++z3_depth_;
    this->VisitStmt(op->body);
    if (z3_loop) --z3_depth_;
  }

  void VisitStmt_(const AttrStmtNode *op) final {
    ICHECK(!IsZ3ScheduleUnitAttrKey(op->attr_key) || z3_depth_ > 0)
        << "NormalizeZ3ScheduleUnits: T.Task/T.Stage is only legal inside an "
           "opted-in tl_z3_pipeline loop";
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const CallNode *op) final {
    if (auto tile_op = ParseOperator(GetRef<Call>(op)); tile_op.defined()) {
      if (const auto *parallel = tile_op.as<ParallelOpNode>()) {
        this->VisitStmt(parallel->GetRoot());
      }
      return;
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  int z3_depth_{0};
};

bool HasExecutableStatement(const Stmt &stmt) {
  if (IsPipelineDeclarationStmt(stmt)) return false;
  if (const auto *evaluate = stmt.as<EvaluateNode>()) {
    if (is_zero(evaluate->value)) return false;
  }
  if (const auto *seq = stmt.as<SeqStmtNode>()) {
    return std::any_of(seq->seq.begin(), seq->seq.end(),
                       [](const Stmt &child) {
                         return HasExecutableStatement(child);
                       });
  }
  if (const auto *attr = stmt.as<AttrStmtNode>()) {
    return HasExecutableStatement(attr->body);
  }
  return true;
}

class NestedScheduleUnitValidator : public StmtExprVisitor {
 public:
  static void Validate(const Stmt &stmt, const std::string &context) {
    NestedScheduleUnitValidator validator(context);
    validator(stmt);
  }

 private:
  explicit NestedScheduleUnitValidator(std::string context)
      : context_(std::move(context)) {}

  void VisitStmt_(const AttrStmtNode *op) final {
    ICHECK(!IsZ3ScheduleUnitAttrKey(op->attr_key))
        << "NormalizeZ3ScheduleUnits: nested T.Task/T.Stage is unsupported in "
        << context_
        << "; place T.Stage outside complete direct-child T.Task regions";
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const ForNode *op) final {
    ICHECK(!op->annotations.count(kZ3Pipeline))
        << "NormalizeZ3ScheduleUnits: a schedule-unit marker cannot cross a "
           "tl_z3_pipeline loop boundary in "
        << context_;
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const CallNode *op) final {
    if (auto tile_op = ParseOperator(GetRef<Call>(op)); tile_op.defined()) {
      if (const auto *parallel = tile_op.as<ParallelOpNode>()) {
        this->VisitStmt(parallel->GetRoot());
      }
      return;
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  std::string context_;
};

/* Collect the storage identities needed by dependency-closed Bind analysis. */
struct ScheduleStorageAccess {
  Array<BufferRegion> reads;
  BufferSet writes;
};

class ScheduleStorageCollector : public StmtExprVisitor {
 public:
  static ScheduleStorageAccess Collect(
      const Stmt &stmt, const Map<Var, Buffer> &buffer_data_to_buffer) {
    ScheduleStorageCollector collector(buffer_data_to_buffer);
    collector(stmt);
    ScheduleStorageAccess result;
    for (const Buffer &buffer : collector.reads_) {
      result.reads.push_back(BufferRegion::FullRegion(buffer));
    }
    result.writes = std::move(collector.writes_);
    return result;
  }

 private:
  explicit ScheduleStorageCollector(
      const Map<Var, Buffer> &buffer_data_to_buffer)
      : buffer_data_to_buffer_(buffer_data_to_buffer) {}

  void HandleTileOp(const TileOperator &tile_op) {
    if (const auto *parallel = tile_op.as<ParallelOpNode>()) {
      this->VisitStmt(parallel->GetRoot());
      return;
    }
    AccessRegions access = tile_op->GetAccessRegions();
    for (const BufferRegion &region : access.reads) {
      reads_.insert(region->buffer);
    }
    for (const BufferRegion &region : access.writes) {
      writes_.insert(region->buffer);
    }
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    writes_.insert(op->buffer);
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const BufferLoadNode *op) final {
    reads_.insert(op->buffer);
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const CallNode *op) final {
    if (auto tile_op = ParseOperator(GetRef<Call>(op)); tile_op.defined()) {
      HandleTileOp(tile_op);
      return;
    }
    if (op->op.same_as(builtin::address_of())) {
      ICHECK(!op->args.empty());
      if (const auto *load = op->args[0].as<BufferLoadNode>()) {
        reads_.insert(load->buffer);
      } else if (const auto *var = op->args[0].as<VarNode>()) {
        auto it = buffer_data_to_buffer_.find(GetRef<Var>(var));
        if (it != buffer_data_to_buffer_.end()) reads_.insert((*it).second);
      }
      return;
    }
    if (op->op.same_as(tl::access_ptr())) {
      ICHECK_EQ(op->args.size(), 3U);
      const auto *load = op->args[0].as<BufferLoadNode>();
      ICHECK(load != nullptr);
      int mask = GetConservativeAccessMask(op->args[2]);
      if (mask & kAccessRead) reads_.insert(load->buffer);
      if (mask & kAccessWrite) writes_.insert(load->buffer);
      StmtExprVisitor::VisitExpr_(op);
      return;
    }
    if (op->op.same_as(builtin::tvm_access_ptr())) {
      ICHECK_GE(op->args.size(), 2U);
      if (const auto *var = op->args[1].as<VarNode>()) {
        auto it = buffer_data_to_buffer_.find(GetRef<Var>(var));
        if (it != buffer_data_to_buffer_.end()) {
          int mask = op->args.size() == 5U
                         ? GetConservativeAccessMask(op->args[4])
                         : kAccessReadWrite;
          if (mask & kAccessRead) reads_.insert((*it).second);
          if (mask & kAccessWrite) writes_.insert((*it).second);
        }
      }
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  Map<Var, Buffer> buffer_data_to_buffer_;
  BufferSet reads_;
  BufferSet writes_;
};

enum ExecutionDomainBit : uint32_t {
  kCopyDomain = 1U << 0,
  kTensorCoreDomain = 1U << 1,
  kVectorDomain = 1U << 2,
  kScalarDomain = 1U << 3,
  kAsyncProxyDomain = 1U << 4,
  kUnknownDomain = 1U << 5,
};

bool IsAsyncProxyCall(const CallNode *op) {
  const auto *op_node = op->op.as<OpNode>();
  if (op_node == nullptr) return false;
  const std::string &name = op_node->name;
  return name.find("async") != std::string::npos ||
         name.find("mbarrier") != std::string::npos ||
         name.find("tma_") != std::string::npos ||
         name.find("ptx_wait_group") != std::string::npos ||
         name.find("ptx_commit_group") != std::string::npos;
}

class ExecutionDomainExtractor : public StmtExprVisitor {
 public:
  static String Extract(const Stmt &stmt) {
    ExecutionDomainExtractor extractor;
    extractor(stmt);
    ICHECK((extractor.domains_ & kUnknownDomain) == 0)
        << "NormalizeZ3ScheduleUnits: cannot prove a single execution_domain "
           "for an unknown tile operation or opaque call; split the body or "
           "add a native domain classifier before using T.Task. Body="
        << stmt;
    ICHECK(extractor.domains_ != 0 &&
           (extractor.domains_ & (extractor.domains_ - 1)) == 0)
        << "NormalizeZ3ScheduleUnits: every T.Task/scheduled statement in a "
           "marked loop must use exactly one execution_domain (copy, "
           "tensor_core, vector, scalar, async_proxy); split cross-domain "
           "bodies into separate tasks. Body="
        << stmt;
    switch (extractor.domains_) {
      case kCopyDomain:
        return "copy";
      case kTensorCoreDomain:
        return "tensor_core";
      case kVectorDomain:
        return "vector";
      case kScalarDomain:
        return "scalar";
      case kAsyncProxyDomain:
        return "async_proxy";
      default:
        LOG(FATAL) << "Unreachable execution domain";
    }
    return String();
  }

 private:
  void MarkScalarOrVector() {
    domains_ |= vector_depth_ > 0 ? kVectorDomain : kScalarDomain;
  }

  void VisitStmt_(const ForNode *op) final {
    bool vector = op->kind == ForKind::kParallel ||
                  op->kind == ForKind::kVectorized ||
                  op->kind == ForKind::kThreadBinding;
    if (vector) ++vector_depth_;
    StmtExprVisitor::VisitStmt_(op);
    if (vector) --vector_depth_;
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    MarkScalarOrVector();
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const BindNode *op) final {
    MarkScalarOrVector();
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const EvaluateNode *op) final {
    if (is_zero(op->value)) return;
    const auto *call = op->value.as<CallNode>();
    if (call != nullptr && ParseOperator(GetRef<Call>(call)).defined()) {
      this->VisitExpr(op->value);
      return;
    }
    if (call != nullptr && IsAsyncProxyCall(call)) {
      this->VisitExpr(op->value);
      return;
    }
    MarkScalarOrVector();
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const CallNode *op) final {
    if (auto tile_op = ParseOperator(GetRef<Call>(op)); tile_op.defined()) {
      if (tile_op.as<CopyNode>()) {
        domains_ |= kCopyDomain;
      } else if (tile_op.as<GemmNode>()) {
        domains_ |= kTensorCoreDomain;
      } else if (tile_op.as<ParallelOpNode>()) {
        domains_ |= kVectorDomain;
      } else {
        domains_ |= kUnknownDomain;
      }
      return;
    }
    if (const auto *op_node = op->op.as<OpNode>()) {
      const std::string &name = op_node->name;
      if (IsAsyncProxyCall(op)) {
        domains_ |= kAsyncProxyDomain;
        return;
      }
      bool known_pure_arithmetic =
          SideEffect(GetRef<Call>(op)) == CallEffectKind::kPure &&
          name.find("extern") == std::string::npos &&
          name.find("packed") == std::string::npos &&
          name.find("random") == std::string::npos &&
          name.find("rng") == std::string::npos &&
          name.find("atomic") == std::string::npos;
      if (!known_pure_arithmetic) domains_ |= kUnknownDomain;
    } else {
      domains_ |= kUnknownDomain;
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  uint32_t domains_{0};
  int vector_depth_{0};
};

struct NormalizedEntry {
  Stmt stmt;
  int64_t stage{-1};
  int stage_group{-1};
};

class ScheduleUnitNormalizer : public StmtExprMutator {
 public:
  static Stmt Rewrite(const PrimFunc &func) {
    MarkerScopeValidator::Validate(func->body);
    ScheduleUnitNormalizer normalizer;
    for (const auto &[_, buffer] : func->buffer_map) {
      normalizer.buffer_data_to_buffer_.Set(buffer->data, buffer);
    }
    return normalizer.VisitStmt(func->body);
  }

 private:
  Stmt VisitStmt_(const SBlockNode *op) final {
    for (const Buffer &buffer : op->alloc_buffers) {
      buffer_data_to_buffer_.Set(buffer->data, buffer);
    }
    SBlock result = Downcast<SBlock>(StmtExprMutator::VisitStmt_(op));
    for (const Buffer &buffer : op->alloc_buffers) {
      buffer_data_to_buffer_.erase(buffer->data);
    }
    return result;
  }

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    ICHECK(!IsZ3ScheduleUnitAttrKey(op->attr_key))
        << "NormalizeZ3ScheduleUnits: T.Task/T.Stage is only legal inside an "
           "opted-in tl_z3_pipeline loop";
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const ForNode *loop) final {
    if (!loop->annotations.count(kZ3Pipeline)) {
      return StmtExprMutator::VisitStmt_(loop);
    }
    ICHECK(GetBool(loop->annotations, kZ3Pipeline, false))
        << "NormalizeZ3ScheduleUnits: tl_z3_pipeline must be True when present";
    if (!ContainsScheduleUnitMarker(loop->body)) return GetRef<For>(loop);
    return NormalizeLoop(loop);
  }

  std::vector<NormalizedEntry> NormalizeList(const Stmt &stmt, int64_t stage,
                                             int stage_group) {
    if (const auto *seq = stmt.as<SeqStmtNode>()) {
      std::vector<NormalizedEntry> result;
      for (const Stmt &child : seq->seq) {
        auto nested = NormalizeList(child, stage, stage_group);
        result.insert(result.end(), nested.begin(), nested.end());
      }
      return result;
    }
    if (const auto *attr = stmt.as<AttrStmtNode>()) {
      if (attr->attr_key == kZ3CudaStageAttr) {
        ICHECK_LT(stage, 0) << "NormalizeZ3ScheduleUnits: nested T.Stage "
                               "markers are not supported";
        ICHECK(is_one(attr->value))
            << "NormalizeZ3ScheduleUnits: tl.cuda_z3_stage value must be 1";
        PrimExpr stage_expr = attr->node.cast<PrimExpr>();
        const auto *value = stage_expr.as<IntImmNode>();
        ICHECK(value != nullptr && value->value >= 0)
            << "NormalizeZ3ScheduleUnits: T.Stage requires a non-negative "
               "constant integer, got "
            << stage_expr;
        stage_seen_ = true;
        int group = next_stage_group_++;
        return NormalizeList(attr->body, value->value, group);
      }
      if (attr->attr_key == kZ3CudaTaskAttr) {
        ValidateTask(attr);
        return {{GetRef<Stmt>(attr), stage, stage_group}};
      }
    }
    NestedScheduleUnitValidator::Validate(
        stmt, "a retained loop, guard, tile-op body, or attribute");
    return {{stmt, stage, stage_group}};
  }

  void ValidateTask(const AttrStmtNode *op) {
    ICHECK(is_one(op->value))
        << "NormalizeZ3ScheduleUnits: tl.cuda_z3_task value must be 1";
    Map<String, Any> metadata = op->node.cast<Map<String, Any>>();
    int64_t latency = -1;
    int64_t ii = -1;
    for (const auto &[key, value] : metadata) {
      ICHECK(key == "latency" || key == "ii")
          << "NormalizeZ3ScheduleUnits: unsupported T.Task metadata key " << key;
      const auto *imm = value.as<IntImmNode>();
      ICHECK(imm != nullptr)
          << "NormalizeZ3ScheduleUnits: T.Task " << key
          << " must be a constant integer";
      if (key == "latency") {
        ICHECK_GE(imm->value, 0)
            << "NormalizeZ3ScheduleUnits: T.Task latency must be non-negative";
        latency = imm->value;
      } else {
        ICHECK_GT(imm->value, 0)
            << "NormalizeZ3ScheduleUnits: T.Task ii must be positive";
        ii = imm->value;
      }
    }
    if (latency >= 0 && ii >= 0) {
      ICHECK_GE(latency, ii)
          << "NormalizeZ3ScheduleUnits: T.Task latency must be >= ii";
    }
    ICHECK(HasExecutableStatement(op->body))
        << "NormalizeZ3ScheduleUnits: T.Task must contain an executable "
           "statement, not only declarations or no-ops";
    NestedScheduleUnitValidator::Validate(op->body, "T.Task");
  }

  Stmt NormalizeLoop(const ForNode *loop) {
    stage_seen_ = false;
    next_stage_group_ = 0;
    std::vector<NormalizedEntry> entries = NormalizeList(loop->body, -1, -1);
    ICHECK(!entries.empty())
        << "NormalizeZ3ScheduleUnits: no schedule units were materialized";

    Array<Stmt> flat_stmts;
    flat_stmts.reserve(entries.size());
    std::vector<Array<BufferRegion>> flat_reads;
    flat_reads.reserve(entries.size());
    BufferSet pipeline_writes;
    for (const NormalizedEntry &entry : entries) {
      flat_stmts.push_back(entry.stmt);
      if (IsPipelineDeclarationStmt(entry.stmt)) {
        flat_reads.push_back({});
        continue;
      }
      ScheduleStorageAccess access =
          ScheduleStorageCollector::Collect(entry.stmt, buffer_data_to_buffer_);
      pipeline_writes.insert(access.writes.begin(), access.writes.end());
      flat_reads.push_back(access.reads);
    }
    ScalarBindAnalysis scalar =
        AnalyzeScalarBinds(flat_stmts, flat_reads, pipeline_writes);

    std::vector<size_t> scheduled_indices;
    for (size_t i = 0; i < entries.size(); ++i) {
      if (IsPipelineDeclarationStmt(entries[i].stmt) ||
          scalar.replayable_binds[i]) {
        continue;
      }
      scheduled_indices.push_back(i);
    }
    ICHECK(!scheduled_indices.empty())
        << "NormalizeZ3ScheduleUnits: markers do not contain a scheduled task";

    if (stage_seen_) {
      const Map<String, Any> &annotations = loop->annotations;
      bool old_manual = GetBool(annotations, kZ3ManualSchedule, false) ||
                        !GetIntegerArray(annotations, kZ3ManualStages).empty();
      ICHECK(!old_manual)
          << "NormalizeZ3ScheduleUnits: T.Stage cannot be combined with the "
             "old manual_schedule/manual_stages schema";
      for (const char *key : {"tl_pipeline_order", "tl_pipeline_stage",
                              "software_pipeline_order",
                              "software_pipeline_stage"}) {
        ICHECK(!annotations.count(key))
            << "NormalizeZ3ScheduleUnits: T.Stage cannot be combined with "
               "handwritten pipeline order/stage annotation "
            << key;
      }
      ICHECK(GetNestedIntegerArrays(annotations, kZ3PipeOrderDeps).empty())
          << "NormalizeZ3ScheduleUnits: T.Stage derives per-pipe issue order "
             "from source order; remove tl_z3_pipeline_pipe_order_deps";
    }

    Array<Integer> manual_stages;
    Array<String> execution_domains;
    int64_t max_stage = 0;
    std::vector<char> group_has_task(next_stage_group_, 0);
    for (size_t flat_index : scheduled_indices) {
      const NormalizedEntry &entry = entries[flat_index];
      execution_domains.push_back(ExecutionDomainExtractor::Extract(entry.stmt));
      if (entry.stage_group >= 0) group_has_task[entry.stage_group] = 1;
      if (stage_seen_) {
        int64_t stage = entry.stage >= 0 ? entry.stage : 0;
        manual_stages.push_back(Integer(stage));
        max_stage = std::max(max_stage, stage);
      }
    }
    for (int group = 0; group < next_stage_group_; ++group) {
      ICHECK(group_has_task[group])
          << "NormalizeZ3ScheduleUnits: T.Stage must wrap at least one "
             "complete scheduled task";
    }

    Map<String, Any> output = loop->annotations;
    output.Set(kZ3ExecutionDomains, execution_domains);
    if (stage_seen_) {
      output.Set(kZ3ManualSchedule, true);
      output.Set(kZ3ManualStages, manual_stages);
      output.erase(kZ3PipeOrderDeps);
      int64_t num_stages = 0;
      if (auto value = output.Get("num_stages"); value.has_value()) {
        const auto *imm = value.value().as<IntImmNode>();
        ICHECK(imm != nullptr)
            << "NormalizeZ3ScheduleUnits: num_stages must be an integer";
        num_stages = imm->value;
        ICHECK_GE(num_stages, 0)
            << "NormalizeZ3ScheduleUnits: num_stages must be non-negative";
      }
      if (num_stages == 0) {
        ICHECK_LT(max_stage, std::numeric_limits<int64_t>::max())
            << "NormalizeZ3ScheduleUnits: default pipeline depth overflows "
               "int64";
        output.Set("num_stages", IntImm(DataType::Int(64), max_stage + 1));
      }
      if (max_stage != 0) {
        ICHECK(GetBool(output, kZ3EnableOffset, false))
            << "NormalizeZ3ScheduleUnits: non-zero T.Stage values require "
               "enable_offset=True";
      }
    }

    return For(loop->loop_var, loop->min, loop->extent, loop->kind,
               MakePipelineBody(flat_stmts), loop->thread_binding, output,
               loop->step, loop->span);
  }

  Map<Var, Buffer> buffer_data_to_buffer_;
  bool stage_seen_{false};
  int next_stage_group_{0};
};

}  // namespace

tvm::transform::Pass NormalizeZ3ScheduleUnits() {
  using namespace tirx::transform;
  auto pass_func = [](PrimFunc func, const IRModule &, PassContext) {
    Stmt body = ScheduleUnitNormalizer::Rewrite(func);
    if (body.same_as(func->body)) return func;
    func.CopyOnWrite()->body = std::move(body);
    return func;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.NormalizeZ3ScheduleUnits", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = reflection;
  refl::GlobalDef().def("tl.transform.NormalizeZ3ScheduleUnits",
                        NormalizeZ3ScheduleUnits);
}

}  // namespace tl
}  // namespace tvm
