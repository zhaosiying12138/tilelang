/*!
 * \file z3_owner_ring_planning.cc
 * \brief Function-level validation for explicit two-owner CUDA rings.
 */

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>
#include <unordered_set>

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/cast.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include "common/warp_specialize.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace {

constexpr const char *kZ3PipelineConfig = "tl.enable_z3_pipeline_planning";
constexpr const char *kZ3OwnerRing = "tl_z3_owner_ring";
constexpr const char *kZ3OwnerRingValidated = "tl_z3_owner_ring_validated";
constexpr const char *kZ3Pipeline = "tl_z3_pipeline";
constexpr const char *kZ3Latencies = "tl_z3_pipeline_latencies";
constexpr const char *kZ3IIs = "tl_z3_pipeline_iis";
constexpr const char *kZ3ResourceFlags = "tl_z3_pipeline_resource_flags";
constexpr const char *kZ3DataDeps = "tl_z3_pipeline_data_deps";
constexpr const char *kZ3ResourceDeps = "tl_z3_pipeline_resource_deps";
constexpr const char *kZ3OwnerExclusionDeps = "tl_z3_pipeline_owner_exclusion_deps";
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
constexpr const char *kZ3RecalculateBufferVersions = "tl_z3_pipeline_recalculate_buffer_versions";
constexpr const char *kZ3Verbose = "tl_z3_pipeline_verbose";
constexpr const char *kZ3DebugDir = "tl_z3_pipeline_debug_dir";
constexpr const char *kZ3AuditII = "tl_z3_pipeline_audit_ii";
constexpr const char *kZ3AuditStarts = "tl_z3_pipeline_audit_starts";

int64_t GetInteger(const Map<String, Any> &annotations, const char *key,
                   int64_t default_value) {
  auto value = annotations.Get(key);
  if (!value.has_value()) return default_value;
  return Downcast<IntImm>(value.value())->value;
}

bool GetBool(const Map<String, Any> &annotations, const char *key,
             bool default_value) {
  auto value = annotations.Get(key);
  if (!value.has_value()) return default_value;
  return value.has_value() ? value.value().cast<bool>() : default_value;
}

String GetString(const Map<String, Any> &annotations, const char *key) {
  auto value = annotations.Get(key);
  ICHECK(value.has_value()) << "Z3OwnerRingPlanning: missing annotation " << key;
  return Downcast<String>(value.value());
}

std::vector<int64_t> GetIntegers(const Map<String, Any> &annotations,
                                 const char *key) {
  std::vector<int64_t> result;
  auto value = annotations.Get(key);
  if (!value.has_value()) return result;
  for (const Integer &item : Downcast<Array<Integer>>(value.value())) {
    result.push_back(item->value);
  }
  return result;
}

std::vector<std::vector<int64_t>> GetNestedIntegers(
    const Map<String, Any> &annotations, const char *key) {
  std::vector<std::vector<int64_t>> result;
  auto value = annotations.Get(key);
  if (!value.has_value()) return result;
  for (const Array<Integer> &group : Downcast<Array<Array<Integer>>>(value.value())) {
    std::vector<int64_t> converted;
    for (const Integer &item : group) converted.push_back(item->value);
    result.push_back(std::move(converted));
  }
  return result;
}

Array<int64_t> ToFFI(const std::vector<int64_t> &values) {
  return Array<int64_t>(values);
}

Array<Array<int64_t>> ToFFI(const std::vector<std::vector<int64_t>> &values) {
  Array<Array<int64_t>> result;
  for (const auto &group : values) result.push_back(Array<int64_t>(group));
  return result;
}

struct OwnerProblem {
  std::vector<int64_t> latencies;
  std::vector<int64_t> iis;
  std::vector<int64_t> resource_flags;
  std::vector<std::vector<int64_t>> owner;
  std::vector<int64_t> buffer_sizes;
  std::vector<std::vector<int64_t>> memory_groups;
  bool enable_offset{false};
  bool recalculate{true};
  int64_t resource_bits{7};
  int64_t seed{42};
  bool verbose{false};
  Optional<String> debug_dir{std::nullopt};
};

OwnerProblem ReadProblem(const For &loop) {
  const auto &annotations = loop->annotations;
  OwnerProblem problem;
  problem.latencies = GetIntegers(annotations, kZ3Latencies);
  problem.iis = GetIntegers(annotations, kZ3IIs);
  problem.resource_flags = GetIntegers(annotations, kZ3ResourceFlags);
  ICHECK_EQ(problem.latencies.size(), 1)
      << "Z3OwnerRingPlanning: each owner must supply exactly one synthetic group task";
  ICHECK_EQ(problem.iis.size(), 1);
  ICHECK_EQ(problem.resource_flags.size(), 1);
  ICHECK(problem.latencies[0] >= 0 && problem.iis[0] > 0 && problem.resource_flags[0] >= 0);
  for (const char *key : {kZ3DataDeps, kZ3ResourceDeps, kZ3PipeOrderDeps, kZ3StageOrderDeps}) {
    ICHECK(GetNestedIntegers(annotations, key).empty())
        << "Z3OwnerRingPlanning: " << key
        << " is outside the two-group contract; encode internal order in WSSchedule";
  }
  problem.owner = GetNestedIntegers(annotations, kZ3OwnerExclusionDeps);
  ICHECK_EQ(problem.owner.size(), 1)
      << "Z3OwnerRingPlanning: exactly one cross-owner exclusion is required";
  ICHECK_EQ(problem.owner[0].size(), 4);
  ICHECK(problem.owner[0][0] == 0 && problem.owner[0][1] == 1)
      << "Z3OwnerRingPlanning: the declared owner order must be (0, 1)";
  ICHECK(problem.owner[0][2] >= 0 && problem.owner[0][3] >= 0);
  problem.buffer_sizes = GetIntegers(annotations, kZ3BufferSizes);
  problem.memory_groups = GetNestedIntegers(annotations, kZ3MemoryGroups);
  problem.enable_offset = GetBool(annotations, kZ3EnableOffset, false);
  problem.recalculate = GetBool(annotations, kZ3RecalculateBufferVersions, true);
  problem.resource_bits = GetInteger(annotations, kZ3ResourceBits, 7);
  problem.seed = GetInteger(annotations, kZ3Seed, 42);
  problem.verbose = GetBool(annotations, kZ3Verbose, false);
  if (auto value = annotations.Get(kZ3DebugDir); value.has_value()) {
    problem.debug_dir = Downcast<String>(value.value());
  }
  ICHECK(!GetBool(annotations, kZ3ManualSchedule, false))
      << "Z3OwnerRingPlanning: manual stage offsets are outside the stage-zero owner contract";
  return problem;
}

void RequireSameProblem(const OwnerProblem &a, const OwnerProblem &b) {
  ICHECK(a.owner == b.owner)
      << "Z3OwnerRingPlanning: both owners must use identical exclusion latencies";
  ICHECK(a.buffer_sizes == b.buffer_sizes && a.memory_groups == b.memory_groups)
      << "Z3OwnerRingPlanning: buffer sizes and memory groups must be shared, not duplicated";
  ICHECK(a.enable_offset == b.enable_offset && a.recalculate == b.recalculate &&
         a.resource_bits == b.resource_bits && a.seed == b.seed &&
         a.verbose == b.verbose && a.debug_dir == b.debug_dir)
      << "Z3OwnerRingPlanning: owner solver options differ between owner loops";
}

class OwnerRingRewriter : public StmtExprMutator {
 public:
  OwnerRingRewriter(const std::vector<For> &owners, const SBlock &schedule_block,
                    int64_t ii, const Array<int64_t> &starts)
      : schedule_block_(schedule_block), ii_(ii) {
    for (int64_t start : starts) {
      starts_.push_back(IntImm(DataType::Int(64), start));
    }
    (void)owners;
  }

 private:
  Stmt VisitStmt_(const ForNode *op) final {
    // The dedicated owner pass has already validated and emitted runtime WS
    // metadata. Keep solver annotations for audit instead of deleting them.
    return GetRef<For>(op);
  }

  Stmt VisitStmt_(const SBlockNode *op) final {
    Stmt rewritten = StmtExprMutator::VisitStmt_(op);
    if (op == schedule_block_.get()) {
      SBlock updated = Downcast<SBlock>(rewritten);
      updated.CopyOnWrite()->annotations.Set(kZ3OwnerRingValidated, Bool(true));
      updated.CopyOnWrite()->annotations.Set(kZ3AuditII,
                                             IntImm(DataType::Int(64), ii_));
      updated.CopyOnWrite()->annotations.Set(kZ3AuditStarts, starts_);
      return updated;
    }
    return rewritten;
  }

  SBlock schedule_block_;
  int64_t ii_;
  Array<IntImm> starts_;
};

PrimFunc ValidateAndConsume(PrimFunc func) {
  std::vector<For> owners;
  SBlock schedule_block{nullptr};
  WSSchedule schedule{nullptr};
  PostOrderVisit(func->body, [&](const ObjectRef &node) {
    if (const auto *loop = node.as<ForNode>()) {
      if (loop->annotations.count(kZ3OwnerRing)) owners.push_back(GetRef<For>(loop));
    } else if (const auto *block = node.as<SBlockNode>()) {
      if (block->annotations.count("tl.ws_schedule")) {
        ICHECK(!schedule_block.defined())
            << "Z3OwnerRingPlanning: multiple WSSchedule blocks are unsupported";
        schedule_block = GetRef<SBlock>(block);
        schedule = Downcast<WSSchedule>(block->annotations.Get("tl.ws_schedule").value());
      }
    }
  });
  if (owners.empty()) {
    ICHECK(!schedule_block.defined())
        << "Z3OwnerRingPlanning: a WSSchedule requires two owner annotations";
    return func;
  }
  ICHECK(owners.size() == 2)
      << "Z3OwnerRingPlanning: expected exactly two owner loops, got " << owners.size();
  ICHECK(schedule_block.defined())
      << "Z3OwnerRingPlanning: explicit owner loops require a WSSchedule";
  std::sort(owners.begin(), owners.end(), [](const For &a, const For &b) {
    return GetInteger(a->annotations, kZ3OwnerRing, -1) < GetInteger(b->annotations, kZ3OwnerRing, -1);
  });
  for (size_t i = 0; i < owners.size(); ++i) {
    ICHECK_EQ(GetInteger(owners[i]->annotations, kZ3OwnerRing, -1), static_cast<int64_t>(i))
        << "Z3OwnerRingPlanning: owner indices must be exactly 0 and 1";
  }

  ICHECK_EQ(schedule->roles.size(), 2);
  ICHECK_EQ(schedule->pipelines.size(), 1);
  WSPipeline pipeline = schedule->pipelines[0];
  ICHECK_EQ(pipeline->owner_scopes.size(), 2);
  ICHECK_GT(pipeline->depth, 0);
  for (size_t i = 0; i < owners.size(); ++i) {
    String scope_id = GetString(owners[i]->annotations, "tl.ws_op_id");
    ICHECK_EQ(scope_id, pipeline->owner_scopes[i])
        << "Z3OwnerRingPlanning: WSSchedule owner scope order does not match the loop order";
    if (owners[i]->annotations.count("num_stages")) {
      ICHECK_EQ(GetInteger(owners[i]->annotations, "num_stages", pipeline->depth), pipeline->depth)
          << "Z3OwnerRingPlanning: num_stages and WSSchedule depth differ";
    }
  }

  OwnerProblem first = ReadProblem(owners[0]);
  OwnerProblem second = ReadProblem(owners[1]);
  RequireSameProblem(first, second);
  auto scheduler = ffi::Function::GetGlobal("tl.cuda.transform.z3_schedule_loop_python");
  ICHECK(scheduler.has_value())
      << "Z3OwnerRingPlanning: tl.cuda.transform.z3_schedule_loop_python is not registered";
  using SolverResult = ffi::Tuple<ffi::Array<int64_t>, ffi::Array<int64_t>, int64_t>;
  SolverResult result = scheduler.value()(
      pipeline->depth,
      Array<int64_t>({first.latencies[0], second.latencies[0]}),
      Array<int64_t>({first.iis[0], second.iis[0]}),
      Array<int64_t>({first.resource_flags[0], second.resource_flags[0]}),
      Array<Array<int64_t>>(), Array<Array<int64_t>>(), ToFFI(first.owner),
      Array<Array<int64_t>>({Array<int64_t>({0, 1})}), ToFFI(first.buffer_sizes), ToFFI(first.memory_groups),
      Array<Array<int64_t>>(), first.enable_offset, Array<int64_t>(), false,
      first.resource_bits, first.seed, first.recalculate, first.verbose, first.debug_dir)
      .cast<SolverResult>();
  Array<int64_t> starts = result.get<0>();
  Array<int64_t> versions = result.get<1>();
  int64_t ii = result.get<2>();
  ICHECK_EQ(starts.size(), 2);
  ICHECK_GT(ii, 0);
  int64_t left_residue = starts[0] % ii;
  int64_t right_residue = starts[1] % ii;
  ICHECK(starts[0] >= 0 && starts[1] >= 0 && left_residue >= 0 && right_residue >= 0);
  ICHECK(starts[0] < ii && starts[1] < ii)
      << "Z3OwnerRingPlanning: owner epochs must remain in stage zero";
  int64_t latency_lr = first.owner[0][2];
  int64_t latency_rl = first.owner[0][3];
  ICHECK(right_residue - left_residue >= latency_lr &&
         left_residue - right_residue + ii >= latency_rl)
      << "Z3OwnerRingPlanning: declared owner order 0 -> 1 does not satisfy "
         "forward and reverse hand-off constraints; refusing to reorder";
  for (int64_t version : versions) {
    ICHECK(version >= 1 && version <= pipeline->depth)
        << "Z3OwnerRingPlanning: solver buffer version exceeds runtime ring depth";
  }
  for (const auto &group : first.memory_groups) {
    ICHECK(!group.empty());
    int64_t used = 0;
    for (size_t member = 1; member < group.size(); ++member) {
      int64_t buffer = group[member];
      ICHECK(buffer >= 0 && buffer < static_cast<int64_t>(versions.size()) &&
             buffer < static_cast<int64_t>(first.buffer_sizes.size()));
      used += versions[buffer] * first.buffer_sizes[buffer];
    }
    ICHECK(used <= group[0])
        << "Z3OwnerRingPlanning: selected versions exceed declared memory capacity";
  }

  func.CopyOnWrite()->body = OwnerRingRewriter(owners, schedule_block, ii, starts)(func->body);
  return func;
}

} // namespace

tvm::transform::Pass Z3OwnerRingPlanning() {
  using namespace tirx::transform;
  auto pass_func = [](PrimFunc func, const IRModule &, PassContext ctx) {
    bool enabled = ctx->GetConfig<Bool>(kZ3PipelineConfig, Bool(false)).value();
    bool has_owner = false;
    PostOrderVisit(func->body, [&](const ObjectRef &node) {
      if (const auto *loop = node.as<ForNode>()) {
        has_owner = has_owner || loop->annotations.count(kZ3OwnerRing);
      }
    });
    ICHECK(!has_owner || enabled)
        << "Z3OwnerRingPlanning: owner annotations require tl.enable_z3_pipeline_planning";
    if (enabled) return ValidateAndConsume(func);
    return func;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.Z3OwnerRingPlanning", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = reflection;
  refl::GlobalDef().def("tl.transform.Z3OwnerRingPlanning", Z3OwnerRingPlanning);
}

} // namespace tl
} // namespace tvm
