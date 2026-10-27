/*!
 * \file z3_task_analysis.cc
 * \brief Opt-in task, dependency, resource, and cost inference for CUDA Z3.
 */

#include "z3_task_analysis.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
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

#include "../cuda/transform/z3_dependency_analysis.h"
#include "../op/builtin.h"
#include "../op/copy.h"
#include "../op/fill.h"
#include "../op/gemm.h"
#include "../op/operator.h"
#include "../op/parallel.h"
#include "../op/utils.h"
#include "common/bind_utils.h"
#include "common/pipeline_utils.h"
#include "common/z3_buffer_versions.h"
#include "common/z3_cost_expression.h"
#include "common/z3_schedule_hints.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

ffi::Any NativeExtent::ToFFI() const {
  ICHECK_GE(upper_bound, 0);
  if (!dynamic) return Integer(upper_bound);
  Map<String, Any> result;
  result.Set("upper_bound", Integer(upper_bound));
  result.Set("reliability", String(reliability));
  return result;
}

Map<String, Any> NativeCostExpr::ToFFI() const {
  Map<String, Any> result;
  switch (kind) {
    case Kind::kLeaf:
      result.Set("kind", String("leaf"));
      result.Set("features", features);
      break;
    case Kind::kSerial: {
      Array<Any> children_ffi;
      for (const NativeCostExpr &child : children) {
        children_ffi.push_back(child.ToFFI());
      }
      result.Set("kind", String("serial"));
      result.Set("children", children_ffi);
      break;
    }
    case Kind::kRepeat:
      ICHECK_EQ(children.size(), 1U);
      result.Set("kind", String("repeat"));
      result.Set("trips", trips.ToFFI());
      result.Set("body", children[0].ToFFI());
      break;
    case Kind::kBranch:
      ICHECK_EQ(children.size(), 2U);
      result.Set("kind", String("branch"));
      result.Set("then", children[0].ToFFI());
      result.Set("else", children[1].ToFFI());
      break;
  }
  return result;
}

namespace {

constexpr const char *kTaskAnalysisConfig = "tl.enable_z3_task_analysis";
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
constexpr const char *kZ3Hardware = "tl_z3_pipeline_hardware";
constexpr const char *kZ3Calibration = "tl_z3_pipeline_calibration";
constexpr const char *kZ3RequireMeasured = "tl_z3_pipeline_require_measured";
constexpr const char *kZ3CostReliability = "tl_z3_pipeline_cost_reliability";
constexpr const char *kZ3CostFallbacks = "tl_z3_pipeline_cost_fallback_used";
constexpr const char *kZ3AsyncSameStage =
    "tl_z3_pipeline_async_same_stage_deps";
constexpr const char *kZ3ManualStages = "tl_z3_pipeline_manual_stages";
constexpr const char *kZ3ManualSchedule = "tl_z3_pipeline_manual_schedule";
constexpr int64_t kAllResources = 127;

bool HasAnnotation(const Map<String, Any> &annotations, const char *key) {
  return annotations.Get(key).has_value();
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

bool GetBool(const Map<String, Any> &annotations, const char *key,
             bool default_value = false) {
  auto value = annotations.Get(key);
  return value.has_value() ? value.value().cast<bool>() : default_value;
}

String GetRequiredPath(const Map<String, Any> &annotations, const char *key) {
  auto value = annotations.Get(key);
  ICHECK(value.has_value())
      << "InferZ3PipelineProblem: missing " << key
      << "; provide an explicit JSON path or explicitly supply every inferred "
         "cost/capacity field";
  String path = Downcast<String>(value.value());
  ICHECK(!path.empty()) << "InferZ3PipelineProblem: " << key
                        << " must not be empty";
  return path;
}

int64_t CheckedAdd(int64_t left, int64_t right, const char *what) {
  ICHECK(left >= 0 && right >= 0 && left <= std::numeric_limits<int64_t>::max() - right)
      << "InferZ3PipelineProblem: int64 overflow in " << what;
  return left + right;
}

int64_t CheckedMul(int64_t left, int64_t right, const char *what) {
  ICHECK(left >= 0 && right >= 0 &&
         (left == 0 || right <= std::numeric_limits<int64_t>::max() / left))
      << "InferZ3PipelineProblem: int64 overflow in " << what;
  return left * right;
}

int64_t StaticNonnegative(const PrimExpr &expr, const char *what) {
  arith::Analyzer analyzer;
  PrimExpr simplified = analyzer.Simplify(expr);
  const int64_t *value = as_const_int(simplified);
  ICHECK(value != nullptr)
      << "InferZ3PipelineProblem: " << what
      << " must be static for automatic task analysis; got " << expr
      << ". Supply explicit cost/buffer fields instead of converting unknown "
         "extents to zero";
  ICHECK_GE(*value, 0) << "InferZ3PipelineProblem: negative " << what;
  return *value;
}

int64_t DTypeBits(DataType dtype) {
  return CheckedMul(dtype.bits(), dtype.lanes(), "dtype bit width");
}

int64_t DTypeBytes(DataType dtype) {
  int64_t bits = DTypeBits(dtype);
  return (CheckedAdd(bits, 7, "rounded dtype size")) / 8;
}

std::string ModelDType(DataType dtype) {
  dtype = dtype.element_of();
  if (dtype.is_float16()) return "f16";
  if (dtype.is_bfloat16()) return "bf16";
  if (dtype.is_tfloat32()) return "tf32";
  if (dtype.is_float() && dtype.bits() == 32) return "f32";
  if (dtype.is_int() && dtype.bits() == 8) return "int8";
  if (dtype.is_int() && dtype.bits() == 32) return "s32";
  std::ostringstream stream;
  stream << dtype;
  return stream.str();
}

std::string ModelScope(const Buffer &buffer) {
  std::string scope = buffer.scope();
  if (scope.empty() || scope == "global") return "global";
  if (scope == "shared" || scope == "shared.dyn") return "shared";
  if (scope == "local" || scope == "local.var" || scope == "local.fragment" ||
      scope == "register") {
    return "register";
  }
  if (scope == "shared.tmem" || scope == "tmem") return "tmem";
  return scope;
}

bool KnownModelScope(const std::string &scope) {
  return scope == "global" || scope == "shared" || scope == "register" ||
         scope == "tmem";
}

NativeExtent MultiplyExtents(const NativeExtent &left,
                             const NativeExtent &right, const char *what) {
  int64_t upper = CheckedMul(left.upper_bound, right.upper_bound, what);
  if (!left.dynamic && !right.dynamic) return NativeExtent::Exact(upper);
  std::string reliability =
      left.reliability == "unknown" || right.reliability == "unknown"
          ? "unknown"
          : (left.reliability == "estimated" ||
                     right.reliability == "estimated"
                 ? "estimated"
                 : "upper_bound");
  return NativeExtent::UpperBound(upper, reliability);
}

NativeExtent MaximumExtent(const NativeExtent &left,
                           const NativeExtent &right) {
  if (!left.dynamic && !right.dynamic) {
    return NativeExtent::Exact(std::max(left.upper_bound, right.upper_bound));
  }
  return NativeExtent::UpperBound(
      std::max(left.upper_bound, right.upper_bound),
      left.reliability == "unknown" || right.reliability == "unknown"
          ? "unknown"
          : "upper_bound");
}

/* Return a provably finite extent without treating a dtype-wide bound as
 * useful dynamic information. A static allocation dimension is an independent
 * cap, matching the actual region rather than replacing its extent wholesale. */
std::optional<NativeExtent> TryFiniteExtent(
    const PrimExpr &expr, arith::Analyzer *analyzer,
    std::optional<int64_t> shape_bound, const char *what) {
  ICHECK(analyzer != nullptr);
  PrimExpr simplified = analyzer->Simplify(expr);
  if (const int64_t *value = as_const_int(simplified)) {
    ICHECK_GE(*value, 0) << "InferZ3PipelineProblem: negative " << what;
    if (shape_bound.has_value()) {
      ICHECK_GE(*shape_bound, 0);
      ICHECK_LE(*value, *shape_bound)
          << "InferZ3PipelineProblem: " << what
          << " exceeds its static buffer-shape bound";
    }
    return NativeExtent::Exact(*value);
  }

  std::optional<int64_t> upper = shape_bound;
  arith::ConstIntBound bound = analyzer->const_int_bound(expr);
  arith::Analyzer unconstrained_analyzer;
  Var unconstrained("z3_unconstrained_extent", expr.dtype());
  int64_t unconstrained_max =
      unconstrained_analyzer.const_int_bound(unconstrained)->max_value;
  bool useful = bound->max_value >= 0 &&
                bound->max_value != arith::ConstIntBound::kPosInf &&
                bound->max_value < unconstrained_max;
  if (useful) {
    upper = upper.has_value() ? std::min(*upper, bound->max_value)
                              : bound->max_value;
  }
  if (!upper.has_value()) return std::nullopt;
  ICHECK_GE(*upper, 0) << "InferZ3PipelineProblem: negative " << what;
  return NativeExtent::UpperBound(*upper);
}

NativeExtent RequireFiniteExtent(const PrimExpr &expr,
                                 arith::Analyzer *analyzer,
                                 std::optional<int64_t> shape_bound,
                                 const char *what) {
  std::optional<NativeExtent> result =
      TryFiniteExtent(expr, analyzer, shape_bound, what);
  ICHECK(result.has_value())
      << "InferZ3PipelineProblem: " << what
      << " requires a finite dynamic extent upper_bound from constraints, "
         "assumptions, static buffer shape, or "
      << kZ3CudaTripBound;
  return result.value();
}

/* The flattener intentionally mirrors the pipeline adapter. Buffer accesses
 * use the same strict collector as nested CUDA Z3 analysis. */

class TaskSeqStmtFlattener : public StmtFunctor<Array<Stmt>(const Stmt &)> {
public:
  using Base = StmtFunctor<Array<Stmt>(const Stmt &)>;
  static Array<Stmt> Flatten(const Array<Stmt> &stmts) {
    TaskSeqStmtFlattener flattener;
    Array<Stmt> flattened;
    for (const Stmt &stmt : stmts) {
      Array<Stmt> nested = flattener(stmt);
      flattened.insert(flattened.end(), nested.begin(), nested.end());
    }
    return flattened;
  }
  Array<Stmt> VisitStmt(const Stmt &stmt) final {
    if (!stmt.as<SeqStmtNode>()) return Array<Stmt>{stmt};
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

struct ScheduledTaskAnalysis {
  Array<Stmt> flat_stmts;
  Array<Stmt> scheduled_stmts;
  std::vector<size_t> scheduled_indices;
  std::vector<int> scheduled_for_flat;
  std::vector<Array<BufferRegion>> reads;
  std::vector<Array<BufferRegion>> writes;
  ScalarBindAnalysis scalar;
};

ScheduledTaskAnalysis AnalyzeScheduledTasks(
    const Stmt &body, const Map<Var, Buffer> &buffer_data_to_buffer,
    bool strict_opaque) {
  Array<Stmt> flat_stmts =
      TaskSeqStmtFlattener::Flatten(NormalizePipelineBody(body));
  BufferSet pipeline_write_buffers;
  std::vector<std::pair<Array<BufferRegion>, Array<BufferRegion>>> accesses;
  accesses.reserve(flat_stmts.size());
  for (const Stmt &stmt : flat_stmts) {
    if (IsPipelineDeclarationStmt(stmt)) {
      accesses.push_back({});
      continue;
    }
    cuda::z3::CudaTaskAnalysisOptions access_options;
    access_options.strict_opaque = strict_opaque;
    access_options.allow_read_state = true;
    access_options.collect_attribute_node = false;
    access_options.explicit_state_contracts = false;
    access_options.diagnostic_prefix = "InferZ3PipelineProblem";
    auto collected = cuda::z3::AnalyzeCudaTaskAccesses(
        stmt, buffer_data_to_buffer, access_options);
    std::pair<Array<BufferRegion>, Array<BufferRegion>> access{
        collected.reads, collected.writes};
    for (const BufferRegion &write : access.second) {
      pipeline_write_buffers.insert(write->buffer);
    }
    accesses.push_back(std::move(access));
  }

  ScheduledTaskAnalysis analysis;
  analysis.flat_stmts = flat_stmts;
  std::vector<Array<BufferRegion>> flat_reads;
  flat_reads.reserve(flat_stmts.size());
  for (const auto &access : accesses) flat_reads.push_back(access.first);
  analysis.scalar =
      AnalyzeScalarBinds(flat_stmts, flat_reads, pipeline_write_buffers);
  analysis.scheduled_for_flat.assign(flat_stmts.size(), -1);
  for (size_t i = 0; i < flat_stmts.size(); ++i) {
    const Stmt &stmt = flat_stmts[i];
    if (IsPipelineDeclarationStmt(stmt)) continue;
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

struct NativeTaskFeatures {
  Array<Any> copies;
  Array<Any> gemms;
  std::map<std::string, int64_t> vector;
  int64_t scalar{0};
  int64_t structural_resource_mask{0};
  bool unknown{false};
  bool upper_bound{false};

  bool Empty() const {
    if (!copies.empty() || !gemms.empty() || scalar != 0 ||
        structural_resource_mask != 0 || unknown) {
      return false;
    }
    for (const auto &[_, count] : vector) {
      if (count != 0) return false;
    }
    return true;
  }

  Map<String, Any> ToFFI() const {
    Map<String, Any> vector_ffi;
    for (const auto &[name, count] : vector) {
      vector_ffi.Set(String(name), Integer(count));
    }
    if (!vector_ffi.Get("read_bytes").has_value()) {
      vector_ffi.Set("read_bytes", Integer(0));
    }
    Map<String, Any> result;
    result.Set("copies", copies);
    result.Set("gemms", gemms);
    result.Set("vector", vector_ffi);
    result.Set("scalar", Integer(scalar));
    result.Set("reliability",
               String(unknown ? "unknown" : upper_bound ? "upper_bound" : "measured"));
    return result;
  }
};

struct TaskCostOverride {
  int64_t latency{-1};
  int64_t ii{-1};
};

TaskCostOverride GetTaskCostOverride(const Stmt &stmt) {
  TaskCostOverride result;
  const auto *task = stmt.as<AttrStmtNode>();
  if (task == nullptr || task->attr_key != kZ3CudaTaskAttr) return result;
  Map<String, Any> metadata = task->node.cast<Map<String, Any>>();
  for (const auto &[key, value] : metadata) {
    ICHECK(key == "latency" || key == "ii")
        << "InferZ3PipelineProblem: unsupported T.Task metadata key " << key;
    const auto *imm = value.as<IntImmNode>();
    ICHECK(imm != nullptr)
        << "InferZ3PipelineProblem: T.Task " << key
        << " must be a constant integer";
    if (key == "latency") {
      ICHECK_GE(imm->value, 0)
          << "InferZ3PipelineProblem: T.Task latency must be non-negative";
      result.latency = imm->value;
    } else {
      ICHECK_GT(imm->value, 0)
          << "InferZ3PipelineProblem: T.Task ii must be positive";
      result.ii = imm->value;
    }
  }
  if (result.latency >= 0 && result.ii >= 0) {
    ICHECK_GE(result.latency, result.ii)
        << "InferZ3PipelineProblem: T.Task latency must be >= ii";
  }
  return result;
}

class TaskFeatureExtractor : public StmtExprVisitor {
public:
  static NativeTaskFeatures Extract(const Stmt &stmt, int64_t scale = 1,
                                    int vector_depth = 0,
                                    arith::Analyzer *analyzer = nullptr) {
    TaskFeatureExtractor extractor;
    arith::Analyzer local_analyzer;
    extractor.scale_ = scale;
    extractor.vector_depth_ = vector_depth;
    extractor.analyzer_ = analyzer == nullptr ? &local_analyzer : analyzer;
    extractor(stmt);
    return std::move(extractor.features_);
  }

private:
  void AddVector(const std::string &name, int64_t count) {
    int64_t &slot = features_.vector[name];
    slot = CheckedAdd(slot, count, "vector operation count");
  }

  void CountOperation(const std::string &name, DataType dtype) {
    int64_t work = CheckedMul(scale_, dtype.lanes(), "operation lane count");
    if (vector_depth_ > 0 || dtype.lanes() > 1) {
      AddVector(name, work);
    } else {
      features_.scalar = CheckedAdd(features_.scalar, work, "scalar operation count");
    }
  }

  void CountUnknown(DataType dtype) {
    int64_t work = CheckedMul(scale_, dtype.lanes(), "unknown operation count");
    AddVector("unknown", work);
    features_.structural_resource_mask |= kAllResources;
    features_.unknown = true;
  }

  void AddMemoryAccess(const std::string &src, const std::string &dst,
                       const NativeExtent &bytes, const std::string &geometry,
                       int64_t width) {
    if (src == "register" && dst == "register") {
      ICHECK(!bytes.dynamic)
          << "InferZ3PipelineProblem: dynamic register vector read bytes "
             "require composed loop extraction";
      int64_t &read_bytes = features_.vector["read_bytes"];
      read_bytes = CheckedAdd(read_bytes, bytes.upper_bound,
                              "register vector read bytes");
      features_.structural_resource_mask |= 16;
      return;
    }
    if (!KnownModelScope(src) || !KnownModelScope(dst)) {
      features_.structural_resource_mask |= kAllResources;
      features_.unknown = true;
    }
    features_.upper_bound = features_.upper_bound || bytes.dynamic;
    Map<String, Any> access;
    access.Set("src_scope", String(src));
    access.Set("dst_scope", String(dst));
    access.Set("bytes", bytes.ToFFI());
    access.Set("geometry", String(geometry));
    access.Set("width", Integer(width));
    access.Set("reliability",
               String(bytes.dynamic ? bytes.reliability : "measured"));
    features_.copies.push_back(std::move(access));
  }

  void AddMemoryAccess(const std::string &src, const std::string &dst,
                       int64_t bytes, const std::string &geometry,
                       int64_t width) {
    AddMemoryAccess(src, dst, NativeExtent::Exact(bytes), geometry, width);
  }

  NativeExtent RegionElements(const Buffer &buffer, const Array<Range> &ranges,
                              int64_t multiplier) {
    NativeExtent elements = NativeExtent::Exact(multiplier);
    for (size_t i = 0; i < ranges.size(); ++i) {
      std::optional<int64_t> shape;
      if (i < buffer->shape.size()) {
        PrimExpr simplified = analyzer_->Simplify(buffer->shape[i]);
        const int64_t *shape_extent = as_const_int(simplified);
        if (shape_extent != nullptr) {
          ICHECK_GE(*shape_extent, 0);
          shape = *shape_extent;
        }
      }
      NativeExtent extent =
          RequireFiniteExtent(ranges[i]->extent, analyzer_, shape,
                              "copy region extent");
      elements = MultiplyExtents(elements, extent, "copy region elements");
    }
    return elements;
  }

  NativeExtent ElementsBytes(const NativeExtent &elements, DataType dtype) {
    NativeExtent bits = MultiplyExtents(
        elements, NativeExtent::Exact(DTypeBits(dtype)), "copy payload bits");
    int64_t bytes = (CheckedAdd(bits.upper_bound, 7, "copy payload rounding")) / 8;
    return bits.dynamic ? NativeExtent::UpperBound(bytes, bits.reliability)
                        : NativeExtent::Exact(bytes);
  }

  int64_t ElementsBytes(int64_t elements, DataType dtype) {
    int64_t bits = CheckedMul(elements, DTypeBits(dtype), "copy payload bits");
    return (CheckedAdd(bits, 7, "copy payload rounding")) / 8;
  }

  void HandleCopy(const CopyNode *copy) {
    NativeExtent src_elements =
        RegionElements(copy->src, copy->src_range, scale_);
    NativeExtent dst_elements =
        RegionElements(copy->dst, copy->dst_range, scale_);
    NativeExtent src_bytes = ElementsBytes(src_elements, copy->src->dtype);
    NativeExtent dst_bytes = ElementsBytes(dst_elements, copy->dst->dtype);
    if (src_bytes.upper_bound != dst_bytes.upper_bound || src_bytes.dynamic ||
        dst_bytes.dynamic) {
      features_.upper_bound = true;
    }
    NativeExtent bytes = MaximumExtent(src_bytes, dst_bytes);
    std::string geometry = "unknown";
    int64_t width = 0;
    if (auto annotated = copy->annotations.Get("coalesced_width")) {
      const auto *imm = annotated->as<IntImmNode>();
      ICHECK(imm != nullptr && imm->value > 0)
          << "InferZ3PipelineProblem: coalesced_width must be a positive IntImm";
      geometry = "coalesced";
      width = ElementsBytes(imm->value, copy->src->dtype);
    }
    AddMemoryAccess(ModelScope(copy->src), ModelScope(copy->dst), bytes,
                    geometry, width);
  }

  void HandleFill(const FillNode *fill) {
    NativeExtent elements = RegionElements(fill->dst, fill->region, scale_);
    NativeExtent bytes = ElementsBytes(elements, fill->dst->dtype);
    AddMemoryAccess("register", ModelScope(fill->dst), bytes, "unknown",
                    DTypeBytes(fill->dst->dtype));
    this->VisitExpr(fill->value);
  }

  void HandleGemm(const GemmNode *gemm) {
    ICHECK(gemm->a_->dtype == gemm->b_->dtype)
        << "InferZ3PipelineProblem: mixed-input-dtype GEMM has no TaskFeatures "
           "representation; provide explicit latency/II/resource fields";
    ICHECK(gemm->m_ > 0 && gemm->n_ > 0 && gemm->k_ > 0)
        << "InferZ3PipelineProblem: GEMM dimensions must be positive";
    Map<String, Any> feature;
    feature.Set("M", Integer(gemm->m_));
    feature.Set("N", Integer(gemm->n_));
    feature.Set("K", Integer(gemm->k_));
    feature.Set("dtype", String(ModelDType(gemm->a_->dtype)));
    feature.Set("acc", String(ModelDType(gemm->c_->dtype)));
    feature.Set("reliability", String("measured"));
    features_.gemms.push_back(std::move(feature));
  }

  void VisitStmt_(const ForNode *op) final {
    int64_t extent = StaticNonnegative(op->extent, "task loop extent");
    int64_t old_scale = scale_;
    scale_ = CheckedMul(scale_, extent, "task loop trip count");
    bool parallel = op->kind == ForKind::kParallel ||
                    op->kind == ForKind::kVectorized ||
                    op->kind == ForKind::kThreadBinding;
    if (parallel) ++vector_depth_;
    this->VisitStmt(op->body);
    if (parallel) --vector_depth_;
    scale_ = old_scale;
  }

  void VisitStmt_(const IfThenElseNode *op) final {
    features_.upper_bound = true;
    this->VisitExpr(op->condition);
    this->VisitStmt(op->then_case);
    if (op->else_case.defined()) this->VisitStmt(op->else_case.value());
  }

  void VisitStmt_(const WhileNode *) final {
    LOG(FATAL) << "InferZ3PipelineProblem: While task costs are not statically "
                  "determinate; provide explicit latency/II/resource fields";
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    int64_t bytes = CheckedMul(DTypeBytes(op->buffer->dtype), scale_,
                               "buffer store bytes");
    if (ModelScope(op->buffer) != "register") {
      AddMemoryAccess("register", ModelScope(op->buffer), bytes, "unknown",
                      DTypeBytes(op->buffer->dtype));
    } else {
      // VectorOps has read-bandwidth but no independent register-write term.
      // Keep the CUDA-core resource without inventing a shared/LSU transfer.
      features_.structural_resource_mask |= 16;
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const BufferLoadNode *op) final {
    int64_t bytes = CheckedMul(DTypeBytes(op->dtype), scale_,
                               "buffer load bytes");
    AddMemoryAccess(ModelScope(op->buffer), "register", bytes, "unknown",
                    DTypeBytes(op->dtype));
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const CallNode *op) final {
    if (auto tile_op = ParseOperator(GetRef<Call>(op)); tile_op.defined()) {
      if (const auto *copy = tile_op.as<CopyNode>()) {
        HandleCopy(copy);
      } else if (const auto *fill = tile_op.as<FillNode>()) {
        HandleFill(fill);
      } else if (const auto *gemm = tile_op.as<GemmNode>()) {
        HandleGemm(gemm);
      } else if (const auto *parallel = tile_op.as<ParallelOpNode>()) {
        this->VisitStmt(parallel->GetRoot());
      } else {
        CountUnknown(op->dtype);
      }
      return;
    }
    if (op->op.same_as(builtin::address_of()) ||
        op->op.same_as(tl::access_ptr()) ||
        op->op.same_as(builtin::tvm_access_ptr())) {
      // Pointer transport is not itself a payload transfer. Region-aware
      // dependency extraction still visits the complete access separately.
      return;
    }
    if (const auto *op_node = op->op.as<OpNode>()) {
      std::string name = op_node->name;
      static const std::unordered_set<std::string> special = {
          "tirx.exp", "tirx.exp10", "tirx.erf", "tirx.tanh",
          "tirx.sigmoid", "tirx.sqrt", "tirx.rsqrt", "tirx.log",
          "tirx.log2", "tirx.log10", "tirx.log1p", "tirx.tan",
          "tirx.cos", "tirx.cosh", "tirx.sin", "tirx.sinh",
          "tirx.asin", "tirx.acos", "tirx.atan", "tirx.acosh",
          "tirx.asinh", "tirx.atanh", "tirx.atan2", "tirx.hypot"};
      static const std::unordered_set<std::string> logic = {
          "tirx.bitwise_and", "tirx.bitwise_or", "tirx.bitwise_xor",
          "tirx.shift_left", "tirx.shift_right", "tirx.popcount",
          "tirx.clz"};
      if (name == "tirx.exp2") {
        CountOperation("exp2", op->dtype);
      } else if (special.count(name)) {
        CountOperation("special", op->dtype);
      } else if (logic.count(name)) {
        CountOperation("logic", op->dtype);
      } else if (name == "tirx.fma") {
        CountOperation("fma", op->dtype);
      } else if (name == "tirx.call_extern" ||
                 name == "tirx.call_pure_extern" ||
                 name == "tirx.call_packed" || name == "tirx.tvm_call_packed") {
        CountUnknown(op->dtype);
      } else {
        // Registered arithmetic intrinsics without an explicit model remain
        // visible as unknown work rather than free operations.
        CountUnknown(op->dtype);
      }
    } else {
      CountUnknown(op->dtype);
    }
    StmtExprVisitor::VisitExpr_(op);
  }

#define TVM_TL_Z3_COUNT_BINARY(NodeType, FeatureName)                         \
  void VisitExpr_(const NodeType##Node *op) final {                           \
    CountOperation(FeatureName, op->dtype);                                   \
    StmtExprVisitor::VisitExpr_(op);                                          \
  }
  TVM_TL_Z3_COUNT_BINARY(Add, "add")
  TVM_TL_Z3_COUNT_BINARY(Sub, "sub")
  TVM_TL_Z3_COUNT_BINARY(Mul, "mul")
  TVM_TL_Z3_COUNT_BINARY(Div, "div")
  TVM_TL_Z3_COUNT_BINARY(Mod, "mod")
  TVM_TL_Z3_COUNT_BINARY(FloorDiv, "div")
  TVM_TL_Z3_COUNT_BINARY(FloorMod, "mod")
  TVM_TL_Z3_COUNT_BINARY(Min, "min_max")
  TVM_TL_Z3_COUNT_BINARY(Max, "min_max")
  TVM_TL_Z3_COUNT_BINARY(EQ, "cmp")
  TVM_TL_Z3_COUNT_BINARY(NE, "cmp")
  TVM_TL_Z3_COUNT_BINARY(LT, "cmp")
  TVM_TL_Z3_COUNT_BINARY(LE, "cmp")
  TVM_TL_Z3_COUNT_BINARY(GT, "cmp")
  TVM_TL_Z3_COUNT_BINARY(GE, "cmp")
  TVM_TL_Z3_COUNT_BINARY(And, "logic")
  TVM_TL_Z3_COUNT_BINARY(Or, "logic")
#undef TVM_TL_Z3_COUNT_BINARY

  void VisitExpr_(const NotNode *op) final {
    CountOperation("logic", op->dtype);
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const CastNode *op) final {
    // VectorOps has no separate cast counter; logic is the calibrated CUDA-core
    // conversion bucket. Keeping it non-free avoids silently dropping casts.
    CountOperation("logic", op->dtype);
    StmtExprVisitor::VisitExpr_(op);
  }

  NativeTaskFeatures features_;
  int64_t scale_{1};
  int vector_depth_{0};
  arith::Analyzer *analyzer_{nullptr};
};

class NativeCostExprExtractor {
public:
  static NativeCostExpr Extract(const Stmt &stmt,
                                bool allow_unbounded_while = false) {
    arith::Analyzer analyzer;
    NativeCostExprExtractor extractor(allow_unbounded_while);
    return extractor.ExtractImpl(stmt, &analyzer, 1, 0);
  }

private:
  explicit NativeCostExprExtractor(bool allow_unbounded_while)
      : allow_unbounded_while_(allow_unbounded_while) {}

  static void AppendSequence(const Stmt &stmt, Array<Stmt> *output) {
    if (!stmt.defined()) return;
    if (const auto *seq = stmt.as<SeqStmtNode>()) {
      for (const Stmt &child : seq->seq) AppendSequence(child, output);
    } else {
      output->push_back(stmt);
    }
  }

  static Stmt MakeSequence(const Array<Stmt> &statements) {
    if (statements.empty()) return Stmt();
    if (statements.size() == 1) return statements[0];
    return SeqStmt(statements);
  }

  static bool UsesVar(const PrimExpr &expr, const Var &wanted) {
    bool found = false;
    PostOrderVisit(expr, [&](const ObjectRef &node) {
      if (const auto *var = node.as<VarNode>()) {
        found = found || GetRef<Var>(var).same_as(wanted);
      }
    });
    return found;
  }

  static int64_t ReadTripBoundValue(const Any &value) {
    const auto *imm = value.as<IntImmNode>();
    ICHECK(imm != nullptr)
        << "InferZ3PipelineProblem: " << kZ3CudaTripBound
        << " must be a non-negative IntImm";
    ICHECK_GE(imm->value, 0)
        << "InferZ3PipelineProblem: " << kZ3CudaTripBound
        << " must be non-negative";
    return imm->value;
  }

  static std::optional<int64_t> ForTripBound(const ForNode *loop) {
    auto value = loop->annotations.Get(kZ3CudaTripBound);
    if (!value.has_value()) return std::nullopt;
    return ReadTripBoundValue(value.value());
  }

  NativeExtent InferForTrips(const ForNode *loop, arith::Analyzer *analyzer,
                             std::optional<int64_t> explicit_bound) {
    std::optional<NativeExtent> extent =
        TryFiniteExtent(loop->extent, analyzer, std::nullopt,
                        "serial loop trip extent");
    PrimExpr step = loop->step.has_value()
                        ? loop->step.value()
                        : IntImm(loop->loop_var.dtype(), 1);
    const int64_t *constant_step = as_const_int(analyzer->Simplify(step));
    int64_t minimum_step = 1;
    bool exact_step = constant_step != nullptr;
    if (constant_step != nullptr) {
      ICHECK_GT(*constant_step, 0)
          << "InferZ3PipelineProblem: serial loop step must be positive for "
             "finite trip inference";
      minimum_step = *constant_step;
    } else {
      arith::ConstIntBound bound = analyzer->const_int_bound(step);
      ICHECK_GT(bound->min_value, 0)
          << "InferZ3PipelineProblem: symbolic serial loop step requires a "
             "provably positive lower bound or an explicit "
          << kZ3CudaTripBound;
      minimum_step = bound->min_value;
    }

    std::optional<NativeExtent> actual;
    if (extent.has_value()) {
      int64_t upper = extent->upper_bound / minimum_step +
                      (extent->upper_bound % minimum_step != 0);
      actual = extent->dynamic || !exact_step
                   ? NativeExtent::UpperBound(upper)
                   : NativeExtent::Exact(upper);
    }
    if (!explicit_bound.has_value()) {
      ICHECK(actual.has_value())
          << "InferZ3PipelineProblem: serial For requires a finite trip upper "
             "bound; use constraints, assumptions, or "
          << kZ3CudaTripBound
          << " instead of converting an unknown extent to one";
      return actual.value();
    }
    if (!actual.has_value()) {
      return NativeExtent::UpperBound(*explicit_bound);
    }
    if (!actual->dynamic) {
      ICHECK_LE(actual->upper_bound, *explicit_bound)
          << "InferZ3PipelineProblem: static serial trip count exceeds "
          << kZ3CudaTripBound;
      return actual.value();
    }
    return NativeExtent::UpperBound(
        std::min(actual->upper_bound, *explicit_bound));
  }

  NativeCostExpr Leaf(const Stmt &stmt, arith::Analyzer *analyzer,
                      int64_t scale, int vector_depth) {
    NativeTaskFeatures features =
        TaskFeatureExtractor::Extract(stmt, scale, vector_depth, analyzer);
    return NativeCostExpr::Leaf(features.ToFFI(),
                                features.structural_resource_mask);
  }

  NativeCostExpr ConditionCost(const PrimExpr &condition,
                               arith::Analyzer *analyzer, int64_t scale,
                               int vector_depth) {
    NativeTaskFeatures features = TaskFeatureExtractor::Extract(
        Evaluate(condition), scale, vector_depth, analyzer);
    if (features.Empty()) return NativeCostExpr::Serial({});
    return NativeCostExpr::Leaf(features.ToFFI(),
                                features.structural_resource_mask);
  }

  std::vector<char> FindInvariantBinds(const Array<Stmt> &statements,
                                       const Var &loop_var) {
    ScalarVarSet local_definitions;
    for (const Stmt &stmt : statements) {
      PostOrderVisit(stmt, [&](const ObjectRef &node) {
        if (const auto *bind = node.as<BindNode>()) {
          local_definitions.insert(bind->var);
        }
      });
    }
    ScalarVarSet invariant_definitions;
    std::vector<char> result(statements.size(), 0);
    for (size_t i = 0; i < statements.size(); ++i) {
      const auto *bind = statements[i].as<BindNode>();
      if (bind == nullptr || !HasCloneableScalarBindType(bind) ||
          !IsProvenDuplicatableScalarValue(bind->value) ||
          SideEffect(bind->value) != CallEffectKind::kPure ||
          UsesVar(bind->value, loop_var)) {
        continue;
      }
      bool depends_on_variant_local = false;
      for (const Var &local : local_definitions) {
        if (!invariant_definitions.count(local) &&
            UsesVar(bind->value, local)) {
          depends_on_variant_local = true;
          break;
        }
      }
      if (depends_on_variant_local) continue;
      result[i] = 1;
      invariant_definitions.insert(bind->var);
    }
    return result;
  }

  NativeCostExpr ExtractSerialFor(const ForNode *loop,
                                  arith::Analyzer *analyzer, int64_t scale,
                                  int vector_depth,
                                  std::optional<int64_t> explicit_bound) {
    NativeExtent trips = InferForTrips(loop, analyzer, explicit_bound);
    std::unique_ptr<arith::Analyzer> context = analyzer->Clone();
    context->Bind(loop->loop_var,
                  Range::FromMinExtent(loop->min, loop->extent));

    Array<Stmt> statements;
    AppendSequence(loop->body, &statements);
    std::vector<char> hoist = FindInvariantBinds(statements, loop->loop_var);
    Array<Stmt> remaining;
    std::vector<NativeCostExpr> prefix;
    for (size_t i = 0; i < statements.size(); ++i) {
      if (!hoist[i]) {
        remaining.push_back(statements[i]);
        continue;
      }
      prefix.push_back(ExtractImpl(statements[i], context.get(), scale,
                                   vector_depth));
      const auto *bind = statements[i].as<BindNode>();
      ICHECK(bind != nullptr);
      context->Bind(bind->var, bind->value);
    }
    NativeCostExpr repeated = NativeCostExpr::Repeat(
        trips, ExtractImpl(MakeSequence(remaining), context.get(), scale,
                           vector_depth));
    if (prefix.empty()) return repeated;
    prefix.push_back(std::move(repeated));
    return NativeCostExpr::Serial(std::move(prefix));
  }

  NativeCostExpr ExtractFor(const ForNode *loop, arith::Analyzer *analyzer,
                            int64_t scale, int vector_depth,
                            std::optional<int64_t> explicit_bound) {
    if (loop->kind == ForKind::kSerial || loop->kind == ForKind::kUnrolled) {
      return ExtractSerialFor(loop, analyzer, scale, vector_depth,
                              explicit_bound);
    }
    ICHECK(!explicit_bound.has_value())
        << "InferZ3PipelineProblem: " << kZ3CudaTripBound
        << " is only valid on serial/unrolled For or While cost recursion";
    int64_t extent = StaticNonnegative(loop->extent, "task loop extent");
    int64_t nested_scale =
        CheckedMul(scale, extent, "task loop trip count");
    bool parallel = loop->kind == ForKind::kParallel ||
                    loop->kind == ForKind::kVectorized ||
                    loop->kind == ForKind::kThreadBinding;
    std::unique_ptr<arith::Analyzer> context = analyzer->Clone();
    context->Bind(loop->loop_var,
                  Range::FromMinExtent(loop->min, loop->extent));
    return ExtractImpl(loop->body, context.get(), nested_scale,
                       vector_depth + (parallel ? 1 : 0));
  }

  NativeCostExpr ExtractWhile(const WhileNode *loop,
                              arith::Analyzer *analyzer, int64_t scale,
                              int vector_depth,
                              std::optional<int64_t> explicit_bound) {
    NativeCostExpr condition =
        ConditionCost(loop->condition, analyzer, scale, vector_depth);
    std::unique_ptr<arith::Analyzer> body_context = analyzer->Clone();
    NativeCostExpr body;
    {
      With<arith::ConstraintContext> scope(body_context.get(), loop->condition);
      body = ExtractImpl(loop->body, body_context.get(), scale, vector_depth);
    }
    NativeCostExpr iteration =
        NativeCostExpr::Serial({std::move(condition), std::move(body)});

    const int64_t *constant_condition =
        as_const_int(analyzer->Simplify(loop->condition));
    if (constant_condition != nullptr && *constant_condition == 0) {
      return NativeCostExpr::Repeat(NativeExtent::Exact(0), iteration);
    }
    if (explicit_bound.has_value()) {
      return NativeCostExpr::Repeat(
          NativeExtent::UpperBound(*explicit_bound), iteration);
    }
    ICHECK(allow_unbounded_while_)
        << "InferZ3PipelineProblem: While requires a finite trip upper bound; "
           "provide "
        << kZ3CudaTripBound
        << " or both T.Task latency and II overrides. Explicit costs do not "
           "waive dependency/state footprints";
    // Only structural resources are consumed from this placeholder tree. The
    // caller replaces both latency and II with task-level overrides.
    iteration.contains_unbounded_while = true;
    return iteration;
  }

  NativeCostExpr ExtractImpl(const Stmt &stmt, arith::Analyzer *analyzer,
                             int64_t scale, int vector_depth) {
    if (!stmt.defined()) return NativeCostExpr::Serial({});
    if (const auto *seq = stmt.as<SeqStmtNode>()) {
      std::unique_ptr<arith::Analyzer> context = analyzer->Clone();
      std::vector<NativeCostExpr> children;
      children.reserve(seq->seq.size());
      for (const Stmt &child : seq->seq) {
        children.push_back(
            ExtractImpl(child, context.get(), scale, vector_depth));
        const auto *bind = child.as<BindNode>();
        if (bind != nullptr &&
            SideEffect(bind->value) == CallEffectKind::kPure) {
          context->Bind(bind->var, bind->value);
        }
      }
      return NativeCostExpr::Serial(std::move(children));
    }
    if (const auto *attr = stmt.as<AttrStmtNode>()) {
      if (attr->attr_key == "tl.assume" ||
          attr->attr_key == "tl.assume_requires_runtime_check") {
        const auto *condition = attr->node.as<PrimExprNode>();
        ICHECK(condition != nullptr)
            << "InferZ3PipelineProblem: assumption node must be a PrimExpr";
        std::unique_ptr<arith::Analyzer> context = analyzer->Clone();
        With<arith::ConstraintContext> scope(
            context.get(), GetRef<PrimExpr>(condition), true);
        return ExtractImpl(attr->body, context.get(), scale, vector_depth);
      }
      if (attr->attr_key == kZ3CudaTripBound) {
        // Accept the conventional node=bound/value=1 spelling as well as a
        // direct value=bound attribute; For annotations always carry the bound
        // directly in their annotation value.
        const Any &bound_value = attr->node.as<IntImmNode>() != nullptr
                                     ? attr->node
                                     : attr->value;
        int64_t bound = ReadTripBoundValue(bound_value);
        if (const auto *loop = attr->body.as<ForNode>()) {
          return ExtractFor(loop, analyzer, scale, vector_depth, bound);
        }
        if (const auto *loop = attr->body.as<WhileNode>()) {
          return ExtractWhile(loop, analyzer, scale, vector_depth, bound);
        }
        LOG(FATAL) << "InferZ3PipelineProblem: " << kZ3CudaTripBound
                   << " must directly wrap a For or While statement";
      }
      return ExtractImpl(attr->body, analyzer, scale, vector_depth);
    }
    if (const auto *loop = stmt.as<ForNode>()) {
      return ExtractFor(loop, analyzer, scale, vector_depth,
                        ForTripBound(loop));
    }
    if (const auto *loop = stmt.as<WhileNode>()) {
      return ExtractWhile(loop, analyzer, scale, vector_depth, std::nullopt);
    }
    if (const auto *branch = stmt.as<IfThenElseNode>()) {
      NativeCostExpr condition = ConditionCost(branch->condition, analyzer,
                                               scale, vector_depth);
      std::unique_ptr<arith::Analyzer> then_context = analyzer->Clone();
      NativeCostExpr then_expr;
      {
        With<arith::ConstraintContext> scope(then_context.get(),
                                             branch->condition);
        then_expr = ExtractImpl(branch->then_case, then_context.get(), scale,
                                vector_depth);
      }
      NativeCostExpr else_expr = NativeCostExpr::Serial({});
      if (branch->else_case.defined()) {
        std::unique_ptr<arith::Analyzer> else_context = analyzer->Clone();
        With<arith::ConstraintContext> scope(else_context.get(),
                                             Not(branch->condition));
        else_expr = ExtractImpl(branch->else_case.value(), else_context.get(),
                                scale, vector_depth);
      }
      return NativeCostExpr::Serial(
          {std::move(condition),
           NativeCostExpr::Branch(std::move(then_expr),
                                  std::move(else_expr))});
    }
    if (const auto *block = stmt.as<SBlockNode>()) {
      std::vector<NativeCostExpr> children;
      if (block->init.defined()) {
        children.push_back(
            ExtractImpl(block->init.value(), analyzer, scale, vector_depth));
      }
      children.push_back(
          ExtractImpl(block->body, analyzer, scale, vector_depth));
      return NativeCostExpr::Serial(std::move(children));
    }
    if (const auto *realize = stmt.as<SBlockRealizeNode>()) {
      return ExtractImpl(realize->block, analyzer, scale, vector_depth);
    }
    if (const auto *evaluate = stmt.as<EvaluateNode>()) {
      if (const auto *call = evaluate->value.as<CallNode>()) {
        if (auto tile_op = ParseOperator(GetRef<Call>(call));
            tile_op.defined()) {
          if (const auto *parallel = tile_op.as<ParallelOpNode>()) {
            return ExtractImpl(parallel->GetRoot(), analyzer, scale,
                               vector_depth);
          }
        }
      }
    }
    return Leaf(stmt, analyzer, scale, vector_depth);
  }

  bool allow_unbounded_while_{false};
};

int CostReliabilityRank(const std::string &reliability) {
  if (reliability == "measured") return 0;
  if (reliability == "estimated") return 1;
  if (reliability == "upper_bound") return 2;
  ICHECK_EQ(reliability, "unknown")
      << "InferZ3PipelineProblem: invalid cost reliability " << reliability;
  return 3;
}

std::string WorstCostReliability(const std::string &left,
                                 const std::string &right) {
  return CostReliabilityRank(left) >= CostReliabilityRank(right) ? left : right;
}

/* Recover the reliability carried by the composed native expression itself.
 * The leaf evaluator may additionally report calibrated geometry fallbacks;
 * those remain visible in fallback_used and are handled at the FFI boundary. */
std::string NativeComposedReliability(const NativeCostExpr &expr) {
  if (expr.IsLeaf()) {
    auto reliability = expr.features.Get("reliability");
    ICHECK(reliability.has_value());
    return reliability.value().cast<String>().operator std::string();
  }
  std::string result = expr.kind == NativeCostExpr::Kind::kRepeat
                           ? expr.trips.reliability
                           : "measured";
  for (const NativeCostExpr &child : expr.children) {
    result = WorstCostReliability(result, NativeComposedReliability(child));
  }
  return result;
}

bool OnlyCalibratedGeometryFallbacks(const Array<String> &fallbacks) {
  if (fallbacks.empty()) return false;
  for (const String &fallback : fallbacks) {
    if (std::string(fallback).rfind("memory_geometry[", 0) != 0) return false;
  }
  return true;
}

struct InferredBuffers {
  Array<Var> vars;
  Array<Integer> sizes;
};

class Z3TaskAnalyzer : public StmtExprMutator {
public:
  static Stmt Substitute(const PrimFunc &func) {
    Z3TaskAnalyzer analyzer;
    for (const auto &[_, buffer] : func->buffer_map) {
      analyzer.buffer_data_to_buffer_.Set(buffer->data, buffer);
    }
    return analyzer.VisitStmt(func->body);
  }

  static bool NeedsInference(const ForNode *loop) {
    static const std::vector<const char *> keys = {
        kZ3Latencies, kZ3IIs, kZ3ResourceFlags, kZ3DataDeps,
        kZ3ResourceDeps, kZ3PipeOrderDeps, kZ3OwnerExclusionDeps,
        kZ3StageOrderDeps, kZ3BufferSizes, kZ3MemoryGroups, kZ3BufferVars};
    if (!loop->annotations.count(kZ3Pipeline)) return false;
    for (const char *key : keys) {
      if (!HasAnnotation(loop->annotations, key)) return true;
    }
    return false;
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
    if (!NeedsInference(loop)) {
      if (loop->annotations.count(kZ3Pipeline)) return GetRef<For>(loop);
      return StmtExprMutator::VisitStmt_(loop);
    }
    return InferLoop(loop);
  }

  InferredBuffers InferBuffers(const ScheduledTaskAnalysis &analysis,
                               bool compute_sizes,
                               const Map<String, Any> &annotations) const {
    InferredBuffers result;
    std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> seen;
    std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> explicitly_requested;
    Z3BufferVersionCountMap requests = ReadZ3BufferVersionCounts(
        annotations, kZ3BufferVersionRequests, "InferZ3PipelineProblem");
    Z3BufferVersionModeMap modes = ReadZ3BufferVersionModes(
        annotations, kZ3BufferVersionModeRequests, "InferZ3PipelineProblem");
    for (const auto &[storage, _] : requests) {
      if (!IsZ3BareOneRequest(storage, requests, modes)) {
        explicitly_requested.insert(storage);
      }
    }
    for (const auto &[storage, _] : modes) explicitly_requested.insert(storage);
    std::unordered_map<Var, std::vector<Buffer>, ObjectPtrHash, ObjectPtrEqual>
        views;
    auto record = [&](const Array<BufferRegion> &regions, bool writes) {
      for (const BufferRegion &region : regions) {
        const Var &storage = region->buffer->data;
        views[storage].push_back(region->buffer);
        if ((!writes || !IsSharedBuffer(region->buffer)) &&
            !explicitly_requested.count(storage)) {
          continue;
        }
        if (seen.insert(storage).second) {
          result.vars.push_back(storage);
        }
      }
    };
    for (size_t i = 0; i < analysis.scheduled_stmts.size(); ++i) {
      record(analysis.reads[i], false);
      record(analysis.writes[i], true);
    }
    if (!compute_sizes) return result;
    for (const Var &var : result.vars) {
      int64_t size = 0;
      for (const Buffer &buffer : views[var]) {
        int64_t elements = 1;
        for (const PrimExpr &dim : buffer->shape) {
          elements = CheckedMul(elements,
                                StaticNonnegative(dim, "buffer shape extent"),
                                "buffer element count");
        }
        size = std::max(size, ElementsBytesForBuffer(elements, buffer));
      }
      result.sizes.push_back(Integer(size));
    }
    return result;
  }

  static int64_t ElementsBytesForBuffer(int64_t elements, const Buffer &buffer) {
    int64_t bits = CheckedMul(elements, DTypeBits(buffer->dtype),
                              "buffer allocation bits");
    return (CheckedAdd(bits, 7, "buffer allocation rounding")) / 8;
  }

  Array<Var> GetExplicitBufferVars(const Map<String, Any> &annotations) const {
    auto value = annotations.Get(kZ3BufferVars);
    return value.has_value() ? Downcast<Array<Var>>(value.value()) : Array<Var>();
  }

  Map<String, int64_t> MemoryCapacities(const Map<String, Any> &annotations) {
    String hardware = GetRequiredPath(annotations, kZ3Hardware);
    auto function = ffi::Function::GetGlobal("tl.cuda.z3_memory_group_capacities");
    ICHECK(function.has_value())
        << "InferZ3PipelineProblem: import tilelang.cuda.task_cost to register "
           "tl.cuda.z3_memory_group_capacities";
    Map<String, Any> raw =
        function.value()(hardware).cast<Map<String, Any>>();
    Map<String, int64_t> result;
    for (const auto &[key, value] : raw) {
      int64_t capacity = value.cast<int64_t>();
      ICHECK_GT(capacity, 0) << "InferZ3PipelineProblem: hardware memory "
                             << "capacity for " << key << " must be positive";
      result.Set(key, capacity);
    }
    return result;
  }

  Buffer BufferForVar(const Var &var, const ScheduledTaskAnalysis &analysis) const {
    auto found = buffer_data_to_buffer_.find(var);
    if (found != buffer_data_to_buffer_.end()) return (*found).second;
    for (size_t i = 0; i < analysis.scheduled_stmts.size(); ++i) {
      for (const auto *regions : {&analysis.reads[i], &analysis.writes[i]}) {
        for (const BufferRegion &region : *regions) {
          if (region->buffer->data.same_as(var)) return region->buffer;
        }
      }
    }
    LOG(FATAL) << "InferZ3PipelineProblem: cannot resolve explicit buffer Var "
               << var << " to a Buffer for memory-group inference";
    return Buffer();
  }

  bool OnlyLoopVar(const PrimExpr &expr, const Var &loop_var) const {
    bool supported = true;
    PostOrderVisit(expr, [&](const ObjectRef &node) {
      if (const auto *var = node.as<VarNode>()) {
        supported = supported && GetRef<Var>(var).same_as(loop_var);
      }
    });
    return supported;
  }

  bool DistinctIterationsDisjoint(const BufferRegion &left,
                                  const BufferRegion &right,
                                  const ForNode *loop) const {
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
    Map<Var, PrimExpr> shift;
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
          *b_extent < 0 || !OnlyLoopVar(a->min, loop->loop_var) ||
          !OnlyLoopVar(b->min, loop->loop_var)) {
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
      if (!OnlyLoopVar(slope, delta)) return false;
      Map<Var, PrimExpr> delta_zero;
      delta_zero.Set(delta, IntImm(delta.dtype(), 0));
      Map<Var, PrimExpr> delta_one;
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

  bool RegionsConflict(const BufferRegion &left, const BufferRegion &right,
                       const ForNode *loop, bool cross_iteration,
                       const Map<Var, arith::IntSet> &thread_domains) const {
    if (cross_iteration) {
      return !DistinctIterationsDisjoint(left, right, loop);
    }
    return PipelineRegionsMayConflict(left, right, thread_domains);
  }

  void AddDataDependency(std::vector<Array<Integer>> *dependencies,
                         int64_t source, int64_t target, int64_t distance,
                         int64_t latency) const {
    ICHECK_GE(latency, 0);
    for (Array<Integer> &existing : *dependencies) {
      if (existing[0]->value == source && existing[1]->value == target &&
          existing[2]->value == distance) {
        if (existing[3]->value < latency) existing.Set(3, Integer(latency));
        return;
      }
    }
    dependencies->push_back(
        Array<Integer>{Integer(source), Integer(target), Integer(distance),
                       Integer(latency)});
  }

  std::vector<Array<Integer>> InferDataDependencies(
      const ForNode *loop, const ScheduledTaskAnalysis &analysis,
      const Array<Integer> &latencies, const Array<Var> &buffer_vars) const {
    size_t n = analysis.scheduled_stmts.size();
    Map<Var, arith::IntSet> thread_domains =
        CollectPipelineThreadDomains(loop->body);
    std::vector<Array<Integer>> dependencies;
    auto buffer_distance = [&](const Var &storage) -> int64_t {
      for (size_t i = 0; i < buffer_vars.size(); ++i) {
        if (buffer_vars[i].same_as(storage)) return -static_cast<int64_t>(i) - 1;
      }
      return 1;
    };
    auto consider = [&](size_t source, size_t target,
                        const Array<BufferRegion> &left,
                        const Array<BufferRegion> &right) {
      for (bool cross : {false, true}) {
        if (!cross && source >= target) continue;
        for (const BufferRegion &lhs : left) {
          for (const BufferRegion &rhs : right) {
            if (!RegionsConflict(lhs, rhs, loop, cross, thread_domains)) {
              continue;
            }
            int64_t distance = cross ? buffer_distance(lhs->buffer->data) : 0;
            AddDataDependency(&dependencies, source, target, distance,
                              latencies[source]->value);
          }
        }
      }
    };
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < n; ++j) {
        consider(i, j, analysis.writes[i], analysis.reads[j]);  // RAW
        consider(i, j, analysis.writes[i], analysis.writes[j]); // WAW
        consider(i, j, analysis.reads[i], analysis.writes[j]);  // WAR
      }
    }
    // Scalar Bind uses are dependencies even when the expression has no
    // BufferRegion. The reverse distance-one edge protects the definition in
    // the next iteration from a longer-latency consumer in this iteration.
    for (const ScalarUseDefEdge &edge : analysis.scalar.edges) {
      ICHECK_LT(edge.definition, edge.use)
          << "InferZ3PipelineProblem: a flat scalar Bind must precede every "
             "use in the same enclosing statement stream";
      int source = analysis.scheduled_for_flat[edge.definition];
      int target = analysis.scheduled_for_flat[edge.use];
      if (source < 0 || target < 0) continue;
      AddDataDependency(&dependencies, source, target, 0,
                        latencies[source]->value);
      AddDataDependency(&dependencies, target, source, 1,
                        latencies[target]->value);
    }
    return dependencies;
  }

  Array<Array<Integer>> ToNestedArrays(
      const std::vector<Array<Integer>> &values) const {
    Array<Array<Integer>> result;
    for (const Array<Integer> &value : values) result.push_back(value);
    return result;
  }

  Array<Array<Integer>> InferResourcePairs(const Array<Integer> &masks) const {
    Array<Array<Integer>> result;
    for (size_t i = 0; i < masks.size(); ++i) {
      for (size_t j = i + 1; j < masks.size(); ++j) {
        if ((masks[i]->value & masks[j]->value) != 0) {
          result.push_back(Array<Integer>{Integer(i), Integer(j)});
        }
      }
    }
    return result;
  }

  void ValidateArraySize(const Array<Integer> &values, size_t n,
                         const char *key) const {
    ICHECK_EQ(values.size(), n)
        << "InferZ3PipelineProblem: explicit " << key << " length "
        << values.size() << " does not match scheduled task count " << n;
  }

  Array<Array<Integer>> ValidateAsyncSameStage(
      const ForNode *loop, const ScheduledTaskAnalysis &analysis,
      const Map<String, Any> &annotations) const {
    Array<Array<Integer>> empty;
    auto flags_value = annotations.Get(kPipelineAsyncProducers);
    auto groups_value = annotations.Get(kPipelineAsyncProducerGroups);
    if (!flags_value.has_value() && !groups_value.has_value()) return empty;
    ICHECK(flags_value.has_value())
        << "InferZ3PipelineProblem: async producer groups require async flags";
    Array<Integer> flags = Downcast<Array<Integer>>(flags_value.value());
    Array<Integer> groups = groups_value.has_value()
                                ? Downcast<Array<Integer>>(groups_value.value())
                                : Array<Integer>();
    size_t n = analysis.scheduled_stmts.size();
    bool flat = flags.size() == analysis.flat_stmts.size();
    ICHECK(flat || flags.size() == n)
        << "InferZ3PipelineProblem: async producer flags must be indexed by "
           "scheduled tasks or the flattened statement stream";
    if (groups_value.has_value()) ICHECK_EQ(groups.size(), flags.size());
    auto at = [&](const Array<Integer> &values, size_t task) -> int64_t {
      return values[flat ? analysis.scheduled_indices[task] : task]->value;
    };
    Map<Var, arith::IntSet> domains = CollectPipelineThreadDomains(loop->body);
    std::unordered_map<int64_t, std::vector<size_t>> expected;
    Array<Array<Integer>> pairs;
    for (size_t producer = 0; producer < n; ++producer) {
      if (at(flags, producer) == 0) continue;
      std::vector<size_t> consumers;
      for (size_t consumer = producer + 1; consumer < n; ++consumer) {
        bool conflict = false;
        for (const BufferRegion &write : analysis.writes[producer]) {
          for (const BufferRegion &read : analysis.reads[consumer]) {
            conflict = conflict || RegionsConflict(write, read, loop, false, domains);
          }
        }
        if (!conflict) continue;
        consumers.push_back(consumer);
        pairs.push_back(Array<Integer>{Integer(producer), Integer(consumer)});
      }
      if (groups_value.has_value()) {
        int64_t group = at(groups, producer);
        ICHECK_GE(group, 0) << "InferZ3PipelineProblem: async producer has a "
                               "negative group id";
        auto [it, inserted] = expected.emplace(group, consumers);
        ICHECK(inserted || it->second == consumers)
            << "InferZ3PipelineProblem: async producers in one group must have "
               "the same direct consumer set";
        ICHECK(!consumers.empty())
            << "InferZ3PipelineProblem: async producer group has no direct "
               "consumer; its commit/wait owner is unknown";
      }
    }
    if (pairs.empty()) return pairs;
    bool manual = GetBool(annotations, kZ3ManualSchedule, false);
    if (!manual) return pairs;
    Array<Integer> stages = GetIntegerArray(annotations, kZ3ManualStages);
    ValidateArraySize(stages, n, kZ3ManualStages);
    for (const Array<Integer> &pair : pairs) {
      ICHECK_LE(stages[pair[0]->value]->value, stages[pair[1]->value]->value)
          << "InferZ3PipelineProblem: async producer " << pair[0]->value
          << " must not occur after consumer " << pair[1]->value
          << "; producer and consumer may share a stage or use ordered stages";
    }
    return pairs;
  }

  Array<Array<Integer>> InferScalarStageOrder(
      const ScheduledTaskAnalysis &analysis) const {
    Array<Array<Integer>> result;
    auto add_pair = [&](int64_t left, int64_t right) {
      for (const auto &pair : result) {
        if (pair[0]->value == left && pair[1]->value == right) return;
      }
      result.push_back(Array<Integer>{Integer(left), Integer(right)});
    };
    for (const ScalarUseDefEdge &edge : analysis.scalar.edges) {
      int source = analysis.scheduled_for_flat[edge.definition];
      int target = analysis.scheduled_for_flat[edge.use];
      if (source < 0 || target < 0 ||
          analysis.scalar.cloneable_binds[edge.definition]) {
        continue;
      }
      // Two directed inequalities express stage equality in the solver schema.
      add_pair(source, target);
      add_pair(target, source);
    }
    return result;
  }

  Stmt InferLoop(const ForNode *loop) {
    const Map<String, Any> &annotations = loop->annotations;
    ICHECK(GetBool(annotations, kZ3Pipeline, false))
        << "InferZ3PipelineProblem: tl_z3_pipeline must be True when present";
    PostOrderVisit(loop->body, [&](const ObjectRef &node) {
      if (const auto *attr = node.as<AttrStmtNode>()) {
        ICHECK(attr->attr_key != kZ3CudaStageAttr)
            << "InferZ3PipelineProblem: unnormalized T.Stage marker; run "
               "NormalizeZ3ScheduleUnits before task inference";
      }
    });
    bool need_latencies = !HasAnnotation(annotations, kZ3Latencies);
    bool need_iis = !HasAnnotation(annotations, kZ3IIs);
    bool need_resources = !HasAnnotation(annotations, kZ3ResourceFlags);
    bool need_data = !HasAnnotation(annotations, kZ3DataDeps);
    bool need_buffer_sizes = !HasAnnotation(annotations, kZ3BufferSizes);
    bool need_buffer_vars = !HasAnnotation(annotations, kZ3BufferVars);
    bool need_groups = !HasAnnotation(annotations, kZ3MemoryGroups);
    ICHECK(need_buffer_sizes == need_buffer_vars)
        << "InferZ3PipelineProblem: buffer_sizes and buffer_vars must either "
           "both be explicit (in one user-defined order) or both be inferred; "
           "a partial override has no safe ordering contract";

    if (need_buffer_vars) {
      for (const Array<Integer> &dep :
           GetNestedIntegerArrays(annotations, kZ3DataDeps)) {
        ICHECK_EQ(dep.size(), 4U)
            << "InferZ3PipelineProblem: data_deps entries must have four fields";
        ICHECK(dep[2]->value >= 0)
            << "InferZ3PipelineProblem: explicit data_deps reference an "
               "automatic buffer id while buffer_vars/buffer_sizes are missing; "
               "provide their ordered values explicitly";
      }
      for (const Array<Integer> &group :
           GetNestedIntegerArrays(annotations, kZ3MemoryGroups)) {
        ICHECK(group.empty())
            << "InferZ3PipelineProblem: explicit memory_groups have buffer "
               "indices but no explicit buffer_vars/buffer_sizes order";
      }
    }

    bool strict_regions = need_data || need_buffer_vars || need_groups;
    ScheduledTaskAnalysis analysis = AnalyzeScheduledTasks(
        loop->body, buffer_data_to_buffer_, strict_regions);
    size_t n = analysis.scheduled_stmts.size();
    ICHECK_GT(n, 0) << "InferZ3PipelineProblem: no scheduled tasks found";
    if (auto domains = annotations.Get(kZ3ExecutionDomains); domains.has_value()) {
      ICHECK_EQ(Downcast<Array<String>>(domains.value()).size(), n)
          << "InferZ3PipelineProblem: execution-domain count does not match "
             "the scheduled task count";
    }

    Array<Integer> latencies = GetIntegerArray(annotations, kZ3Latencies);
    Array<Integer> iis = GetIntegerArray(annotations, kZ3IIs);
    Array<Integer> resource_flags =
        GetIntegerArray(annotations, kZ3ResourceFlags);
    if (!need_latencies) ValidateArraySize(latencies, n, kZ3Latencies);
    if (!need_iis) ValidateArraySize(iis, n, kZ3IIs);
    if (!need_resources) ValidateArraySize(resource_flags, n, kZ3ResourceFlags);

    Map<String, Any> output = annotations;
    if (need_latencies || need_iis || need_resources) {
      std::vector<TaskCostOverride> overrides;
      overrides.reserve(n);
      bool any_model_required = false;
      for (size_t i = 0; i < n; ++i) {
        TaskCostOverride override =
            GetTaskCostOverride(analysis.scheduled_stmts[i]);
        if (!need_latencies && override.latency >= 0) {
          ICHECK_EQ(latencies[i]->value, override.latency)
              << "InferZ3PipelineProblem: T.Task latency override conflicts "
                 "with the loop-level latency array";
        }
        if (!need_iis && override.ii >= 0) {
          ICHECK_EQ(iis[i]->value, override.ii)
              << "InferZ3PipelineProblem: T.Task ii override conflicts with "
                 "the loop-level II array";
        }
        bool model_required =
            (need_latencies && override.latency < 0) ||
            (need_iis && override.ii < 0) || need_resources;
        any_model_required = any_model_required || model_required;
        overrides.push_back(override);
      }

      String hardware;
      String calibration;
      bool require_measured = GetBool(annotations, kZ3RequireMeasured, false);
      Optional<ffi::Function> estimator;
      Optional<ffi::Function> composed_estimator;
      if (any_model_required) {
        hardware = GetRequiredPath(annotations, kZ3Hardware);
        calibration = GetRequiredPath(annotations, kZ3Calibration);
        estimator = ffi::Function::GetGlobal("tl.cuda.estimate_task");
        ICHECK(estimator.has_value())
            << "InferZ3PipelineProblem: import tilelang.cuda.task_cost to "
               "register tl.cuda.estimate_task";
      }

      Array<String> reliabilities;
      Array<Array<String>> fallbacks;
      for (size_t i = 0; i < n; ++i) {
        const TaskCostOverride &override = overrides[i];
        bool model_required =
            (need_latencies && override.latency < 0) ||
            (need_iis && override.ii < 0) || need_resources;
        int64_t model_latency = 0;
        int64_t model_ii = 1;
        int64_t model_mask = 0;
        String reliability = "explicit";
        Array<String> fallback;
        if (model_required) {
          bool allow_unbounded_while =
              override.latency >= 0 && override.ii >= 0;
          NativeCostExpr native = NativeCostExprExtractor::Extract(
              analysis.scheduled_stmts[i], allow_unbounded_while);
          Map<String, Any> result;
          if (native.IsLeaf()) {
            result = estimator.value()(native.features, hardware, calibration,
                                       require_measured)
                         .cast<Map<String, Any>>();
          } else {
            if (!composed_estimator.has_value()) {
              composed_estimator =
                  ffi::Function::GetGlobal("tl.cuda.estimate_composed_task");
              ICHECK(composed_estimator.has_value())
                  << "InferZ3PipelineProblem: import tilelang.cuda.task_cost "
                     "to register tl.cuda.estimate_composed_task";
            }
            result = composed_estimator.value()(native.ToFFI(), hardware,
                                                calibration, require_measured)
                         .cast<Map<String, Any>>();
          }
          auto get_int = [&](const char *key) -> int64_t {
            auto value = result.Get(key);
            ICHECK(value.has_value()) << "Task estimator omitted " << key;
            return value.value().cast<int64_t>();
          };
          model_latency = get_int("latency");
          model_ii = get_int("ii");
          model_mask = get_int("resource_flags") |
                       native.structural_resource_mask;
          auto model_reliability = result.Get("reliability");
          ICHECK(model_reliability.has_value());
          reliability = model_reliability.value().cast<String>();
          auto model_fallback = result.Get("fallback_used");
          ICHECK(model_fallback.has_value());
          fallback = model_fallback.value().cast<Array<String>>();
          if (reliability == "unknown" &&
              NativeComposedReliability(native) == "upper_bound" &&
              OnlyCalibratedGeometryFallbacks(fallback)) {
            // A finite native Extent remains an upper bound when the only
            // additional evaluator fallback is the explicitly calibrated
            // unknown-geometry bandwidth path. Genuine unknown operations,
            // missing paths, and unmeasured calibration still win.
            reliability = "upper_bound";
          }
        }

        int64_t latency = need_latencies
                              ? (override.latency >= 0 ? override.latency
                                                       : model_latency)
                              : latencies[i]->value;
        int64_t ii = need_iis
                         ? (override.ii >= 0 ? override.ii : model_ii)
                         : iis[i]->value;
        int64_t mask = need_resources ? model_mask : resource_flags[i]->value;
        ICHECK(latency >= 0 && ii > 0 && latency >= ii)
            << "InferZ3PipelineProblem: invalid task cost latency=" << latency
            << ", II=" << ii;
        ICHECK(mask >= 0 && mask <= kAllResources)
            << "InferZ3PipelineProblem: resource mask exceeds seven bits";
        if (need_latencies) latencies.push_back(Integer(latency));
        if (need_iis) iis.push_back(Integer(ii));
        if (need_resources) resource_flags.push_back(Integer(mask));
        reliabilities.push_back(reliability);
        fallbacks.push_back(std::move(fallback));
      }
      if (!HasAnnotation(annotations, kZ3CostReliability)) {
        output.Set(kZ3CostReliability, reliabilities);
      }
      if (!HasAnnotation(annotations, kZ3CostFallbacks)) {
        output.Set(kZ3CostFallbacks, fallbacks);
      }
    }

    InferredBuffers inferred;
    bool buffers_needed = need_buffer_vars || need_data || need_groups;
    if (buffers_needed) inferred = InferBuffers(analysis, need_buffer_vars, annotations);
    Array<Var> buffer_vars = need_buffer_vars
                                 ? inferred.vars
                                 : GetExplicitBufferVars(annotations);
    Array<Integer> buffer_sizes = need_buffer_sizes
                                      ? inferred.sizes
                                      : GetIntegerArray(annotations, kZ3BufferSizes);
    ICHECK_EQ(buffer_vars.size(), buffer_sizes.size());

    if (need_latencies || need_iis || need_resources) {
      ValidateArraySize(latencies, n, kZ3Latencies);
      ValidateArraySize(iis, n, kZ3IIs);
      ValidateArraySize(resource_flags, n, kZ3ResourceFlags);
    }
    if (need_latencies) output.Set(kZ3Latencies, latencies);
    if (need_iis) output.Set(kZ3IIs, iis);
    if (need_resources) output.Set(kZ3ResourceFlags, resource_flags);
    if (need_data) {
      output.Set(kZ3DataDeps,
                 ToNestedArrays(InferDataDependencies(loop, analysis, latencies,
                                                      buffer_vars)));
    }
    if (!HasAnnotation(annotations, kZ3ResourceDeps)) {
      output.Set(kZ3ResourceDeps, InferResourcePairs(resource_flags));
    }
    if (!HasAnnotation(annotations, kZ3PipeOrderDeps)) {
      bool manual = GetBool(annotations, kZ3ManualSchedule, false);
      output.Set(kZ3PipeOrderDeps,
                 manual ? InferResourcePairs(resource_flags) : Array<Array<Integer>>());
    }
    if (!HasAnnotation(annotations, kZ3OwnerExclusionDeps)) {
      output.Set(kZ3OwnerExclusionDeps, Array<Array<Integer>>());
    }
    if (!HasAnnotation(annotations, kZ3StageOrderDeps)) {
      output.Set(kZ3StageOrderDeps, InferScalarStageOrder(analysis));
    }
    if (need_buffer_vars) output.Set(kZ3BufferVars, buffer_vars);
    if (need_buffer_sizes) output.Set(kZ3BufferSizes, buffer_sizes);
    if (need_groups) {
      Array<Array<Integer>> groups;
      std::vector<int64_t> shared_indices;
      for (size_t i = 0; i < buffer_vars.size(); ++i) {
        Buffer buffer = BufferForVar(buffer_vars[i], analysis);
        if (IsSharedBuffer(buffer)) shared_indices.push_back(i);
      }
      if (!shared_indices.empty()) {
        Map<String, int64_t> capacities = MemoryCapacities(annotations);
        auto capacity = capacities.Get("shared");
        ICHECK(capacity.has_value())
            << "InferZ3PipelineProblem: hardware spec does not provide an "
               "explicit shared memory-group capacity";
        Array<Integer> group{Integer(capacity.value())};
        for (int64_t index : shared_indices) group.push_back(Integer(index));
        groups.push_back(std::move(group));
      }
      output.Set(kZ3MemoryGroups, groups);
    }

    Array<Array<Integer>> async_pairs =
        ValidateAsyncSameStage(loop, analysis, annotations);
    if (!async_pairs.empty() && !HasAnnotation(annotations, kZ3AsyncSameStage)) {
      output.Set(kZ3AsyncSameStage, async_pairs);
    }

    return For(loop->loop_var, loop->min, loop->extent, loop->kind, loop->body,
               loop->thread_binding, output, loop->step, loop->span);
  }

  Map<Var, Buffer> buffer_data_to_buffer_;
};

bool HasIncompleteZ3Loop(const PrimFunc &func) {
  bool found = false;
  PostOrderVisit(func->body, [&](const ObjectRef &node) {
    if (const auto *loop = node.as<ForNode>()) {
      found = found || Z3TaskAnalyzer::NeedsInference(loop);
    }
  });
  return found;
}

}  // namespace

tvm::transform::Pass InferZ3PipelineProblem() {
  using namespace tirx::transform;
  auto pass_func = [](PrimFunc func, const IRModule &, PassContext ctx) {
    if (!ctx->GetConfig<Bool>(kTaskAnalysisConfig, Bool(false)).value()) {
      return func;
    }
    if (!HasIncompleteZ3Loop(func)) return func;
    func.CopyOnWrite()->body = Z3TaskAnalyzer::Substitute(func);
    return func;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.InferZ3PipelineProblem", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = reflection;
  TVM_REGISTER_PASS_CONFIG_OPTION(kTaskAnalysisConfig, Bool);
  refl::GlobalDef().def("tl.transform.InferZ3PipelineProblem",
                        InferZ3PipelineProblem);
}

}  // namespace tl
}  // namespace tvm
