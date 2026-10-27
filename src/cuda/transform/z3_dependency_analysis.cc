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

#include "z3_dependency_analysis.h"

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>

#include <algorithm>
#include <functional>
#include <set>
#include <sstream>
#include <unordered_set>

#include "op/builtin.h"
#include "op/parallel.h"
#include "op/utils.h"
#include "transform/common/int64_promoter.h"

namespace tvm::tl::cuda::z3 {

using namespace tirx;

namespace {

constexpr SpecialStateMask kLoopControl =
    static_cast<SpecialStateMask>(SpecialState::kLoopControl);
constexpr SpecialStateMask kAsyncProxy =
    static_cast<SpecialStateMask>(CudaSpecialState::kAsyncProxy);
constexpr SpecialStateMask kMBarrierProtocol =
    static_cast<SpecialStateMask>(CudaSpecialState::kMBarrierProtocol);

void AddVar(const Var& var, std::vector<Var>* vars) {
  if (std::find_if(vars->begin(), vars->end(), [&](const Var& other) {
        return other.same_as(var);
      }) == vars->end()) {
    vars->push_back(var);
  }
}

void AppendAccesses(const TaskAccessInfo& input, TaskAccessInfo* output) {
  output->reads.insert(output->reads.end(), input.reads.begin(), input.reads.end());
  output->writes.insert(output->writes.end(), input.writes.begin(), input.writes.end());
  for (const Var& var : input.read_vars) AddVar(var, &output->read_vars);
  for (const Var& var : input.write_vars) AddVar(var, &output->write_vars);
  output->state_reads |= input.state_reads;
  output->state_writes |= input.state_writes;
}

bool NameEndsWith(const std::string& name, const std::string& suffix) {
  return name.size() >= suffix.size() &&
         name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0;
}

class CudaAccessCollector final : public StmtExprVisitor {
 public:
  static TaskAccessInfo Analyze(const Stmt& stmt,
                                const ffi::Map<Var, Buffer>& buffers,
                                const CudaTaskAnalysisOptions& options) {
    CudaAccessCollector collector(buffers, options);
    SBlock wrapper({}, {}, {}, "cuda_z3_access", stmt);
    collector.VisitStmt(wrapper);
    return std::move(collector.accesses_);
  }

 private:
  CudaAccessCollector(const ffi::Map<Var, Buffer>& buffers,
                      const CudaTaskAnalysisOptions& options)
      : buffers_(buffers), options_(options) {}

  static Region PointRegion(const ffi::Array<PrimExpr>& indices) {
    Region result;
    result.reserve(indices.size());
    for (const PrimExpr& index : indices) {
      result.push_back(Range::FromMinExtent(index, make_const(index.dtype(), 1)));
    }
    return result;
  }

  void AppendRegion(const BufferRegion& region, bool write) {
    if (write) {
      accesses_.writes.push_back(region);
    } else {
      accesses_.reads.push_back(region);
    }
  }

  void VisitRegionExpressions(const BufferRegion& region) {
    for (const Range& range : region->region) {
      VisitExpr(range->min);
      VisitExpr(range->extent);
    }
  }

  void HandleTileOp(const TileOperator& tile_op) {
    if (const auto* parallel = tile_op.as<ParallelOpNode>()) {
      VisitStmt(parallel->GetRoot());
      return;
    }
    AccessRegions access = tile_op->GetAccessRegions();
    for (const BufferRegion& region : access.reads) {
      accesses_.reads.push_back(region);
      VisitRegionExpressions(region);
    }
    for (const BufferRegion& region : access.writes) {
      accesses_.writes.push_back(region);
      VisitRegionExpressions(region);
    }
  }

  bool AppendPointerBase(const PrimExpr& pointer, bool write) {
    BufferRegion region;
    if (const auto* load = pointer.as<BufferLoadNode>()) {
      region = BufferRegion::FullRegion(load->buffer);
      for (const PrimExpr& index : load->indices) VisitExpr(index);
      if (load->predicate.defined()) VisitExpr(load->predicate.value());
    } else if (const auto* var = pointer.as<VarNode>()) {
      auto found = buffers_.find(ffi::GetRef<Var>(var));
      if (found != buffers_.end()) {
        region = BufferRegion::FullRegion((*found).second);
      }
    } else {
      return false;
    }
    if (region.defined()) AppendRegion(region, write);
    return region.defined();
  }

  bool HandleExplicitCallContract(const CallNode* op) {
    if (!options_.explicit_state_contracts) return false;
    const auto* op_node = op->op.as<OpNode>();
    if (op_node == nullptr) return false;
    const std::string& name = op_node->name;
    Call call = ffi::GetRef<Call>(op);

    if (op->op.same_as(tl::loop_break())) {
      accesses_.state_writes |= kLoopControl;
      for (const PrimExpr& arg : op->args) VisitExpr(arg);
      return true;
    }

    auto barrier_contract = [&](bool write, SpecialStateMask state,
                                size_t scalar_begin) {
      ICHECK(!op->args.empty()) << options_.diagnostic_prefix << ": " << call
                                << " is missing its barrier pointer";
      bool resolved = AppendPointerBase(op->args[0], write);
      if (!resolved) {
        const auto* pointer_call = op->args[0].as<CallNode>();
        resolved = pointer_call != nullptr &&
                   (pointer_call->op.same_as(builtin::address_of()) ||
                    pointer_call->op.same_as(tl::access_ptr()) ||
                    pointer_call->op.same_as(builtin::tvm_access_ptr()));
        if (resolved) VisitExpr(op->args[0]);
      }
      ICHECK(resolved || !options_.strict_opaque)
          << options_.diagnostic_prefix << ": barrier intrinsic has an opaque base; "
          << "automatic dependency extraction would be incomplete: " << call;
      for (size_t i = scalar_begin; i < op->args.size(); ++i) VisitExpr(op->args[i]);
      if (write) {
        accesses_.state_writes |= state;
      } else {
        accesses_.state_reads |= state;
      }
      return true;
    };

    if (NameEndsWith(name, "mbarrier_wait_parity")) {
      return barrier_contract(false, kMBarrierProtocol, 1);
    }
    if (NameEndsWith(name, "mbarrier_expect_tx") ||
        NameEndsWith(name, "ptx_arrive_barrier_expect_tx") ||
        NameEndsWith(name, "ptx_arrive_barrier") ||
        NameEndsWith(name, "ptx_arrive_cluster_barrier") ||
        NameEndsWith(name, "ptx_cp_async_barrier_noinc") ||
        NameEndsWith(name, "tcgen05_mma_arrive")) {
      return barrier_contract(true, kMBarrierProtocol, 1);
    }
    if (NameEndsWith(name, "ptx_cp_async")) {
      ICHECK_GE(op->args.size(), 3U)
          << options_.diagnostic_prefix << ": malformed cp.async contract: " << call;
      for (const PrimExpr& arg : op->args) VisitExpr(arg);
      accesses_.state_writes |= kAsyncProxy;
      return true;
    }
    if (NameEndsWith(name, "tma_store_arrive") ||
        NameEndsWith(name, "warpgroup_arrive")) {
      for (const PrimExpr& arg : op->args) VisitExpr(arg);
      accesses_.state_writes |= kAsyncProxy;
      return true;
    }
    if (NameEndsWith(name, "tma_store_wait") ||
        NameEndsWith(name, "cp_async_wait_group") ||
        NameEndsWith(name, "ptx_wait_group") ||
        NameEndsWith(name, "commit_group") ||
        NameEndsWith(name, "warpgroup_wait")) {
      for (const PrimExpr& arg : op->args) VisitExpr(arg);
      accesses_.state_reads |= kAsyncProxy;
      accesses_.state_writes |= kAsyncProxy;
      return true;
    }
    return false;
  }

  void VisitExpr_(const VarNode* op) final {
    AddVar(ffi::GetRef<Var>(op), &accesses_.read_vars);
  }

  void VisitStmt_(const BindNode* op) final {
    AddVar(op->var, &accesses_.write_vars);
    VisitExpr(op->value);
  }

  void VisitStmt_(const AttrStmtNode* op) final {
    if (options_.collect_attribute_node) {
      if (const auto* expr = op->node.as<PrimExprNode>()) {
        VisitExpr(ffi::GetRef<PrimExpr>(expr));
      }
    }
    if (op->value.defined()) VisitExpr(op->value);
    VisitStmt(op->body);
  }

  void VisitStmt_(const BufferStoreNode* op) final {
    accesses_.writes.push_back(BufferRegion(op->buffer, PointRegion(op->indices)));
    VisitExpr(op->value);
    for (const PrimExpr& index : op->indices) VisitExpr(index);
    if (op->predicate.defined()) VisitExpr(op->predicate.value());
  }

  void VisitExpr_(const BufferLoadNode* op) final {
    accesses_.reads.push_back(BufferRegion(op->buffer, PointRegion(op->indices)));
    for (const PrimExpr& index : op->indices) VisitExpr(index);
    if (op->predicate.defined()) VisitExpr(op->predicate.value());
  }

  void VisitExpr_(const CallNode* op) final {
    Call call = ffi::GetRef<Call>(op);
    if (auto tile_op = ParseOperator(call); tile_op.defined()) {
      HandleTileOp(tile_op);
      StmtExprVisitor::VisitExpr_(op);
      return;
    }
    if (options_.explicit_state_contracts && op->op.same_as(tl::region())) {
      AccessRegion access = NormalizeToAccessRegion(call);
      if (access.access_mask & kAccessRead) AppendRegion(access.region, false);
      if (access.access_mask & kAccessWrite) AppendRegion(access.region, true);
      // A region descriptor carries no additional payload transfer beyond its
      // contract, but every minimum/extent remains a scalar control input.
      VisitRegionExpressions(access.region);
      VisitExpr(op->args[1]);
      return;
    }
    if (op->op.same_as(builtin::address_of())) {
      ICHECK(!op->args.empty());
      bool resolved = AppendPointerBase(op->args[0], false);
      ICHECK(resolved || !options_.strict_opaque)
          << options_.diagnostic_prefix
          << ": address_of has an opaque base; automatic dependency extraction "
             "would be incomplete";
      return;
    }
    if (op->op.same_as(tl::access_ptr())) {
      ICHECK_EQ(op->args.size(), 3U);
      const auto* load = op->args[0].as<BufferLoadNode>();
      ICHECK(load != nullptr) << "tl.access_ptr base must be a BufferLoad";
      BufferRegion region = BufferRegion::FullRegion(load->buffer);
      int access_mask = GetConservativeAccessMask(op->args[2]);
      if (access_mask & kAccessRead) accesses_.reads.push_back(region);
      if (access_mask & kAccessWrite) accesses_.writes.push_back(region);
      for (const PrimExpr& index : load->indices) VisitExpr(index);
      if (load->predicate.defined()) VisitExpr(load->predicate.value());
      VisitExpr(op->args[1]);
      VisitExpr(op->args[2]);
      return;
    }
    if (op->op.same_as(builtin::tvm_access_ptr())) {
      ICHECK_GE(op->args.size(), 2U);
      const auto* buffer_var = op->args[1].as<VarNode>();
      ICHECK(buffer_var != nullptr);
      auto found = buffers_.find(ffi::GetRef<Var>(buffer_var));
      ICHECK(found != buffers_.end() || !options_.strict_opaque)
          << options_.diagnostic_prefix
          << ": tvm_access_ptr references an unknown buffer; automatic "
             "dependency extraction would be incomplete";
      if (found != buffers_.end()) {
        BufferRegion region = BufferRegion::FullRegion((*found).second);
        int access_mask = op->args.size() == 5U
                              ? GetConservativeAccessMask(op->args[4])
                              : kAccessReadWrite;
        if (access_mask & kAccessRead) accesses_.reads.push_back(region);
        if (access_mask & kAccessWrite) accesses_.writes.push_back(region);
      }
      for (size_t i = 2; i < op->args.size(); ++i) VisitExpr(op->args[i]);
      return;
    }
    if (HandleExplicitCallContract(op)) return;
    CallEffectKind effect = SideEffect(call);
    ICHECK(!options_.strict_opaque || effect == CallEffectKind::kPure ||
           (options_.allow_read_state && effect <= CallEffectKind::kReadState))
        << options_.diagnostic_prefix
        << ": opaque or stateful Call has no complete BufferRegion/state "
           "contract; supply dependencies explicitly: "
        << call;
    StmtExprVisitor::VisitExpr_(op);
  }

  ffi::Map<Var, Buffer> buffers_;
  CudaTaskAnalysisOptions options_;
  TaskAccessInfo accesses_;
};

const ControlNode* HeaderControl(const TaskNode* task) {
  const auto* parent = task->GetParent();
  if (parent == nullptr || !parent->IsControl()) return nullptr;
  const auto* control = static_cast<const ControlNode*>(parent);
  return control->task.get() == task ? control : nullptr;
}

IRStructure* SemanticNode(TaskNode* task) {
  const ControlNode* control = HeaderControl(task);
  return control == nullptr ? static_cast<IRStructure*>(task)
                            : static_cast<IRStructure*>(const_cast<ControlNode*>(control));
}

bool LexicallyPrecedes(TaskNode* writer, TaskNode* reader) {
  IRStructure* lhs = SemanticNode(writer);
  IRStructure* rhs = SemanticNode(reader);
  if (lhs == rhs) return true;
  std::vector<IRStructure*> a = lhs->PathFrom();
  std::vector<IRStructure*> b = rhs->PathFrom();
  size_t common = 0;
  while (common < a.size() && common < b.size() && a[common] == b[common]) ++common;
  if (common == a.size()) return true;
  if (common == b.size()) return false;
  return a[common]->GetIndex() < b[common]->GetIndex();
}

bool IsDirectScalarBind(const TaskNode* task, const Var& var) {
  Stmt stmt = task->stmt;
  while (const auto* attr = stmt.as<AttrStmtNode>()) stmt = attr->body;
  const auto* bind = stmt.as<BindNode>();
  return bind != nullptr && bind->var.same_as(var);
}

std::string TaskLabel(const TaskNode* task) {
  std::ostringstream os;
  if (!task->ws_op_id.empty()) {
    os << task->ws_op_id;
  } else {
    os << "source " << task->GetSourceId();
  }
  if (HeaderControl(task) != nullptr) os << " (control header)";
  return os.str();
}

void ValidateScalarBindingsForTasks(const std::vector<TaskNode*>& tasks) {
  std::unordered_map<Var, TaskNode*, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> writers;
  for (TaskNode* task : tasks) {
    for (const Var& var : task->accesses.write_vars) {
      auto [it, inserted] = writers.emplace(var, task);
      ICHECK(inserted) << "Duplicate scalar Bind definition for Var " << var
                       << " in " << TaskLabel(it->second) << " and " << TaskLabel(task)
                       << "; nested scalar analysis requires unique Var identity";
    }
  }
  for (TaskNode* reader : tasks) {
    for (const Var& var : reader->accesses.read_vars) {
      auto found = writers.find(var);
      if (found == writers.end()) continue;
      TaskNode* writer = found->second;
      if (writer == reader) {
        ICHECK(!IsDirectScalarBind(reader, var))
            << "Cyclic scalar Bind definition involving Var " << var << " in "
            << TaskLabel(reader);
        continue;
      }
      ICHECK(LexicallyPrecedes(writer, reader))
          << "Scalar Var " << var << " is used by " << TaskLabel(reader)
          << " before its lexical Bind definition in " << TaskLabel(writer);
    }
  }
}

void CollectDeclaredBuffers(const ScheduledTIR& scheduled,
                            ffi::Map<Var, Buffer>* buffers) {
  if (scheduled.metadata.kernel_root.defined()) {
    for (const Buffer& buffer : scheduled.metadata.kernel_root->alloc_buffers) {
      if (!buffers->count(buffer->data)) buffers->Set(buffer->data, buffer);
    }
  }
  std::vector<TaskNode*> tasks;
  CollectAllTaskNodes(scheduled.tree, &tasks, true);
  for (const TaskNode* task : tasks) {
    PostOrderVisit(task->stmt, [&](const ffi::ObjectRef& node) {
      if (const auto* decl = node.as<DeclBufferNode>()) {
        if (!buffers->count(decl->buffer->data)) buffers->Set(decl->buffer->data, decl->buffer);
      } else if (const auto* alloc = node.as<AllocBufferNode>()) {
        if (!buffers->count(alloc->buffer->data)) buffers->Set(alloc->buffer->data, alloc->buffer);
      }
    });
    for (const auto* regions : {&task->accesses.reads, &task->accesses.writes}) {
      for (const BufferRegion& region : *regions) {
        if (!buffers->count(region->buffer->data)) {
          buffers->Set(region->buffer->data, region->buffer);
        }
      }
    }
  }
}

std::vector<const ControlNode*> StorageIterationNest(ControlNode* loop,
                                                      const Var& storage) {
  std::vector<const ControlNode*> controls = loop->GetAncestorControls();
  size_t first = 0;
  for (size_t i = controls.size(); i > 1; --i) {
    const ControlNode* parent = controls[i - 2];
    const IRStructure* path_child = controls[i - 1];
    ICHECK_EQ(path_child->GetParent(), parent);
    bool external = parent->task && parent->task->TouchesStorage(storage);
    for (const auto& child : parent->children) {
      if (child.get() != path_child && child->TouchesStorage(storage)) {
        external = true;
        break;
      }
    }
    if (external) {
      first = i - 1;
      break;
    }
  }
  return {controls.begin() + first, controls.end()};
}

bool SameStorage(const std::optional<Var>& lhs, const std::optional<Var>& rhs) {
  if (lhs.has_value() != rhs.has_value()) return false;
  return !lhs.has_value() || lhs.value().same_as(rhs.value());
}

struct RegionAccess {
  TaskNode* task;
  BufferRegion region;
};
struct StateAccess {
  TaskNode* task;
  SpecialStateMask mask;
};
struct NodeAccesses {
  std::vector<RegionAccess> reads;
  std::vector<RegionAccess> writes;
  std::vector<StateAccess> state_reads;
  std::vector<StateAccess> state_writes;
};

std::vector<DepInfo> AnalyzeDependenciesImpl(
    std::vector<IRStructure*> nodes, ControlNode* loop,
    const BufferVersionMap& manual_versions, const MultiBufferOwnerMap& owners,
    DependencyCache* cache);

void CollectCoveredDependencies(IRStructure* node,
                                const BufferVersionMap& manual_versions,
                                const MultiBufferOwnerMap& owners,
                                DependencyCache* cache,
                                std::vector<DepInfo>* covered) {
  if (node == nullptr || !node->IsControl()) return;
  auto* control = static_cast<ControlNode*>(node);
  std::vector<IRStructure*> children;
  for (const auto& child : control->children) children.push_back(child.get());
  std::vector<DepInfo> child_deps = AnalyzeDependenciesImpl(
      children, control, manual_versions, owners, cache);
  covered->insert(covered->end(), child_deps.begin(), child_deps.end());
  for (IRStructure* child : children) {
    CollectCoveredDependencies(child, manual_versions, owners, cache, covered);
  }
}

std::vector<DepInfo> AnalyzeDependenciesImpl(
    std::vector<IRStructure*> nodes, ControlNode* loop,
    const BufferVersionMap& manual_versions, const MultiBufferOwnerMap& owners,
    DependencyCache* cache) {
  if (loop != nullptr) {
    auto found = cache->find(loop);
    if (found != cache->end()) return found->second;
    std::stable_sort(nodes.begin(), nodes.end(), [](const IRStructure* lhs,
                                                    const IRStructure* rhs) {
      return lhs->GetStage() < rhs->GetStage();
    });
  }

  const size_t n = nodes.size();
  std::vector<NodeAccesses> accesses(n);
  std::unordered_map<TaskNode*, size_t> task_node;
  for (size_t i = 0; i < n; ++i) {
    std::vector<TaskNode*> tasks;
    CollectAllTaskNodes(nodes[i], &tasks, true);
    for (TaskNode* task : tasks) {
      task_node[task] = i;
      for (const BufferRegion& region : task->accesses.reads) {
        accesses[i].reads.push_back({task, region});
      }
      for (const BufferRegion& region : task->accesses.writes) {
        accesses[i].writes.push_back({task, region});
      }
      SpecialStateMask reads = task->accesses.state_reads & ~kLoopControl;
      SpecialStateMask writes = task->accesses.state_writes & ~kLoopControl;
      if (reads) accesses[i].state_reads.push_back({task, reads});
      if (writes) accesses[i].state_writes.push_back({task, writes});
    }
    if (loop != nullptr) {
      TaskNode* representative = nodes[i]->IsTask()
                                     ? static_cast<TaskNode*>(nodes[i])
                                     : static_cast<ControlNode*>(nodes[i])->task.get();
      accesses[i].state_reads.push_back({representative, kLoopControl});
      if (representative->accesses.state_writes & kLoopControl) {
        accesses[i].state_writes.push_back({representative, kLoopControl});
      }
    }
  }

  std::vector<DepInfo> deps;
  std::set<DependencyTaskPair> scalar_task_pairs;
  auto add_pair = [&](IRStructure* producer, IRStructure* consumer,
                      const std::optional<Var>& storage, int distance,
                      TaskNode* producer_task, TaskNode* consumer_task,
                      DependencyKind kind = DependencyKind::kData) {
    auto found = std::find_if(deps.begin(), deps.end(), [&](const DepInfo& dep) {
      return dep.prod_node == producer && dep.cons_node == consumer &&
             SameStorage(dep.storage, storage) && dep.distance == distance && dep.kind == kind;
    });
    if (found == deps.end()) {
      deps.push_back({producer, consumer, storage, {}, distance, kind});
      found = std::prev(deps.end());
    }
    DependencyTaskPair pair{producer_task, consumer_task};
    if (std::find(found->task_pairs.begin(), found->task_pairs.end(), pair) ==
        found->task_pairs.end()) {
      found->task_pairs.push_back(pair);
    }
  };

  auto manual_for = [&](const std::optional<Var>& storage) {
    if (!storage.has_value()) return 0;
    auto found = manual_versions.find(storage.value());
    return found == manual_versions.end() ? 0 : (*found).second;
  };
  auto owner_count = [&](const std::optional<Var>& storage) -> size_t {
    if (loop == nullptr || !storage.has_value()) return 0;
    auto found = owners.find(storage.value());
    if (found == owners.end() ||
        std::find(found->second.begin(), found->second.end(), loop) == found->second.end()) {
      return 0;
    }
    return found->second.size();
  };
  auto find_owner = [](const std::vector<ControlNode*>& owner_loops,
                       const TaskNode* task) -> ControlNode* {
    ControlNode* result = nullptr;
    for (ControlNode* owner : owner_loops) {
      if (!task->IsWithin(owner)) continue;
      if (result != nullptr) return nullptr;
      result = owner;
    }
    return result;
  };
  auto owner_exclusion = [&](const std::optional<Var>& storage,
                             const TaskNode* lhs, const TaskNode* rhs) {
    if (!storage.has_value()) return false;
    auto found = owners.find(storage.value());
    if (found == owners.end() || found->second.size() <= 1) return false;
    ControlNode* lhs_owner = find_owner(found->second, lhs);
    ControlNode* rhs_owner = find_owner(found->second, rhs);
    return lhs_owner != nullptr && rhs_owner != nullptr && lhs_owner != rhs_owner;
  };

  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      if (loop == nullptr && i >= j) continue;
      auto consider_regions = [&](const std::vector<RegionAccess>& lhs_accesses,
                                  const std::vector<RegionAccess>& rhs_accesses) {
        for (const RegionAccess& lhs : lhs_accesses) {
          for (const RegionAccess& rhs : rhs_accesses) {
            std::optional<Var> storage;
            if (lhs.region->buffer->data.same_as(rhs.region->buffer->data)) {
              storage = lhs.region->buffer->data;
            }
            bool exclusion = owner_exclusion(storage, lhs.task, rhs.task);
            if (exclusion && i >= j) continue;
            int manual = manual_for(storage);
            if (manual && !exclusion) {
              for (int distance = (i < j ? 0 : 1); distance <= manual; ++distance) {
                if (RegionsMayConflict(lhs.task->outer_ctx, lhs.region,
                                       rhs.task->outer_ctx, rhs.region, loop, distance)) {
                  add_pair(nodes[i], nodes[j], storage, distance, lhs.task, rhs.task);
                  break;
                }
              }
              continue;
            }
            for (int distance : {0, -1}) {
              if (distance == 0 && i >= j) continue;
              if (distance < 0 && loop == nullptr) continue;
              if (RegionsMayConflict(lhs.task->outer_ctx, lhs.region,
                                     rhs.task->outer_ctx, rhs.region, loop, distance,
                                     owner_count(storage))) {
                add_pair(nodes[i], nodes[j], storage, exclusion ? 0 : distance,
                         lhs.task, rhs.task,
                         exclusion ? DependencyKind::kOwnerExclusion : DependencyKind::kData);
                break;
              }
            }
          }
        }
      };
      consider_regions(accesses[i].writes, accesses[j].reads);  // RAW
      consider_regions(accesses[i].writes, accesses[j].writes); // WAW
      consider_regions(accesses[i].reads, accesses[j].writes);  // WAR

      auto consider_state = [&](const std::vector<StateAccess>& lhs_accesses,
                                const std::vector<StateAccess>& rhs_accesses) {
        for (const StateAccess& lhs : lhs_accesses) {
          for (const StateAccess& rhs : rhs_accesses) {
            if (!(lhs.mask & rhs.mask)) continue;
            if (i < j) {
              add_pair(nodes[i], nodes[j], std::nullopt, 0, lhs.task, rhs.task);
            }
            if (loop != nullptr) {
              add_pair(nodes[i], nodes[j], std::nullopt, -1, lhs.task, rhs.task);
            }
          }
        }
      };
      consider_state(accesses[i].state_writes, accesses[j].state_reads);
      consider_state(accesses[i].state_writes, accesses[j].state_writes);
      consider_state(accesses[i].state_reads, accesses[j].state_writes);
    }
  }

  // Scalar Bind RAW plus reverse next-iteration WAR. Exact leaf/header pairs
  // retain consumer-subtree timestamps even when the parent nodes are Controls.
  std::unordered_map<Var, TaskNode*, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> scalar_writers;
  for (const auto& [task, node_index] : task_node) {
    (void)node_index;
    for (const Var& var : task->accesses.write_vars) scalar_writers.emplace(var, task);
  }
  for (size_t reader_index = 0; reader_index < n; ++reader_index) {
    std::vector<TaskNode*> readers;
    CollectAllTaskNodes(nodes[reader_index], &readers, true);
    for (TaskNode* reader : readers) {
      for (const Var& var : reader->accesses.read_vars) {
        auto found = scalar_writers.find(var);
        if (found == scalar_writers.end() || found->second == reader) continue;
        TaskNode* writer = found->second;
        auto writer_index = task_node.find(writer);
        if (writer_index == task_node.end()) continue;
        add_pair(nodes[writer_index->second], nodes[reader_index], std::nullopt, 0,
                 writer, reader);
        scalar_task_pairs.emplace(writer, reader);
        if (loop != nullptr) {
          add_pair(nodes[reader_index], nodes[writer_index->second], std::nullopt, -1,
                   reader, writer);
          scalar_task_pairs.emplace(reader, writer);
        }
      }
    }
  }

  // Drop only same-distance buffer edges and same-iteration scalar edges already
  // covered in the subtree. Cross-iteration scalar and all special-state/opaque
  // effects remain intact.
  for (DepInfo& dep : deps) {
    if (dep.prod_node != dep.cons_node || !dep.prod_node->IsControl() ||
        dep.kind != DependencyKind::kData) {
      continue;
    }
    std::vector<DepInfo> covered;
    CollectCoveredDependencies(dep.prod_node, manual_versions, owners, cache, &covered);
    for (auto pair = dep.task_pairs.begin(); pair != dep.task_pairs.end();) {
      bool covered_scalar = !dep.storage.has_value() && dep.distance == 0 &&
                            scalar_task_pairs.count(*pair) != 0;
      if (!dep.storage.has_value() && !covered_scalar) {
        ++pair;
        continue;
      }
      bool redundant = std::any_of(covered.begin(), covered.end(), [&](const DepInfo& child) {
        return child.kind == DependencyKind::kData && SameStorage(child.storage, dep.storage) &&
               child.distance == dep.distance &&
               std::find(child.task_pairs.begin(), child.task_pairs.end(), *pair) !=
                   child.task_pairs.end();
      });
      if (redundant) {
        pair = dep.task_pairs.erase(pair);
      } else {
        ++pair;
      }
    }
  }
  deps.erase(std::remove_if(deps.begin(), deps.end(),
                            [](const DepInfo& dep) { return dep.task_pairs.empty(); }),
             deps.end());
  for (DepInfo& dep : deps) {
    std::sort(dep.task_pairs.begin(), dep.task_pairs.end());
  }
  if (loop != nullptr) cache->emplace(loop, deps);
  return deps;
}

ffi::String NodeName(const IRStructure* node) {
  if (node->IsTask()) {
    const auto* task = static_cast<const TaskNode*>(node);
    if (!task->ws_op_id.empty()) return task->ws_op_id;
  } else {
    const auto* control = static_cast<const ControlNode*>(node);
    if (!control->task->ws_op_id.empty()) return control->task->ws_op_id;
  }
  std::ostringstream os;
  os << "source_" << node->GetSourceId();
  return ffi::String(os.str());
}

ffi::String ExactTaskName(const TaskNode* task) {
  std::ostringstream os;
  os << TaskLabel(task);
  return ffi::String(os.str());
}

}  // namespace

TaskAccessInfo AnalyzeCudaTaskAccesses(const Stmt& stmt,
                                       const ffi::Map<Var, Buffer>& buffers,
                                       const CudaTaskAnalysisOptions& options) {
  return CudaAccessCollector::Analyze(stmt, buffers, options);
}

void AppendExpressionAccesses(const PrimExpr& expr, TaskAccessInfo* accesses) {
  ICHECK(accesses != nullptr);
  AppendAccesses(AnalyzeCudaTaskAccesses(Evaluate(expr), {}), accesses);
}

void AppendGuardAccesses(const IRStructure& node, TaskAccessInfo* accesses) {
  ICHECK(accesses != nullptr);
  for (const auto& guard : node.GetGuards()) {
    if (guard->IsCondition()) {
      AppendExpressionAccesses(static_cast<const ConditionGuard*>(guard.get())->condition,
                               accesses);
      continue;
    }
    const auto* attribute = static_cast<const AttributeGuard*>(guard.get());
    if (auto expr = attribute->node.try_cast<PrimExpr>(); expr.has_value()) {
      AppendExpressionAccesses(expr.value(), accesses);
    }
    if (attribute->value.defined()) AppendExpressionAccesses(attribute->value, accesses);
  }
}

void RefreshCudaTaskAccesses(ScheduledTIR* scheduled,
                             const ffi::Map<Var, Buffer>& buffers,
                             const CudaTaskAnalysisOptions& options) {
  ICHECK(scheduled != nullptr);
  ffi::Map<Var, Buffer> all_buffers = buffers;
  CollectDeclaredBuffers(*scheduled, &all_buffers);
  std::function<void(IRStructure*)> refresh = [&](IRStructure* node) {
    if (node->IsTask()) {
      auto* task = static_cast<TaskNode*>(node);
      TaskAccessInfo accesses = AnalyzeCudaTaskAccesses(task->stmt, all_buffers, options);
      AppendGuardAccesses(*task, &accesses);
      accesses.state_reads |= kLoopControl;
      task->accesses = std::move(accesses);
      return;
    }
    auto* control = static_cast<ControlNode*>(node);
    ICHECK(control->task != nullptr);
    TaskAccessInfo accesses =
        AnalyzeCudaTaskAccesses(control->task->stmt, all_buffers, options);
    AppendGuardAccesses(*control, &accesses);
    control->task->accesses = std::move(accesses);
    for (auto& child : control->children) refresh(child.get());
  };
  for (auto& node : scheduled->tree) refresh(node.get());
  ValidateNestedScalarBindings(*scheduled);
}

void ValidateNestedScalarBindings(const ScheduledTIR& scheduled) {
  std::vector<TaskNode*> tasks;
  CollectAllTaskNodes(scheduled.tree, &tasks, true);
  ValidateScalarBindingsForTasks(tasks);
}

bool RegionsMayConflict(const ConstrSet& a_ctx, const BufferRegion& a_region,
                        const ConstrSet& b_ctx, const BufferRegion& b_region,
                        ControlNode* loop, int offset, size_t num_storage_owners) {
  const bool cross = offset != 0;
  if (cross && loop == nullptr) return false;
  if (!a_region->buffer->data.same_as(b_region->buffer->data)) return false;
  if (!a_region->buffer.same_as(b_region->buffer)) return true;
  const ffi::Array<Range>& a_ranges = a_region->region;
  const ffi::Array<Range>& b_ranges = b_region->region;
  if (a_ranges.size() != b_ranges.size()) return true;

  arith::Analyzer analyzer;
  analyzer.z3_prover.SetRLimit(50000);
  ffi::Map<Var, PrimExpr> a_sub, b_sub;
  ConstrSet a_local = a_ctx;
  ConstrSet b_local = b_ctx;
  ConstrSet loop_ctx;
  auto seed_iteration_vars = [&](const ControlNode* scope) {
    std::vector<Var> writes = scope->Accesses().write_vars;
    for (const Var& var : writes) {
      if (!a_sub.count(var)) a_sub.Set(var, var.copy_with_suffix("_p"));
      if (!b_sub.count(var)) b_sub.Set(var, var.copy_with_suffix("_c"));
    }
  };
  if (cross) {
    if (offset < 0) {
      std::vector<const ControlNode*> nest =
          num_storage_owners > 1 ? loop->GetAncestorControls()
                                 : StorageIterationNest(loop, a_region->buffer->data);
      ICHECK(!nest.empty());
      seed_iteration_vars(nest.front());
      Var pivot = nest.front()->control->loop_var;
      a_local = a_local.RenameFrom("_p", a_sub, pivot);
      b_local = b_local.RenameFrom("_c", b_sub, pivot);
      if (num_storage_owners <= 1) {
        PrimExpr different = Bool(false);
        for (const ControlNode* control : nest) {
          Var iteration = control->control->loop_var;
          different = different ||
                      (Substitute(iteration, a_sub) != Substitute(iteration, b_sub));
        }
        loop_ctx.AddConstr(different);
      }
    } else {
      seed_iteration_vars(loop);
      Var iteration = loop->control->loop_var;
      a_local = a_local.RenameFrom("_p", a_sub, iteration);
      b_local = b_local.RenameFrom("_c", b_sub, iteration);
      loop_ctx.AddConstr(Substitute(iteration, b_sub) ==
                         Substitute(iteration, a_sub) + IntImm(iteration.dtype(), offset));
    }
  }
  loop_ctx.Merge(a_local).Merge(b_local).Populate(analyzer);
  FreshenMutableReads freshen_a(FreshenMutableReads::Mode::kSnapshot);
  FreshenMutableReads freshen_b(FreshenMutableReads::Mode::kSnapshot);
  Int64Promoter promoter;
  auto widen = [&](const PrimExpr& value) {
    return cast(DataType::Int(64), promoter(value));
  };
  PrimExpr disjoint = Bool(false);
  for (size_t i = 0; i < a_ranges.size(); ++i) {
    PrimExpr a_min = widen(freshen_a(Substitute(a_ranges[i]->min, a_sub)));
    PrimExpr a_extent = widen(freshen_a(Substitute(a_ranges[i]->extent, a_sub)));
    PrimExpr b_min = widen(freshen_b(Substitute(b_ranges[i]->min, b_sub)));
    PrimExpr b_extent = widen(freshen_b(Substitute(b_ranges[i]->extent, b_sub)));
    PrimExpr a_end = a_min + a_extent;
    PrimExpr b_end = b_min + b_extent;
    disjoint = disjoint || (a_extent <= 0) || (b_extent <= 0) ||
               (a_end <= b_min) || (b_end <= a_min);
  }
  disjoint = analyzer.Simplify(disjoint);
  return !(analyzer.CanProve(disjoint) || analyzer.z3_prover.CanProve(disjoint));
}

std::vector<DepInfo> AnalyzeDependencies(std::vector<IRStructure*> nodes,
                                         ControlNode* loop,
                                         const BufferVersionMap& manual_versions,
                                         const MultiBufferOwnerMap& owners,
                                         DependencyCache* dependency_cache) {
  std::vector<TaskNode*> tasks;
  for (IRStructure* node : nodes) CollectAllTaskNodes(node, &tasks, true);
  ValidateScalarBindingsForTasks(tasks);
  DependencyCache local;
  DependencyCache& cache = dependency_cache ? *dependency_cache : local;
  return AnalyzeDependenciesImpl(std::move(nodes), loop, manual_versions, owners, &cache);
}

std::pair<int, int> ResolveOwnerExclusionDistances(const DepInfo& dep) {
  ICHECK(dep.kind == DependencyKind::kOwnerExclusion);
  ICHECK(dep.prod_node != nullptr && dep.cons_node != nullptr);
  ICHECK_GE(dep.prod_node->GetStage(), 0);
  ICHECK_GE(dep.cons_node->GetStage(), 0);
  int forward = dep.prod_node->GetStage() - dep.cons_node->GetStage();
  return {forward, 1 - forward};
}

namespace {

ffi::Array<ffi::Any> InspectNestedDependencies(const SBlock& root) {
  ScheduledTIR scheduled = DecodeScheduledTIR(root, ConstrSet());
  RefreshCudaTaskAccesses(&scheduled);
  DependencyCache cache;
  ffi::Array<ffi::Any> result;
  std::function<void(std::vector<IRStructure*>, ControlNode*, ffi::String)> visit =
      [&](std::vector<IRStructure*> nodes, ControlNode* loop, ffi::String scope) {
        std::vector<DepInfo> deps = AnalyzeDependencies(
            nodes, loop, scheduled.metadata.requested_versions, {}, &cache);
        for (const DepInfo& dep : deps) {
          ffi::Map<ffi::String, ffi::Any> item;
          item.Set("scope", scope);
          item.Set("prod", NodeName(dep.prod_node));
          item.Set("cons", NodeName(dep.cons_node));
          item.Set("storage", dep.storage.has_value()
                                  ? ffi::String(dep.storage.value()->name_hint)
                                  : ffi::String(""));
          item.Set("distance", Integer(dep.distance));
          item.Set("kind", Integer(static_cast<int>(dep.kind)));
          ffi::Array<ffi::String> pairs;
          for (const auto& [producer, consumer] : dep.task_pairs) {
            pairs.push_back(ExactTaskName(producer));
            pairs.push_back(ExactTaskName(consumer));
          }
          item.Set("task_pairs", pairs);
          result.push_back(item);
        }
        for (IRStructure* node : nodes) {
          if (!node->IsControl()) continue;
          auto* control = static_cast<ControlNode*>(node);
          std::vector<IRStructure*> children;
          for (const auto& child : control->children) children.push_back(child.get());
          visit(children, control, NodeName(control));
        }
      };
  std::vector<IRStructure*> roots;
  for (const auto& node : scheduled.tree) roots.push_back(node.get());
  visit(roots, nullptr, "root");
  return result;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.cuda.transform.Z3NestedTaskAnalysis", InspectNestedDependencies);
}

}  // namespace

}  // namespace tvm::tl::cuda::z3
