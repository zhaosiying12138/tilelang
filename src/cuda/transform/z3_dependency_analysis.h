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
 * \file z3_dependency_analysis.h
 * \brief Strict nested CUDA task access and dependency analysis.
 */

#ifndef TVM_TL_CUDA_TRANSFORM_Z3_DEPENDENCY_ANALYSIS_H_
#define TVM_TL_CUDA_TRANSFORM_Z3_DEPENDENCY_ANALYSIS_H_

#include <tvm/ffi/container/map.h>
#include <tvm/tirx/buffer.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/stmt.h>

#include <map>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "z3_schedule_ir.h"

namespace tvm::tl::cuda::z3 {

using DependencyTaskPair = std::pair<TaskNode*, TaskNode*>;
using MultiBufferOwnerMap =
    std::unordered_map<tirx::Var, std::vector<ControlNode*>, ffi::ObjectPtrHash,
                       ffi::ObjectPtrEqual>;

// Additional CUDA state bits are deliberately conservative. They order async
// protocol effects even when two intrinsics use different pointer transports.
enum class CudaSpecialState : uint64_t {
  kAsyncProxy = 1ULL << 1,
  kMBarrierProtocol = 1ULL << 2,
};

struct CudaTaskAnalysisOptions {
  // Reject a call unless its complete buffer/state effect is known. When
  // false, preserve the legacy flat best-effort behavior for explicit overrides.
  bool strict_opaque{true};
  // The legacy flat collector accepted read-state calls. Nested scheduling is
  // stricter because an untracked state read can change control flow.
  bool allow_read_state{false};
  // Nested scalar/control analysis includes expressions stored in AttrStmt.node;
  // the flat wrapper disables this to retain its historical region set exactly.
  bool collect_attribute_node{true};
  // Enable the explicit loop-break/mbarrier/async contracts below.
  bool explicit_state_contracts{true};
  std::string diagnostic_prefix{"CUDA Z3 nested task analysis"};
};

// Collect direct accesses without discarding scalar Bind/control dependencies.
// Buffer aliases are identified by Buffer.data, not by Buffer object identity.
TaskAccessInfo AnalyzeCudaTaskAccesses(
    const tirx::Stmt& stmt, const ffi::Map<tirx::Var, tirx::Buffer>& buffers,
    const CudaTaskAnalysisOptions& options = {});
void AppendExpressionAccesses(const PrimExpr& expr, TaskAccessInfo* accesses);
void AppendGuardAccesses(const IRStructure& node, TaskAccessInfo* accesses);

// Recompute every leaf/header access list and all task guards. Control access
// views remain derived by Slice A. Also validates unique scalar Bind identity
// and lexical use-before-definition across the complete nested tree.
void RefreshCudaTaskAccesses(
    ScheduledTIR* scheduled, const ffi::Map<tirx::Var, tirx::Buffer>& buffers = {},
    const CudaTaskAnalysisOptions& options = {});
void ValidateNestedScalarBindings(const ScheduledTIR& scheduled);

enum class DependencyKind { kData, kOwnerExclusion };

struct DepInfo {
  IRStructure* prod_node;
  IRStructure* cons_node;
  // Buffer storage identity. Scalar and special-state dependencies use nullopt.
  std::optional<tirx::Var> storage;
  // Exact leaf/header accesses; parent Control nodes alone are not timestamps.
  std::vector<DependencyTaskPair> task_pairs;
  // 0 = same iteration, >0 = tested manual-ring distance, -1 = automatic ring.
  int distance;
  DependencyKind kind{DependencyKind::kData};
};

using DependencyCache = std::map<ControlNode*, std::vector<DepInfo>>;

// Analyze one ordered child list. A non-null loop stable-sorts by manual stage;
// the root list retains lexical order. The cache is valid only for one immutable
// scheduling phase and is keyed by the Control whose child list was analyzed.
std::vector<DepInfo> AnalyzeDependencies(
    std::vector<IRStructure*> nodes, ControlNode* loop = nullptr,
    const BufferVersionMap& manual_versions = {},
    const MultiBufferOwnerMap& owners = {}, DependencyCache* cache = nullptr);

// Proof-only region disjointness with independent producer/consumer contexts,
// nested mutable-variable renaming, mutable-read freshening, and int64 endpoints.
bool RegionsMayConflict(const ConstrSet& a_ctx, const tirx::BufferRegion& a_region,
                        const ConstrSet& b_ctx, const tirx::BufferRegion& b_region,
                        ControlNode* loop, int offset,
                        size_t num_storage_owners = 0);

// Owner exclusions are unordered during analysis. Once physical order/stages
// are known, return {forward, reverse} = {stage(first)-stage(second),
// 1-stage(first)+stage(second)} for the stable DepInfo orientation.
std::pair<int, int> ResolveOwnerExclusionDistances(const DepInfo& dep);

}  // namespace tvm::tl::cuda::z3

#endif  // TVM_TL_CUDA_TRANSFORM_Z3_DEPENDENCY_ANALYSIS_H_
