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
 * \file z3_lifetime_analysis.h
 * \brief CUDA epoch sync graphs, positive generation aliases, and exact arenas.
 */

#ifndef TVM_TL_CUDA_TRANSFORM_Z3_LIFETIME_ANALYSIS_H_
#define TVM_TL_CUDA_TRANSFORM_Z3_LIFETIME_ANALYSIS_H_

#include <tvm/target/target.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "transform/common/iteration_order.h"
#include "z3_epoch_domain.h"

namespace tvm::tl::cuda::z3 {

// Site IDs are dense indices into CudaSyncModel::sites. The model deliberately
// contains no WSSync or stock-pipeline IR: Slice F owns projection of those
// planned synchronization operations into this target-neutral contract.
struct CudaSyncSite {
  int domain{0};
};

struct CudaSyncEdge {
  size_t src{0};
  size_t dst{0};
  // Nonnegative distance in the named epoch clock. Zero is same-epoch order.
  int64_t distance{0};
  // Strong means completion order; weak means issue order only.
  bool strong{false};
  int domain{0};
};

struct CudaStorageLifetime {
  tirx::Var storage;
  int domain{0};
  // Kept in the standalone model so low-level proofs cannot accidentally give
  // a counter epoch a lexical period. Production validation also checks this
  // against EpochDomainRegistry::Domain(domain).
  EpochDomainKind domain_kind{EpochDomainKind::kLexical};
  const ControlNode* scope{nullptr};
  bool periodic{false};
  // A lexical ring must state its period explicitly. A counter ring uses
  // counter-relative event offsets and must leave this field empty.
  std::optional<int64_t> lexical_period;
  std::vector<GenerationLifetime> generations;
};

struct CudaSyncModel {
  std::vector<CudaSyncSite> sites;
  std::vector<CudaSyncEdge> edges;
  std::vector<CudaStorageLifetime> storages;
};

// One nonnegative two-state IterationOrder per epoch domain. Constructing with
// a registry validates domain/guard projection; the null-registry form is for
// standalone, already-projected graph proofs and host tests only.
class CudaLifetimeGraph {
 public:
  explicit CudaLifetimeGraph(const CudaSyncModel& model,
                             const EpochDomainRegistry* registry = nullptr);

  std::optional<int64_t> MinimumDistance(int domain, size_t src, size_t dst) const;
  bool HappensBefore(int domain, size_t src, size_t dst, int64_t distance) const;
  bool CanAlias(const CudaStorageLifetime& lhs, const CudaStorageLifetime& rhs) const;

 private:
  const IterationOrder* FindOrder(int domain) const;

  std::unordered_map<int, std::unique_ptr<IterationOrder>> orders_;
  const CudaSyncModel* model_{nullptr};
};

// Analyze only positive pairs. Unlisted pairs are conflicts; an empty result is
// therefore a valid proof result and not a request to reuse storage freely.
BufferAliasMap AnalyzeBufferLifetimeModel(const CudaSyncModel& model);

bool HasPositiveAlias(const BufferAliasMap& aliases, const tirx::Var& lhs,
                      const tirx::Var& rhs);
void ValidatePositiveAliasMap(const BufferAliasMap& aliases);

// Production entry point required by section 10. It always marks alias
// analysis as present, including when the returned map is empty.
BufferAliasMap AnalyzeBufferLifetimes(ScheduledTIR* scheduled,
                                      const MultiBufferPlan& plan,
                                      const CudaSyncModel& sync_model);

// Slice F adapter hook. The adapter extracts actual access/live boundaries and
// planned strong/weak sync sites; the generic engine performs all proof steps.
using CudaSyncModelAdapter =
    std::function<CudaSyncModel(const ScheduledTIR&, const MultiBufferPlan&,
                                const EpochDomainRegistry&)>;
BufferAliasMap AnalyzeBufferLifetimesWithAdapter(ScheduledTIR* scheduled,
                                                 const MultiBufferPlan& plan,
                                                 const CudaSyncModelAdapter& adapter);

struct PhysicalAllocationExpectation {
  tirx::Var storage;
  int64_t versions{1};
  int64_t unversioned_bytes{0};
  // Stable provenance, assigned before physical rewrites. The final physical
  // Var need not be the same Object as storage.
  int64_t allocation_id{-1};
};

enum class PhysicalAllocationKind { kSelected, kFixed, kBarrier };

struct PhysicalAllocationRecord {
  int64_t allocation_id{-1};
  tirx::Var storage;
  PhysicalAllocationKind kind{PhysicalAllocationKind::kSelected};
  // Only leading dimensions that implement the selected ring belong here.
  // Their checked product must equal the expectation's selected version count.
  std::vector<int64_t> leading_version_extents;
  PrimExpr byte_offset;
  PrimExpr byte_size;
  int64_t required_alignment{1};
  // other allocation ID -> generation/lifetime proof token. Overlap requires
  // equal nonempty tokens in both directions in addition to BufferAliasMap.
  std::unordered_map<int64_t, ffi::String> alias_contracts;
};

struct SharedMemoryArenaPlan {
  // Includes selected, fixed, noneligible, and barrier allocations. Holes are
  // represented by offsets/arena_bytes rather than fabricated allocations.
  std::vector<PhysicalAllocationRecord> allocations;
  PrimExpr arena_bytes;
};

// Validate final post-versioning/layout/flattening extents and byte ranges.
// Dynamic sizes or offsets, arithmetic overflow, missing/stale provenance,
// unproved overlap, and missing target capacity are hard errors.
void ValidatePhysicalAllocationContracts(
    const std::vector<PhysicalAllocationExpectation>& expected,
    const BufferAliasMap& positive_alias_proof, const SharedMemoryArenaPlan& actual,
    const Target& target);

}  // namespace tvm::tl::cuda::z3

#endif  // TVM_TL_CUDA_TRANSFORM_Z3_LIFETIME_ANALYSIS_H_
