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

#include "z3_lifetime_analysis.h"

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>

#include <algorithm>
#include <limits>
#include <string>
#include <unordered_set>
#include <utility>

namespace tvm::tl::cuda::z3 {

using namespace tirx;

namespace {

std::string StorageName(const Var& storage) {
  return storage.defined() ? std::string(storage->name_hint) : std::string("<undefined>");
}

bool CanProjectToDomain(int source, int target, const EpochDomainRegistry* registry) {
  if (source == target) return true;
  if (registry == nullptr) return false;
  const EpochDomain& from = registry->Domain(source);
  const EpochDomain& to = registry->Domain(target);
  // Distinct lexical loops are never interchangeable clocks. Counter
  // projection is permitted only along the registry's proved guard closure.
  if (!from.IsCounter() || !to.IsCounter()) return false;
  return registry->CanProjectDomain(source, target, to.owner);
}

void ValidateLifetimeContract(const CudaStorageLifetime& lifetime,
                              const CudaSyncModel& model,
                              const EpochDomainRegistry* registry) {
  ICHECK(lifetime.storage.defined()) << "CUDA Z3 lifetime has no storage identity";
  ICHECK_GT(lifetime.domain, 0) << "CUDA Z3 lifetime has no epoch domain";
  if (registry != nullptr) {
    const EpochDomain& domain = registry->Domain(lifetime.domain);
    ICHECK(domain.kind == lifetime.domain_kind)
        << "CUDA Z3 lifetime domain kind disagrees with the epoch registry";
    // Also validates that the scope belongs to this ScheduledTIR registry.
    (void)registry->UnconditionalLexicalDomain(lifetime.scope);
  }
  if (lifetime.domain_kind == EpochDomainKind::kCounter) {
    ICHECK(!lifetime.lexical_period.has_value())
        << "Counter epoch storage " << StorageName(lifetime.storage)
        << " cannot be assigned a lexical iteration period";
  } else if (lifetime.periodic) {
    ICHECK(lifetime.lexical_period.has_value() && *lifetime.lexical_period > 0)
        << "Periodic lexical storage " << StorageName(lifetime.storage)
        << " requires a positive explicit lexical period";
  } else {
    ICHECK(!lifetime.lexical_period.has_value())
        << "Nonperiodic storage " << StorageName(lifetime.storage)
        << " cannot carry a lexical period";
  }

  for (const GenerationLifetime& generation : lifetime.generations) {
    if (generation.guard.defined()) {
      ICHECK_LE(SideEffect(generation.guard), CallEffectKind::kPure)
          << "CUDA Z3 lifetime guards must be pure snapshots";
    }
    auto validate_events = [&](const std::vector<IterationEvent>& events) {
      for (const IterationEvent& event : events) {
        ICHECK_LT(event.site, model.sites.size())
            << "CUDA Z3 lifetime references unknown sync site " << event.site;
        const CudaSyncSite& site = model.sites[event.site];
        ICHECK(CanProjectToDomain(site.domain, lifetime.domain, registry))
            << "CUDA Z3 lifetime for " << StorageName(lifetime.storage)
            << " crosses unproved epoch-domain projection";
      }
    };
    validate_events(generation.begin_sites);
    validate_events(generation.end_sites);
  }
}

bool MutuallyExclusiveAtScope(const PrimExpr& lhs, const PrimExpr& rhs,
                              const ControlNode* scope) {
  if (!lhs.defined() || !rhs.defined()) return false;
  arith::Analyzer analyzer;
  if (scope != nullptr) scope->GetLoopBodyContext().Populate(analyzer);
  return analyzer.CanProve(!(lhs && rhs));
}

BufferAliasMap AnalyzeWithRegistry(const CudaSyncModel& model,
                                   const EpochDomainRegistry* registry) {
  CudaLifetimeGraph graph(model, registry);
  BufferAliasMap aliases;
  std::unordered_set<Var, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> seen;
  for (const CudaStorageLifetime& lifetime : model.storages) {
    ICHECK(seen.insert(lifetime.storage).second)
        << "Duplicate CUDA Z3 lifetime for storage " << StorageName(lifetime.storage);
    aliases.Set(lifetime.storage, ffi::Array<Var>{});
  }

  auto append_alias = [](BufferAliasMap* result, const Var& from, const Var& to) {
    ffi::Array<Var> values = result->at(from);
    values.push_back(to);
    result->Set(from, std::move(values));
  };
  for (size_t i = 0; i < model.storages.size(); ++i) {
    for (size_t j = i + 1; j < model.storages.size(); ++j) {
      const CudaStorageLifetime& lhs = model.storages[i];
      const CudaStorageLifetime& rhs = model.storages[j];
      if (!graph.CanAlias(lhs, rhs)) continue;
      append_alias(&aliases, lhs.storage, rhs.storage);
      append_alias(&aliases, rhs.storage, lhs.storage);
    }
  }
  ValidatePositiveAliasMap(aliases);
  return aliases;
}

int64_t CheckedProduct(int64_t lhs, int64_t rhs, const char* diagnostic) {
  ICHECK_GE(lhs, 0) << diagnostic;
  ICHECK_GE(rhs, 0) << diagnostic;
  ICHECK_EQ(lhs == 0 || rhs <= std::numeric_limits<int64_t>::max() / lhs, true)
      << diagnostic << ": int64 overflow";
  return lhs * rhs;
}

int64_t ConstantBytes(const PrimExpr& value, const char* diagnostic) {
  ICHECK(value.defined()) << diagnostic << ": missing value";
  const auto* immediate = value.as<IntImmNode>();
  ICHECK(immediate != nullptr) << diagnostic << ": dynamic values are not allowed in an exact plan";
  ICHECK_GE(immediate->value, 0) << diagnostic << ": negative values are not allowed";
  return immediate->value;
}

// ---- FFI model decoding ---------------------------------------------------

ffi::Any Required(const ffi::Map<ffi::String, ffi::Any>& map, const char* key) {
  auto found = map.find(key);
  ICHECK(found != map.end()) << "Missing CUDA Z3 model field '" << key << "'";
  return (*found).second;
}

template <typename T>
T Get(const ffi::Map<ffi::String, ffi::Any>& map, const char* key) {
  return Required(map, key).cast<T>();
}

template <typename T>
T GetOr(const ffi::Map<ffi::String, ffi::Any>& map, const char* key, T fallback) {
  auto found = map.find(key);
  if (found == map.end()) return fallback;
  if (auto value = (*found).second.try_cast<T>()) return *value;
  return fallback;
}

ffi::Map<ffi::String, ffi::Any> GetMap(const ffi::Map<ffi::String, ffi::Any>& map,
                                       const char* key) {
  return Get<ffi::Map<ffi::String, ffi::Any>>(map, key);
}

std::vector<IterationEvent> ParseEvents(const ffi::Any& value) {
  std::vector<IterationEvent> events;
  for (const ffi::Any& item : value.cast<ffi::Array<ffi::Any>>()) {
    ffi::Array<ffi::Any> event = item.cast<ffi::Array<ffi::Any>>();
    ICHECK_EQ(event.size(), 2) << "An iteration event must be [site, offset]";
    events.push_back(IterationEvent{static_cast<size_t>(event[0].cast<int64_t>()),
                                   event[1].cast<int64_t>()});
  }
  return events;
}

GenerationLifetime ParseGeneration(const ffi::Map<ffi::String, ffi::Any>& spec) {
  GenerationLifetime generation;
  generation.begin_sites = ParseEvents(Required(spec, "begins"));
  generation.end_sites = ParseEvents(Required(spec, "ends"));
  generation.guard = GetOr<PrimExpr>(spec, "guard", PrimExpr());
  generation.guard_iteration_invariant =
      GetOr<bool>(spec, "guard_iteration_invariant", true);
  return generation;
}

CudaSyncModel ParseModel(const ffi::Map<ffi::String, ffi::Any>& spec) {
  CudaSyncModel model;
  for (const ffi::Any& item : GetOr<ffi::Array<ffi::Any>>(spec, "sites", {})) {
    auto site = item.cast<ffi::Map<ffi::String, ffi::Any>>();
    model.sites.push_back(CudaSyncSite{static_cast<int>(Get<int64_t>(site, "domain"))});
  }
  for (const ffi::Any& item : GetOr<ffi::Array<ffi::Any>>(spec, "edges", {})) {
    auto edge = item.cast<ffi::Map<ffi::String, ffi::Any>>();
    model.edges.push_back(CudaSyncEdge{
        static_cast<size_t>(Get<int64_t>(edge, "src")),
        static_cast<size_t>(Get<int64_t>(edge, "dst")),
        Get<int64_t>(edge, "distance"),
        Get<bool>(edge, "strong"),
        static_cast<int>(Get<int64_t>(edge, "domain")),
    });
  }
  for (const ffi::Any& item : GetOr<ffi::Array<ffi::Any>>(spec, "storages", {})) {
    auto storage = item.cast<ffi::Map<ffi::String, ffi::Any>>();
    CudaStorageLifetime lifetime;
    lifetime.storage = Get<Var>(storage, "storage");
    lifetime.domain = static_cast<int>(Get<int64_t>(storage, "domain"));
    std::string kind = std::string(GetOr<ffi::String>(storage, "domain_kind", "lexical"));
    ICHECK(kind == "lexical" || kind == "counter")
        << "Unknown CUDA Z3 epoch-domain kind " << kind;
    lifetime.domain_kind = kind == "counter" ? EpochDomainKind::kCounter
                                              : EpochDomainKind::kLexical;
    lifetime.periodic = GetOr<bool>(storage, "periodic", false);
    if (storage.count("lexical_period")) {
      if (auto period = Required(storage, "lexical_period").try_cast<int64_t>()) {
        lifetime.lexical_period = *period;
      }
    }
    for (const ffi::Any& generation :
         GetOr<ffi::Array<ffi::Any>>(storage, "generations", {})) {
      lifetime.generations.push_back(
          ParseGeneration(generation.cast<ffi::Map<ffi::String, ffi::Any>>()));
    }
    model.storages.push_back(std::move(lifetime));
  }
  return model;
}

ffi::Any OptionalInteger(std::optional<int64_t> value) {
  return value ? ffi::Any(Integer(*value)) : ffi::Any(nullptr);
}

}  // namespace

CudaLifetimeGraph::CudaLifetimeGraph(const CudaSyncModel& model,
                                     const EpochDomainRegistry* registry)
    : model_(&model) {
  auto order_for = [&](int domain) -> IterationOrder* {
    ICHECK_GT(domain, 0) << "CUDA Z3 sync sites require positive epoch-domain IDs";
    if (registry != nullptr) (void)registry->Domain(domain);
    auto found = orders_.find(domain);
    if (found != orders_.end()) return found->second.get();
    auto order = std::make_unique<IterationOrder>(model.sites.size());
    IterationOrder* result = order.get();
    orders_.emplace(domain, std::move(order));
    return result;
  };
  for (const CudaSyncSite& site : model.sites) (void)order_for(site.domain);
  for (const CudaStorageLifetime& lifetime : model.storages) {
    (void)order_for(lifetime.domain);
    ValidateLifetimeContract(lifetime, model, registry);
  }
  for (const CudaSyncEdge& edge : model.edges) {
    ICHECK_LT(edge.src, model.sites.size()) << "CUDA Z3 sync edge has unknown source";
    ICHECK_LT(edge.dst, model.sites.size()) << "CUDA Z3 sync edge has unknown destination";
    ICHECK_GE(edge.distance, 0) << "CUDA Z3 sync distances must be nonnegative";
    ICHECK(CanProjectToDomain(model.sites[edge.src].domain, edge.domain, registry) &&
           CanProjectToDomain(model.sites[edge.dst].domain, edge.domain, registry))
        << "CUDA Z3 sync edge crosses unproved epoch-domain projection";
    ICHECK(order_for(edge.domain)->AddEdge(edge.src, edge.dst, edge.distance, edge.strong))
        << "Invalid CUDA Z3 sync edge";
  }
}

const IterationOrder* CudaLifetimeGraph::FindOrder(int domain) const {
  auto found = orders_.find(domain);
  return found == orders_.end() ? nullptr : found->second.get();
}

std::optional<int64_t> CudaLifetimeGraph::MinimumDistance(int domain, size_t src,
                                                          size_t dst) const {
  const IterationOrder* order = FindOrder(domain);
  if (order == nullptr || src >= model_->sites.size() || dst >= model_->sites.size()) {
    return std::nullopt;
  }
  return order->MinimumDistance(src, dst);
}

bool CudaLifetimeGraph::HappensBefore(int domain, size_t src, size_t dst,
                                      int64_t distance) const {
  if (distance < 0) return false;
  auto minimum = MinimumDistance(domain, src, dst);
  if (!minimum) return false;
  // A positive weighted path cannot certify same-epoch order. For later epochs
  // the all-later-distances contract permits every distance at or above minimum.
  return distance == 0 ? *minimum == 0 : distance >= *minimum;
}

bool CudaLifetimeGraph::CanAlias(const CudaStorageLifetime& lhs,
                                 const CudaStorageLifetime& rhs) const {
  if (lhs.storage.same_as(rhs.storage) || lhs.domain != rhs.domain ||
      lhs.domain_kind != rhs.domain_kind || lhs.scope != rhs.scope ||
      lhs.generations.empty() || rhs.generations.empty()) {
    return false;
  }
  const int domain = lhs.domain;
  auto happens_before = [&](size_t src, size_t dst, int64_t distance) {
    return HappensBefore(domain, src, dst, distance);
  };
  auto minimum_distance = [&](size_t src, size_t dst) {
    return MinimumDistance(domain, src, dst);
  };
  auto exclusive = [&](const PrimExpr& a, const PrimExpr& b) {
    return MutuallyExclusiveAtScope(a, b, lhs.scope);
  };
  return CanAliasGenerations(lhs.generations, rhs.generations,
                             lhs.periodic || rhs.periodic, happens_before,
                             minimum_distance, exclusive);
}

BufferAliasMap AnalyzeBufferLifetimeModel(const CudaSyncModel& model) {
  return AnalyzeWithRegistry(model, nullptr);
}

bool HasPositiveAlias(const BufferAliasMap& aliases, const Var& lhs, const Var& rhs) {
  auto found = aliases.find(lhs);
  if (found == aliases.end()) return false;
  for (const Var& candidate : (*found).second) {
    if (candidate.same_as(rhs)) return true;
  }
  return false;
}

void ValidatePositiveAliasMap(const BufferAliasMap& aliases) {
  for (const auto& [storage, compatible] : aliases) {
    ICHECK(storage.defined());
    for (const Var& other : compatible) {
      ICHECK(other.defined() && !storage.same_as(other))
          << "CUDA Z3 positive aliases cannot contain undefined or self edges";
      ICHECK(HasPositiveAlias(aliases, other, storage))
          << "CUDA Z3 positive alias map must be symmetric: "
          << StorageName(storage) << " and " << StorageName(other);
    }
  }
}

BufferAliasMap AnalyzeBufferLifetimes(ScheduledTIR* scheduled,
                                      const MultiBufferPlan& plan,
                                      const CudaSyncModel& sync_model) {
  ICHECK(scheduled != nullptr);
  EpochDomainRegistry registry(*scheduled, plan);
  BufferAliasMap aliases = AnalyzeWithRegistry(sync_model, &registry);
  scheduled->metadata.buffer_aliases = aliases;
  scheduled->metadata.has_buffer_aliases = true;
  return aliases;
}

BufferAliasMap AnalyzeBufferLifetimesWithAdapter(
    ScheduledTIR* scheduled, const MultiBufferPlan& plan,
    const CudaSyncModelAdapter& adapter) {
  ICHECK(scheduled != nullptr);
  ICHECK(static_cast<bool>(adapter)) << "CUDA Z3 sync-model adapter is empty";
  EpochDomainRegistry registry(*scheduled, plan);
  CudaSyncModel model = adapter(*scheduled, plan, registry);
  BufferAliasMap aliases = AnalyzeWithRegistry(model, &registry);
  scheduled->metadata.buffer_aliases = aliases;
  scheduled->metadata.has_buffer_aliases = true;
  return aliases;
}

void ValidatePhysicalAllocationContracts(
    const std::vector<PhysicalAllocationExpectation>& expected,
    const BufferAliasMap& positive_alias_proof, const SharedMemoryArenaPlan& actual,
    const Target& target) {
  ValidatePositiveAliasMap(positive_alias_proof);
  std::unordered_map<int64_t, const PhysicalAllocationExpectation*> expected_by_id;
  std::unordered_set<Var, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> expected_storages;
  for (const PhysicalAllocationExpectation& expectation : expected) {
    ICHECK(expectation.storage.defined()) << "Selected allocation has no storage identity";
    ICHECK_GE(expectation.allocation_id, 0) << "Selected allocation has no stable allocation ID";
    ICHECK_GT(expectation.versions, 0)
        << "Selected allocation " << StorageName(expectation.storage)
        << " must have at least one version";
    ICHECK_GE(expectation.unversioned_bytes, 0);
    ICHECK(expected_by_id.emplace(expectation.allocation_id, &expectation).second)
        << "Duplicate expected physical allocation ID " << expectation.allocation_id;
    ICHECK(expected_storages.insert(expectation.storage).second)
        << "Duplicate expected storage " << StorageName(expectation.storage);
  }

  struct ValidatedAllocation {
    const PhysicalAllocationRecord* record;
    Var contract_storage;
    int64_t offset;
    int64_t bytes;
  };
  std::vector<ValidatedAllocation> validated;
  validated.reserve(actual.allocations.size());
  std::unordered_set<int64_t> actual_ids;
  std::unordered_set<const PhysicalAllocationExpectation*> covered;
  std::unordered_set<Var, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> selected_storages;
  int64_t greatest_end = 0;

  for (const PhysicalAllocationRecord& record : actual.allocations) {
    ICHECK(record.storage.defined()) << "Physical allocation has no final storage identity";
    ICHECK_GE(record.allocation_id, 0) << "Physical allocation has no stable allocation ID";
    ICHECK(actual_ids.insert(record.allocation_id).second)
        << "Duplicate physical allocation ID " << record.allocation_id;
    const int64_t offset = ConstantBytes(record.byte_offset, "Physical allocation offset");
    const int64_t bytes = ConstantBytes(record.byte_size, "Physical allocation byte size");
    ICHECK_GT(record.required_alignment, 0) << "Physical allocation alignment must be positive";
    ICHECK_EQ(offset % record.required_alignment, 0)
        << "Physical allocation " << StorageName(record.storage) << " has unaligned offset "
        << offset << " for required alignment " << record.required_alignment;
    auto end = AddIterationOffsets(offset, bytes);
    ICHECK(end.has_value()) << "Physical allocation byte range overflows int64";
    greatest_end = std::max(greatest_end, *end);

    Var contract_storage = record.storage;
    if (record.kind == PhysicalAllocationKind::kSelected) {
      auto expectation = expected_by_id.find(record.allocation_id);
      ICHECK(expectation != expected_by_id.end())
          << "Unknown or stale selected physical allocation provenance ID "
          << record.allocation_id;
      const PhysicalAllocationExpectation* expected_record = expectation->second;
      ICHECK(covered.insert(expected_record).second)
          << "Duplicate selected allocation for stable ID " << record.allocation_id;
      contract_storage = expected_record->storage;
      ICHECK(selected_storages.insert(contract_storage).second)
          << "Duplicate selected allocation for storage " << StorageName(contract_storage);

      int64_t version_product = 1;
      ICHECK(!record.leading_version_extents.empty())
          << "Selected allocation " << StorageName(contract_storage)
          << " has no leading version dimension (including extent one)";
      for (int64_t extent : record.leading_version_extents) {
        ICHECK_GT(extent, 0) << "Leading version extents must be positive";
        version_product = CheckedProduct(version_product, extent,
                                         "Leading version-dimension product overflow");
      }
      ICHECK_EQ(version_product, expected_record->versions)
          << "Leading version-dimension product for " << StorageName(contract_storage)
          << " is " << version_product << ", expected " << expected_record->versions;
      const int64_t exact_bytes = CheckedProduct(
          expected_record->unversioned_bytes, expected_record->versions,
          "Versioned physical byte-size overflow");
      ICHECK_EQ(bytes, exact_bytes)
          << "Physical byte size for " << StorageName(contract_storage) << " is " << bytes
          << ", expected unversioned_bytes * versions = " << exact_bytes;
    }
    validated.push_back(ValidatedAllocation{&record, contract_storage, offset, bytes});
  }

  ICHECK_EQ(covered.size(), expected.size())
      << "Missing, duplicate, or stale selected physical allocation provenance";
  const int64_t arena_bytes = ConstantBytes(actual.arena_bytes, "Shared-memory arena size");
  for (const ValidatedAllocation& allocation : validated) {
    auto end = AddIterationOffsets(allocation.offset, allocation.bytes);
    ICHECK(end && *end <= arena_bytes)
        << "Physical allocation " << StorageName(allocation.record->storage)
        << " range [" << allocation.offset << ", " << (end ? *end : -1)
        << ") is outside arena of " << arena_bytes << " bytes";
  }

  for (size_t i = 0; i < validated.size(); ++i) {
    const ValidatedAllocation& lhs = validated[i];
    if (lhs.bytes == 0) continue;
    const int64_t lhs_end = lhs.offset + lhs.bytes;  // Checked above.
    for (size_t j = i + 1; j < validated.size(); ++j) {
      const ValidatedAllocation& rhs = validated[j];
      if (rhs.bytes == 0) continue;
      const int64_t rhs_end = rhs.offset + rhs.bytes;  // Checked above.
      if (lhs_end <= rhs.offset || rhs_end <= lhs.offset) continue;
      ICHECK(HasPositiveAlias(positive_alias_proof, lhs.contract_storage,
                              rhs.contract_storage))
          << "Unproved overlap between physical allocations "
          << StorageName(lhs.record->storage) << " and " << StorageName(rhs.record->storage);
      auto lhs_contract = lhs.record->alias_contracts.find(rhs.record->allocation_id);
      auto rhs_contract = rhs.record->alias_contracts.find(lhs.record->allocation_id);
      ICHECK(lhs_contract != lhs.record->alias_contracts.end() &&
             rhs_contract != rhs.record->alias_contracts.end() &&
             !lhs_contract->second.empty() && lhs_contract->second == rhs_contract->second)
          << "Mismatched actual generation/lifetime alias contracts for "
          << StorageName(lhs.record->storage) << " and " << StorageName(rhs.record->storage);
    }
  }

  ffi::Optional<Integer> capacity =
      target->GetAttr<Integer>("max_shared_memory_per_block");
  ICHECK(capacity.has_value())
      << "CUDA Z3 physical validation requires target attribute max_shared_memory_per_block";
  ICHECK_GE(capacity.value()->value, 0);
  ICHECK_LE(arena_bytes, capacity.value()->value)
      << "CUDA Z3 shared-memory arena requires " << arena_bytes
      << " bytes after all merges (including fixed and barrier allocations), exceeding target capacity "
      << capacity.value()->value;
  ICHECK_GE(arena_bytes, greatest_end);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.cuda.transform.Z3AnalyzeLifetimeModel",
                        [](ffi::Map<ffi::String, ffi::Any> spec) {
                          return AnalyzeBufferLifetimeModel(ParseModel(spec));
                        });
  refl::GlobalDef().def("tl.cuda.transform.Z3InspectSeparatingCuts",
                        [](ffi::Map<ffi::String, ffi::Any> spec) {
                          CudaSyncModel model = ParseModel(spec);
                          CudaLifetimeGraph graph(model);
                          GenerationLifetime lhs = ParseGeneration(GetMap(spec, "lhs"));
                          GenerationLifetime rhs = ParseGeneration(GetMap(spec, "rhs"));
                          int domain = static_cast<int>(GetOr<int64_t>(spec, "domain", 1));
                          auto query = [&](size_t src, size_t dst) {
                            return graph.MinimumDistance(domain, src, dst);
                          };
                          auto forward =
                              SeparateIterationEvents(lhs.end_sites, rhs.begin_sites, query);
                          auto reverse =
                              SeparateIterationEvents(rhs.end_sites, lhs.begin_sites, query);
                          std::optional<int64_t> sum;
                          if (forward && reverse) sum = AddIterationOffsets(*forward, *reverse);
                          bool valid = sum && *sum <= 1;
                          return ffi::Array<ffi::Any>{OptionalInteger(forward),
                                                      OptionalInteger(reverse),
                                                      OptionalInteger(sum), valid};
                        });
  refl::GlobalDef().def("tl.cuda.transform.Z3AttachLifetimeAliases",
                        [](const SBlock& root, ffi::Map<ffi::String, ffi::Any> spec) {
                          ScheduledTIR scheduled = DecodeScheduledTIR(root, ConstrSet());
                          MultiBufferPlan plan;
                          for (const auto& [storage, versions] :
                               scheduled.metadata.selected_versions) {
                            MultiBufferInfo info;
                            info.storage = storage;
                            info.num_versions = versions;
                            plan.AddInfo(std::move(info));
                          }
                          (void)AnalyzeBufferLifetimes(&scheduled, plan, ParseModel(spec));
                          return EncodeScheduledTIR(std::move(scheduled));
                        });
}

}  // namespace tvm::tl::cuda::z3
