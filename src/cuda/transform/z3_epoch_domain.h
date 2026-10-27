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
 * \file z3_epoch_domain.h
 * \brief Deterministic lexical and counter epoch-domain projection.
 */

#ifndef TVM_TL_CUDA_TRANSFORM_Z3_EPOCH_DOMAIN_H_
#define TVM_TL_CUDA_TRANSFORM_Z3_EPOCH_DOMAIN_H_

#include <map>
#include <unordered_map>
#include <vector>

#include "z3_buffer_ownership.h"

namespace tvm::tl::cuda::z3 {

enum class EpochDomainKind { kLexical, kCounter };

struct EpochDomain {
  int id{0};
  EpochDomainKind kind{EpochDomainKind::kLexical};
  ControlNode* owner{nullptr};
  PrimExpr active_guard{Bool(true)};
  tirx::Buffer counter;

  bool IsCounter() const { return kind == EpochDomainKind::kCounter; }
};

// Pass-local registry. The ScheduledTIR and MultiBufferPlan must outlive the
// registry; owner and counter objects are intentionally non-owning references.
class EpochDomainRegistry {
 public:
  EpochDomainRegistry(const ScheduledTIR& scheduled, const MultiBufferPlan& plan);

  const EpochDomain& Domain(int id) const;
  int DomainForStorage(const tirx::Var& storage, const ControlNode* scope) const;
  int UnconditionalLexicalDomain(const ControlNode* scope) const;
  bool CanProjectDomain(int source, int target, const ControlNode* scope) const;
  std::vector<int> DomainsForTaskAtScope(const TaskNode& task,
                                         const ControlNode* scope) const;

 private:
  using StorageBindings =
      std::unordered_map<tirx::Var, std::unordered_map<const ControlNode*, int>,
                         ffi::ObjectPtrHash, ffi::ObjectPtrEqual>;

  int InternDomain(EpochDomainKind kind, ControlNode* owner,
                   const PrimExpr& active_guard, const tirx::Buffer& counter = {});
  void AddLexicalScope(ControlNode* scope);
  void RegisterLexicalScopes(const std::vector<std::shared_ptr<IRStructure>>& nodes);
  void RegisterCounterDomains();
  void BuildProjectionClosure();
  void BindCounterStorage(const tirx::Var& storage, ControlNode* owner, int domain);
  int FindCounterBinding(const tirx::Var& storage, const ControlNode* owner) const;

  const ScheduledTIR& scheduled_;
  const MultiBufferPlan& plan_;
  std::vector<EpochDomain> domains_;
  std::unordered_map<const ControlNode*, int> lexical_domain_ids_;
  StorageBindings counter_bindings_;
  std::map<const ControlNode*, std::map<int, std::vector<int>>> projection_targets_;
};

}  // namespace tvm::tl::cuda::z3

#endif  // TVM_TL_CUDA_TRANSFORM_Z3_EPOCH_DOMAIN_H_
