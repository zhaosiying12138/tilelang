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
 * \file z3_buffer_ownership.h
 * \brief Complete write-first storage ownership and logical CUDA epoch clocks.
 */

#ifndef TVM_TL_CUDA_TRANSFORM_Z3_BUFFER_OWNERSHIP_H_
#define TVM_TL_CUDA_TRANSFORM_Z3_BUFFER_OWNERSHIP_H_

#include "z3_role_planning.h"

namespace tvm::tl::cuda::z3 {

// Explicit owner claims use Buffer.data identities. Automatic claims are
// serialized with the same annotation so later scheduling passes can recover
// the frontier without running the write-first proof again.
constexpr const char* kMultiBufferEligible = "multi_buffer_eligible";
// Owner-external fills carry this metadata on their semantic CUDA task. The
// physical rewrite that initializes every version belongs to the lowering slice.
constexpr const char* kMultiBufferBroadcastFill = "tl.multi_buffer_broadcast_fill";

bool CanBroadcastFillToStorage(const tirx::Stmt& stmt, const tirx::Var& storage);
bool IsMultiBufferBroadcastFill(const TaskNode* task, const tirx::Var& storage);

// Infer and annotate the deepest complete owner frontier. A storage with an
// explicit claim, a requested/manual version policy, or a bare one-version
// opt-out is never automatically claimed elsewhere in the kernel.
MultiBufferOwnerMap InferCompleteOwnerFrontier(ScheduledTIR* scheduled);

// Select iteration/counter clocks, assign compatible counter groups, and
// validate complete coverage. The returned plan also contains one-version
// storages; their physical expansion is elided, not their epoch semantics.
MultiBufferPlan PrepareBufferEpochs(ScheduledTIR* scheduled,
                                    const RoleAvailability& availability);

// Revalidate a plan after scheduling or serialization. In particular, owners
// must be ordered and disjoint, counter groups must have one continuous
// structure, and no ordinary access may escape its owner.
void ValidateMultiBufferPlan(const ScheduledTIR& scheduled,
                             const MultiBufferPlan& plan);

}  // namespace tvm::tl::cuda::z3

#endif  // TVM_TL_CUDA_TRANSFORM_Z3_BUFFER_OWNERSHIP_H_
