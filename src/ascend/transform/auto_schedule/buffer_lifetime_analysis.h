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
 * \file buffer_lifetime_analysis.h
 * \brief Ascend forwarding header for target-neutral lifetime primitives.
 */

#ifndef TVM_TL_ASCEND_TRANSFORM_AUTO_SCHEDULE_BUFFER_LIFETIME_ANALYSIS_H_
#define TVM_TL_ASCEND_TRANSFORM_AUTO_SCHEDULE_BUFFER_LIFETIME_ANALYSIS_H_

#include "transform/common/iteration_order.h"

namespace tvm::tl::ascend {

using ::tvm::tl::AddIterationOffsets;
using ::tvm::tl::AllEventsBefore;
using ::tvm::tl::AllHappenBefore;
using ::tvm::tl::CanAliasGenerations;
using ::tvm::tl::CanSeparatePeriodicLifetimes;
using ::tvm::tl::GenerationLifetime;
using ::tvm::tl::HappensBeforeQuery;
using ::tvm::tl::IsLocalGeneration;
using ::tvm::tl::IterationEvent;
using ::tvm::tl::IterationOrder;
using ::tvm::tl::MakeIterationEvents;
using ::tvm::tl::MinimumIterationDistance;
using ::tvm::tl::MutuallyExclusiveQuery;
using ::tvm::tl::SeparateIterationEvents;

}  // namespace tvm::tl::ascend

#endif  // TVM_TL_ASCEND_TRANSFORM_AUTO_SCHEDULE_BUFFER_LIFETIME_ANALYSIS_H_
