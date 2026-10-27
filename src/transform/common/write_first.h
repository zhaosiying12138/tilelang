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
 * \file write_first.h
 * \brief Target-neutral four-state write-before-read lattice.
 */

#ifndef TVM_TL_TRANSFORM_COMMON_WRITE_FIRST_H_
#define TVM_TL_TRANSFORM_COMMON_WRITE_FIRST_H_

namespace tvm::tl {

// Storage-granular classification of the first access in evaluation order.
// kMaybeWriteFirst means untouched-or-write-first: a write is not guaranteed,
// but no read-before-write has been observed. kReadFirst is absorbing.
enum class AccessOrder {
  kUntouched,
  kMaybeWriteFirst,
  kWriteFirst,
  kReadFirst,
};

// Compose two sub-events in evaluation order. A guaranteed write or an unsafe
// read determines the result and absorbs the suffix. A conditional write
// followed by a read remains read-first because the write may have been skipped.
inline AccessOrder SeqCompose(AccessOrder first, AccessOrder second) {
  if (first == AccessOrder::kReadFirst || first == AccessOrder::kWriteFirst) {
    return first;
  }
  if (first == AccessOrder::kUntouched) {
    return second;
  }
  if (second == AccessOrder::kUntouched) {
    return AccessOrder::kMaybeWriteFirst;
  }
  return second;
}

// Merge mutually exclusive, exhaustive then/else branches.
inline AccessOrder MergeIfThenElse(AccessOrder then_order, AccessOrder else_order) {
  if (then_order == AccessOrder::kReadFirst || else_order == AccessOrder::kReadFirst) {
    return AccessOrder::kReadFirst;
  }
  if (then_order == AccessOrder::kUntouched && else_order == AccessOrder::kUntouched) {
    return AccessOrder::kUntouched;
  }
  if (then_order == AccessOrder::kWriteFirst && else_order == AccessOrder::kWriteFirst) {
    return AccessOrder::kWriteFirst;
  }
  return AccessOrder::kMaybeWriteFirst;
}

// Merge a then branch with its implicit untouched else branch.
inline AccessOrder MergeIfThenOnly(AccessOrder then_order) {
  if (then_order == AccessOrder::kReadFirst) {
    return AccessOrder::kReadFirst;
  }
  if (then_order == AccessOrder::kUntouched) {
    return AccessOrder::kUntouched;
  }
  return AccessOrder::kMaybeWriteFirst;
}

// A possibly skipped scope cannot provide a guaranteed first write.
inline AccessOrder DemoteIfNotMustExecute(AccessOrder order, bool must_execute) {
  if (!must_execute && order == AccessOrder::kWriteFirst) {
    return AccessOrder::kMaybeWriteFirst;
  }
  return order;
}

}  // namespace tvm::tl

#endif  // TVM_TL_TRANSFORM_COMMON_WRITE_FIRST_H_
