/*!
 * \file z3_cost_expression.h
 * \brief Strict native cost-expression trees for CUDA Z3 task inference.
 */
#ifndef TVM_TL_TRANSFORM_COMMON_Z3_COST_EXPRESSION_H_
#define TVM_TL_TRANSFORM_COMMON_Z3_COST_EXPRESSION_H_

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/any.h>
#include <tvm/ffi/string.h>

namespace tvm {
namespace tl {

/*! \brief An AttrStmt key or For annotation containing a non-negative trip cap. */
constexpr const char *kZ3CudaTripBound = "tl.cuda_z3_trip_bound";

/*!
 * \brief A finite native extent.
 *
 * Exact extents cross the FFI as integers. Dynamic extents cross as the
 * existing Extent object schema and therefore cannot be mistaken for measured
 * integer trip counts or payloads.
 */
struct NativeExtent {
  int64_t upper_bound{0};
  bool dynamic{false};
  std::string reliability{"measured"};

  static NativeExtent Exact(int64_t value) {
    return {value, false, "measured"};
  }

  static NativeExtent UpperBound(int64_t value,
                                 std::string reliability = "upper_bound") {
    return {value, true, std::move(reliability)};
  }

  ffi::Any ToFFI() const;
};

/*!
 * \brief A leaf/serial/repeat/branch cost tree.
 *
 * The JSON object contains only the public strict schema. Native-only resource
 * and unbounded-While audit bits are aggregated separately and are never
 * stringified into cost features.
 */
struct NativeCostExpr {
  enum class Kind { kLeaf, kSerial, kRepeat, kBranch };

  Kind kind{Kind::kSerial};
  ffi::Map<ffi::String, ffi::Any> features;
  std::vector<NativeCostExpr> children;
  NativeExtent trips{NativeExtent::Exact(0)};
  int64_t structural_resource_mask{0};
  bool contains_unbounded_while{false};

  static NativeCostExpr Leaf(ffi::Map<ffi::String, ffi::Any> features,
                             int64_t structural_resource_mask) {
    NativeCostExpr result;
    result.kind = Kind::kLeaf;
    result.features = std::move(features);
    result.structural_resource_mask = structural_resource_mask;
    return result;
  }

  static NativeCostExpr Serial(std::vector<NativeCostExpr> children) {
    NativeCostExpr result;
    result.kind = Kind::kSerial;
    result.children = std::move(children);
    result.AggregateChildren();
    return result;
  }

  static NativeCostExpr Repeat(const NativeExtent &trips, NativeCostExpr body) {
    NativeCostExpr result;
    result.kind = Kind::kRepeat;
    result.trips = trips;
    result.children.push_back(std::move(body));
    result.AggregateChildren();
    return result;
  }

  static NativeCostExpr Branch(NativeCostExpr then_expr,
                               NativeCostExpr else_expr) {
    NativeCostExpr result;
    result.kind = Kind::kBranch;
    result.children = {std::move(then_expr), std::move(else_expr)};
    result.AggregateChildren();
    return result;
  }

  bool IsLeaf() const { return kind == Kind::kLeaf; }
  ffi::Map<ffi::String, ffi::Any> ToFFI() const;

 private:
  void AggregateChildren() {
    for (const NativeCostExpr &child : children) {
      structural_resource_mask |= child.structural_resource_mask;
      contains_unbounded_while =
          contains_unbounded_while || child.contains_unbounded_while;
    }
  }
};

}  // namespace tl
}  // namespace tvm

#endif  // TVM_TL_TRANSFORM_COMMON_Z3_COST_EXPRESSION_H_
