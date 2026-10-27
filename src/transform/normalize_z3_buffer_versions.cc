/*!
 * \file normalize_z3_buffer_versions.cc
 * \brief Normalize public CUDA Z3 buffer-version requests before inference.
 */

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../op/operator.h"
#include "common/pipeline_utils.h"
#include "common/z3_buffer_versions.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace {

constexpr const char *kPassName = "NormalizeZ3BufferVersions";
constexpr const char *kZ3Pipeline = "tl_z3_pipeline";
constexpr const char *kZ3BufferVars = "tl_z3_pipeline_buffer_vars";

class RequestedAccessCollector : public StmtExprVisitor {
public:
  void VisitStmt_(const BufferStoreNode *op) final {
    writes_.insert(op->buffer->data);
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const BufferLoadNode *op) final {
    reads_.insert(op->buffer->data);
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const CallNode *op) final {
    if (auto tile_op = ParseOperator(GetRef<Call>(op)); tile_op.defined()) {
      AccessRegions access = tile_op->GetAccessRegions();
      for (const BufferRegion &region : access.reads) {
        reads_.insert(region->buffer->data);
      }
      for (const BufferRegion &region : access.writes) {
        writes_.insert(region->buffer->data);
      }
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  bool Touches(const Var &storage) const {
    return reads_.count(storage) || writes_.count(storage);
  }

  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> reads_;
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> writes_;
};

struct LoopRequests {
  const ForNode *loop{nullptr};
  RequestedAccessCollector access;
  Z3BufferVersionCountMap counts;
  Z3BufferVersionModeMap modes;
};

bool IsVersionRequestScope(const Buffer &buffer) {
  const String scope = buffer.scope();
  return scope == "shared" || scope == "shared.dyn" || scope == "local" ||
         scope == "local.var" || scope == "local.fragment" ||
         scope == "register" || scope == "shared.tmem";
}

class RequestNormalizer : public StmtExprMutator {
public:
  static Stmt Rewrite(const PrimFunc &func) {
    RequestNormalizer normalizer;
    normalizer.Collect(func);
    if (!normalizer.has_requests_) return func->body;
    normalizer.ValidateAndProject(func);
    return normalizer.VisitStmt(func->body);
  }

private:
  void Collect(const PrimFunc &func) {
    for (const auto &[_, buffer] : func->buffer_map) {
      buffers_.Set(buffer->data, buffer);
    }
    PostOrderVisit(func->body, [&](const ObjectRef &node) {
      if (const auto *block = node.as<SBlockNode>()) {
        for (const Buffer &buffer : block->alloc_buffers) {
          buffers_.Set(buffer->data, buffer);
        }
        bool has_counts = block->annotations.count(kZ3BufferVersionRequests);
        bool has_modes = block->annotations.count(kZ3BufferVersionModeRequests);
        if (!has_counts && !has_modes) return;
        ICHECK(!block->annotations.count(kZ3BufferVersionsSelected))
            << kPassName << ": cannot add new requests after selected versions "
               "have been published";
        request_blocks_.insert(block);
        has_requests_ = true;
        MergeCounts(ReadZ3BufferVersionCounts(block->annotations,
                                              kZ3BufferVersionRequests, kPassName),
                    &requests_);
        MergeModes(ReadZ3BufferVersionModes(block->annotations,
                                            kZ3BufferVersionModeRequests, kPassName),
                   &mode_requests_);
      } else if (const auto *alloc = node.as<AllocBufferNode>()) {
        buffers_.Set(alloc->buffer->data, alloc->buffer);
      } else if (const auto *decl = node.as<DeclBufferNode>()) {
        buffers_.Set(decl->buffer->data, decl->buffer);
      } else if (const auto *load = node.as<BufferLoadNode>()) {
        if (!buffers_.count(load->buffer->data)) {
          buffers_.Set(load->buffer->data, load->buffer);
        }
      } else if (const auto *store = node.as<BufferStoreNode>()) {
        if (!buffers_.count(store->buffer->data)) {
          buffers_.Set(store->buffer->data, store->buffer);
        }
      } else if (const auto *loop = node.as<ForNode>()) {
        if (!loop->annotations.count(kZ3Pipeline)) return;
        auto loop_requests = std::make_unique<LoopRequests>();
        loop_requests->loop = loop;
        loop_requests->access(loop->body);
        loops_.push_back(std::move(loop_requests));
      }
    });
  }

  static void MergeCounts(const Z3BufferVersionCountMap &source,
                          Z3BufferVersionCountMap *destination) {
    for (const auto &[storage, count] : source) {
      if (auto previous = destination->Get(storage)) {
        ICHECK_EQ(previous.value()->value, count->value)
            << kPassName << ": conflicting alias requests for " << storage;
      } else {
        destination->Set(storage, count);
      }
    }
  }

  static void MergeModes(const Z3BufferVersionModeMap &source,
                         Z3BufferVersionModeMap *destination) {
    for (const auto &[storage, mode] : source) {
      if (auto previous = destination->Get(storage)) {
        ICHECK_EQ(previous.value(), mode)
            << kPassName << ": conflicting alias mode requests for " << storage;
      } else {
        destination->Set(storage, mode);
      }
    }
  }

  void ValidateAndProject(const PrimFunc &func) {
    (void)func;
    std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> storages;
    for (const auto &[storage, _] : requests_) storages.insert(storage);
    for (const auto &[storage, _] : mode_requests_) storages.insert(storage);
    for (const Var &storage : storages) {
      auto buffer = buffers_.Get(storage);
      ICHECK(buffer.has_value())
          << kPassName << ": request refers to unknown Buffer.data Var "
          << storage;
      ICHECK(IsVersionRequestScope(buffer.value()))
          << kPassName << ": request for " << storage
          << " is outside automatic on-chip version scopes; got "
          << buffer.value().scope();

      bool bare_one = IsZ3BareOneRequest(storage, requests_, mode_requests_);
      bool has_write_owner = false;
      for (const auto &loop : loops_) {
        if (!loop->access.Touches(storage)) continue;
        if (loop->access.writes_.count(storage)) has_write_owner = true;
        if (auto count = requests_.Get(storage)) {
          loop->counts.Set(storage, count.value());
        }
        if (auto mode = mode_requests_.Get(storage)) {
          loop->modes.Set(storage, mode.value());
        }
      }
      ICHECK(bare_one || has_write_owner)
          << kPassName << ": request for " << storage
          << " has no writing CUDA Z3 owner loop";
    }

    for (const auto &loop : loops_) {
      auto explicit_vars = loop->loop->annotations.Get(kZ3BufferVars);
      if (!explicit_vars.has_value()) continue;
      Array<Var> vars = Downcast<Array<Var>>(explicit_vars.value());
      std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> available;
      for (const Var &var : vars) available.insert(var);
      for (const auto &[storage, _] : loop->counts) {
        if (IsZ3BareOneRequest(storage, requests_, mode_requests_)) continue;
        ICHECK(available.count(storage))
            << kPassName << ": requested storage " << storage
            << " is absent from explicit tl_z3_pipeline_buffer_vars";
      }
      for (const auto &[storage, _] : loop->modes) {
        ICHECK(available.count(storage))
            << kPassName << ": requested storage " << storage
            << " is absent from explicit tl_z3_pipeline_buffer_vars";
      }
    }
  }

  Stmt VisitStmt_(const ForNode *loop) final {
    For rewritten = Downcast<For>(StmtExprMutator::VisitStmt_(loop));
    for (const auto &request : loops_) {
      if (request->loop != loop) continue;
      Map<String, Any> annotations = rewritten->annotations;
      if (!request->counts.empty()) {
        annotations.Set(kZ3BufferVersionRequests, request->counts);
      }
      if (!request->modes.empty()) {
        annotations.Set(kZ3BufferVersionModeRequests, request->modes);
      }
      rewritten.CopyOnWrite()->annotations = std::move(annotations);
      break;
    }
    return rewritten;
  }

  Stmt VisitStmt_(const SBlockNode *block) final {
    SBlock rewritten = Downcast<SBlock>(StmtExprMutator::VisitStmt_(block));
    if (!request_blocks_.count(block)) return rewritten;
    Map<String, Any> annotations = rewritten->annotations;
    annotations.erase(kZ3BufferVersionRequests);
    annotations.erase(kZ3BufferVersionModeRequests);
    annotations.Set(kZ3AuditVersionRequest, requests_);
    if (!mode_requests_.empty()) {
      annotations.Set(kZ3AuditVersionModeRequest, mode_requests_);
    }
    rewritten.CopyOnWrite()->annotations = std::move(annotations);
    return rewritten;
  }

  Map<Var, Buffer> buffers_;
  Z3BufferVersionCountMap requests_;
  Z3BufferVersionModeMap mode_requests_;
  std::vector<std::unique_ptr<LoopRequests>> loops_;
  std::unordered_set<const SBlockNode *> request_blocks_;
  bool has_requests_{false};
};

} // namespace

tvm::transform::Pass NormalizeZ3BufferVersions() {
  using namespace tirx::transform;
  auto pass_func = [](PrimFunc func, const IRModule &, PassContext) {
    Stmt body = RequestNormalizer::Rewrite(func);
    if (body.same_as(func->body)) return func;
    func.CopyOnWrite()->body = std::move(body);
    return func;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.NormalizeZ3BufferVersions", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = reflection;
  refl::GlobalDef().def("tl.transform.NormalizeZ3BufferVersions",
                        NormalizeZ3BufferVersions);
}

} // namespace tl
} // namespace tvm
