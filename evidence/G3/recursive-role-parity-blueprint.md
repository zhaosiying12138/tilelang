# CUDA recursive scheduling and role/ownership implementation blueprint

## 0. Purpose and implementation rule

This is an implementation plan for the remaining recursive scheduling/ownership cluster in `evidence/G3/parity-84-reaudit-agent.md`: IR-001, IR-003, IR-004, PASS-001 through PASS-007, PASS-013, API-003, and the nested parts of Z3-003/Z3-015. It is intentionally not another parity status report.

The central replacement is:

> Replace the flat `Stmt` stream used for G3 inference and scheduling with a nested Task/Control tree, schedule each child list bottom-up, and project the result into the existing CUDA pipeline and typed warp-specialization lowering paths.

Do **not** port Ascend's AIC/AIV launch model, pipe mask, HBM ports, L0/SF layout rules, flag ABI, or named configuration registers. Do **not** build a second CUDA warp-specialization runtime. The implementation must reuse `PipelinePlanning`/`InjectSoftwarePipeline`, `WSRole`/`WSPipeline`/`WSScope`/`WSSchedule`, `MaterializeWSSchedule`, TVM `ConstrSet`/`arith::Analyzer`, and the current seven-bit CUDA solver resource model.

The implementation is fail-closed. Unsupported semantics must report the task and structural path; they must not disappear in flattening, silently become stage zero, silently clamp a buffer ring, or fall back to an order that was not proved valid.

## 1. Source-derived mapping and boundaries

### 1.1 Generic algorithms to carry over

| Algorithm | Ascend reference to follow | CUDA implementation mapping |
|---|---|---|
| Canonical schedule-unit grammar and codec | `src/ascend/transform/materialize_schedule_units.cc`, `auto_schedule/scheduled_tir.{h,cc}` | CUDA task markers plus a target-neutral `tl.schedule_unit`; retain declarations and loop-local buffer scope explicitly. |
| Task/Control tree, guards, clone, substitution, path order | `auto_schedule/ir_structure.{h,cc}` | New CUDA-side internal tree with role IDs and generic special-state bits, not Ascend `CoreMask` or `ResourcePipe`. |
| Condition/bound snapshots and synthetic while loops | `normalize_control_flow_for_schedule.cc`, `restore_while_loops.cc` | Same algorithms. CUDA already lowers `tl.loop_break` to `break` in `src/cuda/codegen/codegen_cuda.cc`. Restore only after actual pipeline injection. |
| Nested access/dependency analysis | `auto_schedule/task_analysis.h`, `dependency_analysis.{h,cc}` | Reuse current CUDA tile-op access extraction and scalar analysis, add guard/control-header accesses and subtree endpoints. |
| Bottom-up recursive Z3 scheduling | `src/ascend/transform/auto_schedule.cc::{ScheduleList,ScheduleRecursive,Z3SchedulePythonLoop}` | Call the existing `tl.cuda.transform.z3_schedule_loop_python` for each child list; no new solver. |
| Nested timestamps and consumer-site Let closure | `auto_schedule.cc::{GetTaskTimestamp,GetDependencyLatency,CollectLetDependencyClosure_}` | Compute absolute times from nested schedules; clone into the consumer subtree with fresh `Var`s and re-run version/capacity validation. |
| Role candidates, input fixed point, execution demand | `src/ascend/transform/assign_core.cc`, `resolve_core.cc` | Role-set fixed points over existing `WSRole` IDs. No core-mask launch semantics. |
| Four-state write-first and complete owner frontier | `src/ascend/transform/annotate_multi_buffer_eligible.cc` | Same storage-identity analysis and guard proofs over shared/TMEM/private CUDA storage. |
| Iteration/counter clocks and counter groups | `src/ascend/transform/prepare_multi_buffer.cc`, `auto_schedule/multi_buffer.h` | Same clock selection and guard compatibility; lower through existing pipeline clocks or WS persistent role-local counters. |
| Epoch domains and strong/weak lifetime separation | `auto_schedule/multi_buffer.h`, `buffer_lifetime_analysis.h`, `buffer_alias_analysis.h` | Extract the target-neutral iteration/lifetime machinery; map CUDA async completion to strong edges and issue order to weak edges. |
| Exact version and physical-allocation validation | `auto_schedule.cc::MergeSelectedBufferVersion`, `merge_ub_allocations.cc`, `buffer_addresses.h` | Recompute every dependency after augmentation, then validate final physical leading dimensions, offsets, alignment, aliasing, and target capacity. |

### 1.2 Existing CUDA/TVM structures to reuse, not recreate

1. **TVM IR and proof machinery**
   - `tvm::tirx::{Stmt,For,Bind,Buffer,BufferRegion,Range,SBlock}` and `StmtExprVisitor`/`StmtExprMutator`.
   - `src/transform/common/constr_visitor.h` for `ConstrSet`/`ConstrVisitor` and scoped Bind/loop/assume facts.
   - `arith::Analyzer`, its Z3 prover, `ffi::StructuralEqual`/`StructuralHash`, `tirx::Substitute`, `Var::copy_with_suffix`, and `PostOrderVisit`.
   - `src/transform/common/bind_utils.h`: `AnalyzeScalarBinds`, duplicatability checks, scalar use/def identity, and region helpers. Do not create a second scalar SSA graph.
   - `src/transform/common/pipeline_utils.h`: body/declaration normalization and existing region conflict queries.

2. **CUDA instruction and access classification**
   - `ClassifyStmt` and `TileStmtKind` from `src/cuda/transform/auto_schedule/common.h` for Load/MMA/Store/Worker role capabilities.
   - `ParseOperator` and `TileOperator::GetAccessRegions`, with the current strict opaque-call behavior in `src/transform/z3_task_analysis.cc`.
   - `IsLocalBuffer`/`IsFragmentBuffer` for role-private values and `IsSharedBuffer`/`IsTmemBuffer` for CTA-visible pipeline storage.
   - The existing seven solver resource bits and `tilelang/cuda/task_cost.py`. Role bits are separate from solver resource bits.

3. **Pipeline runtime lowering**
   - `src/transform/pipeline_planning.cc` and `src/transform/inject_pipeline.cc`. `PipelineInjector::VisitStmt_(ForNode)` already visits children before injecting the parent, which is exactly the required bottom-up runtime order.
   - `src/cuda/transform/multi_version_buffer_rewriter.cc`, including its existing loop-nest linearization and barrier parity handling.
   - `PipelineBufferReconciler` for views, allocation references, and leading-version reconciliation.

4. **Typed WS runtime**
   - `src/transform/common/warp_specialize.h` and `src/ir.cc`: existing role ranges, scope bodies, pipeline buffer/depth/owner-scope metadata, and sync stage values.
   - `src/cuda/transform/materialize_ws_schedule.cc`: warp-range dispatch, full/empty mbarriers, arrival counts, proxy fences, persistent role-local phase counters, CTA-local checks, and allocation lifetime.
   - `src/cuda/transform/auto_schedule/role_based_scheduler.cc` only as a reference for instruction classification and backward slicing. Do not introduce its `SchedOp`/`SchedScope` as a third scheduling tree.

5. **Solver**
   - `tilelang/cuda/transform/z3_scheduler.py` already accepts arbitrary owner-exclusion pairs and both wraparound directions. No solver feature fork is needed.

### 1.3 Genuinely new components

Only these are new because no current CUDA structure represents them:

- A nested scheduled-tree IR and its codec.
- A recursive per-list planner that retains subtree timing and absolute-time audit records.
- A role-set availability/demand solver over existing `WSRole` IDs.
- An automatic storage-owner/clock plan and epoch-domain registry.
- A CUDA sync-site/lifetime adapter that emits positive alias proofs for the existing allocator.
- Optional guarded sync metadata on `WSSync`; this is a small extension of an existing runtime node, not a new role runtime.

### 1.4 Ascend-only hardware ABI that must remain behind the boundary

Do not copy `CoreMask`, AIC/AIV bits, AIV banks, `cthread`, `ResourcePipe`, `HbmPort`, L0 pitches, L0 data/SF groups, `subcore`/`cube_domains`, Ascend `hard_event`/flag IDs, or Ascend HF32/pad/atomic/MMAD special registers.

CUDA replacements are deliberately different:

- A `RoleId` indexes an existing `WSRole`; warp ranges are used only when emitting `WSSchedule`.
- CTA shared/TMEM visibility and role-private storage determine availability. Cluster/cross-CTA storage remains unsupported unless a separate cluster protocol is later designed.
- `tl.loop_break` is the initial generic special-state instance. Mbarrier and async-group state is represented through explicit buffer/access and sync contracts, not invented “special registers.” Unknown stateful calls remain fatal unless their access contract is explicit.
- CUDA strong completion edges come from actual/planned mbarrier or synchronous completion. TMA, cp.async, and tcgen05 issue order alone is weak.
- Exact physical placement uses CUDA byte offsets, alignment annotations, and `max_shared_memory_per_block`, not Ascend arena addresses or pool constants.

## 2. Target pass architecture

Run this sequence only for modules already opting into CUDA Z3 planning. Unannotated modules and all legacy pass defaults remain unchanged.

```text
UnrollLoop + Simplify
  -> cuda.NormalizeZ3ControlFlow                 (while, condition, bounds)
  -> cuda.MaterializeZ3ScheduleUnits            (nested grammar, stable IDs)
  -> cuda.AssignZ3RoleCandidates                (capability + input fixed point)
  -> cuda.AnnotateZ3BufferOwnership             (write-first + owner frontier)
  -> InferZ3PipelineProblem                     (nested leaves; legacy wrapper preserved)
  -> Z3PipelinePlanning                         (recursive bottom-up solve)
  -> cuda.PrepareZ3BufferEpochs                 (owners, clocks, groups, counters)
  -> cuda.ResolveZ3RoleDemands                  (scalar/special/epoch/counter/exit)
  -> cuda.BuildZ3WSSchedule                     (only when multiple roles need WS)
  -> cuda.AnalyzeZ3BufferLifetimes              (epoch HB + positive alias map)
  -> cuda.LowerZ3ScheduleUnits                  (existing per-loop annotations)
  -> MaterializeWSSchedule                      (when a WS schedule was emitted)
  -> PipelinePlanning                           (consumes explicit order/stage)
  -> InjectSoftwarePipeline                     (children first; selected clock)
  -> RestoreWhileLoops                          (after real pipeline lowering)
  -> Simplify
  -> existing LayoutInference / LowerTileOp / ... / MergeSharedMemoryAllocations
       (physical contract and positive-alias checks)
```

`Z3OwnerRingPlanning` becomes an adapter for legacy explicit two-owner annotations into the same list solver and schedule validator. It must not retain an independent owner semantics implementation after the recursive path lands.

The existing direct calls to `InferZ3PipelineProblem` and `Z3PipelinePlanning` on unmarked flat loops remain compatibility wrappers during migration. Once recursive host/device tests pass, those wrappers construct the same tree and call the same implementation; the private duplicate flat algorithms can then be deleted without changing public annotation behavior.

## 3. Nested scheduled IR and codec (IR-001, IR-003, IR-004)

### 3.1 Files and namespaces

Add:

- `src/cuda/transform/z3_schedule_ir.h`
- `src/cuda/transform/z3_schedule_ir.cc`
- `src/cuda/transform/z3_schedule_units.cc`

Use `namespace tvm::tl::cuda::z3`. These are internal C++ classes, not new TVM object types and not public runtime nodes.

### 3.2 Required C++ contracts

The exact field layout may be adjusted for style, but the following semantics and API boundaries should remain stable across implementation slices.

```cpp
using RoleId = uint8_t;                     // index into the kernel's WSRole array
using RoleMask = uint64_t;                  // bit i = WSRole index i
using SpecialStateMask = uint64_t;          // generic state, not a CUDA launch mask
using BufferVersionMap = ffi::Map<tirx::Var, int>;
using BufferAliasMap = ffi::Map<tirx::Var, ffi::Array<tirx::Var>>;

enum class SpecialState : uint64_t {
  kNone = 0,
  kLoopControl = 1ULL << 0,
};

enum class TaskKind { kOperation, kDeclaration, kPerRoleGroup };

struct TaskAccessInfo {
  ffi::Array<tirx::BufferRegion> reads;
  ffi::Array<tirx::BufferRegion> writes;
  std::vector<tirx::Var> read_vars;
  std::vector<tirx::Var> write_vars;
  SpecialStateMask state_reads{0};
  SpecialStateMask state_writes{0};
};

class IRStructure {
 public:
  enum class Kind { kTask, kControl };
  virtual ~IRStructure() = default;
  virtual Kind GetKind() const = 0;
  virtual std::shared_ptr<IRStructure> Clone() const = 0;
  virtual void SubstituteVar(const tirx::Var& old_var,
                             const tirx::Var& new_var) = 0;

  IRStructure* GetParent();
  const IRStructure* GetParent() const;
  void SetParent(IRStructure* parent);       // non-owning; never this
  bool IsWithin(const IRStructure* ancestor) const;
  IRStructure* GetLowestCommonAncestor(IRStructure* other);
  std::vector<IRStructure*> PathFrom(IRStructure* ancestor = nullptr);
  IRStructure* GetChildOnPathFrom(IRStructure* strict_ancestor);

  const GuardList& GetGuards() const;
  tirx::PrimExpr GetConditionGuard() const;  // conjunction of delta conditions
  bool GuardsTouchStorage(const tirx::Var& storage) const;
  void SetGuards(GuardList guards);

  int GetStage() const;                      // -1 before scheduling
  size_t GetIndex() const;                   // current sibling position
  int64_t GetLatency() const;
  int64_t GetII() const;
  void SetSchedule(int stage, size_t index);
  void SetCost(int64_t latency, int64_t ii);

  virtual const TaskAccessInfo& Accesses() const = 0;
  virtual RoleMask CandidateRoles() const = 0;
  virtual RoleMask ResolvedRoles() const = 0;
};

class TaskNode final : public IRStructure {
 public:
  tirx::Stmt stmt;                           // includes semantic task marker when relevant
  ConstrSet outer_ctx;
  TaskKind task_kind{TaskKind::kOperation};
  bool solver_eligible{true};                // false for declarations/replayable Bind
  ffi::String ws_op_id;

  void SetCandidateRoles(RoleMask roles);
  void SetResolvedRoles(RoleMask roles);
  const ConstrSet& GetContextBeforeConditionGuards() const;
};

class ControlNode final : public IRStructure {
 public:
  tirx::For control;                         // serial or unrolled only
  std::shared_ptr<TaskNode> task;            // synthetic min/extent/step + guards
  std::vector<std::shared_ptr<IRStructure>> children;

  void InitializeLoopBodyContext();
  const ConstrSet& GetLoopBodyContext() const;
  ControlNode* GetParentControl();
  std::vector<const ControlNode*> GetAncestorControls() const;
  bool BodyHasLoopBreak() const;
  bool BodyTouchesStorage(const tirx::Var& storage) const;

  void SetIterationTiming(int64_t ii_per_iter,
                          int64_t first_iteration_span,
                          int64_t total_latency);
  int64_t GetIIperIter() const;
  int64_t GetLastIterationOffset() const;    // total - first span, checked >= 0
};

struct ScheduledTIRMetadata {
  tirx::SBlock kernel_root;
  BufferVersionMap requested_versions;
  BufferVersionMap selected_versions;
  ffi::Map<tirx::Var, ffi::String> version_modes;  // auto/iteration/counter
  BufferAliasMap buffer_aliases;
  bool has_buffer_aliases{false};            // empty proof differs from absent analysis
};

struct ScheduledTIR {
  std::vector<std::shared_ptr<IRStructure>> tree;
  ScheduledTIRMetadata metadata;
};

ScheduledTIR DecodeScheduledTIR(const tirx::SBlock& root,
                                const ConstrSet& outer_ctx);
tirx::SBlock EncodeScheduledTIR(ScheduledTIR scheduled);
void ValidateScheduledTIR(const ScheduledTIR& scheduled);

int CompareIRStructure(const IRStructure* lhs, const IRStructure* rhs);
tirx::PrimExpr CalculateIterationCount(const ControlNode* loop = nullptr);
void CollectAllTaskNodes(IRStructure* node, std::vector<TaskNode*>* tasks,
                         bool include_control_headers = false);
```

`CompareIRStructure` follows the Ascend contract: `0` for identical paths, `-2/+2` for ancestor/descendant, and `-1/+1` for ordinary lexicographic `(stage,index)` order. A null node is the virtual root. `CalculateIterationCount` accumulates in int64, subtracts each loop minimum, divides by step, and uses `ceil(extent/step)` multipliers. It returns a symbolic expression; it must not evaluate a dynamic trip count.

### 3.3 Tree invariants

1. `shared_ptr` children own the tree; `parent_` never owns. Every child and synthetic control header has a valid parent after decode, clone, insertion, reorder, and final projection. Root parents are null.
2. `index` is the current sibling position, not the original solver index. Keep a separate immutable source/solver ID when emitting annotations.
3. A `ControlNode` is atomic in its parent's list and owns an ordered child list. Its synthetic `task` represents only loop bounds, step, and guard accesses; it is not emitted as a second runtime statement.
4. Guards are ordered outermost-first deltas relative to the nearest surviving structural ancestor. `GetConditionGuard()` conjoins only condition guards. Analyzer contexts used for proofs are never used to inject freshened mutable reads into emitted IR.
5. `TaskNode::SubstituteVar` rewrites the statement, all region min/extent expressions, buffer data aliases, scalar uses/defs, guards, `outer_ctx`, role/special metadata that mentions the Var, and pass-local clone/epoch information. Use one buffer remap per substitution so all aliases remain coherent.
6. `ControlNode::Clone` recursively clones children and the header, clones pass-local timing/ownership extras, then repairs all parent pointers and indices. `TaskNode::Clone` never shallow-copies a mutable guard or pass-local object.
7. Role masks refer only to positions in the kernel's `Array<WSRole>`. Require at most 64 roles and reject bit 63 sign confusion when converting through signed annotation integers; serialize masks as decimal strings or two uint32 words if necessary.
8. `kDeclaration` covers loop-local `AllocBuffer`/`DeclBuffer` without inventing a fourth structural node. Declarations stay at their original sequence position, never enter the solver, never receive a role, and are restored before stock injection strips/registers them.
9. A replayable scalar Bind is an operation with `solver_eligible=false`, not a declaration. It remains visible to substitution and ownership analysis and is projected using the existing replayable-Bind contract.
10. Role placement lives on tasks, not schedule-unit wrappers or loops. A control's roles are derived from descendants and its synthetic header demand.
11. A Control may cache its aggregate accesses only as a mutable derived view. Every child/header/guard substitution, clone insertion, and reorder invalidates that cache before the next parent analysis; never retain references into a destroyed child list.

### 3.4 Canonical serialized grammar

Reuse `tl.schedule_unit` with a map payload. The CUDA task markers are `tl.cuda_task`, `tl.cuda_per_role_task`, and `tl.cuda_stage`.

- Every non-declaration schedule unit has integer `stage`, initially `-1` and nonnegative after scheduling.
- A leaf unit contains exactly one outer `tl.cuda_task`/`tl.cuda_per_role_task`, or one marked declaration.
- A control unit contains one serial/unrolled `For` whose body is another schedule-unit list.
- A task marker carries optional `latency`, positive `ii`, `candidate_roles`, `resolved_roles`, and stable `ws_op_id`. Public input may supply only cost; all role fields are compiler-owned.
- A control schedule unit has no core/role mask. The control's synthetic task stores header accesses in memory only.
- `T.Stage` is consumed during materialization; it is not an arbitrary guard. If any child in one list is staged, all unannotated siblings become stage zero. Each child list independently selects manual or automatic mode.
- Only `tl.assume` and `tl.assume_requires_runtime_check` are scheduling attribute guards. `tl.ws_op_id` is consumed into task/scope identity, not treated as a guard. Other attributes must be inside an opaque task or rejected with their key and path.
- An outside `IfThenElse` is flattened into complementary guarded sibling units. An `else` becomes `Not(condition)`. Conditions inside an explicit atomic Task remain structured.
- Parallel/vectorized/thread-bound loops and non-root `SBlock`s are opaque leaf tasks. Serial/unrolled loops are the only recursive controls. A bare `While` is accepted only before normalization; the scheduled decoder rejects it.
- Empty lists receive a no-op task only when needed to preserve grammar. Lowering removes that task.

## 4. Control normalization and public task API (PASS-001, PASS-002, API-003)

### 4.1 Control normalization

Add `src/cuda/transform/z3_normalize_control_flow.cc` and move the target-neutral `src/ascend/transform/restore_while_loops.cc` to `src/transform/restore_while_loops.cc` without changing its FFI name.

Apply in this order:

1. **While rewrite**

   ```text
   while cond: BODY
     =>
   for __while_w in [0, W):
     if not cond: tl.loop_break()
     BODY
   ```

   Tag the `For` with `synthetic_while=1` and do not pipeline the synthetic loop itself. `W = max(2, lcm(1..max_nested_num_stages))`, with checked LCM overflow. This makes enclosing version strides divisible by every possible ring depth and gives dependency analysis distinct adjacent epochs. Do not rewrite while loops inside opaque task leaves.

2. **Condition snapshots**

   Hoist every non-`Var` `if` condition into a preceding scalar `Bind`. This includes the newly introduced `not cond`. The snapshot must dominate every branch and any body mutation.

3. **Loop-bound snapshots**

   Hoist buffer-reading serial/unrolled `min`, `extent`, and `step` expressions into preceding `Bind`s. This preserves evaluate-on-entry semantics and gives dependency/lifetime analysis an explicit task endpoint.

`RestoreWhileLoops` remains bottom-up. Remove the synthetic Var when dead or algebraically eliminated; otherwise allocate a `local.var` counter, initialize it to the original minimum, replace residual uses with its load, and increment after the restored body. Because CUDA's injector performs the actual pipeline rewrite, call restoration **after** `InjectSoftwarePipeline`, not before it.

### 4.2 Public Python surface

Add CUDA-facing helpers following `tilelang/ascend/language/schedule_hint.py`, but emitting CUDA marker keys:

```python
def Task(*, latency: int | None = None, ii: int | None = None):
    """One atomic CUDA scheduler task; no role/warp argument is public."""

def Stage(stage: int):
    """Assign a manual stage to complete tasks in the current child list."""

def PerRoleTask():
    """One logical task with guarded alternative candidates on one resolved role."""
```

Validation:

- `latency >= 0`, `ii > 0`, and when both are present `latency >= ii`; booleans are rejected.
- `stage` is a nonnegative non-bool integer.
- `Task` cannot nest inside `Task`/`PerRoleTask`. `PerRoleTask` cannot nest. `Stage` must wrap complete scheduler tasks and cannot appear inside an atomic task.
- A compound explicit Task must have one role-capability family. CUDA's “single-pipe” contract is therefore a **single role execution contract**, not an invented hardware pipe: reject TMA-load + MMA or other combinations that require incompatible roles. Compound Worker-only serial statements remain valid.
- Every PerRole candidate uses one role-capability family, the same logical producer/consumer direction, compatible access/arrival signatures, and has no unresolved opaque side effect.
- Candidate guards retain their original nesting. For complementary/static guards, prove pairwise exclusion and collective exhaustiveness under `outer_ctx`; reject provable zero/two-execution cases. As in Ascend, arbitrary dynamic exactly-once remains an explicit user contract, but it is checked on device and never treated as a compile-time proof.

`WSSync` receives one backward-compatible extension:

```python
class WSSync(WSInstr):
    def __init__(self, kind, pipeline: str, stage: int = 0, *, guard=None): ...
```

`guard` is an optional `PrimExpr`, reflected as `ffi::Optional<tirx::PrimExpr>` on `WSSyncNode`. The four classmethod constructors accept the same keyword. Existing positional calls and serialized schedules remain valid. `WSPipeline.owner_scopes` changes from “exactly two” to “an ordered nonempty frontier”; a single entry means a counter-clock single owner, two preserves the legacy owner ring, and three or more represents the generalized frontier. Pipeline identity is the physical counter-group identity.

## 5. Nested access, dependency, and role availability (IR-004, PASS-003, Z3-003)

### 5.1 One shared access collector

Refactor the strict parts of `TaskBufferRegionCollector` into a reusable CUDA header consumed by both the legacy flat wrapper and the nested tree:

```cpp
TaskAccessInfo AnalyzeCudaTaskAccesses(
    const tirx::Stmt& stmt,
    const ffi::Map<tirx::Var, tirx::Buffer>& buffers,
    const CudaTaskAnalysisOptions& options);

void AppendExpressionAccesses(const tirx::PrimExpr& expr,
                              TaskAccessInfo* accesses);
void AppendGuardAccesses(const IRStructure& node, TaskAccessInfo* accesses);
```

Requirements:

- Preserve tile-op regions, scalar `Bind` use/def, address-of/access-ptr behavior, and strict opaque/stateful-call failures.
- Include read/write aliases by `Buffer.data`; different views of one storage conservatively conflict unless a layout-aware disjointness proof exists.
- Add all condition guards, attribute-guard expressions, loop min/extent/step, and scalar control expressions to the appropriate task/header. A buffer-dependent guard or bound must never disappear from dependencies.
- Handle mbarrier/arrive/wait and async intrinsics only through explicit access/state contracts. A pointer transport is not proof that an opaque call has no state effect.
- Aggregate a Control's child and header accesses for parent-level dependencies while retaining leaf `task_pairs` for exact timestamps.
- `loop_break` writes `kLoopControl`; every sibling in that loop reads it. A Control masks `kLoopControl` out of writes propagated to outer siblings, but propagates other special-state writes.

Reuse `AnalyzeScalarBinds` for local flat lists, then link a Bind to every consumer subtree, including nested statements, guards, and loop headers. Reject duplicate scalar definition identity and use-before-definition unless the source TIR's lexical Bind scope proves otherwise.

### 5.2 Region/dependency API

Add `src/cuda/transform/z3_dependency_analysis.{h,cc}`:

```cpp
enum class DependencyKind { kData, kOwnerExclusion };

struct DepInfo {
  IRStructure* prod_node;
  IRStructure* cons_node;
  std::optional<tirx::Var> storage;
  std::vector<std::pair<TaskNode*, TaskNode*>> task_pairs;
  int distance;                 // 0, fixed positive, or -1 automatic ring
  DependencyKind kind{DependencyKind::kData};
};

std::vector<DepInfo> AnalyzeDependencies(
    std::vector<IRStructure*> nodes,
    ControlNode* loop,
    const BufferVersionMap& manual_versions,
    const MultiBufferOwnerMap& owners,
    DependencyCache* cache);
```

Follow `RegionsMayConflict` semantics: same/cross-iteration renamings, independent producer/consumer contexts, freshened mutable reads, int64 endpoint promotion, and proof-only removal of disjoint edges. Do not reuse the flat shortcut that only compares the immediate loop Var when nested mutable control variables can be shared accidentally. A fixed manual ring tests distances from zero through the declared count. Automatic rings use `-1` and are resolved to the selected version variable in the solver problem.

For disjoint owners of one storage, emit `kOwnerExclusion` once per unordered pair. After the schedule is known, derive both directed distances from physical order and stage:

```text
forward = stage(first) - stage(second)
reverse = 1 - forward
```

Drop only redundant self-dependencies already present inside the same Control subtree. Do not drop opaque side effects.

### 5.3 Role candidates and input fixed point

Add `src/cuda/transform/z3_role_planning.{h,cc}`:

```cpp
struct CudaRoleTable {
  ffi::Array<WSRole> roles;       // existing warp-range objects
  RoleMask active_mask{0};
};

class RoleAvailability {
 public:
  explicit RoleAvailability(const std::vector<TaskNode*>& tasks);
  RoleMask ValueRoles(const tirx::Var& value,
                      const TaskNode* reader = nullptr) const;
  RoleMask ExpressionRoles(const tirx::PrimExpr& expr,
                           const TaskNode* reader = nullptr) const;
  RoleMask TaskInputRoles(const TaskNode* task) const;
};

void AssignRoleCandidates(ScheduledTIR* scheduled,
                          const CudaRoleTable& role_table,
                          const Target& target);
void ResolveRoleDemands(ScheduledTIR* scheduled,
                        const MultiBufferPlan& plan,
                        const CudaRoleTable& role_table);
```

Initial capability mapping reuses `ClassifyStmt`: eligible TMA/cp.async loads -> Load, eligible TMA stores -> Store, tcgen05 MMA -> MMA, other tile/SIMT work -> Worker. Layout-ineligible copies fall back to Worker exactly as the existing role scheduler does. Pure scalar/private-register producers begin with all active roles as candidates. Tasks with externally visible effects do not broadcast.

Iterate candidate intersection to a fixed point over:

- Scalar and private-register reaching definitions.
- Read/write region address expressions and scalar control expressions.
- Task guards and every ancestor Control header.
- Role-private values; CTA shared/TMEM data itself is visible but still needs a valid synchronization edge.
- Epoch guards and counter-group expressions after they are prepared.
- PerRole candidate intersection. The final group must resolve to one concrete role even if individual candidates retain guard-specific alternatives.

An empty intersection is an error naming the task, role path, and unavailable input. Do not assign an arbitrary role to make the schedule satisfiable.

## 6. Automatic ownership and clocks (PASS-004, PASS-006)

### 6.1 Write-first lattice

Extract the target-neutral parts of Ascend's classifier into `src/transform/common/write_first.h`; leave Ascend hardware callbacks behind. Both backends should use:

```cpp
enum class AccessOrder {
  kUntouched,
  kMaybeWriteFirst,
  kWriteFirst,
  kReadFirst,
};

AccessOrder SeqCompose(AccessOrder first, AccessOrder second);
AccessOrder MergeIfThenElse(AccessOrder then_order, AccessOrder else_order);
AccessOrder MergeIfThenOnly(AccessOrder then_order);
AccessOrder DemoteIfNotMustExecute(AccessOrder order, bool must_execute);
```

Preserve evaluation order: store value, indices, predicate, then the write; loop min/extent/step then body; guard expression before guarded payload. Read-modify-write is read-first. A skipped guaranteed write becomes `kMaybeWriteFirst`. Use manual stage order when stages are present.

For flattened scheduled guards, track `written_guard` and require every later read's active guard to imply it. Merge exhaustive complementary writes with OR and simplify proof-equivalent guards using the current control context. The lattice is storage-granular, as in Ascend; complete structural owner coverage is a separate requirement and must not be confused with a per-element overwritten-region proof.

### 6.2 Complete owner frontier

Add `src/cuda/transform/z3_buffer_ownership.{h,cc}`:

```cpp
struct MultiBufferOwnerInfo {
  ControlNode* loop{nullptr};
  tirx::PrimExpr active_guard{tirx::Bool(true)};
};
using MultiBufferOwnerMap =
    std::unordered_map<tirx::Var, std::vector<ControlNode*>,
                       ffi::ObjectPtrHash, ffi::ObjectPtrEqual>;

enum class ClockKind { kIteration, kCounter };

class MultiBufferInfo {
 public:
  tirx::Var storage;
  int num_versions{1};
  ClockKind clock{ClockKind::kIteration};
  int counter_group_id{0};       // 0 only for iteration clock
  std::vector<MultiBufferOwnerInfo> owners;
  tirx::Buffer counter;          // undefined only for iteration clock
  std::vector<int> owner_stages;

  // Null outside every owner; error on nested ownership for this storage.
  const MultiBufferOwnerInfo* FindOwner(const IRStructure& node) const;
};

class MultiBufferPlan {
 public:
  const std::vector<MultiBufferInfo>& Infos() const;
  const MultiBufferInfo* Find(const tirx::Var& storage) const;
  const MultiBufferInfo* Find(const tirx::Buffer& buffer) const;
  void AddInfo(MultiBufferInfo info);
};

MultiBufferOwnerMap InferCompleteOwnerFrontier(ScheduledTIR* scheduled);
MultiBufferPlan PrepareBufferEpochs(ScheduledTIR* scheduled,
                                    const RoleAvailability& availability);
```

For each storage, recursively compute `{Untouched,Covered,NeedsOwner}`. An ordinary access outside a descendant frontier bubbles `NeedsOwner` upward. A Control may claim the storage only when all ordinary accesses are covered and every plane is not `kReadFirst`. A row-local writer followed by a whole-buffer consumer therefore promotes ownership to the outer loop. Independent siblings can become several owners.

Validate after inference:

- Owners are pairwise disjoint, ordered in final schedule order, and never nested.
- Every ordinary task/header/guard access belongs to exactly one owner. Only an explicitly recognized owner-external broadcast fill may remain outside, and then it initializes every physical version without advancing the epoch.
- Buffer-dependent bounds were normalized; no unnormalized storage access escapes through a Control header.
- Explicit claims/manual rings/single-version opt-outs suppress automatic claims for that storage across the whole kernel. Alias members use one owner set and one version policy.
- One-version plans still carry owner/clock/sync semantics; only physical expansion is elided.

### 6.3 Clock selection and counter groups

`AnalyzeStorageEpoch` ORs direct child condition guards for accesses in one owner and records the latest guard definition. If a guard is defined at/after the first access, or accesses occur at different stages, use the unconditional lexical domain or reject an explicitly requested counter clock. Nested guards cannot escape into an outer epoch.

Select clocks as follows:

- `iteration`: exactly one owner, true active guard, and an affine path with no loop break, condition, or loop-variant bound. Slot is `CalculateIterationCount(owner) mod versions`.
- `counter`: multiple owners, a non-affine path, a non-true active guard, or explicit counter mode. All accesses in each owner have one common stage. Counter clocks cannot silently become iteration clocks when these conditions fail.
- `auto`: use counter only when required and available; single-owner affine cases can use iteration. An explicit mode is validated strictly.

Two storages share a counter group only when they have the same ordered owner loops, equivalent active guard per owner under that owner's `loop_body_ctx`, identical per-owner stages, and guards evaluable on every role participating in the combined protocol. Mutated-memory snapshots that are not proof-equivalent cannot share. Different ring depths may share a counter; slot and parity use each storage's own depth.

Counter protocol invariants:

1. One initialization dominates every owner and every participating role.
2. Exactly one guarded advance per active owner epoch occurs after the owner's last access/release and before any subsequent owner epoch. No access in that epoch observes the incremented value.
3. Every role using any storage in a group can evaluate every owner guard and has the required counter init/advance demand. The guard must denote the same active-epoch decision on every protocol role; a role-varying guard that changes participation is rejected rather than allowed to deadlock the barrier.
4. A role-local WS counter is continuous across all owners; it never resets at an owner boundary. Slot is `counter mod depth`, parity is `(counter / depth) mod 2`.
5. Iteration mode never has a counter group or owner guard. Counter mode always has a positive group ID, a defined counter, and explicit owner guards (possibly true).

For stock injection, pass a per-storage clock-expression map into `PipelineBodyRewriter`; do not let it fall back to immediate-`loop_var` modulo for a prepared plan. For counter clocks, materialize init/advance and substitute counter loads. For WS, use the existing persistent role-local counters and guarded `WSSync` commits. Both lowering paths consume the same `MultiBufferPlan` and have the same advance invariants.

## 7. Role execution demand and loop exits (PASS-007)

Use a monotone `RoleMask` demand set; no Ascend `cube_domains` are needed.

```cpp
struct DemandMaps {
  std::unordered_map<tirx::Var, RoleMask,
                     ffi::ObjectPtrHash, ffi::ObjectPtrEqual> scalar;
  std::unordered_map<tirx::Var, RoleMask,
                     ffi::ObjectPtrHash, ffi::ObjectPtrEqual> registers;
  std::unordered_map<const TaskNode*, RoleMask> special_writers;
  std::unordered_map<const TaskNode*, RoleMask> execution;
};
```

Initialize execution demand from fixed externally visible tasks. Then rescan until no bits change:

1. Readers add demand to scalar/private-register reaching definitions, guard definitions, epoch guards, and counter-group state.
2. Every Control header receives the union of descendant execution demand; its inputs propagate to their writers.
3. Special-state writers inherit readers in the same lexical scope, including writers nested in a Control. `kLoopControl` uses loop-exit demand instead of ordinary special-state propagation.
4. A loop break inherits the enclosing subtree's execution demand and any demand reaching the loop from later consumers of its counters/scalars. All roles participating in the loop/protocol must evaluate its break condition and exit together; a role-specific early exit around shared mbarriers is rejected.
5. Counter/epoch users implicitly read the guard and counter state even before final materialization inserts the loads/stores.
6. Required roles must be a subset of the candidate mask. Pure scalar/private producers may be replicated into every demanded role with fresh per-role SSA; externally visible or multi-role-incompatible effects cannot.
7. After convergence, assign unconstrained pure tasks a deterministic fallback: Worker when legal, otherwise the lowest active role ID. Re-run the fixed point once. No final task has an empty resolved mask.

This is the generic Ascend demand algorithm without subcore-specific projection. Loop-exit union semantics are the smallest CUDA rule consistent with role branches and CTA synchronization.

## 8. Recursive Z3 scheduling and Let/control reconstruction (PASS-005, Z3-003, Z3-015)

### 8.1 Planner API

Refactor `src/transform/z3_pipeline_planning.cc` around a CUDA nested planner exposed in a small internal header so the owner-ring adapter can reuse it:

```cpp
struct ListSolveResult {
  ffi::Array<int64_t> starts;
  ffi::Array<int64_t> versions;
  int64_t ii{0};
  std::vector<int64_t> raw_stages;
  std::vector<int64_t> residues;
  std::vector<int> physical_order;
};

class RecursiveZ3Planner {
 public:
  RecursiveZ3Planner(ScheduledTIR* scheduled, Z3PlannerOptions options);
  void Run();
  ListSolveResult SolveList(const ListProblem& problem);
  ListSolveResult ValidateSchedule(const ListProblem& problem,
                                   ListSolveResult result);
  void ValidateGlobalVersionAndCapacityContracts() const;
 private:
  void ScheduleList(std::vector<std::shared_ptr<IRStructure>>& children,
                    ControlNode* loop);
  void ScheduleNode(std::shared_ptr<IRStructure>& node);
  void FinalizeControlTiming(ControlNode* control,
                             const ListSolveResult& result);
  void MaterializeRequiredScalarClones(ControlNode* control,
                                       ListSolveResult* result);
};
```

### 8.2 Bottom-up algorithm

For each list:

1. Save original order, assign source indices and parents, and determine manual/automatic mode from uniform stage metadata.
2. Recurse into every Control child. Its descendants must be fully scheduled before the parent problem is built.
3. Build leaf/subtree dependencies. For each `DepInfo`, compute latency as the maximum over actual leaf task pairs:

   ```cpp
   int64_t TaskTimestamp(IRStructure* subtree_root,
                         const TaskNode* task,
                         bool producer_side) const;
   int64_t DependencyLatency(const DepInfo& dep, bool reverse = false) const;
   ```

   Walking the path from subtree root to leaf, add each child's start. For a producer, also add each ancestor Control's `last_iteration_offset`; for a consumer, add zero. This covers nested producer/consumer timestamps instead of flattening them into a whitelist.

4. Call the existing loop solver with latencies, IIs, seven-bit resources, data/resource/owner/stage/pipe dependencies, storage sizes, and memory groups. Owner exclusions are no longer rejected or routed through a separate solver implementation.
5. Validate the returned `start = raw_stage * II + residue`, including manual stages, every data/resource/owner inequality, version lower bounds, and memory capacity. Use `__int128` in C++ validation and preserve int64 audit values.
6. Find cross-stage scalar dependencies. Add required consumer clones and re-solve the augmented list to a fixed point, with a checked finite definition/consumer bound and a configurable round limit. Recompute versions and capacity after every augmentation.
7. Normalize the minimum raw stage to zero, derive physical order by residue, descending stage at equal residue, then stable source order. Retain the existing zero-latency topological tie checks for scalar/data/pipe constraints.
8. Set child stages/start metadata and reorder children. Reindex and repair parents.
9. For a Control, compute:

   ```text
   first_iteration_span = max(child.start + child.latency)
   total_latency = (trip_count == 0) ? 0
                 : first_iteration_span + (trip_count - 1) * II
   last_iteration_offset = (trip_count == 0) ? 0
                         : total_latency - first_iteration_span
   control.II = control.latency = total_latency
   control.IIperIter = II
   ```

   Use checked arithmetic and zero-trip-safe formulas. Static trip counts are derived from min/extent/step. If a trip count is not static, require an explicit control total-latency override; never substitute a heuristic such as 100 for proof. An override must be at least `first_iteration_span`, and for static loops must equal the derived value.

The root list is the non-loop straight-line case: only stage zero is valid, dependencies use `loop=null`, and nonidentity reorder must pass the existing safety checks for declarations, scalar Bind scope, buffer dependencies, and non-buffer side effects.

### 8.3 General consumer-subtree scalar clones

Keep the existing cloneable-purity checks. For each needed definition/consumer pair:

1. Clone the complete defining `TaskNode`, including guards, context, reads, cost, and pass-local metadata. Create a fresh `Var` for the Bind result.
2. Place the clone in the consumer's stage immediately before the consumer root in the **same child list**. If the consumer is a Control, substitution covers its header, guards, all descendants, and nested region expressions.
3. Build the whole stage's clone set first, rewire chained clone inputs to fresh Vars, then emit definitions before uses. A diamond shares the clone for one definition/consumer chain; independent consumers receive independent fresh Vars.
4. Add virtual buffer reads and all RAW/WAR/WAW edges for the clone. Include loop/control and async guards instead of rejecting async augmentation merely because the old flat metadata could not represent it.
5. Keep noncloneable or side-effecting definitions in the same stage as every consumer. Zero-latency chains still use topological physical order.
6. Store the actual cloned Bind in the scheduled tree. It becomes an ordinary statement for stock pipeline lowering; do not require `MaterializeScheduledBindClones` to rediscover it from flat metadata.

This direct-tree materialization is the fix for the current restricted flat clone path and its nested/control limitations. The legacy flat clone annotation remains only as a compatibility input until its wrapper is retired.

### 8.4 Version/capacity and absolute-time retention

For every negative-distance edge, recompute:

```text
required_versions = max(1, ceil((latency + start[source] - start[target]) / II))
```

when the numerator is positive. Apply the final selected version to every data and owner inequality. Never preserve a larger solver value if it violates final physical capacity merely for compatibility.

When independent owners select versions for one storage, merge with `max`, propagate to all alias members, and rerun final global validation after all lists are scheduled. A parent solver sees the versions selected by its already-scheduled children. If the final merge makes an earlier schedule invalid or exceeds a pool, fail with both owner paths; do not copy Ascend's documented “accepted overflow limitation.”

Retain existing flat audit annotations for compatibility, and add per-Control records aligned with the complete augmented child list:

```text
tl_z3_pipeline_audit_ii
tl_z3_pipeline_audit_child_paths       # stable source paths, including clones
tl_z3_pipeline_audit_child_starts      # unnormalized absolute starts, int64
tl_z3_pipeline_audit_child_stages
tl_z3_pipeline_audit_child_residues
tl_z3_pipeline_audit_first_span
tl_z3_pipeline_audit_last_iter_offset
```

These records preserve raw start/stage/residue/physical-time reconstruction at every nesting level, not only a flat `cuda_depth`.

### 8.5 Projection into stock pipeline annotations

`cuda.LowerZ3ScheduleUnits` recursively restores each list and emits the existing `tl_pipeline_order`/`tl_pipeline_stage` for every direct operation or Control child, plus `num_stages`, `tl_pipeline_buffer_versions`, and the prepared async/clock metadata required by injection. Declarations and replayable Bind statements retain their existing filtering metadata. Control children already carry their own annotations, so `PipelinePlanning` can consume explicit annotations without learning tree scheduling. When a prepared recursive plan is present, `PipelinePlanning` must not add legacy stage-level `software_pipeline_async_stages` or replace per-statement async/owner metadata; the explicit recursive annotations and existing async fields are authoritative.

The injector then processes child loops before parents. Prepared clock maps and actual clones survive this projection. After injection, strip schedule-unit-only metadata and restore synthetic while loops.

## 9. WS schedule construction and runtime changes (PASS-003, PASS-006, API-003)

Add `src/cuda/transform/z3_ws_schedule_builder.cc`; modify only the WS files assigned to the runtime slice in section 12.

Build `WSSchedule` from the resolved tree:

- Reuse the active `WSRole` table. Factor the existing private `PlanRoles` logic from `role_based_scheduler.cc` into a small `PlanCudaRoleLayout(active_roles, worker_warps, widen_cpasync_load)` helper used by both legacy auto-WS and this builder, preserving warp ranges and register-budget policy instead of duplicating it. Do not infer or change AIC/AIV-like hardware masks.
- Give every Task/Control a stable `tl.ws_op_id`. A `WSScope` exists for the root and every Control with participating descendants. Each role body follows final physical order and includes only resolved tasks and relevant child scopes.
- Pure scalar/private values needed by several roles are emitted with fresh role-local Vars, using the demand map. `MaterializeWSSchedule` continues to perform its existing SSA substitution.
- For each cross-role storage handoff, construct the existing full/empty `WSPipeline`. Keep a single logical producer role and consumer role per ordered protocol; represent fan-out as separately proved protocols or reject it before emission, never as an untracked shared barrier.
- An owner-frontier pipeline lists every ordered owner scope. Counter groups map one-to-one to `WSPipeline`s. A single-owner counter uses one scope; legacy two-owner schedules are unchanged; generalized frontiers use the same continuous phase arithmetic.
- Sync stages come from the scheduled child stage. The selected physical owner order determines acquire/commit versus wait/release directions and forward/reverse hand-offs.
- `WSSync.guard` carries the prepared active-epoch guard. Counter increments and arrive/wait emission inherit it. Guard definitions must dominate the sync and be available on every protocol role.
- For a PerRole group, duplicate the logical sync next to each candidate under that candidate's original guard. Keep a wider outer guard outside candidates when it belongs to an enclosing epoch. Candidates must have compatible `BarrierSidePlan`s so exactly one active candidate produces exactly the planned arrival count.

Generalize `MaterializeWSSchedule` rather than bypassing it:

1. Accept nonempty `owner_scopes` of any length instead of exactly two and derive loops over the vector instead of `first`/`second` indexing.
2. Replace hard-coded two-role checks with validation of the actual protocol roles. A pipeline may involve one fixed producer and one fixed consumer role; other roles need not enter the owner frontier.
3. Permit owner guards and normalized synthetic-while owners when a prepared counter clock exists; the actual While form is restored only after injection. Continue rejecting nested owners and any unproved external storage access.
4. Use `WSSync.guard` around wait/arrive and counter increment. Every epoch must still have exactly one matched open/close per active role; candidates share one logical span.
5. Use the general arrival planner per owner/protocol group. Async atoms are allowed only when their completion is represented in the lifetime graph and every active epoch has a fixed arrival signature. Mismatched signatures are rejected, not hidden by aggregate use sets.
6. Import the prepared counter identity when present, or create the existing role-local counter as an implementation detail. Never emit both a generic advance and a WS advance for the same epoch.
7. Keep CTA-local storage, cluster, proxy-fence, `setmaxnreg`, and barrier initialization checks. These are CUDA ABI constraints, not generic ownership rules.

## 10. Epoch lifetime and exact physical plans (PASS-013, PASS-005)

### 10.1 Shared iteration-order machinery

Move the target-neutral contents of `src/ascend/transform/auto_schedule/buffer_lifetime_analysis.h` into `src/transform/common/iteration_order.h`, leaving a forwarding header for Ascend. Reuse unchanged:

- `IterationEvent` and checked offset addition.
- Two-state `IterationOrder` Dijkstra reachability, where state 1 has crossed a strong completion edge.
- `MinimumIterationDistance` and its all-later-distances contract.
- `SeparateIterationEvents`, `CanSeparatePeriodicLifetimes`, `GenerationLifetime`, and `CanAliasGenerations`.

This code contains no Ascend hardware dependency and should not be duplicated.

### 10.2 Epoch registry and CUDA sync-site adapter

Add `src/cuda/transform/z3_epoch_domain.{h,cc}` for the registry and `src/cuda/transform/z3_lifetime_analysis.{h,cc}` for the CUDA sync-site adapter:

```cpp
enum class EpochDomainKind { kLexical, kCounter };

struct EpochDomain {
  int id{0};
  EpochDomainKind kind;
  ControlNode* owner{nullptr};
  tirx::PrimExpr active_guard;
  tirx::Buffer counter;
};

class EpochDomainRegistry {
 public:
  EpochDomainRegistry(const ScheduledTIR& scheduled,
                      const MultiBufferPlan& plan);
  const EpochDomain& Domain(int id) const;
  int DomainForStorage(const tirx::Var& storage,
                       const ControlNode* scope) const;
  int UnconditionalLexicalDomain(const ControlNode* scope) const;
  bool CanProjectDomain(int source, int target,
                        const ControlNode* scope) const;
  std::vector<int> DomainsForTaskAtScope(const TaskNode& task,
                                         const ControlNode* scope) const;
};

BufferAliasMap AnalyzeBufferLifetimes(ScheduledTIR* scheduled,
                                      const MultiBufferPlan& plan,
                                      const CudaSyncModel& sync_model);
```

Intern lexical domains for every sibling scope and counter domains per concrete owner/guard/counter. Guard implication defines projection; compute transitive closure and deterministic domain IDs. A counter epoch never converts implicitly to a lexical iteration period.

`CudaSyncModel` is the backend adapter:

- A completed synchronous operation or a chain containing a real wait/release/full-empty handshake supplies a strong edge.
- Same-role issue order for TMA/cp.async/tcgen05 before a completion edge is weak. A path containing only weak edges cannot authorize reuse.
- `WSSync` entries and stock pipeline async metadata provide planned strong sites with owner/domain/guard mappings. Both endpoints of a dependency edge use one active-epoch domain; guarded projection requires proof of guard coverage.
- Build generation begin/end sites from actual writes, reads, loop boundaries, and last uses. Promote child summaries only when clock mapping is proved. A changing guard cannot be shifted between generations without an explicit mapping.

For periodic generations, require both separating cuts and `forward + reverse <= 1`. For nonperiodic generations require exactly one direction to be strongly ordered. Empty evidence, overflow, unknown distance, weak-only order, or unproved guards rejects reuse.

Emit `tl.buffer_alias_map` with a presence bit in scheduled metadata. Presence with an empty map means “analysis ran; no unlisted reuse is allowed,” which differs from an absent legacy annotation.

### 10.3 Final physical validation

Extend `MergeSharedMemoryAllocations` for recursive-Z3-produced functions with a metadata contract and a small validator:

```cpp
struct PhysicalAllocationExpectation {
  tirx::Var storage;
  int64_t versions;
  int64_t unversioned_bytes;
};

void ValidatePhysicalAllocationContracts(
    const std::vector<PhysicalAllocationExpectation>& expected,
    const BufferAliasMap& positive_alias_proof,
    const SharedMemoryArenaPlan& actual,
    const Target& target);
```

Validate after versioning/layout/flattening, when actual allocation extents are known:

- Every selected storage maps to exactly one physical allocation provenance record; duplicate/unknown/stale provenance fails. Assign a stable allocation ID before lowering and thread it through `PipelineBufferReconciler`, WS `PlanVersionedBuffers`, layout rewrites, and arena merging; final validation uses that ID rather than assuming the original `Buffer.data` pointer survived remapping.
- The product of leading version dimensions equals the selected count for that ring level. No double expansion or silently dropped extent-one-only pseudo-version is accepted.
- Physical byte size equals the final layout's unversioned byte size times selected versions, with checked sub-byte rounding and alignment. Dynamic sizes are rejected for an automatic exact plan rather than converted to zero.
- Every offset meets the allocation/TMA/MMA alignment contract; every `[offset, offset + bytes)` is within the arena and target `max_shared_memory_per_block`. Holes are legal. Include barrier storage and fixed/noneligible allocations in the total.
- Two allocations may share an address range only when their storage pair is present in the positive alias proof and their actual generation/lifetime contracts match. Existing lexical reuse remains unchanged for legacy functions; recursive-Z3 functions require the proof map.
- Recompute global totals after all merges. A schedule-time memory-group sum is only a solver constraint, not a substitute for final physical validation.

## 11. Row-to-implementation map

| Rows | Implementation closure |
|---|---|
| IR-001 | Canonical nested grammar, declaration handling, guard whitelist, per-list stage normalization, stable role/owner metadata, codec round-trip validation. |
| IR-003 | Task/Control classes, deep clone/substitution/extra propagation, parent/path/LCA APIs, ancestor-aware compare, symbolic nested iteration, flattened leaf iteration that preserves declarations and Control atomicity. |
| IR-004 | Complete buffer/scalar/control/special-state accesses, resource and role availability, storage identity, role candidate and execution-demand fixed points. |
| PASS-001 | Synthetic bounded while, condition/bound snapshots, post-injection restoration, residual live counter and loop-exit demand. |
| PASS-002 | Atomic Task wrapping, guard/stage normalization, single-role execution validation, PerRole group contract, stable IDs. |
| PASS-003 | CUDA instruction-to-role candidates, scalar/private/control availability fixed point, deterministic WSSchedule projection. |
| PASS-004 | Four-state write-first, guard coverage, complete disjoint owner frontier, automatic claims and full external-access validation. |
| PASS-005 | Bottom-up recursive solve, nested dependency timestamps, direct subtree clones, exact selected-version/capacity checks, final physical allocation contracts. |
| PASS-006 | Automatic iteration/counter selection, active guards, compatible counter groups, init/advance rules, continuous N-owner WS phases and stock clock lowering. |
| PASS-007 | Scalar/private/special/epoch/counter demand fixed point and role-uniform loop-exit analysis with deterministic fallback. |
| PASS-013 | Epoch domains, strong/weak sync graph, generation lifetimes, periodic separating-cut proof, positive-only alias map. |
| API-003 | Public Task/Stage/PerRoleTask, candidate guards and single-role checks, guarded sync migration, exact planned arrivals, existing warp-range dispatch. |
| Z3-003 | Buffer and scalar RAW/WAR/WAW plus guard/control/special-state closure, nested task-pair endpoints and clone virtual dependencies. |
| Z3-015 | Nested absolute starts, stage/residue/physical order, Control total span and last-iteration offset, general clone/control reconstruction, int64 per-Control audit records. |

## 12. Parallel implementation slices

The dependency gate is the API, not completion of every test. Once section 3 types and section 6 plan structs land, the analysis, planner, and runtime slices can proceed concurrently. Each slice exclusively owns its listed existing files; new test files avoid shared-test edit conflicts.

### Slice A — shared IR and generic lifetime primitives (foundation)

**New files**

- `src/cuda/transform/z3_schedule_ir.{h,cc}`
- `src/transform/common/write_first.h`
- `src/transform/common/iteration_order.h`
- `testing/python/transform/test_tilelang_cuda_z3_schedule_ir.py`

**Existing files owned**

- `src/ascend/transform/auto_schedule/buffer_lifetime_analysis.h` (forwarding include)
- `src/ascend/transform/annotate_multi_buffer_eligible.cc` (only if needed to consume the shared lattice)
- No other slice edits Ascend files.

**Deliver**

Tree/clone/substitute/path/codec APIs, role masks, generic write-first and iteration-order extraction, round-trip and ancestor/clone tests. Do not register CUDA passes yet.

### Slice B — normalization, task frontend, and restoration

**New files**

- `src/cuda/transform/z3_normalize_control_flow.cc`
- `src/cuda/transform/z3_schedule_units.cc`
- `src/transform/restore_while_loops.cc` (moved)
- `testing/python/transform/test_tilelang_cuda_z3_control_flow.py`
- `testing/python/transform/test_tilelang_cuda_z3_schedule_units.py`

**Existing files owned**

- `src/ascend/transform/restore_while_loops.cc` (delete after move)
- `tilelang/language/annotations.py`
- `tilelang/language/__init__.py` (only public Task/Stage/PerRoleTask exports)

**Depends on:** Slice A.

**Deliver**

All normalization rules, declaration-preserving materialization, public frontend validation, restore tests. No pipeline ordering changes.

### Slice C — nested access/dependencies and role fixed points

**New files**

- `src/cuda/transform/z3_dependency_analysis.{h,cc}`
- `src/cuda/transform/z3_role_planning.{h,cc}`
- `testing/python/transform/test_tilelang_cuda_z3_nested_task_analysis.py`
- `testing/python/transform/test_tilelang_cuda_z3_role_planning.py`

**Existing files owned**

- `src/transform/z3_task_analysis.cc`
- `src/transform/z3_task_analysis.h`

**Depends on:** A; B for end-to-end tests only.

**Deliver**

Shared access collector, nested dependency API, role candidate/input/execution-demand implementation, legacy flat inference wrapper. Do not edit the recursive planner or WS materializer.

### Slice D — ownership, clocks, and epoch registry

**New files**

- `src/cuda/transform/z3_buffer_ownership.{h,cc}`
- `src/cuda/transform/z3_epoch_domain.{h,cc}` (registry and epoch projection)
- `testing/python/transform/test_tilelang_cuda_z3_buffer_ownership.py`
- `testing/python/transform/test_tilelang_cuda_z3_buffer_epochs.py`

**Existing files owned:** none beyond new files.

**Depends on:** A and C's `RoleAvailability` API.

**Deliver**

Write-first frontier, clock selection, counter groups, plan validation, epoch-domain projection. Publish `MultiBufferPlan` and `EpochDomainRegistry` for C/E/F/G.

### Slice E — recursive solver and projection

**New files**

- `testing/python/transform/test_tilelang_cuda_z3_recursive_planning.py`

**Existing files owned**

- `src/transform/z3_pipeline_planning.cc`
- `src/transform/z3_owner_ring_planning.cc`

**Depends on:** A, C, D.

**Deliver**

Bottom-up solver, nested timestamps, direct subtree clones, augmented validation, global version merge, per-Control audit, legacy owner-ring adapter, `cuda.LowerZ3ScheduleUnits` implementation. Do not edit injector files; publish the exact projection/clock-map contract for Slice F.

### Slice F — WS construction and materializer

**New files**

- `src/cuda/transform/z3_ws_schedule_builder.cc`
- `src/cuda/transform/auto_schedule/role_layout.{h,cc}`
- `testing/python/transform/test_tilelang_cuda_z3_ws_recursive.py`

**Existing files owned**

- `src/cuda/transform/materialize_ws_schedule.cc`
- `src/cuda/transform/auto_schedule/role_based_scheduler.cc` (factor out role-layout construction only)
- `src/transform/common/warp_specialize.h`
- `src/ir.cc`
- `tilelang/language/warp_specialize.py`

**Depends on:** A/C/D APIs and E's final order/stage output for integration tests.

**Deliver**

WSSchedule emission, optional `WSSync.guard`, N-owner counter phases, PerRole sync migration, arrival validation, pure scalar role projection. No pipeline or allocator edits.

### Slice G — stock injection clocks and while restoration integration

**New files**

- `testing/python/transform/test_tilelang_cuda_z3_pipeline_projection.py`

**Existing files owned**

- `src/transform/pipeline_planning.cc`
- `src/transform/inject_pipeline.cc`
- `src/cuda/transform/multi_version_buffer_rewriter.cc`

**Depends on:** B, D, E.

**Deliver**

Consume recursive per-loop annotations, use prepared lexical/counter clocks, preserve direct clones, child-before-parent injection, and post-injection restoration hooks. Suppress legacy stage-level async synthesis when a prepared recursive plan already supplies async/owner metadata. Keep legacy behavior when no prepared clock map is present.

### Slice H — lifetime graph and physical allocator

**New files**

- `src/cuda/transform/z3_lifetime_analysis.{h,cc}`
- `testing/python/transform/test_tilelang_cuda_z3_epoch_lifetimes.py`
- `testing/python/transform/test_tilelang_cuda_z3_physical_plan.py`

**Existing files owned**

- `src/transform/merge_shared_memory_allocations.cc`

**Depends on:** A/D and F's `CudaSyncModel` description; G for final physical integration tests.

**Deliver**

Strong/weak site graph, generation aliases, positive alias map, final version/offset/alignment/capacity validator.

### Slice I — registration, pipeline integration, and device acceptance

**New files**

- `testing/python/language/test_tilelang_cuda_z3_recursive_pipeline.py`
- `testing/python/language/test_tilelang_cuda_z3_per_role_integration.py`

**Existing files owned**

- `CMakeLists.txt` and `src/cuda/CMakeLists.txt` only if new subdirectories require glob updates
- `tilelang/cuda/pipeline.py`
- `tilelang/transform/__init__.py`
- `tilelang/cuda/transform/__init__.py`

**Depends on:** all implementation slices.

**Deliver**

Pass registration/order, config gating, build integration, complete host regression and device acceptance. Only this slice changes pipeline ordering, avoiding conflicts with algorithm work.

## 13. Host acceptance tests

All tests below are compile/IR/solver tests and require no GPU. Existing flat Z3, clone, async, owner-ring, stock pipeline, and Ascend host suites must also remain green.

### 13.1 IR, codec, and control flow

`test_tilelang_cuda_z3_schedule_ir.py`

- Round-trip a root with nested Task/Control/declaration nodes, complementary guards, requested/selected versions, aliases, role metadata, and manual stages. Assert structural equality after decode/encode and all parent/index invariants.
- Clone a chain and diamond; mutate a cloned Var and prove source regions/guards/extra metadata are unchanged while every clone pointer/context is repaired.
- Check path/LCA and `CompareIRStructure` ordinary and `-2/+2` ancestor results. Check symbolic iteration for nonzero min, step 2, and int64 extents.
- Reject unsupported attr guards, unmarked serial loops/While, parallel Control, malformed task markers, duplicate `ws_op_id`, mixed staged/unscheduled lists, invalid role masks, and out-of-scope declarations.

`test_tilelang_cuda_z3_control_flow.py`

- Port the Ascend condition and loop-bound snapshot tests: the snapshot precedes every mutation and retains the correct buffer dependency.
- Normalize/restore nested While with a break. Check no counter for a dead synthetic Var, counter initialization/increment for a residual Var, and elimination after version-index simplification.
- Verify While inside an opaque parallel/SBlock task is not rewritten.

`test_tilelang_cuda_z3_schedule_units.py`

- If/else becomes complementary guarded siblings; partial manual stages normalize per child list; nested lists choose manual/auto independently.
- Explicit and inferred PerRole candidates preserve all conditions. Mixed valid compound Worker Tasks remain atomic.
- Reject cross-role compound Task, nested Task/PerRole/Stage, multiple task payloads in one unit, and unmarked non-declaration leaves.

### 13.2 Nested analysis, roles, ownership, and epochs

`test_tilelang_cuda_z3_nested_task_analysis.py`

- Port RAW/WAR/WAW and affine-distance cases across a nested Control and its header. Include guard-only reads, mutated loop-bound snapshots, aliases, and opaque side effects.
- A nested `loop_break` orders all same-loop siblings but does not fence outer siblings; ordinary special-state writes still cross Control boundaries.
- Chained/diamond Let uses in a child Control, its bounds, and its guards produce exact task-pair dependencies and maximum-latency dedup.

`test_tilelang_cuda_z3_role_planning.py`

- Copy destination/instruction selection chooses expected role candidates; layout-ineligible TMA falls back to Worker.
- A role-private guard cannot control an incompatible role. A scalar/global predicate replicates legally; an externally visible effect cannot broadcast.
- Fixed-point cases for counter guards, ancestor bounds, private-register chains, fallback determinism, and empty-intersection diagnostics.
- Port loop-exit tests with two loop roles and a later third-role consumer of a counter/predicate. Assert the break and all control producers include the required downstream role.

`test_tilelang_cuda_z3_buffer_ownership.py`

- Port `write-first`, `read-first`, `same-guard`, `disjoint-guards`, `exhaustive-writes`, `manual-stage-order`, nested alias, row-write/whole-read, and explicit-policy tests from Ascend.
- Verify three disjoint sibling owners form one ordered frontier; nested owners, owner-external ordinary reads, unnormalized bounds, and nested owner claims fail.
- Check owner-external fill initialization of every version and single-version owner semantics.

`test_tilelang_cuda_z3_buffer_epochs.py`

- Port affine/guarded/sibling clock selection, iteration-single-owner rejection, equivalent/different/mutated guard grouping, cross-stage rejection, and late-definition fallback.
- Check one reset and one guarded advance per owner, continuous counters across three owners, independent depths on one counter, and incompatible arrival groups splitting instead of merging.

### 13.3 Recursive planner and projection

`test_tilelang_cuda_z3_recursive_planning.py`

- Patch the solver to capture per-list inputs and return known starts. Assert child lists are solved before parent lists and nested `DependencyLatency` includes producer last-iteration offsets and consumer first-iteration offsets.
- Check Control total span/II, static zero/single-trip handling, dynamic-trip override requirements, int64 starts/versions, and override mismatch rejection.
- Test physical-order ties, zero-latency topological cycles, owner-exclusion forward/reverse validation, and no UNSAT fallback.
- Test direct chain/diamond/zero-latency/async-guard clones through a Control subtree. Assert fresh Vars in loop bounds and guards, defs-before-uses, virtual dependencies, and version/capacity recomputation.
- Merge different selected versions across sibling owners with `max`; reject an invalidated parent schedule and final over-capacity rather than clamping.
- Assert per-Control absolute audit records reconstruct `start`, `raw_stage`, `residue`, and `physical_time` after clones.

`test_tilelang_cuda_z3_pipeline_projection.py`

- Project nested per-loop order/stage into stock `PipelinePlanning`; verify child-first injection and unchanged legacy flat outputs.
- Exercise affine nested clocks with nonzero min/step, guarded counters, and a counter shared across three owners. Assert no immediate-loop modulo fallback.
- Inject then restore a synthetic While and verify exact residual counter placement.

### 13.4 WS, lifetime, and physical plans

`test_tilelang_cuda_z3_ws_recursive.py`

- Reflect old and new `WSSync` constructors and optional guard. Verify one-, two-, and three-owner pipelines, depths 1/2/3, continuous counters, single leading version dimensions, and exact arrival counts.
- Complementary PerRole candidates produce one logical span and guarded sync copies; provably nonexclusive/nonexhaustive candidates fail. An outer epoch guard remains outside candidates.
- Test compatible synchronous and async signatures; mismatched per-owner arrivals, external accesses, nested owners, cross-CTA storage, and role-specific loop exits fail before emission.

`test_tilelang_cuda_z3_epoch_lifetimes.py`

- Port weak-only DMA no-alias failure, strong completion-through-another-role success, ring generation, nested carried lifetime, changing-guard rejection, and local ring alias cases.
- Unit-test separating cuts directly: valid `forward + reverse == 1`, invalid `>1`, missing strong path, unknown distance, and int64 overflow.
- Assert counter domains cannot be assigned a lexical period and presence of an empty alias map disables unproved reuse.

`test_tilelang_cuda_z3_physical_plan.py`

- Preserve holes and exact aligned offsets after version/layout expansion. Check final leading-dimension products, sub-byte sizes, fixed allocations, barriers, and target capacity totals.
- Reject stale/unknown provenance, duplicate allocations, unaligned/out-of-range offsets, double expansion, under/over-version mismatch, dynamic unknown size, and unproved overlap.
- Show a positive separating-cut alias proof permits reuse and the same program with weak-only ordering is rejected.

## 14. Device acceptance tests

Device tests supplement, rather than replace, host proofs. Run exact-output repetitions to expose stale slots and phase drift. Use the existing SM120 integration style for the base path; add SM90 for TMA/mbarrier and SM100 only for a tcgen05-specific case. No GPU execution is part of producing this blueprint.

### D1. Nested affine recursive pipeline (SM120; also compile for SM90)

- Two nested serial loops, inner producer/consumer with a shared ring and an outer consumer of the inner-produced shared-buffer value. Use nonzero inner minimum and step 2 in one variant.
- Force a nontrivial schedule with a Control child between two root tasks and a cross-stage Let consumed in the Control header and body.
- Assert compiled nested annotations first; then compare exact output against a scalar reference for four repeated invocations. Include depths 1, 2, and 3 and ensure no version resets at the outer boundary.

### D2. Dynamic While and loop-exit demand (SM120)

- A persistent While whose condition is loaded from memory, with one branch mutating the predicate source and a role-local epoch counter consumed after the loop.
- Verify the snapshot is taken before mutation, every participating role exits together, residual counter/ring phases remain continuous, and output is exact for four invocations with different early-exit trip counts.
- Include a dead-synthetic-Var variant that checks generated IR has no counter allocation.

### D3. Automatic guarded multi-owner ring (SM90 and SM120)

- Generate three ordered sibling owners from ordinary write-first source IR, not explicit `tl_z3_owner_ring` annotations. Alternate active epochs with a shared predicate and place different buffers with depths 2 and 3 in one compatible counter group.
- Use one producer and one consumer role and require TMA on SM90; use the supported synchronous path on SM120. Compare exact output for eight repetitions, covering forward and reverse hand-offs and odd/even guard skips.
- Verify host IR has one continuous counter per role, one reset, three guarded advances, and no owner-boundary reset before running the kernel.

### D4. PerRole candidates and sync migration (SM90)

- One PerRole group with complementary block-parity candidates feeding an MMA/Worker consumer. Each active candidate performs the same logical producer write and has the same arrival signature.
- Cover both guard polarities across blocks and compare exact outputs. Host IR must show one active logical span, guarded producer sync beside each candidate, and one expected arrival per active epoch.
- Negative compile-only variants have overlapping, nonexhaustive, incompatible-arrival, and cross-role compound candidates.

### D5. Strong/weak async lifetime case (SM90)

- Reuse storage between two asynchronous producer/consumer generations where issue order alone is insufficient. The accepted program includes a real completion chain; the negative variant removes it and must fail host validation before codegen.
- For the accepted program, compare exact output for sixteen invocations and run the same binary under `compute-sanitizer --tool racecheck` in the device CI job when available. Numerical success alone is not the separating-cut proof.

### D6. Existing regressions

- Re-run current owner-ring depths 1/2/3, A+B, multi-async, and flat/nested Bind clone numerical cases. The new direct-tree clone path must compile and produce exact output; the current clone compile failure is an explicit release blocker, not an accepted residual.
- Keep legacy explicit two-owner WSSchedule input as a compatibility case and add a semantically equivalent automatically inferred case to prove both use the same runtime implementation.

## 15. Implementation acceptance gates

1. **Foundation gate:** all IR/codec/clone/path/region/lifetime primitive host tests pass; no Ascend host regression.
2. **Analysis gate:** nested dependency, role fixed point, owner frontier, clock grouping, and negative proof tests pass. No hardware-specific Ascend type appears in CUDA headers.
3. **Planner gate:** recursive solves and augmented validation pass with owner exclusions, nested timestamps, general clones, and final version/capacity checks. The old flat test suite still passes through wrappers.
4. **Lowering gate:** stock child-first injection, WS N-owner counters, guarded sync migration, while restoration, and physical allocation validation pass host inspection.
5. **Proof gate:** weak-only lifetime reuse fails; strong periodic separating cuts produce the exact positive alias map consumed by the allocator.
6. **Device gate:** D1–D5 and existing numerical regressions pass on their declared targets. No row is promoted from host-only evidence without its required runtime case.
7. **Cleanup gate:** legacy flat inference/planning and owner-ring functions call the same nested implementations; no second owner solver, scheduling tree, or WS runtime remains.

A row closes only when its implementation and row-specific acceptance evidence exist. Generic host algorithms, host IR transforms, successful compilation, and device numerics remain separate evidence planes.
