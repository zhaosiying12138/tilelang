# CUDA pipeline buffer-version overrides

## Z3 map contract

`InjectSoftwarePipeline` accepts the loop annotation
`tl_pipeline_buffer_versions` with the following IR-level contract. This is the
map ABI between Z3 pipeline planning and software-pipeline injection; it does
not add a Z3 solver expression or change solver behavior.

| Item | Contract |
| --- | --- |
| Container | `Map<Var, IntImm>` |
| Key | The original `Buffer.data` `Var`, with its identity preserved |
| Value | A positive, compile-time version count |
| Scope | One annotated pipeline loop and only allocations participating in that loop |

For example, the annotation conceptually has this form:

```python
{
    A_shared.data: T.int32(3),
    B_fragment.data: T.int32(1),
}
```

A Z3 annotation producer must not replace the keys with buffer names, string
IDs, `Buffer` objects, or stage-indexed arrays. It should attach the map to the
loop that owns the accesses rather than merging maps from sibling or nested
pipelines.

## Version-selection semantics

An entry is an exact override, not a lower bound or a request for additional
double buffering. `ComputeBufferVersions` returns it before the external-buffer
rule and the automatic conflict/liveness heuristics. Consequently, the pass
must not clamp it to the pipeline stage count or reduce it to the automatically
inferred count.

- `1` keeps the original allocation rank and disables multi-version indexing
  for that buffer, even when automatic inference would choose two versions.
- `N > 1` prepends a leading dimension of extent `N` to the allocation and uses
  the existing logical-iteration `floormod(..., N)` version index.
- Buffers absent from the map use the existing automatic inference unchanged.
- The override is independent of `num_stages`; in particular, an explicit ring
  may be larger than the loop trip count or the number of pipeline stages.

The existing allocation rewrite remains responsible for storage behavior. Local
allocations remain local, outer allocations remain at their outer scope, and
`shared`/`local.fragment` storage scopes are not changed. Existing layout-map
expansion and outer-allocation reconciliation continue to apply. Prologue,
steady-loop, and epilogue accesses use the same remap and existing bound checks;
short loops do not introduce a separate versioning rule.

## Validation and cleanup

The complete map is validated before the current pipeline rewrites allocations.
Errors are reported for a non-map annotation, non-`Var` keys, non-`IntImm`
values, zero or negative counts, counts outside the supported integer range,
unknown data Vars, and buffers that are visible but do not participate in the
current pipeline. Duplicate entries and ambiguous mappings that resolve to
multiple current buffer objects are also rejected. IR `Map` itself is
unique-keyed, so annotation producers must avoid overwriting duplicate source
entries before constructing that map.

`tl_pipeline_buffer_versions` is an internal pipeline-planning annotation. It is
removed with the other pipeline keys when generating prologue/steady/epilogue
loops and on the non-overlapping-stage fallback. It therefore does not leak into
the final loop annotations or trigger the same override again when
`InjectSoftwarePipeline` is run a second time. With no such annotation, the pass
uses the original inference and rewrite path without adding an IR transformation.

## Validation status

`testing/python/transform/test_tilelang_transform_pipeline_buffer_versions.py`
contains host-only structural tests for single and multiple buffers, explicit
1/2/3 versions, disagreement with automatic inference, short loops,
local/outer allocations, shared/fragment scopes, malformed and unknown maps,
default behavior, annotation removal, and a repeated injection pass. These
tests do not perform CUDA code generation or GPU numerical validation. No GPU
numerical-correctness claim is made for this contract.
