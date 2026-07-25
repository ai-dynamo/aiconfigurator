# Collector V3: Op-Centric Collector Management — Design

- **Status:** In review
- **Last updated:** 2026-07-15
- **Source proposal:** *Per-Op Collector Version Management* (Google Doc, 2026-06-28)
- **Release:** v0.11.0 (code freeze 2026-07-28)

## 1. Goal

Make the operation, not the framework version, the unit of collector management.
Every op resolves to exactly one pinned, reproducible runtime (framework default
plus explicit per-op overrides). Collected data carries enough provenance to know
which collector, runtime, and case plan produced it. Any change to a pin or
collector produces an exact changed-operation manifest saying what must be
recollected; everything else is reused deterministically under authored,
backward-safe rules. CI and the support-matrix healer decide from manifest +
provenance alone whether required silicon evidence exists — anything unknown
fails closed.

### Success criteria

1. A collector change identifies exactly which operations and data slices require
   recollection.
2. Unchanged compatible operation data remains reusable.
3. CI rejects missing provenance, missing required silicon data, and unsupported
   silent fallback.

### Non-goals

- The latest collector code working against all historical framework versions.
- Revalidating historical framework versions after each upgrade.
- Forcing all ops to one version during out-of-cycle model onboarding.
- Unifying op naming across frameworks into one global op (identity is always
  the pair *(framework, family)*).

## 2. Op identity

One identity chain, single-sourced, three levels:

```text
registry OpEntry.op  →  PerfFile table  →  family
(collection unit)       (storage unit)     (management unit)
```

- **Registry op** (`collector/<framework>/registry.py`): what you run
  (`collect.py --ops attention_context`). Also the granularity of collector-code
  identity — ops sharing a `collect_*.py` module change together.
- **Table** (`PerfFile` enum, `collector/registry_types.py`): what is stored and
  diffed (`context_attention_perf.parquet`). Provenance is tracked per table.
- **Family**: the unit of pinning, physical layout, reuse, and evidence. The
  table→family mapping is **`collector/op_backend_catalog.yaml`** (introduced by
  PR #1345, op-backend facts) — 12 canonical families covering all 36 tables:
  `attention, encoder_attention, mla, mla_bmm, sparse_attention, moe, gemm,
  mlp, quantize, linear_attention, mhc, comm`.

V3 adds no new identity source. Where the catalog splits (e.g.
`encoder_attention` is its own single-table family, separate from `attention`),
V3 splits.

**Fail-closed identity gate:** a registry op whose `perf_filename` table appears
in no catalog family is a hard validation error — no family means no pin, no
home directory, and no evidence rule. This is the design's "unknown identity"
gate.

**Relationship to #1345:** op-backend facts are observational (which backend an
op actually ran on / can run on) and never gate collection. V3 is normative
(which runtime an op must use; what evidence a change requires). V3 consumes the
catalog vocabulary; it never keys management decisions on observed kernel
backends.

## 3. Physical data layout

```text
aic-core/src/aiconfigurator_core/systems/data/<system>/<family>/<backend>/<version>/
    <table>_perf.parquet     # filenames unchanged (PerfFile enum untouched)
    collection_meta.yaml     # provenance sidecar (committed, required)
    reuse.yaml               # authored reuse declarations (only in declared-reuse dirs)
```

Example: `data/h200_sxm/attention/trtllm/1.3.0rc10/context_attention_perf.parquet`.

- `<family>` is one of the 12 catalog families.
- `<backend>` stays the consumer backend (`trtllm | sglang | vllm`). WideEP data
  lives under its backend at its own version
  (`moe/sglang/0.5.10/wideep_moe_perf.parquet` next to
  `moe/sglang/0.5.14/moe_perf.parquet`); the producing runtime is recorded in
  provenance, not in the path.
- One op's entire history — all backends, all versions — is one subtree.
  A framework upgrade for one family adds exactly one directory; the change is
  visible in `git diff --stat` rather than derived.
- `SHARED_LAYER_REUSE.txt` and `INCOMPLETE.txt` are retired: the former becomes
  `reuse.yaml` (§6), the latter becomes `status: partial` in
  `collection_meta.yaml` (§5).

### Migration

- A generated `git mv` script maps every existing
  `data/<system>/<backend>/<version>/<table>_perf.parquet` to
  `data/<system>/<family>/<backend>/<version>/` using the catalog `op_files`
  mapping. LFS OIDs are unchanged — no re-upload.
- The loader reads **both layouts during one transition window** (legacy path
  deprecated, warning logged), so tools and the GitLab auto-collect pipeline do
  not have to cut over the same day.
- Path-aware consumers to sweep in the same PR: loader discovery,
  `tools/perf_database/audit_kernel_source.py`, `tools/perf_database/parquet_diff.py`,
  `tools/support_matrix/*`, chart tooling, collector finalize output paths.
  The GitLab auto-collect pipeline is updated in lockstep; in-flight data PRs
  rebase after the move.
- Note: `src/aiconfigurator/systems` is a symlink into aic-core since #1322.
  Anything fetching raw file contents by URL (raw.githubusercontent does not
  traverse symlinks) must use the real `aic-core/...` path.

## 4. Manifest v2: per-op runtime pins

`collector/framework_manifest.yaml` moves to `schema_version: 2`: per-framework
`default` runtime plus explicit per-family overrides. WideEP entries flatten to
peer frameworks so resolution is uniform.

```yaml
schema_version: 2
frameworks:
  sglang:
    source_repo: https://github.com/sgl-project/sglang.git
    default:
      version: "0.5.14"
      images:
        default: "lmsysorg/sglang:v0.5.14@sha256:..."
        cu130:   "lmsysorg/sglang:v0.5.14-cu130@sha256:..."
    families:                      # explicit overrides only
      moe:
        version: "0.5.15"
        images: {default: "lmsysorg/sglang:v0.5.15@sha256:..."}
  wideep_sglang:                   # flattened from the v1 `wideep:` section
    base_framework: sglang
    collector_dir: collector/wideep/sglang
    data_backend: sglang           # writes under <family>/sglang/<its version>/
    default:
      version: "0.5.10"
      images: {default: "deepseek-v4-blackwell"}
  trtllm: {...}
  vllm: {...}
  wideep_trtllm: {...}
```

**Resolution:** `(framework, op) → table → family → families[family] or default`
— exactly one runtime per op, always. Two runtimes may contribute tables to the
same family (sglang and wideep_sglang both feed `moe`); pins are therefore per
*(framework, family)*, and each registry belongs to one framework.

**Validation (fail-closed):** a validator run in CI and before collection
asserts every registry op across all frameworks resolves to a version and
image, and enforces the digest rule syntactically: any image reference
containing `/` must carry `@sha256:<64 hex>`; only bare internal image names
(no `/`, e.g. `deepseek-v4-blackwell`) are exempt — in practice, every
public-registry reference is digest-pinned. Unresolvable op or missing image =
hard error.

**Snapshots:** the manifest at a git revision *is* the snapshot. Each
coordinated upgrade tags `collector-snapshot-YYYY-MM` (first tag:
`collector-snapshot-2026-08` when v0.11.0 data settles). Historical support =
check out the snapshot and run its pinned images. No historical mappings
accumulate in the latest manifest.

**Upgrade paths** (unchanged from the source proposal):

1. *Scheduled* (quarterly cadence, §11): umbrella issue, one collection task
   per op, all pins move to the selected baseline where practical, each op
   validated only against its new pin, merge + snapshot tag.
2. *Targeted* (new model): upgrade only the families whose old runtime cannot
   collect the model; unrelated pins unchanged; mixed-version snapshot recorded
   in the manifest.

## 5. Provenance: `collection_meta.yaml`

Written by the collector finalize step into every version dir it produces;
committed with the data. Parquet row schema is unchanged.

```yaml
schema_version: 1
runtime:
  framework: sglang                # or wideep_sglang — the producing runtime
  version: "0.5.14"
  image: "lmsysorg/sglang:v0.5.14"
  image_digest: "sha256:..."
tables:
  moe_perf:
    collector_ref: 0b077da5        # repo SHA the collector ran from
    collector_hash: "sha256:..."   # content hash of the op's module closure
    case_plan_hash: "sha256:..."   # hash of the resolved case set
    collected_at: 2026-07-20
    rows: 12345
    status: complete               # complete | partial
```

- `collector_hash` covers the registry-declared `collect_*.py` module plus its
  declared shared dependencies (e.g. `helper.py`, the family's
  `cases/base_ops/*.yaml`); it is content-based, so it survives rebases.
- `status` is derived, never asserted: at finalize the collector marks a table
  `complete` iff its producing ops' checkpoints hold zero unresolved failed
  cases and no module-level collection failure was recorded — anything else is
  `partial`. The run's observed failure records are the only input; the mere
  presence of provenance fields implies nothing about coverage.
  `case_plan_hash` attests the case set the run actually attempted, so a
  filtered subset/healing run stays distinguishable from a full-plan run even
  when both finish `complete`.
- Scope of `status`: it asserts execution-completeness of the attempted plan
  on that one system dir. Whether the *right* systems and cases were collected
  for a given change is enforced one layer up — the §8 changed-op manifest
  names the required systems and the §9 evidence gate demands the
  corresponding evidence. CI deliberately never re-expands a case plan to
  check per-dir coverage: case generation consults live device memory (§8),
  so the attempted-set attestation is collection-time only.
- "Missing provenance" = a parquet table present with no matching `tables`
  entry. CI fails closed on it (§8); the loader's strict mode refuses it (§7).
- **Legacy tier** (a later amendment): data collected before V3 carries a
  backfilled sidecar with `provenance: legacy` — `runtime: {framework,
  version}` only, per-table `status` (complete unless it had `INCOMPLETE.txt`),
  no hashes. The tier makes the §8 sidecar-coverage gate and the §12.3 support
  bar total over the whole tree while staying honest about unknown identity.
  Strict mode (PR 4) treats `legacy` as warn-not-fail for one release. New
  collections always write full provenance; the legacy tier only shrinks.
- Transient run artifacts (`collection_summary_*.json`, `errors_*.json`) remain
  uncommitted but are retained as CI artifacts on data PRs (§9).

## 6. Data reuse: three channels

Reuse is never wholesale. The loader never reuses "a database" or "a version";
it fills individual missing shapes from approved donors, through exactly three
channels that carry different risk and therefore different requirements.

### 6.1 The substrate: per-table, per-shape merging

For each table (say `moe_perf`), the loader builds a priority-ordered list of
donor version dirs, loads rows from all of them, and merges **by shape key**
(m/n/k, batch, dtype, …): the first source that has a shape wins; later
sources only fill gaps. This is how `perf_database.py` already works (source
list assembly in the shared-layer sourcing code); V3 does not change the merge
— it changes *which sources are admitted and in what order*. Consequences:

- Different shapes of the same table may legitimately come from different
  versions.
- A donor can never shadow the primary: newer valid data always wins for
  shapes it actually has.

### 6.2 Channel 1 — implicit backward fill (same backend, earlier version)

The free, always-on channel. Example: requesting `sglang 0.5.14` gemm on
h200_sxm where a few shapes were only ever collected at 0.5.12:

```text
source order for (h200_sxm, gemm, sglang, requested=0.5.14):
  1. gemm/sglang/0.5.14/     ← primary
  2. gemm/sglang/0.5.12/     ← nearest EARLIER sibling, fills gaps only
  x. gemm/sglang/0.5.15/     ← skipped: newer than requested, not declared
```

The V3 change from today's behavior: ordering becomes *nearest-earlier*
instead of *newest-first*, and newer-than-requested donors are excluded by
default. Rationale: data collected before the requested version existed cannot
have been invalidated by it, so backward fill preserves historical
reproducibility; forward fill silently answers "how fast is 0.5.14" with
0.5.15's kernels.

### 6.3 Channel 2 — declared reuse (`reuse.yaml`, same backend, any direction)

When we *know* data is valid for a version we never collected — typically a
framework upgrade where a family's kernels did not change — we say so
explicitly. Replaces the presence-only `SHARED_LAYER_REUSE.txt`. In the
typical case a declared-reuse version dir carries no parquet of its own, only
a `reuse.yaml`:

```yaml
# data/h200_sxm/moe/sglang/0.5.12/reuse.yaml
schema_version: 1
reuse:
  - table: moe_perf
    from_version: "0.5.14"
    reason: "MoE kernels unchanged 0.5.12→0.5.14; verified <link/commit>"
    approved_by: yimingl
```

The loader inserts the declared donor **ahead of** implicit fallback (right
after the primary), because it is a vetted claim rather than a heuristic. This
is the only way to borrow *forward* (from a newer version) — which is exactly
what today's blank marker dirs do implicitly; V3 keeps the capability but
makes it per-table, reviewable, and reasoned.

A dir may also hold BOTH its own parquet and a declaration for the same
table (self-overlap): per §6.1 the dir's own rows always win the shapes they
cover, so such a declaration only forward-fills shapes the dir's own data is
missing. The migrated tree carries a small set of these (l40s), each with a
`reason` field disclosing exactly that mechanical derivation.

This channel is also what makes per-op pinning cheap: on a scheduled upgrade
to 0.5.15, families whose kernels did not move are not recollected — each gets
a one-entry `reuse.yaml` backed by the evidence policy (§9: kernel-identity
evidence from the #1345 facts registry plus a spot benchmark on one system per
SM generation), and the fleet-wide GPU cost of the upgrade shrinks to the
families that actually changed.

### 6.4 Channel 3 — cross-framework fill (kernel-identity gated)

Cross-framework reuse is legitimate only when it is not really cross-framework
at the kernel level — e.g. trtllm and sglang both dispatching the same cuBLAS
GEMM or the same flashinfer attention kernel. The gate is the existing
`op_kernel_source_manifest.yaml`: rows are admitted from a sibling backend only
when their `kernel_source` is in a `shared`/`shared_fallback` tier whitelisting
the active backend, and only to fill shapes the same-backend chain (channels
1–2) could not. Example: a `gemm_perf` shape missing from all sglang versions
may be filled from trtllm rows whose `kernel_source` is a whitelisted shared
cuBLAS kernel — but never from trtllm rows whose kernel is trtllm-internal.

V3 keeps this mechanism unchanged. What #1345 adds is a stronger verification
story for the whitelist itself: its `kernel_source_backends.yaml` translation
table documents what each label actually is, with code citations.

### 6.5 Summary rules and guardrails

The loader's source ordering (§7) in one list:

1. Same backend only, by default.
2. Requested version first, then declared `reuse.yaml` targets, then the
   nearest **earlier** version.
3. Never a version newer than requested — unless an explicit `reuse.yaml`
   declaration says so.
4. Cross-backend fill only for kernel sources whitelisted by
   `op_kernel_source_manifest.yaml`, and only after channels 1–2.
5. The `comm` family is excluded from sibling-version reuse entirely — NCCL
   curves are topology-bound, so shape-filling across versions is wrong there
   (current NCCL/oneCCL behavior, now stated as policy).

Guardrails:

- **Provenance surfacing:** the loader reports, per table, the admitted
  sources with channel tags (`primary | declared_reuse | fallback |
  cross_backend`), so the support-matrix health classifier can tell "natively
  collected" from "riding on a declaration" without re-deriving anything.
- **CI audit:** a `reuse.yaml` pointing at data that does not exist, or any
  fill pattern outside these channels, fails the PR — that is the
  operational definition of **unsupported silent fallback**.
- **Scope limits:** all reuse runs only in SILICON/HYBRID modes; formula-only
  modes (EMPIRICAL, SOL) are untouched.

## 7. Loader changes (`aic-core/src/aiconfigurator_core/sdk/perf_database.py`)

In increasing order of semantic weight:

1. **Dual-layout discovery** — family tree first, legacy layout as deprecated
   fallback for one transition window.
2. **Effective-source provenance** — `PerfDatabase.data_provenance`: per table,
   the admitted sources with channel tags (`primary | declared_reuse | fallback |
   cross_backend`), consumed by support-matrix health. Granularity is
   admitted sources, not per-row attribution. *(Shipped in PR 4.)*
3. **Reuse rules** — implement §6 ordering. `get_database(system, backend,
   version)` keeps `version` as the *requested* framework version; per-op
   resolution happens inside.
4. **Strict mode** — missing `collection_meta.yaml`, undeclared reuse, or a
   table without a family raises instead of warning. Default **on in CI, off
   for end users** in v0.11.0.

Shared-layer reuse remains enabled only for SILICON/HYBRID modes and disabled
for EMPIRICAL/SOL (unchanged).

**Safety net:** every loader PR must pass the prediction-regression gate
(old-vs-new snapshot comparison, workflows from #1289/#1336), which catches
unintended semantic drift in exactly this layer.

## 8. Changed-operation manifest and CI enforcement

### `tools/perf_database/changed_ops.py`

Diffs two git revisions (typically `origin/main` vs PR head). Per
*(framework, family)* it compares three change signals:

1. **Pin** — manifest v2 version / image digest;
2. **Collector code** — `collector_hash` over the op's module closure;
3. **Case plan** — `case_plan_hash` of the resolved case set.

Output (machine-readable, exhaustive):

```yaml
changed:
  - framework: sglang
    family: attention
    reasons: [pin_version]
    tables: [context_attention_perf, generation_attention_perf]
    systems: [h200_sxm, b200_sxm, gb200]   # systems holding data at the old pin
    action: recollect
unchanged:
  - {framework: sglang, family: moe, tables: [...], systems: [...]}
```

Contract notes (locked during implementation): `unchanged` entries carry exactly
`{framework, family, tables, systems}` — no vacuous `reasons`/`action`. The
`case_plan` reason is computed from the family's case-INPUT files (base-ops
and model-case YAML plus the case-generation modules) hashed at each revision
— GPU-free and deterministic at any rev — while `collection_meta.yaml`'s
`case_plan_hash` remains the collection-time attestation of the expanded case
set. A base revision predating V3 metadata exits with code 3 ("cannot compute
against a pre-V3 baseline"); CI maps it to a neutral skip.

This file is the single input consumed by the evidence resolver (§9), the CI
gate, and the support-matrix healer — same manifest in, same
requirements out.

### CI audit (fail-closed surface)

One audit tool (sibling of `audit_kernel_source.py`), run on every PR touching
`data/`, `collector/`, or the manifest. Hard failures:

- a parquet table without a matching provenance entry;
- a `reuse.yaml` pointing at nonexistent data or violating §6 rules;
- any registry op that does not resolve through manifest v2;
- a table filed under the wrong family;
- pin/collector/case-plan changes without the evidence the policy demands (§9).

The CI audit is the primary gate; loader strict mode is the backstop.

## 9. Evidence policy

Policy-as-code: `collector/evidence_policy.yaml` (thresholds; an authored
`system_generations` map covering the whole fleet — SM103 Ultra folded into
blackwell as a policy decision — and one evidence representative per
generation) plus the pure-function resolver `tools/perf_database/evidence_check.py
--manifest changed_ops.yaml` → required evidence, filtered to the SM
generations the change actually touches (an unmapped system fails closed).
Deterministic: CI and the healer get identical answers from identical
manifests. Exception WAIVERS (`evidence_exceptions.yaml`, approver + expiry)
are applied by the evidence-gate CI check, not the resolver — expiry needs a clock,
which would break the resolver's purity. *(Policy and resolver shipped in PR 4;
the enforcing CI gate — the required check that validates submitted evidence
and waivers against these requirements and blocks under-evidenced data PRs —
is a separate tracked deliverable. Until it lands, the audit workflow uploads
the changed-op manifest as an informational artifact and enforcement is by
human review against the resolver's output.)*

| Change | Required evidence |
|---|---|
| Pin version change | Fresh silicon for the family's tables on every system in the manifest — **or** a `reuse.yaml` declaration backed by kernel-identity evidence (#1345 facts showing unchanged kernels) plus a before/after spot-benchmark on one system per SM generation |
| Collector code change, same pin | Before/after `parquet_diff` on affected tables from an evidence system (one designated system per SM generation, named in `evidence_policy.yaml`); median latency delta beyond threshold (initial: 5%) escalates to full recollection |
| Case plan additions | Collect the new cases only (additive); removals prune with the diff visible |

- **Retention:** collection summary + error JSONs attached as CI artifacts on
  the data PR; provenance committed with the data.
- **Exceptions:** only via an `evidence_exceptions.yaml` entry with approver and
  expiry — explicit, reviewed, auditable.
- The support-matrix healer may *propose* pin or reuse changes, but its PRs pass
  the same evidence gate and human review as any other.

## 10. Delivery plan (v0.11.0, freeze 2026-07-28)

The implementation is split into four sub-scopes, delivered as exactly four
PRs (one per sub-scope) to bound CI time and review load.

| PR | Content | Depends on |
|---|---|---|
| 1 | This design doc + manifest v2 + resolver + validation; wideep flattened | — |
| 2 | Physical reorg: scripted `git mv` + dual-read loader + tools path sweep | #1345 catalog |
| 3 | Provenance writer + `reuse.yaml` + marker migration + `changed_ops.py` + CI audit | PR 2 |
| 4 | Loader reuse rules + strict mode + source diagnostics + evidence policy/resolver | PR 3, regression gate |

- PR 1 proceeds immediately and carries the reviewed design doc; PRs 2→3→4
  are sequential.
- PR 2 needs `op_backend_catalog.yaml` on main — either #1345 merges first or
  the catalog file is co-landed with its author's agreement.
- PRs 3 and 4 span collector/tools/aic-core: each declares the approved
  cross-module contract change per
  `.claude/rules/collector/layer_permissions.md` (approval, not PR count, is
  the requirement).
- **De-scope line:** if the window tightens, PR 4's runtime strict mode drops to
  CI-only enforcement; everything else still lands.
- After release data settles: tag `collector-snapshot-2026-08`.

## 11. Quarterly operation

The operating premise: once per quarter, the collector is pinned to a new
framework baseline and the database is updated with that pinned version's
data. V3 makes that cycle computed and mostly declarative:

1. **Kickoff = one manifest commit.** The cycle's umbrella issue bumps each
   framework's `default` pin (e.g. sglang 0.5.14 → 0.6.2). The goal state of
   every quarter is `families: {}` — overrides accumulated mid-quarter
   converge back into the new default. `changed_ops.py` between main and the
   pin-bump commit emits the quarter's work list mechanically: every
   *(framework, family)* whose pin moved, with tables and systems. No human
   enumerates recollection scope.
2. **The work list splits into "collect" and "declare".** Per changed family,
   the evidence policy (§9) allows exactly two moves: recollect the family's
   tables inside the new pinned image (new
   `data/<system>/<family>/<backend>/<new_version>/` dirs with provenance), or
   declare reuse (`reuse.yaml` ← previous quarter's version) backed by
   kernel-identity evidence from the #1345 facts plus a spot benchmark per SM
   generation. How to choose, per changed family:

   1. **Default: recollect.** Fresh collection is always valid and needs no
      extra justification.
   2. **Reuse is allowed only if BOTH hold:** the kernel-identity facts
      (#1345) show the family's kernels are the same before and after the
      pin bump, AND a spot benchmark on one system per SM generation
      confirms it (median latency delta within the policy threshold).
   3. **If either check fails, recollect.** This is a plan-time decision:
      the data PR ships one move or the other, the evidence gate verifies
      whichever was shipped, and the loader never switches between them at
      query time.

   Why reuse exists at all: when kernels are unchanged, recollection
   produces statistically identical rows — reuse buys the same answer
   without the fleet-wide GPU cost, so the quarter's bill stays
   proportional to what the framework actually changed, not to the size
   of the support matrix.
3. **Per-family progress; partial upgrades are first-class.** Each
   *(framework, family)* is an independently ownable task. A family that
   breaks on the new baseline keeps an explicit `families:` override on the
   old version — visible debt in the manifest and next quarter's backlog —
   without blocking the rest of the cycle.
4. **Fail-closed completion.** *(framework, family, new_version)* counts as
   supported only under the §12.3 bar (complete provenance or a valid reuse
   chain). An uncollected, undeclared family is missing provenance: CI rejects
   it and the support matrix never lists the new version for it.
5. **Quarter close = snapshot tag.** When the work list is empty, tag
   `collector-snapshot-YYYY-MM`. The tag is the quarter's deliverable and its
   reproducibility contract (manifest pins + collector code + data at one
   revision). Prior quarters' data dirs stay in the tree (append-only), which
   keeps historical version requests and channel-1 backward fill working on
   latest main.

## 12. Resolved open questions (from the source proposal)

1. **Cadence** — quarterly (decided; see §11), with targeted mid-quarter
   upgrades for new models. The mechanism itself hard-codes no cadence.
2. **Image digests** — mandatory for any image reference containing `/` (in
   practice, every public-registry reference; multi-arch index digests, never
   platform-child digests); bare internal image names are exempt. Enforced
   syntactically by manifest validation.
3. **Minimum bar to claim version support** — a *(framework, family, version)*
   is supported iff its data dir has complete provenance (`status: complete`
   for all the family's tables) or a valid `reuse.yaml` chain to one that does;
   the support matrix derives from this, so the bar is machine-checkable.

## 13. Open question: data retention across quarters

Quarterly appends grow the LFS tree without bound (~906 parquet files today,
plus each quarter's changed families). This design does not prune. A sensible
policy would be "keep the most recent N quarters of version dirs on main;
older quarters remain reachable only through their `collector-snapshot-*`
tags" — but N, and whether pruning also drops declared-reuse chains that
reference pruned donors, must be decided deliberately. Decide before the
second quarterly cycle, not by accident.
