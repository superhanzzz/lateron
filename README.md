# Lateron: Multi-Horizon Hard-Token Context Compression and Primacy-Zone State Steering for Long-Horizon LLM Agents

**Author:** Christopher Dryland (Independent Researcher, Canberra, Australia)
**Date:** October 23, 2026
**Version:** 1.3.1 (Systems Architecture Specification)
**License:** [Creative Commons Attribution 4.0 International (CC BY 4.0)](LICENSE)

> **Changes in v1.3.1.** Separates the veto registry from prose summarisation: active operator vetoes are structured rows rendered deterministically into the block, never re-summarised by a model (§3.5, §5). Replaces the transaction-held-across-model-calls pattern in `compact()` with optimistic concurrency on a `state_version` field, so an operator pivot arriving mid-compaction aborts the pass instead of being overwritten (§5). Cites the prefix-caching constraint explicitly and reframes placement as preserving the static preamble's eligibility for cache reuse, with provenance and authority separated (§3.6). States the partition trade-off plainly and adds **Arm B3**, a mechanism-matched single-timescale baseline, so the benchmark prices multi-timescale partitioning separately from structured lifecycle management (§2.3, §6). Rebalances the budget to 120/100/90/40/50 = 400 with a structured registry reserve.

> **Changes in v1.3.** Corrects three defects in the §5 skeleton (state loss on distillation, out-of-order stratum reads, non-atomic writes) and aligns `drift_ratio()` with the prose definition; reframes §2.3 to concede that the $\alpha^k$ decay model describes *unstructured* summarisation only, and adds Arm B2, a pinned-invariant baseline; replaces Slot 0 injection with a **Post-Static-Prefix Breakpoint** that preserves host KV prefix caches, and adds a trust boundary fencing L3 as operator-authorised and L1/L2 as non-authoritative telemetry; reserves header overhead in the §2.4 budget.

> **Changes in v1.2.** Staged Rolling-*N* Uniform Summarisation as the competitor baseline (§1.4, §2.1–2.3); promoted rejection telemetry to its own subsection with a seven-day TTL (§3.5); replaced Primacy-Zone residency with a placement rule conceding host-prompt-length dependency (§3.6); replaced LLM-judged drift detection with a deterministic L0 preprocessor metric (§3.4); made L3 normative and two-way governed (§4); made L0 an immutable event-sourced log with a replay cursor (§3.2); pinned tokenisation to `o200k_base`; specified the evaluation protocol (§2.4, §3.3, §6).

---

## Abstract

Long-horizon Large Language Model (LLM) agents degrade in reliability as interaction history grows. While recent models support million-token context windows, production engineering reports indicate that transformer instruction fidelity begins to decay once context fill passes roughly 60–70%, with retrieved memories injected into the middle context zone (15–85%) suffering from pronounced attention dilution. Standard Retrieval-Augmented Generation (RAG) and flat episodic memory stores additionally treat stale implementation noise and high-level system invariants with equal retrieval weight.

This paper presents **Lateron**, an out-of-band, multi-horizon hard-token compression architecture for stateful LLM agents. It separates three contributions that are usually conflated: **structured lifecycle management** (a veto registry, L3 governance, cache-safe placement) and **multi-timescale partitioning** (strata with per-horizon resolution rules). The evaluation protocol prices them separately, because the first is cheap and the second is not, and a paper that claims both while testing neither proves nothing.

Lateron applies **Horizon-Based Resolution Scaling** (adapted from Construal Level Theory) across discrete temporal strata (24-Hour, 7-Day, 30-Day, 90-Day+). As state logs age, low-level execution mechanics are pruned while high-level trajectory invariants are distilled upward. The resulting state is reconciled nightly into a 400-token control block injected after the host's static preamble — or immediately before the active turn — so it occupies an attention peak while leaving the prefix cache eligible for reuse. Consolidation is driven by a deterministic drift metric over an immutable event-sourced log; the invariant stratum is fenced as operator-authenticated and changed only top-down, while the lower strata are fenced as non-authoritative observation.

---

## 1. Introduction: The Production Bottleneck in Long-Horizon Agents

Autonomous and semi-autonomous LLM agents operating over weeks or months face three structural failure modes in production.

### 1.1 Middle-Zone Attention Degradation

Transformer attention distribution is heavily U-shaped. Performance on tasks requiring information from the middle of long contexts degrades as context length grows, even for models explicitly built for long context (Liu et al., 2023). Production engineering reports place the **Primacy Zone** at roughly the first 0–15% of tokens and the **Recency Zone** at the final 5–10%, with the **Middle Zone** (15–85%) receiving degraded attention (RankSquire, 2026). Standard memory pipelines inject retrieved vector chunks into this middle zone, increasing token cost while yielding poor instruction adherence.

### 1.2 Temporal Contradiction and Retrieval Pollution

Append-only vector databases retrieve past logs on semantic similarity rather than temporal validity. As shown in the Evo-Memory benchmark (Wei et al., 2025) and HiMem (Zhang et al., 2026), failing to prune superseded operational states causes agents to hallucinate obsolete constraints or repeat failed tool paths.

### 1.3 Uniform Summarisation Loss

Existing recursive summarisation pipelines compress text uniformly — shrinking transient syntax errors and core project objectives at the same rate until long-term system goals vanish into generic prose.

### 1.4 The Baselines That Must Be Beaten

§1.3 is normally stated as an intuition and left there. It should not be. The dominant production pattern is **Rolling-*N* Uniform Summarisation**, the behaviour behind `/compact`-style context management. A single operator is applied to a sliding window of history under a fixed token budget, with no notion of what kind of information it is compressing.

Three baselines follow, and only the last two are a fair fight:

- **Arm B** — the naked summariser.
- **Arm B2** — 90 tokens of pinned invariants plus a shared rolling summary at the same 400-token total: what a competent engineer ships instead of strata.
- **Arm B3** — B2 plus the *entire structured lifecycle apparatus* but **no timescale partition**: identical veto registry, identical pivot controls, identical placement, one shared descriptive summary.

B3 exists because this paper makes two claims that can succeed or fail independently. If B3 matches Lateron, structured lifecycle management is doing the work and the partition is decoration. If B2 matches B3, the registry and governance are decoration. §6 is built to separate them.

---

## 2. Horizon-Based Resolution Scaling

### 2.1 The Production Default: Rolling-*N* Uniform Summarisation

Define a history $H$ of token length $|H| > N$. Rolling-*N* applies a compression operator $U$ such that $U(H) \rightarrow \hat{H}$ with $|\hat{H}| \le N$, typically by *"summarise the conversation so far, keeping what matters"*, applied cumulatively as the window advances.

Three properties follow:

1. **One operator, no strata.** $U$ cannot treat a resolved stack trace and a non-negotiable deployment constraint as different kinds of object, because nothing in the pipeline tells it that they are.
2. **Compounding decay.** Applied $k$ times, every surviving item is the result of $k$ successive re-summarisations. Salience not reinforced by recency is progressively eroded.
3. **Recency-symmetric attention.** Material that has aged out of the sliding window is either summarised or gone; no stratum gives an old invariant a *protected* representation.

Rolling-*N* is good at what it was designed for: bounding context and preserving recent working state. On its own, it protects the wrong thing at long horizons.

### 2.2 Horizon-Based Resolution Scaling

Lateron replaces the single operator with stratum-specific resolution rules indexed to age, derived from **Construal Level Theory** (Trope & Liberman, 2010): *temporal distance dictates abstraction resolution*. The mapping is a design heuristic, not a cognitive claim. The system does not argue that agents think like people; it argues that the *category* of information worth retaining changes with age, and that a different operator per age band encodes that.

| Temporal Stratum | Resolution Rule | Retained State | Pruned State | Token Cap |
| --- | --- | --- | --- | --- |
| **L0: Event Log** | Immutable, append-only | All telemetry — tool calls, git diffs, session logs, errors, operator vetoes | Nothing (never pruned on disk) | Unbounded (disk) |
| **L1: 7-Day Stratum** | Tactical state (the *how*) | Active task velocity, open blockers, uncommitted branches | Resolved errors, raw conversational turns, intermediate tool outputs, veto prose | ≤ 120 tokens |
| **L2: 30-Day Stratum** | Operational patterns (system dynamics) | Throughput bottlenecks, recurring failure modes, architectural drift, workflow dependencies | Completed task checklists, isolated daily variance, deprecated experiments | ≤ 100 tokens |
| **L3: 90-Day+ Stratum** | Invariant objectives (the *why*) | Core project specification, non-negotiable constraints, primary objective function | All low-level implementation mechanics and historical tactical state | ≤ 90 tokens |
| **Runtime Triage Rule** | Compiled directive | The single priority instruction for the active session | — | ≤ 40 tokens |
| **Veto Registry, Headers & Flags** | Structured, rendered not summarised | Live operator vetoes as rows with scope and expiry; section fences; `DRIFT_HIGH` | — | ≤ 50 tokens |

### 2.3 What Partitioning Costs

Let $H = H_{inv} \cup H_{noise}$. An **unstructured** uniform operator $U_{uniform}$ reduces each component by the same factor:

$$|U_{uniform}(H_{inv})| \approx \alpha \cdot |H_{inv}|, \qquad |U_{uniform}(H_{noise})| \approx \alpha \cdot |H_{noise}|$$

The ratio holds; the absolute invariant mass decays as $\alpha^k$ over $k$ passes. **That model is narrower than v1.2 claimed.** It describes a pipeline in which nothing is told to protect anything. It does not describe an instruction-guided summariser, nor a pinned objective block held outside the rolling window — and Arm B2 is exactly that design. The straw-man version of the argument is withdrawn.

**The trade-off, stated plainly.** Hard multi-timescale partitions buy one property: **tactical churn cannot evict recurring patterns**. A 7-day stratum absorbs the noise of a bad afternoon; a 30-day stratum retains the diagnosis of a bad month; neither can crowd out the other, because their capacities are separate. Against that, two real costs:

1. **Capacity is not fungible.** 120 tokens of L1 capacity cannot be borrowed by L2 during a fortnight of quiet tactical activity, nor can an empty L3 donate 90 tokens to a busy week. A single shared 220-token summary allocates where demand is, which is often better.
2. **Update lag.** An insight must age into its stratum before it can be represented at that resolution. L2 patterns are refreshed on a weekly cadence, so a newly discovered recurring failure stays invisible to the pattern stratum for up to seven days — while it is perfectly visible to L1.

Neither cost is hypothetical, and neither is repaired by the strata themselves. They are the price of the property, and §6's Arm B3 exists to check whether the price was worth paying.

**What the architecture therefore has to justify** is not invariant protection (a pinned block does that) but three mechanisms, two of which are lifecycle rather than partition:

1. **Veto registry (lifecycle).** A bounded, expiring, structured record of what a human rejected, which B2 lacks entirely and B3 shares with C.
2. **Pattern extraction (partition).** The property the partition actually buys, and the only one B3 cannot replicate: L2 holds thirty days of recurring failure modes that never compete with L1 for space.
3. **Pivot governance (lifecycle).** Proposal, sign-off, retirement and baseline reset, shared with B3.

Invariant mass remains bounded from below by the L3 cap:

$$|L_3|_{t+1} \ge \min(\beta_{L3},\, |L_3|_t), \qquad \beta_{L3} = 90 \text{ tokens}$$

but this floor is a *budget guarantee*, and B2 and B3 both have it. It is not evidence for Lateron.

### 2.4 Token Budget and Tokenisation

All caps are counted with `tiktoken`'s `o200k_base` encoding. The budget totals exactly 400 tokens:

- **L1 (7-day tactical):** 120 tokens
- **L2 (30-day patterns):** 100 tokens
- **L3 (90-day invariants):** 90 tokens
- **Runtime Triage Rule:** 40 tokens
- **Veto registry, headers and flags:** 50 tokens

The 50-token reserve is itself split: **25 tokens for rendered veto rows** and 25 for section fences, the `DRIFT_HIGH` line, and per-veto expiry stamps. A budget that omits its own overhead is not a budget, and a registry without a reserved sub-cap silently crowds the strata whenever an operator is having a bad week.

The split is a working default, expected to be tuned per deployment; it is not derived from first principles. Lateron does not control the host system prompt and makes no claim about total context length — only about the size of the block it contributes.

---

## 3. System Architecture & Pipeline Execution

Lateron decouples memory consolidation (asynchronous, out-of-band) from live inference (synchronous, zero-latency).

### 3.1 Passive Telemetry Ingestion

Rather than requiring explicit "save-to-memory" tool calls during live inference — which consume tool-calling turns and increase latency — Lateron ingests passive operational exhaust:

- **Session state logs:** read-only queries against the agent's local SQLite state store (message history, tool execution status, timestamps).
- **Repository and filesystem state:** per-session `git diff --stat` and `--name-only`, commit metadata, and file modification timestamps across active workspaces.

### 3.2 Event-Sourced L0

L0 is an **immutable, append-only event log** in SQLite. Compaction never deletes rows. A `last_compacted_id` cursor marks the high-water mark of processed events; the worker reads forward from the cursor and advances it only in the same atomic write that publishes new stratum state.

Two consequences matter:

1. **Pruning applies to injection, not storage.** What is pruned is the *representation* placed in context. The events remain on disk, so a bad distillation is recoverable rather than permanent.
2. **Deterministic replay.** Because L0 is an ordered log with a cursor, a suspect compaction can be replayed from any prior cursor against a pinned model and prompt, producing the same strata. Memory becomes auditable, which vector stores are not.

### 3.3 Compaction Trigger

Compaction runs on a compound trigger, whichever fires first:

$$\text{compact when } \min(\text{24h wall-clock},\ 8{,}000\ \text{uncompacted tokens})$$

The token clause closes the session-length gap: an agent running for five days without termination does not silently accumulate unbounded uncompacted state.

### 3.4 Deterministic Drift Detection

Drift must not be a second LLM's opinion of a first LLM's work. Lateron computes it before any compaction call, in a preprocessor over L0:

1. L3 declares `target_paths` — glob patterns — and a target repository scope.
2. Over the trailing seven days, count tool **touches** (`kind='touch'`) outside `target_paths` (`touches_out`) against all touches (`touches_total`). The denominator counts touches only: mixing commit rows into it would let a burst of commits dilute a touch-heavy drift signal, and would make the metric move when the operator committed more rather than when the agent wandered more.
3. Compute the **Drift Ratio**:

$$D_{7d} = \frac{\text{touches}_{out}}{\max(1,\ \text{touches}_{total})}$$

4. Separately, with a different window, count commits whose changed path matches `target_paths` (`commits_in_target`). Commits never enter the $D_{7d}$ denominator.
5. Raise `DRIFT_HIGH` when both conditions hold:

$$D_{7d} > 0.75 \quad \text{and} \quad \text{commits}_{in\_target} = 0 \text{ in the last 48h}$$

Both sides of the commit test apply `target_paths` matching. A count of commits with a non-null path measures "the operator committed something", not "on-target work happened".

The flag is arithmetic, auditable and reproducible. The model's only role is to summarise off-target activity into one line attached to the flag — it can neither raise nor clear it. A false positive is threshold tuning, not a reasoning failure.

### 3.5 Rejection Telemetry: The Veto Registry

When a human operator vetoes or aborts an agent-proposed action during a live run, the event is recorded as a row in a **`vetoes` table** — not as prose, and not as an instruction to a summariser.

```
vetoes(id, created_at, last_triggered_at, expires_at, scope, reason, trigger_count, source)
```

`scope` is a structured matcher — an action name, a command prefix, a path glob — so that "do not run the schema migration" is a *testable predicate*, not a sentence a model may soften. `expires_at` is `last_triggered_at + 7 days`, and every fresh veto of the same scope resets the clock and increments `trigger_count`.

Three properties follow from storing rows rather than prose:

- **Renderable, not re-summarisable.** The control block's veto lines are produced by deterministic Python (§5, `render_veto_registry`) directly from live rows. No model paraphrases a veto, so no model can dilute one. A veto that has survived four nightly compactions reads exactly as it did on the first.
- **Locally scoped.** A veto is evidence about the tactical present, not a statement about the mission. Putting it in L3 would make one bad afternoon's judgement a constitutional constraint; putting it in L1 as *data* keeps it bounded and lets it expire.
- **Expiring, with an overflow policy.** At TTL a row stops rendering. If active vetoes exceed the registry's 25-token sub-cap, the renderer keeps the **most recently triggered** entries, drops the rest, and emits a single `+N more vetoes active, not shown` line; the overflow count is also written to L0 as a `veto_overflow` event so the omission is auditable rather than silent. Recency is the right eviction key because a veto that keeps firing is the one the agent keeps needing to see.

Rejected-action loops are among the most expensive failure modes in long-horizon agents: each repetition burns a full tool-call round trip to re-derive a decision a human already made. Lateron treats the human's veto as the cheapest available gradient signal and stores it at the shortest stratum that can act on it — as a row with an expiry, not a paragraph with a tone.

**Consequence for the implementation.** Because the registry is structured, L1 prose no longer carries vetoes at all: L1 distils velocity and blockers, and the registry is rendered beside it. This removes the v1.2 failure mode in which a distillation pass rephrased an active veto into something softer, or dropped it because it was not re-mentioned that day.

### 3.6 Placement and the Trust Boundary

v1.1 claimed injection "into the Primacy Zone", which assumed a short host system prompt. v1.2 replaced it with a dual anchor and made Slot 0 preferred. **Slot 0 is wrong wherever the host caches prefixes**, which is most current inference stacks.

**The constraint.** Prefix caching reuses the KV states of a token-identical prefix; as the Modular LLM Inference Handbook puts it, the reuse "works only when the prefix is exactly identical, including whitespace and formatting. Even a single character difference breaks the cache" (Modular, 2026). A dynamic block placed *ahead of* a static preamble therefore forfeits the preamble's **eligibility for prefix-cache reuse** on every change: the cache diverges at its first token, and the whole preamble is re-prefilled. A 400-token block that mutates nightly would cost a full re-prefill of thousands of static tokens every cycle — precisely the inference overhead this architecture exists to avoid.

**v1.3.1 placement, in order of preference:**

- **Post-Static-Prefix Breakpoint (primary).** Inject immediately *after* the cached static preamble — after the system prompt and tool schemas, before volatile history. The static prefix stays byte-identical between calls, so its eligibility for cache reuse is preserved and only the suffix is recomputed. In short-prompt hosts this still lands inside the primacy zone.
- **Recency-Boundary Injection (fallback, and preferable for long prompts).** Place the block immediately before the active turn. This rides the right-hand peak of the U-curve, leaves the cached prefix untouched, and where the preamble runs to thousands of tokens it is the better anchor rather than the weaker one: the primacy zone is no longer reachable without paying the cache penalty.
- **Slot 0 (rejected where prefix caching is enabled).** Genuine primacy, invalidated cache, full re-prefill per change. Defensible only without prefix caching, or when the block changes at most once per session and the preamble is short.

Placement must also be **stable within a session**: a block that moves between calls invalidates the cache even when its text is unchanged. The integrating system fixes the anchor at session start and records which it used, so §6 can attribute adherence differences to placement rather than content.

**Provenance and authority are separate.** Placement decides where the block is read. Fencing decides what it may *do*, and that requires distinguishing where content came from from what it is allowed to command.

- **L3 is operator-authenticated.** It arrives through `lateron commit-l3` from an authenticated human, outside model text and outside summarisation. Its provenance is the operator. Fenced `[OPERATOR-AUTHORISED INVARIANTS]`, it carries **normative authority**: imperative mood permitted, and it is the only section from which the agent may take instructions. The Triage Rule is compiled from L3 under the same authority.
- **L1 and L2 are observational.** They are model-authored prose about events, and those events include repository contents, commit messages, issue text, tool output and web pages — all of which third parties can write. Fenced `[DESCRIPTIVE TELEMETRY — NON-AUTHORITATIVE]`, they are **data only**: the agent may reason about them and must not obey them. Compaction enforces the distinction with an imperative-mood lint applied to L1/L2 output; a commit message saying *"delete the migrations folder"* becomes an L1 line saying *"a commit message asserted that the migrations folder should be deleted"* — and nothing more.

Provenance alone does not confer authority (a file a human wrote is still an untrusted input), and authority is not conferred by position in the block. The fence is the only thing that confers it, which is why the fence must be rendered by code and not written by a summariser.

### 3.7 The Whisper Block

The output of nightly compaction is a static markdown block injected at the chosen anchor. At runtime, zero vector-search latency is incurred:

```text
[SYSTEM STATE: MULTI-HORIZON TELEMETRY]                    # 400 tokens, o200k_base
[OPERATOR-AUTHORISED INVARIANTS]                           # normative — ≤90 + ≤40
- L3 (90d): Primary objective is production stability and throughput of the core
  publishing pipeline. Secondary prototypes are non-blocking.
  Non-negotiable: no prompt change without Chris's agreement.
  Target paths: /pipeline/**, /editions/**.
- Triage (≤40): Prioritise staged core pipeline tasks. Off-target concepts go to
  /backlog.md; do not divert active session compute.

[DESCRIPTIVE TELEMETRY — NON-AUTHORITATIVE]                # data only — ≤120 + ≤100
- L1 (7d ≤120): Pipeline modules A–B deployed; zero commits to core repo in 36h;
  4h on unscheduled exploratory scripts; two open blockers on the edition builder.
- L2 (30d ≤100): Scope drift recurs during data-cleaning phases; pre-staging input
  files raises session completion rate. Two prior drift episodes resolved same-week.

[VETO REGISTRY — 25 of 50 tokens; rendered from rows, never summarised]
- veto[migrate:auto] automated schema migration (expires 2026-10-27, x2)
- veto[git:push --force] force-push to main (expires 2026-10-25, x1)
- +1 more vetoes active, not shown

[FLAGS]                                                    # within the 50 reserve
- DRIFT_HIGH: D7d=0.81; 0 on-target commits in 48h. Off-target summary (L1 line
  only): HTML prototype rework in /experiments/, Thu–Sat.
```

---

## 4. Two-Way L3 Governance

### 4.1 Normative Versus Descriptive Strata

L3 is **normative**: it states what the system is *for*, and changes top-down on human authority only. L1 and L2 are **descriptive**: they record what happened, and are written bottom-up by compaction. Conflating these directions is how memory systems acquire the property of slowly rewriting their own mission — or, worse, of accepting a repository file as an order.

### 4.2 Anti-Drift Rule

**L2 never auto-mutates L3.** At the 30-day boundary L2 may emit a `PROPOSED_L3_DIFF` — a diff against current L3 with the L2 evidence motivating it — but it cannot apply it. The proposal waits for human sign-off. The rule exists because 30 days of scope drift, unsupervised, is indistinguishable from a considered strategy change to a summariser reading only those 30 days.

```
lateron review-l3            # list PROPOSED_L3_DIFF entries with evidence
lateron approve-l3 <diff_id> # commit the proposal to L3
lateron reject-l3 <diff_id>  # discard; optionally record a veto
```

### 4.3 Day-40 Pivot Rule

The anti-drift rule has a mirror-image risk: a project that genuinely pivots must not be held hostage by a mission statement written on day 1.

```bash
lateron commit-l3 --file new_objective.md --reason "pivot to match-graph pipeline"
```

Run by a human, this immediately: updates L3 in place; retires superseded invariants to a retired list rather than deleting them; flushes contradicted L1/L2 state; and resets the drift baseline so `D_7d` measures against the new target paths. L0 is untouched — the event log remains the record of what actually happened either side of the pivot.

It also **increments `state_version`**, which is what allows a compaction pass that started before the pivot to detect that it is stale and abort rather than overwrite the operator (§5).

The asymmetry is deliberate: the system may only *propose* a change to its mission; only a human may *make* one — in either direction.

---

## 5. Reference Implementation

Three corrections since v1.2, all of which would have destroyed state quietly in production: prior-state merging, ordered stratum reads inside one write, and a drift metric that measures on-target work. v1.3.1 adds two more: the veto registry as structured rows, and optimistic concurrency so that slow model calls never hold a write transaction open.

```python
# Lateron compaction worker (conceptual skeleton, v1.3.1)
import sqlite3, tiktoken

ENC = tiktoken.get_encoding("o200k_base")
BUDGET = {"L1": 120, "L2": 100, "L3": 90, "TRIAGE": 40, "REGISTRY": 50}
VETO_RENDER_BUDGET = 25          # tokens, inside the 50-token registry reserve

class ConcurrencyError(RuntimeError):
    """A pivot or another compaction moved state_version while we were distilling."""

def ntok(s: str) -> int:
    return len(ENC.encode(s))

def drift_ratio(conn, target_globs: list[str], days: int = 7) -> tuple[float, int]:
    """Deterministic. No model involved.

    Ratio is touches/touches, as the prose defines it: commits never enter the
    denominator. The 48h gate is filtered through target_globs, because counting
    commits with any non-null path measures operator activity, not on-target work.
    """
    touches = [p for (p,) in conn.execute(
        "SELECT path FROM events WHERE kind='touch' "
        "AND timestamp > datetime('now', ?)", (f"-{days} days",),
    ).fetchall()]
    touches_total = len(touches)
    touches_out = sum(1 for p in touches if p and not _matches(p, target_globs))

    commits_in_target = sum(1 for (p,) in conn.execute(
        "SELECT path FROM events WHERE kind='commit' "
        "AND timestamp > datetime('now','-48 hours')"
    ).fetchall() if p and _matches(p, target_globs))

    return (touches_out / max(1, touches_total)), commits_in_target

def render_veto_registry(conn) -> tuple[str, int]:
    """Deterministic rendering of live veto rows. A veto is a row, not prose.

    Overflow: keep the most recently triggered entries, count the rest.
    """
    rows = conn.execute(
        "SELECT scope, reason, expires_at, trigger_count FROM vetoes "
        "WHERE expires_at > datetime('now') "
        "ORDER BY last_triggered_at DESC"
    ).fetchall()

    lines, overflow = [], 0
    for scope, reason, exp, hits in rows:
        line = f"- veto[{scope}] {reason} (expires {exp[:10]}, x{hits})"
        if ntok("\n".join(lines + [line])) > VETO_RENDER_BUDGET:
            overflow += 1
            continue
        lines.append(line)
    if overflow:
        lines.append(f"- +{overflow} more vetoes active, not shown")
        _log_event(conn, kind="veto_overflow", value=overflow)   # auditable omission
    return "\n".join(lines), overflow

def compact(db_path: str, max_attempts: int = 3) -> None:
    """Read, think, then write — and only if nobody moved state underneath us."""
    for _ in range(max_attempts):
        # ---- read phase: read-only, no lock held across model calls ----
        ro = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        snapshot = _state_version(ro)
        last = _cursor(ro)
        events = ro.execute(
            "SELECT id, role, content, kind, path, timestamp FROM events "
            "WHERE id > ? ORDER BY id", (last,)
        ).fetchall()
        if not events:
            ro.close()
            return

        prior_l1 = _read_stratum(ro, "L1")
        prior_l2 = _read_stratum(ro, "L2")
        l2_snapshots = _l1_snapshot_history(ro, days=30)   # trailing L1 snapshots
        l3 = _read_stratum(ro, "L3")
        globs = _l3_target_globs(ro)
        D, commits_in = drift_ratio(ro, globs)
        ro.close()

        # ---- compute phase: slow calls, NO transaction open ----
        if D > 0.75 and commits_in == 0:
            _raise_flag(db_path, "DRIFT_HIGH",
                        {"D7d": round(D, 3), "commits_in_target_48h": 0})

        # Merge live prior state, or a pass deletes every open blocker that
        # was not re-mentioned today. Vetoes are not in this text at all.
        l1 = _distil(events, prior=prior_l1, rule="tactical", cap=BUDGET["L1"])
        _assert_descriptive(l1)          # trust boundary: no imperatives in L1

        # L2 distils the trailing L1 snapshots, never the live veto rows, which
        # are rendered directly by render_veto_registry instead.
        l2 = _distil(l2_snapshots, prior=prior_l2, rule="patterns", cap=BUDGET["L2"])
        _assert_descriptive(l2)
        proposal = _maybe_propose_l3_diff(l2, l3)

        # ---- write phase: atomic, and conditional on state_version ----
        w = sqlite3.connect(db_path)
        try:
            with w:
                bumped = w.execute(
                    "UPDATE lateron_meta SET value = value + 1 "
                    "WHERE key = 'state_version' AND value = ?", (snapshot,)
                )
                if bumped.rowcount != 1:
                    raise ConcurrencyError("state moved during distillation")

                _write_stratum(w, "L1", l1)
                _write_stratum(w, "L2", l2)
                if proposal:
                    _store_proposal(w, proposal)   # awaits `approve-l3`
                _advance_cursor(w, events[-1][0])  # L0 rows are never deleted
            return
        except ConcurrencyError:
            continue          # re-read: an operator pivot (or a peer worker) won
        finally:
            w.close()
    raise ConcurrencyError("compaction aborted after repeated version conflicts")
```

Two things this buys. First, a model call can take ninety seconds and hold no write lock, so the agent keeps writing L0 throughout and a `commit-l3` is never blocked. Second, because the pivot *and* every `lateron veto` bump `state_version`, a pass that read the world before the operator's decision cannot publish strata computed against the pre-pivot mission: it detects the version change, rolls back and re-reads. Version-conflict handling is the whole reason the pivot rule in §4.3 is safe to run during a live session.

Note what the skeleton does *not* do: it never deletes L0 rows, never lets `_distil` write L3, never replaces L1 or L2 without prior state, never re-summarises a veto, and never asks a model whether drift occurred.

---

## 6. Evaluation Protocol

Four arms at an equal 400-token budget. The design is nested: it first asks whether structured lifecycle management beats a plain pinned summary (C vs B2), then whether multi-timescale partitioning beats the same lifecycle apparatus without the partition (C vs B3).

### 6.1 Arms

- **A — No Memory.** Context truncated to the working window.
- **B — Rolling-*N* Uniform Summary.** 400 tokens, single summariser, sliding window.
- **B2 — Pinned Invariants + Shared Summary.** 90 tokens pinned L3 + 310-token rolling summary = 400. Protects the mission as well as Lateron's L3 at the same cost, with no strata and no lifecycle machinery.
- **B3 — Mechanism-Matched, Single Timescale.** 90 pinned L3 + 40 triage + 50 structured veto registry, headers and flags + **220 tokens of one shared descriptive summary** = 400. Identical registry, identical pivot controls, identical placement, identical budget — the only difference from C is that the 220 descriptive tokens are **not** partitioned into 120/100 across two timescales.
- **C — Lateron.** Strata and caps per §2.2 and §2.4; anchor per §3.6; fences per §3.6.

B3 is the arm that makes this a paper rather than a brochure. It holds constant every piece of engineering that is easy to describe and hard to justify, and varies only the thing under test.

### 6.2 Tasks and Horizon

Twenty-four tasks in three families, each run across **45 compressed cycles** (one per simulated day), so that the 30-day L2 boundary fires, at least one `PROPOSED_L3_DIFF` is generated and adjudicated, and the day-40 pivot lands with five cycles of aftermath. The 12-day horizon in v1.2 could not fire the mechanisms it was testing; a seven-day TTL and a thirty-day boundary need a calendar, not a sprint.

1. **Long-horizon feature work** — multi-day implementation, stable objective, periodic off-target temptations.
2. **Mid-stream pivot** — objective changed by operator instruction at cycle 40 (tests §4.3 and the §5 concurrency guard against L3 inertia).
3. **Veto-heavy operation** — repeated proposals of a rejected action across the full horizon, including past the TTL boundary, so expiry and re-trigger are observable.

### 6.3 Metrics

- **Invariant retention** — blind rubric scoring against the day-0 invariant set, sampled at cycles 3, 10, 20, 30, 40, 45. Expected near-parity across B2, B3 and C; reported to confirm the baselines are strong, not to win an argument.
- **Veto suppression** — re-proposals per session while a constraint is live, and again after expiry.
- **Pattern extraction** — recall and precision of recurring-failure lines against a human-labelled list of the failure modes actually present across 45 cycles. This is the partition's own test.
- **Pivot governance** — stale-invariant violations and cycles-to-recovery after the cycle-40 pivot, plus operator effort (manual prompt surgery for B/B2 versus one `commit-l3` for B3/C).
- **Drift rate** — $D_{7d}$ per §3.4, validated against human labels.
- **Cost** — tokens per session, wall-clock latency, and **prefix-cache hit rate**, since the placement claim is a cost claim.

### 6.4 Hypotheses and Falsification

- **H1 (lifecycle):** C and B3 both re-propose vetoed actions less often than B2, and are indistinguishable from each other. The registry is the mechanism; the partition is not.
- **H2 (partition):** C recovers more true recurring failure modes than B3 at equal budget, because L2's capacity cannot be consumed by L1 churn.
- **H3 (governance):** C and B3 are indistinguishable on pivot violations and operator effort, and both beat B2. Shared apparatus should behave identically.
- **H4 (invariant parity):** B2, B3 and C are near-parity on invariant retention. Parity confirms the baselines are fair; C winning here would suggest the pinned blocks were mis-specified.
- **H5 (cost):** tokens, latency and prefix-cache hit rate are indistinguishable across B2, B3 and C under §3.6 placement.

**Nested falsification, stated in advance:** if B3 ≥ C on H2, multi-timescale partitioning does not earn its capacity rigidity and weekly lag, and the correct design is B3. If B2 ≥ B3 on H1 and H3 as well, structured lifecycle management does not earn its complexity either, and the correct design is B2. Either result is a publishable finding; neither can be reinterpreted as a partial success after the fact.

### 6.5 Reproduction

Harness, task definitions, prompts, seeds and per-run JSONL logs published alongside this document. Model versions pinned per run; all token counts via `o200k_base`; every run records its anchor (`post-static-prefix` or `recency`), its `D_7d` trace, its veto-overflow count, and its prefix-cache hit rate. A run is reproducible if replaying L0 from cursor 0 with a pinned model yields the same stratum text.

---

## 7. Conclusion

As context windows scale, **context quality rather than raw context capacity sets the practical ceiling on agent reliability**. Uniform rolling summarisation preserves the ratio of signal to noise while destroying the absolute quantity of both, which is why long-horizon agents keep their tone and lose their mission. A pinned objective block fixes that symptom cheaply — so the interesting question is not whether invariants survive, but what else a memory system can do that a pinned block cannot.

Lateron's answer is three mechanisms, only one of which requires partitioning. A structured veto registry holds a human's rejection as an expiring, testable predicate rather than a sentence a summariser may soften. Multi-timescale strata keep tactical churn from evicting the diagnosis of a recurring failure, at the known price of rigid capacity and a weekly lag. Two-way governance lets the system propose a change to its own mission and never make one, while the operator can pivot synchronously and be certain no in-flight compaction overwrites the decision. The block carrying this is 400 tokens, placed so the static preamble stays cache-eligible, and fenced so that summarised repository content can never be mistaken for an order.

This is offered as a specification with a specified falsification test, not as a measured result. The test is built to separate two claims that most memory papers merge, and to lose cleanly if either does not hold.

---

## References

1. Liu, N. F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., & Liang, P. (2023). *Lost in the Middle: How Language Models Use Long Contexts.* arXiv:2307.03172. <https://arxiv.org/abs/2307.03172>
2. Wei, T., Sachdeva, N., Coleman, B., et al. (2025). *Evo-Memory: Benchmarking LLM Agent Test-time Learning with Self-Evolving Memory.* UIUC & Google DeepMind. arXiv:2511.20857 (v2, May 2026). <https://arxiv.org/abs/2511.20857>
3. Li, K., Yu, X., Ni, Z., et al. (2026). *TiMem: Temporal-Hierarchical Memory Consolidation for Long-Horizon Conversational Agents.* Findings of ACL 2026, pp. 21700–21720. <https://aclanthology.org/2026.findings-acl.1091/>
4. Zhang, N., Yang, X., Tan, Z., Deng, W., & Wang, W. (2026). *HiMem: Hierarchical Long-Term Memory for LLM Long-Horizon Agents.* arXiv:2601.06377. <https://arxiv.org/abs/2601.06377>
5. Compresr Research (2026). *Context Compression Architecture: 2026 Complete LLM Guide (Hard-Token, Soft-Token, and KV Cache Designs).* <https://compresr.ai/blog/context-compression-architecture-complete-guide>
6. Ahmed, M. S. (2026). *LLM Architecture 2026: Primacy, Middle, and Recency Context Attention Zones in Production Agents.* RankSquire Engineering. <https://ranksquire.com/2026/04/13/llm-architecture-2026/>
7. Trope, Y., & Liberman, N. (2010). *Construal-level theory of psychological distance.* Psychological Review, 117(2), 440–463. <https://doi.org/10.1037/a0018963>
8. Modular (2026). *Prefix caching.* LLM Inference Handbook. <https://handbook.modular.com/inference-optimization/prefix-caching/>

---

## Citation

```bibtex
@techreport{dryland2026lateron,
  title  = {Lateron: Multi-Horizon Hard-Token Context Compression and
            Primacy-Zone State Steering for Long-Horizon LLM Agents},
  author = {Dryland, Christopher},
  year   = {2026},
  month  = {October},
  note   = {Systems Architecture Specification, v1.3.1},
  url    = {https://github.com/superhanzzz/lateron}
}
```

## Changelog

**v1.3.1** — Separated the veto registry from prose: active vetoes are structured `vetoes` rows (scope, `expires_at`, `trigger_count`) rendered deterministically into the control block by Python, with a 25-token sub-cap, most-recently-triggered retention, and an auditable overflow count (§3.5, §5). Replaced the write-transaction-held-across-model-calls pattern with optimistic concurrency: read phase in read-only mode, `_distil` outside any transaction, then an atomic write gated on a `state_version` check so a concurrent `commit-l3` pivot aborts the stale pass instead of being overwritten (§5, §4.3). Reframed placement as preserving the static preamble's eligibility for prefix-cache reuse, citing the Modular LLM Inference Handbook, and separated provenance (operator-authenticated L3 versus observational L1/L2) from authority conferred by the fence (§3.6). Stated the partition trade-off explicitly — tactical churn cannot evict recurring patterns, against rigid capacity and weekly update lag — and added **Arm B3**, a mechanism-matched single-timescale baseline, so the benchmark prices partitioning separately from lifecycle management (§2.3, §6). Rebalanced to 120/100/90/40/50 = 400 with a structured registry reserve (§2.4).

**v1.3** — Corrected three defects in the §5 skeleton (`_distil` now receives live prior state; L2 distils the freshly written L1 atomically with the cursor advance; `drift_ratio()` filters both sides through `target_paths` and keeps the denominator to touches only). Reframed §2.3 to scope the $\alpha^k$ model to unstructured summarisation. Added Arm B2 and extended the horizon to 45 cycles. Replaced Slot 0 injection with the Post-Static-Prefix Breakpoint. Added the trust boundary with an imperative-mood lint. Reserved header overhead in the budget.

**v1.2** — Staged Rolling-*N* Uniform Summarisation as the explicit baseline; promoted rejection telemetry to §3.5 with a seven-day TTL; made L3 normative and two-way governed (§4); made L0 an immutable event-sourced log with replay (§3.2); pinned tokenisation; specified the evaluation protocol (§6).

**v1.1** — Initial systems architecture specification.

## License

This work is licensed under a [Creative Commons Attribution 4.0 International License](LICENSE).
