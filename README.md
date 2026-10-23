# Lateron: Multi-Horizon Hard-Token Context Compression and Primacy-Zone State Steering for Long-Horizon LLM Agents

**Author:** Christopher Dryland (Independent Researcher, Canberra, Australia)
**Date:** October 23, 2026
**Version:** 1.3 (Systems Architecture Specification)
**License:** [Creative Commons Attribution 4.0 International (CC BY 4.0)](LICENSE)

> **Changes in v1.3.** Corrects three bugs in the §5 skeleton (state loss on distillation, out-of-order stratum reads, non-atomic writes) and aligns `drift_ratio()` with the prose definition; reframes §2.3 to concede that the $\alpha^k$ decay model describes *unstructured* summarisation only, and adds Arm B2 — a pinned-invariant baseline — to §6 so the architecture must beat the strongest available comparison rather than a straw man; replaces Slot 0 injection with a **Post-Static-Prefix Breakpoint** that preserves host KV prefix caches, and adds an explicit **trust boundary** fencing L3 as operator-authorised and L1/L2 as non-authoritative telemetry; reserves 30 tokens of header overhead in the §2.4 budget.

> **Changes in v1.2.** Adds a staged competitor baseline against Rolling-*N* Uniform Summarisation (§1.4, §2.1–2.3); promotes rejection telemetry to its own subsection with a seven-day TTL (§3.5); replaces Primacy-Zone residency with a placement rule that concedes host-prompt-length dependency (§3.6); replaces LLM-judged drift detection with a deterministic L0 preprocessor metric (§3.4); makes L3 normative and two-way governed (§4); makes L0 an immutable event-sourced log with a replay cursor (§3.2); pins tokenisation to `o200k_base`, rebalances the 400-token budget, sets a compound compaction trigger, and specifies the evaluation protocol (§2.4, §3.3, §6). Spelling standardised to British English.

---

## Abstract

Long-horizon Large Language Model (LLM) agents degrade in reliability as interaction history grows. While recent models support million-token context windows, production engineering reports indicate that transformer instruction fidelity begins to decay once context fill passes roughly 60–70%, with retrieved memories injected into the middle context zone (15–85%) suffering from pronounced attention dilution. Furthermore, standard Retrieval-Augmented Generation (RAG) and flat episodic memory stores treat stale implementation noise and high-level system invariants with equal retrieval weight.

This paper presents **Lateron**, an out-of-band, multi-horizon hard-token compression architecture for stateful LLM agents. Its baseline is the dominant production default — Rolling-*N* Uniform Summarisation — in which a single summarisation operator is applied to a sliding window of history, reducing transient implementation noise and durable objective invariants at the same rate. That comparison is not the hardest one available: a pinned objective block plus a rolling summary also protects invariants, and §6 requires Lateron to beat that stronger baseline too, on mechanisms a pinned block does not have.

Lateron applies **Horizon-Based Resolution Scaling** (adapted from Construal Level Theory) across discrete temporal strata (24-Hour, 7-Day, 30-Day, and 90-Day horizons). As state logs age, low-level execution mechanics (*how* tasks were executed) are systematically pruned, while high-level trajectory invariants (*why* the system is operating and its governing constraints) are distilled upward. The resulting state is reconciled nightly into a compact 400-token control block, injected after the host's cached static prefix — or immediately before the active turn — so that it occupies an attention peak while leaving KV prefix caches intact. Consolidation is driven by a deterministic drift metric over an immutable event-sourced log; the invariant stratum is fenced as operator-authorised and changed only top-down, while the lower strata are fenced as non-authoritative telemetry that cannot escalate into instructions.

---

## 1. Introduction: The Production Bottleneck in Long-Horizon Agents

Autonomous and semi-autonomous LLM agents operating over weeks or months face three structural failure modes in production.

### 1.1 Middle-Zone Attention Degradation

Transformer attention distribution is heavily U-shaped. Performance on tasks requiring information from the middle of long contexts degrades as context length grows, even for models explicitly built for long context (Liu et al., 2023). Production engineering reports place the **Primacy Zone** at roughly the first 0–15% of tokens and the **Recency Zone** at the final 5–10%, with the **Middle Zone** (15–85%) receiving degraded attention (RankSquire, 2026). Standard memory pipelines inject retrieved vector chunks into this middle zone, increasing token cost while yielding poor instruction adherence.

### 1.2 Temporal Contradiction and Retrieval Pollution

Append-only vector databases retrieve past logs on semantic similarity rather than temporal validity. As shown in the Evo-Memory benchmark (Wei et al., 2025) and HiMem (Zhang et al., 2026), failing to prune superseded operational states causes agents to hallucinate obsolete constraints or repeat failed tool paths.

### 1.3 Uniform Summarisation Loss

Existing recursive summarisation pipelines compress text uniformly — shrinking transient syntax errors and core project objectives at the same rate until long-term system goals vanish into generic prose.

### 1.4 The Baseline That Must Be Beaten

§1.3 is normally stated as an intuition and left there. It should not be. The dominant production pattern is not vector RAG and not hierarchical memory: it is **Rolling-*N* Uniform Summarisation**, the behaviour behind `/compact`-style context management in agent frameworks and coding assistants. A single operator is applied to a sliding window of history under a fixed token budget, and the operator has no notion of what kind of information it is compressing.

Two baselines follow from that, and only the second is a fair fight. **Arm B** is the naked summariser. **Arm B2** — a 90-token pinned objective block plus a 310-token rolling summary at the same 400-token total — is the version a competent engineer would actually ship, and it defeats the simple form of the argument in §1.3 without any strata at all. Lateron must beat B2, not B, and §6 is written accordingly.

---

## 2. Horizon-Based Resolution Scaling

### 2.1 The Production Default: Rolling-*N* Uniform Summarisation

Define a history $H$ of token length $|H| > N$. Rolling-*N* applies a compression operator $U$ such that $U(H) \rightarrow \hat{H}$ with $|\hat{H}| \le N$, typically by an instruction of the form *"summarise the conversation so far, keeping what matters"*, applied cumulatively as the window advances.

Three properties follow, and all three are load-bearing for the weak form of the argument:

1. **One operator, no strata.** $U$ cannot treat a resolved stack trace and a non-negotiable deployment constraint as different kinds of object, because nothing in the pipeline tells it that they are.
2. **Compounding decay.** Applied $k$ times over a long horizon, every surviving item is the result of $k$ successive re-summarisations. Salience that is not reinforced by recency is progressively eroded.
3. **Recency-symmetric attention.** Because the operator reads a sliding window, material that has aged out of the window is either summarised or gone; there is no stratum in which an old invariant has a *protected* representation.

Rolling-*N* is genuinely good at what it was designed for: bounding context and preserving recent working state. Its weakness is specific — on its own, it protects the wrong thing at long horizons.

### 2.2 Horizon-Based Resolution Scaling

Lateron replaces the single operator with a set of stratum-specific resolution rules indexed to age, derived from **Construal Level Theory** (Trope & Liberman, 2010): *temporal distance dictates abstraction resolution*. The mapping is borrowed as a design heuristic, not asserted as a cognitive mechanism. The system does not claim that agents "think like people"; it claims that the *category* of information worth retaining changes with age, and that a different operator per age band encodes that.

Instead of compressing all information equally, each stratum enforces a filter on what survives compaction:

| Temporal Stratum | Resolution Rule | Retained State | Pruned State | Token Cap |
| --- | --- | --- | --- | --- |
| **L0: Event Log** | Immutable, append-only | All telemetry — tool calls, git diffs, session logs, errors, operator overrides | Nothing (never pruned on disk) | Unbounded (disk) |
| **L1: 7-Day Stratum** | Tactical state (the *how*) | Active task velocity, open blockers, uncommitted branches, operator vetoes | Resolved errors, raw conversational turns, intermediate tool outputs | ≤ 130 tokens |
| **L2: 30-Day Stratum** | Operational patterns (system dynamics) | Throughput bottlenecks, recurring failure modes, architectural drift, workflow dependencies | Completed task checklists, isolated daily variance, deprecated experiments | ≤ 110 tokens |
| **L3: 90-Day+ Stratum** | Invariant objectives (the *why*) | Core project specification, non-negotiable constraints, primary objective function | All low-level implementation mechanics and historical tactical state | ≤ 90 tokens |
| **Runtime Triage Rule** | Compiled directive | The single priority instruction for the active session | — | ≤ 40 tokens |
| **Headers, Flags, TTL Metadata** | Structural overhead | Section fences, `DRIFT_HIGH` state, veto timestamps and expiry stamps | — | ≤ 30 tokens |

### 2.3 Equal-Rate Decay and Its Limits

Let $H = H_{inv} \cup H_{noise}$, where $H_{inv}$ is the token mass carrying objective invariants and $H_{noise}$ the token mass carrying transient implementation detail. An **unstructured** uniform operator $U_{uniform}$ is applied to the union, so to first order each component is reduced by the same factor:

$$|U_{uniform}(H)| \approx \alpha \cdot |H|, \qquad |U_{uniform}(H_{inv})| \approx \alpha \cdot |H_{inv}|$$

The ratio $|U_{uniform}(H_{inv})| / |U_{uniform}(H_{noise})|$ is preserved while the *absolute* invariant mass shrinks by $\alpha$ on every pass, with $\alpha$ set by the budget rather than by importance. Over $k$ nightly passes the invariant fraction decays as $\alpha^k$: at 90 days the agent still says something about the project, but the sentence has been re-summarised eleven times and now constrains nothing.

**That model has a narrow scope, and v1.2 overstated it.** $\alpha^k$ describes what happens when nothing in the pipeline is told to protect anything. It does *not* describe:

- an **instruction-guided** summariser, told explicitly to preserve objectives and non-negotiables; or
- a **pinned objective block** — 90 tokens of invariants held outside the rolling window and re-injected verbatim at every call.

Both mitigate the decay substantially, and Arm B2 (§6.1) is exactly that design. Any claim that stratification is necessary because otherwise invariants evaporate is a claim against a straw man, and this paper withdraws it.

What survives is narrower and more interesting. A pinned block and Lateron both protect L3; protection is therefore **not** the discriminator. The architecture has to justify itself on the three things a pinned block does not do:

1. **Veto suppression (L1).** A pinned block has no mechanism for recording that a human rejected an action on Tuesday, so a pinned-block agent re-proposes it on Wednesday. Lateron carries the veto as a TTL-bound negative constraint.
2. **Pattern extraction (L2).** A pinned block states intent; it does not compress thirty days of failure modes into recurring-diagnosis lines, because it has no stratum at that horizon.
3. **Pivot governance (L3).** A pinned block is edited by hand, ad hoc, with no proposal path, no retirement of superseded invariants, and no reset of the drift baseline. Lateron makes the change synchronous, auditable, and reversible in the event log.

Invariant mass remains bounded from below by the L3 cap rather than multiplied by $\alpha$ each night:

$$|L_3|_{t+1} \ge \min(\beta_{L3},\, |L_3|_t), \qquad \beta_{L3} = 90 \text{ tokens}$$

but this floor is a *budget guarantee*, and Arm B2 has it too. §6 is written so that Lateron fails if it cannot show a measurable advantage on (1), (2) and (3) at matched budget.

### 2.4 Token Budget and Tokenisation

All caps are counted with `tiktoken`'s `o200k_base` encoding, so every number in this document is reproducible. The budget totals exactly 400 tokens:

- **L1 (7-day tactical):** 130 tokens
- **L2 (30-day patterns):** 110 tokens
- **L3 (90-day invariants):** 90 tokens
- **Runtime Triage Rule:** 40 tokens
- **Headers, flags and TTL metadata:** 30 tokens

The 30-token header reserve exists because the block is not only its payload: section fences, the `DRIFT_HIGH` line, and per-veto timestamps are real tokens, and v1.2 left them uncounted. A budget that omits its own overhead is not a budget.

The split is a working default chosen so that the invariant stratum survives all compaction, and is expected to be tuned per deployment; it is not derived from first principles. Lateron does not control the host system prompt and makes no claim about total context length — only about the size of the block it contributes.

---

## 3. System Architecture & Pipeline Execution

Lateron decouples memory consolidation (asynchronous, out-of-band) from live inference (synchronous, zero-latency).

### 3.1 Passive Telemetry Ingestion

Rather than requiring explicit "save-to-memory" tool calls during live inference — which consume tool-calling turns and increase latency — Lateron ingests passive operational exhaust:

- **Session state logs:** read-only queries against the agent's local SQLite state store (message history, tool execution status, timestamps).
- **Repository and filesystem state:** per-session `git diff --stat` and `--name-only`, commit metadata, and file modification timestamps across active workspaces.

### 3.2 Event-Sourced L0

L0 is an **immutable, append-only event log** in SQLite. Compaction never deletes rows. A `last_compacted_id` cursor marks the high-water mark of processed events; the compaction worker reads forward from the cursor and advances it only inside the same transaction that writes the new stratum state.

Two consequences matter:

1. **Pruning applies to injection, not storage.** What is pruned is the *representation* placed in context. The underlying events remain on disk, so a bad nightly distillation is recoverable rather than permanent.
2. **Deterministic replay.** Because L0 is an ordered log with a cursor, a failed or suspect compaction can be replayed from any prior cursor position against a pinned model and prompt, producing the same strata. Memory becomes auditable, which vector stores are not.

### 3.3 Compaction Trigger

Compaction runs on a compound trigger, whichever fires first:

$$\text{compact when } \min(\text{24h wall-clock},\ 8{,}000\ \text{uncompacted tokens})$$

The token clause closes the session-length gap: an agent running for five days without termination does not silently accumulate unbounded uncompacted state, which a wall-clock-only trigger permits.

### 3.4 Deterministic Drift Detection

Drift must not be a second LLM's opinion of a first LLM's work. Lateron computes it **before** any compaction call, in a preprocessor over L0:

1. L3 declares `target_paths` — a set of glob patterns — and a target repository scope.
2. Over the trailing seven days, count tool **touches** (`kind='touch'`) to files outside `target_paths` (`touches_out`) against all touches (`touches_total`). The denominator counts touches only: mixing commit rows into it would let a burst of commits dilute a touch-heavy drift signal, and would make the metric move when the operator committed more, not when the agent wandered more.
3. Compute the **Drift Ratio**:

$$D_{7d} = \frac{\text{touches}_{out}}{\max(1,\ \text{touches}_{total})}$$

4. Separately, and with a different window, count commits whose changed path matches `target_paths` (`commits_in_target`). A commit that touches any in-scope file counts as in-scope; commits are never counted in the $D_{7d}$ denominator.
5. Raise `DRIFT_HIGH` deterministically when both conditions hold:

$$D_{7d} > 0.75 \quad \text{and} \quad \text{commits}_{in\_target} = 0 \text{ in the last 48h}$$

Both sides of the commit test apply `target_paths` matching. A count of "commits with a non-null path" is not a measure of on-target work, and v1.2's skeleton shipped exactly that bug (§5).

The flag is arithmetic, auditable, and reproducible. The model's only role is to summarise the off-target activity into one L1 line and attach it to the flag — it can neither raise nor clear the flag. A false positive is therefore a threshold-tuning problem, not a reasoning failure, and the threshold is a single configurable constant.

### 3.5 Rejection Telemetry: Stratum-Resident Negative Constraints

When a human operator vetoes or aborts an agent-proposed action during a live run, the event is recorded as a **negative constraint in L1** with a **seven-day TTL**.

This is the mechanism that distinguishes Lateron from summarisation-based memory, and it is deliberately *local*. A veto is evidence about the tactical present — the operator did not want this action, in this context, this week — not a permanent statement about the mission. Placing it in L3 would make one bad afternoon's judgement a constitutional constraint on the project. Placing it in L1 makes it a bounded, decaying inhibition:

- **Suppresses action-looping locally.** The agent reads the constraint in its L1 block and does not re-propose the rejected action in subsequent sessions within the window.
- **Does not mutate L3.** The invariant stratum is untouched by operational friction; the mission statement cannot be edited by an annoyed operator at 11pm.
- **Expires cleanly.** At TTL, the constraint drops out unless re-triggered by a fresh veto. If the operator rejects the same action again, the TTL resets — repetition is evidence the constraint is real, and it earns its continued place without any promotion to L3.

Rejected-action loops are among the most expensive failure modes in long-horizon agents, because each repetition burns a full tool-call round trip to re-derive a decision a human already made. Lateron treats the human's veto as the cheapest available gradient signal and stores it at the shortest stratum that can act on it.

**Corollary for the implementation:** because L1 carries state with a lifetime longer than one compaction run, every distillation pass must merge into the existing L1 rather than replace it. A pass that distils only today's events silently deletes every unexpired veto and every open blocker that was not mentioned again today. v1.2's skeleton had this bug; §5 fixes it.

### 3.6 Placement: Post-Static-Prefix Breakpoint, and the Trust Boundary

v1.1 claimed injection "into the Primacy Zone". That claim assumed a short host system prompt, and Lateron does not control host prompt length. v1.2 replaced it with a dual anchor and picked Slot 0 — prepending the block ahead of the static system prompt and tool schemas — as preferred. **That choice is wrong wherever the host uses KV prefix caching**, which is to say in most current inference stacks.

**The cache trap.** A KV prefix cache is keyed on a token prefix. If a dynamic block is placed *ahead of* a static preamble, then every time the block changes, the cached prefix diverges at its first token, and the entire downstream prefix — system prompt plus tool schemas, thousands of tokens — must be re-prefilled. A 400-token block that mutates nightly therefore costs a full re-prefill of the whole preamble every cycle, which is precisely the inference overhead this architecture exists to avoid. Slot 0 wins the attention argument and loses the cost argument, and the cost argument is the one that shows up on the bill.

**v1.3 placement, in order of preference:**

- **Post-Static-Prefix Breakpoint (primary).** Inject immediately *after* the cached static preamble — after the system prompt and tool schemas, before volatile session history. The static prefix is unchanged byte-for-byte between calls, so the prefix cache holds and only the suffix is recomputed. The block still lands far earlier than a middle-zone injection, and in short-prompt hosts it sits squarely in the primacy zone.
- **Recency-Boundary Injection (fallback, and preferred where the prompt is long).** Place the block immediately before the active turn. This rides the right-hand peak of the U-curve and also preserves the prefix cache, since the cached prefix is untouched. Where the host preamble exceeds a few thousand tokens, this is the better anchor, not the weaker one: the primacy zone is no longer reachable without paying the cache penalty.
- **Slot 0 (rejected where prefix caching is enabled).** Genuine primacy, invalidated cache, full re-prefill per change. Defensible only where the host has no prefix caching, or where the block changes at most once per session and the preamble is short.

Placement must also be **stable within a session**. A block that moves between calls invalidates the cache even if its text is unchanged. The integrating system therefore fixes the anchor at session start and records which one it used, so §6 can attribute adherence differences to placement rather than content.

**Trust boundary.** Placement controls where the block is read; fencing controls what it is allowed to *do*. The block mixes content with very different provenance: L3 is written by a human, while L1 and L2 are distilled by a model from L0 — and L0 contains repository contents, commit messages, tool output, issue text and web pages, any of which an outside party can write. Summarised attacker-controlled text must never acquire the force of a governing instruction, so the block carries explicit fences:

- **`[OPERATOR-AUTHORISED INVARIANTS]`** — L3 and the Triage Rule. Normative authority. Imperative mood permitted. This is the only section from which the agent may take instructions.
- **`[DESCRIPTIVE TELEMETRY — NON-AUTHORITATIVE]`** — L1 and L2. Data only. The agent may reason about it but must not treat it as instruction, and compaction enforces this: a lint rejects imperative constructions in L1/L2 output and rewrites them as descriptive statements before they are stored. If a commit message says *"delete the migrations folder"*, the L1 line may say *"a commit message asserted that the migrations folder should be deleted"* — and nothing more.

The rule is cheap to state and cheap to enforce, and its absence is the difference between a memory system and an injection vector that persists for seven days.

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

[DESCRIPTIVE TELEMETRY — NON-AUTHORITATIVE]                # data only — ≤130 + ≤110
- L1 (7d ≤130): Pipeline modules A–B deployed; zero commits to core repo in 36h;
  4h on unscheduled exploratory scripts. Two vetoes live: automated schema
  migration (expires 2026-10-27); force-push to main (expires 2026-10-25).
- L2 (30d ≤110): Scope drift recurs during data-cleaning phases; pre-staging input
  files raises session completion rate. Two prior drift episodes resolved same-week.

[FLAGS]                                                    # ≤30 with headers
- DRIFT_HIGH: D7d=0.81; 0 on-target commits in 48h. Off-target summary (L1 line
  only): HTML prototype rework in /experiments/, Thu–Sat.
```

---

## 4. Two-Way L3 Governance

### 4.1 Normative Versus Descriptive Strata

L3 is **normative**: it states what the system is *for*, and changes top-down, on human authority only. L1 and L2 are **descriptive**: they record what has actually happened, and are written bottom-up by the compaction process. Conflating these two directions is how memory systems acquire the property of slowly rewriting their own mission — or, worse, of accepting a repository file as an order.

### 4.2 Anti-Drift Rule

**L2 never auto-mutates L3.** At the 30-day boundary, L2 may emit a `PROPOSED_L3_DIFF` — a diff against the current L3, with the L2 evidence that motivates it — but it cannot apply the change. The proposal sits in the state block until a human signs it off. The rule exists precisely because 30 days of scope drift, left unsupervised, is indistinguishable from a considered strategy change to a summariser reading only the last 30 days.

Approval is explicit:

```
lateron review-l3            # list PROPOSED_L3_DIFF entries with evidence
lateron approve-l3 <diff_id> # commit the proposal to L3
lateron reject-l3 <diff_id>  # discard; optionally annotate as a negative constraint
```

### 4.3 Day-40 Pivot Rule

The anti-drift rule has a mirror-image risk: a project that genuinely pivots must not be held hostage by a mission statement written on day 1. If the operator's objectives change, the agent needs a synchronous path to say so, not a monthly boundary.

```bash
lateron commit-l3 --file new_objective.md --reason "pivot to match-graph pipeline"
```

This command, run by a human, immediately: updates L3 in place; retires superseded invariants by moving them to a retired list rather than deleting them; flushes contradicted L1 and L2 state (dropping blocks whose content contradicts the new invariant set); and resets the drift baseline so that `D_7d` is measured against the new target paths from that point. L0 is untouched — the event log remains the record of what actually happened before and after the pivot.

The asymmetry is deliberate: the system can only ever *propose* a change to its mission, and only a human can *make* one — in either direction, whether that means accepting drift-driven change or overriding a stale mission.

---

## 5. Reference Implementation

The skeleton below carries four corrections against v1.2, each marked. They are ordinary bugs rather than architectural ones, and each would have quietly destroyed state in production.

```python
# Minimal Lateron compaction worker (conceptual skeleton, v1.3)
import sqlite3, subprocess, tiktoken

ENC = tiktoken.get_encoding("o200k_base")
BUDGET = {"L1": 130, "L2": 110, "L3": 90, "TRIAGE": 40, "HEADERS": 30}

def ntok(s: str) -> int:
    return len(ENC.encode(s))

def due(db_path: str) -> bool:
    """Compound trigger: 24h wall clock OR 8,000 uncompacted tokens."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cur = conn.execute("SELECT value FROM lateron_meta WHERE key='last_compacted_id'").fetchone()
    last = cur[0] if cur else 0
    row = conn.execute(
        "SELECT SUM(tokens), MIN(timestamp) FROM events WHERE id > ?", (last,)
    ).fetchone()
    tokens, oldest = row[0] or 0, row[1]
    if tokens >= 8000:
        return True
    return oldest is not None and _age_hours(oldest) >= 24

def drift_ratio(db_path: str, target_globs: list[str], days: int = 7) -> tuple[float, int]:
    """Deterministic pre-compaction metric. No model involved.

    FIX 3 (v1.3): the ratio is touches/touches, as the prose defines it —
    commits are never in the denominator. The 48h commit count is filtered
    through target_globs; counting rows with any non-null path measured
    "the operator committed something", not "on-target work happened".
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)

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

def compact(db_path: str) -> None:
    """Merge into live state, read strata in order, commit atomically."""
    conn = sqlite3.connect(db_path)
    last = _cursor(conn)
    events = conn.execute(
        "SELECT id, role, content, kind, path, timestamp FROM events "
        "WHERE id > ? ORDER BY id", (last,)
    ).fetchall()
    if not events:
        return

    globs = _l3_target_globs(conn)
    D, commits_in = drift_ratio(db_path, globs)
    if D > 0.75 and commits_in == 0:
        # Flags are written outside the transaction on purpose: a rolled-back
        # compaction must not lose the evidence that triggered it.
        _raise_flag(conn, "DRIFT_HIGH", {"D7d": round(D, 3), "commits_in_target_48h": 0})

    # FIX 1 (v1.3): pass the LIVE L1 as prior state. Distilling only the new
    # events replaces the stratum, deleting every veto and blocker that has
    # a lifetime longer than one compaction run.
    l1 = _distil(events, prior=_read_stratum(conn, "L1"),
                 rule="tactical", cap=BUDGET["L1"])
    _assert_descriptive(l1)          # §3.6 trust boundary: no imperatives in L1

    with conn:                       # FIX 2 (v1.3): strata and cursor move together
        _write_stratum(conn, "L1", l1)

        # Read L1 back inside the transaction so L2 distils the NEW L1, never
        # the stale pre-write row (v1.2 distilled the old one).
        l2 = _distil(_read_stratum(conn, "L1"),
                     prior=_read_stratum(conn, "L2"),
                     rule="patterns", cap=BUDGET["L2"])
        _assert_descriptive(l2)

        # L2 may PROPOSE an L3 change; it may never apply one.
        proposal = _maybe_propose_l3_diff(l2, _read_stratum(conn, "L3"))
        if proposal:
            _store_proposal(conn, proposal)   # awaits `lateron approve-l3`

        _write_stratum(conn, "L2", l2)
        _advance_cursor(conn, events[-1][0])  # L0 rows are never deleted
```

Note what the skeleton does *not* do: it never deletes L0 rows, never lets `_distil` write L3, never replaces L1 or L2 without their prior state, and never asks a model whether drift occurred.

---

## 6. Evaluation Protocol

The architecture is falsifiable, and this section specifies how. Four arms at an **equal 400-token budget** — the comparison that matters, and the one that v1.1 omitted and v1.2 only half-made by testing against a naked summariser.

### 6.1 Arms

- **A — No Memory.** Context truncated to the working window; no consolidation.
- **B — Rolling-*N* Uniform Summary.** The production default (§2.1): single summariser, sliding window, 400-token budget.
- **B2 — Pinned Invariants + Rolling Summary.** **90 tokens of pinned L3 invariants + a 310-token rolling summary**, total 400. This is the strong baseline: it protects the mission exactly as well as Lateron's L3 does, at the same cost, with none of the strata.
- **C — Lateron.** Strata and caps per §2.2 and §2.4, anchor per §3.6 (`post-static-prefix` / `recency`), trust-boundary fences per §3.6.

**The paper's claim narrows to this:** C must beat **B2** — not merely B — and it must do so through the mechanisms B2 lacks: L1 veto suppression, L2 pattern extraction, and L3 pivot governance. If C only matches B2 on invariant retention, stratification has bought nothing that a pinned block did not already buy, and the honest conclusion is to ship B2.

### 6.2 Tasks and Horizon

Twenty-four tasks in three families, each run across **45 compressed cycles** (one cycle per simulated day), so that the 30-day L2 boundary fires, at least one `PROPOSED_L3_DIFF` is generated and adjudicated, and the day-40 pivot lands with room to observe its aftermath:

1. **Long-horizon feature work** — a multi-day implementation task with a stable objective and periodic off-target temptations.
2. **Mid-stream pivot** — the same task, with the objective changed by operator instruction at cycle 40 (tests §4.3 against L3 inertia, with five cycles of aftermath).
3. **Veto-heavy operation** — repeated proposals of a previously rejected action across the full horizon (tests the §3.5 TTL, expiry and re-trigger path, which a 12-day run could not exercise: the TTL is seven days and the interesting behaviour is what happens after it expires).

The 12-day horizon in v1.2 was too short to fire the mechanisms it was testing. A benchmark that never crosses its own boundaries measures nothing.

### 6.3 Metrics

- **Invariant retention** — blind rubric scoring of stated constraints against the day-0 invariant set, sampled at cycles 3, 10, 20, 30, 40 and 45. Expected to be near-parity between B2 and C; reported to confirm the baseline is strong, not to win the argument.
- **Veto suppression** — re-proposals of a vetoed action per session while the constraint is live, and again after TTL expiry. This is H1's discriminator, and B2 has no mechanism for it.
- **Pattern extraction** — recall and precision of L2 recurring-failure lines against a human-labelled list of the failure modes actually present across the 45 cycles.
- **Pivot governance** — stale-invariant violations and cycles-to-recovery after the day-40 pivot, plus the operator effort required (for B2: manual prompt editing; for C: one `commit-l3`).
- **Drift rate** — $D_{7d}$ per §3.4, validated against human labels.
- **Tokens per session** and **wall-clock latency**, including **prefix-cache hit rate** (§3.6) — the placement argument is a cost claim and should be measured as one.

### 6.4 Hypotheses and Falsification

- **H1 (primary):** C re-proposes vetoed actions less often than B2 while constraints are live, because B2 has no L1. A null result removes the strongest reason to build strata.
- **H2:** C's L2 lines recover more true recurring failure modes than B2's rolling summary at equal budget. A null result removes the second reason.
- **H3:** C handles the day-40 pivot with fewer stale-invariant violations and less operator effort than B2. A null result removes the third reason.
- **H4:** C and B2 are near-parity on invariant retention. Parity is the *expected* outcome and confirms B2 is a fair baseline; C beating B2 here would suggest the pinned block was mis-specified rather than that stratification is superior.
- **H5:** C's tokens-per-session and latency are indistinguishable from B2, and C's prefix-cache hit rate is unaffected relative to B2 under the placement rules of §3.6.

**Falsification condition.** If B2 ≥ C on H1, H2 *and* H3, the stratification does not justify itself and the correct architecture is a pinned block with a rolling summary. The paper states this in advance so that a null result cannot be reinterpreted as a partial success after the fact.

### 6.5 Reproduction

Harness, task definitions, prompts, seeds and per-run JSONL logs to be published alongside this document. Model versions pinned per run; all token counts via `o200k_base`; every run records its anchor (`post-static-prefix` or `recency`), its `D_7d` trace, and its prefix-cache hit rate. A run is reproducible if replaying L0 from cursor 0 with a pinned model yields the same stratum text.

---

## 7. Conclusion

As context windows scale, **context quality rather than raw context capacity sets the practical ceiling on agent reliability**. The production default — uniform rolling summarisation — preserves the ratio of signal to noise while destroying the absolute quantity of both, which is why long-horizon agents keep their tone and lose their mission. That diagnosis is necessary but not sufficient: a pinned objective block fixes the symptom cheaply, and any architecture arguing for strata must beat that design on something else.

Lateron's answer is that the strata do three things a pinned block cannot. They hold a human's veto long enough to stop the agent repeating a rejected action, and then let it expire. They compress thirty days of failure modes into recurring diagnosis rather than a list of incidents. And they govern the mission in both directions: the system may propose a change to its invariants but never make one, while the operator can pivot synchronously on demand. The block carrying that state is 400 tokens, injected where it does not invalidate a prefix cache, and fenced so that summarised repository content can never be mistaken for an order.

This is offered as a specification with a specified falsification test, not as a measured result. §6 exists so that the claim can be settled by experiment rather than by argument — and it is written so that the experiment can lose.

---

## References

1. Liu, N. F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., & Liang, P. (2023). *Lost in the Middle: How Language Models Use Long Contexts.* arXiv:2307.03172. <https://arxiv.org/abs/2307.03172>
2. Wei, T., Sachdeva, N., Coleman, B., et al. (2025). *Evo-Memory: Benchmarking LLM Agent Test-time Learning with Self-Evolving Memory.* UIUC & Google DeepMind. arXiv:2511.20857 (v2, May 2026). <https://arxiv.org/abs/2511.20857>
3. Li, K., Yu, X., Ni, Z., et al. (2026). *TiMem: Temporal-Hierarchical Memory Consolidation for Long-Horizon Conversational Agents.* Findings of ACL 2026, pp. 21700–21720. <https://aclanthology.org/2026.findings-acl.1091/>
4. Zhang, N., Yang, X., Tan, Z., Deng, W., & Wang, W. (2026). *HiMem: Hierarchical Long-Term Memory for LLM Long-Horizon Agents.* arXiv:2601.06377. <https://arxiv.org/abs/2601.06377>
5. Compresr Research (2026). *Context Compression Architecture: 2026 Complete LLM Guide (Hard-Token, Soft-Token, and KV Cache Designs).* <https://compresr.ai/blog/context-compression-architecture-complete-guide>
6. Ahmed, M. S. (2026). *LLM Architecture 2026: Primacy, Middle, and Recency Context Attention Zones in Production Agents.* RankSquire Engineering. <https://ranksquire.com/2026/04/13/llm-architecture-2026/>
7. Trope, Y., & Liberman, N. (2010). *Construal-level theory of psychological distance.* Psychological Review, 117(2), 440–463. <https://doi.org/10.1037/a0018963>

---

## Citation

```bibtex
@techreport{dryland2026lateron,
  title  = {Lateron: Multi-Horizon Hard-Token Context Compression and
            Primacy-Zone State Steering for Long-Horizon LLM Agents},
  author = {Dryland, Christopher},
  year   = {2026},
  month  = {October},
  note   = {Systems Architecture Specification, v1.3},
  url    = {https://github.com/superhanzzz/lateron}
}
```

## Changelog

**v1.3** — Corrected three defects in the §5 skeleton: `_distil` now receives live prior state so L1 vetoes and blockers survive a pass; L2 distils the freshly written L1 inside a single atomic transaction with the cursor advance; and `drift_ratio()` filters both sides through `target_paths` and keeps the $D_{7d}$ denominator to touches only. Reframed §2.3 to scope the $\alpha^k$ model to unstructured summarisation and to withdraw the straw-man claim, narrowing the thesis to veto suppression, pattern extraction and pivot governance. Added **Arm B2** (pinned 90-token L3 + 310-token rolling summary) to §6, with the explicit concession that C must beat B2 or the strata do not justify themselves, plus a stated falsification condition. Extended the evaluation horizon from 12 days to 45 cycles so the 30-day `PROPOSED_L3_DIFF` and the day-40 pivot actually fire. Replaced Slot 0 injection with the **Post-Static-Prefix Breakpoint** to preserve KV prefix-cache hit rates, demoted Slot 0 to a rejected option, and added the **trust boundary** fencing L3 as operator-authorised and L1/L2 as non-authoritative telemetry with a lint enforcing it. Reserved 30 tokens of header, flag and TTL overhead in the §2.4 budget (130/110/90/40/30 = 400).

**v1.2** — Staged Rolling-*N* Uniform Summarisation as the explicit baseline (§1.4, §2.1–2.3), with a formal equal-rate decay argument and a falsifiable prediction. Promoted rejection telemetry to §3.5 with a seven-day TTL, local suppression, and re-trigger semantics. Replaced Primacy-Zone residency with a placement rule (§3.6), conceding host-prompt-length dependency. Replaced LLM-judged drift with a deterministic preprocessor metric and threshold (§3.4). Made L3 normative and two-way governed: anti-drift `PROPOSED_L3_DIFF` and synchronous `lateron commit-l3` pivot override (§4). Made L0 an immutable event-sourced log with `last_compacted_id` replay (§3.2). Pinned token counts to `o200k_base`, rebalanced the budget (§2.4), added the compound compaction trigger (§3.3), and specified the evaluation protocol (§6). British English throughout.

**v1.1** — Initial systems architecture specification.

## License

This work is licensed under a [Creative Commons Attribution 4.0 International License](LICENSE).
