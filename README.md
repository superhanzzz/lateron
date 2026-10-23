# Lateron: Multi-Horizon Hard-Token Context Compression and Primacy-Zone State Steering for Long-Horizon LLM Agents

**Author:** Christopher Dryland (Independent Researcher, Canberra, Australia)
**Date:** October 23, 2026
**Version:** 1.2 (Systems Architecture Specification)
**License:** [Creative Commons Attribution 4.0 International (CC BY 4.0)](LICENSE)

> **Changes in v1.2.** Adds a staged competitor baseline against Rolling-*N* Uniform Summarisation (§1.4, §2.1–2.3); promotes rejection telemetry to its own subsection with a seven-day TTL (§3.5); replaces Primacy-Zone residency with a dual-anchor placement rule that concedes host-prompt-length dependency (§3.6); replaces LLM-judged drift detection with a deterministic L0 preprocessor metric (§3.4); makes L3 normative and two-way governed, with an anti-drift rule and a synchronous operator override (§4); makes L0 an immutable event-sourced log with a replay cursor (§3.2); pins tokenisation to o200k_base, rebalances the 400-token budget, sets a compound compaction trigger, and adds a reproducible three-arm evaluation protocol (§2.4, §3.3, §6). Spelling standardised to British English.

---

## Abstract

Long-horizon Large Language Model (LLM) agents degrade in reliability as interaction history grows. While recent models support million-token context windows, production engineering reports indicate that transformer instruction fidelity begins to decay once context fill passes roughly 60–70%, with retrieved memories injected into the middle context zone (15–85%) suffering from pronounced attention dilution. Furthermore, standard Retrieval-Augmented Generation (RAG) and flat episodic memory stores treat stale implementation noise and high-level system invariants with equal retrieval weight.

This paper presents **Lateron**, an out-of-band, multi-horizon hard-token compression architecture for stateful LLM agents. Its design target is the dominant production default — Rolling-*N* Uniform Summarisation, in which a single summarisation operator is applied to a sliding window of history. Uniform compression reduces transient implementation noise and durable objective invariants at the same rate, so the ratio between them is preserved while the absolute quantity of both decays.

Lateron instead applies **Horizon-Based Resolution Scaling** (adapted from Construal Level Theory) across discrete temporal strata (24-Hour, 7-Day, 30-Day, and 90-Day horizons). As state logs age, low-level execution mechanics (*how* tasks were executed) are systematically pruned, while high-level trajectory invariants (*why* the system is operating and its governing constraints) are distilled upward. The resulting state is reconciled nightly into a compact 400-token control block, placed by a dual-anchor rule — at the front of the context where the host architecture permits, or immediately before the active turn otherwise — so that it rides an attention peak rather than sitting stranded behind a long system prompt. Consolidation is driven by a deterministic drift metric over an immutable event-sourced log, and the invariant stratum is governed top-down: it can be proposed for change by the system but only committed by a human operator.

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

If Horizon-Based Resolution Scaling is worth its complexity, it must beat that baseline at an equal token budget. This paper states the comparison explicitly (§2.3) and specifies the experiment that would falsify it (§6).

---

## 2. Horizon-Based Resolution Scaling

### 2.1 The Production Default: Rolling-*N* Uniform Summarisation

Define a history $H$ of token length $|H| > N$. Rolling-*N* applies a compression operator $U$ such that $U(H) \rightarrow \hat{H}$ with $|\hat{H}| \le N$, typically by an instruction of the form *"summarise the conversation so far, keeping what matters"*, applied cumulatively as the window advances.

Three properties follow, and all three are load-bearing:

1. **One operator, no strata.** $U$ cannot treat a resolved stack trace and a non-negotiable deployment constraint as different kinds of object, because nothing in the pipeline tells it that they are.
2. **Compounding decay.** Applied $k$ times over a long horizon, every surviving item is the result of $k$ successive re-summarisations. Salience that is not reinforced by recency is progressively eroded.
3. **Recency-symmetric attention.** Because the operator reads a sliding window, material that has aged out of the window is either summarised or gone; there is no stratum in which an old invariant has a *protected* representation.

Rolling-*N* is genuinely good at what it was designed for: bounding context and preserving recent working state. Its weakness is specific — it protects the wrong thing at long horizons.

### 2.2 Horizon-Based Resolution Scaling

Lateron replaces the single operator with a set of stratum-specific resolution rules indexed to age, derived from **Construal Level Theory** (Trope & Liberman, 2010): *temporal distance dictates abstraction resolution*. The mapping is borrowed as a design heuristic, not asserted as a cognitive mechanism. The system does not claim that agents "think like people"; it claims that the *category* of information worth retaining changes with age, and that a different operator per age band encodes that.

Instead of compressing all information equally, each stratum enforces a filter on what survives compaction:

| Temporal Stratum | Resolution Rule | Retained State | Pruned State | Token Cap |
| --- | --- | --- | --- | --- |
| **L0: Event Log** | Immutable, append-only | All telemetry — tool calls, git diffs, session logs, errors, operator overrides | Nothing (never pruned on disk) | Unbounded (disk) |
| **L1: 7-Day Stratum** | Tactical state (the *how*) | Active task velocity, open blockers, uncommitted branches, operator vetoes | Resolved errors, raw conversational turns, intermediate tool outputs | ≤ 140 tokens |
| **L2: 30-Day Stratum** | Operational patterns (system dynamics) | Throughput bottlenecks, recurring failure modes, architectural drift, workflow dependencies | Completed task checklists, isolated daily variance, deprecated experiments | ≤ 120 tokens |
| **L3: 90-Day+ Stratum** | Invariant objectives (the *why*) | Core project specification, non-negotiable constraints, primary objective function | All low-level implementation mechanics and historical tactical state | ≤ 90 tokens |
| **Runtime Triage Rule** | Compiled directive | The single priority instruction for the active session | — | ≤ 50 tokens |

### 2.3 Why the Axes Differ: Equal-Rate Decay Versus Protected Invariants

Let $H = H_{inv} \cup H_{noise}$, where $H_{inv}$ is the token mass carrying objective invariants and $H_{noise}$ the token mass carrying transient implementation detail. Rolling-*N* applies one operator $U$ to the union, so to first order each component is reduced by the same factor:

$$|U(H)| \approx \alpha \cdot |H|, \qquad |U(H_{inv})| \approx \alpha \cdot |H_{inv}|, \qquad |U(H_{noise})| \approx \alpha \cdot |H_{noise}|$$

The ratio $|U(H_{inv})| / |U(H_{noise})|$ is thus preserved — but the *absolute* invariant mass shrinks by $\alpha$ on every pass, and $\alpha$ is set by the budget, not by importance. Over $k$ nightly passes the invariant fraction decays as $\alpha^k$, which is precisely the observed failure mode: at 90 days the agent still says something about the project, but the sentence has been re-summarised eleven times and now carries a generic claim that no longer constrains anything.

Rolling-*N* preserves the *proportion* of signal to noise while destroying the *quantity* of signal. That is the failure, stated precisely, and it is why the fix has to be a different operator per category rather than a better single operator.

Lateron applies a distinct operator per stratum: $f_{L1}$ prunes aggressively (§2.2), $f_{L3}$ is conservative and identity-preserving on the invariant set. Invariant mass is bounded from below by the L3 cap rather than multiplied by $\alpha$ each night:

$$|L_3|_{t+1} \ge \min(\beta_{L3},\, |L_3|_t), \qquad \beta_{L3} = 90 \text{ tokens}$$

The invariant floor is a *budget guarantee*, not a compression outcome. This is the entire architectural difference, and it is testable: at matched token budget, Lateron should retain more L3-checkable constraints than Rolling-*N* after the same number of passes. §6 specifies how to measure that.

### 2.4 Token Budget and Tokenisation

All caps are counted with `tiktoken`'s `o200k_base` encoding, so every number in this document is reproducible. The budget totals exactly 400 tokens:

- **L1 (7-day tactical):** 140 tokens
- **L2 (30-day patterns):** 120 tokens
- **L3 (90-day invariants):** 90 tokens
- **Runtime Triage Rule:** 50 tokens

The split is a working default chosen so that the invariant stratum survives all compaction, and is expected to be tuned per deployment; it is not derived from first principles. Lateron does not control the host system prompt and makes no claim about total context length — only about the size of the block it contributes.

---

## 3. System Architecture & Pipeline Execution

Lateron decouples memory consolidation (asynchronous, out-of-band) from live inference (synchronous, zero-latency).

### 3.1 Passive Telemetry Ingestion

Rather than requiring explicit "save-to-memory" tool calls during live inference — which consume tool-calling turns and increase latency — Lateron ingests passive operational exhaust:

- **Session state logs:** read-only queries against the agent's local SQLite state store (message history, tool execution status, timestamps).
- **Repository and filesystem state:** per-session `git diff --stat` and `--name-only`, commit metadata, and file modification timestamps across active workspaces.

### 3.2 Event-Sourced L0

L0 is an **immutable, append-only event log** in SQLite. Compaction never deletes rows. A `last_compacted_id` cursor marks the high-water mark of processed events; the compaction worker reads forward from the cursor and advances it only on a successful write of the new stratum state.

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
2. Over the trailing seven days, count tool touches to files outside `target_paths` (`touches_out`) against all touches (`touches_total`), and count commits whose changed paths fall entirely outside `target_paths` (`commits_out`).
3. Compute the **Drift Ratio**:

$$D_{7d} = \frac{\text{touches}_{out}}{\max(1,\ \text{touches}_{total})}$$

4. Raise `DRIFT_HIGH` deterministically when both conditions hold:

$$D_{7d} > 0.75 \quad \text{and} \quad \text{commits}_{in\_target} = 0 \text{ in the last 48h}$$

The flag is arithmetic, auditable, and reproducible. The model's only role is to summarise the off-target activity into one L1 line and attach it to the flag for the operator — it can neither raise nor clear the flag. A false positive is therefore a threshold-tuning problem, not a reasoning failure, and the threshold is a single configurable constant.

### 3.5 Rejection Telemetry: Stratum-Resident Negative Constraints

When a human operator vetoes or aborts an agent-proposed action during a live run, the event is recorded as a **negative constraint in L1** with a **seven-day TTL**.

This is the mechanism that distinguishes Lateron from summarisation-based memory, and it is deliberately *local*. A veto is evidence about the tactical present — the operator did not want this action, in this context, this week — not a permanent statement about the mission. Placing it in L3 would make one bad afternoon's judgement a constitutional constraint on the project. Placing it in L1 makes it a bounded, decaying inhibition:

- **Suppresses action-looping locally.** The agent reads the constraint in its L1 block and does not re-propose the rejected action in subsequent sessions within the window.
- **Does not mutate L3.** The invariant stratum is untouched by operational friction; the mission statement cannot be edited by an annoyed operator at 11pm.
- **Expires cleanly.** At TTL, the constraint drops out unless re-triggered by a fresh veto. If the operator rejects the same action again, the TTL resets — repetition is evidence the constraint is real, and it earns its continued place without any promotion to L3.

Rejected-action loops are among the most expensive failure modes in long-horizon agents, because each repetition burns a full tool-call round trip to re-derive a decision a human already made. Lateron treats the human's veto as the cheapest available gradient signal and stores it at the shortest stratum that can act on it.

### 3.6 Dual-Anchor Injection

v1.1 claimed injection "into the Primacy Zone". That claim is only true if the host system prompt is short, and Lateron does not control host prompt length. A 400-token block appended after a 3,000-token system prompt sits at position 3,000 — in the middle zone, in the trough of the U-curve, which is exactly where this architecture says not to put it.

v1.2 therefore specifies **dual-anchor placement**:

- **Slot 0 (preferred):** prepend the block ahead of the static system prompt and tool schemas, so it occupies the first ~400 tokens of the context and genuinely sits in the primacy zone. This is available only where the host architecture permits it — that is, where prompt assembly is under the integrating system's control and tool schemas tolerate a preceding block.
- **Recency-Boundary Injection (fallback):** write the block immediately before the active turn — after all history, immediately preceding the current user/tool message. This rides the right-hand peak of the U-curve rather than the left. It is a weaker anchor than Slot 0 for instruction persistence across a long generation, but it is categorically better than the trough.

The block is byte-identical in both placements; only position changes. The integrating system selects the anchor at assembly time and records which one was used, so that evaluation (§6) can attribute any difference in adherence to placement rather than to content.

### 3.7 The Whisper Block

The output of nightly compaction is a static markdown block injected at one of the two anchors above. At runtime, zero vector-search latency is incurred:

```text
[SYSTEM STATE: MULTI-HORIZON TELEMETRY]        # 400 tokens, o200k_base
- L1 (7d Tactical, ≤140): Pipeline modules A–B deployed. Zero commits to core
  repo in 36h; 4h on unscheduled exploratory scripts. Operator vetoed automated
  schema migration (2026-10-20, TTL 7d — hold until manual review).
- L2 (30d Pattern, ≤120): Scope drift occurs during data-cleaning phases.
  Pre-staging input files raises session completion rate.
- L3 (90d Invariant, ≤90): Primary objective is production stability and
  throughput of the core publishing pipeline. Secondary prototypes are
  non-blocking. Target paths: /pipeline/**, /editions/**.
- Triage (≤50): Prioritise staged core pipeline tasks. Off-target concepts go
  to /backlog.md; do not divert active session compute.
- DRIFT_HIGH: D7d=0.81, 0 target commits in 48h. Off-target summary: HTML
  prototype rework in /experiments/ (Thu–Sat).
```

---

## 4. Two-Way L3 Governance

### 4.1 Normative Versus Descriptive Strata

L3 is **normative**: it states what the system is *for*, and changes top-down, on human authority only. L1 and L2 are **descriptive**: they record what has actually happened, and are written bottom-up by the compaction process. Conflating these two directions is how memory systems acquire the property of slowly rewriting their own mission.

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

```python
# Minimal Lateron compaction worker (conceptual skeleton)
import sqlite3, subprocess, tiktoken

ENC = tiktoken.get_encoding("o200k_base")
BUDGET = {"L1": 140, "L2": 120, "L3": 90, "TRIAGE": 50}

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
    """Deterministic pre-compaction metric. No model involved."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    rows = conn.execute(
        "SELECT path, kind FROM events "
        "WHERE timestamp > datetime('now', ?) AND kind IN ('touch','commit')",
        (f"-{days} days",),
    ).fetchall()
    total = len(rows)
    out = sum(1 for p, _ in rows if not _matches(p, target_globs))
    in_target_48h = conn.execute(
        "SELECT COUNT(*) FROM events WHERE kind='commit' "
        "AND timestamp > datetime('now','-48 hours') AND path IS NOT NULL"
    ).fetchone()[0]
    return (out / max(1, total)), in_target_48h

def compact(db_path: str) -> None:
    """Reads forward from the cursor; advances it only after a successful write."""
    conn = sqlite3.connect(db_path)
    last = _cursor(conn)
    events = conn.execute(
        "SELECT id, role, content, kind, path, timestamp FROM events "
        "WHERE id > ? ORDER BY id", (last,)
    ).fetchall()
    if not events:
        return

    D, commits_in = drift_ratio(db_path, _l3_target_globs(conn))
    if D > 0.75 and commits_in == 0:
        _raise_flag(conn, "DRIFT_HIGH", {"D7d": round(D, 3), "commits_in_target_48h": 0})

    # One model call per due stratum, each with its own resolution rule and cap.
    l1 = _distil(events, rule="tactical",        cap=BUDGET["L1"])
    l2 = _distil(_read_stratum(conn, "L1"), rule="patterns", cap=BUDGET["L2"])

    # L2 may PROPOSE an L3 change; it may never apply one.
    proposal = _maybe_propose_l3_diff(l2, _read_stratum(conn, "L3"))
    if proposal:
        _store_proposal(conn, proposal)      # awaits `lateron approve-l3`

    _write_stratum(conn, "L1", l1)
    _write_stratum(conn, "L2", l2)
    _advance_cursor(conn, events[-1][0])     # L0 rows are never deleted
```

Note what the skeleton does *not* do: it never deletes L0 rows, never lets `_distil` write L3, and never asks a model whether drift occurred.

---

## 6. Evaluation Protocol

The architecture is falsifiable, and this section specifies how. The benchmark is three arms at an **equal token budget**, which is the comparison that matters and the one v1.1 omitted.

### 6.1 Arms

- **A — No Memory.** Context truncated to the working window; no consolidation.
- **B — Rolling-*N* Uniform Summary.** The production default (§2.1): single summariser, sliding window, **400-token budget matched to Lateron**. Matching the budget is essential; comparing a 400-token Lateron block against an unconstrained summary measures nothing.
- **C — Lateron.** Strata as specified in §2.2, caps per §2.4, anchor per §3.6, with the anchor recorded (`slot0` / `recency`).

### 6.2 Tasks

Twenty-four tasks in three families, each run across **twelve simulated days** with multiple sessions per day:

1. **Long-horizon feature work** — a multi-day implementation task with an explicit, stable objective and periodic off-target temptations.
2. **Mid-stream pivot** — the same task, with the objective changed at day 5 by operator instruction (tests §4.3 against L3 inertia).
3. **Veto-heavy operation** — repeated proposals of a previously rejected action (tests the §3.5 TTL and re-trigger path).

### 6.3 Metrics

- **Invariant retention** — blind rubric scoring of the agent's stated constraints against the day-0 invariant set, at days 3, 6, 9 and 12.
- **Drift rate** — $D_{7d}$ from §3.4, plus human-labelled drift for the tasks where the metric is being validated.
- **Stale-constraint violations** — actions contradicting a constraint the operator retired.
- **Rejected-action re-proposals** — count of re-proposals of a vetoed action per session.
- **Tokens per session** and **wall-clock latency** (verifying the out-of-band claim costs nothing at inference).

### 6.4 Hypotheses and Falsification

- **H1:** at matched budget, C retains more invariants than B at day 12. If B ≥ C, stratification buys nothing and the paper's core claim fails.
- **H2:** the B−C gap widens with horizon length (the $\alpha^k$ prediction of §2.3). A flat gap falsifies the equal-rate decay argument.
- **H3:** C re-proposes vetoed actions less often than A and B. A null result limits the contribution of §3.5.
- **H4:** C's tokens-per-session and latency are statistically indistinguishable from B, since compaction is out-of-band.

### 6.5 Reproduction

Harness, task definitions, prompts, seeds and per-run JSONL logs to be published alongside this document. Model versions pinned per run; all token counts via `o200k_base`; every run records its anchor (Slot 0 or Recency-Boundary) and its Drift Ratio trace. A run is reproducible if replaying L0 from cursor 0 with a pinned model yields the same stratum text.

---

## 7. Conclusion

As context windows scale, **context quality rather than raw context capacity sets the practical ceiling on agent reliability**. The production default — uniform rolling summarisation — preserves the ratio of signal to noise while destroying the absolute quantity of both, which is why long-horizon agents keep their tone and lose their mission.

Lateron replaces that single operator with stratum-specific resolution rules, driven by a deterministic drift metric over an immutable event log, injected at a position chosen from the attention curve rather than assumed into it, and governed so that the system can propose a change to its own objectives but never make one. The result is a 400-token control block with a guaranteed floor of invariant content, no inference-time retrieval cost, and an auditable history underneath it.

This is offered as a specification with a specified falsification test, not as a measured result. §6 exists so that the claim can be settled by experiment rather than by argument.

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
  note   = {Systems Architecture Specification, v1.2},
  url    = {https://github.com/superhanzzz/lateron}
}
```

## Changelog

**v1.2** — Staged Rolling-*N* Uniform Summarisation as the explicit baseline (§1.4, §2.1–2.3), with a formal equal-rate decay argument and a falsifiable prediction. Promoted rejection telemetry to §3.5 with a seven-day TTL, local suppression, and re-trigger semantics. Replaced Primacy-Zone residency with dual-anchor placement (§3.6), conceding host-prompt-length dependency and adding Recency-Boundary Injection as the fallback. Replaced LLM-judged drift with a deterministic preprocessor metric and threshold (§3.4). Made L3 normative and two-way governed: anti-drift `PROPOSED_L3_DIFF` and synchronous `lateron commit-l3` pivot override (§4). Made L0 an immutable event-sourced log with `last_compacted_id` replay (§3.2). Pinned token counts to `o200k_base`, rebalanced the budget to 140/120/90/50 (§2.4), added the compound compaction trigger (§3.3), and specified the three-arm evaluation protocol (§6). British English throughout.

**v1.1** — Initial systems architecture specification.

## License

This work is licensed under a [Creative Commons Attribution 4.0 International License](LICENSE).
