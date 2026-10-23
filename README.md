# Lateron: Multi-Horizon Hard-Token Context Compression and Primacy-Zone State Steering for Long-Horizon LLM Agents

**Author:** Christopher Dryland (Independent Researcher, Canberra, Australia)
**Date:** October 23, 2026
**Version:** 1.1 (Systems Architecture Specification)
**License:** [Creative Commons Attribution 4.0 International (CC BY 4.0)](LICENSE)

---

## Abstract

Long-horizon Large Language Model (LLM) agents degrade in reliability as interaction history grows. While recent models support million-token context windows, production engineering reports indicate that transformer instruction fidelity begins to decay once context fill passes roughly 60–70%, with retrieved memories injected into the middle context zone (15–85%) suffering from pronounced attention dilution. Furthermore, standard Retrieval-Augmented Generation (RAG) and flat episodic memory stores treat stale implementation noise and high-level system invariants with equal retrieval weight.

This paper presents **Lateron**, an out-of-band, multi-horizon hard-token compression architecture for stateful LLM agents. Rather than summarizing history uniformly by recency or entity extraction, Lateron applies **Horizon-Based Resolution Scaling** (adapted from Construal Level Theory) across discrete temporal strata (24-Hour, 7-Day, 30-Day, and 90-Day horizons). As state logs age, low-level execution mechanics (*how* tasks were executed) are systematically pruned, while high-level trajectory invariants (*why* the system is operating and its governing constraints) are distilled upward. The resulting multi-horizon state is reconciled nightly into a compact (~400 token) control block intended for injection into the transformer's high-attention **Primacy Zone**, with the aim of bounding context growth and reducing long-horizon objective drift.

---

## 1. Introduction: The Production Bottleneck in Long-Horizon Agents

Autonomous and semi-autonomous LLM agents operating over weeks or months face three structural failure modes in production:

1. **Middle-Zone Attention Degradation.** Transformer attention distribution is heavily U-shaped. Performance on tasks requiring use of information in the middle of long contexts degrades as context length grows, even for models explicitly built for long context (Liu et al., 2023). Production engineering reports place the **Primacy Zone** at roughly the first 0–15% of tokens and the **Recency Zone** at the final 5–10%, with the **Middle Zone** (15–85%) receiving degraded attention (RankSquire, 2026). Standard memory pipelines inject retrieved vector chunks into this middle zone, increasing token cost while yielding poor instruction adherence.

2. **Temporal Contradiction and Retrieval Pollution.** Append-only vector databases retrieve past logs based on semantic similarity rather than temporal validity. As shown in the Evo-Memory benchmark (Wei et al., 2025) and HiMem (Zhang et al., 2026), failing to prune superseded operational states causes agents to hallucinate obsolete constraints or repeat failed tool paths.

3. **Uniform Summarization Loss.** Existing recursive summarization pipelines compress text uniformly — shrinking both transient syntax errors and core project objectives at the same rate until long-term system goals vanish into generic prose.

---

## 2. Horizon-Based Resolution Scaling

While hierarchical structures like TiMem (Li et al., 2026) organize dialogue into temporal trees, Lateron introduces a strict semantic pruning rule derived from **Construal Level Theory** (Trope & Liberman, 2010): *temporal distance dictates abstraction resolution*. The mapping is borrowed here as a design heuristic rather than asserted as a cognitive mechanism.

Instead of compressing all information equally, each stratum enforces a specific filter on what category of information survives compaction:

| Temporal Stratum | Resolution Rule | Retained State | Pruned State | Token Cap |
| --- | --- | --- | --- | --- |
| **L0: 24-Hour Buffer** | Raw Execution (Full Telemetry) | Tool calls, git diffs, session logs, active errors, operator overrides | None (cleared after nightly compaction) | Ephemeral |
| **L1: 7-Day Stratum** | Tactical State (the *how*) | Active task velocity, open blockers, uncommitted branches, recent operator rejections | Resolved errors, raw conversational turns, intermediate tool outputs | ≤ 150 tokens |
| **L2: 30-Day Stratum** | Operational Patterns (system dynamics) | Throughput bottlenecks, recurring failure modes, architectural drift, workflow dependencies | Completed task checklists, isolated daily variance, deprecated experiments | ≤ 150 tokens |
| **L3: 90-Day+ Stratum** | Invariant Objectives (the *why*) | Core project specification, non-negotiable system constraints, primary objective function | All low-level implementation mechanics and historical tactical state | ≤ 100 tokens |

The token caps above are initial working defaults, not derived constants: they were chosen as round numbers and are expected to be tuned per deployment. By enforcing strict caps and orthogonal resolution rules at each layer, the entire multi-month memory footprint is bounded at roughly **400 tokens** — small enough, where the hosting system prompt is itself short, to sit near the front of the model's context alongside it.

---

## 3. System Architecture & Pipeline Execution

Lateron decouples memory consolidation (asynchronous, out-of-band) from live inference (synchronous, zero-latency).

### 3.1 Passive Telemetry Ingestion

Rather than requiring explicit "save-to-memory" tool calls during live inference — which consume tool-calling turns and increase latency — Lateron ingests passive operational exhaust:

- **Session State Logs:** Read-only queries against the agent's local SQLite state store (e.g. message history, tool execution status, timestamps).
- **Repository & Filesystem State:** Daily `git diff --stat` and file modification timestamps across active workspaces.

### 3.2 Out-of-Band Nightly Compaction

Once per 24-hour cycle (or upon session termination), an out-of-band compaction worker runs a three-step reconciliation pass:

1. **Upward Distillation.** Raw L0 telemetry is distilled into L1 (7-Day). On weekly boundaries, L1 is distilled into L2 (30-Day); on monthly boundaries, L2 updates L3 (90-Day).
2. **Drift Detection (Cross-Stratum Diff).** The compaction model compares L1 tactical activity against L3 invariant objectives. If 7-day execution has drifted into unscheduled scope creep while the primary L3 objective remains stalled, a high-priority drift flag is raised for operator review.
3. **Rejection Telemetry Logging (Anti-Looping).** When a human operator vetoes or aborts an agent-proposed action during live runs, the event is logged in L1 as a temporary **Negative Constraint**. This prevents the agent from repeatedly re-proposing rejected actions in subsequent sessions without requiring permanent changes to L3 goals.

### 3.3 Primacy-Zone Directive Injection ("Whisper" Block)

The output of the nightly compaction is a static markdown block written directly to the agent's primary system prompt header. At runtime, zero vector-search latency is incurred:

```text
[SYSTEM STATE: MULTI-HORIZON TELEMETRY]
- L1 (7d Tactical): Pipeline modules A-B deployed. Zero commits to core repo in 36h;
  4h spent on unscheduled exploratory scripts. Operator rejected automated schema
  migration on Thu (hold until manual review).
- L2 (30d Pattern): Scope drift frequently occurs during data-cleaning phases.
  Pre-staging input files increases session completion rate.
- L3 (90d Invariant): Primary objective is production stability and throughput of the
  core publishing pipeline. Secondary prototypes are non-blocking.
- Runtime Triage Rule: Prioritize staged core pipeline tasks. Log any new exploratory
  concepts to /backlog.md without diverting active session compute.
```

---

## 4. Reference Implementation Specification

The compaction loop requires no external vector database and can be deployed via a single cron script pointing at an existing agent SQLite store:

```python
# Minimal Lateron Nightly Compaction Worker (Conceptual Skeleton)
import sqlite3, subprocess, pathlib

def collect_l0_telemetry(db_path: str, repo_path: str) -> str:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    logs = conn.execute(
        "SELECT role, content, timestamp FROM messages "
        "WHERE timestamp > datetime('now', '-1 day')"
    ).fetchall()
    git_stat = subprocess.getoutput(
        f'git -C {repo_path} log --since="24 hours ago" --oneline --stat'
    )
    return f"SESSION LOGS:\n{logs}\n\nGIT DIFFS:\n{git_stat}"

# Pass collect_l0_telemetry() + existing lateron_state.md to a local/API LLM
# using the L1/L2/L3 Horizon Resolution rules, then write the ~400-token output
# directly into the agent's system prompt / MEMORY.md header.
```

---

## 5. Conclusion

As LLM context windows scale, **context quality rather than raw context capacity sets the practical ceiling on agent reliability**. By replacing append-only vector retrieval with out-of-band Horizon-Based Resolution Scaling, Lateron compresses months of operational telemetry into a ~400 token steering block — reducing inference token overhead by design while aiming to keep autonomous agents anchored to long-horizon objectives. The architecture is offered as a specification to be evaluated, not a measured result.

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
  note   = {Systems Architecture Specification, v1.1},
  url    = {https://github.com/superhanzzz/lateron}
}
```

## License

This work is licensed under a [Creative Commons Attribution 4.0 International License](LICENSE).
