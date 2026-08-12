"""Schema-masked markdown renderer for Phase 3 judging.

The rendered output MUST NOT reveal the mapping schema name to the Phase 3 summarizer
or judge. This is guaranteed by construction (see the note below), not by filtering the
output for schema-name keywords.

All three schemas share the same 5-section structure. Orphan decisions (those with no
goal connection) are rendered in a dedicated "Unattached / Miscellaneous Decisions"
subsection that is structurally identical across all three schemas, so the judge cannot
infer schema identity from orphan handling.

Volume normalization is achieved DOWNSTREAM via the research summary obfuscation layer
(src/evaluation/research_summary.py), which compresses each artifact to a fixed-length
prose summary (~450–600 words) before judging. The raw artifacts vary in size (16k–30k
words) by design — the summarizer normalizes this.

Schema masking is guaranteed BY CONSTRUCTION, not by keyword filtering: every section
header is neutral, GM relation types come from a closed vocabulary that contains no
schema acronyms, and the schema identity ("dm"/"dsm"/"gm") lives only in output
filenames/cache keys — never in the rendered content. No blacklist of acronyms is
applied, so coincidental tokens like "GM" (General Manager) or "DM" in real org data
pass through untouched.
"""
from collections import defaultdict
from pathlib import Path

from ..models import CanonicalGoal, Decision, ORPHAN_GOAL_ID, make_orphan_goal
from .schemas import DMMapping, DSMMapping, GMMapping


def _word_count(text: str) -> int:
    return len(text.split())


def _confidence_histogram(confidences: list[str]) -> str:
    high = confidences.count("high")
    medium = confidences.count("medium")
    low = confidences.count("low")
    return (
        "| Confidence | Count |\n"
        "|------------|-------|\n"
        f"| high       | {high:5} |\n"
        f"| medium     | {medium:5} |\n"
        f"| low        | {low:5} |"
    )


def _render_orphan_subsection(orphan_ids: list[str], decisions_by_id: dict[str, Decision]) -> list[str]:
    """Render an identical-shape orphan block (used by all three schemas)."""
    orphan_goal = make_orphan_goal()
    lines = [
        "",
        f"### {orphan_goal.title}",
        "",
        orphan_goal.description,
        "",
    ]
    for d_id in orphan_ids:
        d = decisions_by_id.get(d_id)
        d_title = d.title if d else d_id
        lines.append(f"- **{d_title}** → *(no goal connection identified)*")
    return lines


# ── DM renderer ───────────────────────────────────────────────────────────────

def _render_dm(
    mapping: DMMapping,
    decisions_by_id: dict[str, Decision],
    goals_by_id: dict[str, CanonicalGoal],
) -> str:
    linked = [e for e in mapping.entries if e.goal_id is not None]
    unlinked_entries = [e for e in mapping.entries if e.goal_id is None]

    # Orphans = decisions with null goal + any decisions absent from entries entirely
    entries_decision_ids = {e.decision_id for e in mapping.entries}
    orphan_ids = (
        [e.decision_id for e in unlinked_entries]
        + [d_id for d_id in decisions_by_id if d_id not in entries_decision_ids]
    )

    sections = [
        f"# Candidate Mapping — condition: {mapping.condition_name}",
        "",
        "## 1. Summary",
        f"Total entries: {len(mapping.entries)} decisions assessed.",
        f"- Linked to a goal: {len(linked)}",
        f"- Unattached decisions: {len(orphan_ids)}",
        "",
        "Each decision is associated with at most one goal.",
        "",
        "## 2. Per-Decision Relationships",
        "",
    ]

    for entry in mapping.entries:
        if entry.goal_id is None:
            continue  # rendered in orphan subsection below
        decision = decisions_by_id.get(entry.decision_id)
        d_title = decision.title if decision else entry.decision_id
        goal = goals_by_id.get(entry.goal_id)
        g_title = goal.title if goal else entry.goal_id
        sections.append(
            f"- **{d_title}** → *{g_title}*  "
            f"[{entry.confidence}] {entry.reasoning}"
        )

    if orphan_ids:
        sections += _render_orphan_subsection(orphan_ids, decisions_by_id)

    sections += [
        "",
        "## 3. Cross-References",
        "",
        "(no cross-references by design — each entry is a single decision-to-goal association)",
        "",
        "## 4. Confidence Distribution",
        "",
        _confidence_histogram([e.confidence for e in mapping.entries]),
        "",
    ]
    return "\n".join(sections)


# ── DSM renderer ──────────────────────────────────────────────────────────────

def _render_dsm(
    mapping: DSMMapping,
    decisions_by_id: dict[str, Decision],
    goals_by_id: dict[str, CanonicalGoal],
) -> str:
    total_scores = sum(len(e.scored_goals) for e in mapping.entries)

    entries_decision_ids = {e.decision_id for e in mapping.entries}
    orphan_ids = (
        [e.decision_id for e in mapping.entries if not e.scored_goals]
        + [d_id for d_id in decisions_by_id if d_id not in entries_decision_ids]
    )

    sections = [
        f"# Candidate Mapping — condition: {mapping.condition_name}",
        "",
        "## 1. Summary",
        f"Total entries: {len(mapping.entries)} decisions assessed, "
        f"{total_scores} scored goal relationships emitted.",
        f"- Unattached decisions: {len(orphan_ids)}",
        "",
        "The candidate mapping emits scored relationships above a "
        f"{mapping.score_threshold:.2f} confidence threshold. "
        "Relationships scoring below this threshold are not shown.",
        "",
        "## 2. Per-Decision Relationships",
        "",
    ]

    for entry in mapping.entries:
        if not entry.scored_goals:
            continue  # rendered in orphan subsection below
        decision = decisions_by_id.get(entry.decision_id)
        d_title = decision.title if decision else entry.decision_id
        scored_lines = []
        for s in sorted(entry.scored_goals, key=lambda x: x.score, reverse=True):
            goal = goals_by_id.get(s.goal_id)
            g_title = goal.title if goal else s.goal_id
            scored_lines.append(f"  - *{g_title}*: {s.score:.2f} — {s.reasoning}")
        sections.append(f"- **{d_title}**")
        sections.extend(scored_lines)

    if orphan_ids:
        sections += _render_orphan_subsection(orphan_ids, decisions_by_id)

    # Score distribution (use buckets as proxy for confidence histogram)
    all_scores = [s.score for e in mapping.entries for s in e.scored_goals]
    high_c = sum(1 for s in all_scores if s >= 0.80)
    med_c = sum(1 for s in all_scores if 0.50 <= s < 0.80)
    low_c = sum(1 for s in all_scores if s < 0.50)

    sections += [
        "",
        "## 3. Cross-References",
        "",
        "(no cross-references by design — each entry is a decision scored against individual goals)",
        "",
        "## 4. Score Distribution",
        "",
        "| Score range | Count |",
        "|-------------|-------|",
        f"| 0.80–1.00   | {high_c:5} |",
        f"| 0.50–0.79   | {med_c:5} |",
        f"| {mapping.score_threshold:.2f}–0.49   | {low_c:5} |",
        "",
    ]
    return "\n".join(sections)


# ── GM renderer ───────────────────────────────────────────────────────────────

def _render_gm(
    mapping: GMMapping,
    decisions_by_id: dict[str, Decision],
    goals_by_id: dict[str, CanonicalGoal],
) -> str:
    # Partition edges
    dec_goal_edges = [
        e for e in mapping.edges
        if (e.source_kind == "decision" and e.target_kind == "goal") or
           (e.source_kind == "goal" and e.target_kind == "decision")
    ]
    cross_edges = [
        e for e in mapping.edges
        if (e.source_kind == e.target_kind)  # goal↔goal (dec↔dec never fires)
    ]

    # Compute orphan decisions (not in any decision↔goal edge)
    connected_decision_ids = set()
    for e in dec_goal_edges:
        if e.source_kind == "decision":
            connected_decision_ids.add(e.source_id)
        else:
            connected_decision_ids.add(e.target_id)
    orphan_ids = [d_id for d_id in decisions_by_id if d_id not in connected_decision_ids]

    sections = [
        f"# Candidate Mapping — condition: {mapping.condition_name}",
        "",
        "## 1. Summary",
        f"Total relationships: {len(mapping.edges)} labeled edges.",
        f"- Decision-to-goal relationships: {len(dec_goal_edges)}",
        f"- Cross-node relationships (goal↔goal): {len(cross_edges)}",
        f"- Unattached decisions: {len(orphan_ids)}",
        "",
        "Relationships are typed using a closed vocabulary of eight relation kinds.",
        "",
        "## 2. Per-Decision Relationships",
        "",
    ]

    # Group decision-anchored edges by decision
    edges_by_decision: dict[str, list] = defaultdict(list)
    for e in dec_goal_edges:
        if e.source_kind == "decision":
            edges_by_decision[e.source_id].append(("→", e.target_id, "goal", e))
        else:
            edges_by_decision[e.target_id].append(("←", e.source_id, "goal", e))

    for d_id, decision in decisions_by_id.items():
        if d_id in connected_decision_ids:
            d_edges = edges_by_decision.get(d_id, [])
            sections.append(f"- **{decision.title}** ({d_id[:8]}…)")
            for direction, other_id, other_kind, edge in d_edges:
                goal = goals_by_id.get(other_id)
                other_title = goal.title if goal else other_id
                sections.append(
                    f"  - `{edge.relation}` {direction} *{other_title}*  "
                    f"[{edge.confidence}] {edge.label} — {edge.reasoning}"
                )

    if orphan_ids:
        sections += _render_orphan_subsection(orphan_ids, decisions_by_id)

    sections += [
        "",
        "## 3. Cross-References",
        "",
    ]

    if cross_edges:
        for edge in cross_edges:
            src = (decisions_by_id.get(edge.source_id) or goals_by_id.get(edge.source_id))
            tgt = (decisions_by_id.get(edge.target_id) or goals_by_id.get(edge.target_id))
            src_title = (src.title if src else edge.source_id)
            tgt_title = (tgt.title if tgt else edge.target_id)
            sections.append(
                f"- [{edge.source_kind}] *{src_title}* `{edge.relation}` "
                f"[{edge.target_kind}] *{tgt_title}*  "
                f"[{edge.confidence}] {edge.label}"
            )
    else:
        sections.append("(no cross-node relationships emitted)")

    sections += [
        "",
        "## 4. Confidence Distribution",
        "",
        _confidence_histogram([e.confidence for e in mapping.edges]),
        "",
    ]
    return "\n".join(sections)


# ── Public API ────────────────────────────────────────────────────────────────

def render_and_save(
    mapping: DMMapping | DSMMapping | GMMapping,
    decisions: list[Decision],
    goals: list[CanonicalGoal],
    output_path: Path,
) -> str:
    """Render the mapping to schema-masked markdown and save to disk.

    Returns the rendered markdown string.
    """
    decisions_by_id = {d.id: d for d in decisions}
    goals_by_id = {g.id: g for g in goals}

    if isinstance(mapping, DMMapping):
        schema_tag = "dm"
        body = _render_dm(mapping, decisions_by_id, goals_by_id)
    elif isinstance(mapping, DSMMapping):
        schema_tag = "dsm"
        body = _render_dsm(mapping, decisions_by_id, goals_by_id)
    elif isinstance(mapping, GMMapping):
        schema_tag = "gm"
        body = _render_gm(mapping, decisions_by_id, goals_by_id)
    else:
        raise TypeError(f"Unknown mapping type: {type(mapping)}")

    wc = _word_count(body)
    md = body + f"\n## 5. Rendering Metadata\n\n- Condition: {mapping.condition_name}\n- Rendering word count: {wc}\n"

    md_path = output_path / f"mapping_{schema_tag}.md"
    md_path.write_text(md)
    print(f"  Rendered → {md_path} ({wc} words)")
    return md
