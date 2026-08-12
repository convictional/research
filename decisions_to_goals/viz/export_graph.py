"""Export Phase-2 mapping artifacts into a browser-loadable graph dataset.

Reads, for each of the 9 cells (3 conditions × 3 schemas):
  - output/<cond>/step5_final_goal_set.pkl  → goal nodes (id, title, stated/unstated)
  - output/shared/decisions.pkl             → decision titles
  - output/<cond>/mapping_<schema>.pkl      → links (decision↔goal[↔goal] relationships)

Writes viz/cell_data.js assigning `window.CELL_DATA`, so the static viz works when
index.html is opened directly from disk (file://) with no server or CORS setup.

Run from the experiments/ directory:
    PYTHONPATH=. uv run python -m decisions_to_goals.viz.export_graph
"""
import json
import pickle
from pathlib import Path

# Cell axes
CONDITIONS = {"unstated": "unstated", "stated": "stated", "mixed": "mixed"}
SCHEMAS = ["dm", "dsm", "gm"]

CONDITION_LABELS = {
    "unstated": "Unstated Goals Only",
    "stated": "Stated Goals Only",
    "mixed": "Mixed (Stated + Unstated)",
}
SCHEMA_LABELS = {
    "dm": "Direct Mapping",
    "dsm": "Direct Score Mapping",
    "gm": "Graph Map",
}

_CONFIDENCE_WEIGHT = {"low": 0.35, "medium": 0.7, "high": 1.0}

_OUTPUT = Path(__file__).resolve().parent.parent / "output"
_VIZ_DIR = Path(__file__).resolve().parent

# Import the orphan sentinel — must match models.py
_ORPHAN_GOAL_ID = "00000000-0000-0000-0000-000000000000"
_ORPHAN_GOAL_TITLE = "Unattached / Miscellaneous Decisions"


def _load(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _short(text: str, limit: int = 70) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _build_cell(cond: str, schema: str, goals_by_id: dict, decisions_by_id: dict) -> dict | None:
    """Build one cell's {nodes, links, meta}. Returns None if the mapping artifact is missing."""
    prefix = CONDITIONS[cond]
    mapping_path = _OUTPUT / prefix / f"mapping_{schema}.pkl"
    if not mapping_path.exists():
        return None
    mapping = _load(mapping_path)

    # (id, kind) -> link list; collect raw links first, then materialize only connected nodes
    raw_links: list[dict] = []
    used: dict[str, str] = {}  # node_id -> kind ("goal"|"decision")
    connected_decision_ids: set[str] = set()

    def mark(node_id: str, kind: str) -> None:
        used[node_id] = kind

    if schema == "dm":
        for e in mapping.entries:
            if e.goal_id is None:
                continue
            mark(e.decision_id, "decision")
            mark(e.goal_id, "goal")
            connected_decision_ids.add(e.decision_id)
            raw_links.append({
                "source": e.decision_id, "target": e.goal_id,
                "relation": "maps_to", "label": e.confidence,
                "value": _CONFIDENCE_WEIGHT.get(e.confidence, 0.7),
            })
    elif schema == "dsm":
        for e in mapping.entries:
            for sg in e.scored_goals:
                mark(e.decision_id, "decision")
                mark(sg.goal_id, "goal")
                connected_decision_ids.add(e.decision_id)
                raw_links.append({
                    "source": e.decision_id, "target": sg.goal_id,
                    "relation": "scored", "label": f"{sg.score:.2f}",
                    "value": round(float(sg.score), 3),
                })
    elif schema == "gm":
        for e in mapping.edges:
            mark(e.source_id, e.source_kind)
            mark(e.target_id, e.target_kind)
            if e.source_kind == "decision":
                connected_decision_ids.add(e.source_id)
            if e.target_kind == "decision":
                connected_decision_ids.add(e.target_id)
            raw_links.append({
                "source": e.source_id, "target": e.target_id,
                "relation": e.relation, "label": e.label or e.relation,
                "value": _CONFIDENCE_WEIGHT.get(e.confidence, 0.7),
            })

    # Orphan decisions — those not connected to any goal
    orphan_decision_ids = [d_id for d_id in decisions_by_id if d_id not in connected_decision_ids]
    n_orphans = len(orphan_decision_ids)

    if orphan_decision_ids:
        # Materialize the synthetic orphan goal node
        used[_ORPHAN_GOAL_ID] = "goal"
        for d_id in orphan_decision_ids:
            used[d_id] = "decision"
            raw_links.append({
                "source": d_id,
                "target": _ORPHAN_GOAL_ID,
                "relation": "unattached",
                "label": "",
                "value": 0.3,
            })

    # Materialize nodes for every id that participated in at least one link
    nodes: list[dict] = []
    for node_id, kind in used.items():
        if kind == "goal":
            if node_id == _ORPHAN_GOAL_ID:
                nodes.append({
                    "id": node_id, "type": "goal", "stated": None, "orphan": True,
                    "name": "Unattached / Misc",
                    "full": _ORPHAN_GOAL_TITLE,
                })
                continue
            g = goals_by_id.get(node_id)
            if g is not None:
                stated = bool(g.is_stated)
                title = g.title
                name = _short(title)
            else:
                stated = None  # referenced by the mapper but absent from the canonical goal set
                title = f"(unknown goal {node_id[:8]})"
                name = title
            nodes.append({
                "id": node_id, "type": "goal", "stated": stated, "orphan": False,
                "name": name, "full": title,
            })
        else:
            d = decisions_by_id.get(node_id)
            title = d.title if d is not None else node_id
            nodes.append({
                "id": node_id, "type": "decision",
                "name": _short(title), "full": title,
            })

    n_goals = sum(1 for n in nodes if n["type"] == "goal" and not n.get("orphan", False))
    n_decisions = sum(1 for n in nodes if n["type"] == "decision")

    return {
        "id": f"{prefix}__{schema}",
        "condition": cond,
        "schema": schema,
        "condition_label": CONDITION_LABELS[cond],
        "schema_label": SCHEMA_LABELS[schema],
        "nodes": nodes,
        "links": raw_links,
        "meta": {
            "n_goals": n_goals,
            "n_decisions": n_decisions,
            "n_orphans": n_orphans,
            "n_links": len(raw_links),
            "goals_in_set": len(goals_by_id),
        },
    }


def build_all() -> dict:
    decisions = _load(_OUTPUT / "shared" / "decisions.pkl")
    decisions_by_id = {d.id: d for d in decisions}

    cells: dict[str, dict] = {}
    for cond, prefix in CONDITIONS.items():
        goal_set_path = _OUTPUT / prefix / "step5_final_goal_set.pkl"
        if not goal_set_path.exists():
            print(f"  [skip] {prefix}: no step5_final_goal_set.pkl")
            continue
        goal_set = _load(goal_set_path)
        goals_by_id = {g.id: g for g in goal_set.goals}
        for schema in SCHEMAS:
            cell = _build_cell(cond, schema, goals_by_id, decisions_by_id)
            if cell is None:
                print(f"  [skip] {prefix}__{schema}: no mapping artifact")
                continue
            cells[cell["id"]] = cell
            m = cell["meta"]
            print(
                f"  {cell['id']}: {m['n_decisions']} decisions, {m['n_goals']} goals, "
                f"{m['n_orphans']} orphans, {m['n_links']} links"
            )
    return cells


def main() -> None:
    print("Building graph data from Phase-2 mapping artifacts...")
    cells = build_all()
    if not cells:
        raise RuntimeError(
            "No cells built. Run the mapping phase first "
            "(make run_experiment ARGS='decisions_to_goals map_all')."
        )
    payload = {
        "conditions": list(CONDITIONS.keys()),
        "schemas": SCHEMAS,
        "condition_labels": CONDITION_LABELS,
        "schema_labels": SCHEMA_LABELS,
        "cells": cells,
    }
    out = _VIZ_DIR / "cell_data.js"
    out.write_text("window.CELL_DATA = " + json.dumps(payload, ensure_ascii=False) + ";\n")
    size_kb = out.stat().st_size / 1024
    print(f"\nWrote {out} ({size_kb:.0f} KB, {len(cells)} cells)")


if __name__ == "__main__":
    main()
