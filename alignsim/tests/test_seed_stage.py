"""Tests for the seed_stage scenario's process-project tech-tree (Stage C).

Guards the hand-authored prerequisite DAG: ids resolve, no cycles, tier shape, and that the
authored tree is fully deterministic (identical across builds / seeds — no RNG).
"""

from alignsim.src.scenarios.seed_stage import create_seed_stage_scenario


def _projects():
    scenario = create_seed_stage_scenario(seed=42)
    return {p.id: p for p in scenario.process_projects}


def test_all_prerequisite_ids_exist():
    projects = _projects()
    for p in projects.values():
        for prereq in p.prerequisites:
            assert prereq in projects, f"{p.id} references missing prerequisite {prereq}"


def test_prerequisite_dag_has_no_cycles():
    """Kahn topo-sort: if fewer than all nodes are visited, there is a cycle."""
    projects = _projects()
    indegree = {pid: len(p.prerequisites) for pid, p in projects.items()}
    dependents: dict[str, list[str]] = {pid: [] for pid in projects}
    for pid, p in projects.items():
        for prereq in p.prerequisites:
            dependents[prereq].append(pid)

    queue = [pid for pid, d in indegree.items() if d == 0]
    visited = 0
    while queue:
        node = queue.pop()
        visited += 1
        for dep in dependents[node]:
            indegree[dep] -= 1
            if indegree[dep] == 0:
                queue.append(dep)
    assert visited == len(projects), "cycle detected in process-project prerequisites"


def test_tier0_projects_have_no_prerequisites():
    projects = _projects()
    for pid in ["PP01", "PP02", "PP03", "PP04", "PP05", "PP06"]:
        assert projects[pid].prerequisites == []


def test_tier1_and_capstone_prerequisites_as_authored():
    projects = _projects()
    assert projects["PP07"].prerequisites == ["PP01"]
    assert projects["PP08"].prerequisites == ["PP02"]
    assert projects["PP09"].prerequisites == ["PP03"]
    assert set(projects["PP10"].prerequisites) == {"PP07", "PP08"}


def test_bonus_max_strictly_increases_along_each_chain():
    """A project's bonus_max exceeds every same-bonus_type prerequisite's (escalating tree)."""
    projects = _projects()
    for p in projects.values():
        for prereq_id in p.prerequisites:
            prereq = projects[prereq_id]
            if prereq.bonus_type == p.bonus_type:
                assert p.bonus_max > prereq.bonus_max, (
                    f"{p.id} bonus_max {p.bonus_max} not > prereq {prereq_id} {prereq.bonus_max}"
                )


def test_floors_increase_by_tier():
    """Permanent floors escalate: tier-1 (PP07-09) > tier-0 base; capstone PP10 is highest."""
    projects = _projects()
    tier1_floors = [projects[p].permanent_floor_fraction for p in ["PP07", "PP08", "PP09"]]
    assert all(f >= 0.25 for f in tier1_floors)
    assert projects["PP10"].permanent_floor_fraction == max(
        p.permanent_floor_fraction for p in projects.values()
    )


def test_authored_tree_is_deterministic_and_seed_independent():
    a = create_seed_stage_scenario(seed=42).process_projects
    b = create_seed_stage_scenario(seed=42).process_projects
    c = create_seed_stage_scenario(seed=999).process_projects
    assert [p.model_dump() for p in a] == [p.model_dump() for p in b]
    assert [p.model_dump() for p in a] == [p.model_dump() for p in c]


# =============================================================================
# v2 ECONOMICS GUARDRAIL
# Locks the v2 "grow, not survive" tune. If these fail, seed_stage's economics
# drifted from the intended calibration — change them deliberately, not by accident.
# =============================================================================

def _v2_scenario():
    return create_seed_stage_scenario(seed=42)


def test_v2_goal_targets():
    g = _v2_scenario().primary_goal
    assert g.mrr_target == 40_000
    assert g.min_runway_turns == 60
    assert g.max_churn_rate == 0.02
    assert g.target_turn == 48


def test_v2_starting_capacity():
    f = _v2_scenario().financials
    assert f.capacity_per_turn == 15
    assert f.eng_capacity == 6
    assert f.sales_capacity == 6
    assert f.marketing_capacity == 3
    assert f.support_capacity == 0
    assert f.ops_capacity == 0


def test_v2_calibration():
    c = _v2_scenario().calibration
    assert c.team_cost_per_capacity == 2200
    assert c.lead_to_prospect_rate == 0.35
    assert c.prospect_to_qualified_rate == 0.55
    assert c.qualified_to_in_deal_rate == 0.48
    assert c.in_deal_to_closed_rate == 0.40


def test_v2_deal_value_monotonic_by_segment():
    """The mid_market-below-growth inversion is fixed: mean deal_value strictly
    increases startup < growth < mid_market < enterprise."""
    by_seg: dict[str, list[int]] = {}
    for c in _v2_scenario().customers:
        by_seg.setdefault(c.segment.value, []).append(c.deal_value)
    means = {seg: sum(v) / len(v) for seg, v in by_seg.items()}
    assert means["startup"] < means["growth"] < means["mid_market"] < means["enterprise"]


def test_v2_desired_price_within_band():
    """Repricing keeps every desired_price_point valid: 50 <= desired <= deal_value."""
    for c in _v2_scenario().customers:
        assert 50 <= c.desired_price_point <= c.deal_value
