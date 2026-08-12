from __future__ import annotations

from pathlib import Path

import pytest

from geo_analyzer.goals import GoalsError, load_goals


def _write(p: Path, body: str) -> Path:
    p.write_text(body.strip())
    return p


class TestLoadGoals:
    def test_happy_path(self, tmp_path: Path) -> None:
        p = _write(
            tmp_path / "goals.yaml",
            """
- id: brand-sov-l3-2026q3
  subject: convictional_brand
  metric: share_of_voice
  tier: L3
  target: 0.25
  created_at: 2026-04-23
  target_date: 2026-09-01

- id: legacy-l4-eoy
  subject: convictional_legacy_dropship
  metric: mention_presence
  tier: L4
  target: 0.05
  direction: below
  created_at: 2026-04-23
  target_date: 2026-12-01
""",
        )
        goals = load_goals(p)
        assert {g.id for g in goals} == {"brand-sov-l3-2026q3", "legacy-l4-eoy"}
        assert goals[1].direction == "below"

    def test_target_date_must_be_after_created_at(self, tmp_path: Path) -> None:
        p = _write(
            tmp_path / "goals.yaml",
            """
- id: bad
  subject: x
  metric: mention_presence
  tier: L1
  target: 0.1
  created_at: 2026-09-01
  target_date: 2026-04-01
""",
        )
        with pytest.raises(GoalsError, match="target_date.*after.*created_at"):
            load_goals(p)

    def test_duplicate_id_rejected(self, tmp_path: Path) -> None:
        p = _write(
            tmp_path / "goals.yaml",
            """
- id: dup
  subject: x
  metric: m
  tier: L1
  target: 0.1
  created_at: 2026-04-01
  target_date: 2026-09-01
- id: dup
  subject: y
  metric: m
  tier: L2
  target: 0.2
  created_at: 2026-04-01
  target_date: 2026-09-01
""",
        )
        with pytest.raises(GoalsError, match="duplicate goal id"):
            load_goals(p)

    def test_missing_file_returns_empty_list(self, tmp_path: Path) -> None:
        # Goals are optional — a project can have zero goals.
        assert load_goals(tmp_path / "goals.yaml") == []
