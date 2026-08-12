from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def fake_catalog_dir(tmp_path: Path) -> Path:
    """A catalog directory with a single valid subject, model, and prompt."""

    cat = tmp_path / "catalog"
    (cat / "prompts").mkdir(parents=True)

    (cat / "subjects.yaml").write_text(
        """
- id: convictional_brand
  kind: brand
  aliases: [convictional]
  definition: Convictional positioning.
  competitors: [lattice]
  owned_domains: [convictional.com]
""".strip()
    )

    (cat / "models.yaml").write_text(
        """
providers:
  openai:
    concurrency: 4
models:
  - id: openai:gpt-5.1:ungrounded
    provider: openai
    model_name: gpt-5.1
    mode: ungrounded
    active: true
    config: {}
    sampling: {n: 1, temperature: 0, seed: 42}
""".strip()
    )

    (cat / "prompts" / "l1_broad.yaml").write_text(
        """
- id: prompt.brand.l1.intro
  tier: L1
  text: What's wrong with how modern companies are run?
  targets: [convictional_brand]
  version: 1
  authored_at: 2026-04-23
""".strip()
    )

    for fname in ("l2_adjacent.yaml", "l3_category.yaml", "l4_brand.yaml"):
        (cat / "prompts" / fname).write_text("[]")

    return cat
