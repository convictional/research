"""manifest.json — run metadata + catalog snapshot, written at run start
and updated on completion. Captures enough context that a stale run is
still interpretable after the catalog changes.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from geo_analyzer.runtime import Run


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run: Run
    subject_ids: list[str]
    prompt_ids: list[str]
    model_ids: list[str]
    catalog_hash: str
    """SHA-256 hex of the concatenated YAML files at load time. Used by
    cross-run analysis to detect catalog drift between runs."""


def write_manifest(path: Path, manifest: Manifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")


def read_manifest(path: Path) -> Manifest:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return Manifest.model_validate(raw)
