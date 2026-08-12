from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from geo_analyzer.cli import app

runner = CliRunner()


def test_validate_seed_catalog_succeeds() -> None:
    project_root = Path(__file__).resolve().parents[1]
    result = runner.invoke(app, ["catalog", "validate", "--catalog-dir", str(project_root / "catalog")])
    assert result.exit_code == 0, result.stdout


def test_validate_missing_catalog_fails(tmp_path: Path) -> None:
    result = runner.invoke(app, ["catalog", "validate", "--catalog-dir", str(tmp_path)])
    assert result.exit_code == 1
    assert "missing required file" in result.stdout or "missing required file" in (result.stderr or "")
