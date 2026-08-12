"""Tests for sandbox setup scripts — verifies CLAUDE.md templating coverage."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

CLAUDE_MD_TEMPLATE = Path(__file__).parent.parent / "player_condition3" / "CLAUDE.md"
LAUNCHER_SCRIPT = Path(__file__).parent.parent / "scripts" / "sandbox_run_condition3.sh"

C4_LAUNCHER_SCRIPT = Path(__file__).parent.parent / "scripts" / "sandbox_run_condition4.sh"
C4_TEMPLATES = {
    "player_condition4a": Path(__file__).parent.parent / "player_condition4a" / "CLAUDE.md",
    "player_condition4b": Path(__file__).parent.parent / "player_condition4b" / "CLAUDE.md",
}


def _extract_apply_function_sed(script_text: str) -> str:
    """Return the body of the bash ``apply_function_sed()`` function (definition line → closing brace)."""
    out: list[str] = []
    capturing = False
    for line in script_text.splitlines():
        if line.startswith("apply_function_sed()"):
            capturing = True
        if capturing:
            out.append(line)
            if line.strip() == "}" and len(out) > 1:
                break
    return "\n".join(out)


class TestCLAUDEMDTemplating:
    def test_all_placeholders_have_replacements(self):
        """Every __PLACEHOLDER__ in CLAUDE.md has a sed replacement in the launcher."""
        template = CLAUDE_MD_TEMPLATE.read_text()
        script = LAUNCHER_SCRIPT.read_text()

        placeholders = set(re.findall(r"__[A-Z_]+__", template))
        assert len(placeholders) >= 7, f"Expected at least 7 placeholders, found: {placeholders}"

        assert "__FUNCTION__" in placeholders
        assert "s/__FUNCTION__/" in script

        per_function = placeholders - {"__FUNCTION__"}
        for ph in per_function:
            assert f"s/{ph}/" in script, f"Placeholder {ph} has no sed replacement in launcher"

    @pytest.mark.parametrize("function", ["engineering", "sales", "marketing", "support", "ops"])
    def test_function_covers_all_placeholders(self, function: str):
        """Each function's case block in apply_function_sed covers all per-function placeholders."""
        template = CLAUDE_MD_TEMPLATE.read_text()
        script = LAUNCHER_SCRIPT.read_text()

        per_function = set(re.findall(r"__[A-Z_]+__", template)) - {"__FUNCTION__"}

        pattern = rf"^\s*{function}\)\s*$(.*?)^\s*;;\s*$"
        match = re.search(pattern, script, re.MULTILINE | re.DOTALL)
        assert match, f"No case block found for {function} in apply_function_sed"
        block = match.group(1)

        for ph in per_function:
            assert ph in block, f"{ph} not replaced in {function}'s case block"

    def test_no_orphan_placeholders_in_template(self):
        """All placeholders follow the __UPPER_CASE__ naming convention."""
        template = CLAUDE_MD_TEMPLATE.read_text()
        placeholders = re.findall(r"__[A-Za-z_]+__", template)
        for ph in placeholders:
            assert ph == ph.upper(), f"Placeholder {ph} should be all-uppercase"


class TestC4CLAUDEMDTemplating:
    """The C4a/4b templates + sandbox_run_condition4.sh stay in sed parity (mirrors C3)."""

    @pytest.mark.parametrize("template_name", ["player_condition4a", "player_condition4b"])
    def test_all_placeholders_have_replacements(self, template_name: str):
        template = C4_TEMPLATES[template_name].read_text()
        script = C4_LAUNCHER_SCRIPT.read_text()

        placeholders = set(re.findall(r"__[A-Z_]+__", template))
        assert "__FUNCTION__" in placeholders
        assert "s/__FUNCTION__/" in script

        for ph in placeholders - {"__FUNCTION__"}:
            assert f"s/{ph}/" in script, f"Placeholder {ph} has no sed replacement in C4 launcher"

    @pytest.mark.parametrize("template_name", ["player_condition4a", "player_condition4b"])
    def test_c4_placeholders_match_c3(self, template_name: str):
        """C4 templates keep the identical placeholder set as C3 (prompt-parity)."""
        c3 = set(re.findall(r"__[A-Z_]+__", CLAUDE_MD_TEMPLATE.read_text()))
        c4 = set(re.findall(r"__[A-Z_]+__", C4_TEMPLATES[template_name].read_text()))
        assert c4 == c3

    @pytest.mark.parametrize("template_name", ["player_condition4a", "player_condition4b"])
    @pytest.mark.parametrize("function", ["engineering", "sales", "marketing", "support", "ops"])
    def test_function_covers_all_placeholders(self, template_name: str, function: str):
        template = C4_TEMPLATES[template_name].read_text()
        script = C4_LAUNCHER_SCRIPT.read_text()
        per_function = set(re.findall(r"__[A-Z_]+__", template)) - {"__FUNCTION__"}

        pattern = rf"^\s*{function}\)\s*$(.*?)^\s*;;\s*$"
        match = re.search(pattern, script, re.MULTILINE | re.DOTALL)
        assert match, f"No case block found for {function} in C4 apply_function_sed"
        block = match.group(1)
        for ph in per_function:
            assert ph in block, f"{ph} not replaced in {function}'s C4 case block"

    def test_c4_apply_function_sed_matches_c3(self):
        """The substituted role/goal/target VALUES stay identical to C3, not just placeholder coverage.

        The coverage tests above catch a MISSING replacement; this catches DRIFT — a future C4 edit
        that reworded a role description, goal metric, or target and silently broke cross-condition
        prompt parity (the exact regression this file exists to prevent).
        """
        c3_block = _extract_apply_function_sed(LAUNCHER_SCRIPT.read_text())
        c4_block = _extract_apply_function_sed(C4_LAUNCHER_SCRIPT.read_text())
        assert c3_block and c4_block, "apply_function_sed block not found in one of the scripts"
        assert c4_block == c3_block, (
            "C4 apply_function_sed drifted from C3 — role/goal/target substitutions must match "
            "across conditions to preserve prompt parity."
        )

    @pytest.mark.parametrize("template_name", ["player_condition4a", "player_condition4b"])
    def test_no_orphan_placeholders_in_template(self, template_name: str):
        """All C4 placeholders follow __UPPER_CASE__; a mixed-case one would evade the [A-Z_] checks."""
        placeholders = re.findall(r"__[A-Za-z_]+__", C4_TEMPLATES[template_name].read_text())
        for ph in placeholders:
            assert ph == ph.upper(), f"Placeholder {ph} should be all-uppercase in {template_name}"
