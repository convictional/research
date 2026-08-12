from __future__ import annotations

from geo_analyzer.reports.html import render_html


class TestRenderHtml:
    def test_wraps_in_html5_doc(self) -> None:
        out = render_html("# Hello", title="Run xyz")
        assert out.startswith("<!DOCTYPE html>")
        assert "<title>Run xyz</title>" in out
        assert "<h1>Hello</h1>" in out

    def test_table_extension_renders_markdown_table(self) -> None:
        md = "| col |\n|---|\n| value |\n"
        out = render_html(md, title="x")
        assert "<table>" in out
        assert "<th>col</th>" in out
        assert "<td>value</td>" in out

    def test_blockquote_renders(self) -> None:
        # Worst-prompts entries use blockquotes for the inline prompt text.
        md = "- `prompt.x` — 50%\n  > What is the prompt?\n"
        out = render_html(md, title="x")
        assert "<blockquote>" in out
        assert "What is the prompt?" in out

    def test_inline_css_present(self) -> None:
        out = render_html("# x", title="x")
        # Should be a self-contained doc — no external stylesheet links.
        assert "<style>" in out
        assert 'rel="stylesheet"' not in out

    def test_title_special_chars_escaped(self) -> None:
        out = render_html("# x", title="<bad>")
        assert "<title>&lt;bad&gt;</title>" in out
