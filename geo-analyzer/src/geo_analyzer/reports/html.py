"""Wrap the markdown summary in a minimal styled HTML page.

Browsers don't render raw markdown, so opening summary.md via `--open-latest`
shows plain text. This module emits summary.html alongside summary.md so the
browser sees the intended layout: tables, code blocks, blockquotes, headings.
"""

from __future__ import annotations

import markdown

# Minimal CSS — just enough for headings, tables, code, and blockquotes to
# render readably. No fonts pulled from CDNs. No JS.
_CSS = """
body {
  max-width: 900px;
  margin: 2rem auto;
  padding: 0 1rem;
  font: 14px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
  color: #1f2328;
  background: #fff;
}
h1 { font-size: 1.75rem; margin: 0 0 1rem; }
h2 {
  font-size: 1.3rem;
  margin: 2rem 0 0.5rem;
  padding-top: 1rem;
  border-top: 1px solid #d0d7de;
}
h3 { font-size: 1.05rem; margin: 1rem 0 0.4rem; color: #57606a; }
p, li { margin: 0.3rem 0; }
ul { padding-left: 1.4rem; }
code {
  background: #f6f8fa;
  padding: 0.1em 0.35em;
  border-radius: 4px;
  font: 0.88em "SF Mono", Menlo, Consolas, monospace;
}
pre {
  background: #f6f8fa;
  padding: 0.8rem;
  border-radius: 6px;
  overflow-x: auto;
}
pre code { background: none; padding: 0; font-size: 0.85em; }
table { border-collapse: collapse; margin: 0.4rem 0 1rem; }
th, td {
  border: 1px solid #d0d7de;
  padding: 0.4rem 0.7rem;
  text-align: left;
}
th { background: #f6f8fa; }
blockquote {
  border-left: 3px solid #d0d7de;
  margin: 0.3rem 0 0.6rem 1.2rem;
  padding: 0 0 0 0.8rem;
  color: #57606a;
}
hr { border: 0; border-top: 1px solid #d0d7de; margin: 1.5rem 0; }
"""


def render_html(markdown_text: str, *, title: str) -> str:
    """Convert markdown to a self-contained HTML5 document.

    Uses python-markdown's `tables` and `fenced_code` extensions so the
    summary's tables and code fences render correctly. The CSS is inlined
    to keep the output a single file (no separate stylesheet to manage).
    """
    body_html = markdown.markdown(
        markdown_text,
        extensions=["tables", "fenced_code"],
        output_format="html",
    )
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        f"<title>{_escape(title)}</title>\n"
        f"<style>{_CSS}</style>\n"
        "</head>\n"
        "<body>\n"
        f"{body_html}\n"
        "</body>\n"
        "</html>\n"
    )


def _escape(s: str) -> str:
    """Minimal HTML escape for the <title> tag (no need for full escaper here)."""
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
