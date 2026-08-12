from datetime import datetime, timezone
from pathlib import Path

from src.models import ParityAnalysis
from src.settings import cost_usd


def _coverage_pct(shared: int, baseline_only: int) -> str:
    total = shared + baseline_only
    if total == 0:
        return "N/A"
    return f"{shared / total * 100:.0f}%"


def _dedup_stat(raw: int, deduped: int) -> str:
    if raw == deduped:
        return str(raw)
    redundancy = (1 - deduped / raw) * 100 if raw else 0
    return f"{raw} → {deduped} ({redundancy:.0f}% redundancy)"


def _percentile(values: list[int], pct: float) -> int | None:
    if not values:
        return None
    sorted_values = sorted(values)
    k = int(pct / 100 * (len(sorted_values) - 1))
    return sorted_values[k]


def _fmt_cost(value: float | None) -> str:
    return f"${value:,.4f}" if value is not None else "N/A"


def _fmt_int(value: int | None) -> str:
    return f"{value:,}" if value is not None else "N/A"


def build_report(
    prompt_results: list[tuple[dict, ParityAnalysis]],
    a_label: str,
    b_label: str,
    a_tag: str,
    b_tag: str,
    a_model_name: str,
    b_model_name: str,
    a_pipeline_latency_ms: int | None = None,
    b_pipeline_latency_ms: int | None = None,
    a_max_concurrent: int | None = None,
    b_max_concurrent: int | None = None,
    a_pacing_ms: int | None = None,
    b_pacing_ms: int | None = None,
) -> str:
    total_shared = sum(len(p.shared) for _, p in prompt_results)
    total_a_only = sum(len(p.a_only) for _, p in prompt_results)
    total_b_only = sum(len(p.b_only) for _, p in prompt_results)
    total_a_deduped = sum(len(p.a_deduped) for _, p in prompt_results)
    total_b_deduped = sum(len(p.b_deduped) for _, p in prompt_results)
    total_a_raw = sum(s["a_raw"] for s, _ in prompt_results)
    total_b_raw = sum(s["b_raw"] for s, _ in prompt_results)
    total_queries = sum(s.get("query_count_total", 0) for s, _ in prompt_results)
    total_a_ok = sum(s.get("a_ok", s.get("query_count_total", 0)) for s, _ in prompt_results)
    total_b_ok = sum(s.get("b_ok", s.get("query_count_total", 0)) for s, _ in prompt_results)
    a_to_b = _coverage_pct(total_shared, total_a_only)
    b_to_a = _coverage_pct(total_shared, total_b_only)
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        f"# Extraction Parity: {a_label} vs {b_label}",
        "",
        f"**Headline**: {a_label} → {b_label}: {a_to_b}, {b_label} → {a_label}: {b_to_a}",
        f"**Legend**: A = {a_label}, B = {b_label}. \"A → B\" = shared / (shared + A-only) — fraction of A's learnings also found in B (B's coverage of A).",
        f"**Calls OK**: {a_label} {total_a_ok}/{total_queries}, {b_label} {total_b_ok}/{total_queries}",
        f"**Generated**: {timestamp}",
        f"**Total**: {a_label} raw={total_a_raw}, {a_label} unique={total_a_deduped}, "
        f"{b_label} raw={total_b_raw}, {b_label} unique={total_b_deduped}, "
        f"Shared={total_shared}, A-only={total_a_only}, B-only={total_b_only}",
        "",
        "---",
        "",
        "## Per-Topic Summary",
        "",
        "| Topic | A-ok | B-ok | A raw | A uniq | B raw | B uniq | Shared | A-only | B-only | A→B % | B→A % |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for stats, _ in prompt_results:
        topic_a_to_b = _coverage_pct(stats["shared"], stats["a_only"])
        topic_b_to_a = _coverage_pct(stats["shared"], stats["b_only"])
        n_q = stats.get("query_count_total", 0)
        a_ok = stats.get("a_ok", n_q)
        b_ok = stats.get("b_ok", n_q)
        lines.append(
            f"| {stats['topic_id']} | {a_ok}/{n_q} | {b_ok}/{n_q} "
            f"| {stats['a_raw']} | {stats['a_deduped']} "
            f"| {stats['b_raw']} | {stats['b_deduped']} "
            f"| {stats['shared']} | {stats['a_only']} | {stats['b_only']} | {topic_a_to_b} | {topic_b_to_a} |"
        )

    lines.extend(["", "---", "", "## Deduplication Stats", ""])
    for stats, _ in prompt_results:
        lines.append(
            f"- **{stats['topic_id']}**: {a_label} {_dedup_stat(stats['a_raw'], stats['a_deduped'])}, "
            f"{b_label} {_dedup_stat(stats['b_raw'], stats['b_deduped'])}"
        )

    # Extraction Token Usage & Cost
    lines.extend(["", "---", "", "## Extraction Token Usage & Cost", ""])
    lines.append("Aggregated across all successful queries. Reflects what the prod extraction step would consume per run.")
    lines.append("")
    lines.append("| Variant | Model | Input tokens | Output tokens | Cost (USD) |")
    lines.append("| --- | --- | --- | --- | --- |")

    def _aggregate_usage(side: str) -> dict | None:
        any_usage = False
        total_in = 0
        total_out = 0
        for stats, _ in prompt_results:
            u = stats.get(f"{side}_usage")
            if u:
                any_usage = True
                total_in += u.get("input_tokens", 0)
                total_out += u.get("output_tokens", 0)
        return {"input_tokens": total_in, "output_tokens": total_out} if any_usage else None

    a_total_usage = _aggregate_usage("a")
    b_total_usage = _aggregate_usage("b")

    def _usage_row(label: str, model: str, usage: dict | None) -> str:
        if usage is None:
            return f"| {label} | {model} | N/A | N/A | N/A |"
        return (
            f"| {label} | {model} | {_fmt_int(usage['input_tokens'])} "
            f"| {_fmt_int(usage['output_tokens'])} "
            f"| {_fmt_cost(cost_usd(usage, model))} |"
        )

    lines.append(_usage_row(a_label, a_model_name, a_total_usage))
    lines.append(_usage_row(b_label, b_model_name, b_total_usage))

    footnotes: list[str] = []
    if a_total_usage is None:
        footnotes.append(f"`{a_tag}` cache predates usage tracking — re-extract to populate.")
    if b_total_usage is None:
        footnotes.append(f"`{b_tag}` cache predates usage tracking — re-extract to populate.")
    if footnotes:
        lines.append("")
        for note in footnotes:
            lines.append(f"> {note}")
    lines.append("")
    lines.append("> Judge calls (dedupe / match / tiebreak) are experiment-pipeline overhead and excluded — they don't run in prod.")

    # Latency
    lines.extend(["", "---", "", "## Latency", "", "### Per-call extraction (successful single API call, milliseconds)", ""])
    lines.append("| Variant | n | p50 | p95 | max |")
    lines.append("| --- | --- | --- | --- | --- |")

    def _latency_row(label: str, side: str) -> tuple[str, bool]:
        values: list[int] = []
        for stats, _ in prompt_results:
            values.extend(stats.get(f"{side}_latencies_ms", []) or [])
        if not values:
            return f"| {label} | 0 | N/A | N/A | N/A |", False
        return (
            f"| {label} | {len(values)} | {_fmt_int(_percentile(values, 50))} "
            f"| {_fmt_int(_percentile(values, 95))} | {_fmt_int(max(values))} |",
            True,
        )

    a_row, a_has_latency = _latency_row(a_label, "a")
    b_row, b_has_latency = _latency_row(b_label, "b")
    lines.append(a_row)
    lines.append(b_row)
    lat_footnotes: list[str] = []
    if not a_has_latency:
        lat_footnotes.append(f"`{a_tag}` cache predates latency tracking — re-extract to populate.")
    if not b_has_latency:
        lat_footnotes.append(f"`{b_tag}` cache predates latency tracking — re-extract to populate.")
    if lat_footnotes:
        lines.append("")
        for note in lat_footnotes:
            lines.append(f"> {note}")

    lines.extend(["", "### Pipeline-overall extraction wall-clock", ""])

    def _pipeline_line(label: str, model_name: str, latency_ms: int | None, max_concurrent: int | None, pacing_ms: int | None) -> str:
        if latency_ms is None:
            return f"- {label}: N/A (extraction did not run this session and meta not cached)"
        seconds = latency_ms / 1000
        if pacing_ms:
            caveat = (
                f" ⚠️ Vertex MaaS rate-limit pacing ({pacing_ms / 1000:.0f}s between requests) inflates this "
                f"— not directly comparable to a provisioned-throughput deployment."
            )
            return f"- {label}: {seconds:.1f}s (sequential, {pacing_ms / 1000:.0f}s pacing){caveat}"
        if max_concurrent:
            return f"- {label}: {seconds:.1f}s (concurrent, max_concurrent = {max_concurrent})"
        return f"- {label}: {seconds:.1f}s"

    lines.append(_pipeline_line(a_label, a_model_name, a_pipeline_latency_ms, a_max_concurrent, a_pacing_ms))
    lines.append(_pipeline_line(b_label, b_model_name, b_pipeline_latency_ms, b_max_concurrent, b_pacing_ms))

    lines.append("")
    lines.append(
        "> **Note:** Per-call latency includes Instructor's internal JSON-validation retries on the successful API call "
        "but excludes the outer 429 backoff loop (Gemma only). For multi-pass Gemma extraction, the per-call latency "
        "shown is the sum of the per-pass latencies."
    )

    # Failed calls
    failed_lines: list[str] = []
    for stats, _ in prompt_results:
        for qid, err in stats.get("a_failures", []):
            failed_lines.append(f"- [{a_label}] {qid}: {err}")
        for qid, err in stats.get("b_failures", []):
            failed_lines.append(f"- [{b_label}] {qid}: {err}")
    if failed_lines:
        lines.extend(["", "---", "", "## Failed Calls", ""])
        lines.extend(failed_lines)

    for stats, parity in prompt_results:
        pid = stats["topic_id"]
        lines.extend(["", "---", "", f"## {pid}", ""])

        lines.append("### Shared Learnings")
        lines.append("")
        lines.append(f"| # | {a_label} Learning | {b_label} Learning | Rationale |")
        lines.append("| --- | --- | --- | --- |")
        for i, pair in enumerate(parity.shared, 1):
            a_text = parity.a_deduped[pair.a_index - 1]
            b_text = parity.b_deduped[pair.b_index - 1]
            lines.append(f"| {i} | {a_text} | {b_text} | {pair.rationale} |")

        lines.extend(["", f"### {a_label}-Only Learnings", ""])
        for learning in parity.a_only:
            lines.append(f"- {learning}")

        lines.extend(["", f"### {b_label}-Only Learnings", ""])
        for learning in parity.b_only:
            lines.append(f"- {learning}")

        if parity.duplicate_warnings:
            lines.extend(["", "### Duplicate Resolution Warnings", ""])
            for w in parity.duplicate_warnings:
                lines.append(f"- {w}")

    return "\n".join(lines) + "\n"


def save_report(content: str, a_tag: str, b_tag: str, output_path: Path) -> Path:
    filename = f"parity_report_{a_tag}_vs_{b_tag}_{datetime.now(tz=timezone.utc).strftime('%Y%m%d_%H%M%S')}.md"
    path = output_path / filename
    path.write_text(content)
    return path
