import asyncio
import json
import logging
import time
from pathlib import Path

import jinja2

from src.db import get_connection, get_organization_id, search_content
from src.dedupe_diff import deduplicate_learnings, derive_parity, match_learnings, resolve_duplicates
from src.extract import GEMMA_DELAY_BETWEEN_REQUESTS, extract_all
from src.models import ExtractionInput, ParityAnalysis
from src.prompts.engine import register_prompt_templates
from src.report import build_report, save_report
from src.research_prompts import PROMPTS
from src.settings import settings

logger = logging.getLogger(__name__)


def _init_prompts() -> None:
    prompts_dir = Path(__file__).parent / "prompts"
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(prompts_dir)),
        undefined=jinja2.StrictUndefined,
    )
    register_prompt_templates(env)


def _variant_tag(
    variant: str, gemma_version: str, haiku_version: str, local_port: int | None, passes: int = 1
) -> str:
    if variant == "sonnet":
        return "sonnet"
    if variant == "haiku":
        return f"haiku_{haiku_version}"
    if variant == "gemma":
        base = f"gemma_local_{gemma_version}" if local_port else f"gemma_{gemma_version}"
        return f"{base}_p{passes}"
    raise ValueError(f"Unknown variant: {variant}")


def _display_name(
    variant: str, gemma_version: str, haiku_version: str, local_port: int | None, passes: int = 1
) -> str:
    if variant == "sonnet":
        return "Sonnet"
    if variant == "haiku":
        return f"Haiku {haiku_version}"
    if variant == "gemma":
        prefix = "Gemma-local" if local_port else "Gemma"
        return f"{prefix} {gemma_version} ({passes}-pass)"
    raise ValueError(f"Unknown variant: {variant}")


def _variant_model_name(variant: str) -> str:
    if variant == "sonnet":
        return settings.sonnet_model
    if variant == "haiku":
        return settings.haiku_model
    if variant == "gemma":
        return settings.gemma_model
    raise ValueError(f"Unknown variant: {variant}")


def _extraction_path(tag: str) -> Path:
    return settings.output_path / f"{tag}_extraction.json"


def save_extraction(tag: str, data: dict[str, dict], meta: dict | None = None) -> None:
    path = _extraction_path(tag)
    payload = dict(data)
    if meta:
        payload["__meta__"] = meta
    path.write_text(json.dumps(payload, indent=2))
    n_ok = sum(1 for v in data.values() if v.get("status") == "success")
    n_failed = len(data) - n_ok
    logger.info(f"Saved {tag} extraction to {path} ({n_ok} OK, {n_failed} failed)")


def load_extraction(tag: str) -> tuple[dict[str, dict], dict]:
    path = _extraction_path(tag)
    if not path.exists():
        raise FileNotFoundError(f"No cached extraction at {path}. Run --extract {tag.split('_')[0]} first.")
    payload = json.loads(path.read_text())
    meta = payload.pop("__meta__", {}) if isinstance(payload, dict) else {}
    data = payload
    total_learnings = sum(len(v.get("learnings", [])) for v in data.values())
    n_ok = sum(1 for v in data.values() if v.get("status") == "success")
    n_failed = len(data) - n_ok
    logger.info(
        f"Loaded {tag} extraction from {path} "
        f"({total_learnings} total learnings, {n_ok} OK, {n_failed} failed)"
    )
    return data, meta


async def run(
    extract: list[str] | None,
    gemma_version: str,
    haiku_version: str,
    prompt_id: str | None,
    diff: tuple[str, str] | None,
    local_port: int | None = None,
    passes: int = 1,
) -> None:
    _init_prompts()

    prompts = PROMPTS
    if prompt_id:
        prompts = [p for p in prompts if p["id"] == prompt_id]
        if not prompts:
            valid = [p["id"] for p in PROMPTS]
            raise ValueError(f"Unknown prompt_id '{prompt_id}'. Valid: {valid}")

    if extract:
        conn = await get_connection()
        try:
            org_id = await get_organization_id(conn)

            inputs: list[ExtractionInput] = []
            for prompt in prompts:
                for query in prompt["queries"]:
                    results = await search_content(
                        conn, query["search_terms"], org_id, limit=settings.max_results_per_query
                    )
                    inputs.append(
                        ExtractionInput(
                            query_id=query["id"],
                            prompt_id=prompt["id"],
                            topic=prompt["topic"],
                            directions=query["goals"],
                            max_learnings=settings.max_learnings_per_query,
                            results=results,
                        )
                    )

            logger.info(f"Built {len(inputs)} extraction inputs across {len(prompts)} prompts")

            search_cache = {inp.query_id: [r.model_dump() for r in inp.results] for inp in inputs}
            (settings.output_path / "search_results.json").write_text(json.dumps(search_cache, indent=2))

            async def _extract_variant(variant: str) -> None:
                display = _display_name(variant, gemma_version, haiku_version, local_port, passes)
                tag = _variant_tag(variant, gemma_version, haiku_version, local_port, passes)
                logger.info(f"Running {display} extraction...")
                start = time.monotonic()
                variant_results = await extract_all(inputs, variant, gemma_version, haiku_version, local_port, passes)
                pipeline_latency_ms = int((time.monotonic() - start) * 1000)
                meta = {
                    "pipeline_latency_ms": pipeline_latency_ms,
                    "model_name": _variant_model_name(variant),
                    "variant": variant,
                    "passes": passes,
                    "max_concurrent": settings.max_concurrent,
                    "gemma_sequential_pacing_ms": int(GEMMA_DELAY_BETWEEN_REQUESTS * 1000)
                    if variant == "gemma" and not local_port
                    else None,
                }
                save_extraction(tag, variant_results, meta=meta)
                logger.info(f"{display} extraction wall-clock: {pipeline_latency_ms / 1000:.1f}s")

            await asyncio.gather(*(_extract_variant(v) for v in extract))
        finally:
            await conn.close()

    if diff:
        variant_a, variant_b = diff
        await _run_diff(prompts, variant_a, variant_b, gemma_version, haiku_version, local_port, passes)


async def _analyze_prompt(
    pid: str,
    query_ids: list[str],
    a_data: dict[str, dict],
    b_data: dict[str, dict],
    variant_a_label: str,
    variant_b_label: str,
) -> tuple[dict, ParityAnalysis]:
    a_raw: list[str] = []
    b_raw: list[str] = []
    a_failures: list[tuple[str, str]] = []
    b_failures: list[tuple[str, str]] = []
    a_input_tokens = 0
    a_output_tokens = 0
    b_input_tokens = 0
    b_output_tokens = 0
    a_latencies: list[int] = []
    b_latencies: list[int] = []
    a_any_usage = False
    b_any_usage = False
    for qid in query_ids:
        a_entry = a_data.get(qid, {"learnings": [], "status": "missing", "error": "not in cache"})
        b_entry = b_data.get(qid, {"learnings": [], "status": "missing", "error": "not in cache"})
        a_raw.extend(a_entry.get("learnings", []))
        b_raw.extend(b_entry.get("learnings", []))
        if a_entry.get("status") != "success":
            a_failures.append((qid, a_entry.get("error") or a_entry.get("status", "unknown")))
        if b_entry.get("status") != "success":
            b_failures.append((qid, b_entry.get("error") or b_entry.get("status", "unknown")))
        a_usage = a_entry.get("usage")
        if a_usage:
            a_input_tokens += a_usage.get("input_tokens", 0)
            a_output_tokens += a_usage.get("output_tokens", 0)
            a_any_usage = True
        b_usage = b_entry.get("usage")
        if b_usage:
            b_input_tokens += b_usage.get("input_tokens", 0)
            b_output_tokens += b_usage.get("output_tokens", 0)
            b_any_usage = True
        if a_entry.get("latency_ms") is not None:
            a_latencies.append(a_entry["latency_ms"])
        if b_entry.get("latency_ms") is not None:
            b_latencies.append(b_entry["latency_ms"])

    a_ok = len(query_ids) - len(a_failures)
    b_ok = len(query_ids) - len(b_failures)
    logger.info(
        f"Deduplicating {pid}: {variant_a_label}={len(a_raw)} learnings from {a_ok}/{len(query_ids)} OK queries, "
        f"{variant_b_label}={len(b_raw)} from {b_ok}/{len(query_ids)} OK queries"
    )
    (_, a_deduped), (_, b_deduped) = await asyncio.gather(
        deduplicate_learnings(a_raw, variant_a_label),
        deduplicate_learnings(b_raw, variant_b_label),
    )

    a_before_pass2 = len(a_deduped)
    b_before_pass2 = len(b_deduped)
    (_, a_deduped), (_, b_deduped) = await asyncio.gather(
        deduplicate_learnings(a_deduped, f"{variant_a_label} (pass 2)"),
        deduplicate_learnings(b_deduped, f"{variant_b_label} (pass 2)"),
    )
    logger.info(f"{variant_a_label} dedupe pass 2 for {pid}: {a_before_pass2} → {len(a_deduped)}")
    logger.info(f"{variant_b_label} dedupe pass 2 for {pid}: {b_before_pass2} → {len(b_deduped)}")

    logger.info(
        f"Matching {pid}: {len(a_deduped)} {variant_a_label} vs {len(b_deduped)} {variant_b_label} de-duped learnings"
    )
    match_result = await match_learnings(a_deduped, b_deduped, variant_a_label, variant_b_label)
    clean_pairs, warnings = await resolve_duplicates(
        match_result.pairs, a_deduped, b_deduped, variant_a_label, variant_b_label
    )
    parity = derive_parity(clean_pairs, a_deduped, b_deduped, warnings)

    stats = {
        "topic_id": pid,
        "a_raw": len(a_raw),
        "a_deduped": len(a_deduped),
        "b_raw": len(b_raw),
        "b_deduped": len(b_deduped),
        "shared": len(parity.shared),
        "a_only": len(parity.a_only),
        "b_only": len(parity.b_only),
        "query_count_total": len(query_ids),
        "a_ok": a_ok,
        "b_ok": b_ok,
        "a_failures": a_failures,
        "b_failures": b_failures,
        "a_usage": {"input_tokens": a_input_tokens, "output_tokens": a_output_tokens} if a_any_usage else None,
        "b_usage": {"input_tokens": b_input_tokens, "output_tokens": b_output_tokens} if b_any_usage else None,
        "a_latencies_ms": a_latencies,
        "b_latencies_ms": b_latencies,
    }
    return stats, parity


async def _run_diff(
    prompts: list[dict],
    variant_a: str,
    variant_b: str,
    gemma_version: str,
    haiku_version: str,
    local_port: int | None,
    passes: int = 1,
) -> None:
    a_tag = _variant_tag(variant_a, gemma_version, haiku_version, local_port, passes)
    b_tag = _variant_tag(variant_b, gemma_version, haiku_version, local_port, passes)
    a_label = _display_name(variant_a, gemma_version, haiku_version, local_port, passes)
    b_label = _display_name(variant_b, gemma_version, haiku_version, local_port, passes)

    a_data, a_meta = load_extraction(a_tag)
    b_data, b_meta = load_extraction(b_tag)

    async def _one(prompt: dict) -> tuple[dict, ParityAnalysis]:
        pid = prompt["id"]
        query_ids = [q["id"] for q in prompt["queries"]]
        stats, parity = await _analyze_prompt(pid, query_ids, a_data, b_data, a_label, b_label)
        for w in parity.duplicate_warnings:
            logger.warning(f"[{pid}] {w}")
        return stats, parity

    prompt_results: list[tuple[dict, ParityAnalysis]] = list(await asyncio.gather(*(_one(p) for p in prompts)))

    report = build_report(
        prompt_results=prompt_results,
        a_label=a_label,
        b_label=b_label,
        a_tag=a_tag,
        b_tag=b_tag,
        a_model_name=_variant_model_name(variant_a),
        b_model_name=_variant_model_name(variant_b),
        a_pipeline_latency_ms=a_meta.get("pipeline_latency_ms"),
        b_pipeline_latency_ms=b_meta.get("pipeline_latency_ms"),
        a_max_concurrent=a_meta.get("max_concurrent"),
        b_max_concurrent=b_meta.get("max_concurrent"),
        a_pacing_ms=a_meta.get("gemma_sequential_pacing_ms"),
        b_pacing_ms=b_meta.get("gemma_sequential_pacing_ms"),
    )

    path = save_report(report, a_tag, b_tag, settings.output_path)
    logger.info(f"Report saved to {path}")
    print(report)
