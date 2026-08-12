"""Request body sampling from database with stratified sampling by prompt length."""

import json

import asyncpg
import numpy as np

from src.benchmark.format_converter import calculate_prompt_length


def _has_complex_content(request_body: dict) -> bool:
    """
    Check if request has complex content that's hard to convert (multi-turn with tool use/results).

    Args:
        request_body: Anthropic request body

    Returns:
        True if request has complex content blocks that should be skipped
    """
    messages = request_body.get("messages", [])

    for msg in messages:
        if not isinstance(msg, dict):
            continue

        if msg.get("role") == "assistant":
            return True

        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") in ["tool_use", "tool_result"]:
                    return True

    return False


async def fetch_request_bodies_from_db(db_name: str) -> list[dict]:
    """
    Fetch all request bodies from the llmrequest table.

    Args:
        db_name: Name of the PostgreSQL database

    Returns:
        List of dicts with 'request_body' and 'length' keys
    """
    conn = await asyncpg.connect(database=db_name, user="decide", host="localhost")

    try:
        rows = await conn.fetch(
            """
            SELECT request_body
            FROM llmrequest
            WHERE request_body IS NOT NULL
            """
        )

        request_bodies = []
        for row in rows:
            request_body_json = row["request_body"]
            if request_body_json:
                try:
                    request_body = json.loads(request_body_json) if isinstance(request_body_json, str) else request_body_json

                    if "messages" in request_body and request_body["messages"]:
                        if _has_complex_content(request_body):
                            continue

                        prompt_length = calculate_prompt_length(request_body)
                        if prompt_length > 0:
                            request_bodies.append({"request_body": request_body, "length": prompt_length})
                except (json.JSONDecodeError, TypeError, AttributeError):
                    continue

        return request_bodies
    finally:
        await conn.close()


def stratified_sample_by_length(request_bodies: list[dict], n_samples: int) -> list[dict]:
    """
    Sample request bodies using stratified sampling by prompt length quartiles.

    Args:
        request_bodies: List of dicts with 'request_body' and 'length' keys
        n_samples: Number of samples to return

    Returns:
        List of sampled request body dicts
    """
    if not request_bodies:
        return []

    if len(request_bodies) <= n_samples:
        return [rb["request_body"] for rb in request_bodies]

    lengths = np.array([rb["length"] for rb in request_bodies])

    quartiles = np.percentile(lengths, [25, 50, 75])

    strata = []
    for rb in request_bodies:
        if rb["length"] <= quartiles[0]:
            strata.append(0)
        elif rb["length"] <= quartiles[1]:
            strata.append(1)
        elif rb["length"] <= quartiles[2]:
            strata.append(2)
        else:
            strata.append(3)

    samples_per_stratum = n_samples // 4
    remainder = n_samples % 4

    sampled_bodies = []

    for stratum_idx in range(4):
        stratum_bodies = [request_bodies[i] for i in range(len(request_bodies)) if strata[i] == stratum_idx]

        if not stratum_bodies:
            continue

        n_to_sample = samples_per_stratum + (1 if stratum_idx < remainder else 0)
        n_to_sample = min(n_to_sample, len(stratum_bodies))

        indices = np.random.choice(len(stratum_bodies), size=n_to_sample, replace=False)
        sampled_bodies.extend([stratum_bodies[i]["request_body"] for i in indices])

    np.random.shuffle(sampled_bodies)

    return sampled_bodies


async def sample_request_bodies_from_db(db_name: str, n_samples: int) -> list[dict]:
    """
    Fetch request bodies from database and return stratified sample by prompt length.

    Args:
        db_name: Name of the PostgreSQL database
        n_samples: Number of request bodies to sample

    Returns:
        List of sampled request body dicts (in Anthropic format)
    """
    request_bodies = await fetch_request_bodies_from_db(db_name)
    return stratified_sample_by_length(request_bodies, n_samples)
