import asyncio
import logging
import time
from datetime import datetime, timezone

import google.auth
import google.auth.transport.requests
import instructor
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI, RateLimitError

from src.models import ExtractionInput, GeneratedResearchQueryReview
from src.prompts.engine import build_prompt
from src.settings import settings

logger = logging.getLogger(__name__)

GEMMA_MAX_RETRIES = 6
GEMMA_BASE_DELAY = 15.0
GEMMA_DELAY_BETWEEN_REQUESTS = 12.0


def _build_system_prompt() -> str:
    return build_prompt("system.md.jinja", now=datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))


async def anthropic_extract(
    input: ExtractionInput,
    model_name: str,
    prompt_template_path: str,
) -> tuple[list[str], dict, int]:
    system_prompt = _build_system_prompt()
    user_prompt = build_prompt(
        prompt_template_path,
        topic=input.topic,
        directions=input.directions,
        max_learnings=input.max_learnings,
        results=[r.model_dump() for r in input.results],
    )

    client = AsyncAnthropic(api_key=settings.anthropic_api_key.get_secret_value())
    instructor_client = instructor.from_anthropic(client)

    start = time.monotonic()
    result, completion = await instructor_client.chat.completions.create_with_completion(
        messages=[{"role": "user", "content": user_prompt}],
        max_tokens=4096,
        model=model_name,
        temperature=0.0,
        response_model=GeneratedResearchQueryReview,
        system=system_prompt,
    )
    latency_ms = int((time.monotonic() - start) * 1000)
    usage = {"input_tokens": completion.usage.input_tokens, "output_tokens": completion.usage.output_tokens}
    logger.info(
        f"{model_name} extracted {len(result.learnings)} learnings for {input.query_id} "
        f"(in={usage['input_tokens']}, out={usage['output_tokens']}, latency={latency_ms}ms)"
    )
    return result.learnings, usage, latency_ms


def _get_vertex_access_token() -> str:
    credentials, _ = google.auth.default()
    credentials.refresh(google.auth.transport.requests.Request())
    return credentials.token


def _build_gemma_client(local_port: int | None) -> tuple[AsyncOpenAI, str]:
    if local_port:
        base_url = f"http://localhost:{local_port}/v1"
        api_key = "not-needed"
    else:
        api_key = _get_vertex_access_token()
        project = settings.google_vertex_project
        location = settings.google_vertex_location
        base_url = f"https://aiplatform.googleapis.com/v1/projects/{project}/locations/{location}/endpoints/openapi"
    return AsyncOpenAI(api_key=api_key, base_url=base_url), base_url


async def _gemma_call(
    instructor_client: instructor.Instructor,
    system_prompt: str,
    user_prompt: str,
    query_id: str,
) -> tuple[GeneratedResearchQueryReview, dict, int]:
    for attempt in range(GEMMA_MAX_RETRIES):
        try:
            start = time.monotonic()
            result, completion = await instructor_client.chat.completions.create_with_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model=settings.gemma_model,
                temperature=1.0,
                top_p=0.95,
                max_tokens=16384,
                response_model=GeneratedResearchQueryReview,
                max_retries=5,
                extra_body={"top_k": 64},
            )
            latency_ms = int((time.monotonic() - start) * 1000)
            usage = {
                "input_tokens": completion.usage.prompt_tokens,
                "output_tokens": completion.usage.completion_tokens,
            }
            return result, usage, latency_ms
        except Exception as e:
            is_rate_limit = isinstance(e, RateLimitError) or "429" in str(e)
            if not is_rate_limit:
                raise
            delay = GEMMA_BASE_DELAY * (2 ** attempt)
            logger.warning(f"Gemma 429 for {query_id}, retry {attempt + 1}/{GEMMA_MAX_RETRIES} in {delay:.0f}s")
            if attempt == GEMMA_MAX_RETRIES - 1:
                raise
            await asyncio.sleep(delay)
    raise RuntimeError(f"Gemma retries exhausted for {query_id}")


async def gemma_extract(
    input: ExtractionInput,
    prompt_version: str = "v4",
    local_port: int | None = None,
    passes: int = 1,
) -> tuple[list[str], dict, int]:
    system_prompt = _build_system_prompt()
    results_data = [r.model_dump() for r in input.results]
    user_prompt = build_prompt(
        f"gemma/{prompt_version}.md.jinja",
        topic=input.topic,
        directions=input.directions,
        max_learnings=input.max_learnings,
        results=results_data,
    )

    client, _ = _build_gemma_client(local_port)
    instructor_client = instructor.from_openai(client, mode=instructor.Mode.JSON)

    result, usage, latency_ms = await _gemma_call(instructor_client, system_prompt, user_prompt, input.query_id)
    learnings = list(result.learnings)
    total_input = usage["input_tokens"]
    total_output = usage["output_tokens"]
    total_latency_ms = latency_ms
    logger.info(
        f"Gemma pass 1 extracted {len(learnings)} learnings for {input.query_id} "
        f"(in={usage['input_tokens']}, out={usage['output_tokens']}, latency={latency_ms}ms)"
    )

    followup_template = f"gemma/{prompt_version}_followup.md.jinja"
    for pass_num in range(2, passes + 1):
        followup_prompt = build_prompt(
            followup_template,
            topic=input.topic,
            directions=input.directions,
            max_learnings=input.max_learnings,
            results=results_data,
            previous_learnings=learnings,
        )
        try:
            result_n, usage_n, latency_n_ms = await _gemma_call(
                instructor_client, system_prompt, followup_prompt, f"{input.query_id}_pass{pass_num}"
            )
        except Exception as e:
            logger.error(
                f"Gemma pass {pass_num} failed for {input.query_id}: {type(e).__name__}: {str(e)[:200]} — "
                f"keeping {len(learnings)} learnings from prior passes"
            )
            break
        total_input += usage_n["input_tokens"]
        total_output += usage_n["output_tokens"]
        total_latency_ms += latency_n_ms
        new_learnings = [l for l in result_n.learnings if l.strip().upper() != "NONE"]
        logger.info(
            f"Gemma pass {pass_num} found {len(new_learnings)} new learnings for {input.query_id} "
            f"(in={usage_n['input_tokens']}, out={usage_n['output_tokens']}, latency={latency_n_ms}ms)"
        )
        if not new_learnings:
            logger.info(f"Gemma pass {pass_num} returned NONE for {input.query_id}, stopping early")
            break
        learnings.extend(new_learnings)

    return learnings, {"input_tokens": total_input, "output_tokens": total_output}, total_latency_ms


def _success(learnings: list[str], usage: dict, latency_ms: int) -> dict:
    return {
        "learnings": learnings,
        "status": "success",
        "error": None,
        "usage": usage,
        "latency_ms": latency_ms,
    }


def _failure(query_id: str, e: Exception) -> dict:
    error_excerpt = f"{type(e).__name__}: {str(e)[:500]}"
    logger.error(f"Extraction failed for {query_id}: {error_excerpt}")
    return {
        "learnings": [],
        "status": "failed",
        "error": error_excerpt,
        "usage": None,
        "latency_ms": None,
    }


async def _extract_one(
    variant: str,
    inp: ExtractionInput,
    gemma_version: str,
    haiku_version: str,
    local_port: int | None,
    passes: int,
) -> tuple[list[str], dict, int]:
    if variant == "sonnet":
        return await anthropic_extract(inp, settings.sonnet_model, "sonnet/extract.md.jinja")
    if variant == "haiku":
        return await anthropic_extract(inp, settings.haiku_model, f"haiku/{haiku_version}.md.jinja")
    if variant == "gemma":
        return await gemma_extract(inp, gemma_version, local_port, passes)
    raise ValueError(f"Unknown variant: {variant}")


async def extract_all(
    inputs: list[ExtractionInput],
    variant: str,
    gemma_version: str = "v4",
    haiku_version: str = "v1",
    local_port: int | None = None,
    passes: int = 1,
) -> dict[str, dict]:
    results: dict[str, dict] = {}

    if variant == "gemma" and not local_port:
        for i, inp in enumerate(inputs):
            if i > 0:
                logger.info(f"Waiting {GEMMA_DELAY_BETWEEN_REQUESTS:.0f}s before next Gemma request...")
                await asyncio.sleep(GEMMA_DELAY_BETWEEN_REQUESTS)
            try:
                learnings, usage, latency_ms = await _extract_one(
                    variant, inp, gemma_version, haiku_version, local_port, passes
                )
                results[inp.query_id] = _success(learnings, usage, latency_ms)
            except Exception as e:
                results[inp.query_id] = _failure(inp.query_id, e)
    else:
        semaphore = asyncio.Semaphore(settings.max_concurrent)

        async def _run(inp: ExtractionInput) -> tuple[str, dict]:
            async with semaphore:
                try:
                    learnings, usage, latency_ms = await _extract_one(
                        variant, inp, gemma_version, haiku_version, local_port, passes
                    )
                    return inp.query_id, _success(learnings, usage, latency_ms)
                except Exception as e:
                    return inp.query_id, _failure(inp.query_id, e)

        tasks = [_run(inp) for inp in inputs]
        for coro in asyncio.as_completed(tasks):
            query_id, entry = await coro
            results[query_id] = entry

    return results
