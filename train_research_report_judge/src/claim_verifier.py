import asyncio

from common.instruct_llm import ainstruct_llm
from common.prompt_template_engine import build_prompt
from src.content_search import hybrid_search
from src.models import ClaimDetail, ClaimVerification, ClaimVerificationRollup
from src.settings import settings

MAX_EVIDENCE_CHARS = 4000


def _build_evidence_context(results: list) -> list[dict[str, str]]:
    context = []
    total_chars = 0
    for r in results:
        content = r.content[:MAX_EVIDENCE_CHARS - total_chars] if total_chars < MAX_EVIDENCE_CHARS else ""
        if not content:
            break
        context.append({"title": r.title, "content": content})
        total_chars += len(content)
    return context


async def _verify_single_claim(claim: ClaimDetail, question: str) -> ClaimVerification:
    search_results = await hybrid_search(claim.claim)
    evidence_context = _build_evidence_context(search_results)

    system_prompt = build_prompt("claim_verification_system.txt.jinja")
    user_prompt = build_prompt(
        "claim_verification_user.txt.jinja",
        question=question,
        claim=claim.claim,
        search_results=evidence_context,
    )

    return await ainstruct_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_model=ClaimVerification,
        llm_model=settings.haiku_model,
        temperature=settings.temperature,
        max_tokens=1024,
    )


def _aggregate_verifications(verifications: list[ClaimVerification]) -> ClaimVerificationRollup:
    supported = sum(1 for v in verifications if v.verdict == "supported")
    partially_supported = sum(1 for v in verifications if v.verdict == "partially_supported")
    unsupported = sum(1 for v in verifications if v.verdict == "unsupported")
    no_evidence = sum(1 for v in verifications if v.verdict == "no_evidence_found")
    avg_confidence = sum(v.confidence for v in verifications) / len(verifications) if verifications else 0.0

    return ClaimVerificationRollup(
        total_claims=len(verifications),
        supported=supported,
        partially_supported=partially_supported,
        unsupported=unsupported,
        no_evidence_found=no_evidence,
        avg_confidence=round(avg_confidence, 3),
        details=verifications,
    )


async def verify_claims(claims: list[ClaimDetail], question: str) -> ClaimVerificationRollup:
    tasks = [_verify_single_claim(claim, question) for claim in claims]
    verifications = await asyncio.gather(*tasks)
    return _aggregate_verifications(verifications)
