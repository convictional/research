"""Diagnostic script: run RAG scorer on a small sample and dump all intermediate signals."""

import asyncio
import json
import random

from common.instruct_llm import set_async_instructor_client
from common.prompt_template_engine import initialize_and_register_prompt_templates
from pathlib import Path

from src.claim_verifier import verify_claims
from src.content_search import close_search, hybrid_search, init_search
from src.data_loader import load_split
from src.format_scorer import score_format
from src.pointwise_scorer import _analyze_claims
from src.rag_scorer import _final_score
from src.settings import settings


async def main():
    set_async_instructor_client(settings.llm_model, settings.anthropic_api_key)
    initialize_and_register_prompt_templates(Path(__file__).resolve().parent.parent / "src" / "prompts")

    reports = load_split("dev")
    await init_search()

    # Sample one from each expert score level
    by_score: dict[int, list] = {}
    for r in reports:
        by_score.setdefault(r.quality_score, []).append(r)

    sample = []
    for score in sorted(by_score.keys()):
        random.seed(42)
        sample.append(random.choice(by_score[score]))

    try:
        for report in sample:
            print(f"\n{'='*80}")
            print(f"Report {report.id} | Expert score: {report.quality_score}")
            print(f"Question: {report.question[:120]}...")
            print(f"Report length: {len(report.research_output)} chars")

            # Step 1: Claim extraction
            claims = await _analyze_claims(report)
            print(f"\nClaims extracted: {len(claims.claims)} (total in report: {claims.total_claims})")
            print(f"Claim summary: {claims.summary[:200]}")

            # Step 2: Quick search test — does the DB have relevant content?
            test_search = await hybrid_search(report.question[:200])
            print(f"\nSearch test (question as query): {len(test_search)} results")
            for sr in test_search[:3]:
                print(f"  [{sr.score:.3f}] {sr.title[:80]} ({len(sr.content)} chars)")

            # Step 3: Verify claims
            rollup = await verify_claims(claims.claims, report.question)
            print(f"\nClaim verification rollup:")
            print(f"  Supported:           {rollup.supported}/{rollup.total_claims}")
            print(f"  Partially supported: {rollup.partially_supported}/{rollup.total_claims}")
            print(f"  Unsupported:         {rollup.unsupported}/{rollup.total_claims}")
            print(f"  No evidence found:   {rollup.no_evidence_found}/{rollup.total_claims}")
            print(f"  Avg confidence:      {rollup.avg_confidence:.2f}")

            for v in rollup.details:
                print(f"  [{v.verdict:>22}] (conf={v.confidence:.2f}) {v.claim[:100]}")

            # Step 4: Format scoring
            fmt = await score_format(report)
            print(f"\nFormat assessment:")
            print(f"  Structure:    {fmt.structure_score}/3")
            print(f"  Length:       {fmt.length_adequacy}")
            print(f"  Tone:         {fmt.tone_score}/3")
            print(f"  Q-A alignment:{fmt.qa_alignment_score}/3")
            print(f"  Notes:        {fmt.notes[:200]}")

            # Step 5: Final score
            final = await _final_score(report.question, rollup, fmt)
            print(f"\nFinal RAG score: {final.quality_score}/3")
            print(f"Justification: {final.justification[:300]}")
            print(f"\nExpert score: {report.quality_score} | RAG score: {final.quality_score} | Delta: {final.quality_score - report.quality_score}")
    finally:
        await close_search()


if __name__ == "__main__":
    asyncio.run(main())
