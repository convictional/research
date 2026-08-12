"""Step 3: Consolidate stated goals and unstated candidates into a canonical goal set.

Embeddings are used here ONLY for deduplication among goals — not for pre-filtering
which goals reach decisions (that is Phase 2's responsibility). This distinction is
intentional: Phase 2 is where the mapping schemas decide which goals reach decisions.
"""
import uuid
from pathlib import Path

from openai import AsyncOpenAI

from common.embeddings import aembed_query, cosine_similarity
from common.io import dump_to_pickle_file, load_pickle_file

from ..cache_log import log_cache_hit

from ..instruct_helper import with_transient_retry
from ..models import CanonicalGoal, CandidateGoal, StatedGoal, StatedGoalEvidence
from ..settings import logger, settings

CACHE_FILENAME = "step3_canonical_goals.pkl"


def _stated_to_canonical(goal: StatedGoal, evidence: StatedGoalEvidence | None) -> CanonicalGoal:
    support_score = evidence.activity_support_score if evidence else 0.0
    return CanonicalGoal(
        id=str(uuid.uuid4()),
        title=goal.title,
        description=goal.description,
        is_stated=True,
        is_unstated=False,
        origin_stated_goal_ids=[goal.id],
        origin_unstated_candidate_ids=[],
        activity_support_score=support_score,
    )


def _candidate_to_canonical(candidate: CandidateGoal, candidate_index: int) -> CanonicalGoal:
    return CanonicalGoal(
        id=str(uuid.uuid4()),
        title=candidate.title,
        description=candidate.description,
        is_stated=False,
        is_unstated=True,
        origin_stated_goal_ids=[],
        origin_unstated_candidate_ids=[f"candidate_{candidate_index}"],
        activity_support_score=0.5,  # default: supporting evidence present but unvalidated
    )


async def _embed_text(client: AsyncOpenAI, text: str) -> list[float]:
    return await with_transient_retry(
        lambda: aembed_query(client, text, settings.embedding_model, settings.embedding_dim),
        label=settings.embedding_model,
    )


def _is_duplicate(
    candidate_embedding: list[float],
    existing_embeddings: list[list[float]],
    threshold: float,
) -> bool:
    for existing_emb in existing_embeddings:
        if cosine_similarity(candidate_embedding, existing_emb) >= threshold:
            return True
    return False


async def run_step3(
    candidates: list[CandidateGoal],
    stated_goals: list[StatedGoal],
    evidence: list[StatedGoalEvidence],
    output_path: Path,
    load_from_cache: bool = True,
) -> list[CanonicalGoal]:
    """Merge stated goals + unstated candidates into a deduplicated canonical goal set.

    Stable UUID4 IDs are assigned here and persisted in the pkl. They must never
    be regenerated on reruns — downstream phases (mapping, judging) reference them.
    """
    cache_path = output_path / CACHE_FILENAME

    if load_from_cache and cache_path.exists():
        log_cache_hit(cache_path)
        return load_pickle_file(cache_path)

    print(f"  Step 3: consolidating {len(stated_goals)} stated + {len(candidates)} candidates...")

    openai_client = AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value())

    evidence_by_goal_id: dict[str, StatedGoalEvidence] = {e.goal_id: e for e in evidence}

    canonical: list[CanonicalGoal] = []
    canonical_embeddings: list[list[float]] = []

    # First: promote all stated goals into canonical goals
    for goal in stated_goals:
        ev = evidence_by_goal_id.get(goal.id)
        cg = _stated_to_canonical(goal, ev)
        embedding = await _embed_text(openai_client, f"{cg.title} {cg.description}")
        canonical.append(cg)
        canonical_embeddings.append(embedding)

    # Second: add unstated candidates that are not near-duplicates of existing canonical goals
    deduped, skipped = 0, 0
    for idx, candidate in enumerate(candidates):
        candidate_text = f"{candidate.title} {candidate.description}"
        candidate_emb = await _embed_text(openai_client, candidate_text)

        if _is_duplicate(candidate_emb, canonical_embeddings, settings.consolidation_similarity_threshold):
            skipped += 1
            continue

        cg = _candidate_to_canonical(candidate, idx)
        canonical.append(cg)
        canonical_embeddings.append(candidate_emb)
        deduped += 1

    dump_to_pickle_file(canonical, cache_path)
    print(
        f"  Step 3: {len(canonical)} canonical goals "
        f"({len(stated_goals)} stated + {deduped} new unstated, {skipped} duplicates removed) → {cache_path}"
    )
    return canonical
