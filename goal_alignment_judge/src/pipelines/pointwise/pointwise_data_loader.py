import random
from collections import Counter
from pathlib import Path

import pandas as pd

from common.embeddings import cosine_similarity
from common.postgres import query_local_postgres

from .pointwise_models import HumanAction, PointwiseExample
from ...settings import DATABASE, logger, settings

CONTENT_BY_SOURCE_IDS_QUERY = """
SELECT c.id, c.source_id, c.title, c.preview_content, c.index_content, c.content_type, c.embedding
FROM content c WHERE c.source_id = ANY($1)
"""

GOALS_BY_IDS_QUERY = """
SELECT g.id, g.title, g.description, c.embedding
FROM goal g
JOIN content c ON c.source_id = CONCAT('gid://decide/Goal/', g.id::text)
WHERE g.id = ANY($1) AND g.deleted_at IS NULL
"""


def load_pin_dismiss_csv(path: Path | None = None) -> list[dict]:
    path = path or settings.pointwise_input_csv
    logger.info(f"Loading pointwise data from {path}")

    df = pd.read_csv(path)

    rows = []
    for _, row in df.iterrows():
        pinned = pd.notna(row.get("pinned_by_id")) and str(row.get("pinned_by_id", "")).strip() != ""
        deleted = pd.notna(row.get("deleted_at")) and str(row.get("deleted_at", "")).strip() != ""

        if pinned:
            action = HumanAction.PINNED
        elif deleted:
            action = HumanAction.DELETED
        else:
            action = HumanAction.NEUTRAL

        rows.append(
            {
                "goal_id": str(row["goal_id"]),
                "goal_title": str(row.get("goal_title", "")),
                "content_id": str(row["content_id"]),
                "content_source_url": str(row.get("content_source_url", "")),
                "human_action": action,
                "original_signal": str(row.get("signal", "")),
                "original_score": float(row.get("alignment_score", 0)),
                "description": str(row.get("description", "")),
                "organization_id": str(row.get("organization_id", "")),
            }
        )

    action_counts = Counter(r["human_action"] for r in rows)
    logger.info(f"Loaded {len(rows)} rows: {dict(action_counts)}")
    return rows


def _parse_embedding(raw: str | list[float] | None) -> list[float] | None:
    if raw is None:
        return None
    if isinstance(raw, list):
        return raw
    raw = str(raw).strip()
    if not raw or raw == "None":
        return None
    return [float(x) for x in raw.strip("[]").split(",")]


async def enrich_from_db(rows: list[dict]) -> list[PointwiseExample]:
    source_urls = list({r["content_source_url"] for r in rows})
    goal_ids = list({r["goal_id"] for r in rows})

    content_query = CONTENT_BY_SOURCE_IDS_QUERY.replace(
        "$1", f"ARRAY[{','.join(repr(u) for u in source_urls)}]::text[]"
    )
    goals_query = GOALS_BY_IDS_QUERY.replace("$1", f"ARRAY[{','.join(repr(gid) for gid in goal_ids)}]::uuid[]")

    content_rows = await query_local_postgres(content_query, logger, database=DATABASE)
    goal_rows = await query_local_postgres(goals_query, logger, database=DATABASE)

    content_map = {str(r["source_id"]): r for r in content_rows}
    goal_map = {str(r["id"]): r for r in goal_rows}

    logger.info(f"Fetched {len(content_map)} content items and {len(goal_map)} goals from DB")

    examples = []
    skipped = 0
    for row in rows:
        content = content_map.get(row["content_source_url"])
        goal = goal_map.get(row["goal_id"])

        if not content or not goal:
            skipped += 1
            continue

        body = content.get("preview_content") or content.get("index_content") or ""

        examples.append(
            PointwiseExample(
                goal_id=row["goal_id"],
                goal_title=str(goal.get("title", row.get("goal_title", ""))),
                goal_description=str(goal.get("description", "")),
                content_id=str(content.get("id", row["content_id"])),
                content_type=str(content.get("content_type", "")),
                content_title=str(content.get("title", "")),
                content_body=str(body),
                human_action=row["human_action"],
                original_signal=row["original_signal"],
                original_score=row["original_score"],
            )
        )

    if skipped:
        logger.warning(f"Skipped {skipped} rows due to missing content or goal in DB")

    return examples, content_map, goal_map


async def generate_hard_negatives(
    examples: list[PointwiseExample],
    content_by_source_id: dict[str, dict],
    goal_rows: dict[str, dict],
) -> list[PointwiseExample]:
    # Build a reverse map: DB content id -> source_id for dedup
    content_id_to_source = {str(c["id"]): sid for sid, c in content_by_source_id.items()}
    existing_pairs = {(ex.content_id, ex.goal_id) for ex in examples}

    all_source_ids = list(content_by_source_id.keys())
    all_goal_ids = list(goal_rows.keys())

    negatives = []
    rng = random.Random(42)

    for goal_id in all_goal_ids:
        goal = goal_rows[goal_id]
        goal_emb = _parse_embedding(goal.get("embedding"))
        if not goal_emb:
            continue

        candidates = []
        for source_id in all_source_ids:
            content = content_by_source_id[source_id]
            db_content_id = str(content["id"])

            if (db_content_id, goal_id) in existing_pairs:
                continue

            content_emb = _parse_embedding(content.get("embedding"))
            if not content_emb:
                continue

            sim = cosine_similarity(goal_emb, content_emb)
            if settings.negative_similarity_min <= sim <= settings.negative_similarity_max:
                candidates.append((db_content_id, content, sim))

        rng.shuffle(candidates)
        for db_cid, content, sim in candidates[: settings.negatives_per_goal]:
            body = content.get("preview_content") or content.get("index_content") or ""
            negatives.append(
                PointwiseExample(
                    goal_id=goal_id,
                    goal_title=str(goal.get("title", "")),
                    goal_description=str(goal.get("description", "")),
                    content_id=db_cid,
                    content_type=str(content.get("content_type", "")),
                    content_title=str(content.get("title", "")),
                    content_body=str(body),
                    human_action=HumanAction.SYNTHETIC_NEGATIVE,
                    original_signal="",
                    original_score=0.0,
                    similarity_score=round(sim, 4),
                )
            )

    logger.info(f"Generated {len(negatives)} synthetic hard negatives")
    return negatives


def split_pointwise(
    examples: list[PointwiseExample],
    train_ratio: float = settings.train_ratio,
    dev_ratio: float = settings.dev_ratio,
    seed: int = 42,
) -> tuple[list[PointwiseExample], list[PointwiseExample], list[PointwiseExample]]:
    rng = random.Random(seed)

    by_goal_action: dict[tuple[str, str], list[PointwiseExample]] = {}
    for ex in examples:
        key = (ex.goal_id, ex.human_action.value)
        by_goal_action.setdefault(key, []).append(ex)

    train, dev, test = [], [], []
    for key, group in sorted(by_goal_action.items()):
        rng.shuffle(group)
        n = len(group)
        n_train = max(1, round(n * train_ratio))
        n_dev = max(1, round(n * dev_ratio))

        if n_train + n_dev >= n:
            n_train = max(1, n - 1)
            n_dev = n - n_train
            train.extend(group[:n_train])
            dev.extend(group[n_train:])
        else:
            train.extend(group[:n_train])
            dev.extend(group[n_train : n_train + n_dev])
            test.extend(group[n_train + n_dev :])

    rng.shuffle(train)
    rng.shuffle(dev)
    rng.shuffle(test)

    logger.info(f"Pointwise split: {len(train)} train, {len(dev)} dev, {len(test)} test")
    return train, dev, test


def subsample_train(
    train: list[PointwiseExample],
    n: int,
    seed: int = 42,
) -> list[PointwiseExample]:
    """Stratified subsample of training data, preserving action class proportions.

    For each action class, selects round(n * class_proportion) items.
    If n >= len(train), returns train unchanged.
    """
    if n >= len(train):
        return train

    rng = random.Random(seed)

    by_action: dict[str, list[PointwiseExample]] = {}
    for ex in train:
        by_action.setdefault(ex.human_action.value, []).append(ex)

    # Compute per-class counts proportional to original distribution
    total = len(train)
    subsample = []
    remaining = n
    action_keys = sorted(by_action.keys())
    for i, action in enumerate(action_keys):
        items = by_action[action]
        rng.shuffle(items)
        if i == len(action_keys) - 1:
            # Last class gets whatever remains to hit exactly n
            take = remaining
        else:
            take = round(n * len(items) / total)
        take = min(take, len(items))
        subsample.extend(items[:take])
        remaining -= take

    rng.shuffle(subsample)
    logger.info(f"Subsampled train: {len(train)} → {len(subsample)} items")
    return subsample


def save_pointwise_split(examples: list[PointwiseExample], name: str, subdir: str | None = None) -> Path:
    base = settings.pointwise_processed_path / subdir if subdir else settings.pointwise_processed_path
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"pointwise_{name}.csv"
    df = pd.DataFrame([ex.model_dump() for ex in examples])
    df.to_csv(path, index=False)
    logger.info(f"Saved {len(examples)} pointwise examples to {path}")
    return path


def load_pointwise_split(name: str, subdir: str | None = None) -> list[PointwiseExample]:
    base = settings.pointwise_processed_path / subdir if subdir else settings.pointwise_processed_path
    path = base / f"pointwise_{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Pointwise split not found: {path}. Run load_pointwise_data first.")

    df = pd.read_csv(path)
    examples = []
    for _, row in df.iterrows():
        examples.append(
            PointwiseExample(
                goal_id=str(row["goal_id"]),
                goal_title=str(row["goal_title"]),
                goal_description=str(row["goal_description"]),
                content_id=str(row["content_id"]),
                content_type=str(row["content_type"]),
                content_title=str(row["content_title"]),
                content_body=str(row["content_body"]),
                human_action=HumanAction(row["human_action"]),
                original_signal=str(row.get("original_signal", "")),
                original_score=float(row.get("original_score", 0)),
                similarity_score=float(row["similarity_score"]) if pd.notna(row.get("similarity_score")) else None,
            )
        )

    return examples


def print_pointwise_summary(examples: list[PointwiseExample], label: str = "All") -> None:
    print(f"\n{'=' * 60}")
    print(f"  {label}: {len(examples)} examples")
    print(f"{'=' * 60}")

    action_dist = Counter(ex.human_action for ex in examples)
    print(f"\n  Action distribution:")
    for action in HumanAction:
        count = action_dist.get(action, 0)
        pct = count / len(examples) * 100 if examples else 0
        bar = "#" * min(count, 40)
        print(f"    {action.value:>20}: {count:>3} ({pct:5.1f}%) {bar}")

    goals = set(ex.goal_id for ex in examples)
    print(f"\n  Unique goals: {len(goals)}")
    print(f"  Unique content items: {len(set(ex.content_id for ex in examples))}")

    scores = [ex.original_score for ex in examples if ex.original_score > 0]
    if scores:
        print(f"  Original scores: mean={sum(scores) / len(scores):.3f}, min={min(scores):.3f}, max={max(scores):.3f}")



def _rebalance_scarce_classes(
    train: list[PointwiseExample],
    dev: list[PointwiseExample],
    test: list[PointwiseExample],
    scarce_actions: tuple[HumanAction, ...] = (HumanAction.PINNED, HumanAction.DELETED),
    move_to_dev: int = 1,
    move_to_test: int = 1,
) -> tuple[list[PointwiseExample], list[PointwiseExample], list[PointwiseExample]]:
    """Move scarce-class examples from train into dev and test to improve evaluation signal."""
    for action in scarce_actions:
        train_items = [ex for ex in train if ex.human_action == action]
        needed = move_to_dev + move_to_test
        if len(train_items) < needed + 1:
            logger.warning(f"Not enough {action.value} in train to rebalance ({len(train_items)} available)")
            continue

        to_move = train_items[:needed]
        for ex in to_move:
            train.remove(ex)

        dev.extend(to_move[:move_to_dev])
        test.extend(to_move[move_to_dev:])
        logger.info(f"Rebalanced {action.value}: moved {move_to_dev} to dev, {move_to_test} to test")

    return train, dev, test


async def load_pointwise_data(
    input_csv: Path | None = None,
) -> tuple[list[PointwiseExample], list[PointwiseExample], list[PointwiseExample]]:
    subdir = input_csv.stem if input_csv else None
    rows = load_pin_dismiss_csv(path=input_csv)
    examples, content_map, goal_map = await enrich_from_db(rows)

    # Hold out few-shot examples before splitting (match on content_id + goal_id pair)
    fewshot_set = {(cid, gid) for cid, gid in settings.fewshot_pairs}
    fewshot = [ex for ex in examples if (ex.content_id, ex.goal_id) in fewshot_set]
    examples = [ex for ex in examples if (ex.content_id, ex.goal_id) not in fewshot_set]
    if fewshot:
        logger.info(f"Held out {len(fewshot)} few-shot examples from splits")

    print_pointwise_summary(examples, "Full pointwise dataset")

    train, dev, test = split_pointwise(examples)
    train, dev, test = _rebalance_scarce_classes(train, dev, test)

    save_pointwise_split(train, "train", subdir=subdir)
    save_pointwise_split(dev, "dev", subdir=subdir)
    save_pointwise_split(test, "test", subdir=subdir)

    print_pointwise_summary(train, "Train")
    print_pointwise_summary(dev, "Dev")
    print_pointwise_summary(test, "Test")

    return train, dev, test
