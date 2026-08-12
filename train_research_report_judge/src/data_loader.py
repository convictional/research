import random
from collections import Counter
from pathlib import Path

import pandas as pd

from src.models import RatedReport
from src.settings import settings, logger


COLUMN_MAP = {
    "id": "id",
    "topics_research_question": "question",
    "topics_research_question_community_relation": "community_relation",
    "topics_research_type": "variant",
    "research_output": "research_output",
    "quality_score": "quality_score",
}


def load_csv(path: Path | None = None) -> list[RatedReport]:
    path = path or settings.raw_data_path
    logger.info(f"Loading data from {path}")

    df = pd.read_csv(path)
    required_cols = list(COLUMN_MAP.keys())
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    reports = []
    for _, row in df.iterrows():
        reports.append(
            RatedReport(
                id=str(row["id"]),
                question=str(row["topics_research_question"]),
                community_relation=str(row["topics_research_question_community_relation"]),
                variant=str(row["topics_research_type"]),
                research_output=str(row["research_output"]),
                quality_score=int(float(row["quality_score"])),
            )
        )

    logger.info(f"Loaded {len(reports)} reports")
    return reports


def stratified_split(
    reports: list[RatedReport],
    train_ratio: float = settings.train_ratio,
    dev_ratio: float = settings.dev_ratio,
    seed: int = 42,
) -> tuple[list[RatedReport], list[RatedReport], list[RatedReport]]:
    random.seed(seed)

    # Group by quality_score x variant for stratification
    groups: dict[str, list[RatedReport]] = {}
    for report in reports:
        key = f"{report.quality_score}_{report.variant}"
        groups.setdefault(key, []).append(report)

    train, dev, test = [], [], []
    for key in sorted(groups.keys()):
        group = groups[key]
        random.shuffle(group)

        n = len(group)
        n_train = max(1, round(n * train_ratio))
        n_dev = max(1, round(n * dev_ratio))

        # Ensure we don't exceed group size
        if n_train + n_dev >= n:
            n_train = max(1, n - 2)
            n_dev = max(1, n - n_train - 1)

        train.extend(group[:n_train])
        dev.extend(group[n_train : n_train + n_dev])
        test.extend(group[n_train + n_dev :])

    random.shuffle(train)
    random.shuffle(dev)
    random.shuffle(test)

    return train, dev, test


def save_split(reports: list[RatedReport], name: str, path: Path | None = None) -> Path:
    path = path or settings.data_path
    file_path = path / f"{name}.csv"
    df = pd.DataFrame([r.model_dump() for r in reports])
    df.to_csv(file_path, index=False)
    logger.info(f"Saved {len(reports)} reports to {file_path}")
    return file_path


def load_split(name: str, path: Path | None = None) -> list[RatedReport]:
    path = path or settings.data_path
    file_path = path / f"{name}.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Split file not found: {file_path}. Run load_data first.")

    df = pd.read_csv(file_path)
    return [RatedReport(**row) for _, row in df.iterrows()]


def print_summary(reports: list[RatedReport], label: str = "All") -> None:
    print(f"\n{'='*60}")
    print(f"  {label}: {len(reports)} reports")
    print(f"{'='*60}")

    score_dist = Counter(r.quality_score for r in reports)
    print(f"\n  Quality score distribution:")
    for score in sorted(score_dist.keys()):
        count = score_dist[score]
        bar = "#" * count
        print(f"    {score}: {count:>4} {bar}")

    variant_dist = Counter(r.variant for r in reports)
    print(f"\n  Variant distribution:")
    for variant in sorted(variant_dist.keys()):
        print(f"    {variant}: {variant_dist[variant]}")

    community_dist = Counter(r.community_relation for r in reports)
    print(f"\n  Community relation:")
    for relation in sorted(community_dist.keys()):
        print(f"    {relation}: {community_dist[relation]}")

    lengths = [len(r.research_output) for r in reports]
    print(f"\n  Report length (chars): min={min(lengths)}, max={max(lengths)}, "
          f"mean={sum(lengths) // len(lengths)}")


def load_and_split() -> tuple[list[RatedReport], list[RatedReport], list[RatedReport]]:
    reports = load_csv()
    train, dev, test = stratified_split(reports)

    save_split(train, "train")
    save_split(dev, "dev")
    save_split(test, "test")

    print_summary(reports, "Full dataset")
    print_summary(train, "Train")
    print_summary(dev, "Dev")
    print_summary(test, "Test")

    return train, dev, test
