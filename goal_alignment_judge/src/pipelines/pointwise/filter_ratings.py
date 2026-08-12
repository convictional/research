from pathlib import Path

import pandas as pd

from ...settings import logger


def _pair_key(df: pd.DataFrame) -> set[tuple[str, str]]:
    """Extract the set of (goal_id, content_source_url) pairs from a ratings CSV."""
    return set(
        zip(df["goal_id"].astype(str), df["content_source_url"].astype(str))
    )


def _filter_df(df: pd.DataFrame, pairs: set[tuple[str, str]]) -> pd.DataFrame:
    """Keep only rows whose (goal_id, content_source_url) is in pairs."""
    mask = df.apply(
        lambda row: (str(row["goal_id"]), str(row["content_source_url"])) in pairs,
        axis=1,
    )
    return df[mask]


def filter_to_common_pairs(
    csv_a: Path,
    csv_b: Path,
    output_a: Path,
    output_b: Path,
) -> tuple[Path, Path]:
    """Intersect two ratings CSVs on (goal_id, content_source_url) and write
    filtered versions of both, so they contain exactly the same item set.

    Args:
        csv_a: First ratings CSV (e.g., Adam's ratings).
        csv_b: Second ratings CSV (e.g., production ratings).
        output_a: Where to write the filtered version of csv_a.
        output_b: Where to write the filtered version of csv_b.

    Returns:
        Tuple of (output_a, output_b) paths.
    """
    df_a = pd.read_csv(csv_a)
    df_b = pd.read_csv(csv_b)

    pairs_a = _pair_key(df_a)
    pairs_b = _pair_key(df_b)
    common = pairs_a & pairs_b

    filtered_a = _filter_df(df_a, common)
    filtered_b = _filter_df(df_b, common)

    for p in (output_a, output_b):
        p.parent.mkdir(parents=True, exist_ok=True)

    filtered_a.to_csv(output_a, index=False)
    filtered_b.to_csv(output_b, index=False)

    only_a = len(pairs_a - common)
    only_b = len(pairs_b - common)
    logger.info(
        f"Common pairs: {len(common)} | "
        f"Only in {csv_a.name}: {only_a} | "
        f"Only in {csv_b.name}: {only_b}"
    )
    logger.info(f"  {csv_a.name}: {len(df_a)} → {len(filtered_a)} rows → {output_a}")
    logger.info(f"  {csv_b.name}: {len(df_b)} → {len(filtered_b)} rows → {output_b}")

    return output_a, output_b
