from pathlib import Path


def log_cache_hit(path: Path) -> None:
    """Log that a cached artifact is being reused instead of regenerated."""
    print(f"Cache hit: {path}")
