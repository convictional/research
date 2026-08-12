import subprocess
from pathlib import Path

from .settings import settings
from .utils.llm import astring_completion
from .utils.tokens import count_tokens

CONTEXT_FILE = settings.root / "codebase_context.md"
LAST_COMMIT_FILE = settings.root / ".last_commit_sha"
MAX_CONTEXT_TOKENS = 4000
BOOTSTRAP_COMMIT_COUNT = 20


async def build_codebase_context() -> str:
    context = _read_context()
    recent_commits = _get_recent_commits()

    if recent_commits:
        context = await _update_context_with_commits(context, recent_commits)
        _write_context(context)
        _save_head_sha()

    if count_tokens(context) > MAX_CONTEXT_TOKENS:
        context = await _condense_context(context)
        _write_context(context)

    return context


def _read_context() -> str:
    if CONTEXT_FILE.exists():
        return CONTEXT_FILE.read_text()
    return ""


def _write_context(context: str) -> None:
    CONTEXT_FILE.write_text(context)


def _get_repo_root() -> Path:
    """Resolve the main repository root, even when running from a worktree."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--git-common-dir"],
            capture_output=True,
            text=True,
            cwd=str(settings.repo_root),
        )
        if result.returncode != 0:
            return settings.repo_root
        git_dir = Path(result.stdout.strip())
        if not git_dir.is_absolute():
            git_dir = settings.repo_root / git_dir
        return git_dir.parent if git_dir.name == ".git" else git_dir
    except Exception:
        return settings.repo_root


def _get_recent_commits() -> str:
    repo_root = _get_repo_root()
    last_sha = _read_last_commit_sha()
    try:
        if last_sha:
            cmd = ["git", "log", f"{last_sha}..HEAD", "--oneline", "--no-merges"]
        else:
            cmd = ["git", "log", f"-{BOOTSTRAP_COMMIT_COUNT}", "--oneline", "--no-merges"]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(repo_root))
        if result.returncode != 0 and last_sha:
            cmd = ["git", "log", f"-{BOOTSTRAP_COMMIT_COUNT}", "--oneline", "--no-merges"]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(repo_root))
        return result.stdout.strip()
    except Exception:
        return ""


def _read_last_commit_sha() -> str:
    if LAST_COMMIT_FILE.exists():
        return LAST_COMMIT_FILE.read_text().strip()
    return ""


def _save_head_sha() -> None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=str(settings.repo_root),
        )
        if result.returncode == 0:
            LAST_COMMIT_FILE.write_text(result.stdout.strip())
    except Exception:
        pass


async def _update_context_with_commits(context: str, commits: str) -> str:
    from .prompts.engine import build_prompt

    system_prompt = build_prompt("codebase_summary.md.jinja")
    user_prompt = f"## Current Context\n\n{context}\n\n## Recent Commits\n\n{commits}"

    updated = await astring_completion(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=4000,
    )
    return updated


async def _condense_context(context: str) -> str:
    from .prompts.engine import build_prompt

    system_prompt = build_prompt("codebase_summary.md.jinja")
    user_prompt = (
        f"The following codebase context has grown too long. "
        f"Condense it to the most important information while preserving key details.\n\n{context}"
    )

    condensed = await astring_completion(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=4000,
    )
    return condensed
