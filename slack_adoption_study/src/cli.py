import asyncio
import logging
import sys
from pathlib import Path

from .settings import Settings
from .utils.management import run_management_command

logger = logging.getLogger(__name__)


async def main():
    """CLI interface for Slack study data management commands."""

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    if len(sys.argv) < 2:
        print("Usage: python -m slack_adoption_study.src.cli <command> [options]")
        print("\nAvailable commands:")
        print("  init_db              - Initialize database schema")
        print("  populate_universe    - Fetch and cache S&P 500 companies")
        print("  backfill_financials  - Download historical financial data")
        print("  cache_status         - Show current cache status")
        print("  clear_cache --confirm - Clear all cached data")
        print("\nExamples:")
        print("  python -m slack_adoption_study.src.cli init_db")
        print("  python -m slack_adoption_study.src.cli backfill_financials")
        print("  python -m slack_adoption_study.src.cli cache_status")
        sys.exit(1)

    command = sys.argv[1]

    # Parse additional arguments
    kwargs = {}

    if '--confirm' in sys.argv:
        kwargs['confirm'] = True

    if '--limit' in sys.argv:
        try:
            limit_idx = sys.argv.index('--limit')
            kwargs['limit'] = int(sys.argv[limit_idx + 1])
        except (ValueError, IndexError):
            print("Error: --limit requires a number")
            sys.exit(1)

    # Load settings
    try:
        settings = Settings()
    except Exception as e:
        logger.error(f"Error loading settings: {e}")
        sys.exit(1)

    # Run the command
    await run_management_command(command, settings, **kwargs)


if __name__ == "__main__":
    asyncio.run(main())
