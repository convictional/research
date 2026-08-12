import logging
import sys
from pathlib import Path
from typing import List

from ..settings import Settings
from ..utils.database import get_db_manager, DatabaseConfig
from ..data_collection.company_universe import fetch_sp500_companies

logger = logging.getLogger(__name__)


async def init_database(settings: Settings) -> None:
    """Initialize the PostgreSQL database schema."""
    logger.info("Initializing database schema")

    try:
        db = await get_db_manager(settings)

        # Execute the schema SQL file
        schema_path = Path(__file__).parent.parent.parent / "schema.sql"
        await db.execute_script(schema_path)

        logger.info("Database schema initialized successfully")

    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise


async def populate_universe(settings: Settings) -> None:
    """Populate the companies table with S&P 500 data."""
    logger.info("Populating company universe")

    try:
        tickers = await fetch_sp500_companies(settings)
        logger.info(f"Successfully populated {len(tickers)} companies")

    except Exception as e:
        logger.error(f"Error populating universe: {e}")
        raise


async def backfill_financials(settings: Settings, tickers: List[str] = None, limit: int = 10) -> None:
    """
    Progressively download historical financial data for companies.

    Args:
        settings: Experiment configuration
        tickers: List of ticker symbols (if None, get from database)
        limit: Maximum number of companies to process
    """
    logger.info(f"Starting financial data backfill (limit: {limit})")

    try:
        db = await get_db_manager(settings)

        # Get tickers to process
        if not tickers:
            companies = await db.get_sp500_companies()
            tickers = [company['ticker'] for company in companies[:limit]]
        else:
            tickers = tickers[:limit]

        logger.info(f"Processing {len(tickers)} companies: {tickers}")

        # Import the financial collection function
        from ..outcomes.financial import collect_outcomes

        successful = 0
        failed = 0

        for i, ticker in enumerate(tickers, 1):
            try:
                logger.info(f"[{i}/{len(tickers)}] Processing {ticker}")

                # Collect financial data (will cache automatically)
                outcomes = await collect_outcomes(settings, [ticker])

                logger.info(f"  → Collected {len(outcomes)} quarters of data for {ticker}")
                successful += 1

            except Exception as e:
                logger.error(f"  → Error processing {ticker}: {e}")
                failed += 1
                continue

        logger.info(f"Backfill complete: {successful} successful, {failed} failed")

    except Exception as e:
        logger.error(f"Error in financial backfill: {e}")
        raise


async def cache_status(settings: Settings) -> None:
    """Show current cache status."""
    logger.info("Checking cache status")

    try:
        db = await get_db_manager(settings)

        # Count companies
        companies = await db.query("SELECT COUNT(*) as count FROM companies")
        company_count = companies[0]['count'] if companies else 0

        # Count financial records
        financials = await db.query("SELECT COUNT(*) as count FROM financial_metrics")
        financial_count = financials[0]['count'] if financials else 0

        # Count stock prices
        prices = await db.query("SELECT COUNT(*) as count FROM stock_prices")
        price_count = prices[0]['count'] if prices else 0

        # Count signals
        signals = await db.query("SELECT COUNT(*) as count FROM adoption_signals")
        signal_count = signals[0]['count'] if signals else 0

        # Count events
        events = await db.query("SELECT COUNT(*) as count FROM adoption_events")
        event_count = events[0]['count'] if events else 0

        # Financial data by ticker
        financial_by_ticker = await db.query("""
            SELECT ticker, COUNT(*) as quarters
            FROM financial_metrics
            GROUP BY ticker
            ORDER BY quarters DESC
            LIMIT 10
        """)

        print("\n" + "="*50)
        print("SLACK STUDY CACHE STATUS")
        print("="*50)
        print(f"Companies:         {company_count:,}")
        print(f"Financial records: {financial_count:,}")
        print(f"Stock prices:      {price_count:,}")
        print(f"Adoption signals:  {signal_count:,}")
        print(f"Adoption events:   {event_count:,}")

        if financial_by_ticker:
            print("\nTop companies by financial data:")
            for row in financial_by_ticker:
                print(f"  {row['ticker']}: {row['quarters']} quarters")

        print("="*50 + "\n")

    except Exception as e:
        logger.error(f"Error checking cache status: {e}")
        raise


async def clear_cache(settings: Settings, confirm: bool = False) -> None:
    """Clear all cached data."""
    if not confirm:
        print("This will delete ALL cached data. Use --confirm to proceed.")
        return

    logger.info("Clearing cache")

    try:
        db = await get_db_manager(settings)

        # Clear all tables
        tables = [
            'analysis_results',
            'adoption_events',
            'adoption_signals',
            'stock_prices',
            'financial_metrics',
            'companies'
        ]

        for table in tables:
            result = await db.execute(f"DELETE FROM {table}")
            logger.info(f"Cleared {table}: {result}")

        logger.info("Cache cleared successfully")

    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise


async def run_management_command(command: str, settings: Settings, **kwargs) -> None:
    """
    Run a data management command.

    Args:
        command: Command to run (init_db, populate_universe, backfill_financials, cache_status, clear_cache)
        settings: Experiment configuration
        **kwargs: Additional arguments for specific commands
    """
    commands = {
        'init_db': init_database,
        'populate_universe': populate_universe,
        'backfill_financials': lambda s: backfill_financials(s, **kwargs),
        'cache_status': cache_status,
        'clear_cache': lambda s: clear_cache(s, **kwargs)
    }

    if command not in commands:
        logger.error(f"Unknown command: {command}")
        logger.info(f"Available commands: {list(commands.keys())}")
        sys.exit(1)

    try:
        await commands[command](settings)
    except KeyboardInterrupt:
        logger.info("Command interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Command failed: {e}")
        sys.exit(1)
