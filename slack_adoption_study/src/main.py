import logging
from datetime import datetime

from .settings import Settings
from .models import ExperimentConfig

logger = logging.getLogger(__name__)


async def collect_company_universe(settings: Settings) -> list[str]:
    """
    Build the universe of public companies for analysis.

    Returns list of ticker symbols from S&P 500.
    """
    logger.info(f"Building {settings.russell_universe} company universe")

    from .data_collection.company_universe import fetch_sp500_companies

    try:
        # Get S&P 500 companies (using as proxy for Russell 3000)
        tickers = await fetch_sp500_companies(settings)
        logger.info(f"Found {len(tickers)} S&P 500 companies")

        # For initial testing, limit to a smaller subset
        if len(tickers) > 50:
            tickers = tickers[:50]
            logger.info(f"Limited to first {len(tickers)} companies for testing")

        return tickers

    except Exception as e:
        logger.error(f"Error fetching company universe: {e}")

        # Fallback to a few well-known tickers
        fallback_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        logger.info(f"Using fallback ticker list: {fallback_tickers}")
        return fallback_tickers


async def collect_adoption_signals(settings: Settings, company_ids: list[str]) -> list:
    """
    Collect adoption signals from multiple data sources.

    Args:
        settings: Experiment configuration
        company_ids: List of company identifiers to collect data for

    Returns:
        List of all adoption signals
    """
    logger.info(f"Collecting adoption signals for {len(company_ids)} companies")

    all_signals = []

    if settings.collect_job_postings:
        logger.info("Collecting job postings signals...")
        from .data_collection import job_postings
        job_signals = await job_postings.collect_signals(settings, company_ids)
        all_signals.extend(job_signals)

    if settings.collect_sec_filings:
        logger.info("Collecting SEC filings signals...")
        from .data_collection import sec_filings
        sec_signals = await sec_filings.collect_signals(settings, company_ids)
        all_signals.extend(sec_signals)

    if settings.collect_press_releases:
        logger.info("Collecting press releases signals...")
        from .data_collection import press_releases
        press_signals = await press_releases.collect_signals(settings, company_ids)
        all_signals.extend(press_signals)

    if settings.collect_tech_stacks:
        logger.info("Collecting tech stack signals...")
        # TODO: Implement tech_stacks.collect_signals()

    logger.info(f"Adoption signals collection completed: {len(all_signals)} total signals")
    return all_signals


async def determine_adoption_events(settings: Settings, all_signals: list) -> list:
    """
    Process raw signals into adoption event determinations.

    Args:
        settings: Experiment configuration
        all_signals: List of adoption signals from all sources

    Returns:
        List of determined adoption events
    """
    logger.info("Determining adoption events from signals")

    from .data_processing import adoption_signals
    adoption_events = await adoption_signals.determine_events(settings, all_signals)

    logger.info(f"Adoption event determination completed: {len(adoption_events)} events")
    return adoption_events


async def collect_outcome_data(settings: Settings, company_ids: list[str]) -> dict:
    """
    Collect outcome variables for analysis.

    Args:
        settings: Experiment configuration
        company_ids: List of company identifiers

    Returns:
        Dictionary of outcome data by type
    """
    logger.info("Collecting outcome data")

    outcomes = {}

    # Financial outcomes (Compustat)
    logger.info("Collecting financial outcomes from Compustat...")
    from .outcomes import financial
    outcomes['financial'] = await financial.collect_outcomes(settings, company_ids)

    # Market outcomes (CRSP)
    logger.info("Collecting market outcomes from CRSP...")
    # TODO: Implement market.collect_outcomes()
    outcomes['market'] = []

    # Innovation outcomes (USPTO)
    logger.info("Collecting innovation outcomes from USPTO...")
    # TODO: Implement innovation.collect_outcomes()
    outcomes['innovation'] = []

    # Employment outcomes (job postings)
    logger.info("Collecting employment outcomes...")
    # TODO: Implement employment.collect_outcomes()
    outcomes['employment'] = []

    logger.info("Outcome data collection completed")
    return outcomes


async def run_causal_analysis(settings: Settings, adoption_events: list, outcomes: dict) -> list:
    """
    Execute causal inference analyses.

    Args:
        settings: Experiment configuration
        adoption_events: Determined adoption events
        outcomes: Outcome data dictionary

    Returns:
        List of analysis results
    """
    logger.info("Running causal inference analyses")

    all_results = []

    # Staggered Difference-in-Differences
    logger.info("Running staggered DiD analysis...")
    from .analysis import staggered_did
    did_results = await staggered_did.run_analysis(settings, adoption_events, outcomes['financial'])
    all_results.extend(did_results)

    # Synthetic Control (for major cases)
    logger.info("Running synthetic control analysis...")
    # TODO: Implement synthetic_control.run_analysis()

    # Event Studies (for market reactions)
    logger.info("Running event study analysis...")
    # TODO: Implement event_study.run_analysis()

    # Heterogeneity Analysis
    logger.info("Running heterogeneity analysis...")
    # TODO: Implement heterogeneity.run_analysis()

    logger.info(f"Causal analysis completed: {len(all_results)} results")
    return all_results


async def generate_results(settings: Settings, analysis_results: list) -> None:
    """
    Generate final results, tables, and visualizations.

    Args:
        settings: Experiment configuration
        analysis_results: Analysis results from causal inference
    """
    logger.info(f"Generating results and visualizations for {len(analysis_results)} analyses")

    # TODO: Implement results generation
    # This includes:
    # 1. Summary statistics tables
    # 2. Event study plots
    # 3. Treatment effect estimates
    # 4. Robustness check results
    # 5. Heterogeneity analysis plots

    # Placeholder: Save results summary
    for result in analysis_results:
        logger.info(
            f"Result: {result.analysis_type} on {result.outcome_variable} - "
            f"Effect: {result.treatment_effect:.4f} (p={result.p_value:.3f})"
        )

    logger.info("Results generation completed")


async def main() -> None:
    """
    Main experiment execution flow.

    Orchestrates the complete Slack adoption study:
    1. Data collection (adoption signals + outcomes)
    2. Signal processing and adoption event determination
    3. Causal inference analysis
    4. Results generation and visualization
    """
    logger.info("=== Starting Slack Adoption Study Experiment ===")
    start_time = datetime.now()

    # Load configuration
    settings = Settings()
    logger.info(f"Study period: {settings.study_period_start} to {settings.study_period_end}")
    logger.info(f"Target universe: {settings.russell_universe}")

    try:
        # Initialize database if needed
        logger.info("Checking database initialization...")
        from .utils.management import init_database
        await init_database(settings)

        logger.info("Database ready")

    except Exception as e:
        logger.warning(f"Database initialization issue (continuing anyway): {e}")

    try:
        # Phase 1: Build company universe
        logger.info("PHASE 1: Building company universe")
        company_ids = await collect_company_universe(settings)
        logger.info(f"Universe contains {len(company_ids)} companies")

        if not company_ids:
            logger.warning("No companies found in universe. Exiting.")
            return

        # Phase 2: Collect adoption signals
        logger.info("PHASE 2: Collecting adoption signals")
        all_signals = await collect_adoption_signals(settings, company_ids)

        # Phase 3: Determine adoption events
        logger.info("PHASE 3: Determining adoption events")
        adoption_events = await determine_adoption_events(settings, all_signals)

        # Phase 4: Collect outcome data
        logger.info("PHASE 4: Collecting outcome data")
        outcomes = await collect_outcome_data(settings, company_ids)

        # Phase 5: Run causal analysis
        logger.info("PHASE 5: Running causal inference analysis")
        analysis_results = await run_causal_analysis(settings, adoption_events, outcomes)

        # Phase 6: Generate results
        logger.info("PHASE 6: Generating results")
        await generate_results(settings, analysis_results)

        # Summary
        duration = datetime.now() - start_time
        logger.info(f"=== Experiment completed successfully in {duration} ===")
        logger.info(f"Results saved to: {settings.output_dir}")

    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        raise
