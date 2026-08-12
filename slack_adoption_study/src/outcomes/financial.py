import logging
import asyncio
from decimal import Decimal
from datetime import datetime, date
from typing import List, Dict, Any

import yfinance as yf
import pandas as pd

from ..models import FinancialOutcome
from ..settings import Settings
from ..utils.database import get_db_manager

logger = logging.getLogger(__name__)


async def collect_outcomes(settings: Settings, company_ids: List[str]) -> List[FinancialOutcome]:
    """
    Collect financial outcome metrics using yfinance with PostgreSQL caching.

    Args:
        settings: Experiment configuration
        company_ids: List of ticker symbols

    Returns:
        List of financial outcomes
    """
    logger.info(f"Collecting financial outcomes for {len(company_ids)} companies using yfinance")

    if not company_ids:
        logger.warning("No company IDs provided")
        return []

    outcomes = []

    for ticker in company_ids:
        try:
            logger.info(f"Processing financial data for {ticker}")

            # Get cached and fresh data
            ticker_outcomes = await _get_financial_data_for_ticker(
                ticker, settings.study_period_start.date(),
                settings.study_period_end.date(), settings
            )

            outcomes.extend(ticker_outcomes)

            # Rate limiting to avoid overwhelming yfinance
            await asyncio.sleep(0.5)

        except Exception as e:
            logger.error(f"Error collecting financial data for {ticker}: {e}")
            continue

    logger.info(f"Collected {len(outcomes)} financial outcomes")
    return outcomes


async def _get_financial_data_for_ticker(
    ticker: str,
    start_date: date,
    end_date: date,
    settings: Settings
) -> List[FinancialOutcome]:
    """
    Get financial data for a single ticker with caching.

    Args:
        ticker: Stock ticker symbol
        start_date: Start date for data
        end_date: End date for data
        settings: Experiment configuration

    Returns:
        List of financial outcomes for this ticker
    """
    db = await get_db_manager(settings)

    # Check what data we already have cached (both quarterly and annual)
    quarterly_data = await db.get_cached_financial_data(ticker, start_date, end_date, 'quarterly')
    annual_data = await db.get_cached_financial_data(ticker, start_date, end_date, 'annual')
    cached_data = quarterly_data + annual_data

    # Determine what we need to fetch from yfinance
    missing_periods = await db.get_missing_periods(ticker, start_date, end_date, 'quarterly')

    # Fetch missing data from yfinance
    if missing_periods:
        logger.info(f"Fetching missing financial data for {ticker}: {len(missing_periods)} periods")

        for period_start, period_end in missing_periods:
            try:
                await _fetch_and_cache_yfinance_data(ticker, period_start, period_end, db)
            except Exception as e:
                logger.error(f"Error fetching yfinance data for {ticker} {period_start}-{period_end}: {e}")
                continue

        # Re-fetch cached data to include newly downloaded data (both quarterly and annual)
        quarterly_data = await db.get_cached_financial_data(ticker, start_date, end_date, 'quarterly')
        annual_data = await db.get_cached_financial_data(ticker, start_date, end_date, 'annual')
        cached_data = quarterly_data + annual_data

    # Convert cached data to FinancialOutcome objects
    outcomes = []
    for record in cached_data:
        try:
            # Calculate derived metrics
            revenue = record.get('revenue')
            employees = record.get('employees')
            revenue_per_employee = None
            if revenue and employees and employees > 0:
                revenue_per_employee = Decimal(str(revenue)) / employees

            # Calculate margins
            gross_margin = None
            operating_margin = None
            if revenue and revenue > 0:
                if record.get('gross_profit'):
                    gross_margin = float(record['gross_profit']) / float(revenue)
                if record.get('operating_income'):
                    operating_margin = float(record['operating_income']) / float(revenue)

            # R&D intensity
            rd_intensity = None
            if revenue and revenue > 0 and record.get('r_and_d_expense'):
                rd_intensity = float(record['r_and_d_expense']) / float(revenue)

            # Create fiscal quarter string
            period_end = record['period_end']
            fiscal_quarter = f"{period_end.year}Q{(period_end.month-1)//3 + 1}"

            outcome = FinancialOutcome(
                company_id=ticker,
                fiscal_quarter=fiscal_quarter,
                fiscal_year=period_end.year,
                revenue=record.get('revenue'),
                employees=record.get('employees'),
                revenue_per_employee=revenue_per_employee,
                gross_profit=record.get('gross_profit'),
                operating_income=record.get('operating_income'),
                sga_expense=record.get('sg_and_a_expense'),
                gross_margin=gross_margin,
                operating_margin=operating_margin,
                sga_efficiency=None,  # TODO: Calculate if SG&A available
                revenue_growth=None,  # TODO: Calculate YoY growth
                rd_expense=record.get('r_and_d_expense'),
                rd_intensity=rd_intensity
            )

            outcomes.append(outcome)

        except Exception as e:
            logger.warning(f"Error processing financial record for {ticker}: {e}")
            continue

    return outcomes


async def _fetch_and_cache_yfinance_data(
    ticker: str,
    start_date: date,
    end_date: date,
    db
) -> None:
    """
    Fetch extended historical financial data from yfinance and cache in PostgreSQL.

    Args:
        ticker: Stock ticker symbol
        start_date: Start date for data
        end_date: End date for data
        db: Database manager instance
    """
    try:
        logger.info(f"Fetching extended yfinance data for {ticker} from {start_date} to {end_date}")

        # Create yfinance Ticker object
        stock = yf.Ticker(ticker)

        # Get company info for employee count and other metadata
        try:
            info = stock.info
        except Exception as e:
            logger.warning(f"Could not fetch info for {ticker}: {e}")
            info = {}

        # Get both quarterly and annual financial statements for maximum historical coverage
        financial_data = {}

        try:
            # Quarterly data (most recent ~4-5 quarters typically)
            financial_data['quarterly'] = {
                'financials': stock.quarterly_financials,
                'balance_sheet': stock.quarterly_balance_sheet,
                'cashflow': stock.quarterly_cashflow
            }
        except Exception as e:
            logger.warning(f"Error fetching quarterly data for {ticker}: {e}")
            financial_data['quarterly'] = {'financials': pd.DataFrame(), 'balance_sheet': pd.DataFrame(), 'cashflow': pd.DataFrame()}

        try:
            # Annual data (goes back further, typically 4-5 years)
            financial_data['annual'] = {
                'financials': stock.financials,
                'balance_sheet': stock.balance_sheet,
                'cashflow': stock.cashflow
            }
        except Exception as e:
            logger.warning(f"Error fetching annual data for {ticker}: {e}")
            financial_data['annual'] = {'financials': pd.DataFrame(), 'balance_sheet': pd.DataFrame(), 'cashflow': pd.DataFrame()}

        # Process all available periods (quarterly first, then annual for older periods)
        all_periods = []

        # Add quarterly periods
        quarterly_financials = financial_data['quarterly']['financials']
        if not quarterly_financials.empty:
            for quarter_date in quarterly_financials.columns:
                all_periods.append({
                    'date': quarter_date,
                    'type': 'quarterly',
                    'financials': quarterly_financials,
                    'balance_sheet': financial_data['quarterly']['balance_sheet'],
                    'cashflow': financial_data['quarterly']['cashflow']
                })

        # Add annual periods (but skip years we already have quarterly data for)
        annual_financials = financial_data['annual']['financials']
        if not annual_financials.empty:
            quarterly_years = set()
            if not quarterly_financials.empty:
                quarterly_years = {pd.to_datetime(d).year for d in quarterly_financials.columns}

            for annual_date in annual_financials.columns:
                annual_year = pd.to_datetime(annual_date).year
                # Only add annual data if we don't have quarterly data for that year
                if annual_year not in quarterly_years:
                    all_periods.append({
                        'date': annual_date,
                        'type': 'annual',
                        'financials': annual_financials,
                        'balance_sheet': financial_data['annual']['balance_sheet'],
                        'cashflow': financial_data['annual']['cashflow']
                    })

        if not all_periods:
            logger.warning(f"No financial data available for {ticker}")
            return

        logger.info(f"Processing {len(all_periods)} periods for {ticker} (quarterly + annual)")

        # Process each period
        for period in all_periods:
            try:
                period_date = period['date']

                # Convert to date
                if hasattr(period_date, 'date'):
                    period_end = period_date.date()
                else:
                    period_end = pd.to_datetime(period_date).date()

                # Check if this period is in our date range
                if not (start_date <= period_end <= end_date):
                    continue

                financials_df = period['financials']
                balance_sheet_df = period['balance_sheet']
                cashflow_df = period['cashflow']

                # Extract financial metrics
                metrics = {
                    'period_end': period_end,
                    'period_type': period['type'],
                    'revenue': _safe_get_value(financials_df, 'Total Revenue', period_date),
                    'gross_profit': _safe_get_value(financials_df, 'Gross Profit', period_date),
                    'operating_income': _safe_get_value(financials_df, 'Operating Income', period_date),
                    'net_income': _safe_get_value(financials_df, 'Net Income', period_date),
                    'r_and_d_expense': _safe_get_value(financials_df, 'Research And Development', period_date),
                    'sg_and_a_expense': _safe_get_value(financials_df, 'Selling General And Administration', period_date),
                }

                # Balance sheet data
                if not balance_sheet_df.empty and period_date in balance_sheet_df.columns:
                    metrics.update({
                        'total_assets': _safe_get_value(balance_sheet_df, 'Total Assets', period_date),
                        'total_debt': _safe_get_value(balance_sheet_df, 'Total Debt', period_date),
                        'shareholders_equity': _safe_get_value(balance_sheet_df, 'Stockholders Equity', period_date),
                    })

                # Cash flow data
                if not cashflow_df.empty and period_date in cashflow_df.columns:
                    metrics.update({
                        'free_cash_flow': _safe_get_value(cashflow_df, 'Free Cash Flow', period_date),
                    })

                # Employee count (usually only available in info, not time series)
                if 'fullTimeEmployees' in info:
                    metrics['employees'] = info['fullTimeEmployees']

                # Store in database
                await db.upsert_financial_metrics(ticker, metrics)
                logger.debug(f"Cached {period['type']} data for {ticker} {period_end}")

            except Exception as e:
                logger.warning(f"Error processing period {period_date} for {ticker}: {e}")
                continue

        logger.info(f"Successfully cached extended financial data for {ticker}")

    except Exception as e:
        logger.error(f"Error fetching extended yfinance data for {ticker}: {e}")
        raise


def _safe_get_value(df: pd.DataFrame, metric_name: str, date_col) -> float | None:
    """
    Safely extract a value from a pandas DataFrame.

    Args:
        df: DataFrame to extract from
        metric_name: Row name to look for
        date_col: Column (date) to extract

    Returns:
        Float value or None if not found/invalid
    """
    try:
        if metric_name in df.index and date_col in df.columns:
            value = df.loc[metric_name, date_col]
            if pd.notna(value) and value != 0:
                return float(value)
    except Exception:
        pass
    return None
