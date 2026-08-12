import logging
from typing import List, Dict
import pandas as pd
import requests
from bs4 import BeautifulSoup

from ..settings import Settings
from ..utils.database import get_db_manager

logger = logging.getLogger(__name__)


async def fetch_sp500_companies(settings: Settings) -> List[str]:
    """
    Fetch S&P 500 companies from Wikipedia and cache in PostgreSQL.

    Args:
        settings: Experiment configuration

    Returns:
        List of ticker symbols
    """
    logger.info("Fetching S&P 500 company list")

    db = await get_db_manager(settings)

    # Check if we already have S&P 500 companies cached
    cached_companies = await db.get_sp500_companies()

    if cached_companies:
        logger.info(f"Found {len(cached_companies)} cached S&P 500 companies")
        return [company['ticker'] for company in cached_companies]

    # Fetch fresh data from Wikipedia
    logger.info("Fetching fresh S&P 500 data from Wikipedia")

    try:
        # Get S&P 500 list from Wikipedia
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        # Parse HTML table
        soup = BeautifulSoup(response.content, 'html.parser')
        table = soup.find('table', {'class': 'wikitable sortable'})

        if not table:
            raise ValueError("Could not find S&P 500 table on Wikipedia")

        companies = []
        rows = table.find_all('tr')[1:]  # Skip header row

        for row in rows:
            cols = row.find_all('td')
            if len(cols) >= 4:  # Ensure we have enough columns
                try:
                    ticker = cols[0].get_text().strip()
                    company_name = cols[1].get_text().strip()
                    sector = cols[2].get_text().strip()
                    industry = cols[3].get_text().strip()

                    # Clean up ticker (remove any extra characters)
                    ticker = ticker.replace('.', '-')  # Handle BRK.B -> BRK-B etc.

                    company_data = {
                        'ticker': ticker,
                        'company_name': company_name,
                        'sector': sector,
                        'industry': industry,
                        'sp500_member': True
                    }

                    companies.append(company_data)

                except Exception as e:
                    logger.warning(f"Error parsing company row: {e}")
                    continue

        logger.info(f"Fetched {len(companies)} companies from Wikipedia")

        # Store in database
        for company_data in companies:
            await db.upsert_company(company_data['ticker'], company_data)

        logger.info(f"Cached {len(companies)} S&P 500 companies in database")

        return [company['ticker'] for company in companies]

    except Exception as e:
        logger.error(f"Error fetching S&P 500 companies: {e}")

        # Fallback: return a small set of well-known companies
        fallback_companies = [
            {'ticker': 'AAPL', 'company_name': 'Apple Inc.', 'sector': 'Technology', 'industry': 'Consumer Electronics', 'sp500_member': True},
            {'ticker': 'MSFT', 'company_name': 'Microsoft Corp.', 'sector': 'Technology', 'industry': 'Software', 'sp500_member': True},
            {'ticker': 'GOOGL', 'company_name': 'Alphabet Inc.', 'sector': 'Technology', 'industry': 'Internet Services', 'sp500_member': True},
            {'ticker': 'AMZN', 'company_name': 'Amazon.com Inc.', 'sector': 'Consumer Discretionary', 'industry': 'E-commerce', 'sp500_member': True},
            {'ticker': 'TSLA', 'company_name': 'Tesla Inc.', 'sector': 'Consumer Discretionary', 'industry': 'Electric Vehicles', 'sp500_member': True},
            {'ticker': 'META', 'company_name': 'Meta Platforms Inc.', 'sector': 'Technology', 'industry': 'Social Media', 'sp500_member': True},
            {'ticker': 'NVDA', 'company_name': 'NVIDIA Corp.', 'sector': 'Technology', 'industry': 'Semiconductors', 'sp500_member': True},
            {'ticker': 'JPM', 'company_name': 'JPMorgan Chase & Co.', 'sector': 'Financial Services', 'industry': 'Banking', 'sp500_member': True},
            {'ticker': 'JNJ', 'company_name': 'Johnson & Johnson', 'sector': 'Healthcare', 'industry': 'Pharmaceuticals', 'sp500_member': True},
            {'ticker': 'V', 'company_name': 'Visa Inc.', 'sector': 'Financial Services', 'industry': 'Payment Processing', 'sp500_member': True},
        ]

        logger.info("Using fallback company list")

        # Store fallback companies
        for company_data in fallback_companies:
            await db.upsert_company(company_data['ticker'], company_data)

        return [company['ticker'] for company in fallback_companies]


async def get_company_info(ticker: str, settings: Settings) -> Dict[str, str]:
    """
    Get company information for a ticker.

    Args:
        ticker: Stock ticker symbol
        settings: Experiment configuration

    Returns:
        Dictionary with company information
    """
    db = await get_db_manager(settings)

    companies = await db.query("SELECT * FROM companies WHERE ticker = $1", ticker)

    if companies:
        return companies[0]
    else:
        # Return basic info if not found
        return {
            'ticker': ticker,
            'company_name': f'Unknown Company ({ticker})',
            'sector': 'Unknown',
            'industry': 'Unknown'
        }


async def map_company_name_to_ticker(company_name: str, settings: Settings) -> str | None:
    """
    Map company name to ticker symbol.

    Args:
        company_name: Company name to search for
        settings: Experiment configuration

    Returns:
        Ticker symbol if found, None otherwise
    """
    db = await get_db_manager(settings)

    # Try exact match first
    results = await db.query(
        "SELECT ticker FROM companies WHERE LOWER(company_name) = LOWER($1)",
        company_name
    )

    if results:
        return results[0]['ticker']

    # Try partial match
    results = await db.query(
        "SELECT ticker FROM companies WHERE LOWER(company_name) LIKE LOWER($1)",
        f'%{company_name}%'
    )

    if results:
        return results[0]['ticker']

    return None
