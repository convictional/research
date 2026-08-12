import asyncio
import json
import logging
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Any
from decimal import Decimal

import asyncpg
from pydantic import BaseModel

from ..settings import Settings

logger = logging.getLogger(__name__)


class DatabaseConfig(BaseModel):
    """PostgreSQL database configuration."""
    host: str = "127.0.0.1"
    port: int = 5432
    user: str = "postgres"
    password: str = ""
    database: str = "slack_study"


class DatabaseManager:
    """Manages PostgreSQL connection and operations for the Slack study experiment."""

    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.pool: Optional[asyncpg.Pool] = None

    async def connect(self) -> None:
        """Initialize connection pool."""
        try:
            self.pool = await asyncpg.create_pool(
                host=self.config.host,
                port=self.config.port,
                user=self.config.user,
                password=self.config.password,
                database=self.config.database,
                min_size=2,
                max_size=10
            )
            logger.info(f"Connected to PostgreSQL database: {self.config.database}")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise

    async def disconnect(self) -> None:
        """Close connection pool."""
        if self.pool:
            await self.pool.close()
            logger.info("Database connection pool closed")

    async def execute_script(self, script_path: Path) -> None:
        """Execute SQL script from file."""
        if not self.pool:
            await self.connect()

        script_content = script_path.read_text()

        async with self.pool.acquire() as conn:
            try:
                await conn.execute(script_content)
                logger.info(f"Successfully executed script: {script_path}")
            except Exception as e:
                logger.error(f"Error executing script {script_path}: {e}")
                raise

    async def query(self, sql: str, *args) -> List[Dict[str, Any]]:
        """Execute query and return results as list of dictionaries."""
        if not self.pool:
            await self.connect()

        async with self.pool.acquire() as conn:
            try:
                records = await conn.fetch(sql, *args)
                return [dict(record) for record in records]
            except Exception as e:
                logger.error(f"Query error: {e}")
                logger.error(f"SQL: {sql}")
                raise

    async def execute(self, sql: str, *args) -> str:
        """Execute command and return status."""
        if not self.pool:
            await self.connect()

        async with self.pool.acquire() as conn:
            try:
                result = await conn.execute(sql, *args)
                return result
            except Exception as e:
                logger.error(f"Execute error: {e}")
                logger.error(f"SQL: {sql}")
                raise

    async def upsert_company(self, ticker: str, company_data: Dict[str, Any]) -> None:
        """Insert or update company record."""
        sql = """
        INSERT INTO companies (ticker, company_name, sector, industry, market_cap, cik, exchange, sp500_member)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        ON CONFLICT (ticker) DO UPDATE SET
            company_name = EXCLUDED.company_name,
            sector = EXCLUDED.sector,
            industry = EXCLUDED.industry,
            market_cap = EXCLUDED.market_cap,
            cik = EXCLUDED.cik,
            exchange = EXCLUDED.exchange,
            sp500_member = EXCLUDED.sp500_member,
            updated_at = CURRENT_TIMESTAMP
        """

        await self.execute(
            sql,
            ticker,
            company_data.get('company_name'),
            company_data.get('sector'),
            company_data.get('industry'),
            company_data.get('market_cap'),
            company_data.get('cik'),
            company_data.get('exchange'),
            company_data.get('sp500_member', False)
        )

    async def upsert_financial_metrics(self, ticker: str, metrics: Dict[str, Any]) -> None:
        """Insert or update financial metrics."""
        sql = """
        INSERT INTO financial_metrics (
            ticker, period_end, period_type, revenue, gross_profit, operating_income,
            net_income, total_assets, total_debt, shareholders_equity, r_and_d_expense,
            sg_and_a_expense, employees, free_cash_flow
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
        ON CONFLICT (ticker, period_end, period_type) DO UPDATE SET
            revenue = EXCLUDED.revenue,
            gross_profit = EXCLUDED.gross_profit,
            operating_income = EXCLUDED.operating_income,
            net_income = EXCLUDED.net_income,
            total_assets = EXCLUDED.total_assets,
            total_debt = EXCLUDED.total_debt,
            shareholders_equity = EXCLUDED.shareholders_equity,
            r_and_d_expense = EXCLUDED.r_and_d_expense,
            sg_and_a_expense = EXCLUDED.sg_and_a_expense,
            employees = EXCLUDED.employees,
            free_cash_flow = EXCLUDED.free_cash_flow,
            updated_at = CURRENT_TIMESTAMP
        """

        await self.execute(
            sql,
            ticker,
            metrics.get('period_end'),
            metrics.get('period_type'),
            metrics.get('revenue'),
            metrics.get('gross_profit'),
            metrics.get('operating_income'),
            metrics.get('net_income'),
            metrics.get('total_assets'),
            metrics.get('total_debt'),
            metrics.get('shareholders_equity'),
            metrics.get('r_and_d_expense'),
            metrics.get('sg_and_a_expense'),
            metrics.get('employees'),
            metrics.get('free_cash_flow')
        )

    async def upsert_stock_prices(self, ticker: str, price_data: List[Dict[str, Any]]) -> None:
        """Batch insert/update stock price data."""
        if not price_data:
            return

        sql = """
        INSERT INTO stock_prices (ticker, date, open, high, low, close, adjusted_close, volume)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        ON CONFLICT (ticker, date) DO UPDATE SET
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close,
            adjusted_close = EXCLUDED.adjusted_close,
            volume = EXCLUDED.volume
        """

        async with self.pool.acquire() as conn:
            async with conn.transaction():
                for price in price_data:
                    await conn.execute(
                        sql,
                        ticker,
                        price['date'],
                        price.get('open'),
                        price.get('high'),
                        price.get('low'),
                        price.get('close'),
                        price.get('adjusted_close'),
                        price.get('volume')
                    )

    async def get_cached_financial_data(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
        period_type: str = 'quarterly'
    ) -> List[Dict[str, Any]]:
        """Get cached financial data for date range."""
        sql = """
        SELECT * FROM financial_metrics
        WHERE ticker = $1
        AND period_type = $2
        AND period_end >= $3
        AND period_end <= $4
        ORDER BY period_end
        """

        return await self.query(sql, ticker, period_type, start_date, end_date)

    async def get_cached_stock_prices(
        self,
        ticker: str,
        start_date: date,
        end_date: date
    ) -> List[Dict[str, Any]]:
        """Get cached stock price data for date range."""
        sql = """
        SELECT * FROM stock_prices
        WHERE ticker = $1
        AND date >= $2
        AND date <= $3
        ORDER BY date
        """

        return await self.query(sql, ticker, start_date, end_date)

    async def get_missing_periods(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
        period_type: str = 'quarterly'
    ) -> List[tuple[date, date]]:
        """Identify missing periods in cached data."""
        cached_data = await self.get_cached_financial_data(ticker, start_date, end_date, period_type)

        if not cached_data:
            return [(start_date, end_date)]

        # Find gaps in cached data
        cached_dates = sorted([record['period_end'] for record in cached_data])
        missing_periods = []

        # Check if we need data before first cached date
        if cached_dates[0] > start_date:
            missing_periods.append((start_date, cached_dates[0] - timedelta(days=1)))

        # Check gaps between cached dates
        for i in range(len(cached_dates) - 1):
            current_date = cached_dates[i]
            next_date = cached_dates[i + 1]

            # If gap is more than expected period length, we have missing data
            expected_gap = timedelta(days=90) if period_type == 'quarterly' else timedelta(days=365)
            if next_date - current_date > expected_gap:
                gap_start = current_date + timedelta(days=1)
                gap_end = next_date - timedelta(days=1)
                missing_periods.append((gap_start, gap_end))

        # Check if we need data after last cached date
        if cached_dates[-1] < end_date:
            missing_periods.append((cached_dates[-1] + timedelta(days=1), end_date))

        return missing_periods

    async def get_sp500_companies(self) -> List[Dict[str, Any]]:
        """Get all S&P 500 companies from cache."""
        sql = "SELECT * FROM companies WHERE sp500_member = TRUE ORDER BY ticker"
        return await self.query(sql)

    async def store_adoption_signal(self, signal_data: Dict[str, Any]) -> None:
        """Store adoption signal in database."""
        sql = """
        INSERT INTO adoption_signals (
            company_id, ticker, source, signal_date, signal_strength,
            raw_count, total_count, details
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        """

        await self.execute(
            sql,
            signal_data['company_id'],
            signal_data.get('ticker'),
            signal_data['source'],
            signal_data['signal_date'],
            signal_data['signal_strength'],
            signal_data.get('raw_count'),
            signal_data.get('total_count'),
            signal_data.get('details')
        )

    async def store_adoption_event(self, event_data: Dict[str, Any]) -> None:
        """Store determined adoption event."""
        # Check if event already exists for this company
        existing = await self.query(
            "SELECT id FROM adoption_events WHERE company_id = $1",
            event_data['company_id']
        )

        if existing:
            # Update existing event
            sql = """
            UPDATE adoption_events SET
                ticker = $2,
                adoption_date = $3,
                confidence = $4,
                primary_signal = $5,
                corroborating_signals = $6,
                adoption_intensity = $7,
                details = $8
            WHERE company_id = $1
            """
        else:
            # Insert new event
            sql = """
            INSERT INTO adoption_events (
                company_id, ticker, adoption_date, confidence, primary_signal,
                corroborating_signals, adoption_intensity, details
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """

        await self.execute(
            sql,
            event_data['company_id'],
            event_data.get('ticker'),
            event_data['adoption_date'],
            event_data['confidence'],
            event_data['primary_signal'],
            event_data.get('corroborating_signals', []),
            event_data.get('adoption_intensity'),
            json.dumps(event_data.get('details')) if event_data.get('details') else None
        )


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


async def get_db_manager(settings: Settings) -> DatabaseManager:
    """Get or create global database manager instance."""
    global _db_manager

    if _db_manager is None:
        config = DatabaseConfig()
        _db_manager = DatabaseManager(config)
        await _db_manager.connect()

    return _db_manager
