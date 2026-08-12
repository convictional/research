-- PostgreSQL schema for Slack adoption study experiment
-- Database: slack_study

-- Companies table - S&P 500 universe with metadata
CREATE TABLE IF NOT EXISTS companies (
    ticker VARCHAR(10) PRIMARY KEY,
    company_name VARCHAR(255) NOT NULL,
    sector VARCHAR(100),
    industry VARCHAR(150),
    market_cap BIGINT,
    cik VARCHAR(20),
    exchange VARCHAR(10),
    sp500_member BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Financial metrics from yfinance
CREATE TABLE IF NOT EXISTS financial_metrics (
    ticker VARCHAR(10),
    period_end DATE,
    period_type VARCHAR(20), -- 'quarterly' or 'annual'
    revenue DECIMAL(20,2),
    gross_profit DECIMAL(20,2),
    operating_income DECIMAL(20,2),
    net_income DECIMAL(20,2),
    total_assets DECIMAL(20,2),
    total_debt DECIMAL(20,2),
    shareholders_equity DECIMAL(20,2),
    r_and_d_expense DECIMAL(20,2),
    sg_and_a_expense DECIMAL(20,2),
    employees INTEGER,
    free_cash_flow DECIMAL(20,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ticker, period_end, period_type),
    FOREIGN KEY (ticker) REFERENCES companies(ticker) ON DELETE CASCADE
);

-- Stock price data from yfinance
CREATE TABLE IF NOT EXISTS stock_prices (
    ticker VARCHAR(10),
    date DATE,
    open DECIMAL(12,4),
    high DECIMAL(12,4),
    low DECIMAL(12,4),
    close DECIMAL(12,4),
    adjusted_close DECIMAL(12,4),
    volume BIGINT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ticker, date),
    FOREIGN KEY (ticker) REFERENCES companies(ticker) ON DELETE CASCADE
);

-- Adoption signals from multiple sources
CREATE TABLE IF NOT EXISTS adoption_signals (
    id SERIAL PRIMARY KEY,
    company_id VARCHAR(50) NOT NULL, -- Can be ticker or placeholder ID
    ticker VARCHAR(10), -- Mapped ticker if available
    source VARCHAR(50) NOT NULL, -- job_postings, sec_filings, press_releases, etc.
    signal_date DATE NOT NULL,
    signal_strength DECIMAL(5,4) NOT NULL CHECK (signal_strength >= 0 AND signal_strength <= 1),
    raw_count INTEGER,
    total_count INTEGER,
    details JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (ticker) REFERENCES companies(ticker) ON DELETE SET NULL
);

-- Determined adoption events (ensemble results)
CREATE TABLE IF NOT EXISTS adoption_events (
    id SERIAL PRIMARY KEY,
    company_id VARCHAR(50) NOT NULL,
    ticker VARCHAR(10), -- Mapped ticker if available
    adoption_date DATE NOT NULL,
    confidence DECIMAL(5,4) NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    primary_signal VARCHAR(50) NOT NULL,
    corroborating_signals TEXT[], -- Array of signal sources
    adoption_intensity DECIMAL(5,4),
    details JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (ticker) REFERENCES companies(ticker) ON DELETE SET NULL
);

-- Analysis results cache
CREATE TABLE IF NOT EXISTS analysis_results (
    id SERIAL PRIMARY KEY,
    analysis_type VARCHAR(50) NOT NULL,
    outcome_variable VARCHAR(100) NOT NULL,
    treatment_effect DECIMAL(12,6),
    standard_error DECIMAL(12,6),
    p_value DECIMAL(8,6),
    confidence_interval_lower DECIMAL(12,6),
    confidence_interval_upper DECIMAL(12,6),
    sample_size INTEGER,
    details JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_companies_sector ON companies(sector);
CREATE INDEX IF NOT EXISTS idx_companies_sp500 ON companies(sp500_member);
CREATE INDEX IF NOT EXISTS idx_financial_ticker_date ON financial_metrics(ticker, period_end);
CREATE INDEX IF NOT EXISTS idx_financial_period_type ON financial_metrics(period_type);
CREATE INDEX IF NOT EXISTS idx_stock_prices_ticker_date ON stock_prices(ticker, date);
CREATE INDEX IF NOT EXISTS idx_adoption_signals_company_source ON adoption_signals(company_id, source);
CREATE INDEX IF NOT EXISTS idx_adoption_signals_ticker ON adoption_signals(ticker);
CREATE INDEX IF NOT EXISTS idx_adoption_signals_date ON adoption_signals(signal_date);
CREATE INDEX IF NOT EXISTS idx_adoption_events_ticker ON adoption_events(ticker);
CREATE INDEX IF NOT EXISTS idx_adoption_events_date ON adoption_events(adoption_date);
CREATE INDEX IF NOT EXISTS idx_analysis_results_type ON analysis_results(analysis_type, outcome_variable);

-- Create triggers to update timestamps
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_companies_updated_at
    BEFORE UPDATE ON companies
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER update_financial_metrics_updated_at
    BEFORE UPDATE ON financial_metrics
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();
