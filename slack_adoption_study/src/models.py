from datetime import datetime
from decimal import Decimal
from typing import Optional

from pydantic import BaseModel, Field, validator


class Company(BaseModel):
    """Core company identifier linking data sources."""
    gvkey: str = Field(..., description="Compustat Global Company Key")
    permno: Optional[int] = Field(None, description="CRSP Permanent Security Number")
    ticker: Optional[str] = Field(None, description="Stock ticker symbol")
    company_name: str = Field(..., description="Company name")
    industry: Optional[str] = Field(None, description="Industry classification")
    sic: Optional[str] = Field(None, description="SIC industry code")
    sector: Optional[str] = Field(None, description="Sector classification")


class AdoptionSignal(BaseModel):
    """Individual adoption signal from a data source."""
    company_id: str = Field(..., description="Company identifier")
    source: str = Field(..., description="Signal source (job_postings, sec_filings, press_releases)")
    signal_date: datetime = Field(..., description="Date of signal observation")
    signal_strength: float = Field(..., ge=0.0, le=1.0, description="Signal strength (0-1)")
    raw_count: Optional[int] = Field(None, description="Raw count (e.g., job postings mentioning Slack)")
    total_count: Optional[int] = Field(None, description="Total count for normalization")
    details: Optional[dict] = Field(None, description="Additional signal-specific details")

    @validator('source')
    def validate_source(cls, v):
        valid_sources = {'job_postings', 'sec_filings', 'press_releases', 'tech_stacks', 'website_trackers'}
        if v not in valid_sources:
            raise ValueError(f'Source must be one of: {valid_sources}')
        return v


class AdoptionEvent(BaseModel):
    """Determined adoption event combining multiple signals."""
    company_id: str = Field(..., description="Company identifier")
    adoption_date: datetime = Field(..., description="Determined adoption date")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in adoption date")
    primary_signal: str = Field(..., description="Primary signal source")
    corroborating_signals: list[str] = Field(default_factory=list, description="Supporting signal sources")
    adoption_intensity: Optional[float] = Field(None, description="Intensity of adoption (0-1)")
    details: dict = Field(default_factory=dict, description="Additional adoption details")


class JobPosting(BaseModel):
    """Job posting data from Lightcast."""
    company_name: str = Field(..., description="Employer name")
    posting_date: datetime = Field(..., description="Job posting date")
    title: str = Field(..., description="Job title")
    skills: list[str] = Field(default_factory=list, description="Required skills")
    location: Optional[str] = Field(None, description="Job location")
    remote_type: Optional[str] = Field(None, description="Remote work type")
    mentions_slack: bool = Field(False, description="Whether posting mentions Slack")
    mentions_teams: bool = Field(False, description="Whether posting mentions Teams")


class SECFiling(BaseModel):
    """SEC filing text extraction."""
    company_id: str = Field(..., description="Company identifier")
    filing_type: str = Field(..., description="Filing type (10-K, 10-Q, 8-K)")
    filing_date: datetime = Field(..., description="Filing date")
    period_end: Optional[datetime] = Field(None, description="Period end date")
    mentions_slack: bool = Field(False, description="Whether filing mentions Slack")
    first_mention_context: Optional[str] = Field(None, description="Context around first Slack mention")
    mention_count: int = Field(0, description="Number of Slack mentions")


class FinancialOutcome(BaseModel):
    """Financial performance metrics from Compustat."""
    company_id: str = Field(..., description="Company identifier (gvkey)")
    fiscal_quarter: str = Field(..., description="Fiscal quarter (YYYYQ format)")
    fiscal_year: int = Field(..., description="Fiscal year")

    # Core metrics
    revenue: Optional[Decimal] = Field(None, description="Total revenue (SALE)")
    employees: Optional[int] = Field(None, description="Number of employees (EMP)")
    revenue_per_employee: Optional[Decimal] = Field(None, description="Revenue per employee")

    # Profitability metrics
    gross_profit: Optional[Decimal] = Field(None, description="Gross profit (GP)")
    operating_income: Optional[Decimal] = Field(None, description="Operating income (OIADP)")
    sga_expense: Optional[Decimal] = Field(None, description="SG&A expenses")
    gross_margin: Optional[float] = Field(None, description="Gross margin (GP/SALE)")
    operating_margin: Optional[float] = Field(None, description="Operating margin (OIADP/SALE)")
    sga_efficiency: Optional[float] = Field(None, description="SG&A efficiency (1 - SGA/SALE)")

    # Growth metrics
    revenue_growth: Optional[float] = Field(None, description="YoY revenue growth")

    # Innovation metrics
    rd_expense: Optional[Decimal] = Field(None, description="R&D expense (XRD)")
    rd_intensity: Optional[float] = Field(None, description="R&D intensity (XRD/SALE)")


class MarketOutcome(BaseModel):
    """Stock market performance metrics from CRSP."""
    company_id: str = Field(..., description="Company identifier")
    permno: int = Field(..., description="CRSP permanent number")
    date: datetime = Field(..., description="Trading date")
    return_1d: Optional[float] = Field(None, description="1-day return")
    return_5d: Optional[float] = Field(None, description="5-day return")
    return_10d: Optional[float] = Field(None, description="10-day return")
    abnormal_return_1d: Optional[float] = Field(None, description="1-day abnormal return")
    abnormal_return_5d: Optional[float] = Field(None, description="5-day abnormal return")
    abnormal_return_10d: Optional[float] = Field(None, description="10-day abnormal return")
    trading_volume: Optional[int] = Field(None, description="Trading volume")


class PatentOutcome(BaseModel):
    """Innovation metrics from USPTO PatentsView."""
    company_id: str = Field(..., description="Company identifier")
    year: int = Field(..., description="Patent year")
    patents_granted: int = Field(0, description="Number of patents granted")
    patents_applied: int = Field(0, description="Number of patents applied for")
    patents_per_rd: Optional[float] = Field(None, description="Patents per R&D dollar")
    citation_weighted_patents: Optional[float] = Field(None, description="Citation-weighted patent count")


class EmploymentOutcome(BaseModel):
    """Employment and hiring dynamics."""
    company_id: str = Field(..., description="Company identifier")
    period: str = Field(..., description="Time period (YYYY-MM)")
    total_postings: int = Field(0, description="Total job postings")
    remote_postings: int = Field(0, description="Remote job postings")
    hybrid_postings: int = Field(0, description="Hybrid job postings")
    remote_share: Optional[float] = Field(None, description="Share of remote postings")
    collaboration_tools_postings: int = Field(0, description="Postings mentioning collaboration tools")


class AnalysisResult(BaseModel):
    """Results from causal inference analysis."""
    analysis_type: str = Field(..., description="Analysis method (staggered_did, synthetic_control, event_study)")
    outcome_variable: str = Field(..., description="Outcome variable analyzed")
    treatment_effect: Optional[float] = Field(None, description="Estimated treatment effect")
    standard_error: Optional[float] = Field(None, description="Standard error")
    p_value: Optional[float] = Field(None, description="P-value")
    confidence_interval: Optional[tuple[float, float]] = Field(None, description="95% confidence interval")
    sample_size: int = Field(..., description="Analysis sample size")
    pre_trend_test: Optional[dict] = Field(None, description="Pre-trend test results")
    robustness_checks: dict = Field(default_factory=dict, description="Robustness check results")
    details: dict = Field(default_factory=dict, description="Additional analysis details")


class ExperimentConfig(BaseModel):
    """Experiment configuration parameters."""
    russell_universe: str = Field("russell_3000", description="Russell index universe")
    study_period_start: datetime = Field(..., description="Study period start date")
    study_period_end: datetime = Field(..., description="Study period end date")

    # Adoption signal thresholds
    job_posting_threshold: float = Field(0.01, description="Minimum share of postings mentioning Slack")
    persistence_months: int = Field(6, description="Months of persistence required")
    corroboration_window_months: int = Field(6, description="Window for signal corroboration")

    # Analysis parameters
    pre_treatment_periods: int = Field(8, description="Pre-treatment periods for analysis")
    post_treatment_periods: int = Field(12, description="Post-treatment periods for analysis")

    # Data source flags
    use_job_postings: bool = Field(True, description="Use job postings data")
    use_sec_filings: bool = Field(True, description="Use SEC filings data")
    use_press_releases: bool = Field(True, description="Use press releases data")
    use_tech_stacks: bool = Field(False, description="Use tech stack data")
