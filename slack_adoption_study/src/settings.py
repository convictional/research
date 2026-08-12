from datetime import datetime
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Configuration settings for the Slack adoption study experiment."""

    # API Keys
    anthropic_api_key: str
    openai_api_key: str
    lightcast_client_id: str | None = None
    lightcast_client_secret: str | None = None

    # Google Cloud
    gcp_project: str = ""  # set via GCP_PROJECT in the environment
    bigquery_dataset: str = "experiments"

    # Data Sources
    compustat_dataset: str = "compustat"
    crsp_dataset: str = "crsp"
    patents_dataset: str = "patents"

    # Experiment Parameters
    russell_universe: str = "russell_3000"
    study_period_start: datetime = datetime(2015, 1, 1)
    study_period_end: datetime = datetime(2025, 12, 31)

    # Adoption Signal Thresholds
    job_posting_slack_threshold: float = 0.01  # 1% of postings must mention Slack
    job_posting_persistence_months: int = 6    # Must persist for 6 months
    min_monthly_postings: int = 3              # Minimum postings per month
    corroboration_window_months: int = 6       # Window for signal corroboration ±6 months

    # Analysis Windows
    pre_treatment_quarters: int = 8   # 2 years pre-treatment
    post_treatment_quarters: int = 12 # 3 years post-treatment
    event_study_days: list[int] = [-10, -5, -3, -1, 0, 1, 3, 5, 10]  # Event study windows

    # Data Collection Flags
    collect_job_postings: bool = True
    collect_sec_filings: bool = True
    collect_press_releases: bool = True
    collect_tech_stacks: bool = False
    collect_website_trackers: bool = False

    # SEC EDGAR
    sec_user_agent: str = "Academic Research researcher@example.com"  # SEC requires a real contact address; set your own
    sec_rate_limit: float = 0.1  # 10 requests per second max

    # Job Postings (Lightcast)
    lightcast_api_base: str = "https://emsiservices.com/jpa"
    lightcast_rate_limit: float = 1.0  # 1 request per second

    # Analysis Parameters
    cluster_se_two_way: bool = True  # Two-way clustering (firm and time)
    winsorize_outcomes: bool = True
    winsorize_percentile: float = 0.01  # Winsorize at 1st and 99th percentiles
    min_pre_periods: int = 4  # Minimum pre-treatment periods required

    # Teams Competition Analysis
    teams_unbundling_date: datetime = datetime(2024, 4, 1)
    salesforce_acquisition_date: datetime = datetime(2021, 7, 21)

    # File Paths
    data_dir: Path = Path("data")
    output_dir: Path = Path("output")
    cache_dir: Path = Path(".cache")

    # Logging
    log_level: str = "INFO"
    log_to_file: bool = True

    class Config:
        env_file = ".env.secrets"
        case_sensitive = False
        extra = "ignore"  # Ignore extra environment variables

    def model_post_init(self, __context) -> None:
        """Create directories if they don't exist."""
        self.data_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)
