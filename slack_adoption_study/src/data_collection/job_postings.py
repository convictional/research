import asyncio
import logging
from datetime import datetime, timedelta
from typing import AsyncGenerator

import httpx
from pydantic import BaseModel

from ..models import JobPosting, AdoptionSignal
from ..settings import Settings

logger = logging.getLogger(__name__)


class LightcastAuth(BaseModel):
    """Lightcast API authentication response."""
    access_token: str
    token_type: str
    expires_in: int


class LightcastClient:
    """Client for Lightcast Job Postings Analytics API."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.base_url = settings.lightcast_api_base
        self.rate_limit = settings.lightcast_rate_limit
        self.access_token = None
        self.token_expires = None

    async def authenticate(self) -> None:
        """Authenticate with Lightcast API to get access token."""
        if not self.settings.lightcast_client_id or not self.settings.lightcast_client_secret:
            raise ValueError("Lightcast credentials not configured")

        auth_url = f"{self.base_url}/auth"
        auth_data = {
            "client_id": self.settings.lightcast_client_id,
            "client_secret": self.settings.lightcast_client_secret,
            "grant_type": "client_credentials",
            "scope": "emsi_open"
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(auth_url, data=auth_data)
            response.raise_for_status()

            auth = LightcastAuth(**response.json())
            self.access_token = auth.access_token
            self.token_expires = datetime.now() + timedelta(seconds=auth.expires_in - 300)

        logger.info("Successfully authenticated with Lightcast API")

    async def _ensure_authenticated(self) -> None:
        """Ensure we have a valid access token."""
        if not self.access_token or datetime.now() >= self.token_expires:
            await self.authenticate()

    async def search_postings(
        self,
        company_name: str,
        start_date: datetime,
        end_date: datetime,
        skills: list[str] = None
    ) -> AsyncGenerator[JobPosting, None]:
        """
        Search for job postings by company and date range.

        Args:
            company_name: Company name to search for
            start_date: Start date for search
            end_date: End date for search
            skills: Optional list of skills to filter by

        Yields:
            JobPosting objects matching the criteria
        """
        await self._ensure_authenticated()

        search_url = f"{self.base_url}/postings"

        # Build query parameters
        params = {
            "q": f'company_name:"{company_name}"',
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "format": "json",
            "size": 1000  # Maximum batch size
        }

        if skills:
            skills_query = " OR ".join([f'skills:"{skill}"' for skill in skills])
            params["q"] += f" AND ({skills_query})"

        headers = {"Authorization": f"Bearer {self.access_token}"}

        async with httpx.AsyncClient(timeout=30.0) as client:
            offset = 0

            while True:
                params["from"] = offset

                # Rate limiting
                await asyncio.sleep(self.rate_limit)

                try:
                    response = await client.get(search_url, params=params, headers=headers)
                    response.raise_for_status()

                    data = response.json()
                    postings = data.get("data", [])

                    if not postings:
                        break

                    for posting_data in postings:
                        try:
                            posting = self._parse_posting(posting_data)
                            yield posting
                        except Exception as e:
                            logger.warning(f"Failed to parse posting: {e}")
                            continue

                    offset += len(postings)

                    # Check if we've reached the end
                    if len(postings) < params["size"]:
                        break

                except httpx.HTTPStatusError as e:
                    logger.error(f"HTTP error searching postings: {e}")
                    break
                except Exception as e:
                    logger.error(f"Error searching postings: {e}")
                    break

    def _parse_posting(self, data: dict) -> JobPosting:
        """Parse raw posting data into JobPosting model."""
        skills = data.get("skills", [])
        if isinstance(skills, str):
            skills = [s.strip() for s in skills.split(",")]
        elif isinstance(skills, list):
            skills = [str(s).strip() for s in skills]
        else:
            skills = []

        # Check for Slack and Teams mentions
        skills_text = " ".join(skills).lower()
        title_text = data.get("title", "").lower()
        description_text = data.get("description", "").lower()

        all_text = f"{skills_text} {title_text} {description_text}"

        mentions_slack = any(term in all_text for term in ["slack", "slack technologies", "slack channel"])
        mentions_teams = any(term in all_text for term in ["teams", "microsoft teams", "ms teams"])

        return JobPosting(
            company_name=data.get("company_name", ""),
            posting_date=datetime.fromisoformat(data.get("posting_date", "")),
            title=data.get("title", ""),
            skills=skills,
            location=data.get("location", ""),
            remote_type=data.get("remote_type", ""),
            mentions_slack=mentions_slack,
            mentions_teams=mentions_teams
        )


async def collect_signals(settings: Settings, company_ids: list[str]) -> list[AdoptionSignal]:
    """
    Collect Slack adoption signals from job postings data.

    Args:
        settings: Experiment configuration
        company_ids: List of company identifiers

    Returns:
        List of adoption signals from job postings
    """
    logger.info(f"Collecting job posting signals for {len(company_ids)} companies")

    if not settings.lightcast_client_id:
        logger.warning("Lightcast credentials not configured, skipping job postings collection")
        return []

    client = LightcastClient(settings)
    signals = []

    # TODO: Need to map company_ids (gvkeys) to company names for Lightcast search
    # This would typically involve querying Compustat for company names

    # For each company, collect monthly posting counts
    for company_id in company_ids:
        try:
            # TODO: Get company name from company_id
            company_name = f"Company_{company_id}"  # Placeholder

            logger.info(f"Processing {company_name}")

            # Collect data month by month to build time series
            current_date = settings.study_period_start

            while current_date < settings.study_period_end:
                month_end = min(
                    current_date.replace(day=28) + timedelta(days=4),
                    settings.study_period_end
                )

                try:
                    # Count total postings and Slack-mentioning postings
                    total_postings = 0
                    slack_postings = 0

                    async for posting in client.search_postings(
                        company_name=company_name,
                        start_date=current_date,
                        end_date=month_end
                    ):
                        total_postings += 1
                        if posting.mentions_slack:
                            slack_postings += 1

                    # Create adoption signal if we have enough data
                    if total_postings >= settings.min_monthly_postings:
                        signal_strength = slack_postings / total_postings

                        if signal_strength >= settings.job_posting_slack_threshold:
                            signal = AdoptionSignal(
                                company_id=company_id,
                                source="job_postings",
                                signal_date=current_date,
                                signal_strength=signal_strength,
                                raw_count=slack_postings,
                                total_count=total_postings,
                                details={
                                    "company_name": company_name,
                                    "month": current_date.strftime("%Y-%m")
                                }
                            )
                            signals.append(signal)

                            logger.info(
                                f"{company_name} {current_date.strftime('%Y-%m')}: "
                                f"{slack_postings}/{total_postings} = {signal_strength:.3f}"
                            )

                except Exception as e:
                    logger.warning(f"Error processing {company_name} {current_date}: {e}")

                # Move to next month
                if current_date.month == 12:
                    current_date = current_date.replace(year=current_date.year + 1, month=1)
                else:
                    current_date = current_date.replace(month=current_date.month + 1)

        except Exception as e:
            logger.error(f"Error processing company {company_id}: {e}")
            continue

    logger.info(f"Collected {len(signals)} job posting signals")
    return signals
