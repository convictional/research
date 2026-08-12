import asyncio
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator, Optional

import httpx
from bs4 import BeautifulSoup
from pydantic import BaseModel

from ..models import SECFiling, AdoptionSignal
from ..settings import Settings
from common.instruct_llm import ainstruct_llm, set_async_instructor_client
from common.prompt_template_engine import initialize_and_register_prompt_templates, build_prompt

logger = logging.getLogger(__name__)


class SECAnalysisResult(BaseModel):
    """Response model for SEC filing Slack mention analysis."""
    mentions_slack: bool
    confidence: float
    reasoning: str


class SECClient:
    """Client for SEC EDGAR database."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.base_url = "https://www.sec.gov"
        self.rate_limit = settings.sec_rate_limit
        self.user_agent = settings.sec_user_agent

    async def search_filings(
        self,
        cik: str,
        form_types: list[str] = None,
        start_date: datetime = None,
        end_date: datetime = None
    ) -> AsyncGenerator[dict, None]:
        """
        Search for SEC filings by CIK and criteria.

        Args:
            cik: Company CIK identifier
            form_types: List of form types to search (e.g., ['10-K', '10-Q', '8-K'])
            start_date: Start date for search
            end_date: End date for search

        Yields:
            Filing metadata dictionaries
        """
        if not form_types:
            form_types = ['10-K', '10-Q', '8-K']

        search_url = f"{self.base_url}/cgi-bin/browse-edgar"

        params = {
            'action': 'getcompany',
            'CIK': cik,
            'type': ','.join(form_types),
            'dateb': end_date.strftime('%Y%m%d') if end_date else '',
            'owner': 'exclude',
            'output': 'xml',
            'count': '100'
        }

        headers = {'User-Agent': self.user_agent}

        async with httpx.AsyncClient(timeout=30.0) as client:
            start = 0

            while True:
                params['start'] = start

                # Rate limiting
                await asyncio.sleep(self.rate_limit)

                try:
                    response = await client.get(search_url, params=params, headers=headers)
                    response.raise_for_status()

                    # Parse XML response
                    soup = BeautifulSoup(response.content, 'xml')
                    entries = soup.find_all('entry')

                    if not entries:
                        break

                    for entry in entries:
                        try:
                            filing_date = datetime.strptime(
                                entry.find('filing-date').text.strip(),
                                '%Y-%m-%d'
                            )

                            # Filter by date range
                            if start_date and filing_date < start_date:
                                continue
                            if end_date and filing_date > end_date:
                                continue

                            filing_data = {
                                'cik': cik,
                                'form_type': entry.find('filing-type').text.strip(),
                                'filing_date': filing_date,
                                'period_end': self._parse_period_end(entry),
                                'documents_url': entry.find('filing-href').text.strip(),
                                'company_name': entry.find('company-name').text.strip() if entry.find('company-name') else ''
                            }

                            yield filing_data

                        except Exception as e:
                            logger.warning(f"Failed to parse filing entry: {e}")
                            continue

                    start += len(entries)

                    # Check if we've reached the end
                    if len(entries) < 100:
                        break

                except httpx.HTTPStatusError as e:
                    logger.error(f"HTTP error searching filings: {e}")
                    break
                except Exception as e:
                    logger.error(f"Error searching filings: {e}")
                    break

    def _parse_period_end(self, entry) -> Optional[datetime]:
        """Parse period end date from filing entry."""
        try:
            period_end = entry.find('period-of-report')
            if period_end:
                return datetime.strptime(period_end.text.strip(), '%Y-%m-%d')
        except:
            pass
        return None

    async def get_filing_text(self, documents_url: str) -> Optional[str]:
        """
        Download and extract text from a SEC filing.

        Args:
            documents_url: URL to filing documents page

        Returns:
            Extracted text content or None if failed
        """
        headers = {'User-Agent': self.user_agent}

        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                # Get documents page
                response = await client.get(documents_url, headers=headers)
                response.raise_for_status()

                # Find the main document link (usually .htm or .txt)
                soup = BeautifulSoup(response.content, 'html.parser')
                doc_links = soup.find_all('a', href=True)

                main_doc_url = None
                for link in doc_links:
                    href = link.get('href')
                    if href and ('.htm' in href or '.txt' in href):
                        if not href.startswith('http'):
                            href = f"{self.base_url}{href}"
                        main_doc_url = href
                        break

                if not main_doc_url:
                    logger.warning(f"No main document found for {documents_url}")
                    return None

                # Rate limiting
                await asyncio.sleep(self.rate_limit)

                # Download main document
                response = await client.get(main_doc_url, headers=headers)
                response.raise_for_status()

                # Extract text content
                if '.htm' in main_doc_url:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    # Remove script and style elements
                    for script in soup(["script", "style"]):
                        script.decompose()
                    text = soup.get_text()
                else:
                    text = response.text

                # Clean up text
                text = re.sub(r'\s+', ' ', text).strip()
                return text

            except Exception as e:
                logger.error(f"Error downloading filing text: {e}")
                return None


async def analyze_slack_mentions(filing_text: str, settings: Settings) -> dict:
    """
    Use LLM to analyze Slack mentions in SEC filing text.

    Args:
        filing_text: Full text of SEC filing
        settings: Experiment configuration

    Returns:
        Analysis results including mention count and context
    """
    # Search for obvious Slack mentions first
    slack_patterns = [
        r'\bSlack\b',
        r'\bSlack Technologies\b',
        r'\bSlack channel\b',
        r'\bSlack integration\b',
        r'\bSlack platform\b'
    ]

    mentions = []
    for pattern in slack_patterns:
        for match in re.finditer(pattern, filing_text, re.IGNORECASE):
            start = max(0, match.start() - 200)
            end = min(len(filing_text), match.end() + 200)
            context = filing_text[start:end].strip()
            mentions.append({
                'pattern': pattern,
                'position': match.start(),
                'context': context
            })

    if not mentions:
        return {
            'mentions_slack': False,
            'mention_count': 0,
            'first_mention_context': None,
            'analysis_confidence': 1.0
        }

    # Use LLM to analyze the mentions for relevance
    try:
        # Set up LLM client
        set_async_instructor_client(
            "claude-3-haiku-20240307",
            settings.anthropic_api_key
        )

        # Build prompts
        system_prompt = build_prompt("sec_extraction/system.txt.jinja")
        user_prompt = build_prompt(
            "sec_extraction/user.txt.jinja",
            filing_text_excerpt=mentions[0]['context'],
            mention_count=len(mentions)
        )

        # Get LLM analysis
        analysis_result = await ainstruct_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=SECAnalysisResult,
            llm_model="claude-3-haiku-20240307"
        )

        return {
            'mentions_slack': analysis_result.mentions_slack,
            'mention_count': len(mentions),
            'first_mention_context': mentions[0]['context'] if mentions else None,
            'analysis_confidence': analysis_result.confidence,
            'analysis_reasoning': analysis_result.reasoning
        }

    except Exception as e:
        logger.warning(f"LLM analysis failed, using pattern matching: {e}")
        return {
            'mentions_slack': True,
            'mention_count': len(mentions),
            'first_mention_context': mentions[0]['context'],
            'analysis_confidence': 0.5,  # Lower confidence for pattern matching only
            'analysis_reasoning': 'LLM analysis failed, using pattern matching'
        }


async def collect_signals(settings: Settings, company_ids: list[str]) -> list[AdoptionSignal]:
    """
    Collect Slack adoption signals from SEC filings.

    Args:
        settings: Experiment configuration
        company_ids: List of company identifiers (gvkeys)

    Returns:
        List of adoption signals from SEC filings
    """
    logger.info(f"Collecting SEC filing signals for {len(company_ids)} companies")

    # Initialize prompt templates
    prompts_path = Path(__file__).parent.parent / "prompts"
    initialize_and_register_prompt_templates(prompts_path)

    client = SECClient(settings)
    signals = []

    # TODO: Need to map gvkeys to CIKs for SEC searches
    # This would typically involve querying a lookup table

    for company_id in company_ids:
        try:
            # TODO: Get CIK from company_id
            cik = f"000{company_id}".zfill(10)  # Placeholder CIK format

            logger.info(f"Processing SEC filings for company {company_id} (CIK: {cik})")

            filing_count = 0
            async for filing_data in client.search_filings(
                cik=cik,
                form_types=['10-K', '10-Q', '8-K'],
                start_date=settings.study_period_start,
                end_date=settings.study_period_end
            ):
                try:
                    filing_count += 1
                    if filing_count % 10 == 0:
                        logger.info(f"Processed {filing_count} filings for {company_id}")

                    # Download filing text
                    filing_text = await client.get_filing_text(filing_data['documents_url'])
                    if not filing_text:
                        continue

                    # Analyze for Slack mentions
                    analysis = await analyze_slack_mentions(filing_text, settings)

                    # Create SEC filing record
                    sec_filing = SECFiling(
                        company_id=company_id,
                        filing_type=filing_data['form_type'],
                        filing_date=filing_data['filing_date'],
                        period_end=filing_data['period_end'],
                        mentions_slack=analysis['mentions_slack'],
                        first_mention_context=analysis['first_mention_context'],
                        mention_count=analysis['mention_count']
                    )

                    # Create adoption signal if Slack is mentioned
                    if analysis['mentions_slack']:
                        signal = AdoptionSignal(
                            company_id=company_id,
                            source="sec_filings",
                            signal_date=filing_data['filing_date'],
                            signal_strength=min(analysis['mention_count'] / 10.0, 1.0),  # Normalize count
                            raw_count=analysis['mention_count'],
                            details={
                                'filing_type': filing_data['form_type'],
                                'period_end': filing_data['period_end'].isoformat() if filing_data['period_end'] else None,
                                'context': analysis['first_mention_context'],
                                'confidence': analysis['analysis_confidence'],
                                'reasoning': analysis.get('analysis_reasoning', '')
                            }
                        )
                        signals.append(signal)

                        logger.info(
                            f"Found Slack mention in {company_id} {filing_data['form_type']} "
                            f"({filing_data['filing_date'].strftime('%Y-%m-%d')}): "
                            f"{analysis['mention_count']} mentions"
                        )

                except Exception as e:
                    logger.warning(f"Error processing filing for {company_id}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error processing SEC filings for company {company_id}: {e}")
            continue

    logger.info(f"Collected {len(signals)} SEC filing signals")
    return signals
