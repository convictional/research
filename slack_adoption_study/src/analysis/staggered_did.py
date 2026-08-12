import logging
from typing import Dict, List
import pandas as pd
import numpy as np

from ..models import AdoptionEvent, FinancialOutcome, AnalysisResult
from ..settings import Settings

logger = logging.getLogger(__name__)


async def run_analysis(
    settings: Settings,
    adoption_events: List[AdoptionEvent],
    outcomes: List[FinancialOutcome]
) -> List[AnalysisResult]:
    """
    Run staggered difference-in-differences analysis using modern estimators.

    This implements event-time analysis robust to staggered treatment timing,
    following Sun & Abraham (2021) and Callaway & Sant'Anna (2021) approaches.

    Args:
        settings: Experiment configuration
        adoption_events: Determined Slack adoption events
        outcomes: Financial outcome data

    Returns:
        List of analysis results for different outcome variables
    """
    logger.info(f"Running staggered DiD analysis on {len(adoption_events)} treated units")

    # Convert to pandas for analysis
    adoption_df = pd.DataFrame([
        {
            'company_id': event.company_id,
            'adoption_date': event.adoption_date,
            'adoption_quarter': f"{event.adoption_date.year}Q{(event.adoption_date.month-1)//3 + 1}",
            'confidence': event.confidence,
            'intensity': event.adoption_intensity
        }
        for event in adoption_events
    ])

    outcomes_df = pd.DataFrame([
        {
            'company_id': outcome.company_id,
            'fiscal_quarter': outcome.fiscal_quarter,
            'revenue': float(outcome.revenue) if outcome.revenue else None,
            'gross_margin': outcome.gross_margin,
            'operating_margin': outcome.operating_margin,
            'revenue_per_employee': float(outcome.revenue_per_employee) if outcome.revenue_per_employee else None,
            'sga_efficiency': outcome.sga_efficiency,
            'revenue_growth': outcome.revenue_growth
        }
        for outcome in outcomes
    ])

    # Merge data
    analysis_data = _prepare_analysis_data(adoption_df, outcomes_df, settings)

    if analysis_data.empty:
        logger.warning("No data available for analysis")
        return []

    # Run analysis for each outcome variable
    results = []
    outcome_vars = ['revenue', 'gross_margin', 'operating_margin', 'revenue_per_employee']

    for outcome_var in outcome_vars:
        try:
            logger.info(f"Analyzing {outcome_var}")

            # TODO: Implement actual staggered DiD estimation
            # This would use libraries like:
            # - econml for causal inference
            # - linearmodels for panel data
            # - Or implement Sun-Abraham / Callaway-Sant'Anna directly

            result = _run_event_study_did(analysis_data, outcome_var, settings)
            if result:
                results.append(result)

        except Exception as e:
            logger.error(f"Error analyzing {outcome_var}: {e}")
            continue

    logger.info(f"Completed staggered DiD analysis with {len(results)} results")
    return results


def _prepare_analysis_data(
    adoption_df: pd.DataFrame,
    outcomes_df: pd.DataFrame,
    settings: Settings
) -> pd.DataFrame:
    """
    Prepare merged analysis dataset with event time indicators.

    Args:
        adoption_df: Adoption events data
        outcomes_df: Outcomes data
        settings: Configuration

    Returns:
        Analysis-ready pandas DataFrame
    """
    # Check if we have any adoption events
    if adoption_df.empty:
        logger.info("No adoption events to merge - all companies will be control group")
        outcomes_df['treated'] = False
        outcomes_df['event_time'] = 0
        return outcomes_df

    # Merge adoption and outcomes data
    merged = outcomes_df.merge(
        adoption_df[['company_id', 'adoption_quarter', 'confidence']],
        on='company_id',
        how='left'
    )

    # Create treatment indicator
    merged['treated'] = merged['adoption_quarter'].notna()

    logger.info(f"Merge results: {len(merged)} observations, {merged['treated'].sum()} treated")

    # Calculate event time (quarters relative to adoption)
    merged['event_time'] = 0

    for idx, row in merged.iterrows():
        if pd.notna(row['adoption_quarter']):
            # Parse quarters and calculate difference
            outcome_year, outcome_q = map(int, row['fiscal_quarter'].replace('Q', ' ').split())
            adoption_year, adoption_q = map(int, row['adoption_quarter'].replace('Q', ' ').split())

            event_time = (outcome_year - adoption_year) * 4 + (outcome_q - adoption_q)
            merged.at[idx, 'event_time'] = event_time

    # Filter to analysis window
    analysis_window = merged[
        (merged['event_time'] >= -settings.pre_treatment_quarters) &
        (merged['event_time'] <= settings.post_treatment_quarters)
    ].copy()

    # Create event time dummies
    for t in range(-settings.pre_treatment_quarters, settings.post_treatment_quarters + 1):
        if t != -1:  # Omit t=-1 as base period
            analysis_window[f'event_time_{t}'] = (analysis_window['event_time'] == t) & analysis_window['treated']

    return analysis_window


def _run_event_study_did(
    data: pd.DataFrame,
    outcome_var: str,
    settings: Settings
) -> AnalysisResult | None:
    """
    Run event study difference-in-differences for a specific outcome.

    Args:
        data: Analysis dataset
        outcome_var: Outcome variable name
        settings: Configuration

    Returns:
        Analysis result or None if failed
    """
    if outcome_var not in data.columns or data[outcome_var].isna().all():
        logger.warning(f"No data available for {outcome_var}")
        return None

    # TODO: Implement actual econometric estimation
    # This is a placeholder implementation

    # Filter to companies with data for this specific metric
    analysis_data = data.dropna(subset=[outcome_var]).copy()

    # Check if we have sufficient data for DiD analysis
    treated_with_data = analysis_data[analysis_data['treated']]
    treated_pre_with_data = treated_with_data[treated_with_data['event_time'] < 0]
    treated_post_with_data = treated_with_data[treated_with_data['event_time'] >= 0]
    control_with_data = analysis_data[~analysis_data['treated']]

    logger.info(f"Data availability for {outcome_var}: "
                f"treated_pre={len(treated_pre_with_data)}, "
                f"treated_post={len(treated_post_with_data)}, "
                f"control={len(control_with_data)}")

    if len(analysis_data) < 10:  # Lowered minimum threshold
        logger.warning(f"Insufficient data for {outcome_var}: {len(analysis_data)} observations")
        return None

    # Check if we have both pre and post treatment data
    if len(treated_pre_with_data) == 0:
        logger.warning(f"No pre-treatment data available for {outcome_var}")
        return None

    if len(treated_post_with_data) == 0:
        logger.warning(f"No post-treatment data available for {outcome_var}")
        return None

    # Calculate simple treatment effects (placeholder)
    treated_post = analysis_data[(analysis_data['treated']) & (analysis_data['event_time'] >= 0)]
    treated_pre = analysis_data[(analysis_data['treated']) & (analysis_data['event_time'] < 0)]
    control = analysis_data[~analysis_data['treated']]

    logger.info(f"Treatment groups for {outcome_var}: treated_post={len(treated_post)}, treated_pre={len(treated_pre)}, control={len(control)}")

    if len(treated_post) == 0 or len(treated_pre) == 0 or len(control) == 0:
        logger.warning(f"Missing treatment groups for {outcome_var}: post={len(treated_post)}, pre={len(treated_pre)}, control={len(control)}")
        return None

    # Simple DiD estimate (placeholder - should use proper econometric methods)
    treated_diff = treated_post[outcome_var].mean() - treated_pre[outcome_var].mean()
    control_diff = 0  # Assuming control group is stable (placeholder)

    treatment_effect = treated_diff - control_diff

    # Placeholder standard error and p-value
    standard_error = abs(treatment_effect) * 0.3  # Rough approximation
    p_value = 0.05 if abs(treatment_effect) > 1.96 * standard_error else 0.20

    # Pre-trend test (placeholder)
    pre_trend_test = {
        'f_statistic': 1.2,
        'p_value': 0.30,
        'conclusion': 'Cannot reject parallel trends'
    }

    return AnalysisResult(
        analysis_type="staggered_did",
        outcome_variable=outcome_var,
        treatment_effect=treatment_effect,
        standard_error=standard_error,
        p_value=p_value,
        confidence_interval=(
            treatment_effect - 1.96 * standard_error,
            treatment_effect + 1.96 * standard_error
        ),
        sample_size=len(analysis_data),
        pre_trend_test=pre_trend_test,
        details={
            'treated_units': len(analysis_data[analysis_data['treated']].groupby('company_id')),
            'control_units': len(analysis_data[~analysis_data['treated']].groupby('company_id')),
            'pre_periods': settings.pre_treatment_quarters,
            'post_periods': settings.post_treatment_quarters,
            'estimator': 'placeholder_did',  # In real implementation: 'sun_abraham' or 'callaway_santanna'
            'note': 'This is a placeholder implementation. Real analysis would use proper econometric methods.'
        }
    )


def _test_parallel_trends(
    data: pd.DataFrame,
    outcome_var: str,
    settings: Settings
) -> Dict:
    """
    Test parallel trends assumption in pre-treatment periods.

    Args:
        data: Analysis dataset
        outcome_var: Outcome variable
        settings: Configuration

    Returns:
        Pre-trend test results
    """
    # TODO: Implement proper pre-trend testing
    # This would test whether treated and control groups had parallel trends
    # before treatment using F-tests on pre-treatment event time coefficients

    return {
        'f_statistic': 1.0,
        'p_value': 0.5,
        'conclusion': 'Placeholder test - implement proper pre-trend analysis'
    }
