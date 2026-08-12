from pydantic import BaseModel
from typing import List, Dict, Any
import numpy as np
import pandas as pd

from ..instruct_llm import ainstruct_llm
from ..helpers.print_section import print_section, Colors
from .forecasting_models import ForecastResult


class InterpretationResult(BaseModel):
    """Structure for LLM interpretation results"""

    summary: str
    key_insights: List[str]
    recommendations: List[str]
    confidence_assessment: str


class ResultsInterpreter:
    """Agent responsible for interpreting results and providing business insights"""

    def _analyze_growth_patterns(self, forecast: ForecastResult) -> Dict[str, Any]:
        """Analyze growth patterns in the forecast"""
        values = np.array(forecast.forecast)
        growth_rate = (values[-1] - values[0]) / values[0] * 100
        volatility = np.std(values) / np.mean(values) * 100

        return {
            "growth_rate": growth_rate,
            "volatility": volatility,
            "min_forecast": np.min(values),
            "max_forecast": np.max(values),
            "avg_forecast": np.mean(values),
            "forecast_length": len(values),
        }

    def _calculate_confidence_levels(self, forecast: ForecastResult) -> Dict[str, Any]:
        """Calculate confidence levels based on confidence intervals"""
        ci_widths = [upper - lower for lower, upper in forecast.confidence_interval]
        relative_uncertainty = np.mean(ci_widths) / np.mean(forecast.forecast) * 100

        return {
            "relative_uncertainty": relative_uncertainty,
            "avg_ci_width": np.mean(ci_widths),
            "max_ci_width": max(ci_widths),
            "min_ci_width": min(ci_widths),
        }

    async def interpret_results(
        self,
        forecast: ForecastResult,
        context: str,
        evaluation_metrics: Dict[str, Any] = None,
        training_data: pd.Series = None,  # Add parameter
    ) -> Dict[str, Any]:
        """Interpret forecasting results using LLM analysis"""
        print_section("Results Interpretation", "Analyzing forecast implications", Colors.GREEN)

        # Prepare analysis data
        growth_patterns = self._analyze_growth_patterns(forecast)
        confidence_metrics = self._calculate_confidence_levels(forecast)

        # Prepare training data summary
        training_summary = ""
        if training_data is not None:
            training_summary = f"""
            Training Data Summary:
            - Most Recent Values:
            {training_data.tail().to_string()}

            - Statistics:
            {training_data.describe().to_string()}
            """

        system_prompt = """You are an expert data scientist specializing in time series forecasting.
        You are a student of Nate Silver and have a knack for translating data into actionable insights for business.
        You have also had a long and varied career forecasting just about everything in business, from
        stock prices, inventory levels, buying patterns, product growth, etc.
        The target_column name in the 'training_data_summary' is the column which you are evaluating a forecast
        for. The 'business_context' is a string that provides additional context about the business model.
        Analyze the provided forecast results and metrics to generate business insights and recommendations.
        Focus on practical implications for resource planning, revenue projections, and risk management within the
        context of what forecast you are evaluating.
        Provide clear, actionable insights that business stakeholders can use for decision-making."""

        user_prompt = f"""
        Business Context:
        {context}

        {training_summary}

        Forecast Method: {forecast.method_used}

        Growth Patterns:
        - Growth Rate: {growth_patterns['growth_rate']:.2f}%
        - Volatility: {growth_patterns['volatility']:.2f}%
        - Average Forecast: ${growth_patterns['avg_forecast']:,.2f}
        - Forecast Range: ${growth_patterns['min_forecast']:,.2f} to ${growth_patterns['max_forecast']:,.2f}
        - Forecast Length: {growth_patterns['forecast_length']} periods

        Confidence Metrics:
        - Relative Uncertainty: {confidence_metrics['relative_uncertainty']:.2f}%
        - Average CI Width: ${confidence_metrics['avg_ci_width']:,.2f}

        Model Evaluation Metrics:
        {evaluation_metrics if evaluation_metrics else 'No evaluation metrics provided'}

        Provide a structured analysis including:
        1. A concise summary of the forecast implications
        2. Key business insights derived from the data
        3. Specific recommendations for action
        4. Assessment of forecast confidence and risks
        """

        result = await ainstruct_llm(
            system_prompt=system_prompt, user_prompt=user_prompt, response_model=InterpretationResult
        )
        interpretation = result[0]  # Extract interpretation from tuple

        formatted_interpretation = {
            "summary": interpretation.summary,
            "key_insights": interpretation.key_insights,
            "recommendations": interpretation.recommendations,
            "confidence_assessment": interpretation.confidence_assessment,
            "forecast_method": forecast.method_used,
            "metrics": {"growth_patterns": growth_patterns, "confidence_metrics": confidence_metrics},
        }

        print_section(
            "Business Insights",
            f"Summary: {interpretation.summary}\n\n"
            f"Key Insights:\n" + "\n".join([f"- {i}" for i in interpretation.key_insights]) + "\n\n"
            "Recommendations:\n" + "\n".join([f"- {r}" for r in interpretation.recommendations]) + "\n\n"
            f"Confidence Assessment:\n{interpretation.confidence_assessment}",
            Colors.GREEN,
        )

        return formatted_interpretation
