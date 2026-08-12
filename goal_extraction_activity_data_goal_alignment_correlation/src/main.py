from .settings import settings
from common.prompt_template_engine import initialize_and_register_prompt_templates
from .raw_data_processing import process_raw_data
from .extract_individual_goals import extract_individual_goals
from .expert_agreement_validation_unstated_goals import validate_expert_agreement_for_unstated_goals
from .metrics_analysis import metrics_analysis
from .rater_feedback_analysis import rater_feedback_analysis


async def main():
    # prompt templates
    initialize_and_register_prompt_templates(settings.root / "src" / "prompts")

    # Load and process raw goal mining data
    process_raw_data()

    # Extract individual goals from processed data
    await extract_individual_goals()

    # Validate expert agreement for unstated goals
    await validate_expert_agreement_for_unstated_goals()

    # Run metrics analysis
    metrics_analysis()

    # Run rater feedback analysis
    await rater_feedback_analysis()
