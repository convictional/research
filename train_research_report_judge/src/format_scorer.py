from common.instruct_llm import ainstruct_llm
from common.prompt_template_engine import build_prompt
from src.models import FormatAssessment, RatedReport
from src.settings import settings


async def score_format(report: RatedReport) -> FormatAssessment:
    system_prompt = build_prompt("format_scorer_system.txt.jinja")
    user_prompt = build_prompt(
        "format_scorer_user.txt.jinja",
        question=report.question,
        research_output=report.research_output,
    )

    return await ainstruct_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_model=FormatAssessment,
        llm_model=settings.haiku_model,
        temperature=settings.temperature,
        max_tokens=1024,
    )
