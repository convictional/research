from jinja2 import Environment, FileSystemLoader

from .settings import settings
from .prompts.engine import register_prompt_templates
from .helpers.json import to_json

from .analysis.analyze_gen_models import run_analysis
from .analysis.analyze_entity_overlap import analyze_entity_document_overlap
from .analysis.analyze_fact_docs_distribution import analyze_fact_docs_distribution


async def main():
    # prompt templates
    prompt_templates = Environment(loader=FileSystemLoader(searchpath=settings.root / "src" / "prompts"))
    prompt_templates.filters["to_json"] = to_json
    register_prompt_templates(prompt_templates)

    # Build and save named entity knowledge store
    # await build_named_entity_knowledge_store_csv()

    # Analyze fact distribution
    # analyze_fact_distribution()

    # Analyze entity document overlap
    # analyze_entity_document_overlap()

    # Analyze fact distribution
    analyze_fact_docs_distribution()

    # Analyze entity entropy
    # analyze_entity_entropy()

    # Analyze generative models
    # await run_analysis(fact_length_cutoff=1)
