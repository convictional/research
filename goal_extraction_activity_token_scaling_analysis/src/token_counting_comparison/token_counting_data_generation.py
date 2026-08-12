import asyncio
import tiktoken
import pandas as pd
import yaml
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
from typing import List

from ..settings import settings, OPENAI_GPT4, CLAUDE_SONNET
from common.prompt_template_engine import build_prompt
from common.async_helper import limited_task, execute_tasks_with_manual_pbar


def load_synthetic_test_data() -> List[str]:
    """
    Load synthetic test data from YAML file.
    """
    yaml_file = settings.input_data_path / "test_data_token_count_comparison.yaml"
    with open(yaml_file, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return data['test_samples']


def count_tokens_tiktoken(text: str) -> int:
    """
    Count tokens using tiktoken library for GPT-4.
    """
    encoding = tiktoken.encoding_for_model("gpt-4")
    return len(encoding.encode(text))


async def make_openai_token_count_call(prompt: str) -> int:
    """Make a single OpenAI API call and extract input token count."""
    client = AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value())
    try:
        response = await client.chat.completions.create(
            model=OPENAI_GPT4,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,  # Minimize output tokens for cost
            temperature=0
        )
        await asyncio.sleep(5)  # Sleep after response to avoid rate limits
        return response.usage.prompt_tokens
    except Exception as e:
        print(f"OpenAI API call failed: {e}")
        return 0


async def make_anthropic_token_count_call(prompt: str) -> int:
    """Make a single Anthropic API call and extract input token count."""
    client = AsyncAnthropic(api_key=settings.anthropic_api_key.get_secret_value())
    try:
        response = await client.messages.create(
            model=CLAUDE_SONNET,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,  # Minimize output tokens for cost
            temperature=0
        )
        await asyncio.sleep(5)  # Sleep after response to avoid rate limits
        return response.usage.input_tokens
    except Exception as e:
        print(f"Anthropic API call failed: {e}")
        return 0


async def count_tokens_with_api_batching(
    prompts: List[str],
    api_call_func,
    api_name: str,
    max_concurrent_tasks: int = 10,
    delay_between_tasks: float = 10.0
) -> List[int]:
    """
    Generic function to count tokens using API calls with batching.
    """
    # Set up concurrent execution
    semaphore = asyncio.Semaphore(max_concurrent_tasks)
    tasks = [
        limited_task(
            api_call_func(prompt),
            semaphore,
            delay_between_tasks
        )
        for prompt in prompts
    ]

    print(f"Making {api_name} API calls for token counting...")
    token_counts = await execute_tasks_with_manual_pbar(tasks)
    return token_counts


async def count_tokens_openai_api_with_prompts(
    prompts: List[str],
    max_concurrent_tasks: int = 30,
    delay_between_tasks: float = 0.1
) -> List[int]:
    """Count tokens using OpenAI GPT-4 API calls with batching."""
    return await count_tokens_with_api_batching(
        prompts, make_openai_token_count_call, "OpenAI", max_concurrent_tasks, delay_between_tasks
    )


async def count_tokens_anthropic_api_with_prompts(
    prompts: List[str],
    max_concurrent_tasks: int = 30,
    delay_between_tasks: float = 0.1
) -> List[int]:
    """Count tokens using Anthropic Claude API calls with batching."""
    return await count_tokens_with_api_batching(
        prompts, make_anthropic_token_count_call, "Anthropic", max_concurrent_tasks, delay_between_tasks
    )


async def run_token_counting_data_generation():
    """
    Generate token counting comparison data using tiktoken, OpenAI, and Anthropic APIs.
    Exports results to CSV. Skips if file already exists.
    """
    # Check if output file already exists
    output_dir = settings.output_path / "token_count_comparison"
    output_file = output_dir / "token_comparison.csv"

    if output_file.exists():
        print(f"Token comparison CSV already exists at: {output_file}")
        print("Skipping token counting experiment.")
        return

    print("Starting token counting comparison experiment...")

    # Load test data from YAML
    text_samples = load_synthetic_test_data()
    print(f"Loaded {len(text_samples)} text samples from YAML")

    # Build prompts for each text sample (same as what gets sent to API)
    prompts = [
        build_prompt("token_counting/user_prompt.txt.jinja", text_sample=sample)
        for sample in text_samples
    ]

    # Count tokens with tiktoken using the rendered prompts
    print("Counting tokens with tiktoken...")
    tiktoken_counts = [count_tokens_tiktoken(prompt) for prompt in prompts]

    # Count tokens with OpenAI API using the same rendered prompts
    gpt4_counts = await count_tokens_openai_api_with_prompts(prompts)

    # Count tokens with Anthropic API using the same rendered prompts
    sonnet_counts = await count_tokens_anthropic_api_with_prompts(prompts)

    # Create comparison DataFrame
    comparison_data = pd.DataFrame({
        'num_tiktoken_tokens': tiktoken_counts,
        'num_gpt_4_tokens': gpt4_counts,
        'num_sonnet_tokens': sonnet_counts
    })

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export to CSV
    comparison_data.to_csv(output_file, index=False)

    print(f"Token comparison results exported to: {output_file}")
    print(f"CSV contains {len(comparison_data)} records")

    return comparison_data
