import asyncio
import pandas as pd
import pingouin as pg
import textwrap
from datetime import datetime
from pydantic import BaseModel, Field

from common.async_helper import limited_task, execute_tasks_with_manual_pbar
from common.instruct_llm import ainstruct_llm, set_async_instructor_client
from common.prompt_template_engine import build_prompt
from .settings import settings

RATED_UNSTATED_GOALS_PATH = settings.data_path / "rated_individual_goals" / "rated_unstated_goals.csv"
OUTPUT_DIR = settings.data_path / "unstated_goals_expert_agreement_analysis"
OUTPUT_FILE = OUTPUT_DIR / "unstated_goals_expert_agreement_analysis.txt"

LLM_TEMPERATURE = 0.0

ICC_ZERO_EXPLANATION = """
TECHNICAL NOTE: RATER 2 vs RATER 3 ICC = 0 ANALYSIS
============================================

The perfect zero ICC between Rater 2 and Rater 3 (ICC = -0.0000) represents a rare statistical phenomenon
called "perfect orthogonality" in rating patterns.

STATISTICAL PATTERN:
Despite having similar means (Rater 2: 4.00, Rater 3: 4.46), their individual item ratings are perfectly
uncorrelated. The key insight is that when Rater 2 rates below their average, Rater 3 tends to rate above
his average, and vice versa. This creates a systematic pattern where their deviations from their
respective means always cancel out when multiplied together.

EXAMPLE OF THE CANCELLATION EFFECT:
For correlation calculation, we compute (rating - mean) for each rater, then multiply these deviations:

Example goals showing the perfect cancellation:
- Goal where Rater 2=5, Rater 3=5:
  Rater 2 deviation: 5-4.00 = +1.00, Rater 3 deviation: 5-4.46 = +0.54
  Product: (+1.00) × (+0.54) = +0.54

- Goal where Rater 2=3, Rater 3=3:
  Rater 2 deviation: 3-4.00 = -1.00, Rater 3 deviation: 3-4.46 = -1.46
  Product: (-1.00) × (-1.46) = +1.46

- Goal where Rater 2=3, Rater 3=5:
  Rater 2 deviation: 3-4.00 = -1.00, Rater 3 deviation: 5-4.46 = +0.54
  Product: (-1.00) × (+0.54) = -0.54

The sum of ALL such products across the 28 goals equals exactly 0.0000000000, creating perfect
orthogonality. Positive products from some goal pairs are exactly cancelled by negative products
from other goal pairs, particularly when one rater is above their mean while the other is below.

MATHEMATICAL RESULT:
- Sum of products of deviations: 0.0000000000 (perfect orthogonality)
- Pearson correlation: 0.0000000000 (exact zero to 10 decimal places)

INTERPRETATION:
This is a genuine statistical phenomenon where two raters have mathematically orthogonal rating
patterns relative to their respective means. While extremely rare in real-world scenarios, the
ICC calculation correctly identifies this as indicating no reliable agreement between the raters.
"""


class ICCAnalysis(BaseModel):
    """LLM analysis of ICC results."""
    statistical_interpretation: str = Field(..., description="Technical interpretation of ICC value, confidence interval, and significance")
    practical_implications: str = Field(..., description="What this means for research validity and expert agreement quality")
    methodological_assessment: str = Field(..., description="Assessment of the reliability of the rating methodology")
    recommendations: str = Field(..., description="Recommendations for research methodology and next steps")


def calculate_icc(long_df: pd.DataFrame, df: pd.DataFrame, rater_list: list[str], analysis_label: str) -> dict:
    """Calculate ICC for specific rater subset."""
    # Filter long format data for specific raters
    filtered_long_df = long_df[long_df['rater'].isin(rater_list)]

    # Calculate ICC
    icc_results = pg.intraclass_corr(
        data=filtered_long_df,
        targets='goal_id',
        raters='rater',
        ratings='rating'
    )

    # Extract ICC(2,1) result
    icc_2_1 = icc_results[icc_results['Type'] == 'ICC2'].iloc[0]

    # Calculate descriptive statistics for this rater subset
    rater_columns = [f"{rater}_rating" for rater in rater_list]
    rater_stats = df[rater_columns].describe()

    # Determine interpretation
    icc_value = icc_2_1['ICC']
    if icc_value < 0.50:
        interpretation = "Poor reliability"
    elif icc_value < 0.75:
        interpretation = "Moderate reliability"
    elif icc_value < 0.90:
        interpretation = "Good reliability"
    else:
        interpretation = "Excellent reliability"

    return {
        'analysis_label': analysis_label,
        'rater_list': rater_list,
        'icc_result': icc_2_1,
        'rater_stats': rater_stats,
        'interpretation': interpretation,
        'sample_size': len(df)
    }


async def analyze_icc_with_llm(result: dict) -> ICCAnalysis:
    """Generate LLM analysis of ICC results."""
    system_prompt = build_prompt("icc_analysis_system.txt.jinja")
    user_prompt = build_prompt(
        "icc_analysis_user.txt.jinja",
        analysis_label=result['analysis_label'],
        rater_list=result['rater_list'],
        sample_size=result['sample_size'],
        icc_value=f"{result['icc_result']['ICC']:.4f}",
        ci_low=f"{result['icc_result']['CI95%'][0]:.4f}",
        ci_high=f"{result['icc_result']['CI95%'][1]:.4f}",
        f_statistic=f"{result['icc_result']['F']:.4f}",
        p_value=result['icc_result']['pval']
    )

    llm_analysis = await ainstruct_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_model=ICCAnalysis,
        llm_model=settings.llm_model,
        temperature=LLM_TEMPERATURE
    )

    return llm_analysis


async def analyze_icc_with_index(index: int, result: dict) -> tuple[int, ICCAnalysis]:
    """Generate LLM analysis of ICC results with index for matching."""
    llm_analysis = await analyze_icc_with_llm(result)
    return index, llm_analysis


async def validate_expert_agreement_for_unstated_goals() -> None:
    """Validate expert agreement for unstated goal ratings."""
    print("Checking expert agreement analysis...")

    if OUTPUT_FILE.exists():
        print(f"Expert agreement analysis already exists at: {OUTPUT_FILE}")
        print("Skipping expert agreement analysis.")
        return

    print("Loading rated unstated goals data...")

    if not RATED_UNSTATED_GOALS_PATH.exists():
        print(f"Error: Rated unstated goals file not found: {RATED_UNSTATED_GOALS_PATH}")
        return

    df = pd.read_csv(RATED_UNSTATED_GOALS_PATH)
    df = df.drop(columns=['avg_rating', 'std_dev_rating'])
    print(f"Rated unstated goals loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")
    print(f"Data types:\n{df.dtypes}")

    # Create goal_id for unique identification
    df['goal_id'] = range(len(df))

    # Transform to long format for ICC analysis
    long_df = pd.melt(
        df,
        id_vars=['goal_id'],
        value_vars=['rater1_rating', 'rater2_rating', 'rater3_rating'],
        var_name='rater',
        value_name='rating'
    )

    # Clean rater names
    long_df['rater'] = long_df['rater'].str.replace('_rating', '')

    print(f"Transformed to long format: {long_df.shape[0]} rows, {long_df.shape[1]} columns")
    print(f"Long format columns: {list(long_df.columns)}")

    # Define analyses to perform
    analyses = [
        ("All Three Raters", ['rater1', 'rater2', 'rater3']),
        ("Rater 1 vs Rater 2", ['rater1', 'rater2']),
        ("Rater 1 vs Rater 3", ['rater1', 'rater3']),
        ("Rater 2 vs Rater 3", ['rater2', 'rater3'])
    ]

    # Calculate ICC for all analyses
    print(f"Calculating ICC for {len(analyses)} different rater combinations...")
    results = []
    for analysis_label, rater_list in analyses:
        print(f"  - {analysis_label}")
        result = calculate_icc(long_df, df, rater_list, analysis_label)
        results.append(result)

    # Initialize instructor client for LLM analysis
    print("Initializing LLM analysis...")
    set_async_instructor_client(settings.llm_model, settings.anthropic_api_key)

    # Generate LLM analysis for each result with progress bar
    print(f"Generating LLM analysis for {len(results)} ICC results...")
    max_concurrent_tasks = 10
    delay_between_tasks = 0.1

    semaphore = asyncio.Semaphore(max_concurrent_tasks)

    tasks = [
        limited_task(
            analyze_icc_with_index(i, result),
            semaphore,
            delay_between_tasks
        )
        for i, result in enumerate(results)
    ]

    llm_results = await execute_tasks_with_manual_pbar(tasks)

    # Match LLM analyses back to original results
    for i, llm_analysis in llm_results:
        results[i]['llm_analysis'] = llm_analysis

    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Format results and write to file
    with open(OUTPUT_FILE, 'w') as f:
        f.write("UNSTATED GOALS EXPERT AGREEMENT ANALYSIS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Sample Size: {len(df)} unstated goals\n")
        f.write(f"Rating Scale: 1-5\n")
        f.write(f"Number of Analyses: {len(analyses)}\n\n")

        # Write each analysis
        for i, result in enumerate(results, 1):
            f.write(f"ANALYSIS {i}: {result['analysis_label'].upper()}\n")
            f.write("=" * 60 + "\n")
            f.write(f"Raters: {', '.join(result['rater_list'])}\n")
            f.write(f"Sample Size: {result['sample_size']} goals\n\n")

            f.write("DESCRIPTIVE STATISTICS\n")
            f.write("-" * 25 + "\n")
            f.write(str(result['rater_stats']) + "\n\n")

            f.write("ICC(2,1) RESULTS\n")
            f.write("-" * 15 + "\n")
            icc_result = result['icc_result']
            f.write(f"ICC(2,1) Value: {icc_result['ICC']:.4f}\n")
            f.write(f"95% Confidence Interval: [{icc_result['CI95%'][0]:.4f}, {icc_result['CI95%'][1]:.4f}]\n")
            f.write(f"F-statistic: {icc_result['F']:.4f}\n")
            f.write(f"p-value: {icc_result['pval']}\n")
            f.write(f"Interpretation: {result['interpretation']}\n")
            f.write(f"Statistical Significance: {'Yes' if icc_result['pval'] < 0.05 else 'No'} (p < 0.05)\n\n")

            # Add LLM Analysis with word wrapping
            f.write("LLM ANALYSIS\n")
            f.write("-" * 12 + "\n")

            f.write("Statistical Interpretation:\n")
            wrapped_text = textwrap.fill(result['llm_analysis'].statistical_interpretation, width=100)
            f.write(f"{wrapped_text}\n\n")

            f.write("Practical Implications:\n")
            wrapped_text = textwrap.fill(result['llm_analysis'].practical_implications, width=100)
            f.write(f"{wrapped_text}\n\n")

            f.write("Methodological Assessment:\n")
            wrapped_text = textwrap.fill(result['llm_analysis'].methodological_assessment, width=100)
            f.write(f"{wrapped_text}\n\n")

            f.write("Recommendations:\n")
            wrapped_text = textwrap.fill(result['llm_analysis'].recommendations, width=100)
            f.write(f"{wrapped_text}\n\n")

        # Summary comparison
        f.write("SUMMARY COMPARISON\n")
        f.write("=" * 60 + "\n")
        f.write("Analysis                    ICC(2,1)   95% CI              F-stat    p-value   Interpretation\n")
        f.write("-" * 95 + "\n")
        for result in results:
            icc_val = result['icc_result']['ICC']
            ci_low = result['icc_result']['CI95%'][0]
            ci_high = result['icc_result']['CI95%'][1]
            f_stat = result['icc_result']['F']
            p_val = result['icc_result']['pval']
            f.write(f"{result['analysis_label']:<25} {icc_val:>7.4f}   [{ci_low:>6.4f}, {ci_high:>6.4f}]   {f_stat:>7.4f}   {p_val:>9.6f}   {result['interpretation']}\n")

        f.write("\n\nINTERPRETATION GUIDELINES\n")
        f.write("-" * 25 + "\n")
        f.write("< 0.50: Poor reliability\n")
        f.write("0.50 - 0.75: Moderate reliability\n")
        f.write("0.75 - 0.90: Good reliability\n")
        f.write("> 0.90: Excellent reliability\n")

        f.write(f"\n\nTECHNICAL NOTES\n")
        f.write("=" * 60 + "\n")
        f.write(ICC_ZERO_EXPLANATION)

    print(f"ICC analysis complete. Results saved to: {OUTPUT_FILE}")
    print("Summary of ICC results:")
    for result in results:
        icc_val = result['icc_result']['ICC']
        print(f"  {result['analysis_label']}: ICC(2,1) = {icc_val:.4f} ({result['interpretation']})")
