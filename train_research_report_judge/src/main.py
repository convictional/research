import argparse
import shutil
from pathlib import Path

from common.instruct_llm import set_async_instructor_client
from common.prompt_template_engine import initialize_and_register_prompt_templates
from src.settings import settings, logger


async def cmd_load_data(args: argparse.Namespace) -> None:
    from src.data_loader import load_and_split

    load_and_split()


async def cmd_discover_rubric(args: argparse.Namespace) -> None:
    from src.data_loader import load_split
    from src.rubric_discovery import discover_rubric

    train = load_split("train")
    version = args.version if hasattr(args, "version") and args.version else 1
    await discover_rubric(train, version=version)


def _apply_trial_overrides(args: argparse.Namespace) -> None:
    """Apply CLI trial feature overrides to settings."""
    if hasattr(args, "ensemble_n") and args.ensemble_n is not None:
        settings.ensemble_n = args.ensemble_n
    if hasattr(args, "no_claims") and args.no_claims:
        settings.claim_analysis_enabled = False
    if hasattr(args, "no_metadata") and args.no_metadata:
        settings.include_metadata = False
    if hasattr(args, "rag") and args.rag:
        settings.rag_verification_enabled = True


async def cmd_evaluate_scorer(args: argparse.Namespace) -> None:
    from src.data_loader import load_split
    from src.evaluator import evaluate, save_results
    from src.pointwise_scorer import score_reports
    from src.rubric_discovery import load_rubric

    _apply_trial_overrides(args)

    rubric = load_rubric(version=args.rubric_version)
    reports = load_split(args.split)
    train = load_split("train")

    scored = await score_reports(reports, rubric, train)
    result = evaluate(scored, split=args.split)
    save_results(result, rubric.version)


async def cmd_analyze_disagreements(args: argparse.Namespace) -> None:
    from src.data_loader import load_split
    from src.disagreement_analyzer import analyze_disagreements
    from src.pointwise_scorer import score_reports
    from src.rubric_discovery import load_rubric

    _apply_trial_overrides(args)

    rubric = load_rubric(version=args.rubric_version)
    reports = load_split(args.split)
    train = load_split("train")

    scored = await score_reports(reports, rubric, train)
    await analyze_disagreements(scored, rubric, min_gap=args.min_gap)


async def cmd_full_pipeline(args: argparse.Namespace) -> None:
    from src.data_loader import load_and_split, load_split
    from src.disagreement_analyzer import analyze_disagreements
    from src.evaluator import evaluate, meets_targets, save_results
    from src.pointwise_scorer import score_reports
    from src.rubric_discovery import discover_rubric, refine_rubric

    # Step 1: Load and split data
    print("\n" + "=" * 60)
    print("  PHASE 1: Loading data")
    print("=" * 60)
    train, dev, test = load_and_split()

    # Step 2: Discover initial rubric
    print("\n" + "=" * 60)
    print("  PHASE 2: Discovering rubric")
    print("=" * 60)
    rubric = await discover_rubric(train, version=1)

    # Step 3: Iterative scoring and refinement on dev set
    best_result = None
    best_rubric = rubric

    for round_num in range(1, settings.max_iteration_rounds + 1):
        print("\n" + "=" * 60)
        print(f"  PHASE 3: Evaluation round {round_num}/{settings.max_iteration_rounds}")
        print("=" * 60)

        scored = await score_reports(dev, rubric, train)
        result = evaluate(scored, split="dev")
        save_results(result, rubric.version)

        result_spearman = result.spearman_continuous if result.spearman_continuous is not None else result.spearman_correlation
        best_spearman = best_result.spearman_continuous if best_result and best_result.spearman_continuous is not None else (best_result.spearman_correlation if best_result else -1)
        if best_result is None or result_spearman > best_spearman:
            best_result = result
            best_rubric = rubric

        if meets_targets(result):
            print(f"\n  All targets met at round {round_num}!")
            break

        if round_num < settings.max_iteration_rounds:
            print(f"\n  Targets not met, analyzing disagreements for refinement...")
            analysis = await analyze_disagreements(scored, rubric)
            if analysis and (analysis.rubric_changes or analysis.prompt_changes):
                rubric = await refine_rubric(rubric, analysis.rubric_changes, analysis.prompt_changes)
            else:
                print("  No actionable disagreements found, stopping iteration")
                break

    # Step 4: Final evaluation on test set
    print("\n" + "=" * 60)
    print(f"  PHASE 4: Final test set evaluation (rubric v{best_rubric.version})")
    print("=" * 60)

    test_reports = load_split("test")
    scored_test = await score_reports(test_reports, best_rubric, train)
    test_result = evaluate(scored_test, split="test")
    save_results(test_result, best_rubric.version)

    print(f"\n  Best dev rubric: v{best_rubric.version}")
    print(f"  Test Spearman (integer): {test_result.spearman_correlation:.4f}")
    if test_result.spearman_continuous is not None:
        print(f"  Test Spearman (continuous): {test_result.spearman_continuous:.4f}")
    print(f"  Test MAE: {test_result.mae:.4f}")
    print(f"  Test adjacent match: {test_result.adjacent_match_rate:.4f}")


async def cmd_export_service(args: argparse.Namespace) -> None:
    from src.rubric_discovery import load_rubric

    rubric = load_rubric(version=args.rubric_version)

    service_path = settings.service_path
    rubric_dest = service_path / "rubric.json"
    rubric_dest.write_text(rubric.model_dump_json(indent=2))

    # Copy prompt templates
    prompts_src = Path(__file__).parent / "prompts"
    for template in [
        "pointwise_scorer_system.txt.jinja",
        "pointwise_scorer_user.txt.jinja",
        "critic_system.txt.jinja",
        "qa_alignment_gate_user.txt.jinja",
        "claim_analysis_system.txt.jinja",
        "claim_analysis_user.txt.jinja",
    ]:
        shutil.copy(prompts_src / template, service_path / template)

    print(f"\n  Exported scorer config to {service_path}/")
    print(f"  - rubric.json (v{rubric.version})")
    print(f"  - pointwise_scorer_system.txt.jinja")
    print(f"  - pointwise_scorer_user.txt.jinja")
    print(f"\n  Usage:")
    print(f'  make run_experiment ARGS="train_research_report_judge score --query \'...\' --report-file report.md"')


async def cmd_diagnose_rag(args: argparse.Namespace) -> None:
    import random

    from src.claim_verifier import verify_claims
    from src.content_search import close_search, hybrid_search, init_search
    from src.data_loader import load_split
    from src.format_scorer import score_format
    from src.pointwise_scorer import _analyze_claims
    from src.rag_scorer import _final_score

    reports = load_split(args.split)
    await init_search()

    by_score: dict[int, list] = {}
    for r in reports:
        by_score.setdefault(r.quality_score, []).append(r)

    sample = []
    for score in sorted(by_score.keys()):
        random.seed(42)
        sample.append(random.choice(by_score[score]))

    try:
        for report in sample:
            print(f"\n{'='*80}")
            print(f"Report {report.id} | Expert score: {report.quality_score}")
            print(f"Question: {report.question[:120]}...")
            print(f"Report length: {len(report.research_output)} chars")

            claims = await _analyze_claims(report)
            print(f"\nClaims extracted: {len(claims.claims)} (total in report: {claims.total_claims})")
            print(f"Claim summary: {claims.summary[:200]}")

            test_search = await hybrid_search(report.question[:200])
            print(f"\nSearch test (question as query): {len(test_search)} results")
            for sr in test_search[:3]:
                print(f"  [{sr.score:.3f}] {sr.title[:80]} ({len(sr.content)} chars)")

            rollup = await verify_claims(claims.claims, report.question)
            print(f"\nClaim verification rollup:")
            print(f"  Supported:           {rollup.supported}/{rollup.total_claims}")
            print(f"  Partially supported: {rollup.partially_supported}/{rollup.total_claims}")
            print(f"  Unsupported:         {rollup.unsupported}/{rollup.total_claims}")
            print(f"  No evidence found:   {rollup.no_evidence_found}/{rollup.total_claims}")
            print(f"  Avg confidence:      {rollup.avg_confidence:.2f}")

            for v in rollup.details:
                print(f"  [{v.verdict:>22}] (conf={v.confidence:.2f}) {v.claim[:100]}")

            fmt = await score_format(report)
            print(f"\nFormat assessment:")
            print(f"  Structure:    {fmt.structure_score}/3")
            print(f"  Length:       {fmt.length_adequacy}")
            print(f"  Tone:         {fmt.tone_score}/3")
            print(f"  Q-A alignment:{fmt.qa_alignment_score}/3")
            print(f"  Notes:        {fmt.notes[:200]}")

            final = await _final_score(report.question, rollup, fmt)
            print(f"\nFinal RAG score: {final.quality_score}/3")
            print(f"Justification: {final.justification[:300]}")
            print(f"\nExpert: {report.quality_score} | RAG: {final.quality_score} | Delta: {final.quality_score - report.quality_score}")
    finally:
        await close_search()


async def cmd_evaluate_rag_scorer(args: argparse.Namespace) -> None:
    from src.data_loader import load_split
    from src.evaluator import evaluate, save_results
    from src.rag_scorer import score_reports_with_rag

    reports = load_split(args.split)
    scored = await score_reports_with_rag(reports)
    result = evaluate(scored, split=args.split)
    save_results(result, rubric_version="rag_v1")


async def cmd_score(args: argparse.Namespace) -> None:
    from src.data_loader import load_split
    from src.pointwise_scorer import score_single, _select_calibration_examples, _compute_train_distribution
    from src.rubric_discovery import load_rubric

    rubric = load_rubric(version=args.rubric_version)
    train = load_split("train")
    calibration_examples = _select_calibration_examples(train)
    train_distribution = _compute_train_distribution(train)

    report_text = Path(args.report_file).read_text()
    result = await score_single(args.query, report_text, rubric, calibration_examples, train_distribution)

    print(f"\n  Quality Score: {result.quality_score}/3")
    print(f"  Justification: {result.overall_justification}")
    for ds in result.dimension_scores:
        print(f"    {ds.dimension}: {ds.score}/3 - {ds.justification}")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Train LLM as a Judge for Research Report Outputs")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("load_data", help="Load CSV and create train/dev/test splits")

    rubric_parser = subparsers.add_parser("discover_rubric", help="Discover quality rubric from training data")
    rubric_parser.add_argument("--version", type=int, default=1, help="Rubric version number")

    eval_parser = subparsers.add_parser("evaluate_scorer", help="Score reports and compute alignment metrics")
    eval_parser.add_argument("--split", default="dev", choices=["dev", "test"])
    eval_parser.add_argument("--rubric-version", type=int, default=None, help="Rubric version (default: latest)")
    eval_parser.add_argument("--ensemble-n", type=int, default=None, help="Override ensemble N (default: from settings)")
    eval_parser.add_argument("--no-claims", action="store_true", help="Disable claim analysis (Trial 9)")
    eval_parser.add_argument("--no-metadata", action="store_true", help="Disable metadata/contrastive pairs (Trial 10)")
    eval_parser.add_argument("--rag", action="store_true", help="Enable RAG claim verification (Trial 11)")

    disagree_parser = subparsers.add_parser("analyze_disagreements", help="Analyze scorer-expert disagreements")
    disagree_parser.add_argument("--split", default="dev", choices=["dev", "test"])
    disagree_parser.add_argument("--rubric-version", type=int, default=None)
    disagree_parser.add_argument("--min-gap", type=int, default=2, help="Minimum score gap to analyze")
    disagree_parser.add_argument("--ensemble-n", type=int, default=None)
    disagree_parser.add_argument("--no-claims", action="store_true")
    disagree_parser.add_argument("--no-metadata", action="store_true")

    subparsers.add_parser("full_pipeline", help="Run all phases end-to-end with iteration")

    export_parser = subparsers.add_parser("export_service", help="Export final scorer config for reuse")
    export_parser.add_argument("--rubric-version", type=int, default=None)

    diag_parser = subparsers.add_parser("diagnose_rag", help="Diagnostic: dump intermediate RAG signals for sample reports")
    diag_parser.add_argument("--split", default="dev", choices=["dev", "test"])

    rag_parser = subparsers.add_parser("evaluate_rag_scorer", help="Evaluate RAG claim-verification scorer")
    rag_parser.add_argument("--split", default="dev", choices=["dev", "test"])

    score_parser = subparsers.add_parser("score", help="Score a single report")
    score_parser.add_argument("--query", required=True, help="Research question")
    score_parser.add_argument("--report-file", required=True, help="Path to report markdown file")
    score_parser.add_argument("--rubric-version", type=int, default=None)

    args = parser.parse_args()

    # Initialize LLM client and prompt templates for commands that need them
    commands_needing_llm = {"discover_rubric", "evaluate_scorer", "analyze_disagreements", "full_pipeline", "score", "evaluate_rag_scorer", "diagnose_rag"}
    if args.command in commands_needing_llm:
        set_async_instructor_client(settings.llm_model, settings.anthropic_api_key)
        initialize_and_register_prompt_templates(Path(__file__).parent / "prompts")

    command_map = {
        "load_data": cmd_load_data,
        "discover_rubric": cmd_discover_rubric,
        "evaluate_scorer": cmd_evaluate_scorer,
        "analyze_disagreements": cmd_analyze_disagreements,
        "full_pipeline": cmd_full_pipeline,
        "export_service": cmd_export_service,
        "evaluate_rag_scorer": cmd_evaluate_rag_scorer,
        "diagnose_rag": cmd_diagnose_rag,
        "score": cmd_score,
    }

    await command_map[args.command](args)
