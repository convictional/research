import argparse
import asyncio
import logging
import sys

import src.main as src_main

VARIANTS = ("sonnet", "haiku", "gemma")


def _parse_extract(value: str) -> list[str]:
    variants = [v.strip() for v in value.split(",") if v.strip()]
    if not variants:
        raise argparse.ArgumentTypeError("--extract requires at least one variant")
    for v in variants:
        if v not in VARIANTS:
            raise argparse.ArgumentTypeError(f"Unknown variant '{v}'. Choices: {', '.join(VARIANTS)}")
    return variants


def _parse_diff(value: str) -> tuple[str, str]:
    parts = [v.strip() for v in value.split(",") if v.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("--diff requires exactly two variants separated by a comma (A,B)")
    for v in parts:
        if v not in VARIANTS:
            raise argparse.ArgumentTypeError(f"Unknown variant '{v}'. Choices: {', '.join(VARIANTS)}")
    if parts[0] == parts[1]:
        raise argparse.ArgumentTypeError("--diff requires two distinct variants")
    return parts[0], parts[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extraction parity experiment (any pair from sonnet/haiku/gemma)",
    )
    parser.add_argument(
        "--extract",
        type=_parse_extract,
        default=None,
        help="Comma-separated variants to extract (e.g. 'sonnet,gemma' or 'haiku'). Choices: sonnet, haiku, gemma.",
    )
    parser.add_argument(
        "--diff",
        type=_parse_diff,
        default=None,
        help="Comma-separated pair to compare, A,B (e.g. 'sonnet,gemma'). A is the baseline side of the headline metric.",
    )
    parser.add_argument(
        "--gemma-prompt-version",
        default="v4",
        help="Gemma prompt version (default: v4). Ignored unless gemma is in --extract or --diff.",
    )
    parser.add_argument(
        "--haiku-prompt-version",
        default="v1",
        help="Haiku prompt version (default: v1). Ignored unless haiku is in --extract or --diff.",
    )
    parser.add_argument(
        "--prompt-id",
        default=None,
        help="Run for a single prompt (e.g. q3_software_capitalization)",
    )
    parser.add_argument(
        "--local-port",
        type=int,
        default=None,
        help="Port for local llama-server Gemma (e.g. 8080). Bypasses Vertex AI.",
    )
    parser.add_argument(
        "--passes",
        type=int,
        default=1,
        help="Number of Gemma extraction passes (default: 1). Additional passes find missed learnings.",
    )

    args = parser.parse_args()

    if not args.extract and not args.diff:
        parser.print_help()
        sys.exit(1)

    try:
        asyncio.run(
            src_main.run(
                extract=args.extract,
                gemma_version=args.gemma_prompt_version,
                haiku_version=args.haiku_prompt_version,
                prompt_id=args.prompt_id,
                diff=args.diff,
                local_port=args.local_port,
                passes=args.passes,
            )
        )
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main()
