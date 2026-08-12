import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm.gpt_oss import stream_gpt_oss_response


async def test_single_request() -> None:
    """Simple test of a single API request."""
    import dotenv
    import os
    from pathlib import Path

    dotenv.load_dotenv(Path(__file__).parent.parent / ".env.secrets")

    print(f"Project ID: {os.getenv('GOOGLE_CLOUD_PROJECT')}")

    headers = {
        "Authorization": f"Bearer {os.getenv('GOOGLE_ACCESS_TOKEN')}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "openai/gpt-oss-120b-maas",
        "stream": True,
        "messages": [
            {
                "role": "user",
                "content": "Write a one-liner about convictional.com's mission.",
            }
        ],
    }

    response_text = await stream_gpt_oss_response(headers, payload)
    print("\nFull Response:")
    print(response_text)


def main() -> None:
    """Entry point that routes to benchmark CLI if args are present, otherwise runs simple test."""
    if len(sys.argv) > 1:
        from src.cli import main as cli_main

        cli_main()
    else:
        print("Running simple test (use CLI args to run benchmark)...")
        print("Example: python -m src.main --requests 10 --concurrency 2\n")
        asyncio.run(test_single_request())


if __name__ == "__main__":
    main()
