import pandas as pd
from typing import Any


# ANSI escape codes for colors
class Colors:
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    WHITE = "\033[97m"
    RESET = "\033[0m"


def print_section(title: str, content: Any = None, color: str = Colors.CYAN) -> None:
    """Print a section header and optional content with consistent formatting"""
    print("\n" + "=" * 80)
    print(f"{color}{title}{Colors.RESET}")
    print("=" * 80)
    if content is not None:
        if isinstance(content, pd.DataFrame):
            print(f"\n{content.head().to_string()}")
            print(f"\nShape: {content.shape}")
        else:
            print(f"\n{content}")
