# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

- Run experiment: `make run_experiment ARGS="<experiment_name> <command>"`
- Install deps: `make install` (will install uv if not present, then sync dependencies)
- Direct uv commands: `uv sync` (install dependencies), `uv run python <script>` (run with virtual environment)

## Code Style Guidelines

### Python
- **Imports**: Standard lib first, third-party second, internal modules last (grouped with blank lines)
- **Line length**: 119 characters max (configured in pyproject.toml)
- **Type hints**: Required for all function parameters and return values
- **Naming**: snake_case for functions/variables, PascalCase for classes, UPPER_CASE for constants
- **Async**: Use async/await for I/O operations; 'a'-prefixed functions for async variants (e.g., `aembed`)
- **Error handling**: Catch specific exceptions with clear context and fallbacks
- **Documentation**: Use docstrings for functions and classes (Google style preferred)

### Architecture
- Respect strict architectural boundaries between experiment modules
- Use Pydantic for data validation and configuration
- Each experiment should be self-contained in its own directory
- Prefer existing utilities in the common directory over duplicating functionality
- Always import modules at the top of the file

## Dependency Management

Python dependencies are managed with `uv`:
- Add dependency: `uv add [name]`
- Add dev dependency: `uv add --group development [name]`
- Install dependencies: `make install` or `uv sync`
- Update lockfile: `uv lock`
