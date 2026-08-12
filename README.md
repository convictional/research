# Research Experiments

This directory is the research log of **Convictional** and spans approximately 2024 to mid-2026.

What follows is roughly two years of applied LLM research, kept as ~50 self-contained "mini
codebases". Some are substantial engineering efforts; some are a single markdown file recording
what was tried and what happened. The research largely explores applied AI for knowledge work (e.g. deep research), knowledge-retrieval and structures for RAG, and goal-alignment of knowledge-work teams. No attempt has been made to tidy them into one.

## Read this before you read anything else

**These are point-in-time experiments and none of them are maintained.** Each was written against
the model versions, prices, libraries, and product assumptions of the week it was run. Dates in
this directory run from roughly 2024 through mid-2026. Model comparisons in particular go stale
fast — treat every benchmark number as a snapshot, not a current claim.

**The input data has been removed.** Almost every experiment here ran against Convictional's own
production data: meeting transcripts, decision records, goal boards, email, engineering activity.
That data contained personal information about employees, customers, and third parties, and it was
deleted before publication rather than anonymized. Where a writeup quoted real content, the quote
has been replaced with a clearly-labelled synthetic equivalent, and the substitution is noted in
place. **Consequence: most experiments here cannot be reproduced from this repository.** You can
read the method and the code; you cannot re-run the result. Individual READMEs say what is missing.

**Nothing here was production code.** No test coverage requirements and minimal maintainability
needs, code reviews focused on design, consistentcy testing and interpretation. There was little
architectural consistency across experiments. Several experiments contain approaches that were
tried and abandoned, which is the point of keeping them.

## Layout

Each subdirectory is one experiment or one group of related experiments. They are independent: a
subdirectory may be a full mini-codebase with its own `pyproject.toml`, or just SQL queries, or just
writing. The only convention is a per-directory `README.md` describing what it is.

`common/` holds shared helpers (LLM clients, prompt templating, embeddings, IO) that several of the
Python experiments import.

`CLAUDE.md` files scattered through the tree are instructions for AI coding agents working in this
repo. They are kept because they document conventions, and because a lot of this code was written
with agent assistance.

## Running something

Most experiments will not run without the data that was removed, and many need cloud credentials.
With that caveat:

### Prerequisites

- Python (see `requires-python` in `pyproject.toml`) and [`uv`](https://docs.astral.sh/uv/).
- For anything touching BigQuery or Vertex AI, the
  [`gcloud` CLI](https://cloud.google.com/sdk/docs/install), authenticated.
- API keys for whichever providers a given experiment uses (Anthropic, OpenAI, Google).

Some experiments store large files (figures, graphs, CSVs) with [Git LFS](https://git-lfs.com/).
Install it and run `git lfs install` before cloning if you want those, otherwise you will get
pointer files:

```bash
brew install git-lfs   # or your platform's package manager
git lfs install
```

### Configuration

Copy the example environment file and fill in what you need:

```bash
cp .env.example .env
```

Put credentials in a separate `.env.secrets` file in this directory — it is gitignored and is
never committed:

```
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
GOOGLE_API_KEY=...
```

Both files are optional as far as `make` is concerned; individual experiments will fail with a
clear error if a key they need is missing.

### Commands

From this directory:

```bash
make install                                    # install dependencies, create the venv
make auth                                       # authenticate with Google Cloud
make run_experiment ARGS="knowledge_graphs"      # run an experiment's __main__.py
make help                                        # list targets
```

`make run_experiment` relies on the experiment having a `__main__.py` entrypoint. Experiments with
their own `pyproject.toml` (`geo-analyzer`, `alignsim`, `graphify_exploration`, and others) have
their own Makefile or CLI instead — see their READMEs.

Dependencies are managed with `uv` against the shared `pyproject.toml` here. A few experiments
needed conflicting versions and so carry their own.
