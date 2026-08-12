# Both of these are gitignored and absent on a fresh clone, so the includes are
# optional (`-include`). Copy `.env.example` to `.env` and create `.env.secrets`
# with your API keys; individual experiments document which ones they need.
-include .env
-include .env.secrets
export

.PHONY: help
help: ## Display available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

.PHONY: install
install: ## Ensures the project dependencies are installed
	@if ! uv --version >/dev/null 2>&1; then \
		echo "Installing uv..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	else \
		echo "uv is already installed"; \
	fi
	PYTHONPATH=. uv sync
	PYTHONPATH=. uv run pre-commit install

.PHONY: auth
auth: ## Authenticates the user with the Google Cloud Platform
	gcloud auth application-default login

.PHONY: run_experiment
run_experiment: ## Runs an experiment. Usage: make run_experiment ARGS="knowledge_graphs my_command"
	PYTHONPATH=. uv run python $(ARGS)
