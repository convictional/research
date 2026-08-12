.PHONY: help install refresh clean

help: ## Display available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Create local venv, install graphifyy, copy SKILL.md to .claude/skills/graphify/
	uv sync
	@mkdir -p .claude/skills/graphify
	@.venv/bin/python -c "import graphify, pathlib, shutil; shutil.copy(pathlib.Path(graphify.__file__).parent / 'skill.md', '.claude/skills/graphify/SKILL.md'); print('Copied SKILL.md from', pathlib.Path(graphify.__file__).parent / 'skill.md')"
	@echo ""
	@echo "Install complete."
	@echo "Activate the venv (. .venv/bin/activate) and launch Claude — /graphify will be available."

refresh: ## Pull latest graphifyy and re-copy SKILL.md (graphify ships daily)
	uv lock --upgrade-package graphifyy
	uv sync
	@.venv/bin/python -c "import graphify, pathlib, shutil; shutil.copy(pathlib.Path(graphify.__file__).parent / 'skill.md', '.claude/skills/graphify/SKILL.md')"
	@echo "Refreshed graphifyy and SKILL.md."

clean: ## Remove .venv, graphify-out/, and the installed graphify skill
	rm -rf .venv graphify-out .claude/skills/graphify
	@echo "Cleaned. Run 'make install' to re-install."
