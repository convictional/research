PYTHON := $(shell dirname $(abspath $(lastword $(MAKEFILE_LIST))))/../.venv/bin/python3
EXPERIMENTS_DIR := $(shell dirname $(abspath $(lastword $(MAKEFILE_LIST))))/../

.PHONY: run_experiment

# Usage: make run_experiment ARGS="decisions_to_goals <subcommand> [flags]"
# The first word of ARGS is the experiment module name; remaining words are passed as subcommand + flags.
run_experiment:
	cd "$(EXPERIMENTS_DIR)" && $(PYTHON) -m $(wordlist 1,1,$(ARGS)) $(wordlist 2,$(words $(ARGS)),$(ARGS))
