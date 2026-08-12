#!/bin/bash
# Post-submit hook: injects reflection context after each turn submission.
# Claude Code reads the hookSpecificOutput and uses additionalContext as guidance.
echo '{"hookSpecificOutput":{"hookEventName":"PostToolUse","additionalContext":"Turn complete. Before proceeding: 1) Review what happened — any rejections? deals closed? bugs? churn? 2) Update notes/strategy.md with learnings and key metrics. 3) Check which customers are closest to closing (compute satisfaction). 4) Plan your next turn before observing."}}'
