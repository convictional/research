#!/bin/bash
# Post-submit hook: injects reflection context after each turn submission.
echo '{"hookSpecificOutput":{"hookEventName":"PostToolUse","additionalContext":"Turn submitted. Before proceeding: 1) Review events and rejections from the response. 2) Share what you did and learned on the relevant Post (comment, or open a new Post) — other agents need this info. 3) Read any unread Posts (./game post read <id>) and check ./game goals. 4) Update notes/strategy.md with key metrics and plan. 5) Coordinate your next turn with the team."}}'
