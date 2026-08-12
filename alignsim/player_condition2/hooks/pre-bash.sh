#!/bin/bash
set -euo pipefail

# PreToolUse allowlist for Bash commands.
#
# Only permits game CLI calls and basic utilities for piping/formatting output.
# Everything else (file I/O, code execution) should go through Claude Code's
# native Read/Write/Edit tools, which are separately constrainable.
#
# Two layers:
#   1. Deny patterns — hard blocks for game internals (catches creative workarounds)
#   2. Allowlist — per-segment first-word check; anything not listed is blocked

INPUT=$(cat)
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // empty')

if [ -z "$COMMAND" ]; then
  echo '{"decision":"block","reason":"Empty command."}'
  exit 0
fi

# ---------------------------------------------------------------------------
# Layer 1: Deny patterns — block references to game internals regardless of
# command structure. Catches subshells, variable expansion, etc.
# ---------------------------------------------------------------------------
DENY_PATTERNS=(
  'alignsim/src'
  'GAME_MECHANICS'
  'BENCHMARK\.md'
  'engine\.pkl'
  'pickle'
  'import alignsim'
  'from alignsim'
)

for pattern in "${DENY_PATTERNS[@]}"; do
  if echo "$COMMAND" | grep -qiE "$pattern"; then
    echo "{\"decision\":\"block\",\"reason\":\"Blocked: references game internals. Use ./game CLI commands to interact with the game.\"}"
    exit 0
  fi
done

# ---------------------------------------------------------------------------
# Layer 2: Allowlist — split on pipe/chain operators, check each segment
# ---------------------------------------------------------------------------
SEGMENTS=$(echo "$COMMAND" | sed 's/|/\n/g; s/&&/\n/g; s/;/\n/g')

while IFS= read -r segment; do
  segment=$(echo "$segment" | xargs 2>/dev/null || true)
  [ -z "$segment" ] && continue

  first_word=$(echo "$segment" | awk '{print $1}')

  case "$first_word" in
    ./game)                              ;; # Game CLI
    jq|sort|uniq|grep|tr|cut|wc|tee)     ;; # Piping/formatting game output
    head|tail)                           ;; # Truncating game output
    cat)
      # cat: only relative paths, no traversal above workspace
      if echo "$segment" | grep -qE '(^|\s)(\.\./|/|~)'; then
        echo '{"decision":"block","reason":"cat: only relative paths within the workspace are allowed. Use Read tool for other files."}'
        exit 0
      fi
      ;;
    ls|pwd|echo|printf|mkdir)            ;; # Basic workspace utilities
    *)
      echo "{\"decision\":\"block\",\"reason\":\"'$first_word' is not on the allowlist. Permitted: ./game, jq, cat (relative paths), ls, pwd, echo, sort, grep, head, tail, cut, tr, wc, tee, mkdir.\"}"
      exit 0
      ;;
  esac
done <<< "$SEGMENTS"

# All segments passed — allow
exit 0
