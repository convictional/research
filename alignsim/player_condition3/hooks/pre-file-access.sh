#!/bin/bash
set -euo pipefail

# PreToolUse allowlist for file-access tools (Read, Glob, Grep, Write, Edit).
#
# Restricts file access to the agent's workspace (read from .workspace_root).
# Blocks access to game source, engine internals, and anything outside the workspace.

INPUT=$(cat)
TOOL_NAME=$(echo "$INPUT" | jq -r '.tool_name // empty')

case "$TOOL_NAME" in
  Read)
    TARGET=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')
    ;;
  Glob)
    TARGET=$(echo "$INPUT" | jq -r '.tool_input.path // empty')
    ;;
  Grep)
    TARGET=$(echo "$INPUT" | jq -r '.tool_input.path // empty')
    ;;
  Write|Edit)
    TARGET=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')
    ;;
  *)
    TARGET=""
    ;;
esac

PATTERN=$(echo "$INPUT" | jq -r '.tool_input.pattern // empty')

# ---------------------------------------------------------------------------
# Deny: references to game internals in any field
# ---------------------------------------------------------------------------
DENY_PATTERNS=(
  'alignsim/src'
  'GAME_MECHANICS'
  'BENCHMARK\.md'
  'engine\.pkl'
  'pickle'
  '/experiments/'
)

COMBINED="$TARGET $PATTERN"
for deny in "${DENY_PATTERNS[@]}"; do
  if echo "$COMBINED" | grep -qiE "$deny"; then
    echo '{"decision":"block","reason":"Blocked: references game internals. Use ./game CLI commands to interact with the game."}'
    exit 0
  fi
done

# ---------------------------------------------------------------------------
# Allow: paths within the player workspace only
# ---------------------------------------------------------------------------

if [ -z "$TARGET" ]; then
  exit 0
fi

if [[ ! "$TARGET" =~ ^[/~] ]]; then
  if echo "$TARGET" | grep -qE '(^|/)\.\.(/|$)'; then
    echo '{"decision":"block","reason":"Path traversal outside workspace is not allowed."}'
    exit 0
  fi
  exit 0
fi

RESOLVED="${TARGET/#\~/$HOME}"
RESOLVED=$(realpath -m "$RESOLVED" 2>/dev/null || echo "$RESOLVED")

WORKSPACE_ROOT=$(cat .workspace_root 2>/dev/null || true)
if [ -n "$WORKSPACE_ROOT" ]; then
  WORKSPACE_ROOT=$(realpath -m "$WORKSPACE_ROOT" 2>/dev/null || echo "$WORKSPACE_ROOT")
  [[ "$RESOLVED" == "$WORKSPACE_ROOT"* ]] && exit 0
elif [[ "$RESOLVED" == "$HOME/game"* ]]; then
  exit 0
fi

echo "{\"decision\":\"block\",\"reason\":\"File access is restricted to your workspace. Path '$TARGET' is outside the workspace.\"}"
exit 0
