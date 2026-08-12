#!/bin/bash
set -euo pipefail

# PreToolUse allowlist for file-access tools (Read, Glob, Grep, Write, Edit).
#
# Restricts file access to the player workspace (~/game/).
# Blocks access to game source, engine internals, and anything outside the workspace.

INPUT=$(cat)
TOOL_NAME=$(echo "$INPUT" | jq -r '.tool_name // empty')

# Extract the path-like field depending on the tool
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

# Also check pattern fields (Glob pattern, Grep pattern) for source references
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

# Empty path = current directory (~/game/) — always fine
if [ -z "$TARGET" ]; then
  exit 0
fi

# Resolve relative paths: if it doesn't start with / or ~, it's relative to cwd (~/game/) — fine
if [[ ! "$TARGET" =~ ^[/~] ]]; then
  # Block traversal above workspace
  if echo "$TARGET" | grep -qE '(^|/)\.\.(/|$)'; then
    echo '{"decision":"block","reason":"Path traversal outside workspace is not allowed."}'
    exit 0
  fi
  exit 0
fi

# Absolute paths: must be under ~/game/
# Expand ~ to $HOME for comparison
RESOLVED="${TARGET/#\~/$HOME}"

if [[ "$RESOLVED" == "$HOME/game"* ]]; then
  exit 0
fi

echo "{\"decision\":\"block\",\"reason\":\"File access is restricted to the player workspace (~/game/). Path '$TARGET' is outside the workspace.\"}"
exit 0
