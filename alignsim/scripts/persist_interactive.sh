#!/bin/bash
set -euo pipefail

# persist_interactive.sh — recover an interactive (e.g. human-guided) Condition 2 run.
#
# Interactive runs drop you into the sandbox shell and never execute the autonomous
# wrap-up (score → collect → persist), so their game state stays in the sandbox at
# ~/game/<run_id>/state and nothing lands in the DB. This pulls that state out, scores
# it ON THE HOST (so the composite uses the current scoring code), assembles a results
# dir exactly like the autonomous collector, and persists it — tagged with a player_type
# (default human_guided) so a human-in-the-loop run is never pooled with the autonomous
# llm_agent grid.
#
# Usage: ./scripts/persist_interactive.sh <run_id> [--player-type human_guided|llm_agent|human]

SANDBOX_NAME="decide-sandbox"
SANDBOX_SSH="limactl shell $SANDBOX_NAME --"

RUN_ID=""
PLAYER_TYPE="human_guided"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --player-type) PLAYER_TYPE="$2"; shift 2 ;;
    -h|--help) echo "Usage: $0 <run_id> [--player-type human_guided|llm_agent|human]"; exit 0 ;;
    -*) echo "Unknown flag: $1" >&2; exit 1 ;;
    *) [ -z "$RUN_ID" ] && RUN_ID="$1" || { echo "Unexpected argument: $1" >&2; exit 1; }; shift ;;
  esac
done

[ -z "$RUN_ID" ] && { echo "Usage: $0 <run_id> [--player-type ...]" >&2; exit 1; }
command -v limactl >/dev/null 2>&1 || { echo "Error: Lima not installed (brew install lima)." >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ALIGNSIM_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
EXPERIMENTS_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_DIR="$ALIGNSIM_DIR/results/$RUN_ID"

echo "==> Recovering interactive run: $RUN_ID (player_type=$PLAYER_TYPE)"

# The game dir lives in the sandbox home; $HOME must expand IN the VM, so it is escaped.
$SANDBOX_SSH bash -lc "test -d \$HOME/game/$RUN_ID/state" \
  || { echo "Error: \$HOME/game/$RUN_ID/state not found in sandbox '$SANDBOX_NAME'." >&2; exit 1; }

mkdir -p "$RESULTS_DIR/state"

# Pull the game state (engine.pkl + jsonl) out of the sandbox — tar over the pipe is binary-safe.
$SANDBOX_SSH bash -lc "tar -cf - -C \$HOME/game/$RUN_ID/state ." | tar -xf - -C "$RESULTS_DIR/state"
$SANDBOX_SSH bash -lc "cat \$HOME/game/$RUN_ID/run_metadata.json" > "$RESULTS_DIR/run_metadata.json"
[ -s "$RESULTS_DIR/run_metadata.json" ] || { echo "Error: run_metadata.json missing/empty in sandbox." >&2; exit 1; }

# Token usage: interactive runs have no autonomous stdout capture, but the harness leaves its own
# session transcript behind. For claude-code that's ~/.claude/projects/<mangled-cwd>/*.jsonl (the run
# dir path with every non-alphanumeric turned to a dash); for pi it's pi-session.jsonl in the run dir.
# parse_transcript_tokens dedups the claude session log by message.id (it repeats each message and has
# no result event), so summing it gives real per-message input+output. Named session.jsonl so run_logger
# prefers it (same convention the autonomous collectors now use).
HARNESS=$(jq -r '.harness // .agent_cli // "claude-code"' "$RESULTS_DIR/run_metadata.json" 2>/dev/null || echo "claude-code")
if [ "$HARNESS" = "pi" ]; then
  $SANDBOX_SSH bash -lc "cat \$HOME/game/$RUN_ID/pi-session.jsonl 2>/dev/null || true" > "$RESULTS_DIR/pi-session.jsonl"
  [ -s "$RESULTS_DIR/pi-session.jsonl" ] || rm -f "$RESULTS_DIR/pi-session.jsonl"
else
  MANGLED=$(printf '%s' "$RUN_ID" | sed 's/[^a-zA-Z0-9]/-/g')
  $SANDBOX_SSH bash -lc "cat ~/.claude/projects/*${MANGLED}/*.jsonl 2>/dev/null || true" > "$RESULTS_DIR/session.jsonl"
  [ -s "$RESULTS_DIR/session.jsonl" ] || rm -f "$RESULTS_DIR/session.jsonl"
fi
if [ -s "$RESULTS_DIR/session.jsonl" ] || [ -s "$RESULTS_DIR/pi-session.jsonl" ]; then
  echo "  Collected session transcript for token usage."
else
  echo "  No session transcript found — token_usage will be empty."
fi

# Score on the HOST so the composite uses the current scoring code (the sandbox checkout
# may predate scoring changes). game_cli status reads the pickled final engine state.
echo "==> Scoring on host"
( cd "$ALIGNSIM_DIR" && PYTHONPATH="$EXPERIMENTS_DIR" \
    uv run python -m alignsim.src.game_cli status --state-dir "$RESULTS_DIR/state" ) > "$RESULTS_DIR/final_status.json"

# Files the collector places at the results-dir root (persist-results reads them there).
for f in _internal_scores.json turn_record.jsonl game_log.jsonl; do
  [ -f "$RESULTS_DIR/state/$f" ] && cp "$RESULTS_DIR/state/$f" "$RESULTS_DIR/$f"
done

if command -v jq >/dev/null 2>&1; then
  GAME_OVER=$(jq -r '.game_over // false' "$RESULTS_DIR/final_status.json" 2>/dev/null || echo "?")
  [ "$GAME_OVER" != "true" ] && echo "  Note: game_over=$GAME_OVER — scoring a game that isn't finished."
fi

echo "==> Persisting to DB"
( cd "$EXPERIMENTS_DIR" && \
    uv run python -m alignsim persist-results \
      --results-dir "$RESULTS_DIR" --player-type "$PLAYER_TYPE" --run-mode interactive )

echo "==> Done. Persisted $RUN_ID as player_type=$PLAYER_TYPE (run_mode=interactive)."
