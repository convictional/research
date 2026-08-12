#!/bin/bash
set -euo pipefail

# benchmark.sh — Run multiple AlignSim games with incrementing seeds.
#
# Usage:
#   ./scripts/benchmark.sh --condition c2 --runs 5 --model claude-opus-4-6
#   ./scripts/benchmark.sh --condition c3 --runs 3 --model claude-sonnet-4-6 --parallel
#   ./scripts/benchmark.sh --condition c2 --runs 5 --seed-start 200 --dry-run
#   ./scripts/benchmark.sh --condition c3 --runs 3 --model claude-opus-4-7 --thinking high
#
# Iterates through seeds and delegates to sandbox_run_condition2.sh or
# sandbox_run_condition3.sh. Use --parallel to launch all runs concurrently
# with staggered starts.
#
# --thinking forwards a reasoning level (off|minimal|low|medium|high|xhigh; default high) to all
# runs. It drives BOTH harnesses — Pi's --thinking flag and Claude Code's effortLevel — so reasoning
# is on at the chosen level for every reasoning-capable model (Gemma is the lone non-reasoner and
# ignores it).
#
# --auth forwards the auth mode (api-key|subscription) to all runs. Because it is
# parsed once and forwarded to every seed, all runs in a session share the same
# auth. subscription mode (claude-code only) bills a claude.ai subscription via
# CLAUDE_CODE_OAUTH_TOKEN; see sandbox_run_condition2.sh for setup. Note: running
# many runs in --parallel against one subscription will hit plan usage limits far
# sooner than API billing would — c3 especially, where each run is 3-5 agents.

CONDITION=""
RUNS=""
SEED_START=100
PARALLEL=false
DRY_RUN=false

PASSTHROUGH_FLAGS=()

G='\033[32m'
B='\033[1m'
RED='\033[31m'
R='\033[0m'
step() { echo -e "\n${B}==> $1${R}"; }
done_msg() { echo -e "${G}${B}✓ $1${R}"; }
err() { echo -e "${RED}✗ $1${R}" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --condition)   CONDITION="$2"; shift 2 ;;
    --runs)        RUNS="$2"; shift 2 ;;
    --seed-start)  SEED_START="$2"; shift 2 ;;
    --parallel)    PARALLEL=true; shift ;;
    --dry-run)     DRY_RUN=true; shift ;;
    --model)       PASSTHROUGH_FLAGS+=(--model "$2"); shift 2 ;;
    --harness)     PASSTHROUGH_FLAGS+=(--harness "$2"); shift 2 ;;
    --scenario)    PASSTHROUGH_FLAGS+=(--scenario "$2"); shift 2 ;;
    --max-turns)   PASSTHROUGH_FLAGS+=(--max-turns "$2"); shift 2 ;;
    --skip-db)     PASSTHROUGH_FLAGS+=(--skip-db); shift ;;
    --thinking)    PASSTHROUGH_FLAGS+=(--thinking "$2"); shift 2 ;;
    --auth)        PASSTHROUGH_FLAGS+=(--auth "$2"); shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

[ -z "$CONDITION" ] && err "Missing --condition (c2, c3, c4a, or c4b)"
[ -z "$RUNS" ] && err "Missing --runs N"
[[ "$RUNS" =~ ^[0-9]+$ ]] || err "--runs must be a positive integer"
[ "$RUNS" -gt 0 ] || err "--runs must be > 0"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "$CONDITION" in
  c2|condition2)
    TARGET_SCRIPT="$SCRIPT_DIR/sandbox_run_condition2.sh"
    CONDITION_LABEL="condition2"
    ;;
  c3|condition3)
    TARGET_SCRIPT="$SCRIPT_DIR/sandbox_run_condition3.sh"
    CONDITION_LABEL="condition3"
    ;;
  c4a|condition4a)
    TARGET_SCRIPT="$SCRIPT_DIR/sandbox_run_condition4.sh"
    CONDITION_LABEL="condition4a"
    PASSTHROUGH_FLAGS+=(--substrate channels)
    ;;
  c4b|condition4b)
    TARGET_SCRIPT="$SCRIPT_DIR/sandbox_run_condition4.sh"
    CONDITION_LABEL="condition4b"
    PASSTHROUGH_FLAGS+=(--substrate convictional)
    ;;
  *) err "Unknown condition '$CONDITION'. Valid: c2, c3, c4a, c4b" ;;
esac

[ -f "$TARGET_SCRIPT" ] || err "Target script not found: $TARGET_SCRIPT"

# ---------------------------------------------------------------------------
# Session setup
# ---------------------------------------------------------------------------

SESSION_ID="bench_${CONDITION_LABEL}_$(date +%Y%m%d_%H%M%S)_$$"
LOG_DIR="$SCRIPT_DIR/../results/benchmarks/$SESSION_ID"
mkdir -p "$LOG_DIR"

echo -e "${B}AlignSim Benchmark${R}"
echo "  Condition:  $CONDITION_LABEL"
echo "  Runs:       $RUNS"
echo "  Seeds:      $SEED_START .. $((SEED_START + RUNS - 1))"
echo "  Parallel:   $PARALLEL"
echo "  Logs:       $LOG_DIR/"
echo "  Passthrough: ${PASSTHROUGH_FLAGS[*]+"${PASSTHROUGH_FLAGS[*]}"}"
echo ""

# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------

if [ "$DRY_RUN" = true ]; then
  step "Dry run — commands that would execute:"
  for i in $(seq 0 $((RUNS - 1))); do
    SEED=$((SEED_START + i))
    echo "  $TARGET_SCRIPT --seed $SEED ${PASSTHROUGH_FLAGS[*]+"${PASSTHROUGH_FLAGS[*]}"}"
  done
  echo ""
  echo "  No runs executed."
  exit 0
fi

# ---------------------------------------------------------------------------
# Tracking arrays
# ---------------------------------------------------------------------------

declare -a SEEDS=()
declare -a EXIT_CODES=()
declare -a DURATIONS=()
declare -a RUN_IDS=()
declare -a PIDS=()
declare -a START_TIMES=()

BENCH_START=$(date +%s)

extract_run_id() {
  local log_file="$1"
  local rid
  rid=$(grep 'Run ID:' "$log_file" 2>/dev/null | tail -1 | sed 's/.*Run ID:[[:space:]]*//' | tr -d '[:space:]')
  echo "${rid:-unknown}"
}

# ---------------------------------------------------------------------------
# Sequential mode
# ---------------------------------------------------------------------------

if [ "$PARALLEL" = false ]; then
  for i in $(seq 0 $((RUNS - 1))); do
    SEED=$((SEED_START + i))
    SEEDS+=("$SEED")
    LOG_FILE="$LOG_DIR/seed_${SEED}.log"

    step "Run $((i + 1))/$RUNS (seed=$SEED)"
    RUN_START=$(date +%s)

    set +e
    "$TARGET_SCRIPT" --seed "$SEED" ${PASSTHROUGH_FLAGS[@]+"${PASSTHROUGH_FLAGS[@]}"} 2>&1 | tee "$LOG_FILE"
    EXIT_CODE=${PIPESTATUS[0]}
    set -e

    RUN_END=$(date +%s)
    DURATION=$((RUN_END - RUN_START))

    EXIT_CODES+=("$EXIT_CODE")
    DURATIONS+=("$DURATION")
    RUN_IDS+=("$(extract_run_id "$LOG_FILE")")

    if [ "$EXIT_CODE" -eq 0 ]; then
      done_msg "Seed $SEED completed in ${DURATION}s"
    else
      echo -e "${RED}✗ Seed $SEED failed (exit $EXIT_CODE) after ${DURATION}s${R}"
    fi
  done

# ---------------------------------------------------------------------------
# Parallel mode
# ---------------------------------------------------------------------------

else
  STAGGER_DELAY=30

  for i in $(seq 0 $((RUNS - 1))); do
    SEED=$((SEED_START + i))
    SEEDS+=("$SEED")
    LOG_FILE="$LOG_DIR/seed_${SEED}.log"
    START_TIMES+=("$(date +%s)")

    step "Launching seed=$SEED (run $((i + 1))/$RUNS)"
    "$TARGET_SCRIPT" --seed "$SEED" ${PASSTHROUGH_FLAGS[@]+"${PASSTHROUGH_FLAGS[@]}"} > "$LOG_FILE" 2>&1 &
    PIDS+=($!)

    # Stagger starts (skip delay after last launch)
    if [ "$i" -lt $((RUNS - 1)) ]; then
      echo "  Waiting ${STAGGER_DELAY}s before next launch..."
      sleep "$STAGGER_DELAY"
    fi
  done

  done_msg "All $RUNS runs launched"

  echo ""
  echo "  Runs are backgrounded — no output until they finish."
  echo "  Process IDs:"
  for i in $(seq 0 $((RUNS - 1))); do
    echo "    seed=${SEEDS[$i]}  pid=${PIDS[$i]}  log=$LOG_DIR/seed_${SEEDS[$i]}.log"
  done
  echo ""
  echo "  To check progress, shell into the sandbox and look at the run dir:"
  echo "    limactl shell lima-decide"
  echo "    cd ~/game && ls -t | head    # find the latest run dirs"
  echo "    tail -1 <run_dir>/orchestrator_data/turn_record.jsonl | python3 -m json.tool"
  echo ""
  echo "  Or tail a log from the host:"
  echo "    tail -f $LOG_DIR/seed_<N>.log"
  echo ""

  step "Waiting for runs to complete..."

  # Wait for each process individually to capture exit codes
  for i in $(seq 0 $((RUNS - 1))); do
    set +e
    wait "${PIDS[$i]}"
    EXIT_CODE=$?
    set -e

    SEED="${SEEDS[$i]}"
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - ${START_TIMES[$i]}))

    EXIT_CODES+=("$EXIT_CODE")
    DURATIONS+=("$DURATION")
    RUN_IDS+=("$(extract_run_id "$LOG_DIR/seed_${SEED}.log")")

    if [ "$EXIT_CODE" -eq 0 ]; then
      done_msg "Seed $SEED completed in ${DURATION}s"
    else
      echo -e "${RED}✗ Seed $SEED failed (exit $EXIT_CODE) after ${DURATION}s${R}"
    fi
  done
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

BENCH_END=$(date +%s)
BENCH_DURATION=$((BENCH_END - BENCH_START))

PASS_COUNT=0
FAIL_COUNT=0
for code in "${EXIT_CODES[@]}"; do
  if [ "$code" -eq 0 ]; then
    PASS_COUNT=$((PASS_COUNT + 1))
  else
    FAIL_COUNT=$((FAIL_COUNT + 1))
  fi
done

echo ""
echo -e "${B}╔══════════════════════════════════════════════════════════════════╗${R}"
echo -e "${B}║                      Benchmark Summary                         ║${R}"
echo -e "${B}╠══════════════════════════════════════════════════════════════════╣${R}"
printf "${B}║${R}  %-8s  %-8s  %-10s  %-34s${B}║${R}\n" "Seed" "Status" "Duration" "Run ID"
echo -e "${B}╠══════════════════════════════════════════════════════════════════╣${R}"

for i in $(seq 0 $((RUNS - 1))); do
  SEED="${SEEDS[$i]}"
  CODE="${EXIT_CODES[$i]}"
  DUR="${DURATIONS[$i]}"
  RID="${RUN_IDS[$i]}"

  # Format duration as Xm Ys
  DUR_MIN=$((DUR / 60))
  DUR_SEC=$((DUR % 60))
  DUR_FMT="${DUR_MIN}m ${DUR_SEC}s"

  if [ "$CODE" -eq 0 ]; then
    STATUS="${G}PASS${R}"
  else
    STATUS="${RED}FAIL($CODE)${R}"
  fi

  printf "${B}║${R}  %-8s  %-19s  %-10s  %-34s${B}║${R}\n" "$SEED" "$STATUS" "$DUR_FMT" "$RID"
done

echo -e "${B}╠══════════════════════════════════════════════════════════════════╣${R}"

BENCH_MIN=$((BENCH_DURATION / 60))
BENCH_SEC=$((BENCH_DURATION % 60))
printf "${B}║${R}  Total: %d passed, %d failed | Wall time: %dm %ds                ${B}║${R}\n" \
  "$PASS_COUNT" "$FAIL_COUNT" "$BENCH_MIN" "$BENCH_SEC"
printf "${B}║${R}  Logs: %-57s${B}║${R}\n" "$LOG_DIR/"
echo -e "${B}╚══════════════════════════════════════════════════════════════════╝${R}"

if [ "$FAIL_COUNT" -gt 0 ]; then
  exit 1
fi
