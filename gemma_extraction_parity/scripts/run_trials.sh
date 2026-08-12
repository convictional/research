#!/usr/bin/env bash
# Run N trials of extract + 3 pairwise diffs sequentially.
#
# Each trial produces 3 extraction caches and 3 parity reports. After every
# trial those artifacts are moved into output/trial_${N}/ so the next trial
# starts with a clean output/ and we end up with N independent samples for
# characterizing extraction + judge variance.
#
# Usage:
#   scripts/run_trials.sh                  # 3 trials starting at 1
#   scripts/run_trials.sh 5                # 5 trials starting at 1
#   scripts/run_trials.sh 3 2              # 3 trials starting at 2 (resume)
#
# Resumable: if output/trial_${N}/ already exists, that trial is skipped.

set -euo pipefail

CYCLES="${1:-3}"
START="${2:-1}"

GEMMA_VERSION="v6_1"
HAIKU_VERSION="v1"
PASSES="2"

EXP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
EXPERIMENTS_DIR="$(cd "$EXP_DIR/.." && pwd)"
OUTPUT_DIR="$EXP_DIR/output"
LOG_DIR="$OUTPUT_DIR/trial_logs"
mkdir -p "$LOG_DIR"

EXTRACTION_FILES=(
    "sonnet_extraction.json"
    "haiku_${HAIKU_VERSION}_extraction.json"
    "gemma_${GEMMA_VERSION}_p${PASSES}_extraction.json"
    "search_results.json"
)

run_cmd() {
    local label="$1"; shift
    local log="$1"; shift
    echo "    [$(date '+%H:%M:%S')] ▶ ${label}"
    (cd "$EXPERIMENTS_DIR" && PYTHONPATH=. uv run python gemma_extraction_parity "$@") \
        2>&1 | tee -a "$log"
    echo "    [$(date '+%H:%M:%S')] ✓ ${label}"
}

for ((i = START; i < START + CYCLES; i++)); do
    TRIAL_DIR="$OUTPUT_DIR/trial_${i}"
    LOG="$LOG_DIR/trial_${i}.log"

    if [ -d "$TRIAL_DIR" ]; then
        echo "[$(date '+%H:%M:%S')] ⏭  Trial ${i} already exists at $TRIAL_DIR — skipping"
        continue
    fi

    echo "============================================================"
    echo "[$(date '+%H:%M:%S')] ▶ Trial ${i} (logging to $LOG)"
    echo "============================================================"

    # Clear any stale extraction artifacts from output/ before extracting.
    for f in "${EXTRACTION_FILES[@]}"; do
        rm -f "$OUTPUT_DIR/$f"
    done

    run_cmd "Trial ${i}: extract sonnet,haiku,gemma (passes=$PASSES)" "$LOG" \
        --extract sonnet,haiku,gemma \
        --haiku-prompt-version "$HAIKU_VERSION" \
        --gemma-prompt-version "$GEMMA_VERSION" \
        --passes "$PASSES"

    run_cmd "Trial ${i}: diff sonnet vs gemma" "$LOG" \
        --diff sonnet,gemma \
        --gemma-prompt-version "$GEMMA_VERSION" \
        --passes "$PASSES"

    run_cmd "Trial ${i}: diff haiku vs gemma" "$LOG" \
        --diff haiku,gemma \
        --haiku-prompt-version "$HAIKU_VERSION" \
        --gemma-prompt-version "$GEMMA_VERSION" \
        --passes "$PASSES"

    run_cmd "Trial ${i}: diff sonnet vs haiku" "$LOG" \
        --diff sonnet,haiku \
        --haiku-prompt-version "$HAIKU_VERSION"

    mkdir -p "$TRIAL_DIR"
    for f in "${EXTRACTION_FILES[@]}"; do
        [ -f "$OUTPUT_DIR/$f" ] && mv "$OUTPUT_DIR/$f" "$TRIAL_DIR/"
    done
    # Move parity reports produced during this trial. They're auto-timestamped,
    # so we just sweep anything matching parity_*.md that landed in output/
    # since the trial started.
    find "$OUTPUT_DIR" -maxdepth 1 -name "parity_*.md" -newer "$LOG" -exec mv {} "$TRIAL_DIR/" \;

    echo "[$(date '+%H:%M:%S')] ✓ Trial ${i} complete → $TRIAL_DIR"
    ls "$TRIAL_DIR"
done

echo
echo "============================================================"
echo "All trials complete. Artifacts per trial:"
for ((i = START; i < START + CYCLES; i++)); do
    echo "  output/trial_${i}/"
done
