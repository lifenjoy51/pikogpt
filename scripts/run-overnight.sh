#!/bin/bash
# Overnight pipeline: train TinyHelen -> sample from best checkpoint.
# Writes timestamped progress to run-logs/. Touches DONE marker on exit.
set -u
cd "$(dirname "$0")/.."
mkdir -p run-logs

LOG=run-logs/overnight.log
DONE=run-logs/overnight.done

rm -f "$DONE"

echo "=== $(date) START OVERNIGHT ===" | tee -a "$LOG"

# 1) Training
echo "=== $(date) TRAIN BEGIN ===" | tee -a "$LOG"
./gradlew --no-daemon runTinyHelenTrain 2>&1 | tee -a "$LOG"
TRAIN_EXIT=${PIPESTATUS[0]}
echo "=== $(date) TRAIN END (exit=$TRAIN_EXIT) ===" | tee -a "$LOG"

# 2) Sampling (only if training ok-ish — still run even on partial failure; sampler will error clearly)
echo "=== $(date) SAMPLE BEGIN ===" | tee -a "$LOG"
./gradlew --no-daemon runTinyHelenSample 2>&1 | tee -a "$LOG"
SAMPLE_EXIT=${PIPESTATUS[0]}
echo "=== $(date) SAMPLE END (exit=$SAMPLE_EXIT) ===" | tee -a "$LOG"

echo "=== $(date) OVERNIGHT DONE (train=$TRAIN_EXIT sample=$SAMPLE_EXIT) ===" | tee -a "$LOG"
# Touch marker with exit codes
echo "train=$TRAIN_EXIT sample=$SAMPLE_EXIT" > "$DONE"
