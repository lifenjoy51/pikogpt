#!/bin/bash
# Vec backend end-to-end run: runTinyHelenTrainVec → runTinyHelenSampleVec.
# Writes tee'd progress to run-logs/vec-overnight.log and marks
# run-logs/vec-overnight.done on exit (with individual task exit codes).
set -u
cd "$(dirname "$0")/.."
mkdir -p run-logs

LOG=run-logs/vec-overnight.log
DONE=run-logs/vec-overnight.done

rm -f "$DONE"
: > "$LOG"

echo "=== $(date) START VEC RUN ===" | tee -a "$LOG"

echo "=== $(date) TRAIN BEGIN ===" | tee -a "$LOG"
./gradlew --no-daemon runTinyHelenTrainVec 2>&1 | tee -a "$LOG"
TRAIN_EXIT=${PIPESTATUS[0]}
echo "=== $(date) TRAIN END (exit=$TRAIN_EXIT) ===" | tee -a "$LOG"

echo "=== $(date) SAMPLE BEGIN ===" | tee -a "$LOG"
./gradlew --no-daemon runTinyHelenSampleVec 2>&1 | tee -a "$LOG"
SAMPLE_EXIT=${PIPESTATUS[0]}
echo "=== $(date) SAMPLE END (exit=$SAMPLE_EXIT) ===" | tee -a "$LOG"

echo "=== $(date) VEC RUN DONE (train=$TRAIN_EXIT sample=$SAMPLE_EXIT) ===" | tee -a "$LOG"
echo "train=$TRAIN_EXIT sample=$SAMPLE_EXIT" > "$DONE"
