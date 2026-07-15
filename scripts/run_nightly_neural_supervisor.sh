#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=/home/wis/Desarrollo/RNE_v16_worktrees/neural-agent-suite
PYTHON=/home/wis/Desarrollo/RNE_v16/.venv/bin/python
ENV_FILE=/home/wis/Desarrollo/RNE_v16/.env
SCHEDULER_LOG_ROOT=/home/wis/Desarrollo/RNE_v16/rnfe_artifacts/integral_campaigns/scheduler_logs

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"
mkdir -p "$SCHEDULER_LOG_ROOT"
SCHEDULER_LOG="$SCHEDULER_LOG_ROOT/nightly-$(date +%Y%m%d-%H%M%S).log"

exec "$PYTHON" scripts/supervise_integral_neural_campaign.py \
  --env-file "$ENV_FILE" \
  --max-attempts 4 \
  --max-wall-minutes 600 \
  --minimum-free-gb 20 \
  --maximum-gpu-temperature-c 88 \
  --monitor-interval-s 15 \
  --auto-stage-qualified-shadow \
  >>"$SCHEDULER_LOG" 2>&1
