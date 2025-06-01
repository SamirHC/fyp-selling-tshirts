#!/bin/bash

set -e

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="$BASE_DIR/venv/bin/activate"
ETL_SCRIPT="$BASE_DIR/scripts/run_etl.sh"
FEATURE_SCRIPT="$BASE_DIR/scripts/run_feature_extraction.sh"

ETL_SCHEDULE="0 * * * *"      # Every hour, 0th minute
FEATURE_SCHEDULE="0 * * * *"  # Every hour, 0th minute

{
    echo "$ETL_SCHEDULE $ETL_SCRIPT"
    echo "$FEATURE_SCHEDULE $FEATURE_SCRIPT --load"
} | crontab -

echo "Cron jobs setup."
