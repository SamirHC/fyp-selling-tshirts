#!/bin/bash

set -e

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="$BASE_DIR/venv/bin/activate"
ETL_PATH="$BASE_DIR/src/data_collection/etl.py"


echo "Running ETL Pipeline"

cd $BASE_DIR
source $VENV_PATH
python $ETL_PATH --ebay_browse
deactivate

echo "ETL Pipeline completed"
