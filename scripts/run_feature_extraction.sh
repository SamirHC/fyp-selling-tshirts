#!/bin/bash

set -e

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="$BASE_DIR/venv/bin/activate"
SCRIPT_PATH="$BASE_DIR/src/feature_extract/load_features.py"


echo "Running Feature Extraction Pipeline"

cd $BASE_DIR
source $VENV_PATH
python $SCRIPT_PATH $@
deactivate

echo "Feature Extraction Pipeline completed"
