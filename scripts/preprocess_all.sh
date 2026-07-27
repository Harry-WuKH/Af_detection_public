#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

"${PYTHON_BIN}" preprocess/preprocess_afdb.py
"${PYTHON_BIN}" preprocess/preprocess_mitdb.py
"${PYTHON_BIN}" preprocess/preprocess_ltafdb.py

