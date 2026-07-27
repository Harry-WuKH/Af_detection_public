#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PROJECT_ROOT

PATH_CONFIG="${PROJECT_ROOT}/configs/paths.env"
if [[ -f "${PATH_CONFIG}" ]]; then
    source "${PATH_CONFIG}"
else
    source "${PROJECT_ROOT}/configs/paths.env.example"
fi

export RAW_DATA_ROOT AFDB_RAW_DIR MITDB_RAW_DIR LTAFDB_RAW_DIR
export PROCESSED_DATA_ROOT RESULTS_ROOT PYTHON_BIN
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

RUNS="${RUNS:-3}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-64}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-8}"
BASELINE_MODELS="${BASELINE_MODELS:-resnet18 vgg19 inception}"
DANN_LAMBDA="${DANN_LAMBDA:-0.001}"
DANN_DOMAIN_STEPS="${DANN_DOMAIN_STEPS:-5}"
GRADCAM_FS="${GRADCAM_FS:-500}"
GRADCAM_SIGNAL_LENGTH="${GRADCAM_SIGNAL_LENGTH:-5000}"

segments_path() {
    local dataset="$1"
    echo "${PROCESSED_DATA_ROOT}/${dataset}/${dataset}_segments_reduced3_overlap50_100s_bp.npy"
}

labels_path() {
    local dataset="$1"
    echo "${PROCESSED_DATA_ROOT}/${dataset}/${dataset}_labels_reduced3_overlap50_100s_bp.npy"
}

require_file() {
    local path="$1"
    if [[ ! -f "${path}" ]]; then
        echo "Missing required file: ${path}" >&2
        exit 1
    fi
}

mkdir -p "${PROCESSED_DATA_ROOT}" "${RESULTS_ROOT}"
cd "${PROJECT_ROOT}"
