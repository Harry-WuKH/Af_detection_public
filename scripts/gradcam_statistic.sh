#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

if [[ $# -ne 4 ]]; then
    echo "Usage: $0 BASELINE_CHECKPOINT DANN_CHECKPOINT DATASET OUTPUT_DIR" >&2
    exit 2
fi

baseline_checkpoint="$1"
dann_checkpoint="$2"
dataset="$3"
output_dir="$4"

"${PYTHON_BIN}" -m scripts.gradcam_region_wilcoxon \
    --base_weight "${baseline_checkpoint}" \
    --dann_weight "${dann_checkpoint}" \
    --data_path "$(segments_path "${dataset}")" \
    --label_path "$(labels_path "${dataset}")" \
    --out_dir "${output_dir}" \
    --fs "${GRADCAM_FS}" \
    --signal_length "${GRADCAM_SIGNAL_LENGTH}" \
    --max_valid_samples 10000 \
    --shuffle_valid_indices \
    --analysis_case AF \
    --focus_p_alternative two-sided \
    --focus_qrs_alternative two-sided

