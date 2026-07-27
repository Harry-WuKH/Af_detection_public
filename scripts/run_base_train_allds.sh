#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

read -r -a baseline_models <<< "${BASELINE_MODELS}"

for architecture in "${baseline_models[@]}"; do
    case "${architecture}" in
        resnet18) module="scripts.train_baseline" ;;
        vgg19) module="scripts.train_baseline_vgg19" ;;
        inception) module="scripts.train_baseline_inception" ;;
        *) echo "Unknown baseline architecture: ${architecture}" >&2; exit 2 ;;
    esac

    for dataset in AFDB MITDB LTAFDB; do
        segments="$(segments_path "${dataset}")"
        labels="$(labels_path "${dataset}")"
        require_file "${segments}"
        require_file "${labels}"

        echo "Starting ${architecture} baseline training: ${dataset}"
        for ((run = 1; run <= RUNS; run++)); do
            "${PYTHON_BIN}" -m "${module}" \
                --src_segments "${segments}" \
                --src_labels "${labels}" \
                --save_ckpt "${RESULTS_ROOT}/baseline/${architecture}/${dataset}/best_f1.pth" \
                --save_src_roc "${RESULTS_ROOT}/baseline/${architecture}/${dataset}/roc.png" \
                --epochs "${EPOCHS}" \
                --bs "${BATCH_SIZE}" \
                --lr_feat_ex 1e-4 \
                --lr_classifier 1e-4 \
                --weight_decay 1e-4 \
                --early_stop_patience "${EARLY_STOP_PATIENCE}" \
                --seed -1
        done
    done
done
