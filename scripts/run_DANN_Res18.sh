#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

datasets=(AFDB MITDB LTAFDB)

for source in "${datasets[@]}"; do
    for target in "${datasets[@]}"; do
        [[ "${source}" == "${target}" ]] && continue

        src_segments="$(segments_path "${source}")"
        src_labels="$(labels_path "${source}")"
        tgt_segments="$(segments_path "${target}")"
        require_file "${src_segments}"
        require_file "${src_labels}"
        require_file "${tgt_segments}"

        echo "Starting DANN training: ${source} -> ${target}"
        for ((run = 1; run <= RUNS; run++)); do
            "${PYTHON_BIN}" -m scripts.train_DANN_res18 \
                --src_segments "${src_segments}" \
                --src_labels "${src_labels}" \
                --tgt_segments "${tgt_segments}" \
                --save_ckpt "${RESULTS_ROOT}/DANN_Res_18/${source}_to_${target}/best_f1.pth" \
                --save_src_roc "${RESULTS_ROOT}/DANN_Res_18/${source}_to_${target}/roc_src.png" \
                --epochs "${EPOCHS}" \
                --bs "${BATCH_SIZE}" \
                --lr_feat_ex 1e-3 \
                --lr_classifier 1e-4 \
                --lr_domain 1e-4 \
                --lambda_adv "${DANN_LAMBDA}" \
                --n_domain "${DANN_DOMAIN_STEPS}" \
                --early_stop_patience "${EARLY_STOP_PATIENCE}" \
                --seed -1
        done
    done
done

