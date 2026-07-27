#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

datasets=(AFDB MITDB LTAFDB)

for source in "${datasets[@]}"; do
    for target in "${datasets[@]}"; do
        [[ "${source}" == "${target}" ]] && continue
        checkpoint_root="${RESULTS_ROOT}/DANN_Res_18/${source}_to_${target}"
        [[ ! -d "${checkpoint_root}" ]] && continue

        while IFS= read -r -d '' checkpoint; do
            "${PYTHON_BIN}" -m scripts.test_eval \
                --test_ds "${target}" \
                --segments "$(segments_path "${target}")" \
                --labels "$(labels_path "${target}")" \
                --train_ds "${source}" \
                --da_ckpt "${checkpoint}" \
                --backbone resnet18 \
                --save_dir "${RESULTS_ROOT}/evaluation/DANN_Res_18" \
                --bs "${BATCH_SIZE}"
        done < <(find "${checkpoint_root}" -name best_f1.pth -print0)
    done
done
