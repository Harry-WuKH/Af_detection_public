#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

read -r -a baseline_models <<< "${BASELINE_MODELS}"

for architecture in "${baseline_models[@]}"; do
    if [[ "${architecture}" == "resnet18" ]]; then
        checkpoint_name="best_da_model.pth"
    else
        checkpoint_name="best_f1.pth"
    fi

    for source in AFDB MITDB LTAFDB; do
        checkpoint_root="${RESULTS_ROOT}/baseline/${architecture}/${source}"
        [[ ! -d "${checkpoint_root}" ]] && continue

        while IFS= read -r -d '' checkpoint; do
            for target in AFDB MITDB LTAFDB; do
                [[ "${source}" == "${target}" ]] && continue
                "${PYTHON_BIN}" -m scripts.test_eval \
                    --test_ds "${target}" \
                    --segments "$(segments_path "${target}")" \
                    --labels "$(labels_path "${target}")" \
                    --train_ds "${source}" \
                    --da_ckpt "${checkpoint}" \
                    --backbone "${architecture}" \
                    --save_dir "${RESULTS_ROOT}/evaluation/baseline/${architecture}" \
                    --bs "${BATCH_SIZE}"
            done
        done < <(find "${checkpoint_root}" -name "${checkpoint_name}" -print0)
    done
done
