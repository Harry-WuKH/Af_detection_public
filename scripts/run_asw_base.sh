#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

log_file="${RESULTS_ROOT}/asw_batch/asw_batch_diagnosis_res18.csv"
mkdir -p "$(dirname "${log_file}")"
echo "Architecture,Seed_Folder,Src_Data,Tgt_Data,ASW_celltype,ASW_batch" > "${log_file}"

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
            seed_name="$(basename "$(dirname "${checkpoint}")")"
            for target in AFDB MITDB LTAFDB; do
                [[ "${source}" == "${target}" ]] && continue
                result="$("${PYTHON_BIN}" -m scripts.eval_asw --model_path "${checkpoint}" --backbone "${architecture}" --src "${source}" --tgt "${target}")"
                cell_asw="$(echo "${result}" | awk '/ASW_celltype/ {print $2}')"
                batch_asw="$(echo "${result}" | awk '/ASW_batch/ {print $2}')"
                echo "${architecture},${seed_name},${source},${target},${cell_asw},${batch_asw}" >> "${log_file}"
            done
        done < <(find "${checkpoint_root}" -name "${checkpoint_name}" -print0)
    done
done

echo "ASW results saved to ${log_file}"
