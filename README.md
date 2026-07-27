# Cross-dataset AF detection

This repository contains the public AF-detection pipeline for AFDB, MITDB, and LTAFDB. It keeps the original preprocessing and experiment implementations while replacing machine-specific paths with one editable configuration file.

The public training scope is:

- source-only ResNet-18, VGG-19, and InceptionNet baselines;
- Domain-Adversarial Neural Network (DANN);
- cross-dataset evaluation;
- Grad-CAM, paired Wilcoxon analysis, and ASW batch analysis.

## Repository layout

```text
configs/       path configuration
preprocess/    separate AFDB, MITDB, and LTAFDB preprocessing scripts
models/        ResNet, VGG-19, InceptionNet, classifier, DANN, and Grad-CAM
scripts/       original Python entrypoints and shell experiment runners
data/          local raw and processed data (ignored by Git)
outputs/       checkpoints and reports (ignored by Git)
```

## Setup

```bash
git clone https://github.com/Harry-WuKH/Af_detection_public.git
cd Af_detection_public
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp configs/paths.env.example configs/paths.env
```

Edit `configs/paths.env`. In the common case, only these paths need changing:

```bash
AFDB_RAW_DIR="/absolute/path/to/afdb"
MITDB_RAW_DIR="/absolute/path/to/mitdb"
LTAFDB_RAW_DIR="/absolute/path/to/ltafdb"
```

Each directory must contain the WFDB record files (`.dat`, `.hea`, and `.atr`). The ECG databases themselves are not included in this repository.

## Preprocessing

The three scripts remain separate because the original database-specific lead selection and preprocessing order are different. Their signal processing, segmentation, and label rules are unchanged.

```bash
bash scripts/preprocess_all.sh
```

Default outputs are written to:

```text
data/processed/AFDB/AFDB_segments_reduced3_overlap50_100s_bp.npy
data/processed/AFDB/AFDB_labels_reduced3_overlap50_100s_bp.npy
data/processed/MITDB/MITDB_segments_reduced3_overlap50_100s_bp.npy
data/processed/MITDB/MITDB_labels_reduced3_overlap50_100s_bp.npy
data/processed/LTAFDB/LTAFDB_segments_reduced3_overlap50_100s_bp.npy
data/processed/LTAFDB/LTAFDB_labels_reduced3_overlap50_100s_bp.npy
```

The original defaults are 128 Hz, 10-second windows, 50% overlap, and reduced rhythm labels: `0 = SINUS`, `1 = AFIB`, `2 = OTHER`. Training and evaluation retain only labels 0 and 1 (Non-AF / AF)

## Training

Run all source-only ResNet-18, VGG-19, and InceptionNet experiments:

```bash
bash scripts/run_base_train_allds.sh
```

Run all six directed DANN transfers:

```bash
bash scripts/run_DANN_Res18.sh
```

Set experiment overrides without editing code:

```bash
RUNS=1 EPOCHS=5 BATCH_SIZE=32 bash scripts/run_DANN_Res18.sh
```

Select only specific baseline architectures when needed:

```bash
BASELINE_MODELS="vgg19 inception" RUNS=1 bash scripts/run_base_train_allds.sh
```

## Evaluation

```bash
bash scripts/run_base_eval_allds.sh
bash scripts/run_DANN_eval_allds.sh
```

The runners discover timestamped seed folders produced by the original training scripts and write cross-dataset reports under `outputs/evaluation/`.

## Analysis

Run Grad-CAM region statistics and the paired Wilcoxon tests with baseline and DANN checkpoints:

```bash
bash scripts/gradcam_statistic.sh \
  outputs/baseline/resnet18/AFDB/<seed-folder>/best_da_model.pth \
  outputs/DANN_Res_18/AFDB_to_MITDB/<seed-folder>/best_f1.pth \
  MITDB \
  outputs/analysis/AFDB_to_MITDB
```

The retained Grad-CAM region implementation targets ResNet-18 layers. Run the original ASW batch analysis over all configured baseline architectures:

```bash
bash scripts/run_asw_base.sh
```

For a single Grad-CAM comparison, run `python -m scripts.visualize_gradcam --help`.

## Reproducibility notes

- Shell scripts resolve the repository root from their own location, so they can be launched from any working directory.
- Raw data, generated NumPy files, checkpoints, and outputs are excluded by `.gitignore`.
- Keep dataset access terms and PhysioNet citations with any redistributed results.
