import os
import argparse
import random
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from scipy.signal import find_peaks
from scipy.stats import wilcoxon

from models.resnet_backbone import ResNet1D_18_backbone
from models.classifier import ClassifierHead
from models.gradcam import GradCAM_1D


# ==========================================
# Full Model
# ==========================================
class AF_Detector_Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ResNet1D_18_backbone(in_ch=1)
        self.classifier = ClassifierHead(
            in_dim=512, hidden_dim=256, num_classes=2
        )

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits


def load_model_weights(model, weight_path, device):
    checkpoint = torch.load(weight_path, map_location=device)
    if "resnet_backbone" in checkpoint:
        model.backbone.load_state_dict(checkpoint["resnet_backbone"])
        model.classifier.load_state_dict(checkpoint["classifier"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model


# ==========================================
# ECG / R-peak utils
# ==========================================

def detect_rpeaks_simple(ecg_signal, fs=128):
    """
    A simple R-peak detector for today’s analysis.
    It is not a clinical-grade delineator, but enough for relative comparison.
    """
    x = np.asarray(ecg_signal, dtype=np.float32)

    # Energy-like envelope
    sq = x ** 2
    win = max(1, int(0.12 * fs))  # ~120 ms
    kernel = np.ones(win, dtype=np.float32) / win
    env = np.convolve(sq, kernel, mode="same")

    # Peak candidates from envelope
    min_distance = int(0.3 * fs)  # 250 ms refractory
    prominence = max(0.2, 0.5 * np.std(env))
    cand_peaks, _ = find_peaks(env, distance=min_distance, prominence=prominence)

    # Refine on original signal using abs amplitude near candidate
    refined = []
    refine_radius = int(0.08 * fs)  # 80 ms
    abs_x = np.abs(x)

    for p in cand_peaks:
        s = max(0, p - refine_radius)
        e = min(len(x), p + refine_radius + 1)
        local = np.argmax(abs_x[s:e])
        refined_peak = s + local
        refined.append(refined_peak)

    if len(refined) == 0:
        return np.array([], dtype=np.int64)

    refined = np.array(sorted(set(refined)), dtype=np.int64)

    # Remove peaks that are too close after refinement
    final_peaks = [refined[0]]
    for p in refined[1:]:
        if p - final_peaks[-1] >= min_distance:
            final_peaks.append(p)

    return np.array(final_peaks, dtype=np.int64)

def plot_rpeaks_only(ecg_signal, rpeaks, gt_label, pred_base, pred_dann, save_path):
    fig, ax = plt.subplots(figsize=(18, 3.5))

    xs = np.arange(len(ecg_signal))
    ax.plot(xs, ecg_signal, color="k", linewidth=1.2)

    if len(rpeaks) > 0:
        ax.scatter(rpeaks, ecg_signal[rpeaks], c="lime", s=28, zorder=5, label="R-peaks")

    ax.set_xlim(0, len(ecg_signal))
    ax.set_xlabel("Samples")
    ax.set_ylabel("Amplitude")
    ax.set_title(
        f"ECG with R-peak markers | GT={gt_label} | Source-Only={pred_base} | DANN={pred_dann}",
        fontsize=12,
        fontweight="bold"
    )
    ax.legend(loc="upper right")
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

def plot_ecg_with_regions_only(
    ecg_signal,
    rpeaks,
    p_mask,
    qrs_mask,
    gt_label,
    pred_base,
    pred_dann,
    save_path
):
    fig, ax = plt.subplots(figsize=(18, 3.8))

    xs = np.arange(len(ecg_signal))

    # White background
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    # ECG waveform
    ax.plot(xs, ecg_signal, color="black", linewidth=1.2, label="ECG")

    # -------- P-region shading --------
    p_indices = np.where(p_mask == 1)[0]
    if len(p_indices) > 0:
        groups = np.split(p_indices, np.where(np.diff(p_indices) != 1)[0] + 1)
        first = True
        for g in groups:
            ax.axvspan(
                g[0], g[-1],
                color="cornflowerblue",
                alpha=0.22,
                label="P-region" if first else None
            )
            first = False

    # -------- QRS-region shading --------
    q_indices = np.where(qrs_mask == 1)[0]
    if len(q_indices) > 0:
        groups = np.split(q_indices, np.where(np.diff(q_indices) != 1)[0] + 1)
        first = True
        for g in groups:
            ax.axvspan(
                g[0], g[-1],
                color="salmon",
                alpha=0.25,
                label="QRS-region" if first else None
            )
            first = False

    # -------- R-peaks --------
    if len(rpeaks) > 0:
        ax.scatter(
            rpeaks,
            ecg_signal[rpeaks],
            c="limegreen",
            s=32,
            zorder=5,
            label="R-peaks"
        )

    ax.set_xlim(0, len(ecg_signal))
    ax.set_xlabel("Samples")
    ax.set_ylabel("Amplitude")
    ax.set_title(
        f"ECG with R-peaks / P-region / QRS-region | GT={gt_label} | "
        f"Source-Only={pred_base} | DANN={pred_dann}",
        fontsize=12,
        fontweight="bold"
    )

    ax.legend(loc="upper right", ncol=4, frameon=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def build_region_masks(signal_length, rpeaks, fs=128,
                       p_left_ms=200, p_right_ms=44,
                       qrs_left_ms=40, qrs_right_ms=60):
    """
    pre-R atrial window: [R - 200 ms, R - 44 ms]
    QRS window         : [R -  40 ms, R + 60 ms]
    """
    p_mask = np.zeros(signal_length, dtype=np.uint8)
    qrs_mask = np.zeros(signal_length, dtype=np.uint8)

    p_left = int(round(p_left_ms * fs / 1000.0))
    p_right = int(round(p_right_ms * fs / 1000.0))
    qrs_left = int(round(qrs_left_ms * fs / 1000.0))
    qrs_right = int(round(qrs_right_ms * fs / 1000.0))

    for r in rpeaks:
        p_start = max(0, r - p_left)
        p_end = max(0, r - p_right)
        if p_end > p_start:
            p_mask[p_start:p_end] = 1

        q_start = max(0, r - qrs_left)
        q_end = min(signal_length, r + qrs_right)
        if q_end > q_start:
            qrs_mask[q_start:q_end] = 1

    return p_mask, qrs_mask


def normalize_cam_mass(cam):
    cam = np.asarray(cam, dtype=np.float32)
    cam = np.abs(cam)
    s = cam.sum()
    if s < 1e-12:
        return np.zeros_like(cam)
    return cam / s


def compute_cam_metrics(cam, p_mask, qrs_mask):
    cam = normalize_cam_mass(cam)

    focus_p = cam[p_mask == 1].sum()
    focus_qrs = cam[qrs_mask == 1].sum()

    return {
        "focus_p": float(focus_p),
        "focus_qrs": float(focus_qrs),
    }

# ==========================================
# Plotting
# ==========================================
def plot_example(ecg_signal, cam_base, cam_dann, rpeaks, p_mask, qrs_mask,
                 gt_label, pred_base, pred_dann, save_path):
    fig, axes = plt.subplots(2, 1, figsize=(18, 7), sharex=True)
    length = len(ecg_signal)
    xs = np.arange(length)

    ymin = np.min(ecg_signal) - 0.5
    ymax = np.max(ecg_signal) + 0.5

    panels = [
        (axes[0], cam_base, f"Source-Only | GT={gt_label} | Pred={pred_base}"),
        (axes[1], cam_dann, f"DANN | GT={gt_label} | Pred={pred_dann}")
    ]

    im = None
    for ax, cam, title in panels:
        im = ax.imshow(
            cam[np.newaxis, :],
            cmap="jet",
            aspect="auto",
            alpha=0.85,
            vmin=0.0,
            vmax=1.0,
            extent=[0, length, ymin, ymax]
        )
        ax.plot(xs, ecg_signal, color="k", linewidth=1.2)

        p_indices = np.where(p_mask == 1)[0]
        if len(p_indices) > 0:
            groups = np.split(p_indices, np.where(np.diff(p_indices) != 1)[0] + 1)
            for g in groups:
                ax.axvspan(g[0], g[-1], color="gray", alpha=0.18)

        q_indices = np.where(qrs_mask == 1)[0]
        if len(q_indices) > 0:
            groups = np.split(q_indices, np.where(np.diff(q_indices) != 1)[0] + 1)
            for g in groups:
                ax.axvspan(g[0], g[-1], color="red", alpha=0.10)

        ax.scatter(rpeaks, ecg_signal[rpeaks], c="lime", s=20, zorder=5, label="R-peaks")
        ax.set_ylabel("Amplitude")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlim(0, length)

    axes[1].set_xlabel("Samples")

    fig.subplots_adjust(right=0.90, hspace=0.30)
    cbar_ax = fig.add_axes([0.915, 0.12, 0.015, 0.76])
    fig.colorbar(im, cax=cbar_ax, label="Grad-CAM")

    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def p_to_stars(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return "ns"

def add_sig_bracket(ax, x1, x2, y, h, text):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="black", linewidth=1.2)
    ax.text((x1 + x2) / 2, y + h, text, ha="center", va="bottom",
            fontsize=12, fontweight="bold")

def plot_box_scatter_metric(df, metric, p_value, save_path, ylabel=None):
    x = df[f"{metric}_baseline"].values
    y = df[f"{metric}_dann"].values

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    fig, ax = plt.subplots(figsize=(5.5, 5.0))

    data = [x, y]
    positions = [1, 2]

    ax.boxplot(
        data,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showfliers=True,
        medianprops=dict(color="black", linewidth=1.3),
        boxprops=dict(facecolor="lightgray", edgecolor="dimgray", linewidth=1.1),
        whiskerprops=dict(color="dimgray", linewidth=1.0),
        capprops=dict(color="dimgray", linewidth=1.0),
        flierprops=dict(marker='d', markersize=4, markerfacecolor='gray',
                        markeredgecolor='gray', alpha=0.8)
    )

    rng = np.random.default_rng(42)
    for pos, arr in zip(positions, data):
        jitter = rng.normal(0, 0.07, size=len(arr))
        ax.scatter(
            np.full(len(arr), pos) + jitter,
            arr,
            s=8,
            c="black",
            alpha=0.8,
            linewidths=0
        )

    ymax = np.max(np.concatenate([x, y]))
    ymin = np.min(np.concatenate([x, y]))
    yrange = max(ymax - ymin, 1e-6)

    bracket_y = ymax + 0.08 * yrange
    bracket_h = 0.03 * yrange
    add_sig_bracket(ax, 1, 2, bracket_y, bracket_h, p_to_stars(p_value))

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Source-Only", "DANN"], fontsize=11)
    ax.set_ylabel(ylabel if ylabel is not None else metric, fontsize=12)
    ax.set_title(metric, fontsize=13, fontweight="bold")
    ax.set_ylim(ymin - 0.05 * yrange, ymax + 0.18 * yrange)
    ax.grid(axis="y", alpha=0.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_violin_metric(df, metric, p_value, save_path, ylabel=None):
    x = df[f"{metric}_baseline"].values
    y = df[f"{metric}_dann"].values

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    fig, ax = plt.subplots(figsize=(5.5, 5.0))

    data = [x, y]
    positions = [1, 2]

    # -----------------------------
    # Violin plot: distribution shape
    # -----------------------------
    parts = ax.violinplot(
        data,
        positions=positions,
        widths=0.75,
        showmeans=False,
        showmedians=False,
        showextrema=False
    )

    for body in parts["bodies"]:
        body.set_facecolor("lightgray")
        body.set_edgecolor("dimgray")
        body.set_alpha(0.65)
        body.set_linewidth(1.0)

    # -----------------------------
    # Narrow boxplot overlay
    # This keeps median/IQR but does not block the middle too much.
    # -----------------------------
    ax.boxplot(
        data,
        positions=positions,
        widths=0.18,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=1.4),
        boxprops=dict(facecolor="white", edgecolor="black", linewidth=1.0, alpha=0.85),
        whiskerprops=dict(color="black", linewidth=1.0),
        capprops=dict(color="black", linewidth=1.0)
    )

    # -----------------------------
    # Optional: raw points, but very faint
    # To avoid overcrowding, randomly display at most 800 points per group.
    # -----------------------------
    rng = np.random.default_rng(42)
    max_points = 800

    for pos, arr in zip(positions, data):
        if len(arr) > max_points:
            show_idx = rng.choice(len(arr), size=max_points, replace=False)
            arr_show = arr[show_idx]
        else:
            arr_show = arr

        jitter = rng.normal(0, 0.045, size=len(arr_show))

        ax.scatter(
            np.full(len(arr_show), pos) + jitter,
            arr_show,
            s=5,
            c="black",
            alpha=0.18,
            linewidths=0,
            zorder=3
        )

    # -----------------------------
    # Significance bracket
    # -----------------------------
    all_values = np.concatenate([x, y])
    ymax = np.max(all_values)
    ymin = np.min(all_values)
    yrange = max(ymax - ymin, 1e-6)

    bracket_y = ymax + 0.08 * yrange
    bracket_h = 0.03 * yrange
    add_sig_bracket(ax, 1, 2, bracket_y, bracket_h, p_to_stars(p_value))

    # -----------------------------
    # Axis style
    # -----------------------------
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Source-Only", "DANN"], fontsize=11)
    ax.set_ylabel(ylabel if ylabel is not None else metric, fontsize=12)
    ax.set_title(metric, fontsize=13, fontweight="bold")

    ax.set_ylim(ymin - 0.05 * yrange, ymax + 0.18 * yrange)
    ax.grid(axis="y", alpha=0.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()




# ==========================================
# Statistics
# ==========================================
def summarize_metric(df, metric, focus_p_alternative, focus_qrs_alternative):
    x = df[f"{metric}_baseline"].values
    y = df[f"{metric}_dann"].values

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if len(x) == 0:
        return None

    if metric == "focus_p":
        alternative = focus_p_alternative
    elif metric == "focus_qrs":
        alternative = focus_qrs_alternative
    else:
        alternative = "two-sided"

    try:
        stat = wilcoxon(y, x, alternative=alternative)
        p_value = float(stat.pvalue)
    except ValueError:
        p_value = 1.0

    if alternative == "greater":
        h1_text = "DANN > Source-Only"
    else:
        h1_text = "DANN != Source-Only"
    

    return {
        "metric": metric,
        "n": int(len(x)),
        "baseline_mean": float(np.mean(x)),
        "baseline_std": float(np.std(x)),
        "dann_mean": float(np.mean(y)),
        "dann_std": float(np.std(y)),
        "mean_diff_dann_minus_baseline": float(np.mean(y - x)),
        "median_diff_dann_minus_baseline": float(np.median(y - x)),
        "p_value": p_value,
        "alternative": alternative,
        "h1_text": h1_text,
    }


def find_valid_indices_batch(
    data,
    labels,
    model_base,
    model_dann,
    device,
    gt_label,
    pred_label,
    case_name,
    signal_length=1280,
    batch_size=256,
):
    labels = np.asarray(labels)
    target_indices = np.where(labels == gt_label)[0]

    valid_len_indices = []
    for idx in target_indices:
        ecg_signal = np.squeeze(data[idx])
        if len(ecg_signal) == signal_length:
            valid_len_indices.append(idx)

    valid_len_indices = np.array(valid_len_indices, dtype=np.int64)

    print(f"Total labels: {len(labels)}")
    print(f"GT={case_name} segments: {len(target_indices)}")
    print(f"GT={case_name} with correct length ({signal_length}): {len(valid_len_indices)}")

    valid_indices = []

    model_base.eval()
    model_dann.eval()

    with torch.no_grad():
        for start in tqdm(
            range(0, len(valid_len_indices), batch_size),
            desc=f"Batch prefilter valid {case_name}",
            ncols=120
        ):
            batch_ids = valid_len_indices[start:start + batch_size]
            batch_np = np.asarray(data[batch_ids], dtype=np.float32)

            if batch_np.ndim == 2:
                batch_np = batch_np[:, None, :]
            elif batch_np.ndim == 3:
                if batch_np.shape[1] == 1:
                    pass
                elif batch_np.shape[2] == 1:
                    batch_np = np.transpose(batch_np, (0, 2, 1))
                else:
                    raise ValueError(f"Unexpected batch shape: {batch_np.shape}")
            else:
                raise ValueError(f"Unexpected batch ndim: {batch_np.ndim}")

            batch_tensor = torch.from_numpy(batch_np).to(device)

            logits_base = model_base(batch_tensor)
            logits_dann = model_dann(batch_tensor)

            pred_base = logits_base.argmax(dim=1).cpu().numpy()
            pred_dann = logits_dann.argmax(dim=1).cpu().numpy()

            keep_mask = (pred_base == pred_label) & (pred_dann == pred_label)
            kept_ids = batch_ids[keep_mask]
            valid_indices.extend(kept_ids.tolist())

    print(f"Both models predicted {case_name}: {len(valid_indices)}")
    return valid_indices


# ==========================================
# Main
# ==========================================
def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.out_dir, exist_ok=True)
    example_dir = os.path.join(args.out_dir, "examples")
    os.makedirs(example_dir, exist_ok=True)

    print("Loading models...")
    model_base = load_model_weights(AF_Detector_Model().to(device), args.base_weight, device)
    model_dann = load_model_weights(AF_Detector_Model().to(device), args.dann_weight, device)

    # You may switch to model.backbone.layer4[-1].conv2 if your block supports it.
    target_layer_base = model_base.backbone.layer4[-1]
    target_layer_dann = model_dann.backbone.layer4[-1]

    cam_base_extractor = GradCAM_1D(model_base, target_layer_base)
    cam_dann_extractor = GradCAM_1D(model_dann, target_layer_dann)

    print("Loading data...")
    data = np.load(args.data_path)
    labels = np.load(args.label_path)

    if args.analysis_case == "AF":
        case_name = "AF"
        gt_label_target = 1
        pred_label_target = 1
        target_class = 1
        per_segment_csv = "gradcam_af_stats.csv"
        report_title = "Grad-CAM analysis on AF segments"
    elif args.analysis_case == "Normal":
        case_name = "Normal"
        gt_label_target = 0
        pred_label_target = 0
        target_class = 0
        per_segment_csv = "gradcam_normal_stats.csv"
        report_title = "Grad-CAM analysis on Normal segments"
    else:
        raise ValueError(f"Unsupported analysis_case: {args.analysis_case}")

    print("Data shape:", data.shape)
    print("Labels shape:", labels.shape)


    print(f"Phase 1: batch prefilter valid {case_name} indices...")
    valid_indices = find_valid_indices_batch(
        data=data,
        labels=labels,
        model_base=model_base,
        model_dann=model_dann,
        device=device,
        gt_label=gt_label_target,
        pred_label=pred_label_target,
        case_name=case_name,
        signal_length=args.signal_length,
        batch_size=args.pred_batch_size,
    )

    if len(valid_indices) == 0:
        print(f"No valid {case_name} samples found after batch prefilter.")
        return

    valid_indices = np.array(valid_indices, dtype=np.int64)

    if args.shuffle_valid_indices:
        rng = np.random.default_rng(args.seed)
        rng.shuffle(valid_indices)

    if args.max_valid_samples is not None:
        original_n = len(valid_indices)
        valid_indices = valid_indices[:args.max_valid_samples]
        print(f"Using only {len(valid_indices)} / {original_n} valid {case_name} samples for Grad-CAM analysis.")
    else:
        print(f"Using all {len(valid_indices)} valid {case_name} samples for Grad-CAM analysis.")

    rows = []
    example_count = 0
    count_valid = 0

    print(f"Phase 2: Grad-CAM analysis on valid {case_name} indices...")
    for idx in tqdm(valid_indices, desc=f"Grad-CAM on valid {case_name}", ncols=120):
        ecg_signal = np.squeeze(data[idx]).astype(np.float32)
        gt = int(labels[idx])

        input_tensor = torch.tensor(ecg_signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

        # After batch prefilter, both predictions are already the target case
        pred_base = pred_label_target
        pred_dann = pred_label_target

        rpeaks = detect_rpeaks_simple(ecg_signal, fs=args.fs)
        if len(rpeaks) < 2:
            continue

        p_mask, qrs_mask = build_region_masks(
            signal_length=len(ecg_signal),
            rpeaks=rpeaks,
            fs=args.fs,
            p_left_ms=args.p_left_ms,
            p_right_ms=args.p_right_ms,
            qrs_left_ms=args.qrs_left_ms,
            qrs_right_ms=args.qrs_right_ms,
        )

        if p_mask.sum() == 0 or qrs_mask.sum() == 0:
            continue

        cam_base, _ = cam_base_extractor.generate_cam(input_tensor, target_class=target_class)
        cam_dann, _ = cam_dann_extractor.generate_cam(input_tensor, target_class=target_class)

        m_base = compute_cam_metrics(cam_base, p_mask, qrs_mask)
        m_dann = compute_cam_metrics(cam_dann, p_mask, qrs_mask)

        row = {
            "idx": idx,
            "gt": gt,
            "pred_base": pred_base,
            "pred_dann": pred_dann,
            "num_rpeaks": len(rpeaks),
        }

        for k, v in m_base.items():
            row[f"{k}_baseline"] = v
        for k, v in m_dann.items():
            row[f"{k}_dann"] = v

        rows.append(row)
        count_valid += 1

        if example_count < args.num_examples:
            save_path_gradcam = os.path.join(example_dir, f"example_idx{idx}_gradcam.png")
            save_path_plain = os.path.join(example_dir, f"example_idx{idx}_rpeaks_only.png")
            save_path_regions = os.path.join(example_dir, f"example_idx{idx}_regions_only.png")

            plot_example(
                ecg_signal=ecg_signal,
                cam_base=cam_base,
                cam_dann=cam_dann,
                rpeaks=rpeaks,
                p_mask=p_mask,
                qrs_mask=qrs_mask,
                gt_label=gt,
                pred_base=pred_base,
                pred_dann=pred_dann,
                save_path=save_path_gradcam,
            )

            plot_rpeaks_only(
                ecg_signal=ecg_signal,
                rpeaks=rpeaks,
                gt_label=gt,
                pred_base=pred_base,
                pred_dann=pred_dann,
                save_path=save_path_plain,
            )
            plot_ecg_with_regions_only(
                ecg_signal=ecg_signal,
                rpeaks=rpeaks,
                p_mask=p_mask,
                qrs_mask=qrs_mask,
                gt_label=gt,
                pred_base=pred_base,
                pred_dann=pred_dann,
                save_path=save_path_regions,
            )

            example_count += 1

    print(f"Prefilter valid {case_name} indices: {len(valid_indices)}")
    print(f"Valid {case_name} segments for paired analysis after R-peak/mask checks: {count_valid}")

    if len(rows) == 0:
        print(f"No valid {case_name} samples found for paired analysis.")
        return

    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.out_dir, per_segment_csv)
    df.to_csv(csv_path, index=False)
    print(f"Saved per-segment metrics to: {csv_path}")

    metrics_to_test = ["focus_p", "focus_qrs"]
    summary_rows = []

    for metric in metrics_to_test:
        result = summarize_metric(
            df,
            metric,
            focus_p_alternative=args.focus_p_alternative,
            focus_qrs_alternative=args.focus_qrs_alternative,
        )
        if result is not None:
            summary_rows.append(result)

            plot_violin_metric(
                df=df,
                metric=metric,
                p_value=result["p_value"],
                save_path=os.path.join(args.out_dir, f"violin_{metric}.png"),
                ylabel=metric
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(args.out_dir, "summary_stats.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved summary stats to: {summary_csv}")

    txt_path = os.path.join(args.out_dir, "summary_report.txt")
    with open(txt_path, "w") as f:
        f.write(report_title + "\n")
        f.write("=" * 60 + "\n")
        f.write(f"N valid {case_name} segments: {len(df)}\n\n")
        for _, row in summary_df.iterrows():
            f.write(f"Metric: {row['metric']}\n")
            f.write(f"  N: {row['n']}\n")
            f.write(f"  Source-Only mean ± std: {row['baseline_mean']:.6f} ± {row['baseline_std']:.6f}\n")
            f.write(f"  DANN mean ± std:     {row['dann_mean']:.6f} ± {row['dann_std']:.6f}\n")
            f.write(f"  Mean diff (DANN - Source-Only): {row['mean_diff_dann_minus_baseline']:.6f}\n")
            f.write(f"  Median diff (DANN - Source-Only): {row['median_diff_dann_minus_baseline']:.6f}\n")
            f.write(f"  Wilcoxon p-value (H1: {row['h1_text']}): {row['p_value']:.6e}\n")
            f.write("\n")

    print(f"Saved summary report to: {txt_path}")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grad-CAM ECG region focus statistics")
    parser.add_argument("--base_weight", type=str, required=True)
    parser.add_argument("--dann_weight", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--label_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--fs", type=int, default=128)
    parser.add_argument("--signal_length", type=int, default=1280)

    parser.add_argument("--p_left_ms", type=float, default=200.0)
    parser.add_argument("--p_right_ms", type=float, default=44.0)
    parser.add_argument("--qrs_left_ms", type=float, default=40.0)
    parser.add_argument("--qrs_right_ms", type=float, default=60.0)
    parser.add_argument("--pred_batch_size", type=int, default=128)

    parser.add_argument("--num_examples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=69)

    parser.add_argument("--max_valid_samples", type=int, default=10000)
    parser.add_argument("--shuffle_valid_indices", action="store_true")

    parser.add_argument(
    "--analysis_case",
    type=str,
    default="Normal",
    choices=["AF", "Normal"],
    help="Which case to analyze."
    )

    parser.add_argument(
        "--focus_p_alternative",
        type=str,
        default="two-sided",
        choices=["greater", "two-sided"],
        help="Wilcoxon alternative for focus_p."
    )

    parser.add_argument(
        "--focus_qrs_alternative",
        type=str,
        default="two-sided",
        choices=["greater", "two-sided"],
        help="Wilcoxon alternative for focus_qrs."
    )

    args = parser.parse_args()
    main(args)
