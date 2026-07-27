# scripts/eval_cross_da.py
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, roc_curve, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

from models.resnet_backbone import ResNet1D_18_backbone
from models.vgg19_backbone import VGG19_1D_backbone
from models.inception_backbone import InceptionBackbone1D
from models.classifier import ClassifierHead


BACKBONES = {
    "resnet18": ResNet1D_18_backbone,
    "vgg19": VGG19_1D_backbone,
    "inception": InceptionBackbone1D,
}


def evaluate_metrics(y_true, y_prob, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp + 1e-12)
    weighted_acc = ( rec + specificity ) / 2
    try:
        auroc = roc_auc_score(y_true, y_prob[:, 1])
    except Exception:
        auroc = float("nan")
    return {
        "acc": acc,
        "precision": prec,
        "recall_sensitivity": rec,
        "specificity": specificity,
        "f1": f1,
        "wacc": weighted_acc,
        "auroc": auroc
    }, cm


def save_log(save_dir, train_name, test_name, seed, ckpt_path, metrics, cm, n_samples, pos_label=1):
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, "log.txt")

    tn, fp, fn, tp = cm.ravel()

    log_str = (
        f"\n{'='*60}\n"
        f"Train: {train_name}\n"
        f"Test:  {test_name}\n"
        f"Seed:  {seed}\n"
        f"N = {n_samples} (labels in {{0,1}}, positive={pos_label})\n\n"
        f"ACC  = {metrics['acc']:.4f}\n"
        f"PREC = {metrics['precision']:.4f}\n"
        f"SEN  = {metrics['recall_sensitivity']:.4f}\n"
        f"SPEC = {metrics['specificity']:.4f}\n"
        f"F1   = {metrics['f1']:.4f}\n"
        f"WACC = {metrics['wacc']:.4f}\n"
        f"AUC  = {metrics['auroc']:.4f}\n\n"
        f"Confusion Matrix (rows=True, cols=Pred, labels=[0,1]):\n"
        f"[[{tn:6d} {fp:6d}]\n"
        f" [{fn:6d} {tp:6d}]]\n"
    )

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(log_str)

    print(f"Metrics written to: {log_path}")



def plot_roc(y_true, y_prob, out_path, title="ROC"):
    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
        plt.figure()
        plt.plot(fpr, tpr, linewidth=2)
        plt.plot([0, 1], [0, 1], '--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(title)
        plt.tight_layout()
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=200)
        plt.close()
    except Exception as e:
        print(f"[WARN] ROC plot failed: {e}")


# ------- plot confusion matrix -------
def plot_confmat(cm, out_path, labels=("Non-AF", "AF"), title="Confusion Matrix"):
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_ds", required=True, help="test dataset name (e.g. AFDB)")
    ap.add_argument("--segments", required=True, help="test segments .npy (N,T) or (N,1,T)")
    ap.add_argument("--labels", required=True, help="test labels .npy (N,), using labels {0,1}")
    ap.add_argument("--train_ds", required=True, help="train dataset name (e.g. MITDB)")
    ap.add_argument("--da_ckpt", required=True, help="DA model checkpoint (.pth)")
    ap.add_argument("--backbone", choices=sorted(BACKBONES), default="resnet18")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--save_dir", default="results/DA_cross_eval/")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X = np.load(args.segments)   # (N,T) or (N,1,T)
    y = np.load(args.labels)     # (N,)
    mask = np.isin(y, [0, 1])
    X, y = X[mask], y[mask].astype(np.int64)
    print(f"[Info] data={X.shape}, labels={np.unique(y, return_counts=True)}")

    if X.ndim == 2:
        X = X[:, np.newaxis, :]
    elif X.ndim == 3 and X.shape[1] == 1:
        pass
    else:
        raise ValueError(f"X must be (N,T) or (N,1,T), got {X.shape}")

    X_t = torch.from_numpy(X).float()
    y_t = torch.from_numpy(y).long()
    dl = DataLoader(TensorDataset(X_t, y_t), batch_size=args.bs, shuffle=False)

    ckpt = torch.load(args.da_ckpt, map_location=device)

    feat_dim = ckpt.get("feat_dim", 512)

    backbone = BACKBONES[args.backbone](in_ch=1).to(device)
    backbone.load_state_dict(ckpt["resnet_backbone"])
    backbone.eval()

    classifier = ClassifierHead(in_dim=feat_dim, num_classes=2).to(device)
    classifier.load_state_dict(ckpt["classifier"])
    classifier.eval()

    all_prob, all_pred, all_true = [], [], []
    with torch.no_grad():
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            feat = backbone(xb)                    # (B, feat_dim)
            logits = classifier(feat)              # (B, 2)
            prob = torch.softmax(logits, dim=1).cpu().numpy()
            pred = prob.argmax(1)
            all_prob.append(prob)
            all_pred.append(pred)
            all_true.append(yb.cpu().numpy())

    all_prob = np.concatenate(all_prob, 0)
    all_pred = np.concatenate(all_pred, 0)
    all_true = np.concatenate(all_true, 0)

    metrics, cm = evaluate_metrics(all_true, all_prob, all_pred)
    print(f"ACC={metrics['acc']:.4f}  SEN={metrics['recall_sensitivity']:.4f}  "
          f"SPEC={metrics['specificity']:.4f}  WACC={metrics['wacc']:.4f}"
          f"AUC={metrics['auroc']:.4f} F1={metrics['f1']:.4f}")
    
    print("Confusion matrix:\n", cm)

    train_name = os.path.basename(args.train_ds)
    test_name = os.path.basename(args.test_ds)

    pair_name = f"{train_name}_to_{test_name}"
    result_path = os.path.join(args.save_dir, pair_name)
    os.makedirs(result_path, exist_ok=True)
    seed_dir = os.path.dirname(args.da_ckpt)
    seed_tag = os.path.basename(seed_dir)
    seed_result_path = os.path.join(result_path, seed_tag)
    os.makedirs(seed_result_path, exist_ok=True)

    cm_title = f"{train_name} → {test_name}  {args.backbone}_CM"
    cm_path = os.path.join(seed_result_path, "cm.png")
    plot_confmat(cm, cm_path, title=cm_title)
    print("Confusion matrix saved to:", cm_path)

    roc_path = os.path.join(seed_result_path, "roc.png")
    plot_roc(all_true, all_prob, roc_path, title=f"{train_name} → {test_name} ROC")
    print("ROC saved to:", roc_path)

    save_log(
        save_dir=seed_result_path,
        train_name=train_name,
        test_name=test_name,
        seed=seed_tag,
        ckpt_path=args.da_ckpt,
        metrics=metrics,
        cm=cm,
        n_samples=len(all_true),
    )


if __name__ == "__main__":
    main()
