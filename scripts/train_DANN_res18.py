# scripts/train_DANN_res18.py
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import random
from torch.utils.data import TensorDataset, DataLoader, random_split, WeightedRandomSampler
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, roc_curve, confusion_matrix
)
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm.auto import tqdm

from models.resnet_backbone import ResNet1D_18_backbone
from models.classifier import ClassifierHead
from models.discriminator import DomainDiscriminator


"""
Iterative DANN version for cross-database ECG AF detection.

Step 1:
    Update domain discriminator D
    min_D L_dom

Step 2:
    Update backbone F + classifier C
    min_{F,C} L_cls - lambda_adv * L_dom

This version uses iterative adversarial domain training.
"""


# -------------------------
# Evaluation Index
# -------------------------
def evaluate_metrics(y_true, y_prob, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp + 1e-12)
    weight_acc = (rec + specificity) / 2
    try:
        auroc = roc_auc_score(y_true, y_prob[:, 1])
    except Exception:
        auroc = float("nan")

    return {
        "acc": acc,
        "precision": prec,
        "recall_sensitivity": rec,
        "specificity": specificity,
        "w_acc": weight_acc,
        "f1": f1,
        "auroc": auroc
    }, cm


def plot_roc(y_true, y_prob, out_path, title="ROC"):
    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
        plt.figure()
        plt.plot(fpr, tpr, linewidth=2)
        plt.plot([0, 1], [0, 1], "--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(title)
        plt.tight_layout()
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=200)
        plt.close()
    except Exception as e:
        print(f"[WARN] ROC plot failed: {e}")


def make_src_dataloaders(X, y, val_ratio, bs, seed=1337):
    """
    Source domain: with labels, for supervised classification + source validation.
    Use WeightedRandomSampler to make source train batches more balanced.
    """
    m = np.isin(y, [0, 1])
    X = X[m]
    y = y[m].astype(np.int64)

    if X.ndim == 2:
        X = X[:, np.newaxis, :]
    elif X.ndim == 3 and X.shape[1] == 1:
        pass
    else:
        raise ValueError(f"X should be (N,T) or (N,1,T), got {X.shape}")

    N = len(y)
    val_len = int(N * val_ratio)
    tr_len = N - val_len

    X_t = torch.from_numpy(X).float()
    y_t = torch.from_numpy(y).long()
    ds = TensorDataset(X_t, y_t)

    tr_ds, va_ds = random_split(
        ds, [tr_len, val_len],
        generator=torch.Generator().manual_seed(seed)
    )

    tr_indices = tr_ds.indices
    tr_labels = y_t[tr_indices].numpy()

    class_counts = np.bincount(tr_labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[tr_labels]
    sample_weights = torch.from_numpy(sample_weights).double()

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    tr_dl = DataLoader(
        tr_ds,
        batch_size=bs,
        sampler=sampler,
        shuffle=False,
        drop_last=True
    )

    va_dl = DataLoader(
        va_ds,
        batch_size=bs,
        shuffle=False,
        drop_last=False
    )

    return tr_dl, va_dl, y


def make_tgt_dataloader(X, bs):
    """
    Target domain: unlabeled, only for domain adaptation training.
    """
    if X.ndim == 2:
        X = X[:, np.newaxis, :]
    elif X.ndim == 3 and X.shape[1] == 1:
        pass
    else:
        raise ValueError(f"Xt should be (N,T) or (N,1,T), got {X.shape}")

    X_t = torch.from_numpy(X).float()
    ds = TensorDataset(X_t)
    dl = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True)
    return dl


def save_epoch_curve(ep_list, cls_list, domD_list, advdom_list, out_path):
    plt.figure()
    plt.plot(ep_list, cls_list, label="L_cls (train epoch avg)")
    plt.plot(ep_list, domD_list, label="L_dom_D (train epoch avg)")
    plt.plot(ep_list, advdom_list, label="L_dom_F (train epoch avg)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Epoch-level Curves: iterative DANN")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def create_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(log_dir, f"train_log_{timestamp}.txt")
    print(f"[Info] Logging to {log_path}")

    def write_log(msg: str):
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    return write_log


def set_requires_grad(model, flag: bool):
    for p in model.parameters():
        p.requires_grad_(flag)


def main():
    ap = argparse.ArgumentParser()

    # source domain
    ap.add_argument("--src_segments", required=True, help="source segments .npy (N,1,T) or (N,T)")
    ap.add_argument("--src_labels",   required=True, help="source labels .npy (N,), {0,1}")

    # target domain
    ap.add_argument("--tgt_segments", required=True, help="target segments .npy (N,1,T) or (N,T)")

    ap.add_argument("--save_ckpt",    required=True)
    ap.add_argument("--save_src_roc", required=True)

    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--bs",        type=int,   default=64)
    ap.add_argument("--epochs",    type=int,   default=50)

    ap.add_argument("--lr_feat_ex",    type=float, default=1e-3)
    ap.add_argument("--lr_classifier", type=float, default=1e-4)
    ap.add_argument("--lr_domain",     type=float, default=1e-4)

    ap.add_argument("--weight_decay",       type=float, default=1e-4)
    ap.add_argument("--lambda_adv",         type=float, default=1.0, help="maximum adversarial strength")
    ap.add_argument("--n_domain",           type=int,   default=1, help="number of D updates per step")
    ap.add_argument("--seed",               type=int,   default=1337)
    ap.add_argument("--early_stop_patience",type=int,   default=8)

    args = ap.parse_args()

    if args.seed == -1:
        args.seed = random.randint(0, 10000)
        print(f"[Info] No seed provided. Generated random seed: {args.seed}")
    else:
        print(f"[Info] Using provided seed: {args.seed}")

    run_tag = f"seed{args.seed}_" + datetime.now().strftime("%Y%m%d-%H%M%S")

    base_dir = os.path.dirname(args.save_ckpt)
    if base_dir == "":
        base_dir = "."

    output_dir = os.path.join(base_dir, run_tag)
    os.makedirs(output_dir, exist_ok=True)

    ckpt_f1   = os.path.join(output_dir, "best_f1.pth")
    ckpt_auc  = os.path.join(output_dir, "best_auc.pth")
    ckpt_wacc = os.path.join(output_dir, "best_wacc.pth")

    args.save_ckpt = ckpt_f1
    args.save_src_roc = os.path.join(output_dir, "roc_src.png")
    logger = create_logger(output_dir)

    print(f"[Info] OUTPUT_DIR   = {output_dir}")
    print(f"[Info] save_ckpt    = {args.save_ckpt}")
    print(f"[Info] save_src_roc = {args.save_src_roc}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # ---------- load data ----------
    print("[Info] loading source & target data...")
    Xs = np.load(args.src_segments)
    ys = np.load(args.src_labels)
    Xt = np.load(args.tgt_segments)

    src_tr_dl, src_va_dl, ys_all = make_src_dataloaders(
        Xs, ys, val_ratio=args.val_ratio, bs=args.bs, seed=args.seed
    )
    tgt_tr_dl = make_tgt_dataloader(Xt, bs=args.bs)

    print(f"[Info] Source N={len(ys_all)}, Target N={len(Xt)} (unlabeled in training)")

    # ---------- build model ----------
    backbone = ResNet1D_18_backbone(in_ch=1).to(device)
    feat_dim = 512 if not hasattr(backbone, "out_dim") else backbone.out_dim

    classifier = ClassifierHead(in_dim=feat_dim, num_classes=2).to(device)
    domain_D = DomainDiscriminator(
        in_dim=feat_dim,
        hidden_dim=256,
        n_domains=2
    ).to(device)

    # ---------- loss / optimizer ----------
    loss_cls = nn.CrossEntropyLoss()
    loss_dom = nn.CrossEntropyLoss()

    opt_F = torch.optim.Adam(
        backbone.parameters(),
        lr=args.lr_feat_ex,
        weight_decay=args.weight_decay
    )
    opt_C = torch.optim.Adam(
        classifier.parameters(),
        lr=args.lr_classifier,
        weight_decay=args.weight_decay
    )
    opt_D = torch.optim.Adam(
        domain_D.parameters(),
        lr=args.lr_domain,
        weight_decay=args.weight_decay
    )

    sched_F = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt_F, mode='max', factor=0.5, patience=5
    )
    sched_C = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt_C, mode='max', factor=0.5, patience=5
    )
    sched_D = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt_D, mode='max', factor=0.5, patience=5
    )

    best_f1 = -1.0
    best_auc = -1.0
    best_wacc = -1.0

    no_improve_count = 0
    patience = args.early_stop_patience
    burn_in = 8

    def save_state(path, epoch):
        state = {
            "resnet_backbone": {k: v.detach().cpu() for k, v in backbone.state_dict().items()},
            "classifier": {k: v.detach().cpu() for k, v in classifier.state_dict().items()},
            "domain_discriminator": {k: v.detach().cpu() for k, v in domain_D.state_dict().items()},
            "feat_dim": feat_dim,
            "epoch": epoch,
        }
        torch.save(state, path)

    ep_list = []
    cls_list = []
    domD_list = []
    advdom_list = []

    # ---------- training ----------
    for ep in range(1, args.epochs + 1):
        backbone.train()
        classifier.train()
        domain_D.train()

        run_cls = 0.0
        run_dom_D = 0.0
        run_dom_F = 0.0
        dom_acc_total = 0.0
        dom_steps = 0

        tgt_iter = iter(tgt_tr_dl)
        src_iter = iter(src_tr_dl)
        num_steps = len(src_tr_dl)

        pbar = tqdm(range(num_steps), desc=f"Ep {ep}/{args.epochs}", ncols=120)

        for step in pbar:
            # -------------------------
            # 1) update domain_D
            # -------------------------
            loss_D_step = 0.0
            dom_acc_step = 0.0

            for _ in range(args.n_domain):
                try:
                    xs_batch_D, ys_batch_D = next(src_iter)
                except StopIteration:
                    src_iter = iter(src_tr_dl)
                    xs_batch_D, ys_batch_D = next(src_iter)

                try:
                    (xt_batch_D,) = next(tgt_iter)
                except StopIteration:
                    tgt_iter = iter(tgt_tr_dl)
                    (xt_batch_D,) = next(tgt_iter)

                xs_batch_D = xs_batch_D.to(device)
                xt_batch_D = xt_batch_D.to(device)

                with torch.no_grad():
                    feat_s_D = backbone(xs_batch_D)
                    feat_t_D = backbone(xt_batch_D)

                dom_feat_D = torch.cat([feat_s_D.detach(), feat_t_D.detach()], dim=0)
                dom_label_s_D = torch.zeros(feat_s_D.size(0), dtype=torch.long, device=device)
                dom_label_t_D = torch.ones(feat_t_D.size(0), dtype=torch.long, device=device)
                dom_label_D = torch.cat([dom_label_s_D, dom_label_t_D], dim=0)

                dom_logits_D = domain_D(dom_feat_D, use_grl=False)
                L_dom_D = loss_dom(dom_logits_D, dom_label_D)

                opt_D.zero_grad(set_to_none=True)
                L_dom_D.backward()
                opt_D.step()

                loss_D_step += L_dom_D.item()

                with torch.no_grad():
                    dom_pred_D = dom_logits_D.argmax(dim=1)
                    dom_acc_D = (dom_pred_D == dom_label_D).float().mean().item()
                    dom_acc_step += dom_acc_D

            loss_D_step /= args.n_domain
            dom_acc_step /= args.n_domain

            # -------------------------
            # 2) update backbone + classifier
            # -------------------------
            try:
                xs_batch, ys_batch = next(src_iter)
            except StopIteration:
                src_iter = iter(src_tr_dl)
                xs_batch, ys_batch = next(src_iter)

            try:
                (xt_batch,) = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(tgt_tr_dl)
                (xt_batch,) = next(tgt_iter)

            xs_batch = xs_batch.to(device)
            ys_batch = ys_batch.to(device)
            xt_batch = xt_batch.to(device)

            lambda_adv = args.lambda_adv

            # freeze D when updating F/C
            set_requires_grad(domain_D, False)

            feat_s = backbone(xs_batch)
            feat_t = backbone(xt_batch)

            logits_s = classifier(feat_s)
            L_cls = loss_cls(logits_s, ys_batch)

            dom_feat_F = torch.cat([feat_s, feat_t], dim=0)
            dom_label_s_F = torch.zeros(feat_s.size(0), dtype=torch.long, device=device)
            dom_label_t_F = torch.ones(feat_t.size(0), dtype=torch.long, device=device)
            dom_label_F = torch.cat([dom_label_s_F, dom_label_t_F], dim=0)

            # no GRL here, because adversarial sign is handled manually
            dom_logits_F = domain_D(dom_feat_F, use_grl=False)
            L_dom_F = loss_dom(dom_logits_F, dom_label_F)

            # iterative DANN objective for backbone/classifier
            L_total = L_cls - lambda_adv * L_dom_F

            opt_F.zero_grad(set_to_none=True)
            opt_C.zero_grad(set_to_none=True)
            L_total.backward()
            opt_F.step()
            opt_C.step()

            set_requires_grad(domain_D, True)

            run_cls += L_cls.item()
            run_dom_D += loss_D_step
            run_dom_F += L_dom_F.item()
            dom_acc_total += dom_acc_step
            dom_steps += 1

            pbar.set_postfix({
                "lambda": f"{lambda_adv:.5f}",
                "L_cls": f"{L_cls.item():.4f}",
                "L_domD": f"{loss_D_step:.4f}",
                "L_domF": f"{L_dom_F.item():.4f}",
                "domAcc": f"{dom_acc_step:.3f}",
            })

        tr_cls_loss = run_cls / max(1, len(src_tr_dl))
        tr_dom_D_loss = run_dom_D / max(1, len(src_tr_dl))
        tr_dom_F_loss = run_dom_F / max(1, len(src_tr_dl))
        mean_dom_acc = dom_acc_total / max(1, dom_steps)

        print(
            f"[Train-Summary] Ep {ep:03d}/{args.epochs} | "
            f"lambda={lambda_adv:.6f}"
            f"cls_loss={tr_cls_loss:.5f} "
            f"domD_loss={tr_dom_D_loss:.5f} "
            f"domF_loss={tr_dom_F_loss:.5f} "
            f"dom_acc={mean_dom_acc:.4f} "
        )

        ep_list.append(ep)
        cls_list.append(tr_cls_loss)
        domD_list.append(tr_dom_D_loss)
        advdom_list.append(tr_dom_F_loss)

        # ---------- eval on source ----------
        backbone.eval()
        classifier.eval()

        vrun = 0.0
        all_prob, all_pred, all_true = [], [], []

        with torch.no_grad():
            for xb, yb in src_va_dl:
                xb, yb = xb.to(device), yb.to(device)
                feat = backbone(xb)
                logits = classifier(feat)
                prob = torch.softmax(logits, dim=1).cpu().numpy()
                pred = prob.argmax(1)
                loss = loss_cls(logits, yb)

                vrun += loss.item()
                all_prob.append(prob)
                all_pred.append(pred)
                all_true.append(yb.cpu().numpy())

        va_loss = vrun / max(1, len(src_va_dl))
        all_prob = np.concatenate(all_prob, 0)
        all_pred = np.concatenate(all_pred, 0)
        all_true = np.concatenate(all_true, 0)

        src_metrics, src_cm = evaluate_metrics(all_true, all_prob, all_pred)

        print(
            f"[DA-Iter] Ep {ep:03d}/{args.epochs} | "
            f"train_cls={tr_cls_loss:.5f} "
            f"train_domD={tr_dom_D_loss:.5f} "
            f"train_domF={tr_dom_F_loss:.5f} "
            f"dom_acc={mean_dom_acc:.4f} | "
            f"SRC_val: loss={va_loss:.5f} "
            f"ACC={src_metrics['acc']:.4f} "
            f"SEN={src_metrics['recall_sensitivity']:.4f} "
            f"SPEC={src_metrics['specificity']:.4f} "
            f"WACC={src_metrics['w_acc']:.4f} "
            f"F1={src_metrics['f1']:.4f} "
            f"AUC={src_metrics['auroc']:.4f}"
        )

        logger(
            f"[SRC-VAL] Ep {ep:03d} | "
            f"loss={va_loss:.6f} "
            f"ACC={src_metrics['acc']:.4f} "
            f"SEN={src_metrics['recall_sensitivity']:.4f} "
            f"SPEC={src_metrics['specificity']:.4f} "
            f"WACC={src_metrics['w_acc']:.4f} "
            f"F1={src_metrics['f1']:.4f} "
            f"AUC={src_metrics['auroc']:.4f} "
            f"DOM_ACC={mean_dom_acc:.4f}"
        )

        src_f1 = float(src_metrics["f1"])
        src_wacc = float(src_metrics["w_acc"])
        src_auc = float(src_metrics["auroc"]) if not np.isnan(src_metrics["auroc"]) else -1.0

        sched_F.step(src_f1)
        sched_C.step(src_f1)
        sched_D.step(src_f1)

        f1_improved = False

        if src_f1 > best_f1:
            best_f1 = src_f1
            f1_improved = True
            save_state(ckpt_f1, ep)
            print(f"  ✅ saved BestF1={best_f1:.4f} → {ckpt_f1}")
            logger(f"[BEST_F1] Ep {ep:03d} | F1={best_f1:.4f}")

        if src_auc > best_auc:
            best_auc = src_auc
            save_state(ckpt_auc, ep)
            print(f"  ✅ saved BestAUC={best_auc:.4f} → {ckpt_auc}")
            logger(f"[BEST_AUC] Ep {ep:03d} | AUC={best_auc:.4f}")

        if src_wacc > best_wacc:
            best_wacc = src_wacc
            save_state(ckpt_wacc, ep)
            print(f"  ✅ saved BestWACC={best_wacc:.4f} → {ckpt_wacc}")
            logger(f"[BEST_WACC] Ep {ep:03d} | WACC={best_wacc:.4f}")

        if ep <= burn_in:
            no_improve_count = 0
        else:
            if f1_improved:
                no_improve_count = 0
            else:
                no_improve_count += 1
                print(f"No F1 improvement for {no_improve_count} epochs")

                if no_improve_count >= patience:
                    print(f"Early stopping triggered after {ep} epochs (patience={patience})")
                    logger(f"[EarlyStop] triggered at Ep {ep:03d} (patience={patience}) best_f1={best_f1:.4f}")
                    break

    # ---------- save curve ----------
    fig_dir = os.path.join(output_dir, "figs")
    os.makedirs(fig_dir, exist_ok=True)

    out_png = os.path.join(fig_dir, "epoch_iterative_dann_curves.png")
    save_epoch_curve(ep_list, cls_list, domD_list, advdom_list, out_png)
    print(f"✅ saved epoch curve → {out_png}")
    logger(f"[PLOT] saved epoch curve → {out_png}")

    # ---------- final ROC ----------
    plot_roc(all_true, all_prob, args.save_src_roc, title="Source ROC (Iterative DANN)")
    print(f"✅ saved SRC ROC figure → {args.save_src_roc}")
    logger(f"[ROC] saved SRC ROC figure → {args.save_src_roc}")


if __name__ == "__main__":
    main()
