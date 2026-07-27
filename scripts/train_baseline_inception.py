# scripts/train_baseline_inception.py
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
from models.inception_backbone import InceptionBackbone1D
from models.classifier import ClassifierHead
from datetime import datetime
from tqdm.auto import tqdm

'''
This version update how to calculate the 
distance between source and target.
src.detach() frozen source data distribution 
and move the target distribution.
'''

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

'''
def make_src_dataloaders(X, y, val_ratio, bs, seed=1337):
    """
    X: (N,T) or (N,1,T)
    y: (N,), in {0,1}
    """
    # m is in {0,1}
    m = np.isin(y, [0, 1]) 
    X = X[m]
    y = y[m] #discard the other class, only use AF/Non-AF
    y = y.astype(np.int64 )

    if X.ndim == 2:
        X = X[:, np.newaxis, :]
    elif X.ndim == 3 and X.shape[1] == 1:
        pass
    else:
        raise ValueError(f"X must be (N,T) or (N,1,T), got {X.shape}")

    N = len(y)
    val_len = int(N * val_ratio)
    tr_len = N - val_len

    X_t = torch.from_numpy(X).float()
    y_t = torch.from_numpy(y).long()
    ds = TensorDataset(X_t, y_t)
    tr_ds, va_ds = random_split(
        ds, [tr_len, val_len], generator=torch.Generator().manual_seed(seed)
    )

    tr_dl = DataLoader(tr_ds, batch_size=bs, shuffle=True, drop_last=True)
    va_dl = DataLoader(va_ds, batch_size=bs, shuffle=False, drop_last=False)
    return tr_dl, va_dl, y
'''

def make_src_dataloaders(X, y, val_ratio, bs, seed=1337):
    """
    """
    # m is in {0,1}
    m = np.isin(y, [0, 1]) 
    X = X[m]
    y = y[m] 
    y = y.astype(np.int64)

    if X.ndim == 2:
        X = X[:, np.newaxis, :]
    elif X.ndim == 3 and X.shape[1] == 1:
        pass
    else:
        raise ValueError(f"X must be (N,T) or (N,1,T), got {X.shape}")

    N = len(y)
    val_len = int(N * val_ratio)
    tr_len = N - val_len

    X_t = torch.from_numpy(X).float()
    y_t = torch.from_numpy(y).long()
    ds = TensorDataset(X_t, y_t)
    
    tr_ds, va_ds = random_split(
        ds, [tr_len, val_len], generator=torch.Generator().manual_seed(seed)
    )

    tr_indices = tr_ds.indices 
    tr_labels = y_t[tr_indices].numpy() 
    
    class_counts = np.bincount(tr_labels)
    class_weights = 1. / class_counts
    
    sample_weights = class_weights[tr_labels]
    sample_weights = torch.from_numpy(sample_weights).double()
    
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    tr_dl = DataLoader(tr_ds, batch_size=bs, sampler=sampler, shuffle=False, drop_last=True, num_workers=8, pin_memory=True)
    
    va_dl = DataLoader(va_ds, batch_size=bs, shuffle=False, drop_last=False, num_workers=8, pin_memory=True)
    
    return tr_dl, va_dl, y


def compute_class_weight(y: np.ndarray, device):
    classes, counts = np.unique(y, return_counts=True) #for class [0,1] for counts [9000 1000]
    w = counts.sum() / (2.0 * counts)
    class_weight = torch.tensor(w, dtype=torch.float32, device=device)
    print(f"[Info] class_weight: {class_weight.cpu().numpy().tolist()} (order labels={classes.tolist()})")
    return class_weight


def create_logger(log_dir):
    """
    logger for debugggg
    """
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
    # source domain (With_label)
    ap.add_argument("--src_segments", required=True, help="source segments .npy (N,1,T) or (N,T)")
    ap.add_argument("--src_labels",   required=True,   help="source labels .npy (N,), {0,1}")

    ap.add_argument("--save_ckpt",    required=True)
    ap.add_argument("--save_src_roc", required=True)

    ap.add_argument("--val_ratio",          type=float, default=0.1)
    ap.add_argument("--bs",                 type=int,   default=64)
    ap.add_argument("--epochs",             type=int,   default=50)

    ap.add_argument("--lr_feat_ex",         type=float, default=1e-4)
    ap.add_argument("--lr_classifier",      type=float, default=1e-4)


    ap.add_argument("--weight_decay",       type=float, default=1e-4)

    ap.add_argument("--seed",               type=int,   default=-1)
    ap.add_argument("--early_stop_patience",type=int,   default=8, help="the patience for early stop, if val_f1 not update for n time -> stop")

    args = ap.parse_args()

    if args.seed == -1: # Not given seed -> random seed
        args.seed = random.randint(0, 10000)
        print(f"[Info] No seed provided. Generated random seed: {args.seed}")
    else:
        print(f"[Info] Using provided seed: {args.seed}")

    #folder name
    run_tag = f"seed{args.seed}_" + datetime.now().strftime("%Y%m%d-%H%M%S")

    base_dir = os.path.dirname(args.save_ckpt)
    if base_dir == "":
        base_dir = "."  

    #Under Source_to_Target file build a timestamp folder to store result
    #like AFDB_to_MITDB/20251122/ <- out_dir
    output_dir = os.path.join(base_dir, run_tag)
    os.makedirs(output_dir, exist_ok=True)

    # model roc_png store in out_dir
    args.save_ckpt = os.path.join(output_dir, "best_f1.pth")
    args.save_src_roc = os.path.join(output_dir, "roc_src.png")
    logger = create_logger(output_dir)

    print(f"[Info] OUTPUT_DIR   = {output_dir}")
    print(f"[Info] save_ckpt    = {args.save_ckpt}")
    print(f"[Info] save_src_roc = {args.save_src_roc}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #for reproduce the same result
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False

    for k, v in vars(args).items():
        logger(f"{k}: {v}")

    # ---------- Load DATA ----------
    print("[Info] loading source & target data...") 
    Xs = np.load(args.src_segments)   # (Ns,1,T) or (Ns,T)
    ys = np.load(args.src_labels)     # (Ns,)

    src_tr_dl, src_va_dl, ys_all = make_src_dataloaders(
        Xs, ys, val_ratio=args.val_ratio, bs=args.bs, seed=args.seed
    )

    print(f"[Info] Source N={len(ys_all)} (unlabeled in training)")

    # ---------- build model ----------
    # InceptionBackbone1D + ClassifierHead
    backbone = InceptionBackbone1D(in_ch=1).to(device)
    feat_dim = 512 if not hasattr(backbone, "out_dim") else backbone.out_dim
    classifier = ClassifierHead(in_dim=feat_dim, num_classes=2).to(device)
    # ---------- loss / optimizer ----------
    #class_weight = compute_class_weight(ys_all, device=device) #will return like [5.5, 0.5]
    loss_cls = nn.CrossEntropyLoss() # only the classification loss with weight
    
    # this is to collect all the parameter for update
    # still need further check, don't know whther the backbone and discriminator 
    # should share the same function or not 

    opt_F = torch.optim.Adam(backbone.parameters(),   lr=args.lr_feat_ex, weight_decay=args.weight_decay)
    opt_C = torch.optim.Adam(classifier.parameters(), lr=args.lr_classifier, weight_decay=args.weight_decay)
    #Using Adam Optimization, maybe can try AdamW in the future experiment

    # Add LR scheduler 
    sch_F = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_F, mode='max', factor=0.5, patience=3)
    sch_C = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_C, mode='max', factor=0.5, patience=3)

    best_src_f1 = -1.0
    best_state = None # for initialization
    patience = args.early_stop_patience 
    no_improve_count = 0

    ep_list = []
    cls_list = []

    # ---------- training ----------
    for ep in range(1, args.epochs + 1):
        backbone.train()
        classifier.train()

        run_cls = 0.0    
        src_iter = iter(src_tr_dl)
        num_steps = len(src_tr_dl)

        pbar = tqdm(range(num_steps), desc=f"Ep {ep}/{args.epochs}", ncols=120)

        #---------------------------
        # 1. training 
        #---------------------------
        for step in pbar:

            try:
                xs_batch, ys_batch = next(src_iter)
            except StopIteration:
                src_iter = iter(src_tr_dl)
                xs_batch, ys_batch = next(src_iter)
            xs_batch, ys_batch = xs_batch.to(device), ys_batch.to(device)

            opt_F.zero_grad()
            opt_C.zero_grad()

            feat_s = backbone(xs_batch)
            logits_s = classifier(feat_s)
            L_cls = loss_cls(logits_s, ys_batch)
            
            # ----- Update ResNet Backbone -------
            L_cls.backward() 
            opt_F.step()
            opt_C.step()

            '''
            logger(
                f"[Backbone] Ep {ep:03d} Step {step:03d}/{num_steps} | "
                f"L_cls={L_cls.item():.6f} "
            )
            '''
            
            #SUM ALL THE BATCH VALUE
            run_cls   += L_cls.item() 
        
        #The average loss in each BATCH
        #The average performance in this EPOCHS
        tr_cls_loss = run_cls   / max(1, len(src_tr_dl))

        print(
            f"[Train-Summary] Ep {ep:03d}/{args.epochs} | "
            f"cls_loss={tr_cls_loss:.5f} "
        )

        logger(
            f"[Train-Summary] Ep {ep:03d}/{args.epochs} | "
            f"cls_loss={tr_cls_loss:.6f} "
        )

        ep_list.append(ep)
        cls_list.append(tr_cls_loss)


        # ---------- eval on source (only) ----------
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
        src_f1 = src_metrics["f1"]

        sch_F.step(src_f1)
        sch_C.step(src_f1)



        print(
            f"[Train] Ep {ep:03d}/{args.epochs}\n"
            f"SRC_val: loss={va_loss:.3f}  ACC={src_metrics['acc']:.4f}  "
            f"SEN={src_metrics['recall_sensitivity']:.4f}  "
            f"SPEC={src_metrics['specificity']:.4f}  "
            f"WACC={src_metrics['w_acc']:.4f}"
            f"F1={src_metrics['f1']:.4f}  AUC={src_metrics['auroc']:.4f}"
        )

        logger(
            f"[SRC-VAL] Ep {ep:03d} | "
            f"loss={va_loss:.6f} "
            f"ACC={src_metrics['acc']:.4f} "
            f"SEN={src_metrics['recall_sensitivity']:.4f} "
            f"SPEC={src_metrics['specificity']:.4f} "
            f"WACC={src_metrics['w_acc']:.4f} "
            f"F1={src_metrics['f1']:.4f} "
            f"AUC={src_metrics['auroc']:.4f}"
        )

        if src_f1 > best_src_f1:
            best_src_f1 = src_f1
            best_state = {
                "resnet_backbone": {k: v.detach().cpu() for k, v in backbone.state_dict().items()},
                "classifier": {k: v.detach().cpu() for k, v in classifier.state_dict().items()},
                "feat_dim": feat_dim,
            }
            os.makedirs(os.path.dirname(args.save_ckpt), exist_ok=True)
            torch.save(best_state, args.save_ckpt)
            print(f"  ✅ saved best DA model (src F1={best_src_f1:.4f}) → {args.save_ckpt}")
            logger(f"[BEST] Ep {ep:03d} | src_F1={best_src_f1:.4f} saved={args.save_ckpt}")
            no_improve_count = 0   # reset patience
        
        else:
            no_improve_count += 1
            print(f"No Improvement for {no_improve_count} epochs =(")

            if no_improve_count >= patience:
                print(f"Early stopping triggered after {ep} epochs (patience={patience})")
                logger(f"[EarlyStop] triggered at Ep {ep:03d} (patience={patience})")
                break


    # ---- plot epoch-level (L_cls vs adv_loss) ----
    print("\n[Info] Reloading best model for final reporting...")
    checkpoint = torch.load(args.save_ckpt)
    backbone.load_state_dict(checkpoint["resnet_backbone"])
    classifier.load_state_dict(checkpoint["classifier"])
    
    backbone.eval()
    classifier.eval()
    
    final_prob, final_true = [], []
    with torch.no_grad():
        for xb, yb in src_va_dl:
            xb = xb.to(device)
            feat = backbone(xb)
            logits = classifier(feat)
            prob = torch.softmax(logits, dim=1).cpu().numpy()
            final_prob.append(prob)
            final_true.append(yb.numpy())

    final_prob = np.concatenate(final_prob, 0)
    final_true = np.concatenate(final_true, 0)

    # Plot Loss Curve
    fig_dir = os.path.join(output_dir, "figs")
    os.makedirs(fig_dir, exist_ok=True)
    out_png = os.path.join(fig_dir, "epoch_cls_loss.png")

    plt.figure()
    plt.plot(ep_list, cls_list, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    logger(f"[PLOT] saved loss curve → {out_png}")

    # Plot ROC (Best Model)
    plot_roc(final_true, final_prob, args.save_src_roc, title=f"Source ROC (Best F1={best_src_f1:.3f})")
    print(f"✅ saved SRC ROC figure (Best Model) → {args.save_src_roc}")
    logger(f"[ROC] saved SRC ROC figure → {args.save_src_roc}")

if __name__ == "__main__":
    main()
