import os
import torch
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import argparse

from models.resnet_backbone import ResNet1D_50_backbone, ResNet1D_18_backbone
from models.vgg19_backbone import VGG19_1D_backbone
from models.inception_backbone import InceptionBackbone1D


BACKBONES = {
    "resnet18": ResNet1D_18_backbone,
    "vgg19": VGG19_1D_backbone,
    "inception": InceptionBackbone1D,
}

def get_balanced_indices(labels, n_per_class=1000):
    """Sample up to a fixed number of examples from each class."""
    indices = []
    unique_labels = np.unique(labels)
    for label in unique_labels:
        label_indices = np.where(labels == label)[0]
        count = min(len(label_indices), n_per_class)
        sampled = np.random.choice(label_indices, count, replace=False)
        indices.extend(sampled)
    return np.array(indices)

def evaluate_metrics(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint = torch.load(args.model_path, map_location=device)
    backbone = BACKBONES[args.backbone](in_ch=1).to(device)
    backbone.load_state_dict(checkpoint["resnet_backbone"])
    backbone.eval()
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_root = os.environ.get("PROCESSED_DATA_ROOT", os.path.join(project_root, "data", "processed"))
    datasets = {
        "AFDB": {
            "seg": os.path.join(data_root, "AFDB", "AFDB_segments_reduced3_overlap50_100s_bp.npy"),
            "lab": os.path.join(data_root, "AFDB", "AFDB_labels_reduced3_overlap50_100s_bp.npy")
        },
        "LTAFDB": {
            "seg": os.path.join(data_root, "LTAFDB", "LTAFDB_segments_reduced3_overlap50_100s_bp.npy"),
            "lab": os.path.join(data_root, "LTAFDB", "LTAFDB_labels_reduced3_overlap50_100s_bp.npy")
        },
        "MITDB": {
            "seg": os.path.join(data_root, "MITDB", "MITDB_segments_reduced3_overlap50_100s_bp.npy"),
            "lab": os.path.join(data_root, "MITDB", "MITDB_labels_reduced3_overlap50_100s_bp.npy")
        }
    }

    all_features = []
    all_cell_labels = []
    all_batch_labels = []

    for d_name in [args.src, args.tgt]:
        segs = np.load(datasets[d_name]["seg"], mmap_mode='r')
        labs = np.load(datasets[d_name]["lab"], mmap_mode='r')
        
        idx = get_balanced_indices(labs, n_per_class=1500)
        
        feat_list = []
        with torch.no_grad():
            for i in range(0, len(idx), 64):
                batch_idx = idx[i:i+64]
                x = torch.from_numpy(segs[batch_idx].astype(np.float32)).to(device)
                if x.ndim == 2: x = x.unsqueeze(1)
                f = backbone(x)
                feat_list.append(f.cpu().numpy())
        
        all_features.append(np.concatenate(feat_list, axis=0))
        all_cell_labels.append(labs[idx])
        all_batch_labels.append([d_name] * len(idx))

    X = np.concatenate(all_features, axis=0)
    y_cell = np.concatenate(all_cell_labels, axis=0)
    y_batch = np.concatenate(all_batch_labels, axis=0)

    X_scaled = StandardScaler().fit_transform(X)
    X_pca = PCA(n_components=32).fit_transform(X_scaled)

    asw_cell = silhouette_score(X_pca, y_cell)
    asw_batch = silhouette_score(X_pca, y_batch)

    print(f"Results for {args.src} -> {args.tgt}:")
    print(f"  ASW_celltype: {asw_cell:.4f}")
    print(f"  ASW_batch:    {asw_batch:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--backbone", choices=sorted(BACKBONES), default="resnet18")
    parser.add_argument("--src", type=str, choices=["AFDB", "LTAFDB", "MITDB"], required=True)
    parser.add_argument("--tgt", type=str, choices=["AFDB", "LTAFDB", "MITDB"], required=True)
    args = parser.parse_args()
    evaluate_metrics(args)
