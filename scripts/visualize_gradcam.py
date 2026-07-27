import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import random

from models.resnet_backbone import ResNet1D_18_backbone
from models.classifier import ClassifierHead 

from models.gradcam import GradCAM_1D

# ==========================================
# ==========================================
def plot_comparison_cam(ecg_signal, cam_base, cam_da, true_label, save_path=None):
    fig, axes = plt.subplots(2, 1, figsize=(16, 6), sharex=True)
    length = len(ecg_signal)
    time_steps = np.arange(length)
    
    class_name = "AF" if true_label == 1 else "Normal (Non-AF)"
    
    ax1 = axes[0]
    im1 = ax1.imshow(cam_base[np.newaxis, :], cmap='jet', aspect='auto', alpha=0.9,
                     vmin=0.0, vmax=1.0, extent=[0, length, np.min(ecg_signal)-0.5, np.max(ecg_signal)+0.5])
    ax1.plot(time_steps, ecg_signal, color='k', linewidth=1.5, alpha=0.9)
    ax1.set_xlim(0, length)
    ax1.set_ylabel('Amplitude')
    ax1.set_title(f'Baseline Model (Failed) | Ground Truth: {class_name}', fontsize=12, fontweight='bold')
    
    ax2 = axes[1]
    im2 = ax2.imshow(cam_da[np.newaxis, :], cmap='jet', aspect='auto', alpha=0.9,
                     vmin=0.0, vmax=1.0, extent=[0, length, np.min(ecg_signal)-0.5, np.max(ecg_signal)+0.5])
    ax2.plot(time_steps, ecg_signal, color='k', linewidth=1.5, alpha=0.9)
    ax2.set_xlim(0, length)
    ax2.set_ylabel('Amplitude')
    ax2.set_xlabel('Time (Samples)')
    ax2.set_title(f'DANN Model (Succeeded) | Ground Truth: {class_name}', fontsize=12, fontweight='bold')
    
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(im1, cax=cbar_ax, label='Importance Weight')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison heatmap saved to {save_path}")
    plt.show()

# ==========================================
# ==========================================
class AF_Detector_Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ResNet1D_18_backbone(in_ch=1)
        self.classifier = ClassifierHead(in_dim=512, hidden_dim=256, num_classes=2)

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
# ==========================================
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading Baseline Model...")
    model_base = load_model_weights(AF_Detector_Model().to(device), args.base_weight, device)
    
    print("Loading DANN Model...")
    model_dann = load_model_weights(AF_Detector_Model().to(device), args.dann_weight, device)

    cam_extractor_base = GradCAM_1D(model_base, model_base.backbone.layer4[-1])
    cam_extractor_dann = GradCAM_1D(model_dann, model_dann.backbone.layer4[-1])

    print("Loading Dataset...")
    data = np.load(args.data_path)     
    labels = np.load(args.label_path)  
    
    target_viz_label = args.target_class 
    target_indices = np.where(labels == target_viz_label)[0].tolist()
    
    random.shuffle(target_indices)
    
    print(f"Searching for a case where Baseline fails but DANN succeeds (Target Class: {target_viz_label})...")
    found_idx = -1
    
    with torch.no_grad():
        for idx in target_indices:
            ecg_signal_1d = np.squeeze(data[idx])
            input_tensor = torch.tensor(ecg_signal_1d, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            
            pred_base = model_base(input_tensor).argmax(dim=1).item()
            pred_dann = model_dann(input_tensor).argmax(dim=1).item()
            
            if pred_base != target_viz_label and pred_dann == target_viz_label:
                found_idx = idx
                print(f"\n[BINGO!] Found a perfect case at Index: {found_idx}")
                print(f"Ground Truth: {target_viz_label} | Baseline Pred: {pred_base} | DANN Pred: {pred_dann}")
                break
                
    if found_idx == -1:
        print("Could not find any sample matching the criteria in this dataset.")
        return

    ecg_signal_1d = np.squeeze(data[found_idx])
    input_tensor = torch.tensor(ecg_signal_1d, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    
    cam_weights_base, _ = cam_extractor_base.generate_cam(input_tensor, target_class=target_viz_label)
    cam_weights_dann, _ = cam_extractor_dann.generate_cam(input_tensor, target_class=target_viz_label)

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    save_name = os.path.join(save_dir, f"compare_idx{found_idx}_Target{target_viz_label}.png")
    
    plot_comparison_cam(ecg_signal_1d, cam_weights_base, cam_weights_dann, true_label=target_viz_label, save_path=save_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find and compare Grad-CAM for Baseline vs DANN")
    parser.add_argument('--base_weight', type=str, required=True, help='Path to Baseline model weights')
    parser.add_argument('--dann_weight', type=str, required=True, help='Path to DANN model weights')
    
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--label_path', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    
    parser.add_argument('--target_class', type=int, default=1, choices=[0, 1])
    
    args = parser.parse_args()
    main(args)
