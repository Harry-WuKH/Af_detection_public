import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class GradCAM_1D:
    """
    """
    def __init__(self, model, target_layer):
        """
        Args:
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_tensor, target_class=None):
        """
        Args:
        Returns:
        """
        self.model.eval()
        
        # 1. Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
            
        self.model.zero_grad()
        target_score = output[0, target_class]
        target_score.backward()
        
        weights = torch.mean(self.gradients, dim=2, keepdim=True) 
        
        cam = torch.sum(weights * self.activations, dim=1).squeeze(0)
        
        cam = F.relu(cam)
        
        cam = cam - torch.min(cam)
        cam_max = torch.max(cam)
        if cam_max > 0:
            cam = cam / cam_max
            
        cam_resized = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0), 
            size=input_tensor.shape[2], 
            mode='linear', 
            align_corners=False
        )
        
        return cam_resized.squeeze().cpu().numpy(), target_class



def plot_1d_cam(ecg_signal, cam, target_class, save_path=None):
    """
    """
    fig, ax = plt.subplots(figsize=(16, 3))
    
    length = len(ecg_signal)
    
    
    im = ax.imshow(
        cam[np.newaxis, :], 
        cmap='jet',           
        aspect='auto', 
        alpha=0.9,
        vmin=0.0, vmax=1.0,
        extent=[0, length, np.min(ecg_signal)-0.5, np.max(ecg_signal)+0.5]
    )
    
    ax.plot(np.arange(length), ecg_signal, color='k', linewidth=1.5, alpha=0.9)
    
    ax.set_xlim(0, length)
    ax.set_ylabel('Amplitude')
    
    class_name = "AF" if target_class == 1 else "Normal (Non-AF)"
    ax.set_title(f'Grad-CAM Visualization | Target: {class_name}', fontsize=12, fontweight='bold')
    
    cbar = fig.colorbar(im, ax=ax, orientation='vertical', pad=0.01)
    cbar.set_label('Importance', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
