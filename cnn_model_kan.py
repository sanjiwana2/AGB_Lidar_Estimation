import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the efficient-kan implementation (must be saved as efficient_kan.py)
from Model.efficient_KAN import KAN


# ==========================================
# 🔧 GLOBAL UTILITY FUNCTIONS
# ==========================================

def safe_input_size(input_size):
    """Extracts the last element if input_size is a tuple/list."""
    return input_size[-1] if isinstance(input_size, (tuple, list)) else input_size


def compute_flattened_size(network, input_shape):
    """Dynamically computes the flattened feature size after forward pass."""
    device = next(network.parameters()).device
    with torch.no_grad():
        dummy_input = torch.zeros(2, *input_shape).to(device)
        output = network(dummy_input)
        return output.view(2, -1).size(1)


# ==========================================
# 🧠 SQUEEZE-AND-EXCITATION (SE) BLOCKS
# ==========================================

class SEBlock1D(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = x.mean(dim=2)
        return x * self.fc(y).view(b, c, 1)


class SEBlock2D(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = x.mean(dim=(2, 3))
        return x * self.fc(y).view(b, c, 1, 1)


# ==========================================
# 🚀 KAN & CNN-KAN ARCHITECTURES
# ==========================================

class CNN1D_KAN(nn.Module):
    def __init__(self, input_size, num_classes=1, kan_hidden=128, use_se=True, grid_size=5, spline_order=3, dropout=0.3):
        super().__init__()
        
        input_size = safe_input_size(input_size)
        
        # Build 1D Convolutional Back-bone pipeline
        modules = []
        configs = [(1, 128), (128, 256), (256, 512)]
        
        for in_ch, out_ch in configs:
            modules.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
            ])
            if use_se:
                modules.append(SEBlock1D(out_ch))
            modules.extend([
                nn.MaxPool1d(2),
                nn.Dropout(dropout)
            ])
            
        self.cnn = nn.Sequential(*modules)
        
        # Calculate dynamic shapes safely
        self.flattened_size = compute_flattened_size(self.cnn, (1, input_size))
        
        self.kan = KAN(
            layers_hidden=[self.flattened_size, kan_hidden, num_classes],
            grid_size=grid_size,
            spline_order=spline_order
        )

    def forward(self, x, update_grid=False):
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add channel dimension if missing: (B, L) -> (B, 1, L)
        x = self.cnn(x)
        x = x.flatten(1)
        return self.kan(x, update_grid=update_grid)


class CNN2D_KAN(nn.Module):
    def __init__(self, input_size=(1, 12, 62), num_classes=1, kan_hidden=128, use_se=False, grid_size=5, spline_order=3):
        super().__init__()

        # Build 2D Convolutional Back-bone pipeline
        modules = []
        configs = [(1, 64), (64, 128), (128, 256)]
        
        for in_ch, out_ch in configs:
            modules.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU()
            ])
            if use_se:
                modules.append(SEBlock2D(out_ch))
            modules.append(nn.MaxPool2d(2))

        self.cnn = nn.Sequential(*modules)
        
        # Calculate dynamic shapes safely
        self.flattened_size = compute_flattened_size(self.cnn, input_size)
        
        self.kan = KAN(
            layers_hidden=[self.flattened_size, kan_hidden, num_classes],
            grid_size=grid_size,
            spline_order=spline_order
        )

    def forward(self, x, update_grid=False):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        return self.kan(x, update_grid=update_grid)
