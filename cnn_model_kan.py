import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 🔥 SE BLOCKS
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
        b, c, l = x.size()
        y = x.mean(dim=2)
        y = self.fc(y).view(b, c, 1)
        return x * y


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
        b, c, h, w = x.size()
        y = x.mean(dim=(2, 3))
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


# ==========================================
# 🔥 IMPROVED KAN LAYER (STABLE)
# ==========================================

class KANLayer(nn.Module):
    def __init__(self, in_features, out_features, hidden=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.LayerNorm(hidden),   # 🔥 stabilizer
            nn.SiLU(),

            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),

            nn.Linear(hidden, out_features)
        )

    def forward(self, x):
        return self.net(x)


# ==========================================
# 🔧 HELPER FUNCTION (CRITICAL FIX)
# ==========================================

def safe_input_size(input_size):
    if isinstance(input_size, tuple):
        return input_size[-1]
    return input_size


# ==========================================
# 🚀 1D CNN (NO SE)
# ==========================================

class CNN1D_KAN(nn.Module):
    def __init__(self, input_size, num_classes=1):
        super().__init__()

        input_size = safe_input_size(input_size)

        self.conv1 = nn.Conv1d(1, 128, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.pool1 = nn.MaxPool1d(2)
        self.drop1 = nn.Dropout(0.3)

        self.conv2 = nn.Conv1d(128, 256, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(256)
        self.pool2 = nn.MaxPool1d(2)
        self.drop2 = nn.Dropout(0.3)

        self.conv3 = nn.Conv1d(256, 512, 3, padding=1)
        self.bn3 = nn.BatchNorm1d(512)
        self.pool3 = nn.MaxPool1d(2)
        self.drop3 = nn.Dropout(0.3)

        self.flattened_size = self._get_flattened_size(input_size)
        self.kan = KANLayer(self.flattened_size, num_classes)

    def _get_flattened_size(self, input_size):
        device = next(self.parameters()).device
        with torch.no_grad():
            x = torch.zeros(2, 1, input_size).to(device)
            x = self.pool1(F.relu(self.bn1(self.conv1(x))))
            x = self.pool2(F.relu(self.bn2(self.conv2(x))))
            x = self.pool3(F.relu(self.bn3(self.conv3(x))))
            return x.view(2, -1).size(1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.drop1(x)
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.drop2(x)
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.drop3(x)
        x = x.view(x.size(0), -1)
        return self.kan(x)


# ==========================================
# 🚀 1D CNN (WITH SE)
# ==========================================

class CNN1D_KAN_SE(nn.Module):
    def __init__(self, input_size, num_classes=1):
        super().__init__()

        input_size = safe_input_size(input_size)

        self.conv1 = nn.Conv1d(1, 128, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.se1 = SEBlock1D(128)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(128, 256, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(256)
        self.se2 = SEBlock1D(256)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(256, 512, 3, padding=1)
        self.bn3 = nn.BatchNorm1d(512)
        self.se3 = SEBlock1D(512)
        self.pool3 = nn.MaxPool1d(2)

        self.flattened_size = self._get_flattened_size(input_size)
        self.kan = KANLayer(self.flattened_size, num_classes)

    def _get_flattened_size(self, input_size):
        device = next(self.parameters()).device
        with torch.no_grad():
            x = torch.zeros(2, 1, input_size).to(device)
            x = self.pool1(self.se1(F.relu(self.bn1(self.conv1(x)))))
            x = self.pool2(self.se2(F.relu(self.bn2(self.conv2(x)))))
            x = self.pool3(self.se3(F.relu(self.bn3(self.conv3(x)))))
            return x.view(2, -1).size(1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool1(self.se1(F.relu(self.bn1(self.conv1(x)))))
        x = self.pool2(self.se2(F.relu(self.bn2(self.conv2(x)))))
        x = self.pool3(self.se3(F.relu(self.bn3(self.conv3(x)))))
        x = x.view(x.size(0), -1)
        return self.kan(x)


# ==========================================
# 🚀 2D CNN (NO SE)
# ==========================================

class CNN2D_KAN(nn.Module):
    def __init__(self, input_size=(1, 12, 62), num_classes=1):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.flattened_size = self._get_flattened_size(input_size)
        self.kan = KANLayer(self.flattened_size, num_classes)

    def _get_flattened_size(self, shape):
        device = next(self.parameters()).device
        with torch.no_grad():
            x = torch.zeros(2, *shape).to(device)
            x = self.cnn(x)
            return x.view(2, -1).size(1)

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        return self.kan(x)


# ==========================================
# 🚀 2D CNN (WITH SE)
# ==========================================

class CNN2D_KAN_SE(nn.Module):
    def __init__(self, input_size=(1, 12, 62), num_classes=1):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            SEBlock2D(64),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            SEBlock2D(128),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            SEBlock2D(256),
            nn.MaxPool2d(2)
        )

        self.flattened_size = self._get_flattened_size(input_size)
        self.kan = KANLayer(self.flattened_size, num_classes)

    def _get_flattened_size(self, shape):
        device = next(self.parameters()).device
        with torch.no_grad():
            x = torch.zeros(2, *shape).to(device)
            x = self.cnn(x)
            return x.view(2, -1).size(1)

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        return self.kan(x)