import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# SE BLOCKS
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
# 1D MODELS (MLP VERSION)
# ==========================================

class CNN1D_MLP(nn.Module):
    def __init__(self, input_size, num_classes=1):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(1, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            SEBlock1D(128),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),

            nn.Conv1d(128, 256, 3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            SEBlock1D(256),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),

            nn.Conv1d(256, 512, 3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            SEBlock1D(512),
            nn.MaxPool1d(2),
            nn.Dropout(0.3)
        )

        self.flattened_size = self._get_flattened_size(input_size)

        self.mlp = nn.Sequential(
            nn.Linear(self.flattened_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, num_classes)
        )

    def _get_flattened_size(self, input_size):
        with torch.no_grad():
            x = torch.zeros(2, 1, input_size)
            x = self.features(x)
            return x.view(2, -1).size(1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.mlp(x)


class CNN1D_SE_MLP(nn.Module):
    def __init__(self, input_size, num_classes=1):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(1, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            SEBlock1D(128),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),

            nn.Conv1d(128, 256, 3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            SEBlock1D(256),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),

            nn.Conv1d(256, 512, 3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            SEBlock1D(512),
            nn.MaxPool1d(2),
            nn.Dropout(0.3)
        )

        self.flattened_size = self._get_flattened_size(input_size)

        self.mlp = nn.Sequential(
            nn.Linear(self.flattened_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, num_classes)
        )

    def _get_flattened_size(self, input_size):
        with torch.no_grad():
            x = torch.zeros(2, 1, input_size)
            x = self.features(x)
            return x.view(2, -1).size(1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.mlp(x)

# ==========================================
# 2D MODELS (MLP VERSION)
# ==========================================

class CNN2D_MLP(nn.Module):
    def __init__(self, input_size, num_classes=1):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            SEBlock2D(64),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),

            nn.Conv2d(64, 128, 2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            SEBlock2D(128),
            nn.MaxPool2d(2),
            nn.Dropout(0.3)
        )

        self.flattened_size = self._get_flattened_size(input_size)

        self.mlp = nn.Sequential(
            nn.Linear(self.flattened_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, num_classes)
        )

    def _get_flattened_size(self, shape):
        with torch.no_grad():
            x = torch.zeros(2, *shape)
            x = self.features(x)
            return x.view(2, -1).size(1)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.mlp(x)


class CNN2D_SE_MLP(nn.Module):
    def __init__(self, input_size, num_classes=1):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            SEBlock2D(64),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),

            nn.Conv2d(64, 128, 2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            SEBlock2D(128),
            nn.MaxPool2d(2),
            nn.Dropout(0.3)
        )

        self.flattened_size = self._get_flattened_size(input_size)

        self.mlp = nn.Sequential(
            nn.Linear(self.flattened_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, num_classes)
        )

    def _get_flattened_size(self, shape):
        with torch.no_grad():
            x = torch.zeros(2, *shape)
            x = self.features(x)
            return x.view(2, -1).size(1)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.mlp(x)