import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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
        y = x.mean(dim=2)              # Global Average Pooling
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
        y = x.mean(dim=(2, 3))         # Global Average Pooling
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

# ==========================================
# 1D MODELS
# ==========================================

class CNN1D(nn.Module):
    def __init__(self, input_size, num_classes=1):
        super(CNN1D, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(128)  
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.3)  
        
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(256)  
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0.3)  

        self.conv3 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(512)  
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout(0.3)

        self.flattened_size = self._get_flattened_size(input_size)
        
        self.fc1 = nn.Linear(self.flattened_size, 256)
        self.dropout_fc = nn.Dropout(0.5) 
        self.fc2 = nn.Linear(256, num_classes)

    def _get_flattened_size(self, input_size):
        with torch.no_grad():
            dummy_input = torch.zeros(2, 1, input_size) # Changed to batch size 2 for BatchNorm stability
            x = self.pool1(F.relu(self.bn1(self.conv1(dummy_input))))
            x = self.pool2(F.relu(self.bn2(self.conv2(x))))
            x = self.pool3(F.relu(self.bn3(self.conv3(x))))
            flattened_size = x.view(x.size(0), -1).size(1)
        return flattened_size

    def forward(self, x):
        x = x.unsqueeze(1)  
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout1(x)
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout2(x)
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout3(x)
        x = x.view(x.size(0), -1)  
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.fc2(x)
        return x

class CNNLSTM(nn.Module):
    def __init__(self, input_size, num_classes=1):
        super(CNNLSTM, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),  
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.3),  
            
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),  
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.3),  
            
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),  
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.3)  
        )
        
        self.lstm = nn.LSTM(input_size=512, hidden_size=256, num_layers=1, batch_first=True)
        self.dropout_lstm = nn.Dropout(0.5)  
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = self.dropout_lstm(x)  
        x = x[:, -1, :]
        x = self.fc(x)
        return x

class CNN1D_SE(nn.Module):
    def __init__(self, input_size, num_classes=1):
        super().__init__()

        self.conv1 = nn.Conv1d(1, 128, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.se1 = SEBlock1D(128)
        self.pool1 = nn.MaxPool1d(2)
        self.drop1 = nn.Dropout(0.3)

        self.conv2 = nn.Conv1d(128, 256, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(256)
        self.se2 = SEBlock1D(256)
        self.pool2 = nn.MaxPool1d(2)
        self.drop2 = nn.Dropout(0.3)

        self.conv3 = nn.Conv1d(256, 512, 3, padding=1)
        self.bn3 = nn.BatchNorm1d(512)
        self.se3 = SEBlock1D(512)
        self.pool3 = nn.MaxPool1d(2)
        self.drop3 = nn.Dropout(0.3)

        self.flattened_size = self._get_flattened_size(input_size)

        self.fc1 = nn.Linear(self.flattened_size, 256)
        self.drop_fc = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def _get_flattened_size(self, input_size):
        with torch.no_grad():
            x = torch.zeros(2, 1, input_size)
            x = self.pool1(self.se1(F.relu(self.bn1(self.conv1(x)))))
            x = self.pool2(self.se2(F.relu(self.bn2(self.conv2(x)))))
            x = self.pool3(self.se3(F.relu(self.bn3(self.conv3(x)))))
            return x.view(2, -1).size(1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool1(self.se1(F.relu(self.bn1(self.conv1(x)))))
        x = self.drop1(x)
        x = self.pool2(self.se2(F.relu(self.bn2(self.conv2(x)))))
        x = self.drop2(x)
        x = self.pool3(self.se3(F.relu(self.bn3(self.conv3(x)))))
        x = self.drop3(x)

        x = x.view(x.size(0), -1)
        x = self.drop_fc(F.relu(self.fc1(x)))
        return self.fc2(x)

class CNNLSTM_SE(nn.Module):
    def __init__(self, input_size, num_classes=1):
        super().__init__()

        self.cnn = nn.Sequential(
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

        self.lstm = nn.LSTM(512, 256, batch_first=True)
        self.drop = nn.Dropout(0.5)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = self.drop(x[:, -1, :])
        return self.fc(x)

# ==========================================
# 2D MODELS
# ==========================================

class CNN2D(nn.Module):
    def __init__(self, input_size, num_classes=1):
        super(CNN2D, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=2, padding=1),  
            nn.BatchNorm2d(64), # Changed from InstanceNorm
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  
            nn.Dropout(0.3)
        )
        
        self.cnn_out_size = self._get_cnn_output_size(input_size)
        self.dropout_fc = nn.Dropout(0.5) # Added missing dropout
        self.fc = nn.Linear(self.cnn_out_size, num_classes)

    def _get_cnn_output_size(self, shape):
        with torch.no_grad():
            dummy_input = torch.zeros(2, *shape) # Changed to batch size 2 for BatchNorm stability
            output = self.cnn(dummy_input)
        return int(torch.prod(torch.tensor(output.shape[1:])))  

    def forward(self, x):
        x = self.cnn(x)  
        x = x.view(x.size(0), -1)  
        x = self.dropout_fc(x) # Applied dropout before final layer
        x = self.fc(x)
        return x

class CNNLSTM2D(nn.Module):
    def __init__(self, input_size, num_classes=1):
        super(CNNLSTM2D, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=2, padding=1),
            nn.BatchNorm2d(64), # Changed from InstanceNorm
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3)
        )
        
        self.cnn_out_size = self._get_cnn_output_size(input_size) 
        
        self.lstm = nn.LSTM(input_size=self.cnn_out_size, hidden_size=32, num_layers=1, batch_first=True)
        self.dropout_lstm = nn.Dropout(0.5) # Added missing dropout
        self.fc = nn.Linear(32, num_classes)

    def _get_cnn_output_size(self, shape):
        with torch.no_grad():
            dummy_input = torch.zeros(2, *shape) # Changed to batch size 2 for BatchNorm stability
            output = self.cnn(dummy_input)
        return int(torch.prod(torch.tensor(output.shape[1:])))  

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)  
        x = x.unsqueeze(1)  
        x, _ = self.lstm(x)
        x = self.dropout_lstm(x) # Applied dropout after LSTM
        x = x[:, -1, :]  
        x = self.fc(x)
        return x

class CNN2D_SE(nn.Module):
    def __init__(self, input_size, num_classes=1):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            SEBlock2D(64),
            nn.MaxPool2d(2),
            nn.Dropout(0.3)
        )

        self.out_size = self._get_out_size(input_size)
        self.drop = nn.Dropout(0.5)
        self.fc = nn.Linear(self.out_size, num_classes)

    def _get_out_size(self, shape):
        with torch.no_grad():
            x = torch.zeros(2, *shape)
            x = self.cnn(x)
            return x.view(2, -1).size(1)

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.drop(x)
        return self.fc(x)

class CNNLSTM2D_SE(nn.Module):
    def __init__(self, input_size, num_classes=1):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            SEBlock2D(64),
            nn.MaxPool2d(2),
            nn.Dropout(0.3)
        )

        self.out_size = self._get_out_size(input_size)

        self.lstm = nn.LSTM(self.out_size, 32, batch_first=True)
        self.drop = nn.Dropout(0.5)
        self.fc = nn.Linear(32, num_classes)

    def _get_out_size(self, shape):
        with torch.no_grad():
            x = torch.zeros(2, *shape)
            x = self.cnn(x)
            return x.view(2, -1).size(1)

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1).unsqueeze(1)
        x, _ = self.lstm(x)
        x = self.drop(x[:, -1, :])
        return self.fc(x)