import torch
import torch.nn as nn
import torch.nn.functional as F
from config import ACTION_SPACE, CNN_CHANNELS

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class ChessResNet(nn.Module):
    def __init__(self, num_res_blocks=6, num_channels=128):
        super().__init__()
        
        # --- Input Block ---
        # Wejście: 18 kanałów (z config.py/encoding.py)
        self.conv_input = nn.Conv2d(18, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(num_channels)
        
        # --- Residual Tower ---
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_channels) for _ in range(num_res_blocks)
        ])
        
        # --- Policy Head ---
        # Redukcja kanałów do 2, potem spłaszczenie
        self.policy_conv = nn.Conv2d(num_channels, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, ACTION_SPACE)
        
        # --- Value Head ---
        # Redukcja kanałów do 1, potem FC -> 256 -> 1
        self.value_conv = nn.Conv2d(num_channels, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        # x: (B, 18, 8, 8)
        out = F.relu(self.bn_input(self.conv_input(x)))
        
        for block in self.res_blocks:
            out = block(out)
            
        # Policy Head
        p = F.relu(self.policy_bn(self.policy_conv(out)))
        p = p.view(p.size(0), -1)  # Flatten
        policy_logits = self.policy_fc(p)
        
        # Value Head
        v = F.relu(self.value_bn(self.value_conv(out)))
        v = v.view(v.size(0), -1)  # Flatten
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v)) # Tanh dla wartości w zakresie [-1, 1]
        
        return policy_logits, value