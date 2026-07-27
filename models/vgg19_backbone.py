import torch
import torch.nn as nn


class VGG19_1D_backbone(nn.Module):
    def __init__(self, in_ch=1):
        super().__init__()
        self.out_dim = 512

        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(in_ch, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Conv1d(64, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool1d(2),

            # Block 2
            nn.Conv1d(64, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Conv1d(128, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.MaxPool1d(2),

            # Block 3
            nn.Conv1d(128, 256, 3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            nn.Conv1d(256, 256, 3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            nn.Conv1d(256, 256, 3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            nn.Conv1d(256, 256, 3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            nn.MaxPool1d(2),

            # Block 4
            nn.Conv1d(256, 512, 3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),

            nn.Conv1d(512, 512, 3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),

            nn.Conv1d(512, 512, 3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),

            nn.Conv1d(512, 512, 3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),

            nn.MaxPool1d(2),

            # Block 5
            nn.Conv1d(512, 512, 3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),

            nn.Conv1d(512, 512, 3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),

            nn.Conv1d(512, 512, 3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),

            nn.Conv1d(512, 512, 3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),

            nn.MaxPool1d(2),
        )

        self.gap = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.features(x)   # [B, 512, L]
        x = self.gap(x)        # [B, 512, 1]
        x = x.squeeze(-1)      # [B, 512]
        return x
