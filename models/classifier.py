'''

Just Classifier Head
Should combine with resnet backbone
The ckpt will load resnet classifier critics Parameter

'''
import torch
import torch.nn as nn
'''
class ClassifierHead(nn.Module):
    def __init__(self, in_dim, num_classes=2):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)


    def forward(self, feat):
        return self.fc(feat)
'''

class ClassifierHead_2048_256_2(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
        )


    def forward(self, feat):
        return self.net(feat)

class ClassifierHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
        )


    def forward(self, feat):
        return self.net(feat)

ClassifierHead_512_256_2 = ClassifierHead
