import torch
import torch.nn as nn

def _same_pad_1d(k: int) -> int:
    assert k % 2 == 1, "Use odd kernel sizes to emulate 'same' padding."
    return (k - 1) // 2

class ConvBN(nn.Module):
    def __init__(self, in_ch, out_ch, k, s):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, k, s, padding=_same_pad_1d(k), bias=False)
        self.bn   = nn.BatchNorm1d(out_ch)
    def forward(self, x):
        return self.bn(self.conv(x))

class Bottleneck1D(nn.Module):
    expansion = 4
    def __init__(self, in_ch, mid_ch, k=3, s=1):
        super().__init__()
        out_ch = mid_ch * self.expansion
        self.conv1 = ConvBN(in_ch, mid_ch, k=1, s=1)
        self.conv2 = ConvBN(mid_ch, mid_ch, k=k, s=s)
        self.conv3 = ConvBN(mid_ch, out_ch, k=1, s=1)
        self.relu  = nn.ReLU(inplace=True)
        self.proj  = None
        if s != 1 or in_ch != out_ch:
            self.proj = ConvBN(in_ch, out_ch, k=1, s=s)
    def forward(self, x):
        shortcut = x if self.proj is None else self.proj(x)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        x = self.relu(x + shortcut)
        return x

class ResNet1D_50_backbone(nn.Module):
    """
    """
    def __init__(self, in_ch=1, ks=(3,3,3,3)):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, 64, kernel_size=7, stride=2, padding=_same_pad_1d(7), bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)  # 'same' pool
        )
        self.layer1 = self._make_layer(64,  64, blocks=3, stride_first=1, k=ks[0])  # output: 256 ch
        self.layer2 = self._make_layer(256, 128, blocks=4, stride_first=2, k=ks[1]) # output: 512 ch
        self.layer3 = self._make_layer(512, 256, blocks=6, stride_first=2, k=ks[2]) # output: 1024 ch
        self.layer4 = self._make_layer(1024,512, blocks=3, stride_first=2, k=ks[3]) # output: 2048 ch
        self.gap    = nn.AdaptiveAvgPool1d(1)

    def _make_layer(self, in_ch, mid_ch, blocks, stride_first, k):
        layers = [Bottleneck1D(in_ch, mid_ch, k=k, s=stride_first)]
        out_ch = mid_ch * Bottleneck1D.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck1D(out_ch, mid_ch, k=k, s=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.gap(x).squeeze(-1)
        return x


#===========================ResNet 18 Backbone ======================
# ------------------------------------------------------------------
# BasicBlock 
# ------------------------------------------------------------------

class BasicBlock1D(nn.Module):
    """
    BasicBlock for ResNet-18 and ResNet-34 architectures in 1D.
    3*3 Conv -> BN -> ReLU -> 3*3 Conv -> BN -> Add -> ReLU
    """
    expansion = 1

    def __init__(self, in_ch, out_ch, k=3, s=1):
        super().__init__()
        self.conv1 = ConvBN(in_ch, out_ch, k=k, s=s)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = ConvBN(out_ch, out_ch, k=k, s=1)
        
        self.proj = None
        if s != 1 or in_ch != out_ch * self.expansion:
            self.proj = ConvBN(in_ch, out_ch * self.expansion, k=1, s=s)

    def forward(self, x):
        shortcut = x if self.proj is None else self.proj(x)
        
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        
        x = self.relu(x + shortcut)
        return x

class ResNet1D_18_backbone(nn.Module):
    """
    Layers: [2, 2, 2, 2] blocks
    """
    def __init__(self, in_ch=1, ks=(3,3,3,3)):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, 64, kernel_size=7, stride=2, padding=_same_pad_1d(7), bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        
        self.layer1 = self._make_layer(64,  64,  blocks=2, stride_first=1, k=ks[0]) # out: 64
        self.layer2 = self._make_layer(64,  128, blocks=2, stride_first=2, k=ks[1]) # out: 128
        self.layer3 = self._make_layer(128, 256, blocks=2, stride_first=2, k=ks[2]) # out: 256
        self.layer4 = self._make_layer(256, 512, blocks=2, stride_first=2, k=ks[3]) # out: 512
        
        self.gap = nn.AdaptiveAvgPool1d(1)

    def _make_layer(self, in_ch, mid_ch, blocks, stride_first, k):
        layers = [BasicBlock1D(in_ch, mid_ch, k=k, s=stride_first)]
        
        out_ch = mid_ch * BasicBlock1D.expansion
        
        for _ in range(1, blocks):
            layers.append(BasicBlock1D(out_ch, mid_ch, k=k, s=1))
            
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x).squeeze(-1) # Output shape: (Batch, 512)
        return x


class ResNet1D_101_backbone(nn.Module):
    """
    Layers: [3, 4, 23, 3]
    """
    def __init__(self, in_ch=1, ks=(3, 3, 3, 3)):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(
                in_ch,
                64,
                kernel_size=7,
                stride=2,
                padding=_same_pad_1d(7),
                bias=False
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        # ResNet-101 configuration: [3, 4, 23, 3]
        self.layer1 = self._make_layer(64,   64,  blocks=3,  stride_first=1, k=ks[0])   # out: 256
        self.layer2 = self._make_layer(256,  128, blocks=4,  stride_first=2, k=ks[1])   # out: 512
        self.layer3 = self._make_layer(512,  256, blocks=23, stride_first=2, k=ks[2])   # out: 1024
        self.layer4 = self._make_layer(1024, 512, blocks=3,  stride_first=2, k=ks[3])   # out: 2048

        self.gap = nn.AdaptiveAvgPool1d(1)

    def _make_layer(self, in_ch, mid_ch, blocks, stride_first, k):
        layers = [Bottleneck1D(in_ch, mid_ch, k=k, s=stride_first)]
        out_ch = mid_ch * Bottleneck1D.expansion

        for _ in range(1, blocks):
            layers.append(Bottleneck1D(out_ch, mid_ch, k=k, s=1))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x).squeeze(-1)   # Output shape: (Batch, 2048)
        return x
